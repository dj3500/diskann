/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Steiner augmentation of a graph (`flab steiner`): auxiliary router nodes with protected
//! out-edges appended to a graph. The rows of the base graph's nodes (data points and its own
//! auxiliary nodes, e.g. a frozen start) are copied unchanged and no node of the base graph gets
//! an edge to a router: routers are reached only from the start table, so a search that does
//! not start at a router is unchanged.
//!
//! - `tree`: for every selected label, a b-ary tree over its members in one shared spatial
//!   order (the depth-first leaf order of a random-projection tree over all points): the
//!   members, sorted by that order, in chunks of `b` under leaf routers; routers grouped `b` at
//!   a time up to one root, which becomes the label's start. A label with at most `b` members
//!   gets one router over all of them (a posting scan expressed as graph expansion). With
//!   `--partition kmeans` the tree is instead built top-down over the label's own members:
//!   a node's members are split by k-means (k = min(b, ⌈m / b⌉), a few Lloyd iterations, seeded)
//!   until at most `b` remain under a leaf router.
//! - `connector`: for every selected label, one entry per source strongly connected component
//!   of the subgraph its members induce (the member nearest to the component's centroid);
//!   with several such components, a b-ary tree of routers over these entries whose root
//!   becomes a start of the label. The label's previous starts are kept.
//! - `pairs`: a tree (as `tree`) over the joint members of every selected stored pair of a
//!   combination-postings file (`--pairs`; a pseudo-label for the conjunction of two labels),
//!   appended to the base graph (typically a graph that already has label trees); the label
//!   starts are kept and the roots go to `pair_roots.flab` (one per directory slot of the
//!   pairs file, `u32::MAX` without a tree), the starts of the kernel's `tree_pairs` route.
//!
//! Router vectors: the centroid of the router's data-point descendants (`centroid`), or the
//! descendant nearest to that centroid (`medoid`: a member's vector, as for a router that would
//! only store the member's id).

use std::{path::PathBuf, time::Instant};

use anyhow::{ensure, Context, Result};
use clap::{Args, ValueEnum};
use serde_json::json;

use crate::{
    container,
    diag::{row_f32, rp_leaves, scc},
    graph::{AuxKind, AuxNodes, Graph},
    labels::LabelStore,
    pairs::PairIndex,
    parallel::par_map,
    run::data::{dataset_dtype, load_graph},
    starts::Starts,
    vectors::{DType, Element, Matrix},
};

/// Magic of `pair_roots.flab` (the roots of the pair trees by directory slot).
pub const PAIR_ROOTS_MAGIC: &[u8; 8] = b"FLABPRT\0";
const PAIR_ROOTS_VERSION: u32 = 1;

/// Writes `pair_roots.flab`: params `[n_pairs, n_entries, dict_hash, n_points]` of the pairs
/// file, then the root of every directory slot (`u32::MAX`: no tree).
pub fn write_pair_roots(path: &std::path::Path, px: &PairIndex, roots: &[u32]) -> Result<()> {
    ensure!(roots.len() == px.n_pairs(), "one root per stored pair");
    let params = [
        px.n_pairs() as u64,
        px.n_entries() as u64,
        px.dict_hash(),
        px.n_points() as u64,
        0,
        0,
        0,
        0,
    ];
    container::write(
        container::create(path)?,
        PAIR_ROOTS_MAGIC,
        PAIR_ROOTS_VERSION,
        &params,
        &[bytemuck::cast_slice(roots)],
    )
    .with_context(|| format!("writing {}", path.display()))
}

/// Reads `pair_roots.flab` and checks that it belongs to the pairs file `px`.
pub fn read_pair_roots(path: &std::path::Path, px: &PairIndex) -> Result<Vec<u32>> {
    let mut r = container::Reader::open(path, PAIR_ROOTS_MAGIC)
        .with_context(|| format!("loading pair roots from {}", path.display()))?;
    ensure!(
        r.version() == PAIR_ROOTS_VERSION,
        "unsupported pair roots version"
    );
    let [n_pairs, n_entries, dict_hash, n_points, ..] = *r.params();
    ensure!(
        n_pairs as usize == px.n_pairs()
            && n_entries as usize == px.n_entries()
            && dict_hash == px.dict_hash()
            && n_points as usize == px.n_points(),
        "{} was built for another combination-postings file",
        path.display()
    );
    r.section(0, px.n_pairs())
}

/// Construction of `flab steiner`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Construction {
    /// Per-label b-ary trees over the members in one shared spatial order.
    Tree,
    /// Per-label routers over one entry per source component of the induced subgraph.
    Connector,
    /// Trees over the joint members of the stored pairs of a combination-postings file.
    Pairs,
}

/// How a `tree` partitions a label's members.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Partition {
    /// Chunks of `b` in the shared spatial order, grouped bottom-up.
    Order,
    /// Top-down k-means over the label's own members.
    Kmeans,
}

/// Router vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum RouterVector {
    /// The centroid of the router's data-point descendants.
    Centroid,
    /// The descendant nearest to that centroid.
    Medoid,
}

/// Arguments of `flab steiner`.
#[derive(Debug, Args)]
pub struct SteinerArgs {
    /// Dataset directory (base vectors, `labels.flab`).
    #[arg(long)]
    pub dataset: PathBuf,
    /// Base graph (directory with `graph.flab`, or `graph.flab`); its auxiliary nodes (e.g. a
    /// frozen start) are kept.
    #[arg(long)]
    pub graph: PathBuf,
    /// Base start table (its global starts and the per-label starts of labels without
    /// routers are kept).
    #[arg(long)]
    pub starts: PathBuf,
    /// Construction.
    #[arg(long, value_enum, default_value = "tree")]
    pub construction: Construction,
    /// `pairs`: the combination-postings file (`flab pairs build`) whose stored pairs get trees.
    #[arg(long)]
    pub pairs: Option<PathBuf>,
    /// Fan-out `b` of the routers (leaf chunks of `b` members).
    #[arg(long, default_value_t = 32)]
    pub block: usize,
    /// Only labels with more than this many members get routers.
    #[arg(long, default_value_t = 4)]
    pub min_size: usize,
    /// Only labels with at most this many members get routers: a number or `n/<k>`.
    #[arg(long, default_value = "n/100")]
    pub max_size: String,
    /// `tree`: give routers to exactly the labels listed in this file (one label id per line,
    /// e.g. `flab select --track graph`'s `trees.txt`) instead of the size range.
    #[arg(long, conflicts_with = "pairs")]
    pub labels_file: Option<PathBuf>,
    /// Router vectors.
    #[arg(long, value_enum, default_value = "centroid")]
    pub router: RouterVector,
    /// `tree`: how a label's members are partitioned.
    #[arg(long, value_enum, default_value = "order")]
    pub partition: Partition,
    /// `--partition kmeans`: Lloyd iterations per split.
    #[arg(long, default_value_t = 4)]
    pub kmeans_iters: usize,
    /// `--partition kmeans`: the iterations of a split of more than this many members times
    /// its k run on an evenly strided sample of that size; one pass then assigns every member
    /// (0: every iteration on all members).
    #[arg(long, default_value_t = 0)]
    pub kmeans_sample: usize,
    /// `tree`: keep the label's previous starts next to its root.
    #[arg(long)]
    pub keep_starts: bool,
    /// Leaf size of the shared spatial order (random-projection tree).
    #[arg(long, default_value_t = 64)]
    pub order_block: usize,
    /// Seed of the shared spatial order.
    #[arg(long, default_value_t = 7)]
    pub seed: u64,
    /// Threads.
    #[arg(short = 't', long, default_value_t = 1)]
    pub threads: usize,
    /// Output directory (`graph.flab`, `starts.flab`, `meta.json`).
    #[arg(long)]
    pub out_dir: PathBuf,
}

/// Runs `flab steiner`.
pub fn run(args: &SteinerArgs) -> Result<()> {
    match dataset_dtype(&args.dataset)? {
        DType::U8 => run_typed::<u8>(args),
        DType::F32 => run_typed::<f32>(args),
    }
}

/// The label ids of a list file (one per line; blank lines and `#` comments ignored), sorted and
/// deduplicated, each below `n_labels`.
pub fn read_label_list(path: &std::path::Path, n_labels: usize) -> Result<Vec<u32>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading the label list {}", path.display()))?;
    parse_label_list(&text, n_labels, &path.display().to_string())
}

fn parse_label_list(text: &str, n_labels: usize, origin: &str) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let t = line.split('#').next().unwrap_or("").trim();
        if t.is_empty() {
            continue;
        }
        let l: u32 = t
            .parse()
            .with_context(|| format!("{origin}:{}: not a label id: {t:?}", i + 1))?;
        ensure!(
            (l as usize) < n_labels,
            "{origin}:{}: label {l} out of range ({n_labels} labels)",
            i + 1
        );
        out.push(l);
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// A label-size bound: `n/k` (= ⌈n/k⌉ of `n` points) or a number of points.
pub fn parse_size(s: &str, n: usize) -> Result<usize> {
    let s = s.trim();
    if let Some(k) = s.strip_prefix("n/") {
        let k: usize = k.parse().with_context(|| format!("bad size {s:?}"))?;
        ensure!(k > 0, "bad size {s:?}");
        return Ok(n.div_ceil(k));
    }
    s.parse().with_context(|| format!("bad size {s:?}"))
}

/// A child of a router: a data point or another router of the same label (local index).
#[derive(Debug, Clone, Copy)]
enum Child {
    Point(u32),
    Router(u32),
}

/// The routers of one label, root last.
struct LabelRouters {
    label: u32,
    /// Per router: its children, and its vector (centroid) or the data point it copies.
    routers: Vec<(Vec<Child>, RouterVec)>,
    /// Entries of the label that are data points (`connector` with one source component).
    point_starts: Vec<u32>,
}

enum RouterVec {
    Centroid(Vec<f32>),
    Point(u32),
}

/// Centroid (f64 sums → f32) and, for `medoid`, the member nearest to it.
fn centroid_of<T: Element>(base: &Matrix<T>, members: &[u32], row: &mut [f32]) -> Vec<f32> {
    let d = base.ncols();
    let mut sum = vec![0f64; d];
    for &x in members {
        row_f32(base.row(x as usize), row);
        for (s, &v) in sum.iter_mut().zip(row.iter()) {
            *s += f64::from(v);
        }
    }
    let c = members.len().max(1) as f64;
    sum.iter().map(|&s| (s / c) as f32).collect()
}

fn nearest_to<T: Element>(base: &Matrix<T>, members: &[u32], c: &[f32], row: &mut [f32]) -> u32 {
    let mut best = (f32::INFINITY, u32::MAX);
    for &x in members {
        row_f32(base.row(x as usize), row);
        let d: f32 = row.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
        if (d, x) < best {
            best = (d, x);
        }
    }
    best.1
}

/// A b-ary tree of routers over `leaves` (data points, in order): chunks of `b` under leaf
/// routers, then routers grouped `b` at a time up to one root (last).
fn tree_over<T: Element>(
    base: &Matrix<T>,
    leaves: &[u32],
    b: usize,
    rv: RouterVector,
    row: &mut [f32],
) -> Vec<(Vec<Child>, RouterVec)> {
    let mut routers: Vec<(Vec<Child>, RouterVec)> = Vec::new();
    // Per router of the current level: its local index and its data-point descendants.
    let mut level: Vec<(u32, Vec<u32>)> = Vec::new();
    for chunk in leaves.chunks(b) {
        let c = centroid_of(base, chunk, row);
        let v = match rv {
            RouterVector::Centroid => RouterVec::Centroid(c),
            RouterVector::Medoid => RouterVec::Point(nearest_to(base, chunk, &c, row)),
        };
        routers.push((chunk.iter().map(|&x| Child::Point(x)).collect(), v));
        level.push(((routers.len() - 1) as u32, chunk.to_vec()));
    }
    while level.len() > 1 {
        let mut next = Vec::new();
        for group in level.chunks(b) {
            let desc: Vec<u32> = group.iter().flat_map(|(_, d)| d.iter().copied()).collect();
            let c = centroid_of(base, &desc, row);
            let v = match rv {
                RouterVector::Centroid => RouterVec::Centroid(c),
                RouterVector::Medoid => RouterVec::Point(nearest_to(base, &desc, &c, row)),
            };
            routers.push((group.iter().map(|(r, _)| Child::Router(*r)).collect(), v));
            next.push(((routers.len() - 1) as u32, desc));
        }
        level = next;
    }
    routers
}

/// A b-ary tree of routers over `members` built top-down by k-means on their own vectors
/// (`vecs`: the members' rows as `f32`, in the same order): a node with more than `b` members
/// is split into k = min(b, ⌈m / b⌉) clusters (Lloyd's iterations from k members spread over
/// the node, seeded by `seed`); empty clusters are dropped and a split that leaves all members
/// in one cluster falls back to consecutive chunks. Routers are pushed children first, the root
/// last; their vectors are the centroids of their descendants (`medoid`: the descendant nearest
/// to it).
#[expect(clippy::too_many_arguments, reason = "the tree's inputs and options")]
fn kmeans_tree<T: Element>(
    base: &Matrix<T>,
    members: &[u32],
    vecs: &[f32],
    b: usize,
    rv: RouterVector,
    iters: usize,
    sample: usize,
    seed: u64,
    row: &mut [f32],
) -> Vec<(Vec<Child>, RouterVec)> {
    let d = base.ncols();
    let mut routers: Vec<(Vec<Child>, RouterVec)> = Vec::new();
    // Work stack of (member positions, parent slot to fill); built post-order by recursion on
    // an explicit stack: (positions, children collected so far, pending child groups).
    fn vector_of<T: Element>(
        base: &Matrix<T>,
        ids: &[u32],
        c: Vec<f32>,
        rv: RouterVector,
        row: &mut [f32],
    ) -> RouterVec {
        match rv {
            RouterVector::Centroid => RouterVec::Centroid(c),
            RouterVector::Medoid => RouterVec::Point(nearest_to(base, ids, &c, row)),
        }
    }
    let centroid = |pos: &[usize]| -> Vec<f32> {
        let mut sum = vec![0f64; d];
        for &p in pos {
            for (s, &v) in sum.iter_mut().zip(&vecs[p * d..(p + 1) * d]) {
                *s += f64::from(v);
            }
        }
        let c = pos.len().max(1) as f64;
        sum.iter().map(|&s| (s / c) as f32).collect()
    };
    let split = |pos: &[usize], salt: u64| -> Vec<Vec<usize>> {
        let m = pos.len();
        let k = b.min(m.div_ceil(b)).max(2);
        // Initial centres: k members evenly spaced in a seeded rotation of the node.
        let off = (seed ^ salt).wrapping_mul(0x9E37_79B9_7F4A_7C15) as usize % m;
        let mut centres: Vec<f32> = (0..k)
            .flat_map(|j| {
                let p = pos[(off + j * m / k) % m];
                vecs[p * d..(p + 1) * d].to_vec()
            })
            .collect();
        // Lloyd's iterations on `fit` (all positions, or a strided sample), then one pass that
        // assigns every position to its nearest centre.
        let fit: Vec<usize> = if sample > 0 && m > sample * k {
            (0..sample * k).map(|i| pos[i * m / (sample * k)]).collect()
        } else {
            pos.to_vec()
        };
        let nearest = |v: &[f32], centres: &[f32]| -> usize {
            let mut best = (f32::INFINITY, 0usize);
            for j in 0..k {
                let c = &centres[j * d..(j + 1) * d];
                let dist: f32 = v.iter().zip(c).map(|(x, y)| (x - y) * (x - y)).sum();
                if dist < best.0 {
                    best = (dist, j);
                }
            }
            best.1
        };
        let mut assign = vec![0usize; fit.len()];
        for it in 0..iters.max(1) {
            for (a, &p) in assign.iter_mut().zip(&fit) {
                *a = nearest(&vecs[p * d..(p + 1) * d], &centres);
            }
            if it + 1 == iters.max(1) {
                break;
            }
            let mut sums = vec![0f64; k * d];
            let mut counts = vec![0usize; k];
            for (&a, &p) in assign.iter().zip(&fit) {
                counts[a] += 1;
                for (s, &v) in sums[a * d..(a + 1) * d]
                    .iter_mut()
                    .zip(&vecs[p * d..(p + 1) * d])
                {
                    *s += f64::from(v);
                }
            }
            for j in 0..k {
                if counts[j] > 0 {
                    for t in 0..d {
                        centres[j * d + t] = (sums[j * d + t] / counts[j] as f64) as f32;
                    }
                }
            }
        }
        if fit.len() < m {
            assign = pos
                .iter()
                .map(|&p| nearest(&vecs[p * d..(p + 1) * d], &centres))
                .collect();
        }
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (&a, &p) in assign.iter().zip(pos) {
            groups[a].push(p);
        }
        groups.retain(|g| !g.is_empty());
        if groups.len() < 2 {
            return pos.chunks(m.div_ceil(k)).map(<[usize]>::to_vec).collect();
        }
        groups
    };
    // Post-order construction with an explicit stack: each frame holds a node's positions, its
    // child groups still to build and the router indices of the children built so far.
    struct Frame {
        pos: Vec<usize>,
        pending: Vec<Vec<usize>>,
        built: Vec<u32>,
    }
    let all: Vec<usize> = (0..members.len()).collect();
    let mut stack: Vec<Frame> = Vec::new();
    let mut salt = 0u64;
    let mut push_node = |pos: Vec<usize>,
                         stack: &mut Vec<Frame>,
                         routers: &mut Vec<(Vec<Child>, RouterVec)>,
                         row: &mut [f32]|
     -> Option<u32> {
        if pos.len() <= b {
            let ids: Vec<u32> = pos.iter().map(|&p| members[p]).collect();
            let v = vector_of(base, &ids, centroid(&pos), rv, row);
            routers.push((ids.iter().map(|&x| Child::Point(x)).collect(), v));
            return Some((routers.len() - 1) as u32);
        }
        salt += 1;
        let mut pending = split(&pos, salt);
        pending.reverse();
        stack.push(Frame {
            pos,
            pending,
            built: Vec::new(),
        });
        None
    };
    let mut done: Option<u32> = push_node(all, &mut stack, &mut routers, row);
    while let Some(top) = stack.last_mut() {
        if let Some(r) = done.take() {
            top.built.push(r);
        }
        if let Some(g) = top.pending.pop() {
            done = push_node(g, &mut stack, &mut routers, row);
            continue;
        }
        let f = stack.pop().expect("non-empty");
        let ids: Vec<u32> = f.pos.iter().map(|&p| members[p]).collect();
        let v = vector_of(base, &ids, centroid(&f.pos), rv, row);
        routers.push((f.built.iter().map(|&r| Child::Router(r)).collect(), v));
        done = Some((routers.len() - 1) as u32);
    }
    routers
}

/// The routers of label `l`: a tree over its members in the shared order (`tree`; `leaf` =
/// every point's leaf in that order), or over one entry per source component of the subgraph
/// its members induce in `g0` (`connector`; `local` is all `u32::MAX` on entry and on return).
#[expect(
    clippy::too_many_arguments,
    reason = "one label's share of the builder's inputs"
)]
fn label_routers<T: Element>(
    construction: Construction,
    g0: &Graph,
    base: &Matrix<T>,
    members: &[u32],
    leaf: &[u32],
    b: usize,
    rv: RouterVector,
    kmeans: Option<(usize, usize, u64)>,
    row: &mut [f32],
    local: &mut [u32],
) -> (Vec<(Vec<Child>, RouterVec)>, Vec<u32>) {
    match construction {
        Construction::Tree | Construction::Pairs => match kmeans {
            Some((iters, sample, seed)) if members.len() > b => {
                let d = base.ncols();
                let mut vecs = vec![0f32; members.len() * d];
                for (i, &x) in members.iter().enumerate() {
                    row_f32(base.row(x as usize), &mut vecs[i * d..(i + 1) * d]);
                }
                (
                    kmeans_tree(base, members, &vecs, b, rv, iters, sample, seed, row),
                    Vec::new(),
                )
            }
            _ => {
                let mut m: Vec<u32> = members.to_vec();
                m.sort_unstable_by_key(|&x| (leaf[x as usize], x));
                (tree_over(base, &m, b, rv, row), Vec::new())
            }
        },
        Construction::Connector => {
            for (j, &x) in members.iter().enumerate() {
                local[x as usize] = j as u32;
            }
            let (comp, ncomp) = scc(g0, members, local);
            let mut has_in = vec![false; ncomp];
            for (j, &x) in members.iter().enumerate() {
                for &w in g0.neighbors(x) {
                    let lw = local.get(w as usize).copied().unwrap_or(u32::MAX);
                    if lw != u32::MAX && comp[lw as usize] != comp[j] {
                        has_in[comp[lw as usize] as usize] = true;
                    }
                }
            }
            for &x in members {
                local[x as usize] = u32::MAX;
            }
            let mut sources: Vec<Vec<u32>> = vec![Vec::new(); ncomp];
            for (j, &x) in members.iter().enumerate() {
                if !has_in[comp[j] as usize] {
                    sources[comp[j] as usize].push(x);
                }
            }
            let reps: Vec<u32> = sources
                .iter()
                .filter(|m| !m.is_empty())
                .map(|m| {
                    let c = centroid_of(base, m, row);
                    nearest_to(base, m, &c, row)
                })
                .collect();
            if reps.len() <= 1 {
                (Vec::new(), reps)
            } else {
                (tree_over(base, &reps, b, rv, row), Vec::new())
            }
        }
    }
}

/// The augmented graph and its per-label starts.
struct Assembled {
    graph: Graph,
    /// (label, start) pairs, deduplicated in order.
    pairs: Vec<(u32, u32)>,
    /// `pairs`: the root of every directory slot's tree (`u32::MAX`: none).
    pair_roots: Vec<u32>,
    labels_with_routers: usize,
    routers: usize,
    router_edges: u64,
}

/// The base graph's nodes (rows unchanged, ids kept), then the routers label by label (each
/// label's root last). Per-label starts: `tree`: the root (and the base starts with
/// `keep_starts`); `connector`: the base starts, then the root or the single source entry;
/// labels without routers keep their base starts.
#[expect(clippy::too_many_arguments, reason = "the assembly's inputs")]
fn assemble<T: Element>(
    g0: &Graph,
    base: &Matrix<T>,
    n_labels: usize,
    st0: &Starts,
    per_label: &[LabelRouters],
    construction: Construction,
    keep_starts: bool,
    b: usize,
    n_slots: usize,
) -> Result<Assembled> {
    let n = g0.n_data();
    let n0 = g0.n_nodes();
    let d = base.ncols();
    let n_routers: usize = per_label.iter().map(|r| r.routers.len()).sum();
    let n_aux = g0.n_aux() + n_routers;
    let mut vecs: Vec<T> = Vec::with_capacity(n_aux * d);
    let mut kinds: Vec<AuxKind> = Vec::with_capacity(n_aux);
    for id in n as u32..n0 as u32 {
        let v = g0
            .aux_vector::<T>(id)
            .context("the base graph's auxiliary vectors are missing")?;
        vecs.extend_from_slice(v);
        kinds.push(g0.aux_kind(id).unwrap_or(AuxKind::Frozen));
    }
    kinds.resize(n_aux, AuxKind::Router);
    for lr in per_label {
        for (_, v) in &lr.routers {
            match v {
                RouterVec::Centroid(c) => vecs.extend(from_f32::<T>(c)),
                RouterVec::Point(x) => vecs.extend_from_slice(base.row(*x as usize)),
            }
        }
    }
    let aux = AuxNodes::new(kinds, &Matrix::from_slice(&vecs, n_aux, d))?;
    let mut g = Graph::new(n, g0.r_max().max(b), aux)?;
    for v in 0..n0 as u32 {
        g.set_neighbors(v, g0.neighbors(v))?;
    }
    g.set_starts(st0.global().to_vec())?;
    let mut next = n0 as u32;
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    let mut with_routers = vec![false; n_labels];
    let mut pair_roots = vec![u32::MAX; n_slots];
    let mut router_edges = 0u64;
    let mut nbr: Vec<u32> = Vec::new();
    let base_starts = |l: u32| -> &[u32] {
        if (l as usize) < st0.n_labels() {
            st0.label_starts(l)
        } else {
            &[]
        }
    };
    for lr in per_label {
        let first = next;
        for (children, _) in &lr.routers {
            nbr.clear();
            nbr.extend(children.iter().map(|c| match *c {
                Child::Point(x) => x,
                Child::Router(r) => first + r,
            }));
            router_edges += nbr.len() as u64;
            g.set_neighbors(next, &nbr)?;
            next += 1;
        }
        let root = (!lr.routers.is_empty()).then(|| next - 1);
        let l = lr.label;
        match construction {
            Construction::Tree => {
                if let Some(r) = root {
                    with_routers[l as usize] = true;
                    pairs.push((l, r));
                    if keep_starts {
                        pairs.extend(base_starts(l).iter().map(|&s| (l, s)));
                    }
                }
            }
            Construction::Connector => {
                with_routers[l as usize] = true;
                pairs.extend(base_starts(l).iter().map(|&s| (l, s)));
                if let Some(r) = root {
                    pairs.push((l, r));
                }
                pairs.extend(lr.point_starts.iter().map(|&s| (l, s)));
            }
            Construction::Pairs => {
                if let Some(r) = root {
                    pair_roots[l as usize] = r;
                }
            }
        }
    }
    ensure!(next as usize == n + n_aux, "router count mismatch");
    for l in 0..st0.n_labels().min(n_labels) as u32 {
        if !with_routers[l as usize] {
            pairs.extend(st0.label_starts(l).iter().map(|&s| (l, s)));
        }
    }
    let mut seen = std::collections::HashSet::new();
    pairs.retain(|p| seen.insert(*p));
    Ok(Assembled {
        graph: g,
        pairs,
        pair_roots,
        labels_with_routers: with_routers.iter().filter(|&&x| x).count(),
        routers: n_routers,
        router_edges,
    })
}

fn run_typed<T: Element>(args: &SteinerArgs) -> Result<()> {
    let t0 = Instant::now();
    std::fs::create_dir_all(&args.out_dir)?;
    let base_file = crate::run::data::base_file(&args.dataset)?;
    let base = Matrix::<T>::read_bin(&base_file)?;
    let labels = LabelStore::read(&args.dataset.join("labels.flab"))?;
    let n = base.nrows();
    ensure!(labels.n_points() == n, "labels and base differ in size");
    let lg = load_graph(&args.graph, None, T::DTYPE)?;
    let g0 = &lg.graph;
    ensure!(
        g0.n_data() == n,
        "the base graph has {} data points, the base {n}",
        g0.n_data()
    );
    let st0 = Starts::read(&args.starts)?;
    if st0.n_labels() > 0 {
        st0.check_dictionary(&labels)?;
    }
    let b = args.block.max(2);
    let max_size = parse_size(&args.max_size, n)?;
    let px = match (args.construction, &args.pairs) {
        (Construction::Pairs, Some(p)) => {
            let px = PairIndex::read(p)?;
            ensure!(
                px.dict_hash() == labels.dict_hash() && px.n_points() == n,
                "the combination postings belong to another label store"
            );
            Some(px)
        }
        (Construction::Pairs, None) => anyhow::bail!("--construction pairs needs --pairs"),
        _ => None,
    };
    // Labels, or for `pairs` the directory slots of the stored pairs.
    let size_of = |x: u32| match &px {
        Some(px) => px.slot_list(x as usize).len(),
        None => labels.posting_len(x),
    };
    let n_items = px.as_ref().map_or(labels.n_labels(), PairIndex::n_pairs);
    let selected: Vec<u32> = match &args.labels_file {
        Some(path) => {
            ensure!(
                args.construction == Construction::Tree,
                "--labels-file selects the labels of --construction tree"
            );
            read_label_list(path, labels.n_labels())?
        }
        None => (0..n_items as u32)
            .filter(|&x| {
                let s = size_of(x);
                s > args.min_size && s <= max_size
            })
            .collect(),
    };
    let d = base.ncols();
    let mut leaf = vec![0u32; n];
    if matches!(args.construction, Construction::Tree | Construction::Pairs)
        && args.partition == Partition::Order
    {
        let all: Vec<u32> = (0..n as u32).collect();
        rp_leaves(&base, &all, args.order_block, args.seed, &mut leaf);
    }
    let t_order = t0.elapsed().as_secs_f64();
    let per_label: Vec<LabelRouters> = par_map(
        selected.len(),
        args.threads,
        || (vec![0f32; d], vec![u32::MAX; n]),
        |i, (row, local)| {
            let l = selected[i];
            let kmeans = (args.partition == Partition::Kmeans).then_some((
                args.kmeans_iters,
                args.kmeans_sample,
                args.seed ^ u64::from(l),
            ));
            let members = match &px {
                Some(px) => px.slot_list(l as usize),
                None => labels.postings(l),
            };
            let (routers, point_starts) = label_routers(
                args.construction,
                g0,
                &base,
                members,
                &leaf,
                b,
                args.router,
                kmeans,
                row,
                local,
            );
            LabelRouters {
                label: l,
                routers,
                point_starts,
            }
        },
    );
    let t_routers = t0.elapsed().as_secs_f64();
    let a = assemble(
        g0,
        &base,
        labels.n_labels(),
        &st0,
        &per_label,
        args.construction,
        args.keep_starts,
        b,
        if px.is_some() { n_items } else { 0 },
    )?;
    let mut starts = Starts::new(&a.graph);
    starts.set_label_starts(&labels, &a.pairs)?;
    a.graph.write(&args.out_dir.join("graph.flab"))?;
    starts.write(&args.out_dir.join("starts.flab"))?;
    if let Some(px) = &px {
        write_pair_roots(&args.out_dir.join("pair_roots.flab"), px, &a.pair_roots)?;
    }
    let elem = std::mem::size_of::<T>();
    let r_max = a.graph.r_max();
    let (n_aux, router_edges) = (a.routers, a.router_edges);
    let meta = json!({
        "tool": "flab steiner",
        "dataset": args.dataset, "graph": args.graph, "starts": args.starts,
        "construction": format!("{:?}", args.construction).to_lowercase(),
        "block": b, "labels_file": args.labels_file,
        // A label list replaces the size bounds.
        "min_size": args.labels_file.is_none().then_some(args.min_size),
        "max_size": args.labels_file.is_none().then_some(max_size),
        "router": format!("{:?}", args.router).to_lowercase(), "keep_starts": args.keep_starts,
        "partition": format!("{:?}", args.partition).to_lowercase(), "kmeans_iters": args.kmeans_iters,
        "kmeans_sample": args.kmeans_sample,
        "order_block": args.order_block, "seed": args.seed, "threads": args.threads,
        "n_data": n, "base_aux": g0.n_aux(), "r_max": r_max,
        "labels_selected": selected.len(),
        "pairs": args.pairs, "pairs_with_trees": a.pair_roots.iter().filter(|&&r| r != u32::MAX).count(),
        "labels_with_routers": a.labels_with_routers,
        "routers": n_aux, "router_edges": router_edges,
        "label_start_entries": a.pairs.len(),
        "bytes": {
            "router_vectors": n_aux * d * elem,
            "router_edges_csr": router_edges * 4 + (n_aux as u64 + 1) * 8,
            "router_rows_fixed": n_aux * (r_max * 4 + 4),
            "id_backed_router_vectors": n_aux * 4,
        },
        "time_s": {"order": t_order, "routers": t_routers - t_order, "total": t0.elapsed().as_secs_f64()},
    });
    std::fs::write(
        args.out_dir.join("meta.json"),
        serde_json::to_string_pretty(&meta)? + "\n",
    )?;
    eprintln!(
        "flab steiner: {} routers ({} edges) for {} labels in {:.1} s -> {}",
        n_aux,
        router_edges,
        selected.len(),
        t0.elapsed().as_secs_f64(),
        args.out_dir.display()
    );
    Ok(())
}

/// `f32` values as the element type (`u8`: rounded and clamped).
fn from_f32<T: Element>(c: &[f32]) -> Vec<T> {
    match T::DTYPE {
        DType::U8 => {
            let v: Vec<u8> = c
                .iter()
                .map(|&x| x.round().clamp(0.0, 255.0) as u8)
                .collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        DType::F32 => bytemuck::cast_slice(c).to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::labels::LabelStoreBuilder;

    fn line_base(n: usize) -> Matrix<f32> {
        Matrix::from_slice(&(0..n).map(|i| i as f32).collect::<Vec<_>>(), n, 1)
    }

    /// The data points reached from router `r` (DFS over router edges), with multiplicity, and
    /// the largest out-degree of a router on the way.
    fn reached(g: &Graph, r: u32) -> (Vec<u32>, usize) {
        let n = g.n_data() as u32;
        let (mut out, mut fan) = (Vec::new(), 0);
        let mut stack = vec![r];
        while let Some(v) = stack.pop() {
            fan = fan.max(g.neighbors(v).len());
            for &w in g.neighbors(v) {
                if w < n {
                    out.push(w);
                } else {
                    stack.push(w);
                }
            }
        }
        out.sort_unstable();
        (out, fan)
    }

    #[test]
    fn a_tree_covers_every_member_once_with_bounded_fan_out() {
        let base = line_base(100);
        let leaves: Vec<u32> = (0..100).collect();
        let mut row = vec![0f32; 1];
        let routers = tree_over(&base, &leaves, 4, RouterVector::Centroid, &mut row);
        // 25 leaf routers, then 7, 2 and the root (last).
        assert_eq!(routers.len(), 25 + 7 + 2 + 1);
        let mut seen = vec![0u32; 100];
        let mut stack = vec![routers.len() - 1];
        while let Some(r) = stack.pop() {
            assert!(routers[r].0.len() <= 4);
            for c in &routers[r].0 {
                match *c {
                    Child::Point(x) => seen[x as usize] += 1,
                    Child::Router(j) => {
                        assert!((j as usize) < r, "children precede their parent");
                        stack.push(j as usize);
                    }
                }
            }
        }
        assert!(seen.iter().all(|&s| s == 1));
        match &routers[0].1 {
            RouterVec::Centroid(c) => assert_eq!(c, &[1.5]),
            RouterVec::Point(_) => panic!("centroid expected"),
        }
        // The medoid of {0, 1, 2, 3}: 1 and 2 are equally near 1.5; the smaller id wins.
        let routers = tree_over(&base, &leaves, 4, RouterVector::Medoid, &mut row);
        match &routers[0].1 {
            RouterVec::Point(x) => assert_eq!(*x, 1),
            RouterVec::Centroid(_) => panic!("medoid expected"),
        }
    }

    #[test]
    fn a_kmeans_tree_covers_every_member_once_and_splits_clusters_first() {
        // Two clusters on a line, 0..50 and 1000..1050, members interleaved by id.
        let n = 100;
        let xs: Vec<f32> = (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    (i / 2) as f32
                } else {
                    1000.0 + (i / 2) as f32
                }
            })
            .collect();
        let base = Matrix::from_slice(&xs, n, 1);
        let members: Vec<u32> = (0..n as u32).collect();
        let mut row = vec![0f32; 1];
        let routers = kmeans_tree(
            &base,
            &members,
            &xs,
            4,
            RouterVector::Centroid,
            4,
            0,
            7,
            &mut row,
        );
        // Fitting on a strided sample of 2 x k members per split also separates the clusters.
        let sampled = kmeans_tree(
            &base,
            &members,
            &xs,
            4,
            RouterVector::Centroid,
            4,
            2,
            7,
            &mut row,
        );
        let top = sampled.last().expect("a root");
        for c in &top.0 {
            let Child::Router(j) = *c else {
                panic!("a router child expected")
            };
            let RouterVec::Centroid(v) = &sampled[j as usize].1 else {
                panic!("centroid expected")
            };
            assert!(v[0] < 60.0 || v[0] > 990.0, "{v:?}");
        }
        let root = routers.len() - 1;
        let mut seen = vec![0u32; n];
        let mut stack = vec![(root, None::<bool>)];
        while let Some((r, side)) = stack.pop() {
            assert!(routers[r].0.len() <= 4);
            for c in &routers[r].0 {
                match *c {
                    Child::Point(x) => {
                        seen[x as usize] += 1;
                        // Below the root's children every point is on its child's side.
                        if let Some(high) = side {
                            assert_eq!(xs[x as usize] >= 1000.0, high, "point {x}");
                        }
                    }
                    Child::Router(j) => {
                        assert!((j as usize) < r, "children precede their parent");
                        let s = side.or_else(|| match &routers[j as usize].1 {
                            RouterVec::Centroid(c) => Some(c[0] >= 500.0),
                            RouterVec::Point(_) => None,
                        });
                        stack.push((j as usize, s));
                    }
                }
            }
        }
        assert!(seen.iter().all(|&s| s == 1));
        match &routers[root].1 {
            RouterVec::Centroid(c) => assert!((c[0] - 524.5).abs() < 1e-3, "{c:?}"),
            RouterVec::Point(_) => panic!("centroid expected"),
        }
    }

    /// 12 points on a line (edges i → i ± 1) and a frozen start (id 12, x = 5.5, edges to 5
    /// and 6); label `a=1` on {0, 3, 6, 9, 11} (no two adjacent), `b=1` on {0, 1, 2} (a path);
    /// base starts: `a=1` → 3, `b=1` → 0.
    fn toy() -> (Matrix<f32>, Graph, LabelStore, Starts) {
        let n = 12;
        let base = line_base(n);
        let aux = AuxNodes::new(vec![AuxKind::Frozen], &Matrix::from_slice(&[5.5f32], 1, 1));
        let mut g = Graph::new(n, 4, aux.unwrap()).unwrap();
        for i in 0..n as u32 {
            let nb: Vec<u32> = [i.checked_sub(1), (i + 1 < n as u32).then_some(i + 1)]
                .into_iter()
                .flatten()
                .collect();
            g.set_neighbors(i, &nb).unwrap();
        }
        g.set_neighbors(12, &[5, 6]).unwrap();
        g.set_starts(vec![12]).unwrap();
        let mut b = LabelStoreBuilder::new();
        for i in 0..n {
            let a = if [0, 3, 6, 9, 11].contains(&i) {
                "a=1"
            } else {
                "a=2"
            };
            let c = if i < 3 { "b=1" } else { "b=2" };
            b.push_point([("a", a), ("b", c)]).unwrap();
        }
        let labels = b.finish().unwrap();
        let id = |s: &str| labels.name_index()[s];
        let mut st = Starts::new(&g);
        st.set_label_starts(&labels, &[(id("a=1"), 3), (id("b=1"), 0)])
            .unwrap();
        (base, g, labels, st)
    }

    fn augment(construction: Construction, keep_starts: bool) -> (Graph, LabelStore, Assembled) {
        let (base, g0, labels, st0) = toy();
        let leaf: Vec<u32> = (0..12).collect();
        let (mut row, mut local) = (vec![0f32; 1], vec![u32::MAX; 12]);
        let per_label: Vec<LabelRouters> = ["a=1", "b=1"]
            .iter()
            .map(|s| {
                let l = labels.name_index()[*s];
                let (routers, point_starts) = label_routers(
                    construction,
                    &g0,
                    &base,
                    labels.postings(l),
                    &leaf,
                    2,
                    RouterVector::Centroid,
                    None,
                    &mut row,
                    &mut local,
                );
                assert!(local.iter().all(|&x| x == u32::MAX));
                LabelRouters {
                    label: l,
                    routers,
                    point_starts,
                }
            })
            .collect();
        let a = assemble(
            &g0,
            &base,
            labels.n_labels(),
            &st0,
            &per_label,
            construction,
            keep_starts,
            2,
            0,
        )
        .unwrap();
        // The base graph's rows are unchanged and none of its nodes links to a router.
        for v in 0..13 {
            assert_eq!(a.graph.neighbors(v), g0.neighbors(v));
            assert!(a.graph.neighbors(v).iter().all(|&w| w < 13));
        }
        assert_eq!(a.graph.n_nodes(), 13 + a.routers);
        (g0, labels, a)
    }

    fn starts_of(a: &Assembled, l: u32) -> Vec<u32> {
        a.pairs.iter().filter(|p| p.0 == l).map(|p| p.1).collect()
    }

    #[test]
    fn tree_roots_reach_exactly_their_labels_members() {
        let (_, labels, a) = augment(Construction::Tree, false);
        let id = |s: &str| labels.name_index()[s];
        // a=1: 5 members under 3 leaf routers, then 2 routers and the root; b=1: 2 + 1.
        assert_eq!(a.routers, 6 + 3);
        // A tree over s members with R routers has s + R − 1 edges, one to each member and to
        // each router below the root (`flab select --tree-bytes compact` counts them so).
        assert_eq!(a.router_edges, (5 + 6 - 1) + (3 + 3 - 1));
        for (name, members) in [("a=1", vec![0, 3, 6, 9, 11]), ("b=1", vec![0, 1, 2])] {
            let s = starts_of(&a, id(name));
            assert_eq!(s.len(), 1, "{name}: the root replaces the base starts");
            assert!(s[0] >= 13);
            let (got, fan) = reached(&a.graph, s[0]);
            assert_eq!(got, members, "{name}");
            assert!(fan <= 2);
        }
        let (_, _, kept) = augment(Construction::Tree, true);
        assert_eq!(starts_of(&kept, id("a=1"))[1..], [3]);
        assert_eq!(a.graph.starts(), &[12]);
    }

    #[test]
    fn pair_trees_reach_exactly_their_joint_members_and_keep_the_label_starts() {
        let (base, g0, labels, st0) = toy();
        let leaf: Vec<u32> = (0..12).collect();
        let (mut row, mut local) = (vec![0f32; 1], vec![u32::MAX; 12]);
        // Two pseudo-labels (directory slots 0 and 1 of a pairs file with 3 stored pairs).
        let joints: [(u32, Vec<u32>); 2] = [(0, vec![0, 3, 6]), (2, vec![9, 11])];
        let per_slot: Vec<LabelRouters> = joints
            .iter()
            .map(|(slot, members)| {
                let (routers, point_starts) = label_routers(
                    Construction::Pairs,
                    &g0,
                    &base,
                    members,
                    &leaf,
                    2,
                    RouterVector::Centroid,
                    None,
                    &mut row,
                    &mut local,
                );
                LabelRouters {
                    label: *slot,
                    routers,
                    point_starts,
                }
            })
            .collect();
        let a = assemble(
            &g0,
            &base,
            labels.n_labels(),
            &st0,
            &per_slot,
            Construction::Pairs,
            false,
            2,
            3,
        )
        .unwrap();
        assert_eq!(a.pair_roots[1], u32::MAX);
        // s + R − 1 edges per tree (3 and 2 members).
        assert_eq!(a.router_edges, (3 + 2 + a.routers - 2) as u64);
        for (slot, members) in &joints {
            let r = a.pair_roots[*slot as usize];
            assert!(r >= 13);
            assert_eq!(&reached(&a.graph, r).0, members);
        }
        // The label starts stay the base table's.
        for l in 0..labels.n_labels() as u32 {
            assert_eq!(starts_of(&a, l), st0.label_starts(l));
        }
    }

    #[test]
    fn a_label_list_is_sorted_deduplicated_and_range_checked() {
        let text = "7\n# a comment\n2 # trailing\n\n  7\n0\n";
        assert_eq!(parse_label_list(text, 8, "list").unwrap(), vec![0, 2, 7]);
        let err = parse_label_list("3\n8\n", 8, "list").unwrap_err();
        assert!(
            err.to_string().contains("list:2: label 8 out of range"),
            "{err}"
        );
        assert!(parse_label_list("x\n", 8, "list").is_err());
    }

    #[test]
    fn connectors_add_one_entry_per_source_component() {
        let (_, labels, a) = augment(Construction::Connector, false);
        let id = |s: &str| labels.name_index()[s];
        // a=1 induces 5 singleton components: a tree over all 5 (as entries) after the base
        // start 3.
        let s = starts_of(&a, id("a=1"));
        assert_eq!(s.len(), 2);
        assert_eq!(s[0], 3);
        assert_eq!(reached(&a.graph, s[1]).0, [0, 3, 6, 9, 11]);
        // b=1 is one strongly connected path: no routers, its entry (the member nearest the
        // centroid, 1) follows the base start 0.
        assert_eq!(starts_of(&a, id("b=1")), [0, 1]);
        assert_eq!(a.routers, 3 + 2 + 1);
    }
}
