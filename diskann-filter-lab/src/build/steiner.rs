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
//!   gets one router over all of them (a posting scan expressed as graph expansion).
//! - `connector`: for every selected label, one entry per source strongly connected component
//!   of the subgraph its members induce (the member nearest to the component's centroid);
//!   with several such components, a b-ary tree of routers over these entries whose root
//!   becomes a start of the label. The label's previous starts are kept.
//!
//! Router vectors: the centroid of the router's data-point descendants (`centroid`), or the
//! descendant nearest to that centroid (`medoid`: a member's vector, as for a router that would
//! only store the member's id).

use std::{path::PathBuf, time::Instant};

use anyhow::{ensure, Context, Result};
use clap::{Args, ValueEnum};
use serde_json::json;

use crate::{
    diag::{row_f32, rp_leaves, scc},
    graph::{AuxKind, AuxNodes, Graph},
    labels::LabelStore,
    parallel::par_map,
    run::data::{dataset_dtype, load_graph},
    starts::Starts,
    vectors::{DType, Element, Matrix},
};

/// Construction of `flab steiner`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Construction {
    /// Per-label b-ary trees over the members in one shared spatial order.
    Tree,
    /// Per-label routers over one entry per source component of the induced subgraph.
    Connector,
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
    /// Fan-out `b` of the routers (leaf chunks of `b` members).
    #[arg(long, default_value_t = 32)]
    pub block: usize,
    /// Only labels with more than this many members get routers.
    #[arg(long, default_value_t = 4)]
    pub min_size: usize,
    /// Only labels with at most this many members get routers: a number or `n/<k>`.
    #[arg(long, default_value = "n/100")]
    pub max_size: String,
    /// Router vectors.
    #[arg(long, value_enum, default_value = "centroid")]
    pub router: RouterVector,
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

fn parse_size(s: &str, n: usize) -> Result<usize> {
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
    row: &mut [f32],
    local: &mut [u32],
) -> (Vec<(Vec<Child>, RouterVec)>, Vec<u32>) {
    match construction {
        Construction::Tree => {
            let mut m: Vec<u32> = members.to_vec();
            m.sort_unstable_by_key(|&x| (leaf[x as usize], x));
            (tree_over(base, &m, b, rv, row), Vec::new())
        }
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
    let selected: Vec<u32> = (0..labels.n_labels() as u32)
        .filter(|&l| {
            let s = labels.posting_len(l);
            s > args.min_size && s <= max_size
        })
        .collect();
    let d = base.ncols();
    let mut leaf = vec![0u32; n];
    if args.construction == Construction::Tree {
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
            let (routers, point_starts) = label_routers(
                args.construction,
                g0,
                &base,
                labels.postings(l),
                &leaf,
                b,
                args.router,
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
    )?;
    let mut starts = Starts::new(&a.graph);
    starts.set_label_starts(&labels, &a.pairs)?;
    a.graph.write(&args.out_dir.join("graph.flab"))?;
    starts.write(&args.out_dir.join("starts.flab"))?;
    let elem = std::mem::size_of::<T>();
    let r_max = a.graph.r_max();
    let (n_aux, router_edges) = (a.routers, a.router_edges);
    let meta = json!({
        "tool": "flab steiner",
        "dataset": args.dataset, "graph": args.graph, "starts": args.starts,
        "construction": format!("{:?}", args.construction).to_lowercase(),
        "block": b, "min_size": args.min_size, "max_size": max_size,
        "router": format!("{:?}", args.router).to_lowercase(), "keep_starts": args.keep_starts,
        "order_block": args.order_block, "seed": args.seed, "threads": args.threads,
        "n_data": n, "base_aux": g0.n_aux(), "r_max": r_max,
        "labels_selected": selected.len(),
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
