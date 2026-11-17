/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Navigation diagnostics of graph-only filtered search (`flab diag`, untimed).
//!
//! Per panel row: the exact answers, the points of the binding clause c\* (the clause with the
//! smallest summed posting size), the reachability of the answers under the kernel's traversal
//! models, the classification of the answers a search missed, searches with oracle and
//! per-component entries, the work a traced search spent before it found the answers, and
//! block statistics of one shared spatial order.
//!
//! Traversal models: an unbounded search over an admission set `A` from the query's starts,
//! which are always expanded (as with `start_admission = "force"`):
//! - `strict1` / `strict2`: `A` = the full matches; one hop, or with the kernel's capped
//!   second hop (through the `h` rejected neighbours closest to the query of each expanded
//!   node);
//! - `proj1` / `proj2`: `A` = the points of c\* (admission `projection(c*)`);
//! - relaxed: every node, i.e. reachability from the graph's global starts.
//!
//! Auxiliary nodes are admitted in every model (the kernel navigates them unconditionally) but
//! are skipped by the second hop, as in the kernel. The unbounded models ignore the order in
//! which the kernel marks rejected neighbours visited, so they slightly over-approximate the
//! second hop: an answer they do not reach is unreachable for the kernel.
//!
//! A missed answer is *unreachable* (not reached by the policy's model), *cut* (strict policy:
//! reached when navigating on c\* alone but not on the full matches), *collection loss* (the
//! search offered it) or *undiscovered* (reachable, never offered).

use std::{fmt::Write as _, path::PathBuf, time::Instant};

use anyhow::{bail, ensure, Context, Result};
use clap::Args;
use serde_json::json;

use crate::{
    bitsets::LabelBitsets,
    graph::Graph,
    kernel::{search::Plan, trace::Trace, Index, Kernel, Query, Searcher},
    labels::LabelStore,
    parallel::par_map,
    predicate::Cnf,
    run::data::{dataset_dtype, load_graph, Dataset, Rows},
    vectors::{DType, Element, Matrix},
};

/// Arguments of `flab diag`.
#[derive(Debug, Args)]
pub struct DiagArgs {
    /// Dataset directory (base vectors, `labels.flab`, the panel's query sets).
    #[arg(long)]
    pub dataset: PathBuf,
    /// Graph directory (`graph.flab`, `starts.flab`) or `graph.flab`.
    #[arg(long)]
    pub graph: PathBuf,
    /// Start table (default: the graph directory's `starts.flab`).
    #[arg(long)]
    pub starts: Option<PathBuf>,
    /// Panel TSV (columns `qset`, `qidx`, `band`, `dilution`, `stratum`, `traffic`).
    #[arg(long)]
    pub panel: PathBuf,
    /// Panel split to keep.
    #[arg(long, default_value = "discovery")]
    pub split: String,
    /// Answers per query.
    #[arg(long, default_value_t = 10)]
    pub k: usize,
    /// Policies: comma list of `p5hc_nb`, `p5hg_bm`, `kha`.
    #[arg(long, default_value = "p5hc_nb,p5hg_bm,kha")]
    pub policies: String,
    /// List sizes of the searches (comma list).
    #[arg(long, default_value = "40,160,640")]
    pub efforts: String,
    /// List size of the traced searches (work decomposition); 0: none.
    #[arg(long, default_value_t = 0)]
    pub trace_effort: usize,
    /// Keep only rows of these panel bands (comma list; `traffic` = traffic rows).
    #[arg(long)]
    pub bands: Option<String>,
    /// At most this many rows (after filtering; every `⌈n / max⌉`-th row).
    #[arg(long)]
    pub max_rows: Option<usize>,
    /// Reachability models and components only for rows whose binding-clause union has at
    /// most this many points.
    #[arg(long, default_value_t = 300_000)]
    pub bfs_cap: usize,
    /// Second-hop cap `h` of the traversal models.
    #[arg(long, default_value_t = 32)]
    pub two_hop: usize,
    /// Block size `b` of the shared spatial order (random-projection tree leaves).
    #[arg(long, default_value_t = 64)]
    pub block: usize,
    /// At most this many per-component entries per search (`scc` variant).
    #[arg(long, default_value_t = 256)]
    pub max_entries: usize,
    /// Worker threads (rows in parallel).
    #[arg(short = 't', long, default_value_t = 1)]
    pub threads: usize,
    /// Seed of the spatial order.
    #[arg(long, default_value_t = 7)]
    pub seed: u64,
    /// Output directory (`rows.tsv`, `runs.tsv`, `labels.tsv`, `diag.json`).
    #[arg(long)]
    pub out: PathBuf,
}

/// Runs `flab diag`.
pub fn run(args: &DiagArgs) -> Result<()> {
    match dataset_dtype(&args.dataset)? {
        DType::U8 => run_typed::<u8>(args),
        DType::F32 => run_typed::<f32>(args),
    }
}

/// The kernel configuration of a policy (B2b's definitions).
pub fn policy_kernel(name: &str) -> Result<Kernel> {
    let kv: &[(&str, &str)] = match name {
        "p5hc_nb" => &[
            ("starts", "label(c*)+global"),
            ("start_admission", "force"),
            ("admission", "projection(c*)"),
            ("laziness", "eager"),
            ("collector", "heap"),
            ("terminator", "exhaust"),
            ("expander", "two_hop(32)"),
            ("predicate", "bitmap"),
            ("materializer", "dense_nav"),
        ],
        "p5hg_bm" => &[
            ("starts", "label(c*)+global"),
            ("start_admission", "force"),
            ("admission", "strict"),
            ("laziness", "eager"),
            ("collector", "final_beam"),
            ("terminator", "exhaust"),
            ("expander", "two_hop(32)"),
            ("predicate", "bitmap"),
            ("materializer", "dense_static"),
        ],
        "kha" => &[
            ("starts", "label(c*)+global"),
            ("collector", "heap"),
            ("terminator", "adaptive_l(1000, 16)"),
            ("laziness", "distance_first"),
        ],
        _ => bail!("unknown policy {name:?} (p5hc_nb, p5hg_bm, kha)"),
    };
    let mut k = Kernel::default();
    for (key, value) in kv {
        k.set(key, value)?;
    }
    k.validate()?;
    Ok(k)
}

fn parse_list(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(|x| {
            x.trim()
                .parse::<usize>()
                .with_context(|| format!("bad number {x:?}"))
        })
        .collect()
}

/// The panel fields kept per row.
#[derive(Debug, Clone)]
struct RowMeta {
    qset: String,
    qidx: usize,
    band: String,
    dilution: String,
    stratum: String,
    traffic: String,
}

fn panel_meta(rows: &Rows) -> Result<Vec<RowMeta>> {
    let header = rows
        .panel_header
        .as_deref()
        .context("the panel has no header")?;
    let cols: Vec<&str> = header.split('\t').collect();
    let col = |n: &str| cols.iter().position(|&c| c == n);
    let (b, d, s, t) = (col("band"), col("dilution"), col("stratum"), col("traffic"));
    let get = |f: &[&str], c: Option<usize>| c.map_or_else(String::new, |c| f[c].to_owned());
    Ok(rows
        .rows
        .iter()
        .zip(&rows.panel_lines)
        .map(|((q, i), line)| {
            let f: Vec<&str> = line.split('\t').collect();
            RowMeta {
                qset: q.clone(),
                qidx: *i,
                band: get(&f, b),
                dilution: get(&f, d),
                stratum: get(&f, s),
                traffic: get(&f, t),
            }
        })
        .collect())
}

#[inline(always)]
fn bit(b: &[u64], v: u32) -> bool {
    b[(v >> 6) as usize] >> (v & 63) & 1 == 1
}

#[inline(always)]
fn set_bit(b: &mut [u64], v: u32) -> bool {
    let (w, m) = ((v >> 6) as usize, 1u64 << (v & 63));
    let was = b[w] & m != 0;
    b[w] |= m;
    was
}

#[inline(always)]
fn clear_bit(b: &mut [u64], v: u32) {
    b[(v >> 6) as usize] &= !(1u64 << (v & 63));
}

/// A row of `T` as `f32`.
pub fn row_f32<T: Element>(row: &[T], out: &mut [f32]) {
    match T::DTYPE {
        DType::U8 => {
            let r: &[u8] = bytemuck::cast_slice(row);
            for (o, &x) in out.iter_mut().zip(r) {
                *o = f32::from(x);
            }
        }
        DType::F32 => out.copy_from_slice(bytemuck::cast_slice(row)),
    }
}

/// SplitMix64.
pub struct SplitMix(pub u64);

impl SplitMix {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// Leaves of a random-projection tree over `ids` (median splits along the direction between
/// two random points of the node, until at most `b` points): the leaf of every listed point,
/// numbered in depth-first order (the leaves of one subtree are contiguous), and the number
/// of leaves. `leaf` is indexed by point id.
pub fn rp_leaves<T: Element>(
    base: &Matrix<T>,
    ids: &[u32],
    b: usize,
    seed: u64,
    leaf: &mut [u32],
) -> usize {
    let d = base.ncols();
    let mut perm = ids.to_vec();
    let mut next = 0u32;
    let mut stack = vec![(0usize, perm.len())];
    let mut rng = SplitMix(seed);
    let (mut va, mut vc, mut dir) = (vec![0f32; d], vec![0f32; d], vec![0f32; d]);
    let mut row = vec![0f32; d];
    let mut proj: Vec<(f32, u32)> = Vec::new();
    while let Some((lo, hi)) = stack.pop() {
        let m = hi - lo;
        if m <= b.max(1) {
            for &p in &perm[lo..hi] {
                leaf[p as usize] = next;
            }
            next += 1;
            continue;
        }
        let a = perm[lo + (rng.next_u64() % m as u64) as usize];
        let c = perm[lo + (rng.next_u64() % m as u64) as usize];
        row_f32(base.row(a as usize), &mut va);
        row_f32(base.row(c as usize), &mut vc);
        let mut norm = 0f32;
        for j in 0..d {
            dir[j] = va[j] - vc[j];
            norm += dir[j] * dir[j];
        }
        if norm == 0.0 {
            for x in dir.iter_mut() {
                *x = (rng.next_u64() % 2001) as f32 / 1000.0 - 1.0;
            }
        }
        proj.clear();
        for &p in &perm[lo..hi] {
            row_f32(base.row(p as usize), &mut row);
            let s: f32 = row.iter().zip(&dir).map(|(x, y)| x * y).sum();
            proj.push((s, p));
        }
        let mid = m / 2;
        proj.select_nth_unstable_by(mid, |x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
        for (i, &(_, p)) in proj.iter().enumerate() {
            perm[lo + i] = p;
        }
        stack.push((lo + mid, hi));
        stack.push((lo, lo + mid));
    }
    next as usize
}

/// Strongly connected components of the subgraph of `graph` induced by `nodes`
/// (`local[v]` = the position of `v` in `nodes`, `u32::MAX` outside): the component of every
/// listed node and the number of components (iterative Tarjan).
fn scc(graph: &Graph, nodes: &[u32], local: &[u32]) -> (Vec<u32>, usize) {
    const NONE: u32 = u32::MAX;
    let n = nodes.len();
    let mut index = vec![NONE; n];
    let mut low = vec![0u32; n];
    let mut on = vec![false; n];
    let mut comp = vec![NONE; n];
    let mut stack: Vec<u32> = Vec::new();
    let mut call: Vec<(u32, usize)> = Vec::new();
    let (mut next, mut ncomp) = (0u32, 0usize);
    for root in 0..n as u32 {
        if index[root as usize] != NONE {
            continue;
        }
        call.push((root, 0));
        index[root as usize] = next;
        low[root as usize] = next;
        next += 1;
        stack.push(root);
        on[root as usize] = true;
        while let Some(&mut (v, ref mut pos)) = call.last_mut() {
            let nb = graph.neighbors(nodes[v as usize]);
            let mut pushed = false;
            while *pos < nb.len() {
                let w = nb[*pos];
                *pos += 1;
                let lw = if (w as usize) < local.len() {
                    local[w as usize]
                } else {
                    NONE
                };
                if lw == NONE {
                    continue;
                }
                if index[lw as usize] == NONE {
                    index[lw as usize] = next;
                    low[lw as usize] = next;
                    next += 1;
                    stack.push(lw);
                    on[lw as usize] = true;
                    call.push((lw, 0));
                    pushed = true;
                    break;
                } else if on[lw as usize] {
                    low[v as usize] = low[v as usize].min(index[lw as usize]);
                }
            }
            if pushed {
                continue;
            }
            call.pop();
            if let Some(&(p, _)) = call.last() {
                low[p as usize] = low[p as usize].min(low[v as usize]);
            }
            if low[v as usize] == index[v as usize] {
                while let Some(w) = stack.pop() {
                    on[w as usize] = false;
                    comp[w as usize] = ncomp as u32;
                    if w == v {
                        break;
                    }
                }
                ncomp += 1;
            }
        }
    }
    (comp, ncomp)
}

/// Shared read-only state.
struct Ctx<'a, T: Element> {
    args: &'a DiagArgs,
    ds: &'a Dataset<T>,
    graph: &'a Graph,
    ix: Index<'a, T>,
    plans: Vec<(String, Plan)>,
    efforts: Vec<usize>,
    meta: Vec<RowMeta>,
    global_reach: Vec<u64>,
    /// Leaf of the shared spatial order of every data point.
    leaf: Vec<u32>,
    n_leaves: usize,
}

/// Per-thread scratch.
struct Scratch {
    st: Searcher,
    in_c: Vec<u64>,
    in_m: Vec<u64>,
    reach: Vec<u64>,
    reach_p: Vec<u64>,
    dist: Vec<f32>,
    dist_touched: Vec<u32>,
    local: Vec<u32>,
    queue: Vec<u32>,
    reached: Vec<u32>,
    reached_p: Vec<u32>,
    rejected: Vec<(f32, u32)>,
    leaf_cnt: Vec<u16>,
    leaf_stamp: Vec<u32>,
    leaf_touched: Vec<u32>,
    stamp: u32,
    cvec: Vec<f32>,
    rvec: Vec<f32>,
}

impl<T: Element> Ctx<'_, T> {
    fn scratch(&self) -> Scratch {
        let n_nodes = self.graph.n_nodes();
        let words = n_nodes.div_ceil(64);
        let plans: Vec<&Plan> = self.plans.iter().map(|(_, p)| p).collect();
        let d = self.ds.base.ncols();
        Scratch {
            st: Searcher::for_index(&self.ix, &plans),
            in_c: vec![0; words],
            in_m: vec![0; words],
            reach: vec![0; words],
            reach_p: vec![0; words],
            dist: vec![f32::NAN; n_nodes],
            dist_touched: Vec::new(),
            local: vec![u32::MAX; n_nodes],
            queue: Vec::new(),
            reached: Vec::new(),
            reached_p: Vec::new(),
            rejected: Vec::new(),
            leaf_cnt: vec![0; self.n_leaves],
            leaf_stamp: vec![0; self.n_leaves],
            leaf_touched: Vec::new(),
            stamp: 0,
            cvec: vec![0.0; d],
            rvec: vec![0.0; d],
        }
    }

    #[inline]
    fn dist(&self, sc: &mut Scratch, q: &[T], v: u32) -> f32 {
        let c = sc.dist[v as usize];
        if !c.is_nan() {
            return c;
        }
        let d = T::squared_l2(q, self.ix.vectors.get(v));
        sc.dist[v as usize] = d;
        sc.dist_touched.push(v);
        d
    }

    /// The traversal model's reach from `starts` over the points of c\* (`adm_is_c`) or the
    /// full matches (aux nodes always admitted), one hop or with the capped second hop: fills
    /// `reach_p` / `reached_p` (`into_p`) or `reach` / `reached` (the starts included).
    fn reach_model(
        &self,
        sc_dist: &mut Scratch,
        q: &[T],
        starts: &[u32],
        adm_is_c: bool,
        two_hop: bool,
        into_p: bool,
    ) {
        let g = self.graph;
        let n_data = g.n_data() as u32;
        let h = self.args.two_hop;
        let mut reach = std::mem::take(if into_p {
            &mut sc_dist.reach_p
        } else {
            &mut sc_dist.reach
        });
        let mut reached = std::mem::take(if into_p {
            &mut sc_dist.reached_p
        } else {
            &mut sc_dist.reached
        });
        let adm = std::mem::take(if adm_is_c {
            &mut sc_dist.in_c
        } else {
            &mut sc_dist.in_m
        });
        let mut queue = std::mem::take(&mut sc_dist.queue);
        let mut rejected = std::mem::take(&mut sc_dist.rejected);
        for &v in &reached {
            clear_bit(&mut reach, v);
        }
        reached.clear();
        queue.clear();
        let admitted = |v: u32| v >= n_data || bit(&adm, v);
        for &s in starts {
            if !set_bit(&mut reach, s) {
                reached.push(s);
                queue.push(s);
            }
        }
        while let Some(u) = queue.pop() {
            rejected.clear();
            for &w in g.neighbors(u) {
                if admitted(w) {
                    if !set_bit(&mut reach, w) {
                        reached.push(w);
                        queue.push(w);
                    }
                } else if two_hop && w < n_data {
                    rejected.push((0.0, w));
                }
            }
            if two_hop && !rejected.is_empty() {
                for r in rejected.iter_mut() {
                    r.0 = self.dist(sc_dist, q, r.1);
                }
                rejected.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                rejected.truncate(h);
                for &(_, z) in rejected.iter() {
                    for &w in g.neighbors(z) {
                        if w < n_data && bit(&adm, w) && !set_bit(&mut reach, w) {
                            reached.push(w);
                            queue.push(w);
                        }
                    }
                }
            }
        }
        if adm_is_c {
            sc_dist.in_c = adm;
        } else {
            sc_dist.in_m = adm;
        }
        sc_dist.queue = queue;
        sc_dist.rejected = rejected;
        if into_p {
            sc_dist.reach_p = reach;
            sc_dist.reached_p = reached;
        } else {
            sc_dist.reach = reach;
            sc_dist.reached = reached;
        }
    }

    /// Representatives (member nearest to the component's centroid) of the source components
    /// of the subgraph induced by `nodes` that hold no node of `covered` (bitset), largest
    /// first, at most `max`; also the component counts (components, sources, uncovered
    /// sources, largest component).
    fn source_entries(
        &self,
        sc: &mut Scratch,
        nodes: &[u32],
        covered: &[u64],
        max: usize,
    ) -> (Vec<u32>, [usize; 4]) {
        for (i, &v) in nodes.iter().enumerate() {
            sc.local[v as usize] = i as u32;
        }
        let (comp, ncomp) = scc(self.graph, nodes, &sc.local);
        let mut has_in = vec![false; ncomp];
        let mut size = vec![0usize; ncomp];
        let mut hit = vec![false; ncomp];
        for (i, &v) in nodes.iter().enumerate() {
            let c = comp[i] as usize;
            size[c] += 1;
            if bit(covered, v) {
                hit[c] = true;
            }
            for &w in self.graph.neighbors(v) {
                let lw = sc.local.get(w as usize).copied().unwrap_or(u32::MAX);
                if lw != u32::MAX && comp[lw as usize] as usize != c {
                    has_in[comp[lw as usize] as usize] = true;
                }
            }
        }
        for &v in nodes {
            sc.local[v as usize] = u32::MAX;
        }
        let sources: Vec<usize> = (0..ncomp).filter(|&c| !has_in[c]).collect();
        let mut uncovered: Vec<usize> = sources.iter().copied().filter(|&c| !hit[c]).collect();
        uncovered.sort_by_key(|&c| std::cmp::Reverse(size[c]));
        let largest = size.iter().copied().max().unwrap_or(0);
        let counts = [ncomp, sources.len(), uncovered.len(), largest];
        uncovered.truncate(max);
        // Members of the chosen components.
        let mut slot = vec![usize::MAX; ncomp];
        for (j, &c) in uncovered.iter().enumerate() {
            slot[c] = j;
        }
        let d = self.ds.base.ncols();
        let mut sums = vec![0f64; uncovered.len() * d];
        let mut members: Vec<Vec<u32>> = vec![Vec::new(); uncovered.len()];
        for (i, &v) in nodes.iter().enumerate() {
            let j = slot[comp[i] as usize];
            if j != usize::MAX {
                members[j].push(v);
                row_f32(self.ds.base.row(v as usize), &mut sc.rvec);
                for (s, &x) in sums[j * d..(j + 1) * d].iter_mut().zip(&sc.rvec) {
                    *s += f64::from(x);
                }
            }
        }
        let mut reps = Vec::with_capacity(uncovered.len());
        for (j, m) in members.iter().enumerate() {
            let cnt = m.len() as f64;
            for (o, &s) in sc.cvec.iter_mut().zip(&sums[j * d..(j + 1) * d]) {
                *o = (s / cnt) as f32;
            }
            let mut best = (f32::INFINITY, u32::MAX);
            for &v in m {
                row_f32(self.ds.base.row(v as usize), &mut sc.rvec);
                let dd: f32 = sc
                    .rvec
                    .iter()
                    .zip(&sc.cvec)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                if (dd, v) < best {
                    best = (dd, v);
                }
            }
            reps.push(best.1);
        }
        (reps, counts)
    }
}

/// Outcome of one search of a row.
struct SearchOut {
    hits: usize,
    n_dist: u64,
    n_pred: u64,
    n_expand: u64,
    n_two_hop: u64,
    n_start: u64,
    /// Missed answers by class: unreachable, cut, undiscovered, collection loss.
    miss: [usize; 4],
}

/// Clears the row's bitsets and distance cache.
fn cleanup(sc: &mut Scratch, u_c: &[u32]) {
    for &x in u_c {
        clear_bit(&mut sc.in_c, x);
        clear_bit(&mut sc.in_m, x);
    }
    for &v in &sc.dist_touched {
        sc.dist[v as usize] = f32::NAN;
    }
    sc.dist_touched.clear();
}

fn analyze_row<T: Element>(ctx: &Ctx<'_, T>, sc: &mut Scratch, i: usize) -> (String, String) {
    let args = ctx.args;
    let m = &ctx.meta[i];
    let qs = &ctx.ds.qsets[&m.qset];
    let qv = qs.vectors.row(m.qidx);
    let cnf: Cnf<'_> = qs.preds.get(m.qidx);
    let labels: &LabelStore = &ctx.ds.labels;
    let g = ctx.graph;
    let n_data = g.n_data();
    let k = args.k;
    let mut row_line = format!(
        "{}\t{}\t{}\t{}\t{}\t{}",
        m.qset, m.qidx, m.band, m.dilution, m.stratum, m.traffic
    );
    let mut runs = String::new();
    // Binding clause and its points.
    let nc = cnf.n_clauses();
    let empty = nc == 0 || cnf.clauses().any(|c| c.is_empty());
    if empty {
        let _ = write!(row_line, "\t{nc}\t0\t0");
        return (row_line + "\n", runs);
    }
    let usum = |c: &[u32]| -> u64 { c.iter().map(|&l| labels.posting_len(l) as u64).sum() };
    let cstar = (0..nc)
        .min_by_key(|&j| (usum(cnf.clause(j)), j))
        .unwrap_or(0);
    let mut u_c: Vec<u32> = Vec::new();
    for &l in cnf.clause(cstar) {
        for &p in labels.postings(l) {
            if !set_bit(&mut sc.in_c, p) {
                u_c.push(p);
            }
        }
    }
    let mut matches: Vec<(f32, u32)> = Vec::new();
    for &x in &u_c {
        if cnf.matches(labels.row(x)) {
            set_bit(&mut sc.in_m, x);
            let d = ctx.dist(sc, qv, x);
            matches.push((d, x));
        }
    }
    matches.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let n_m = matches.len();
    let _ = write!(row_line, "\t{nc}\t{}\t{n_m}", u_c.len());
    if n_m == 0 {
        cleanup(sc, &u_c);
        return (row_line + "\n", runs);
    }
    let k_eff = k.min(n_m);
    let d_k = matches[k_eff - 1].0;
    let tol = d_k + d_k.abs() * 1e-6;
    let gt: Vec<u32> = matches[..k_eff].iter().map(|&(_, x)| x).collect();
    // Dilution floor: c* members within the K-th answer's distance; blocks of the shared
    // spatial order holding them (c* points sorted by leaf, chunks of b).
    let b = args.block.max(1);
    let mut by_leaf: Vec<(u32, u32)> = u_c.iter().map(|&x| (ctx.leaf[x as usize], x)).collect();
    by_leaf.sort_unstable();
    let mut m_k = 0usize;
    let mut chunks_touched = 0usize;
    let mut leaves_touched = 0usize;
    let mut last_leaf = u32::MAX;
    let mut leaf_hit = false;
    let mut chunk_hit = false;
    for (j, &(lf, x)) in by_leaf.iter().enumerate() {
        if j % b == 0 {
            chunks_touched += usize::from(chunk_hit);
            chunk_hit = false;
        }
        if lf != last_leaf {
            leaves_touched += usize::from(leaf_hit);
            leaf_hit = false;
            last_leaf = lf;
        }
        let d = ctx.dist(sc, qv, x);
        if d <= tol {
            m_k += 1;
            chunk_hit = true;
            leaf_hit = true;
        }
    }
    chunks_touched += usize::from(chunk_hit);
    leaves_touched += usize::from(leaf_hit);
    let n_chunks = u_c.len().div_ceil(b);
    // Union summaries at the leaves of the shared order (and groups of 64 leaves): leaves
    // whose union of labels meets every clause vs leaves holding c* points vs holding a match.
    sc.leaf_touched.clear();
    for j in 0..nc {
        sc.stamp = sc.stamp.wrapping_add(1).max(1);
        let stamp = sc.stamp;
        for &l in cnf.clause(j) {
            for &p in labels.postings(l) {
                let lf = ctx.leaf[p as usize] as usize;
                if sc.leaf_stamp[lf] != stamp {
                    sc.leaf_stamp[lf] = stamp;
                    if sc.leaf_cnt[lf] == 0 {
                        sc.leaf_touched.push(lf as u32);
                    }
                    sc.leaf_cnt[lf] += 1;
                }
            }
        }
    }
    let mut sum_leaves = 0usize;
    let mut sum_groups: Vec<u32> = Vec::new();
    for &lf in &sc.leaf_touched {
        if sc.leaf_cnt[lf as usize] as usize == nc {
            sum_leaves += 1;
            sum_groups.push(lf / 64);
        }
        sc.leaf_cnt[lf as usize] = 0;
    }
    sum_groups.sort_unstable();
    sum_groups.dedup();
    let mut c_leaves: Vec<u32> = u_c.iter().map(|&x| ctx.leaf[x as usize]).collect();
    c_leaves.sort_unstable();
    c_leaves.dedup();
    let mut m_leaves: Vec<u32> = matches.iter().map(|&(_, x)| ctx.leaf[x as usize]).collect();
    m_leaves.sort_unstable();
    m_leaves.dedup();
    let mut m_groups: Vec<u32> = m_leaves.iter().map(|&l| l / 64).collect();
    m_groups.dedup();
    let _ = write!(
        row_line,
        "\t{k_eff}\t{d_k}\t{m_k}\t{n_chunks}\t{chunks_touched}\t{leaves_touched}\t{}\t{sum_leaves}\t{}\t{}\t{}",
        c_leaves.len(),
        m_leaves.len(),
        sum_groups.len(),
        m_groups.len()
    );
    // Starts of the policies (label(c*) + global).
    let mut starts: Vec<u32> = Vec::new();
    if let Some(t) = ctx.ix.label_starts {
        for &l in cnf.clause(cstar) {
            if (l as usize) < t.n_labels() {
                starts.extend_from_slice(t.label_starts(l));
            }
        }
    }
    starts.extend_from_slice(ctx.ix.global);
    starts.sort_unstable();
    starts.dedup();
    // Reachability models (rows with a small enough binding clause).
    let do_bfs = u_c.len() <= args.bfs_cap;
    let gt_global = gt.iter().filter(|&&t| bit(&ctx.global_reach, t)).count();
    // proj2 → reach_p (kept for the cut class); strict2 → reach.
    let mut model_cols = String::new();
    let mut entries_c: Vec<u32> = Vec::new();
    let mut entries_m: Vec<u32> = Vec::new();
    if do_bfs {
        // strict1, proj1, proj2 (kept in reach_p), strict2 (kept in reach).
        let mut counts = Vec::new();
        for (adm_c, two) in [(false, false), (true, false), (true, true), (false, true)] {
            ctx.reach_model(sc, qv, &starts, adm_c, two, adm_c && two);
            let (reach, reached) = if adm_c && two {
                (&sc.reach_p, &sc.reached_p)
            } else {
                (&sc.reach, &sc.reached)
            };
            let adm = if adm_c { &sc.in_c } else { &sc.in_m };
            let a_reached = reached
                .iter()
                .filter(|&&v| (v as usize) < n_data && bit(adm, v))
                .count();
            let m_reached = reached
                .iter()
                .filter(|&&v| (v as usize) < n_data && bit(&sc.in_m, v))
                .count();
            let gt_reached = gt.iter().filter(|&&t| bit(reach, t)).count();
            counts.push((a_reached, m_reached, gt_reached));
        }
        // Source components of the 1-hop induced subgraphs (full matches; c* points) that the
        // two-hop models do not reach, and one entry per such component.
        let m_nodes: Vec<u32> = matches.iter().map(|&(_, x)| x).collect();
        let cover = sc.reach.clone();
        let (reps, cnt) = ctx.source_entries(sc, &m_nodes, &cover, args.max_entries);
        let _ = write!(
            model_cols,
            "\t{}\t{}\t{}\t{}",
            cnt[0], cnt[1], cnt[2], cnt[3]
        );
        entries_m = reps;
        let cover = sc.reach_p.clone();
        let (reps, cnt) = ctx.source_entries(sc, &u_c, &cover, args.max_entries);
        let _ = write!(
            model_cols,
            "\t{}\t{}\t{}\t{}",
            cnt[0], cnt[1], cnt[2], cnt[3]
        );
        entries_c = reps;
        for (a, mr, gr) in &counts {
            let _ = write!(model_cols, "\t{a}\t{mr}\t{gr}");
        }
    }
    let _ = write!(
        row_line,
        "\t{}\t{gt_global}\t{}",
        u8::from(do_bfs),
        if do_bfs {
            model_cols.trim_start_matches('\t')
        } else {
            ""
        }
    );
    // Searches.
    let mut ids = vec![0u32; k];
    let mut dists = vec![0f32; k];
    for (pname, plan) in &ctx.plans {
        let strict = pname == "p5hg_bm";
        let relaxed = pname == "kha";
        let mut efforts: Vec<(usize, bool)> = ctx.efforts.iter().map(|&l| (l, false)).collect();
        if args.trace_effort > 0 {
            efforts.push((args.trace_effort, true));
        }
        for &(l, traced) in &efforts {
            let variants: &[&str] = if traced {
                &["trace"]
            } else {
                &["base", "oracle", "scc"]
            };
            for &var in variants {
                sc.st.extra_starts.clear();
                match var {
                    "oracle" => sc.st.extra_starts.push(gt[0]),
                    "scc" => {
                        if !do_bfs {
                            continue;
                        }
                        let e = if strict { &entries_m } else { &entries_c };
                        sc.st.extra_starts.extend_from_slice(e);
                    }
                    _ => {}
                }
                if traced {
                    sc.st.trace = Some(Trace::new());
                }
                let q = Query {
                    vector: qv,
                    cnf,
                    view: None,
                };
                let ctr = sc
                    .st
                    .search(plan, &ctx.ix, &q, l, &mut ids, &mut dists, None);
                let trace = sc.st.trace.take();
                sc.st.extra_starts.clear();
                // Recall (tie-aware).
                let mut hits = 0usize;
                let mut returned: Vec<u32> = Vec::new();
                for &x in ids.iter().take(ctr.n_out as usize) {
                    if (x as usize) < n_data && bit(&sc.in_m, x) {
                        returned.push(x);
                        if ctx.dist(sc, qv, x) <= tol {
                            hits += 1;
                        }
                    }
                }
                let hits = hits.min(k_eff);
                let mut out = SearchOut {
                    hits,
                    n_dist: ctr.n_dist,
                    n_pred: ctr.n_pred,
                    n_expand: ctr.n_expand,
                    n_two_hop: ctr.n_two_hop,
                    n_start: ctr.n_start,
                    miss: [0; 4],
                };
                if var == "base" {
                    let mut missed: Vec<u32> = gt
                        .iter()
                        .copied()
                        .filter(|t| !returned.contains(t))
                        .collect();
                    let n_missed = k_eff - hits;
                    if missed.len() > n_missed {
                        missed.drain(..missed.len() - n_missed);
                    }
                    for &t in &missed {
                        let class = if relaxed {
                            usize::from(bit(&ctx.global_reach, t)) * 2
                        } else if !do_bfs {
                            4
                        } else if strict {
                            if bit(&sc.reach, t) {
                                2
                            } else if bit(&sc.reach_p, t) {
                                1
                            } else {
                                0
                            }
                        } else if bit(&sc.reach_p, t) {
                            2
                        } else {
                            0
                        };
                        let class = if class == 2 && sc.st.visited(t) {
                            3
                        } else {
                            class
                        };
                        if class < 4 {
                            out.miss[class] += 1;
                        }
                    }
                }
                // Work until the first / last answer found (traced runs).
                let (mut w_first, mut w_last, mut w_total) = (-1i64, -1i64, 0i64);
                if let Some(t) = trace {
                    let mut found = 0usize;
                    let mut seen: Vec<u32> = Vec::new();
                    for e in &t.events {
                        let dist_ev = (e.kind == "offer" || e.kind == "start") && e.d.is_some();
                        if dist_ev {
                            w_total += 1;
                        }
                        if (e.kind == "offer" || e.kind == "start" || e.kind == "answer")
                            && gt.contains(&e.node)
                            && !seen.contains(&e.node)
                        {
                            seen.push(e.node);
                            found += 1;
                            if found == 1 {
                                w_first = w_total;
                            }
                            w_last = w_total;
                        }
                    }
                }
                let _ = writeln!(
                    runs,
                    "{}\t{}\t{pname}\t{l}\t{var}\t{k_eff}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{w_first}\t{w_last}\t{w_total}",
                    m.qset,
                    m.qidx,
                    out.hits,
                    out.n_dist,
                    out.n_pred,
                    out.n_expand,
                    out.n_two_hop,
                    out.n_start,
                    out.miss[0],
                    out.miss[1],
                    out.miss[2],
                    out.miss[3],
                );
            }
        }
    }
    // Clear the model bitsets.
    for &v in &sc.reached {
        clear_bit(&mut sc.reach, v);
    }
    sc.reached.clear();
    for &v in &sc.reached_p {
        clear_bit(&mut sc.reach_p, v);
    }
    sc.reached_p.clear();
    cleanup(sc, &u_c);
    (row_line + "\n", runs)
}

fn run_typed<T: Element>(args: &DiagArgs) -> Result<()> {
    let t0 = Instant::now();
    std::fs::create_dir_all(&args.out)?;
    let mut rows = Rows::from_panel(&args.panel, Some(&args.split), None, None)?;
    // Row filter.
    if let Some(bands) = &args.bands {
        let want: Vec<&str> = bands.split(',').map(str::trim).collect();
        let meta = panel_meta(&rows)?;
        let keep: Vec<bool> = meta
            .iter()
            .map(|m| {
                want.iter().any(|&w| {
                    (w == "traffic" && m.traffic == "1") || (m.stratum == "1" && m.band == w)
                })
            })
            .collect();
        let mut it = keep.iter();
        rows.rows.retain(|_| *it.next().unwrap_or(&false));
        let mut it = keep.iter();
        rows.panel_lines.retain(|_| *it.next().unwrap_or(&false));
    }
    if let Some(mx) = args.max_rows {
        if rows.rows.len() > mx && mx > 0 {
            let step = rows.rows.len().div_ceil(mx);
            let idx: Vec<usize> = (0..rows.rows.len()).step_by(step).collect();
            rows.rows = idx.iter().map(|&i| rows.rows[i].clone()).collect();
            rows.panel_lines = idx.iter().map(|&i| rows.panel_lines[i].clone()).collect();
        }
    }
    ensure!(!rows.rows.is_empty(), "no rows left");
    let meta = panel_meta(&rows)?;
    let ds = Dataset::<T>::load(&args.dataset, &rows.qsets(), &[])?;
    let lg = load_graph(&args.graph, args.starts.as_deref(), T::DTYPE)?;
    if let Some(s) = &lg.starts {
        if s.n_labels() > 0 {
            s.check_dictionary(&ds.labels)?;
        }
    }
    let n = ds.base.nrows();
    let bitsets = LabelBitsets::build(&ds.labels, n.div_ceil(64) as u64);
    let ix = Index {
        graph: &lg.graph,
        vectors: lg.graph.vectors(&ds.base)?,
        labels: &ds.labels,
        global: &lg.global,
        label_starts: lg.starts.as_ref().filter(|s| s.n_labels() > 0),
        sample: lg.sample.as_ref(),
        label_view: false,
        view_postings: None,
        eps_min: 0.0,
        bitsets: Some(&bitsets),
        costs: None,
        sketches: None,
    };
    let mut plans = Vec::new();
    for p in args
        .policies
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
    {
        let kernel = policy_kernel(p)?;
        plans.push((p.to_owned(), Plan::new(&kernel, &ix)?));
    }
    let global_reach = lg.graph.reachable(&lg.global);
    let all: Vec<u32> = (0..n as u32).collect();
    let mut leaf = vec![0u32; n];
    let n_leaves = rp_leaves(&ds.base, &all, args.block, args.seed, &mut leaf);
    let t_setup = t0.elapsed().as_secs_f64();
    let ctx = Ctx {
        args,
        ds: &ds,
        graph: &lg.graph,
        ix,
        plans,
        efforts: parse_list(&args.efforts)?,
        meta,
        global_reach,
        leaf,
        n_leaves,
    };
    let out = par_map(
        ctx.meta.len(),
        args.threads,
        || ctx.scratch(),
        |i, sc| analyze_row(&ctx, sc, i),
    );
    let mut rows_tsv = String::from(
        "qset\tqidx\tband\tdilution\tstratum\ttraffic\tn_clauses\tu_cstar\tn_match\tk_eff\td_k\tm_k\tn_chunks\tchunks_touched\tleaves_touched\tleaves_cstar\tleaves_summary\tleaves_match\tgroups_summary\tgroups_match\tbfs\tgt_global\t\
         scc_m\tsrc_m\tsrc_m_uncov\tscc_m_max\tscc_c\tsrc_c\tsrc_c_uncov\tscc_c_max\t\
         strict1_a\tstrict1_m\tstrict1_gt\tproj1_a\tproj1_m\tproj1_gt\tproj2_a\tproj2_m\tproj2_gt\tstrict2_a\tstrict2_m\tstrict2_gt\n",
    );
    let mut runs_tsv = String::from(
        "qset\tqidx\tpolicy\teffort\tvariant\tk_eff\thits\tn_dist\tn_pred\tn_expand\tn_two_hop\tn_start\tmiss_unreach\tmiss_cut\tmiss_undisc\tmiss_collect\tw_first\tw_last\tw_total\n",
    );
    for (r, s) in &out {
        rows_tsv.push_str(r);
        runs_tsv.push_str(s);
    }
    std::fs::write(args.out.join("rows.tsv"), rows_tsv)?;
    std::fs::write(args.out.join("runs.tsv"), runs_tsv)?;
    // Per-label spread over the leaves of the shared order.
    let labels = &ds.labels;
    let mut lab = String::from("label\tcategory\tsize\tleaves\tmin_leaves\n");
    let mut seen: Vec<u32> = vec![u32::MAX; ctx.n_leaves];
    for l in 0..labels.n_labels() as u32 {
        let p = labels.postings(l);
        if p.len() < 2 {
            continue;
        }
        let mut cnt = 0usize;
        for &x in p {
            let lf = ctx.leaf[x as usize] as usize;
            if seen[lf] != l {
                seen[lf] = l;
                cnt += 1;
            }
        }
        let _ = writeln!(
            lab,
            "{l}\t{}\t{}\t{cnt}\t{}",
            labels.category(l),
            p.len(),
            p.len().div_ceil(args.block.max(1))
        );
    }
    std::fs::write(args.out.join("labels.tsv"), lab)?;
    let meta = json!({
        "tool": "flab diag",
        "dataset": args.dataset, "graph": args.graph, "starts": args.starts, "panel": args.panel,
        "split": args.split, "k": args.k, "policies": args.policies, "efforts": args.efforts,
        "trace_effort": args.trace_effort, "bands": args.bands, "max_rows": args.max_rows,
        "bfs_cap": args.bfs_cap, "two_hop": args.two_hop, "block": args.block,
        "max_entries": args.max_entries, "threads": args.threads, "seed": args.seed,
        "rows": ctx.meta.len(), "n_leaves": ctx.n_leaves, "setup_s": t_setup,
        "elapsed_s": t0.elapsed().as_secs_f64(),
        "graph_nodes": lg.graph.n_nodes(), "graph_aux": lg.graph.n_aux(),
    });
    std::fs::write(
        args.out.join("diag.json"),
        serde_json::to_string_pretty(&meta)? + "\n",
    )?;
    eprintln!(
        "flab diag: {} rows in {:.1} s -> {}",
        ctx.meta.len(),
        t0.elapsed().as_secs_f64(),
        args.out.display()
    );
    Ok(())
}
