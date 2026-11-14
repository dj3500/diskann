/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! `flab calibrate`: the unit costs of the cost model ([`crate::kernel::cost::UnitCosts`]) on
//! one machine and dataset, measured with the kernel's own code on base data only (no query
//! of any workload is read).
//!
//! - **Density curves** (`dist_ns`, `pred_ns`): for each candidate density ρ (fraction of the
//!   data points), a fresh seeded random subset of the ids in increasing order (≥ 256 points):
//!   ns per distance (vectors prefetched 8 ahead, as the exact path) and ns per full-predicate
//!   check (row offsets and rows prefetched ahead; the static bitsets at `θ`, rows for rarer
//!   labels), for *calibration predicates* made from the labels of random base points (one or
//!   two clauses: a label of one category, then one or two labels of another category). Median
//!   over repetitions (at least `reps`, and enough subsets for 20,000 points). Like the
//!   graph-search costs, these are steady-state costs (no cache flush; a fresh subset per
//!   repetition keeps sparse candidates out of the caches).
//! - **Postings**: the k-way merge with deduplication of real posting lists (ns per entry and
//!   binary level), galloping intersection of the calibration predicates (ns per modelled
//!   probe), bitmap word operations, scatters and enumeration.
//! - **Probe**: ns per sampled point of a sketch probe (every clause evaluated on random points,
//!   prefetched like the probe).
//! - **Graph search** (when a graph is given): random base points as queries, global start,
//!   heap(K = 10) collection, mode-1 predicate; list sizes 10…2560. A predicate-free search
//!   (deferred evaluation) gives ns per distance computation (adjacency, visited set and beam
//!   included); the eager search with the calibration predicates gives ns per predicate
//!   evaluation; a least-squares line of distance computations against L gives `α`, `β`.

use std::{path::Path, time::Instant};

use anyhow::{ensure, Result};
use serde_json::{json, Value};

use crate::{
    bitsets::LabelBitsets,
    kernel::{
        collect::{Answers, Collector},
        config::{Collector as Coll, Kernel, Laziness},
        cost::UnitCosts,
        eval::{Compiled, Eval, EvalCtx, RowScan, StaticBits, Weights},
        exact,
        materialize::{merge_distinct, Materialized},
        search::{Counters, Index, Plan, Query, Searcher},
    },
    labels::LabelStore,
    predicate::{PredicateSet, PredicateSetBuilder},
    run::{data, sys},
    vectors::{Element, Matrix},
};

/// Options of `flab calibrate`.
#[derive(Debug, Clone)]
pub struct Options {
    /// Repetitions per measurement (median).
    pub reps: usize,
    /// Calibration predicates and queries.
    pub queries: usize,
    pub seed: u64,
    /// Static bitsets threshold (`None`: ⌈n/64⌉, the exact path's default; `Some(0)`: none).
    pub theta: Option<u64>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            reps: 3,
            queries: 64,
            seed: 7,
            theta: None,
        }
    }
}

/// The candidate densities of the curves.
pub const DENSITIES: [f64; 6] = [1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }

    /// A sorted random subset of `0..n` with about `m` elements (Bernoulli sampling).
    fn subset(&mut self, n: usize, m: usize) -> Vec<u32> {
        if m >= n {
            return (0..n as u32).collect();
        }
        let p = m as f64 / n as f64;
        let cut = (p * u64::MAX as f64) as u64;
        (0..n as u32).filter(|_| self.next() < cut).collect()
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_unstable_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Calibration predicates from the labels of random base points.
fn predicates(labels: &LabelStore, count: usize, rng: &mut Rng) -> PredicateSet {
    let mut pb = PredicateSetBuilder::new(labels);
    let n = labels.n_points();
    let mut made = 0;
    let mut tries = 0;
    while made < count && tries < 100 * count {
        tries += 1;
        let row = labels.row(rng.below(n) as u32);
        if row.is_empty() {
            continue;
        }
        let first = row[rng.below(row.len())];
        let cat = labels.category(first);
        let others: Vec<u32> = row
            .iter()
            .copied()
            .filter(|&l| labels.category(l) != cat)
            .collect();
        let mut clauses = vec![vec![labels.label_name(first).to_owned()]];
        if !others.is_empty() {
            let second = others[rng.below(others.len())];
            let range = labels.category_labels(labels.category(second));
            let extra = range.start + rng.below((range.end - range.start) as usize) as u32;
            let mut c2 = vec![labels.label_name(second).to_owned()];
            if extra != second {
                c2.push(labels.label_name(extra).to_owned());
            }
            clauses.push(c2);
        }
        pb.push(&clauses);
        made += 1;
    }
    pb.finish()
}

/// Measures the unit costs; returns the report (`costs` and the raw measurements).
pub fn calibrate<T: Element>(
    dataset: &Path,
    graph: Option<&Path>,
    starts: Option<&Path>,
    opts: &Options,
) -> Result<Value> {
    let t_all = Instant::now();
    ensure!(
        opts.reps >= 1 && opts.queries >= 1,
        "reps and queries must be positive"
    );
    let ds = data::Dataset::<T>::load(dataset, &[], &[])?;
    let labels = &ds.labels;
    let base = &ds.base;
    let n = base.nrows();
    ensure!(n >= 1000, "calibration needs at least 1,000 points");
    let theta = opts.theta.unwrap_or(n.div_ceil(64) as u64);
    let bitsets = (theta > 0).then(|| LabelBitsets::build(labels, theta));
    std::hint::black_box(data::touch(base.as_slice()));
    let mut rng = Rng(opts.seed);
    let preds = predicates(labels, opts.queries, &mut rng);
    let mut compiled = Compiled::new();
    let mut bits = vec![0u64; labels.n_labels()];
    let mut mat = Materialized::new(n, true);
    let mut answers = Answers::new();
    let qvec: Vec<T> = base.row(rng.below(n)).to_vec();
    let mut details = serde_json::Map::new();

    // Density curves.
    let mut dist_ns = Vec::new();
    let mut pred_ns = Vec::new();
    let mut curve_rows = Vec::new();
    for &rho in &DENSITIES {
        let m = ((rho * n as f64) as usize).max(256).min(n);
        // Sparse densities: enough fresh subsets for at least 20,000 points.
        let subsets = opts.reps.max(20_000usize.div_ceil(m));
        let (mut dv, mut pv) = (Vec::new(), Vec::new());
        for rep in 0..subsets {
            let subset = rng.subset(n, m);
            let vectors = crate::graph::NodeVectors::data(base);
            answers.reset(Collector::Heap, 10, 10);
            let mut ctr = Counters::default();
            let t = Instant::now();
            exact::distances(&subset, &vectors, &qvec, &mut answers, false, &mut ctr);
            dv.push(t.elapsed().as_nanos() as f64 / subset.len() as f64);

            let qi = (rep * DENSITIES.len() + pv.len()) % preds.len().max(1);
            compiled.compile(labels, preds.get(qi), Weights::Unit, None, &mut bits);
            let mut cx = EvalCtx::new(labels, &compiled, &bits, &[], &[]);
            let t = Instant::now();
            match &bitsets {
                Some(b) => {
                    compiled.split_static(labels, b, b.theta(), false);
                    cx = EvalCtx::new(labels, &compiled, &bits, &[], &[]).with_static(b);
                    exact::filter::<StaticBits>(&cx, &subset, &mut mat.matches, &mut ctr);
                }
                None => exact::filter::<RowScan>(&cx, &subset, &mut mat.matches, &mut ctr),
            }
            pv.push(t.elapsed().as_nanos() as f64 / subset.len() as f64);
            compiled.clear_bits(&mut bits);
        }
        let (d, p) = (median(&mut dv), median(&mut pv));
        curve_rows.push(json!({"density": rho, "points": m, "dist_ns": d, "pred_ns": p}));
        dist_ns.push(d);
        pred_ns.push(p);
    }
    details.insert("curves".into(), Value::Array(curve_rows));

    // k-way merge with deduplication of real postings (labels with 1K-100K points).
    let mid: Vec<u32> = (0..labels.n_labels() as u32)
        .filter(|&l| (1000..100_000).contains(&labels.posting_len(l)))
        .collect();
    let mut merge_samples = Vec::new();
    let (mut heap, mut pos) = (Vec::new(), Vec::new());
    for &k in &[2usize, 8, 32] {
        if mid.len() < k {
            continue;
        }
        let mut v = Vec::new();
        for _ in 0..opts.reps {
            let set: Vec<u32> = (0..k).map(|_| mid[rng.below(mid.len())]).collect();
            let total: usize = set.iter().map(|&l| labels.posting_len(l)).sum();
            mat.cands.clear();
            let cands = &mut mat.cands;
            let t = Instant::now();
            merge_distinct(
                k,
                |i| labels.postings(set[i]),
                &mut heap,
                &mut pos,
                |p| {
                    cands.push(p);
                    true
                },
            );
            v.push(t.elapsed().as_nanos() as f64 / (total as f64 * (k as f64).log2()));
        }
        merge_samples.push(median(&mut v));
    }
    let merge_ns = if merge_samples.is_empty() {
        UnitCosts::default().merge_ns
    } else {
        median(&mut merge_samples.clone())
    };
    details.insert("merge_ns_by_k".into(), json!(merge_samples));

    // Galloping intersections of the two-clause calibration predicates (merge cost removed).
    let mut gallop_v = Vec::new();
    for qi in 0..preds.len() {
        let cnf = preds.get(qi);
        if cnf.n_clauses() < 2 {
            continue;
        }
        compiled.compile(labels, cnf, Weights::Unit, None, &mut bits);
        if compiled.is_empty() {
            compiled.clear_bits(&mut bits);
            continue;
        }
        let a = compiled.upper_bound(0) as f64;
        let mut probes = 0.0;
        for j in 1..compiled.n_clauses() {
            for &l in compiled.clause(j) {
                let p = labels.posting_len(l) as f64;
                probes += a * (1.0 + (1.0 + p / a.max(1.0)).log2());
            }
        }
        let k0 = compiled.clause(0).len().max(1) as f64;
        let merge = if k0 > 1.0 {
            a * merge_ns * k0.log2()
        } else {
            0.0
        };
        let t = Instant::now();
        mat.build_sparse(labels, &compiled, n);
        let ns = t.elapsed().as_nanos() as f64 - merge;
        if probes >= 1000.0 && ns > 0.0 {
            gallop_v.push(ns / probes);
        }
        compiled.clear_bits(&mut bits);
    }
    let gallop_ns = if gallop_v.is_empty() {
        UnitCosts::default().gallop_ns
    } else {
        median(&mut gallop_v)
    };

    // Bitmap words, scatters and enumeration.
    let words = n.div_ceil(64);
    let src: Vec<u64> = (0..words).map(|_| rng.next()).collect();
    let mut dst = vec![0u64; words];
    let mut wv = Vec::new();
    for _ in 0..opts.reps {
        let t = Instant::now();
        for (d, s) in dst.iter_mut().zip(&src) {
            *d |= *s;
        }
        std::hint::black_box(&dst);
        wv.push(t.elapsed().as_nanos() as f64 / words as f64);
    }
    let word_ns = median(&mut wv);
    let big = (0..labels.n_labels() as u32)
        .max_by_key(|&l| labels.posting_len(l))
        .unwrap_or(0);
    let mut sv = Vec::new();
    for _ in 0..opts.reps {
        dst.fill(0);
        let post = labels.postings(big);
        let t = Instant::now();
        for &p in post {
            dst[(p >> 6) as usize] |= 1u64 << (p & 63);
        }
        std::hint::black_box(&dst);
        sv.push(t.elapsed().as_nanos() as f64 / post.len().max(1) as f64);
    }
    let scatter_ns = median(&mut sv);
    // Enumeration of a bitmap with 10 % of the bits set (per word).
    let sparse_bits: Vec<u64> = (0..words)
        .map(|_| {
            let mut w = 0u64;
            for b in 0..64 {
                if rng.below(10) == 0 {
                    w |= 1 << b;
                }
            }
            w
        })
        .collect();
    let mut ev = Vec::new();
    for _ in 0..opts.reps {
        mat.matches.clear();
        let t = Instant::now();
        for (w, &x) in sparse_bits.iter().enumerate() {
            let mut x = x;
            while x != 0 {
                mat.matches.push(((w as u32) << 6) | x.trailing_zeros());
                x &= x - 1;
            }
        }
        std::hint::black_box(&mat.matches);
        ev.push(t.elapsed().as_nanos() as f64 / words as f64);
    }
    let enum_ns = median(&mut ev);

    // Probe: every clause on random points, prefetched like the probe (128 points per query).
    let mut pv = Vec::new();
    for qi in 0..preds.len() {
        compiled.compile(labels, preds.get(qi), Weights::Unit, None, &mut bits);
        if compiled.is_empty() {
            compiled.clear_bits(&mut bits);
            continue;
        }
        let pts: Vec<u32> = (0..128).map(|_| rng.below(n) as u32).collect();
        let cx = match &bitsets {
            Some(b) => {
                compiled.split_static(labels, b, b.theta(), false);
                EvalCtx::new(labels, &compiled, &bits, &[], &[]).with_static(b)
            }
            None => EvalCtx::new(labels, &compiled, &bits, &[], &[]),
        };
        let t = Instant::now();
        let failed = match &bitsets {
            Some(_) => probe_points::<StaticBits>(&cx, &pts),
            None => probe_points::<RowScan>(&cx, &pts),
        };
        std::hint::black_box(failed);
        pv.push(t.elapsed().as_nanos() as f64 / pts.len() as f64);
        compiled.clear_bits(&mut bits);
    }
    let probe_ns = if pv.is_empty() {
        UnitCosts::default().probe_ns
    } else {
        median(&mut pv)
    };

    // Graph search.
    let mut graph_doc = Value::Null;
    let (mut graph_dist_ns, mut graph_pred_ns, mut graph_alpha, mut graph_beta) = {
        let d = UnitCosts::default();
        (
            d.graph_dist_ns,
            d.graph_pred_ns,
            d.graph_alpha,
            d.graph_beta,
        )
    };
    if let Some(gpath) = graph {
        let g = data::load_graph(gpath, starts, T::DTYPE)?;
        ensure!(g.graph.n_data() == n, "the graph has another point count");
        let ix = Index {
            graph: &g.graph,
            vectors: g.graph.vectors(base)?,
            labels,
            global: &g.global,
            label_starts: None,
            sample: None,
            label_view: false,
            view_postings: None,
            eps_min: 0.0,
            bitsets: None,
            costs: None,
            sketches: None,
        };
        let mut eager = Kernel {
            collector: Coll::Heap,
            ..Kernel::default()
        };
        eager.laziness = Laziness::Eager;
        let deferred = Kernel {
            laziness: Laziness::Deferred,
            ..Kernel::default()
        };
        let plans = [
            Plan::new(&deferred, &ix)?.generic(),
            Plan::new(&eager, &ix)?.generic(),
        ];
        let mut st = Searcher::for_index(&ix, &[&plans[0], &plans[1]]);
        let qrows: Vec<usize> = (0..opts.queries).map(|_| rng.below(n)).collect();
        let qm = Matrix::from_slice(
            &qrows
                .iter()
                .flat_map(|&r| base.row(r).iter().copied())
                .collect::<Vec<T>>(),
            qrows.len(),
            base.ncols(),
        );
        let ls = [10usize, 20, 40, 80, 160, 320, 640, 1280, 2560];
        let (mut ids, mut ds) = (vec![0u32; 10], vec![0f32; 10]);
        let mut sums = [[0f64; 3]; 2];
        let mut by_l = Vec::new();
        for &l in &ls {
            let mut per = [[0f64; 3]; 2];
            for (pi, plan) in plans.iter().enumerate() {
                let mut tv = Vec::new();
                let mut nd = 0.0;
                let mut np = 0.0;
                for _ in 0..opts.reps {
                    let t = Instant::now();
                    nd = 0.0;
                    np = 0.0;
                    for (i, &r) in qrows.iter().enumerate() {
                        let _ = r;
                        let q = Query {
                            vector: qm.row(i),
                            cnf: preds.get(i % preds.len().max(1)),
                            view: None,
                        };
                        let c = st.search(plan, &ix, &q, l, &mut ids, &mut ds, None);
                        nd += c.n_dist as f64;
                        np += c.n_pred as f64;
                    }
                    tv.push(t.elapsed().as_nanos() as f64);
                }
                let t = median(&mut tv);
                per[pi] = [t, nd, np];
                for (a, b) in sums[pi].iter_mut().zip(per[pi]) {
                    *a += b;
                }
            }
            let q = qrows.len() as f64;
            by_l.push(json!({
                "L": l,
                "deferred": {"ns_per_query": per[0][0] / q, "n_dist": per[0][1] / q},
                "eager": {"ns_per_query": per[1][0] / q, "n_dist": per[1][1] / q, "n_pred": per[1][2] / q},
            }));
        }
        graph_dist_ns = sums[0][0] / sums[0][1].max(1.0);
        graph_pred_ns = ((sums[1][0] - graph_dist_ns * sums[1][1]) / sums[1][2].max(1.0)).max(0.0);
        // Least squares of the eager search's distance computations per query against L.
        let xs: Vec<f64> = ls.iter().map(|&l| l as f64).collect();
        let ys: Vec<f64> = by_l
            .iter()
            .map(|v| v["eager"]["n_dist"].as_f64().unwrap_or(0.0))
            .collect();
        let k = xs.len() as f64;
        let (mx, my) = (xs.iter().sum::<f64>() / k, ys.iter().sum::<f64>() / k);
        let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
        let sxx: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
        graph_alpha = (sxy / sxx).max(0.0);
        graph_beta = (my - graph_alpha * mx).max(0.0);
        graph_doc = json!({"graph": gpath, "queries": qrows.len(), "by_L": by_l});
    }

    let costs = UnitCosts {
        density: DENSITIES.to_vec(),
        dist_ns,
        pred_ns,
        merge_ns,
        gallop_ns,
        word_ns,
        scatter_ns,
        enum_ns,
        probe_ns,
        sketch_ns: merge_ns,
        graph_dist_ns,
        graph_pred_ns,
        graph_alpha,
        graph_beta,
    };
    costs.validate()?;
    details.insert("graph".into(), graph_doc);
    Ok(json!({
        "tool": "flab calibrate",
        "dataset": dataset,
        "dtype": T::NAME,
        "n_points": n,
        "dim": base.ncols(),
        "theta": theta,
        "bitsets": bitsets.as_ref().map(|b| json!({"n_bitsets": b.n_bitsets(), "bytes": b.bytes()})),
        "options": {"reps": opts.reps, "queries": opts.queries, "seed": opts.seed},
        "costs": costs,
        "details": Value::Object(details),
        "host": sys::host_info(),
        "cpu": sys::current_cpu(),
        "affinity": sys::affinity(),
        "load_avg": sys::load_average(),
        "time": sys::utc_timestamp(false),
        "elapsed_s": t_all.elapsed().as_secs_f64(),
    }))
}

/// Every clause of the query on `pts` (rows prefetched ahead, as the probe does).
fn probe_points<P: Eval>(cx: &EvalCtx<'_>, pts: &[u32]) -> u64 {
    const AHEAD: usize = 8;
    let m = pts.len();
    for &v in &pts[..m.min(2 * AHEAD)] {
        P::prefetch_offsets(cx, v);
    }
    for &v in &pts[..m.min(AHEAD)] {
        P::prefetch(cx, v);
    }
    let mut acc = 0u64;
    for i in 0..m {
        if i + 2 * AHEAD < m {
            P::prefetch_offsets(cx, pts[i + 2 * AHEAD]);
        }
        if i + AHEAD < m {
            P::prefetch(cx, pts[i + AHEAD]);
        }
        acc = acc.wrapping_add(P::failed_all(cx, pts[i]));
    }
    acc
}
