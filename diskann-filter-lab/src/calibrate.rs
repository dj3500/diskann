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
//! - **Member checks by residual shape** (`check_base_ns`, `check_word_ns`,
//!   `check_prefetch_ns`, `check_row_ns`): at each density, checks of fresh random candidates
//!   (a scan touches each candidate once) for `[ℓ₀] ∧ …` with `ℓ₀` the rarest label (the
//!   skipped `c*`) and residual clauses of 1, 4 or 16 random labels with a bitset, of two
//!   labels without one (every row read), or of one label without a bitset followed by a rare
//!   bitset label (every row prefetched, few read); least squares of ns per candidate on the
//!   bitset words probed gives the per-candidate and per-word costs, the two row workloads
//!   the per-prefetch and per-read costs. The model multiplies them by the words, prefetches
//!   and reads a query's residual clauses need (their labels and the probe's survival
//!   estimates); the sketch probe pays the same per sampled point.
//! - **Per query** (`query_ns`): compiling a calibration predicate, its static split and
//!   posting bounds.
//! - **Graph search** (when a graph is given): random base points as queries, starting as the
//!   crossover's graph route does (`label(c*)+global` when the start table has per-label
//!   starts, else the global start), heap(K = 10) collection, mode-1 predicate; list sizes
//!   10…2560. A predicate-free search (deferred evaluation) gives ns per distance computation
//!   (adjacency, visited set and beam included); the eager search with the calibration
//!   predicates gives ns per predicate evaluation and the distance computations per list size
//!   (`D(L)`, also summarized by a least-squares line `α`, `β`); `heap_bound` searches at
//!   K = 10 and 100 give the predicate evaluations beyond `K/r` per list size
//!   (`extra(K, L)`), `distance_first` searches their share of the distances. The
//!   predicate-free search also gives ns per distance by list size (`graph_dist_ns_curve` at
//!   `graph_dist_ns_l`), extended to L = 40,960 with a quarter of the queries: maintaining a
//!   long beam costs more per distance.
//! - **Effort curve** (when a graph is given; `calibrate/effort.rs`): the unfiltered list size for mean
//!   recall R at K' neighbours on midpoints of random base pairs (`effort_r`, `effort_k`,
//!   `effort_l`), read by `terminator = effort_curve`.

mod effort;

pub use effort::EffortOptions;

use std::{path::Path, time::Instant};

use anyhow::{ensure, Result};
use serde_json::{json, Value};

use crate::{
    bitsets::LabelBitsets,
    kernel::{
        collect::{Answers, Collector},
        config::{Collector as Coll, Kernel, Laziness, StartSource},
        cost::{Estimates, UnitCosts},
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
    /// The effort curve (with a graph; no queries and no pairs: none).
    pub effort: EffortOptions,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            reps: 3,
            queries: 64,
            seed: 7,
            theta: None,
            effort: EffortOptions::default(),
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

/// Calibration predicates from the labels of random base points: one of the point's labels,
/// ANDed with a clause of a label of another category of the point (and possibly a second
/// label of that category). With `within`, a point without another category's label ANDs two
/// of its labels instead (singleton clauses, as the queries of single-category label sets).
fn predicates(labels: &LabelStore, count: usize, within: bool, rng: &mut Rng) -> PredicateSet {
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
        } else if within && row.len() > 1 {
            let mut second = row[rng.below(row.len() - 1)];
            if second == first {
                second = row[row.len() - 1];
            }
            clauses.push(vec![labels.label_name(second).to_owned()]);
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
    let preds = predicates(labels, opts.queries, false, &mut rng);
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

    // Checks of members of union(c0) (as the exact path checks its candidates) and galloping
    // steps by gap, on a separate random stream (the measurements above keep batch B6's).
    let mut rng2 = Rng(opts.seed ^ 0x5eed_a1a1);
    let (pred_row_ns, pred_bits_ns, member_doc) =
        member_curves(labels, bitsets.as_ref(), n, opts, &mut rng2, &pred_ns);
    details.insert("member_checks".into(), member_doc);
    let (gallop_gap, gallop_gap_ns, gallop_doc) = gallop_curve(labels, n, word_ns, &mut rng2);
    details.insert("gallop_by_gap".into(), gallop_doc);
    // Cold member checks by residual shape and the per-query compile cost, on a third stream.
    let mut rng3 = Rng(opts.seed ^ 0x5eed_a1b0);
    let (check_base_ns, check_word_ns, check_prefetch_ns, check_row_ns, shape_doc) =
        shape_curves(labels, bitsets.as_ref(), n, opts, &mut rng3);
    details.insert("check_shape".into(), shape_doc);
    let query_ns = query_cost(labels, bitsets.as_ref(), &preds);

    // Graph search.
    let mut graph_doc = Value::Null;
    let (mut heap_bound_k, mut heap_bound_extra) = (Vec::new(), Vec::new());
    let (mut heap_bound_l, mut heap_bound_extra_kl) = (Vec::new(), Vec::new());
    let (mut graph_l, mut graph_l_dists) = (Vec::new(), Vec::new());
    let mut distance_first_frac = 0.0;
    let (mut graph_dist_ns_l, mut graph_dist_ns_curve) = (Vec::new(), Vec::new());
    let (mut effort_r, mut effort_k, mut effort_l) = (Vec::new(), Vec::new(), Vec::new());
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
        // With per-label starts, the searches start as the crossover's graph route does.
        let label_starts = g.starts.as_ref().filter(|s| s.n_labels() > 0);
        let start_policy = if label_starts.is_some() {
            vec![StartSource::LabelSmallest, StartSource::Global]
        } else {
            vec![StartSource::Global]
        };
        let ix = Index {
            graph: &g.graph,
            vectors: g.graph.vectors(base)?,
            labels,
            global: &g.global,
            label_starts,
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
            starts: start_policy.clone(),
            ..Kernel::default()
        };
        eager.laziness = Laziness::Eager;
        let deferred = Kernel {
            laziness: Laziness::Deferred,
            starts: start_policy.clone(),
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
        // ns per distance of the predicate-free search by list size, up to the largest list
        // sizes that planned searches reach (L₀ × 16): beam maintenance grows with L. The
        // large sizes use a quarter of the queries.
        for v in &by_l {
            let l = v["L"].as_f64().unwrap_or(0.0);
            let (t, nd) = (
                v["deferred"]["ns_per_query"].as_f64().unwrap_or(0.0),
                v["deferred"]["n_dist"].as_f64().unwrap_or(0.0),
            );
            if nd > 0.0 {
                graph_dist_ns_l.push(l);
                graph_dist_ns_curve.push(t / nd);
            }
        }
        let few = (qrows.len() / 4).max(1);
        for l in [5120usize, 10240, 20480, 40960] {
            let mut tv = Vec::new();
            let mut nd = 0.0;
            for _ in 0..opts.reps {
                let t = Instant::now();
                nd = 0.0;
                for i in 0..few {
                    let q = Query {
                        vector: qm.row(i),
                        cnf: preds.get(i % preds.len().max(1)),
                        view: None,
                    };
                    nd += st
                        .search(&plans[0], &ix, &q, l, &mut ids, &mut ds, None)
                        .n_dist as f64;
                }
                tv.push(t.elapsed().as_nanos() as f64);
            }
            if nd > 0.0 {
                graph_dist_ns_l.push(l as f64);
                graph_dist_ns_curve.push(median(&mut tv) / nd);
            }
        }
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
        // The measured curve itself (the line misses its curvature: ×0.5–1.4 at the ends).
        if ys.iter().all(|&y| y > 0.0) {
            graph_l = xs.clone();
            graph_l_dists = ys.clone();
        }

        // Laziness: predicate evaluations of `heap_bound` searches beyond K/r (r: the
        // predicate's match rate on random points; searches capped by their distance count
        // or with r < 1 % left out) and the share of distances evaluated by `distance_first`.
        let rates: Vec<f64> = (0..preds.len())
            .map(|qi| sampled_rate(labels, &preds, qi, &mut rng2, &mut compiled, &mut bits))
            .collect();
        let lazy = |laziness| Kernel {
            collector: Coll::Heap,
            laziness,
            starts: start_policy.clone(),
            ..Kernel::default()
        };
        let (hb, dfk) = (lazy(Laziness::HeapBound), lazy(Laziness::DistanceFirst));
        let lazy_plans = [
            Plan::new(&hb, &ix)?.generic(),
            Plan::new(&dfk, &ix)?.generic(),
        ];
        let mut st = Searcher::for_index(&ix, &[&lazy_plans[0], &lazy_plans[1]]);
        let mut fracs = Vec::new();
        let mut lazy_rows = Vec::new();
        let mut table = Vec::new();
        for &kk in &[10usize, 100] {
            let (mut ids, mut ds) = (vec![0u32; kk], vec![0f32; kk]);
            let mut extra = Vec::new();
            let mut row = Vec::new();
            for &l in &ls {
                let (mut hp, mut hd) = (0.0, 0.0);
                let mut extra_l = Vec::new();
                for i in 0..qrows.len() {
                    let pi = i % preds.len().max(1);
                    let q = Query {
                        vector: qm.row(i),
                        cnf: preds.get(pi),
                        view: None,
                    };
                    let c = st.search(&lazy_plans[0], &ix, &q, l, &mut ids, &mut ds, None);
                    hp += c.n_pred as f64;
                    hd += c.n_dist as f64;
                    if rates[pi] >= 0.01 && c.n_pred < c.n_dist {
                        extra_l.push(c.n_pred as f64 - kk as f64 / rates[pi]);
                    }
                    let c = st.search(&lazy_plans[1], &ix, &q, l, &mut ids, &mut ds, None);
                    if c.n_dist > 0 {
                        fracs.push(c.n_pred as f64 / c.n_dist as f64);
                    }
                }
                extra.extend_from_slice(&extra_l);
                row.push((!extra_l.is_empty()).then(|| median(&mut extra_l).max(0.0)));
                let q = qrows.len().max(1) as f64;
                lazy_rows.push(json!({"K": kk, "L": l, "heap_bound_n_pred": hp / q,
                    "heap_bound_n_dist": hd / q}));
            }
            if !extra.is_empty() {
                heap_bound_k.push(kk as f64);
                heap_bound_extra.push(median(&mut extra).max(0.0));
                table.push(row);
            }
        }
        // extra(K, L) where every heap size has a value at the list size.
        let full: Vec<usize> = (0..ls.len())
            .filter(|&j| table.iter().all(|r| r[j].is_some()))
            .collect();
        if !table.is_empty() && full.len() >= 2 {
            heap_bound_l = full.iter().map(|&j| ls[j] as f64).collect();
            heap_bound_extra_kl = table
                .iter()
                .map(|r| full.iter().map(|&j| r[j].unwrap_or(0.0)).collect())
                .collect();
        }
        if !fracs.is_empty() {
            distance_first_frac = median(&mut fracs).clamp(0.0, 1.0);
        }
        graph_doc = json!({"graph": gpath, "queries": qrows.len(), "by_L": by_l,
            "starts": if label_starts.is_some() { "label(c*)+global" } else { "global" },
            "dist_ns_by_L": {"L": graph_dist_ns_l, "ns": graph_dist_ns_curve},
            "laziness": {"rates": rates, "by_K_L": lazy_rows, "heap_bound_k": heap_bound_k,
                "heap_bound_extra": heap_bound_extra, "heap_bound_l": heap_bound_l,
                "heap_bound_extra_kl": heap_bound_extra_kl,
                "distance_first_frac": distance_first_frac}});
        if opts.effort.queries > 0 || opts.effort.pairs.is_some() {
            let e = effort::effort_curve(base, &ix, labels, &opts.effort)?;
            (effort_r, effort_k, effort_l) = (e.r, e.k, e.l);
            details.insert("effort".into(), e.doc);
        }
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
        pred_row_ns,
        pred_bits_ns,
        gallop_gap,
        gallop_gap_ns,
        heap_bound_k,
        heap_bound_extra,
        distance_first_frac,
        graph_l,
        graph_l_dists,
        heap_bound_l,
        heap_bound_extra_kl,
        check_base_ns,
        check_word_ns,
        check_prefetch_ns,
        check_row_ns,
        query_ns,
        graph_dist_ns_l,
        graph_dist_ns_curve,
        effort_r,
        effort_k,
        effort_l,
        // `reads_at_reach` comes from the kernel configuration (`row_reads`), not calibrated.
        ..UnitCosts::default()
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

/// A sorted sample of about `m` elements of the sorted list `u` (all of it if `m ≥ |u|`).
fn sample_sorted(u: &[u32], m: usize, rng: &mut Rng, out: &mut Vec<u32>) {
    out.clear();
    if m >= u.len() {
        out.extend_from_slice(u);
    } else if 4 * m >= u.len() {
        let cut = (m as f64 / u.len() as f64 * u64::MAX as f64) as u64;
        out.extend(u.iter().copied().filter(|_| rng.next() < cut));
    } else {
        let mut idx: Vec<usize> = (0..m).map(|_| rng.below(u.len())).collect();
        idx.sort_unstable();
        idx.dedup();
        out.extend(idx.into_iter().map(|i| u[i]));
    }
}

/// Points and passes per (density, workload) of [`shape_curves`].
const SHAPE_POINTS: usize = 20_000;
const SHAPE_MAX_PASSES: usize = 48;
/// Candidates per pass at most (a window of the id space keeps the density).
const SHAPE_MAX_CANDS: usize = 100_000;

/// The workloads of [`shape_curves`]: the residual clauses after `[ℓ₀]`.
#[derive(Clone, Copy, PartialEq)]
enum Shape {
    /// One clause of `k` labels with a bitset.
    Bits(usize),
    /// One clause of two labels without a bitset (every member's row read).
    Rows,
    /// A clause of one label without a bitset, then a rarely matching bitset label (every
    /// member's row prefetched, few read).
    Prefetch,
}

/// The bitset words probed and the rows read checking `v` on the residual clauses of `q`, as
/// the static evaluator does (an unsatisfied clause without row labels ends the check; the
/// row is read once if a clause with row labels is left unsatisfied by its bitsets).
fn check_work(q: &Compiled, b: &LabelBitsets, v: u32) -> (usize, usize) {
    let (w, bit) = ((v >> 6) as usize, v & 63);
    let (mut words, mut row) = (0, false);
    for j in 1..q.n_clauses() {
        let mut hit = false;
        for &s in q.static_slots(j) {
            words += 1;
            if (b.word(s, w) >> bit) & 1 != 0 {
                hit = true;
                break;
            }
        }
        if !hit {
            if q.row_labels(j).is_empty() {
                return (words, 0);
            }
            row = true;
        }
    }
    (words, usize::from(row))
}

/// Checks of members of `union(c*)` by residual shape, at each density of [`DENSITIES`]:
/// `(base, word, prefetch, row, report)`, ns per candidate, per static-bitset word probed, per
/// row prefetched and per row read and checked. Each pass compiles `[ℓ₀] ∧ …` with `ℓ₀` the
/// rarest label (so `c* = [ℓ₀]`, skipped as for members) and the residual clauses of a
/// [`Shape`] (random labels), checks fresh random candidates of the density (a window of the
/// id space for dense ones: a scan touches each candidate once; no cache flush, as the other
/// curves: the steady state of a stream of queries) and counts the words probed and rows read
/// (untimed). Least squares of ns per candidate on the words probed (bitset workloads) gives
/// `base` and `word`; the rows workload (`base + prefetch + row`) and the prefetch workload
/// (`base + words·word + prefetch + reads·row`) give `prefetch` and `row`. Without bitsets
/// every check reads the row (`word` = `prefetch` = 0).
fn shape_curves(
    labels: &LabelStore,
    bitsets: Option<&LabelBitsets>,
    n: usize,
    opts: &Options,
    rng: &mut Rng,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Value) {
    let nd = DENSITIES.len();
    let nl = labels.n_labels() as u32;
    let zeros = || vec![0.0; nd];
    let Some(rare) = (0..nl)
        .filter(|&l| labels.posting_len(l) > 0)
        .min_by_key(|&l| (labels.posting_len(l), l))
    else {
        return (zeros(), zeros(), zeros(), zeros(), Value::Null);
    };
    let floor = labels.posting_len(rare);
    let (with_bits, without): (Vec<u32>, Vec<u32>) = (0..nl)
        .filter(|&l| l != rare && labels.posting_len(l) > floor)
        .partition(|&l| bitsets.is_some_and(|b| b.slot(l).is_some()));
    // The rarest quarter of the bitset labels: members seldom match them.
    let mut sparse_bits = with_bits.clone();
    sparse_bits.sort_by_key(|&l| (labels.posting_len(l), l));
    sparse_bits.truncate(with_bits.len().div_ceil(4));
    let mut compiled = Compiled::new();
    let mut bits = vec![0u64; labels.n_labels()];
    let mut out = Vec::new();
    let mut ctr = Counters::default();
    let (mut base, mut word, mut prefetch, mut row) = (zeros(), zeros(), zeros(), zeros());
    let mut doc = Vec::new();
    let pick = |pool: &[u32], rng: &mut Rng| pool[rng.below(pool.len())];
    for (di, &rho) in DENSITIES.iter().enumerate() {
        let m = ((rho * n as f64) as usize)
            .clamp(64, SHAPE_MAX_CANDS)
            .min(n);
        let window = ((m as f64 / rho).ceil() as usize).clamp(m, n);
        // (ns per candidate, words probed, rows read) per pass of each workload.
        let (mut wpts, mut rpts, mut ppts) = (Vec::new(), Vec::new(), Vec::new());
        let workloads: Vec<Shape> = if bitsets.is_some() && !with_bits.is_empty() {
            vec![
                Shape::Bits(1),
                Shape::Bits(4),
                Shape::Bits(16),
                Shape::Rows,
                Shape::Prefetch,
            ]
        } else {
            vec![Shape::Rows]
        };
        for &shape in &workloads {
            let feasible = match shape {
                Shape::Bits(_) => !with_bits.is_empty(),
                Shape::Rows => !without.is_empty(),
                Shape::Prefetch => !without.is_empty() && !sparse_bits.is_empty(),
            };
            if !feasible {
                continue;
            }
            let passes = opts
                .reps
                .max(SHAPE_POINTS.div_ceil(m))
                .min(SHAPE_MAX_PASSES);
            for _ in 0..passes {
                let name = |l: u32| labels.label_name(l).to_owned();
                let mut clauses = vec![vec![name(rare)]];
                match shape {
                    Shape::Bits(k) => {
                        let mut c: Vec<u32> = (0..k).map(|_| pick(&with_bits, rng)).collect();
                        c.sort_unstable();
                        c.dedup();
                        clauses.push(c.into_iter().map(name).collect());
                    }
                    Shape::Rows => {
                        let mut c = vec![pick(&without, rng), pick(&without, rng)];
                        c.sort_unstable();
                        c.dedup();
                        clauses.push(c.into_iter().map(name).collect());
                    }
                    Shape::Prefetch => {
                        clauses.push(vec![name(pick(&without, rng))]);
                        clauses.push(vec![name(pick(&sparse_bits, rng))]);
                    }
                }
                let mut pb = PredicateSetBuilder::new(labels);
                pb.push(&clauses);
                let preds = pb.finish();
                compiled.compile(labels, preds.get(0), Weights::Unit, None, &mut bits);
                if compiled.is_empty() || compiled.n_clauses() < 2 {
                    compiled.clear_bits(&mut bits);
                    continue;
                }
                if let Some(b) = bitsets {
                    compiled.split_static(labels, b, b.theta(), false);
                }
                let start = rng.below(n - window + 1);
                let mut cands: Vec<u32> =
                    (0..m).map(|_| (start + rng.below(window)) as u32).collect();
                cands.sort_unstable();
                cands.dedup();
                let (mut w, mut r) = (0usize, 0usize);
                for &v in &cands {
                    let (dw, dr) = match bitsets {
                        Some(b) => check_work(&compiled, b, v),
                        None => (0, 1),
                    };
                    w += dw;
                    r += dr;
                }
                let mut cx = EvalCtx::new(labels, &compiled, &bits, &[], &[]).cstar_members();
                if let Some(b) = bitsets {
                    cx = cx.with_static(b);
                }
                let t = Instant::now();
                match bitsets {
                    Some(_) => exact::filter::<StaticBits>(&cx, &cands, &mut out, &mut ctr),
                    None => exact::filter::<RowScan>(&cx, &cands, &mut out, &mut ctr),
                }
                let ns = t.elapsed().as_nanos() as f64;
                std::hint::black_box(&out);
                let c = cands.len().max(1) as f64;
                let pt = (ns / c, w as f64 / c, r as f64 / c);
                match shape {
                    Shape::Bits(_) => wpts.push(pt),
                    Shape::Rows => rpts.push(pt),
                    Shape::Prefetch => ppts.push(pt),
                }
                compiled.clear_bits(&mut bits);
            }
        }
        // Least squares on the bitset workloads: ns = base + word · words.
        let (b0, w0) = if wpts.len() >= 2 {
            let k = wpts.len() as f64;
            let mx = wpts.iter().map(|p| p.1).sum::<f64>() / k;
            let my = wpts.iter().map(|p| p.0).sum::<f64>() / k;
            let sxy: f64 = wpts.iter().map(|p| (p.1 - mx) * (p.0 - my)).sum();
            let sxx: f64 = wpts.iter().map(|p| (p.1 - mx) * (p.1 - mx)).sum();
            let slope = if sxx > 0.0 { (sxy / sxx).max(0.0) } else { 0.0 };
            ((my - slope * mx).max(0.0), slope)
        } else {
            (0.0, 0.0)
        };
        let med = |v: &[(f64, f64, f64)], f: fn(&(f64, f64, f64)) -> f64| -> Option<f64> {
            let mut x: Vec<f64> = v.iter().map(f).collect();
            (!x.is_empty()).then(|| median(&mut x))
        };
        // rows: t_r = base + prefetch + row (every member's row read);
        // prefetch: t_p = base + w_p·word + prefetch + r_p·row (few read).
        let (p0, r0) = match (med(&rpts, |p| p.0), med(&ppts, |p| p.0)) {
            (Some(tr), Some(tp)) => {
                let wp = med(&ppts, |p| p.1).unwrap_or(0.0);
                let rp = med(&ppts, |p| p.2).unwrap_or(0.0);
                let read = if rp < 0.9 {
                    ((tr - tp + wp * w0) / (1.0 - rp)).max(0.0)
                } else {
                    tr - b0
                };
                let read = read.min((tr - b0).max(0.0));
                ((tr - b0 - read).max(0.0), read)
            }
            (Some(tr), None) => (0.0, (tr - b0).max(0.0)),
            _ => (0.0, 0.0),
        };
        (base[di], word[di], prefetch[di], row[di]) = (b0, w0, p0, r0);
        doc.push(
            json!({"density": rho, "candidates_per_pass": m, "bitset_passes": wpts.len(),
            "row_passes": rpts.len(), "prefetch_passes": ppts.len(), "base_ns": b0, "word_ns": w0,
            "prefetch_ns": p0, "row_ns": r0,
            "bitset_points": wpts.iter().map(|p| [p.0, p.1]).collect::<Vec<_>>(),
            "row_ns_per_cand": rpts.iter().map(|p| p.0).collect::<Vec<_>>(),
            "prefetch_points": ppts.iter().map(|p| [p.0, p.1, p.2]).collect::<Vec<_>>()}),
        );
    }
    let report = json!({"rare_label_postings": floor, "bitset_labels": with_bits.len(),
        "row_labels": without.len(), "by_density": doc});
    (base, word, prefetch, row, report)
}

/// Per query: compiling a calibration predicate, splitting it against the static bitsets and
/// its posting bounds (median ns).
fn query_cost(labels: &LabelStore, bitsets: Option<&LabelBitsets>, preds: &PredicateSet) -> f64 {
    let mut compiled = Compiled::new();
    let mut bits = vec![0u64; labels.n_labels()];
    let mut v = Vec::new();
    for qi in 0..preds.len() {
        let t = Instant::now();
        compiled.compile(labels, preds.get(qi), Weights::Unit, None, &mut bits);
        if let Some(b) = bitsets {
            compiled.split_static(labels, b, b.theta(), false);
        }
        std::hint::black_box(Estimates::from_bounds(&compiled, labels.n_points()));
        compiled.clear_bits(&mut bits);
        v.push(t.elapsed().as_nanos() as f64);
    }
    if v.is_empty() {
        0.0
    } else {
        median(&mut v)
    }
}

/// Points and predicates measured per (class, density) before a member-check value counts.
const MEMBER_POINTS: usize = 20_000;
const MEMBER_PREDICATES: usize = 3;

/// Checks of members of `union(c0)`, the way the exact path checks its `scan` and `df`
/// candidates (residual clauses, prefetched for them), at each density of [`DENSITIES`]:
/// fresh calibration predicates, each sampled at the densities its union reaches, split into
/// predicates whose residual clauses read rows and bitset-only ones. Densities no predicate
/// reaches take the nearest measured density of the class; a class without measurements
/// takes the other class's curve, and `fallback` without any. Returns `(rows, bits, report)`.
fn member_curves(
    labels: &LabelStore,
    bitsets: Option<&LabelBitsets>,
    n: usize,
    opts: &Options,
    rng: &mut Rng,
    fallback: &[f64],
) -> (Vec<f64>, Vec<f64>, Value) {
    // Residual checks need two clauses: single-category label sets AND two labels.
    let preds = predicates(labels, 4 * opts.queries, labels.n_categories() == 1, rng);
    let nd = DENSITIES.len();
    let mut samples: [Vec<Vec<f64>>; 2] = [vec![Vec::new(); nd], vec![Vec::new(); nd]];
    let mut points = [vec![0usize; nd], vec![0usize; nd]];
    let mut used = [vec![0usize; nd], vec![0usize; nd]];
    let mut compiled = Compiled::new();
    let mut bits = vec![0u64; labels.n_labels()];
    let mut mat = Materialized::new(n, false);
    let (mut union, mut cands) = (Vec::new(), Vec::new());
    let (mut heap, mut pos) = (Vec::new(), Vec::new());
    let mut ctr = Counters::default();
    for qi in 0..preds.len() {
        compiled.compile(labels, preds.get(qi), Weights::Unit, None, &mut bits);
        if compiled.is_empty() || compiled.n_clauses() < 2 {
            compiled.clear_bits(&mut bits);
            continue;
        }
        if let Some(b) = bitsets {
            compiled.split_static(labels, b, b.theta(), false);
        }
        let class = usize::from(!compiled.residual_reads_rows(bitsets.is_some()));
        let upper = compiled.upper_bound(0) as usize;
        let need: Vec<usize> = (0..nd)
            .filter(|&di| {
                let m = (DENSITIES[di] * n as f64) as usize;
                m <= upper
                    && (points[class][di] < MEMBER_POINTS || used[class][di] < MEMBER_PREDICATES)
            })
            .collect();
        if need.is_empty() {
            compiled.clear_bits(&mut bits);
            continue;
        }
        let c0 = compiled.clause(0);
        union.clear();
        merge_distinct(
            c0.len(),
            |i| labels.postings(c0[i]),
            &mut heap,
            &mut pos,
            |p| {
                union.push(p);
                true
            },
        );
        let mut cx = EvalCtx::new(labels, &compiled, &bits, &[], &[]).cstar_members();
        if let Some(b) = bitsets {
            cx = cx.with_static(b);
        }
        for di in need {
            let m = ((DENSITIES[di] * n as f64) as usize).max(64);
            if m > union.len() {
                continue;
            }
            used[class][di] += 1;
            for _ in 0..opts.reps.max(MEMBER_POINTS.div_ceil(m).min(64)) {
                sample_sorted(&union, m, rng, &mut cands);
                let t = Instant::now();
                match bitsets {
                    Some(_) => exact::filter::<StaticBits>(&cx, &cands, &mut mat.matches, &mut ctr),
                    None => exact::filter::<RowScan>(&cx, &cands, &mut mat.matches, &mut ctr),
                }
                samples[class][di].push(t.elapsed().as_nanos() as f64 / cands.len().max(1) as f64);
                points[class][di] += cands.len();
                if points[class][di] >= MEMBER_POINTS && samples[class][di].len() >= opts.reps {
                    break;
                }
            }
        }
        compiled.clear_bits(&mut bits);
    }
    let measured: Vec<Vec<Option<f64>>> = samples
        .iter_mut()
        .map(|cls| {
            cls.iter_mut()
                .map(|v| (!v.is_empty()).then(|| median(v)))
                .collect()
        })
        .collect();
    let fill = |cls: &[Option<f64>]| -> Option<Vec<f64>> {
        let have: Vec<usize> = (0..cls.len()).filter(|&i| cls[i].is_some()).collect();
        if have.is_empty() {
            return None;
        }
        Some(
            (0..cls.len())
                .map(|i| {
                    let j = *have
                        .iter()
                        .min_by_key(|&&j| (j as isize - i as isize).unsigned_abs())
                        .unwrap_or(&have[0]);
                    cls[j].unwrap_or(0.0)
                })
                .collect(),
        )
    };
    let (rows, bitsv) = match (fill(&measured[0]), fill(&measured[1])) {
        (Some(r), Some(b)) => (r, b),
        (Some(r), None) => (r.clone(), r),
        (None, Some(b)) => (b.clone(), b),
        (None, None) => (fallback.to_vec(), fallback.to_vec()),
    };
    let report = json!({
        "predicates": preds.len(),
        "rows": (0..nd).map(|di| json!({"density": DENSITIES[di], "ns": measured[0][di],
            "points": points[0][di], "predicates": used[0][di]})).collect::<Vec<_>>(),
        "bits": (0..nd).map(|di| json!({"density": DENSITIES[di], "ns": measured[1][di],
            "points": points[1][di], "predicates": used[1][di]})).collect::<Vec<_>>(),
    });
    (rows, bitsv, report)
}

/// Gaps (posting entries per galloped candidate) of the galloping-step curve.
const GALLOP_GAPS: [f64; 10] = [
    1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0, 262144.0,
];

/// Galloping steps by gap, with the gallop plan's own code
/// ([`Materialized::build_sparse`]) on two-clause predicates `[ℓ₀] ∧ [ℓ₁]` of label pairs whose
/// posting lengths have the ratio `p/s ≈ gap` (`s = |ℓ₀|` from 256, or 16, to 250,000): ns per modelled
/// step `s·(1 + log2(1 + p/s))`, the copy of the candidates (`s` words) removed. Every label is
/// used once (cold lists, as in one query); the median over up to five pairs per gap, made
/// non-decreasing in the gap (a longer jump never costs less). Gaps without two measurements
/// are left out. Returns `(gaps, ns, report)`.
fn gallop_curve(
    labels: &LabelStore,
    n: usize,
    word_ns: f64,
    rng: &mut Rng,
) -> (Vec<f64>, Vec<f64>, Value) {
    const PAIRS: usize = 5;
    const MIN_CANDS: f64 = 16.0;
    const MAX_CANDS: f64 = 250_000.0;
    let mut by_len: Vec<(usize, u32)> = (0..labels.n_labels() as u32)
        .map(|l| (labels.posting_len(l), l))
        .filter(|&(p, _)| p as f64 >= MIN_CANDS)
        .collect();
    by_len.sort_unstable();
    let range = |lo: f64, hi: f64| {
        (
            by_len.partition_point(|&(p, _)| (p as f64) < lo),
            by_len.partition_point(|&(p, _)| (p as f64) <= hi),
        )
    };
    // Pairs per gap (distinct labels throughout): candidate lists of at least 256 points when
    // possible (less timing noise), else of at least 16; the longer list is drawn among those
    // that can have a partner at this gap.
    let smallest = by_len.first().map_or(MIN_CANDS, |&(p, _)| p as f64);
    let mut used = vec![false; labels.n_labels()];
    let mut pairs: Vec<(usize, u32, u32)> = Vec::new();
    for (gi, &g) in GALLOP_GAPS.iter().enumerate() {
        let mut found = 0;
        for min_s in [256.0, MIN_CANDS] {
            let (lo, hi) = range(min_s.max(smallest) * g / 1.5, MAX_CANDS * g);
            for _ in 0..400 {
                if found == PAIRS || lo >= hi {
                    break;
                }
                let (p, l1) = by_len[lo + rng.below(hi - lo)];
                let target = p as f64 / g;
                let (a, b) = range((target / 1.5).max(min_s), target * 1.5);
                if a >= b {
                    continue;
                }
                let (_, l0) = by_len[a + rng.below(b - a)];
                if l0 == l1 || used[l0 as usize] || used[l1 as usize] {
                    continue;
                }
                used[l0 as usize] = true;
                used[l1 as usize] = true;
                pairs.push((gi, l0, l1));
                found += 1;
            }
        }
    }
    let mut pb = PredicateSetBuilder::new(labels);
    for &(_, l0, l1) in &pairs {
        pb.push(&[vec![labels.label_name(l0)], vec![labels.label_name(l1)]]);
    }
    let set = pb.finish();
    let mut compiled = Compiled::new();
    let mut bits = vec![0u64; labels.n_labels()];
    let mut mat = Materialized::new(n, false);
    let mut per_gap: Vec<Vec<f64>> = vec![Vec::new(); GALLOP_GAPS.len()];
    for (i, &(gi, _, _)) in pairs.iter().enumerate() {
        compiled.compile(labels, set.get(i), Weights::Unit, None, &mut bits);
        if compiled.is_empty() || compiled.n_clauses() != 2 {
            compiled.clear_bits(&mut bits);
            continue;
        }
        // The Boolean order puts the smaller list first: it gives the candidates.
        let s = labels.posting_len(compiled.clause(0)[0]) as f64;
        let p = labels.posting_len(compiled.clause(1)[0]) as f64;
        let t = Instant::now();
        mat.build_sparse(labels, &compiled, n);
        let ns = t.elapsed().as_nanos() as f64 - s * word_ns;
        compiled.clear_bits(&mut bits);
        let steps = s * (1.0 + (1.0 + p / s).log2());
        if ns > 0.0 {
            per_gap[gi].push(ns / steps);
        }
    }
    let (mut gaps, mut ns, mut rows) = (Vec::new(), Vec::new(), Vec::new());
    let mut floor = 0.0f64;
    for (gi, v) in per_gap.iter_mut().enumerate() {
        if v.len() < 2 {
            continue;
        }
        let m = median(v);
        floor = floor.max(m);
        gaps.push(GALLOP_GAPS[gi]);
        ns.push(floor);
        rows.push(
            json!({"gap": GALLOP_GAPS[gi], "median_ns_per_step": m, "ns_per_step": floor,
            "pairs": v.len()}),
        );
    }
    (gaps, ns, json!(rows))
}

/// Match rate of predicate `qi` of `preds` on 4,096 random points (mode 1).
fn sampled_rate(
    labels: &LabelStore,
    preds: &PredicateSet,
    qi: usize,
    rng: &mut Rng,
    compiled: &mut Compiled,
    bits: &mut [u64],
) -> f64 {
    const POINTS: usize = 4096;
    compiled.compile(labels, preds.get(qi), Weights::Unit, None, bits);
    if compiled.is_empty() {
        compiled.clear_bits(bits);
        return 0.0;
    }
    let cx = EvalCtx::new(labels, compiled, bits, &[], &[]);
    let n = labels.n_points();
    let hits = (0..POINTS)
        .filter(|_| RowScan::full(&cx, rng.below(n) as u32))
        .count();
    compiled.clear_bits(bits);
    hits as f64 / POINTS as f64
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
