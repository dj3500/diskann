/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Per-label stitched subgraphs merged into a label-blind graph (`flab build --stitch`).
//!
//! For every label with at most `θ` points (and at least 2), a Vamana graph over the label's
//! points alone, as StitchedVamana's per-label graphs: degree `R_small`, list `L_small`,
//! α-RobustPrune, the label's points inserted one at a time in a seeded random order from the
//! label's medoid, back-edges appended up to `⌊1.3 · R_small⌋` entries or re-pruned. Then the row
//! of every data point that is in some per-label graph is rebuilt from its row in the label-blind
//! graph together with its out-neighbors in the per-label graphs of its labels: the union, in
//! pool order and truncated to `maxc` (as StitchedVamana's final prune), is pruned to `R` with the
//! given rule — one degree budget. There is no back-edge pass after the merge (as
//! StitchedVamana). Other rows (and auxiliary nodes) keep their label-blind rows.

use std::time::Instant;

use anyhow::{ensure, Context, Result};
use diskann_providers::utils::ParallelIteratorInPool;
use rayon::prelude::*;
use serde_json::{json, Value};

use super::{
    adjacency::Adjacency,
    builder::{insertion_order, slots, Order},
    candidates::LabelSubset,
    medoid_index,
    prune::{fresh_sources, normalize_pool, reprune, Candidate, PruneRule, PruneScratch},
    search::Searcher,
    thread_pool, BuildElement, LabelInfo, PerThread, Points,
};
use crate::{graph::Graph, labels::LabelStore, vectors::Matrix};

/// Parameters of [`stitch`].
#[derive(Debug, Clone)]
pub struct StitchParams {
    /// Labels that get a per-label graph (at least 2 points).
    pub subset: LabelSubset,
    pub r_small: usize,
    pub l_small: usize,
    /// The rule of the merge (its degree is the final degree bound).
    pub rule: PruneRule,
    pub maxc: usize,
    pub threads: usize,
    pub seed: u64,
}

/// Counters and timings of a stitch.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct StitchStats {
    pub wall_s: f64,
    pub label_graphs_s: f64,
    pub merge_s: f64,
    pub labels: u64,
    /// Points inserted over all per-label graphs (`Σ |P_ℓ|`).
    pub label_points: u64,
    pub label_edges: u64,
    pub n_dist_label_graphs: u64,
    /// Merged pool entries over all data points (after repeats and `maxc`).
    pub n_pool_entries: u64,
    pub n_dist_merge: u64,
    pub n_label_checks: u64,
}

/// Builds the per-label graph of `members` (global ids, sorted) and returns its rows as global
/// ids (row `i` belongs to `members[i]`), with the number of distance computations.
fn label_graph<T: BuildElement>(
    base: &Matrix<T>,
    members: &[u32],
    params: &StitchParams,
    label: u32,
) -> (Vec<Vec<u32>>, u64) {
    let m = members.len();
    let dim = base.ncols();
    let mut data = Vec::with_capacity(m * dim);
    for &p in members {
        data.extend_from_slice(base.row(p as usize));
    }
    let local = Matrix::from_slice(&data, m, dim);
    let points = Points::new(&local, &[]);
    let start = medoid_index(&local) as u32;
    let slots = slots(params.r_small, 1.3);
    let rule = PruneRule::Alpha {
        r: params.r_small,
        alpha: params.rule.alpha(),
    };
    let order = insertion_order(
        m,
        Order::Random(params.seed ^ u64::from(label).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
    );
    let mut adjacency = Adjacency::new(m, slots);
    let mut searcher = Searcher::new(m);
    let mut scratch = PruneScratch::default();
    let mut out = Vec::with_capacity(params.r_small);
    let mut fresh = Vec::new();
    let mut n_dist = 0u64;
    // the start first, then the others (the start must have a row before it is searched from)
    let mut seq = Vec::with_capacity(m);
    seq.push(start);
    seq.extend(order.into_iter().filter(|&i| i != start));
    for &i in &seq[1..] {
        n_dist += searcher.search(&adjacency, &points, points.get(i), start, params.l_small);
        let mut pool = std::mem::take(&mut searcher.expanded);
        normalize_pool(&mut pool, i, params.maxc);
        rule.prune(&points, None, i, &pool, &mut scratch, &mut out);
        searcher.expanded = pool;
        adjacency.set(i, out.iter().map(|c| c.id));
        for c in &out {
            let t = c.id;
            fresh_sources(adjacency.neighbors(t), std::iter::once(i), &mut fresh);
            if fresh.is_empty() {
                continue;
            }
            if adjacency.degree(t) < slots {
                adjacency.extend(t, &fresh);
            } else {
                let row: Vec<u32> = adjacency.neighbors(t).to_vec();
                let mut re = Vec::with_capacity(params.r_small);
                reprune(
                    &rule,
                    &points,
                    None,
                    None,
                    t,
                    &row,
                    &fresh,
                    params.maxc,
                    &mut scratch,
                    &mut re,
                );
                adjacency.set(t, re.iter().map(|c| c.id));
            }
        }
    }
    n_dist += scratch.n_dist + scratch.n_dist_source;
    let rows = (0..m as u32)
        .map(|i| {
            adjacency
                .neighbors(i)
                .iter()
                .map(|&j| members[j as usize])
                .collect()
        })
        .collect();
    (rows, n_dist)
}

/// Merges per-label graphs of the labels of `params.subset` into `blind` (see the module
/// documentation). `blind` must be a graph over the rows of `base`.
pub fn stitch<T: BuildElement>(
    base: &Matrix<T>,
    store: &LabelStore,
    blind: &Graph,
    params: &StitchParams,
) -> Result<(Graph, StitchStats, Value)> {
    let t0 = Instant::now();
    let n = base.nrows();
    ensure!(
        blind.n_data() == n && store.n_points() == n,
        "the graph has {} data points, the base {n}, the labels {}",
        blind.n_data(),
        store.n_points()
    );
    let theta = params.subset.theta(n);
    let mut labels: Vec<u32> = (0..store.n_labels() as u32)
        .filter(|&l| {
            let len = store.posting_len(l);
            len >= 2 && theta.is_none_or(|t| len <= t)
        })
        .collect();
    // largest first, for load balance
    labels.sort_unstable_by_key(|&l| std::cmp::Reverse(store.posting_len(l)));
    let pool = thread_pool(params.threads)?;
    let mut stats = StitchStats {
        labels: labels.len() as u64,
        label_points: labels.iter().map(|&l| store.posting_len(l) as u64).sum(),
        ..StitchStats::default()
    };

    let t_labels = Instant::now();
    let graphs: Vec<(Vec<Vec<u32>>, u64)> = labels
        .par_iter()
        .map(|&l| label_graph(base, store.postings(l), params, l))
        .collect_in_pool(pool.as_ref());
    // per-point candidate lists (CSR by a counting sort over every label graph)
    let mut counts = vec![0u64; n + 1];
    for (&l, (rows, _)) in labels.iter().zip(&graphs) {
        for (&p, row) in store.postings(l).iter().zip(rows) {
            counts[p as usize + 1] += row.len() as u64;
        }
    }
    for p in 0..n {
        counts[p + 1] += counts[p];
    }
    let mut fill = counts.clone();
    let mut nbrs = vec![0u32; counts[n] as usize];
    for (&l, (rows, nd)) in labels.iter().zip(&graphs) {
        stats.n_dist_label_graphs += nd;
        for (&p, row) in store.postings(l).iter().zip(rows) {
            let at = fill[p as usize] as usize;
            nbrs[at..at + row.len()].copy_from_slice(row);
            fill[p as usize] += row.len() as u64;
        }
    }
    drop(fill);
    drop(graphs);
    stats.label_edges = nbrs.len() as u64;
    stats.label_graphs_s = t_labels.elapsed().as_secs_f64();

    let t_merge = Instant::now();
    let aux = blind
        .aux()
        .vectors::<T>()
        .context("the graph's auxiliary vectors are not of the base's element type")?
        .to_vec();
    let points = Points::new(base, &aux);
    let info = params.rule.needs_labels().then(|| LabelInfo::new(store));
    let scratch = PerThread::new(params.threads, || {
        (Vec::<Candidate>::new(), PruneScratch::default())
    });
    let rule = params.rule;
    let merged: Vec<(Vec<u32>, u64)> = (0..n as u32)
        .into_par_iter()
        .map(|p| {
            scratch.with(|(cands, prune_scratch)| {
                let extra = &nbrs[counts[p as usize] as usize..counts[p as usize + 1] as usize];
                if extra.is_empty() {
                    return (blind.neighbors(p).to_vec(), 0);
                }
                cands.clear();
                let q = points.get(p);
                for &v in blind.neighbors(p).iter().chain(extra) {
                    cands.push(Candidate::new(v, T::squared_l2(q, points.get(v))));
                }
                prune_scratch.n_dist_source += cands.len() as u64;
                normalize_pool(cands, p, params.maxc);
                let mut out = Vec::with_capacity(rule.degree());
                rule.prune(&points, info.as_ref(), p, cands, prune_scratch, &mut out);
                (
                    out.iter().map(|c| c.id).collect::<Vec<u32>>(),
                    cands.len() as u64,
                )
            })
        })
        .collect_in_pool(pool.as_ref());
    let rows: Vec<Vec<u32>> = merged
        .into_iter()
        .map(|(row, len)| {
            stats.n_pool_entries += len;
            row
        })
        .collect();
    for (_, s) in scratch.into_inner() {
        stats.n_dist_merge += s.n_dist + s.n_dist_source;
        stats.n_label_checks += s.n_label_checks;
    }
    let slots = blind.r_max().max(rule.degree());
    let mut adjacency = Adjacency::new(blind.n_nodes(), slots);
    for (p, row) in rows.iter().enumerate() {
        adjacency.set(p as u32, row.iter().copied());
    }
    for v in n as u32..blind.n_nodes() as u32 {
        adjacency.set(v, blind.neighbors(v).iter().copied());
    }
    let graph = adjacency.into_graph(n, blind.aux().clone(), blind.starts().to_vec())?;
    stats.merge_s = t_merge.elapsed().as_secs_f64();
    stats.wall_s = t0.elapsed().as_secs_f64();
    let info = json!({
        "kind": "stitched per-label graphs merged into a label-blind graph",
        "subset": params.subset.to_string(),
        "theta": theta,
        "r_small": params.r_small,
        "l_small": params.l_small,
        "merge_rule": rule.to_string(),
        "maxc": params.maxc,
        "seed": params.seed,
    });
    Ok((graph, stats, info))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{
        build, builder::BuildParams, test_util::random_points, BudgetOrder, Candidates, SplitMix64,
    };

    #[test]
    fn stitching_makes_a_rare_label_navigable_and_keeps_the_blind_rows() {
        let n = 2000;
        let base = random_points(n, 8, 7);
        let mut rng = SplitMix64::new(8);
        let mut b = crate::labels::LabelStoreBuilder::new();
        for i in 0..n {
            let g = format!("g{}", rng.below(8));
            let r = if i % 25 == 0 { "rare" } else { "common" };
            b.push_point([("cg", g.as_str()), ("cr", r)]).unwrap();
        }
        let store = b.finish().unwrap();
        let rare = store.name_index()["rare"];
        let params = BuildParams {
            rule: PruneRule::Alpha { r: 24, alpha: 1.2 },
            l_build: 32,
            max_batch: 64,
            threads: 2,
            candidates: Candidates::Blind,
            ..BuildParams::default()
        };
        let blind = build(&base, None, &params).unwrap().graph;
        let stitch_params = |threads| StitchParams {
            subset: LabelSubset::Cap(100),
            r_small: 8,
            l_small: 16,
            rule: PruneRule::SourceContainment {
                r: 24,
                alpha: 1.2,
                theta: Some(100),
                budget: None,
                order: BudgetOrder::Rarest,
            },
            maxc: 750,
            threads,
            seed: 0,
        };
        let (g1, stats, _) = stitch(&base, &store, &blind, &stitch_params(1)).unwrap();
        let (g3, _, _) = stitch(&base, &store, &blind, &stitch_params(3)).unwrap();
        assert_eq!(g1, g3, "independent of the thread count");
        assert_eq!(stats.labels, 1);
        assert_eq!(stats.label_points, 80);
        assert_eq!(g1.n_nodes(), blind.n_nodes());
        assert_eq!(g1.starts(), blind.starts());
        // rare members reach each other through rare members only
        let members = store.postings(rare);
        let is_member = |v: u32| members.binary_search(&v).is_ok();
        let reach = |g: &Graph| {
            let mut seen = vec![false; g.n_nodes()];
            let mut stack = vec![members[0]];
            seen[members[0] as usize] = true;
            let mut k = 0;
            while let Some(v) = stack.pop() {
                k += 1;
                for &w in g.neighbors(v) {
                    if !seen[w as usize] && is_member(w) {
                        seen[w as usize] = true;
                        stack.push(w);
                    }
                }
            }
            k as f64 / members.len() as f64
        };
        assert!(
            reach(&g1) >= 0.99 && reach(&blind) < 0.5,
            "reach blind {}, stitched {}",
            reach(&blind),
            reach(&g1)
        );
        // rows outside every per-label graph are the label-blind rows; members are pruned to R
        for p in 0..n as u32 {
            if is_member(p) {
                assert!(g1.neighbors(p).len() <= 24);
            } else {
                assert_eq!(g1.neighbors(p), blind.neighbors(p));
            }
        }
    }
}
