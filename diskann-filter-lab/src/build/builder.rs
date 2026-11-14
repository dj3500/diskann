/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Parallel Vamana construction with batch-parallel insertion.
//!
//! The data points are inserted in a fixed order, in batches whose sizes double from 1 up to
//! `max_batch` (prefix doubling) and then stay at `max_batch`. For every batch:
//!
//! 1. *Search and prune* (parallel over the batch): each point runs a greedy search with list
//!    size `L_build` from the frozen start node over the graph as it was before the batch;
//!    the expanded nodes, in pool order and truncated to `maxc`, are its candidate pool,
//!    which the pruning rule reduces to its out-neighbors.
//! 2. *Update*: the new rows are set; every new edge `p → t` proposes the back-edge `t → p`.
//!    Per target, the proposals of the batch (sorted by source id) are appended to its row
//!    when the row stays within `slots = ⌊slack · R⌋` entries; otherwise the row and the new
//!    sources are re-pruned to `R` with the same rule (parallel over such targets). This is
//!    upstream's `multi_insert` with `IntraBatchCandidates::None` and `add_edge_and_prune`.
//!
//! A batch therefore depends only on the graph before it, and the result only on the
//! insertion order and the batch schedule, not on the number of threads. The start node is
//! an auxiliary node holding a copy of the medoid (upstream's `StartPointStrategy::Medoid`
//! with one frozen point); it is a regular graph node that can be a neighbor. Prefix
//! doubling replaces upstream's multi-insert bootstrap: the first points are inserted
//! (almost) one at a time.

use std::time::Instant;

use anyhow::{ensure, Result};
use diskann_providers::utils::ParallelIteratorInPool;
use rayon::prelude::*;
use serde::Serialize;
use serde_json::json;

use super::{
    adjacency::Adjacency,
    medoid_index,
    pools::Pools,
    prune::{fresh_sources, normalize_pool, reprune, Candidate, PruneRule, PruneScratch},
    search::Searcher,
    thread_pool, BuildElement, LabelInfo, PerThread, Points, SplitMix64,
};
use crate::{
    graph::{AuxKind, AuxNodes, Graph},
    labels::LabelStore,
    vectors::{Element, Matrix},
};

/// Insertion order of the data points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// Ids in increasing order.
    Identity,
    /// A seeded uniform permutation (Fisher–Yates with SplitMix64).
    Random(u64),
}

/// Parameters of a build.
#[derive(Debug, Clone)]
pub struct BuildParams {
    /// Pruning rule of the insertions and back-edge re-prunes; its degree is `R`.
    pub rule: PruneRule,
    /// Candidate list size of the insertion search.
    pub l_build: usize,
    /// Rows hold up to `⌊slack · R⌋` neighbors before a back-edge triggers a re-prune.
    pub slack: f32,
    /// Pool truncation (upstream's maximum occlusion size).
    pub maxc: usize,
    /// Largest batch of the schedule.
    pub max_batch: usize,
    pub order: Order,
    pub threads: usize,
    /// Record every node's candidate pool ([`Pools`]).
    pub capture: bool,
    /// Re-prune every row longer than `R` at the end (the C++ library's cleanup pass;
    /// upstream's in-memory build does not).
    pub final_prune: bool,
}

impl Default for BuildParams {
    fn default() -> Self {
        Self {
            rule: PruneRule::Alpha { r: 64, alpha: 1.2 },
            l_build: 100,
            slack: 1.3,
            maxc: 750,
            max_batch: 128,
            order: Order::Random(0),
            threads: 1,
            capture: false,
            final_prune: false,
        }
    }
}

impl BuildParams {
    /// Row capacity: `⌊slack · R⌋` (at least `R`).
    pub fn slots(&self) -> usize {
        slots(self.rule.degree(), self.slack)
    }
}

pub(crate) fn slots(degree: usize, slack: f32) -> usize {
    ((degree as f32 * slack) as usize).max(degree)
}

/// Batch boundaries over `n` insertions: sizes 1, 2, 4, … up to `max_batch`, then
/// `max_batch`.
pub fn schedule(n: usize, max_batch: usize) -> Vec<u64> {
    let mut bounds = vec![0u64];
    let (mut at, mut size) = (0usize, 1usize);
    while at < n {
        at = (at + size.min(max_batch.max(1))).min(n);
        bounds.push(at as u64);
        size = size.saturating_mul(2);
    }
    bounds
}

/// The insertion order of `n` points.
pub fn insertion_order(n: usize, order: Order) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..n as u32).collect();
    if let Order::Random(seed) = order {
        let mut rng = SplitMix64::new(seed);
        for i in (1..n).rev() {
            let j = rng.below(i as u64 + 1) as usize;
            ids.swap(i, j);
        }
    }
    ids
}

/// Counters and timings of a build.
#[derive(Debug, Clone, Default, Serialize)]
pub struct BuildStats {
    /// Wall time of the whole build (medoid, batches, final prune, assembly).
    pub wall_s: f64,
    pub medoid_s: f64,
    /// Parallel search-and-prune phases.
    pub search_prune_s: f64,
    /// Update phases (row writes, back-edges, re-prunes).
    pub update_s: f64,
    pub final_prune_s: f64,
    pub assemble_s: f64,
    pub n_batches: usize,
    pub medoid: u32,
    /// Distance computations of the insertion searches.
    pub n_dist_search: u64,
    /// Expanded nodes of the insertion searches (= search-pool entries before truncation).
    pub n_expanded: u64,
    /// Occlusion distance computations of the insertion prunes.
    pub n_dist_prune: u64,
    /// Distance computations of back-edge re-prunes (and the final prune): node to
    /// candidates, plus occlusion tests.
    pub n_dist_reprune: u64,
    pub n_label_checks: u64,
    /// Targets whose row took a batch's back-edges by appending.
    pub n_appends: u64,
    /// Targets re-pruned because a batch's back-edges would overflow the row.
    pub n_reprunes: u64,
    pub n_final_prunes: u64,
}

/// A built graph, its candidate pools (when recorded) and the build's counters.
#[derive(Debug)]
pub struct BuildOutput {
    pub graph: Graph,
    pub pools: Option<Pools>,
    pub stats: BuildStats,
}

struct Inserted {
    out: Vec<Candidate>,
    pool: Option<Vec<Candidate>>,
    n_dist: u64,
    n_expanded: u64,
    n_dist_prune: u64,
}

/// Builds a graph over the rows of `base` (see the module documentation). `labels` is needed
/// by label-aware rules.
pub fn build<T: BuildElement>(
    base: &Matrix<T>,
    labels: Option<&LabelStore>,
    params: &BuildParams,
) -> Result<BuildOutput> {
    let t0 = Instant::now();
    let n = base.nrows();
    ensure!(n > 0 && base.ncols() > 0, "empty base");
    ensure!(n < u32::MAX as usize - 1, "too many points for 32-bit ids");
    ensure!(
        params.l_build >= 1 && params.maxc >= 1 && params.max_batch >= 1 && params.threads >= 1,
        "L_build, maxc, the batch size and the thread count must be positive"
    );
    let rule = params.rule;
    let label_info = match labels {
        Some(store) if rule.needs_labels() => {
            ensure!(
                store.n_points() == n,
                "the label store has {} points, the base {n}",
                store.n_points()
            );
            Some(LabelInfo::new(store))
        }
        None if rule.needs_labels() => anyhow::bail!("rule {rule} needs labels"),
        _ => None,
    };
    let labels = label_info.as_ref();

    let mut stats = BuildStats::default();
    let medoid = medoid_index(base);
    stats.medoid = medoid as u32;
    let start_vector = base.row(medoid).to_vec();
    stats.medoid_s = t0.elapsed().as_secs_f64();
    let points = Points::new(base, &start_vector);
    let (start, n_nodes, slots) = (n as u32, n + 1, params.slots());
    let order = insertion_order(n, params.order);
    let batches = schedule(n, params.max_batch);
    stats.n_batches = batches.len() - 1;

    let pool = thread_pool(params.threads)?;
    let scratch = PerThread::new(params.threads, || {
        (Searcher::new(n_nodes), PruneScratch::default())
    });
    let mut adjacency = Adjacency::new(n_nodes, slots);
    let mut search_pools: Vec<Vec<Candidate>> = Vec::new();
    let mut proposals: Vec<Vec<Candidate>> = Vec::new();
    if params.capture {
        search_pools.resize(n, Vec::new());
        proposals.resize(n_nodes, Vec::new());
    }
    let mut edges: Vec<(u32, u32, f32)> = Vec::new();
    let mut jobs: Vec<(u32, usize, usize)> = Vec::new();
    let mut fresh = Vec::new();

    for b in 0..stats.n_batches {
        let (lo, hi) = (batches[b] as usize, batches[b + 1] as usize);
        let t_search = Instant::now();
        let graph = &adjacency;
        let inserted: Vec<Inserted> = (lo..hi)
            .into_par_iter()
            .map(|i| {
                let p = order[i];
                scratch.with(|(searcher, prune_scratch)| {
                    let n_dist =
                        searcher.search(graph, &points, points.get(p), start, params.l_build);
                    let n_expanded = searcher.expanded.len() as u64;
                    let mut candidates = std::mem::take(&mut searcher.expanded);
                    normalize_pool(&mut candidates, p, params.maxc);
                    let before = prune_scratch.n_dist;
                    let mut out = Vec::with_capacity(rule.degree());
                    rule.prune(&points, labels, p, &candidates, prune_scratch, &mut out);
                    let n_dist_prune = prune_scratch.n_dist - before;
                    let pool = if params.capture {
                        Some(candidates)
                    } else {
                        searcher.expanded = candidates;
                        None
                    };
                    Inserted {
                        out,
                        pool,
                        n_dist,
                        n_expanded,
                        n_dist_prune,
                    }
                })
            })
            .collect_in_pool(pool.as_ref());
        stats.search_prune_s += t_search.elapsed().as_secs_f64();

        let t_update = Instant::now();
        edges.clear();
        for (k, ins) in inserted.iter().enumerate() {
            let p = order[lo + k];
            adjacency.set(p, ins.out.iter().map(|c| c.id));
            edges.extend(ins.out.iter().map(|c| (c.id, p, c.dist)));
            stats.n_dist_search += ins.n_dist;
            stats.n_expanded += ins.n_expanded;
            stats.n_dist_prune += ins.n_dist_prune;
        }
        if params.capture {
            for (k, ins) in inserted.into_iter().enumerate() {
                search_pools[order[lo + k] as usize] = ins.pool.unwrap_or_default();
            }
        }
        edges.sort_unstable_by_key(|e| (e.0, e.1));
        jobs.clear();
        let mut i = 0;
        while i < edges.len() {
            let t = edges[i].0;
            let j = i + edges[i..].iter().take_while(|e| e.0 == t).count();
            fresh_sources(
                adjacency.neighbors(t),
                edges[i..j].iter().map(|e| e.1),
                &mut fresh,
            );
            if !fresh.is_empty() {
                if adjacency.degree(t) + fresh.len() <= slots {
                    adjacency.extend(t, &fresh);
                    stats.n_appends += 1;
                } else {
                    jobs.push((t, i, j));
                }
            }
            if params.capture {
                proposals[t as usize].extend(edges[i..j].iter().map(|e| Candidate::new(e.1, e.2)));
            }
            i = j;
        }
        let graph = &adjacency;
        let rows: Vec<Vec<u32>> = jobs
            .par_iter()
            .map(|&(t, i, j)| {
                scratch.with(|(_, prune_scratch)| {
                    let mut fresh = std::mem::take(&mut prune_scratch.fresh);
                    fresh_sources(
                        graph.neighbors(t),
                        edges[i..j].iter().map(|e| e.1),
                        &mut fresh,
                    );
                    let mut out = Vec::with_capacity(rule.degree());
                    reprune(
                        &rule,
                        &points,
                        labels,
                        t,
                        graph.neighbors(t),
                        &fresh,
                        params.maxc,
                        prune_scratch,
                        &mut out,
                    );
                    prune_scratch.fresh = fresh;
                    out.iter().map(|c| c.id).collect()
                })
            })
            .collect_in_pool(pool.as_ref());
        for (&(t, _, _), row) in jobs.iter().zip(&rows) {
            adjacency.set(t, row.iter().copied());
        }
        stats.n_reprunes += jobs.len() as u64;
        stats.update_s += t_update.elapsed().as_secs_f64();
    }

    if params.final_prune {
        let t_final = Instant::now();
        stats.n_final_prunes = final_prune(
            &mut adjacency,
            &rule,
            &points,
            labels,
            params.maxc,
            &pool,
            &scratch,
            |s| &mut s.1,
        );
        stats.final_prune_s = t_final.elapsed().as_secs_f64();
    }

    let t_assemble = Instant::now();
    for (_, s) in scratch.into_inner() {
        stats.n_dist_reprune += s.n_dist + s.n_dist_source;
        stats.n_label_checks += s.n_label_checks;
    }
    stats.n_dist_reprune -= stats.n_dist_prune;
    let dim = base.ncols();
    let aux = AuxNodes::new(
        vec![AuxKind::Frozen],
        &Matrix::from_slice(&start_vector, 1, dim),
    )?;
    let graph = adjacency.into_graph(n, aux, vec![start])?;
    let pools = if params.capture {
        let meta = json!({
            "builder": "flab build (batch-parallel Vamana)",
            "rule": rule.to_string(),
            "l_build": params.l_build,
            "slack": params.slack,
            "slots": slots,
            "maxc": params.maxc,
            "max_batch": params.max_batch,
            "n_batches": stats.n_batches,
            "order": match params.order {
                Order::Identity => json!("identity"),
                Order::Random(seed) => json!({"random": seed}),
            },
            "final_prune": params.final_prune,
            "start": {"node": start, "medoid_of": medoid},
        });
        Some(assemble_pools(
            n,
            order,
            batches,
            search_pools,
            proposals,
            slots,
            params.maxc,
            bytemuck::cast_slice(&start_vector).to_vec(),
            T::DTYPE,
            dim,
            meta.to_string(),
        ))
    } else {
        None
    };
    stats.assemble_s = t_assemble.elapsed().as_secs_f64();
    stats.wall_s = t0.elapsed().as_secs_f64();
    Ok(BuildOutput {
        graph,
        pools,
        stats,
    })
}

/// Re-prunes every row longer than the rule's degree; returns the number of such rows.
/// `prune_scratch` selects the prune scratch within a thread's scratch.
#[expect(
    clippy::too_many_arguments,
    reason = "internal step shared by build and replay"
)]
pub(crate) fn final_prune<T: Element, S: Send>(
    adjacency: &mut Adjacency,
    rule: &PruneRule,
    points: &Points<'_, T>,
    labels: Option<&LabelInfo<'_>>,
    maxc: usize,
    pool: &diskann_providers::utils::RayonThreadPool,
    scratch: &PerThread<S>,
    prune_scratch: impl Fn(&mut S) -> &mut PruneScratch + Sync,
) -> u64 {
    let degree = rule.degree();
    let targets: Vec<u32> = (0..adjacency.n_nodes() as u32)
        .filter(|&v| adjacency.degree(v) > degree)
        .collect();
    let graph = &*adjacency;
    let rows: Vec<Vec<u32>> = targets
        .par_iter()
        .map(|&t| {
            scratch.with(|s| {
                let mut out = Vec::with_capacity(degree);
                let prune_scratch = prune_scratch(s);
                reprune(
                    rule,
                    points,
                    labels,
                    t,
                    graph.neighbors(t),
                    &[],
                    maxc,
                    prune_scratch,
                    &mut out,
                );
                out.iter().map(|c| c.id).collect()
            })
        })
        .collect_in_pool(pool.as_ref());
    for (&t, row) in targets.iter().zip(&rows) {
        adjacency.set(t, row.iter().copied());
    }
    targets.len() as u64
}

#[expect(
    clippy::too_many_arguments,
    reason = "assembles every field of the pools"
)]
fn assemble_pools(
    n_data: usize,
    order: Vec<u32>,
    batches: Vec<u64>,
    search_pools: Vec<Vec<Candidate>>,
    proposals: Vec<Vec<Candidate>>,
    slots: usize,
    maxc: usize,
    aux_bytes: Vec<u8>,
    dtype: crate::vectors::DType,
    dim: usize,
    meta: String,
) -> Pools {
    let n_nodes = proposals.len();
    let total: usize = search_pools.iter().map(Vec::len).sum::<usize>()
        + proposals.iter().map(Vec::len).sum::<usize>();
    let mut offsets = Vec::with_capacity(n_nodes + 1);
    let mut n_search = Vec::with_capacity(n_nodes);
    let mut ids = Vec::with_capacity(total);
    let mut dists = Vec::with_capacity(total);
    offsets.push(0u64);
    let mut search = search_pools.into_iter();
    for prop in proposals {
        let s = search.next().unwrap_or_default();
        n_search.push(s.len() as u32);
        for c in s.iter().chain(&prop) {
            ids.push(c.id);
            dists.push(c.dist);
        }
        offsets.push(ids.len() as u64);
    }
    Pools {
        n_data,
        order,
        batches,
        offsets,
        n_search,
        ids,
        dists,
        slots,
        maxc,
        dtype,
        dim,
        aux_bytes,
        meta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::test_util::random_points;

    #[test]
    fn schedule_doubles_then_saturates() {
        assert_eq!(schedule(0, 4), vec![0]);
        assert_eq!(schedule(5, 4), vec![0, 1, 3, 5]);
        assert_eq!(schedule(20, 4), vec![0, 1, 3, 7, 11, 15, 19, 20]);
        let order = insertion_order(10, Order::Random(3));
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..10).collect::<Vec<u32>>());
        assert_ne!(order, sorted);
        assert_eq!(order, insertion_order(10, Order::Random(3)));
    }

    #[test]
    fn build_is_independent_of_the_thread_count() {
        let base = random_points(1500, 8, 11);
        let params = |threads| BuildParams {
            rule: PruneRule::Alpha { r: 12, alpha: 1.2 },
            l_build: 24,
            max_batch: 64,
            threads,
            ..BuildParams::default()
        };
        let one = build(&base, None, &params(1)).unwrap();
        let three = build(&base, None, &params(3)).unwrap();
        assert_eq!(one.graph, three.graph);
        let g = &one.graph;
        assert_eq!((g.n_data(), g.n_nodes(), g.r_max()), (1500, 1501, 15));
        assert_eq!(g.starts(), &[1500]);
        let stats = g.stats();
        assert_eq!(stats.reachable_data, 1500);
        assert_eq!((stats.self_loops, stats.duplicate_edges), (0, 0));
        assert!(one.stats.n_reprunes > 0 && one.stats.n_appends > 0);
    }

    #[test]
    fn build_finds_near_neighbors() {
        let base = random_points(2000, 8, 5);
        let params = BuildParams {
            rule: PruneRule::Alpha { r: 16, alpha: 1.2 },
            l_build: 32,
            max_batch: 32,
            threads: 2,
            ..BuildParams::default()
        };
        let out = build(&base, None, &params).unwrap();
        // Search with a larger list from the start: the true nearest neighbor of each probe
        // must be found almost always.
        let aux = out.graph.aux().vectors::<u8>().unwrap().to_vec();
        let points = Points::new(&base, &aux);
        let mut adjacency = Adjacency::new(out.graph.n_nodes(), out.graph.r_max());
        for v in 0..out.graph.n_nodes() as u32 {
            adjacency.set(v, out.graph.neighbors(v).iter().copied());
        }
        let mut searcher = Searcher::new(out.graph.n_nodes());
        let mut found = 0;
        for q in (0..2000u32).step_by(40) {
            let query: Vec<u8> = base
                .row(q as usize)
                .iter()
                .map(|x| x.wrapping_add(1))
                .collect();
            let truth = (0..2000u32)
                .min_by(|&a, &b| {
                    u8::squared_l2(&query, base.row(a as usize))
                        .total_cmp(&u8::squared_l2(&query, base.row(b as usize)))
                        .then(a.cmp(&b))
                })
                .unwrap();
            searcher.search(&adjacency, &points, &query, 2000, 64);
            let d_truth = u8::squared_l2(&query, base.row(truth as usize));
            if searcher.expanded.iter().any(|c| c.dist <= d_truth) {
                found += 1;
            }
        }
        assert!(found >= 48, "found {found} of 50");
    }
}
