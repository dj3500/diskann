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
//! With label-aware candidate generation ([`Candidates`], module `candidates`) the pool of step
//! 1 is the union of the label-blind search's pool and the pools of label-filtered searches, or
//! the pool of one penalized search. A node can then receive back-edges before its own
//! insertion (a label's start node is found by filtered searches before it is inserted); its
//! row at insertion is pruned from its pool together with those neighbors.
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
    candidates::{
        nn_scale, penalty_weight, test_bit, Candidates, FilterPlan, MemberMarks, Penalty,
        PenaltyScratch, PruneKey,
    },
    medoid_index,
    pools::Pools,
    prune::{fresh_sources, normalize_pool, reprune, Candidate, PruneRule, PruneScratch},
    search::{Ranking, Searcher},
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
    /// Candidate generation of the insertions (label-blind by default).
    pub candidates: Candidates,
    /// Sample size of the nearest-neighbor distance scale of penalized insertion.
    pub nn_sample: usize,
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
            candidates: Candidates::Blind,
            nn_sample: 256,
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
    /// Label-filtered insertion searches, their distance computations and member tests.
    pub n_filtered_searches: u64,
    pub n_dist_filtered: u64,
    pub n_member_tests: u64,
    /// Label-penalty evaluations (`f(p, c)`) of penalized insertion searches.
    pub n_penalty_evals: u64,
    /// Entries of the pools handed to the insertion prunes (after merging and repeats).
    pub n_pool_entries: u64,
    /// Insertions whose node already had neighbors (back-edges received before its
    /// insertion), and those neighbors.
    pub n_row_merges: u64,
    pub n_row_merge_entries: u64,
}

/// A built graph, its candidate pools (when recorded) and the build's counters.
#[derive(Debug)]
pub struct BuildOutput {
    pub graph: Graph,
    pub pools: Option<Pools>,
    pub stats: BuildStats,
    /// Setup of the candidate generation (filtered labels, penalty scale, ...).
    pub candidates: serde_json::Value,
}

struct Inserted {
    out: Vec<Candidate>,
    pool: Option<Vec<Candidate>>,
    n_dist: u64,
    n_expanded: u64,
    n_dist_prune: u64,
    n_filtered: u64,
    n_dist_filtered: u64,
    n_member_tests: u64,
    n_pool: u64,
    n_merged: u64,
}

/// Admits the members of one label (a bitset over the nodes).
struct Members<'b> {
    bits: &'b [u64],
}

impl Ranking for Members<'_> {
    #[inline(always)]
    fn admit(&mut self, v: u32) -> bool {
        test_bit(self.bits, v)
    }
}

/// Orders by `d² + λ·f(p, c)` for the point `p` marked in `scratch`.
struct Penalized<'b> {
    penalty: &'b Penalty<'b>,
    scratch: &'b PenaltyScratch,
}

impl Ranking for Penalized<'_> {
    #[inline(always)]
    fn prepare(&mut self, v: u32) {
        self.penalty.prefetch_row(v);
    }

    #[inline(always)]
    fn key(&mut self, v: u32, d: f32) -> f32 {
        self.penalty.key(self.scratch, v, d)
    }
}

/// Per-thread scratch of a build.
struct Scratch {
    searcher: Searcher,
    prune: PruneScratch,
    marks: MemberMarks,
    extra: Vec<Candidate>,
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
    let gen = params.candidates;
    let needs_labels = rule.needs_labels() || gen.needs_labels();
    let store = match labels {
        Some(store) if needs_labels => {
            ensure!(
                store.n_points() == n,
                "the label store has {} points, the base {n}",
                store.n_points()
            );
            Some(store)
        }
        None if needs_labels => anyhow::bail!("rule {rule} with candidates {gen} needs labels"),
        _ => None,
    };
    let label_info = store.filter(|_| rule.needs_labels()).map(LabelInfo::new);
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
    let t_setup = Instant::now();
    let (plan, penalty, mut gen_info) = match gen {
        Candidates::Blind => (None, None, json!({"kind": "blind"})),
        Candidates::Filtered {
            subset,
            budget,
            l_filter,
        } => {
            let store = store.expect("checked to have labels");
            let plan = FilterPlan::new(
                base,
                store,
                subset,
                budget.unwrap_or(rule.degree()),
                l_filter.unwrap_or(params.l_build),
                params.threads,
            )?;
            let mut info = gen.to_json();
            info["plan"] = plan.stats.clone();
            (Some(plan), None, info)
        }
        Candidates::Penalized { lambda, key } => {
            let store = store.expect("checked to have labels");
            let (scale, quartiles) = nn_scale(base, params.nn_sample, 0, &pool);
            let weight = penalty_weight(base, lambda, scale);
            let mut info = gen.to_json();
            info["nn_d2_median"] = json!(scale);
            info["nn_d2_quartiles"] = json!(quartiles);
            info["nn_sample"] = json!(params.nn_sample);
            info["weight"] = json!(weight);
            (None, Some(Penalty::new(store, weight, key)), info)
        }
    };
    gen_info["setup_s"] = json!(t_setup.elapsed().as_secs_f64());
    let filtered = plan.is_some();
    let scratch = PerThread::new(params.threads, || Scratch {
        searcher: Searcher::new(n_nodes),
        prune: PruneScratch::default(),
        marks: if filtered {
            MemberMarks::new(n_nodes)
        } else {
            MemberMarks::default()
        },
        extra: Vec::new(),
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
                scratch.with(|s| {
                    let Scratch {
                        searcher,
                        prune: prune_scratch,
                        marks,
                        extra,
                    } = s;
                    let query = points.get(p);
                    let penalized_key = penalty.as_ref().is_some_and(|pen| {
                        pen.begin(&mut prune_scratch.penalty, p);
                        pen.key == PruneKey::Penalized
                    });
                    let n_dist = match &penalty {
                        None => searcher.search(graph, &points, query, start, params.l_build),
                        Some(pen) => searcher.search_ranked(
                            graph,
                            &points,
                            query,
                            start,
                            params.l_build,
                            &mut Penalized {
                                penalty: pen,
                                scratch: &prune_scratch.penalty,
                            },
                        ),
                    };
                    let n_expanded = searcher.expanded.len() as u64;
                    let mut candidates = std::mem::take(&mut searcher.expanded);
                    if penalized_key {
                        for (c, &k) in candidates.iter_mut().zip(&searcher.expanded_keys) {
                            c.dist = k;
                        }
                    }
                    normalize_pool(&mut candidates, p, params.maxc);
                    let (mut n_filtered, mut n_dist_filtered, mut n_member_tests) = (0, 0, 0);
                    extra.clear();
                    if let Some(plan) = &plan {
                        let store = store.expect("checked to have labels");
                        for &l in plan.labels_of(p) {
                            let s = plan.starts[l as usize];
                            if s == u32::MAX {
                                continue;
                            }
                            let members = store.postings(l);
                            let own = plan.static_bits(l).is_none();
                            if own {
                                marks.set(members);
                            }
                            let bits = plan.static_bits(l).unwrap_or_else(|| marks.bits());
                            n_dist_filtered += searcher.search_ranked(
                                graph,
                                &points,
                                query,
                                s,
                                plan.l_filter,
                                &mut Members { bits },
                            );
                            if own {
                                marks.clear(members);
                            }
                            n_filtered += 1;
                            n_member_tests += searcher.n_admit;
                            let found = searcher.expanded.len().min(params.maxc);
                            extra.extend_from_slice(&searcher.expanded[..found]);
                        }
                    }
                    // Neighbors received before the insertion (filtered builds only).
                    let n_merged = graph.degree(p) as u64;
                    for &v in graph.neighbors(p) {
                        let d = T::squared_l2(query, points.get(v));
                        let d = match (&penalty, penalized_key) {
                            (Some(pen), true) => pen.key(&prune_scratch.penalty, v, d),
                            _ => d,
                        };
                        extra.push(Candidate::new(v, d));
                    }
                    if !extra.is_empty() {
                        candidates.extend_from_slice(extra);
                        normalize_pool(&mut candidates, p, usize::MAX);
                    }
                    if let Some(pen) = &penalty {
                        pen.end(&mut prune_scratch.penalty);
                    }
                    let n_pool = candidates.len() as u64;
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
                        n_dist: n_dist + n_merged,
                        n_expanded,
                        n_dist_prune,
                        n_filtered,
                        n_dist_filtered,
                        n_member_tests,
                        n_pool,
                        n_merged,
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
            stats.n_filtered_searches += ins.n_filtered;
            stats.n_dist_filtered += ins.n_dist_filtered;
            stats.n_member_tests += ins.n_member_tests;
            stats.n_pool_entries += ins.n_pool;
            stats.n_row_merges += u64::from(ins.n_merged > 0);
            stats.n_row_merge_entries += ins.n_merged;
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
                scratch.with(|s| {
                    let prune_scratch = &mut s.prune;
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
                        penalty.as_ref(),
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
            penalty.as_ref(),
            params.maxc,
            &pool,
            &scratch,
            |s| &mut s.prune,
        );
        stats.final_prune_s = t_final.elapsed().as_secs_f64();
    }

    let t_assemble = Instant::now();
    for s in scratch.into_inner() {
        stats.n_dist_reprune += s.prune.n_dist + s.prune.n_dist_source;
        stats.n_label_checks += s.prune.n_label_checks;
    }
    stats.n_dist_reprune -= stats.n_dist_prune;
    if penalty.is_some() {
        // every distance of a penalized search evaluated the penalty once
        stats.n_penalty_evals = stats.n_dist_search - stats.n_row_merge_entries;
    }
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
            "candidates": gen_info.clone(),
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
        candidates: gen_info,
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
    penalty: Option<&Penalty<'_>>,
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
                    penalty,
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
    use crate::labels::LabelStore;

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

    /// Random 8-d points with two categories: `cg` (8 values spread at random) and `cr`
    /// (`rare` on every 25th point, else `common`).
    fn labeled(n: usize, seed: u64) -> (Matrix<u8>, LabelStore) {
        let base = random_points(n, 8, seed);
        let mut rng = SplitMix64::new(seed + 1);
        let mut b = crate::labels::LabelStoreBuilder::new();
        for i in 0..n {
            let g = format!("g{}", rng.below(8));
            let r = if i % 25 == 0 { "rare" } else { "common" };
            b.push_point([("cg", g.as_str()), ("cr", r)]).unwrap();
        }
        (base, b.finish().unwrap())
    }

    /// Fraction of the members of `label` reachable from its first member's medoid start by
    /// edges between members.
    fn label_reach(graph: &Graph, store: &LabelStore, base: &Matrix<u8>, label: u32) -> f64 {
        let members = store.postings(label);
        let starts = crate::label_starts::build(
            base,
            store,
            crate::label_starts::Strategy::Medoid,
            &crate::label_starts::Params::default(),
        )
        .unwrap();
        let start = starts.starts[label as usize][0];
        let is_member = |v: u32| members.binary_search(&v).is_ok();
        let mut seen = vec![false; graph.n_nodes()];
        let mut stack = vec![start];
        seen[start as usize] = true;
        let mut reached = 0;
        while let Some(v) = stack.pop() {
            reached += 1;
            for &w in graph.neighbors(v) {
                if !seen[w as usize] && is_member(w) {
                    seen[w as usize] = true;
                    stack.push(w);
                }
            }
        }
        reached as f64 / members.len() as f64
    }

    /// Share of the data points' out-edges `p → c` with `c` sharing a label with `p` in every
    /// category of `p`.
    fn compatible_edge_share(graph: &Graph, store: &LabelStore) -> f64 {
        let (mut ok, mut all) = (0u64, 0u64);
        for p in 0..graph.n_data() as u32 {
            let fp = store.row(p);
            for &c in graph.neighbors(p) {
                if c as usize >= graph.n_data() {
                    continue;
                }
                let fc = store.row(c);
                let cats: std::collections::BTreeSet<u32> =
                    fp.iter().map(|&l| store.category(l)).collect();
                let sat = cats
                    .iter()
                    .filter(|&&k| {
                        fp.iter()
                            .any(|&l| store.category(l) == k && fc.contains(&l))
                    })
                    .count();
                ok += u64::from(sat == cats.len());
                all += 1;
            }
        }
        ok as f64 / all as f64
    }

    fn labeled_params(threads: usize, candidates: Candidates, rule: PruneRule) -> BuildParams {
        BuildParams {
            rule,
            l_build: 32,
            max_batch: 64,
            threads,
            candidates,
            nn_sample: 64,
            ..BuildParams::default()
        }
    }

    #[test]
    fn zero_penalty_reproduces_the_label_blind_build() {
        let (base, store) = labeled(1200, 3);
        let rule = PruneRule::Alpha { r: 16, alpha: 1.2 };
        let blind = build(&base, None, &labeled_params(2, Candidates::Blind, rule)).unwrap();
        for key in [PruneKey::Raw, PruneKey::Penalized] {
            let gen = Candidates::Penalized {
                lambda: crate::build::Lambda::Relative(0.0),
                key,
            };
            let pen = build(&base, Some(&store), &labeled_params(2, gen, rule)).unwrap();
            assert_eq!(pen.graph, blind.graph, "{key}");
            assert_eq!(pen.stats.n_row_merges, 0);
        }
    }

    #[test]
    fn filtered_insertion_makes_a_rare_label_navigable() {
        let (base, store) = labeled(2000, 7);
        let rare = store.name_index()["rare"];
        assert_eq!(store.posting_len(rare), 80);
        let r = 24;
        let alpha = PruneRule::Alpha { r, alpha: 1.2 };
        let blind = build(
            &base,
            Some(&store),
            &labeled_params(2, Candidates::Blind, alpha),
        )
        .unwrap();
        let gen = Candidates::Filtered {
            subset: crate::build::LabelSubset::Cap(100),
            budget: None,
            l_filter: None,
        };
        let sc = PruneRule::SourceContainment {
            r,
            alpha: 1.2,
            theta: Some(100),
            budget: None,
        };
        let one = build(&base, Some(&store), &labeled_params(1, gen, sc)).unwrap();
        let three = build(&base, Some(&store), &labeled_params(3, gen, sc)).unwrap();
        assert_eq!(one.graph, three.graph, "independent of the thread count");
        // Only the rare label is below the cap: one filtered search per rare point.
        assert_eq!(one.stats.n_filtered_searches, 80);
        assert_eq!(one.candidates["plan"]["labels_searched"], 1);
        let reach_blind = label_reach(&blind.graph, &store, &base, rare);
        let reach_filtered = label_reach(&one.graph, &store, &base, rare);
        assert!(
            reach_filtered >= 0.99 && reach_blind < 0.5,
            "rare label reach: blind {reach_blind}, filtered {reach_filtered}"
        );
        // Plain α-prune on the same pools also keeps some label edges, but fewer.
        let plain = build(&base, Some(&store), &labeled_params(2, gen, alpha)).unwrap();
        let reach_plain = label_reach(&plain.graph, &store, &base, rare);
        assert!(reach_plain <= reach_filtered && reach_plain > reach_blind);
    }

    #[test]
    fn lexicographic_penalty_links_label_compatible_points() {
        let (base, store) = labeled(1500, 11);
        let rule = PruneRule::Alpha { r: 16, alpha: 1.2 };
        let blind = build(
            &base,
            Some(&store),
            &labeled_params(2, Candidates::Blind, rule),
        )
        .unwrap();
        let gen = |key| Candidates::Penalized {
            lambda: crate::build::Lambda::Infinite,
            key,
        };
        let raw = build(
            &base,
            Some(&store),
            &labeled_params(2, gen(PruneKey::Raw), rule),
        )
        .unwrap();
        let pen1 = build(
            &base,
            Some(&store),
            &labeled_params(1, gen(PruneKey::Penalized), rule),
        )
        .unwrap();
        let pen3 = build(
            &base,
            Some(&store),
            &labeled_params(3, gen(PruneKey::Penalized), rule),
        )
        .unwrap();
        assert_eq!(pen1.graph, pen3.graph, "independent of the thread count");
        let (b, r, p) = (
            compatible_edge_share(&blind.graph, &store),
            compatible_edge_share(&raw.graph, &store),
            compatible_edge_share(&pen1.graph, &store),
        );
        assert!(
            b < r && r < p && p > 0.5,
            "compatible edges: blind {b}, raw {r}, penalized {p}"
        );
        assert!(pen1.stats.n_penalty_evals > 0);
        assert!(pen1.candidates["weight"].as_f64().unwrap() > 4.0 * 255.0 * 255.0 * 8.0);
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
