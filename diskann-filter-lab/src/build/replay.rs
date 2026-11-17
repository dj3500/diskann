/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Prune replay: the adjacency rebuilt from recorded candidate pools with a pruning rule,
//! without any search.
//!
//! - [`ReplayMode::Incremental`] replays the build's insertion process: every point's
//!   out-neighbors are the rule applied to its recorded search pool; the back-edges they
//!   propose reach each target batch by batch, by source id within a batch, and are appended
//!   while the row stays within `⌊slack · R⌋` entries, else the row and the new sources are
//!   re-pruned with the rule, exactly as in the builder. With the rule of the build this
//!   reproduces the build's graph exactly. With another rule it is a proxy for an
//!   incremental build with that rule: the search pools stay those of the recorded build
//!   (which navigated the recorded graph).
//! - [`ReplayMode::Union`] prunes every node once over its whole recorded pool (search pool
//!   and back-edge proposals, without repeated ids), with no back-edges afterwards: a
//!   stitch-then-prune construction over the recorded pools.
//!
//! In the incremental mode a target's row evolves only through its own back-edges, and the
//! out-neighbors of every point are fixed by its recorded pool, so all initial prunes run in
//! parallel and every target's back-edge sequence is replayed independently.

use std::time::Instant;

use anyhow::{ensure, Result};
use diskann_providers::utils::ParallelIteratorInPool;
use rayon::prelude::*;
use serde::Serialize;

use super::{
    adjacency::Adjacency,
    builder::{final_prune, slots},
    pools::Pools,
    prune::{fresh_sources, normalize_pool, reprune, Candidate, PruneRule, PruneScratch},
    thread_pool, LabelInfo, PerThread, Points,
};
use crate::{
    graph::{AuxKind, AuxNodes, Graph},
    labels::LabelStore,
    vectors::{Element, Matrix},
};

/// How a replay uses the recorded pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayMode {
    /// The build's insertion process on the recorded search pools (back-edges included).
    Incremental,
    /// One prune of each node's whole recorded pool, no back-edges.
    Union,
}

impl ReplayMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Incremental => "incremental",
            Self::Union => "union",
        }
    }
}

/// Parameters of a replay.
#[derive(Debug, Clone)]
pub struct ReplayParams {
    pub rule: PruneRule,
    pub mode: ReplayMode,
    /// Row capacity `⌊slack · R⌋` of the incremental mode (use the build's to replay it).
    pub slack: f32,
    /// Pool truncation (at most the build's).
    pub maxc: usize,
    pub threads: usize,
    /// Re-prune rows longer than `R` at the end (incremental mode).
    pub final_prune: bool,
}

/// Counters and timings of a replay.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ReplayStats {
    pub wall_s: f64,
    /// Initial prunes (incremental) or the single prune of every node (union).
    pub prune_s: f64,
    /// Back-edge replay (incremental).
    pub backedge_s: f64,
    pub final_prune_s: f64,
    pub assemble_s: f64,
    /// Occlusion distance computations of the prunes over the recorded pools.
    pub n_dist_prune: u64,
    /// Distance computations of back-edge re-prunes and the final prune.
    pub n_dist_reprune: u64,
    pub n_label_checks: u64,
    pub n_appends: u64,
    pub n_reprunes: u64,
    pub n_final_prunes: u64,
    /// Candidates handed to the prunes over recorded pools.
    pub n_pool_entries: u64,
}

/// A replayed graph and its counters.
#[derive(Debug)]
pub struct ReplayOutput {
    pub graph: Graph,
    pub stats: ReplayStats,
}

/// Rebuilds the adjacency from `pools` with `params.rule` (see the module documentation).
/// `base` must be the build's base vectors; `labels` is needed by label-aware rules.
pub fn replay<T: Element>(
    pools: &Pools,
    base: &Matrix<T>,
    labels: Option<&LabelStore>,
    params: &ReplayParams,
) -> Result<ReplayOutput> {
    let t0 = Instant::now();
    let n = pools.n_data();
    ensure!(
        base.nrows() == n && base.ncols() == pools.dim(),
        "the base is {} × {}, the pools were recorded on {n} × {}",
        base.nrows(),
        base.ncols(),
        pools.dim()
    );
    ensure!(
        params.maxc >= 1 && params.threads >= 1,
        "maxc and the thread count must be positive"
    );
    let rule = params.rule;
    let label_info = match labels {
        Some(store) if rule.needs_labels() => {
            ensure!(
                store.n_points() == n,
                "the label store has {} points, the pools {n}",
                store.n_points()
            );
            Some(LabelInfo::new(store))
        }
        None if rule.needs_labels() => anyhow::bail!("rule {rule} needs labels"),
        _ => None,
    };
    let labels = label_info.as_ref();
    let aux = pools.aux_vectors::<T>()?;
    let points = Points::new(base, &aux);
    let n_nodes = pools.n_nodes();
    ensure!(
        points.n_nodes() == n_nodes,
        "auxiliary vectors do not match the node count"
    );
    let maxc = params.maxc;
    let pool = thread_pool(params.threads)?;
    let scratch = PerThread::new(params.threads, || {
        (Vec::<Candidate>::new(), PruneScratch::default())
    });
    let mut stats = ReplayStats::default();

    // Prunes over recorded pools: the search pools of the data points (incremental) or the
    // whole rows of all nodes (union).
    let t_prune = Instant::now();
    let pruned_nodes = match params.mode {
        ReplayMode::Incremental => n,
        ReplayMode::Union => n_nodes,
    };
    let outs: Vec<Vec<u32>> = (0..pruned_nodes as u32)
        .into_par_iter()
        .map(|v| {
            scratch.with(|(candidates, prune_scratch)| {
                candidates.clear();
                let (ids, dists) = match params.mode {
                    ReplayMode::Incremental => pools.search(v),
                    ReplayMode::Union => pools.row(v),
                };
                candidates.extend(ids.iter().zip(dists).map(|(&i, &d)| Candidate::new(i, d)));
                match params.mode {
                    // Recorded search pools are already in pool order without repeats.
                    ReplayMode::Incremental => candidates.truncate(maxc),
                    ReplayMode::Union => normalize_pool(candidates, v, maxc),
                }
                let mut out = Vec::with_capacity(rule.degree());
                rule.prune(&points, labels, v, candidates, prune_scratch, &mut out);
                out.iter().map(|c| c.id).collect()
            })
        })
        .collect_in_pool(pool.as_ref());
    stats.n_pool_entries = (0..pruned_nodes as u32)
        .map(|v| match params.mode {
            ReplayMode::Incremental => pools.search(v).0.len().min(maxc) as u64,
            ReplayMode::Union => pools.row(v).0.len() as u64,
        })
        .sum();
    stats.prune_s = t_prune.elapsed().as_secs_f64();

    let (rows, slots) = match params.mode {
        ReplayMode::Union => (outs, rule.degree()),
        ReplayMode::Incremental => {
            let t_back = Instant::now();
            let slots = slots(rule.degree(), params.slack);
            let rows = replay_backedges(
                pools, &outs, &rule, &points, labels, slots, maxc, &pool, &scratch, &mut stats,
            );
            stats.backedge_s = t_back.elapsed().as_secs_f64();
            (rows, slots)
        }
    };

    let t_assemble = Instant::now();
    let mut adjacency = Adjacency::new(n_nodes, slots);
    for (v, row) in rows.iter().enumerate() {
        adjacency.set(v as u32, row.iter().copied());
    }
    drop(rows);
    let mut assemble_s = t_assemble.elapsed().as_secs_f64();
    if params.final_prune && params.mode == ReplayMode::Incremental {
        let t_final = Instant::now();
        let final_scratch = PerThread::new(params.threads, PruneScratch::default);
        stats.n_final_prunes = final_prune(
            &mut adjacency,
            &rule,
            &points,
            labels,
            None,
            maxc,
            &pool,
            &final_scratch,
            |s| s,
        );
        for s in final_scratch.into_inner() {
            stats.n_dist_reprune += s.n_dist + s.n_dist_source;
            stats.n_label_checks += s.n_label_checks;
        }
        stats.final_prune_s = t_final.elapsed().as_secs_f64();
    }
    let t_graph = Instant::now();
    for (_, s) in scratch.into_inner() {
        stats.n_dist_prune += s.n_dist;
        stats.n_dist_reprune += s.n_dist_source;
        stats.n_label_checks += s.n_label_checks;
    }
    let n_aux = n_nodes - n;
    let aux_nodes = AuxNodes::new(
        vec![AuxKind::Frozen; n_aux],
        &Matrix::from_slice(&aux, n_aux, base.ncols()),
    )?;
    let graph = adjacency.into_graph(n, aux_nodes, (n as u32..n_nodes as u32).collect())?;
    assemble_s += t_graph.elapsed().as_secs_f64();
    stats.assemble_s = assemble_s;
    stats.wall_s = t0.elapsed().as_secs_f64();
    Ok(ReplayOutput { graph, stats })
}

/// The back-edge phase of the incremental mode: per target, the initial row (its own prune;
/// empty for auxiliary nodes) followed by the back-edges of every later batch.
#[expect(clippy::too_many_arguments, reason = "internal step of the replay")]
fn replay_backedges<T: Element>(
    pools: &Pools,
    outs: &[Vec<u32>],
    rule: &PruneRule,
    points: &Points<'_, T>,
    labels: Option<&LabelInfo<'_>>,
    slots: usize,
    maxc: usize,
    pool: &diskann_providers::utils::RayonThreadPool,
    scratch: &PerThread<(Vec<Candidate>, PruneScratch)>,
    stats: &mut ReplayStats,
) -> Vec<Vec<u32>> {
    let n_nodes = pools.n_nodes();
    // Proposals grouped by target (counting sort), each as `batch << 32 | source`, in
    // insertion order within a target.
    let mut offsets = vec![0u64; n_nodes + 1];
    for out in outs {
        for &t in out {
            offsets[t as usize + 1] += 1;
        }
    }
    for v in 0..n_nodes {
        offsets[v + 1] += offsets[v];
    }
    let mut fill = offsets.clone();
    let mut keys = vec![0u64; offsets[n_nodes] as usize];
    for b in 0..pools.n_batches() {
        for &p in pools.batch(b) {
            for &t in &outs[p as usize] {
                let at = &mut fill[t as usize];
                keys[*at as usize] = (b as u64) << 32 | u64::from(p);
                *at += 1;
            }
        }
    }
    drop(fill);
    let results: Vec<(Vec<u32>, u64, u64)> = (0..n_nodes)
        .into_par_iter()
        .map(|t| {
            scratch.with(|(_, prune_scratch)| {
                let mut row: Vec<u32> = outs.get(t).cloned().unwrap_or_default();
                let mut segment = keys[offsets[t] as usize..offsets[t + 1] as usize].to_vec();
                segment.sort_unstable();
                let (mut appends, mut reprunes) = (0u64, 0u64);
                let mut fresh = std::mem::take(&mut prune_scratch.fresh);
                let mut out = Vec::with_capacity(rule.degree());
                let mut i = 0;
                while i < segment.len() {
                    let batch = segment[i] >> 32;
                    let j = i + segment[i..]
                        .iter()
                        .take_while(|&&k| k >> 32 == batch)
                        .count();
                    fresh_sources(&row, segment[i..j].iter().map(|&k| k as u32), &mut fresh);
                    if !fresh.is_empty() {
                        if row.len() + fresh.len() <= slots {
                            row.extend_from_slice(&fresh);
                            appends += 1;
                        } else {
                            reprune(
                                rule,
                                points,
                                labels,
                                None,
                                t as u32,
                                &row,
                                &fresh,
                                maxc,
                                prune_scratch,
                                &mut out,
                            );
                            row.clear();
                            row.extend(out.iter().map(|c| c.id));
                            reprunes += 1;
                        }
                    }
                    i = j;
                }
                prune_scratch.fresh = fresh;
                (row, appends, reprunes)
            })
        })
        .collect_in_pool(pool.as_ref());
    let mut rows = Vec::with_capacity(n_nodes);
    for (row, appends, reprunes) in results {
        stats.n_appends += appends;
        stats.n_reprunes += reprunes;
        rows.push(row);
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        build::{build, test_util::random_points, BuildParams},
        labels::LabelStoreBuilder,
    };

    fn recorded(rule: PruneRule) -> (Matrix<u8>, crate::build::BuildOutput) {
        let base = random_points(1200, 8, 21);
        let params = BuildParams {
            rule,
            l_build: 24,
            max_batch: 32,
            threads: 3,
            capture: true,
            ..BuildParams::default()
        };
        let out = build(&base, None, &params).unwrap();
        (base, out)
    }

    fn params(rule: PruneRule, mode: ReplayMode) -> ReplayParams {
        ReplayParams {
            rule,
            mode,
            slack: 1.3,
            maxc: 750,
            threads: 2,
            final_prune: false,
        }
    }

    #[test]
    fn incremental_replay_with_the_build_rule_reproduces_the_build() {
        let rule = PruneRule::Alpha { r: 12, alpha: 1.2 };
        let (base, out) = recorded(rule);
        let pools = out.pools.unwrap();
        let replayed = replay(&pools, &base, None, &params(rule, ReplayMode::Incremental)).unwrap();
        assert_eq!(replayed.graph, out.graph);
        assert_eq!(replayed.stats.n_reprunes, out.stats.n_reprunes);
    }

    #[test]
    fn pools_roundtrip_and_union_replay() {
        let rule = PruneRule::Alpha { r: 12, alpha: 1.2 };
        let (base, out) = recorded(rule);
        let pools = out.pools.unwrap();
        // Every node's final neighbors were in its recorded pool.
        for v in 0..pools.n_nodes() as u32 {
            let (ids, _) = pools.row(v);
            assert!(out.graph.neighbors(v).iter().all(|u| ids.contains(u)));
        }
        let mut bytes = Vec::new();
        pools.write_to(&mut bytes).unwrap();
        assert_eq!(bytes.len() as u64, pools.file_bytes());
        let back = Pools::read_from(std::io::Cursor::new(&bytes)).unwrap();
        assert_eq!(back, pools);
        let union = replay(&back, &base, None, &params(rule, ReplayMode::Union)).unwrap();
        let g = &union.graph;
        assert_eq!((g.n_nodes(), g.r_max()), (1201, 12));
        // Without back-edges after the prune a point can lose every in-edge (e.g. an exact
        // duplicate occluded by its twin in every pool), so reachability is not guaranteed.
        assert!(g.stats().reachable_data >= 1190);
        assert!(g.degree(5) <= 12);
    }

    #[test]
    fn label_aware_replay_keeps_more_edges() {
        let rule = PruneRule::Alpha { r: 12, alpha: 1.2 };
        let (base, out) = recorded(rule);
        let pools = out.pools.unwrap();
        let mut b = LabelStoreBuilder::new();
        for p in 0..1200usize {
            let a = format!("a{}", p % 7);
            let c = format!("c{}", p % 3);
            b.push_point([("a", a.as_str()), ("c", c.as_str())])
                .unwrap();
        }
        let store = b.finish().unwrap();
        let lc = PruneRule::LabelContainment { r: 12, alpha: 1.2 };
        let blind = replay(&pools, &base, None, &params(rule, ReplayMode::Incremental)).unwrap();
        let aware = replay(
            &pools,
            &base,
            Some(&store),
            &params(lc, ReplayMode::Incremental),
        )
        .unwrap();
        assert!(aware.graph.n_edges() > blind.graph.n_edges());
        assert!(aware.stats.n_label_checks > 0);
        assert!(replay(&pools, &base, None, &params(lc, ReplayMode::Incremental)).is_err());
    }
}
