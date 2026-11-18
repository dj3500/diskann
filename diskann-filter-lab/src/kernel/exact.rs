/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Exact paths (`ExactPath` role): answers computed from the postings.
//!
//! Version 1 (routing rule `min_c U_c ≤ T_exact`):
//! - `scan_smallest_clause`: merge the postings of the clause with the smallest `U_c`
//!   (distinct points in increasing id order), check the other clauses on each candidate
//!   (Boolean, with the run's predicate evaluator: mode 1 or mode 2 on the label row, or the
//!   static bitsets), compute distances for the survivors (prefetching their vectors) and
//!   keep the exact top-K by (distance, id);
//! - `intersect_then_scan`: galloping intersection of the clauses' posting unions first
//!   ([`super::materialize::Materialized::build_sparse`]), then distances.
//!
//! Version 2 (`planned`, [`run_planned`]): one of the plans of [`super::cost::ExactPlan`],
//! chosen per query by the cost model; candidate rows and vectors are prefetched in a software
//! pipeline, and checks use the static bitsets when the index has them. With
//! `exact_checks = "residual"` (the default) the candidates, all members of `union(c*)`, are
//! checked on the residual clauses only, and the prefetches serve those clauses (the rows
//! when one of them has labels without a bitset, the first residual clause's bitset word);
//! `"full"` checks every clause with the graph search's prefetch policy (batch B6):
//! - `scan`: candidates of `c*` (its posting list, or the k-way merge of its labels'
//!   postings), predicate first, distances of the matches;
//! - `df` (distance first): distances of every candidate; the predicate only for candidates
//!   inside the answer bound (heap not full, or distance ≤ the heap's K-th: ties are checked);
//! - `gallop`: galloping intersection, then distances;
//! - `bitmap`: dense match bitmap (static bitsets ORed word by word, rare labels scattered),
//!   built and enumerated in id order one L1-sized block at a time
//!   ([`Materialized::build_bitmap`]; a block's remaining clauses are skipped once it is
//!   empty), then distances;
//! - `rowand`: the words holding members of `union(c*)`, each ANDed with the residual
//!   clauses word by word (bitset words; galloping cursors in the postings of labels without
//!   a bitset; [`Materialized::build_rowand`]), enumerated, then distances.
//!
//! Every plan processes *every* candidate, so the answers are the exact top-K. With a
//! *seeded* answer heap (the graph search's answers, after a fallback) the plans keep the
//! seed's answers, skip points already held (graph-visited points are not known to be
//! exact-processed, e.g. after lazy predicate skips), and use the seed's K-th distance as the
//! initial bound of `df`.

use super::{
    collect::{Answers, Collector},
    cost::ExactPlan,
    eval::{check, ClauseSearch, Eval, EvalCtx, RowScan, StaticBits},
    materialize::{merge_distinct, Materialized},
    search::{tic, toc, Counters, EvalMode},
};
use crate::{graph::NodeVectors, vectors::Element};

/// Exact-path role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExactPath {
    None,
    /// Route to the posting scan when `min_c U_c ≤ T`.
    ScanSmallestClause(u64),
    /// Route to the posting intersection when `min_c U_c ≤ T`.
    IntersectThenScan(u64),
    /// Exact path v2 (the cost model's cheapest plan); with the `threshold` crossover, route
    /// when `min_c U_c ≤ T` (other crossovers decide by cost and ignore `T`).
    Planned(u64),
}

impl ExactPath {
    /// The routing threshold `T_exact`, if any.
    pub fn threshold(self) -> Option<u64> {
        match self {
            Self::None => None,
            Self::ScanSmallestClause(t) | Self::IntersectThenScan(t) | Self::Planned(t) => Some(t),
        }
    }

    /// Whether this is the exact path v2.
    pub fn is_planned(self) -> bool {
        matches!(self, Self::Planned(_))
    }
}

const BATCH: usize = 16;

/// Distances of a batch of matches (prefetched first) into the answer heap.
#[inline]
fn flush<T: Element>(
    batch: &[u32],
    vectors: &NodeVectors<'_, T>,
    query: &[T],
    answers: &mut Answers,
    ctr: &mut Counters,
) {
    for &p in batch {
        diskann_vector::prefetch_hint_max::<4, _>(vectors.get(p));
    }
    for &p in batch {
        let d = T::squared_l2(query, vectors.get(p));
        ctr.n_dist += 1;
        answers.offer(d, p);
    }
}

/// Runs an exact path; the answers are left in `answers` (a `heap(K)` collector).
/// `scan_smallest_clause` checks candidates with the evaluator of `eval` (mode 2 or the
/// static bitsets; mode 1 otherwise, as no match set is materialized for exact queries).
#[expect(
    clippy::too_many_arguments,
    reason = "the exact path reads every per-query input"
)]
pub fn run<T: Element>(
    path: ExactPath,
    eval: EvalMode,
    cx: &EvalCtx<'_>,
    vectors: &NodeVectors<'_, T>,
    n_data: usize,
    query: &[T],
    k: usize,
    mat: &mut Materialized,
    answers: &mut Answers,
    ctr: &mut Counters,
) {
    answers.reset(Collector::Heap, k, k);
    match path {
        ExactPath::None | ExactPath::Planned(_) => {}
        ExactPath::ScanSmallestClause(_) => match eval {
            EvalMode::ClauseSearch => {
                scan::<T, ClauseSearch>(cx, vectors, query, mat, answers, ctr)
            }
            EvalMode::StaticBits => scan::<T, StaticBits>(cx, vectors, query, mat, answers, ctr),
            EvalMode::RowScan | EvalMode::Dense | EvalMode::Sparse => {
                scan::<T, RowScan>(cx, vectors, query, mat, answers, ctr);
            }
        },
        ExactPath::IntersectThenScan(_) => {
            ctr.n_exact_cand += mat.build_sparse(cx.labels, cx.q, n_data);
            for chunk in mat.sparse.chunks(BATCH) {
                flush(chunk, vectors, query, answers, ctr);
            }
        }
    }
}

/// `scan_smallest_clause` with the evaluator `P`.
fn scan<T: Element, P: Eval>(
    cx: &EvalCtx<'_>,
    vectors: &NodeVectors<'_, T>,
    query: &[T],
    mat: &mut Materialized,
    answers: &mut Answers,
    ctr: &mut Counters,
) {
    let mut batch = [0u32; BATCH];
    let mut n = 0;
    let q = cx.q;
    let c0 = q.clause(0);
    let labels = cx.labels;
    let others = q.n_clauses() > 1;
    let (heap, pos) = mat.merge_scratch();
    merge_distinct(
        c0.len(),
        |i| labels.postings(c0[i]),
        heap,
        pos,
        |p| {
            ctr.n_exact_cand += 1;
            let ok = !others || {
                ctr.n_pred += 1;
                P::full(cx, p)
            };
            if ok {
                batch[n] = p;
                n += 1;
                if n == BATCH {
                    flush(&batch, vectors, query, answers, ctr);
                    n = 0;
                }
            }
            true
        },
    );
    flush(&batch[..n], vectors, query, answers, ctr);
}

/// Evaluator of the exact path v2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExactEval {
    /// Mode 1 on the label rows.
    RowScan,
    /// The static per-label bitsets (the query split against them), rows for rare labels.
    StaticBits,
}

/// Prefetch distances (in candidates) of the exact path's pipeline: row offsets (and the first
/// bitset word), rows, vectors.
const AHEAD_OFFSETS: usize = 16;
const AHEAD_ROWS: usize = 8;
const AHEAD_VECTORS: usize = 8;

#[inline(always)]
fn offer_answer(answers: &mut Answers, seeded: bool, d: f32, v: u32) {
    if seeded {
        answers.offer_unique(d, v);
    } else {
        answers.offer(d, v);
    }
}

/// Distances of `ids` (prefetched `AHEAD_VECTORS` ahead) into the answer heap.
#[inline]
pub(crate) fn distances<T: Element>(
    ids: &[u32],
    vectors: &NodeVectors<'_, T>,
    query: &[T],
    answers: &mut Answers,
    seeded: bool,
    ctr: &mut Counters,
) {
    let n = ids.len();
    for &v in &ids[..n.min(AHEAD_VECTORS)] {
        diskann_vector::prefetch_hint_max::<4, _>(vectors.get(v));
    }
    for i in 0..n {
        if i + AHEAD_VECTORS < n {
            diskann_vector::prefetch_hint_max::<4, _>(vectors.get(ids[i + AHEAD_VECTORS]));
        }
        let v = ids[i];
        let d = T::squared_l2(query, vectors.get(v));
        ctr.n_dist += 1;
        offer_answer(answers, seeded, d, v);
    }
}

/// The candidates of `cands` that satisfy the full predicate, into `out` (row offsets and
/// rows prefetched ahead).
#[inline]
pub(crate) fn filter<P: Eval>(
    cx: &EvalCtx<'_>,
    cands: &[u32],
    out: &mut Vec<u32>,
    ctr: &mut Counters,
) {
    out.clear();
    let n = cands.len();
    for &v in &cands[..n.min(AHEAD_OFFSETS)] {
        P::prefetch_offsets(cx, v);
    }
    for &v in &cands[..n.min(AHEAD_ROWS)] {
        P::prefetch(cx, v);
    }
    for i in 0..n {
        if i + AHEAD_OFFSETS < n {
            P::prefetch_offsets(cx, cands[i + AHEAD_OFFSETS]);
        }
        if i + AHEAD_ROWS < n {
            P::prefetch(cx, cands[i + AHEAD_ROWS]);
        }
        let v = cands[i];
        if check::<P>(cx, v) {
            out.push(v);
        }
    }
    ctr.n_pred += n as u64;
}

/// Candidates per block of the pipelined `df` (`df_pipeline`): a block's distances cover the
/// latency of the previous block's row prefetches.
const DF_BLOCK: usize = 32;

/// Distance first: every candidate's distance; the predicate only inside the answer bound.
/// `pipeline`: a candidate inside the bound when its distance is computed has its row offsets
/// prefetched, its row prefetched after its block, and is checked after the next block's
/// distances if it is still inside the bound. The bound only shrinks and the checks keep the
/// candidate order, so the checks, answers and counters are those of the direct loop.
#[inline]
#[expect(
    clippy::too_many_arguments,
    reason = "the exact path's per-query inputs"
)]
fn distance_first<T: Element, P: Eval>(
    cx: &EvalCtx<'_>,
    cands: &[u32],
    checks: bool,
    vectors: &NodeVectors<'_, T>,
    query: &[T],
    answers: &mut Answers,
    seeded: bool,
    pipeline: Option<&mut DfPipeline>,
    ctr: &mut Counters,
) {
    let n = cands.len();
    for &v in &cands[..n.min(AHEAD_VECTORS)] {
        diskann_vector::prefetch_hint_max::<4, _>(vectors.get(v));
    }
    if let (true, Some(pl)) = (checks, pipeline) {
        pl.pending.clear();
        pl.current.clear();
        for start in (0..n).step_by(DF_BLOCK) {
            for i in start..(start + DF_BLOCK).min(n) {
                if i + AHEAD_VECTORS < n {
                    diskann_vector::prefetch_hint_max::<4, _>(
                        vectors.get(cands[i + AHEAD_VECTORS]),
                    );
                }
                let v = cands[i];
                let d = T::squared_l2(query, vectors.get(v));
                ctr.n_dist += 1;
                if answers.is_full() && d > answers.worst() {
                    continue;
                }
                P::prefetch_offsets(cx, v);
                pl.current.push((d, v));
            }
            df_checks::<P>(cx, &pl.pending, answers, seeded, ctr);
            for &(_, v) in &pl.current {
                P::prefetch(cx, v);
            }
            std::mem::swap(&mut pl.pending, &mut pl.current);
            pl.current.clear();
        }
        df_checks::<P>(cx, &pl.pending, answers, seeded, ctr);
        return;
    }
    for i in 0..n {
        if i + AHEAD_VECTORS < n {
            diskann_vector::prefetch_hint_max::<4, _>(vectors.get(cands[i + AHEAD_VECTORS]));
        }
        let v = cands[i];
        let d = T::squared_l2(query, vectors.get(v));
        ctr.n_dist += 1;
        // Strictly beyond a full heap's K-th distance: cannot be an answer (equal distances
        // may still enter by id, so they are checked).
        if answers.is_full() && d > answers.worst() {
            continue;
        }
        if checks {
            ctr.n_pred += 1;
            if !check::<P>(cx, v) {
                continue;
            }
        }
        offer_answer(answers, seeded, d, v);
    }
}

/// The pipelined `df`'s checks of `block` (candidates in order with their distances), each only
/// if still inside the answer bound.
#[inline]
fn df_checks<P: Eval>(
    cx: &EvalCtx<'_>,
    block: &[(f32, u32)],
    answers: &mut Answers,
    seeded: bool,
    ctr: &mut Counters,
) {
    for &(d, v) in block {
        if answers.is_full() && d > answers.worst() {
            continue;
        }
        ctr.n_pred += 1;
        if check::<P>(cx, v) {
            offer_answer(answers, seeded, d, v);
        }
    }
}

/// Scratch of the pipelined `df`: the candidates of the current block inside the answer bound
/// and those of the previous block, awaiting their checks.
#[derive(Debug)]
pub struct DfPipeline {
    current: Vec<(f32, u32)>,
    pending: Vec<(f32, u32)>,
}

impl DfPipeline {
    /// Scratch for blocks of [`DF_BLOCK`] candidates (allocated now: no allocation during a
    /// search).
    pub fn new() -> Self {
        Self {
            current: Vec::with_capacity(DF_BLOCK),
            pending: Vec::with_capacity(DF_BLOCK),
        }
    }
}

impl Default for DfPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Collects the candidates of `c*` (the first clause in Boolean order) into `mat.cands`,
/// unless `c*` has a single non-empty label (its posting list is then read in place:
/// returns it). Without clauses every data point is a candidate.
fn candidates<'a>(cx: &EvalCtx<'a>, n_data: usize, mat: &mut Materialized) -> Option<&'a [u32]> {
    let q = cx.q;
    let labels = cx.labels;
    mat.cands.clear();
    if q.n_clauses() == 0 {
        mat.cands.extend(0..n_data as u32);
        return None;
    }
    let c0 = q.clause(0);
    let mut nonempty = c0.iter().filter(|&&l| labels.posting_len(l) > 0);
    if let (Some(&l), None) = (nonempty.next(), nonempty.next()) {
        return Some(labels.postings(l));
    }
    let cands = &mut mat.cands;
    let (heap, pos) = (&mut mat.heap, &mut mat.pos);
    merge_distinct(
        c0.len(),
        |i| labels.postings(c0[i]),
        heap,
        pos,
        |p| {
            cands.push(p);
            true
        },
    );
    None
}

/// Runs the exact path v2 with `plan`; the answers are left in `answers` (a `heap(K)`
/// collector). With `seeded`, `answers` already holds a `heap(K)` of answers of another
/// search (kept; see the module documentation). `df`: the pipelined `df` loop's scratch
/// (`df_pipeline`; `None`: the direct loop).
#[expect(
    clippy::too_many_arguments,
    reason = "the exact path reads every per-query input"
)]
pub fn run_planned<T: Element>(
    plan: ExactPlan,
    eval: ExactEval,
    cx: &EvalCtx<'_>,
    vectors: &NodeVectors<'_, T>,
    n_data: usize,
    query: &[T],
    k: usize,
    mat: &mut Materialized,
    answers: &mut Answers,
    seeded: bool,
    df: Option<&mut DfPipeline>,
    ctr: &mut Counters,
) {
    if !seeded {
        answers.reset(Collector::Heap, k, k);
    }
    match eval {
        ExactEval::RowScan => planned::<T, RowScan>(
            plan, cx, vectors, n_data, query, mat, answers, seeded, df, ctr,
        ),
        ExactEval::StaticBits => planned::<T, StaticBits>(
            plan, cx, vectors, n_data, query, mat, answers, seeded, df, ctr,
        ),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the exact path reads every per-query input"
)]
fn planned<T: Element, P: Eval>(
    plan: ExactPlan,
    cx: &EvalCtx<'_>,
    vectors: &NodeVectors<'_, T>,
    n_data: usize,
    query: &[T],
    mat: &mut Materialized,
    answers: &mut Answers,
    seeded: bool,
    df: Option<&mut DfPipeline>,
    ctr: &mut Counters,
) {
    let checks = cx.q.n_clauses() > 1;
    match plan {
        ExactPlan::Scan | ExactPlan::DistanceFirst => {
            let t = tic(ctr);
            let inplace = candidates(cx, n_data, mat);
            let cands: &[u32] = inplace.unwrap_or(&mat.cands);
            toc(t, &mut ctr.t_cand_ns);
            ctr.n_exact_cand += cands.len() as u64;
            if plan == ExactPlan::DistanceFirst {
                let t = tic(ctr);
                distance_first::<T, P>(cx, cands, checks, vectors, query, answers, seeded, df, ctr);
                toc(t, &mut ctr.t_dist_ns);
            } else if checks {
                let t = tic(ctr);
                filter::<P>(cx, cands, &mut mat.matches, ctr);
                toc(t, &mut ctr.t_check_ns);
                let t = tic(ctr);
                distances(&mat.matches, vectors, query, answers, seeded, ctr);
                toc(t, &mut ctr.t_dist_ns);
            } else {
                let t = tic(ctr);
                distances(cands, vectors, query, answers, seeded, ctr);
                toc(t, &mut ctr.t_dist_ns);
            }
        }
        ExactPlan::Gallop => {
            let t = tic(ctr);
            ctr.n_exact_cand += mat.build_sparse(cx.labels, cx.q, n_data);
            toc(t, &mut ctr.t_check_ns);
            let t = tic(ctr);
            distances(&mat.sparse, vectors, query, answers, seeded, ctr);
            toc(t, &mut ctr.t_dist_ns);
        }
        ExactPlan::Bitmap => {
            let t = tic(ctr);
            let sb = (cx.stat.n_points() == n_data).then_some(cx.stat);
            ctr.n_exact_cand += mat.build_bitmap(cx.labels, cx.q, n_data, sb);
            toc(t, &mut ctr.t_check_ns);
            let t = tic(ctr);
            distances(&mat.matches, vectors, query, answers, seeded, ctr);
            toc(t, &mut ctr.t_dist_ns);
        }
        ExactPlan::RowAnd => {
            let t = tic(ctr);
            mat.ensure_dense(n_data);
            // The query was split against the static bitsets iff the evaluator reads them.
            let split = P::NAME == StaticBits::NAME && cx.stat.n_points() == n_data;
            ctr.n_exact_cand += mat.build_rowand(cx.labels, cx.q, n_data, split.then_some(cx.stat));
            toc(t, &mut ctr.t_check_ns);
            let t = tic(ctr);
            distances(&mat.matches, vectors, query, answers, seeded, ctr);
            toc(t, &mut ctr.t_dist_ns);
        }
    }
}
