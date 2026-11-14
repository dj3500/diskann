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
//! pipeline, and checks use the static bitsets when the index has them:
//! - `scan`: candidates of `c*` (its posting list, or the k-way merge of its labels'
//!   postings), predicate first, distances of the matches;
//! - `df` (distance first): distances of every candidate; the predicate only for candidates
//!   inside the answer bound (heap not full, or distance ≤ the heap's K-th: ties are checked);
//! - `gallop`: galloping intersection, then distances;
//! - `bitmap`: dense match bitmap (static bitsets ORed word by word, rare labels scattered),
//!   enumerated in id order, then distances.
//!
//! Every plan processes *every* candidate, so the answers are the exact top-K. With a
//! *seeded* answer heap (the graph search's answers, after a fallback) the plans keep the
//! seed's answers, skip points already held (graph-visited points are not known to be
//! exact-processed, e.g. after lazy predicate skips), and use the seed's K-th distance as the
//! initial bound of `df`.

use super::{
    collect::{Answers, Collector},
    cost::ExactPlan,
    eval::{ClauseSearch, Eval, EvalCtx, RowScan, StaticBits},
    materialize::{merge_distinct, Materialized},
    search::{Counters, EvalMode},
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
        if P::full(cx, v) {
            out.push(v);
        }
    }
    ctr.n_pred += n as u64;
}

/// Distance first: every candidate's distance; the predicate only inside the answer bound.
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
    ctr: &mut Counters,
) {
    let n = cands.len();
    for &v in &cands[..n.min(AHEAD_VECTORS)] {
        diskann_vector::prefetch_hint_max::<4, _>(vectors.get(v));
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
            if !P::full(cx, v) {
                continue;
            }
        }
        offer_answer(answers, seeded, d, v);
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
/// search (kept; see the module documentation).
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
    ctr: &mut Counters,
) {
    if !seeded {
        answers.reset(Collector::Heap, k, k);
    }
    match eval {
        ExactEval::RowScan => {
            planned::<T, RowScan>(plan, cx, vectors, n_data, query, mat, answers, seeded, ctr);
        }
        ExactEval::StaticBits => {
            planned::<T, StaticBits>(plan, cx, vectors, n_data, query, mat, answers, seeded, ctr);
        }
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
    ctr: &mut Counters,
) {
    let checks = cx.q.n_clauses() > 1;
    match plan {
        ExactPlan::Scan | ExactPlan::DistanceFirst => {
            let inplace = candidates(cx, n_data, mat);
            let cands: &[u32] = inplace.unwrap_or(&mat.cands);
            ctr.n_exact_cand += cands.len() as u64;
            if plan == ExactPlan::DistanceFirst {
                distance_first::<T, P>(cx, cands, checks, vectors, query, answers, seeded, ctr);
            } else if checks {
                filter::<P>(cx, cands, &mut mat.matches, ctr);
                distances(&mat.matches, vectors, query, answers, seeded, ctr);
            } else {
                distances(cands, vectors, query, answers, seeded, ctr);
            }
        }
        ExactPlan::Gallop => {
            ctr.n_exact_cand += mat.build_sparse(cx.labels, cx.q, n_data);
            distances(&mat.sparse, vectors, query, answers, seeded, ctr);
        }
        ExactPlan::Bitmap => {
            mat.ensure_dense(n_data);
            if cx.stat.n_points() == n_data {
                mat.build_dense_static(cx.labels, cx.q, n_data, cx.stat);
            } else {
                mat.build_dense(cx.labels, cx.q, n_data);
            }
            let words = n_data.div_ceil(64);
            let out = &mut mat.matches;
            out.clear();
            for (w, &x) in mat.dense[..words].iter().enumerate() {
                let mut x = x;
                while x != 0 {
                    out.push(((w as u32) << 6) | x.trailing_zeros());
                    x &= x - 1;
                }
            }
            ctr.n_exact_cand += out.len() as u64;
            distances(&mat.matches, vectors, query, answers, seeded, ctr);
        }
    }
}
