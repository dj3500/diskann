/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Exact paths (`ExactPath` role): answers computed from the postings, used for queries whose
//! smallest clause upper bound is at most `T_exact`.
//!
//! - `scan_smallest_clause`: merge the postings of the clause with the smallest `U_c`
//!   (distinct points in increasing id order), check the other clauses on each candidate
//!   (Boolean, with the run's predicate evaluator: mode 1 or mode 2 on the label row, or the
//!   static bitsets), compute distances for the survivors (prefetching their vectors) and
//!   keep the exact top-K by (distance, id);
//! - `intersect_then_scan`: galloping intersection of the clauses' posting unions first
//!   ([`super::materialize::Materialized::build_sparse`]), then distances.

use super::{
    collect::{Answers, Collector},
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
}

impl ExactPath {
    /// The routing threshold `T_exact`, if any.
    pub fn threshold(self) -> Option<u64> {
        match self {
            Self::None => None,
            Self::ScanSmallestClause(t) | Self::IntersectThenScan(t) => Some(t),
        }
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
        ExactPath::None => {}
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
