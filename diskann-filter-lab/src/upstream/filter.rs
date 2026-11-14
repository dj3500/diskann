/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Per-query predicate providers implementing upstream's
//! [`QueryLabelProvider`](diskann::graph::ext::labeled::QueryLabelProvider).
//!
//! A provider is compiled for one predicate at a time ([`PredicateFilter::compile`]); all
//! per-query work (clause ordering, bitmap construction) happens there, so callers can time it
//! together with the search. Ids at or above `n_data` (frozen start points and other auxiliary
//! nodes) never match, and they are never used to index label rows or bitmaps.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use diskann::graph::ext::labeled::QueryLabelProvider;

use crate::{
    labels::LabelStore,
    predicate::{intersects, Cnf},
};

/// An event counter for single-threaded use behind `&self`.
///
/// The increment is a relaxed load followed by a relaxed store rather than an atomic
/// read-modify-write, so it compiles to plain moves. Concurrent use loses counts but is safe.
#[derive(Debug, Default)]
pub struct Counter(AtomicU64);

impl Counter {
    #[inline(always)]
    pub fn bump(&self) {
        self.0
            .store(self.0.load(Ordering::Relaxed) + 1, Ordering::Relaxed);
    }

    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    pub fn reset(&self) {
        self.0.store(0, Ordering::Relaxed);
    }
}

/// Predicate evaluations made since the last [`PredicateFilter::compile`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FilterCounts {
    /// Calls of `is_match`.
    pub calls: u64,
    /// Calls that returned `true`.
    pub matches: u64,
}

/// A [`QueryLabelProvider`] that is prepared for one predicate at a time.
pub trait PredicateFilter: QueryLabelProvider<u32> + 'static {
    /// Prepares the provider for `cnf` and resets its counters.
    fn compile(&mut self, cnf: Cnf<'_>);

    /// Evaluations since the last [`Self::compile`].
    fn counts(&self) -> FilterCounts;

    /// Label or posting entries touched by the last [`Self::compile`].
    fn prep_entries(&self) -> u64;
}

/// Clause order shared by the providers: ascending total posting length (an upper bound on the
/// number of points satisfying the clause), so the clause most likely to fail comes first;
/// ties keep the predicate's order.
fn order_clauses(labels: &LabelStore, cnf: Cnf<'_>, order: &mut Vec<(u64, usize)>) {
    order.clear();
    order.extend(cnf.clauses().enumerate().map(|(i, clause)| {
        let total = clause
            .iter()
            .map(|&l| labels.posting_len(l) as u64)
            .sum::<u64>();
        (total, i)
    }));
    order.sort_unstable();
}

/// Evaluates the predicate on the point's sorted label row (forward store), clause by clause
/// in [`order_clauses`] order, with early exit.
#[derive(Debug)]
pub struct ForwardFilter {
    labels: Arc<LabelStore>,
    n_data: u32,
    order: Vec<(u64, usize)>,
    bounds: Vec<usize>,
    clause_labels: Vec<u32>,
    prep_entries: u64,
    calls: Counter,
    matches: Counter,
}

impl ForwardFilter {
    /// A provider over `labels`; ids `>= labels.n_points()` are auxiliary.
    pub fn new(labels: Arc<LabelStore>) -> Self {
        let n_data = labels.n_points() as u32;
        Self {
            labels,
            n_data,
            order: Vec::new(),
            bounds: vec![0],
            clause_labels: Vec::new(),
            prep_entries: 0,
            calls: Counter::default(),
            matches: Counter::default(),
        }
    }

    #[inline]
    fn evaluate(&self, id: u32) -> bool {
        if id >= self.n_data {
            return false;
        }
        let row = self.labels.row(id);
        self.bounds
            .windows(2)
            .all(|w| intersects(row, &self.clause_labels[w[0]..w[1]]))
    }
}

impl QueryLabelProvider<u32> for ForwardFilter {
    #[inline]
    fn is_match(&self, id: u32) -> bool {
        self.calls.bump();
        let matched = self.evaluate(id);
        if matched {
            self.matches.bump();
        }
        matched
    }
}

impl PredicateFilter for ForwardFilter {
    fn compile(&mut self, cnf: Cnf<'_>) {
        order_clauses(&self.labels, cnf, &mut self.order);
        self.bounds.truncate(1);
        self.clause_labels.clear();
        for &(_, i) in &self.order {
            self.clause_labels.extend_from_slice(cnf.clause(i));
            self.bounds.push(self.clause_labels.len());
        }
        self.prep_entries = self.clause_labels.len() as u64;
        self.calls.reset();
        self.matches.reset();
    }

    fn counts(&self) -> FilterCounts {
        FilterCounts {
            calls: self.calls.get(),
            matches: self.matches.get(),
        }
    }

    fn prep_entries(&self) -> u64 {
        self.prep_entries
    }
}

/// Materializes the matching set as a dense bitmap over `0..n_data` from the postings
/// (per clause: union of its labels' postings; across clauses: intersection), then answers
/// with one bit test.
///
/// Clauses are processed in [`order_clauses`] order, and construction stops early once the
/// intersection is empty. Construction costs the total posting length of the processed
/// clauses plus a few passes over `n_data / 8` bytes.
#[derive(Debug)]
pub struct BitmapFilter {
    labels: Arc<LabelStore>,
    n_data: u32,
    words: Vec<u64>,
    /// Scratch bitmap of one clause; all zero between uses.
    clause_words: Vec<u64>,
    order: Vec<(u64, usize)>,
    prep_entries: u64,
    calls: Counter,
    matches: Counter,
}

impl BitmapFilter {
    /// A provider over `labels`; ids `>= labels.n_points()` are auxiliary.
    pub fn new(labels: Arc<LabelStore>) -> Self {
        let n_data = labels.n_points() as u32;
        let n_words = labels.n_points().div_ceil(64);
        Self {
            labels,
            n_data,
            words: vec![0; n_words],
            clause_words: vec![0; n_words],
            order: Vec::new(),
            prep_entries: 0,
            calls: Counter::default(),
            matches: Counter::default(),
        }
    }

    fn scatter(labels: &LabelStore, clause: &[u32], words: &mut [u64]) -> u64 {
        let mut entries = 0;
        for &l in clause {
            let postings = labels.postings(l);
            entries += postings.len() as u64;
            for &p in postings {
                words[(p >> 6) as usize] |= 1u64 << (p & 63);
            }
        }
        entries
    }

    #[inline]
    fn evaluate(&self, id: u32) -> bool {
        id < self.n_data && (self.words[(id >> 6) as usize] >> (id & 63)) & 1 == 1
    }
}

impl QueryLabelProvider<u32> for BitmapFilter {
    #[inline]
    fn is_match(&self, id: u32) -> bool {
        self.calls.bump();
        let matched = self.evaluate(id);
        if matched {
            self.matches.bump();
        }
        matched
    }
}

impl PredicateFilter for BitmapFilter {
    fn compile(&mut self, cnf: Cnf<'_>) {
        self.calls.reset();
        self.matches.reset();
        self.prep_entries = 0;
        order_clauses(&self.labels, cnf, &mut self.order);
        let Some(&(_, first)) = self.order.first() else {
            // No clauses: every data point matches.
            self.words.fill(u64::MAX);
            let tail = self.n_data % 64;
            if tail != 0 {
                if let Some(last) = self.words.last_mut() {
                    *last = (1u64 << tail) - 1;
                }
            }
            return;
        };
        self.words.fill(0);
        self.prep_entries = Self::scatter(&self.labels, cnf.clause(first), &mut self.words);
        for &(_, i) in &self.order[1..] {
            self.prep_entries += Self::scatter(&self.labels, cnf.clause(i), &mut self.clause_words);
            let mut any = 0;
            for (w, c) in self.words.iter_mut().zip(self.clause_words.iter_mut()) {
                *w &= *c;
                *c = 0;
                any |= *w;
            }
            if any == 0 {
                break;
            }
        }
    }

    fn counts(&self) -> FilterCounts {
        FilterCounts {
            calls: self.calls.get(),
            matches: self.matches.get(),
        }
    }

    fn prep_entries(&self) -> u64 {
        self.prep_entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{labels::LabelStoreBuilder, predicate::PredicateSetBuilder};

    fn store() -> Arc<LabelStore> {
        let mut b = LabelStoreBuilder::new();
        let points: [&[(&str, &str)]; 5] = [
            &[("c", "c=a"), ("d", "d=x")],
            &[("c", "c=b"), ("d", "d=x")],
            &[("c", "c=a"), ("c", "c=b"), ("d", "d=y")],
            &[("d", "d=y")],
            &[],
        ];
        for labels in points {
            b.push_point(labels.iter().copied()).unwrap();
        }
        Arc::new(b.finish().unwrap())
    }

    #[test]
    fn providers_agree_and_reject_auxiliary_ids() {
        let labels = store();
        let mut pb = PredicateSetBuilder::new(&labels);
        pb.push(&[vec!["c=a", "c=b"], vec!["d=y"]]);
        pb.push(&[vec!["d=x"]]);
        pb.push(&[vec!["c=nowhere"], vec!["d=x"]]);
        pb.push::<&str>(&[]);
        let set = pb.finish();
        let expected: [&[u32]; 4] = [&[2], &[0, 1], &[], &[0, 1, 2, 3, 4]];

        let mut forward = ForwardFilter::new(labels.clone());
        let mut bitmap = BitmapFilter::new(labels.clone());
        for (q, want) in expected.iter().enumerate() {
            forward.compile(set.get(q));
            bitmap.compile(set.get(q));
            for id in 0..8u32 {
                let m = want.contains(&id);
                assert_eq!(forward.is_match(id), m, "forward q{q} id{id}");
                assert_eq!(bitmap.is_match(id), m, "bitmap q{q} id{id}");
            }
            let counts = FilterCounts {
                calls: 8,
                matches: want.len() as u64,
            };
            assert_eq!(forward.counts(), counts);
            assert_eq!(bitmap.counts(), counts);
        }
        // Compiling resets the counters.
        forward.compile(set.get(0));
        assert_eq!(forward.counts(), FilterCounts::default());
    }

    #[test]
    fn most_selective_clause_first() {
        let labels = store();
        let mut pb = PredicateSetBuilder::new(&labels);
        // d=x has 2 points; c=a and c=b have 2 each (4 posting entries): d=x goes first.
        pb.push(&[vec!["c=a", "c=b"], vec!["d=x"]]);
        let set = pb.finish();
        let mut forward = ForwardFilter::new(labels.clone());
        forward.compile(set.get(0));
        assert_eq!(forward.bounds, [0, 1, 3]);
        assert_eq!(forward.prep_entries(), 3);
        let mut bitmap = BitmapFilter::new(labels);
        bitmap.compile(set.get(0));
        assert_eq!(bitmap.prep_entries(), 2 + 4);
    }
}
