/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Per-query materialization of the match set from postings (`Materializer` role), and the
//! k-way merge of sorted lists used by the materializers, the exact paths and the sample
//! starts.
//!
//! - **dense**: a bitmap over the data points, the union of each clause's postings ANDed
//!   across clauses (the buffers are zeroed per query, inside the timer);
//! - **dense_static**: the same bitmap, with the labels that have a static bitset
//!   ([`crate::bitsets::LabelBitsets`]) ORed in word by word instead of scattered from their
//!   postings;
//! - **sparse**: the sorted list of matches, by merging the postings of the clause with the
//!   smallest upper bound and filtering it against the other clauses with galloping searches
//!   in their postings;
//! - **dense_nav**: the bitmap of the *navigation* clauses only (all clauses, or the projected
//!   ones, e.g. the binding clause under `projection(c*)`), with the static bitsets when the
//!   index has them; the evaluator [`super::eval::NavBitmap`] completes the full predicate on
//!   the row for the bitmap's members.
//!
//! Static per-label bitsets (a sidecar built once, not per query) are in
//! [`crate::bitsets`]; the `static_bits(θ)` evaluator ([`super::eval::StaticBits`]) reads them.

use super::eval::{gallop_lower_bound, Compiled};
use crate::{bitsets::LabelBitsets, labels::LabelStore};

/// Materializer role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Materializer {
    None,
    Dense,
    Sparse,
    /// Dense, built with the index's static bitsets.
    DenseStatic,
    /// Dense bitmap of the navigation clauses (static bitsets when the index has them).
    DenseNav,
}

/// Merges `n` sorted lists (`list(i)`) and calls `f` on every distinct value in increasing
/// order until `f` returns `false`. `heap` and `pos` are reusable scratch.
pub fn merge_distinct<'s, L, F>(
    n: usize,
    list: L,
    heap: &mut Vec<(u32, u32)>,
    pos: &mut Vec<usize>,
    mut f: F,
) where
    L: Fn(usize) -> &'s [u32],
    F: FnMut(u32) -> bool,
{
    if n == 1 {
        for &x in list(0) {
            if !f(x) {
                return;
            }
        }
        return;
    }
    heap.clear();
    pos.clear();
    pos.resize(n, 1);
    for i in 0..n {
        if let Some(&x) = list(i).first() {
            heap.push((x, i as u32));
        }
    }
    for i in (0..heap.len() / 2).rev() {
        sift_down(heap, i);
    }
    let mut last = None;
    while let Some(&(x, i)) = heap.first() {
        if last != Some(x) {
            last = Some(x);
            if !f(x) {
                return;
            }
        }
        let li = list(i as usize);
        let p = &mut pos[i as usize];
        if *p < li.len() {
            heap[0] = (li[*p], i);
            *p += 1;
        } else {
            let tail = heap.pop().unwrap_or((0, 0));
            if heap.is_empty() {
                break;
            }
            heap[0] = tail;
        }
        sift_down(heap, 0);
    }
}

#[inline]
fn sift_down(h: &mut [(u32, u32)], mut i: usize) {
    let n = h.len();
    loop {
        let l = 2 * i + 1;
        if l >= n {
            break;
        }
        let r = l + 1;
        let c = if r < n && h[r] < h[l] { r } else { l };
        if h[i] <= h[c] {
            break;
        }
        h.swap(i, c);
        i = c;
    }
}

/// Buffers of the materializers (allocated once per run).
#[derive(Debug, Clone, Default)]
pub struct Materialized {
    /// Dense match bitmap over the data points.
    pub dense: Vec<u64>,
    tmp: Vec<u64>,
    /// Sorted matches.
    pub sparse: Vec<u32>,
    tmp_sparse: Vec<u32>,
    cursors: Vec<usize>,
    pub(crate) heap: Vec<(u32, u32)>,
    pub(crate) pos: Vec<usize>,
    /// Candidates of the exact path v2 (merged postings).
    pub(crate) cands: Vec<u32>,
    /// Matches of the exact path v2 (filtered or enumerated candidates).
    pub(crate) matches: Vec<u32>,
}

impl Materialized {
    /// Buffers for `n_data` points (the dense bitmaps are allocated and written now, so that
    /// no page fault happens during a search).
    pub fn new(n_data: usize, dense: bool) -> Self {
        let words = if dense { n_data.div_ceil(64) } else { 0 };
        let mut m = Self::default();
        m.dense.resize(words, 0);
        m.tmp.resize(words, 0);
        m
    }

    /// Allocates the dense buffers for `n_data` points if they are missing (normally done by
    /// [`Self::new`], outside the timed path).
    pub fn ensure_dense(&mut self, n_data: usize) {
        let words = n_data.div_ceil(64);
        if self.dense.len() < words {
            self.dense.resize(words, 0);
            self.tmp.resize(words, 0);
        }
    }

    /// Builds the dense match bitmap of `q`; returns the number of posting entries read.
    pub fn build_dense(&mut self, labels: &LabelStore, q: &Compiled, n_data: usize) -> u64 {
        let words = n_data.div_ceil(64);
        assert!(
            self.dense.len() >= words,
            "dense buffers were not allocated"
        );
        let dense = &mut self.dense[..words];
        let mut read = 0u64;
        if q.n_clauses() == 0 {
            dense.fill(u64::MAX);
            if !n_data.is_multiple_of(64) {
                dense[words - 1] = (1u64 << (n_data % 64)) - 1;
            }
            return 0;
        }
        dense.fill(0);
        for &l in q.clause(0) {
            for &p in labels.postings(l) {
                dense[(p >> 6) as usize] |= 1u64 << (p & 63);
            }
            read += labels.posting_len(l) as u64;
        }
        let tmp = &mut self.tmp[..words];
        for j in 1..q.n_clauses() {
            tmp.fill(0);
            for &l in q.clause(j) {
                for &p in labels.postings(l) {
                    tmp[(p >> 6) as usize] |= 1u64 << (p & 63);
                }
                read += labels.posting_len(l) as u64;
            }
            for (a, b) in dense.iter_mut().zip(tmp.iter()) {
                *a &= *b;
            }
        }
        read
    }

    /// [`Self::build_dense`] with static bitsets: the labels with a bitset in `sb` are ORed in
    /// word by word, the others scattered from their postings. Returns the posting entries
    /// plus the bitset words read.
    pub fn build_dense_static(
        &mut self,
        labels: &LabelStore,
        q: &Compiled,
        n_data: usize,
        sb: &LabelBitsets,
    ) -> u64 {
        let words = n_data.div_ceil(64);
        assert!(
            self.dense.len() >= words,
            "dense buffers were not allocated"
        );
        assert_eq!(sb.n_points(), n_data, "bitsets of another point set");
        if q.n_clauses() == 0 {
            return self.build_dense(labels, q, n_data);
        }
        let mut read = 0u64;
        for j in 0..q.n_clauses() {
            let clause = q.clause(j);
            // A clause of one bitset label after the first: AND it in directly.
            if j > 0 && clause.len() == 1 {
                if let Some(s) = sb.slot(clause[0]) {
                    for (a, &b) in self.dense[..words].iter_mut().zip(sb.bitset(s)) {
                        *a &= b;
                    }
                    read += words as u64;
                    continue;
                }
            }
            let target = if j == 0 {
                &mut self.dense[..words]
            } else {
                &mut self.tmp[..words]
            };
            target.fill(0);
            for &l in clause {
                match sb.slot(l) {
                    Some(s) => {
                        for (t, &b) in target.iter_mut().zip(sb.bitset(s)) {
                            *t |= b;
                        }
                        read += words as u64;
                    }
                    None => {
                        for &p in labels.postings(l) {
                            target[(p >> 6) as usize] |= 1u64 << (p & 63);
                        }
                        read += labels.posting_len(l) as u64;
                    }
                }
            }
            if j > 0 {
                for (a, b) in self.dense[..words].iter_mut().zip(&self.tmp[..words]) {
                    *a &= *b;
                }
            }
        }
        read
    }

    /// The dense bitmap of the navigation clauses of `q` ([`Compiled::nav`]: all clauses, or
    /// the projected ones): the union of each clause's labels, ANDed across the clauses; labels
    /// with a bitset in `sb` are ORed in word by word, the others scattered from their postings.
    /// Without a projection it is the match bitmap of [`Self::build_dense`]; with no navigation
    /// clause every point is set. Returns the posting entries plus the bitset words read.
    pub fn build_dense_nav(
        &mut self,
        labels: &LabelStore,
        q: &Compiled,
        n_data: usize,
        sb: Option<&LabelBitsets>,
    ) -> u64 {
        let words = n_data.div_ceil(64);
        assert!(
            self.dense.len() >= words,
            "dense buffers were not allocated"
        );
        let sb = sb.filter(|b| b.n_points() == n_data);
        if q.n_clauses() == 0 || q.nav().is_empty() {
            let dense = &mut self.dense[..words];
            dense.fill(u64::MAX);
            if !n_data.is_multiple_of(64) {
                dense[words - 1] = (1u64 << (n_data % 64)) - 1;
            }
            return 0;
        }
        let mut read = 0u64;
        for (i, &j) in q.nav().iter().enumerate() {
            let target = if i == 0 {
                &mut self.dense[..words]
            } else {
                &mut self.tmp[..words]
            };
            target.fill(0);
            for &l in q.clause(j as usize) {
                match sb.and_then(|b| b.slot(l).map(|s| b.bitset(s))) {
                    Some(bits) => {
                        for (t, &b) in target.iter_mut().zip(bits) {
                            *t |= b;
                        }
                        read += words as u64;
                    }
                    None => {
                        for &p in labels.postings(l) {
                            target[(p >> 6) as usize] |= 1u64 << (p & 63);
                        }
                        read += labels.posting_len(l) as u64;
                    }
                }
            }
            if i > 0 {
                for (a, b) in self.dense[..words].iter_mut().zip(&self.tmp[..words]) {
                    *a &= *b;
                }
            }
        }
        read
    }

    /// Builds the sorted match list of `q` (galloping intersection); returns the number of
    /// candidates examined.
    pub fn build_sparse(&mut self, labels: &LabelStore, q: &Compiled, n_data: usize) -> u64 {
        self.sparse.clear();
        if q.n_clauses() == 0 {
            self.sparse.extend(0..n_data as u32);
            return n_data as u64;
        }
        let c0 = q.clause(0);
        let sparse = &mut self.sparse;
        merge_distinct(
            c0.len(),
            |i| labels.postings(c0[i]),
            &mut self.heap,
            &mut self.pos,
            |p| {
                sparse.push(p);
                true
            },
        );
        let mut examined = self.sparse.len() as u64;
        for j in 1..q.n_clauses() {
            let clause = q.clause(j);
            self.cursors.clear();
            self.cursors.resize(clause.len(), 0);
            self.tmp_sparse.clear();
            for &p in &self.sparse {
                let mut hit = false;
                for (c, &l) in self.cursors.iter_mut().zip(clause) {
                    let post = labels.postings(l);
                    *c = gallop_lower_bound(post, *c, p);
                    if *c < post.len() && post[*c] == p {
                        hit = true;
                        break;
                    }
                }
                if hit {
                    self.tmp_sparse.push(p);
                }
            }
            examined += self.sparse.len() as u64;
            std::mem::swap(&mut self.sparse, &mut self.tmp_sparse);
            if self.sparse.is_empty() {
                break;
            }
        }
        examined
    }

    /// Merge scratch (for callers that merge other lists).
    pub(crate) fn merge_scratch(&mut self) -> (&mut Vec<(u32, u32)>, &mut Vec<usize>) {
        (&mut self.heap, &mut self.pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{kernel::eval::Weights, labels::LabelStoreBuilder, predicate::PredicateSetBuilder};

    #[test]
    fn merge_yields_distinct_sorted_values() {
        let lists: [&[u32]; 4] = [&[1, 4, 9], &[], &[2, 4, 10, 11], &[0, 9, 12]];
        let (mut heap, mut pos) = (Vec::new(), Vec::new());
        let mut out = Vec::new();
        merge_distinct(
            lists.len(),
            |i| lists[i],
            &mut heap,
            &mut pos,
            |x| {
                out.push(x);
                true
            },
        );
        assert_eq!(out, [0, 1, 2, 4, 9, 10, 11, 12]);
        let mut first3 = Vec::new();
        merge_distinct(
            lists.len(),
            |i| lists[i],
            &mut heap,
            &mut pos,
            |x| {
                first3.push(x);
                first3.len() < 3
            },
        );
        assert_eq!(first3, [0, 1, 2]);
        let mut single = Vec::new();
        merge_distinct(
            1,
            |_| &[5u32, 6][..],
            &mut heap,
            &mut pos,
            |x| {
                single.push(x);
                true
            },
        );
        assert_eq!(single, [5, 6]);
    }

    #[test]
    fn dense_and_sparse_agree_with_the_predicate() {
        let mut b = LabelStoreBuilder::new();
        for i in 0..200u32 {
            let a = format!("a={}", i % 7);
            let c = format!("c={}", (i / 3) % 5);
            let mut row = vec![("a", a.as_str()), ("c", c.as_str())];
            let extra = format!("a={}", (i * 3) % 7);
            if i % 4 == 0 {
                row.push(("a", extra.as_str()));
            }
            b.push_point(row).unwrap();
        }
        let s = b.finish().unwrap();
        let mut pb = PredicateSetBuilder::new(&s);
        pb.push(&[vec!["a=1", "a=3"], vec!["c=0", "c=4"]]);
        pb.push(&[vec!["a=2"]]);
        pb.push::<&str>(&[]);
        let preds = pb.finish();
        let mut bits = vec![0u64; s.n_labels()];
        let mut q = Compiled::new();
        let mut m = Materialized::new(200, true);
        let sb = LabelBitsets::build(&s, 38);
        assert!(sb.n_bitsets() > 0 && sb.n_bitsets() < s.n_labels());
        for i in 0..preds.len() {
            q.compile(&s, preds.get(i), Weights::Unit, None, &mut bits);
            m.build_dense_static(&s, &q, 200, &sb);
            let with_bitsets = m.dense.clone();
            m.build_dense(&s, &q, 200);
            assert_eq!(m.dense, with_bitsets, "query {i}");
            m.build_sparse(&s, &q, 200);
            let expect: Vec<u32> = (0..200)
                .filter(|&p| preds.get(i).matches(s.row(p)))
                .collect();
            assert_eq!(m.sparse, expect, "query {i}");
            let from_dense: Vec<u32> = (0..200u32)
                .filter(|&p| m.dense[(p >> 6) as usize] >> (p & 63) & 1 == 1)
                .collect();
            assert_eq!(from_dense, expect, "query {i}");
            // Without a projection the navigation bitmap is the match bitmap (with or without
            // the bitsets); under projection(c*) it is the union of the binding clause.
            for b in [Some(&sb), None] {
                m.build_dense_nav(&s, &q, 200, b);
                assert_eq!(m.dense, with_bitsets, "query {i} (dense_nav)");
            }
            q.project_smallest();
            m.build_dense_nav(&s, &q, 200, Some(&sb));
            let c0: Vec<u32> = (0..200u32)
                .filter(|&p| q.n_clauses() == 0 || q.clause(0).iter().any(|l| s.row(p).contains(l)))
                .collect();
            let from_nav: Vec<u32> = (0..200u32)
                .filter(|&p| m.dense[(p >> 6) as usize] >> (p & 63) & 1 == 1)
                .collect();
            assert_eq!(from_nav, c0, "query {i} (dense_nav, projection(c*))");
            q.clear_bits(&mut bits);
        }
    }
}
