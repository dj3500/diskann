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
//! - **row AND** (the exact plan `rowand`): the words of the data-point bitmap that hold
//!   members of the first clause (`c*`), each ANDed with the other clauses word by word (the
//!   static bitset words of their frequent labels; the postings of their other labels through
//!   galloping cursors), stopping at the first clause that clears it, then enumerated.
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
    /// Row AND: a bitmap over the data points and its summary (one bit per word), all zero
    /// between queries; the words of `c*`'s members and their bits; cursors in postings.
    rw_bits: Vec<u64>,
    rw_summary: Vec<u64>,
    rw_words: Vec<u32>,
    rw_masks: Vec<u64>,
    rw_cursors: Vec<usize>,
}

/// Row AND: bitset words are prefetched this many surviving words ahead, for the first
/// `RW_PREFETCH` bitset labels of the clause.
const RW_AHEAD: usize = 16;
const RW_PREFETCH: usize = 4;

impl Materialized {
    /// Buffers for `n_data` points (the dense bitmaps are allocated and written now, so that
    /// no page fault happens during a search).
    pub fn new(n_data: usize, dense: bool) -> Self {
        let words = if dense { n_data.div_ceil(64) } else { 0 };
        let mut m = Self::default();
        m.dense.resize(words, 0);
        m.tmp.resize(words, 0);
        m.rw_bits.resize(words, 0);
        m.rw_summary.resize(words.div_ceil(64), 0);
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
        if self.rw_bits.len() < words {
            self.rw_bits.resize(words, 0);
            self.rw_summary.resize(words.div_ceil(64), 0);
        }
    }

    /// Bytes of the dense buffers (both bitmaps, the row AND's bitmap and summary).
    pub fn dense_bytes(&self) -> u64 {
        8 * (self.dense.len() + self.tmp.len() + self.rw_bits.len() + self.rw_summary.len()) as u64
    }

    /// Row AND of `q` (see the module documentation): the sorted matches into
    /// `self.matches`. With `sb` (the query split against those static bitsets by
    /// [`Compiled::split_static`]) the split's bitset labels are read in their bitsets and its
    /// other labels in their postings; without, every label in its postings. Returns the
    /// members of `union(c*)` (the candidates). The buffers must be allocated
    /// ([`Self::ensure_dense`]).
    pub fn build_rowand(
        &mut self,
        labels: &LabelStore,
        q: &Compiled,
        n_data: usize,
        sb: Option<&LabelBitsets>,
    ) -> u64 {
        let words = n_data.div_ceil(64);
        assert!(
            self.rw_bits.len() >= words,
            "dense buffers were not allocated"
        );
        self.matches.clear();
        let nc = q.n_clauses();
        if nc == 0 {
            self.matches.extend(0..n_data as u32);
            return n_data as u64;
        }
        let hot = |j: usize| -> &[u32] {
            match sb {
                Some(_) => q.static_slots(j),
                None => &[],
            }
        };
        let rare = |j: usize| -> &[u32] {
            match sb {
                Some(_) => q.row_labels(j),
                None => q.clause(j),
            }
        };
        let (wl, wm) = (&mut self.rw_words, &mut self.rw_masks);
        wl.clear();
        wm.clear();
        // The words of c*'s members and their bits, in increasing order.
        let (h0, r0) = (hot(0), rare(0));
        let mut lists = r0.iter().filter(|&&l| labels.posting_len(l) > 0);
        match (h0, lists.next(), lists.next()) {
            ([], None, _) => {}
            ([], Some(&l), None) => {
                // One posting list: its entries grouped by word.
                let (mut cur, mut m) = (u32::MAX, 0u64);
                for &p in labels.postings(l) {
                    let w = p >> 6;
                    if w != cur {
                        if m != 0 {
                            wl.push(cur);
                            wm.push(m);
                        }
                        cur = w;
                        m = 0;
                    }
                    m |= 1u64 << (p & 63);
                }
                if m != 0 {
                    wl.push(cur);
                    wm.push(m);
                }
            }
            ([], Some(_), Some(_)) => {
                // Several lists: scattered into the bitmap and its summary, collected in word
                // order (clearing both).
                let (bits, summary) = (&mut self.rw_bits, &mut self.rw_summary);
                for &l in r0 {
                    for &p in labels.postings(l) {
                        let w = (p >> 6) as usize;
                        bits[w] |= 1u64 << (p & 63);
                        summary[w >> 6] |= 1u64 << (w & 63);
                    }
                }
                for (i, s) in summary[..words.div_ceil(64)].iter_mut().enumerate() {
                    let mut x = *s;
                    *s = 0;
                    while x != 0 {
                        let w = (i << 6) | x.trailing_zeros() as usize;
                        x &= x - 1;
                        wl.push(w as u32);
                        wm.push(bits[w]);
                        bits[w] = 0;
                    }
                }
            }
            (_, first, _) => {
                // Bitset labels: every word streamed, the other labels scattered first.
                let sb = sb.expect("bitset labels come from the static bitsets");
                let bits = &mut self.rw_bits;
                let scattered = first.is_some();
                for &l in r0 {
                    for &p in labels.postings(l) {
                        bits[(p >> 6) as usize] |= 1u64 << (p & 63);
                    }
                }
                for (w, b) in bits[..words].iter_mut().enumerate() {
                    let mut x = *b;
                    if scattered {
                        *b = 0;
                    }
                    for &s in h0 {
                        x |= sb.word(s, w);
                    }
                    if x != 0 {
                        wl.push(w as u32);
                        wm.push(x);
                    }
                }
            }
        }
        let cands: u64 = wm.iter().map(|m| u64::from(m.count_ones())).sum();
        // Clause by clause, the surviving words (compacted in place) ANDed with the clause:
        // its bitset words (most frequent label first, until the word's members are covered;
        // prefetched ahead for the first labels), then the postings of its labels without a
        // bitset through cursors, which move only when a word reaches the clause's next
        // entry (a lower bound: the words before it hold none of those entries).
        let (cursors, mut len) = (&mut self.rw_cursors, wl.len());
        for j in 1..nc {
            if len == 0 {
                break;
            }
            let (hs, rs) = (hot(j), rare(j));
            cursors.clear();
            cursors.resize(rs.len(), 0);
            // Lower bound of the next word holding an entry of `rs` (0: not known before the
            // first word that needs them).
            let mut next = 0u32;
            let ahead = &hs[..hs.len().min(RW_PREFETCH)];
            let mut kept = 0;
            for i in 0..len {
                if let (Some(sb), Some(&a)) = (sb, wl.get(i + RW_AHEAD)) {
                    if i + RW_AHEAD < len {
                        for &s in ahead {
                            super::prefetch(sb.word_ptr(s, a as usize));
                        }
                    }
                }
                let w = wl[i];
                let x = wm[i];
                let mut y = 0u64;
                if let Some(sb) = sb {
                    for &s in hs {
                        y |= sb.word(s, w as usize);
                        if x & !y == 0 {
                            break;
                        }
                    }
                }
                if x & !y != 0 && next <= w {
                    let lo = w << 6;
                    let mut nx = u32::MAX;
                    for (t, &l) in rs.iter().enumerate() {
                        let post = labels.postings(l);
                        let mut p = gallop_lower_bound(post, cursors[t], lo);
                        while let Some(&v) = post.get(p) {
                            if v >> 6 != w {
                                nx = nx.min(v >> 6);
                                break;
                            }
                            y |= 1u64 << (v & 63);
                            p += 1;
                        }
                        cursors[t] = p;
                    }
                    next = nx;
                }
                let x = x & y;
                if x != 0 {
                    wl[kept] = w;
                    wm[kept] = x;
                    kept += 1;
                }
            }
            len = kept;
        }
        let out = &mut self.matches;
        for (&w, &x) in wl[..len].iter().zip(&wm[..len]) {
            let mut x = x;
            while x != 0 {
                out.push((w << 6) | x.trailing_zeros());
                x &= x - 1;
            }
        }
        cands
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
            q.clear_bits(&mut bits);
        }
    }

    #[test]
    fn rowand_agrees_with_the_predicate() {
        // Labels of skewed frequencies in three categories (some with a static bitset), and
        // random CNF predicates of 1 to 4 clauses of 1 to 4 labels.
        let n = 5000u32;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move |m: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % m
        };
        let mut b = LabelStoreBuilder::new();
        for _ in 0..n {
            let mut row = Vec::new();
            for (cat, k) in [("a", 12u64), ("b", 40), ("c", 6)] {
                let picks = 1 + rnd(3);
                for _ in 0..picks {
                    // Zipf-like: small values are frequent.
                    let v = rnd(k).min(rnd(k)).min(rnd(k));
                    row.push((cat, format!("{cat}={v}")));
                }
            }
            row.sort();
            row.dedup();
            b.push_point(row.iter().map(|(c, l)| (*c, l.as_str())))
                .unwrap();
        }
        let s = b.finish().unwrap();
        let mut names: Vec<String> = s.name_index().keys().map(|k| k.to_string()).collect();
        names.sort();
        let mut pb = PredicateSetBuilder::new(&s);
        for _ in 0..300 {
            let clauses: Vec<Vec<&str>> = (0..1 + rnd(4))
                .map(|_| {
                    (0..1 + rnd(4))
                        .map(|_| names[rnd(names.len() as u64) as usize].as_str())
                        .collect()
                })
                .collect();
            pb.push(&clauses);
        }
        pb.push::<&str>(&[]);
        let preds = pb.finish();
        let sb = LabelBitsets::build(&s, u64::from(n).div_ceil(64));
        assert!(sb.n_bitsets() > 0 && sb.n_bitsets() < s.n_labels());
        let mut bits = vec![0u64; s.n_labels()];
        let mut q = Compiled::new();
        let mut m = Materialized::new(n as usize, true);
        let mut shapes = [0usize; 3];
        for i in 0..preds.len() {
            q.compile(&s, preds.get(i), Weights::Unit, None, &mut bits);
            let expect: Vec<u32> = (0..n).filter(|&p| preds.get(i).matches(s.row(p))).collect();
            m.build_rowand(&s, &q, n as usize, None);
            assert_eq!(m.matches, expect, "query {i} without bitsets");
            q.split_static(&s, &sb, sb.theta(), false);
            let cands = m.build_rowand(&s, &q, n as usize, Some(&sb));
            assert_eq!(m.matches, expect, "query {i} with bitsets");
            assert!(m.rw_bits.iter().chain(&m.rw_summary).all(|&w| w == 0));
            if q.n_clauses() > 0 {
                let mut union: Vec<u32> = q
                    .clause(0)
                    .iter()
                    .flat_map(|&l| s.postings(l).iter().copied())
                    .collect();
                union.sort_unstable();
                union.dedup();
                assert_eq!(cands, union.len() as u64, "query {i}");
                let lists = q
                    .row_labels(0)
                    .iter()
                    .filter(|&&l| s.posting_len(l) > 0)
                    .count();
                shapes[match (q.static_slots(0).is_empty(), lists) {
                    (false, _) => 2,
                    (true, 0 | 1) => 0,
                    (true, _) => 1,
                }] += 1;
            }
            q.clear_bits(&mut bits);
        }
        assert!(shapes.iter().all(|&c| c > 0), "every c* shape: {shapes:?}");
    }
}
