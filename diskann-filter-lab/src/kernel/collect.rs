/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Answer collection: `final_beam`, `heap(K)` and `all(L₀)`.
//!
//! Answers are kept as packed `(true distance, id)` keys ([`super::pack`]), so their order is
//! the order by distance, then id.

use super::{pack, unpack};

/// Collector role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Collector {
    /// The matching members of the final beam (post-filter); the predicate is evaluated (or
    /// completed, under a clause projection) at the end.
    FinalBeam,
    /// A bounded max-heap of the `K` best matches among all evaluated nodes.
    Heap,
    /// Every evaluated match, sorted at the end and truncated to `L₀`, then `K` (upstream
    /// inline search).
    All,
}

/// Collected answers of one query.
#[derive(Debug, Clone, Default)]
pub struct Answers {
    kind: Option<Collector>,
    k: usize,
    l0: usize,
    /// `heap(K)`: binary max-heap of packed answers; `all`/`final_beam`: unordered list.
    items: Vec<u64>,
}

impl Answers {
    pub fn new() -> Self {
        Self::default()
    }

    /// Empties the collector for a query with `k` answers and initial list size `l0`.
    pub fn reset(&mut self, kind: Collector, k: usize, l0: usize) {
        self.kind = Some(kind);
        self.k = k;
        self.l0 = l0;
        self.items.clear();
    }

    /// Number of answers held.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Whether a bounded heap holds `K` answers.
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.kind == Some(Collector::Heap) && self.items.len() >= self.k
    }

    /// Distance of the worst answer of a full heap.
    #[inline(always)]
    pub fn worst(&self) -> f32 {
        self.items.first().map_or(f32::INFINITY, |&p| unpack(p).0)
    }

    /// Offers a match. Returns `(inserted, evicted id)` for `heap(K)`; other collectors keep
    /// every offered match.
    #[inline]
    pub fn offer(&mut self, d: f32, id: u32) -> (bool, Option<u32>) {
        let p = pack(d, id);
        match self.kind {
            Some(Collector::Heap) => {
                if self.items.len() < self.k {
                    self.items.push(p);
                    sift_up(&mut self.items);
                    (true, None)
                } else if self.k > 0 && p < self.items[0] {
                    let old = self.items[0] as u32;
                    self.items[0] = p;
                    sift_down(&mut self.items);
                    (true, Some(old))
                } else {
                    (false, None)
                }
            }
            _ => {
                self.items.push(p);
                (true, None)
            }
        }
    }

    /// [`Self::offer`] for `heap(K)` that ignores a point already held (merging answers found
    /// by another search into a seeded heap). Returns whether the point was inserted.
    #[inline]
    pub fn offer_unique(&mut self, d: f32, id: u32) -> bool {
        let p = pack(d, id);
        if self.kind == Some(Collector::Heap)
            && self.items.len() >= self.k
            && (self.k == 0 || p >= self.items[0])
        {
            return false;
        }
        if self.items.iter().any(|&x| x as u32 == id) {
            return false;
        }
        self.offer(d, id).0
    }

    /// Whether point `id` is held.
    pub fn contains(&self, id: u32) -> bool {
        self.items.iter().any(|&x| x as u32 == id)
    }

    /// The `K` of the collector.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Writes the answers, sorted by (distance, id), into `ids`/`dists` (at most
    /// `ids.len()`; `all` keeps the first `L₀` first) and returns their number. The caller
    /// pads the rest.
    pub fn finish(&mut self, ids: &mut [u32], dists: &mut [f32]) -> usize {
        self.items.sort_unstable();
        if self.kind == Some(Collector::All) {
            self.items.truncate(self.l0);
        }
        let n = self.items.len().min(ids.len()).min(self.k);
        for (i, &p) in self.items[..n].iter().enumerate() {
            let (d, id) = unpack(p);
            ids[i] = id;
            dists[i] = d;
        }
        n
    }

    /// The answers held, sorted by (distance, id) (for tests and traces).
    pub fn sorted(&self) -> Vec<(f32, u32)> {
        let mut v = self.items.clone();
        v.sort_unstable();
        v.into_iter().map(unpack).collect()
    }
}

#[inline]
fn sift_up(h: &mut [u64]) {
    let mut i = h.len() - 1;
    while i > 0 {
        let parent = (i - 1) / 2;
        if h[parent] >= h[i] {
            break;
        }
        h.swap(parent, i);
        i = parent;
    }
}

#[inline]
fn sift_down(h: &mut [u64]) {
    let n = h.len();
    let mut i = 0;
    loop {
        let l = 2 * i + 1;
        if l >= n {
            break;
        }
        let r = l + 1;
        let c = if r < n && h[r] > h[l] { r } else { l };
        if h[i] >= h[c] {
            break;
        }
        h.swap(i, c);
        i = c;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_keeps_the_k_best_by_distance_then_id() {
        let mut a = Answers::new();
        a.reset(Collector::Heap, 3, 10);
        for (d, id) in [(5.0, 1), (1.0, 2), (3.0, 3), (3.0, 0), (9.0, 4), (1.0, 5)] {
            a.offer(d, id);
        }
        assert!(a.is_full());
        assert_eq!(a.worst(), 3.0);
        assert_eq!(
            a.offer(3.0, 7),
            (false, None),
            "(3, 7) is worse than (3, 0)"
        );
        assert_eq!(a.offer(2.0, 9), (true, Some(0)));
        let (mut ids, mut ds) = ([0u32; 5], [0f32; 5]);
        assert_eq!(a.finish(&mut ids, &mut ds), 3);
        assert_eq!(&ids[..3], &[2, 5, 9]);
        assert_eq!(&ds[..3], &[1.0, 1.0, 2.0]);
    }

    #[test]
    fn all_truncates_to_l0_then_k() {
        let mut a = Answers::new();
        a.reset(Collector::All, 2, 3);
        for (d, id) in [(4.0, 1), (2.0, 2), (3.0, 3), (1.0, 4)] {
            a.offer(d, id);
        }
        let (mut ids, mut ds) = ([0u32; 2], [0f32; 2]);
        assert_eq!(a.finish(&mut ids, &mut ds), 2);
        assert_eq!(ids, [4, 2]);
    }
}
