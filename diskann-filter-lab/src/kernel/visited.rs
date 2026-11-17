/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Visited set over node ids: a bitset with a list of the words in use, emptied in time
//! proportional to the number of words touched (no per-query clearing of the whole set).

/// A visited set over `0..n`. One bit per node keeps the set small enough to stay in the
/// private cache (125 KiB per million nodes), unlike a 4-byte stamp per node; the words that
/// became non-zero are listed, so that emptying the set only rewrites those.
#[derive(Debug, Clone, Default)]
pub struct Visited {
    bits: Vec<u64>,
    /// Indices of the non-zero words of `bits` (capacity for all words: never reallocates).
    touched: Vec<u32>,
    n: usize,
}

impl Visited {
    /// A set over `0..n`. The arrays are written element by element (not allocated as lazily
    /// mapped zero pages), so no page fault happens during a search.
    #[expect(
        clippy::slow_vector_initialization,
        reason = "writes every page now, so that no page fault happens during a timed search"
    )]
    pub fn new(n: usize) -> Self {
        let words = n.div_ceil(64);
        let bits = crate::hugepages::filled(words, 0);
        let mut touched = Vec::with_capacity(words);
        touched.resize(words, 0);
        touched.clear();
        Self { bits, touched, n }
    }

    /// Number of ids covered.
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Empties the set.
    #[inline]
    pub fn clear(&mut self) {
        for &w in &self.touched {
            self.bits[w as usize] = 0;
        }
        self.touched.clear();
    }

    /// Whether `v` is in the set.
    #[inline(always)]
    pub fn contains(&self, v: u32) -> bool {
        self.bits[(v >> 6) as usize] & (1 << (v & 63)) != 0
    }

    /// Inserts `v`.
    #[inline(always)]
    pub fn insert(&mut self, v: u32) {
        self.test_and_set(v);
    }

    /// Inserts `v` and returns whether it was already present.
    #[inline(always)]
    pub fn test_and_set(&mut self, v: u32) -> bool {
        let i = (v >> 6) as usize;
        let m = 1u64 << (v & 63);
        let w = self.bits[i];
        if w & m != 0 {
            return true;
        }
        if w == 0 {
            // At most one push per word: `touched` was allocated for all of them.
            self.touched.push(i as u32);
        }
        self.bits[i] = w | m;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clearing_empties_the_set() {
        let mut v = Visited::new(200);
        assert!(!v.test_and_set(3));
        assert!(v.test_and_set(3) && v.contains(3) && !v.contains(4));
        v.insert(4);
        v.insert(130);
        v.insert(199);
        assert!(v.contains(4) && v.contains(130) && v.contains(199));
        assert_eq!(v.touched.len(), 3, "one entry per word in use");
        v.clear();
        assert!((0..200).all(|i| !v.contains(i)));
        assert!(v.touched.is_empty() && v.bits.iter().all(|&w| w == 0));
        for i in 0..200 {
            assert!(!v.test_and_set(i));
        }
        assert_eq!(v.touched.len(), 4);
        v.clear();
        assert!(!v.contains(64));
    }
}
