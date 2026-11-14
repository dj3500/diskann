/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Epoch-stamped visited set over node ids (no per-query clearing).

/// A visited set over `0..n` that is emptied in O(1) by starting a new epoch.
#[derive(Debug, Clone)]
pub struct Visited {
    stamps: Vec<u32>,
    epoch: u32,
}

impl Default for Visited {
    /// The empty set over no ids.
    fn default() -> Self {
        Self {
            stamps: Vec::new(),
            epoch: 1,
        }
    }
}

impl Visited {
    /// A set over `0..n`. The stamp array is written element by element (not allocated as
    /// lazily mapped zero pages), so no page fault happens during a search.
    #[expect(
        clippy::slow_vector_initialization,
        reason = "writes every page now, so that no page fault happens during a timed search"
    )]
    pub fn new(n: usize) -> Self {
        let mut stamps = Vec::with_capacity(n);
        stamps.resize(n, 0);
        Self { stamps, epoch: 1 }
    }

    /// Number of ids covered.
    pub fn len(&self) -> usize {
        self.stamps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stamps.is_empty()
    }

    /// Empties the set.
    #[inline]
    pub fn clear(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamps.fill(0);
            self.epoch = 1;
        }
    }

    /// Whether `v` is in the set.
    #[inline(always)]
    pub fn contains(&self, v: u32) -> bool {
        self.stamps[v as usize] == self.epoch
    }

    /// Inserts `v`.
    #[inline(always)]
    pub fn insert(&mut self, v: u32) {
        self.stamps[v as usize] = self.epoch;
    }

    /// Inserts `v` and returns whether it was already present.
    #[inline(always)]
    pub fn test_and_set(&mut self, v: u32) -> bool {
        let epoch = self.epoch;
        let s = &mut self.stamps[v as usize];
        if *s == epoch {
            true
        } else {
            *s = epoch;
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epochs_empty_the_set() {
        let mut v = Visited::new(10);
        assert!(!v.test_and_set(3));
        assert!(v.test_and_set(3) && v.contains(3) && !v.contains(4));
        v.clear();
        assert!(!v.contains(3));
        v.insert(4);
        assert!(v.contains(4));
        // Wrap-around of the epoch counter clears all stamps.
        v.epoch = u32::MAX;
        v.insert(5);
        v.clear();
        assert_eq!(v.epoch, 1);
        assert!(!v.contains(5) && !v.contains(4));
    }
}
