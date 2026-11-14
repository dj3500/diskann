/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The navigation beam: candidates sorted by key, with expanded flags.
//!
//! Semantics follow upstream DiskANN's `NeighborPriorityQueue`:
//! - a new entry is inserted **before** existing entries with an equal key;
//! - when the beam is full, a new entry whose key exceeds the last key is dropped; otherwise
//!   it is inserted and the last entry is evicted (also when the keys are equal);
//! - only the first `search_l` positions are expandable, and a cursor points at the first
//!   unexpanded entry.
//!
//! In resizable mode (the reference executor) nothing is evicted and every entry is
//! expandable.

/// What is known about an entry's predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum State {
    /// Not evaluated (deferred evaluation).
    Unknown = 0,
    /// Evaluated and not a match (or not admissible as an answer).
    NoMatch = 1,
    /// A match of the full predicate.
    Match = 2,
    /// Satisfies the projected clauses; the remaining clauses are checked by the collector.
    Partial = 3,
    /// An auxiliary node (never an answer).
    Aux = 4,
}

/// A beam entry (the key is stored separately).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Entry {
    pub id: u32,
    /// True squared distance to the query.
    pub d: f32,
    pub state: State,
    pub expanded: bool,
}

/// Result of [`Beam::insert`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inserted {
    /// The beam is full and the key exceeds the last key.
    Dropped,
    /// Inserted at `rank`; `evicted` is the id of the entry that fell off the end.
    At { rank: usize, evicted: Option<u32> },
}

/// Sorted candidate list with expanded flags (see the module documentation).
#[derive(Debug, Clone, Default)]
pub struct Beam {
    keys: Vec<u64>,
    entries: Vec<Entry>,
    capacity: usize,
    search_l: usize,
    cursor: usize,
    resizable: bool,
}

impl Beam {
    pub fn new() -> Self {
        Self::default()
    }

    /// Empties the beam and sets its capacity and expandable prefix. With `resizable`, the
    /// beam grows without bound, every entry is expandable and nothing is evicted.
    pub fn reset(&mut self, capacity: usize, search_l: usize, resizable: bool) {
        assert!(capacity > 0, "the beam capacity must be positive");
        self.keys.clear();
        self.entries.clear();
        self.capacity = capacity;
        self.search_l = if resizable { usize::MAX } else { search_l };
        self.resizable = resizable;
        self.cursor = 0;
        let reserve = capacity.min(1 << 20);
        if self.keys.capacity() < reserve {
            self.keys.reserve(reserve - self.keys.len());
            self.entries.reserve(reserve - self.entries.len());
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of expandable positions (the current L).
    pub fn search_l(&self) -> usize {
        self.search_l
    }

    pub fn is_resizable(&self) -> bool {
        self.resizable
    }

    /// Whether a new entry with a larger key than the last would be dropped.
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        !self.resizable && self.keys.len() == self.capacity
    }

    /// Key of the last entry.
    #[inline(always)]
    pub fn worst_key(&self) -> Option<u64> {
        self.keys.last().copied()
    }

    /// Inserts an entry (see the module documentation for the rules).
    #[inline]
    pub fn insert(&mut self, key: u64, d: f32, id: u32, state: State) -> Inserted {
        let size = self.keys.len();
        let full = size == self.capacity;
        if full {
            if self.resizable {
                self.capacity += (self.capacity >> 1).max(1);
            } else if self.keys[size - 1] < key {
                return Inserted::Dropped;
            }
        }
        let rank = lower_bound(&self.keys, key);
        let mut evicted = None;
        if full && !self.resizable {
            self.keys.pop();
            evicted = self.entries.pop().map(|e| e.id);
        }
        self.keys.insert(rank, key);
        self.entries.insert(
            rank,
            Entry {
                id,
                d,
                state,
                expanded: false,
            },
        );
        if rank < self.cursor {
            self.cursor = rank;
        }
        Inserted::At { rank, evicted }
    }

    /// Whether an unexpanded entry lies within the expandable prefix.
    #[inline(always)]
    pub fn has_unexpanded(&self) -> bool {
        self.cursor < self.search_l.min(self.keys.len())
    }

    /// Key of the closest unexpanded entry.
    #[inline(always)]
    pub fn peek_key(&self) -> Option<u64> {
        self.has_unexpanded().then(|| self.keys[self.cursor])
    }

    /// Marks the closest unexpanded entry as expanded and returns it.
    #[inline]
    pub fn pop(&mut self) -> Option<Entry> {
        if !self.has_unexpanded() {
            return None;
        }
        let at = self.cursor;
        self.entries[at].expanded = true;
        self.cursor += 1;
        while self.cursor < self.keys.len() && self.entries[self.cursor].expanded {
            self.cursor += 1;
        }
        Some(self.entries[at])
    }

    /// Marks the entry of node `id`, if present, as expanded (e.g. a node whose neighbours were
    /// already offered); returns whether it was found unexpanded.
    pub fn mark_expanded(&mut self, id: u32) -> bool {
        let Some(at) = self.entries.iter().position(|e| e.id == id) else {
            return false;
        };
        if self.entries[at].expanded {
            return false;
        }
        self.entries[at].expanded = true;
        if at == self.cursor {
            while self.cursor < self.keys.len() && self.entries[self.cursor].expanded {
                self.cursor += 1;
            }
        }
        true
    }

    /// Sets the expandable prefix and the capacity to `l` (adaptive L; never shrinks the
    /// stored entries).
    pub fn resize(&mut self, l: usize) {
        if self.resizable {
            return;
        }
        self.search_l = l;
        self.capacity = l.max(self.keys.len());
        if self.keys.capacity() < self.capacity {
            self.keys.reserve(self.capacity - self.keys.len());
            self.entries.reserve(self.capacity - self.entries.len());
        }
    }

    /// Keys in beam order.
    pub fn keys(&self) -> &[u64] {
        &self.keys
    }

    /// Entries in beam order.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }
}

/// Index of the first key `>= key`.
#[inline(always)]
fn lower_bound(keys: &[u64], key: u64) -> usize {
    // Branchless binary search.
    let mut base = 0usize;
    let mut size = keys.len();
    while size > 1 {
        let half = size / 2;
        let mid = base + half;
        // SAFETY: `mid < base + size <= keys.len()`.
        let k = unsafe { *keys.get_unchecked(mid) };
        base = if k < key { mid } else { base };
        size -= half;
    }
    // SAFETY: `base < keys.len()` when the slice is non-empty.
    if !keys.is_empty() && unsafe { *keys.get_unchecked(base) } < key {
        base + 1
    } else {
        base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(b: &Beam) -> Vec<u32> {
        b.entries().iter().map(|e| e.id).collect()
    }

    #[test]
    fn lower_bound_matches_partition_point() {
        let keys: Vec<u64> = vec![1, 3, 3, 3, 7, 9, 9, 12];
        for key in 0..14 {
            assert_eq!(
                lower_bound(&keys, key),
                keys.partition_point(|&k| k < key),
                "{key}"
            );
        }
        assert_eq!(lower_bound(&[], 5), 0);
        assert_eq!(lower_bound(&[5], 5), 0);
        assert_eq!(lower_bound(&[5], 6), 1);
    }

    #[test]
    fn ties_go_first_and_full_beam_evicts() {
        let mut b = Beam::new();
        b.reset(3, 3, false);
        assert_eq!(
            b.insert(5, 5.0, 1, State::Match),
            Inserted::At {
                rank: 0,
                evicted: None
            }
        );
        b.insert(5, 5.0, 2, State::Match);
        assert_eq!(ids(&b), [2, 1], "a new entry goes before an equal key");
        b.insert(7, 7.0, 3, State::NoMatch);
        assert!(b.is_full());
        assert_eq!(b.insert(8, 8.0, 4, State::Match), Inserted::Dropped);
        // Equal to the last key: inserted, the old last entry is evicted.
        assert_eq!(
            b.insert(7, 7.0, 5, State::Match),
            Inserted::At {
                rank: 2,
                evicted: Some(3)
            }
        );
        assert_eq!(ids(&b), [2, 1, 5]);
        assert_eq!(b.worst_key(), Some(7));
    }

    #[test]
    fn cursor_follows_insertions_and_search_l() {
        let mut b = Beam::new();
        b.reset(4, 2, false);
        b.insert(10, 10.0, 1, State::Match);
        b.insert(20, 20.0, 2, State::Match);
        b.insert(30, 30.0, 3, State::Match);
        assert_eq!(b.pop().map(|e| e.id), Some(1));
        assert_eq!(b.pop().map(|e| e.id), Some(2));
        // Position 2 is outside the expandable prefix.
        assert!(!b.has_unexpanded() && b.pop().is_none());
        b.insert(15, 15.0, 4, State::Match);
        assert_eq!(b.peek_key(), Some(15));
        assert_eq!(b.pop().map(|e| e.id), Some(4));
        b.resize(4);
        assert_eq!(b.pop().map(|e| e.id), Some(3));
        assert!(b.pop().is_none());
        assert!(b.entries().iter().all(|e| e.expanded));
    }

    #[test]
    fn resizable_beam_never_evicts() {
        let mut b = Beam::new();
        b.reset(2, 2, true);
        for i in 0..10u32 {
            b.insert(u64::from(100 - i), 0.0, i, State::Unknown);
        }
        assert_eq!(b.len(), 10);
        assert!(!b.is_full());
        let mut popped = Vec::new();
        while let Some(e) = b.pop() {
            popped.push(e.id);
        }
        assert_eq!(popped, (0..10).rev().collect::<Vec<_>>());
    }

    #[test]
    fn mark_expanded_advances_the_cursor() {
        let mut b = Beam::new();
        b.reset(4, 4, false);
        b.insert(1, 1.0, 7, State::Match);
        b.insert(2, 2.0, 8, State::Match);
        assert!(b.mark_expanded(7));
        assert!(!b.mark_expanded(7) && !b.mark_expanded(99));
        assert_eq!(b.pop().map(|e| e.id), Some(8));
    }
}
