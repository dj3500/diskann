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
//!
//! Layout: keys (`u32` for scores that are one `f32`, `u64` for lexicographic keys), slots
//! `(id, state, expanded)` of 8 bytes, and the true distances only when they differ from the
//! key (with distance keys the distance is the key), so an insertion moves 12 bytes per entry
//! behind it for distance keys, as upstream's queue does.

use super::funkey;

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

/// A beam entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Entry {
    pub id: u32,
    /// True squared distance to the query.
    pub d: f32,
    pub state: State,
    pub expanded: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Slot {
    id: u32,
    state: State,
    expanded: bool,
}

/// Key types of the beam.
pub trait BeamKey: Copy + Ord + Send + Sync + std::fmt::Debug + 'static {
    /// The key as a `u64` (order preserving).
    fn wide(self) -> u64;
}

impl BeamKey for u32 {
    #[inline(always)]
    fn wide(self) -> u64 {
        u64::from(self)
    }
}

impl BeamKey for u64 {
    #[inline(always)]
    fn wide(self) -> u64 {
        self
    }
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
#[derive(Debug, Clone)]
pub struct Beam<K> {
    keys: Vec<K>,
    slots: Vec<Slot>,
    dists: Vec<f32>,
    store_d: bool,
    capacity: usize,
    search_l: usize,
    cursor: usize,
    resizable: bool,
}

impl<K: BeamKey> Default for Beam<K> {
    fn default() -> Self {
        Self {
            keys: Vec::new(),
            slots: Vec::new(),
            dists: Vec::new(),
            store_d: true,
            capacity: 0,
            search_l: 0,
            cursor: 0,
            resizable: false,
        }
    }
}

impl<K: BeamKey> Beam<K> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Empties the beam and sets its capacity and expandable prefix. With `resizable`, the
    /// beam grows without bound, every entry is expandable and nothing is evicted. Without
    /// `store_d`, the true distance of an entry is the `f32` encoded by its key.
    pub fn reset(&mut self, capacity: usize, search_l: usize, resizable: bool, store_d: bool) {
        assert!(capacity > 0, "the beam capacity must be positive");
        self.keys.clear();
        self.slots.clear();
        self.dists.clear();
        self.store_d = store_d;
        self.capacity = capacity;
        self.search_l = if resizable { usize::MAX } else { search_l };
        self.resizable = resizable;
        self.cursor = 0;
        let reserve = capacity.min(1 << 20);
        if self.keys.capacity() < reserve {
            self.keys.reserve(reserve);
            self.slots.reserve(reserve);
            self.dists.reserve(reserve);
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
    pub fn worst_key(&self) -> Option<K> {
        self.keys.last().copied()
    }

    /// Inserts an entry (see the module documentation for the rules).
    #[inline(always)]
    pub fn insert(&mut self, key: K, d: f32, id: u32, state: State) -> Inserted {
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
            evicted = self.slots.pop().map(|e| e.id);
            if self.store_d {
                self.dists.pop();
            }
        }
        self.keys.insert(rank, key);
        self.slots.insert(
            rank,
            Slot {
                id,
                state,
                expanded: false,
            },
        );
        if self.store_d {
            self.dists.insert(rank, d);
        }
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
    pub fn peek_key(&self) -> Option<K> {
        self.has_unexpanded().then(|| self.keys[self.cursor])
    }

    /// Id of the closest unexpanded entry (the node the next [`Beam::pop`] returns unless a
    /// closer node is inserted first).
    #[inline(always)]
    pub fn peek_id(&self) -> Option<u32> {
        self.has_unexpanded().then(|| self.slots[self.cursor].id)
    }

    /// Marks the closest unexpanded entry as expanded and returns its id.
    #[inline]
    pub fn pop(&mut self) -> Option<u32> {
        if !self.has_unexpanded() {
            return None;
        }
        let at = self.cursor;
        self.slots[at].expanded = true;
        self.cursor += 1;
        while self.cursor < self.keys.len() && self.slots[self.cursor].expanded {
            self.cursor += 1;
        }
        Some(self.slots[at].id)
    }

    /// Marks the entry of node `id`, if present, as expanded (e.g. a node whose neighbours were
    /// already offered); returns whether it was found unexpanded.
    pub fn mark_expanded(&mut self, id: u32) -> bool {
        let Some(at) = self.slots.iter().position(|e| e.id == id) else {
            return false;
        };
        if self.slots[at].expanded {
            return false;
        }
        self.slots[at].expanded = true;
        if at == self.cursor {
            while self.cursor < self.keys.len() && self.slots[self.cursor].expanded {
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
            let more = self.capacity - self.keys.len();
            self.keys.reserve(more);
            self.slots.reserve(more);
            self.dists.reserve(more);
        }
    }

    /// Keys in beam order.
    pub fn keys(&self) -> &[K] {
        &self.keys
    }

    /// Entry at position `i`.
    #[inline]
    pub fn entry(&self, i: usize) -> Entry {
        let s = self.slots[i];
        Entry {
            id: s.id,
            d: if self.store_d {
                self.dists[i]
            } else {
                funkey(self.keys[i].wide() as u32)
            },
            state: s.state,
            expanded: s.expanded,
        }
    }

    /// Entries in beam order.
    pub fn entries(&self) -> impl ExactSizeIterator<Item = Entry> + '_ {
        (0..self.keys.len()).map(|i| self.entry(i))
    }
}

/// The beam operations of the post-filter loop (distance keys, distances decoded from the
/// keys), implemented by [`Beam<u32>`] and [`PackedBeam`].
pub trait PostfilterBeam: Default {
    /// [`Beam::reset`] with `search_l = capacity`, not resizable, distances from the keys.
    fn start(&mut self, capacity: usize);
    fn offer(&mut self, key: u32, id: u32, state: State) -> Inserted;
    fn pop(&mut self) -> Option<u32>;
    fn peek_id(&self) -> Option<u32>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn entry(&self, i: usize) -> Entry;
    fn search_l(&self) -> usize;
}

impl PostfilterBeam for Beam<u32> {
    #[inline]
    fn start(&mut self, capacity: usize) {
        self.reset(capacity, capacity, false, false);
    }

    #[inline(always)]
    fn offer(&mut self, key: u32, id: u32, state: State) -> Inserted {
        self.insert(key, 0.0, id, state)
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<u32> {
        Beam::pop(self)
    }

    #[inline(always)]
    fn peek_id(&self) -> Option<u32> {
        Beam::peek_id(self)
    }

    #[inline(always)]
    fn len(&self) -> usize {
        Beam::len(self)
    }

    #[inline(always)]
    fn entry(&self, i: usize) -> Entry {
        Beam::entry(self, i)
    }

    #[inline(always)]
    fn search_l(&self) -> usize {
        Beam::search_l(self)
    }
}

/// Bits of a [`PackedBeam`] entry below the key: the node id, its state and the expanded flag.
const PACKED_ID: u64 = (1 << 28) - 1;
/// Graphs with at most this many nodes fit [`PackedBeam`]'s 28-bit ids.
pub const PACKED_MAX_NODES: usize = 1 << 28;
const PACKED_STATE_SHIFT: u32 = 28;
const PACKED_EXPANDED: u64 = 1 << 31;

/// A beam of distance keys whose entries are one `u64` each: the key in the high 32 bits, then
/// the expanded flag, the state (3 bits) and the node id (28 bits, ids below 2^28). The same
/// semantics as [`Beam<u32>`] without stored distances (ties go first, a full beam evicts its
/// last entry, the cursor points at the first unexpanded entry), but an insertion searches and
/// moves one array of 8-byte entries instead of a key array and a slot array (12 bytes per
/// entry in two moves).
#[derive(Debug, Clone, Default)]
pub struct PackedBeam {
    entries: Vec<u64>,
    capacity: usize,
    cursor: usize,
}

impl PackedBeam {
    #[inline(always)]
    fn state_of(e: u64) -> State {
        match (e >> PACKED_STATE_SHIFT) & 7 {
            0 => State::Unknown,
            1 => State::NoMatch,
            2 => State::Match,
            3 => State::Partial,
            _ => State::Aux,
        }
    }
}

impl PostfilterBeam for PackedBeam {
    fn start(&mut self, capacity: usize) {
        assert!(capacity > 0, "the beam capacity must be positive");
        self.entries.clear();
        self.capacity = capacity;
        self.cursor = 0;
        let reserve = capacity.min(1 << 20);
        if self.entries.capacity() < reserve {
            self.entries.reserve(reserve);
        }
    }

    #[inline(always)]
    fn offer(&mut self, key: u32, id: u32, state: State) -> Inserted {
        debug_assert!(u64::from(id) <= PACKED_ID, "node id {id} exceeds 28 bits");
        let size = self.entries.len();
        let full = size == self.capacity;
        if full && (self.entries[size - 1] >> 32) < u64::from(key) {
            return Inserted::Dropped;
        }
        // The first entry whose key is not below `key` (entries with an equal key compare
        // greater than the key with zero low bits): the new entry goes before them.
        let probe = u64::from(key) << 32;
        let rank = lower_bound(&self.entries, probe);
        let evicted = if full {
            self.entries.pop().map(|e| (e & PACKED_ID) as u32)
        } else {
            None
        };
        self.entries.insert(
            rank,
            probe | (u64::from(state as u8) << PACKED_STATE_SHIFT) | u64::from(id),
        );
        if rank < self.cursor {
            self.cursor = rank;
        }
        Inserted::At { rank, evicted }
    }

    #[inline]
    fn pop(&mut self) -> Option<u32> {
        if self.cursor >= self.entries.len() {
            return None;
        }
        let at = self.cursor;
        self.entries[at] |= PACKED_EXPANDED;
        self.cursor += 1;
        while self.cursor < self.entries.len() && self.entries[self.cursor] & PACKED_EXPANDED != 0 {
            self.cursor += 1;
        }
        Some((self.entries[at] & PACKED_ID) as u32)
    }

    #[inline(always)]
    fn peek_id(&self) -> Option<u32> {
        self.entries
            .get(self.cursor)
            .map(|&e| (e & PACKED_ID) as u32)
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    fn entry(&self, i: usize) -> Entry {
        let e = self.entries[i];
        Entry {
            id: (e & PACKED_ID) as u32,
            d: funkey((e >> 32) as u32),
            state: Self::state_of(e),
            expanded: e & PACKED_EXPANDED != 0,
        }
    }

    #[inline(always)]
    fn search_l(&self) -> usize {
        self.capacity
    }
}

/// Index of the first key `>= key`.
#[inline(always)]
fn lower_bound<K: BeamKey>(keys: &[K], key: K) -> usize {
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
    use crate::kernel::fkey;

    fn ids<K: BeamKey>(b: &Beam<K>) -> Vec<u32> {
        b.entries().map(|e| e.id).collect()
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
        assert_eq!(lower_bound::<u32>(&[], 5), 0);
        assert_eq!(lower_bound(&[5u32], 5), 0);
        assert_eq!(lower_bound(&[5u32], 6), 1);
    }

    /// The packed beam behaves exactly like `Beam<u32>` without stored distances: the same
    /// insertion results, pops, peeks and entries on random sequences with many tied keys.
    #[test]
    fn the_packed_beam_matches_the_beam() {
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |n: u64| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng % n
        };
        for cap in [1usize, 2, 5, 17, 64] {
            let (mut a, mut b) = (Beam::<u32>::new(), PackedBeam::default());
            PostfilterBeam::start(&mut a, cap);
            b.start(cap);
            for step in 0..2000 {
                if next(4) == 0 {
                    assert_eq!(
                        PostfilterBeam::peek_id(&a),
                        b.peek_id(),
                        "cap {cap} step {step}"
                    );
                    assert_eq!(
                        PostfilterBeam::pop(&mut a),
                        b.pop(),
                        "cap {cap} step {step}"
                    );
                } else {
                    let key = fkey(next(40) as f32);
                    let id = next(1 << 28) as u32;
                    let state = [State::Unknown, State::Aux, State::Match][next(3) as usize];
                    assert_eq!(
                        a.offer(key, id, state),
                        b.offer(key, id, state),
                        "cap {cap} step {step}"
                    );
                }
                assert_eq!(PostfilterBeam::len(&a), b.len());
                for i in 0..b.len() {
                    assert_eq!(
                        PostfilterBeam::entry(&a, i),
                        b.entry(i),
                        "cap {cap} step {step} at {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn ties_go_first_and_full_beam_evicts() {
        let mut b = Beam::<u64>::new();
        b.reset(3, 3, false, true);
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
            b.insert(7, 7.5, 5, State::Match),
            Inserted::At {
                rank: 2,
                evicted: Some(3)
            }
        );
        assert_eq!(ids(&b), [2, 1, 5]);
        assert_eq!(b.worst_key(), Some(7));
        assert_eq!(b.entry(2).d, 7.5);
    }

    #[test]
    fn distances_from_keys() {
        let mut b = Beam::<u32>::new();
        b.reset(4, 4, false, false);
        b.insert(fkey(2.5), 2.5, 1, State::Unknown);
        b.insert(fkey(0.5), 0.5, 2, State::Unknown);
        let d: Vec<f32> = b.entries().map(|e| e.d).collect();
        assert_eq!(d, [0.5, 2.5]);
    }

    #[test]
    fn cursor_follows_insertions_and_search_l() {
        let mut b = Beam::<u32>::new();
        b.reset(4, 2, false, true);
        b.insert(10, 10.0, 1, State::Match);
        b.insert(20, 20.0, 2, State::Match);
        b.insert(30, 30.0, 3, State::Match);
        assert_eq!(b.pop(), Some(1));
        assert_eq!(b.pop(), Some(2));
        // Position 2 is outside the expandable prefix.
        assert!(!b.has_unexpanded() && b.pop().is_none());
        b.insert(15, 15.0, 4, State::Match);
        assert_eq!(b.peek_key(), Some(15));
        assert_eq!(b.pop(), Some(4));
        b.resize(4);
        assert_eq!(b.pop(), Some(3));
        assert!(b.pop().is_none());
        assert!(b.entries().all(|e| e.expanded));
    }

    #[test]
    fn resizable_beam_never_evicts() {
        let mut b = Beam::<u32>::new();
        b.reset(2, 2, true, true);
        for i in 0..10u32 {
            b.insert(100 - i, 0.0, i, State::Unknown);
        }
        assert_eq!(b.len(), 10);
        assert!(!b.is_full());
        let mut popped = Vec::new();
        while let Some(id) = b.pop() {
            popped.push(id);
        }
        assert_eq!(popped, (0..10).rev().collect::<Vec<_>>());
    }

    #[test]
    fn mark_expanded_advances_the_cursor() {
        let mut b = Beam::<u32>::new();
        b.reset(4, 4, false, true);
        b.insert(1, 1.0, 7, State::Match);
        b.insert(2, 2.0, 8, State::Match);
        assert!(b.mark_expanded(7));
        assert!(!b.mark_expanded(7) && !b.mark_expanded(99));
        assert_eq!(b.pop(), Some(8));
    }
}
