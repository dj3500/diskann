/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Greedy search for insertion candidates (upstream DiskANN's `search_internal` with beam
//! width 1, as used by `insert` and `multi_insert`), with an optional [`Ranking`]: which nodes
//! may enter the list (label-filtered insertion) and the key that orders it (penalized
//! insertion).

use super::{adjacency::Adjacency, prune::Candidate, Points};
use crate::vectors::Element;

/// How an insertion search admits and orders nodes. The default methods give the label-blind
/// search (every node admitted, ordered by squared distance).
pub(crate) trait Ranking {
    /// Whether node `v` may enter the list; checked before its distance is computed. A
    /// rejected node is marked visited and never evaluated again.
    #[inline(always)]
    fn admit(&mut self, _v: u32) -> bool {
        true
    }

    /// Called for every admitted unvisited node before the distances of an expansion are
    /// computed (e.g. to prefetch its label row).
    #[inline(always)]
    fn prepare(&mut self, _v: u32) {}

    /// The list key of node `v` at squared distance `d`; never below `d` (a node whose
    /// distance alone cannot enter the full list is rejected without computing its key).
    #[inline(always)]
    fn key(&mut self, _v: u32, d: f32) -> f32 {
        d
    }
}

/// The label-blind ranking.
pub(crate) struct Blind;

impl Ranking for Blind {}

#[derive(Debug, Clone, Copy)]
struct Slot {
    key: f32,
    dist: f32,
    id: u32,
    expanded: bool,
}

/// Per-thread search state: a visited bitset over all nodes (small enough to stay in cache;
/// cleared through the list of visited nodes), the candidate list and the expanded nodes of
/// the last search.
#[derive(Debug)]
pub(crate) struct Searcher {
    visited: Vec<u64>,
    touched: Vec<u32>,
    beam: Vec<Slot>,
    cursor: usize,
    fresh: Vec<u32>,
    /// Nodes expanded by the last search, in expansion order, with their squared distances.
    pub(crate) expanded: Vec<Candidate>,
    /// The list keys of [`Self::expanded`] (equal to the distances for the blind ranking).
    pub(crate) expanded_keys: Vec<f32>,
    /// Admission tests of the last search.
    pub(crate) n_admit: u64,
}

impl Searcher {
    pub(crate) fn new(n_nodes: usize) -> Self {
        Self {
            visited: vec![0; n_nodes.div_ceil(64)],
            touched: Vec::new(),
            beam: Vec::new(),
            cursor: 0,
            fresh: Vec::new(),
            expanded: Vec::new(),
            expanded_keys: Vec::new(),
            n_admit: 0,
        }
    }

    /// Greedy best-first search for `query` from `start` with list size `l`. As upstream:
    /// the candidate list holds `l + 1` entries (list size plus the start point), a new
    /// entry goes before entries at an equal distance, and the search stops when every
    /// entry of the list is expanded. Returns the number of distance computations; the
    /// expanded nodes are in [`Self::expanded`].
    pub(crate) fn search<T: Element>(
        &mut self,
        adjacency: &Adjacency,
        points: &Points<'_, T>,
        query: &[T],
        start: u32,
        l: usize,
    ) -> u64 {
        self.search_ranked(adjacency, points, query, start, l, &mut Blind)
    }

    /// [`Self::search`] with a [`Ranking`]: only admitted nodes (the start included) enter the
    /// list, which is ordered by the ranking's key (ties: a new entry goes before entries
    /// with an equal key). Returns the number of distance computations.
    pub(crate) fn search_ranked<T: Element, R: Ranking>(
        &mut self,
        adjacency: &Adjacency,
        points: &Points<'_, T>,
        query: &[T],
        start: u32,
        l: usize,
        ranking: &mut R,
    ) -> u64 {
        let capacity = l + 1;
        self.beam.clear();
        self.cursor = 0;
        self.expanded.clear();
        self.expanded_keys.clear();
        self.touched.clear();
        self.n_admit = 1;
        self.mark(start);
        if !ranking.admit(start) {
            self.reset_visited();
            return 0;
        }
        let d0 = T::squared_l2(query, points.get(start));
        self.insert(capacity, start, ranking.key(start, d0), d0);
        let mut n_dist = 1u64;
        while self.cursor < self.beam.len() {
            let node = self.beam[self.cursor];
            self.beam[self.cursor].expanded = true;
            self.expanded.push(Candidate::new(node.id, node.dist));
            self.expanded_keys.push(node.key);
            self.cursor += 1;
            while self.cursor < self.beam.len() && self.beam[self.cursor].expanded {
                self.cursor += 1;
            }
            self.fresh.clear();
            for &v in adjacency.neighbors(node.id) {
                if self.mark(v) {
                    self.n_admit += 1;
                    if ranking.admit(v) {
                        self.fresh.push(v);
                        points.prefetch(v);
                        ranking.prepare(v);
                    }
                }
            }
            n_dist += self.fresh.len() as u64;
            for i in 0..self.fresh.len() {
                let v = self.fresh[i];
                let d = T::squared_l2(query, points.get(v));
                // keys are never below the distance: a node the full list rejects by its
                // distance is rejected by its key, so its key is not computed
                let len = self.beam.len();
                if len == capacity && self.beam[len - 1].key < d {
                    continue;
                }
                self.insert(capacity, v, ranking.key(v, d), d);
            }
        }
        self.reset_visited();
        n_dist
    }

    fn reset_visited(&mut self) {
        for &v in &self.touched {
            self.visited[v as usize / 64] = 0;
        }
    }

    /// Marks `v` visited; whether it was not visited before.
    #[inline]
    fn mark(&mut self, v: u32) -> bool {
        let (word, bit) = (v as usize / 64, 1u64 << (v % 64));
        let w = &mut self.visited[word];
        if *w & bit != 0 {
            return false;
        }
        if *w == 0 {
            self.touched.push(v);
        }
        *w |= bit;
        true
    }

    #[inline]
    fn insert(&mut self, capacity: usize, id: u32, key: f32, dist: f32) {
        let len = self.beam.len();
        if len == capacity && self.beam[len - 1].key < key {
            return;
        }
        let pos = self.beam.partition_point(|s| s.key < key);
        if len == capacity {
            self.beam.pop();
        }
        self.beam.insert(
            pos,
            Slot {
                key,
                dist,
                id,
                expanded: false,
            },
        );
        if pos < self.cursor {
            self.cursor = pos;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vectors::Matrix;

    #[test]
    fn walks_a_path_graph_to_the_query() {
        // Points on a line, edges i <-> i + 1; the search from 0 must reach the far end.
        let n = 50;
        let xs: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let m = Matrix::from_slice(&xs, n, 1);
        let points = Points::new(&m, &[]);
        let mut adj = Adjacency::new(n, 2);
        for i in 0..n as u32 {
            let mut row = Vec::new();
            if i > 0 {
                row.push(i - 1);
            }
            if (i as usize) < n - 1 {
                row.push(i + 1);
            }
            adj.set(i, row.into_iter());
        }
        let mut s = Searcher::new(n);
        let n_dist = s.search(&adj, &points, &[47.2], 0, 3);
        let mut best: Vec<u32> = s.expanded.iter().map(|c| c.id).collect();
        best.sort_unstable_by(|a, b| {
            (xs[*a as usize] - 47.2)
                .abs()
                .total_cmp(&(xs[*b as usize] - 47.2).abs())
        });
        assert_eq!(best[0], 47);
        assert_eq!(n_dist as usize, n);
        assert_eq!(s.expanded[0].id, 0);
    }
}
