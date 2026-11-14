/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Greedy search for insertion candidates (upstream DiskANN's `search_internal` with beam
//! width 1, as used by `insert` and `multi_insert`).

use super::{adjacency::Adjacency, prune::Candidate, Points};
use crate::vectors::Element;

#[derive(Debug, Clone, Copy)]
struct Slot {
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
    /// Nodes expanded by the last search, in expansion order, with their distances.
    pub(crate) expanded: Vec<Candidate>,
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
        let capacity = l + 1;
        self.beam.clear();
        self.cursor = 0;
        self.expanded.clear();
        self.touched.clear();
        self.mark(start);
        self.insert(capacity, start, T::squared_l2(query, points.get(start)));
        let mut n_dist = 1u64;
        while self.cursor < self.beam.len() {
            let node = self.beam[self.cursor];
            self.beam[self.cursor].expanded = true;
            self.expanded.push(Candidate::new(node.id, node.dist));
            self.cursor += 1;
            while self.cursor < self.beam.len() && self.beam[self.cursor].expanded {
                self.cursor += 1;
            }
            self.fresh.clear();
            for &v in adjacency.neighbors(node.id) {
                if self.mark(v) {
                    self.fresh.push(v);
                    points.prefetch(v);
                }
            }
            n_dist += self.fresh.len() as u64;
            for i in 0..self.fresh.len() {
                let v = self.fresh[i];
                self.insert(capacity, v, T::squared_l2(query, points.get(v)));
            }
        }
        for &v in &self.touched {
            self.visited[v as usize / 64] = 0;
        }
        n_dist
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
    fn insert(&mut self, capacity: usize, id: u32, dist: f32) {
        let len = self.beam.len();
        if len == capacity && self.beam[len - 1].dist < dist {
            return;
        }
        let pos = self.beam.partition_point(|s| s.dist < dist);
        if len == capacity {
            self.beam.pop();
        }
        self.beam.insert(
            pos,
            Slot {
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
