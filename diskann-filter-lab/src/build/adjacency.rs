/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The mutable adjacency of a build: fixed-stride rows of `slots` neighbor ids.

use anyhow::{ensure, Result};

use crate::graph::{AuxNodes, Graph, NO_NEIGHBOR};

#[derive(Debug, Clone)]
pub(crate) struct Adjacency {
    slots: usize,
    degrees: Vec<u32>,
    /// Row `v` is `ids[v * slots..(v + 1) * slots]`; slots after the degree hold
    /// [`NO_NEIGHBOR`], as in `graph.flab`.
    ids: Vec<u32>,
}

impl Adjacency {
    pub(crate) fn new(n_nodes: usize, slots: usize) -> Self {
        Self {
            slots,
            degrees: vec![0; n_nodes],
            ids: vec![NO_NEIGHBOR; n_nodes * slots],
        }
    }

    pub(crate) fn n_nodes(&self) -> usize {
        self.degrees.len()
    }

    #[inline]
    pub(crate) fn degree(&self, v: u32) -> usize {
        self.degrees[v as usize] as usize
    }

    #[inline]
    pub(crate) fn neighbors(&self, v: u32) -> &[u32] {
        let start = v as usize * self.slots;
        &self.ids[start..start + self.degrees[v as usize] as usize]
    }

    /// Replaces the row of `v`.
    pub(crate) fn set(&mut self, v: u32, list: impl ExactSizeIterator<Item = u32>) {
        let n = list.len();
        assert!(
            n <= self.slots,
            "node {v}: {n} neighbors exceed {} slots",
            self.slots
        );
        let start = v as usize * self.slots;
        let row = &mut self.ids[start..start + self.slots];
        for (slot, id) in row.iter_mut().zip(list) {
            *slot = id;
        }
        row[n..].fill(NO_NEIGHBOR);
        self.degrees[v as usize] = n as u32;
    }

    /// Appends `extra` to the row of `v`.
    pub(crate) fn extend(&mut self, v: u32, extra: &[u32]) {
        let d = self.degrees[v as usize] as usize;
        assert!(d + extra.len() <= self.slots, "node {v}: row overflow");
        let start = v as usize * self.slots + d;
        self.ids[start..start + extra.len()].copy_from_slice(extra);
        self.degrees[v as usize] = (d + extra.len()) as u32;
    }

    #[cfg(test)]
    pub(crate) fn n_edges(&self) -> u64 {
        self.degrees.iter().map(|&d| u64::from(d)).sum()
    }

    /// The graph over `n_data` data points and the auxiliary nodes `aux` (rows of `slots`
    /// slots).
    pub(crate) fn into_graph(
        self,
        n_data: usize,
        aux: AuxNodes,
        starts: Vec<u32>,
    ) -> Result<Graph> {
        ensure!(
            n_data + aux.len() == self.n_nodes(),
            "{} nodes but {n_data} data points and {} auxiliary nodes",
            self.n_nodes(),
            aux.len()
        );
        let mut graph = Graph::new(n_data, self.slots, aux)?;
        {
            let (degrees, rows) = graph.raw_mut();
            degrees.copy_from_slice(&self.degrees);
            rows.copy_from_slice(&self.ids);
        }
        graph.validate()?;
        graph.set_starts(starts)?;
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_keep_padding() {
        let mut a = Adjacency::new(3, 4);
        a.set(1, [2u32, 0].into_iter());
        a.extend(1, &[2]);
        assert_eq!(a.neighbors(1), &[2, 0, 2]);
        a.set(1, [0u32].into_iter());
        assert_eq!(a.neighbors(1), &[0]);
        assert_eq!(a.n_edges(), 1);
        let g = a.into_graph(3, AuxNodes::untyped(), vec![0]).unwrap();
        assert_eq!(g.neighbors(1), &[0]);
        assert_eq!(g.row(1), &[0, NO_NEIGHBOR, NO_NEIGHBOR, NO_NEIGHBOR]);
    }
}
