/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Candidate pools recorded by a build, for prune replay (`pools.flab`, magic
//! `FLABPOL\0`).
//!
//! For every node the file keeps everything the build pruned over: the *search pool* (the
//! nodes expanded by the node's insertion search, in pool order and truncated to `maxc`, i.e.
//! exactly the list handed to the pruning rule) followed by the *back-edge proposals* (every
//! node that later chose it as an out-neighbor, in arrival order: batch by batch, by id
//! within a batch), each with its squared distance. It also keeps the insertion order, the
//! batch schedule and the vectors of the auxiliary nodes, so a replay needs only the base
//! vectors (and labels for label-aware rules).
//!
//! Parameters: `n_data`, `n_nodes`, `n_entries`, `n_batches`, `slots` (row capacity of the
//! build), `maxc` (pool truncation), element type code (1 = `u8`, 2 = `f32`) `| dim << 8`,
//! length of the metadata. Sections: insertion order (`u32 × n_data`), batch boundaries into
//! it (`u64 × (n_batches + 1)`), row offsets (`u64 × (n_nodes + 1)`), search-pool length per
//! node (`u32 × n_nodes`), ids (`u32 × n_entries`), squared distances (`f32 × n_entries`),
//! auxiliary vectors (`(n_nodes − n_data) × dim` elements), metadata (JSON text: the build's
//! parameters).

use std::{
    io::{Read, Seek, Write},
    path::Path,
};

use anyhow::{ensure, Context, Result};

use crate::{
    container,
    vectors::{DType, Element},
};

/// File magic of `pools.flab`.
pub const MAGIC: &[u8; 8] = b"FLABPOL\0";
/// Version of the `pools.flab` format.
pub const VERSION: u32 = 1;

/// Candidate pools of every node of a build.
#[derive(Debug, Clone, PartialEq)]
pub struct Pools {
    pub(crate) n_data: usize,
    /// Insertion order of the data points.
    pub(crate) order: Vec<u32>,
    /// Batch boundaries: batch `b` inserts `order[batches[b]..batches[b + 1]]`.
    pub(crate) batches: Vec<u64>,
    /// Row offsets per node into `ids` / `dists`.
    pub(crate) offsets: Vec<u64>,
    /// Length of the search pool at the start of each row.
    pub(crate) n_search: Vec<u32>,
    pub(crate) ids: Vec<u32>,
    pub(crate) dists: Vec<f32>,
    /// Row capacity of the build (maximum degree with slack).
    pub(crate) slots: usize,
    /// Pool truncation of the build.
    pub(crate) maxc: usize,
    pub(crate) dtype: DType,
    pub(crate) dim: usize,
    /// Vectors of the auxiliary nodes (`n_nodes − n_data` rows), as bytes.
    pub(crate) aux_bytes: Vec<u8>,
    /// The build's parameters (JSON text).
    pub(crate) meta: String,
}

fn dtype_code(d: DType) -> u64 {
    match d {
        DType::U8 => 1,
        DType::F32 => 2,
    }
}

impl Pools {
    pub fn n_data(&self) -> usize {
        self.n_data
    }

    pub fn n_nodes(&self) -> usize {
        self.n_search.len()
    }

    pub fn n_entries(&self) -> usize {
        self.ids.len()
    }

    pub fn n_batches(&self) -> usize {
        self.batches.len() - 1
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    pub fn maxc(&self) -> usize {
        self.maxc
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Parameters of the build that recorded the pools (JSON text).
    pub fn meta(&self) -> &str {
        &self.meta
    }

    pub fn order(&self) -> &[u32] {
        &self.order
    }

    /// The points inserted by batch `b`.
    pub fn batch(&self, b: usize) -> &[u32] {
        &self.order[self.batches[b] as usize..self.batches[b + 1] as usize]
    }

    fn range(&self, v: u32) -> (usize, usize, usize) {
        let (a, b) = (
            self.offsets[v as usize] as usize,
            self.offsets[v as usize + 1] as usize,
        );
        (a, a + self.n_search[v as usize] as usize, b)
    }

    /// Search pool of node `v` (pool order): ids and distances.
    pub fn search(&self, v: u32) -> (&[u32], &[f32]) {
        let (a, s, _) = self.range(v);
        (&self.ids[a..s], &self.dists[a..s])
    }

    /// Back-edge proposals to node `v` (arrival order): ids and distances.
    pub fn proposals(&self, v: u32) -> (&[u32], &[f32]) {
        let (_, s, b) = self.range(v);
        (&self.ids[s..b], &self.dists[s..b])
    }

    /// Search pool and proposals of node `v`.
    pub fn row(&self, v: u32) -> (&[u32], &[f32]) {
        let (a, _, b) = self.range(v);
        (&self.ids[a..b], &self.dists[a..b])
    }

    /// Vectors of the auxiliary nodes, row-major.
    pub fn aux_vectors<T: Element>(&self) -> Result<Vec<T>> {
        ensure!(
            T::DTYPE == self.dtype,
            "the pools were recorded on {} vectors, not {}",
            self.dtype.name(),
            T::NAME
        );
        let mut out = vec![T::zeroed(); self.aux_bytes.len() / std::mem::size_of::<T>()];
        bytemuck::cast_slice_mut::<T, u8>(&mut out).copy_from_slice(&self.aux_bytes);
        Ok(out)
    }

    /// Size of the file in bytes (header and section padding included).
    pub fn file_bytes(&self) -> u64 {
        let sections = [
            4 * self.order.len(),
            8 * self.batches.len(),
            8 * self.offsets.len(),
            4 * self.n_search.len(),
            4 * self.ids.len(),
            4 * self.dists.len(),
            self.aux_bytes.len(),
            self.meta.len(),
        ];
        let header = (16 + 8 * container::N_PARAMS + 16 * sections.len()).div_ceil(64) * 64;
        header as u64
            + sections
                .iter()
                .map(|&s| (s as u64).div_ceil(64) * 64)
                .sum::<u64>()
    }

    /// Writes `pools.flab`.
    pub fn write(&self, path: &Path) -> Result<()> {
        self.write_to(container::create(path)?)
            .with_context(|| format!("writing {}", path.display()))
    }

    /// Writes the `pools.flab` format to any writer.
    pub fn write_to<W: Write>(&self, w: W) -> Result<()> {
        let params = [
            self.n_data as u64,
            self.n_nodes() as u64,
            self.n_entries() as u64,
            self.n_batches() as u64,
            self.slots as u64,
            self.maxc as u64,
            dtype_code(self.dtype) | (self.dim as u64) << 8,
            self.meta.len() as u64,
        ];
        container::write(
            w,
            MAGIC,
            VERSION,
            &params,
            &[
                bytemuck::cast_slice(&self.order),
                bytemuck::cast_slice(&self.batches),
                bytemuck::cast_slice(&self.offsets),
                bytemuck::cast_slice(&self.n_search),
                bytemuck::cast_slice(&self.ids),
                bytemuck::cast_slice(&self.dists),
                &self.aux_bytes,
                self.meta.as_bytes(),
            ],
        )
    }

    /// Reads and validates `pools.flab`.
    pub fn read(path: &Path) -> Result<Self> {
        container::Reader::open(path, MAGIC)
            .and_then(Self::load)
            .with_context(|| format!("loading candidate pools from {}", path.display()))
    }

    /// Reads and validates the `pools.flab` format from any reader.
    pub fn read_from<R: Read + Seek>(r: R) -> Result<Self> {
        Self::load(container::Reader::new(r, MAGIC)?)
    }

    fn load<R: Read + Seek>(mut r: container::Reader<R>) -> Result<Self> {
        ensure!(
            r.version() == VERSION,
            "unsupported version {}",
            r.version()
        );
        let [n_data, n_nodes, n_entries, n_batches, slots, maxc, element, meta_len] = *r.params();
        let (n_data, n_nodes, n_entries) = (n_data as usize, n_nodes as usize, n_entries as usize);
        let dim = element >> 8;
        let dtype = match element & 0xff {
            1 => DType::U8,
            2 => DType::F32,
            other => anyhow::bail!("unknown element type code {other}"),
        };
        ensure!(
            n_data <= n_nodes,
            "{n_data} data points but {n_nodes} nodes"
        );
        let aux_len = (n_nodes - n_data) * dim as usize * dtype.size();
        let pools = Self {
            n_data,
            order: r.section(0, n_data)?,
            batches: r.section(1, n_batches as usize + 1)?,
            offsets: r.section(2, n_nodes + 1)?,
            n_search: r.section(3, n_nodes)?,
            ids: r.section(4, n_entries)?,
            dists: r.section(5, n_entries)?,
            slots: slots as usize,
            maxc: maxc as usize,
            dtype,
            dim: dim as usize,
            aux_bytes: r.section(6, aux_len)?,
            meta: String::from_utf8(r.section(7, meta_len as usize)?)
                .context("metadata is not UTF-8")?,
        };
        pools.validate()?;
        Ok(pools)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let n = self.n_nodes();
        ensure!(
            self.offsets.first() == Some(&0)
                && self.offsets.last() == Some(&(self.ids.len() as u64))
                && self.offsets.windows(2).all(|w| w[0] <= w[1]),
            "row offsets are not monotone over the entries"
        );
        for v in 0..n {
            ensure!(
                u64::from(self.n_search[v]) <= self.offsets[v + 1] - self.offsets[v],
                "node {v}: search pool longer than its row"
            );
        }
        ensure!(
            self.ids.iter().all(|&i| (i as usize) < n),
            "a pool entry is out of range ({n} nodes)"
        );
        ensure!(
            self.batches.first() == Some(&0)
                && self.batches.last() == Some(&(self.order.len() as u64))
                && self.batches.windows(2).all(|w| w[0] <= w[1]),
            "batch boundaries do not cover the insertion order"
        );
        let mut seen = vec![false; self.n_data];
        for &p in &self.order {
            ensure!(
                (p as usize) < self.n_data && !std::mem::replace(&mut seen[p as usize], true),
                "the insertion order is not a permutation of the data points"
            );
        }
        ensure!(self.dim > 0, "dimension 0");
        Ok(())
    }
}
