/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Loading in-memory indexes saved by upstream (`DiskANNIndex::save_with`, e.g. the
//! `graph-index-build` benchmark job with `save_path`).
//!
//! A saved index with prefix `P` consists of the graph file `P` and the vectors `P.data`
//! (DiskANN `.bin` layout, data points followed by the frozen start points). The graph file
//! starts with a 24-byte header: `u64` file size, `u32` maximum observed degree, `u32` start
//! point id, `u64` number of frozen points. Data points have ids `0..n_data`; the frozen points
//! have ids `n_data..n_data + n_frozen`.
//!
//! The index is configured exactly like the benchmark's `Load` source: pruned degree = maximum
//! observed degree, `MaxDegree::same()`, `l_build` 1, squared L2.

use std::{fs::File, io::Read, num::NonZeroUsize, path::Path};

use anyhow::{ensure, Context, Result};
use diskann::{
    graph::{config, DiskANNIndex},
    utils::VectorRepr,
};
use diskann_providers::{
    model::{configuration::IndexConfiguration, graph::provider::async_::inmem},
    storage::{FileStorageProvider, LoadWith},
    utils::load_metadata_from_file,
};
use diskann_vector::distance::Metric;

/// The in-memory full-precision provider used by upstream's benchmark for saved indexes.
pub type Provider<T> = inmem::FullPrecisionProvider<T>;

/// Header of an upstream graph file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphHeader {
    pub file_size: u64,
    pub max_observed_degree: u32,
    pub start: u32,
    pub n_frozen: u64,
}

impl GraphHeader {
    pub fn read(path: &Path) -> Result<Self> {
        let mut bytes = [0u8; 24];
        File::open(path)
            .and_then(|mut f| f.read_exact(&mut bytes))
            .with_context(|| format!("reading the graph header of {}", path.display()))?;
        let u64_at = |i: usize| u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
        let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        Ok(Self {
            file_size: u64_at(0),
            max_observed_degree: u32_at(8),
            start: u32_at(12),
            n_frozen: u64_at(16),
        })
    }
}

/// A loaded index with its node-space layout.
pub struct LoadedIndex<T: VectorRepr> {
    pub index: DiskANNIndex<Provider<T>>,
    pub header: GraphHeader,
    /// Number of data points (ids `0..n_data`).
    pub n_data: usize,
    /// Frozen start points (ids `>= n_data`).
    pub start_ids: Vec<u32>,
    pub dim: usize,
}

/// Loads the index saved with prefix `prefix` on a current-thread runtime.
pub fn load<T: VectorRepr>(prefix: &str) -> Result<LoadedIndex<T>> {
    let header = GraphHeader::read(Path::new(prefix))?;
    let n_frozen = NonZeroUsize::new(header.n_frozen as usize)
        .context("the saved graph has no frozen start point")?;
    let storage = FileStorageProvider;
    let data_path = format!("{prefix}.data");
    let metadata = load_metadata_from_file(&storage, &data_path)
        .with_context(|| format!("reading the header of {data_path}"))?;
    let config = config::Builder::new(
        header.max_observed_degree as usize,
        config::MaxDegree::same(),
        1,
        Metric::L2.into(),
    )
    .build()?;
    let index_config = IndexConfiguration::new(
        Metric::L2,
        metadata.ndims(),
        metadata.npoints(),
        n_frozen,
        1,
        config,
    );
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    let index = runtime
        .block_on(DiskANNIndex::<Provider<T>>::load_with(
            &storage,
            &(prefix, index_config),
        ))
        .with_context(|| format!("loading the index {prefix}"))?;
    let start_ids = index.provider().starting_points()?;
    let n_data = index.provider().total_points() - start_ids.len();
    ensure!(
        start_ids.iter().all(|&s| s as usize >= n_data) && start_ids.contains(&header.start),
        "unexpected start points {start_ids:?} for {n_data} data points"
    );
    Ok(LoadedIndex {
        index,
        header,
        n_data,
        start_ids,
        dim: metadata.ndims(),
    })
}
