/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Compares a saved DiskANN in-memory index as read by the upstream graph loader
//! (`SimpleNeighborProviderAsync::load_direct` of `diskann-providers`) with the lab's import of
//! the same files, and optionally with a `graph.flab` imported earlier.
//!
//! ```sh
//! cargo run --release -p diskann-filter-lab --example upstream_check -- \
//!     <prefix> [--dtype u8|f32] [--flab graph.flab]
//! ```
//!
//! `<prefix>` is the graph file; its vectors are read from `<prefix>.data`.

use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use diskann::graph::AdjacencyList;
use diskann_filter_lab::{diskann_io, graph::Graph, vectors::DType};
use diskann_providers::{
    model::graph::provider::async_::SimpleNeighborProviderAsync, storage::FileStorageProvider,
};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (mut prefix, mut dtype, mut flab) = (None, DType::U8, None);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dtype" => {
                dtype = match args.next().as_deref() {
                    Some("u8") => DType::U8,
                    Some("f32") => DType::F32,
                    other => bail!("unknown --dtype {other:?}"),
                }
            }
            "--flab" => flab = Some(PathBuf::from(args.next().context("--flab needs a path")?)),
            _ if prefix.is_none() => prefix = Some(arg),
            _ => bail!("unexpected argument {arg:?}"),
        }
    }
    let prefix =
        prefix.context("usage: upstream_check <prefix> [--dtype u8|f32] [--flab graph.flab]")?;
    let data = format!("{prefix}.data");
    let (g, layout) =
        diskann_io::read_index_files(Path::new(&prefix), Some((Path::new(&data), dtype)))?;
    let upstream = SimpleNeighborProviderAsync::load_direct(&FileStorageProvider, &prefix)?;

    let mut list = AdjacencyList::new();
    let mut differing = 0;
    for id in 0..layout.lists {
        upstream.get_neighbors_sync(id, &mut list)?;
        let ours = if id < g.n_nodes() {
            g.neighbors(id as u32)
        } else {
            &[]
        };
        if ours != &*list {
            differing += 1;
            if differing <= 5 {
                println!("node {id}: lab {ours:?}, upstream {:?}", &*list);
            }
        }
    }
    println!(
        "{prefix}: {} adjacency lists ({} unused), {} nodes ({} frozen), {} edges; lists differing from the upstream loader: {differing}",
        layout.lists,
        layout.unused_lists,
        g.n_nodes(),
        g.n_aux(),
        g.n_edges()
    );
    if let Some(path) = flab {
        let same = Graph::read(&path)? == g;
        println!("{} equals the import: {same}", path.display());
        ensure!(same, "{} differs from the import", path.display());
    }
    ensure!(differing == 0, "{differing} adjacency lists differ");
    Ok(())
}
