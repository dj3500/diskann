/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Round trip through the Rust in-memory index of `diskann-providers`: build a small index,
//! save it with `SaveWith`, import the saved files with `diskann_io::read_index`, and compare
//! every adjacency list, the starts and the frozen point with what the index holds in memory and
//! with what the upstream graph loader reads back.

use std::io::{Cursor, Read};

use diskann::{
    graph::{config, AdjacencyList, StartPointStrategy},
    provider::DefaultContext,
    utils::ONE,
};
use diskann_filter_lab::{
    diskann_io,
    graph::{AuxKind, Graph},
    vectors::DType,
};
use diskann_providers::{
    index::diskann_async,
    model::graph::provider::async_::{
        common::{FullPrecision, NoDeletes},
        inmem::{DefaultProviderParameters, SetStartPoints},
        SimpleNeighborProviderAsync,
    },
    storage::{AsyncIndexMetadata, SaveWith, StorageReadProvider, VirtualStorageProvider},
};
use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::distance::Metric;

#[test]
fn upstream_saved_index_imports_identically() {
    let (n, dim, degree, max_degree) = (1500, 12, 16, 20);
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let data = rowmajor::Owned::from_fn(n, dim, |_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 56) as u8
    });

    let config = config::Builder::new_with(
        degree,
        config::MaxDegree::new(max_degree),
        32,
        Metric::L2.into(),
        |b| {
            b.alpha(1.2);
        },
    )
    .build()
    .unwrap();
    let params = DefaultProviderParameters {
        max_points: n,
        frozen_points: ONE,
        metric: Metric::L2,
        dim,
        prefetch_lookahead: None,
        prefetch_cache_line_level: None,
        max_degree: max_degree as u32,
    };
    let index = diskann_async::new_index::<u8, _>(config, params, NoDeletes).unwrap();
    let medoid = StartPointStrategy::Medoid.compute(data.as_view()).unwrap();
    index.provider().set_start_points(medoid.rows()).unwrap();
    let storage = VirtualStorageProvider::new_memory();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    runtime.block_on(async {
        for (i, row) in data.rows().enumerate() {
            index
                .insert(&FullPrecision, &DefaultContext, &(i as u32), row)
                .await
                .unwrap();
        }
        index
            .save_with(&storage, &AsyncIndexMetadata::new("/index"))
            .await
            .unwrap();
    });

    let file = |path: &str| {
        let mut bytes = Vec::new();
        storage
            .open_reader(path)
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        Cursor::new(bytes)
    };
    let (g, layout) =
        diskann_io::read_index(file("/index"), Some((file("/index.data"), DType::U8))).unwrap();

    // n data points and one frozen start point; the file ends with one unused empty list.
    let frozen = n as u32;
    assert_eq!((g.n_data(), g.n_nodes(), g.n_aux()), (n, n + 1, 1));
    assert_eq!((layout.lists, layout.unused_lists), (n + 2, 1));
    assert_eq!(layout.header.max_degree as usize, max_degree);
    assert_eq!(g.r_max(), max_degree);
    assert_eq!(g.starts(), &index.provider().starting_points().unwrap()[..]);
    assert_eq!(g.starts(), &[frozen]);
    assert_eq!(g.aux_kind(frozen), Some(AuxKind::Frozen));
    assert_eq!(g.aux_vector::<u8>(frozen).unwrap(), medoid.row(0));
    assert!(g.n_edges() > n as u64);

    // Every list as the index holds it and as upstream's loader reads the file back.
    let loaded = SimpleNeighborProviderAsync::load_direct(&storage, "/index").unwrap();
    let mut list = AdjacencyList::new();
    for id in 0..=frozen {
        index
            .provider()
            .neighbors()
            .get_neighbors_sync(id as usize, &mut list)
            .unwrap();
        assert_eq!(g.neighbors(id), &*list, "node {id} in memory");
        loaded.get_neighbors_sync(id as usize, &mut list).unwrap();
        assert_eq!(g.neighbors(id), &*list, "node {id} loaded");
    }
    loaded.get_neighbors_sync(n + 1, &mut list).unwrap();
    assert!(list.is_empty());

    // The lab format keeps all of it.
    let mut bytes = Vec::new();
    g.write_to(&mut bytes).unwrap();
    assert_eq!(Graph::read_from(Cursor::new(bytes)).unwrap(), g);
}
