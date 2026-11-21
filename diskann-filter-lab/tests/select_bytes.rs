/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The selector's index bytes (`select::IndexBytes`, `B` of the elasticity rule) on a tiny
//! dataset written to cargo's target directory: vectors `n·d·elem`, the graph as adjacency
//! lists `4·(nodes + edges)` in both file formats, router vectors, the label store file and the
//! static bitsets at `⌈n/64⌉`.

use std::path::Path;

use diskann_filter_lab::{
    bitsets::LabelBitsets,
    graph::{AuxKind, AuxNodes, Graph},
    labels::LabelStoreBuilder,
    select::IndexBytes,
    vectors::Matrix,
};

const N: usize = 300;
const DIM: usize = 6;

#[test]
fn index_bytes_count_the_reports_structures_in_both_graph_formats() {
    let dir =
        Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("flab-ixb-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let base = Matrix::<u8>::from_slice(
        &(0..N * DIM).map(|i| (i * 7 % 61) as u8).collect::<Vec<_>>(),
        N,
        DIM,
    );
    base.write_bin(&dir.join("base.u8bin")).unwrap();
    let mut b = LabelStoreBuilder::new();
    for i in 0..N {
        let g = format!("g={}", i % 5);
        let r = format!("r={}", i % 60);
        b.push_point([("g", g.as_str()), ("r", r.as_str())])
            .unwrap();
    }
    let labels = b.finish().unwrap();
    labels.write(&dir.join("labels.flab")).unwrap();
    let labels_file = std::fs::metadata(dir.join("labels.flab")).unwrap().len();
    // θ = ⌈300/64⌉ = 5: the five g labels (60 points each) and every r label (5 points).
    let bitsets = LabelBitsets::build(&labels, N.div_ceil(64) as u64).bytes();
    assert_eq!(bitsets, 65 * 5 * 8);

    // graph.flab with a frozen node and two routers: their vectors count, the frozen one's not.
    let kinds = vec![AuxKind::Frozen, AuxKind::Router, AuxKind::Router];
    let aux = AuxNodes::new(kinds, &Matrix::<u8>::zeros(3, DIM)).unwrap();
    let mut g = Graph::new(N, 4, aux).unwrap();
    let mut edges = 0u64;
    for v in 0..N + 3 {
        let nb: Vec<u32> = (1..=(v % 4) as u32)
            .map(|d| ((v as u32) + d) % N as u32)
            .collect();
        edges += nb.len() as u64;
        g.set_neighbors(v as u32, &nb).unwrap();
    }
    g.set_starts(vec![N as u32]).unwrap();
    let gdir = dir.join("graph");
    std::fs::create_dir_all(&gdir).unwrap();
    g.write(&gdir.join("graph.flab")).unwrap();
    let want = IndexBytes {
        vectors: (N * DIM) as u64,
        graph: 4 * ((N + 3) as u64 + edges),
        router_vectors: 2 * DIM as u64,
        labels: labels_file,
        bitsets,
    };
    for path in [gdir.clone(), gdir.join("graph.flab")] {
        assert_eq!(IndexBytes::of(&dir, &path, &labels, 1).unwrap(), want);
    }

    // The same adjacency as a DiskANN in-memory index: header (file size, degree bound, start,
    // frozen points), then per node its degree and neighbours; no router vectors.
    let (mut body, mut d_edges) = (Vec::new(), 0u64);
    for v in 0..N + 1 {
        let nb: Vec<u32> = (1..=(v % 4) as u32)
            .map(|d| ((v as u32) + d) % N as u32)
            .collect();
        d_edges += nb.len() as u64;
        body.extend_from_slice(&(nb.len() as u32).to_le_bytes());
        for x in nb {
            body.extend_from_slice(&x.to_le_bytes());
        }
    }
    let mut file = Vec::new();
    file.extend_from_slice(&(24 + body.len() as u64).to_le_bytes());
    file.extend_from_slice(&3u32.to_le_bytes());
    file.extend_from_slice(&(N as u32).to_le_bytes());
    file.extend_from_slice(&1u64.to_le_bytes());
    file.extend_from_slice(&body);
    std::fs::write(dir.join("index"), &file).unwrap();
    let got = IndexBytes::of(&dir, &dir.join("index"), &labels, 1).unwrap();
    assert_eq!(got.graph, 4 * ((N + 1) as u64 + d_edges));
    assert_eq!(got.router_vectors, 0);
    assert_eq!(
        (got.vectors, got.labels, got.bitsets),
        (want.vectors, want.labels, want.bitsets)
    );
    std::fs::remove_dir_all(&dir).ok();
}
