/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Readers for saved DiskANN in-memory indices: the graph file `<prefix>` and the vector file
//! `<prefix>.data`, as written by the C++ library (`Index::save`) and by the Rust
//! `diskann-providers` crate (`SaveWith<AsyncIndexMetadata>` of an in-memory index). Both use
//! the same layout.
//!
//! **Graph file** (little-endian): a 24-byte header (`u64` file size, `u32` maximum degree,
//! `u32` start id, `u64` number of frozen points `F`), then for every node in id order a `u32`
//! degree `k` followed by `k` `u32` neighbor ids. Frozen points are the last `F` nodes. The C++
//! library records the largest observed degree, the Rust crate its configured maximum degree.
//!
//! **Vector file**: the `.bin` layout (`u32` rows, `u32` dimension, row-major elements) with
//! one row per node; the vectors of the frozen points are the last `F` rows.
//!
//! The Rust in-memory provider allocates one adjacency list more than it has nodes
//! (`DefaultProvider::new_empty` sizes its neighbor store for `max_points + F` nodes plus one
//! start point), so its graph files end with one empty list beyond the `N + F` rows of the
//! vector file. That list belongs to no node.
//!
//! [`read_index`] loads both files into a [`Graph`] without intermediate copies: the node count
//! is the number of rows of the vector file (or of adjacency lists without it), the first
//! `n_nodes - F` nodes are the data points, the frozen points become auxiliary nodes of kind
//! [`AuxKind::Frozen`], `r_max` is the header's maximum degree, and the starts are the header's
//! start id followed by the frozen points (the start set of both libraries).

use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use anyhow::{bail, ensure, Context, Result};

use crate::{
    container,
    graph::{self, AuxKind, AuxNodes, Graph},
    vectors::{DType, Element, Matrix},
};

/// Size of the graph file header in bytes.
pub const HEADER_BYTES: u64 = 24;

/// Layout of a DiskANN graph file found by [`read_index`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub header: Header,
    /// Number of adjacency lists in the file.
    pub lists: usize,
    /// Trailing empty lists beyond the nodes (1 for files of the Rust in-memory provider).
    pub unused_lists: usize,
}

/// Header of a DiskANN graph file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Size of the whole file in bytes.
    pub file_size: u64,
    pub max_degree: u32,
    pub start: u32,
    pub n_frozen: u64,
}

impl Header {
    pub fn read<R: Read>(r: &mut R) -> Result<Self> {
        let mut b = [0u8; HEADER_BYTES as usize];
        r.read_exact(&mut b).context("truncated graph header")?;
        let u64_at = |at: usize| u64::from_le_bytes(b[at..at + 8].try_into().unwrap_or_default());
        let u32_at = |at: usize| u32::from_le_bytes(b[at..at + 4].try_into().unwrap_or_default());
        Ok(Self {
            file_size: u64_at(0),
            max_degree: u32_at(8),
            start: u32_at(12),
            n_frozen: u64_at(16),
        })
    }

    /// The header in the file layout.
    pub fn to_bytes(&self) -> [u8; HEADER_BYTES as usize] {
        let mut b = [0u8; HEADER_BYTES as usize];
        b[..8].copy_from_slice(&self.file_size.to_le_bytes());
        b[8..12].copy_from_slice(&self.max_degree.to_le_bytes());
        b[12..16].copy_from_slice(&self.start.to_le_bytes());
        b[16..].copy_from_slice(&self.n_frozen.to_le_bytes());
        b
    }
}

fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

/// Reads the header of a `.bin` vector file: (rows, dimension).
pub fn read_bin_header<R: Read>(r: &mut R) -> Result<(usize, usize)> {
    let rows = read_u32(r).context("truncated vector file header")?;
    let dim = read_u32(r).context("truncated vector file header")?;
    Ok((rows as usize, dim as usize))
}

/// Walks the adjacency lists without storing them: returns their number after checking every
/// degree against the header and the list boundaries against the file size.
fn count_lists<R: Read + Seek>(r: &mut BufReader<R>, h: &Header) -> Result<usize> {
    let mut pos = HEADER_BYTES;
    let mut n = 0usize;
    while pos < h.file_size {
        ensure!(
            h.file_size - pos >= 4,
            "truncated degree of node {n} at byte {pos}"
        );
        let k = read_u32(r)?;
        ensure!(
            k <= h.max_degree,
            "node {n}: degree {k} exceeds the header's maximum degree {}",
            h.max_degree
        );
        let bytes = 4 * u64::from(k);
        ensure!(
            h.file_size - pos - 4 >= bytes,
            "node {n}: truncated adjacency list at byte {pos}"
        );
        r.seek_relative(bytes as i64)?;
        pos += 4 + bytes;
        n += 1;
    }
    Ok(n)
}

fn read_rows<T: Element, D: Read>(d: &mut D, rows: usize, dim: usize) -> Result<Matrix<T>> {
    let mut m = Matrix::<T>::zeros(rows, dim);
    d.read_exact(bytemuck::cast_slice_mut(m.as_mut_slice()))?;
    Ok(m)
}

/// Reads the vectors of the last `n_frozen` of the `rows` rows of a vector file whose header
/// has been read.
fn read_frozen<D: Read + Seek>(
    d: &mut D,
    dtype: DType,
    (rows, dim): (usize, usize),
    n_frozen: usize,
) -> Result<AuxNodes> {
    let row_bytes = (dim * dtype.size()) as u64;
    d.seek(SeekFrom::Start(8 + (rows - n_frozen) as u64 * row_bytes))?;
    let kinds = vec![AuxKind::Frozen; n_frozen];
    match dtype {
        DType::U8 => AuxNodes::new(kinds, &read_rows::<u8, _>(d, n_frozen, dim)?),
        DType::F32 => AuxNodes::new(kinds, &read_rows::<f32, _>(d, n_frozen, dim)?),
    }
}

/// Reads a DiskANN graph file and, if given, its vector file with element type `dtype`. The
/// vector file is required when the graph has frozen points (for their vectors); otherwise it
/// records the element type and dimension and fixes the node count.
pub fn read_index<R: Read + Seek, D: Read + Seek>(
    graph: R,
    data: Option<(D, DType)>,
) -> Result<(Graph, Layout)> {
    let mut r = BufReader::with_capacity(1 << 22, graph);
    let file_len = r.seek(SeekFrom::End(0))?;
    r.seek(SeekFrom::Start(0))?;
    let header = Header::read(&mut r)?;
    ensure!(
        header.file_size == file_len,
        "the header says {} bytes but the file has {file_len}",
        header.file_size
    );
    let lists = count_lists(&mut r, &header)?;
    let n_frozen = usize::try_from(header.n_frozen).unwrap_or(usize::MAX);
    let (n_nodes, aux) = match data {
        Some((mut d, dtype)) => {
            let shape = bin_shape(&mut d, dtype).context("reading the vector file")?;
            let rows = shape.0;
            ensure!(
                lists == rows || lists == rows + 1,
                "the graph has {lists} adjacency lists but the vector file has {rows} rows"
            );
            ensure!(
                n_frozen <= rows,
                "{n_frozen} frozen points but {rows} vectors"
            );
            let aux =
                read_frozen(&mut d, dtype, shape, n_frozen).context("reading the vector file")?;
            (rows, aux)
        }
        None => {
            ensure!(
                n_frozen == 0,
                "the graph has {n_frozen} frozen points; their vectors need the vector file"
            );
            (lists, AuxNodes::untyped())
        }
    };
    ensure!(
        (header.start as usize) < n_nodes,
        "start {} out of range ({n_nodes} nodes)",
        header.start
    );

    let r_max = header.max_degree as usize;
    let mut g = Graph::new(n_nodes - n_frozen, r_max, aux)?;
    r.seek(SeekFrom::Start(HEADER_BYTES))?;
    let (degrees, adjacency) = g.raw_mut();
    for (id, degree) in degrees.iter_mut().enumerate() {
        let k = read_u32(&mut r)? as usize;
        ensure!(k <= r_max, "node {id}: the file changed while reading");
        let row = &mut adjacency[id * r_max..id * r_max + k];
        r.read_exact(bytemuck::cast_slice_mut(row))?;
        if let Some(&v) = row.iter().find(|&&v| v as usize >= n_nodes) {
            bail!("node {id}: neighbor {v} out of range ({n_nodes} nodes)");
        }
        *degree = k as u32;
    }
    if lists > n_nodes {
        ensure!(
            read_u32(&mut r)? == 0,
            "the adjacency list after the last node is not empty"
        );
    }
    let first_frozen = (n_nodes - n_frozen) as u32;
    let mut starts = vec![header.start];
    starts.extend((first_frozen..n_nodes as u32).filter(|&v| v != header.start));
    g.set_starts(starts)?;
    let layout = Layout {
        header,
        lists,
        unused_lists: lists - n_nodes,
    };
    Ok((g, layout))
}

/// [`read_index`] from files.
pub fn read_index_files(graph: &Path, data: Option<(&Path, DType)>) -> Result<(Graph, Layout)> {
    let open = |p: &Path| File::open(p).with_context(|| format!("opening {}", p.display()));
    let data = match data {
        Some((p, dtype)) => Some((BufReader::new(open(p)?), dtype)),
        None => None,
    };
    read_index(open(graph)?, data)
        .with_context(|| format!("reading the DiskANN graph {}", graph.display()))
}

/// The vector file of the DiskANN index with graph file `graph`: `<graph>.data`.
pub fn data_path(graph: &Path) -> PathBuf {
    let mut p = graph.as_os_str().to_owned();
    p.push(".data");
    PathBuf::from(p)
}

/// Shape of a `.bin` vector file from its header and size: (rows, dimension, element type, `u8`
/// or `f32`).
pub fn vector_file_shape(path: &Path) -> Result<(usize, usize, DType)> {
    let mut f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let len = f.metadata()?.len();
    let (rows, dim) = read_bin_header(&mut f)?;
    let cells = (rows * dim) as u64;
    match [DType::U8, DType::F32]
        .into_iter()
        .find(|t| len == 8 + cells * t.size() as u64)
    {
        Some(t) => Ok((rows, dim, t)),
        None => bail!(
            "{}: {len} bytes are no {rows} × {dim} u8 or f32 vectors",
            path.display()
        ),
    }
}

/// The node layout [`read_index`] gives a DiskANN graph file, without reading the lists when
/// the vector file's row count `rows` is known: the header and the node count (`rows`; without
/// it the number of lists, which needs a graph without frozen points: the lists of the Rust
/// in-memory provider's files end with an unused one that only the vector file tells apart).
pub fn index_nodes<R: Read + Seek>(graph: R, rows: Option<usize>) -> Result<(Header, usize)> {
    let mut r = BufReader::with_capacity(1 << 22, graph);
    let file_len = r.seek(SeekFrom::End(0))?;
    r.seek(SeekFrom::Start(0))?;
    let header = Header::read(&mut r)?;
    ensure!(
        header.file_size == file_len,
        "the header says {} bytes but the file has {file_len}",
        header.file_size
    );
    let n_frozen = usize::try_from(header.n_frozen).unwrap_or(usize::MAX);
    let n_nodes = match rows {
        Some(rows) => rows,
        None => {
            ensure!(
                n_frozen == 0,
                "the graph has {n_frozen} frozen points; its node count needs the vector file"
            );
            count_lists(&mut r, &header)?
        }
    };
    ensure!(
        n_frozen <= n_nodes,
        "{n_frozen} frozen points but {n_nodes} nodes"
    );
    ensure!(
        (header.start as usize) < n_nodes,
        "start {} out of range ({n_nodes} nodes)",
        header.start
    );
    Ok((header, n_nodes))
}

/// Reads the adjacency lists of the `n_nodes` nodes of a DiskANN graph file (see
/// [`index_nodes`]) in one sequential pass and calls `row(id, neighbors)` for each, with the
/// checks of [`read_index`] (degrees, neighbor ids, at most one trailing empty list); reads no
/// vectors and keeps no lists.
pub fn scan_index<R: Read + Seek>(
    graph: R,
    n_nodes: usize,
    mut row: impl FnMut(usize, &[u32]) -> Result<()>,
) -> Result<()> {
    let mut r = BufReader::with_capacity(1 << 22, graph);
    let file_len = r.seek(SeekFrom::End(0))?;
    r.seek(SeekFrom::Start(0))?;
    let header = Header::read(&mut r)?;
    let r_max = header.max_degree as usize;
    let (mut pos, mut list) = (HEADER_BYTES, Vec::with_capacity(r_max));
    for id in 0..n_nodes {
        ensure!(
            file_len - pos >= 4,
            "the graph has {id} adjacency lists for {n_nodes} nodes"
        );
        let k = read_u32(&mut r)? as usize;
        ensure!(
            k <= r_max && file_len - pos - 4 >= 4 * k as u64,
            "node {id}: degree {k} exceeds the maximum degree {r_max} or the file"
        );
        list.resize(k, 0);
        r.read_exact(bytemuck::cast_slice_mut(&mut list))?;
        if let Some(&v) = list.iter().find(|&&v| v as usize >= n_nodes) {
            bail!("node {id}: neighbor {v} out of range ({n_nodes} nodes)");
        }
        row(id, &list)?;
        pos += 4 + 4 * k as u64;
    }
    ensure!(
        pos == file_len || (file_len - pos == 4 && read_u32(&mut r)? == 0),
        "the graph has more adjacency lists than {n_nodes} nodes and one unused empty list"
    );
    Ok(())
}

/// Loads a graph from `graph.flab`, or directly from a DiskANN graph file (with its vector
/// file, see [`read_index`]).
pub fn open(path: &Path, data: Option<(&Path, DType)>) -> Result<Graph> {
    if &container::peek_magic(path)? == graph::MAGIC {
        Graph::read(path)
    } else {
        Ok(read_index_files(path, data)?.0)
    }
}

/// Row-wise comparison of a vector file with a base file (see [`compare_rows`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowComparison {
    /// Rows of the base, all compared.
    pub rows: usize,
    /// Compared rows that differ.
    pub mismatched_rows: usize,
    pub first_mismatch: Option<usize>,
    /// Rows of the vector file after the base's rows (e.g. frozen points).
    pub extra_rows: usize,
    /// Whether the two files are byte-identical.
    pub byte_identical: bool,
}

/// Compares the first rows of the vector file `data` with every row of `base`; both have
/// element type `dtype` and the same dimension, and `data` has at least as many rows.
pub fn compare_rows<A: Read + Seek, B: Read + Seek>(
    data: A,
    base: B,
    dtype: DType,
) -> Result<RowComparison> {
    let (mut a, mut b) = (
        BufReader::with_capacity(1 << 22, data),
        BufReader::with_capacity(1 << 22, base),
    );
    let (rows_a, dim_a) = bin_shape(&mut a, dtype)?;
    let (rows, dim) = bin_shape(&mut b, dtype)?;
    ensure!(
        dim_a == dim && rows_a >= rows,
        "{rows_a} × {dim_a} vectors cannot extend {rows} × {dim} base vectors"
    );
    let row_bytes = dim * dtype.size();
    let chunk_rows = ((1 << 22) / row_bytes.max(1)).max(1);
    let (mut buf_a, mut buf_b) = (Vec::new(), Vec::new());
    let (mut mismatched, mut first) = (0, None);
    let mut done = 0;
    while done < rows {
        let n = chunk_rows.min(rows - done);
        buf_a.resize(n * row_bytes, 0);
        buf_b.resize(n * row_bytes, 0);
        a.read_exact(&mut buf_a)?;
        b.read_exact(&mut buf_b)?;
        if buf_a != buf_b {
            for (i, (x, y)) in buf_a
                .chunks_exact(row_bytes)
                .zip(buf_b.chunks_exact(row_bytes))
                .enumerate()
            {
                if x != y {
                    mismatched += 1;
                    first = first.or(Some(done + i));
                }
            }
        }
        done += n;
    }
    Ok(RowComparison {
        rows,
        mismatched_rows: mismatched,
        first_mismatch: first,
        extra_rows: rows_a - rows,
        byte_identical: mismatched == 0 && rows_a == rows,
    })
}

/// Reads the header of a `.bin` vector file and checks the file size; returns (rows, dimension).
fn bin_shape<R: Read + Seek>(r: &mut R, dtype: DType) -> Result<(usize, usize)> {
    let len = r.seek(SeekFrom::End(0))?;
    r.seek(SeekFrom::Start(0))?;
    let (rows, dim) = read_bin_header(r)?;
    ensure!(
        len == 8 + (rows * dim * dtype.size()) as u64,
        "a vector file has {len} bytes, not 8 + {rows} × {dim} {} elements",
        dtype.name()
    );
    Ok((rows, dim))
}

/// [`compare_rows`] on files.
pub fn compare_row_files(data: &Path, base: &Path, dtype: DType) -> Result<RowComparison> {
    let open = |p: &Path| File::open(p).with_context(|| format!("opening {}", p.display()));
    compare_rows(open(data)?, open(base)?, dtype)
        .with_context(|| format!("comparing {} with {}", data.display(), base.display()))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    /// A DiskANN graph file with the given lists.
    fn graph_file(max_degree: u32, start: u32, n_frozen: u64, lists: &[&[u32]]) -> Vec<u8> {
        let mut body = Vec::new();
        for list in lists {
            body.extend_from_slice(&(list.len() as u32).to_le_bytes());
            for v in *list {
                body.extend_from_slice(&v.to_le_bytes());
            }
        }
        let h = Header {
            file_size: HEADER_BYTES + body.len() as u64,
            max_degree,
            start,
            n_frozen,
        };
        let mut file = h.to_bytes().to_vec();
        file.extend_from_slice(&body);
        file
    }

    fn bin_file<T: bytemuck::Pod>(rows: u32, dim: u32, data: &[T]) -> Vec<u8> {
        let mut file = rows.to_le_bytes().to_vec();
        file.extend_from_slice(&dim.to_le_bytes());
        file.extend_from_slice(bytemuck::cast_slice(data));
        file
    }

    const LISTS: [&[u32]; 4] = [&[1, 2], &[0], &[], &[0, 1, 2]];

    #[test]
    fn reads_a_graph_without_frozen_points() {
        let file = graph_file(3, 2, 0, &LISTS);
        let (g, layout) = read_index::<_, Cursor<Vec<u8>>>(Cursor::new(file), None).unwrap();
        let h = layout.header;
        assert_eq!((h.max_degree, h.start, h.n_frozen), (3, 2, 0));
        assert_eq!((layout.lists, layout.unused_lists), (4, 0));
        assert_eq!(
            (g.n_data(), g.n_nodes(), g.r_max(), g.starts()),
            (4, 4, 3, &[2][..])
        );
        for (id, list) in LISTS.iter().enumerate() {
            assert_eq!(g.neighbors(id as u32), *list);
        }
        assert_eq!((g.dtype(), g.n_edges()), (None, 6));

        // With the vector file: element type and dimension are recorded.
        let data = bin_file::<f32>(4, 1, &[0.5, 1.5, 2.5, 3.5]);
        let file = graph_file(3, 2, 0, &LISTS);
        let (g, _) = read_index(Cursor::new(file), Some((Cursor::new(data), DType::F32))).unwrap();
        assert_eq!((g.dtype(), g.dim(), g.n_aux()), (Some(DType::F32), 1, 0));
    }

    #[test]
    fn frozen_points_become_auxiliary_starts() {
        let file = graph_file(3, 3, 1, &LISTS);
        let data = bin_file::<u8>(4, 2, &[1, 2, 3, 4, 5, 6, 7, 8]);
        let (g, _) = read_index(Cursor::new(file), Some((Cursor::new(data), DType::U8))).unwrap();
        assert_eq!((g.n_data(), g.n_nodes(), g.starts()), (3, 4, &[3][..]));
        assert_eq!(g.aux_kind(3), Some(AuxKind::Frozen));
        assert_eq!(g.aux_vector::<u8>(3), Some(&[7, 8][..]));
        assert_eq!(g.neighbors(3), &[0, 1, 2]);

        // Files of the Rust in-memory provider end with one empty list that is not a node.
        let file = graph_file(3, 3, 1, &[LISTS[0], LISTS[1], LISTS[2], LISTS[3], &[]]);
        let data = bin_file::<u8>(4, 2, &[1, 2, 3, 4, 5, 6, 7, 8]);
        let (h, layout) =
            read_index(Cursor::new(file), Some((Cursor::new(data), DType::U8))).unwrap();
        assert_eq!(h, g);
        assert_eq!((layout.lists, layout.unused_lists), (5, 1));

        // A start that is a data point comes first, then every frozen point.
        let file = graph_file(3, 0, 2, &LISTS);
        let data = bin_file::<u8>(4, 1, &[1, 2, 3, 4]);
        let (g, _) = read_index(Cursor::new(file), Some((Cursor::new(data), DType::U8))).unwrap();
        assert_eq!((g.n_data(), g.starts()), (2, &[0, 2, 3][..]));
    }

    #[test]
    fn rejects_inconsistent_files() {
        let read = |file: Vec<u8>| read_index::<_, Cursor<Vec<u8>>>(Cursor::new(file), None);
        let mut wrong_size = graph_file(3, 0, 0, &LISTS);
        wrong_size[0] += 4;
        assert!(read(wrong_size).is_err());
        assert!(read(graph_file(2, 0, 0, &LISTS)).is_err());
        assert!(read(graph_file(3, 0, 0, &[&[1, 4], &[0]])).is_err());
        assert!(read(graph_file(3, 7, 0, &LISTS)).is_err());
        let mut truncated = graph_file(3, 0, 0, &LISTS);
        truncated.truncate(truncated.len() - 4);
        let size = truncated.len() as u64;
        truncated[..8].copy_from_slice(&size.to_le_bytes());
        assert!(read(truncated).is_err());
        // Frozen points need the vector file, with one row per node (and at most one unused
        // empty list after them).
        assert!(read(graph_file(3, 3, 1, &LISTS)).is_err());
        let with_rows = |rows: u32, lists: &[&[u32]]| {
            let data = bin_file::<u8>(rows, 1, &vec![0; rows as usize]);
            let file = graph_file(3, 1, 1, lists);
            read_index(Cursor::new(file), Some((Cursor::new(data), DType::U8)))
        };
        assert!(with_rows(4, &LISTS).is_ok());
        assert!(with_rows(5, &LISTS).is_err());
        assert!(with_rows(2, &LISTS).is_err());
        assert!(with_rows(3, &LISTS).is_err());
        assert!(with_rows(3, &[&[1], &[0], &[], &[]]).is_ok());
    }

    /// The lists `scan_index` reads, with the node count `index_nodes` gives.
    fn scan(file: &[u8], rows: Option<usize>) -> Result<(Header, Vec<Vec<u32>>)> {
        let (header, n_nodes) = index_nodes(Cursor::new(file), rows)?;
        let mut lists = Vec::new();
        scan_index(Cursor::new(file), n_nodes, |id, list| {
            assert_eq!(id, lists.len());
            lists.push(list.to_vec());
            Ok(())
        })?;
        Ok((header, lists))
    }

    #[test]
    fn scanning_reads_the_nodes_read_index_reads() {
        // A frozen point (node 3) and the Rust in-memory provider's unused list: 4 nodes, as
        // with the vector file's 4 rows.
        let file = graph_file(3, 3, 1, &[LISTS[0], LISTS[1], LISTS[2], LISTS[3], &[]]);
        let data = bin_file::<u8>(4, 2, &[1, 2, 3, 4, 5, 6, 7, 8]);
        let (g, _) = read_index(Cursor::new(&file), Some((Cursor::new(data), DType::U8))).unwrap();
        let (h, lists) = scan(&file, Some(4)).unwrap();
        assert_eq!((h.n_frozen, lists.len()), (1, g.n_nodes()));
        for (id, list) in lists.iter().enumerate() {
            assert_eq!(g.neighbors(id as u32), &list[..]);
        }
        // Without frozen points the lists are the nodes.
        let file = graph_file(3, 2, 0, &LISTS);
        assert_eq!(scan(&file, None).unwrap().1.len(), 4);
        // The node count of a graph with frozen points needs the vector file's rows; lists
        // missing for them, a second extra list or a non-empty extra list are errors.
        let file = graph_file(3, 3, 1, &LISTS);
        assert!(scan(&file, None).is_err());
        assert!(scan(&file, Some(4)).is_ok());
        assert!(scan(&file, Some(5)).is_err());
        let lists: [&[u32]; 4] = [&[1], &[0], &[], &[]];
        assert!(scan(&graph_file(3, 1, 1, &lists), Some(3)).is_ok());
        assert!(scan(&graph_file(3, 1, 1, &lists), Some(2)).is_err());
        let lists: [&[u32]; 4] = [&[1], &[0], &[], &[0]];
        assert!(scan(&graph_file(3, 1, 1, &lists), Some(3)).is_err());
    }

    #[test]
    fn row_comparison() {
        let base = bin_file::<u8>(3, 2, &[1, 2, 3, 4, 5, 6]);
        let cmp = |data: Vec<u8>| compare_rows(Cursor::new(data), Cursor::new(&base), DType::U8);
        let same = cmp(base.clone()).unwrap();
        assert_eq!(
            (same.mismatched_rows, same.extra_rows, same.byte_identical),
            (0, 0, true)
        );
        let extended = cmp(bin_file::<u8>(4, 2, &[1, 2, 3, 4, 5, 6, 0, 0])).unwrap();
        assert_eq!((extended.mismatched_rows, extended.extra_rows), (0, 1));
        assert!(!extended.byte_identical);
        let changed = cmp(bin_file::<u8>(3, 2, &[1, 2, 3, 9, 5, 6])).unwrap();
        assert_eq!(
            (changed.mismatched_rows, changed.first_mismatch),
            (1, Some(1))
        );
        assert!(cmp(bin_file::<u8>(2, 2, &[1, 2, 3, 4])).is_err());
        assert!(cmp(bin_file::<u8>(3, 3, &[0; 9])).is_err());
    }
}
