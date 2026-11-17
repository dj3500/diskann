/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Graphs over a node space of data points and auxiliary nodes (format `graph.flab`).
//!
//! Node ids `0..n_data` are the data points: they index the base vectors and the label store.
//! Ids `n_data..n_nodes` are **auxiliary nodes** (frozen start points, centroids, routers).
//! Their vectors are stored with the graph, in the element type of the data. Auxiliary nodes
//! have no labels and are never answers: [`NodeLabels`] returns [`Node::Auxiliary`] for them
//! instead of indexing a label row, and [`NodeVectors`] serves their vectors next to the base
//! rows.
//!
//! Adjacency lists are fixed-stride rows of `r_max` slots (unused slots hold [`NO_NEIGHBOR`])
//! plus one degree per node, so [`Graph::neighbors`] is a slice of one contiguous,
//! 64-byte-aligned array.

use std::{
    io::{Read, Seek, Write},
    path::Path,
};

use anyhow::{bail, ensure, Context, Result};
use bytemuck::{Pod, Zeroable};
use serde_json::{json, Value};

use crate::{
    container,
    labels::LabelStore,
    predicate::Cnf,
    vectors::{DType, Element, Matrix},
};

/// File magic of `graph.flab`.
pub const MAGIC: &[u8; 8] = b"FLABGRF\0";
/// Version of the `graph.flab` format.
pub const VERSION: u32 = 1;
/// Content of the unused slots of an adjacency row.
pub const NO_NEIGHBOR: u32 = u32::MAX;

/// Kind of an auxiliary node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuxKind {
    /// A start point that is not a data point (DiskANN's frozen points).
    Frozen,
    /// A cluster centroid.
    Centroid,
    /// A routing node.
    Router,
}

impl AuxKind {
    /// Code in the file format.
    pub fn code(self) -> u8 {
        match self {
            Self::Frozen => 0,
            Self::Centroid => 1,
            Self::Router => 2,
        }
    }

    pub fn from_code(code: u8) -> Result<Self> {
        Ok(match code {
            0 => Self::Frozen,
            1 => Self::Centroid,
            2 => Self::Router,
            _ => bail!("unknown auxiliary node kind {code}"),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Frozen => "frozen",
            Self::Centroid => "centroid",
            Self::Router => "router",
        }
    }
}

/// Element type code in the header: 0 = not recorded, 1 = `u8`, 2 = `f32`.
fn dtype_code(dtype: Option<DType>) -> u64 {
    match dtype {
        None => 0,
        Some(DType::U8) => 1,
        Some(DType::F32) => 2,
    }
}

fn dtype_from_code(code: u64) -> Result<Option<DType>> {
    Ok(match code {
        0 => None,
        1 => Some(DType::U8),
        2 => Some(DType::F32),
        _ => bail!("unknown element type code {code}"),
    })
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C, align(64))]
struct Line([u32; 16]);

/// `u32` array in 64-byte-aligned storage.
#[derive(Debug, Clone)]
struct Aligned {
    lines: Vec<Line>,
    len: usize,
}

impl Aligned {
    fn filled(len: usize, value: u32) -> Self {
        Self {
            lines: vec![Line([value; 16]); len.div_ceil(16)],
            len,
        }
    }

    #[inline]
    fn as_slice(&self) -> &[u32] {
        &bytemuck::cast_slice(&self.lines)[..self.len]
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [u32] {
        &mut bytemuck::cast_slice_mut(&mut self.lines)[..self.len]
    }
}

/// The auxiliary nodes of a graph: their kinds and vectors (one row per node, in the element
/// type of the data). Also records the element type and dimension of the data, if known.
#[derive(Debug, Clone)]
pub struct AuxNodes {
    kinds: Vec<AuxKind>,
    dtype: Option<DType>,
    dim: usize,
    /// One row of `dim` elements per node, as bytes.
    bytes: Matrix<u8>,
}

impl AuxNodes {
    /// No auxiliary nodes and no recorded element type (a graph without its vectors).
    pub fn untyped() -> Self {
        Self {
            kinds: Vec::new(),
            dtype: None,
            dim: 0,
            bytes: Matrix::zeros(0, 0),
        }
    }

    /// Auxiliary nodes of the given kinds; row `i` of `vectors` is the vector of node
    /// `n_data + i`. With zero rows this records only the element type and dimension.
    pub fn new<T: Element>(kinds: Vec<AuxKind>, vectors: &Matrix<T>) -> Result<Self> {
        ensure!(
            kinds.len() == vectors.nrows(),
            "{} auxiliary node kinds but {} vectors",
            kinds.len(),
            vectors.nrows()
        );
        let row_bytes = vectors.ncols() * std::mem::size_of::<T>();
        Ok(Self {
            kinds,
            dtype: Some(T::DTYPE),
            dim: vectors.ncols(),
            bytes: Matrix::from_slice(
                bytemuck::cast_slice(vectors.as_slice()),
                vectors.nrows(),
                row_bytes,
            ),
        })
    }

    pub fn len(&self) -> usize {
        self.kinds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kinds.is_empty()
    }

    pub fn kinds(&self) -> &[AuxKind] {
        &self.kinds
    }

    /// Element type of the data, if recorded.
    pub fn dtype(&self) -> Option<DType> {
        self.dtype
    }

    /// Dimension of the data (0 if not recorded).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// All auxiliary vectors (row-major) if their element type is `T`.
    pub fn vectors<T: Element>(&self) -> Option<&[T]> {
        (self.dtype == Some(T::DTYPE)).then(|| bytemuck::cast_slice(self.bytes.as_slice()))
    }
}

/// A directed graph over data points and auxiliary nodes, with fixed-stride adjacency rows and
/// a list of start (entry) nodes.
#[derive(Debug, Clone)]
pub struct Graph {
    n_data: u32,
    r_max: usize,
    degrees: Vec<u32>,
    adjacency: Aligned,
    starts: Vec<u32>,
    aux: AuxNodes,
}

impl Graph {
    /// A graph without edges or starts over `n_data` data points followed by the auxiliary
    /// nodes `aux` (ids `n_data..n_data + aux.len()`), with rows of `r_max` slots.
    pub fn new(n_data: usize, r_max: usize, aux: AuxNodes) -> Result<Self> {
        let n_nodes = n_data.checked_add(aux.len()).context("too many nodes")?;
        ensure!(
            n_nodes < NO_NEIGHBOR as usize,
            "too many nodes ({n_nodes}) for 32-bit ids"
        );
        let slots = n_nodes
            .checked_mul(r_max)
            .context("adjacency array too large")?;
        Ok(Self {
            n_data: n_data as u32,
            r_max,
            degrees: vec![0; n_nodes],
            adjacency: Aligned::filled(slots, NO_NEIGHBOR),
            starts: Vec::new(),
            aux,
        })
    }

    /// Number of data points (ids `0..n_data`).
    #[inline]
    pub fn n_data(&self) -> usize {
        self.n_data as usize
    }

    /// Number of nodes, data points and auxiliary nodes.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.degrees.len()
    }

    /// Number of auxiliary nodes (ids `n_data..n_nodes`).
    pub fn n_aux(&self) -> usize {
        self.aux.len()
    }

    /// Slots per adjacency row (the maximum degree).
    pub fn r_max(&self) -> usize {
        self.r_max
    }

    /// Total number of edges.
    pub fn n_edges(&self) -> u64 {
        self.degrees.iter().map(|&d| u64::from(d)).sum()
    }

    #[inline]
    pub fn degree(&self, id: u32) -> usize {
        self.degrees[id as usize] as usize
    }

    /// Out-neighbors of node `id`.
    #[inline]
    pub fn neighbors(&self, id: u32) -> &[u32] {
        let start = id as usize * self.r_max;
        &self.adjacency.as_slice()[start..start + self.degrees[id as usize] as usize]
    }

    /// The whole adjacency row of node `id` (`r_max` slots, unused ones hold [`NO_NEIGHBOR`]),
    /// e.g. for prefetching.
    #[inline]
    pub fn row(&self, id: u32) -> &[u32] {
        let start = id as usize * self.r_max;
        &self.adjacency.as_slice()[start..start + self.r_max]
    }

    /// Prefetch hint for node `id`'s out-degree and the first cache lines of its adjacency row
    /// (at most `r_max` slots), so that an expansion that comes soon finds them in cache.
    #[inline(always)]
    pub fn prefetch_row(&self, id: u32) {
        diskann_vector::prefetch_hint_max::<1, _>(&self.degrees[id as usize..id as usize + 1]);
        diskann_vector::prefetch_hint_max::<4, _>(self.row(id));
    }

    /// Start (entry) nodes in priority order.
    pub fn starts(&self) -> &[u32] {
        &self.starts
    }

    /// Whether `id` is a data point (and may be an answer).
    #[inline]
    pub fn is_data(&self, id: u32) -> bool {
        id < self.n_data
    }

    /// Kind of auxiliary node `id`; `None` for a data point.
    ///
    /// # Panics
    ///
    /// Panics if `id >= n_nodes`.
    #[inline]
    pub fn aux_kind(&self, id: u32) -> Option<AuxKind> {
        id.checked_sub(self.n_data)
            .map(|i| self.aux.kinds[i as usize])
    }

    pub fn aux(&self) -> &AuxNodes {
        &self.aux
    }

    /// Element type of the data, if recorded.
    pub fn dtype(&self) -> Option<DType> {
        self.aux.dtype
    }

    /// Dimension of the data (0 if not recorded).
    pub fn dim(&self) -> usize {
        self.aux.dim
    }

    /// Vector of auxiliary node `id`; `None` for a data point.
    ///
    /// # Panics
    ///
    /// Panics if `id >= n_nodes` or if the element type of the graph is not `T`.
    #[inline]
    pub fn aux_vector<T: Element>(&self, id: u32) -> Option<&[T]> {
        let i = id.checked_sub(self.n_data)? as usize;
        let dim = self.aux.dim;
        let all = self.aux.vectors::<T>().unwrap_or_else(|| {
            panic!(
                "the auxiliary vectors are {}, not {}",
                self.aux.dtype.map_or("not recorded", DType::name),
                T::NAME
            )
        });
        Some(&all[i * dim..(i + 1) * dim])
    }

    /// Replaces the out-neighbors of node `id`.
    pub fn set_neighbors(&mut self, id: u32, neighbors: &[u32]) -> Result<()> {
        let n = self.n_nodes();
        ensure!((id as usize) < n, "node {id} out of range ({n} nodes)");
        ensure!(
            neighbors.len() <= self.r_max,
            "node {id}: {} neighbors exceed r_max = {}",
            neighbors.len(),
            self.r_max
        );
        if let Some(&v) = neighbors.iter().find(|&&v| v as usize >= n) {
            bail!("node {id}: neighbor {v} out of range ({n} nodes)");
        }
        let start = id as usize * self.r_max;
        let row = &mut self.adjacency.as_mut_slice()[start..start + self.r_max];
        row[..neighbors.len()].copy_from_slice(neighbors);
        row[neighbors.len()..].fill(NO_NEIGHBOR);
        self.degrees[id as usize] = neighbors.len() as u32;
        Ok(())
    }

    /// Sets the start (entry) nodes, in priority order.
    pub fn set_starts(&mut self, starts: Vec<u32>) -> Result<()> {
        let n = self.n_nodes();
        if let Some(&s) = starts.iter().find(|&&s| s as usize >= n) {
            bail!("start {s} out of range ({n} nodes)");
        }
        self.starts = starts;
        Ok(())
    }

    /// Degrees and adjacency rows for bulk filling. The caller must restore the invariants
    /// checked by [`Self::validate`].
    pub(crate) fn raw_mut(&mut self) -> (&mut [u32], &mut [u32]) {
        (&mut self.degrees, self.adjacency.as_mut_slice())
    }

    /// Checks degrees, neighbor ids, row padding and starts.
    pub(crate) fn validate(&self) -> Result<()> {
        let n = self.n_nodes();
        for id in 0..n as u32 {
            let degree = self.degree(id);
            ensure!(
                degree <= self.r_max,
                "node {id}: degree {degree} exceeds r_max = {}",
                self.r_max
            );
            let row = self.row(id);
            if let Some(&v) = row[..degree].iter().find(|&&v| v as usize >= n) {
                bail!("node {id}: neighbor {v} out of range ({n} nodes)");
            }
            ensure!(
                row[degree..].iter().all(|&v| v == NO_NEIGHBOR),
                "node {id}: unused row slots are not {NO_NEIGHBOR}"
            );
        }
        if let Some(&s) = self.starts.iter().find(|&&s| s as usize >= n) {
            bail!("start {s} out of range ({n} nodes)");
        }
        Ok(())
    }

    /// Bounds-safe label access by node id, from the label store of the data points.
    pub fn labels<'a>(&'a self, store: &'a LabelStore) -> Result<NodeLabels<'a>> {
        ensure!(
            store.n_points() == self.n_data(),
            "the label store has {} points but the graph has {} data points",
            store.n_points(),
            self.n_data()
        );
        Ok(NodeLabels { graph: self, store })
    }

    /// Vectors by node id: rows of `base` for data points, stored vectors for auxiliary nodes.
    pub fn vectors<'a, T: Element>(&'a self, base: &'a Matrix<T>) -> Result<NodeVectors<'a, T>> {
        ensure!(
            base.nrows() == self.n_data(),
            "the base has {} vectors but the graph has {} data points",
            base.nrows(),
            self.n_data()
        );
        let aux: &[T] = match self.aux.dtype {
            None => &[],
            Some(dtype) => {
                ensure!(
                    dtype == T::DTYPE,
                    "the graph was built on {} vectors, the base has {}",
                    dtype.name(),
                    T::NAME
                );
                ensure!(
                    self.aux.dim == base.ncols(),
                    "the graph was built on {}-dimensional vectors, the base has {}",
                    self.aux.dim,
                    base.ncols()
                );
                self.aux.vectors().unwrap_or_default()
            }
        };
        Ok(NodeVectors {
            base: base.as_slice(),
            aux,
            dim: base.ncols(),
            n_data: self.n_data,
        })
    }

    /// Writes `graph.flab` (see the crate `README.md` for the layout).
    pub fn write(&self, path: &Path) -> Result<()> {
        self.write_to(container::create(path)?)
            .with_context(|| format!("writing {}", path.display()))
    }

    /// Writes the `graph.flab` format to any writer.
    pub fn write_to<W: Write>(&self, w: W) -> Result<()> {
        let kinds: Vec<u8> = self.aux.kinds.iter().map(|k| k.code()).collect();
        let params = [
            u64::from(self.n_data),
            self.n_nodes() as u64,
            self.r_max as u64,
            self.starts.len() as u64,
            dtype_code(self.aux.dtype),
            self.aux.dim as u64,
            0,
            self.n_edges(),
        ];
        container::write(
            w,
            MAGIC,
            VERSION,
            &params,
            &[
                bytemuck::cast_slice(&self.degrees),
                bytemuck::cast_slice(self.adjacency.as_slice()),
                bytemuck::cast_slice(&self.starts),
                &kinds,
                self.aux.bytes.as_slice(),
            ],
        )
    }

    /// Reads and validates `graph.flab`.
    pub fn read(path: &Path) -> Result<Self> {
        container::Reader::open(path, MAGIC)
            .and_then(Self::load)
            .with_context(|| format!("loading a graph from {}", path.display()))
    }

    /// Reads and validates the `graph.flab` format from any reader.
    pub fn read_from<R: Read + Seek>(r: R) -> Result<Self> {
        Self::load(container::Reader::new(r, MAGIC)?)
    }

    fn load<R: Read + Seek>(mut r: container::Reader<R>) -> Result<Self> {
        ensure!(
            r.version() == VERSION,
            "unsupported version {}",
            r.version()
        );
        let [n_data, n_nodes, r_max, n_starts, dtype, dim, flags, n_edges] = *r.params();
        ensure!(flags == 0, "unknown flags {flags:#x}");
        ensure!(
            n_data <= n_nodes,
            "{n_data} data points but only {n_nodes} nodes"
        );
        let dtype = dtype_from_code(dtype)?;
        let n_aux = (n_nodes - n_data) as usize;
        ensure!(
            dtype.is_some() || (n_aux == 0 && dim == 0),
            "auxiliary nodes or a dimension without an element type"
        );
        let kinds = r
            .section::<u8>(3, n_aux)?
            .into_iter()
            .map(AuxKind::from_code)
            .collect::<Result<Vec<_>>>()?;
        let row_bytes = (dim as usize)
            .checked_mul(dtype.map_or(0, DType::size))
            .context("dimension too large")?;
        let mut bytes = Matrix::<u8>::zeros(n_aux, row_bytes);
        r.section_into(4, bytes.as_mut_slice())?;
        let aux = AuxNodes {
            kinds,
            dtype,
            dim: dim as usize,
            bytes,
        };
        let mut graph = Self::new(n_data as usize, r_max as usize, aux)?;
        r.section_into(0, &mut graph.degrees)?;
        r.section_into(1, graph.adjacency.as_mut_slice())?;
        graph.starts = r.section(2, n_starts as usize)?;
        graph.validate()?;
        ensure!(
            graph.n_edges() == n_edges,
            "the header says {n_edges} edges, the rows hold {}",
            graph.n_edges()
        );
        Ok(graph)
    }

    /// Nodes reachable from `sources` along out-edges (the sources included), as a bitset over
    /// node ids.
    pub fn reachable(&self, sources: &[u32]) -> Vec<u64> {
        let mut seen = vec![0u64; self.n_nodes().div_ceil(64)];
        let mut stack = Vec::new();
        let mut visit = |v: u32, stack: &mut Vec<u32>| {
            let (w, b) = (v as usize / 64, 1u64 << (v % 64));
            if seen[w] & b == 0 {
                seen[w] |= b;
                stack.push(v);
            }
        };
        for &s in sources {
            visit(s, &mut stack);
        }
        while let Some(v) = stack.pop() {
            for &u in self.neighbors(v) {
                visit(u, &mut stack);
            }
        }
        seen
    }

    /// Structural statistics (degrees, rows, in-degrees, reachability from the starts).
    pub fn stats(&self) -> GraphStats {
        let (n, n_data) = (self.n_nodes(), self.n_data());
        let mut histogram = vec![0u64; self.r_max + 1];
        let mut in_degrees = vec![0u64; n];
        let (mut self_loops, mut duplicates, mut to_aux) = (0, 0, 0);
        let mut scratch = Vec::with_capacity(self.r_max);
        for id in 0..n as u32 {
            let nbrs = self.neighbors(id);
            if self.is_data(id) {
                histogram[nbrs.len()] += 1;
            }
            for &v in nbrs {
                in_degrees[v as usize] += 1;
                self_loops += u64::from(v == id);
                to_aux += u64::from(!self.is_data(v));
            }
            scratch.clear();
            scratch.extend_from_slice(nbrs);
            scratch.sort_unstable();
            duplicates += scratch.windows(2).filter(|w| w[0] == w[1]).count() as u64;
        }
        let mut out_degrees: Vec<u64> = self.degrees[..n_data]
            .iter()
            .map(|&d| u64::from(d))
            .collect();
        let reachable = self.reachable(&self.starts);
        let is_set = |v: usize| reachable[v / 64] >> (v % 64) & 1 == 1;
        let in_data = &mut in_degrees[..n_data];
        GraphStats {
            n_edges: self.n_edges(),
            zero_out_degree: histogram[0] as usize,
            full_rows: histogram[self.r_max] as usize,
            degree: Summary::of(&mut out_degrees),
            zero_in_degree: in_data.iter().filter(|&&d| d == 0).count(),
            in_degree: Summary::of(in_data),
            degree_histogram: histogram,
            self_loops,
            duplicate_edges: duplicates,
            edges_to_aux: to_aux,
            aux_degrees: self.degrees[n_data..].to_vec(),
            reachable_nodes: reachable.iter().map(|w| w.count_ones() as usize).sum(),
            reachable_data: (0..n_data).filter(|&v| is_set(v)).count(),
        }
    }
}

/// Graphs are equal when their node spaces, rows, starts and auxiliary nodes are.
impl PartialEq for Graph {
    fn eq(&self, other: &Self) -> bool {
        self.n_data == other.n_data
            && self.r_max == other.r_max
            && self.degrees == other.degrees
            && self.adjacency.as_slice() == other.adjacency.as_slice()
            && self.starts == other.starts
            && self.aux.kinds == other.aux.kinds
            && self.aux.dtype == other.aux.dtype
            && self.aux.dim == other.aux.dim
            && self.aux.bytes.as_slice() == other.aux.bytes.as_slice()
    }
}

impl Eq for Graph {}

/// Nearest-rank summary of a distribution of counts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    pub min: u64,
    pub p10: u64,
    pub p50: u64,
    pub p90: u64,
    pub max: u64,
    pub mean: f64,
}

impl Summary {
    /// Summary of `values` (sorted in place); `None` if empty.
    pub fn of(values: &mut [u64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        values.sort_unstable();
        let at = |p: f64| values[((values.len() - 1) as f64 * p).round() as usize];
        Some(Self {
            min: at(0.0),
            p10: at(0.1),
            p50: at(0.5),
            p90: at(0.9),
            max: at(1.0),
            mean: values.iter().sum::<u64>() as f64 / values.len() as f64,
        })
    }

    pub fn to_json(&self) -> Value {
        json!({
            "min": self.min, "p10": self.p10, "p50": self.p50, "p90": self.p90,
            "max": self.max, "mean": self.mean,
        })
    }
}

impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "min {} p10 {} p50 {} p90 {} max {} mean {:.2}",
            self.min, self.p10, self.p50, self.p90, self.max, self.mean
        )
    }
}

/// Structural statistics of a graph (see [`Graph::stats`]). Degree statistics are over the
/// data points.
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub n_edges: u64,
    /// Out-degrees of the data points.
    pub degree: Option<Summary>,
    /// Number of data points per out-degree `0..=r_max`.
    pub degree_histogram: Vec<u64>,
    /// Data points without out-neighbors.
    pub zero_out_degree: usize,
    /// Data points with full rows (degree `r_max`).
    pub full_rows: usize,
    /// In-degrees of the data points (edges from any node).
    pub in_degree: Option<Summary>,
    /// Data points without in-neighbors.
    pub zero_in_degree: usize,
    pub self_loops: u64,
    /// Repeated entries within a row (beyond the first occurrence).
    pub duplicate_edges: u64,
    /// Edges into auxiliary nodes.
    pub edges_to_aux: u64,
    /// Out-degrees of the auxiliary nodes.
    pub aux_degrees: Vec<u32>,
    /// Nodes reachable from the starts (the starts included).
    pub reachable_nodes: usize,
    /// Data points reachable from the starts.
    pub reachable_data: usize,
}

impl GraphStats {
    pub fn to_json(&self) -> Value {
        let summary = |s: &Option<Summary>| s.as_ref().map_or(Value::Null, Summary::to_json);
        json!({
            "n_edges": self.n_edges,
            "out_degree": summary(&self.degree),
            "out_degree_histogram": self.degree_histogram,
            "zero_out_degree": self.zero_out_degree,
            "full_rows": self.full_rows,
            "in_degree": summary(&self.in_degree),
            "zero_in_degree": self.zero_in_degree,
            "self_loops": self.self_loops,
            "duplicate_edges": self.duplicate_edges,
            "edges_to_aux": self.edges_to_aux,
            "aux_out_degrees": self.aux_degrees,
            "reachable_nodes": self.reachable_nodes,
            "reachable_data": self.reachable_data,
        })
    }
}

/// A node's labels: the sorted label ids of a data point, or a marker for an auxiliary node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Node<'a> {
    Data(&'a [u32]),
    Auxiliary(AuxKind),
}

/// Bounds-safe label access by node id (see [`Graph::labels`]): auxiliary nodes never index
/// the label store and never match a predicate.
#[derive(Debug, Clone, Copy)]
pub struct NodeLabels<'a> {
    graph: &'a Graph,
    store: &'a LabelStore,
}

impl<'a> NodeLabels<'a> {
    /// Labels of node `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id >= n_nodes`.
    #[inline]
    pub fn get(&self, id: u32) -> Node<'a> {
        match self.graph.aux_kind(id) {
            None => Node::Data(self.store.row(id)),
            Some(kind) => Node::Auxiliary(kind),
        }
    }

    /// Sorted label ids of data point `id`; `None` for an auxiliary node.
    #[inline]
    pub fn row(&self, id: u32) -> Option<&'a [u32]> {
        self.graph.is_data(id).then(|| self.store.row(id))
    }

    /// Whether node `id` matches `cnf`; auxiliary nodes never match.
    #[inline]
    pub fn matches(&self, cnf: Cnf<'_>, id: u32) -> bool {
        self.row(id).is_some_and(|row| cnf.matches(row))
    }

    /// Number of clauses of `cnf` that node `id` fails; `None` for an auxiliary node.
    #[inline]
    pub fn failed_clauses(&self, cnf: Cnf<'_>, id: u32) -> Option<usize> {
        self.row(id).map(|row| cnf.failed_clauses(row))
    }
}

/// Vectors by node id (see [`Graph::vectors`]).
#[derive(Debug, Clone, Copy)]
pub struct NodeVectors<'a, T> {
    base: &'a [T],
    aux: &'a [T],
    dim: usize,
    n_data: u32,
}

impl<'a, T: Element> NodeVectors<'a, T> {
    /// The vectors of the data points of `base` (no auxiliary nodes).
    pub fn data(base: &'a Matrix<T>) -> Self {
        Self {
            base: base.as_slice(),
            aux: &[],
            dim: base.ncols(),
            n_data: base.nrows() as u32,
        }
    }
}

impl<'a, T> NodeVectors<'a, T> {
    /// Vector of node `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id >= n_nodes`.
    #[inline]
    pub fn get(&self, id: u32) -> &'a [T] {
        let (rows, i) = if id < self.n_data {
            (self.base, id as usize)
        } else {
            (self.aux, (id - self.n_data) as usize)
        };
        &rows[i * self.dim..(i + 1) * self.dim]
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::{labels::LabelStoreBuilder, predicate::PredicateSetBuilder};

    /// Three data points and one frozen node (id 3) with 2-dimensional `u8` vectors.
    pub(crate) fn sample() -> Graph {
        let frozen = Matrix::<u8>::from_slice(&[7, 9], 1, 2);
        let aux = AuxNodes::new(vec![AuxKind::Frozen], &frozen).unwrap();
        let mut g = Graph::new(3, 3, aux).unwrap();
        g.set_neighbors(0, &[1, 2]).unwrap();
        g.set_neighbors(1, &[0]).unwrap();
        g.set_neighbors(3, &[0, 1, 2]).unwrap();
        g.set_starts(vec![3]).unwrap();
        g
    }

    #[test]
    fn rows_and_node_space() {
        let g = sample();
        assert_eq!(
            (g.n_data(), g.n_nodes(), g.n_aux(), g.r_max()),
            (3, 4, 1, 3)
        );
        assert_eq!(g.neighbors(0), &[1, 2]);
        assert_eq!(g.row(1), &[0, NO_NEIGHBOR, NO_NEIGHBOR]);
        assert!(g.neighbors(2).is_empty());
        assert_eq!((g.n_edges(), g.starts()), (6, &[3][..]));
        assert!(g.is_data(2) && !g.is_data(3));
        assert_eq!(
            (g.aux_kind(2), g.aux_kind(3)),
            (None, Some(AuxKind::Frozen))
        );
        assert_eq!(g.aux_vector::<u8>(3), Some(&[7, 9][..]));
        assert_eq!(g.aux_vector::<u8>(0), None);
        assert_eq!(g.adjacency.as_slice().as_ptr() as usize % 64, 0);

        let mut g = g;
        assert!(g.set_neighbors(0, &[1, 2, 3, 0]).is_err());
        assert!(g.set_neighbors(0, &[4]).is_err());
        assert!(g.set_starts(vec![4]).is_err());
        g.set_neighbors(0, &[2]).unwrap();
        assert_eq!(g.row(0), &[2, NO_NEIGHBOR, NO_NEIGHBOR]);
    }

    #[test]
    fn auxiliary_nodes_have_no_labels_and_never_match() {
        let g = sample();
        let mut b = LabelStoreBuilder::new();
        for row in [&["c=a"][..], &["c=b"], &[]] {
            b.push_point(row.iter().map(|&l| ("c", l))).unwrap();
        }
        let store = b.finish().unwrap();
        let mut pb = PredicateSetBuilder::new(&store);
        pb.push(&[vec!["c=a", "c=b"]]);
        pb.push::<&str>(&[]);
        let preds = pb.finish();
        let (any_label, everything) = (preds.get(0), preds.get(1));

        let labels = g.labels(&store).unwrap();
        assert_eq!(labels.get(0), Node::Data(store.row(0)));
        assert_eq!(labels.get(3), Node::Auxiliary(AuxKind::Frozen));
        assert_eq!(labels.row(3), None);
        assert!(labels.matches(any_label, 1) && !labels.matches(any_label, 2));
        // The predicate without clauses matches every data point, but no auxiliary node.
        assert!(labels.matches(everything, 2) && !labels.matches(everything, 3));
        assert_eq!(labels.failed_clauses(any_label, 2), Some(1));
        assert_eq!(labels.failed_clauses(any_label, 3), None);

        let mut short = LabelStoreBuilder::new();
        short.push_point([("c", "c=a")]).unwrap();
        assert!(g.labels(&short.finish().unwrap()).is_err());
    }

    #[test]
    fn node_vectors_serve_base_and_auxiliary_rows() {
        let g = sample();
        let base = Matrix::<u8>::from_slice(&[1, 2, 3, 4, 5, 6], 3, 2);
        let v = g.vectors(&base).unwrap();
        assert_eq!((v.get(1), v.get(3), v.dim()), (&[3, 4][..], &[7, 9][..], 2));
        assert!(g.vectors(&Matrix::<f32>::zeros(3, 2)).is_err());
        assert!(g.vectors(&Matrix::<u8>::zeros(3, 3)).is_err());
        assert!(g.vectors(&Matrix::<u8>::zeros(4, 2)).is_err());
    }

    #[test]
    #[should_panic(expected = "not f32")]
    fn aux_vector_checks_the_element_type() {
        sample().aux_vector::<f32>(3);
    }

    #[test]
    fn file_roundtrip_and_validation() {
        let g = sample();
        let mut buf = Vec::new();
        g.write_to(&mut buf).unwrap();
        let h = Graph::read_from(Cursor::new(&buf)).unwrap();
        assert_eq!(h, g);
        assert_eq!((h.n_data(), h.n_nodes(), h.r_max()), (3, 4, 3));
        for id in 0..4 {
            assert_eq!(h.neighbors(id), g.neighbors(id));
        }
        assert_eq!(h.starts(), g.starts());
        assert_eq!((h.dtype(), h.dim()), (Some(DType::U8), 2));
        assert_eq!(h.aux().kinds(), &[AuxKind::Frozen]);
        assert_eq!(h.aux_vector::<u8>(3), Some(&[7, 9][..]));

        // A graph without recorded vectors.
        let mut u = Graph::new(2, 1, AuxNodes::untyped()).unwrap();
        u.set_neighbors(0, &[1]).unwrap();
        let mut ubuf = Vec::new();
        u.write_to(&mut ubuf).unwrap();
        let u = Graph::read_from(Cursor::new(&ubuf)).unwrap();
        assert_eq!((u.dtype(), u.neighbors(0)), (None, &[1][..]));

        // Corruptions: a neighbor out of range, a padding slot, the edge count. The section
        // table starts at byte 80; entry 1 is the adjacency section.
        let at = u64::from_le_bytes(buf[96..104].try_into().unwrap()) as usize;
        let mut bad = buf.clone();
        bad[at..at + 4].copy_from_slice(&9u32.to_le_bytes());
        assert!(Graph::read_from(Cursor::new(&bad)).is_err());
        let mut bad = buf.clone();
        bad[at + 8..at + 12].copy_from_slice(&1u32.to_le_bytes());
        assert!(Graph::read_from(Cursor::new(&bad)).is_err());
        let mut bad = buf.clone();
        bad[16 + 7 * 8..16 + 8 * 8].copy_from_slice(&5u64.to_le_bytes());
        assert!(Graph::read_from(Cursor::new(&bad)).is_err());
    }

    #[test]
    fn stats_and_reachability() {
        let mut g = sample();
        g.set_neighbors(1, &[0, 1, 0]).unwrap();
        let s = g.stats();
        assert_eq!(s.n_edges, 8);
        assert_eq!(s.degree_histogram, vec![1, 0, 1, 1]);
        assert_eq!((s.zero_out_degree, s.full_rows), (1, 1));
        assert_eq!((s.self_loops, s.duplicate_edges, s.edges_to_aux), (1, 1, 0));
        assert_eq!(s.degree.unwrap().max, 3);
        assert_eq!(s.aux_degrees, vec![3]);
        assert_eq!((s.reachable_nodes, s.reachable_data), (4, 3));
        assert_eq!(s.zero_in_degree, 0);

        g.set_starts(vec![2]).unwrap();
        let s = g.stats();
        assert_eq!((s.reachable_nodes, s.reachable_data), (1, 1));
        assert_eq!(g.reachable(&[1]), vec![0b111]);
    }
}
