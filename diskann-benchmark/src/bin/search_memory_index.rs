/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! CLI tool to search an in-memory DiskANN index, measuring recall and QPS.
//!
//! Mimics the old C++ `apps/search_memory_index` command, with additional
//! support for filtered search using BetaFilter or MultihopSearch strategies,
//! and a brute-force fallback for highly selective queries.
//!
//! # Example (unfiltered)
//!
//! ```bash
//! search_memory_index \
//!   --data_type float --dist_fn l2 \
//!   --index_path_prefix data/sift/index_R32_L50 \
//!   --query_file data/sift/sift_query.fbin \
//!   --gt_file data/sift/sift_gt.bin \
//!   -K 10 -L 10 20 30 40 50 100 \
//!   --result_path data/sift/results
//! ```
//!
//! # Example (filtered with BetaFilter)
//!
//! ```bash
//! search_memory_index \
//!   --data_type float --dist_fn l2 \
//!   --index_path_prefix data/sift/index_R32_L50 \
//!   --query_file data/sift/sift_query.fbin \
//!   --gt_file data/sift/sift_gt.bin \
//!   --data_labels data/sift/base_labels.jsonl \
//!   --query_labels data/sift/query_labels.jsonl \
//!   --filter_strategy beta --beta 0.5 \
//!   -K 10 -L 10 20 30 40 50 100
//! ```
//!
//! # Example (filtered with MultihopSearch)
//!
//! ```bash
//! search_memory_index \
//!   --data_type float --dist_fn l2 \
//!   --index_path_prefix data/sift/index_R32_L50 \
//!   --query_file data/sift/sift_query.fbin \
//!   --gt_file data/sift/sift_gt.bin \
//!   --data_labels data/sift/base_labels.jsonl \
//!   --query_labels data/sift/query_labels.jsonl \
//!   --filter_strategy multihop \
//!   -K 10 -L 10 20 30 40 50 100
//! ```
//!
//! # Example (filtered with brute-force fallback for selective queries)
//!
//! ```bash
//! search_memory_index \
//!   --data_type float --dist_fn l2 \
//!   --index_path_prefix data/sift/index_R32_L50 \
//!   --data_path data/sift/sift_base.fbin \
//!   --query_file data/sift/sift_query.fbin \
//!   --gt_file data/sift/sift_gt.bin \
//!   --data_labels data/sift/base_labels.jsonl \
//!   --query_labels data/sift/query_labels.jsonl \
//!   --filter_strategy beta --beta 0.5 \
//!   --brute_force_threshold 500 \
//!   -K 10 -L 10 20 30 40 50 100
//! ```

use std::collections::BinaryHeap;
use std::collections::HashSet;
use std::io::Read;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use diskann::graph::index::QueryLabelProvider;
use diskann::graph::search_output_buffer;
use diskann::graph::{self, config, DiskANNIndex};
use diskann::neighbor::Neighbor;
use diskann::provider::DefaultContext;
use diskann::utils::{IntoUsize, VectorRepr};
use diskann_label_filter::attribute::Attribute;
use diskann_label_filter::encoded_attribute_provider::attribute_encoder::AttributeEncoder;
use diskann_label_filter::encoded_attribute_provider::encoded_filter_expr::EncodedFilterExpr;
use diskann_label_filter::encoded_attribute_provider::roaring_attribute_store::RoaringAttributeStore;
use diskann_label_filter::parser::ast::{ASTExpr, CompareOp};
use diskann_label_filter::read_and_parse_queries;
use diskann_providers::model::configuration::IndexConfiguration;
use diskann_providers::model::graph::provider::async_::common::FullPrecision;
use diskann_providers::model::graph::provider::async_::inmem::FullPrecisionProvider;
use diskann_providers::model::graph::provider::layers::BetaFilter;
use diskann_providers::storage::{FileStorageProvider, LoadWith, StorageReadProvider};
use diskann_providers::utils::load_metadata_from_file;
use diskann_utils::future::AsyncFriendly;
use diskann_utils::views::Matrix;
use diskann_vector::distance::{DistanceProvider, Metric};
use half::f16;

// Reuse the save_and_load helpers from benchmark
mod save_and_load_helpers {
    use std::io::Read;
    use std::mem::size_of;
    use std::num::NonZeroUsize;

    use diskann::ANNResult;
    use diskann_providers::storage::StorageReadProvider;

    pub fn get_graph_num_frozen_points(
        storage_provider: &impl StorageReadProvider,
        graph_file: &str,
    ) -> ANNResult<NonZeroUsize> {
        let mut file = storage_provider.open_reader(graph_file)?;
        let mut usize_buffer = [0; size_of::<usize>()];
        let mut u32_buffer = [0; size_of::<u32>()];

        file.read_exact(&mut usize_buffer)?;
        file.read_exact(&mut u32_buffer)?;
        file.read_exact(&mut u32_buffer)?;
        file.read_exact(&mut usize_buffer)?;
        let file_frozen_pts = usize::from_le_bytes(usize_buffer);

        NonZeroUsize::new(file_frozen_pts).ok_or_else(|| {
            diskann::ANNError::log_index_config_error(
                "num_frozen_pts".to_string(),
                "num_frozen_pts is zero in saved file".to_string(),
            )
        })
    }

    pub fn get_graph_max_observed_degree(
        storage_provider: &impl StorageReadProvider,
        graph_file: &str,
    ) -> ANNResult<u32> {
        let mut file = storage_provider.open_reader(graph_file)?;
        let mut usize_buffer = [0; size_of::<usize>()];
        let mut u32_buffer = [0; size_of::<u32>()];

        file.read_exact(&mut usize_buffer)?;
        file.read_exact(&mut u32_buffer)?;
        let max_observed_degree = u32::from_le_bytes(u32_buffer);

        Ok(max_observed_degree)
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DataType {
    Float,
    #[value(alias("fp16"))]
    Float16,
    #[value(alias("uint8"))]
    Uint8,
    #[value(alias("int8"))]
    Int8,
}

/// Which filter strategy to use during search.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum FilterStrategy {
    /// No filter (post-filter only: run unfiltered search then discard non-matching results).
    #[default]
    None,
    /// BetaFilter: multiply distance by beta for matching vectors to bias search toward them.
    /// Soft filter — non-matching results may still appear, but are ranked lower.
    Beta,
    /// MultihopSearch: two-hop expansion through non-matching nodes to discover matching
    /// neighbors. Hard filter — only matching vectors enter the result set.
    Multihop,
    /// InlineBeta: like Beta, but label lookup happens inline during graph traversal via
    /// RoaringAttributeStore + DocumentProvider. No bitmap precomputation. Per-query label
    /// evaluation cost is included in the reported latency.
    #[value(alias("inline_beta"))]
    InlineBeta,
}

#[derive(Debug, Parser)]
#[command(name = "search_memory_index", about = "Search an in-memory DiskANN index and report recall/QPS")]
struct Args {
    /// Data type of the vectors.
    #[arg(long = "data_type", default_value = "float")]
    data_type: DataType,

    /// Distance function: l2, mips, or cosine.
    #[arg(long = "dist_fn", default_value = "l2")]
    dist_fn: Metric,

    /// Path prefix of the saved index.
    #[arg(long = "index_path_prefix", required = true)]
    index_path_prefix: String,

    /// Path to query vectors in .bin or .fbin format.
    #[arg(long = "query_file", required = true)]
    query_file: String,

    /// Path to ground truth file in .bin format.
    /// Use "null" to skip recall computation.
    #[arg(long = "gt_file", required = true)]
    gt_file: String,

    /// Number of nearest neighbors to search for (K).
    #[arg(short = 'K', default_value = "10")]
    recall_at: usize,

    /// Search list size(s) (L). Provide one or more values.
    #[arg(short = 'L', long = "search_list", num_args = 1.., required = true)]
    l_search: Vec<usize>,

    /// Number of search threads.
    #[arg(short = 'T', long = "num_threads", default_value = "1")]
    num_threads: usize,

    /// Optional prefix for writing result files.
    #[arg(long = "result_path")]
    result_path: Option<String>,

    /// Number of search repetitions for more stable QPS measurement.
    #[arg(long = "search_reps", default_value = "1")]
    search_reps: usize,

    /// Base vector JSONL labels file (for filtered search).
    #[arg(long = "data_labels")]
    data_labels: Option<String>,

    /// Query JSONL predicates file (for filtered search).
    #[arg(long = "query_labels")]
    query_labels: Option<String>,

    /// Filter strategy to use for filtered search.
    /// "none" = post-filter only, "beta" = BetaFilter, "multihop" = MultihopSearch.
    #[arg(long = "filter_strategy", default_value = "none")]
    filter_strategy: FilterStrategy,

    /// Beta parameter for BetaFilter strategy. Must be in (0, 1].
    /// Lower values bias more aggressively toward matching vectors.
    #[arg(long = "beta", default_value = "0.5")]
    beta: f32,

    /// Path to base vectors in .bin or .fbin format (required for brute-force fallback).
    /// When --brute_force_threshold is set and a query's filter matches fewer points
    /// than the threshold, a brute-force scan over matching points is used instead
    /// of the graph index.
    #[arg(long = "data_path")]
    data_path: Option<String>,

    /// If a query's filter bitmap matches fewer than this many points, use brute-force
    /// search over those points instead of the graph index. Set to 0 to disable.
    /// Requires --data_path to be specified.
    #[arg(long = "brute_force_threshold", default_value = "0")]
    brute_force_threshold: usize,
}

fn load_data<T: Copy + bytemuck::Pod>(path: &str) -> Result<Matrix<T>> {
    let data =
        diskann_utils::io::read_bin::<T>(&mut FileStorageProvider.open_reader(path)?)?;
    Ok(data)
}

/// Load groundtruth: [num_queries: u32, dim: u32, then num_queries * dim u32 IDs,
/// optionally followed by num_queries * dim f32 distances].
/// Returns (ids, optional distances, num_queries, dim).
fn load_groundtruth(path: &str) -> Result<(Vec<u32>, Option<Vec<f32>>, usize, usize)> {
    let provider = FileStorageProvider;
    let mut file = provider
        .open_reader(path)
        .with_context(|| format!("Opening ground truth file: {}", path))?;

    let actual_file_size = std::fs::metadata(path)?.len() as usize;

    let (num_queries, dim) = {
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)?;
        let nq = u32::from_le_bytes(buf) as usize;
        file.read_exact(&mut buf)?;
        let d = u32::from_le_bytes(buf) as usize;
        (nq, d)
    };

    let mut gt = vec![0u32; num_queries * dim];
    let gt_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut gt);
    file.read_exact(gt_bytes)?;

    // Check if distances are appended after IDs
    let expected_with_dists = 2 * 4 + num_queries * dim * 4 + num_queries * dim * 4;
    let gt_dists = if actual_file_size == expected_with_dists {
        let mut dists = vec![0f32; num_queries * dim];
        let dist_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut dists);
        file.read_exact(dist_bytes)?;
        Some(dists)
    } else {
        None
    };

    Ok((gt, gt_dists, num_queries, dim))
}

/// Compute K-recall@N with tie-aware scoring.
///
/// When `gt_dists` is provided, points tied at the K-th GT distance boundary
/// are treated as interchangeable: the algorithm credits up to the number of
/// "slots" available in that tie band, matching the C++ `calculate_recall`.
fn compute_recall(
    num_queries: usize,
    gt: &[u32],
    gt_dists: Option<&[f32]>,
    gt_dim: usize,
    results: &[u32],
    results_dim: usize,
    recall_k: usize,
) -> f64 {
    let k = recall_k.min(gt_dim);
    let n = recall_k.min(results_dim);
    let mut total = 0.0f64;

    for q in 0..num_queries {
        let gt_vec = &gt[q * gt_dim..q * gt_dim + gt_dim];
        let res_vec = &results[q * results_dim..q * results_dim + n];

        // Count valid GT entries (skip u32::MAX sentinels)
        let mut num_pts_in_gt = 0usize;
        while num_pts_in_gt < k && gt_vec[num_pts_in_gt] != u32::MAX {
            num_pts_in_gt += 1;
        }

        if num_pts_in_gt == 0 {
            // No feasible base point exists → 100% recall by convention
            total += 1.0;
            continue;
        }

        let res_set: HashSet<u32> = res_vec.iter().copied().collect();

        let cur_recall = if num_pts_in_gt < k || gt_dists.is_none() {
            // No tiebreaking needed (or possible)
            let mut count = 0usize;
            for j in 0..num_pts_in_gt {
                if res_set.contains(&gt_vec[j]) {
                    count += 1;
                }
            }
            count
        } else {
            // Tie-aware scoring
            let gt_dist_vec = &gt_dists.unwrap()[q * gt_dim..q * gt_dim + gt_dim];
            let boundary_dist = gt_dist_vec[k - 1];

            // Find tiebreaker_start: first index with distance == boundary_dist
            let mut tiebreaker_start = k - 1;
            while tiebreaker_start >= 1 && gt_dist_vec[tiebreaker_start - 1] == boundary_dist {
                tiebreaker_start -= 1;
            }

            // Find tiebreaker_end: last index (exclusive) with distance == boundary_dist
            let mut tiebreaker_end = k;
            while tiebreaker_end < gt_dim && gt_dist_vec[tiebreaker_end] == boundary_dist {
                tiebreaker_end += 1;
            }

            // Non-tied part: count normally
            let mut count = 0usize;
            for j in 0..tiebreaker_start {
                if res_set.contains(&gt_vec[j]) {
                    count += 1;
                }
            }

            // Tied part: credit at most (k - tiebreaker_start) matches
            let mut tie_recall = 0usize;
            for j in tiebreaker_start..tiebreaker_end {
                if res_set.contains(&gt_vec[j]) {
                    tie_recall += 1;
                }
            }
            count + tie_recall.min(k - tiebreaker_start)
        };

        total += cur_recall as f64 / num_pts_in_gt as f64;
    }

    total / num_queries as f64 * 100.0
}

/// Build an IndexConfiguration from the saved graph metadata.
fn load_config(index_path: &str, metric: Metric) -> Result<IndexConfiguration> {
    let sp = FileStorageProvider;
    let num_frozen_pts = save_and_load_helpers::get_graph_num_frozen_points(&sp, index_path)?;
    let max_observed_degree =
        save_and_load_helpers::get_graph_max_observed_degree(&sp, index_path)?;
    let metadata = load_metadata_from_file(&sp, &format!("{}.data", index_path))?;

    let config = config::Builder::new(
        max_observed_degree.into_usize(),
        config::MaxDegree::same(),
        1, // L is set per-query; 1 is placeholder
        metric.into(),
    )
    .build()
    .context("Failed to build config from saved index")?;

    Ok(IndexConfiguration::new(
        metric,
        metadata.ndims(),
        metadata.npoints(),
        num_frozen_pts,
        1,
        config,
    ))
}

// ============================================================================
// Label JSONL parsing. Both filter strategies (Bitmap64 inverted index and
// inline-beta encoded attribute store) stream the base label file once at
// startup and share this byte-level parser.
// ============================================================================

/// Parse a JSONL label line of the form
///   `{"doc_id": N, "<field>": true, ..., "<field>": false, ...}`
/// Calls `on_true_field(name)` for each top-level key whose value is JSON
/// `true`. Returns the doc_id on success, or `None` if the line is malformed.
///
/// Non-bool values (numbers, strings) are scanned past but not indexed,
/// matching the convention of the bitmap inverted-index builder.
fn parse_label_line<F>(line: &[u8], mut on_true_field: F) -> Option<u32>
where
    F: FnMut(&str),
{
    let mut i = 0usize;
    let n = line.len();
    let mut doc_id: Option<u32> = None;

    while i < n {
        let q1 = match line[i..].iter().position(|&b| b == b'"') {
            Some(p) => p,
            None => break,
        };
        i += q1 + 1;
        if i >= n {
            break;
        }
        let key_start = i;
        let q2 = match line[i..].iter().position(|&b| b == b'"') {
            Some(p) => p,
            None => break,
        };
        i += q2;
        let key_end = i;
        i += 1;

        let key = match std::str::from_utf8(&line[key_start..key_end]) {
            Ok(s) => s,
            Err(_) => return doc_id,
        };

        while i < n && line[i] != b':' {
            i += 1;
        }
        if i >= n {
            break;
        }
        i += 1;
        while i < n && (line[i] == b' ' || line[i] == b'\t') {
            i += 1;
        }
        if i >= n {
            break;
        }

        let v = line[i];
        if v == b't' {
            if i + 4 > n || &line[i..i + 4] != b"true" {
                return doc_id;
            }
            i += 4;
            if key != "doc_id" {
                on_true_field(key);
            }
        } else if v == b'f' {
            if i + 5 > n || &line[i..i + 5] != b"false" {
                return doc_id;
            }
            i += 5;
        } else if v == b'"' {
            i += 1;
            while i < n && line[i] != b'"' {
                i += 1;
            }
            if i < n {
                i += 1;
            }
        } else if v.is_ascii_digit() || v == b'-' {
            let start = i;
            while i < n
                && (line[i].is_ascii_digit()
                    || matches!(line[i], b'.' | b'e' | b'E' | b'+' | b'-'))
            {
                i += 1;
            }
            if key == "doc_id" {
                doc_id = std::str::from_utf8(&line[start..i]).ok()?.parse().ok();
            }
        } else {
            return doc_id;
        }
    }

    doc_id
}

/// Fixed-size dense bitmap backed by `Vec<u64>`.
///
/// Designed for fast per-query filter computation: all hot-path operations
/// (`union_with`, `intersect_with`, `complement_in_place`, `copy_from`, `clear`)
/// are in-place and allocation-free, working on 64-bit words to enable
/// auto-vectorization (vs `bit_set::BitSet`'s 32-bit words).
#[derive(Clone, Debug)]
struct Bitmap64 {
    words: Vec<u64>,
    universe: usize,
}

impl Bitmap64 {
    /// New all-zero bitmap with capacity for `universe` bits.
    fn with_universe(universe: usize) -> Self {
        let n_words = universe.div_ceil(64);
        Self {
            words: vec![0u64; n_words],
            universe,
        }
    }

    /// Empty bitmap (zero universe). Used as a placeholder; resize before use.
    fn empty() -> Self {
        Self {
            words: Vec::new(),
            universe: 0,
        }
    }

    /// Mask off trailing bits beyond `universe` in the last word.
    #[inline]
    fn mask_tail(&mut self) {
        if self.words.is_empty() {
            return;
        }
        let extra = self.words.len() * 64 - self.universe;
        if extra > 0 {
            let last = self.words.len() - 1;
            self.words[last] &= (!0u64) >> extra;
        }
    }

    /// Insert a bit, growing the underlying storage as needed.
    /// Used during inverted-index construction when the universe size is not
    /// yet known. After construction, call `resize_universe` to normalize.
    fn insert_grow(&mut self, idx: usize) {
        let word_idx = idx >> 6;
        if word_idx >= self.words.len() {
            self.words.resize(word_idx + 1, 0);
        }
        self.words[word_idx] |= 1u64 << (idx & 63);
        if idx + 1 > self.universe {
            self.universe = idx + 1;
        }
    }

    /// Pad/truncate to the given universe size. Trailing bits beyond universe
    /// are cleared.
    fn resize_universe(&mut self, universe: usize) {
        let n_words = universe.div_ceil(64);
        self.words.resize(n_words, 0);
        self.universe = universe;
        self.mask_tail();
    }

    #[inline]
    fn contains(&self, idx: usize) -> bool {
        let word_idx = idx >> 6;
        word_idx < self.words.len() && (self.words[word_idx] >> (idx & 63)) & 1 != 0
    }

    /// In-place `self |= other`. Both bitmaps must have the same universe.
    fn union_with(&mut self, other: &Self) {
        debug_assert_eq!(self.words.len(), other.words.len());
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            *a |= *b;
        }
    }

    /// In-place `self &= other`. Both bitmaps must have the same universe.
    fn intersect_with(&mut self, other: &Self) {
        debug_assert_eq!(self.words.len(), other.words.len());
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            *a &= *b;
        }
    }

    /// In-place `self = !self` (within universe).
    fn complement_in_place(&mut self) {
        for w in &mut self.words {
            *w = !*w;
        }
        self.mask_tail();
    }

    /// In-place `self = other`. No allocation when sizes match.
    fn copy_from(&mut self, other: &Self) {
        debug_assert_eq!(self.words.len(), other.words.len());
        self.words.copy_from_slice(&other.words);
        self.universe = other.universe;
    }

    /// In-place `self = all zeros`.
    fn clear_bits(&mut self) {
        for w in &mut self.words {
            *w = 0;
        }
    }

    /// In-place `self = all ones` (within universe).
    fn fill_ones(&mut self) {
        for w in &mut self.words {
            *w = !0u64;
        }
        self.mask_tail();
    }

    /// Number of set bits (popcount).
    fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Iterator over set bit indices.
    fn iter(&self) -> Bitmap64Iter<'_> {
        Bitmap64Iter::new(&self.words)
    }
}

struct Bitmap64Iter<'a> {
    words: &'a [u64],
    word_idx: usize,
    remaining: u64,
}

impl<'a> Bitmap64Iter<'a> {
    fn new(words: &'a [u64]) -> Self {
        Self {
            words,
            word_idx: 0,
            remaining: words.first().copied().unwrap_or(0),
        }
    }
}

impl Iterator for Bitmap64Iter<'_> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        while self.remaining == 0 {
            self.word_idx += 1;
            if self.word_idx >= self.words.len() {
                return None;
            }
            self.remaining = self.words[self.word_idx];
        }
        let tz = self.remaining.trailing_zeros() as usize;
        self.remaining &= self.remaining - 1;
        Some(self.word_idx * 64 + tz)
    }
}

/// Compute per-query filter bitmaps from JSONL label files using an inverted index.
///
/// Instead of evaluating each query's AST against every base document (O(Q*N) JSON evals),
/// this builds an inverted index from base labels (field_name → BitSet of doc_ids) and
/// then evaluates each query by combining posting lists: OR = union, AND = intersection.
///
/// This is orders of magnitude faster for large datasets (seconds vs hours for 10K queries × 1M base).
fn compute_filter_bitmaps(
    data_labels_path: &str,
    query_labels_path: &str,
) -> Result<(Vec<Bitmap64>, Vec<usize>)> {
    use diskann_label_filter::read_and_parse_queries;
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    // Phase 1: Stream the base label JSONL once and build a Vec<Bitmap64>
    // keyed by attr_id, plus a parallel attr-name table. Each line's true
    // fields are interned into a small id_buf during parsing; the doc_id is
    // only known at end-of-line so we apply the buffered ids in a second tiny
    // pass over id_buf.
    println!("  Building inverted index from base labels ...");
    let inv_start = Instant::now();

    let src_file = File::open(data_labels_path)
        .with_context(|| format!("Opening base labels file: {}", data_labels_path))?;
    let mut reader = BufReader::with_capacity(1 << 22, src_file);
    let mut line_buf: Vec<u8> = Vec::with_capacity(512);

    let mut attr_ids: HashMap<String, u32> = HashMap::with_capacity(1 << 16);
    let mut attr_names: Vec<String> = Vec::new();
    let mut attr_bitmaps: Vec<Bitmap64> = Vec::new();
    let mut id_buf: Vec<u32> = Vec::with_capacity(16);

    let mut max_doc_id: u32 = 0;
    let mut num_docs: u32 = 0;
    let mut last_print = inv_start;
    loop {
        line_buf.clear();
        let n = reader.read_until(b'\n', &mut line_buf)?;
        if n == 0 {
            break;
        }
        let line = if line_buf.last() == Some(&b'\n') {
            &line_buf[..line_buf.len() - 1]
        } else {
            &line_buf[..]
        };

        id_buf.clear();
        let doc_id = match parse_label_line(line, |field: &str| {
            if let Some(&id) = attr_ids.get(field) {
                id_buf.push(id);
            } else {
                let id = attr_names.len() as u32;
                attr_names.push(field.to_string());
                attr_ids.insert(field.to_string(), id);
                attr_bitmaps.push(Bitmap64::empty());
                id_buf.push(id);
            }
        }) {
            Some(d) => d,
            None => continue,
        };

        for &aid in &id_buf {
            attr_bitmaps[aid as usize].insert_grow(doc_id as usize);
        }
        if doc_id > max_doc_id {
            max_doc_id = doc_id;
        }
        num_docs += 1;
        if num_docs % 500_000 == 0 {
            let now = Instant::now();
            if (now - last_print).as_secs_f64() > 1.0 {
                eprint!(
                    "\r  Building inverted index: {} docs, {} distinct fields, {:.1}s",
                    num_docs,
                    attr_names.len(),
                    inv_start.elapsed().as_secs_f64()
                );
                last_print = now;
            }
        }
    }
    eprintln!(
        "\r  Building inverted index: {} docs, {} distinct fields, {:.1}s",
        num_docs,
        attr_names.len(),
        inv_start.elapsed().as_secs_f64()
    );

    let num_base = (max_doc_id as usize) + 1;
    for bm in &mut attr_bitmaps {
        bm.resize_universe(num_base);
    }

    // Transpose attr_id -> field name keying so the AST evaluator (which
    // looks up by `field: String`) keeps working unchanged.
    let mut inverted_index: HashMap<String, Bitmap64> = HashMap::with_capacity(attr_names.len());
    for (name, bm) in attr_names.drain(..).zip(attr_bitmaps.drain(..)) {
        inverted_index.insert(name, bm);
    }

    let inv_elapsed = inv_start.elapsed();
    eprintln!(
        "\r  Inverted index built: {} base docs, {} distinct fields, {:.2}s",
        num_base,
        inverted_index.len(),
        inv_elapsed.as_secs_f64()
    );

    // Phase 2: Evaluate queries using the inverted index.
    // Each query AST is $and of $or of {field: {$eq: true}} leaf comparisons.
    println!("  Evaluating queries against inverted index ...");
    let eval_start = Instant::now();
    let parsed_queries = read_and_parse_queries(query_labels_path)?;

    // Pre-allocate result vector: one Bitmap64 per query at the full universe.
    // This is the dominant memory cost (10K * 5MB = 50GB for 40M-bit queries)
    // but it's allocated up front, not per query.
    let num_queries = parsed_queries.len();
    let mut bitmaps: Vec<Bitmap64> = (0..num_queries)
        .map(|_| Bitmap64::with_universe(num_base))
        .collect();

    // Scratch buffers for nested AND/OR/NOT evaluation. Each level of nesting
    // that has multiple subexpressions needs one scratch slot. Our queries are
    // AND-of-OR-of-terminals (depth 2), so 4 buffers is ample.
    let scratch_depth = 8;
    let mut scratch: Vec<Bitmap64> = (0..scratch_depth)
        .map(|_| Bitmap64::with_universe(num_base))
        .collect();

    // Build a precomputed all-zeros template for terminal-not-found case.
    // We reference `inverted_index` entries directly when present.
    for (i, (_query_id, query_expr)) in parsed_queries.iter().enumerate() {
        evaluate_ast_into(
            query_expr,
            &inverted_index,
            num_base,
            &mut bitmaps[i],
            &mut scratch[..],
        );
    }

    let eval_elapsed = eval_start.elapsed();
    println!(
        "  Query evaluation: {} queries in {:.2}s",
        bitmaps.len(),
        eval_elapsed.as_secs_f64()
    );

    let counts: Vec<usize> = bitmaps.iter().map(|bm| bm.count_ones()).collect();
    Ok((bitmaps, counts))
}

/// Evaluate an `ASTExpr` against an inverted index, writing the result into `out`.
///
/// Uses caller-provided scratch buffers to avoid all per-call allocation.
/// For an AST of nesting depth D with N-ary AND/OR nodes, at most D scratch
/// buffers are consumed (one per non-trivial AND/OR/NOT level).
fn evaluate_ast_into(
    expr: &diskann_label_filter::ASTExpr,
    index: &std::collections::HashMap<String, Bitmap64>,
    universe: usize,
    out: &mut Bitmap64,
    scratch: &mut [Bitmap64],
) {
    use diskann_label_filter::{ASTExpr, CompareOp};
    match expr {
        ASTExpr::And(subs) => {
            if subs.is_empty() {
                // Empty AND = vacuous truth = all bits set.
                out.fill_ones();
                return;
            }
            // First subexpression writes directly into `out` using the full scratch.
            evaluate_ast_into(&subs[0], index, universe, out, scratch);
            if subs.len() > 1 {
                // Subsequent subexpressions write into one slot, AND'd into `out`.
                let (slot, rest) = scratch
                    .split_first_mut()
                    .expect("not enough scratch buffers for AST nesting depth");
                for sub in &subs[1..] {
                    evaluate_ast_into(sub, index, universe, slot, rest);
                    out.intersect_with(slot);
                }
            }
        }
        ASTExpr::Or(subs) => {
            if subs.is_empty() {
                out.clear_bits();
                return;
            }
            evaluate_ast_into(&subs[0], index, universe, out, scratch);
            if subs.len() > 1 {
                let (slot, rest) = scratch
                    .split_first_mut()
                    .expect("not enough scratch buffers for AST nesting depth");
                for sub in &subs[1..] {
                    evaluate_ast_into(sub, index, universe, slot, rest);
                    out.union_with(slot);
                }
            }
        }
        ASTExpr::Not(sub) => {
            evaluate_ast_into(sub, index, universe, out, scratch);
            out.complement_in_place();
        }
        ASTExpr::Compare { field, op } => match op {
            CompareOp::Eq(val) if val.as_bool() == Some(true) => {
                if let Some(bm) = index.get(field) {
                    out.copy_from(bm);
                } else {
                    out.clear_bits();
                }
            }
            _ => {
                // Other operators / non-boolean Eq aren't used in our encoding.
                out.clear_bits();
            }
        },
    }
}

/// Brute-force KNN over a subset of points identified by a bitmap.
/// Returns the top-K point IDs sorted by distance.
fn brute_force_knn<T>(
    query: &[T],
    base_data: &Matrix<T>,
    bitmap: &Bitmap64,
    k: usize,
    metric: Metric,
) -> Vec<u32>
where
    T: DistanceProvider<T> + Copy + bytemuck::Pod + 'static,
{
    let dim = base_data.ncols();
    let dist_fn = T::distance_comparer(metric, Some(dim));

    // Use a max-heap of size K to maintain the top-K closest points.
    let mut heap: BinaryHeap<Neighbor<u32>> = BinaryHeap::new();

    for doc_id in bitmap.iter() {
        if doc_id >= base_data.nrows() {
            continue;
        }
        let vec = base_data.row(doc_id);
        let d = dist_fn.call(query, vec);
        let neighbor = Neighbor::new(doc_id as u32, d);
        if heap.len() < k {
            heap.push(neighbor);
        } else if let Some(worst) = heap.peek() {
            if d < worst.distance {
                heap.pop();
                heap.push(neighbor);
            }
        }
    }

    // Extract in sorted order (closest first)
    let mut results: Vec<Neighbor<u32>> = heap.into_vec();
    results.sort_unstable_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.iter().map(|n| n.id).collect()
}

/// Wrapper implementing QueryLabelProvider for a Bitmap64.
#[derive(Debug)]
struct BitmapLabelProvider(Bitmap64);

impl QueryLabelProvider<u32> for BitmapLabelProvider {
    fn is_match(&self, vec_id: u32) -> bool {
        self.0.contains(vec_id as usize)
    }
}

/// Inline label provider: evaluates an encoded filter against a RoaringAttributeStore
/// per node visit. No precomputed bitmaps — label lookups happen during graph traversal.
struct InlineLabelProvider {
    store: Arc<RoaringAttributeStore<u32>>,
    encoded_filter: Arc<EncodedFilterExpr>,
}

impl std::fmt::Debug for InlineLabelProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InlineLabelProvider").finish()
    }
}

impl QueryLabelProvider<u32> for InlineLabelProvider {
    fn is_match(&self, vec_id: u32) -> bool {
        self.store.matches_filter(&vec_id, &self.encoded_filter)
    }
}

/// Label provider that matches nothing. Used for queries whose predicate references
/// attributes that don't exist in the base dataset and would otherwise be empty.
#[derive(Debug)]
struct AlwaysFalseLabelProvider;

impl QueryLabelProvider<u32> for AlwaysFalseLabelProvider {
    fn is_match(&self, _vec_id: u32) -> bool {
        false
    }
}

/// Result of cleaning a query AST against the encoder's attribute map.
enum CleanedAst {
    /// Expression after pruning unknown leaves; safe to encode.
    Expr(ASTExpr),
    /// The whole expression is unsatisfiable (no document can match).
    Empty,
}

/// Walk a parsed query AST and prune `field == value` leaves whose attribute is not
/// present in `encoder`. Such leaves can never match any base point, so:
///   * inside an OR, they are dropped;
///   * inside an AND, they make the whole AND unsatisfiable.
/// Other compare operators (Ne / Lt / Lte / Gt / Gte) are left untouched so the
/// downstream encoder/evaluator handles them as before.
fn clean_ast(expr: &ASTExpr, encoder: &AttributeEncoder) -> CleanedAst {
    match expr {
        ASTExpr::Compare {
            field,
            op: CompareOp::Eq(value),
        } => match Attribute::from_json_value(field, value) {
            Ok(attr) => {
                if encoder.get(&attr).is_some() {
                    CleanedAst::Expr(expr.clone())
                } else {
                    CleanedAst::Empty
                }
            }
            Err(_) => CleanedAst::Empty,
        },
        ASTExpr::Compare { .. } => CleanedAst::Expr(expr.clone()),
        ASTExpr::And(children) => {
            let mut out: Vec<ASTExpr> = Vec::with_capacity(children.len());
            for c in children {
                match clean_ast(c, encoder) {
                    CleanedAst::Expr(e) => out.push(e),
                    CleanedAst::Empty => return CleanedAst::Empty,
                }
            }
            match out.len() {
                0 => CleanedAst::Empty,
                1 => CleanedAst::Expr(out.into_iter().next().unwrap()),
                _ => CleanedAst::Expr(ASTExpr::And(out)),
            }
        }
        ASTExpr::Or(children) => {
            let mut out: Vec<ASTExpr> = Vec::with_capacity(children.len());
            for c in children {
                if let CleanedAst::Expr(e) = clean_ast(c, encoder) {
                    out.push(e);
                }
            }
            match out.len() {
                0 => CleanedAst::Empty,
                1 => CleanedAst::Expr(out.into_iter().next().unwrap()),
                _ => CleanedAst::Expr(ASTExpr::Or(out)),
            }
        }
        ASTExpr::Not(inner) => match clean_ast(inner, encoder) {
            CleanedAst::Expr(e) => CleanedAst::Expr(ASTExpr::Not(Box::new(e))),
            // NOT(unsatisfiable) is a tautology, which we cannot easily express.
            // Conservatively pass the original through so the encoder can decide.
            CleanedAst::Empty => CleanedAst::Expr(expr.clone()),
        },
    }
}

type FPIndex<T> = DiskANNIndex<FullPrecisionProvider<T>>;

fn search_and_report<T>(args: &Args) -> Result<()>
where
    T: VectorRepr
        + DistanceProvider<T>
        + AsyncFriendly
        + Copy
        + bytemuck::Pod
        + std::fmt::Debug
        + Send
        + Sync
        + 'static,
    [T]: Send + Sync,
{
    if matches!(args.filter_strategy, FilterStrategy::InlineBeta) {
        return search_inline_beta::<T>(args);
    }

    // Validate args
    if !matches!(args.filter_strategy, FilterStrategy::None)
        && (args.data_labels.is_none() || args.query_labels.is_none())
    {
        anyhow::bail!(
            "--filter_strategy {:?} requires --data_labels and --query_labels",
            args.filter_strategy
        );
    }
    if matches!(args.filter_strategy, FilterStrategy::Beta | FilterStrategy::InlineBeta)
        && (args.beta <= 0.0 || args.beta > 1.0)
    {
        anyhow::bail!("--beta must be in (0.0, 1.0], got {}", args.beta);
    }
    if args.brute_force_threshold > 0 && args.data_path.is_none() {
        anyhow::bail!("--brute_force_threshold > 0 requires --data_path");
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(args.num_threads)
        .build()
        .context("Failed to create tokio runtime")?;

    // Load index
    println!("Loading index from {} ...", args.index_path_prefix);
    let index_config = load_config(&args.index_path_prefix, args.dist_fn)?;
    let index: FPIndex<T> = rt.block_on(async {
        FPIndex::<T>::load_with(
            &FileStorageProvider,
            &(args.index_path_prefix.as_str(), index_config),
        )
        .await
        .map_err(|e| anyhow::anyhow!("Failed to load index: {}", e))
    })?;
    let index = Arc::new(index);
    println!("Index loaded.");

    // Load queries
    println!("Loading queries from {} ...", args.query_file);
    let queries: Matrix<T> = load_data(&args.query_file)?;
    let num_queries = queries.nrows();
    println!(
        "Loaded {} queries of dimension {}",
        num_queries,
        queries.ncols()
    );

    // Load ground truth
    let (gt, gt_dists, _gt_nq, gt_dim, has_gt) = if args.gt_file == "null" {
        println!("No ground truth file provided. Recall will not be computed.");
        (vec![], None, 0, 0, false)
    } else {
        println!("Loading ground truth from {} ...", args.gt_file);
        let (gt, gt_dists, nq, dim) = load_groundtruth(&args.gt_file)?;
        println!("Ground truth: {} queries, {} neighbors each", nq, dim);
        if gt_dists.is_some() {
            println!("  (GT file includes distances — tie-aware recall enabled)");
        }
        assert_eq!(
            nq, num_queries,
            "Mismatch: ground truth has {} queries but query file has {}",
            nq, num_queries
        );
        (gt, gt_dists, nq, dim, true)
    };

    // Load filter bitmaps if provided
    let mut bitmap_time_ms: f64 = 0.0;
    let filter_bitmaps: Option<Vec<Bitmap64>> =
        match (&args.data_labels, &args.query_labels) {
            (Some(dl), Some(ql)) => {
                println!("Computing filter bitmaps ...");
                let bitmap_start = Instant::now();
                let (bitmaps, counts) = compute_filter_bitmaps(dl, ql)?;
                bitmap_time_ms = bitmap_start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(
                    bitmaps.len(),
                    num_queries,
                    "Mismatch: {} query predicates but {} queries",
                    bitmaps.len(),
                    num_queries
                );
                // Print bitmap statistics
                let total: usize = counts.iter().sum();
                let min = counts.iter().copied().min().unwrap_or(0);
                let max = counts.iter().copied().max().unwrap_or(0);
                let mean = total as f64 / counts.len() as f64;
                println!(
                    "  {} per-query filter bitmaps: min={}, max={}, mean={:.1} matching points",
                    bitmaps.len(),
                    min,
                    max,
                    mean
                );
                println!(
                    "  Bitmap computation time: {:.2} ms",
                    bitmap_time_ms
                );
                if args.brute_force_threshold > 0 {
                    let bf_count = counts
                        .iter()
                        .filter(|&&c| c < args.brute_force_threshold)
                        .count();
                    println!(
                        "  {} queries ({:.1}%) will use brute-force (threshold={})",
                        bf_count,
                        bf_count as f64 / num_queries as f64 * 100.0,
                        args.brute_force_threshold
                    );
                }
                Some(bitmaps)
            }
            (None, None) => None,
            _ => anyhow::bail!(
                "Both --data_labels and --query_labels must be provided together for filtered search"
            ),
        };

    // Load base data for brute-force fallback if needed
    let base_data: Option<Arc<Matrix<T>>> = if args.brute_force_threshold > 0 {
        match &args.data_path {
            Some(dp) => {
                println!("Loading base vectors from {} for brute-force fallback ...", dp);
                let data: Matrix<T> = load_data(dp)?;
                println!(
                    "  Loaded {} base vectors of dimension {}",
                    data.nrows(),
                    data.ncols()
                );
                Some(Arc::new(data))
            }
            None => None,
        }
    } else {
        None
    };

    let k = args.recall_at;
    let recall_header = format!("Recall@{}", k);
    let strategy_label = match args.filter_strategy {
        FilterStrategy::None => {
            if filter_bitmaps.is_some() {
                "post-filter"
            } else {
                "unfiltered"
            }
        }
        FilterStrategy::Beta => "beta-filter",
        FilterStrategy::Multihop => "multihop",
        FilterStrategy::InlineBeta => unreachable!("handled by search_inline_beta"),
    };

    // Print table header
    println!();
    println!("Strategy: {} ", strategy_label);
    if matches!(args.filter_strategy, FilterStrategy::Beta) {
        println!("Beta: {}", args.beta);
    }
    if bitmap_time_ms > 0.0 {
        println!("Bitmap precomputation: {:.2} ms", bitmap_time_ms);
    }
    println!();
    if args.brute_force_threshold > 0 {
        println!(
            "{:>8}  {:>10}  {:>12}  {:>14}  {:>10}  {:>8}",
            "Ls", "QPS", "Mean Lat(us)", "p99 Lat(us)", recall_header, "BF Qrys"
        );
        println!("{}", "=".repeat(74));
    } else {
        println!(
            "{:>8}  {:>10}  {:>12}  {:>14}  {:>10}",
            "Ls", "QPS", "Mean Lat(us)", "p99 Lat(us)", recall_header
        );
        println!("{}", "=".repeat(62));
    }

    let metric = args.dist_fn;

    for &l_search in &args.l_search {
        if l_search < k {
            eprintln!("Warning: L={} < K={}, skipping", l_search, k);
            continue;
        }

        // Storage for all results across reps (use first rep's IDs for recall)
        let mut all_ids: Vec<u32> = vec![0u32; num_queries * k];
        let mut query_latencies_us = Vec::with_capacity(num_queries * args.search_reps);
        let mut brute_force_count: usize = 0;

        // For post-filter mode (filter_strategy=none with bitmaps), search for L
        // candidates so we have more to filter from; for other modes, search for K.
        let is_postfilter_mode = matches!(args.filter_strategy, FilterStrategy::None)
            && filter_bitmaps.is_some();
        let search_k = if is_postfilter_mode { l_search } else { k };

        let graph_search = graph::search::Knn::new(search_k, l_search, None).map_err(|e| {
            anyhow::anyhow!("Invalid search params K={} L={}: {}", search_k, l_search, e)
        })?;

        for rep in 0..args.search_reps {
            let mut rep_latencies: Vec<f64> = Vec::with_capacity(num_queries);

            rt.block_on(async {
                let context = DefaultContext;

                for q in 0..num_queries {
                    let query = queries.row(q);
                    let q_start = Instant::now();

                    // Check if brute-force should be used for this query
                    let use_brute_force = args.brute_force_threshold > 0
                        && filter_bitmaps.is_some()
                        && filter_bitmaps.as_ref().unwrap()[q].count_ones()
                            < args.brute_force_threshold;

                    let mut ids = vec![0u32; search_k];

                    if use_brute_force {
                        // Brute-force search over matching points
                        let bitmap = &filter_bitmaps.as_ref().unwrap()[q];
                        let base = base_data.as_ref().unwrap();
                        let bf_ids = brute_force_knn(query, base, bitmap, k, metric);
                        let copy_len = bf_ids.len().min(k);
                        ids[..copy_len].copy_from_slice(&bf_ids[..copy_len]);
                        if rep == 0 {
                            brute_force_count += 1;
                        }
                    } else {
                        // Graph-based search
                        let mut dists = vec![0.0f32; search_k];
                        let mut output =
                            search_output_buffer::IdDistance::new(&mut ids, &mut dists);

                        match args.filter_strategy {
                            FilterStrategy::Beta
                                if filter_bitmaps.is_some() =>
                            {
                                let bitmap = &filter_bitmaps.as_ref().unwrap()[q];
                                let label_provider: Arc<dyn QueryLabelProvider<u32>> =
                                    Arc::new(BitmapLabelProvider(bitmap.clone()));
                                let beta_strategy = BetaFilter::new(
                                    FullPrecision,
                                    label_provider,
                                    args.beta,
                                );
                                let _stats = index
                                    .search(
                                        graph_search,
                                        &beta_strategy,
                                        &context,
                                        query,
                                        &mut output,
                                    )
                                    .await
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "BetaFilter search failed for query {}: {}",
                                            q,
                                            e
                                        )
                                    })?;
                            }
                            FilterStrategy::Multihop
                                if filter_bitmaps.is_some() =>
                            {
                                let bitmap = &filter_bitmaps.as_ref().unwrap()[q];
                                let label_provider: &dyn QueryLabelProvider<u32> =
                                    &BitmapLabelProvider(bitmap.clone());
                                let multihop = graph::search::MultihopSearch::new(
                                    graph_search,
                                    label_provider,
                                );
                                let _stats = index
                                    .search(
                                        multihop,
                                        &FullPrecision,
                                        &context,
                                        query,
                                        &mut output,
                                    )
                                    .await
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "MultihopSearch failed for query {}: {}",
                                            q,
                                            e
                                        )
                                    })?;
                            }
                            _ => {
                                // Unfiltered search (or filtered with post-filter only)
                                let _stats = index
                                    .search(
                                        graph_search,
                                        &FullPrecision,
                                        &context,
                                        query,
                                        &mut output,
                                    )
                                    .await
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "Search failed for query {}: {}",
                                            q,
                                            e
                                        )
                                    })?;
                            }
                        }
                    }

                    let q_elapsed = q_start.elapsed();
                    rep_latencies.push(q_elapsed.as_micros() as f64);

                    // Keep the IDs from the first repetition for recall computation
                    if rep == 0 {
                        if let Some(ref bitmaps) = filter_bitmaps {
                            // Post-filter: keep only matching IDs, take best K
                            let bm = &bitmaps[q];
                            let mut write_pos = 0;
                            for &id in ids.iter() {
                                if id == 0 && write_pos == 0 {
                                    // Skip uninitialized zero entries at the start
                                    // only if they don't match the bitmap
                                    if !bm.contains(0) {
                                        continue;
                                    }
                                }
                                if bm.contains(id as usize) {
                                    all_ids[q * k + write_pos] = id;
                                    write_pos += 1;
                                    if write_pos >= k {
                                        break;
                                    }
                                }
                            }
                        } else {
                            all_ids[q * k..q * k + k].copy_from_slice(&ids[..k]);
                        }
                    }
                }
                Ok::<(), anyhow::Error>(())
            })?;

            query_latencies_us.extend_from_slice(&rep_latencies);
        }

        // Compute stats
        let total_queries = query_latencies_us.len();
        let mean_lat = query_latencies_us.iter().sum::<f64>() / total_queries as f64;

        // p99 latency
        let mut sorted = query_latencies_us.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p99_idx = ((total_queries as f64 * 0.99) as usize).min(total_queries - 1);
        let p99_lat = sorted[p99_idx];

        // QPS: total queries / total wall time
        let total_time_s = query_latencies_us.iter().sum::<f64>() / 1_000_000.0;
        let qps = total_queries as f64 / total_time_s;

        // Recall
        let recall = if has_gt {
            compute_recall(num_queries, &gt, gt_dists.as_deref(), gt_dim, &all_ids, k, k)
        } else {
            f64::NAN
        };

        // Print results row
        if args.brute_force_threshold > 0 {
            if has_gt {
                println!(
                    "{:>8}  {:>10.2}  {:>12.2}  {:>14.2}  {:>10.2}  {:>8}",
                    l_search, qps, mean_lat, p99_lat, recall, brute_force_count
                );
            } else {
                println!(
                    "{:>8}  {:>10.2}  {:>12.2}  {:>14.2}  {:>10}  {:>8}",
                    l_search, qps, mean_lat, p99_lat, "N/A", brute_force_count
                );
            }
        } else if has_gt {
            println!(
                "{:>8}  {:>10.2}  {:>12.2}  {:>14.2}  {:>10.2}",
                l_search, qps, mean_lat, p99_lat, recall
            );
        } else {
            println!(
                "{:>8}  {:>10.2}  {:>12.2}  {:>14.2}  {:>10}",
                l_search, qps, mean_lat, p99_lat, "N/A"
            );
        }

        // Save results if requested
        if let Some(ref result_prefix) = args.result_path {
            let result_file = format!("{}_{}", result_prefix, l_search);
            write_results_bin(&result_file, &all_ids, num_queries, k)?;
        }
    }

    println!();
    Ok(())
}

/// Inline-beta search path: loads labels into RoaringAttributeStore and encodes
/// query predicates into EncodedFilterExpr. During graph traversal, each node's
/// labels are looked up via efficient roaring bitmap operations (no JSON parsing).
/// Label lookup cost is included in per-query latency.
fn search_inline_beta<T>(args: &Args) -> Result<()>
where
    T: VectorRepr
        + DistanceProvider<T>
        + AsyncFriendly
        + Copy
        + bytemuck::Pod
        + std::fmt::Debug
        + Send
        + Sync
        + 'static,
    [T]: Send + Sync,
{
    let dl = args.data_labels.as_ref().ok_or_else(|| {
        anyhow::anyhow!("--filter_strategy inline_beta requires --data_labels")
    })?;
    let ql = args.query_labels.as_ref().ok_or_else(|| {
        anyhow::anyhow!("--filter_strategy inline_beta requires --query_labels")
    })?;
    if args.beta <= 0.0 || args.beta > 1.0 {
        anyhow::bail!("--beta must be in (0.0, 1.0], got {}", args.beta);
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(args.num_threads)
        .build()
        .context("Failed to create tokio runtime")?;

    // Load index (plain FPIndex — no DocumentProvider wrapper needed)
    println!("Loading index from {} ...", args.index_path_prefix);
    let index_config = load_config(&args.index_path_prefix, args.dist_fn)?;
    let fp_index: FPIndex<T> = rt.block_on(async {
        FPIndex::<T>::load_with(
            &FileStorageProvider,
            &(args.index_path_prefix.as_str(), index_config),
        )
        .await
        .map_err(|e| anyhow::anyhow!("Failed to load index: {}", e))
    })?;
    let index = Arc::new(fp_index);
    println!("Index loaded.");

    // Stream the base label JSONL once, interning field names into attr_ids
    // and collecting per-doc (doc_id, attr_ids) pairs. The store then bulk-
    // builds each RoaringTreemap from a sorted iterator under one set of
    // locks.
    println!("Loading base labels from {} ...", dl);
    let label_start = Instant::now();

    let src_file = std::fs::File::open(dl)
        .map_err(|e| anyhow::anyhow!("Opening base labels file {}: {}", dl, e))?;
    let mut reader = std::io::BufReader::with_capacity(1 << 22, src_file);
    let mut line_buf: Vec<u8> = Vec::with_capacity(512);

    let mut attr_ids: std::collections::HashMap<String, u32> =
        std::collections::HashMap::with_capacity(1 << 16);
    let mut attr_names: Vec<String> = Vec::new();
    let mut docs_buf: Vec<(u32, Vec<u64>)> = Vec::with_capacity(1 << 25);
    let mut ids_u64: Vec<u64> = Vec::with_capacity(16);

    let mut num_docs: u32 = 0;
    let mut last_print = label_start;
    use std::io::BufRead;
    loop {
        line_buf.clear();
        let n = reader
            .read_until(b'\n', &mut line_buf)
            .map_err(|e| anyhow::anyhow!("Reading base labels: {}", e))?;
        if n == 0 {
            break;
        }
        let line = if line_buf.last() == Some(&b'\n') {
            &line_buf[..line_buf.len() - 1]
        } else {
            &line_buf[..]
        };

        ids_u64.clear();
        let doc_id = match parse_label_line(line, |field: &str| {
            if let Some(&id) = attr_ids.get(field) {
                ids_u64.push(id as u64);
            } else {
                let id = attr_names.len() as u32;
                attr_names.push(field.to_string());
                attr_ids.insert(field.to_string(), id);
                ids_u64.push(id as u64);
            }
        }) {
            Some(d) => d,
            None => continue,
        };

        if !ids_u64.is_empty() {
            docs_buf.push((doc_id, std::mem::take(&mut ids_u64)));
            // Restore capacity for the next iteration without re-allocating.
            ids_u64.reserve(16);
        }

        num_docs += 1;
        if num_docs % 500_000 == 0 {
            let now = Instant::now();
            if (now - last_print).as_secs_f64() > 1.0 {
                eprint!(
                    "\r  Streaming base labels: {} docs, {} distinct fields, {:.1}s",
                    num_docs,
                    attr_names.len(),
                    label_start.elapsed().as_secs_f64()
                );
                last_print = now;
            }
        }
    }
    eprintln!(
        "\r  Streaming base labels: {} docs, {} distinct fields, {:.1}s",
        num_docs,
        attr_names.len(),
        label_start.elapsed().as_secs_f64()
    );

    // Build attribute strings in attr_id order (matches the encoder convention).
    let attr_strings: Vec<Attribute> = attr_names
        .iter()
        .map(|name| Attribute::from_json_value(name, &serde_json::Value::Bool(true)))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("Failed to build Attribute from field name: {}", e))?;

    let num_docs_cache = num_docs as usize;
    let roaring_store = RoaringAttributeStore::<u32>::new();
    roaring_store
        .bulk_insert_encoded(&attr_strings, docs_buf.into_iter())
        .map_err(|e| anyhow::anyhow!("Failed to bulk-load attributes: {}", e))?;
    let label_load_ms = label_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  Loaded {} base documents into RoaringAttributeStore ({:.2} ms)",
        num_docs_cache, label_load_ms
    );
    let roaring_store = Arc::new(roaring_store);

    // Parse query predicates into ASTExpr, then encode into EncodedFilterExpr
    println!("Parsing query predicates from {} ...", ql);
    let query_exprs = read_and_parse_queries(ql)
        .map_err(|e| anyhow::anyhow!("Failed to parse query predicates: {}", e))?;
    println!("  Parsed {} query predicates", query_exprs.len());

    let attr_map = roaring_store.attribute_map();
    let mut empty_filter_count: usize = 0;
    let mut pruned_filter_count: usize = 0;
    let encoded_filters: Vec<Option<Arc<EncodedFilterExpr>>> = {
        let encoder_guard = attr_map.read().map_err(|_| {
            anyhow::anyhow!("Failed to acquire read lock on attribute encoder")
        })?;
        let mut out: Vec<Option<Arc<EncodedFilterExpr>>> =
            Vec::with_capacity(query_exprs.len());
        for (_qid, ast) in query_exprs.iter() {
            match clean_ast(ast, &encoder_guard) {
                CleanedAst::Empty => {
                    empty_filter_count += 1;
                    out.push(None);
                }
                CleanedAst::Expr(cleaned) => {
                    if &cleaned != ast {
                        pruned_filter_count += 1;
                    }
                    let ef = EncodedFilterExpr::new(&cleaned, attr_map.clone())
                        .map_err(|e| anyhow::anyhow!("Failed to encode filter: {}", e))?;
                    out.push(Some(Arc::new(ef)));
                }
            }
        }
        out
    };
    println!(
        "  Encoded {} query filters ({} pruned of unknown labels, {} unsatisfiable)",
        encoded_filters.len(),
        pruned_filter_count,
        empty_filter_count
    );

    // Load queries
    println!("Loading queries from {} ...", args.query_file);
    let queries: Matrix<T> = load_data(&args.query_file)?;
    let num_queries = queries.nrows();
    println!(
        "Loaded {} queries of dimension {}",
        num_queries,
        queries.ncols()
    );
    assert_eq!(
        query_exprs.len(),
        num_queries,
        "Mismatch: {} query predicates but {} queries",
        query_exprs.len(),
        num_queries
    );

    // Load ground truth
    let (gt, gt_dists, _gt_nq, gt_dim, has_gt) = if args.gt_file == "null" {
        println!("No ground truth file provided. Recall will not be computed.");
        (vec![], None, 0, 0, false)
    } else {
        println!("Loading ground truth from {} ...", args.gt_file);
        let (gt, gt_dists, nq, dim) = load_groundtruth(&args.gt_file)?;
        println!("Ground truth: {} queries, {} neighbors each", nq, dim);
        if gt_dists.is_some() {
            println!("  (GT file includes distances — tie-aware recall enabled)");
        }
        assert_eq!(
            nq, num_queries,
            "Mismatch: ground truth has {} queries but query file has {}",
            nq, num_queries
        );
        (gt, gt_dists, nq, dim, true)
    };

    let k = args.recall_at;
    let recall_header = format!("Recall@{}", k);

    // Print table header
    println!();
    println!("Strategy: inline-beta (roaring encoded lookups)");
    println!("Beta: {}", args.beta);
    println!("Label load time: {:.2} ms", label_load_ms);
    println!();
    println!(
        "{:>8}  {:>10}  {:>12}  {:>14}  {:>10}",
        "Ls", "QPS", "Mean Lat(us)", "p99 Lat(us)", recall_header
    );
    println!("{}", "=".repeat(62));

    for &l_search in &args.l_search {
        if l_search < k {
            eprintln!("Warning: L={} < K={}, skipping", l_search, k);
            continue;
        }

        let mut all_ids: Vec<u32> = vec![0u32; num_queries * k];
        let mut query_latencies_us = Vec::with_capacity(num_queries * args.search_reps);

        let graph_search = graph::search::Knn::new(k, l_search, None).map_err(|e| {
            anyhow::anyhow!("Invalid search params K={} L={}: {}", k, l_search, e)
        })?;

        for rep in 0..args.search_reps {
            let mut rep_latencies: Vec<f64> = Vec::with_capacity(num_queries);

            rt.block_on(async {
                let context = DefaultContext;

                for q in 0..num_queries {
                    let query_vec = queries.row(q);

                    // Build inline label provider for this query (cheap: just Arc clones).
                    // Queries whose predicate references attributes absent from the base
                    // dataset get an always-false provider so they match nothing.
                    let label_provider: Arc<dyn QueryLabelProvider<u32>> =
                        match &encoded_filters[q] {
                            Some(ef) => Arc::new(InlineLabelProvider {
                                store: roaring_store.clone(),
                                encoded_filter: ef.clone(),
                            }),
                            None => Arc::new(AlwaysFalseLabelProvider),
                        };
                    let beta_strategy =
                        BetaFilter::new(FullPrecision, label_provider, args.beta);

                    let q_start = Instant::now();

                    let mut ids = vec![0u32; k];
                    let mut dists = vec![0.0f32; k];
                    let mut output =
                        search_output_buffer::IdDistance::new(&mut ids, &mut dists);

                    let _stats = index
                        .search(
                            graph_search,
                            &beta_strategy,
                            &context,
                            query_vec,
                            &mut output,
                        )
                        .await
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "InlineBeta search failed for query {}: {}",
                                q,
                                e
                            )
                        })?;

                    let q_elapsed = q_start.elapsed();
                    rep_latencies.push(q_elapsed.as_micros() as f64);

                    if rep == 0 {
                        all_ids[q * k..q * k + k].copy_from_slice(&ids[..k]);
                    }
                }
                Ok::<(), anyhow::Error>(())
            })?;

            query_latencies_us.extend_from_slice(&rep_latencies);
        }

        // Compute stats
        let total_queries = query_latencies_us.len();
        let mean_lat = query_latencies_us.iter().sum::<f64>() / total_queries as f64;

        let mut sorted = query_latencies_us.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p99_idx = ((total_queries as f64 * 0.99) as usize).min(total_queries - 1);
        let p99_lat = sorted[p99_idx];

        let total_time_s = query_latencies_us.iter().sum::<f64>() / 1_000_000.0;
        let qps = total_queries as f64 / total_time_s;

        let recall = if has_gt {
            compute_recall(num_queries, &gt, gt_dists.as_deref(), gt_dim, &all_ids, k, k)
        } else {
            f64::NAN
        };

        if has_gt {
            println!(
                "{:>8}  {:>10.2}  {:>12.2}  {:>14.2}  {:>10.2}",
                l_search, qps, mean_lat, p99_lat, recall
            );
        } else {
            println!(
                "{:>8}  {:>10.2}  {:>12.2}  {:>14.2}  {:>10}",
                l_search, qps, mean_lat, p99_lat, "N/A"
            );
        }

        if let Some(ref result_prefix) = args.result_path {
            let result_file = format!("{}_{}", result_prefix, l_search);
            write_results_bin(&result_file, &all_ids, num_queries, k)?;
        }
    }

    println!();
    Ok(())
}

/// Write results in the standard DiskANN binary format:
/// [num_queries: u32] [k: u32] [num_queries * k u32 IDs]
fn write_results_bin(path: &str, ids: &[u32], num_queries: usize, k: usize) -> Result<()> {
    use std::io::Write;
    let mut file =
        std::fs::File::create(path).with_context(|| format!("Creating result file: {}", path))?;
    file.write_all(&(num_queries as u32).to_le_bytes())?;
    file.write_all(&(k as u32).to_le_bytes())?;
    let id_bytes: &[u8] = bytemuck::cast_slice(ids);
    file.write_all(id_bytes)?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Search parameters:");
    println!("  data_type:        {:?}", args.data_type);
    println!("  dist_fn:          {}", args.dist_fn);
    println!("  index_path:       {}", args.index_path_prefix);
    println!("  query_file:       {}", args.query_file);
    println!("  gt_file:          {}", args.gt_file);
    println!("  K:                {}", args.recall_at);
    println!("  L values:         {:?}", args.l_search);
    println!("  threads:          {}", args.num_threads);
    println!("  search_reps:      {}", args.search_reps);
    println!("  filter_strategy:  {:?}", args.filter_strategy);
    if matches!(args.filter_strategy, FilterStrategy::Beta | FilterStrategy::InlineBeta) {
        println!("  beta:             {}", args.beta);
    }
    if let Some(ref p) = args.data_labels {
        println!("  data_labels:      {}", p);
    }
    if let Some(ref p) = args.query_labels {
        println!("  query_labels:     {}", p);
    }
    if args.brute_force_threshold > 0 {
        println!("  bf_threshold:     {}", args.brute_force_threshold);
        if let Some(ref p) = args.data_path {
            println!("  data_path:        {}", p);
        }
    }
    println!();

    match args.data_type {
        DataType::Float => search_and_report::<f32>(&args),
        DataType::Float16 => search_and_report::<f16>(&args),
        DataType::Uint8 => search_and_report::<u8>(&args),
        DataType::Int8 => search_and_report::<i8>(&args),
    }
}
