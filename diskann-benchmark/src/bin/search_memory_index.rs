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
use bit_set::BitSet;
use clap::{Parser, ValueEnum};
use diskann::graph::index::QueryLabelProvider;
use diskann::graph::search_output_buffer;
use diskann::graph::{self, config, DiskANNIndex};
use diskann::neighbor::Neighbor;
use diskann::provider::DefaultContext;
use diskann::utils::{IntoUsize, VectorRepr};
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

/// Load groundtruth: [num_queries: u32, dim: u32, then num_queries * dim u32 IDs]
fn load_groundtruth(path: &str) -> Result<(Vec<u32>, usize, usize)> {
    let provider = FileStorageProvider;
    let mut file = provider
        .open_reader(path)
        .with_context(|| format!("Opening ground truth file: {}", path))?;

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

    Ok((gt, num_queries, dim))
}

/// Compute K-recall@N: fraction of top-K ground truth neighbors found in top-N results.
fn compute_recall(
    num_queries: usize,
    gt: &[u32],
    gt_dim: usize,
    results: &[u32],
    results_dim: usize,
    recall_k: usize,
) -> f64 {
    let k = recall_k.min(gt_dim);
    let n = recall_k.min(results_dim);
    let mut total = 0usize;
    for q in 0..num_queries {
        let gt_set: HashSet<u32> = gt[q * gt_dim..q * gt_dim + k].iter().copied().collect();
        let res_set: HashSet<u32> = results[q * results_dim..q * results_dim + n]
            .iter()
            .copied()
            .collect();
        total += gt_set.intersection(&res_set).count();
    }
    total as f64 / (num_queries as f64 * k as f64) * 100.0
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

/// Compute per-query filter bitmaps from JSONL label files.
/// Returns (bitmaps, per_query_match_counts).
fn compute_filter_bitmaps(
    data_labels_path: &str,
    query_labels_path: &str,
) -> Result<(Vec<BitSet>, Vec<usize>)> {
    use diskann_label_filter::{eval_query_expr, read_and_parse_queries, read_baselabels};

    let base_labels = read_baselabels(data_labels_path)?;
    let parsed_queries = read_and_parse_queries(query_labels_path)?;

    let bitmaps: Vec<BitSet> = parsed_queries
        .iter()
        .map(|(_query_id, query_expr)| {
            let mut bitmap = BitSet::new();
            for base_label in base_labels.iter() {
                if eval_query_expr(query_expr, &base_label.label) {
                    bitmap.insert(base_label.doc_id);
                }
            }
            bitmap
        })
        .collect();

    let counts: Vec<usize> = bitmaps.iter().map(|bm| bm.len()).collect();
    Ok((bitmaps, counts))
}

/// Brute-force KNN over a subset of points identified by a bitmap.
/// Returns the top-K point IDs sorted by distance.
fn brute_force_knn<T>(
    query: &[T],
    base_data: &Matrix<T>,
    bitmap: &BitSet,
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

/// Wrapper implementing QueryLabelProvider for a BitSet.
#[derive(Debug)]
struct BitmapLabelProvider(BitSet);

impl QueryLabelProvider<u32> for BitmapLabelProvider {
    fn is_match(&self, vec_id: u32) -> bool {
        self.0.contains(vec_id as usize)
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
    // Validate args
    if !matches!(args.filter_strategy, FilterStrategy::None)
        && (args.data_labels.is_none() || args.query_labels.is_none())
    {
        anyhow::bail!(
            "--filter_strategy {:?} requires --data_labels and --query_labels",
            args.filter_strategy
        );
    }
    if matches!(args.filter_strategy, FilterStrategy::Beta)
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
    let (gt, _gt_nq, gt_dim, has_gt) = if args.gt_file == "null" {
        println!("No ground truth file provided. Recall will not be computed.");
        (vec![], 0, 0, false)
    } else {
        println!("Loading ground truth from {} ...", args.gt_file);
        let (gt, nq, dim) = load_groundtruth(&args.gt_file)?;
        println!("Ground truth: {} queries, {} neighbors each", nq, dim);
        assert_eq!(
            nq, num_queries,
            "Mismatch: ground truth has {} queries but query file has {}",
            nq, num_queries
        );
        (gt, nq, dim, true)
    };

    // Load filter bitmaps if provided
    let mut bitmap_time_ms: f64 = 0.0;
    let filter_bitmaps: Option<Vec<BitSet>> =
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
                        && filter_bitmaps.as_ref().unwrap()[q].len()
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
            compute_recall(num_queries, &gt, gt_dim, &all_ids, k, k)
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
    if matches!(args.filter_strategy, FilterStrategy::Beta) {
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
