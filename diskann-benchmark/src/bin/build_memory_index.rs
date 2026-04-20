/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! CLI tool to build an in-memory DiskANN index and save it to disk.
//!
//! Mimics the old C++ `apps/build_memory_index` command.
//!
//! # Example
//!
//! ```bash
//! build_memory_index \
//!   --data_type float --dist_fn l2 \
//!   --data_path data/sift/sift_base.fbin \
//!   --index_path_prefix data/sift/index_R32_L50 \
//!   -R 32 -L 50 --alpha 1.2 -T 8
//! ```

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use diskann::graph::{self, config, StartPointStrategy};
use diskann::provider::DefaultContext;
use diskann::utils::VectorRepr;
use diskann::ANNError;
use diskann_providers::index::diskann_async;
use diskann_providers::model::graph::provider::async_::common::{FullPrecision, NoDeletes};
use diskann_providers::model::graph::provider::async_::inmem::{
    DefaultProviderParameters, SetStartPoints,
};
use diskann_providers::storage::{
    AsyncIndexMetadata, FileStorageProvider, SaveWith, StorageReadProvider,
};
use diskann_utils::future::AsyncFriendly;
use diskann_utils::sampling::WithApproximateNorm;
use diskann_utils::views::{Matrix, MatrixView};
use diskann_vector::distance::Metric;
use half::f16;

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

#[derive(Debug, Parser)]
#[command(name = "build_memory_index", about = "Build an in-memory DiskANN index and save it to disk")]
struct Args {
    /// Data type of the vectors.
    #[arg(long = "data_type", default_value = "float")]
    data_type: DataType,

    /// Distance function: l2, mips, or cosine.
    #[arg(long = "dist_fn", default_value = "l2")]
    dist_fn: Metric,

    /// Path to the input vectors in .bin or .fbin format.
    #[arg(long = "data_path", required = true)]
    data_path: String,

    /// Output path prefix for the saved index.
    #[arg(long = "index_path_prefix", required = true)]
    index_path_prefix: String,

    /// Max graph degree (R).
    #[arg(short = 'R', long = "max_degree", default_value = "64")]
    max_degree: usize,

    /// Build search list size (L).
    #[arg(short = 'L', long = "Lbuild", default_value = "100")]
    l_build: usize,

    /// Graph diameter parameter (alpha). Typical values: 1.0 to 1.5.
    #[arg(long = "alpha", default_value = "1.2")]
    alpha: f32,

    /// Number of build threads.
    #[arg(short = 'T', long = "num_threads", default_value = "0")]
    num_threads: usize,
}

fn load_data<T: Copy + bytemuck::Pod>(path: &str) -> Result<Matrix<T>> {
    let data = diskann_utils::io::read_bin::<T>(
        &mut FileStorageProvider.open_reader(path)?,
    )?;
    Ok(data)
}

fn set_start_points<DP, T>(
    provider: &DP,
    data: MatrixView<'_, T>,
) -> Result<()>
where
    DP: SetStartPoints<[T]>,
    T: graph::SampleableForStart + WithApproximateNorm + AsyncFriendly,
{
    let start_points = StartPointStrategy::Medoid
        .compute(data)
        .map_err(|e| {
            ANNError::new(
                diskann::ANNErrorKind::DiskANN(diskann::error::DiskANNError::StartPointComputeError),
                e,
            )
        })?;
    provider.set_start_points(start_points.row_iter())?;
    Ok(())
}

/// Build and save the index for a given element type.
fn build_and_save<T>(args: &Args) -> Result<()>
where
    T: VectorRepr
        + graph::SampleableForStart
        + WithApproximateNorm
        + AsyncFriendly
        + Copy
        + bytemuck::Pod
        + std::fmt::Debug
        + Send
        + Sync
        + 'static,
    [T]: Send + Sync,
{
    let total_start = Instant::now();

    // Load data
    println!("Loading data from {} ...", args.data_path);
    let data: Arc<Matrix<T>> = Arc::new(load_data(&args.data_path)?);
    let npoints = data.nrows();
    let ndims = data.ncols();
    println!("Loaded {} vectors of dimension {}", npoints, ndims);

    // Build config
    let metric: Metric = args.dist_fn;
    let exact_max_degree = (args.max_degree as f32 * 1.3) as usize;
    let config = config::Builder::new_with(
        args.max_degree,
        config::MaxDegree::new(exact_max_degree),
        args.l_build,
        metric.into(),
        |builder| {
            builder.alpha(args.alpha).backedge_ratio(1.0);
        },
    )
    .build()
    .context("Failed to build index configuration")?;

    let params = DefaultProviderParameters {
        max_points: npoints,
        frozen_points: std::num::NonZero::new(StartPointStrategy::Medoid.count()).unwrap(),
        metric,
        dim: ndims,
        max_degree: exact_max_degree as u32,
        prefetch_lookahead: None,
        prefetch_cache_line_level: None,
    };

    // Create index
    println!("Creating index (R={}, L={}, alpha={}) ...", args.max_degree, args.l_build, args.alpha);
    let index = diskann_async::new_index::<T, _>(config, params, NoDeletes)?;

    // Set start points
    set_start_points(index.provider(), data.as_view())?;

    // Determine thread count
    let num_threads = if args.num_threads == 0 {
        num_cpus::get()
    } else {
        args.num_threads
    };
    println!("Building index with {} threads ...", num_threads);

    // Insert all vectors
    let build_start = Instant::now();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_threads)
        .build()
        .context("Failed to create tokio runtime")?;

    rt.block_on(async {
        // Insert vectors sequentially (matching the C++ single-threaded insert)
        // but using the async runtime for the underlying graph operations.
        let context = DefaultContext;
        for i in 0..npoints {
            index
                .insert(FullPrecision, &context, &(i as u32), data.row(i))
                .await
                .map_err(|e| anyhow::anyhow!("Failed to insert vector {}: {}", i, e))?;

            if (i + 1) % 10000 == 0 || i + 1 == npoints {
                eprint!("\r  Inserted {}/{} vectors", i + 1, npoints);
            }
        }
        eprintln!();
        Ok::<(), anyhow::Error>(())
    })?;

    let build_elapsed = build_start.elapsed();
    println!(
        "Index build completed in {:.2}s ({:.0} vectors/sec)",
        build_elapsed.as_secs_f64(),
        npoints as f64 / build_elapsed.as_secs_f64()
    );

    // Save index
    println!("Saving index to {} ...", args.index_path_prefix);
    rt.block_on(async {
        index
            .save_with(
                &FileStorageProvider,
                &AsyncIndexMetadata::new(&args.index_path_prefix),
            )
            .await
            .map_err(|e| anyhow::anyhow!("Failed to save index: {}", e))
    })?;

    let total_elapsed = total_start.elapsed();
    println!(
        "Done. Total time: {:.2}s. Index saved to: {}",
        total_elapsed.as_secs_f64(),
        args.index_path_prefix
    );

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Parameters:");
    println!("  data_type:  {:?}", args.data_type);
    println!("  dist_fn:    {}", args.dist_fn);
    println!("  data_path:  {}", args.data_path);
    println!("  R:          {}", args.max_degree);
    println!("  L_build:    {}", args.l_build);
    println!("  alpha:      {}", args.alpha);
    println!(
        "  threads:    {}",
        if args.num_threads == 0 {
            format!("{} (auto)", num_cpus::get())
        } else {
            args.num_threads.to_string()
        }
    );
    println!("  save_path:  {}", args.index_path_prefix);
    println!();

    match args.data_type {
        DataType::Float => build_and_save::<f32>(&args),
        DataType::Float16 => build_and_save::<f16>(&args),
        DataType::Uint8 => build_and_save::<u8>(&args),
        DataType::Int8 => build_and_save::<i8>(&args),
    }
}
