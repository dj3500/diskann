/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! `flab-upstream`: upstream DiskANN's filtered searches on an upstream-saved index, through a
//! thin timed adapter with the lab's predicate providers and run outputs.

use std::{path::PathBuf, sync::Arc};

use anyhow::{bail, ensure, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use diskann::utils::VectorRepr;
use diskann_filter_lab::{
    labels::LabelStore,
    upstream::{
        index::{self, LoadedIndex},
        queries::{dataset_dtype, Queries, QuerySource},
        run::{self, ProviderKind, RunContext, RunSpec},
        search::Algorithm,
        sys,
    },
    vectors::{DType, Element},
};
use serde_json::{json, Value};

#[derive(Parser)]
#[command(
    name = "flab-upstream",
    version,
    about = "Upstream DiskANN filtered searches through a thin timed adapter"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Timed single-threaded runs of upstream algorithms (pin the process with taskset).
    Run(Box<RunArgs>),
    /// Summarize an upstream-saved index: header, node space and degrees.
    Info {
        /// Index prefix (graph file; vectors in `<prefix>.data`).
        #[arg(long)]
        index: String,
        #[arg(long, value_enum)]
        dtype: DTypeArg,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum DTypeArg {
    U8,
    F32,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum AlgorithmArg {
    /// Unfiltered `Knn` with list size L, then the best K matches of the L candidates.
    KnnPostfilter,
    /// `InlineFilterSearch` with fixed L.
    Inline,
    /// `InlineFilterSearch` with upstream's adaptive L.
    InlineAdaptive,
    /// `MultihopFilterSearch`.
    Multihop,
    /// The `BetaFilter` layer over `Knn`, post-filtered to matches with true distances.
    Beta,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ProviderArg {
    /// Evaluate the CNF on the point's label row (most selective clause first).
    Forward,
    /// Build a dense match bitmap from postings per query, then test bits.
    Bitmap,
}

#[derive(Args)]
struct RunArgs {
    /// Index prefix (graph file; vectors in `<prefix>.data`).
    #[arg(long)]
    index: String,
    /// Dataset directory (`labels.flab`, `q<S>.{u8bin,fbin}`, `q<S>.fpred`).
    #[arg(long)]
    dataset: PathBuf,
    /// Run every query of query set S of the dataset.
    #[arg(long, conflicts_with_all = ["panel", "queries"])]
    qset: Option<String>,
    /// Run the rows of a panel file (TSV with columns `qset`, `qidx`, optional `split`).
    #[arg(long, conflicts_with = "queries")]
    panel: Option<PathBuf>,
    /// Explicit query vectors (requires --predicates).
    #[arg(long, requires = "predicates")]
    queries: Option<PathBuf>,
    /// Explicit predicates of --queries.
    #[arg(long, requires = "queries")]
    predicates: Option<PathBuf>,
    /// Use only the first N queries.
    #[arg(long)]
    max_queries: Option<usize>,
    /// Algorithms (comma-separated).
    #[arg(long, value_enum, value_delimiter = ',', required = true)]
    algorithms: Vec<AlgorithmArg>,
    /// Predicate providers (comma-separated).
    #[arg(long, value_enum, value_delimiter = ',', default_value = "forward")]
    providers: Vec<ProviderArg>,
    /// Numbers of results K (comma-separated). Each K runs the efforts L >= K of --efforts,
    /// or its own list given as `K:L1/L2/...` (e.g. `10:10/20/50,100:100/200/500`).
    #[arg(short, value_delimiter = ',', default_value = "10")]
    k: Vec<String>,
    /// Search list sizes L (comma-separated).
    #[arg(long, value_delimiter = ',')]
    efforts: Vec<usize>,
    /// Adaptive L: distance computations before the match rate is estimated.
    #[arg(long, default_value_t = 1000)]
    adaptive_samples: usize,
    /// Adaptive L: maximum multiplier of L.
    #[arg(long, default_value_t = 16.0)]
    adaptive_scale: f64,
    /// Beta: distance multiplier of matching points, in (0, 1].
    #[arg(long, default_value_t = 0.5)]
    beta: f32,
    /// Timed repetitions (interleaved across all configurations).
    #[arg(long, default_value_t = 3)]
    reps: usize,
    /// Directory receiving one run directory per configuration.
    #[arg(long)]
    runs_dir: PathBuf,
    /// Trial ledger (JSON lines, appended).
    #[arg(long)]
    ledger: PathBuf,
    /// Free-form tag recorded in run.json.
    #[arg(long)]
    tag: Option<String>,
    /// Element type; inferred from the dataset's base vectors when omitted.
    #[arg(long, value_enum)]
    dtype: Option<DTypeArg>,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Run(args) => {
            let dtype = match args.dtype {
                Some(DTypeArg::U8) => DType::U8,
                Some(DTypeArg::F32) => DType::F32,
                None => dataset_dtype(&args.dataset)?,
            };
            match dtype {
                DType::U8 => run_typed::<u8>(&args),
                DType::F32 => run_typed::<f32>(&args),
            }
        }
        Command::Info { index, dtype } => match dtype {
            DTypeArg::U8 => info::<u8>(&index),
            DTypeArg::F32 => info::<f32>(&index),
        },
    }
}

/// Parses `K` (efforts: those of `default` that are >= K) or `K:L1/L2/...`.
fn parse_k(spec: &str, default: &[usize]) -> Result<(usize, Vec<usize>)> {
    let (k, efforts) = match spec.split_once(':') {
        Some((k, list)) => (
            k.trim().parse::<usize>()?,
            list.split('/')
                .map(|l| l.trim().parse::<usize>())
                .collect::<Result<Vec<_>, _>>()?,
        ),
        None => {
            let k = spec.trim().parse::<usize>()?;
            (k, default.iter().copied().filter(|&l| l >= k).collect())
        }
    };
    ensure!(
        !efforts.is_empty() && efforts.iter().all(|&l| l >= k),
        "K = {k} needs at least one effort, and every effort must be >= K"
    );
    Ok((k, efforts))
}

fn index_description<T: VectorRepr + Element>(loaded: &LoadedIndex<T>, prefix: &str) -> Value {
    let meta_path = std::path::Path::new(prefix).with_file_name("meta.json");
    let artifact = std::fs::read_to_string(&meta_path)
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok());
    json!({
        "prefix": prefix,
        "format": "upstream in-memory index (graph + .data)",
        "dtype": T::NAME,
        "dim": loaded.dim,
        "n_data": loaded.n_data,
        "start_ids": loaded.start_ids,
        "max_observed_degree": loaded.header.max_observed_degree,
        "graph_file_bytes": loaded.header.file_size,
        "artifact_meta": artifact,
    })
}

fn run_typed<T: VectorRepr + Element>(args: &RunArgs) -> Result<()> {
    ensure!(args.reps > 0, "--reps must be positive");
    let source = match (&args.qset, &args.panel, &args.queries, &args.predicates) {
        (Some(s), None, None, None) => QuerySource::Set(s.clone()),
        (None, Some(p), None, None) => QuerySource::Panel(p.clone()),
        (None, None, Some(v), Some(p)) => QuerySource::Files {
            vectors: v.clone(),
            predicates: p.clone(),
        },
        _ => bail!("give exactly one of --qset, --panel, or --queries with --predicates"),
    };

    let t = std::time::Instant::now();
    let labels = Arc::new(LabelStore::read(&args.dataset.join("labels.flab"))?);
    let loaded = index::load::<T>(&args.index)?;
    let queries = Queries::<T>::load(
        &args.dataset,
        &source,
        &labels,
        loaded.dim,
        args.max_queries,
    )?;
    eprintln!(
        "loaded index ({} data points, start {:?}), labels and {} queries in {:.1} s",
        loaded.n_data,
        loaded.start_ids,
        queries.len(),
        t.elapsed().as_secs_f64()
    );

    let mut specs = Vec::new();
    for &algorithm in &args.algorithms {
        let algorithm = match algorithm {
            AlgorithmArg::KnnPostfilter => Algorithm::KnnPostfilter,
            AlgorithmArg::Inline => Algorithm::Inline,
            AlgorithmArg::InlineAdaptive => {
                Algorithm::inline_adaptive(args.adaptive_samples, args.adaptive_scale)?
            }
            AlgorithmArg::Multihop => Algorithm::Multihop,
            AlgorithmArg::Beta => Algorithm::beta(args.beta)?,
        };
        for &provider in &args.providers {
            for k_spec in &args.k {
                let (k, efforts) = parse_k(k_spec, &args.efforts)?;
                specs.push(RunSpec {
                    algorithm: algorithm.clone(),
                    provider: match provider {
                        ProviderArg::Forward => ProviderKind::Forward,
                        ProviderArg::Bitmap => ProviderKind::Bitmap,
                    },
                    k,
                    efforts,
                });
            }
        }
    }

    let dataset_name = args
        .dataset
        .canonicalize()
        .unwrap_or_else(|_| args.dataset.clone())
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let panel = match &source {
        QuerySource::Panel(p) => {
            let bytes = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
            Some((p.display().to_string(), sys::fnv1a_hex(&bytes)))
        }
        _ => None,
    };
    let queries_json = match &source {
        QuerySource::Set(s) => json!({"kind": "qset", "qset": s}),
        QuerySource::Files {
            vectors,
            predicates,
        } => json!({"kind": "files", "vectors": vectors, "predicates": predicates}),
        QuerySource::Panel(p) => json!({
            "kind": "panel",
            "panel": p,
            "panel_hash": panel.as_ref().map(|x| &x.1),
        }),
    };
    let mut queries_json = queries_json;
    queries_json["max_queries"] = json!(args.max_queries);
    queries_json["sets"] = json!(queries.set_names());
    let ctx = RunContext {
        runs_dir: args.runs_dir.clone(),
        ledger: args.ledger.clone(),
        dataset_name: dataset_name.clone(),
        dataset: json!({
            "name": dataset_name,
            "dir": args.dataset,
            "labels_dict_hash": format!("{:016x}", labels.dict_hash()),
            "n_points": labels.n_points(),
            "n_labels": labels.n_labels(),
        }),
        queries: queries_json,
        index: index_description(&loaded, &args.index),
        tool: json!({
            "name": "flab-upstream",
            "version": env!("CARGO_PKG_VERSION"),
            "commit": option_env!("FLAB_GIT_COMMIT"),
            "executable": std::env::current_exe().ok(),
            "command_line": std::env::args().collect::<Vec<_>>(),
        }),
        panel,
        reps: args.reps,
        tag: args.tag.clone(),
    };

    let outcomes = run::run(&loaded, &labels, &queries, &specs, &ctx)?;
    println!(
        "{:<44} {:>6} {:>10} {:>10} {:>10} {:>9} {:>10} {:>10} {:>7} {:>5}",
        "run", "L", "QPS", "wall_us", "cpu_us", "prep_us", "n_dist", "n_pred", "n_out", "nondet"
    );
    for (spec, outcome) in specs.iter().zip(&outcomes) {
        let name = format!(
            "{}/{}/k{}",
            spec.algorithm.name(),
            spec.provider.name(),
            spec.k
        );
        for s in &outcome.summary {
            let wall = s.median_wall_ns();
            println!(
                "{:<44} {:>6} {:>10.1} {:>10.1} {:>10.1} {:>9.1} {:>10.1} {:>10.1} {:>7.2} {:>5}",
                name,
                s.effort,
                1e9 / wall,
                wall / 1e3,
                s.median_cpu_ns() / 1e3,
                s.mean_prep_ns / 1e3,
                s.mean_dist,
                s.mean_pred,
                s.mean_out,
                s.nondeterministic_rows
            );
        }
        println!("  -> {}", outcome.dir.display());
    }
    Ok(())
}

fn info<T: VectorRepr + Element>(prefix: &str) -> Result<()> {
    let loaded = index::load::<T>(prefix)?;
    let neighbors = loaded.index.provider().neighbors();
    let n_nodes = loaded.n_data + loaded.start_ids.len();
    let mut buf = diskann::graph::AdjacencyList::new();
    let (mut total, mut max, mut zero) = (0u64, 0usize, 0usize);
    for id in 0..n_nodes {
        neighbors.get_neighbors_sync(id, &mut buf)?;
        total += buf.len() as u64;
        max = max.max(buf.len());
        zero += usize::from(buf.is_empty());
    }
    let out = json!({
        "header": {
            "file_size": loaded.header.file_size,
            "max_observed_degree": loaded.header.max_observed_degree,
            "start": loaded.header.start,
            "n_frozen": loaded.header.n_frozen,
        },
        "dtype": T::NAME,
        "dim": loaded.dim,
        "n_data": loaded.n_data,
        "start_ids": loaded.start_ids,
        "degree": {
            "mean": total as f64 / n_nodes as f64,
            "max": max,
            "nodes_without_edges": zero,
            "edges": total,
        },
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_k;

    #[test]
    fn efforts_per_k() {
        assert_eq!(parse_k("10", &[5, 10, 50]).unwrap(), (10, vec![10, 50]));
        assert_eq!(
            parse_k("100:100/500", &[10]).unwrap(),
            (100, vec![100, 500])
        );
        assert!(parse_k("100", &[10, 50]).is_err());
        assert!(parse_k("10:5/20", &[]).is_err());
    }
}
