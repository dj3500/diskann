/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! `flab build`: build a graph (optionally recording its candidate pools) or replay recorded
//! pools with a pruning rule; writes `graph.flab` and `meta.json` into the output directory.

use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{ensure, Context, Result};
use clap::ValueEnum;
use serde_json::{json, Value};

use super::{
    build,
    builder::{slots, BuildParams, Order},
    replay,
    stitch::{stitch, StitchParams},
    BuildElement, Candidates, Pools, PruneRule, ReplayMode, ReplayParams,
};
use crate::{
    diskann_io,
    graph::Graph,
    labels::LabelStore,
    run::sys,
    vectors::{DType, Matrix},
};

/// Arguments of `flab build`.
#[derive(Debug, Clone, clap::Args)]
pub struct BuildArgs {
    /// Dataset directory (`base.u8bin` or `base.fbin`, `labels.flab`).
    #[arg(long)]
    pub dataset: Option<PathBuf>,
    /// Base vectors (default: the dataset's base file).
    #[arg(long)]
    pub base: Option<PathBuf>,
    /// Label store of label-aware rules (default: the dataset's `labels.flab`).
    #[arg(long)]
    pub labels: Option<PathBuf>,
    /// Pruning rule: `alpha(R, α)`, `label_containment(R, α)`, `source_containment(R, α)`,
    /// `coverage(R, α, B[, k])` or `split(R_geo, R_lab[, α])`.
    #[arg(long, default_value = "alpha(64, 1.2)")]
    pub prune: String,
    /// Candidate generation of the insertions: `blind`, `filtered(<subset>[, <budget>|R[,
    /// <L_f>]])` (subset `all`, `<θ>` or `n/<k>`: one label-filtered search per label of the
    /// subset, rarest first, at most `budget` per point, default R) or
    /// `penalized(<λ>|inf[, raw|penalized])` (λ in units of the median squared
    /// nearest-neighbor distance).
    #[arg(long, default_value = "blind")]
    pub candidates: String,
    /// Sample size of the nearest-neighbor distance scale of penalized insertion.
    #[arg(long, default_value_t = 256)]
    pub nn_sample: usize,
    /// Stitch per-label Vamana graphs of the labels of this subset (`all`, `<θ>`, `n/<k>`) into
    /// the label-blind graph `--stitch-into` instead of building (the rows of their points are
    /// re-pruned with `--prune`).
    #[arg(long, requires = "stitch_into")]
    pub stitch: Option<String>,
    /// The label-blind graph of `--stitch` (`graph.flab`).
    #[arg(long)]
    pub stitch_into: Option<PathBuf>,
    /// Degree of the per-label graphs of `--stitch`.
    #[arg(long, default_value_t = 32)]
    pub stitch_r: usize,
    /// List size of the per-label graphs of `--stitch`.
    #[arg(long, default_value_t = 100)]
    pub stitch_l: usize,
    /// Replay these recorded pools with the rule instead of building (no searches).
    #[arg(long)]
    pub replay: Option<PathBuf>,
    /// How a replay uses the pools.
    #[arg(long, value_enum, default_value_t = ReplayModeArg::Incremental)]
    pub replay_mode: ReplayModeArg,
    /// Candidate list size of the insertion search.
    #[arg(long, default_value_t = 100)]
    pub l_build: usize,
    /// Rows hold up to ⌊slack · R⌋ neighbors (default 1.3; a replay defaults to its build's).
    #[arg(long)]
    pub slack: Option<f32>,
    /// Pool truncation (default 750; a replay defaults to its build's).
    #[arg(long)]
    pub maxc: Option<usize>,
    /// Largest insertion batch (batch sizes double from 1 up to it).
    #[arg(long, default_value_t = 128)]
    pub batch: usize,
    /// Insertion order.
    #[arg(long, value_enum, default_value_t = OrderArg::Random)]
    pub order: OrderArg,
    /// Seed of the random insertion order.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    #[arg(short = 't', long, default_value_t = 1)]
    pub threads: usize,
    /// Record the candidate pools of the build into this file (`pools.flab`).
    #[arg(long)]
    pub pools: Option<PathBuf>,
    /// Re-prune rows longer than R at the end.
    #[arg(long)]
    pub final_prune: bool,
    /// Report the edge overlap with this graph (`graph.flab`, or a DiskANN index prefix).
    #[arg(long)]
    pub compare: Option<PathBuf>,
    /// Output directory (`graph.flab`, `meta.json`).
    #[arg(short = 'o', long)]
    pub out_dir: PathBuf,
    /// Code version recorded in `meta.json` (default: the `FLAB_COMMIT` environment variable).
    #[arg(long)]
    pub commit: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ReplayModeArg {
    /// The build's insertion process on the recorded search pools, back-edges included.
    Incremental,
    /// One prune of every node's whole recorded pool, no back-edges.
    Union,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OrderArg {
    Random,
    Identity,
}

/// Runs `flab build`.
pub fn run(args: &BuildArgs) -> Result<()> {
    let base = match (&args.base, &args.dataset) {
        (Some(b), _) => b.clone(),
        (None, Some(d)) => ["base.u8bin", "base.fbin"]
            .iter()
            .map(|f| d.join(f))
            .find(|p| p.exists())
            .with_context(|| format!("no base.u8bin or base.fbin in {}", d.display()))?,
        (None, None) => anyhow::bail!("give --dataset or --base"),
    };
    match DType::from_path(&base)? {
        DType::U8 => run_typed::<u8>(args, &base),
        DType::F32 => run_typed::<f32>(args, &base),
    }
}

fn file_info(path: &Path) -> Value {
    json!({
        "path": path,
        "bytes": std::fs::metadata(path).map(|m| m.len()).ok(),
    })
}

fn run_typed<T: BuildElement>(args: &BuildArgs, base_path: &Path) -> Result<()> {
    let t_total = Instant::now();
    let created = sys::utc_timestamp(false);
    let load_start = sys::load_average();
    let rule: PruneRule = args.prune.parse()?;
    let candidates: Candidates = args.candidates.parse()?;
    anyhow::ensure!(
        args.replay.is_none() || candidates == Candidates::Blind,
        "a replay uses recorded pools: --candidates applies to builds only"
    );
    let labels_path = args
        .labels
        .clone()
        .or_else(|| args.dataset.as_ref().map(|d| d.join("labels.flab")));

    let t_load = Instant::now();
    let base = Matrix::<T>::read_bin(base_path)?;
    let labels = if rule.needs_labels() || candidates.needs_labels() || args.stitch.is_some() {
        let path = labels_path
            .as_ref()
            .context("label-aware rules and candidates need --labels or --dataset")?;
        Some(LabelStore::read(path)?)
    } else {
        None
    };
    let pools = args.replay.as_deref().map(Pools::read).transpose()?;
    let load_s = t_load.elapsed().as_secs_f64();
    std::fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("creating {}", args.out_dir.display()))?;

    let cpu_start = sys::process_cpu_s();
    let t_build = Instant::now();
    let (graph, mode_json, params_json, stats_json) = if let Some(subset) = &args.stitch {
        let store = labels
            .as_ref()
            .context("--stitch needs labels (--labels or --dataset)")?;
        let into = args
            .stitch_into
            .as_ref()
            .context("--stitch needs --stitch-into")?;
        let blind = Graph::read(into)?;
        let params = StitchParams {
            subset: subset.parse()?,
            r_small: args.stitch_r,
            l_small: args.stitch_l,
            rule,
            maxc: args.maxc.unwrap_or(750),
            threads: args.threads,
            seed: args.seed,
        };
        eprintln!(
            "stitching per-label graphs of labels {} (R {}, L {}) into {} with {rule}, {} threads",
            params.subset,
            params.r_small,
            params.l_small,
            into.display(),
            args.threads
        );
        let (graph, stats, mut info) = stitch(&base, store, &blind, &params)?;
        info["stitch_into"] = json!(into);
        (graph, "stitch", info, serde_json::to_value(&stats)?)
    } else if let Some(pools) = &pools {
        let built: Value = serde_json::from_str(pools.meta()).unwrap_or(Value::Null);
        let slack = args
            .slack
            .or_else(|| built["slack"].as_f64().map(|s| s as f32))
            .unwrap_or(1.3);
        let maxc = args.maxc.unwrap_or(pools.maxc()).min(pools.maxc());
        let mode = match args.replay_mode {
            ReplayModeArg::Incremental => ReplayMode::Incremental,
            ReplayModeArg::Union => ReplayMode::Union,
        };
        let params = ReplayParams {
            rule,
            mode,
            slack,
            maxc,
            threads: args.threads,
            final_prune: args.final_prune,
        };
        eprintln!(
            "replaying {} ({} nodes, {} pool entries, {} batches) with {rule} ({} mode, {} threads)",
            args.replay.as_ref().map_or(String::new(), |p| p.display().to_string()),
            pools.n_nodes(),
            pools.n_entries(),
            pools.n_batches(),
            mode.name(),
            args.threads
        );
        let out = replay(pools, &base, labels.as_ref(), &params)?;
        let params_json = json!({
            "rule": rule.to_string(),
            "replay_mode": mode.name(),
            "slack": slack,
            "slots": match mode {
                ReplayMode::Incremental => slots(rule.degree(), slack),
                ReplayMode::Union => rule.degree(),
            },
            "maxc": maxc,
            "final_prune": args.final_prune,
            "recorded_build": built,
        });
        (
            out.graph,
            "replay",
            params_json,
            serde_json::to_value(&out.stats)?,
        )
    } else {
        let params = BuildParams {
            rule,
            l_build: args.l_build,
            slack: args.slack.unwrap_or(1.3),
            maxc: args.maxc.unwrap_or(750),
            max_batch: args.batch,
            order: match args.order {
                OrderArg::Random => Order::Random(args.seed),
                OrderArg::Identity => Order::Identity,
            },
            threads: args.threads,
            capture: args.pools.is_some(),
            final_prune: args.final_prune,
            candidates,
            nn_sample: args.nn_sample,
        };
        eprintln!(
            "building over {} × {} {} points with {rule}, candidates {candidates}, L_build {}, batches ≤ {}, {} threads{}",
            base.nrows(),
            base.ncols(),
            T::NAME,
            params.l_build,
            params.max_batch,
            params.threads,
            if params.capture {
                ", recording pools"
            } else {
                ""
            }
        );
        let out = build(&base, labels.as_ref(), &params)?;
        let params_json = json!({
            "rule": rule.to_string(),
            "l_build": params.l_build,
            "slack": params.slack,
            "slots": params.slots(),
            "maxc": params.maxc,
            "max_batch": params.max_batch,
            "schedule": "prefix doubling 1, 2, 4, ... up to max_batch",
            "order": match params.order {
                Order::Identity => json!("identity"),
                Order::Random(seed) => json!({"random": seed}),
            },
            "final_prune": params.final_prune,
            "start": "frozen copy of the medoid (auxiliary node n_data)",
            "candidates": out.candidates,
        });
        let mut stats = serde_json::to_value(&out.stats)?;
        if let (Some(path), Some(pools)) = (&args.pools, &out.pools) {
            let t_write = Instant::now();
            if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
                std::fs::create_dir_all(dir)?;
            }
            pools.write(path)?;
            let mean = |f: &dyn Fn(u32) -> usize| {
                (0..pools.n_nodes() as u32).map(f).sum::<usize>() as f64 / pools.n_nodes() as f64
            };
            let report = json!({
                "artifact": path.parent().and_then(Path::file_name).map(|s| s.to_string_lossy()),
                "kind": "candidate pools for prune replay (pools.flab)",
                "created": created,
                "commit": commit(args),
                "graph": args.out_dir,
                "file": file_info(path),
                "n_data": pools.n_data(),
                "n_nodes": pools.n_nodes(),
                "n_entries": pools.n_entries(),
                "n_batches": pools.n_batches(),
                "mean_search_pool": mean(&|v| pools.search(v).0.len()),
                "mean_proposals": mean(&|v| pools.proposals(v).0.len()),
                "write_s": t_write.elapsed().as_secs_f64(),
                "build": serde_json::from_str::<Value>(pools.meta())?,
            });
            let meta = path.with_file_name("meta.json");
            std::fs::write(&meta, serde_json::to_string_pretty(&report)? + "\n")?;
            stats["pools_write_s"] = json!(t_write.elapsed().as_secs_f64());
            eprintln!(
                "pools: {} entries ({:.2} GB) -> {}",
                pools.n_entries(),
                pools.file_bytes() as f64 / 1e9,
                path.display()
            );
        }
        (out.graph, "build", params_json, stats)
    };
    let build_s = t_build.elapsed().as_secs_f64();
    let cpu_s = sys::process_cpu_s().zip(cpu_start).map(|(b, a)| b - a);
    let load_end = sys::load_average();

    let t_write = Instant::now();
    let graph_path = args.out_dir.join("graph.flab");
    graph.write(&graph_path)?;
    ensure!(
        Graph::read(&graph_path)? == graph,
        "{} does not read back identically",
        graph_path.display()
    );
    let write_s = t_write.elapsed().as_secs_f64();
    let compare = match &args.compare {
        None => Value::Null,
        Some(path) => {
            let data = path.with_extension("data");
            let other = if data.exists() {
                diskann_io::open(path, Some((&data, T::DTYPE)))?
            } else {
                diskann_io::open(path, None)?
            };
            let mut c = edge_overlap(&graph, &other);
            c["graph"] = json!(path);
            c
        }
    };

    let stats = graph.stats();
    let n_data = graph.n_data();
    let data_edges: u64 = (0..n_data as u32).map(|v| graph.degree(v) as u64).sum();
    let meta = json!({
        "artifact": args.out_dir.file_name().map(|s| s.to_string_lossy()),
        "kind": match mode_json {
            "replay" => "graph.flab from prune replay (flab build --replay)",
            "stitch" => "graph.flab from per-label graphs stitched into a label-blind graph (flab build --stitch)",
            _ => "graph.flab from the lab's batch-parallel Vamana builder (flab build)",
        },
        "tool": "flab build",
        "flab_version": env!("CARGO_PKG_VERSION"),
        "commit": commit(args),
        "command_line": std::env::args().collect::<Vec<_>>(),
        "created": created,
        "mode": mode_json,
        "inputs": {
            "dataset": args.dataset,
            "base": file_info(base_path),
            "labels": labels.as_ref().and(labels_path.as_deref()).map(file_info),
            "pools": args.replay.as_deref().map(file_info),
        },
        "params": params_json,
        "build": {
            "threads": args.threads,
            "wall_s": build_s,
            "cpu_s": cpu_s,
            "thread_s": build_s * args.threads as f64,
            "load_s": load_s,
            "write_s": write_s,
            "total_s": t_total.elapsed().as_secs_f64(),
            "peak_rss_bytes": sys::peak_rss_bytes(),
            "load_avg_start": load_start,
            "load_avg_end": load_end,
            "host": sys::host_info(),
            "stats": stats_json,
        },
        "graph": {
            "n_data": n_data,
            "n_nodes": graph.n_nodes(),
            "r_max": graph.r_max(),
            "starts": graph.starts(),
            "n_edges": graph.n_edges(),
            "mean_degree_data": data_edges as f64 / n_data as f64,
            "bytes": {
                "graph_flab": std::fs::metadata(&graph_path)?.len(),
                "edges_u32": 4 * graph.n_edges(),
                "per_data_point_u32_edges": 4.0 * data_edges as f64 / n_data as f64,
            },
            "stats": stats.to_json(),
        },
        "compare": compare,
    });
    let meta_path = args.out_dir.join("meta.json");
    std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)? + "\n")?;
    eprintln!(
        "{mode_json}: {:.1} s wall ({} threads{}), {} edges, mean degree {:.2}, max {} -> {}",
        build_s,
        args.threads,
        cpu_s.map_or(String::new(), |c| format!(", {c:.1} s CPU")),
        graph.n_edges(),
        data_edges as f64 / n_data as f64,
        stats.degree.as_ref().map_or(0, |d| d.max),
        args.out_dir.display()
    );
    if !compare.is_null() {
        eprintln!(
            "compare: {} of {} edges in common ({:.4}), identical graph: {}",
            compare["common_edges"],
            compare["edges_other"],
            compare["fraction_of_other"],
            compare["identical"]
        );
    }
    Ok(())
}

fn commit(args: &BuildArgs) -> Option<String> {
    args.commit
        .clone()
        .or_else(|| std::env::var("FLAB_COMMIT").ok())
}

/// Directed-edge overlap of two graphs over the same node space.
pub fn edge_overlap(a: &Graph, b: &Graph) -> Value {
    if a.n_nodes() != b.n_nodes() || a.n_data() != b.n_data() {
        return json!({"error": format!("node spaces differ: {} / {} vs {} / {}", a.n_data(), a.n_nodes(), b.n_data(), b.n_nodes())});
    }
    let (mut common, mut same_set, mut same_list) = (0u64, 0u64, 0u64);
    let mut jaccard = 0.0f64;
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for v in 0..a.n_nodes() as u32 {
        let (ra, rb) = (a.neighbors(v), b.neighbors(v));
        same_list += u64::from(ra == rb);
        x.clear();
        x.extend_from_slice(ra);
        x.sort_unstable();
        y.clear();
        y.extend_from_slice(rb);
        y.sort_unstable();
        let (mut i, mut j, mut c) = (0, 0, 0u64);
        while i < x.len() && j < y.len() {
            match x[i].cmp(&y[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    c += 1;
                    i += 1;
                    j += 1;
                }
            }
        }
        common += c;
        same_set += u64::from(x == y);
        let union = (x.len() + y.len()) as u64 - c;
        jaccard += if union == 0 {
            1.0
        } else {
            c as f64 / union as f64
        };
    }
    let n = a.n_nodes() as f64;
    json!({
        "edges_this": a.n_edges(),
        "edges_other": b.n_edges(),
        "common_edges": common,
        "fraction_of_this": common as f64 / a.n_edges().max(1) as f64,
        "fraction_of_other": common as f64 / b.n_edges().max(1) as f64,
        "mean_jaccard": jaccard / n,
        "rows_same_set": same_set,
        "rows_same_list": same_list,
        "identical": a == b,
    })
}
