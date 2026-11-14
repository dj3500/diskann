/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! `flab run <experiment.toml>`: runs the configurations of an experiment over a query panel
//! and an effort grid, and writes one run directory per (configuration, K).
//!
//! Measurement protocol (timed mode):
//! - the process must be pinned to one CPU (`taskset -c <cpu>`); otherwise the run refuses
//!   to start. The CPU, NUMA node, affinity, memory policy, niceness and load average are
//!   recorded at the start and the end, with a calibration (a fixed distance kernel and a
//!   fixed small search);
//! - all inputs are loaded and pre-faulted before timing, and every scratch buffer is
//!   allocated (and written) once per run;
//! - per K: an untimed warm-up of every configuration over the whole panel at the median
//!   effort; then for each repetition and effort, the configurations in a seeded shuffled order
//!   (re-shuffled per (repetition, effort)), the queries in panel order;
//! - per query: `cpu_t0 → wall_t0 → search (predicate compilation, routing, starts, pilot,
//!   materialization, the loop, collection, scratch resets, output copy) → wall_t1 →
//!   cpu_t1`, with `CLOCK_THREAD_CPUTIME_ID` read outside the wall-clock interval;
//! - configurations that are exactly the `postfilter` or `upstream-inline` preset run their
//!   hand-specialized loop (same answers and counters, a few percent faster than the
//!   configurable loop); `fast_paths = false` runs everything in the configurable loop, and
//!   `run.json` records the loop used (`config.fast_path`).
//!
//! Outputs per run directory: `run.json` (tool, dataset, panel, index, the resolved
//! configuration, K, efforts, mode, host, placement, load, calibration, per-effort summary),
//! `res_<effort>.bin` (DiskANN truthset layout: `u32 nq`, `u32 K`, ids padded with
//! `u32::MAX`, true squared-L2 distances, rows in panel order), `perquery.tsv`, the panel rows
//! used (`panel.tsv`), traces (trace mode); and one line per run in the ledger.

pub mod data;
pub mod spec;
pub mod sys;

use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{ensure, Context, Result};
use hashbrown::HashMap;
use serde_json::{json, Value};

use self::{
    data::{fnv1a, Dataset, LoadedGraph, Rows},
    spec::{ConfigSpec, Experiment, Mode},
};
use crate::{
    kernel::{
        config::StartSource,
        entry::SampleIndex,
        search::{Counters, FastPath, Index, Plan, Query, Searcher},
        trace::Trace,
    },
    vectors::{DType, Element},
};

/// Command-line options of `flab run`.
#[derive(Debug, Clone, Default)]
pub struct Options {
    /// Overrides the experiment's `runs_dir`.
    pub runs_dir: Option<PathBuf>,
    /// Overrides the experiment's `ledger`.
    pub ledger: Option<PathBuf>,
    /// Validate and print the plan without searching.
    pub dry_run: bool,
    /// Code version recorded in the outputs.
    pub commit: Option<String>,
    /// The command line (recorded).
    pub command_line: Vec<String>,
}

/// Runs an experiment file; returns the run directories written.
pub fn run(path: &Path, opts: &Options) -> Result<Vec<PathBuf>> {
    let exp = Experiment::read(path)?;
    let ledger = opts.ledger.clone().or_else(|| exp.ledger.clone());
    let result = match data::dataset_dtype(&exp.dataset)? {
        DType::U8 => drive::<u8>(&exp, opts),
        DType::F32 => drive::<f32>(&exp, opts),
    };
    if let (Err(e), Some(ledger)) = (&result, &ledger) {
        if !opts.dry_run {
            let line = json!({
                "time": sys::utc_timestamp(false),
                "status": "failed",
                "experiment": exp.name,
                "file": exp.file,
                "dataset": exp.dataset,
                "mode": exp.mode.name(),
                "tool": "flab run",
                "error": format!("{e:#}"),
            });
            append_ledger(ledger, &line)?;
        }
    }
    result
}

fn append_ledger(path: &Path, line: &Value) -> Result<()> {
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening the ledger {}", path.display()))?;
    writeln!(f, "{line}")?;
    Ok(())
}

/// A seeded shuffle (SplitMix64 + Fisher–Yates).
fn shuffle<T>(items: &mut [T], seed: u64) {
    let mut state = seed;
    let mut next = move || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    for i in (1..items.len()).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}

/// One per-query record.
#[derive(Debug, Clone, Copy)]
struct Rec {
    effort: usize,
    rep: usize,
    row: usize,
    wall_ns: Option<u64>,
    cpu_ns: Option<u64>,
    ctr: Counters,
    reference: Option<(u64, u64)>,
}

/// Results of one configuration at one K.
struct Outcome {
    efforts: Vec<usize>,
    skipped: Vec<usize>,
    ids: BTreeMap<usize, Vec<u32>>,
    dists: BTreeMap<usize, Vec<f32>>,
    recs: Vec<Rec>,
    nondeterministic: BTreeMap<usize, usize>,
    traces: Vec<(usize, usize, Trace)>,
}

/// Key of a loaded graph (graph path and start-table path).
type GraphKey = (PathBuf, Option<PathBuf>);

fn graph_key(c: &ConfigSpec) -> GraphKey {
    (c.graph.clone(), c.starts_sidecar.clone())
}

/// `ε_min`: median, over a fixed sample of data points, of the smallest positive distance to
/// their graph out-neighbours (an upper bound on the nearest-neighbour distance).
fn eps_min<T: Element>(ds: &Dataset<T>, g: &LoadedGraph) -> f32 {
    let n = g.graph.n_data() as u64;
    let mut vals = Vec::new();
    let mut state = 0x5eed_u64;
    for _ in 0..1000.min(n) {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let p = ((state >> 33) % n) as u32;
        let v = ds.base.row(p as usize);
        let m = g
            .graph
            .neighbors(p)
            .iter()
            .filter(|&&u| (u as usize) < ds.base.nrows())
            .map(|&u| T::squared_l2(v, ds.base.row(u as usize)))
            .filter(|&d| d > 0.0)
            .fold(f32::INFINITY, f32::min);
        if m.is_finite() {
            vals.push(m);
        }
    }
    if vals.is_empty() {
        return 1.0;
    }
    vals.sort_unstable_by(f32::total_cmp);
    vals[vals.len() / 2]
}

/// The fixed distance kernel of the calibration: ns per squared-L2 distance (best of 5
/// passes over the first 256 KiB of the base).
fn calibrate_kernel<T: Element>(ds: &Dataset<T>) -> f64 {
    let dim = ds.base.ncols();
    let rows = ((256 * 1024) / (dim * std::mem::size_of::<T>()).max(1))
        .clamp(2, ds.base.nrows().max(2))
        .min(ds.base.nrows());
    if rows < 2 {
        return 0.0;
    }
    let q = ds.base.row(0);
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        let mut acc = 0.0f32;
        for r in 0..rows {
            acc += T::squared_l2(q, ds.base.row(r));
        }
        std::hint::black_box(acc);
        best = best.min(t.elapsed().as_nanos() as f64 / rows as f64);
    }
    best
}

struct Ctx<'a, T: Element> {
    exp: &'a Experiment,
    ds: &'a Dataset<T>,
    rows: &'a Rows,
    indices: Vec<Index<'a, T>>,
    plans: Vec<Plan>,
    queries: Vec<Vec<Query<'a, T>>>,
}

impl<T: Element> Ctx<'_, T> {
    /// The fixed small search of the calibration: the first configuration at the smallest
    /// effort over the first 64 rows, best of 3 passes (mean wall ns per query).
    fn calibrate_search(&self, searchers: &mut HashMap<GraphKey, Searcher>) -> Value {
        let c = &self.exp.configs[0];
        let st = searchers.get_mut(&graph_key(c)).expect("searcher");
        let l = self.exp.efforts.iter().copied().min().unwrap_or(10);
        let k = 10.min(l);
        let n = self.rows.len().min(64);
        let (mut ids, mut d) = (vec![0u32; k], vec![0f32; k]);
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t = Instant::now();
            for q in &self.queries[0][..n] {
                st.search(
                    &self.plans[0],
                    &self.indices[0],
                    q,
                    l,
                    &mut ids,
                    &mut d,
                    None,
                );
            }
            best = best.min(t.elapsed().as_nanos() as f64 / n as f64);
        }
        json!({
            "search": {"config": c.name, "L": l, "K": k, "rows": n, "mean_wall_ns_best_of_3": best},
            "kernel_ns_per_dist": calibrate_kernel(self.ds),
            "load_avg": sys::load_average(),
            "cpu": sys::current_cpu(),
            "time": sys::utc_timestamp(false),
        })
    }
}

fn drive<T: Element>(exp: &Experiment, opts: &Options) -> Result<Vec<PathBuf>> {
    let started = Instant::now();
    let started_at = sys::utc_timestamp(false);
    if exp.mode == Mode::Timed && !opts.dry_run {
        let a = sys::affinity();
        ensure!(
            a.as_ref().is_some_and(|a| a.len() == 1),
            "timed runs must be pinned to one CPU (e.g. `taskset -c <cpu> flab run ...`); the affinity is {a:?}"
        );
    }
    let runs_dir = opts
        .runs_dir
        .clone()
        .or_else(|| exp.runs_dir.clone())
        .context("no runs directory (set `runs_dir` or pass --runs-dir)")?;
    let ledger = opts.ledger.clone().or_else(|| exp.ledger.clone());

    // Rows and inputs.
    let mut rows = match &exp.panel {
        Some(p) => Rows::from_panel(p, exp.split.as_deref(), exp.qsets.as_deref(), exp.limit)?,
        None => Rows::from_qset(exp.queries.as_deref().unwrap_or_default(), 0, None),
    };
    let views: BTreeSet<String> = exp.configs.iter().filter_map(|c| c.view.clone()).collect();
    let views: Vec<String> = views.into_iter().collect();
    let qset_names = match &exp.panel {
        Some(_) => rows.qsets(),
        None => vec![exp.queries.clone().unwrap_or_default()],
    };
    let ds = Dataset::<T>::load(&exp.dataset, &qset_names, &views)?;
    if exp.panel.is_none() {
        let q = &qset_names[0];
        rows = Rows::from_qset(q, ds.qsets[q].vectors.nrows(), exp.limit);
    }
    for (q, i) in &rows.rows {
        ensure!(
            *i < ds.qsets[q].vectors.nrows(),
            "row {i} of query set {q} does not exist"
        );
    }

    // Graphs, start tables and sample indices.
    let mut graphs: HashMap<GraphKey, LoadedGraph> = HashMap::new();
    for c in &exp.configs {
        let key = graph_key(c);
        if !graphs.contains_key(&key) {
            let g = data::load_graph(&c.graph, c.starts_sidecar.as_deref(), T::DTYPE)
                .with_context(|| format!("config {:?}", c.name))?;
            ensure!(
                g.graph.n_data() == ds.base.nrows(),
                "{}: {} data points, the dataset has {}",
                c.graph.display(),
                g.graph.n_data(),
                ds.base.nrows()
            );
            graphs.insert(key.clone(), g);
        }
        let g = graphs.get_mut(&key).expect("loaded");
        if c.kernel
            .starts
            .iter()
            .any(|s| matches!(s, StartSource::Sample(_)))
            && g.sample.is_none()
        {
            let table = g.starts.as_ref().with_context(|| {
                format!("config {:?}: sample starts need a start table", c.name)
            })?;
            g.sample = Some(SampleIndex::build(table, &ds.labels)?);
        }
    }
    let needs_pilot: Vec<&ConfigSpec> = exp
        .configs
        .iter()
        .filter(|c| c.kernel.scorer.needs_pilot())
        .collect();
    let (eps, eps_method) = match (exp.eps_min, needs_pilot.first()) {
        (Some(e), _) => (e, "given".to_owned()),
        (None, Some(c)) => (
            eps_min(&ds, &graphs[&graph_key(c)]),
            format!(
                "median over 1000 sampled points of the smallest positive distance to a graph neighbour ({})",
                c.graph.display()
            ),
        ),
        (None, None) => (0.0, "not needed".to_owned()),
    };

    // Indices, plans and per-row queries.
    let mut indices = Vec::new();
    let mut plans = Vec::new();
    for c in &exp.configs {
        let g = &graphs[&graph_key(c)];
        let (label_view, view_postings) = match &c.view {
            Some(v) => {
                let (hash, n_labels, post) = &ds.views[v];
                if let Some(t) = &g.starts {
                    if t.n_labels() > 0 {
                        ensure!(
                            t.dict_hash() == *hash && t.n_labels() == *n_labels,
                            "config {:?}: the per-label starts do not refer to view {v:?}",
                            c.name
                        );
                    }
                }
                (true, Some(&post[..]))
            }
            None => (false, None),
        };
        let ix = Index {
            graph: &g.graph,
            vectors: g.graph.vectors(&ds.base)?,
            labels: &ds.labels,
            global: &g.global,
            label_starts: g.starts.as_ref().filter(|s| s.n_labels() > 0),
            sample: g.sample.as_ref(),
            label_view,
            view_postings,
            eps_min: eps,
        };
        let mut plan = Plan::new(&c.kernel, &ix).with_context(|| format!("config {:?}", c.name))?;
        if !exp.fast_paths {
            plan = plan.generic();
        }
        indices.push(ix);
        plans.push(plan);
    }
    let queries: Vec<Vec<Query<'_, T>>> = exp
        .configs
        .iter()
        .map(|c| {
            rows.rows
                .iter()
                .map(|(q, i)| {
                    let set = &ds.qsets[q];
                    Query {
                        vector: set.vectors.row(*i),
                        cnf: set.preds.get(*i),
                        view: c.view.as_ref().map(|v| set.views[v].get(*i)),
                    }
                })
                .collect()
        })
        .collect();

    eprintln!(
        "flab run {}: {} configurations, {} rows, K {:?}, efforts {:?}, {} reps, mode {}",
        exp.name,
        exp.configs.len(),
        rows.len(),
        exp.ks,
        exp.efforts,
        exp.reps,
        exp.mode.name()
    );
    if opts.dry_run {
        for (c, p) in exp.configs.iter().zip(&plans) {
            eprintln!(
                "  {}: {} on {}",
                c.name,
                p.kernel.to_json(),
                c.graph.display()
            );
        }
        return Ok(Vec::new());
    }

    let mut searchers: HashMap<GraphKey, Searcher> = HashMap::new();
    for (c, ix) in exp.configs.iter().zip(&indices) {
        let key = graph_key(c);
        if !searchers.contains_key(&key) {
            let mine: Vec<&Plan> = exp
                .configs
                .iter()
                .zip(&plans)
                .filter(|(o, _)| graph_key(o) == key)
                .map(|(_, p)| p)
                .collect();
            searchers.insert(key, Searcher::for_index(ix, &mine));
        }
    }
    std::hint::black_box(data::prefault(&ds, graphs.values()));

    let cx = Ctx {
        exp,
        ds: &ds,
        rows: &rows,
        indices,
        plans,
        queries,
    };
    let host_start = sys::host_info();
    let calib_start = (exp.mode == Mode::Timed).then(|| cx.calibrate_search(&mut searchers));
    let mut per_k = Vec::new();
    for &k in &exp.ks {
        let load_start = sys::load_average();
        let outcomes = run_k(&cx, &mut searchers, k)?;
        per_k.push((k, outcomes, load_start, sys::load_average()));
    }
    let calib_end = (exp.mode == Mode::Timed).then(|| cx.calibrate_search(&mut searchers));
    let host_end = sys::host_info();

    // Outputs.
    let mut dirs = Vec::new();
    let splits: Vec<String> = split_values(&rows);
    for (k, outcomes, load_start, load_end) in per_k {
        for (ci, out) in outcomes.into_iter().enumerate() {
            let c = &exp.configs[ci];
            let g = &graphs[&graph_key(c)];
            let config_json = json!({
                "name": c.name,
                "preset": c.preset,
                "overrides": c.overrides,
                "kernel": cx.plans[ci].kernel.to_json(),
                "two_hop_cap": cx.plans[ci].two_hop_cap(),
                "view": c.view,
                // Trace and reference searches always run the configurable loop.
                "fast_path": match exp.mode {
                    Mode::Timed => cx.plans[ci].fast_path().name(),
                    Mode::Trace | Mode::Reference => FastPath::None.name(),
                },
            });
            let hash_input = json!({
                "experiment": exp.name, "config": config_json, "K": k, "efforts": out.efforts,
                "dataset": exp.dataset, "graph": c.graph, "starts": g.starts_path,
                "panel": rows.source.as_ref().map(|s| &s.1), "split": exp.split,
                "qsets": exp.qsets, "limit": exp.limit, "queries": exp.queries,
                "mode": exp.mode.name(), "reps": exp.reps, "seed": exp.seed,
            });
            let config_hash = format!("{:016x}", fnv1a(hash_input.to_string().as_bytes()));
            let safe: String = c
                .name
                .chars()
                .map(|ch| {
                    if ch.is_ascii_alphanumeric() || ch == '-' || ch == '.' {
                        ch
                    } else {
                        '_'
                    }
                })
                .collect();
            let run_id = format!(
                "{}-lab-{}-k{}-{}",
                sys::utc_timestamp(true),
                safe,
                k,
                &config_hash[..8]
            );
            let dir = runs_dir.join(&run_id);
            ensure!(!dir.exists(), "{} exists", dir.display());
            std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
            let summary = write_run_files(&dir, &rows, k, &out)?;
            // A whole query set has no panel; a prefix of one (`limit`) is described by the
            // rows written to `panel.tsv`, so that evaluators read exactly the rows used.
            let panel_json = match (&rows.source, exp.limit) {
                (Some((src, sha)), _) => json!({
                    "file": dir.join("panel.tsv"),
                    "sha256": data::sha256_file(&dir.join("panel.tsv"))?,
                    "source": src,
                    "source_sha256": sha,
                    "split": exp.split,
                    "qsets": exp.qsets,
                    "limit": exp.limit,
                }),
                (None, Some(limit)) => json!({
                    "file": dir.join("panel.tsv"),
                    "sha256": data::sha256_file(&dir.join("panel.tsv"))?,
                    "source": Value::Null,
                    "qsets": [exp.queries],
                    "limit": limit,
                }),
                (None, None) => Value::Null,
            };
            let tool = json!({
                "name": "flab run",
                "version": env!("CARGO_PKG_VERSION"),
                "commit": opts.commit,
                "executable": std::env::current_exe().ok(),
                "command_line": opts.command_line,
            });
            let dataset = json!({
                "dir": exp.dataset,
                "name": exp.dataset.file_name().map(|n| n.to_string_lossy().into_owned()),
                "base": ds.base_file,
                "dtype": T::NAME,
                "n_points": ds.base.nrows(),
                "dim": ds.base.ncols(),
                "n_labels": ds.labels.n_labels(),
                "labels_dict_hash": format!("{:016x}", ds.labels.dict_hash()),
                "view": c.view,
            });
            let index = json!({
                "graph": c.graph,
                "starts": g.starts_path,
                "n_data": g.graph.n_data(),
                "n_nodes": g.graph.n_nodes(),
                "n_aux": g.graph.n_aux(),
                "r_max": g.graph.r_max(),
                "n_edges": g.graph.n_edges(),
                "global_starts": g.global,
                "per_label_starts": g.starts.as_ref().map(|s| s.n_label_entries()),
                "sample": g.sample.as_ref().map(SampleIndex::len),
                "sample_index_bytes": g.sample.as_ref().map(SampleIndex::bytes),
            });
            let protocol = json!({
                "reps": if exp.mode == Mode::Timed { exp.reps } else { 1 },
                "seed": exp.seed,
                "interleaving": "per K: for each repetition and effort, the configurations in a seeded shuffled order (re-shuffled per (rep, effort)); queries in panel order",
                "warmup": exp.warmup && exp.mode == Mode::Timed,
                "warmup_effort": "the median applicable effort, whole panel, untimed",
                "prefault": "every page of the base, labels, postings, queries and graph rows read before timing; scratch written at allocation",
            });
            let mut run = serde_json::Map::new();
            let mut put = |key: &str, v: Value| {
                run.insert(key.to_owned(), v);
            };
            put("run_id", json!(run_id));
            put("tool", tool);
            put(
                "experiment",
                json!({"name": exp.name, "file": exp.file, "tag": exp.tag}),
            );
            put("mode", json!(exp.mode.name()));
            put("dataset", dataset);
            put("panel", panel_json);
            put(
                "qset",
                json!(if rows.source.is_none() {
                    exp.queries.clone()
                } else {
                    None
                }),
            );
            put("rows", json!(rows.len()));
            put("splits", json!(splits));
            put("index", index);
            put("config", config_json);
            put("config_hash", json!(config_hash));
            put("K", json!(k));
            put("efforts", json!(out.efforts));
            put("effort_name", json!("L (search list size)"));
            put("skipped_efforts", json!(out.skipped));
            put(
                "skipped_reason",
                json!("the collector returns at most L answers (L < K)"),
            );
            put("reps", protocol["reps"].clone());
            put("protocol", protocol);
            put("eps_min", json!({"value": eps, "method": eps_method}));
            put("host", json!({"start": host_start, "end": host_end}));
            put("load_avg_start", json!(load_start));
            put("load_avg_end", json!(load_end));
            put(
                "calibration",
                json!({"start": calib_start, "end": calib_end}),
            );
            put("peak_rss_bytes", json!(sys::peak_rss_bytes()));
            put("timing", timing_doc());
            put("counters", counter_docs());
            put("results", summary);
            put("started_at", json!(started_at));
            put("elapsed_s", json!(started.elapsed().as_secs_f64()));
            put("status", json!("ok"));
            put("error", Value::Null);
            let run = Value::Object(run);
            std::fs::write(
                dir.join("run.json"),
                serde_json::to_string_pretty(&run)? + "\n",
            )?;
            if let Some(ledger) = &ledger {
                append_ledger(
                    ledger,
                    &json!({
                        "run_id": run_id,
                        "time": sys::utc_timestamp(false),
                        "status": "ok",
                        "tool": "flab run",
                        "experiment": exp.name,
                        "config": c.name,
                        "config_hash": config_hash,
                        "dataset": exp.dataset.file_name().map(|n| n.to_string_lossy().into_owned()),
                        "panel": rows.source.as_ref().map(|s| &s.0),
                        "panel_hash": rows.source.as_ref().map(|s| &s.1),
                        "split": splits,
                        "K": k,
                        "efforts": out.efforts,
                        "mode": exp.mode.name(),
                        "run_dir": dir,
                        "error": Value::Null,
                    }),
                )?;
            }
            eprintln!("  wrote {}", dir.display());
            dirs.push(dir);
        }
    }
    Ok(dirs)
}

fn timing_doc() -> Value {
    json!({
        "wall_clock": "monotonic clock (std::time::Instant)",
        "cpu_clock": "CLOCK_THREAD_CPUTIME_ID, read outside the wall-clock interval",
        "threads": 1,
        "per_query_scope": "predicate compilation and clause bits, exact-path routing, start selection (incl. sample intersection), pilot, materialization, the search loop, collection, scratch resets and the output copy",
    })
}

fn counter_docs() -> Value {
    json!({
        "n_dist": "distance computations, starts, pilot and auxiliary nodes included",
        "n_pred": "predicate evaluations (navigation, final checks, sample-start checks)",
        "n_expand": "nodes popped and expanded",
        "n_hops": "adjacency lists read (expansions and second-hop reads)",
        "n_match_seen": "evaluated nodes satisfying the navigation predicate",
        "n_out": "answers returned (<= K)",
        "route": "graph | exact | empty",
        "n_aux_dist": "distance computations to auxiliary nodes",
        "n_lazy": "nodes skipped by distance_first",
        "n_aux_expand": "expansions of auxiliary nodes",
        "n_two_hop": "nodes inserted through the second hop",
        "n_evict": "beam entries evicted",
        "n_exact_cand": "exact-path candidates",
        "n_start": "starts selected (deduplicated)",
        "final_L": "expandable beam size at the end (adaptive L)",
    })
}

/// Distinct values of the panel rows' `split` column.
fn split_values(rows: &Rows) -> Vec<String> {
    let Some(header) = &rows.panel_header else {
        return Vec::new();
    };
    let Some(c) = header.split('\t').position(|h| h == "split") else {
        return Vec::new();
    };
    let set: BTreeSet<String> = rows
        .panel_lines
        .iter()
        .filter_map(|l| l.split('\t').nth(c).map(str::to_owned))
        .collect();
    set.into_iter().collect()
}

fn run_k<T: Element>(
    cx: &Ctx<'_, T>,
    searchers: &mut HashMap<GraphKey, Searcher>,
    k: usize,
) -> Result<Vec<Outcome>> {
    let exp = cx.exp;
    let n = cx.rows.len();
    let mut outs: Vec<Outcome> = cx
        .plans
        .iter()
        .map(|p| {
            let (efforts, skipped): (Vec<usize>, Vec<usize>) = exp
                .efforts
                .iter()
                .partition(|&&l| !(p.kernel.needs_l_at_least_k() && l < k));
            Outcome {
                ids: efforts
                    .iter()
                    .map(|&l| (l, vec![u32::MAX; n * k]))
                    .collect(),
                dists: efforts
                    .iter()
                    .map(|&l| (l, vec![f32::MAX; n * k]))
                    .collect(),
                efforts,
                skipped,
                recs: Vec::new(),
                nondeterministic: BTreeMap::new(),
                traces: Vec::new(),
            }
        })
        .collect();
    let (mut tmp_ids, mut tmp_d) = (vec![0u32; k], vec![0f32; k]);

    match exp.mode {
        Mode::Timed => {
            if exp.warmup {
                for (ci, c) in exp.configs.iter().enumerate() {
                    let efforts = &outs[ci].efforts;
                    if efforts.is_empty() {
                        continue;
                    }
                    let l = efforts[efforts.len() / 2];
                    let st = searchers.get_mut(&graph_key(c)).expect("searcher");
                    for q in &cx.queries[ci] {
                        st.search(
                            &cx.plans[ci],
                            &cx.indices[ci],
                            q,
                            l,
                            &mut tmp_ids,
                            &mut tmp_d,
                            None,
                        );
                    }
                }
            }
            let mut order: Vec<usize> = (0..exp.configs.len()).collect();
            for rep in 0..exp.reps {
                for &l in &exp.efforts {
                    order.sort_unstable();
                    let seed = exp.seed ^ fnv1a(format!("{k}/{rep}/{l}").as_bytes());
                    shuffle(&mut order, seed);
                    for &ci in &order {
                        if !outs[ci].efforts.contains(&l) {
                            continue;
                        }
                        let c = &exp.configs[ci];
                        let st = searchers.get_mut(&graph_key(c)).expect("searcher");
                        let (plan, ix) = (&cx.plans[ci], &cx.indices[ci]);
                        let out = &mut outs[ci];
                        let mut nondet = 0;
                        for (row, q) in cx.queries[ci].iter().enumerate() {
                            let (ids, dists): (&mut [u32], &mut [f32]) = if rep == 0 {
                                (
                                    &mut out.ids.get_mut(&l).expect("effort")
                                        [row * k..(row + 1) * k],
                                    &mut out.dists.get_mut(&l).expect("effort")
                                        [row * k..(row + 1) * k],
                                )
                            } else {
                                (&mut tmp_ids, &mut tmp_d)
                            };
                            let cpu0 = sys::thread_cpu_ns();
                            let wall0 = Instant::now();
                            let ctr = st.search(plan, ix, q, l, ids, dists, None);
                            let wall = wall0.elapsed().as_nanos() as u64;
                            let cpu = sys::thread_cpu_ns() - cpu0;
                            if rep > 0
                                && (out.ids[&l][row * k..(row + 1) * k] != tmp_ids[..]
                                    || out.dists[&l][row * k..(row + 1) * k]
                                        .iter()
                                        .zip(&tmp_d)
                                        .any(|(a, b)| a.to_bits() != b.to_bits()))
                            {
                                nondet += 1;
                            }
                            out.recs.push(Rec {
                                effort: l,
                                rep,
                                row,
                                wall_ns: Some(wall),
                                cpu_ns: Some(cpu),
                                ctr,
                                reference: None,
                            });
                        }
                        *out.nondeterministic.entry(l).or_insert(0) += nondet;
                    }
                }
            }
        }
        Mode::Trace | Mode::Reference => {
            let traced: BTreeSet<usize> = match &exp.trace_rows {
                Some(r) => r.iter().copied().filter(|&i| i < n).collect(),
                None => {
                    let mut all: Vec<usize> = (0..n).collect();
                    shuffle(&mut all, exp.seed ^ 0x7ace);
                    all.into_iter().take(exp.trace_sample).collect()
                }
            };
            for (ci, c) in exp.configs.iter().enumerate() {
                let st = searchers.get_mut(&graph_key(c)).expect("searcher");
                let (plan, ix) = (&cx.plans[ci], &cx.indices[ci]);
                let out = &mut outs[ci];
                for l in out.efforts.clone() {
                    for (row, q) in cx.queries[ci].iter().enumerate() {
                        let ids = &mut out.ids.get_mut(&l).expect("effort")[row * k..(row + 1) * k];
                        let dists =
                            &mut out.dists.get_mut(&l).expect("effort")[row * k..(row + 1) * k];
                        let (ctr, reference) = if exp.mode == Mode::Reference {
                            let base = st.search(plan, ix, q, l, ids, dists, None);
                            let budget = ((base.n_dist as f64 * exp.reference_factor) as u64)
                                .max(exp.reference_min);
                            let ctr = st.search(plan, ix, q, l, ids, dists, Some(budget));
                            (ctr, Some((budget, base.n_dist)))
                        } else {
                            let trace = traced.contains(&row);
                            st.trace = trace.then(Trace::new);
                            let ctr = st.search(plan, ix, q, l, ids, dists, None);
                            if let Some(t) = st.trace.take() {
                                out.traces.push((l, row, t));
                            }
                            (ctr, None)
                        };
                        out.recs.push(Rec {
                            effort: l,
                            rep: 0,
                            row,
                            wall_ns: None,
                            cpu_ns: None,
                            ctr,
                            reference,
                        });
                    }
                }
            }
        }
    }
    Ok(outs)
}

/// Writes `res_<effort>.bin`, `perquery.tsv`, `panel.tsv` and traces; returns the per-effort
/// summary for `run.json`.
fn write_run_files(dir: &Path, rows: &Rows, k: usize, out: &Outcome) -> Result<Value> {
    let n = rows.len();
    for &l in &out.efforts {
        let mut buf = Vec::with_capacity(8 + n * k * 8);
        buf.extend_from_slice(&(n as u32).to_le_bytes());
        buf.extend_from_slice(&(k as u32).to_le_bytes());
        buf.extend_from_slice(bytemuck::cast_slice(&out.ids[&l]));
        buf.extend_from_slice(bytemuck::cast_slice(&out.dists[&l]));
        std::fs::write(dir.join(format!("res_{l}.bin")), buf)?;
    }
    // The rows used, as a panel the evaluator can read.
    let mut panel = String::new();
    match &rows.panel_header {
        Some(h) => {
            panel.push_str(h);
            panel.push('\n');
            for line in &rows.panel_lines {
                panel.push_str(line);
                panel.push('\n');
            }
        }
        None => {
            panel.push_str("qset\tqidx\n");
            for (q, i) in &rows.rows {
                panel.push_str(&format!("{q}\t{i}\n"));
            }
        }
    }
    std::fs::write(dir.join("panel.tsv"), panel)?;

    let names = Counters::NAMES;
    let mut tsv = String::from("effort\trep\trow\twall_ns\tcpu_ns");
    for name in &names[..6] {
        tsv.push('\t');
        tsv.push_str(name);
    }
    tsv.push_str("\troute");
    for name in &names[6..] {
        tsv.push('\t');
        tsv.push_str(name);
    }
    let reference = out.recs.iter().any(|r| r.reference.is_some());
    if reference {
        tsv.push_str("\tref_budget\tbase_n_dist");
    }
    tsv.push('\n');
    for r in &out.recs {
        let v = r.ctr.values();
        let opt = |x: Option<u64>| x.map_or_else(String::new, |x| x.to_string());
        tsv.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}",
            r.effort,
            r.rep,
            r.row,
            opt(r.wall_ns),
            opt(r.cpu_ns)
        ));
        for x in &v[..6] {
            tsv.push_str(&format!("\t{x}"));
        }
        tsv.push('\t');
        tsv.push_str(r.ctr.route.name());
        for x in &v[6..] {
            tsv.push_str(&format!("\t{x}"));
        }
        if let Some((b, d)) = r.reference {
            tsv.push_str(&format!("\t{b}\t{d}"));
        } else if reference {
            tsv.push_str("\t\t");
        }
        tsv.push('\n');
    }
    std::fs::write(dir.join("perquery.tsv"), tsv)?;

    if !out.traces.is_empty() {
        let tdir = dir.join("trace");
        std::fs::create_dir_all(&tdir)?;
        for (l, row, t) in &out.traces {
            let f = std::fs::File::create(tdir.join(format!("{l}_{row}.tsv")))?;
            t.write_tsv(std::io::BufWriter::new(f))?;
        }
    }

    let mut summary = serde_json::Map::new();
    for &l in &out.efforts {
        let recs: Vec<&Rec> = out.recs.iter().filter(|r| r.effort == l).collect();
        let reps: BTreeSet<usize> = recs.iter().map(|r| r.rep).collect();
        let mean_wall: Vec<f64> = reps
            .iter()
            .filter_map(|&rep| {
                let w: Vec<u64> = recs
                    .iter()
                    .filter(|r| r.rep == rep)
                    .filter_map(|r| r.wall_ns)
                    .collect();
                (!w.is_empty()).then(|| w.iter().sum::<u64>() as f64 / w.len() as f64)
            })
            .collect();
        let mean_cpu: Vec<f64> = reps
            .iter()
            .filter_map(|&rep| {
                let w: Vec<u64> = recs
                    .iter()
                    .filter(|r| r.rep == rep)
                    .filter_map(|r| r.cpu_ns)
                    .collect();
                (!w.is_empty()).then(|| w.iter().sum::<u64>() as f64 / w.len() as f64)
            })
            .collect();
        let median = |v: &[f64]| {
            let mut s = v.to_vec();
            s.sort_unstable_by(f64::total_cmp);
            s.get(s.len() / 2).copied()
        };
        let first: Vec<&&Rec> = recs.iter().filter(|r| r.rep == 0).collect();
        let mean_ctr = |i: usize| {
            first.iter().map(|r| r.ctr.values()[i] as f64).sum::<f64>() / first.len().max(1) as f64
        };
        let mut counters = serde_json::Map::new();
        for (i, name) in Counters::NAMES.iter().enumerate() {
            counters.insert((*name).to_owned(), json!(mean_ctr(i)));
        }
        let mw = median(&mean_wall);
        summary.insert(
            l.to_string(),
            json!({
                "file": format!("res_{l}.bin"),
                "mean_wall_ns_per_rep": mean_wall,
                "mean_cpu_ns_per_rep": mean_cpu,
                "median_mean_wall_ns": mw,
                "qps_wall": mw.map(|w| 1e9 / w),
                "qps_cpu": median(&mean_cpu).map(|w| 1e9 / w),
                "mean_counters_rep0": counters,
                "nondeterministic_rows": out.nondeterministic.get(&l).copied().unwrap_or(0),
            }),
        );
    }
    Ok(Value::Object(summary))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_is_seeded() {
        let mut a: Vec<usize> = (0..10).collect();
        let mut b = a.clone();
        shuffle(&mut a, 7);
        shuffle(&mut b, 7);
        assert_eq!(a, b);
        let mut c: Vec<usize> = (0..10).collect();
        shuffle(&mut c, 8);
        assert_ne!(a, c);
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }
}
