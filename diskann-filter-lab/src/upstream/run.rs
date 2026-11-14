/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Timed runs of upstream algorithms with per-query outputs.
//!
//! One invocation runs several configurations ([`RunSpec`]: algorithm × provider × K × effort
//! grid) over the same index and queries, single-threaded on the calling thread:
//!
//! 1. one untimed warm-up pass per configuration and effort; its results are the reference;
//! 2. `reps` timed passes, interleaved: repetition-major over all configurations and efforts;
//! 3. per query, everything is inside the timer: predicate compilation (clause ordering, the
//!    bitmap provider's construction from postings), the upstream search with its default
//!    post-processing, the adapter's post-filter (`knn-postfilter`, `beta`) and the copy of
//!    the results.
//!
//! Each configuration writes a run directory with `run.json`, `res_<effort>.bin` (DiskANN
//! truthset layout, rows in query order, true squared-L2 distances), `perquery.tsv`, and
//! appends one line to the ledger.

use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use anyhow::{Context, Result};
use diskann::utils::VectorRepr;
use serde_json::{json, Value};

use super::{
    filter::{BitmapFilter, FilterCounts, ForwardFilter, PredicateFilter},
    index::LoadedIndex,
    queries::Queries,
    search::{Algorithm, QueryStats, Searcher},
    sys,
};
use crate::{labels::LabelStore, predicate::Cnf, vectors::Element};

/// How the predicate is answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    /// [`ForwardFilter`]: CNF evaluated on the point's label row.
    Forward,
    /// [`BitmapFilter`]: dense per-query bitmap built from postings.
    Bitmap,
}

impl ProviderKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Bitmap => "bitmap",
        }
    }
}

/// One configuration of a run.
#[derive(Debug, Clone)]
pub struct RunSpec {
    pub algorithm: Algorithm,
    pub provider: ProviderKind,
    pub k: usize,
    /// Search list sizes `L` (each `>= k`).
    pub efforts: Vec<usize>,
}

impl RunSpec {
    /// The resolved configuration, as recorded in `run.json` and hashed into the run id.
    pub fn config(&self) -> Value {
        let (adaptive, beta) = match &self.algorithm {
            Algorithm::InlineAdaptive {
                sample_count,
                scale_factor,
                ..
            } => (
                json!({"sample_count": sample_count, "scale_factor": scale_factor}),
                Value::Null,
            ),
            Algorithm::Beta { beta } => (Value::Null, json!(beta)),
            _ => (Value::Null, Value::Null),
        };
        json!({
            "algorithm": self.algorithm.name(),
            "adaptive_l": adaptive,
            "beta": beta,
            "provider": self.provider.name(),
            "clause_order": "ascending total posting length (most selective first)",
            "k": self.k,
            "efforts": self.efforts,
            "effort_kind": "L (search list size)",
            "beam_width": 1,
            "post_processor": "strategy default (start points and deleted ids removed)",
        })
    }
}

/// Fixed context shared by all configurations of an invocation.
#[derive(Debug, Clone)]
pub struct RunContext {
    pub runs_dir: PathBuf,
    pub ledger: PathBuf,
    pub dataset_name: String,
    /// Dataset, query source, index and tool descriptions recorded in `run.json`.
    pub dataset: Value,
    pub queries: Value,
    pub index: Value,
    pub tool: Value,
    /// Panel file and hash, if any (ledger).
    pub panel: Option<(String, String)>,
    pub reps: usize,
    pub tag: Option<String>,
}

/// Outcome of one configuration.
#[derive(Debug)]
pub struct RunOutcome {
    pub run_id: String,
    pub dir: PathBuf,
    pub summary: Vec<EffortSummary>,
}

/// Aggregates of one configuration at one effort.
#[derive(Debug, Clone)]
pub struct EffortSummary {
    pub effort: usize,
    /// Per timed repetition: mean wall and CPU time per query (ns).
    pub mean_wall_ns: Vec<f64>,
    pub mean_cpu_ns: Vec<f64>,
    pub mean_prep_ns: f64,
    pub mean_dist: f64,
    pub mean_pred: f64,
    pub mean_out: f64,
    /// Rows whose results differed between a timed repetition and the warm-up pass.
    pub nondeterministic_rows: usize,
}

impl EffortSummary {
    /// Median over repetitions of the mean wall time per query (ns).
    pub fn median_wall_ns(&self) -> f64 {
        median(&self.mean_wall_ns)
    }

    pub fn median_cpu_ns(&self) -> f64 {
        median(&self.mean_cpu_ns)
    }
}

fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    match v.len() {
        0 => f64::NAN,
        n if n % 2 == 1 => v[n / 2],
        n => 0.5 * (v[n / 2 - 1] + v[n / 2]),
    }
}

#[derive(Debug)]
enum AnyFilter {
    Forward(Arc<ForwardFilter>),
    Bitmap(Arc<BitmapFilter>),
}

/// Measurements of one query.
#[derive(Debug, Clone, Copy, Default)]
struct Record {
    wall_ns: u64,
    cpu_ns: u64,
    prep_ns: u64,
    stats: QueryStats,
    counts: FilterCounts,
    prep_entries: u64,
}

#[expect(
    clippy::too_many_arguments,
    reason = "one timed query; all inputs are needed"
)]
fn timed_query<T: VectorRepr, F: PredicateFilter>(
    searcher: &mut Searcher<'_, T>,
    algorithm: &Algorithm,
    l: usize,
    query: &[T],
    cnf: Cnf<'_>,
    filter: &mut Arc<F>,
    ids: &mut [u32],
    distances: &mut [f32],
) -> Result<Record> {
    let cpu0 = sys::thread_cpu_ns();
    let t0 = Instant::now();
    Arc::get_mut(filter)
        .context("the predicate provider is still shared after the previous query")?
        .compile(cnf);
    let prep = t0.elapsed();
    let stats = searcher.search(algorithm, l, query, filter, ids, distances)?;
    let wall = t0.elapsed();
    let cpu1 = sys::thread_cpu_ns();
    Ok(Record {
        wall_ns: wall.as_nanos() as u64,
        cpu_ns: cpu1.saturating_sub(cpu0),
        prep_ns: prep.as_nanos() as u64,
        stats,
        counts: filter.counts(),
        prep_entries: filter.prep_entries(),
    })
}

/// Per-configuration state during a run.
#[derive(Debug)]
struct State {
    spec: RunSpec,
    run_id: String,
    dir: PathBuf,
    config_hash: String,
    filter: AnyFilter,
    /// Per effort: reference results of the warm-up pass (`nq × k`).
    ids: Vec<Vec<u32>>,
    distances: Vec<Vec<f32>>,
    /// Per effort: rows that differed in some timed repetition.
    differs: Vec<Vec<bool>>,
    /// Per effort and repetition: sums of wall, CPU.
    wall: Vec<Vec<u64>>,
    cpu: Vec<Vec<u64>>,
    /// Per effort: sums over all timed queries of prep time, distances, predicates, outputs.
    sums: Vec<[u64; 4]>,
    perquery: BufWriter<File>,
}

const PERQUERY_HEADER: &str =
    "effort\trep\trow\twall_ns\tcpu_ns\tn_dist\tn_pred\tn_expand\tn_hops\t\
n_match_seen\tn_out\troute\tup_cmps\tup_hops\tup_results\tn_start\tpost_pred\textra_dist\t\
prep_ns\tprep_entries";

impl State {
    fn n_dist(&self, r: &Record, n_start: u32) -> u64 {
        let start = match self.spec.algorithm {
            // Inline and multihop do not count the start-point distances in `cmps`.
            Algorithm::Inline | Algorithm::InlineAdaptive { .. } | Algorithm::Multihop => n_start,
            _ => 0,
        };
        u64::from(r.stats.up_cmps) + u64::from(start) + u64::from(r.stats.extra_dist)
    }

    fn write_row(
        &mut self,
        effort: usize,
        rep: usize,
        row: usize,
        r: &Record,
        n_start: u32,
    ) -> Result<()> {
        // Multihop's `hops` mixes expansions with second-hop adjacency fetches.
        let n_expand = match self.spec.algorithm {
            Algorithm::Multihop => String::new(),
            _ => r.stats.up_hops.to_string(),
        };
        let n_dist = self.n_dist(r, n_start);
        writeln!(
            self.perquery,
            "{effort}\t{rep}\t{row}\t{}\t{}\t{n_dist}\t{}\t{n_expand}\t{}\t{}\t{}\tgraph\t{}\t{}\t{}\t{n_start}\t{}\t{}\t{}\t{}",
            r.wall_ns,
            r.cpu_ns,
            r.counts.calls,
            r.stats.up_hops,
            r.counts.matches,
            r.stats.n_out,
            r.stats.up_cmps,
            r.stats.up_hops,
            r.stats.up_results,
            r.stats.post_pred,
            r.stats.extra_dist,
            r.prep_ns,
            r.prep_entries,
        )?;
        Ok(())
    }
}

fn unique_dir(runs_dir: &Path, base: &str) -> Result<(String, PathBuf)> {
    for i in 1.. {
        let id = if i == 1 {
            base.to_owned()
        } else {
            format!("{base}-{i}")
        };
        let dir = runs_dir.join(&id);
        match std::fs::create_dir(&dir) {
            Ok(()) => return Ok((id, dir)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                return Err(e).with_context(|| format!("creating {}", dir.display()));
            }
        }
    }
    unreachable!()
}

fn append_ledger(path: &Path, line: &Value) -> Result<()> {
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening the ledger {}", path.display()))?;
    f.write_all(format!("{line}\n").as_bytes())?;
    Ok(())
}

fn write_truthset(path: &Path, k: usize, ids: &[u32], distances: &[f32]) -> Result<()> {
    let nq = ids.len() / k;
    let mut w =
        BufWriter::new(File::create(path).with_context(|| format!("creating {}", path.display()))?);
    w.write_all(&(nq as u32).to_le_bytes())?;
    w.write_all(&(k as u32).to_le_bytes())?;
    w.write_all(bytemuck::cast_slice(ids))?;
    w.write_all(bytemuck::cast_slice(distances))?;
    w.flush()?;
    Ok(())
}

/// Runs every configuration of `specs` (see the module documentation) and writes their run
/// directories and ledger lines. A failure is recorded in the ledger (and in `run.json`) of
/// every configuration before it is returned.
pub fn run<T: VectorRepr + Element>(
    index: &LoadedIndex<T>,
    labels: &Arc<LabelStore>,
    queries: &Queries<T>,
    specs: &[RunSpec],
    ctx: &RunContext,
) -> Result<Vec<RunOutcome>> {
    anyhow::ensure!(!queries.is_empty(), "no queries");
    anyhow::ensure!(
        labels.n_points() == index.n_data,
        "the labels have {} points but the index has {} data points",
        labels.n_points(),
        index.n_data
    );
    for spec in specs {
        anyhow::ensure!(
            spec.k > 0 && spec.efforts.iter().all(|&l| l >= spec.k),
            "{}: every L must be at least K = {}",
            spec.algorithm.name(),
            spec.k
        );
    }
    std::fs::create_dir_all(&ctx.runs_dir)
        .with_context(|| format!("creating {}", ctx.runs_dir.display()))?;
    let runs_dir = ctx.runs_dir.canonicalize()?;
    let stamp = sys::utc_timestamp(true);
    let nq = queries.len();
    let mut states = Vec::with_capacity(specs.len());
    for spec in specs {
        let config = json!({
            "config": spec.config(),
            "dataset": ctx.dataset,
            "queries": ctx.queries,
            "index": ctx.index["prefix"],
        });
        let config_hash = sys::fnv1a_hex(config.to_string().as_bytes());
        let base = format!(
            "{stamp}-up-{}-{}-k{}-{}",
            spec.algorithm.name(),
            spec.provider.name(),
            spec.k,
            &config_hash[..8]
        );
        let (run_id, dir) = unique_dir(&runs_dir, &base)?;
        let mut perquery = BufWriter::new(File::create(dir.join("perquery.tsv"))?);
        writeln!(perquery, "{PERQUERY_HEADER}")?;
        let filter = match spec.provider {
            ProviderKind::Forward => {
                AnyFilter::Forward(Arc::new(ForwardFilter::new(labels.clone())))
            }
            ProviderKind::Bitmap => AnyFilter::Bitmap(Arc::new(BitmapFilter::new(labels.clone()))),
        };
        let ne = spec.efforts.len();
        states.push(State {
            spec: spec.clone(),
            run_id,
            dir,
            config_hash,
            filter,
            ids: vec![vec![u32::MAX; nq * spec.k]; ne],
            distances: vec![vec![f32::MAX; nq * spec.k]; ne],
            differs: vec![vec![false; nq]; ne],
            wall: vec![vec![0; ctx.reps]; ne],
            cpu: vec![vec![0; ctx.reps]; ne],
            sums: vec![[0; 4]; ne],
            perquery,
        });
    }

    let started = sys::utc_timestamp(false);
    let t_start = Instant::now();
    let load_start = sys::load_average();
    let host_start = sys::host_info();
    let calibration_start = sys::calibrate();
    let result = execute(index, queries, &mut states, ctx.reps);
    let calibration_end = sys::calibrate();
    let load_end = sys::load_average();
    let host_end = sys::host_info();
    let elapsed = t_start.elapsed().as_secs_f64();

    let error = result.as_ref().err().map(|e| format!("{e:#}"));
    let splits: Option<Vec<String>> = {
        let mut s: Vec<String> = queries.splits.iter().flatten().cloned().collect();
        s.sort();
        s.dedup();
        (!s.is_empty()).then_some(s)
    };
    let interleaved: Vec<String> = states.iter().map(|s| s.run_id.clone()).collect();
    let mut outcomes = Vec::new();
    for state in &mut states {
        state.perquery.flush()?;
        let mut summaries = Vec::new();
        let mut results = serde_json::Map::new();
        if error.is_none() {
            for (e, &effort) in state.spec.efforts.iter().enumerate() {
                let file = format!("res_{effort}.bin");
                write_truthset(
                    &state.dir.join(&file),
                    state.spec.k,
                    &state.ids[e],
                    &state.distances[e],
                )?;
                let n_timed = (nq * ctx.reps) as f64;
                let s = EffortSummary {
                    effort,
                    mean_wall_ns: state.wall[e]
                        .iter()
                        .map(|&w| w as f64 / nq as f64)
                        .collect(),
                    mean_cpu_ns: state.cpu[e].iter().map(|&c| c as f64 / nq as f64).collect(),
                    mean_prep_ns: state.sums[e][0] as f64 / n_timed,
                    mean_dist: state.sums[e][1] as f64 / n_timed,
                    mean_pred: state.sums[e][2] as f64 / n_timed,
                    mean_out: state.sums[e][3] as f64 / n_timed,
                    nondeterministic_rows: state.differs[e].iter().filter(|&&d| d).count(),
                };
                results.insert(
                    effort.to_string(),
                    json!({
                        "file": file,
                        "mean_wall_ns_per_rep": s.mean_wall_ns,
                        "mean_cpu_ns_per_rep": s.mean_cpu_ns,
                        "median_mean_wall_ns": s.median_wall_ns(),
                        "qps_wall": 1e9 / s.median_wall_ns(),
                        "qps_cpu": 1e9 / s.median_cpu_ns(),
                        "mean_prep_ns": s.mean_prep_ns,
                        "mean_n_dist": s.mean_dist,
                        "mean_n_pred": s.mean_pred,
                        "mean_n_out": s.mean_out,
                        "nondeterministic_rows": s.nondeterministic_rows,
                    }),
                );
                summaries.push(s);
            }
        }
        let status = if error.is_none() { "ok" } else { "failed" };
        let run_json = json!({
            "run_id": state.run_id,
            "status": status,
            "error": error,
            "mode": "timed",
            "tool": ctx.tool,
            "tag": ctx.tag,
            "dataset": ctx.dataset,
            "queries": ctx.queries,
            "n_queries": nq,
            "splits": splits,
            "index": ctx.index,
            "config": state.spec.config(),
            "config_hash": state.config_hash,
            "reps": ctx.reps,
            "warmup_passes": 1,
            "interleaving": "repetition-major over the configurations and efforts of one invocation",
            "interleaved_runs": interleaved,
            "timing": {
                "threads": 1,
                "wall_clock": "monotonic clock (std::time::Instant)",
                "cpu_clock": "CLOCK_THREAD_CPUTIME_ID",
                "per_query_scope": "predicate compilation (clause ordering; bitmap construction from postings), upstream search with the strategy's default post-processing, the adapter's post-filter (knn-postfilter, beta) and the result copy",
            },
            "counters": {
                "n_dist": "upstream cmps + start-point distances where cmps excludes them (inline, multihop) + adapter distance computations (beta: true distances of the returned matches)",
                "n_pred": "predicate evaluations (is_match calls), including the adapter's post-filter",
                "n_expand": "upstream hops (expanded nodes; beam width 1); empty for multihop, whose hops include second-hop adjacency fetches",
                "n_hops": "upstream hops",
                "n_match_seen": "predicate evaluations that returned true (frozen start points never match)",
                "n_out": "results returned (<= K)",
                "route": "graph (no exact path)",
                "prep_ns": "wall time of predicate compilation, included in wall_ns",
                "prep_entries": "forward: clause labels copied; bitmap: posting entries scattered",
            },
            "host": {"start": host_start, "end": host_end},
            "load_avg_start": load_start,
            "load_avg_end": load_end,
            "calibration": {
                "kernel": "squared L2, 64-d u8, 256 KiB buffer, best of 5 passes",
                "start_ns_per_dist": calibration_start,
                "end_ns_per_dist": calibration_end,
            },
            "started_at": started,
            "elapsed_s": elapsed,
            "peak_rss_bytes": sys::peak_rss_bytes(),
            "results": results,
        });
        std::fs::write(
            state.dir.join("run.json"),
            serde_json::to_string_pretty(&run_json)?,
        )?;
        append_ledger(
            &ctx.ledger,
            &json!({
                "run_id": state.run_id,
                "time": sys::utc_timestamp(false),
                "status": status,
                "error": error,
                "tool": "flab-upstream",
                "config_hash": state.config_hash,
                "dataset": ctx.dataset_name,
                "panel": ctx.panel.as_ref().map(|p| &p.0),
                "panel_hash": ctx.panel.as_ref().map(|p| &p.1),
                "split": splits,
                "algorithm": state.spec.algorithm.name(),
                "provider": state.spec.provider.name(),
                "k": state.spec.k,
                "efforts": state.spec.efforts,
                "run_dir": state.dir,
            }),
        )?;
        outcomes.push(RunOutcome {
            run_id: state.run_id.clone(),
            dir: state.dir.clone(),
            summary: summaries,
        });
    }
    result.map(|()| outcomes)
}

fn execute<T: VectorRepr + Element>(
    index: &LoadedIndex<T>,
    queries: &Queries<T>,
    states: &mut [State],
    reps: usize,
) -> Result<()> {
    let max_l = states
        .iter()
        .flat_map(|s| s.spec.efforts.iter().copied())
        .max()
        .unwrap_or(1);
    let max_k = states.iter().map(|s| s.spec.k).max().unwrap_or(1);
    let mut searcher = Searcher::new(&index.index, max_l);
    let n_start = index.start_ids.len() as u32;
    let mut row_ids = vec![0u32; max_k];
    let mut row_distances = vec![0f32; max_k];
    let nq = queries.len();

    // `rep == None` is the untimed warm-up pass; its results are the reference.
    for rep in std::iter::once(None).chain((0..reps).map(Some)) {
        for state in states.iter_mut() {
            let k = state.spec.k;
            let algorithm = state.spec.algorithm.clone();
            for (e, &l) in state.spec.efforts.clone().iter().enumerate() {
                for row in 0..nq {
                    let (ids, distances) = match rep {
                        None => (
                            &mut state.ids[e][row * k..(row + 1) * k],
                            &mut state.distances[e][row * k..(row + 1) * k],
                        ),
                        Some(_) => (&mut row_ids[..k], &mut row_distances[..k]),
                    };
                    let (query, cnf) = (queries.vector(row), queries.predicate(row));
                    let r = match &mut state.filter {
                        AnyFilter::Forward(f) => {
                            timed_query(&mut searcher, &algorithm, l, query, cnf, f, ids, distances)
                        }
                        AnyFilter::Bitmap(f) => {
                            timed_query(&mut searcher, &algorithm, l, query, cnf, f, ids, distances)
                        }
                    }
                    .with_context(|| {
                        format!(
                            "{} / {}: L = {l}, row {row}",
                            algorithm.name(),
                            state.spec.provider.name()
                        )
                    })?;
                    let Some(rep) = rep else { continue };
                    let reference = &state.ids[e][row * k..(row + 1) * k];
                    if reference != &row_ids[..k] {
                        state.differs[e][row] = true;
                    }
                    state.wall[e][rep] += r.wall_ns;
                    state.cpu[e][rep] += r.cpu_ns;
                    let n_dist = state.n_dist(&r, n_start);
                    let sums = &mut state.sums[e];
                    sums[0] += r.prep_ns;
                    sums[1] += n_dist;
                    sums[2] += r.counts.calls;
                    sums[3] += u64::from(r.stats.n_out);
                    state.write_row(l, rep, row, &r, n_start)?;
                }
            }
        }
    }
    Ok(())
}
