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
    bitsets::LabelBitsets,
    islands::IslandIndex,
    kernel::{
        config::{
            Crossover, EstimateSource, Materializer, PredicateMode, StartSource, Terminator,
            TreeRoute,
        },
        cost::UnitCosts,
        entry::SampleIndex,
        profile,
        search::{Counters, FastPath, Index, Plan, Query, Searcher},
        trace::Trace,
    },
    pairs::PairIndex,
    provenance::{self, BaseShape, Checker, Expect, GraphFp, GraphPart, LabelsFp},
    sketches::LabelSketches,
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
    /// A sidecar or calibration built from other inputs (provenance) is a warning, not an
    /// error.
    pub allow_provenance_mismatch: bool,
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

/// The draw in [0, 1) of row `row` of a `mix` configuration (splitmix64 of `seed ^ row·φ`).
pub fn mix_draw(seed: u64, row: usize) -> f64 {
    let mut z = seed ^ (row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
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
    // Provenance: the fingerprints every sidecar and calibration is compared with.
    let labels_fp = LabelsFp::of(&ds.labels);
    let base_shape = BaseShape::of(&ds.base);
    let cpu_model = provenance::cpu_model();
    let mut prov = Checker::new(opts.allow_provenance_mismatch);
    let labels_only = Expect {
        labels: Some(&labels_fp),
        ..Expect::default()
    };

    // Graphs, start tables and sample indices.
    let mut graphs: HashMap<GraphKey, LoadedGraph> = HashMap::new();
    let mut graph_files = std::collections::HashMap::new();
    for c in &exp.configs {
        let key = graph_key(c);
        if !graphs.contains_key(&key) {
            let g = data::load_graph_shared(
                &c.graph,
                c.starts_sidecar.as_deref(),
                T::DTYPE,
                &mut graph_files,
            )
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
    let mut graph_fps: HashMap<PathBuf, GraphFp> = HashMap::new();
    for g in graphs.values() {
        if !graph_fps.contains_key(&g.path) {
            graph_fps.insert(g.path.clone(), GraphFp::of(&g.graph));
        }
        if let (Some(s), Some(p)) = (&g.starts, &g.starts_path) {
            // A table of the graph's node count may name its auxiliary nodes; a table of
            // another graph holds data-point starts only (checked by the loader).
            let e = Expect {
                labels: Some(&labels_fp),
                graph: (s.n_nodes() == g.graph.n_nodes())
                    .then(|| (&graph_fps[&g.path], GraphPart::Full)),
                ..Expect::default()
            };
            prov.file(
                &format!("start table {} of graph {}", p.display(), g.path.display()),
                p,
                crate::starts::MAGIC,
                crate::starts::N_SECTIONS,
                &e,
            )?;
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

    // Static per-label bitsets (untimed T1 sidecar) for `static_bits(θ)` and `dense_static`
    // configurations: loaded, or built at the smallest θ of the `static_bits(θ)`
    // configurations (⌈n/64⌉ without one: a bitset then costs at most as many words as its
    // postings).
    let theta_min = exp
        .configs
        .iter()
        .filter_map(|c| match c.kernel.predicate {
            PredicateMode::StaticBits(t) => Some(t),
            _ => None,
        })
        .min()
        .or_else(|| {
            exp.configs
                .iter()
                .any(|c| {
                    matches!(
                        c.kernel.materializer,
                        Materializer::DenseStatic | Materializer::DenseNav
                    ) || uses_exact_bitsets(c)
                        || matches!(c.kernel.tree_route, TreeRoute::StaticBits | TreeRoute::Cost)
                })
                .then(|| ds.labels.n_points().div_ceil(64) as u64)
        });
    let (bitsets, bitsets_json) = match (theta_min, &exp.bitsets_sidecar) {
        (None, _) => (None, Value::Null),
        (Some(_), Some(path)) => {
            let t = Instant::now();
            let b = LabelBitsets::read(path)?;
            prov.file(
                &format!("bitsets_sidecar {}", path.display()),
                path,
                crate::bitsets::MAGIC,
                crate::bitsets::N_SECTIONS,
                &labels_only,
            )?;
            b.check(&ds.labels)
                .with_context(|| format!("{} vs the dataset's labels", path.display()))?;
            let doc = json!({
                "source": path, "theta": b.theta(), "n_bitsets": b.n_bitsets(),
                "bytes": b.bytes(), "load_s": t.elapsed().as_secs_f64(),
            });
            (Some(b), doc)
        }
        (Some(theta), None) => {
            let t = Instant::now();
            let b = LabelBitsets::build(&ds.labels, theta);
            let doc = json!({
                "source": "built in memory", "theta": b.theta(), "n_bitsets": b.n_bitsets(),
                "bytes": b.bytes(), "build_s": t.elapsed().as_secs_f64(),
            });
            (Some(b), doc)
        }
    };

    // Unit costs of the cost model (per configuration) and bottom-b sketches (untimed
    // sidecars).
    let mut cost_docs: BTreeMap<PathBuf, Value> = BTreeMap::new();
    let mut cost_files: BTreeMap<PathBuf, (UnitCosts, Value)> = BTreeMap::new();
    for c in &exp.configs {
        if let Some(path) = &c.calibration {
            if !cost_files.contains_key(path) {
                let (u, file) = read_costs_doc(path)?;
                let doc = json!({"source": path, "costs": &u});
                cost_files.insert(path.clone(), (u, doc));
                cost_docs.insert(path.clone(), file);
            }
            // A calibration is bound to the data points' graph (the routers of a Steiner graph
            // leave its unit costs and effort curve unchanged), the base and the machine.
            let g = &graphs[&graph_key(c)];
            let e = Expect {
                labels: Some(&labels_fp),
                graph: Some((&graph_fps[&g.path], GraphPart::Data)),
                base: Some(&base_shape),
                cpu_model: cpu_model.as_deref(),
            };
            prov.calibration(
                &format!(
                    "calibration {} with graph {}",
                    path.display(),
                    g.path.display()
                ),
                &cost_docs[path],
                &e,
            )?;
        }
    }
    // `effort_curve`: target recalls beyond the measured effort curve plan its largest measured
    // list (the curve's maximum) — said once per configuration, before the runs.
    for c in &exp.configs {
        let Some(costs) = c.calibration.as_ref().map(|p| &cost_files[p].0) else {
            continue;
        };
        for &k in &exp.ks {
            let capped =
                effort_curve_capped(c, costs, c.efforts.as_ref().unwrap_or(&exp.efforts), k);
            if !capped.is_empty() {
                eprintln!(
                    "warning: config {:?}, K = {k}: effort(s) {capped:?} reach beyond the calibrated effort curve (measured up to L = {}) for some match rates: those lists are capped there, at the curve's maximum recall",
                    c.name, costs.effort_l_top
                );
            }
        }
    }
    // Sketches per file (the experiment's, or a configuration's own).
    let mut sketch_files: BTreeMap<PathBuf, (LabelSketches, Value)> = BTreeMap::new();
    for c in &exp.configs {
        if let Some(path) = &c.sketches_sidecar {
            if !sketch_files.contains_key(path) {
                let t = Instant::now();
                let sk = LabelSketches::read(path)?;
                prov.file(
                    &format!("sketches_sidecar {}", path.display()),
                    path,
                    crate::sketches::MAGIC,
                    crate::sketches::N_SECTIONS,
                    &labels_only,
                )?;
                sk.check_sample(&ds.labels, 97)
                    .with_context(|| format!("{} vs the dataset's labels", path.display()))?;
                let doc = json!({
                    "source": path, "b": sk.b(), "seed": sk.seed(), "entries": sk.n_entries(),
                    "bytes": sk.bytes(), "load_s": t.elapsed().as_secs_f64(),
                });
                sketch_files.insert(path.clone(), (sk, doc));
            }
        }
    }
    let sketches_doc = |c: &ConfigSpec| -> Value {
        c.sketches_sidecar
            .as_ref()
            .map_or(Value::Null, |p| sketch_files[p].1.clone())
    };
    // Combination postings per file (untimed sidecars).
    let mut pair_files: BTreeMap<PathBuf, (PairIndex, Value)> = BTreeMap::new();
    for c in &exp.configs {
        if let Some(path) = &c.pairs_sidecar {
            if !pair_files.contains_key(path) {
                let t = Instant::now();
                let px = PairIndex::read(path)?;
                prov.file(
                    &format!("pairs_sidecar {}", path.display()),
                    path,
                    crate::pairs::MAGIC,
                    crate::pairs::N_SECTIONS,
                    &labels_only,
                )?;
                px.check_sample(&ds.labels, 997)
                    .with_context(|| format!("{} vs the dataset's labels", path.display()))?;
                let doc = json!({
                    "source": path, "t_big": px.t_big(), "n_big": px.n_big(),
                    "pairs": px.n_pairs(), "entries": px.n_entries(), "bytes": px.bytes(),
                    "load_s": t.elapsed().as_secs_f64(),
                });
                pair_files.insert(path.clone(), (px, doc));
            }
        }
    }
    let pairs_doc = |c: &ConfigSpec| -> Value {
        c.pairs_sidecar
            .as_ref()
            .map_or(Value::Null, |p| pair_files[p].1.clone())
    };
    // Per-label sub-graphs per file (untimed sidecars).
    let mut island_files: BTreeMap<PathBuf, (IslandIndex, Value)> = BTreeMap::new();
    for c in &exp.configs {
        if let Some(path) = &c.islands_sidecar {
            if !island_files.contains_key(path) {
                let t = Instant::now();
                let isl = IslandIndex::read(path)?;
                prov.file(
                    &format!("islands_sidecar {}", path.display()),
                    path,
                    crate::islands::MAGIC,
                    crate::islands::N_SECTIONS,
                    &labels_only,
                )?;
                isl.check(&ds.labels)
                    .with_context(|| format!("{} vs the dataset's labels", path.display()))?;
                let doc = json!({
                    "source": path, "islands": isl.n_islands(), "members": isl.n_members(),
                    "edges": isl.n_edges(), "degree_bound": isl.degree_bound(),
                    "bytes": isl.bytes(), "load_s": t.elapsed().as_secs_f64(),
                });
                island_files.insert(path.clone(), (isl, doc));
            }
        }
    }
    let islands_doc = |c: &ConfigSpec| -> Value {
        c.islands_sidecar
            .as_ref()
            .map_or(Value::Null, |p| island_files[p].1.clone())
    };
    // Roots of the pair trees per file (untimed sidecars; `tree_pairs`).
    let mut pair_root_files: BTreeMap<PathBuf, Vec<u32>> = BTreeMap::new();
    for c in &exp.configs {
        if let (Some(path), Some(px)) = (&c.pair_roots, &c.pairs_sidecar) {
            if !pair_root_files.contains_key(path) {
                let roots = crate::build::steiner::read_pair_roots(path, &pair_files[px].0)?;
                pair_root_files.insert(path.clone(), roots);
            }
            // Router ids of the pair-tree graph: bound to every node of the graph.
            let g = &graphs[&graph_key(c)];
            let e = Expect {
                labels: Some(&labels_fp),
                graph: Some((&graph_fps[&g.path], GraphPart::Full)),
                ..Expect::default()
            };
            prov.file(
                &format!(
                    "pair_roots {} with graph {}",
                    path.display(),
                    g.path.display()
                ),
                path,
                crate::build::steiner::PAIR_ROOTS_MAGIC,
                crate::build::steiner::PAIR_ROOTS_SECTIONS,
                &e,
            )?;
        }
    }

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
            bitsets: bitsets.as_ref(),
            costs: c.calibration.as_ref().map(|p| &cost_files[p].0),
            sketches: c.sketches_sidecar.as_ref().map(|p| &sketch_files[p].0),
            pairs: c.pairs_sidecar.as_ref().map(|p| &pair_files[p].0),
            islands: c.islands_sidecar.as_ref().map(|p| &island_files[p].0),
            pair_roots: c.pair_roots.as_ref().map(|p| &pair_root_files[p][..]),
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
                        cnf: match (&c.view, c.view_predicates) {
                            (Some(v), true) => set.views[v].get(*i),
                            _ => set.preds.get(*i),
                        },
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
            let mut config_json = json!({
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
            if c.view_predicates {
                config_json["predicates"] = json!("view");
            }
            let mut hash_input = json!({
                "experiment": exp.name, "config": config_json, "K": k, "efforts": out.efforts,
                "dataset": exp.dataset, "graph": c.graph, "starts": g.starts_path,
                "panel": rows.source.as_ref().map(|s| &s.1), "split": exp.split,
                "qsets": exp.qsets, "limit": exp.limit, "queries": exp.queries,
                "mode": exp.mode.name(), "reps": exp.reps, "seed": exp.seed,
                "bitsets": (matches!(c.kernel.predicate, PredicateMode::StaticBits(_))
                    || matches!(
                        c.kernel.materializer,
                        Materializer::DenseStatic | Materializer::DenseNav
                    )
                    || uses_exact_bitsets(c)
                    || matches!(c.kernel.tree_route, TreeRoute::StaticBits | TreeRoute::Cost))
                    .then(|| bitsets_json["source"].clone()),
                "calibration": uses_costs(c).then(|| c.calibration.clone()).flatten(),
                "sketches": (c.kernel.estimates == EstimateSource::Sketch)
                    .then(|| sketches_doc(c)["source"].clone()),
            });
            if c.pairs_sidecar.is_some() {
                hash_input["pairs"] = pairs_doc(c)["source"].clone();
            }
            if c.islands_sidecar.is_some() {
                hash_input["islands"] = islands_doc(c)["source"].clone();
            }
            if let Some(p) = &c.pair_roots {
                hash_input["pair_roots"] = json!(p);
            }
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
                "bitsets": bitsets_json,
                "calibration": c.calibration.as_ref().filter(|_| uses_costs(c))
                    .map_or(Value::Null, |p| cost_files[p].1.clone()),
                "sketches": if c.kernel.estimates == EstimateSource::Sketch {
                    sketches_doc(c)
                } else {
                    Value::Null
                },
                "pairs": pairs_doc(c),
                "islands": islands_doc(c),
                "fingerprints": {"labels": &labels_fp, "graph": graph_fps.get(&g.path)},
            });
            let protocol = json!({
                "reps": if exp.mode == Mode::Timed { exp.reps } else { 1 },
                "seed": exp.seed,
                "interleaving": "per K: for each repetition and effort, the configurations in a seeded shuffled order (re-shuffled per (rep, effort)); queries in panel order",
                "warmup": exp.warmup && exp.mode == Mode::Timed,
                "warmup_effort": "the median applicable effort, whole panel, untimed",
                "hot": exp.hot && exp.mode == Mode::Timed,
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
                json!("the collector returns at most L answers (L < K), or the configuration's own efforts"),
            );
            if let Some(costs) = c.calibration.as_ref().map(|p| &cost_files[p].0) {
                let capped = effort_curve_capped(c, costs, &out.efforts, k);
                if !capped.is_empty() {
                    put(
                        "effort_curve_capped",
                        json!({"efforts": capped, "l_top": costs.effort_l_top,
                            "note": "target recalls beyond the measured effort curve at K or more neighbours (K / estimated match rate): those lists are capped at its largest measured list size"}),
                    );
                }
            }
            put("provenance", prov.to_json());
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
        "route": "graph | exact | empty | fallback | postfilter | island",
        "n_aux_dist": "distance computations to auxiliary nodes",
        "n_lazy": "nodes skipped by distance_first",
        "n_aux_expand": "expansions of auxiliary nodes",
        "n_two_hop": "nodes inserted through the second hop",
        "n_evict": "beam entries evicted",
        "n_late": "configured starts a post-filter search admitted after it ended (by_rate_late)",
        "n_exact_cand": "exact-path candidates",
        "n_start": "starts selected (deduplicated)",
        "final_L": "expandable beam size at the end (adaptive L, or the planned posting_l)",
        "n_probe": "predicate evaluations of the sketch probe (also in n_pred)",
        "plan": "exact plan chosen: 0 none / version-1 path, 1 scan, 2 distance-first, 3 gallop, 4 bitmap, 5 rowand, 6 pair",
        "n_pair_lookups": "directory lookups in the combination postings (the pair plan's clause pair)",
        "fallback": "guarded trial: 0 none, 1 forecast, 2 budget, 3 incomplete heap, 4 witness",
        "est_exact_ns": "estimated exact-path cost (model ns; 0 if not computed)",
        "est_graph_ns": "nominal graph cost at the planned L (model ns; 0 if not computed)",
        "a_hat": "estimate of |union(c*)| (0 if not computed)",
        "m_hat": "estimate of the match count (0 if not computed)",
        "graph_ns": "model cost of the graph work of a guarded trial (ns)",
    })
}

/// Whether a configuration's exact path or probe uses the static bitsets.
fn uses_exact_bitsets(c: &ConfigSpec) -> bool {
    !c.kernel.static_bitsets_off
        && (c.kernel.exact.is_planned() || c.kernel.estimates == EstimateSource::Sketch)
}

/// Whether a configuration reads the unit costs.
fn uses_costs(c: &ConfigSpec) -> bool {
    c.kernel.exact.is_planned() || c.kernel.crossover != Crossover::Threshold
}

/// Reads unit costs: a `flab calibrate` report (its `costs` member) or a bare cost object.
pub fn read_costs(path: &Path) -> Result<UnitCosts> {
    read_costs_doc(path).map(|(c, _)| c)
}

/// [`read_costs`] and the whole document (its provenance, [`Checker::calibration`]).
pub fn read_costs_doc(path: &Path) -> Result<(UnitCosts, Value)> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let v: Value =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let c = costs_from_json(&v).with_context(|| format!("unit costs in {}", path.display()))?;
    Ok((c, v))
}

/// The efforts of an `effort_curve` configuration whose target recall lies beyond the measured
/// effort curve at `k` or more neighbours, i.e. whose lists are capped (for every query, or for
/// the selective ones).
fn effort_curve_capped(
    c: &ConfigSpec,
    costs: &UnitCosts,
    efforts: &[usize],
    k: usize,
) -> Vec<usize> {
    if !matches!(c.kernel.terminator, Terminator::EffortCurve { .. }) || costs.effort_l.is_empty() {
        return Vec::new();
    }
    efforts
        .iter()
        .copied()
        .filter(|&e| costs.effort_capped(e as f64 / 1000.0, k as f64))
        .collect()
}

/// The unit costs of a calibration document (`flab calibrate`'s JSON, or the bare costs). An
/// effort curve without its largest measured list size (older calibrations) takes it from the
/// document's measured list sizes (`details.effort.l`), so that it is capped there too.
pub fn costs_from_json(v: &Value) -> Result<UnitCosts> {
    let mut c: UnitCosts = serde_json::from_value(v.get("costs").cloned().unwrap_or(v.clone()))?;
    if c.effort_l_top <= 0.0 && !c.effort_l.is_empty() {
        let measured = v
            .pointer("/details/effort/l")
            .and_then(Value::as_array)
            .and_then(|ls| ls.iter().filter_map(Value::as_f64).reduce(f64::max));
        if let Some(top) = measured {
            c.effort_l_top = top;
        }
    }
    c.validate()?;
    Ok(c)
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
    let own_efforts: Vec<Option<Vec<usize>>> =
        exp.configs.iter().map(|c| c.efforts.clone()).collect();
    let mut outs: Vec<Outcome> = cx
        .plans
        .iter()
        .enumerate()
        .map(|(ci, p)| {
            let (efforts, skipped): (Vec<usize>, Vec<usize>) =
                exp.efforts.iter().partition(|&&l| {
                    !(p.kernel.needs_l_at_least_k() && l < k)
                        && own_efforts[ci].as_ref().is_none_or(|v| v.contains(&l))
                });
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
    let (mut hot_ids, mut hot_d) = (vec![0u32; k], vec![0f32; k]);

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
                        // Phase cycle counts of the timed searches (profiling builds only).
                        let mut phases = [0u64; profile::SLOTS];
                        profile::take();
                        for (row, q) in cx.queries[ci].iter().enumerate() {
                            let (plan, ix, q, l_run) = match &c.mix {
                                Some(m) => {
                                    let (j, lj) = if mix_draw(m.seed, row) < m.p {
                                        m.a
                                    } else {
                                        m.b
                                    };
                                    (&cx.plans[j], &cx.indices[j], &cx.queries[j][row], lj)
                                }
                                None => (plan, ix, q, l),
                            };
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
                            if exp.hot {
                                st.search(plan, ix, q, l_run, &mut hot_ids, &mut hot_d, None);
                                profile::take();
                            }
                            let cpu0 = sys::thread_cpu_ns();
                            let wall0 = Instant::now();
                            let ctr = st.search(plan, ix, q, l_run, ids, dists, None);
                            let wall = wall0.elapsed().as_nanos() as u64;
                            let cpu = sys::thread_cpu_ns() - cpu0;
                            if profile::ENABLED {
                                for (a, x) in phases.iter_mut().zip(profile::take()) {
                                    *a += x;
                                }
                            }
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
                        if profile::ENABLED {
                            let fields: Vec<String> = profile::NAMES
                                .iter()
                                .zip(&phases)
                                .map(|(n, v)| format!("{n}={v}"))
                                .collect();
                            println!(
                                "phase-profile config={} K={k} L={l} rep={rep} rows={} {}",
                                c.name,
                                cx.queries[ci].len(),
                                fields.join(" ")
                            );
                        }
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
    fn older_calibrations_cap_the_effort_curve_at_their_measured_lists() {
        // A calibration document without `effort_l_top` (written before it existed): the largest
        // of its measured list sizes (`details.effort.l`) caps the curve; a recorded value stays.
        let mut costs = serde_json::to_value(UnitCosts {
            effort_r: vec![0.5, 0.99],
            effort_k: vec![10.0],
            effort_l: vec![vec![10.0], vec![500.0]],
            ..UnitCosts::default()
        })
        .unwrap();
        costs.as_object_mut().unwrap().remove("effort_l_top");
        let old = json!({"costs": costs, "details": {"effort": {"l": [10, 20, 40, 80]}}});
        let c = costs_from_json(&old).unwrap();
        assert_eq!(c.effort_l_top, 80.0);
        assert_eq!(c.effort_list(0.99, 10.0), Some(80.0));
        let mut recorded = costs.clone();
        recorded["effort_l_top"] = json!(200.0);
        assert_eq!(
            costs_from_json(&json!({"costs": recorded}))
                .unwrap()
                .effort_l_top,
            200.0
        );
        assert_eq!(costs_from_json(&costs).unwrap().effort_l_top, 0.0);
    }

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
