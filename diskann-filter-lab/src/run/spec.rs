/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The experiment file of `flab run` (TOML).
//!
//! ```toml
//! [experiment]
//! name = "e001"
//! dataset = "<dataset dir>"        # base.{u8bin,fbin}, labels.flab, q<set>.{u8bin,fbin,fpred}
//! graph = "<graph dir or file>"    # graph.flab (+ starts.flab) or a DiskANN index prefix
//! starts_sidecar = "<starts.flab>" # optional; default: starts.flab next to the graph
//! view = "<name>"                  # optional label view: labels.<name>.flab, q<set>.<name>.fpred
//! queries = "q2Khard"              # without a panel: every query of q2Khard.*
//! panel = "panels/screen.tsv"      # or a panel (relative to the dataset dir), with filters:
//! split = "discovery"              #   keep rows of this split,
//! qsets = ["10K", "2Khard"]        #   of these query sets,
//! limit = 500                      #   the first rows only
//! k = [10, 100]                    # one run per K
//! efforts = [10, 20, 40, 80]       # L values
//! reps = 3
//! mode = "timed"                   # timed | trace | reference
//! seed = 1                         # order of the configurations per (rep, effort)
//! hot = false                      # profiling: repeat each search untimed right before its
//!                                  # timed run (the time with a cached working set)
//! runs_dir = "<dir>"               # run directories are created here
//! ledger = "<file>"                # one JSON line per run
//! bitsets_sidecar = "<bitsets.flab>" # optional static bitsets of `static_bits(θ)` configs
//!                                  # (default: built in memory at the smallest θ, untimed)
//! calibration = "<costs.json>"     # unit costs of the cost model (`flab calibrate`): needed by
//!                                  # the planned exact path and the cost crossovers
//! sketches_sidecar = "<sketches.flab>" # bottom-b sketches (`estimates = "sketch"`)
//! pairs_sidecar = "<pairs.flab>"   # combination postings (the exact path's `pair` plan)
//!
//! [[config]]
//! name = "inline"
//! preset = "upstream-inline"
//!
//! [[config]]
//! name = "inline+label-starts+heap"
//! preset = "upstream-inline"
//! starts = "label(c*)"             # any role override of `Kernel::set`
//! collector = "heap"
//! graph = "<other graph>"          # per-configuration graph, starts sidecar and view
//! efforts = [100]                  # per-configuration subset of the efforts (e.g. one effort
//!                                  # for an L-independent exact path)
//! calibration = "<costs.json>"     # per-configuration unit costs (e.g. another calibration)
//! sketches_sidecar = "<sketches.flab>" # per-configuration sketches (e.g. another priority seed)
//! pairs_sidecar = "<pairs.flab>"   # per-configuration combination postings (e.g. a budget)
//! ```

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;

use crate::kernel::{Kernel, Preset};

/// Run mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Timed repetitions under the measurement protocol.
    Timed,
    /// Untimed; event traces of sampled rows.
    Trace,
    /// Untimed; the reference executor (unbounded beam, distance budget).
    Reference,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Timed => "timed",
            Self::Trace => "trace",
            Self::Reference => "reference",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawExperiment {
    name: String,
    dataset: PathBuf,
    graph: Option<PathBuf>,
    starts_sidecar: Option<PathBuf>,
    view: Option<String>,
    queries: Option<String>,
    panel: Option<String>,
    split: Option<String>,
    qsets: Option<Vec<String>>,
    limit: Option<usize>,
    k: KList,
    efforts: Vec<usize>,
    reps: Option<usize>,
    mode: Option<String>,
    seed: Option<u64>,
    runs_dir: Option<PathBuf>,
    ledger: Option<PathBuf>,
    warmup: Option<bool>,
    hot: Option<bool>,
    tag: Option<String>,
    trace_rows: Option<Vec<usize>>,
    trace_sample: Option<usize>,
    reference_factor: Option<f64>,
    reference_min: Option<u64>,
    eps_min: Option<f32>,
    fast_paths: Option<bool>,
    bitsets_sidecar: Option<PathBuf>,
    calibration: Option<PathBuf>,
    sketches_sidecar: Option<PathBuf>,
    pairs_sidecar: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum KList {
    One(usize),
    Many(Vec<usize>),
}

#[derive(Debug, Clone, Deserialize)]
struct RawFile {
    experiment: RawExperiment,
    #[serde(default)]
    config: Vec<BTreeMap<String, toml::Value>>,
}

/// One configuration of an experiment.
#[derive(Debug, Clone)]
pub struct ConfigSpec {
    pub name: String,
    pub preset: Option<String>,
    pub kernel: Kernel,
    pub graph: PathBuf,
    pub starts_sidecar: Option<PathBuf>,
    pub view: Option<String>,
    /// The efforts of this configuration (a subset of the experiment's; all if `None`).
    pub efforts: Option<Vec<usize>>,
    /// Unit costs of this configuration (the experiment's unless overridden).
    pub calibration: Option<PathBuf>,
    /// Bottom-b sketches of this configuration (the experiment's unless overridden).
    pub sketches_sidecar: Option<PathBuf>,
    /// Combination postings of this configuration (the experiment's unless overridden).
    pub pairs_sidecar: Option<PathBuf>,
    /// The overrides as given (for the record).
    pub overrides: BTreeMap<String, String>,
}

/// A parsed and validated experiment file.
#[derive(Debug, Clone)]
pub struct Experiment {
    pub name: String,
    pub file: PathBuf,
    pub dataset: PathBuf,
    pub queries: Option<String>,
    pub panel: Option<PathBuf>,
    pub split: Option<String>,
    pub qsets: Option<Vec<String>>,
    pub limit: Option<usize>,
    pub ks: Vec<usize>,
    pub efforts: Vec<usize>,
    pub reps: usize,
    pub mode: Mode,
    pub seed: u64,
    pub runs_dir: Option<PathBuf>,
    pub ledger: Option<PathBuf>,
    pub warmup: bool,
    /// Profiling: each timed search runs right after an untimed search of the same row and
    /// effort, so that it finds its working set in the caches (the time without cold misses).
    pub hot: bool,
    pub tag: Option<String>,
    pub trace_rows: Option<Vec<usize>>,
    pub trace_sample: usize,
    pub reference_factor: f64,
    pub reference_min: u64,
    pub eps_min: Option<f32>,
    /// Whether configurations that are exactly a preset with a hand-specialized loop use it
    /// (default); `false` runs every configuration in the configurable loop.
    pub fast_paths: bool,
    /// Static per-label bitsets of the dataset's labels for `static_bits(θ)` configurations
    /// (built in memory when absent).
    pub bitsets_sidecar: Option<PathBuf>,
    /// Unit costs of the cost model (`flab calibrate` output).
    pub calibration: Option<PathBuf>,
    /// Bottom-`b` sketches of the dataset's labels.
    pub sketches_sidecar: Option<PathBuf>,
    /// Combination postings of the dataset's labels (`pairs.flab`).
    pub pairs_sidecar: Option<PathBuf>,
    pub configs: Vec<ConfigSpec>,
}

fn value_text(key: &str, v: &toml::Value) -> Result<String> {
    Ok(match v {
        toml::Value::String(s) => s.clone(),
        toml::Value::Integer(i) => i.to_string(),
        toml::Value::Float(f) => f.to_string(),
        toml::Value::Boolean(b) => b.to_string(),
        toml::Value::Array(items) => items
            .iter()
            .map(|i| value_text(key, i))
            .collect::<Result<Vec<_>>>()?
            .join("+"),
        _ => bail!("option {key:?} must be a string, a number or a list of strings"),
    })
}

/// Resolves `path` relative to `base` (absolute paths are kept).
fn resolve(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        base.join(path)
    }
}

impl Experiment {
    /// Reads and validates an experiment file. Relative paths are resolved against the file's
    /// directory, panel paths against the dataset directory.
    pub fn read(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let dir = path.parent().unwrap_or(Path::new("."));
        Self::parse(&text, dir, path)
    }

    /// Parses an experiment from TOML text.
    pub fn parse(text: &str, dir: &Path, file: &Path) -> Result<Self> {
        let raw: RawFile = toml::from_str(text).context("parsing the experiment file")?;
        let e = raw.experiment;
        let dataset = resolve(dir, &e.dataset);
        let ks = match e.k {
            KList::One(k) => vec![k],
            KList::Many(v) => v,
        };
        ensure!(
            !ks.is_empty() && ks.iter().all(|&k| k > 0),
            "k must be positive"
        );
        ensure!(
            !e.efforts.is_empty() && e.efforts.iter().all(|&l| l > 0),
            "efforts must be positive"
        );
        let mode = match e.mode.as_deref().unwrap_or("timed") {
            "timed" => Mode::Timed,
            "trace" => Mode::Trace,
            "reference" => Mode::Reference,
            m => bail!("unknown mode {m:?} (timed, trace, reference)"),
        };
        ensure!(
            e.queries.is_some() != e.panel.is_some(),
            "give exactly one of `queries` and `panel`"
        );
        if e.panel.is_none() {
            ensure!(
                e.split.is_none() && e.qsets.is_none(),
                "`split` and `qsets` filter a panel"
            );
        }
        let reps = e.reps.unwrap_or(3);
        ensure!(reps >= 1, "reps must be at least 1");
        ensure!(!raw.config.is_empty(), "no [[config]] entries");

        let mut configs = Vec::new();
        let mut names = std::collections::HashSet::new();
        for table in raw.config {
            let mut table = table;
            let name = match table.remove("name") {
                Some(toml::Value::String(s)) => s,
                _ => bail!("every [[config]] needs a string `name`"),
            };
            ensure!(names.insert(name.clone()), "duplicate config name {name:?}");
            let text =
                |t: &mut BTreeMap<String, toml::Value>, key: &str| -> Result<Option<String>> {
                    t.remove(key).map(|v| value_text(key, &v)).transpose()
                };
            let preset = text(&mut table, "preset")?;
            let graph = text(&mut table, "graph")?
                .map(PathBuf::from)
                .or_else(|| e.graph.clone())
                .with_context(|| format!("config {name:?}: no graph"))?;
            let starts_sidecar = text(&mut table, "starts_sidecar")?
                .map(PathBuf::from)
                .or_else(|| e.starts_sidecar.clone());
            let view = text(&mut table, "view")?.or_else(|| e.view.clone());
            let calibration = text(&mut table, "calibration")?
                .map(PathBuf::from)
                .or_else(|| e.calibration.clone());
            let sketches_sidecar = text(&mut table, "sketches_sidecar")?
                .map(PathBuf::from)
                .or_else(|| e.sketches_sidecar.clone());
            let pairs_sidecar = text(&mut table, "pairs_sidecar")?
                .map(PathBuf::from)
                .or_else(|| e.pairs_sidecar.clone());
            let efforts = match table.remove("efforts") {
                None => None,
                Some(toml::Value::Array(items)) => {
                    let mut v = Vec::new();
                    for item in items {
                        match item {
                            toml::Value::Integer(l) if l > 0 => v.push(l as usize),
                            _ => bail!("config {name:?}: efforts must be positive integers"),
                        }
                    }
                    ensure!(
                        !v.is_empty() && v.iter().all(|l| e.efforts.contains(l)),
                        "config {name:?}: efforts must be a non-empty subset of the experiment's"
                    );
                    Some(v)
                }
                Some(_) => bail!("config {name:?}: efforts must be a list of integers"),
            };
            let mut kernel = match &preset {
                Some(p) => Preset::parse(p)
                    .with_context(|| format!("config {name:?}"))?
                    .kernel(),
                None => Kernel::default(),
            };
            let mut overrides = BTreeMap::new();
            for (key, value) in &table {
                let v = value_text(key, value)?;
                kernel
                    .set(key, &v)
                    .with_context(|| format!("config {name:?}"))?;
                overrides.insert(key.clone(), v);
            }
            kernel
                .validate()
                .with_context(|| format!("config {name:?}"))?;
            if let Some(p) = &preset {
                if Preset::parse(p)?.uses_view() {
                    ensure!(
                        view.is_some(),
                        "config {name:?}: preset {p} takes its per-label starts from a label view (set `view`)"
                    );
                }
            }
            configs.push(ConfigSpec {
                name,
                preset,
                kernel,
                graph: resolve(dir, &graph),
                starts_sidecar: starts_sidecar.map(|p| resolve(dir, &p)),
                view,
                efforts,
                calibration: calibration.map(|p| resolve(dir, &p)),
                sketches_sidecar: sketches_sidecar.map(|p| resolve(dir, &p)),
                pairs_sidecar: pairs_sidecar.map(|p| resolve(dir, &p)),
                overrides,
            });
        }
        let panel = e.panel.as_ref().map(|p| {
            let p = Path::new(p);
            if p.is_absolute() {
                p.to_owned()
            } else {
                dataset.join(p)
            }
        });
        Ok(Self {
            name: e.name,
            file: file.to_owned(),
            dataset,
            queries: e
                .queries
                .map(|q| q.strip_prefix('q').map_or(q.clone(), str::to_owned)),
            panel,
            split: e.split,
            qsets: e.qsets,
            limit: e.limit,
            ks,
            efforts: e.efforts,
            reps,
            mode,
            seed: e.seed.unwrap_or(0),
            runs_dir: e.runs_dir.map(|p| resolve(dir, &p)),
            ledger: e.ledger.map(|p| resolve(dir, &p)),
            warmup: e.warmup.unwrap_or(true),
            hot: e.hot.unwrap_or(false),
            tag: e.tag,
            trace_rows: e.trace_rows,
            trace_sample: e.trace_sample.unwrap_or(10),
            reference_factor: e.reference_factor.unwrap_or(20.0),
            reference_min: e.reference_min.unwrap_or(1000),
            eps_min: e.eps_min,
            fast_paths: e.fast_paths.unwrap_or(true),
            bitsets_sidecar: e.bitsets_sidecar.map(|p| resolve(dir, &p)),
            calibration: e.calibration.map(|p| resolve(dir, &p)),
            sketches_sidecar: e.sketches_sidecar.map(|p| resolve(dir, &p)),
            pairs_sidecar: e.pairs_sidecar.map(|p| resolve(dir, &p)),
            configs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_presets_overrides_and_defaults() {
        let text = r#"
            [experiment]
            name = "e"
            dataset = "ds"
            graph = "g"
            panel = "panels/screen.tsv"
            split = "discovery"
            k = 10
            efforts = [10, 20]

            [[config]]
            name = "inline"
            preset = "upstream-inline"

            [[config]]
            name = "inline+heap"
            preset = "upstream-inline"
            starts = ["label(c*)", "global"]
            collector = "heap"
            beam_width = 2
            graph = "/abs/other"
        "#;
        let e = Experiment::parse(text, Path::new("/x"), Path::new("/x/e.toml")).unwrap();
        assert_eq!((e.ks.clone(), e.reps, e.mode), (vec![10], 3, Mode::Timed));
        assert_eq!(
            e.panel.as_deref(),
            Some(Path::new("/x/ds/panels/screen.tsv"))
        );
        assert_eq!(e.configs[0].graph, Path::new("/x/g"));
        assert_eq!(e.configs[1].graph, Path::new("/abs/other"));
        assert_eq!(e.configs[1].kernel.beam_width, 2);
        assert_eq!(e.configs[1].overrides["starts"], "label(c*)+global");

        let bad = |cfg: &str| {
            let t = format!(
                "[experiment]\nname='e'\ndataset='d'\ngraph='g'\nqueries='q1'\nk=[10]\nefforts=[10]\n{cfg}"
            );
            Experiment::parse(&t, Path::new("/x"), Path::new("/x/e.toml")).is_err()
        };
        assert!(!bad("[[config]]\nname='a'\npreset='postfilter'"));
        assert!(bad("[[config]]\nname='a'\npreset='nope'"));
        assert!(bad("[[config]]\nname='a'\ncolour='red'"));
        assert!(bad(
            "[[config]]\nname='a'\nlaziness='deferred'\nscorer='bonus(0.5)'"
        ));
        assert!(bad("[[config]]\nname='a'\n[[config]]\nname='a'"));
        assert!(bad("[[config]]\nname='a'\npreset='cp'"), "cp needs a view");
        assert!(bad(""), "no configs");
        assert!(!bad("[[config]]\nname='a'\nefforts=[10]"));
        assert!(
            bad("[[config]]\nname='a'\nefforts=[20]"),
            "not an experiment effort"
        );
        assert!(bad("[[config]]\nname='a'\nefforts=[]"));
    }
}
