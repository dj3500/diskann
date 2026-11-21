/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! `flab`: command-line tool of the filter lab (conversion, ground truth, inspection).

use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{bail, ensure, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use diskann_filter_lab::{
    bitsets::{self, LabelBitsets},
    calibrate, container, diskann_io,
    graph::{self, Graph, GraphStats},
    gt::{self, DiskannTruth, GroundTruth},
    islands::{self, IslandIndex},
    jsonl, label_starts, labels,
    labels::LabelStore,
    pairs::{Cover, PairIndex},
    predicate::{self, PredicateSet, PredicateSetBuilder},
    run,
    sketches::{self, LabelSketches},
    starts::{self, Starts},
    text,
    vectors::{self, DType, Element, Matrix},
    workload,
};
use serde_json::{json, Value};

#[derive(Parser)]
#[command(
    name = "flab",
    version,
    about = "Filter lab: label/predicate conversion, exact filtered ground truth, graph import"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Convert labels, predicates or vectors into the lab's formats.
    #[command(subcommand)]
    Convert(Convert),
    /// Exact filtered k-NN ground truth with complete distance-tie bands.
    Gt(GtArgs),
    /// Compare an extended ground truth with a DiskANN ground truth file, up to tie order.
    GtCompare(GtCompareArgs),
    /// Import a saved DiskANN in-memory index (C++ library or Rust `diskann-providers`) as
    /// `graph.flab`.
    ImportGraph(ImportGraphArgs),
    /// Build the start-node table (`starts.flab`) of a graph from C++ build outputs.
    ImportStarts(ImportStartsArgs),
    /// Start-node tables computed from the data.
    #[command(subcommand)]
    Starts(StartsCommand),
    /// Static per-label bitsets (`bitsets.flab`, for the `static_bits(θ)` predicate mode).
    #[command(subcommand)]
    Bitsets(BitsetsCommand),
    /// Bottom-b priority sketches of the label postings (`sketches.flab`, for the sketch
    /// estimates of the cost crossover).
    #[command(subcommand)]
    Sketches(SketchesCommand),
    /// Combination postings (`pairs.flab`: joint postings of selected label pairs within a
    /// memory budget, for the exact path's `pair` plan).
    #[command(subcommand)]
    Pairs(PairsCommand),
    /// Per-label sub-graphs (`islands.flab`: a Vamana graph over each selected label's members
    /// within a memory budget, for the `island` route of the cost crossover).
    #[command(subcommand)]
    Islands(IslandsCommand),
    /// One selector for the auxiliary structures: combination postings and per-label
    /// sub-graphs chosen together by expected saving per query per byte on base-derived queries
    /// under one memory budget (writes `pairs.flab` / `islands.flab` files and a report).
    Select(SelectArgs),
    /// Measure the unit costs of the cost model on this machine and dataset (JSON).
    Calibrate(CalibrateArgs),
    /// Add the tree effort curve of a Steiner graph's router trees (`flab steiner`) to a
    /// calibration of its base graph (read by `tree_effort = curve`).
    TreeCalibrate(TreeCalibrateArgs),
    /// Run an experiment file (configurations × panel × effort grid) with the lab search
    /// kernel and write run directories (see `run` in the crate documentation).
    Run(RunArgs),
    /// Build a graph with the lab's parallel Vamana builder (optionally recording candidate
    /// pools), or replay recorded pools with a pruning rule.
    Build(diskann_filter_lab::build::cli::BuildArgs),
    /// Relabel the data points (an isomorphic id layout, e.g. a k-means tree order): writes
    /// permuted copies of the base, labels, ground truth and sketches (links to the query
    /// files) and of the graph and its start table.
    Permute(PermuteArgs),
    /// Navigation diagnostics of graph-only search (reachability of the answers under the
    /// kernel's traversal models, missed-answer classes, oracle and per-component entries,
    /// work decomposition, block statistics; untimed).
    Diag(diskann_filter_lab::diag::DiagArgs),
    /// Steiner augmentation of a graph: auxiliary router nodes with protected out-edges,
    /// reached only from the start table (per-label trees or component connectors).
    Steiner(diskann_filter_lab::build::steiner::SteinerArgs),
    /// Summarize a lab file (labels, predicates, ground truth, graph, starts) or a vector file.
    Info {
        path: PathBuf,
        /// Element type of a vector file; inferred from the extension when omitted.
        #[arg(long, value_enum)]
        dtype: Option<DTypeArg>,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum Format {
    /// Comma-separated labels per point; `&`-separated clauses of `|`-separated labels.
    Text,
    /// JSON lines (`doc_id` + Boolean attributes; `$and` / `$or` / `$eq` filters).
    Jsonl,
}

#[derive(Clone, Copy, ValueEnum)]
enum EvaluationArg {
    /// Per query, whichever touches fewer label entries.
    Auto,
    /// Forward-store rows of the smallest clause's points.
    Forward,
    /// Bitsets from the postings of every clause.
    Postings,
}

#[derive(Clone, Copy, ValueEnum)]
enum DTypeArg {
    U8,
    F32,
}

fn dtype(arg: Option<DTypeArg>, path: &Path) -> Result<DType> {
    match arg {
        Some(DTypeArg::U8) => Ok(DType::U8),
        Some(DTypeArg::F32) => Ok(DType::F32),
        None => DType::from_path(path),
    }
}

#[derive(Subcommand)]
enum Convert {
    /// Base labels (one line per point) to `labels.flab`.
    Labels {
        #[arg(long, value_enum)]
        format: Format,
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Query predicates (one line per query) to a predicate file, against the dictionary of
    /// `--labels`.
    Predicates {
        #[arg(long, value_enum)]
        format: Format,
        /// The `labels.flab` of the base points.
        #[arg(long)]
        labels: PathBuf,
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Copy a vector file (u32 n, u32 d, row-major data), optionally scaling every row to unit
    /// L2 norm.
    Vectors {
        /// Element type; inferred from the input extension when omitted.
        #[arg(long, value_enum)]
        dtype: Option<DTypeArg>,
        /// Scale every row to unit L2 norm (f32 only; zero rows stay zero).
        #[arg(long)]
        normalize: bool,
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
}

#[derive(Args)]
struct GtArgs {
    /// Base vectors (`.u8bin` or `.fbin`).
    #[arg(long)]
    base: PathBuf,
    /// Query vectors (same element type and dimension as the base).
    #[arg(long)]
    queries: PathBuf,
    /// `labels.flab` of the base points.
    #[arg(long)]
    labels: PathBuf,
    /// Predicates of the queries (compiled against `--labels`).
    #[arg(long)]
    predicates: PathBuf,
    /// Element type; inferred from the base file extension when omitted.
    #[arg(long, value_enum)]
    dtype: Option<DTypeArg>,
    /// Number of neighbors K.
    #[arg(short, default_value_t = 100)]
    k: usize,
    #[arg(short, long, default_value_t = 8)]
    threads: usize,
    /// How candidates are checked against the predicate (all choices are exact).
    #[arg(long, value_enum, default_value = "auto")]
    eval: EvaluationArg,
    /// Output: standard DiskANN ground truth (u32 nq, u32 K, ids, distances; padded).
    #[arg(long)]
    out_bin: Option<PathBuf>,
    /// Output: extended ground truth (tie bands, match counts, smallest clause).
    #[arg(long)]
    out_ext: Option<PathBuf>,
}

#[derive(Args)]
struct GtCompareArgs {
    /// Extended ground truth (ours).
    #[arg(long)]
    ext: PathBuf,
    /// DiskANN ground truth file with distances.
    #[arg(long)]
    reference: PathBuf,
    /// Compare the top-k (default: the smaller K of the two files).
    #[arg(short)]
    k: Option<usize>,
    /// Number of mismatching queries to print.
    #[arg(long, default_value_t = 10)]
    show: usize,
}

#[derive(Args)]
struct ImportGraphArgs {
    /// The graph file of the index (`<prefix>`).
    #[arg(long)]
    graph: PathBuf,
    /// The vector file of the index (`<prefix>.data`); required when the graph has frozen
    /// points.
    #[arg(long)]
    data: Option<PathBuf>,
    /// Base vectors of the dataset; the first rows of `--data` must equal them.
    #[arg(long)]
    base: Option<PathBuf>,
    /// Element type of the vectors; inferred from the extension of `--base` or `--data` when
    /// omitted.
    #[arg(long, value_enum)]
    dtype: Option<DTypeArg>,
    /// Import even if the vectors of the index differ from `--base`.
    #[arg(long)]
    allow_base_mismatch: bool,
    #[arg(short, long)]
    output: PathBuf,
    /// Write a JSON report (inputs, header, node space, statistics, checks).
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Args)]
struct ImportStartsArgs {
    /// The graph (`graph.flab`); its starts become the global starts.
    #[arg(long)]
    graph: PathBuf,
    /// Label store that the per-label starts refer to (the graph's label view).
    #[arg(long)]
    labels: Option<PathBuf>,
    /// C++ per-label start points (`<prefix>_labels_to_medoids.txt`); needs `--label-map` and
    /// `--labels`.
    #[arg(long)]
    medoids: Option<PathBuf>,
    /// C++ label map (`<prefix>_labels_map.txt`: `name<TAB>label` lines).
    #[arg(long)]
    label_map: Option<PathBuf>,
    /// Sample of data points in order (`<prefix>_sample_ids.bin`).
    #[arg(long)]
    sample: Option<PathBuf>,
    /// Start node of each sample entry (`<prefix>_sample_labels.txt_map.bin`); by default the
    /// entry itself. Entries beyond the sample are ignored.
    #[arg(long)]
    sample_map: Option<PathBuf>,
    #[arg(short, long)]
    output: PathBuf,
    /// Write a JSON report (inputs, counts, checks).
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Subcommand)]
enum StartsCommand {
    /// Per-label starts of every label of a label store, for a graph over the same points:
    /// writes `starts-<tag>.flab` (with the graph's global starts) and a JSON report per
    /// strategy.
    Build(StartsBuildArgs),
}

#[derive(Args)]
struct StartsBuildArgs {
    /// Dataset directory (`base.u8bin` or `base.fbin`, `labels.flab`).
    #[arg(long)]
    dataset: PathBuf,
    /// Label store of the per-label table (default: `<dataset>/labels.flab`).
    #[arg(long)]
    labels: Option<PathBuf>,
    /// The graph: a directory with `graph.flab`, a `graph.flab`, or a DiskANN index prefix
    /// (vectors in `<prefix>.data`). Its starts become the global starts.
    #[arg(long)]
    graph: PathBuf,
    /// Strategies, comma-separated or repeated: `medoid` (the member closest to the label's
    /// centroid), `random(k)` (k seeded random members; `random` = 1), `kmeans(k)` (up to k
    /// k-means representatives; `kmeans` = 4).
    #[arg(long = "strategy", value_delimiter = ',', required = true)]
    strategies: Vec<String>,
    /// Labels with at most this many points use all of them (medoid, kmeans).
    #[arg(long, default_value_t = 10_000)]
    exact_max: usize,
    /// Members sampled (seeded) from larger labels.
    #[arg(long, default_value_t = 10_000)]
    sample: usize,
    /// Lloyd iterations of kmeans.
    #[arg(long, default_value_t = 10)]
    iters: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(short, long, default_value_t = 4)]
    threads: usize,
    /// Output directory (created if needed).
    #[arg(long)]
    out_dir: PathBuf,
}

#[derive(Subcommand)]
enum BitsetsCommand {
    /// Build the bitsets of every label with at least θ points.
    Build {
        /// The label store.
        #[arg(long)]
        labels: PathBuf,
        /// Threshold θ (points per label).
        #[arg(long)]
        theta: u64,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Memory of the bitsets for a list of thresholds (without building them).
    Stats {
        #[arg(long)]
        labels: PathBuf,
        /// Thresholds θ (points per label), comma-separated; default: n/1000, n/100, n/32,
        /// n/10.
        #[arg(long, value_delimiter = ',')]
        theta: Vec<u64>,
    },
}

#[derive(Subcommand)]
enum SketchesCommand {
    /// Build the sketches of every label for each b: writes `sketches-b<b>.flab` and
    /// `sketches-b<b>.json` (memory, build time).
    Build {
        /// The label store.
        #[arg(long)]
        labels: PathBuf,
        /// Entries per label (comma-separated list).
        #[arg(short, long, value_delimiter = ',', required = true)]
        b: Vec<usize>,
        /// Seed of the point priorities.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(short, long, default_value_t = 4)]
        threads: usize,
        /// Output directory (created if needed).
        #[arg(long)]
        out_dir: PathBuf,
    },
    /// Memory of the sketches for a list of b (without building them).
    Stats {
        #[arg(long)]
        labels: PathBuf,
        #[arg(short, long, value_delimiter = ',', required = true)]
        b: Vec<usize>,
    },
}

#[derive(Subcommand)]
enum PairsCommand {
    /// Select and build the combination postings for each budget: writes
    /// `pairs-nested-b<per-mille>.flab` and `.json` (sizes, build time, and the coverage of the
    /// `--report` queries); `pairs-<prior>-nested-b<per-mille>.*` under a prior other than
    /// `uniform`, and without `-nested` for `--plain` selections.
    Build {
        /// The label store.
        #[arg(long)]
        labels: PathBuf,
        /// Budgets as fractions of the label postings' bytes (comma-separated list).
        #[arg(short, long, value_delimiter = ',', required = true)]
        budget: Vec<f64>,
        #[arg(short, long, default_value_t = 4)]
        threads: usize,
        /// Output directory (created if needed).
        #[arg(long)]
        out_dir: PathBuf,
        /// Predicate files whose coverage is reported (queries with a covered clause pair and
        /// their candidates against the smallest clause bound).
        #[arg(long, value_delimiter = ',')]
        report: Vec<PathBuf>,
        /// Which pairs queries are assumed to use: `uniform` (every co-occurring pair; rank by
        /// gain per entry), `rare` (as likely as the smaller label is frequent) or `data` (as
        /// likely as a uniformly drawn point carries both labels).
        #[arg(long, default_value = "uniform")]
        prior: diskann_filter_lab::pairs::Prior,
        /// Nested selections (the default): a budget's selection keeps every pair of the
        /// smaller budgets of its chain (the 1-2-5 ladder from 5 % below it), so that the
        /// selections are monotone in the budget; budgets of one chain are built together.
        #[arg(long, conflicts_with = "plain")]
        nested: bool,
        /// Independent selections per budget (no smaller budget's pairs kept).
        #[arg(long)]
        plain: bool,
    },
}

#[derive(Subcommand)]
enum IslandsCommand {
    /// Select the labels (gain per byte under the cost model) for each budget and build their
    /// islands with the lab's Vamana builder (the union once): writes `islands-b<per-mille>.flab`
    /// and `islands-b<per-mille>.json`, and `gains.tsv` (every label's selection terms).
    Build {
        /// Dataset directory (`base.u8bin` or `base.fbin`, `labels.flab`).
        #[arg(long)]
        dataset: PathBuf,
        /// Unit costs of the cost model (`flab calibrate` output).
        #[arg(long)]
        calibration: PathBuf,
        /// Budgets as fractions of `256·n` bytes (an adjacency of 64 slots per point).
        #[arg(short, long, value_delimiter = ',', required = true)]
        budget: Vec<f64>,
        /// Answers `K` of the selection's cost terms.
        #[arg(short, long, default_value_t = 10)]
        k: usize,
        /// Degree bound `R` of the islands.
        #[arg(long, default_value_t = 32)]
        r: usize,
        #[arg(long, default_value_t = 64)]
        l_build: usize,
        #[arg(long, default_value_t = 1.2)]
        alpha: f32,
        /// Islands built at a time (one thread each).
        #[arg(short, long, default_value_t = 2)]
        threads: usize,
        /// Output directory (created if needed).
        #[arg(long)]
        out_dir: PathBuf,
        /// Which labels queries are assumed to search in: `label` (every label equally likely;
        /// rank by gain per byte), `data` (as likely as frequent; rank by gain) or `queries`
        /// (base-derived queries: greedy by expected saving per query per byte; terms in
        /// `gains-queries.tsv`). Files `islands-<prior>-b*` for `data` and `queries`.
        #[arg(long, default_value = "label")]
        prior: islands::Prior,
        /// Base-derived queries drawn by `--prior queries`.
        #[arg(long, default_value_t = 200_000)]
        queries: usize,
        /// Seed of the base-derived queries.
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Probability that a base-derived query's clause is a single label of its point (else
        /// a random subset of the point's labels in the category).
        #[arg(long, default_value_t = 0.5)]
        singleton: f64,
        /// Select only: write the selections (`islands-<prior>-b*.selection.json`) without
        /// building islands.
        #[arg(long)]
        dry_run: bool,
        /// Only these labels are candidates (a file of label ids, one per line; e.g. a
        /// counterfactual or oracle selection); output files `islands-restricted-b*`.
        #[arg(long)]
        restrict: Option<PathBuf>,
    },
}

#[derive(Args)]
struct SelectArgs {
    /// Dataset directory (`base.u8bin` or `base.fbin`, `labels.flab`).
    #[arg(long)]
    dataset: PathBuf,
    /// Unit costs of the cost model (`flab calibrate` output).
    #[arg(long)]
    calibration: PathBuf,
    /// The graph file the index searches: with the base vectors and the label store, its
    /// bytes are the index bytes of the value floor.
    #[arg(long)]
    graph: PathBuf,
    /// Memory budgets in bytes per data point (one selection each; with `--stop elasticity` a
    /// ceiling).
    #[arg(short, long, value_delimiter = ',', required = true)]
    budget: Vec<f64>,
    /// When the greedy stops: `elasticity` (an item is taken only if it lowers the workload's
    /// mean query time by at least the fraction by which it grows the index, `Δt/t ≥ ΔB/B`; the
    /// budget is a ceiling) or `budget` (fill the budget, items below the `--kappa` floor
    /// excluded).
    #[arg(long, value_enum, default_value = "elasticity")]
    stop: StopArg,
    /// `--stop budget`: value floor, an item must speed the workload up by at least `kappa`
    /// times the fraction by which it grows the index, both measured without auxiliary
    /// structures (expected saving per query per byte at least `kappa · mean cost / index
    /// bytes`; 0: no floor).
    #[arg(long, default_value_t = 0.0)]
    kappa: f64,
    /// Share of the independent queries in the workload: with probability ½ a label drawn
    /// uniformly among those that occur, as `{ℓ}`, otherwise a pair of labels of different
    /// categories drawn uniformly among the co-occurring ones, as `{a} ∧ {b}`; the query vector
    /// a data point drawn independently (0: base-derived queries only).
    #[arg(long, default_value_t = 0.5)]
    uniform: f64,
    /// Answers `K` of the cost terms.
    #[arg(short, long, default_value_t = 10)]
    k: usize,
    /// Degree bound `R` of the islands.
    #[arg(long, default_value_t = 32)]
    r: usize,
    #[arg(long, default_value_t = 64)]
    l_build: usize,
    #[arg(long, default_value_t = 1.2)]
    alpha: f32,
    #[arg(short, long, default_value_t = 2)]
    threads: usize,
    /// Output directory (created if needed).
    #[arg(long)]
    out_dir: PathBuf,
    /// Base-derived queries of the workload.
    #[arg(long, default_value_t = 200_000)]
    queries: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Probability that a kept category's clause is a single label (else a random subset).
    #[arg(long, default_value_t = 0.5)]
    singleton: f64,
    /// Probability that a kept category's clause comes from another data point.
    #[arg(long, default_value_t = 0.0)]
    fresh: f64,
    /// Select only: write the reports without building the structures.
    #[arg(long)]
    dry_run: bool,
    /// Predicate sets whose coverage by the selected pairs is reported (diagnostics only).
    #[arg(long, value_delimiter = ',')]
    report: Vec<PathBuf>,
    /// Write the pair candidates (labels, joint sizes, context weights; counted without a
    /// floor's pruning when `--kappa 0`) to this file, for `--load-candidates`.
    #[arg(long)]
    save_candidates: Option<PathBuf>,
    /// Read the pair candidates from a file written by `--save-candidates` (same label store and
    /// `--singleton`; written with `--kappa 0`, it serves every `--kappa`) instead of counting
    /// them; the floor's label pruning is applied as when counting.
    #[arg(long)]
    load_candidates: Option<PathBuf>,
    /// Also write the first `M` base-derived queries and `M` independent queries drawn i.i.d.
    /// from the workload (`--seed`) as query sets: `workload-{d,u}.fpred` (predicates),
    /// `workload-{d,u}.<base extension>` (the vectors of their points) and `workload-{d,u}.tsv`.
    #[arg(long)]
    dump_workload: Option<usize>,
    /// Whose auxiliary structures: `hybrid` (islands and combination postings, valued against
    /// the exact path and the graph route) or `graph` (track G: a router tree per label and a
    /// pair tree per label pair, valued against the graph router alone; writes the trees'
    /// labels, `sel-b<B>.trees.txt`, for `flab steiner --labels-file`, and the pairs).
    #[arg(long, value_enum, default_value = "hybrid")]
    track: SelectTrack,
    /// `--track graph`: reference recall of the router's and the tree route's costs.
    #[arg(long, default_value_t = 0.9)]
    tree_recall: f64,
    /// `--track graph`: reference router frontier `F` of the tree-route cost (the declared
    /// frontier table's at the reference recall).
    #[arg(long, default_value_t = 512.0)]
    tree_frontier: f64,
    /// `--track graph`: the tree route's expansions from the reference frontier (`frontier`) or
    /// from the calibration's tree effort curve (`curve`; `flab tree-calibrate`).
    #[arg(long, value_enum, default_value = "frontier")]
    tree_effort: SelectTreeEffort,
    /// `--track graph`: price both routes from route calibration curves (JSON: per label-size
    /// class, the router's and the tree route's mean recall and time at each frontier level, on
    /// queries placed independently of the label (`U`), on members (`D`) and on label pairs
    /// (`P`); `select_g::RouteCurves`) at the reference recall `--tree-recall`.
    #[arg(long)]
    route_curves: Option<PathBuf>,
    /// `--track graph` with `--route-curves`: the price of a route that never reaches the
    /// reference recall — `must` (a fixed large cost: a structure that makes the query answerable
    /// is a must-have) or `at-max` (its time at the highest measured level).
    #[arg(long, value_enum, default_value = "must")]
    unreachable: SelectUnreachable,
    /// `--track graph`: also select pair trees (`false`: router trees per label only).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pair_trees: bool,
    /// `--track graph`: mean router fan-out `b̄` of the k-means trees (fan-out bound 64).
    #[arg(long, default_value_t = 26.0)]
    tree_fanout: f64,
    /// `--track graph`: a label with fewer sampled queries is valued by its size class (the
    /// workload's closed form; see `select_g::TreeGreedy::with_classes`).
    #[arg(long, default_value_t = 30)]
    tree_min_records: u32,
    /// `--track graph`: the label trees are fixed by a size cap — every label of 1 to SIZE
    /// points (`n/10` = ⌈n/10⌉), as `flab steiner --construction tree --min-size 0 --max-size
    /// SIZE` builds them — taken before the greedy and part of the index, and only pair trees
    /// are selected, valued against the routes these trees give.
    #[arg(long)]
    fixed_trees_max_size: Option<String>,
    /// The index's bytes without the selected structures (the `B` of the elasticity rule and of
    /// the floor). Default: the base vectors, the label store and the graph file (and with
    /// `--fixed-trees-max-size` the fixed trees' modelled bytes).
    #[arg(long)]
    index_bytes: Option<u64>,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SelectTrack {
    Hybrid,
    Graph,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SelectTreeEffort {
    Frontier,
    Curve,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SelectUnreachable {
    Must,
    AtMax,
}

/// Degree bound of a graph file (`graph.flab`, a directory holding one, or a DiskANN index).
fn graph_degree_bound(path: &Path) -> Result<usize> {
    let p = if path.is_dir() {
        path.join("graph.flab")
    } else {
        path.to_owned()
    };
    if container::peek_magic(&p).ok().as_ref() == Some(graph::MAGIC) {
        let r = container::Reader::open(&p, graph::MAGIC)?;
        Ok(r.params()[2] as usize)
    } else {
        let mut f = std::fs::File::open(&p).with_context(|| format!("opening {}", p.display()))?;
        Ok(diskann_io::Header::read(&mut f)?.max_degree as usize)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum StopArg {
    Elasticity,
    Budget,
}

#[derive(Clone, Copy, ValueEnum)]
enum OrderArg {
    /// Unchanged ids (a copy in the lab's formats).
    Identity,
    /// Breadth-first from the graph's first start.
    Bfs,
    /// k-means tree (`--fanout` clusters per split, leaves of at most `--leaf` points).
    Kmeans,
}

#[derive(Args)]
struct PermuteArgs {
    /// Dataset directory (base vectors, `labels.flab`, query files, ground truth `*.fgt`).
    #[arg(long)]
    dataset: PathBuf,
    /// The graph: a directory with `graph.flab`, a `graph.flab`, or a DiskANN index prefix.
    #[arg(long)]
    graph: PathBuf,
    /// Start table of the graph (default: the graph's `starts.flab`, if any).
    #[arg(long)]
    starts_sidecar: Option<PathBuf>,
    #[arg(long, value_enum)]
    order: OrderArg,
    /// Clusters per split of the k-means tree.
    #[arg(long, default_value_t = 64)]
    fanout: usize,
    /// Most points of a k-means tree leaf.
    #[arg(long, default_value_t = 256)]
    leaf: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Sketch sizes to build over the permuted labels (`sketches-b<b>.flab`); the priorities
    /// are functions of the new ids.
    #[arg(short = 'b', long, value_delimiter = ',')]
    sketches: Vec<usize>,
    /// Seed of the sketch priorities.
    #[arg(long, default_value_t = 0)]
    sketch_seed: u64,
    #[arg(short, long, default_value_t = 4)]
    threads: usize,
    /// Output dataset directory.
    #[arg(long)]
    out_dataset: PathBuf,
    /// Output graph directory (`graph.flab`, `starts.flab`).
    #[arg(long)]
    out_graph: PathBuf,
}

#[derive(Args)]
struct TreeCalibrateArgs {
    /// Dataset directory (`base.u8bin` or `base.fbin`, `labels.flab`).
    #[arg(long)]
    dataset: PathBuf,
    /// The Steiner graph (a directory with `graph.flab`, or a `graph.flab`).
    #[arg(long)]
    graph: PathBuf,
    /// Its start table (the routers' roots as per-label starts).
    #[arg(long)]
    starts: PathBuf,
    /// The calibration of the base graph (`flab calibrate` output) to extend.
    #[arg(long)]
    calibration: PathBuf,
    /// Queries per label-size class (one label of the class sampled in proportion to its size
    /// each).
    #[arg(long, default_value_t = 200)]
    queries: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Largest router frontier measured (above it the curve is extrapolated).
    #[arg(long, default_value_t = 65_536)]
    l_max: usize,
    /// Calibration queries: `label` (midpoints of two members of the sampled label) or `base`
    /// (midpoints of two random data points).
    #[arg(long, default_value = "label")]
    queries_from: String,
    /// Output JSON.
    #[arg(short, long)]
    output: PathBuf,
}

#[derive(Args)]
struct CalibrateArgs {
    /// Dataset directory (`base.u8bin` or `base.fbin`, `labels.flab`).
    #[arg(long)]
    dataset: PathBuf,
    /// The graph (for the graph-search costs): a directory with `graph.flab`, a `graph.flab`,
    /// or a DiskANN index prefix.
    #[arg(long)]
    graph: Option<PathBuf>,
    /// Start table of the graph (optional).
    #[arg(long)]
    starts: Option<PathBuf>,
    /// Static bitsets threshold θ of the exact path (default ⌈n/64⌉; 0: no bitsets).
    #[arg(long)]
    theta: Option<u64>,
    #[arg(long, default_value_t = 3)]
    reps: usize,
    /// Calibration predicates and queries (from random base points).
    #[arg(long, default_value_t = 64)]
    queries: usize,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    /// Midpoint queries of the effort curve (with a graph; 0: no effort curve).
    #[arg(long, default_value_t = 200)]
    effort_queries: usize,
    /// Seed of the effort curve's base pairs.
    #[arg(long, default_value_t = 0)]
    effort_seed: u64,
    /// Largest list size the effort curve measures (above it the curve is extrapolated).
    #[arg(long, default_value_t = 40_960)]
    effort_lmax: usize,
    /// JSON file with the effort curve's base pairs (`{"pairs": [[a, b], ...]}`) instead of
    /// random ones.
    #[arg(long)]
    effort_pairs: Option<PathBuf>,
    /// Output JSON.
    #[arg(short, long)]
    output: PathBuf,
}

#[derive(Args)]
struct RunArgs {
    /// The experiment file (TOML).
    experiment: PathBuf,
    /// Directory for the run directories (overrides the experiment's `runs_dir`).
    #[arg(long)]
    runs_dir: Option<PathBuf>,
    /// Ledger file (overrides the experiment's `ledger`).
    #[arg(long)]
    ledger: Option<PathBuf>,
    /// Validate the experiment and print the resolved configurations without searching.
    #[arg(long)]
    dry_run: bool,
    /// Code version recorded in the outputs (default: the `FLAB_COMMIT` environment variable).
    #[arg(long)]
    commit: Option<String>,
}

fn run_experiment(a: &RunArgs) -> Result<()> {
    let opts = run::Options {
        runs_dir: a.runs_dir.clone(),
        ledger: a.ledger.clone(),
        dry_run: a.dry_run,
        commit: a
            .commit
            .clone()
            .or_else(|| std::env::var("FLAB_COMMIT").ok()),
        command_line: std::env::args().collect(),
    };
    let dirs = run::run(&a.experiment, &opts)?;
    for d in dirs {
        println!("{}", d.display());
    }
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Run(args) => run_experiment(&args),
        Command::Build(args) => diskann_filter_lab::build::cli::run(&args),
        Command::Diag(args) => diskann_filter_lab::diag::run(&args),
        Command::Steiner(args) => diskann_filter_lab::build::steiner::run(&args),
        Command::Convert(c) => convert(c),
        Command::Gt(args) => match dtype(args.dtype, &args.base)? {
            DType::U8 => run_gt::<u8>(&args),
            DType::F32 => run_gt::<f32>(&args),
        },
        Command::GtCompare(args) => run_compare(&args),
        Command::ImportGraph(args) => import_graph(&args),
        Command::ImportStarts(args) => import_starts(&args),
        Command::Starts(StartsCommand::Build(args)) => {
            match DType::from_path(&run::data::base_file(&args.dataset)?)? {
                DType::U8 => build_starts::<u8>(&args),
                DType::F32 => build_starts::<f32>(&args),
            }
        }
        Command::Bitsets(c) => bitsets_command(c),
        Command::Sketches(c) => sketches_command(c),
        Command::Pairs(c) => pairs_command(c),
        Command::Islands(c) => {
            let IslandsCommand::Build { dataset, .. } = &c;
            match DType::from_path(&run::data::base_file(dataset)?)? {
                DType::U8 => islands_command::<u8>(c),
                DType::F32 => islands_command::<f32>(c),
            }
        }
        Command::Select(a) => match DType::from_path(&run::data::base_file(&a.dataset)?)? {
            DType::U8 => select_command::<u8>(a),
            DType::F32 => select_command::<f32>(a),
        },
        Command::Calibrate(a) => {
            let opts = calibrate::Options {
                reps: a.reps,
                queries: a.queries,
                seed: a.seed,
                theta: a.theta,
                effort: calibrate::EffortOptions {
                    queries: a.effort_queries,
                    seed: a.effort_seed,
                    l_max: a.effort_lmax,
                    pairs: match &a.effort_pairs {
                        Some(p) => Some(read_pairs(p)?),
                        None => None,
                    },
                },
            };
            let base = run::data::base_file(&a.dataset)?;
            let report = match DType::from_path(&base)? {
                DType::U8 => calibrate::calibrate::<u8>(
                    &a.dataset,
                    a.graph.as_deref(),
                    a.starts.as_deref(),
                    &opts,
                )?,
                DType::F32 => calibrate::calibrate::<f32>(
                    &a.dataset,
                    a.graph.as_deref(),
                    a.starts.as_deref(),
                    &opts,
                )?,
            };
            if let Some(dir) = a.output.parent() {
                if !dir.as_os_str().is_empty() {
                    std::fs::create_dir_all(dir)?;
                }
            }
            std::fs::write(&a.output, serde_json::to_string_pretty(&report)? + "\n")?;
            println!(
                "calibrated in {:.1} s -> {}",
                report["elapsed_s"].as_f64().unwrap_or(0.0),
                a.output.display()
            );
            Ok(())
        }
        Command::TreeCalibrate(a) => {
            let t0 = std::time::Instant::now();
            let opts = calibrate::TreeOptions {
                queries: a.queries,
                seed: a.seed,
                l_max: a.l_max,
                queries_from: calibrate::TreeQueries::parse(&a.queries_from)?,
            };
            let input: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&a.calibration)?)?;
            let base = run::data::base_file(&a.dataset)?;
            let doc = match DType::from_path(&base)? {
                DType::U8 => calibrate::tree_calibrate::<u8>(
                    &a.dataset,
                    &a.graph,
                    Some(&a.starts),
                    &input,
                    &opts,
                )?,
                DType::F32 => calibrate::tree_calibrate::<f32>(
                    &a.dataset,
                    &a.graph,
                    Some(&a.starts),
                    &input,
                    &opts,
                )?,
            };
            if let Some(dir) = a.output.parent() {
                if !dir.as_os_str().is_empty() {
                    std::fs::create_dir_all(dir)?;
                }
            }
            std::fs::write(&a.output, serde_json::to_string_pretty(&doc)? + "\n")?;
            println!(
                "tree effort curve in {:.1} s -> {}",
                t0.elapsed().as_secs_f64(),
                a.output.display()
            );
            Ok(())
        }
        Command::Permute(args) => match DType::from_path(&run::data::base_file(&args.dataset)?)? {
            DType::U8 => permute::<u8>(&args),
            DType::F32 => permute::<f32>(&args),
        },
        Command::Info { path, dtype: d } => info(&path, d),
    }
}

/// Fractions of the edges `u → v` between data points with `|new(u) − new(v)| < 2^s` for each
/// `s` of `shifts` (the share of neighbors whose vectors, rows and visited bits lie close to
/// the expanded node's).
fn edge_spans(g: &Graph, p: &diskann_filter_lab::permute::Permutation, shifts: &[u32]) -> Vec<f64> {
    let n = g.n_data() as u32;
    let mut hits = vec![0u64; shifts.len()];
    let mut edges = 0u64;
    for u in 0..n {
        let nu = i64::from(p.new_id(u));
        for &v in g.neighbors(u) {
            if v < n {
                edges += 1;
                let span = (nu - i64::from(p.new_id(v))).unsigned_abs();
                for (h, &s) in hits.iter_mut().zip(shifts) {
                    *h += u64::from(span < (1u64 << s));
                }
            }
        }
    }
    hits.iter()
        .map(|&h| h as f64 / edges.max(1) as f64)
        .collect()
}

fn permute<T: Element>(a: &PermuteArgs) -> Result<()> {
    use diskann_filter_lab::permute::{permute_rows, Order, Permutation};
    let t0 = Instant::now();
    let src = std::fs::canonicalize(&a.dataset)?;
    std::fs::create_dir_all(&a.out_dataset)?;
    std::fs::create_dir_all(&a.out_graph)?;
    ensure!(
        std::fs::canonicalize(&a.out_dataset)? != src,
        "the output dataset directory is the input"
    );
    let base_file = run::data::base_file(&a.dataset)?;
    let base = Matrix::<T>::read_bin(&base_file)?;
    let lg = run::data::load_graph(&a.graph, a.starts_sidecar.as_deref(), T::DTYPE)?;
    let order = match a.order {
        OrderArg::Identity => Order::Identity,
        OrderArg::Bfs => Order::Bfs,
        OrderArg::Kmeans => Order::Kmeans {
            fanout: a.fanout,
            leaf: a.leaf,
            seed: a.seed,
        },
    };
    let t = Instant::now();
    let p = order.permutation(&lg.graph, &base, a.threads)?;
    let order_s = secs(t);
    let shifts = [6u32, 9, 12, 15];
    let identity = Permutation::identity(p.len());
    let spans_before = edge_spans(&lg.graph, &identity, &shifts);
    let spans_after = edge_spans(&lg.graph, &p, &shifts);
    println!(
        "order {} of {} points in {order_s:.1} s; edges with |Δid| < 2^{shifts:?}: {spans_before:.3?} -> {spans_after:.3?}",
        order.name(),
        p.len()
    );

    let base_name = base_file
        .file_name()
        .context("base file name")?
        .to_string_lossy()
        .into_owned();
    permute_rows(&base, &p).write_bin(&a.out_dataset.join(&base_name))?;
    drop(base);
    lg.graph
        .permuted(&p)?
        .write(&a.out_graph.join("graph.flab"))?;
    if let Some(s) = &lg.starts {
        s.permuted(&p)?.write(&a.out_graph.join("starts.flab"))?;
    }
    drop(lg);

    let store = LabelStore::read(&a.dataset.join("labels.flab"))?.permute(&p, a.threads)?;
    store.write(&a.out_dataset.join("labels.flab"))?;
    let mut sketch_files = Vec::new();
    for &b in &a.sketches {
        let out = a.out_dataset.join(format!("sketches-b{b}.flab"));
        LabelSketches::build(&store, b, a.sketch_seed, a.threads).write(&out)?;
        sketch_files.push(out);
    }
    drop(store);

    let (mut truths, mut links) = (Vec::new(), Vec::new());
    let mut names: Vec<String> = std::fs::read_dir(&a.dataset)?
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect();
    names.sort();
    for name in &names {
        let from = a.dataset.join(name);
        if name.ends_with(".fgt") {
            let gt = GroundTruth::read(&from)?.permuted(&p)?;
            gt.write(&a.out_dataset.join(name))?;
            let bin = name.trim_end_matches(".fgt").to_owned() + ".bin";
            if names.contains(&bin) {
                gt.write_diskann(&a.out_dataset.join(&bin))?;
            }
            truths.push(name.clone());
        } else if name.starts_with('q')
            && [".u8bin", ".fbin", ".fpred", ".txt", ".meta.tsv"]
                .iter()
                .any(|x| name.ends_with(x))
        {
            let to = a.out_dataset.join(name);
            if to.symlink_metadata().is_ok() {
                std::fs::remove_file(&to)?;
            }
            std::os::unix::fs::symlink(std::fs::canonicalize(&from)?, &to)?;
            links.push(name.clone());
        }
    }
    let map: &[u8] = bytemuck::cast_slice(p.new_of_old());
    std::fs::write(a.out_dataset.join("new_of_old.u32"), map)?;
    let report = json!({
        "tool": "flab permute",
        "dataset": src,
        "graph": a.graph,
        "starts_sidecar": a.starts_sidecar,
        "order": order.name(),
        "seed": a.seed,
        "n_points": p.len(),
        "order_s": order_s,
        "edge_span_shifts": shifts,
        "edge_spans_before": spans_before,
        "edge_spans_after": spans_after,
        "ground_truth": truths,
        "links": links,
        "sketches": sketch_files,
        "sketch_seed": a.sketch_seed,
        "out_graph": a.out_graph,
        "new_of_old": "new_of_old.u32 (u32 little-endian: the new id of every old id)",
        "elapsed_s": secs(t0),
    });
    let text = serde_json::to_string_pretty(&report)? + "\n";
    std::fs::write(a.out_dataset.join("permutation.json"), &text)?;
    std::fs::write(a.out_graph.join("permutation.json"), &text)?;
    println!(
        "permuted in {:.1} s -> {}",
        secs(t0),
        a.out_dataset.display()
    );
    Ok(())
}

fn secs(since: Instant) -> f64 {
    since.elapsed().as_secs_f64()
}

/// The base pairs of an effort calibration from a JSON file `{"pairs": [[a, b], ...]}`.
fn read_pairs(path: &Path) -> Result<Vec<(u32, u32)>> {
    let v: Value = serde_json::from_str(
        &std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?,
    )?;
    let pairs = v["pairs"]
        .as_array()
        .with_context(|| format!("{}: no \"pairs\" array", path.display()))?;
    pairs
        .iter()
        .map(|p| {
            let ab: Vec<u32> = p
                .as_array()
                .map(|x| {
                    x.iter()
                        .filter_map(|y| y.as_u64())
                        .map(|y| y as u32)
                        .collect()
                })
                .unwrap_or_default();
            ensure!(ab.len() == 2, "{}: a pair must be [a, b]", path.display());
            Ok((ab[0], ab[1]))
        })
        .collect()
}

fn convert(c: Convert) -> Result<()> {
    let start = Instant::now();
    match c {
        Convert::Labels {
            format,
            input,
            output,
        } => {
            let store = match format {
                Format::Text => text::read_base_labels(&input)?,
                Format::Jsonl => jsonl::read_base_labels(&input)?,
            };
            store.write(&output)?;
            eprintln!(
                "converted {} -> {} in {:.1} s",
                input.display(),
                output.display(),
                secs(start)
            );
            print_labels(&store);
        }
        Convert::Predicates {
            format,
            labels,
            input,
            output,
        } => {
            let store = LabelStore::read(&labels)?;
            let mut builder = PredicateSetBuilder::new(&store);
            match format {
                Format::Text => text::push_predicates(&input, &mut builder)?,
                Format::Jsonl => jsonl::push_predicates(&input, &mut builder)?,
            };
            let unknown = builder.unknown_names();
            if !unknown.is_empty() {
                let shown: Vec<String> = unknown
                    .iter()
                    .take(20)
                    .map(|(name, count)| format!("{name} ({count})"))
                    .collect();
                println!(
                    "labels absent from the dictionary ({} distinct, occurrences in parentheses): {}{}",
                    unknown.len(),
                    shown.join(", "),
                    if unknown.len() > 20 { ", ..." } else { "" }
                );
            }
            let set = builder.finish();
            set.write(&output)?;
            eprintln!(
                "converted {} -> {} in {:.1} s",
                input.display(),
                output.display(),
                secs(start)
            );
            print_predicates(&set);
        }
        Convert::Vectors {
            dtype: d,
            normalize,
            input,
            output,
        } => {
            match dtype(d, &input)? {
                DType::U8 => {
                    ensure!(!normalize, "--normalize needs f32 vectors");
                    Matrix::<u8>::read_bin(&input)?.write_bin(&output)?;
                }
                DType::F32 => {
                    let mut m = Matrix::<f32>::read_bin(&input)?;
                    if normalize {
                        let norms = vectors::normalize_rows(&mut m);
                        let zero = norms.iter().filter(|&&x| x == 0.0).count();
                        let mut sorted = norms;
                        sorted.sort_by(f64::total_cmp);
                        let at = |p: f64| sorted[((sorted.len() - 1) as f64 * p).round() as usize];
                        if !sorted.is_empty() {
                            println!(
                                "original L2 norms: min {:.4} p50 {:.4} max {:.4}; zero rows: {zero}",
                                at(0.0),
                                at(0.5),
                                at(1.0)
                            );
                        }
                    }
                    m.write_bin(&output)?;
                }
            }
            eprintln!(
                "copied {} -> {}{} in {:.1} s",
                input.display(),
                output.display(),
                if normalize { " (normalized)" } else { "" },
                secs(start)
            );
        }
    }
    Ok(())
}

fn run_gt<T: Element>(a: &GtArgs) -> Result<()> {
    ensure!(
        a.out_bin.is_some() || a.out_ext.is_some(),
        "nothing to write: give --out-bin and/or --out-ext"
    );
    let start = Instant::now();
    let base = Matrix::<T>::read_bin(&a.base)?;
    let queries = Matrix::<T>::read_bin(&a.queries)?;
    let labels = LabelStore::read(&a.labels)?;
    let predicates = PredicateSet::read(&a.predicates)?;
    eprintln!(
        "loaded {} base and {} query vectors ({}-dim {}), {} labels, {} predicates in {:.1} s",
        base.nrows(),
        queries.nrows(),
        base.ncols(),
        T::NAME,
        labels.n_labels(),
        predicates.len(),
        secs(start)
    );
    if T::NAME == "u8" && base.ncols() > 258 {
        eprintln!(
            "warning: u8 distances above 2^24 are rounded in f32; distance ties may be inexact"
        );
    }
    let nq = queries.nrows();
    let step = (nq / 20).max(1);
    let start = Instant::now();
    let progress = |done: usize| {
        if done.is_multiple_of(step) {
            eprintln!("  {done}/{nq} queries, {:.1} s", secs(start));
        }
    };
    let params = gt::Params {
        k: a.k,
        threads: a.threads,
        evaluation: match a.eval {
            EvaluationArg::Auto => gt::Evaluation::Auto,
            EvaluationArg::Forward => gt::Evaluation::Forward,
            EvaluationArg::Postings => gt::Evaluation::Postings,
        },
    };
    let truth = gt::compute(&base, &queries, &labels, &predicates, &params, &progress)?;
    let elapsed = secs(start);
    eprintln!(
        "ground truth of {nq} queries in {elapsed:.1} s on {} threads ({:.2} ms per query and thread)",
        a.threads,
        1e3 * elapsed * a.threads.min(nq.max(1)) as f64 / nq.max(1) as f64
    );
    if let Some(path) = &a.out_bin {
        truth.write_diskann(path)?;
    }
    if let Some(path) = &a.out_ext {
        truth.write(path)?;
    }
    print_gt(&truth);
    Ok(())
}

fn run_compare(a: &GtCompareArgs) -> Result<()> {
    let ours = GroundTruth::read(&a.ext)?;
    let reference = DiskannTruth::read(&a.reference)?;
    let k = a.k.unwrap_or(ours.k().min(reference.k));
    let c = gt::compare(&ours, &reference, k)?;
    println!(
        "top-{k} of {} queries: identical {}, equal up to ties {} (same ids in another order: {}; other ids within the k-th distance tie: {}), mismatching {}",
        c.n_queries,
        c.identical,
        c.equal_up_to_ties,
        c.same_id_set,
        c.equal_up_to_ties - c.same_id_set,
        c.mismatches.len()
    );
    for kind in [
        gt::MismatchKind::Count,
        gt::MismatchKind::Duplicate,
        gt::MismatchKind::Membership,
        gt::MismatchKind::Distances,
    ] {
        let n = c.mismatches.iter().filter(|m| m.kind == kind).count();
        if n > 0 {
            println!("  {kind:?}: {n}");
        }
    }
    for m in c.mismatches.iter().take(a.show) {
        println!("  query {}: {:?}: {}", m.query, m.kind, m.detail);
    }
    if !c.mismatches.is_empty() {
        bail!("{} queries disagree", c.mismatches.len());
    }
    Ok(())
}

/// Size and modification time of an input file, for reports.
fn file_info(path: &Path) -> Result<Value> {
    let meta = std::fs::metadata(path).with_context(|| format!("reading {}", path.display()))?;
    let modified = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs_f64());
    Ok(json!({
        "path": path.display().to_string(),
        "bytes": meta.len(),
        "modified_unix": modified,
    }))
}

fn write_report(path: Option<&Path>, report: &Value) -> Result<()> {
    if let Some(path) = path {
        let text = serde_json::to_string_pretty(report)?;
        std::fs::write(path, text + "\n").with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

fn graph_json(g: &Graph) -> Value {
    json!({
        "n_data": g.n_data(),
        "n_nodes": g.n_nodes(),
        "n_aux": g.n_aux(),
        "aux_kinds": g.aux().kinds().iter().map(|k| k.name()).collect::<Vec<_>>(),
        "r_max": g.r_max(),
        "dtype": g.dtype().map(DType::name),
        "dim": g.dim(),
        "starts": g.starts(),
    })
}

fn import_graph(a: &ImportGraphArgs) -> Result<()> {
    let start = Instant::now();
    let dtype = match (a.dtype, &a.base, &a.data) {
        (Some(arg), ..) => Some(dtype(Some(arg), &a.graph)?),
        (None, Some(base), _) => Some(DType::from_path(base)?),
        (None, None, Some(data)) => Some(DType::from_path(data)?),
        (None, None, None) => None,
    };
    let data = match (&a.data, dtype) {
        (Some(path), Some(t)) => Some((path.as_path(), t)),
        _ => None,
    };
    let (g, layout) = diskann_io::read_index_files(&a.graph, data)?;
    let header = layout.header;
    eprintln!(
        "read {}: {} adjacency lists ({} unused), {} nodes ({} frozen), max degree {}, start {} in {:.1} s",
        a.graph.display(),
        layout.lists,
        layout.unused_lists,
        g.n_nodes(),
        header.n_frozen,
        header.max_degree,
        header.start,
        secs(start)
    );

    let mut base_check = Value::Null;
    if let Some(base) = &a.base {
        let (Some(data), Some(t)) = (&a.data, dtype) else {
            bail!("--base needs --data");
        };
        let c = diskann_io::compare_row_files(data, base, t)?;
        let ok = c.rows == g.n_data() && c.extra_rows == g.n_aux() && c.mismatched_rows == 0;
        println!(
            "vectors of the index vs base: {} base rows, {} mismatching (first: {:?}), {} extra rows; byte-identical files: {}",
            c.rows, c.mismatched_rows, c.first_mismatch, c.extra_rows, c.byte_identical
        );
        base_check = json!({
            "base": file_info(base)?,
            "rows": c.rows,
            "mismatched_rows": c.mismatched_rows,
            "first_mismatch": c.first_mismatch,
            "extra_rows": c.extra_rows,
            "byte_identical": c.byte_identical,
            "rows_identical": ok,
        });
        if !ok {
            let message = format!(
                "the index's vectors do not match the base ({} data points vs {} base rows, {} mismatching rows)",
                g.n_data(),
                c.rows,
                c.mismatched_rows
            );
            ensure!(a.allow_base_mismatch, "{message}");
            eprintln!("WARNING: {message}");
        }
    }

    g.write(&a.output)?;
    ensure!(
        Graph::read(&a.output)? == g,
        "{} does not read back identically",
        a.output.display()
    );
    let seconds = secs(start);
    eprintln!(
        "wrote and verified {} in {seconds:.1} s",
        a.output.display()
    );
    let stats = g.stats();
    print_graph(&g, &stats);
    write_report(
        a.report.as_deref(),
        &json!({
            "tool": "flab import-graph",
            "flab_version": env!("CARGO_PKG_VERSION"),
            "inputs": {
                "graph": file_info(&a.graph)?,
                "data": a.data.as_deref().map(file_info).transpose()?,
            },
            "header": {
                "file_size": header.file_size,
                "max_degree": header.max_degree,
                "start": header.start,
                "n_frozen": header.n_frozen,
                "lists": layout.lists,
                "unused_lists": layout.unused_lists,
            },
            "graph": graph_json(&g),
            "stats": stats.to_json(),
            "base_check": base_check,
            "output": file_info(&a.output)?,
            "seconds": seconds,
        }),
    )
}

fn import_starts(a: &ImportStartsArgs) -> Result<()> {
    let start = Instant::now();
    let g = Graph::read(&a.graph)?;
    let mut s = Starts::new(&g);
    let mut report = json!({
        "tool": "flab import-starts",
        "flab_version": env!("CARGO_PKG_VERSION"),
        "graph": file_info(&a.graph)?,
        "global": s.global(),
    });

    let store = a.labels.as_deref().map(LabelStore::read).transpose()?;
    if let Some(medoids) = &a.medoids {
        let (Some(map), Some(store), Some(labels_path)) = (&a.label_map, &store, &a.labels) else {
            bail!("--medoids needs --label-map and --labels");
        };
        let pairs = starts::read_cxx_medoids(medoids)?;
        let names = starts::read_cxx_label_map(map)?;
        let starts::JoinedMedoids {
            pairs: joined,
            absent,
        } = starts::join_medoids(&pairs, &names, store)?;
        s.set_label_starts(store, &joined)?;
        let with_start = (0..store.n_labels() as u32)
            .filter(|&l| !s.label_starts(l).is_empty())
            .count();
        let mismatches = s.label_mismatches(store)?;
        let auxiliary = joined.iter().filter(|p| !g.is_data(p.1)).count();
        println!(
            "per-label starts: {} lines, {} labels in the map; {} mapped to {with_start} of {} labels of the store ({} labels without a start); {} names absent from the store{}",
            pairs.len(),
            names.len(),
            joined.len(),
            store.n_labels(),
            store.n_labels() - with_start,
            absent.len(),
            if absent.is_empty() {
                String::new()
            } else {
                format!(" (e.g. {})", absent[..absent.len().min(5)].join(", "))
            }
        );
        println!(
            "  starts that are data points without their label: {mismatches}; auxiliary starts: {auxiliary}"
        );
        report["label_starts"] = json!({
            "medoids": file_info(medoids)?,
            "label_map": file_info(map)?,
            "labels": file_info(labels_path)?,
            "dict_hash": format!("{:016x}", store.dict_hash()),
            "lines": pairs.len(),
            "map_size": names.len(),
            "mapped": joined.len(),
            "labels_with_start": with_start,
            "labels_without_start": store.n_labels() - with_start,
            "absent_names": absent.len(),
            "absent_examples": &absent[..absent.len().min(20)],
            "starts_without_their_label": mismatches,
            "auxiliary_starts": auxiliary,
        });
    } else if a.label_map.is_some() {
        bail!("--label-map needs --medoids");
    }

    if let Some(sample_path) = &a.sample {
        let sample = starts::read_id_file(sample_path)?;
        let mut map_info = Value::Null;
        let map = match &a.sample_map {
            None => None,
            Some(path) => {
                let mut map = starts::read_id_file(path)?;
                ensure!(
                    map.len() >= sample.len(),
                    "the sample map has {} entries, the sample {}",
                    map.len(),
                    sample.len()
                );
                let unused = map.len() - sample.len();
                map.truncate(sample.len());
                let differing = sample.iter().zip(&map).filter(|(x, y)| x != y).count();
                let first = sample.iter().zip(&map).position(|(x, y)| x != y);
                println!(
                    "sample map: {} entries ({unused} beyond the sample, unused); differs from the sample at {differing} positions (first: {first:?})",
                    sample.len() + unused
                );
                map_info = json!({
                    "file": file_info(path)?,
                    "entries": sample.len() + unused,
                    "unused_entries": unused,
                    "differing_positions": differing,
                    "first_difference": first,
                });
                Some(map)
            }
        };
        let ascending = sample.windows(2).all(|w| w[0] < w[1]);
        println!(
            "sample: {} entries ({:.2}% of the data points), strictly ascending: {ascending}",
            sample.len(),
            100.0 * sample.len() as f64 / g.n_data().max(1) as f64
        );
        report["sample"] = json!({
            "file": file_info(sample_path)?,
            "entries": sample.len(),
            "ascending": ascending,
            "map": map_info,
        });
        s.set_sample(sample, map)?;
    } else if a.sample_map.is_some() {
        bail!("--sample-map needs --sample");
    }

    s.write(&a.output)?;
    ensure!(
        Starts::read(&a.output)? == s,
        "{} does not read back identically",
        a.output.display()
    );
    eprintln!(
        "wrote and verified {} in {:.1} s",
        a.output.display(),
        secs(start)
    );
    print_starts(&s);
    report["output"] = file_info(&a.output)?;
    report["seconds"] = json!(secs(start));
    write_report(a.report.as_deref(), &report)
}

fn build_starts<T: Element>(a: &StartsBuildArgs) -> Result<()> {
    let strategies = a
        .strategies
        .iter()
        .map(|s| label_starts::Strategy::parse(s))
        .collect::<Result<Vec<_>>>()?;
    let start = Instant::now();
    let base_path = run::data::base_file(&a.dataset)?;
    let labels_path = a
        .labels
        .clone()
        .unwrap_or_else(|| a.dataset.join("labels.flab"));
    let base = Matrix::<T>::read_bin(&base_path)?;
    let store = LabelStore::read(&labels_path)?;
    let g = run::data::load_graph(&a.graph, None, T::DTYPE)?;
    ensure!(
        g.graph.n_data() == base.nrows() && store.n_points() == base.nrows(),
        "the graph has {} data points, the base {}, the labels {}",
        g.graph.n_data(),
        base.nrows(),
        store.n_points()
    );
    let load_s = secs(start);
    eprintln!(
        "loaded {} ({} × {} {}), {} labels, graph {} ({} nodes, global starts {:?}) in {load_s:.1} s",
        base_path.display(),
        base.nrows(),
        base.ncols(),
        T::NAME,
        store.n_labels(),
        a.graph.display(),
        g.graph.n_nodes(),
        g.graph.starts()
    );
    std::fs::create_dir_all(&a.out_dir)
        .with_context(|| format!("creating {}", a.out_dir.display()))?;
    let params = label_starts::Params {
        exact_max: a.exact_max,
        sample: a.sample,
        iters: a.iters,
        seed: a.seed,
        threads: a.threads,
    };
    for strategy in strategies {
        let t = Instant::now();
        let cpu0 = run::sys::process_cpu_s();
        let built = label_starts::build(&base, &store, strategy, &params)?;
        let wall = secs(t);
        let cpu = match (cpu0, run::sys::process_cpu_s()) {
            (Some(a), Some(b)) => Some(b - a),
            _ => None,
        };
        let mut s = Starts::new(&g.graph);
        s.set_label_starts(&store, &built.pairs())?;
        let out = a.out_dir.join(format!("starts-{}.flab", strategy.tag()));
        s.write(&out)?;
        ensure!(
            Starts::read(&out)? == s,
            "{} does not read back identically",
            out.display()
        );
        let mismatches = s.label_mismatches(&store)?;
        ensure!(mismatches == 0, "{mismatches} starts lack their label");
        let mut nodes: Vec<u32> = built.starts.iter().flatten().copied().collect();
        nodes.sort_unstable();
        nodes.dedup();
        let with_start = built.starts.iter().filter(|v| !v.is_empty()).count();
        eprintln!(
            "{strategy}: {} entries for {with_start} of {} labels ({} distinct nodes, {} labels sampled), {:.1} s wall{} -> {}",
            s.n_label_entries(),
            store.n_labels(),
            nodes.len(),
            built.sampled,
            wall,
            cpu.map_or(String::new(), |c| format!(", {c:.1} s CPU on {} threads", a.threads)),
            out.display()
        );
        let report = json!({
            "tool": "flab starts build",
            "flab_version": env!("CARGO_PKG_VERSION"),
            "strategy": strategy.to_string(),
            "params": {
                "exact_max": a.exact_max,
                "sample": a.sample,
                "iters": a.iters,
                "seed": a.seed,
                "threads": a.threads,
            },
            "inputs": {
                "dataset": a.dataset,
                "base": file_info(&base_path)?,
                "labels": file_info(&labels_path)?,
                "graph": a.graph,
            },
            "dict_hash": format!("{:016x}", store.dict_hash()),
            "graph": {
                "n_data": g.graph.n_data(),
                "n_nodes": g.graph.n_nodes(),
                "global": s.global(),
            },
            "counts": {
                "labels": store.n_labels(),
                "labels_with_start": with_start,
                "entries": s.n_label_entries(),
                "distinct_nodes": nodes.len(),
                "labels_sampled": built.sampled,
                "distance_computations": built.n_dist,
                "starts_without_their_label": mismatches,
            },
            "cost": {
                "load_s": load_s,
                "build_wall_s": wall,
                "build_cpu_s": cpu,
                "threads": a.threads,
                "peak_rss_bytes": run::sys::peak_rss_bytes(),
            },
            "output": file_info(&out)?,
        });
        write_report(
            Some(&a.out_dir.join(format!("starts-{}.json", strategy.tag()))),
            &report,
        )?;
    }
    Ok(())
}

fn bitsets_command(c: BitsetsCommand) -> Result<()> {
    match c {
        BitsetsCommand::Build {
            labels,
            theta,
            output,
        } => {
            let store = LabelStore::read(&labels)?;
            let t = Instant::now();
            let b = LabelBitsets::build(&store, theta);
            let build_s = secs(t);
            b.write(&output)?;
            ensure!(
                LabelBitsets::read(&output)? == b,
                "{} does not read back identically",
                output.display()
            );
            println!(
                "θ {}: {} of {} labels with a bitset, {:.1} MiB; built in {build_s:.2} s -> {}",
                b.theta(),
                b.n_bitsets(),
                store.n_labels(),
                b.bytes() as f64 / (1u64 << 20) as f64,
                output.display()
            );
        }
        BitsetsCommand::Stats { labels, theta } => {
            let store = LabelStore::read(&labels)?;
            let n = store.n_points() as u64;
            let thetas = if theta.is_empty() {
                vec![n / 1000, n / 100, n / 32, n / 10]
            } else {
                theta
            };
            let postings = 4 * store.n_entries() as u64;
            println!(
                "{} points, {} labels, {} entries (postings {:.1} MiB)",
                n,
                store.n_labels(),
                store.n_entries(),
                postings as f64 / (1u64 << 20) as f64
            );
            println!(
                "{:>10} {:>10} {:>12} {:>16}",
                "theta", "bitsets", "MiB", "entries covered"
            );
            for t in thetas {
                let (k, bytes, covered) = bitsets::footprint(&store, t);
                println!(
                    "{t:>10} {k:>10} {:>12.1} {:>15.1}%",
                    bytes as f64 / (1u64 << 20) as f64,
                    100.0 * covered as f64 / store.n_entries().max(1) as f64
                );
            }
        }
    }
    Ok(())
}

fn islands_command<T: diskann_filter_lab::build::BuildElement>(c: IslandsCommand) -> Result<()> {
    let IslandsCommand::Build {
        dataset,
        calibration,
        budget,
        k,
        r,
        l_build,
        alpha,
        threads,
        out_dir,
        prior,
        queries,
        seed,
        singleton,
        dry_run,
        restrict,
    } = c;
    let stem = match (prior, &restrict) {
        (_, Some(_)) => "islands-restricted".to_owned(),
        (islands::Prior::Label, None) => "islands".to_owned(),
        (p, None) => format!("islands-{}", p.as_str()),
    };
    let store = LabelStore::read(&dataset.join("labels.flab"))?;
    let costs = run::read_costs(&calibration)?;
    std::fs::create_dir_all(&out_dir)?;
    let gains = islands::gains(&store, &costs, k, r);
    let mut tsv = String::from("label\tsize\texact_ns\tbase_ns\tisland_ns\tgain_ns\tbytes\n");
    for g in &gains {
        tsv += &format!(
            "{}\t{}\t{:.0}\t{:.0}\t{:.0}\t{:.0}\t{}\n",
            g.label, g.size, g.exact_ns, g.base_ns, g.island_ns, g.gain_ns, g.bytes
        );
    }
    if restrict.is_none() {
        std::fs::write(out_dir.join("gains.tsv"), tsv)?;
    }
    let gains: Vec<islands::Gain> = match &restrict {
        Some(path) => {
            let keep: std::collections::BTreeSet<u32> = std::fs::read_to_string(path)?
                .split_whitespace()
                .map(str::parse)
                .collect::<Result<_, _>>()
                .with_context(|| format!("label ids in {}", path.display()))?;
            gains
                .into_iter()
                .filter(|g| keep.contains(&g.label))
                .collect()
        }
        None => gains,
    };
    let unit = 256 * store.n_points() as u64;
    let sample = (prior == islands::Prior::Queries).then(|| {
        let t = Instant::now();
        let model = islands::QueryModel {
            n_queries: queries,
            seed,
            singleton,
            ..Default::default()
        };
        let s = islands::QuerySample::build(&store, &costs, k, model, threads);
        println!(
            "{queries} base-derived queries in {:.1} s; mean cost without islands {:.0} ns",
            secs(t),
            s.mean_base_ns()
        );
        s
    });
    let cand_set: std::collections::HashSet<u32> = gains.iter().map(|g| g.label).collect();
    let candidates = |l: u32| cand_set.contains(&l);
    let sels: Vec<(f64, Vec<u32>)> = budget
        .iter()
        .map(|&f| {
            let b = (f * unit as f64) as u64;
            let sel = match &sample {
                Some(s) => {
                    let (sel, terms) =
                        s.select(&store, b, r, |l| restrict.is_none() || candidates(l));
                    let mut tsv =
                        String::from("label\tsize\tqueries\talone_ns\tmarginal_ns\tbytes\torder\n");
                    for t in &terms {
                        tsv += &format!(
                            "{}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{}\n",
                            t.label,
                            t.size,
                            t.queries,
                            t.alone_ns,
                            t.marginal_ns,
                            t.bytes,
                            t.order.map_or(String::new(), |o| o.to_string())
                        );
                    }
                    let tag = (f * 1000.0).round() as u64;
                    std::fs::write(out_dir.join(format!("gains-queries-b{tag}.tsv")), tsv)
                        .expect("writing the selection terms");
                    sel
                }
                None => islands::select_with(&gains, b, prior),
            };
            (f, sel)
        })
        .collect();
    if dry_run {
        for (f, sel) in &sels {
            let tag = (f * 1000.0).round() as u64;
            let sizes: Vec<usize> = sel.iter().map(|&l| store.posting_len(l)).collect();
            let names: Vec<&str> = sel.iter().map(|&l| store.label_name(l)).collect();
            let bytes: u64 = sizes.iter().map(|&s| islands::island_bytes(s, r)).sum();
            std::fs::write(
                out_dir.join(format!("{stem}-b{tag}.selection.json")),
                serde_json::to_string_pretty(&json!({
                    "budget_fraction": f,
                    "budget_bytes": (f * unit as f64) as u64,
                    "expected_bytes": bytes,
                    "prior": prior,
                    "queries": sample.as_ref().map(|s| s.model),
                    "labels": sel,
                    "names": names,
                    "sizes": sizes,
                }))? + "\n",
            )?;
            println!(
                "budget {f}: {} labels, {} members, {:.1} MiB expected",
                sel.len(),
                sizes.iter().sum::<usize>(),
                bytes as f64 / (1u64 << 20) as f64
            );
        }
        return Ok(());
    }
    let base = Matrix::<T>::read_bin(&run::data::base_file(&dataset)?)?;
    let mut all: Vec<u32> = sels.iter().flat_map(|(_, s)| s.iter().copied()).collect();
    all.sort_unstable();
    all.dedup();
    let params = diskann_filter_lab::build::BuildParams {
        rule: diskann_filter_lab::build::PruneRule::Alpha { r, alpha },
        l_build,
        ..Default::default()
    };
    let t = Instant::now();
    let (union, stats, per_island) = IslandIndex::build(&base, &store, &all, &params, threads)?;
    let wall = secs(t);
    println!(
        "{} islands ({} members, {} edges, {} members patched, {:.1} MiB) in {wall:.1} s wall, {:.1} s summed over islands, {} threads",
        stats.islands,
        stats.members,
        stats.edges,
        stats.patched,
        stats.bytes as f64 / (1u64 << 20) as f64,
        per_island.iter().sum::<f64>(),
        threads
    );
    for (f, sel) in sels {
        let ix = union.subset(&sel);
        let tag = format!("{}", (f * 1000.0).round() as u64);
        let out = out_dir.join(format!("{stem}-b{tag}.flab"));
        ix.write(&out)?;
        ensure!(
            IslandIndex::read(&out)? == ix,
            "{} does not read back identically",
            out.display()
        );
        ix.check(&store)?;
        let build_s: f64 = all
            .iter()
            .zip(&per_island)
            .filter(|(l, _)| sel.binary_search(l).is_ok())
            .map(|(_, s)| s)
            .sum();
        let sizes: Vec<usize> = sel.iter().map(|&l| store.posting_len(l)).collect();
        let report = json!({
            "dataset": dataset,
            "calibration": calibration,
            "prior": prior,
            "queries": sample.as_ref().map(|s| s.model),
            "restrict": restrict,
            "n_points": store.n_points(),
            "budget_fraction": f,
            "budget_bytes": (f * unit as f64) as u64,
            "k": k, "r": r, "l_build": l_build, "alpha": alpha,
            "candidates": gains.iter().filter(|g| g.gain_ns > 0.0).count(),
            "islands": ix.n_islands(),
            "members": ix.n_members(),
            "edges": ix.n_edges(),
            "bytes": ix.bytes(),
            "build_cpu_s": build_s,
            "build_wall_s_union": wall,
            "patched_union": stats.patched,
            "sizes": sizes,
            "labels": sel,
            "output": out,
        });
        std::fs::write(
            out_dir.join(format!("{stem}-b{tag}.json")),
            serde_json::to_string_pretty(&report)? + "\n",
        )?;
        println!(
            "budget {f}: {} islands, {} members, {} edges, {:.1} MiB, {build_s:.1} CPU-s -> {}",
            ix.n_islands(),
            ix.n_members(),
            ix.n_edges(),
            ix.bytes() as f64 / (1u64 << 20) as f64,
            out.display()
        );
    }
    Ok(())
}

fn file_bytes(p: &Path) -> Result<u64> {
    Ok(std::fs::metadata(p)
        .with_context(|| format!("size of {}", p.display()))?
        .len())
}

const CANDS_MAGIC: u64 = 0x5345_4c43_414e_4431; // "SELCAND1": the candidates only
const CONTEXTS_MAGIC: u64 = 0x5345_4c43_414e_4432; // "SELCAND2": every co-occurring pair

/// Writes the pair contexts of `flab select` (every co-occurring pair of labels of different
/// categories; header: magic, points, labels, singleton, count; then `f` per label, then
/// `(a, b, joint, g)` per pair, little-endian).
fn write_candidates(
    path: &Path,
    store: &LabelStore,
    singleton: f64,
    f: &[f64],
    cands: &[diskann_filter_lab::select::PairCand],
) -> Result<()> {
    let mut b: Vec<u8> = Vec::with_capacity(40 + 8 * f.len() + 20 * cands.len());
    for x in [
        CONTEXTS_MAGIC,
        store.n_points() as u64,
        store.n_labels() as u64,
        singleton.to_bits(),
        cands.len() as u64,
    ] {
        b.extend_from_slice(&x.to_le_bytes());
    }
    for x in f {
        b.extend_from_slice(&x.to_le_bytes());
    }
    for c in cands {
        b.extend_from_slice(&c.a.to_le_bytes());
        b.extend_from_slice(&c.b.to_le_bytes());
        b.extend_from_slice(&c.joint.to_le_bytes());
        b.extend_from_slice(&c.g.to_le_bytes());
    }
    std::fs::write(path, b).with_context(|| format!("writing {}", path.display()))
}

/// Reads [`write_candidates`]' file, checking it belongs to `store` and `singleton`; also
/// files of earlier versions that hold only the candidates (labels of 64 to `n − 1` points),
/// flagged by `false`.
fn read_candidates(
    path: &Path,
    store: &LabelStore,
    singleton: f64,
) -> Result<(Vec<f64>, Vec<diskann_filter_lab::select::PairCand>, bool)> {
    let b = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().expect("8 bytes"));
    ensure!(
        b.len() >= 40 && (u64_at(0) == CANDS_MAGIC || u64_at(0) == CONTEXTS_MAGIC),
        "not a candidates file"
    );
    let complete = u64_at(0) == CONTEXTS_MAGIC;
    ensure!(
        u64_at(8) == store.n_points() as u64
            && u64_at(16) == store.n_labels() as u64
            && f64::from_bits(u64_at(24)) == singleton,
        "the candidates belong to another label store or singleton probability"
    );
    let (nl, nc) = (store.n_labels(), u64_at(32) as usize);
    ensure!(
        b.len() == 40 + 8 * nl + 20 * nc,
        "truncated candidates file"
    );
    let f: Vec<f64> = (0..nl)
        .map(|i| f64::from_bits(u64_at(40 + 8 * i)))
        .collect();
    let base = 40 + 8 * nl;
    let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().expect("4 bytes"));
    let cands = (0..nc)
        .map(|i| {
            let o = base + 20 * i;
            diskann_filter_lab::select::PairCand {
                a: u32_at(o),
                b: u32_at(o + 4),
                joint: u32_at(o + 8),
                g: f64::from_bits(u64_at(o + 12)),
            }
        })
        .collect();
    Ok((f, cands, complete))
}

/// `flab select --dump-workload M`: the workload's first `M` base-derived queries and `M`
/// independent queries drawn i.i.d. ([`workload::Walk::draw`]), as query sets of the vectors of
/// their points (gathered from the base file) and their predicates.
fn dump_workload<T: diskann_filter_lab::build::BuildElement>(
    a: &SelectArgs,
    store: &LabelStore,
    base_file: &Path,
    model: &islands::QueryModel,
    contexts: &[diskann_filter_lab::select::PairCand],
    m: usize,
) -> Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let walk = workload::Walk::new(store, contexts);
    let ext = base_file
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("bin")
        .to_string();
    let mut f = std::fs::File::open(base_file)?;
    let mut head = [0u8; 8];
    f.read_exact(&mut head)?;
    let nrows = u32::from_le_bytes(head[0..4].try_into()?) as usize;
    let dim = u32::from_le_bytes(head[4..8].try_into()?) as usize;
    ensure!(
        nrows == store.n_points(),
        "{} holds {nrows} vectors, the label store {} points",
        base_file.display(),
        store.n_points()
    );
    let row_bytes = dim * std::mem::size_of::<T>();
    for (name, drawn) in [
        (
            "d",
            (0..m)
                .filter_map(|i| workload::draw_data(store, model, i))
                .collect::<Vec<_>>(),
        ),
        (
            "u",
            (0..m)
                .filter_map(|i| walk.draw(model, i))
                .collect::<Vec<_>>(),
        ),
    ] {
        let mut pb = PredicateSetBuilder::new(store);
        let mut tsv = String::from("query\tpart\tpoint\tpredicate\n");
        let mut vecs = Vec::with_capacity(8 + drawn.len() * row_bytes);
        vecs.extend_from_slice(&(drawn.len() as u32).to_le_bytes());
        vecs.extend_from_slice(&(dim as u32).to_le_bytes());
        let mut row = vec![0u8; row_bytes];
        for (q, d) in drawn.iter().enumerate() {
            let names: Vec<Vec<&str>> = d
                .clauses
                .iter()
                .map(|c| c.iter().map(|&l| store.label_name(l)).collect())
                .collect();
            pb.push(&names);
            tsv += &format!(
                "{q}\t{}\t{}\t{}\n",
                serde_json::to_value(d.part)?.as_str().unwrap_or("?"),
                d.point,
                names
                    .iter()
                    .map(|c| c.join("|"))
                    .collect::<Vec<_>>()
                    .join(" & ")
            );
            f.seek(SeekFrom::Start(8 + d.point as u64 * row_bytes as u64))?;
            f.read_exact(&mut row)?;
            vecs.extend_from_slice(&row);
        }
        let stem = a.out_dir.join(format!("workload-{name}"));
        pb.finish().write(&stem.with_extension("fpred"))?;
        std::fs::write(stem.with_extension(&ext), vecs)?;
        std::fs::write(stem.with_extension("tsv"), tsv)?;
        println!(
            "workload: {} {} queries to {}.{{fpred,{ext},tsv}}",
            drawn.len(),
            if name == "d" {
                "base-derived"
            } else {
                "independent"
            },
            stem.display()
        );
    }
    Ok(())
}

fn select_command<T: diskann_filter_lab::build::BuildElement>(a: SelectArgs) -> Result<()> {
    use diskann_filter_lab::select;
    let t0 = Instant::now();
    let store = LabelStore::read(&a.dataset.join("labels.flab"))?;
    let costs = run::read_costs(&a.calibration)?;
    std::fs::create_dir_all(&a.out_dir)?;
    let base_file = run::data::base_file(&a.dataset)?;
    let file_index_bytes = file_bytes(&base_file)?
        + file_bytes(&a.dataset.join("labels.flab"))?
        + file_bytes(&a.graph)?;
    let n = store.n_points();
    let graph_track = a.track == SelectTrack::Graph;
    let fixed_max = match &a.fixed_trees_max_size {
        Some(s) => {
            ensure!(
                graph_track,
                "--fixed-trees-max-size fixes track G's label trees (--track graph)"
            );
            Some(diskann_filter_lab::build::steiner::parse_size(s, n)?)
        }
        None => None,
    };
    let is_fixed = |l: u32| fixed_max.is_some_and(|m| (1..=m).contains(&store.posting_len(l)));
    ensure!(
        (0.0..1.0).contains(&a.uniform),
        "--uniform must be in [0, 1)"
    );
    let mut model = islands::QueryModel {
        n_queries: a.queries,
        seed: a.seed,
        singleton: a.singleton,
        fresh: a.fresh,
        uniform: a.uniform,
        ..Default::default()
    };
    let tree_model = if graph_track {
        ensure!(
            a.tree_effort == SelectTreeEffort::Frontier || !costs.tree_r.is_empty(),
            "--tree-effort curve needs a calibration with a tree effort curve (flab tree-calibrate)"
        );
        let r_max = graph_degree_bound(&a.graph)?;
        let mut bf = std::fs::File::open(&base_file)?;
        let (_, dim) = diskann_io::read_bin_header(&mut bf)?;
        Some(diskann_filter_lab::select_g::TreeModel {
            k: a.k,
            recall: a.tree_recall,
            frontier: a.tree_frontier,
            curve: a.tree_effort == SelectTreeEffort::Curve,
            fanout: a.tree_fanout,
            router_bytes: diskann_filter_lab::select_g::TreeModel::router_bytes_of(
                r_max,
                dim * T::DTYPE.size(),
            ),
        })
    } else {
        None
    };
    let curves: Option<diskann_filter_lab::select_g::RouteCurves> = match &a.route_curves {
        Some(path) => {
            ensure!(
                graph_track,
                "--route-curves prices track G's routes (--track graph)"
            );
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?;
            Some(
                serde_json::from_str(&text)
                    .with_context(|| format!("parsing {}", path.display()))?,
            )
        }
        None => None,
    };
    let pricing = tree_model
        .as_ref()
        .map(|tm| diskann_filter_lab::select_g::Pricing {
            tm: *tm,
            c: &costs,
            curves: curves.as_ref(),
            n,
            at_max: a.unreachable == SelectUnreachable::AtMax,
        });
    // Every co-occurring pair of labels of different categories: the pair contexts of the
    // workload, and (labels of 64 to n − 1 points) the candidates.
    let t = Instant::now();
    let (f, contexts, complete) = match &a.load_candidates {
        Some(path) => read_candidates(path, &store, a.singleton)?,
        None => {
            let (counts, f) = select::label_weights(&store, a.singleton, a.threads);
            let contexts = select::pair_contexts(&store, &counts, a.singleton, a.threads);
            (f, contexts, true)
        }
    };
    ensure!(
        complete || a.uniform == 0.0,
        "pair contexts need every co-occurring pair; {} holds only the candidates (count again)",
        a.load_candidates
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default()
    );
    if let Some(path) = &a.save_candidates {
        write_candidates(path, &store, a.singleton, &f, &contexts)?;
    }
    let count_s = secs(t);
    if a.uniform > 0.0 {
        model.contexts = contexts.len() as u64;
    }
    if let Some(m) = a.dump_workload {
        ensure!(
            complete,
            "--dump-workload needs every co-occurring pair (count again or load a SELCAND2 file)"
        );
        dump_workload::<T>(&a, &store, &base_file, &model, &contexts, m)?;
    }
    let t = Instant::now();
    let independent = (a.uniform > 0.0)
        .then(|| std::sync::Arc::new(workload::independent(&store, &contexts, &model, a.threads)));
    let walk_s = secs(t);
    let t = Instant::now();
    let (sample, tree_sample) = match &pricing {
        Some(pr) => (
            None,
            Some(diskann_filter_lab::select_g::TreeSample::build_mixed(
                &store,
                pr,
                model,
                independent.clone(),
                a.threads,
            )),
        ),
        None => (
            Some(islands::QuerySample::build_mixed(
                &store,
                &costs,
                a.k,
                model,
                independent.clone(),
                a.threads,
            )),
            None,
        ),
    };
    let sample_s = secs(t);
    let (mean, (mean_derived, mean_u)) = match (&sample, &tree_sample) {
        (Some(s), _) => (s.mean_base_ns(), s.part_means(&s.base_ns)),
        (_, Some(s)) => (s.mean_base_ns(), s.part_means(&s.base_ns)),
        _ => unreachable!("one sample"),
    };
    let (n_single, n_pair, n_walk) = independent
        .as_deref()
        .map_or((0, 0, 0), |u| (u.n_single, u.n_pair, u.n_walk));
    // `--fixed-trees-max-size`: the fixed trees (their number, expected saving per query and
    // modelled bytes; the same in every greedy run), the mean cost the greedy starts from.
    let fixed = match (fixed_max, &tree_sample, &pricing) {
        (Some(_), Some(ts), Some(pr)) => Some(
            diskann_filter_lab::select_g::TreeGreedy::with_classes(
                ts,
                &store,
                pr,
                &f,
                a.tree_min_records,
            )
            .fix(is_fixed),
        ),
        _ => None,
    };
    let index_bytes = a
        .index_bytes
        .unwrap_or(file_index_bytes + fixed.map_or(0, |(_, _, b)| b));
    let mean_start = mean - fixed.map_or(0.0, |(_, s, _)| s);
    if let Some((nt, s, b)) = fixed {
        println!(
            "fixed label trees (1 to {} points): {nt}, {:.1} MiB modelled, expected saving {s:.0} of {mean:.0} ns per query; the greedy starts at {mean_start:.0} ns",
            fixed_max.unwrap_or(0),
            b as f64 / (1u64 << 20) as f64
        );
    }
    let stop = match a.stop {
        StopArg::Elasticity => select::Stop::Elasticity,
        StopArg::Budget => select::Stop::Budget,
    };
    let floor = match stop {
        select::Stop::Budget => a.kappa * mean_start / index_bytes as f64,
        select::Stop::Elasticity => 0.0,
    };
    println!(
        "{} base-derived queries; independent: {n_single} labels, {n_pair} co-occurring pairs (of {}), {n_walk} three-label walks ({walk_s:.1} s); sample in {sample_s:.1} s; mean cost {mean:.0} ns (base-derived {mean_derived:.0}; independent 1/2/3 labels {:.0} / {:.0} / {:.0}); index {:.2} GiB; stop {}{}",
        a.queries,
        contexts.len(),
        mean_u[0],
        mean_u[1],
        mean_u[2],
        index_bytes as f64 / (1u64 << 30) as f64,
        match stop {
            select::Stop::Elasticity => "elasticity",
            select::Stop::Budget => "budget",
        },
        if stop == select::Stop::Budget {
            format!(" (floor {floor:.3e} ns per query per byte, kappa {})", a.kappa)
        } else {
            String::new()
        }
    );
    let t = Instant::now();
    // Labels whose bound on any pair's value reaches the floor (all of them without a floor).
    let keep: Vec<bool> = {
        let f_max = f.iter().copied().fold(0.0, f64::max);
        (0..store.n_labels() as u32)
            .map(|l| {
                // `--track graph`: the bound is the exact path's; track G's pairs are valued
                // against the router.
                graph_track
                    || floor <= 0.0
                    || select::pair_value_bound(&store, &costs, &model, &f, f_max, l) >= floor
            })
            .collect()
    };
    let n_keep = keep.iter().filter(|&&k| k).count();
    let cands: Vec<select::PairCand> = contexts
        .iter()
        .filter(|c| {
            select::is_candidate_label(&store, c.a)
                && select::is_candidate_label(&store, c.b)
                && keep[c.a as usize]
                && keep[c.b as usize]
        })
        .copied()
        .collect();
    let values = match &pricing {
        Some(_) if !a.pair_trees => Vec::new(),
        Some(pr) => diskann_filter_lab::select_g::pair_tree_values(
            pr, &store, &model, &f, &cands, fixed_max,
        ),
        None => select::pair_values(&store, &costs, a.k, &model, &f, &cands),
    };
    let pairs_s = count_s + secs(t);
    println!(
        "{n_keep} of {} labels can pass the floor; {} co-occurring pairs in different categories, {} candidates, {} with a positive saving, in {pairs_s:.1} s{}",
        store.n_labels(),
        contexts.len(),
        cands.len(),
        values.len(),
        match &a.load_candidates {
            Some(path) => format!(" (counted pairs read from {})", path.display()),
            None => String::new(),
        }
    );
    let mut tsv = String::from("a\tb\tsize_a\tsize_b\tjoint\tprob\tsaving_ns\tbytes\tvalue\n");
    for v in values.iter().take(200_000) {
        tsv += &format!(
            "{}\t{}\t{}\t{}\t{}\t{:.4e}\t{:.0}\t{}\t{:.4e}\n",
            v.cand.a,
            v.cand.b,
            store.posting_len(v.cand.a),
            store.posting_len(v.cand.b),
            v.cand.joint,
            v.prob,
            v.saving_ns,
            v.bytes,
            v.value
        );
    }
    std::fs::write(a.out_dir.join("pair-values.tsv"), tsv)?;
    if let (Some(ts), Some(pr)) = (&tree_sample, &pricing) {
        // Track G: every label tree's value before the greedy (its saving per query on its
        // base-derived queries, pair contexts and single label; per byte).
        let g = diskann_filter_lab::select_g::TreeGreedy::with_classes(
            ts,
            &store,
            pr,
            &f,
            a.tree_min_records,
        );
        let mut tsv =
            String::from("label\tsize\tqueries\tcontexts\tderived_ns\tsaving_ns\tbytes\tvalue\n");
        for t in &g.terms {
            tsv += &format!(
                "{}\t{}\t{}\t{}\t{:.3e}\t{:.3e}\t{}\t{:.4e}\n",
                t.label,
                t.size,
                t.queries,
                t.contexts,
                t.derived_ns,
                t.saving_ns,
                t.bytes,
                t.saving_ns / t.bytes as f64
            );
        }
        std::fs::write(a.out_dir.join("tree-values.tsv"), tsv)?;
    }
    let preds = a
        .report
        .iter()
        .map(|p| {
            let ps = PredicateSet::read(p)?;
            ps.check_dictionary(&store)?;
            Ok((p.clone(), ps))
        })
        .collect::<Result<Vec<_>>>()?;
    let tag = |b: f64| format!("{b}");
    let mut sels = Vec::new();
    for &b in &a.budget {
        ensure!(b >= 0.0 && b.is_finite(), "budgets must be non-negative");
        let params = select::Params {
            k: a.k,
            r: a.r,
            budget_bytes: (b * n as f64) as u64,
            floor,
            model,
            stop,
            mean_ns: mean_start,
            index_bytes,
        };
        let t = Instant::now();
        let sel = match (&sample, &tree_sample, &pricing) {
            (Some(sample), _, _) => select::select(&store, sample, &values, &params),
            (_, Some(ts), Some(pr)) => select::select_with(
                &store,
                || {
                    let mut g = diskann_filter_lab::select_g::TreeGreedy::with_classes(
                        ts,
                        &store,
                        pr,
                        &f,
                        a.tree_min_records,
                    );
                    if fixed_max.is_some() {
                        g.fix(is_fixed);
                    }
                    g
                },
                &values,
                &params,
            ),
            _ => unreachable!("one sample"),
        };
        let select_s = secs(t);
        let best = &sel.best;
        println!(
            "budget {b} B/point: {} {} ({:.1} MiB, {:.0} ns per query), {} {} (T {:?}, {} big labels, {:.1} MiB incl. matrix, {:.0} ns per query); expected saving {:.0} of {mean_start:.0} ns, {:.1} MiB; cut {:.3e}; {select_s:.1} s",
            best.islands.len(),
            if graph_track { "trees" } else { "islands" },
            best.island_bytes as f64 / (1u64 << 20) as f64,
            best.island_saving_ns,
            sel.pairs.len(),
            if graph_track { "pair trees" } else { "pairs" },
            best.t_big,
            sel.big_labels.len(),
            (best.bytes() - best.island_bytes) as f64 / (1u64 << 20) as f64,
            best.pair_saving_ns,
            best.saving_ns(),
            best.bytes() as f64 / (1u64 << 20) as f64,
            best.cut
        );
        println!(
            "  {:.2} B/point spent; mean cost {:.0} → {:.0} ns (−{:.1} %), index +{:.2} %",
            best.bytes() as f64 / n as f64,
            mean_start,
            mean_start - best.saving_ns(),
            100.0 * best.saving_ns() / mean_start,
            100.0 * best.bytes() as f64 / index_bytes as f64
        );
        sels.push((b, params, sel, select_s));
    }
    let common = json!({
        "dataset": a.dataset,
        "calibration": a.calibration,
        "graph": a.graph,
        "n_points": n,
        "index_bytes": index_bytes,
        "kappa": a.kappa,
        "stop": stop,
        "floor_ns_per_query_byte": floor,
        "mean_cost_ns": mean,
        "mean_cost_derived_ns": mean_derived,
        "mean_cost_independent_ns": mean_u,
        "pair_contexts": contexts.len(),
        "independent": {
            "labels": n_single, "pairs": n_pair, "walks": n_walk,
            "k_max": independent.as_deref().map_or(0, |u| u.k_max),
            "seconds": walk_s,
        },
        "model": model,
        "k": a.k, "r": a.r, "l_build": a.l_build, "alpha": a.alpha,
        "threads": a.threads,
        "sample_s": sample_s,
        "pairs_s": pairs_s,
        "labels_passing_bound": n_keep,
        "pair_candidates": cands.len(),
        "pair_values": values.len(),
        "track": if graph_track { "graph" } else { "hybrid" },
        "tree_model": tree_model,
        "route_curves": a.route_curves,
        "unreachable": if a.unreachable == SelectUnreachable::AtMax { "at-max" } else { "must" },
        "pair_trees": a.pair_trees,
        "fixed_trees": fixed.map(|(nt, s, b)| json!({
            "max_size": fixed_max, "labels": nt, "expected_saving_ns": s, "model_bytes": b,
        })),
        "mean_cost_start_ns": mean_start,
        "index_bytes_given": a.index_bytes.is_some(),
    });
    let report_of = |b: f64, sel: &select::Selection, select_s: f64, extra: Value| -> Value {
        let mut r = common.clone();
        let names: Vec<&str> = sel
            .best
            .islands
            .iter()
            .map(|&l| store.label_name(l))
            .collect();
        let sizes: Vec<usize> = sel
            .best
            .islands
            .iter()
            .map(|&l| store.posting_len(l))
            .collect();
        r["budget_bytes_per_point"] = json!(b);
        r["budget_bytes"] = json!((b * n as f64) as u64);
        r["select_s"] = json!(select_s);
        r["best"] = json!(sel.best);
        r["bytes"] = json!(sel.best.bytes());
        r["bytes_per_point"] = json!(sel.best.bytes() as f64 / n as f64);
        r["expected_saving_ns"] = json!(sel.best.saving_ns());
        r["runs"] = json!(sel.runs);
        // `--track graph`: the per-label kind is the router trees.
        let kind = if graph_track { "tree" } else { "island" };
        r[format!("{kind}_names")] = json!(names);
        r[format!("{kind}_sizes")] = json!(sizes);
        r["n_pairs"] = json!(sel.pairs.len());
        r["n_big"] = json!(sel.big_labels.len());
        r["build"] = extra;
        r
    };
    if a.dry_run {
        for (b, _, sel, select_s) in &sels {
            let r = report_of(*b, sel, *select_s, Value::Null);
            std::fs::write(
                a.out_dir.join(format!("sel-b{}.selection.json", tag(*b))),
                serde_json::to_string_pretty(&r)? + "\n",
            )?;
        }
        println!("dry run in {:.1} s", secs(t0));
        return Ok(());
    }
    // Islands: the union once, then each budget's subset (`--track graph`: none; the trees'
    // labels are written for `flab steiner --labels-file`).
    let mut all: Vec<u32> = sels
        .iter()
        .filter(|_| !graph_track)
        .flat_map(|(_, _, s, _)| s.best.islands.iter().copied())
        .collect();
    all.sort_unstable();
    all.dedup();
    let built_islands = if all.is_empty() {
        None
    } else {
        let base = Matrix::<T>::read_bin(&base_file)?;
        let params = diskann_filter_lab::build::BuildParams {
            rule: diskann_filter_lab::build::PruneRule::Alpha {
                r: a.r,
                alpha: a.alpha,
            },
            l_build: a.l_build,
            ..Default::default()
        };
        let t = Instant::now();
        let (union, stats, per_island) =
            IslandIndex::build(&base, &store, &all, &params, a.threads)?;
        let wall = secs(t);
        println!(
            "{} islands ({} members, {:.1} MiB) in {wall:.1} s wall, {:.1} s summed over islands",
            stats.islands,
            stats.members,
            stats.bytes as f64 / (1u64 << 20) as f64,
            per_island.iter().sum::<f64>()
        );
        Some((union, per_island, wall))
    };
    for (b, params, sel, select_s) in &sels {
        let mut build = json!({});
        if graph_track {
            // The trees' labels: those taken, or (`--fixed-trees-max-size`) the fixed ones.
            let trees: Vec<u32> = match fixed_max {
                Some(_) => (0..store.n_labels() as u32)
                    .filter(|&l| is_fixed(l))
                    .collect(),
                None => sel.best.islands.clone(),
            };
            let out = a.out_dir.join(format!("sel-b{}.trees.txt", tag(*b)));
            let text: String = trees.iter().map(|l| format!("{l}\n")).collect();
            std::fs::write(&out, text)?;
            build["trees"] =
                json!({ "output": out, "labels": trees.len(), "fixed": fixed_max.is_some() });
        }
        if let Some((union, per_island, wall)) = &built_islands {
            if !sel.best.islands.is_empty() {
                let ix = union.subset(&sel.best.islands);
                let out = a.out_dir.join(format!("sel-b{}.islands.flab", tag(*b)));
                ix.write(&out)?;
                ensure!(
                    IslandIndex::read(&out)? == ix,
                    "{} does not read back",
                    out.display()
                );
                ix.check(&store)?;
                let cpu: f64 = all
                    .iter()
                    .zip(per_island)
                    .filter(|(l, _)| sel.best.islands.binary_search(l).is_ok())
                    .map(|(_, s)| s)
                    .sum();
                build["islands"] = json!({
                    "output": out, "islands": ix.n_islands(), "members": ix.n_members(),
                    "edges": ix.n_edges(), "bytes": ix.bytes(), "build_cpu_s": cpu,
                    "build_wall_s_union": wall,
                });
            }
        }
        if let (Some(t_big), false) = (sel.best.t_big, sel.pairs.is_empty()) {
            let t = Instant::now();
            let (ix, stats) = PairIndex::build_selected(
                &store,
                t_big,
                &sel.big_labels,
                &sel.pairs,
                params.budget_bytes,
                sel.best.cut,
                a.threads,
            )?;
            let wall = secs(t);
            let out = a.out_dir.join(format!("sel-b{}.pairs.flab", tag(*b)));
            ix.write(&out)?;
            ensure!(
                PairIndex::read(&out)? == ix,
                "{} does not read back",
                out.display()
            );
            ix.check_sample(&store, 97)?;
            let cov: Vec<Value> = preds
                .iter()
                .map(|(path, ps)| pair_coverage(&ix, &store, ps, path))
                .collect();
            println!(
                "budget {b}: pairs T {t_big}, {} big labels, {} stored ({} entries), {:.1} MiB in {wall:.1} s",
                stats.n_big,
                stats.stored_pairs,
                stats.stored_entries,
                stats.bytes as f64 / (1u64 << 20) as f64
            );
            for c in &cov {
                println!("  {c}");
            }
            build["pairs"] = json!({
                "output": out, "stats": stats, "build_wall_s": wall, "threads": a.threads,
                "coverage": cov,
            });
        }
        let r = report_of(*b, sel, *select_s, build);
        std::fs::write(
            a.out_dir.join(format!("sel-b{}.json", tag(*b))),
            serde_json::to_string_pretty(&r)? + "\n",
        )?;
    }
    println!("done in {:.1} s", secs(t0));
    Ok(())
}

fn pairs_command(c: PairsCommand) -> Result<()> {
    match c {
        PairsCommand::Build {
            labels,
            budget,
            threads,
            out_dir,
            report,
            prior,
            nested: _,
            plain,
        } => {
            let store = LabelStore::read(&labels)?;
            std::fs::create_dir_all(&out_dir)?;
            let postings = 4 * store.n_entries() as u64;
            let mut stem = match prior {
                diskann_filter_lab::pairs::Prior::Uniform => "pairs".to_owned(),
                p => format!("pairs-{}", p.as_str()),
            };
            if !plain {
                stem += "-nested";
            }
            ensure!(
                budget.iter().all(|&f| f > 0.0 && f.is_finite()),
                "budgets must be positive"
            );
            let preds = report
                .iter()
                .map(|p| {
                    let ps = PredicateSet::read(p)?;
                    ps.check_dictionary(&store)?;
                    Ok((p.clone(), ps))
                })
                .collect::<Result<Vec<_>>>()?;
            // Builds: one chain per remaining largest budget, writing every requested budget
            // that is a step of it (plain: one budget per build).
            let mut todo = budget;
            todo.sort_by(f64::total_cmp);
            todo.dedup();
            while let Some(&top) = todo.last() {
                let chain = if plain {
                    vec![top]
                } else {
                    diskann_filter_lab::pairs::nested_chain(top)
                };
                let want: Vec<bool> = chain.iter().map(|f| todo.contains(f)).collect();
                todo.retain(|f| !chain.contains(f));
                let bytes: Vec<u64> = chain.iter().map(|f| (f * postings as f64) as u64).collect();
                let t = Instant::now();
                let built = PairIndex::build_chain(&store, &bytes, threads, prior, &want);
                let build_s = secs(t);
                for (f, step) in chain.iter().zip(built) {
                    let Some((ix, stats)) = step else { continue };
                    let f = *f;
                    let tag = format!("{}", (f * 1000.0).round() as u64);
                    let out = out_dir.join(format!("{stem}-b{tag}.flab"));
                    ix.write(&out)?;
                    ensure!(
                        PairIndex::read(&out)? == ix,
                        "{} does not read back identically",
                        out.display()
                    );
                    ix.check_sample(&store, 97)?;
                    let mut cov = Vec::new();
                    for (path, ps) in &preds {
                        cov.push(pair_coverage(&ix, &store, ps, path));
                    }
                    let report = json!({
                        "labels": labels,
                        "n_points": store.n_points(),
                        "n_labels": store.n_labels(),
                        "budget_fraction": f,
                        "nested": !plain,
                        "chain": chain,
                        "stats": stats,
                        "build_s": build_s,
                        "build_note": "build_s: the whole chain's build (shared by its outputs)",
                        "threads": threads,
                        "coverage": cov,
                        "output": out,
                    });
                    std::fs::write(
                        out_dir.join(format!("{stem}-b{tag}.json")),
                        serde_json::to_string_pretty(&report)? + "\n",
                    )?;
                    println!(
                        "budget {f}: T {} ({} big labels, matrix {:.1} MiB), {} co-occurring big pairs, {} stored ({} entries), {:.1} MiB = {:.1} % of the postings; {build_s:.1} s (chain {chain:?}) -> {}",
                        stats.t_big,
                        stats.n_big,
                        stats.matrix_bytes as f64 / (1u64 << 20) as f64,
                        stats.cooc_pairs,
                        stats.stored_pairs,
                        stats.stored_entries,
                        stats.bytes as f64 / (1u64 << 20) as f64,
                        100.0 * stats.bytes as f64 / postings.max(1) as f64,
                        out.display()
                    );
                    for c in &cov {
                        println!("  {c}");
                    }
                }
            }
        }
    }
    Ok(())
}

/// Coverage of a predicate set by combination postings: queries with at least two clauses, and
/// among them those whose best clause pair has fewer candidates than the smallest clause bound
/// (by ×2 and ×10), with exact covers, and the median ratio bound / candidates.
fn pair_coverage(ix: &PairIndex, store: &LabelStore, ps: &PredicateSet, path: &Path) -> Value {
    let mut cov = Cover::default();
    let (mut multi, mut exact, mut x2, mut x10) = (0usize, 0usize, 0usize, 0usize);
    let mut ratios = Vec::new();
    for q in 0..ps.len() {
        let cnf = ps.get(q);
        if cnf.n_clauses() < 2 || cnf.is_unsatisfiable() {
            continue;
        }
        multi += 1;
        let bound = (0..cnf.n_clauses())
            .map(|i| {
                cnf.clause(i)
                    .iter()
                    .map(|&l| store.posting_len(l) as u64)
                    .sum::<u64>()
            })
            .min()
            .unwrap_or(0);
        if ix.cover(
            cnf.n_clauses(),
            |i| cnf.clause(i),
            |l| store.posting_len(l),
            &mut cov,
        ) {
            exact += usize::from(cov.exact);
            let r = bound as f64 / (cov.entries.max(1)) as f64;
            x2 += usize::from(r >= 2.0);
            x10 += usize::from(r >= 10.0);
            ratios.push(r);
        } else {
            ratios.push(1.0);
        }
    }
    ratios.sort_unstable_by(f64::total_cmp);
    let q = |f: f64| {
        ratios
            .get(((ratios.len() as f64 - 1.0) * f) as usize)
            .copied()
            .unwrap_or(1.0)
    };
    json!({
        "predicates": path,
        "multi_clause": multi,
        "exact_covers": exact,
        "saves_x2": x2,
        "saves_x10": x10,
        "ratio_p50": q(0.5),
        "ratio_p90": q(0.9),
        "ratio_p99": q(0.99),
    })
}

fn sketches_command(c: SketchesCommand) -> Result<()> {
    match c {
        SketchesCommand::Build {
            labels,
            b,
            seed,
            threads,
            out_dir,
        } => {
            let store = LabelStore::read(&labels)?;
            std::fs::create_dir_all(&out_dir)?;
            let postings = 4 * store.n_entries() as u64;
            for b in b {
                let t = Instant::now();
                let sk = LabelSketches::build(&store, b, seed, threads);
                let build_s = secs(t);
                let out = out_dir.join(format!("sketches-b{b}.flab"));
                sk.write(&out)?;
                ensure!(
                    LabelSketches::read(&out)? == sk,
                    "{} does not read back identically",
                    out.display()
                );
                let complete = (0..store.n_labels() as u32)
                    .filter(|&l| store.posting_len(l) <= b)
                    .count();
                let report = json!({
                    "labels": labels,
                    "n_points": store.n_points(),
                    "n_labels": store.n_labels(),
                    "b": b,
                    "seed": seed,
                    "entries": sk.n_entries(),
                    "bytes": sk.bytes(),
                    "postings_bytes": postings,
                    "labels_complete": complete,
                    "build_s": build_s,
                    "threads": threads,
                    "output": out,
                });
                std::fs::write(
                    out_dir.join(format!("sketches-b{b}.json")),
                    serde_json::to_string_pretty(&report)? + "\n",
                )?;
                println!(
                    "b {b}: {} entries, {:.1} MiB ({:.1} % of the postings), {complete} of {} labels complete; built in {build_s:.2} s -> {}",
                    sk.n_entries(),
                    sk.bytes() as f64 / (1u64 << 20) as f64,
                    100.0 * sk.bytes() as f64 / postings.max(1) as f64,
                    store.n_labels(),
                    out.display()
                );
            }
        }
        SketchesCommand::Stats { labels, b } => {
            let store = LabelStore::read(&labels)?;
            let postings = 4 * store.n_entries() as u64;
            println!(
                "{} points, {} labels, postings {:.1} MiB",
                store.n_points(),
                store.n_labels(),
                postings as f64 / (1u64 << 20) as f64
            );
            println!(
                "{:>8} {:>12} {:>10} {:>12}",
                "b", "entries", "MiB", "% postings"
            );
            for b in b {
                let entries: u64 = (0..store.n_labels() as u32)
                    .map(|l| store.posting_len(l).min(b) as u64)
                    .sum();
                let bytes = 8 * (entries + store.n_labels() as u64 + 1);
                println!(
                    "{b:>8} {entries:>12} {:>10.1} {:>11.1}%",
                    bytes as f64 / (1u64 << 20) as f64,
                    100.0 * bytes as f64 / postings.max(1) as f64
                );
            }
        }
    }
    Ok(())
}

fn info(path: &Path, d: Option<DTypeArg>) -> Result<()> {
    let magic = container::peek_magic(path)?;
    if &magic == labels::MAGIC {
        print_labels(&LabelStore::read(path)?);
    } else if &magic == predicate::MAGIC {
        print_predicates(&PredicateSet::read(path)?);
    } else if &magic == gt::MAGIC {
        print_gt(&GroundTruth::read(path)?);
    } else if &magic == graph::MAGIC {
        let g = Graph::read(path)?;
        print_graph(&g, &g.stats());
    } else if &magic == starts::MAGIC {
        print_starts(&Starts::read(path)?);
    } else if &magic == sketches::MAGIC {
        let sk = LabelSketches::read(path)?;
        println!(
            "bottom-{} sketches of {} labels over {} points: {} entries, {:.1} MiB, seed {}, dictionary hash {:016x}",
            sk.b(),
            sk.n_labels(),
            sk.n_points(),
            sk.n_entries(),
            sk.bytes() as f64 / (1u64 << 20) as f64,
            sk.seed(),
            sk.dict_hash()
        );
    } else if &magic == bitsets::MAGIC {
        let b = LabelBitsets::read(path)?;
        println!(
            "static bitsets of {} points: θ {}, {} labels with a bitset, {:.1} MiB, dictionary hash {:016x}",
            b.n_points(),
            b.theta(),
            b.n_bitsets(),
            b.bytes() as f64 / (1u64 << 20) as f64,
            b.dict_hash()
        );
    } else {
        match dtype(d, path)? {
            DType::U8 => print_vectors(&Matrix::<u8>::read_bin(path)?),
            DType::F32 => print_vectors(&Matrix::<f32>::read_bin(path)?),
        }
    }
    Ok(())
}

/// Nearest-rank percentiles (min, p10, p50, p90, max).
fn percentiles(mut v: Vec<u64>) -> String {
    if v.is_empty() {
        return "-".into();
    }
    v.sort_unstable();
    let at = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    format!(
        "min {} p10 {} p50 {} p90 {} max {}",
        at(0.0),
        at(0.1),
        at(0.5),
        at(0.9),
        at(1.0)
    )
}

fn print_labels(s: &LabelStore) {
    let n = s.n_points();
    let row_lens: Vec<u64> = (0..n as u32).map(|p| s.row(p).len() as u64).collect();
    println!(
        "points {n}, labels {}, categories {}, entries {} (mean {:.2} labels per point; {}), dictionary hash {:016x}",
        s.n_labels(),
        s.n_categories(),
        s.n_entries(),
        s.n_entries() as f64 / n.max(1) as f64,
        percentiles(row_lens.clone()),
        s.dict_hash()
    );
    println!(
        "points without labels: {}",
        row_lens.iter().filter(|&&l| l == 0).count()
    );
    let mut points_with = vec![0u64; s.n_categories()];
    for p in 0..n as u32 {
        let mut last = u32::MAX;
        for &l in s.row(p) {
            let c = s.category(l);
            if c != last {
                points_with[c as usize] += 1;
                last = c;
            }
        }
    }
    for (c, &points) in points_with.iter().enumerate() {
        let range = s.category_labels(c as u32);
        let entries: usize = range.clone().map(|l| s.posting_len(l)).sum();
        let posting_lens: Vec<u64> = range.clone().map(|l| s.posting_len(l) as u64).collect();
        println!(
            "category {c} {:?}: {} labels (ids {}..{}), {entries} entries, {points} points; posting lengths {}",
            s.category_name(c as u32),
            range.len(),
            range.start,
            range.end,
            percentiles(posting_lens)
        );
    }
}

fn print_predicates(set: &PredicateSet) {
    let nq = set.len();
    let mut clauses = Vec::with_capacity(nq);
    let mut clause_sizes = Vec::new();
    let (mut with_unknown, mut unknown, mut unsatisfiable, mut empty) = (0, 0u64, 0, 0);
    for q in 0..nq {
        let cnf = set.get(q);
        clauses.push(cnf.n_clauses() as u64);
        clause_sizes.extend(cnf.clauses().map(|c| c.len() as u64));
        let u = set.unknown_labels(q);
        with_unknown += usize::from(u > 0);
        unknown += u64::from(u);
        unsatisfiable += usize::from(cnf.is_unsatisfiable());
        empty += usize::from(cnf.n_clauses() == 0);
    }
    println!(
        "predicates {nq}, dictionary hash {:016x}\n  clauses per predicate: {}\n  labels per clause: {}",
        set.dict_hash(),
        percentiles(clauses),
        percentiles(clause_sizes)
    );
    println!(
        "  with unknown labels: {with_unknown} ({unknown} occurrences dropped); unsatisfiable (an empty clause): {unsatisfiable}; without clauses: {empty}"
    );
}

fn print_gt(gt: &GroundTruth) {
    let (nq, k) = (gt.n_queries(), gt.k());
    let matches: Vec<u64> = (0..nq).map(|q| u64::from(gt.n_matches(q))).collect();
    let unions: Vec<u64> = (0..nq).map(|q| u64::from(gt.min_clause_union(q))).collect();
    let zero = matches.iter().filter(|&&m| m == 0).count();
    let below = matches.iter().filter(|&&m| (m as usize) < k).count();
    println!(
        "ground truth: {nq} queries, K = {k}, {} points, dictionary hash {:016x}",
        gt.n_points(),
        gt.dict_hash()
    );
    println!("  matches |M_q|: {}", percentiles(matches));
    println!("  queries with no match: {zero}; with fewer than K matches: {below}");
    println!(
        "  smallest clause, points satisfying it: {}",
        percentiles(unions)
    );
    let mut per_mille: Vec<u64> = (0..nq)
        .filter(|&q| gt.min_clause_union(q) > 0)
        .map(|q| 1000 * u64::from(gt.n_matches(q)) / u64::from(gt.min_clause_union(q)))
        .collect();
    per_mille.sort_unstable();
    println!(
        "  |M_q| / smallest clause (per mille, queries with a non-empty smallest clause): {}",
        percentiles(per_mille)
    );
    for kk in [1, 10, k].into_iter().filter(|&kk| kk <= k) {
        let (mut tied, mut longest, mut total) = (0, 0, 0);
        for q in 0..nq {
            let len = gt.band_at(q, kk).0.len();
            let top = (gt.n_matches(q) as usize).min(kk);
            tied += usize::from(len > top);
            longest = longest.max(len);
            total += len;
        }
        println!(
            "  k = {kk}: queries with distance ties across position k: {tied}; band entries total {total}, longest {longest}"
        );
    }
}

fn print_graph(g: &Graph, st: &GraphStats) {
    let mut kinds: Vec<(&str, usize)> = Vec::new();
    for k in g.aux().kinds() {
        match kinds.iter_mut().find(|e| e.0 == k.name()) {
            Some(e) => e.1 += 1,
            None => kinds.push((k.name(), 1)),
        }
    }
    let kinds: Vec<String> = kinds.iter().map(|(k, n)| format!("{n} {k}")).collect();
    println!(
        "graph: {} nodes = {} data points + {} auxiliary{}; r_max {}; {} edges; vectors: {}",
        g.n_nodes(),
        g.n_data(),
        g.n_aux(),
        if kinds.is_empty() {
            String::new()
        } else {
            format!(" ({})", kinds.join(", "))
        },
        g.r_max(),
        st.n_edges,
        match g.dtype() {
            Some(t) => format!("{} × {}", t.name(), g.dim()),
            None => "not recorded".into(),
        }
    );
    let shown: Vec<String> = g.starts().iter().take(10).map(u32::to_string).collect();
    println!(
        "  starts ({}): {}{}",
        g.starts().len(),
        shown.join(", "),
        if g.starts().len() > 10 { ", ..." } else { "" }
    );
    let summary = |s: &Option<graph::Summary>| s.map_or("-".into(), |s| s.to_string());
    println!(
        "  out-degree of data points: {}; zero: {}; full rows (= r_max): {}",
        summary(&st.degree),
        st.zero_out_degree,
        st.full_rows
    );
    println!(
        "  in-degree of data points: {}; zero: {}",
        summary(&st.in_degree),
        st.zero_in_degree
    );
    println!(
        "  self loops: {}; repeated row entries: {}; edges into auxiliary nodes: {}",
        st.self_loops, st.duplicate_edges, st.edges_to_aux
    );
    if !st.aux_degrees.is_empty() {
        let shown: Vec<String> = st.aux_degrees.iter().take(10).map(u32::to_string).collect();
        println!("  out-degrees of auxiliary nodes: {}", shown.join(", "));
    }
    println!(
        "  reachable from the starts: {} nodes, {} of {} data points ({:.4}%)",
        st.reachable_nodes,
        st.reachable_data,
        g.n_data(),
        100.0 * st.reachable_data as f64 / g.n_data().max(1) as f64
    );
}

fn print_starts(s: &Starts) {
    let shown: Vec<String> = s.global().iter().take(10).map(u32::to_string).collect();
    println!(
        "starts of a graph with {} nodes ({} data points): global ({}): {}",
        s.n_nodes(),
        s.n_data(),
        s.global().len(),
        shown.join(", ")
    );
    if s.n_labels() > 0 {
        let lens: Vec<u64> = (0..s.n_labels() as u32)
            .map(|l| s.label_starts(l).len() as u64)
            .collect();
        let with_start = lens.iter().filter(|&&n| n > 0).count();
        let mut distinct: Vec<u32> = (0..s.n_labels() as u32)
            .flat_map(|l| s.label_starts(l).iter().copied())
            .collect();
        distinct.sort_unstable();
        distinct.dedup();
        println!(
            "  per-label starts over {} labels (dictionary hash {:016x}): {} entries, {with_start} labels with a start, {} distinct nodes; starts per label: {}",
            s.n_labels(),
            s.dict_hash(),
            s.n_label_entries(),
            distinct.len(),
            percentiles(lens)
        );
    }
    if !s.sample().is_empty() {
        let differing = s
            .sample()
            .iter()
            .zip(s.sample_start())
            .filter(|(x, y)| x != y)
            .count();
        println!(
            "  sample: {} entries; entries whose start node is another node: {differing}",
            s.sample().len()
        );
    }
}

fn print_vectors<T: Element>(m: &Matrix<T>) {
    println!("vectors: {} × {} {}", m.nrows(), m.ncols(), T::NAME);
}
