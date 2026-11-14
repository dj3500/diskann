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
    container, diskann_io,
    graph::{self, Graph, GraphStats},
    gt::{self, DiskannTruth, GroundTruth},
    jsonl, labels,
    labels::LabelStore,
    predicate::{self, PredicateSet, PredicateSetBuilder},
    starts::{self, Starts},
    text,
    vectors::{self, DType, Element, Matrix},
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

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Convert(c) => convert(c),
        Command::Gt(args) => match dtype(args.dtype, &args.base)? {
            DType::U8 => run_gt::<u8>(&args),
            DType::F32 => run_gt::<f32>(&args),
        },
        Command::GtCompare(args) => run_compare(&args),
        Command::ImportGraph(args) => import_graph(&args),
        Command::ImportStarts(args) => import_starts(&args),
        Command::Info { path, dtype: d } => info(&path, d),
    }
}

fn secs(since: Instant) -> f64 {
    since.elapsed().as_secs_f64()
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
