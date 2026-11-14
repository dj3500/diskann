/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Micro-benchmarks of the lab kernel's components (the evaluators, materializers, exact paths
//! and the distance kernel of `diskann_filter_lab::kernel`, called directly, not
//! re-implemented). Single-threaded; pin the process to one CPU. Every section writes one TSV
//! line per measurement.
//!
//! - `checks`: ns per predicate check for every evaluator (mode 1, mode 2, the dense match
//!   bitmap, static bitsets at several thresholds; Boolean and complete), per query, per
//!   row-length bucket of the checked points, under three access patterns: `warm` (64 points
//!   checked 16 times), `cold` (1,024 fresh random points of the bucket, no prefetch) and
//!   `cold_pf` (the same with the kernel's two-stage prefetch over groups of 32, as when
//!   expanding a node: the evaluator's `prefetch_offsets` for the group, then `prefetch`
//!   eight points ahead). Labels and predicates only (no vectors).
//! - `materialize`: per query, the dense match bitmap (from postings, and with static bitsets
//!   for the labels with at least `θ` points) and the sparse intersection (first run after a
//!   cache flush, and the median of three warm runs).
//! - `exact`: per query, the exact paths (`scan_smallest_clause` with each evaluator,
//!   `intersect_then_scan`) after a cache flush, with candidate and distance counts.
//! - `distances`: ns per squared-L2 distance on random base rows (`warm`, `cold`, `cold_pf`).
//!
//! Usage: `kernel_microbench <section> --dataset <dir> --out <file.tsv> [options]`
//! (`--help` per section).

use std::{
    fs::File,
    hint::black_box,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{ensure, Context, Result};
use clap::{Args, Parser, Subcommand};
use diskann_filter_lab::{
    bitsets::LabelBitsets,
    graph::{AuxNodes, Graph},
    kernel::{
        collect::Answers,
        eval::{ClauseSearch, Compiled, DenseBitmap, Eval, EvalCtx, RowScan, StaticBits, Weights},
        exact::{self, ExactPath},
        materialize::Materialized,
        search::{Counters, EvalMode},
    },
    labels::LabelStore,
    predicate::PredicateSet,
    run::{data, sys},
    vectors::{DType, Element, Matrix},
};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    section: Section,
}

#[derive(Subcommand)]
enum Section {
    /// ns per predicate check by evaluator, row length and access pattern.
    Checks(ChecksArgs),
    /// Per-query materializations by match-set size.
    Materialize(MatArgs),
    /// Exact paths per candidate.
    Exact(ExactArgs),
    /// Distance computations.
    Distances(DistArgs),
}

#[derive(Args)]
struct Common {
    /// Dataset directory (`labels.flab`, `q<set>.fpred`; base vectors for `exact`,
    /// `distances`).
    #[arg(long)]
    dataset: PathBuf,
    /// Label store (default `<dataset>/labels.flab`).
    #[arg(long)]
    labels: Option<PathBuf>,
    /// Query sets (`q<set>.fpred`), comma-separated.
    #[arg(long, value_delimiter = ',', default_value = "10K")]
    qsets: Vec<String>,
    /// Queries per query set (a seeded random sample of the satisfiable ones).
    #[arg(long, default_value_t = 300)]
    queries: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Name of the dataset in the output.
    #[arg(long)]
    tag: String,
    /// Output TSV.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Args)]
struct ChecksArgs {
    #[command(flatten)]
    common: Common,
    /// Static-bitset thresholds θ (points per label); default n/1000, n/100, n/32.
    #[arg(long, value_delimiter = ',')]
    thetas: Vec<u64>,
}

#[derive(Args)]
struct MatArgs {
    #[command(flatten)]
    common: Common,
    /// Warm repetitions per query (after the first, cold run).
    #[arg(long, default_value_t = 3)]
    warm_reps: usize,
    /// Static-bitset threshold of `dense_static` (default ⌈n/64⌉).
    #[arg(long)]
    theta: Option<u64>,
}

#[derive(Args)]
struct ExactArgs {
    #[command(flatten)]
    common: Common,
    /// Static-bitset threshold of the `static_bits` evaluator (default n/100).
    #[arg(long)]
    theta: Option<u64>,
    /// Queries per band of the candidate count (`min_c U_c`).
    #[arg(long, default_value_t = 30)]
    per_band: usize,
    /// K of the answer heap.
    #[arg(short, default_value_t = 10)]
    k: usize,
    /// Skip queries whose candidate count `min_c U_c` exceeds this.
    #[arg(long, default_value_t = u64::MAX)]
    max_u: u64,
}

#[derive(Args)]
struct DistArgs {
    #[command(flatten)]
    common: Common,
    /// Chunks of 1,024 distances per pattern.
    #[arg(long, default_value_t = 200)]
    chunks: usize,
}

fn main() -> Result<()> {
    let affinity = sys::affinity();
    if affinity.as_ref().is_none_or(|a| a.len() != 1) {
        eprintln!("warning: not pinned to one CPU (affinity {affinity:?})");
    }
    match Cli::parse().section {
        Section::Checks(a) => checks(&a),
        Section::Materialize(a) => materialize(&a),
        Section::Exact(a) => match data::dataset_dtype(&a.common.dataset)? {
            DType::U8 => exact_paths::<u8>(&a),
            DType::F32 => exact_paths::<f32>(&a),
        },
        Section::Distances(a) => match data::dataset_dtype(&a.common.dataset)? {
            DType::U8 => distances::<u8>(&a),
            DType::F32 => distances::<f32>(&a),
        },
    }
}

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

fn load_labels(c: &Common) -> Result<LabelStore> {
    let path = c
        .labels
        .clone()
        .unwrap_or_else(|| c.dataset.join("labels.flab"));
    LabelStore::read(&path)
}

/// Named query sets.
type QuerySets = Vec<(String, PredicateSet)>;

/// The query sets and a seeded sample of their satisfiable queries: `(set, index)`.
fn load_queries(c: &Common, labels: &LabelStore) -> Result<(QuerySets, Vec<(usize, usize)>)> {
    let mut rng = Rng(c.seed);
    let mut sets = Vec::new();
    let mut picked = Vec::new();
    for (si, q) in c.qsets.iter().enumerate() {
        let q = q.trim_start_matches('q');
        let p = PredicateSet::read(&c.dataset.join(format!("q{q}.fpred")))?;
        p.check_dictionary(labels)?;
        let mut idx: Vec<usize> = (0..p.len())
            .filter(|&i| {
                let cnf = p.get(i);
                cnf.n_clauses() > 0 && !cnf.is_unsatisfiable()
            })
            .collect();
        rng.shuffle(&mut idx);
        idx.truncate(c.queries);
        picked.extend(idx.into_iter().map(|i| (si, i)));
        sets.push((q.to_owned(), p));
    }
    Ok((sets, picked))
}

/// Reads every cache line of a 64 MiB buffer (evicts the private caches and most of the
/// shared L3 between timed operations).
struct Flusher(Vec<u64>);

impl Flusher {
    fn new() -> Self {
        Self((0..(64 << 20) / 8).map(|i| i as u64).collect())
    }

    fn flush(&self) {
        let mut acc = 0u64;
        for line in self.0.chunks(8) {
            acc = acc.wrapping_add(line[0]);
        }
        black_box(acc);
    }
}

fn writer(path: &Path, header: &str) -> Result<BufWriter<File>> {
    let mut w =
        BufWriter::new(File::create(path).with_context(|| format!("creating {}", path.display()))?);
    writeln!(w, "{header}")?;
    Ok(w)
}

/// Shape of a compiled query: clauses, labels, widest clause.
fn shape(q: &Compiled) -> (usize, usize, usize) {
    let width = (0..q.n_clauses())
        .map(|j| q.clause(j).len())
        .max()
        .unwrap_or(0);
    (q.n_clauses(), q.total_labels(), width)
}

const CHUNK: usize = 1024;
const WARM: usize = 64;
const GROUP: usize = 32;
const LOOKAHEAD: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pattern {
    Warm,
    Cold,
    ColdPf,
}

impl Pattern {
    fn name(self) -> &'static str {
        match self {
            Self::Warm => "warm",
            Self::Cold => "cold",
            Self::ColdPf => "cold_pf",
        }
    }
}

/// Checks `pts` `reps` times with evaluator `P`; returns (ns, matches).
#[inline(never)]
fn time_checks<P: Eval>(
    cx: &EvalCtx<'_>,
    pts: &[u32],
    complete: bool,
    prefetch: bool,
    reps: usize,
) -> (u64, u64) {
    let t = Instant::now();
    let mut matches = 0u64;
    for _ in 0..reps {
        if prefetch {
            for group in pts.chunks(GROUP) {
                for &v in group {
                    P::prefetch_offsets(cx, v);
                }
                let la = LOOKAHEAD.min(group.len());
                for &v in &group[..la] {
                    P::prefetch(cx, v);
                }
                for i in 0..group.len() {
                    if i + la < group.len() {
                        P::prefetch(cx, group[i + la]);
                    }
                    matches += u64::from(black_box(P::eval(cx, group[i], complete)).m);
                }
            }
        } else {
            for &v in pts {
                matches += u64::from(black_box(P::eval(cx, v, complete)).m);
            }
        }
    }
    (t.elapsed().as_nanos() as u64, matches)
}

/// Evaluator variants of the `checks` section.
#[derive(Clone, Copy)]
enum Variant {
    Mode1(bool),
    Mode2(bool),
    Bitmap,
    Static(usize, bool),
}

const BUCKETS: [(usize, usize, &str); 8] = [
    (1, 4, "1-4"),
    (5, 8, "5-8"),
    (9, 16, "9-16"),
    (17, 32, "17-32"),
    (33, 64, "33-64"),
    (65, 128, "65-128"),
    (129, 512, "129-512"),
    (513, usize::MAX, ">512"),
];

fn checks(a: &ChecksArgs) -> Result<()> {
    let c = &a.common;
    let t0 = Instant::now();
    let labels = load_labels(c)?;
    let n = labels.n_points();
    let (sets, picked) = load_queries(c, &labels)?;
    let thetas: Vec<u64> = if a.thetas.is_empty() {
        vec![n as u64 / 1000, n as u64 / 100, n as u64 / 32]
    } else {
        a.thetas.clone()
    };
    let mut bitsets = Vec::new();
    for &t in &thetas {
        let tb = Instant::now();
        let b = LabelBitsets::build(&labels, t);
        eprintln!(
            "static bitsets θ {t}: {} labels, {:.1} MiB, built in {:.2} s",
            b.n_bitsets(),
            b.bytes() as f64 / (1u64 << 20) as f64,
            tb.elapsed().as_secs_f64()
        );
        bitsets.push(b);
    }

    // Point pools: every bucket of row lengths, plus all points; shuffled.
    let mut rng = Rng(c.seed ^ 0x5eed);
    let mut pools: Vec<(String, Vec<u32>)> = Vec::new();
    let mut all: Vec<u32> = (0..n as u32).collect();
    rng.shuffle(&mut all);
    for &(lo, hi, name) in &BUCKETS {
        let mut p: Vec<u32> = all
            .iter()
            .copied()
            .filter(|&v| (lo..=hi).contains(&labels.row(v).len()))
            .collect();
        eprintln!("bucket {name}: {} points", p.len());
        if p.len() >= 4 * CHUNK {
            rng.shuffle(&mut p);
            pools.push((name.to_owned(), p));
        }
    }
    pools.push(("all".to_owned(), all));
    let mut cursors = vec![0usize; pools.len()];
    eprintln!(
        "{} queries; inputs ready in {:.1} s",
        picked.len(),
        t0.elapsed().as_secs_f64()
    );

    let mut variants = vec![
        Variant::Mode1(false),
        Variant::Mode1(true),
        Variant::Mode2(false),
        Variant::Mode2(true),
        Variant::Bitmap,
    ];
    for i in 0..thetas.len() {
        variants.push(Variant::Static(i, false));
        variants.push(Variant::Static(i, true));
    }

    let mut w = writer(
        &c.out,
        "ds\tqset\tqidx\tclauses\tlabels\twidth\tu_min\tn_match\tvariant\ttheta\tcomplete\tbucket\tpattern\tchecks\tns\tmatches\tbitset_labels\trow_labels",
    )?;
    let mut bits = vec![0u64; labels.n_labels()];
    let mut q = Compiled::new();
    let mut mat = Materialized::new(n, true);
    let started = Instant::now();
    for (done, &(si, qi)) in picked.iter().enumerate() {
        let (qname, preds) = &sets[si];
        q.compile(&labels, preds.get(qi), Weights::Unit, None, &mut bits);
        mat.build_dense(&labels, &q, n);
        let n_match: u64 = mat.dense.iter().map(|w| u64::from(w.count_ones())).sum();
        let (clauses, n_labels, width) = shape(&q);
        for &var in &variants {
            let (name, theta, complete) = match var {
                Variant::Mode1(c) => ("mode1", 0, c),
                Variant::Mode2(c) => ("mode2", 0, c),
                Variant::Bitmap => ("bitmap", 0, false),
                Variant::Static(i, c) => {
                    q.split_static(&labels, &bitsets[i], thetas[i], c);
                    ("static_bits", thetas[i], c)
                }
            };
            let (big, small) = match var {
                Variant::Static(..) => {
                    let big: usize = (0..clauses).map(|j| q.static_slots(j).len()).sum();
                    (big, n_labels - big)
                }
                _ => (0, n_labels),
            };
            let mut cx = EvalCtx::new(&labels, &q, &bits, &mat.dense, &[]);
            if let Variant::Static(i, _) = var {
                cx = cx.with_static(&bitsets[i]);
            }
            for (b, (bucket, pool)) in pools.iter().enumerate() {
                for pattern in [Pattern::Warm, Pattern::Cold, Pattern::ColdPf] {
                    let (pts, reps) = match pattern {
                        Pattern::Warm => (&pool[..WARM], CHUNK / WARM),
                        Pattern::Cold | Pattern::ColdPf => {
                            if cursors[b] + CHUNK > pool.len() {
                                cursors[b] = 0;
                            }
                            let s = &pool[cursors[b]..cursors[b] + CHUNK];
                            cursors[b] += CHUNK;
                            (s, 1)
                        }
                    };
                    let pf = pattern == Pattern::ColdPf;
                    let run = |pts: &[u32], reps: usize| match var {
                        Variant::Mode1(c) => time_checks::<RowScan>(&cx, pts, c, pf, reps),
                        Variant::Mode2(c) => time_checks::<ClauseSearch>(&cx, pts, c, pf, reps),
                        Variant::Bitmap => time_checks::<DenseBitmap>(&cx, pts, false, pf, reps),
                        Variant::Static(_, c) => time_checks::<StaticBits>(&cx, pts, c, pf, reps),
                    };
                    if pattern == Pattern::Warm {
                        run(pts, 1);
                    }
                    let (ns, m) = run(pts, reps);
                    writeln!(
                        w,
                        "{}\t{qname}\t{qi}\t{clauses}\t{n_labels}\t{width}\t{}\t{n_match}\t{name}\t{theta}\t{}\t{bucket}\t{}\t{}\t{ns}\t{m}\t{big}\t{small}",
                        c.tag,
                        q.min_upper_bound(),
                        u8::from(complete),
                        pattern.name(),
                        pts.len() * reps,
                    )?;
                }
            }
        }
        q.clear_bits(&mut bits);
        if (done + 1) % 100 == 0 {
            eprintln!(
                "  {}/{} queries, {:.1} s",
                done + 1,
                picked.len(),
                started.elapsed().as_secs_f64()
            );
        }
    }
    w.flush()?;
    eprintln!(
        "checks done in {:.1} s; cpu {:?}, load {:?}",
        started.elapsed().as_secs_f64(),
        sys::current_cpu(),
        sys::load_average()
    );
    Ok(())
}

fn median(v: &mut [u64]) -> u64 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn materialize(a: &MatArgs) -> Result<()> {
    let c = &a.common;
    let labels = load_labels(c)?;
    let n = labels.n_points();
    let (sets, picked) = load_queries(c, &labels)?;
    let theta = a.theta.unwrap_or(n.div_ceil(64) as u64);
    let tb = Instant::now();
    let sb = LabelBitsets::build(&labels, theta);
    eprintln!(
        "static bitsets θ {theta}: {} labels, {:.1} MiB, built in {:.2} s",
        sb.n_bitsets(),
        sb.bytes() as f64 / (1u64 << 20) as f64,
        tb.elapsed().as_secs_f64()
    );
    let flusher = Flusher::new();
    let mut w = writer(
        &c.out,
        "ds\tqset\tqidx\tclauses\tlabels\twidth\tu_min\tu_sum\tn_match\tsparse_cand\tdense_first_ns\tsparse_first_ns\tdense_warm_ns\tsparse_warm_ns\ttheta\tstatic_read\tdense_static_first_ns\tdense_static_warm_ns",
    )?;
    let mut bits = vec![0u64; labels.n_labels()];
    let mut q = Compiled::new();
    let mut mat = Materialized::new(n, true);
    let started = Instant::now();
    for &(si, qi) in &picked {
        let (qname, preds) = &sets[si];
        q.compile(&labels, preds.get(qi), Weights::Unit, None, &mut bits);
        let u_sum: u64 = (0..q.n_clauses()).map(|j| q.upper_bound(j)).sum();
        flusher.flush();
        let t = Instant::now();
        black_box(mat.build_dense(&labels, &q, n));
        let dense_first = t.elapsed().as_nanos() as u64;
        flusher.flush();
        let t = Instant::now();
        let cand = black_box(mat.build_sparse(&labels, &q, n));
        let sparse_first = t.elapsed().as_nanos() as u64;
        let n_match = mat.sparse.len();
        flusher.flush();
        let t = Instant::now();
        let static_read = black_box(mat.build_dense_static(&labels, &q, n, &sb));
        let static_first = t.elapsed().as_nanos() as u64;
        let (mut dw, mut sw, mut stw) = (vec![0], vec![0], vec![0]);
        if a.warm_reps > 0 {
            dw.clear();
            sw.clear();
            stw.clear();
        }
        for _ in 0..a.warm_reps {
            let t = Instant::now();
            black_box(mat.build_dense(&labels, &q, n));
            dw.push(t.elapsed().as_nanos() as u64);
            let t = Instant::now();
            black_box(mat.build_sparse(&labels, &q, n));
            sw.push(t.elapsed().as_nanos() as u64);
            let t = Instant::now();
            black_box(mat.build_dense_static(&labels, &q, n, &sb));
            stw.push(t.elapsed().as_nanos() as u64);
        }
        let (clauses, n_labels, width) = shape(&q);
        writeln!(
            w,
            "{}\t{qname}\t{qi}\t{clauses}\t{n_labels}\t{width}\t{}\t{u_sum}\t{n_match}\t{cand}\t{dense_first}\t{sparse_first}\t{}\t{}\t{theta}\t{static_read}\t{static_first}\t{}",
            c.tag,
            q.min_upper_bound(),
            median(&mut dw),
            median(&mut sw),
            median(&mut stw)
        )?;
        q.clear_bits(&mut bits);
    }
    w.flush()?;
    eprintln!(
        "materialize: {} queries in {:.1} s; cpu {:?}, load {:?}",
        picked.len(),
        started.elapsed().as_secs_f64(),
        sys::current_cpu(),
        sys::load_average()
    );
    Ok(())
}

const BANDS: [(u64, u64, &str); 5] = [
    (1, 100, "1-100"),
    (101, 1_000, "101-1K"),
    (1_001, 10_000, "1K-10K"),
    (10_001, 100_000, "10K-100K"),
    (100_001, u64::MAX, ">100K"),
];

fn exact_paths<T: Element>(a: &ExactArgs) -> Result<()> {
    let c = &a.common;
    let labels = load_labels(c)?;
    let n = labels.n_points();
    let base = Matrix::<T>::read_bin(&data::base_file(&c.dataset)?)?;
    ensure!(base.nrows() == n, "base and labels differ in size");
    let graph = Graph::new(n, 1, AuxNodes::untyped())?;
    let vectors = graph.vectors(&base)?;
    let theta = a.theta.unwrap_or(n as u64 / 100);
    let bitsets = LabelBitsets::build(&labels, theta);
    // Every satisfiable query of the sets, then up to `per_band` per band of `min_c U_c`.
    let all = Common {
        queries: usize::MAX,
        dataset: c.dataset.clone(),
        labels: c.labels.clone(),
        qsets: c.qsets.clone(),
        seed: c.seed,
        tag: c.tag.clone(),
        out: c.out.clone(),
    };
    let (sets, mut picked) = load_queries(&all, &labels)?;
    Rng(c.seed ^ 0xe7ac7).shuffle(&mut picked);
    let mut bits = vec![0u64; labels.n_labels()];
    let mut q = Compiled::new();
    let mut per_band = vec![Vec::new(); BANDS.len()];
    for &(si, qi) in &picked {
        q.compile(&labels, sets[si].1.get(qi), Weights::Unit, None, &mut bits);
        let u = q.min_upper_bound();
        q.clear_bits(&mut bits);
        if u > a.max_u {
            continue;
        }
        if let Some(b) = BANDS.iter().position(|&(lo, hi, _)| (lo..=hi).contains(&u)) {
            if per_band[b].len() < a.per_band {
                per_band[b].push((si, qi, b));
            }
        }
    }
    for (b, list) in per_band.iter().enumerate() {
        eprintln!("band {}: {} queries", BANDS[b].2, list.len());
    }
    let qvecs: Vec<Matrix<T>> = sets
        .iter()
        .map(|(name, _)| Matrix::<T>::read_bin(&c.dataset.join(format!("q{name}.{}", ext::<T>()))))
        .collect::<Result<_>>()?;
    let flusher = Flusher::new();
    let mut w = writer(
        &c.out,
        "ds\tqset\tqidx\tband\tclauses\tlabels\tu_min\tpath\tevaluator\ttheta\tns\tn_exact_cand\tn_pred\tn_dist\tn_out",
    )?;
    let mut mat = Materialized::new(n, false);
    let mut answers = Answers::new();
    let (mut ids, mut dists) = (vec![0u32; a.k], vec![0f32; a.k]);
    let paths: [(&str, ExactPath, EvalMode); 4] = [
        (
            "scan_smallest_clause",
            ExactPath::ScanSmallestClause(u64::MAX),
            EvalMode::RowScan,
        ),
        (
            "scan_smallest_clause",
            ExactPath::ScanSmallestClause(u64::MAX),
            EvalMode::ClauseSearch,
        ),
        (
            "scan_smallest_clause",
            ExactPath::ScanSmallestClause(u64::MAX),
            EvalMode::StaticBits,
        ),
        (
            "intersect_then_scan",
            ExactPath::IntersectThenScan(u64::MAX),
            EvalMode::RowScan,
        ),
    ];
    let started = Instant::now();
    for list in &per_band {
        for (rot, &(si, qi, b)) in list.iter().enumerate() {
            let (qname, preds) = &sets[si];
            q.compile(&labels, preds.get(qi), Weights::Unit, None, &mut bits);
            q.split_static(&labels, &bitsets, theta, false);
            let (clauses, n_labels, _) = shape(&q);
            let cx = EvalCtx::new(&labels, &q, &bits, &[], &[]).with_static(&bitsets);
            let qv = qvecs[si].row(qi);
            let mut reference: Option<Vec<u32>> = None;
            for i in 0..paths.len() {
                let (pname, path, eval) = paths[(i + rot) % paths.len()];
                flusher.flush();
                let mut ctr = Counters::default();
                let t = Instant::now();
                exact::run(
                    path,
                    eval,
                    &cx,
                    &vectors,
                    n,
                    qv,
                    a.k,
                    &mut mat,
                    &mut answers,
                    &mut ctr,
                );
                let out = answers.finish(&mut ids, &mut dists);
                let ns = t.elapsed().as_nanos() as u64;
                match &reference {
                    None => reference = Some(ids[..out].to_vec()),
                    Some(r) => ensure!(r[..] == ids[..out], "exact paths disagree on {qname}/{qi}"),
                }
                let ename = match eval {
                    EvalMode::ClauseSearch => "mode2",
                    EvalMode::StaticBits => "static_bits",
                    _ => "mode1",
                };
                writeln!(
                    w,
                    "{}\t{qname}\t{qi}\t{}\t{clauses}\t{n_labels}\t{}\t{pname}\t{ename}\t{}\t{ns}\t{}\t{}\t{}\t{out}",
                    c.tag,
                    BANDS[b].2,
                    q.min_upper_bound(),
                    if eval == EvalMode::StaticBits { theta } else { 0 },
                    ctr.n_exact_cand,
                    ctr.n_pred,
                    ctr.n_dist
                )?;
            }
            q.clear_bits(&mut bits);
        }
    }
    w.flush()?;
    eprintln!(
        "exact: done in {:.1} s; cpu {:?}, load {:?}",
        started.elapsed().as_secs_f64(),
        sys::current_cpu(),
        sys::load_average()
    );
    Ok(())
}

fn ext<T: Element>() -> &'static str {
    match T::DTYPE {
        DType::U8 => "u8bin",
        DType::F32 => "fbin",
    }
}

#[inline(never)]
fn time_distances<T: Element>(base: &Matrix<T>, q: &[T], rows: &[u32], prefetch: bool) -> u64 {
    let t = Instant::now();
    let mut acc = 0.0f32;
    if prefetch {
        for group in rows.chunks(GROUP) {
            let la = LOOKAHEAD.min(group.len());
            for &v in &group[..la] {
                diskann_vector::prefetch_hint_max::<4, _>(base.row(v as usize));
            }
            for i in 0..group.len() {
                if i + la < group.len() {
                    diskann_vector::prefetch_hint_max::<4, _>(base.row(group[i + la] as usize));
                }
                acc += black_box(T::squared_l2(q, base.row(group[i] as usize)));
            }
        }
    } else {
        for &v in rows {
            acc += black_box(T::squared_l2(q, base.row(v as usize)));
        }
    }
    black_box(acc);
    t.elapsed().as_nanos() as u64
}

fn distances<T: Element>(a: &DistArgs) -> Result<()> {
    let c = &a.common;
    let base = Matrix::<T>::read_bin(&data::base_file(&c.dataset)?)?;
    std::hint::black_box(data::touch(base.as_slice()));
    let n = base.nrows();
    let mut rng = Rng(c.seed);
    let q: Vec<T> = base.row((rng.next() % n as u64) as usize).to_vec();
    let mut w = writer(&c.out, "ds\tdtype\tdim\tpattern\tchunk\tdistances\tns")?;
    let warm: Vec<u32> = (0..WARM).map(|_| (rng.next() % n as u64) as u32).collect();
    for chunk in 0..a.chunks {
        for pattern in [Pattern::Warm, Pattern::Cold, Pattern::ColdPf] {
            let (ns, count) = match pattern {
                Pattern::Warm => {
                    let mut total = 0;
                    time_distances(&base, &q, &warm, false);
                    for _ in 0..CHUNK / WARM {
                        total += time_distances(&base, &q, &warm, false);
                    }
                    (total, CHUNK)
                }
                Pattern::Cold | Pattern::ColdPf => {
                    let rows: Vec<u32> =
                        (0..CHUNK).map(|_| (rng.next() % n as u64) as u32).collect();
                    (
                        time_distances(&base, &q, &rows, pattern == Pattern::ColdPf),
                        CHUNK,
                    )
                }
            };
            writeln!(
                w,
                "{}\t{}\t{}\t{}\t{chunk}\t{count}\t{ns}",
                c.tag,
                T::NAME,
                base.ncols(),
                pattern.name()
            )?;
        }
    }
    w.flush()?;
    eprintln!(
        "distances: cpu {:?}, load {:?}",
        sys::current_cpu(),
        sys::load_average()
    );
    Ok(())
}
