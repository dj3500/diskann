/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Kernel configuration: one variant per role, their text forms, validation of role
//! combinations, and the reference presets of the baselines.
//!
//! Text forms (used in experiment files and reports): `name` or `name(arg, ...)`, e.g.
//! `bonus(0.5)`, `adaptive_l(1000, 16)`, `projection(C3)`, `sample(10)`. Start sources are
//! joined with `+` (`sample(10)+global`).

use std::fmt;

use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};

pub use super::{collect::Collector, eval::Weights, exact::ExactPath, materialize::Materializer};

/// A source of start nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StartSource {
    /// The graph's global starts.
    Global,
    /// Per-label starts of the labels of the clause with the smallest `U_c` (`label(c*)`) of
    /// the start predicate (the label view's predicate if the run has a view, else the
    /// predicate).
    LabelSmallest,
    /// Per-label starts of every label of the start predicate (`label(all)`; `product` for a
    /// product-label view).
    LabelAll,
    /// The first `k` sample entries (in sample order) whose point fully matches the predicate.
    Sample(usize),
    /// Every auxiliary node.
    Aux,
}

/// Whether starts obey the admission rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StartAdmission {
    /// Starts always enter the beam (upstream); they are answers only if they match.
    Force,
    /// A start enters only if admitted (C++).
    Obey,
}

/// Which node a selected sample entry starts from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleStart {
    /// The sample point itself (it matches by construction).
    Intended,
    /// The start node recorded for the entry in the sidecar (e.g. the ids an earlier build's
    /// search actually used).
    Recorded,
}

/// Which evaluated nodes may enter the navigation beam.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Admission {
    All,
    /// Full matches only.
    Strict,
    /// Nodes satisfying the clauses whose labels all lie in the named categories (only these
    /// clauses are evaluated during the search).
    Projection(Vec<String>),
    /// Nodes failing at most `τ` clauses (unit count).
    Cutoff(u32),
}

/// Order of distance and predicate evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Laziness {
    /// Distance, then predicate.
    Eager,
    /// Distance only; the predicate runs on the final beam (post-filter).
    Deferred,
    /// Predicate first; rejected nodes get no distance.
    PredicateFirst,
    /// Distance first; the predicate is skipped when the node can neither enter the beam nor
    /// the answers.
    DistanceFirst,
}

/// Scorer role (see [`super::score`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scorer {
    Distance,
    Bonus(f32),
    Additive(f32),
    NormalizedAdditive(f32),
    Multiplicative(f32),
    Shifted(f32),
    ShellAdaptive(f32),
    Lexicographic,
}

/// Monomorphized scorer families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    Distance,
    Bonus,
    Additive,
    Multiplicative,
    Lexicographic,
}

/// Expander role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Expander {
    OneHop,
    /// Upstream multihop: the step's rejected neighbours, sorted by distance and truncated to
    /// `cap` (default: half the graph's row width, i.e. upstream's `max_degree_with_slack / 2`
    /// for upstream-built graphs), expand their matching neighbours.
    TwoHop(Option<usize>),
}

/// Terminator role.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Terminator {
    /// Stop when no unexpanded node remains.
    Exhaust,
    /// Upstream adaptive L (sample size, maximum multiplier).
    AdaptiveL { sample: usize, scale: f64 },
    /// Stop after this many distance computations.
    Budget(u64),
    /// Stop when the answer heap is full and no unexpanded node can be closer than its worst.
    MatchAware,
}

/// Predicate evaluation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PredicateMode {
    /// Mode 1: row scan with clause bits.
    RowScan,
    /// Mode 2: per-clause galloping search.
    ClauseSearch,
    /// Lookup in the materialized match set.
    Bitmap,
    /// Static per-label bitsets for the labels with at least `θ` points, the row for the
    /// others (needs bitsets built with a threshold of at most `θ`).
    StaticBits(u64),
}

/// Beam capacity convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Padding {
    /// Upstream: capacity and expandable prefix `L + |starts|`; answers come from the first `L`
    /// non-auxiliary entries.
    Upstream,
    /// C++ OR-filtered search: capacity `max(L, |starts|)`; answers come from the whole beam.
    Cxx,
    /// C++ multi-filter search: capacity exactly `L` (starts beyond it compete for the slots
    /// like any node); answers come from the whole beam.
    Exact,
}

/// A complete kernel configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Kernel {
    pub starts: Vec<StartSource>,
    pub start_admission: StartAdmission,
    pub sample_start: SampleStart,
    pub admission: Admission,
    pub laziness: Laziness,
    pub scorer: Scorer,
    pub collector: Collector,
    pub expander: Expander,
    pub terminator: Terminator,
    pub predicate: PredicateMode,
    pub materializer: Materializer,
    pub exact: ExactPath,
    pub weights: Weights,
    /// Nodes expanded per step (`W`).
    pub beam_width: usize,
    pub padding: Padding,
    /// `g` used to key auxiliary nodes.
    pub aux_g: f32,
}

impl Default for Kernel {
    /// The plain search: global start (force), everything admitted, eager evaluation, distance
    /// keys, final-beam collection, one hop, exhaustive termination, mode 1.
    fn default() -> Self {
        Self {
            starts: vec![StartSource::Global],
            start_admission: StartAdmission::Force,
            sample_start: SampleStart::Intended,
            admission: Admission::All,
            laziness: Laziness::Eager,
            scorer: Scorer::Distance,
            collector: Collector::FinalBeam,
            expander: Expander::OneHop,
            terminator: Terminator::Exhaust,
            predicate: PredicateMode::RowScan,
            materializer: Materializer::None,
            exact: ExactPath::None,
            weights: Weights::Unit,
            beam_width: 1,
            padding: Padding::Upstream,
            aux_g: 0.0,
        }
    }
}

/// Splits `name(a, b)` into the name and its arguments.
fn call(s: &str) -> Result<(String, Vec<String>)> {
    let s = s.trim();
    match s.find('(') {
        None => Ok((s.to_owned(), Vec::new())),
        Some(i) => {
            ensure!(s.ends_with(')'), "unbalanced parentheses in {s:?}");
            let args = s[i + 1..s.len() - 1]
                .split(',')
                .map(|a| a.trim().to_owned())
                .filter(|a| !a.is_empty())
                .collect();
            Ok((s[..i].trim().to_owned(), args))
        }
    }
}

fn num<T: std::str::FromStr>(s: &str, what: &str) -> Result<T> {
    s.parse().ok().with_context(|| format!("bad {what} {s:?}"))
}

fn arity(name: &str, args: &[String], n: usize) -> Result<()> {
    ensure!(
        args.len() == n,
        "{name} takes {n} argument(s), got {}",
        args.len()
    );
    Ok(())
}

impl StartSource {
    pub fn parse(s: &str) -> Result<Self> {
        let (name, args) = call(s)?;
        Ok(match (name.as_str(), args.as_slice()) {
            ("global", []) => Self::Global,
            ("label", [a]) if a == "c*" || a == "smallest" => Self::LabelSmallest,
            ("label", [a]) if a == "all" => Self::LabelAll,
            ("product", []) => Self::LabelAll,
            ("sample", [k]) => Self::Sample(num(k, "sample size")?),
            ("aux", []) => Self::Aux,
            _ => bail!("unknown start source {s:?} (global, label(c*), label(all), product, sample(k), aux)"),
        })
    }

    /// Parses `a+b+c`.
    pub fn parse_list(s: &str) -> Result<Vec<Self>> {
        s.split('+').map(Self::parse).collect()
    }
}

impl fmt::Display for StartSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Global => write!(f, "global"),
            Self::LabelSmallest => write!(f, "label(c*)"),
            Self::LabelAll => write!(f, "label(all)"),
            Self::Sample(k) => write!(f, "sample({k})"),
            Self::Aux => write!(f, "aux"),
        }
    }
}

impl StartAdmission {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim() {
            "force" => Self::Force,
            "obey" => Self::Obey,
            _ => bail!("unknown start admission {s:?} (force, obey)"),
        })
    }
}

impl fmt::Display for StartAdmission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Force => "force",
            Self::Obey => "obey",
        })
    }
}

impl SampleStart {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim() {
            "intended" => Self::Intended,
            "recorded" => Self::Recorded,
            _ => bail!("unknown sample start {s:?} (intended, recorded)"),
        })
    }
}

impl fmt::Display for SampleStart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Intended => "intended",
            Self::Recorded => "recorded",
        })
    }
}

impl Admission {
    pub fn parse(s: &str) -> Result<Self> {
        let (name, args) = call(s)?;
        Ok(match name.as_str() {
            "all" => {
                arity(&name, &args, 0)?;
                Self::All
            }
            "strict" => {
                arity(&name, &args, 0)?;
                Self::Strict
            }
            "projection" => {
                ensure!(
                    !args.is_empty(),
                    "projection needs at least one category name"
                );
                Self::Projection(args)
            }
            "cutoff" => {
                arity(&name, &args, 1)?;
                Self::Cutoff(num(&args[0], "cutoff")?)
            }
            _ => bail!("unknown admission {s:?} (all, strict, projection(categories), cutoff(τ))"),
        })
    }
}

impl fmt::Display for Admission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::All => write!(f, "all"),
            Self::Strict => write!(f, "strict"),
            Self::Projection(c) => write!(f, "projection({})", c.join(", ")),
            Self::Cutoff(t) => write!(f, "cutoff({t})"),
        }
    }
}

impl Laziness {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim() {
            "eager" => Self::Eager,
            "deferred" => Self::Deferred,
            "predicate_first" => Self::PredicateFirst,
            "distance_first" => Self::DistanceFirst,
            _ => bail!("unknown laziness {s:?} (eager, deferred, predicate_first, distance_first)"),
        })
    }
}

impl fmt::Display for Laziness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Eager => "eager",
            Self::Deferred => "deferred",
            Self::PredicateFirst => "predicate_first",
            Self::DistanceFirst => "distance_first",
        })
    }
}

impl Scorer {
    pub fn parse(s: &str) -> Result<Self> {
        let (name, args) = call(s)?;
        let one = |what: &str| -> Result<f32> {
            arity(&name, &args, 1)?;
            num(&args[0], what)
        };
        Ok(match name.as_str() {
            "distance" => {
                arity(&name, &args, 0)?;
                Self::Distance
            }
            "bonus" => Self::Bonus(one("β")?),
            "additive" => Self::Additive(one("λ")?),
            "normalized_additive" => Self::NormalizedAdditive(one("η")?),
            "multiplicative" => Self::Multiplicative(one("η")?),
            "shifted" => Self::Shifted(one("η")?),
            "shell_adaptive" => Self::ShellAdaptive(one("κ")?),
            "lexicographic" => {
                arity(&name, &args, 0)?;
                Self::Lexicographic
            }
            _ => bail!(
                "unknown scorer {s:?} (distance, bonus(β), additive(λ), normalized_additive(η), \
                 multiplicative(η), shifted(η), shell_adaptive(κ), lexicographic)"
            ),
        })
    }

    pub fn family(self) -> Family {
        match self {
            Self::Distance => Family::Distance,
            Self::Bonus(_) => Family::Bonus,
            Self::Additive(_) | Self::NormalizedAdditive(_) => Family::Additive,
            Self::Multiplicative(_) | Self::Shifted(_) | Self::ShellAdaptive(_) => {
                Family::Multiplicative
            }
            Self::Lexicographic => Family::Lexicographic,
        }
    }

    /// Whether keys depend on the failed-clause count.
    pub fn uses_g(self) -> bool {
        !matches!(self, Self::Distance | Self::Bonus(_))
    }

    /// Whether the scorer needs the per-query scales of the pilot.
    pub fn needs_pilot(self) -> bool {
        matches!(
            self,
            Self::NormalizedAdditive(_)
                | Self::Multiplicative(_)
                | Self::Shifted(_)
                | Self::ShellAdaptive(_)
        )
    }
}

impl fmt::Display for Scorer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Distance => write!(f, "distance"),
            Self::Bonus(b) => write!(f, "bonus({b})"),
            Self::Additive(l) => write!(f, "additive({l})"),
            Self::NormalizedAdditive(e) => write!(f, "normalized_additive({e})"),
            Self::Multiplicative(e) => write!(f, "multiplicative({e})"),
            Self::Shifted(e) => write!(f, "shifted({e})"),
            Self::ShellAdaptive(k) => write!(f, "shell_adaptive({k})"),
            Self::Lexicographic => write!(f, "lexicographic"),
        }
    }
}

fn parse_collector(s: &str) -> Result<Collector> {
    Ok(match s.trim() {
        "final_beam" => Collector::FinalBeam,
        "heap" => Collector::Heap,
        "all" => Collector::All,
        _ => bail!("unknown collector {s:?} (final_beam, heap, all)"),
    })
}

fn collector_name(c: Collector) -> &'static str {
    match c {
        Collector::FinalBeam => "final_beam",
        Collector::Heap => "heap",
        Collector::All => "all",
    }
}

impl Expander {
    pub fn parse(s: &str) -> Result<Self> {
        let (name, args) = call(s)?;
        Ok(match (name.as_str(), args.as_slice()) {
            ("one_hop", []) => Self::OneHop,
            ("two_hop", []) => Self::TwoHop(None),
            ("two_hop", [c]) => Self::TwoHop(Some(num(c, "two-hop cap")?)),
            _ => bail!("unknown expander {s:?} (one_hop, two_hop, two_hop(cap))"),
        })
    }
}

impl fmt::Display for Expander {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OneHop => write!(f, "one_hop"),
            Self::TwoHop(None) => write!(f, "two_hop"),
            Self::TwoHop(Some(c)) => write!(f, "two_hop({c})"),
        }
    }
}

impl Terminator {
    pub fn parse(s: &str) -> Result<Self> {
        let (name, args) = call(s)?;
        Ok(match (name.as_str(), args.as_slice()) {
            ("exhaust", []) => Self::Exhaust,
            ("adaptive_l", []) => Self::AdaptiveL {
                sample: 1000,
                scale: 16.0,
            },
            ("adaptive_l", [n, m]) => Self::AdaptiveL {
                sample: num(n, "adaptive-L sample")?,
                scale: num(m, "adaptive-L scale")?,
            },
            ("budget", [w]) => Self::Budget(num(w, "budget")?),
            ("match_aware", []) => Self::MatchAware,
            _ => bail!(
                "unknown terminator {s:?} (exhaust, adaptive_l(sample, scale), budget(W), match_aware)"
            ),
        })
    }
}

impl fmt::Display for Terminator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exhaust => write!(f, "exhaust"),
            Self::AdaptiveL { sample, scale } => write!(f, "adaptive_l({sample}, {scale})"),
            Self::Budget(w) => write!(f, "budget({w})"),
            Self::MatchAware => write!(f, "match_aware"),
        }
    }
}

impl PredicateMode {
    pub fn parse(s: &str) -> Result<Self> {
        let (name, args) = call(s)?;
        Ok(match (name.as_str(), args.as_slice()) {
            ("mode1" | "row_scan", []) => Self::RowScan,
            ("mode2" | "clause_search", []) => Self::ClauseSearch,
            ("bitmap", []) => Self::Bitmap,
            ("static_bits", [t]) => Self::StaticBits(num(t, "θ_bits")?),
            _ => bail!("unknown predicate mode {s:?} (mode1, mode2, bitmap, static_bits(θ))"),
        })
    }
}

impl fmt::Display for PredicateMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RowScan => write!(f, "mode1"),
            Self::ClauseSearch => write!(f, "mode2"),
            Self::Bitmap => write!(f, "bitmap"),
            Self::StaticBits(t) => write!(f, "static_bits({t})"),
        }
    }
}

fn parse_materializer(s: &str) -> Result<Materializer> {
    Ok(match s.trim() {
        "none" => Materializer::None,
        "dense" => Materializer::Dense,
        "sparse" => Materializer::Sparse,
        _ => bail!("unknown materializer {s:?} (none, dense, sparse)"),
    })
}

fn materializer_name(m: Materializer) -> &'static str {
    match m {
        Materializer::None => "none",
        Materializer::Dense => "dense",
        Materializer::Sparse => "sparse",
    }
}

fn parse_exact(s: &str) -> Result<ExactPath> {
    let (name, args) = call(s)?;
    Ok(match (name.as_str(), args.as_slice()) {
        ("none", []) => ExactPath::None,
        ("scan_smallest_clause", [t]) => ExactPath::ScanSmallestClause(num(t, "T_exact")?),
        ("intersect_then_scan", [t]) => ExactPath::IntersectThenScan(num(t, "T_exact")?),
        _ => bail!(
            "unknown exact path {s:?} (none, scan_smallest_clause(T), intersect_then_scan(T))"
        ),
    })
}

fn exact_name(e: ExactPath) -> String {
    match e {
        ExactPath::None => "none".into(),
        ExactPath::ScanSmallestClause(t) => format!("scan_smallest_clause({t})"),
        ExactPath::IntersectThenScan(t) => format!("intersect_then_scan({t})"),
    }
}

fn parse_weights(s: &str) -> Result<Weights> {
    Ok(match s.trim() {
        "unit" => Weights::Unit,
        "surprisal" => Weights::Surprisal,
        _ => bail!("unknown weights {s:?} (unit, surprisal)"),
    })
}

fn parse_padding(s: &str) -> Result<Padding> {
    Ok(match s.trim() {
        "upstream" => Padding::Upstream,
        "cxx" => Padding::Cxx,
        "exact" => Padding::Exact,
        _ => bail!("unknown padding {s:?} (upstream, cxx, exact)"),
    })
}

impl Kernel {
    /// Applies one role override `key = value` (the keys of an experiment file's `[[config]]`
    /// table).
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "starts" => self.starts = StartSource::parse_list(value)?,
            "start_admission" => self.start_admission = StartAdmission::parse(value)?,
            "sample_start" => self.sample_start = SampleStart::parse(value)?,
            "admission" => self.admission = Admission::parse(value)?,
            "laziness" => self.laziness = Laziness::parse(value)?,
            "scorer" => self.scorer = Scorer::parse(value)?,
            "collector" => self.collector = parse_collector(value)?,
            "expander" => self.expander = Expander::parse(value)?,
            "terminator" => self.terminator = Terminator::parse(value)?,
            "predicate" => self.predicate = PredicateMode::parse(value)?,
            "materializer" => self.materializer = parse_materializer(value)?,
            "exact" => self.exact = parse_exact(value)?,
            "weights" => self.weights = parse_weights(value)?,
            "beam_width" => self.beam_width = num(value, "beam width")?,
            "padding" => self.padding = parse_padding(value)?,
            "aux_g" => self.aux_g = num(value, "aux_g")?,
            _ => bail!("unknown kernel option {key:?}"),
        }
        Ok(())
    }

    /// Whether the predicate evaluation must return the failed-clause count.
    pub fn needs_g(&self) -> bool {
        self.scorer.uses_g() || matches!(self.admission, Admission::Cutoff(_))
    }

    /// Rejects illegal role combinations.
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.starts.is_empty(), "no start source");
        ensure!(self.beam_width >= 1, "the beam width must be at least 1");
        ensure!(
            self.aux_g.is_finite() && self.aux_g >= 0.0,
            "aux_g must be finite and >= 0"
        );
        for s in &self.starts {
            if let StartSource::Sample(k) = s {
                ensure!(*k >= 1, "sample(k) needs k >= 1");
            }
        }
        match self.scorer {
            Scorer::Bonus(b) => ensure!(b > 0.0 && b.is_finite(), "bonus(β) needs β > 0"),
            Scorer::Additive(x)
            | Scorer::NormalizedAdditive(x)
            | Scorer::Multiplicative(x)
            | Scorer::Shifted(x)
            | Scorer::ShellAdaptive(x) => {
                ensure!(x.is_finite(), "scorer parameters must be finite")
            }
            Scorer::Distance | Scorer::Lexicographic => {}
        }
        match self.terminator {
            Terminator::AdaptiveL { sample, scale } => {
                ensure!(
                    sample >= 10,
                    "adaptive_l needs a sample of at least 10 (upstream)"
                );
                ensure!(
                    scale >= 1.0,
                    "adaptive_l needs a scale of at least 1 (upstream)"
                );
            }
            Terminator::Budget(w) => ensure!(w > 0, "budget(W) needs W > 0"),
            Terminator::MatchAware => ensure!(
                self.collector == Collector::Heap,
                "match_aware termination requires the heap collector"
            ),
            Terminator::Exhaust => {}
        }
        if let Expander::TwoHop(Some(c)) = self.expander {
            ensure!(c >= 1, "two_hop(cap) needs cap >= 1");
        }
        let two_hop = matches!(self.expander, Expander::TwoHop(_));
        match self.laziness {
            Laziness::Deferred => {
                ensure!(
                    self.admission == Admission::All
                        && self.scorer == Scorer::Distance
                        && self.collector == Collector::FinalBeam,
                    "deferred evaluation requires admission all, the distance scorer and the final_beam collector"
                );
                ensure!(!two_hop, "deferred evaluation cannot expand two hops (nothing is rejected)");
            }
            Laziness::DistanceFirst => {
                ensure!(!two_hop, "distance_first is unsafe with two_hop (a skipped node may be a bridge)");
                ensure!(
                    self.collector != Collector::All,
                    "distance_first cannot skip nodes that the `all` collector keeps"
                );
            }
            Laziness::PredicateFirst => ensure!(
                !two_hop,
                "predicate_first is illegal with two_hop (the second hop ranks rejected nodes by distance)"
            ),
            Laziness::Eager => {}
        }
        if let PredicateMode::StaticBits(t) = self.predicate {
            ensure!(t >= 1, "static_bits(θ) needs θ >= 1");
        }
        let bitmap = self.predicate == PredicateMode::Bitmap;
        ensure!(
            bitmap == (self.materializer != Materializer::None),
            "the bitmap predicate mode and a materializer go together"
        );
        if bitmap {
            ensure!(
                !self.needs_g(),
                "a materialized match set gives no failed-clause count (scorer {} / admission {})",
                self.scorer,
                self.admission
            );
            ensure!(
                !matches!(self.admission, Admission::Projection(_)),
                "a clause projection cannot be evaluated on a materialized match set"
            );
        }
        if let Admission::Projection(c) = &self.admission {
            ensure!(!c.is_empty(), "projection needs at least one category");
        }
        Ok(())
    }

    /// The configuration as JSON (text forms of every role).
    pub fn to_json(&self) -> Value {
        json!({
            "starts": self.starts.iter().map(ToString::to_string).collect::<Vec<_>>().join("+"),
            "start_admission": self.start_admission.to_string(),
            "sample_start": self.sample_start.to_string(),
            "admission": self.admission.to_string(),
            "laziness": self.laziness.to_string(),
            "scorer": self.scorer.to_string(),
            "collector": collector_name(self.collector),
            "expander": self.expander.to_string(),
            "terminator": self.terminator.to_string(),
            "predicate": self.predicate.to_string(),
            "materializer": materializer_name(self.materializer),
            "exact": exact_name(self.exact),
            "weights": match self.weights { Weights::Unit => "unit", Weights::Surprisal => "surprisal" },
            "beam_width": self.beam_width,
            "padding": match self.padding {
                Padding::Upstream => "upstream",
                Padding::Cxx => "cxx",
                Padding::Exact => "exact",
            },
            "aux_g": self.aux_g,
        })
    }

    /// Whether efforts below `k` are not applicable (the collector returns at most `L`
    /// answers).
    pub fn needs_l_at_least_k(&self) -> bool {
        matches!(self.collector, Collector::FinalBeam | Collector::All)
    }
}

/// Reference presets of the baselines (`kernel-spec.md` §4).
#[derive(Debug, Clone, PartialEq)]
pub enum Preset {
    /// Unfiltered search, post-filter of the final beam.
    Postfilter,
    /// Upstream inline search: every node navigates, every match is collected.
    UpstreamInline,
    /// Upstream inline search with adaptive L.
    UpstreamInlineAdaptive { sample: usize, scale: f64 },
    /// Upstream multihop search.
    UpstreamMultihop,
    /// Upstream beta search (matches keyed `β·d`).
    UpstreamBeta(f32),
    /// Beta search with an exact posting scan for `min_c U_c ≤ T`.
    BetaBf { beta: f32, t: u64 },
    /// C++ multi-filter search (penalty `λ` per failed clause, cut-off `τ`, sample starts).
    Filt { lambda: f32, tau: u32 },
    /// C++ OR-filtered search on a market-label graph: navigation on the projected
    /// categories' clauses, post-filter of the final beam.
    Sfmarket(Vec<String>),
    /// C++ OR-filtered search on a product-label graph (starts from the product-label view).
    Cp,
}

impl Preset {
    /// Parses a preset name with optional arguments.
    pub fn parse(s: &str) -> Result<Self> {
        let (name, args) = call(s)?;
        let f = |i: usize, what: &str| -> Result<f32> { num(&args[i], what) };
        Ok(match name.as_str() {
            "postfilter" => {
                arity(&name, &args, 0)?;
                Self::Postfilter
            }
            "upstream-inline" | "inline" => {
                arity(&name, &args, 0)?;
                Self::UpstreamInline
            }
            "upstream-inline-adaptive" | "inline-adaptive" => match args.len() {
                0 => Self::UpstreamInlineAdaptive {
                    sample: 1000,
                    scale: 16.0,
                },
                2 => Self::UpstreamInlineAdaptive {
                    sample: num(&args[0], "sample")?,
                    scale: num(&args[1], "scale")?,
                },
                _ => bail!("{name} takes 0 or 2 arguments"),
            },
            "upstream-multihop" | "multihop" => {
                arity(&name, &args, 0)?;
                Self::UpstreamMultihop
            }
            "upstream-beta" | "inline-beta" | "beta" => match args.len() {
                0 => Self::UpstreamBeta(0.5),
                1 => Self::UpstreamBeta(f(0, "β")?),
                _ => bail!("{name} takes 0 or 1 argument"),
            },
            "bf" => {
                arity(&name, &args, 1)?;
                Self::BetaBf {
                    beta: 0.5,
                    t: num(&args[0], "T")?,
                }
            }
            "inline-beta-bf" => {
                arity(&name, &args, 2)?;
                Self::BetaBf {
                    beta: f(0, "β")?,
                    t: num(&args[1], "T")?,
                }
            }
            "filt" => match args.len() {
                0 => Self::Filt {
                    lambda: 10.0,
                    tau: 1,
                },
                2 => Self::Filt {
                    lambda: f(0, "λ")?,
                    tau: num(&args[1], "τ")?,
                },
                _ => bail!("filt takes 0 or 2 arguments (λ, τ)"),
            },
            "sfmarket" => {
                ensure!(
                    !args.is_empty(),
                    "sfmarket needs the categories of its navigation clause, e.g. sfmarket(<category>)"
                );
                Self::Sfmarket(args)
            }
            "cp" | "CP" => {
                arity(&name, &args, 0)?;
                Self::Cp
            }
            _ => bail!(
                "unknown preset {s:?} (postfilter, upstream-inline, upstream-inline-adaptive, \
                 upstream-multihop, upstream-beta(β), bf(T), inline-beta-bf(β, T), filt(λ, τ), \
                 sfmarket(categories), cp)"
            ),
        })
    }

    /// The preset's kernel configuration.
    pub fn kernel(&self) -> Kernel {
        let base = Kernel::default();
        let cxx = |starts: Vec<StartSource>, admission: Admission| Kernel {
            starts,
            start_admission: StartAdmission::Obey,
            admission,
            laziness: Laziness::PredicateFirst,
            padding: Padding::Cxx,
            ..Kernel::default()
        };
        match self {
            Self::Postfilter => Kernel {
                laziness: Laziness::Deferred,
                ..base
            },
            Self::UpstreamInline => Kernel {
                collector: Collector::All,
                ..base
            },
            Self::UpstreamInlineAdaptive { sample, scale } => Kernel {
                collector: Collector::All,
                terminator: Terminator::AdaptiveL {
                    sample: *sample,
                    scale: *scale,
                },
                ..base
            },
            Self::UpstreamMultihop => Kernel {
                admission: Admission::Strict,
                expander: Expander::TwoHop(None),
                ..base
            },
            Self::UpstreamBeta(beta) => Kernel {
                scorer: Scorer::Bonus(*beta),
                ..base
            },
            Self::BetaBf { beta, t } => Kernel {
                scorer: Scorer::Bonus(*beta),
                exact: ExactPath::ScanSmallestClause(*t),
                ..base
            },
            Self::Filt { lambda, tau } => Kernel {
                scorer: Scorer::Additive(*lambda),
                padding: Padding::Exact,
                ..cxx(
                    vec![StartSource::Sample(10), StartSource::Global],
                    Admission::Cutoff(*tau),
                )
            },
            Self::Sfmarket(categories) => cxx(
                vec![StartSource::Global, StartSource::LabelSmallest],
                Admission::Projection(categories.clone()),
            ),
            Self::Cp => cxx(
                vec![StartSource::Global, StartSource::LabelAll],
                Admission::Strict,
            ),
        }
    }

    /// Whether the preset takes its per-label starts from a label view.
    pub fn uses_view(&self) -> bool {
        matches!(self, Self::Sfmarket(_) | Self::Cp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_forms_roundtrip() {
        for s in ["global", "label(c*)", "label(all)", "sample(10)", "aux"] {
            assert_eq!(StartSource::parse(s).unwrap().to_string(), s);
        }
        assert_eq!(
            StartSource::parse("product").unwrap(),
            StartSource::LabelAll
        );
        assert_eq!(
            StartSource::parse_list("sample(10)+global").unwrap(),
            [StartSource::Sample(10), StartSource::Global]
        );
        for s in ["all", "strict", "projection(C3, C4)", "cutoff(1)"] {
            assert_eq!(Admission::parse(s).unwrap().to_string(), s);
        }
        for s in [
            "distance",
            "bonus(0.5)",
            "additive(10)",
            "normalized_additive(0.25)",
            "multiplicative(1)",
            "shifted(2)",
            "shell_adaptive(1.5)",
            "lexicographic",
        ] {
            assert_eq!(Scorer::parse(s).unwrap().to_string(), s);
        }
        for s in [
            "exhaust",
            "adaptive_l(1000, 16)",
            "budget(500)",
            "match_aware",
        ] {
            assert_eq!(Terminator::parse(s).unwrap().to_string(), s);
        }
        for s in ["mode1", "mode2", "bitmap", "static_bits(31250)"] {
            assert_eq!(PredicateMode::parse(s).unwrap().to_string(), s);
        }
        assert!(PredicateMode::parse("static_bits").is_err());
        assert!(Scorer::parse("bonus").is_err());
        assert!(Scorer::parse("bonus(x)").is_err());
        assert!(Terminator::parse("adaptive_l(5)").is_err());
        assert!(StartSource::parse("label(foo)").is_err());
    }

    #[test]
    fn presets_are_valid_and_overridable() {
        for p in [
            "postfilter",
            "upstream-inline",
            "upstream-inline-adaptive",
            "upstream-inline-adaptive(500, 8)",
            "upstream-multihop",
            "upstream-beta(0.5)",
            "inline-beta(0.3)",
            "bf(5000)",
            "inline-beta-bf(0.5, 5000)",
            "filt",
            "filt(10, 1)",
            "sfmarket(C3)",
            "cp",
        ] {
            let k = Preset::parse(p).unwrap().kernel();
            k.validate().unwrap_or_else(|e| panic!("{p}: {e}"));
        }
        let mut k = Preset::parse("upstream-inline").unwrap().kernel();
        k.set("starts", "label(c*)").unwrap();
        k.set("collector", "heap").unwrap();
        k.validate().unwrap();
        assert!(k.set("color", "red").is_err());
        assert!(Preset::parse("sfmarket").is_err());
    }

    #[test]
    fn illegal_combinations_are_rejected() {
        let bad = |edits: &[(&str, &str)]| {
            let mut k = Kernel::default();
            for (key, value) in edits {
                k.set(key, value).unwrap();
            }
            k.validate().is_err()
        };
        assert!(bad(&[("laziness", "deferred"), ("scorer", "bonus(0.5)")]));
        assert!(bad(&[("laziness", "deferred"), ("collector", "heap")]));
        assert!(bad(&[("laziness", "deferred"), ("admission", "strict")]));
        assert!(bad(&[
            ("laziness", "distance_first"),
            ("expander", "two_hop")
        ]));
        assert!(bad(&[("laziness", "distance_first"), ("collector", "all")]));
        assert!(bad(&[
            ("laziness", "predicate_first"),
            ("expander", "two_hop")
        ]));
        assert!(bad(&[("terminator", "match_aware")]));
        assert!(bad(&[("predicate", "bitmap")]));
        assert!(bad(&[("materializer", "dense")]));
        assert!(bad(&[
            ("predicate", "bitmap"),
            ("materializer", "dense"),
            ("scorer", "additive(10)")
        ]));
        assert!(bad(&[
            ("predicate", "bitmap"),
            ("materializer", "sparse"),
            ("admission", "cutoff(1)")
        ]));
        assert!(bad(&[
            ("predicate", "bitmap"),
            ("materializer", "dense"),
            ("admission", "projection(C3)")
        ]));
        assert!(bad(&[("terminator", "adaptive_l(5, 2)")]));
        assert!(bad(&[("beam_width", "0")]));
        assert!(bad(&[("predicate", "static_bits(0)")]));
        assert!(bad(&[
            ("predicate", "static_bits(100)"),
            ("materializer", "dense")
        ]));
        assert!(!bad(&[
            ("predicate", "static_bits(100)"),
            ("scorer", "multiplicative(1.2)")
        ]));
        assert!(bad(&[("scorer", "bonus(0)")]));
        assert!(!bad(&[
            ("predicate", "bitmap"),
            ("materializer", "dense"),
            ("scorer", "bonus(0.5)")
        ]));
        assert!(!bad(&[
            ("terminator", "match_aware"),
            ("collector", "heap")
        ]));
        assert!(!bad(&[
            ("laziness", "distance_first"),
            ("collector", "heap")
        ]));
    }
}
