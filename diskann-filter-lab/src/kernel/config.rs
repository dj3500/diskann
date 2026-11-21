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

pub use super::{
    collect::Collector,
    cost::{ExactPlan, PlanSet},
    eval::Weights,
    exact::ExactPath,
    materialize::Materializer,
};

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
    /// None of its own: only the searcher's extra starts (the pair-tree route's roots).
    Extra,
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
    /// Nodes satisfying the query's clause with the smallest `U_c` (`projection(c*)`, the
    /// binding clause, chosen per query from the posting sizes; only that clause is evaluated
    /// during the search).
    ProjectionSmallest,
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
    /// Distance first; the predicate is skipped whenever the node cannot be an answer (the
    /// `heap(K)` collector is full and the distance exceeds its K-th), but the node still
    /// navigates (distance keys, admission `all`).
    HeapBound,
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
    /// Only auxiliary nodes (routers) are expanded; data points are evaluated and collected
    /// but never expanded (a search over router trees: their leaves' members are scanned).
    AuxOnly,
    /// As `aux_only`, and data points never enter the list either: the list is a frontier of
    /// routers only (its size bounds the routers kept; members go to the collector).
    AuxFrontier,
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
    /// Posting-based adaptive L: the list size `L₀ × m`, with upstream's multiplier `m` of the
    /// estimated match rate (`|M̂| / n` from the `estimates` role), clamped to `scale`, fixed
    /// before navigation; then `exhaust`.
    PostingL { scale: f64 },
    /// Calibrated effort: the effort is a target recall `R` in per mille, and the list size is
    /// the graph's calibrated unfiltered effort for recall `R` at `K / r̂` neighbours (`r̂ =
    /// |M̂| / n`: a filtered search at match rate `r` as an unfiltered search for `K / r`
    /// neighbours), fixed before navigation; then `exhaust`. Needs the effort curve of the
    /// unit costs and a cost crossover. `effort_curve(cap)`: at most `cap` × the curve's list
    /// size for an unselective query (`r̂ = 1`); `effort_curve`: no cap.
    EffortCurve { cap: f64 },
}

/// How a query chooses between the graph search and the exact path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Crossover {
    /// The exact path when `min_c U_c ≤ T` of the `exact` role (fixed threshold).
    Threshold,
    /// The exact path when its estimated cost is at most the nominal graph cost.
    Cost,
    /// `cost`, then a bounded graph trial with online fallback to the exact path (needs the
    /// `heap` collector).
    Guarded,
    /// Graph only (track G): the cost model's estimates set the list size (`posting_l`,
    /// `effort_curve`) and choose the post-filter route, but the exact path is never taken
    /// (the hopeless gate does not apply).
    Graph,
}

/// Source of the per-query estimates of the cost model and of `posting_l`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EstimateSource {
    /// Posting bounds only (`A = min(n, P*)`, clause survival `min(1, U_j/n)` multiplied).
    Bounds,
    /// A probe of the bottom-`b` sketches (a uniform sample of `union(c*)`).
    Sketch,
}

/// What the planned exact path and the probe check on their points, all members of
/// `union(c*)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExactChecks {
    /// The residual clauses (clause 0 holds), with prefetches for them.
    Residual,
    /// Every clause, with the graph search's prefetch policy (batch B6).
    Full,
}

/// Whether graph-routed queries of the cost crossover may take a post-filter search instead
/// of the configured graph search: the graph's global starts, no predicate checks during the
/// traversal, and the final beam checked in distance order until `K` matches are found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PostfilterRoute {
    /// Never.
    Off,
    /// When its nominal cost is at most that of the configured graph search, at a list size
    /// that holds the `K` nearest matches with high probability at the estimated match rate.
    Cost,
    /// Always (a control).
    Always,
}

/// Start nodes of the post-filter route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PostfilterStarts {
    /// The graph's global starts.
    Global,
    /// The configured `starts` of the kernel (e.g. per-label starts of `c*`).
    Configured,
    /// The configured starts for a narrow predicate (estimated match rate below
    /// [`BROAD_RATE`](super::cost::BROAD_RATE), where the planner also enlarges the list), the
    /// global starts for a broad one: its matches are everywhere, so per-label starts carry no
    /// information about where they are, and the route is an unfiltered search, from the entry
    /// point the graph was built to be navigable from.
    ByRate,
    /// `by_rate`, and a broad predicate's search, once it ends, offers the configured starts to
    /// its list: a start closer to the query than the list's last entry (the search from the
    /// global starts missed the query's neighbourhood) is admitted and the search continues
    /// from it. A start the list does not admit costs one distance computation.
    ByRateLate,
    /// `by_rate`, and a broad predicate's search from the global starts is followed by a second
    /// search from the configured starts with the same list size, sharing the visited nodes (it
    /// stops where the first search has been); the answers are the `K` nearest matches of both.
    /// Never fewer or farther answers than `by_rate`; the second search costs little when both
    /// reach the query's neighbourhood.
    ByRateBoth,
    /// `by_rate_both` with the second search's list size `K` (at most the first one's): the
    /// smallest search that can return `K` answers of its own, a hedge against a first search
    /// held in the wrong region whose cost does not grow with the first search's list.
    ByRateBothK,
}

impl PostfilterStarts {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim() {
            "global" => Self::Global,
            "configured" => Self::Configured,
            "by_rate" => Self::ByRate,
            "by_rate_late" => Self::ByRateLate,
            "by_rate_both" => Self::ByRateBoth,
            "by_rate_both_k" => Self::ByRateBothK,
            _ => bail!(
                "unknown postfilter_starts {s:?} (global, configured, by_rate, by_rate_late, \
                 by_rate_both, by_rate_both_k)"
            ),
        })
    }

    /// Whether the route starts from the configured starts at estimated match rate `rate`.
    pub fn configured_at(self, rate: f64) -> bool {
        match self {
            Self::Global => false,
            Self::Configured => true,
            Self::ByRate | Self::ByRateLate | Self::ByRateBoth | Self::ByRateBothK => {
                rate < super::cost::BROAD_RATE
            }
        }
    }

    /// Whether a search from the global starts at estimated match rate `rate` offers the
    /// configured starts once it ends.
    pub fn late_at(self, rate: f64) -> bool {
        self == Self::ByRateLate && !self.configured_at(rate)
    }

    /// Whether a search from the global starts at estimated match rate `rate` is followed by a
    /// second search from the configured starts.
    pub fn both_at(self, rate: f64) -> bool {
        matches!(self, Self::ByRateBoth | Self::ByRateBothK) && !self.configured_at(rate)
    }

    /// The list size of the second search (`by_rate_both*`) after a first one with list size
    /// `l`, for `K = k` answers.
    pub fn second_list(self, l: usize, k: usize) -> usize {
        match self {
            Self::ByRateBothK => k.clamp(1, l.max(1)),
            _ => l,
        }
    }
}

impl fmt::Display for PostfilterStarts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Global => "global",
            Self::Configured => "configured",
            Self::ByRate => "by_rate",
            Self::ByRateLate => "by_rate_late",
            Self::ByRateBoth => "by_rate_both",
            Self::ByRateBothK => "by_rate_both_k",
        })
    }
}

impl PostfilterRoute {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim() {
            "off" => Self::Off,
            "cost" => Self::Cost,
            "always" => Self::Always,
            _ => bail!("unknown postfilter route {s:?} (off, cost, always)"),
        })
    }
}

impl fmt::Display for PostfilterRoute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Off => "off",
            Self::Cost => "cost",
            Self::Always => "always",
        })
    }
}

impl ExactChecks {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim() {
            "residual" => Self::Residual,
            "full" => Self::Full,
            _ => bail!("unknown exact_checks {s:?} (residual, full)"),
        })
    }
}

impl fmt::Display for ExactChecks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Residual => "residual",
            Self::Full => "full",
        })
    }
}

/// Fallback signals of the `guarded` crossover.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fallbacks {
    /// While the heap is incomplete: forecast of the remaining graph cost from the arrival
    /// rate of answers, against the exact cost.
    pub forecast: bool,
    /// Graph spend reaching the estimated exact cost.
    pub budget: bool,
    /// At nominal termination: a sampled match closer than the graph's K-th answer.
    pub witness: bool,
}

impl Fallbacks {
    pub const ALL: Self = Self {
        forecast: true,
        budget: true,
        witness: true,
    };

    /// Parses `forecast+budget+witness` (any subset) or `none`.
    pub fn parse(s: &str) -> Result<Self> {
        let mut f = Self {
            forecast: false,
            budget: false,
            witness: false,
        };
        if s.trim() == "none" {
            return Ok(f);
        }
        for p in s.split('+') {
            match p.trim() {
                "forecast" => f.forecast = true,
                "budget" => f.budget = true,
                "witness" => f.witness = true,
                _ => bail!("unknown fallback signal {p:?} (forecast, budget, witness, none)"),
            }
        }
        Ok(f)
    }
}

impl fmt::Display for Fallbacks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut v = Vec::new();
        if self.forecast {
            v.push("forecast");
        }
        if self.budget {
            v.push("budget");
        }
        if self.witness {
            v.push("witness");
        }
        if v.is_empty() {
            f.write_str("none")
        } else {
            f.write_str(&v.join("+"))
        }
    }
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
    pub crossover: Crossover,
    pub estimates: EstimateSource,
    /// Plans the exact path v2 may choose from.
    pub exact_plans: PlanSet,
    pub fallback: Fallbacks,
    pub exact_checks: ExactChecks,
    /// Per-phase wall-clock timers of a query (probe, exact path phases, graph search) into
    /// its counters: a diagnostic, off by default (a few clock reads per query).
    pub phase_times: bool,
    pub postfilter: PostfilterRoute,
    pub postfilter_starts: PostfilterStarts,
    /// The cost crossovers route to the exact path when the planned graph search is hopeless
    /// (fewer than one match met at the upper confidence bound of the match count; A1c);
    /// `off` for the ablation. Inert under the graph crossover.
    pub hopeless_gate: bool,
    /// How the cost model counts the row reads of member checks.
    pub row_reads: RowReads,
    /// Prefetch the adjacency row of the next node to expand while the current node's
    /// neighbours are evaluated (A7; off reproduces the earlier loops exactly). Changes no
    /// answer, only memory timing.
    pub row_prefetch: bool,
    /// The post-filter loop's beam packs each entry into one `u64` (`PackedBeam`, A7): the same
    /// beam semantics, one array to search and move per insertion. Changes no answer.
    pub packed_beam: bool,
    /// The exact plan `df` counts its predicate evaluations at the lower confidence bound of the
    /// match count (A7b): its answer bound saves checks only once `K` matches are found, so a
    /// match count over-estimated past `K` must not make it look cheap. Off: the estimate.
    pub df_lower: bool,
    /// A cost crossover plans an inline graph search (`posting_l`) at a list size whose
    /// evaluated nodes reach the rank of the `K`-th match at the estimated match rate, as the
    /// post-filter route does (A7b); off: `L₀ × m(M̂ / n)` only.
    pub inline_rank: bool,
    /// The exact plan `df` checks a candidate inside the answer bound one block of candidates
    /// after its distance, with its row offsets and row prefetched meanwhile (A7b), as `scan`
    /// prefetches its checks; off: each check reads its row at once. The same checks and
    /// answers, only memory timing.
    pub df_pipeline: bool,
    /// What the `threshold` crossover compares with `T` of `exact`: `min_c U_c` (off, the
    /// default: the exact path when `min_c U_c ≤ T`) or, on, the sketch estimate `M̂` of the
    /// match count (the graph route when `M̂ ≥ T`, else the exact path; an ablation that
    /// separates better match-count estimates from pricing by the cost model).
    pub threshold_estimate: bool,
    /// `static_bitsets = "off"`: the exact path, the probe and the post-filter route check
    /// clauses on the label rows, never with the static bitsets (the plans keep their forms
    /// without them: `bitmap` scatters postings, `rowand` gallops); an ablation of the stored
    /// dense rows.
    pub static_bitsets_off: bool,
    /// The tree route for tree-eligible queries (`off` by default).
    pub tree_route: TreeRoute,
    /// The tree route's list size (`list` by default).
    pub tree_effort: TreeEffort,
    /// When a tree-eligible query takes the tree route (`always` by default).
    pub tree_choice: TreeChoice,
    /// `cstar_members = "on"`: every node the search evaluates is a member of `union(c*)` (a
    /// router-tree search from c\*'s starts: `starts = "label(c*)"`, forced starts, only
    /// routers expanded), so checks cover the other clauses only (off by default).
    pub cstar_members: bool,
    /// The pair-tree route for queries whose clause pair is exactly covered by pair trees
    /// (`flab steiner --construction pairs`; the run's `pairs_sidecar` and `pair_roots`) with
    /// fewer members than c\* (off by default).
    pub tree_pairs: TreePairs,
}

/// How the cost model counts a member check's row reads (the static evaluator reads a
/// member's row only when a clause with labels without a bitset is not already satisfied by
/// its bitset labels).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RowReads {
    /// As the evaluator: when such a clause's bitset labels miss.
    Clause,
    /// Whenever the check reaches such a clause (the A1b model; an ablation).
    Reach,
}

impl RowReads {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim() {
            "clause" => Self::Clause,
            "reach" => Self::Reach,
            v => bail!("unknown row_reads {v:?} (clause, reach)"),
        })
    }
}

impl fmt::Display for RowReads {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Clause => "clause",
            Self::Reach => "reach",
        })
    }
}

/// The tree route: a query whose binding clause c\* has only labels whose starts are all
/// auxiliary nodes (router-tree roots, `flab steiner`) — or no points — is *tree-eligible*
/// and runs a tree search instead of the configured search: starts `label(c*)`, only
/// auxiliary nodes expanded (`aux_only`), strict admission with predicate-first checks, a
/// heap of the K best matches, the same list size. Other queries take the configured route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TreeRoute {
    Off,
    /// Member checks on the label rows.
    RowScan,
    /// Member checks on a per-query dense match bitmap built with the static bitsets.
    Bitmap,
    /// Member checks of the clauses other than c\* (which every member of c\*'s trees
    /// satisfies; `cstar_members`) with the static bitsets, the row only for labels without
    /// a bitset.
    StaticBits,
    /// Per query, the member check (`row_scan`, `static_bits` or `bitmap`) with the lowest
    /// model cost from the calibrated unit costs (see `Plan::tree_checks`).
    Cost,
}

impl TreeRoute {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim() {
            "off" => Self::Off,
            "row_scan" => Self::RowScan,
            "bitmap" => Self::Bitmap,
            "static_bits" => Self::StaticBits,
            "cost" => Self::Cost,
            v => bail!("unknown tree_route {v:?} (off, row_scan, bitmap, static_bits, cost)"),
        })
    }
}

impl fmt::Display for TreeRoute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Off => "off",
            Self::RowScan => "row_scan",
            Self::Bitmap => "bitmap",
            Self::StaticBits => "static_bits",
            Self::Cost => "cost",
        })
    }
}

/// The list size of the tree route's search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TreeEffort {
    /// The configuration's list size (the tree search's list holds routers and matches).
    List,
    /// The calibrated tree effort curve (`tree_*` in the unit costs) at the configuration's
    /// target recall (`effort_curve` efforts) for `K / ρ̂` members, `ρ̂` = the estimated
    /// share of the binding clause's members that match (`M̂ / A`); the list holds routers
    /// only.
    Curve,
    /// As `curve`, at most the router's `effort_curve(cap)` times the curve's list size at `K`
    /// members (the graph route's cap on selective queries, applied to the tree route).
    Capped,
    /// One router frontier `F` for every query and effort (`fixed(F)`, `F ≥ 1`): the list holds
    /// routers only. A configuration's grid of efforts then pairs each graph effort with the
    /// same tree frontier; needs no tree effort curve.
    Fixed(u32),
}

impl TreeEffort {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim() {
            "list" => Self::List,
            "curve" => Self::Curve,
            "capped" => Self::Capped,
            v => match v.strip_prefix("fixed(").and_then(|r| r.strip_suffix(')')) {
                Some(f) => {
                    let f: u32 = f
                        .trim()
                        .parse()
                        .with_context(|| format!("tree_effort {v:?}: F must be an integer"))?;
                    ensure!(f >= 1, "tree_effort {v:?}: F must be at least 1");
                    Self::Fixed(f)
                }
                None => bail!("unknown tree_effort {v:?} (list, curve, capped, fixed(F))"),
            },
        })
    }
}

impl fmt::Display for TreeEffort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::List => f.write_str("list"),
            Self::Curve => f.write_str("curve"),
            Self::Capped => f.write_str("capped"),
            Self::Fixed(n) => write!(f, "fixed({n})"),
        }
    }
}

/// The pair-tree route (`tree_pairs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TreePairs {
    /// No pair route (`off`).
    Off,
    /// A covered query searches the pair trees when their model cost is below the route it
    /// takes otherwise — its label tree when it is tree-eligible, else the crossover's route —
    /// each tree priced at the query's dilution (`on`): the frontier scaled to the members per
    /// estimated match, at most the whole tree, its router expansions' member checks and
    /// distances; the pair route then searches that frontier.
    Cost,
    /// Every covered query searches the pair trees at the tree route's list size (`always`:
    /// the route before the cost check, kept to reproduce it).
    Always,
}

impl TreePairs {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim() {
            "off" => Self::Off,
            "on" => Self::Cost,
            "always" => Self::Always,
            v => bail!("unknown tree_pairs {v:?} (on, always, off)"),
        })
    }
}

/// When a tree-eligible query takes the tree route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TreeChoice {
    /// Every tree-eligible query.
    Always,
    /// When the tree search's model cost (its router frontier's expansions, their member
    /// checks and the distances of the members expected to match) is below the planned
    /// route's (the crossover's estimate for the route it chose).
    Cost,
}

impl TreeChoice {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim() {
            "always" => Self::Always,
            "cost" => Self::Cost,
            v => bail!("unknown tree_choice {v:?} (always, cost)"),
        })
    }
}

impl fmt::Display for TreeChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Always => "always",
            Self::Cost => "cost",
        })
    }
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
            crossover: Crossover::Threshold,
            estimates: EstimateSource::Bounds,
            exact_plans: PlanSet::ALL,
            fallback: Fallbacks::ALL,
            exact_checks: ExactChecks::Residual,
            phase_times: false,
            postfilter: PostfilterRoute::Off,
            postfilter_starts: PostfilterStarts::Global,
            hopeless_gate: true,
            row_reads: RowReads::Clause,
            row_prefetch: false,
            packed_beam: false,
            df_lower: false,
            inline_rank: false,
            df_pipeline: false,
            threshold_estimate: false,
            static_bitsets_off: false,
            tree_route: TreeRoute::Off,
            tree_effort: TreeEffort::List,
            tree_choice: TreeChoice::Always,
            cstar_members: false,
            tree_pairs: TreePairs::Off,
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
            ("extra", []) => Self::Extra,
            _ => bail!("unknown start source {s:?} (global, label(c*), label(all), product, sample(k), aux, extra)"),
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
            Self::Extra => write!(f, "extra"),
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
            "projection" if args.len() == 1 && args[0] == "c*" => Self::ProjectionSmallest,
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
            Self::ProjectionSmallest => write!(f, "projection(c*)"),
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
            "heap_bound" => Self::HeapBound,
            _ => bail!(
                "unknown laziness {s:?} (eager, deferred, predicate_first, distance_first, heap_bound)"
            ),
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
            Self::HeapBound => "heap_bound",
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
            ("aux_only", []) => Self::AuxOnly,
            ("aux_frontier", []) => Self::AuxFrontier,
            _ => bail!(
                "unknown expander {s:?} (one_hop, two_hop, two_hop(cap), aux_only, aux_frontier)"
            ),
        })
    }
}

impl fmt::Display for Expander {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OneHop => write!(f, "one_hop"),
            Self::TwoHop(None) => write!(f, "two_hop"),
            Self::TwoHop(Some(c)) => write!(f, "two_hop({c})"),
            Self::AuxOnly => write!(f, "aux_only"),
            Self::AuxFrontier => write!(f, "aux_frontier"),
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
            ("posting_l", []) => Self::PostingL { scale: 16.0 },
            ("posting_l", [m]) => Self::PostingL {
                scale: num(m, "posting-L scale")?,
            },
            ("effort_curve", []) => Self::EffortCurve { cap: f64::INFINITY },
            ("effort_curve", [c]) => Self::EffortCurve {
                cap: num(c, "effort-curve cap")?,
            },
            _ => bail!(
                "unknown terminator {s:?} (exhaust, adaptive_l(sample, scale), budget(W), match_aware, posting_l(scale), effort_curve(cap))"
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
            Self::PostingL { scale } => write!(f, "posting_l({scale})"),
            Self::EffortCurve { cap } if cap.is_infinite() => write!(f, "effort_curve"),
            Self::EffortCurve { cap } => write!(f, "effort_curve({cap})"),
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
        "dense_static" => Materializer::DenseStatic,
        "dense_nav" => Materializer::DenseNav,
        _ => bail!("unknown materializer {s:?} (none, dense, sparse, dense_static, dense_nav)"),
    })
}

fn materializer_name(m: Materializer) -> &'static str {
    match m {
        Materializer::None => "none",
        Materializer::Dense => "dense",
        Materializer::Sparse => "sparse",
        Materializer::DenseStatic => "dense_static",
        Materializer::DenseNav => "dense_nav",
    }
}

fn parse_exact(s: &str) -> Result<ExactPath> {
    let (name, args) = call(s)?;
    Ok(match (name.as_str(), args.as_slice()) {
        ("none", []) => ExactPath::None,
        ("scan_smallest_clause", [t]) => ExactPath::ScanSmallestClause(num(t, "T_exact")?),
        ("intersect_then_scan", [t]) => ExactPath::IntersectThenScan(num(t, "T_exact")?),
        ("planned", []) => ExactPath::Planned(0),
        ("planned", [t]) => ExactPath::Planned(num(t, "T_exact")?),
        _ => bail!(
            "unknown exact path {s:?} (none, scan_smallest_clause(T), intersect_then_scan(T), planned(T))"
        ),
    })
}

fn exact_name(e: ExactPath) -> String {
    match e {
        ExactPath::None => "none".into(),
        ExactPath::ScanSmallestClause(t) => format!("scan_smallest_clause({t})"),
        ExactPath::IntersectThenScan(t) => format!("intersect_then_scan({t})"),
        ExactPath::Planned(t) => format!("planned({t})"),
    }
}

fn parse_weights(s: &str) -> Result<Weights> {
    Ok(match s.trim() {
        "unit" => Weights::Unit,
        "surprisal" => Weights::Surprisal,
        _ => bail!("unknown weights {s:?} (unit, surprisal)"),
    })
}

impl Crossover {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim() {
            "threshold" => Self::Threshold,
            "cost" => Self::Cost,
            "guarded" => Self::Guarded,
            "graph" => Self::Graph,
            _ => bail!("unknown crossover {s:?} (threshold, cost, guarded, graph)"),
        })
    }
}

impl fmt::Display for Crossover {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Threshold => "threshold",
            Self::Cost => "cost",
            Self::Guarded => "guarded",
            Self::Graph => "graph",
        })
    }
}

impl EstimateSource {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim() {
            "bounds" => Self::Bounds,
            "sketch" => Self::Sketch,
            _ => bail!("unknown estimates {s:?} (bounds, sketch)"),
        })
    }
}

impl fmt::Display for EstimateSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Bounds => "bounds",
            Self::Sketch => "sketch",
        })
    }
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
            "crossover" => self.crossover = Crossover::parse(value)?,
            "estimates" => self.estimates = EstimateSource::parse(value)?,
            "exact_plans" => self.exact_plans = PlanSet::parse(value)?,
            "fallback" => self.fallback = Fallbacks::parse(value)?,
            "exact_checks" => self.exact_checks = ExactChecks::parse(value)?,
            "phase_times" => {
                self.phase_times = match value.trim() {
                    "on" => true,
                    "off" => false,
                    v => bail!("unknown phase_times {v:?} (on, off)"),
                }
            }
            "postfilter" => self.postfilter = PostfilterRoute::parse(value)?,
            "postfilter_starts" => self.postfilter_starts = PostfilterStarts::parse(value)?,
            "hopeless_gate" => {
                self.hopeless_gate = match value.trim() {
                    "on" => true,
                    "off" => false,
                    v => bail!("unknown hopeless_gate {v:?} (on, off)"),
                }
            }
            "row_reads" => self.row_reads = RowReads::parse(value)?,
            "row_prefetch" => {
                self.row_prefetch = match value.trim() {
                    "on" => true,
                    "off" => false,
                    v => bail!("unknown row_prefetch {v:?} (on, off)"),
                }
            }
            "packed_beam" => {
                self.packed_beam = match value.trim() {
                    "on" => true,
                    "off" => false,
                    v => bail!("unknown packed_beam {v:?} (on, off)"),
                }
            }
            "static_bitsets" => {
                self.static_bitsets_off = match value.trim() {
                    "on" => false,
                    "off" => true,
                    v => bail!("unknown static_bitsets {v:?} (on, off)"),
                }
            }
            "threshold_on" => {
                self.threshold_estimate = match value.trim() {
                    "bound" => false,
                    "estimate" => true,
                    v => bail!("unknown threshold_on {v:?} (bound, estimate)"),
                }
            }
            "df_evals" => {
                self.df_lower = match value.trim() {
                    "lower" => true,
                    "estimate" => false,
                    v => bail!("unknown df_evals {v:?} (estimate, lower)"),
                }
            }
            "inline_rank" => {
                self.inline_rank = match value.trim() {
                    "on" => true,
                    "off" => false,
                    v => bail!("unknown inline_rank {v:?} (on, off)"),
                }
            }
            "df_pipeline" => {
                self.df_pipeline = match value.trim() {
                    "on" => true,
                    "off" => false,
                    v => bail!("unknown df_pipeline {v:?} (on, off)"),
                }
            }
            "tree_route" => self.tree_route = TreeRoute::parse(value)?,
            "tree_effort" => self.tree_effort = TreeEffort::parse(value)?,
            "tree_choice" => self.tree_choice = TreeChoice::parse(value)?,
            "cstar_members" => {
                self.cstar_members = match value.trim() {
                    "on" => true,
                    "off" => false,
                    v => bail!("unknown cstar_members {v:?} (on, off)"),
                }
            }
            "tree_pairs" => self.tree_pairs = TreePairs::parse(value)?,
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
            Terminator::PostingL { scale } => {
                ensure!(scale >= 1.0, "posting_l needs a scale of at least 1");
            }
            Terminator::EffortCurve { cap } => {
                ensure!(
                    self.crossover != Crossover::Threshold,
                    "effort_curve needs a cost crossover (the estimates and the calibrated curve)"
                );
                ensure!(cap >= 1.0, "effort_curve(cap) needs cap >= 1");
            }
            Terminator::Exhaust => {}
        }
        if let Expander::TwoHop(Some(c)) = self.expander {
            ensure!(c >= 1, "two_hop(cap) needs cap >= 1");
        }
        let aux_expander = matches!(self.expander, Expander::AuxOnly | Expander::AuxFrontier);
        ensure!(
            !(aux_expander && self.scorer.needs_pilot()),
            "aux_only / aux_frontier cannot use a pilot scorer (the pilot expands the global start)"
        );
        ensure!(
            self.expander != Expander::AuxFrontier || self.collector == Collector::Heap,
            "aux_frontier collects matches in the heap collector only (they never enter the list)"
        );
        match self.tree_effort {
            TreeEffort::List => {}
            TreeEffort::Fixed(_) => ensure!(
                self.tree_route != TreeRoute::Off,
                "tree_effort = fixed(F) needs the tree route"
            ),
            TreeEffort::Curve | TreeEffort::Capped => ensure!(
                self.tree_route != TreeRoute::Off
                    && matches!(self.terminator, Terminator::EffortCurve { .. }),
                "tree_effort = curve / capped needs the tree route and effort_curve efforts (a target recall)"
            ),
        }
        ensure!(
            self.tree_choice == TreeChoice::Always
                || (self.tree_route != TreeRoute::Off
                    && self.tree_effort != TreeEffort::List
                    && matches!(self.crossover, Crossover::Cost | Crossover::Graph)),
            "tree_choice = cost needs the tree route, a tree effort (curve, capped or fixed) and a cost crossover (cost or graph)"
        );
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
            Laziness::HeapBound => ensure!(
                !two_hop
                    && self.admission == Admission::All
                    && self.scorer == Scorer::Distance
                    && self.collector == Collector::Heap,
                "heap_bound requires admission all, the distance scorer, the heap collector and one_hop (keys must not depend on the predicate)"
            ),
            Laziness::Eager => {}
        }
        ensure!(!self.exact_plans.is_empty(), "exact_plans is empty");
        if self.crossover != Crossover::Threshold {
            ensure!(
                self.exact.is_planned(),
                "crossover {} needs the planned exact path (exact = \"planned\")",
                self.crossover
            );
        }
        if self.threshold_estimate {
            ensure!(
                self.crossover == Crossover::Threshold
                    && self.estimates == EstimateSource::Sketch
                    && self.exact.threshold().is_some(),
                "threshold_on = \"estimate\" needs the threshold crossover, sketch estimates and an exact path with a threshold"
            );
        }
        if self.crossover == Crossover::Guarded {
            ensure!(
                self.collector == Collector::Heap,
                "the guarded crossover needs the heap collector (its answers seed the fallback)"
            );
        }
        if self.postfilter != PostfilterRoute::Off && self.crossover == Crossover::Threshold {
            // The graph side of the threshold router is the post-filter route at the fixed list.
            ensure!(
                self.postfilter == PostfilterRoute::Always
                    && (self.threshold_estimate
                        || matches!(
                            self.postfilter_starts,
                            PostfilterStarts::Global | PostfilterStarts::Configured
                        )),
                "under the threshold crossover the post-filter route is `always`, and its rate-dependent starts need threshold_on = \"estimate\""
            );
        } else if self.postfilter != PostfilterRoute::Off {
            ensure!(
                matches!(
                    self.crossover,
                    Crossover::Cost | Crossover::Guarded | Crossover::Graph
                ),
                "the post-filter route needs a cost crossover (its estimates and costs)"
            );
        }
        if let PredicateMode::StaticBits(t) = self.predicate {
            ensure!(t >= 1, "static_bits(θ) needs θ >= 1");
        }
        ensure!(
            !self.static_bitsets_off
                || !(matches!(self.predicate, PredicateMode::StaticBits(_))
                    || self.materializer == Materializer::DenseStatic),
            "static_bitsets = \"off\" excludes the static_bits predicate and the dense_static materializer"
        );
        ensure!(
            !(self.static_bitsets_off
                && matches!(
                    self.tree_route,
                    TreeRoute::Bitmap | TreeRoute::StaticBits | TreeRoute::Cost
                )),
            "static_bitsets = \"off\" excludes tree_route = \"bitmap\", \"static_bits\" and \"cost\" (their member checks use the static bitsets)"
        );
        ensure!(
            self.tree_pairs == TreePairs::Off
                || (self.tree_route != TreeRoute::Off && self.tree_choice == TreeChoice::Always),
            "tree_pairs needs the tree route with tree_choice = always"
        );
        ensure!(
            self.tree_pairs != TreePairs::Cost
                || (self.exact.is_planned() && self.crossover != Crossover::Threshold),
            "tree_pairs = \"on\" prices its route after the crossover: a planned exact path (its unit costs) and crossover cost or graph"
        );
        ensure!(
            self.tree_route != TreeRoute::Cost
                || (self.tree_choice == TreeChoice::Always && self.exact.is_planned()),
            "tree_route = \"cost\" needs tree_choice = always and a planned exact path (its unit costs)"
        );
        ensure!(
            !self.cstar_members
                || (self.starts == [StartSource::LabelSmallest]
                    && self.start_admission == StartAdmission::Force
                    && self.admission == Admission::Strict
                    && matches!(self.expander, Expander::AuxOnly | Expander::AuxFrontier)
                    && matches!(
                        self.predicate,
                        PredicateMode::RowScan | PredicateMode::StaticBits(_)
                    )),
            "cstar_members = \"on\" needs a router-tree search from c*: starts = \"label(c*)\", start_admission = \"force\", admission = \"strict\", expander aux_only or aux_frontier, predicate row_scan or static_bits(θ)"
        );
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
                self.materializer == Materializer::DenseNav
                    || !matches!(
                        self.admission,
                        Admission::Projection(_) | Admission::ProjectionSmallest
                    ),
                "a clause projection cannot be evaluated on a materialized match set (dense_nav \
                 materializes the projected clauses)"
            );
        }
        if let Admission::Projection(c) = &self.admission {
            ensure!(!c.is_empty(), "projection needs at least one category");
        }
        Ok(())
    }

    /// The configuration as JSON (text forms of every role).
    pub fn to_json(&self) -> Value {
        let mut j = json!({
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
            "crossover": self.crossover.to_string(),
            "estimates": self.estimates.to_string(),
            "exact_plans": self.exact_plans.to_string(),
            "fallback": self.fallback.to_string(),
            "exact_checks": self.exact_checks.to_string(),
        });
        // Only when on: the hashes of configurations without timers do not change.
        if self.phase_times {
            j["phase_times"] = json!("on");
        }
        // Only when set, so that the JSON (and hashes) of other configurations do not change.
        if self.postfilter != PostfilterRoute::Off {
            j["postfilter"] = json!(self.postfilter.to_string());
        }
        if self.postfilter_starts != PostfilterStarts::Global {
            j["postfilter_starts"] = json!(self.postfilter_starts.to_string());
        }
        if !self.hopeless_gate {
            j["hopeless_gate"] = json!("off");
        }
        if self.row_prefetch {
            j["row_prefetch"] = json!("on");
        }
        if self.packed_beam {
            j["packed_beam"] = json!("on");
        }
        if self.df_lower {
            j["df_evals"] = json!("lower");
        }
        if self.inline_rank {
            j["inline_rank"] = json!("on");
        }
        if self.df_pipeline {
            j["df_pipeline"] = json!("on");
        }
        if self.row_reads != RowReads::Clause {
            j["row_reads"] = json!(self.row_reads.to_string());
        }
        if self.threshold_estimate {
            j["threshold_on"] = json!("estimate");
        }
        if self.static_bitsets_off {
            j["static_bitsets"] = json!("off");
        }
        if self.tree_route != TreeRoute::Off {
            j["tree_route"] = json!(self.tree_route.to_string());
        }
        if self.tree_effort != TreeEffort::List {
            j["tree_effort"] = json!(self.tree_effort.to_string());
        }
        if self.tree_choice != TreeChoice::Always {
            j["tree_choice"] = json!(self.tree_choice.to_string());
        }
        if self.cstar_members {
            j["cstar_members"] = json!("on");
        }
        match self.tree_pairs {
            TreePairs::Off => {}
            TreePairs::Cost => j["tree_pairs"] = json!("on"),
            TreePairs::Always => j["tree_pairs"] = json!("always"),
        }
        j
    }

    /// The kernels of the tree route's member checks under `tree_route = "cost"`, in the
    /// order of `Plan::tree_checks`: row scan, static bitsets skipping c\*, per-query bitmap.
    pub fn tree_check_kernels(&self) -> Vec<Kernel> {
        if self.tree_route != TreeRoute::Cost {
            return Vec::new();
        }
        [TreeRoute::RowScan, TreeRoute::StaticBits, TreeRoute::Bitmap]
            .into_iter()
            .filter_map(|r| {
                Kernel {
                    tree_route: r,
                    ..self.clone()
                }
                .tree_kernel()
            })
            .collect()
    }

    /// The kernels of the pair-tree route (`tree_pairs`): the tree route's kernel (or, under
    /// `tree_route = "cost"`, each of `tree_check_kernels`) started only at the searcher's
    /// extra starts (the covering pairs' roots); their checks cover the clauses outside the
    /// covered pair (set per query), not via `cstar_members`.
    pub fn pair_kernels(&self) -> Vec<Kernel> {
        if self.tree_pairs == TreePairs::Off {
            return Vec::new();
        }
        let ks = if self.tree_route == TreeRoute::Cost {
            self.tree_check_kernels()
        } else {
            self.tree_kernel().into_iter().collect()
        };
        ks.into_iter()
            .map(|k| Kernel {
                starts: vec![StartSource::Extra],
                cstar_members: false,
                ..k
            })
            .collect()
    }

    /// The kernel of the tree route's search (`None` when the route is off).
    pub fn tree_kernel(&self) -> Option<Kernel> {
        let (predicate, materializer) = match self.tree_route {
            TreeRoute::Off => return None,
            TreeRoute::RowScan => (PredicateMode::RowScan, Materializer::None),
            TreeRoute::Bitmap => (PredicateMode::Bitmap, Materializer::DenseStatic),
            // θ = 0: the index's bitsets' θ (set by `Plan::new`).
            TreeRoute::StaticBits => (PredicateMode::StaticBits(0), Materializer::None),
            // The first of `tree_kernels` (the others are chosen per query).
            TreeRoute::Cost => (PredicateMode::RowScan, Materializer::None),
        };
        Some(Kernel {
            starts: vec![StartSource::LabelSmallest],
            start_admission: StartAdmission::Force,
            admission: Admission::Strict,
            laziness: Laziness::PredicateFirst,
            collector: Collector::Heap,
            expander: match self.tree_effort {
                TreeEffort::List => Expander::AuxOnly,
                TreeEffort::Curve | TreeEffort::Capped | TreeEffort::Fixed(_) => {
                    Expander::AuxFrontier
                }
            },
            terminator: Terminator::Exhaust,
            predicate,
            materializer,
            weights: self.weights,
            phase_times: self.phase_times,
            cstar_members: self.tree_route == TreeRoute::StaticBits,
            ..Kernel::default()
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
        for s in [
            "all",
            "strict",
            "projection(C3, C4)",
            "projection(c*)",
            "cutoff(1)",
        ] {
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
            "posting_l(16)",
            "effort_curve",
            "effort_curve(16)",
        ] {
            assert_eq!(Terminator::parse(s).unwrap().to_string(), s);
        }
        assert_eq!(
            Terminator::parse("posting_l").unwrap(),
            Terminator::PostingL { scale: 16.0 }
        );
        for s in ["threshold", "cost", "guarded", "graph"] {
            assert_eq!(Crossover::parse(s).unwrap().to_string(), s);
        }
        for s in ["bounds", "sketch"] {
            assert_eq!(EstimateSource::parse(s).unwrap().to_string(), s);
        }
        for s in [
            "forecast+budget+witness",
            "budget",
            "none",
            "forecast+witness",
        ] {
            assert_eq!(Fallbacks::parse(s).unwrap().to_string(), s);
        }
        assert!(Fallbacks::parse("panic").is_err());
        assert_eq!(parse_exact("planned").unwrap(), ExactPath::Planned(0));
        assert_eq!(
            exact_name(parse_exact("planned(500)").unwrap()),
            "planned(500)"
        );
        assert_eq!(
            Laziness::parse("heap_bound").unwrap().to_string(),
            "heap_bound"
        );
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
        assert!(!bad(&[
            ("predicate", "bitmap"),
            ("materializer", "dense_nav"),
            ("admission", "projection(c*)"),
            ("collector", "heap")
        ]));
        assert!(bad(&[
            ("predicate", "bitmap"),
            ("materializer", "dense_nav"),
            ("admission", "cutoff(1)")
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
        assert!(
            bad(&[("laziness", "heap_bound")]),
            "needs the heap collector"
        );
        assert!(bad(&[
            ("laziness", "heap_bound"),
            ("collector", "heap"),
            ("scorer", "bonus(0.5)")
        ]));
        assert!(!bad(&[("laziness", "heap_bound"), ("collector", "heap")]));
        assert!(
            bad(&[("crossover", "cost")]),
            "needs the planned exact path"
        );
        assert!(!bad(&[("crossover", "cost"), ("exact", "planned")]));
        assert!(bad(&[("crossover", "guarded"), ("exact", "planned")]));
        assert!(!bad(&[
            ("crossover", "guarded"),
            ("exact", "planned"),
            ("collector", "heap")
        ]));
        assert!(bad(&[("terminator", "posting_l(0.5)")]));
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
