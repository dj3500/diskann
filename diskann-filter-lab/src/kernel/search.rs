/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The per-query flow and the search loop (see the module documentation of [`super`]).

use std::marker::PhantomData;

use anyhow::{bail, ensure, Result};

use super::{
    beam::{Beam, BeamKey, Inserted, State},
    collect::{Answers, Collector},
    config::{
        Admission, Expander, Family, Kernel, Laziness, Padding, PredicateMode, Preset, Scorer,
    },
    config::{Crossover, EstimateSource, ExactChecks, StartAdmission, StartSource, Terminator},
    cost::{l_multiplier, Estimates, ExactPlan, PlanInputs, UnitCosts},
    entry::{self, SampleIndex, StartPredicate},
    eval::{
        ClauseSearch, Compiled, DenseBitmap, Eval, EvalCtx, PredOut, RowScan, SparseList,
        StaticBits,
    },
    exact::{self, ExactEval},
    materialize::{Materialized, Materializer},
    planner::{Decision, Fallback, Guard, Probe},
    score::{
        self, AdditiveScore, BonusScore, DistanceScore, LexicographicScore, MultiplicativeScore,
        Score,
    },
    trace::Trace,
    visited::Visited,
};
use crate::{
    bitsets::LabelBitsets,
    graph::{Graph, NodeVectors},
    labels::LabelStore,
    predicate::Cnf,
    sketches::LabelSketches,
    starts::Starts,
    vectors::Element,
};

mod special;

pub use special::FastPath;

/// Vectors prefetched ahead of their distance computation.
const LOOKAHEAD: usize = 8;

/// Everything a search reads: the graph and its node space, vectors, labels and start tables.
#[derive(Debug, Clone, Copy)]
pub struct Index<'a, T> {
    pub graph: &'a Graph,
    /// Vectors by node id (data points and auxiliary nodes).
    pub vectors: NodeVectors<'a, T>,
    /// Labels of the data points (the predicate's dictionary).
    pub labels: &'a LabelStore,
    /// Global starts in priority order.
    pub global: &'a [u32],
    /// Per-label starts over the start predicate's dictionary.
    pub label_starts: Option<&'a Starts>,
    /// Per-label sample postings (`sample(k)` starts).
    pub sample: Option<&'a SampleIndex>,
    /// Whether the per-label starts refer to a label view (the queries then carry the view's
    /// predicate).
    pub label_view: bool,
    /// Posting offsets of the label view's labels (to pick the view clause with the smallest
    /// upper bound), if known.
    pub view_postings: Option<&'a [u64]>,
    /// Positive base-derived minimum of the per-query scales (`ε_min`; 0 if unknown).
    pub eps_min: f32,
    /// Static per-label bitsets of `labels` (the `static_bits(θ)` predicate mode, and the
    /// exact path v2's checks and bitmaps).
    pub bitsets: Option<&'a LabelBitsets>,
    /// Unit costs of the cost model (the planned exact path and the crossovers).
    pub costs: Option<&'a UnitCosts>,
    /// Bottom-`b` sketches of `labels` (the `sketch` estimates).
    pub sketches: Option<&'a LabelSketches>,
}

/// One query: its vector, predicate and, with a label view, the view's predicate.
#[derive(Debug, Clone, Copy)]
pub struct Query<'a, T> {
    pub vector: &'a [T],
    pub cnf: Cnf<'a>,
    pub view: Option<Cnf<'a>>,
}

/// How a query was answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Route {
    /// Graph search.
    #[default]
    Graph,
    /// Exact path.
    Exact,
    /// A clause is empty: no answers.
    Empty,
    /// A guarded graph trial, then the exact path.
    Fallback,
}

impl Route {
    pub fn name(self) -> &'static str {
        match self {
            Self::Graph => "graph",
            Self::Exact => "exact",
            Self::Empty => "empty",
            Self::Fallback => "fallback",
        }
    }
}

/// Per-query counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counters {
    /// Distance computations (auxiliary nodes and the pilot included).
    pub n_dist: u64,
    /// Distance computations to auxiliary nodes.
    pub n_aux_dist: u64,
    /// Predicate evaluations (navigation, final checks and sample-start checks).
    pub n_pred: u64,
    /// Nodes whose predicate and insertion were skipped by `distance_first`.
    pub n_lazy: u64,
    /// Nodes expanded (popped from the beam).
    pub n_expand: u64,
    /// Adjacency lists read (expansions and second-hop reads).
    pub n_hops: u64,
    /// Expansions of auxiliary nodes.
    pub n_aux_expand: u64,
    /// Nodes inserted through the second hop.
    pub n_two_hop: u64,
    /// Evaluated nodes satisfying the navigation predicate.
    pub n_match_seen: u64,
    /// Beam entries evicted by insertions.
    pub n_evict: u64,
    /// Candidates examined by the exact path.
    pub n_exact_cand: u64,
    /// Answers returned.
    pub n_out: u64,
    /// Starts selected (after deduplication).
    pub n_start: u64,
    /// Final expandable beam size (L after adaptive resizing).
    pub final_l: u64,
    pub route: Route,
    /// Predicate evaluations of the sketch probe (also counted in `n_pred`).
    pub n_probe: u64,
    /// Exact plan chosen ([`ExactPlan`] code; 0 for none or the version-1 paths).
    pub plan: u64,
    /// Why a guarded trial fell back ([`Fallback`] code).
    pub fallback: u64,
    /// Estimated cost of the exact path and nominal graph cost (model ns; 0 if not computed).
    pub est_exact_ns: u64,
    pub est_graph_ns: u64,
    /// Estimates of `|union(c*)|` and of the match count (0 if not computed).
    pub a_hat: u64,
    pub m_hat: u64,
    /// Model cost of the graph work of a guarded trial (ns).
    pub graph_ns: u64,
}

impl Counters {
    /// Adds the event counts of `o` (not `n_out`, `n_start`, `final_l` or the route).
    #[inline(always)]
    pub fn add(&mut self, o: &Counters) {
        self.n_dist += o.n_dist;
        self.n_aux_dist += o.n_aux_dist;
        self.n_pred += o.n_pred;
        self.n_lazy += o.n_lazy;
        self.n_expand += o.n_expand;
        self.n_hops += o.n_hops;
        self.n_aux_expand += o.n_aux_expand;
        self.n_two_hop += o.n_two_hop;
        self.n_match_seen += o.n_match_seen;
        self.n_evict += o.n_evict;
        self.n_exact_cand += o.n_exact_cand;
        self.n_probe += o.n_probe;
    }

    /// Names of the numeric counters, in [`Self::values`] order.
    pub const NAMES: [&'static str; 22] = [
        "n_dist",
        "n_pred",
        "n_expand",
        "n_hops",
        "n_match_seen",
        "n_out",
        "n_aux_dist",
        "n_lazy",
        "n_aux_expand",
        "n_two_hop",
        "n_evict",
        "n_exact_cand",
        "n_start",
        "final_L",
        "n_probe",
        "plan",
        "fallback",
        "est_exact_ns",
        "est_graph_ns",
        "a_hat",
        "m_hat",
        "graph_ns",
    ];

    /// The numeric counters, in [`Self::NAMES`] order.
    pub fn values(&self) -> [u64; 22] {
        [
            self.n_dist,
            self.n_pred,
            self.n_expand,
            self.n_hops,
            self.n_match_seen,
            self.n_out,
            self.n_aux_dist,
            self.n_lazy,
            self.n_aux_expand,
            self.n_two_hop,
            self.n_evict,
            self.n_exact_cand,
            self.n_start,
            self.final_l,
            self.n_probe,
            self.plan,
            self.fallback,
            self.est_exact_ns,
            self.est_graph_ns,
            self.a_hat,
            self.m_hat,
            self.graph_ns,
        ]
    }
}

/// Predicate evaluation as a compile-time choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvalMode {
    RowScan,
    ClauseSearch,
    Dense,
    Sparse,
    StaticBits,
}

/// A kernel configuration resolved against an index (validated once per run).
#[derive(Debug, Clone)]
pub struct Plan {
    pub kernel: Kernel,
    projection: Option<Vec<u32>>,
    two_hop_cap: usize,
    eval: EvalMode,
    /// `θ` of the `static_bits(θ)` mode.
    static_theta: u64,
    family: Family,
    pilot: bool,
    complete: bool,
    n_aux: usize,
    fast: FastPath,
    /// Unit costs (planned exact path, crossovers).
    costs: Option<UnitCosts>,
    /// Evaluator of the planned exact path and of the probe.
    exact_eval: ExactEval,
}

impl Plan {
    /// Validates `kernel` and resolves it against `ix`.
    pub fn new<T: Element>(kernel: &Kernel, ix: &Index<'_, T>) -> Result<Self> {
        kernel.validate()?;
        let labels = ix.labels;
        let projection = match &kernel.admission {
            Admission::Projection(names) => {
                let mut ids = Vec::new();
                for name in names {
                    let c = (0..labels.n_categories() as u32)
                        .find(|&c| labels.category_name(c) == name.as_str());
                    match c {
                        Some(c) => ids.push(c),
                        None => bail!("projection: no category named {name:?} in the labels"),
                    }
                }
                ids.sort_unstable();
                ids.dedup();
                Some(ids)
            }
            _ => None,
        };
        let two_hop_cap = match kernel.expander {
            Expander::TwoHop(Some(c)) => c,
            _ => (ix.graph.r_max() / 2).max(1),
        };
        let eval = match (kernel.predicate, kernel.materializer) {
            (PredicateMode::RowScan, _) => EvalMode::RowScan,
            (PredicateMode::ClauseSearch, _) => EvalMode::ClauseSearch,
            (PredicateMode::Bitmap, Materializer::Dense | Materializer::DenseStatic) => {
                EvalMode::Dense
            }
            (PredicateMode::Bitmap, _) => EvalMode::Sparse,
            (PredicateMode::StaticBits(_), _) => EvalMode::StaticBits,
        };
        let static_theta = match kernel.predicate {
            PredicateMode::StaticBits(t) => {
                let b = ix.bitsets.ok_or_else(|| {
                    anyhow::anyhow!(
                        "predicate {} needs static per-label bitsets",
                        kernel.predicate
                    )
                })?;
                ensure!(
                    b.dict_hash() == labels.dict_hash() && b.n_points() == labels.n_points(),
                    "the static bitsets refer to another label store"
                );
                ensure!(
                    b.theta() <= t,
                    "predicate {} needs bitsets of every label with at least {t} points; they were built with θ = {}",
                    kernel.predicate,
                    b.theta()
                );
                t
            }
            _ => 0,
        };
        if kernel.materializer == Materializer::DenseStatic {
            let b = ix.bitsets.ok_or_else(|| {
                anyhow::anyhow!("the dense_static materializer needs static per-label bitsets")
            })?;
            ensure!(
                b.dict_hash() == labels.dict_hash() && b.n_points() == labels.n_points(),
                "the static bitsets refer to another label store"
            );
        }
        for s in &kernel.starts {
            match s {
                StartSource::Global => ensure!(
                    !ix.global.is_empty(),
                    "the start source `global` needs global starts"
                ),
                StartSource::LabelSmallest | StartSource::LabelAll => {
                    let t = ix.label_starts.ok_or_else(|| {
                        anyhow::anyhow!("the start source `{s}` needs a table of per-label starts")
                    })?;
                    ensure!(
                        ix.label_view || (t.dict_hash() == labels.dict_hash() && t.n_labels() == labels.n_labels()),
                        "the per-label starts refer to another dictionary than the labels; give the run a label view"
                    );
                }
                StartSource::Sample(_) => ensure!(
                    ix.sample.is_some(),
                    "the start source `{s}` needs a start table with a sample"
                ),
                StartSource::Aux => {}
            }
        }
        if kernel.scorer.needs_pilot() {
            ensure!(
                ix.eps_min > 0.0 && ix.eps_min.is_finite(),
                "scorer {} needs a positive ε_min of the index",
                kernel.scorer
            );
            ensure!(!ix.global.is_empty(), "the pilot needs a global start");
        }
        if let Terminator::AdaptiveL { .. } = kernel.terminator {
            ensure!(
                kernel.laziness != Laziness::Deferred,
                "adaptive_l counts matches, which deferred evaluation does not see"
            );
        }
        let costs = if kernel.exact.is_planned() || kernel.crossover != Crossover::Threshold {
            let c = ix.costs.ok_or_else(|| {
                anyhow::anyhow!(
                    "the planned exact path and the cost crossovers need unit costs (a calibration)"
                )
            })?;
            c.validate()?;
            Some(c.clone())
        } else {
            None
        };
        if kernel.estimates == EstimateSource::Sketch {
            let sk = ix.sketches.ok_or_else(|| {
                anyhow::anyhow!("the sketch estimates need bottom-b sketches of the labels")
            })?;
            ensure!(
                sk.dict_hash() == labels.dict_hash() && sk.n_points() == labels.n_points(),
                "the sketches refer to another label store"
            );
        }
        if let Some(b) = ix.bitsets {
            ensure!(
                b.dict_hash() == labels.dict_hash() && b.n_points() == labels.n_points(),
                "the static bitsets refer to another label store"
            );
        }
        let exact_eval = if ix.bitsets.is_some()
            && (kernel.exact.is_planned() || kernel.estimates == EstimateSource::Sketch)
        {
            ExactEval::StaticBits
        } else {
            ExactEval::RowScan
        };
        let fast = if eval != EvalMode::RowScan {
            FastPath::None
        } else if *kernel == Preset::Postfilter.kernel() {
            FastPath::Postfilter
        } else if *kernel == Preset::UpstreamInline.kernel() {
            FastPath::Inline
        } else {
            FastPath::None
        };
        Ok(Self {
            kernel: kernel.clone(),
            projection,
            two_hop_cap,
            eval,
            static_theta,
            family: kernel.scorer.family(),
            pilot: kernel.scorer.needs_pilot(),
            complete: kernel.needs_g(),
            n_aux: ix.graph.n_aux(),
            fast,
            costs,
            exact_eval,
        })
    }

    /// Whether the plan may build dense bitmaps (materializers or the exact `bitmap` plan).
    pub fn needs_dense(&self) -> bool {
        matches!(
            self.kernel.materializer,
            Materializer::Dense | Materializer::DenseStatic
        ) || (self.kernel.exact.is_planned() && self.kernel.exact_plans.contains(ExactPlan::Bitmap))
    }

    /// The same plan without its hand-specialized fast path (every query runs the
    /// configurable loop).
    pub fn generic(mut self) -> Self {
        self.fast = FastPath::None;
        self
    }

    /// The fast path this plan uses for graph searches (see [`FastPath`]).
    pub fn fast_path(&self) -> FastPath {
        self.fast
    }

    /// The two-hop truncation used by this plan.
    pub fn two_hop_cap(&self) -> usize {
        self.two_hop_cap
    }

    /// Scorer parameters of a query.
    fn score_params(&self, q: &Compiled, s: &Scales) -> score::Params {
        let k = &self.kernel;
        let nav_w = q.nav_weight();
        let force = k.start_admission == StartAdmission::Force;
        let g_adm = match &k.admission {
            Admission::All => nav_w,
            Admission::Strict | Admission::Projection(_) => 0.0,
            Admission::Cutoff(t) => (*t as f32 * q.max_nav_weight()).min(nav_w),
        };
        let g_aux = if self.n_aux > 0 { k.aux_g } else { 0.0 };
        let g_max = g_adm.max(if force { nav_w } else { 0.0 }).max(g_aux);
        let mut p = score::Params {
            g_max,
            nonmatch_in_beam: !matches!(k.admission, Admission::Strict | Admission::Projection(_))
                || force
                || self.n_aux > 0,
            ..score::Params::default()
        };
        match k.scorer {
            Scorer::Distance | Scorer::Lexicographic => {}
            Scorer::Bonus(b) => p.beta = b,
            Scorer::Additive(l) => p.c = l,
            Scorer::NormalizedAdditive(e) => p.c = e * s.s,
            Scorer::Multiplicative(e) => {
                p.eps = s.eps;
                p.eta = e;
            }
            Scorer::Shifted(e) => {
                p.eps = s.s;
                p.eta = e;
            }
            Scorer::ShellAdaptive(kappa) => {
                p.eps = s.eps;
                p.eta = if nav_w > 0.0 {
                    kappa * (1.0 + s.rho).ln() / nav_w
                } else {
                    0.0
                };
            }
        }
        p
    }
}

/// Per-query scales from the pilot.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Scales {
    /// Median pilot distance (at least `ε_min`).
    pub s: f32,
    /// `max(0.02·s, ε_min)`.
    pub eps: f32,
    /// `(d₍₃₂₎ − d₍₁₎) / d₍₁₎` over the first 32 pilot distances (`d₍₁₎` floored at `ε_min`).
    pub rho: f32,
}

impl Scales {
    fn from_pilot(ds: &[f32], eps_min: f32, sorted: &mut Vec<f32>) -> Self {
        if ds.is_empty() {
            return Self {
                s: eps_min,
                eps: eps_min,
                rho: 0.0,
            };
        }
        sorted.clear();
        sorted.extend_from_slice(&ds[..ds.len().min(32)]);
        sorted.sort_unstable_by(f32::total_cmp);
        let (lo, hi) = (sorted[0], sorted[sorted.len() - 1]);
        let rho = (hi - lo) / lo.max(eps_min);
        sorted.clear();
        sorted.extend_from_slice(ds);
        sorted.sort_unstable_by(f32::total_cmp);
        let n = sorted.len();
        let median = if n % 2 == 1 {
            sorted[n / 2]
        } else {
            0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
        };
        let s = median.max(eps_min);
        Self {
            s,
            eps: (0.02 * s).max(eps_min),
            rho,
        }
    }
}

/// A second-hop candidate (laid out like upstream's `Neighbor`, so that the unstable sort by
/// distance orders ties as upstream does).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub(crate) struct Cand {
    id: u32,
    d: f32,
}

/// Reusable per-run scratch and the per-query flow.
#[derive(Debug)]
pub struct Searcher {
    beam32: Beam<u32>,
    beam64: Beam<u64>,
    visited: Visited,
    bits: Vec<u64>,
    compiled: Compiled,
    mat: Materialized,
    answers: Answers,
    starts: Vec<u32>,
    pilot: Vec<(u32, f32)>,
    pilot_d: Vec<f32>,
    sorted: Vec<f32>,
    nbrs: Vec<u32>,
    rejected: Vec<Cand>,
    frontier: Vec<u32>,
    heap: Vec<(u32, u32)>,
    pos: Vec<usize>,
    probe: Probe,
    last_estimates: Estimates,
    /// When set, the next searches record their events here (untimed diagnostics).
    pub trace: Option<Trace>,
}

impl Searcher {
    /// Scratch for an index with `n_nodes` nodes, `n_data` data points and `n_labels` labels;
    /// `dense` allocates the dense-bitmap buffers.
    #[expect(
        clippy::slow_vector_initialization,
        reason = "writes every page now, so that no page fault happens during a timed search"
    )]
    pub fn new(n_nodes: usize, n_data: usize, n_labels: usize, dense: bool) -> Self {
        let mut bits = Vec::with_capacity(n_labels);
        bits.resize(n_labels, 0);
        Self {
            beam32: Beam::new(),
            beam64: Beam::new(),
            visited: Visited::new(n_nodes),
            bits,
            compiled: Compiled::new(),
            mat: Materialized::new(n_data, dense),
            answers: Answers::new(),
            starts: Vec::new(),
            pilot: Vec::new(),
            pilot_d: Vec::new(),
            sorted: Vec::new(),
            nbrs: Vec::new(),
            rejected: Vec::new(),
            frontier: Vec::new(),
            heap: Vec::new(),
            pos: Vec::new(),
            probe: Probe::new(),
            last_estimates: Estimates::default(),
            trace: None,
        }
    }

    /// Scratch sized for `ix` and the materializers of `plans`.
    pub fn for_index<T: Element>(ix: &Index<'_, T>, plans: &[&Plan]) -> Self {
        let dense = plans.iter().any(|p| p.needs_dense());
        Self::new(
            ix.graph.n_nodes(),
            ix.graph.n_data(),
            ix.labels.n_labels(),
            dense,
        )
    }

    /// The final beam of the last graph search with `f32` keys (diagnostics).
    pub fn beam(&self) -> &Beam<u32> {
        &self.beam32
    }

    /// Answers query `q` with list size `l`, writing `ids.len()` = K results (ids and true
    /// squared distances, sorted by (distance, id), padded with `u32::MAX` / `f32::MAX`).
    /// With `reference = Some(W)`, the beam is unbounded (nothing evicted, every entry
    /// expandable) and the search stops after `W` distance computations.
    ///
    /// # Panics
    ///
    /// Panics if `ids` and `dists` differ in length, or if the plan needs a label-view
    /// predicate that the query lacks.
    #[expect(
        clippy::too_many_arguments,
        reason = "the per-query inputs and outputs"
    )]
    pub fn search<T: Element>(
        &mut self,
        plan: &Plan,
        ix: &Index<'_, T>,
        q: &Query<'_, T>,
        l: usize,
        ids: &mut [u32],
        dists: &mut [f32],
        reference: Option<u64>,
    ) -> Counters {
        assert_eq!(ids.len(), dists.len());
        let k = ids.len();
        let kernel = &plan.kernel;
        let mut ctr = Counters::default();
        self.compiled.compile(
            ix.labels,
            q.cnf,
            kernel.weights,
            plan.projection.as_deref(),
            &mut self.bits,
        );
        let stat = (plan.eval == EvalMode::StaticBits).then(|| {
            let b = ix
                .bitsets
                .expect("a static_bits plan needs the static bitsets of the index");
            self.compiled
                .split_static(ix.labels, b, plan.static_theta, plan.complete);
            b
        });
        // The planned exact path and the probe check clauses with the static bitsets when
        // the index has them (the graph search may use another evaluator).
        let exact_stat = match (stat, plan.exact_eval, ix.bitsets) {
            (Some(b), _, _) => Some(b),
            (None, ExactEval::StaticBits, Some(b)) => {
                self.compiled.split_static(ix.labels, b, b.theta(), false);
                Some(b)
            }
            _ => None,
        };
        let n = if self.compiled.is_empty() {
            ctr.route = Route::Empty;
            0
        } else if kernel.crossover == Crossover::Threshold {
            let routed = kernel
                .exact
                .threshold()
                .is_some_and(|t| self.compiled.min_upper_bound() <= t);
            if routed && kernel.exact.is_planned() {
                ctr.route = Route::Exact;
                let costs = plan.costs.as_ref().expect("validated");
                let est = self.estimates(plan, ix, exact_stat, &mut ctr);
                let (p, e) = PlanInputs {
                    q: &self.compiled,
                    labels: ix.labels,
                    bitsets: exact_stat,
                    k,
                }
                .best(costs, kernel.exact_plans, &est, false, false);
                record(&mut ctr, p, e, 0.0, &est);
                self.exact_v2(plan, ix, q, k, p, exact_stat, false, &mut ctr);
                self.answers.finish(ids, dists)
            } else if routed {
                ctr.route = Route::Exact;
                let mut cx = EvalCtx::new(ix.labels, &self.compiled, &self.bits, &[], &[]);
                if let Some(b) = stat {
                    cx = cx.with_static(b);
                }
                exact::run(
                    kernel.exact,
                    plan.eval,
                    &cx,
                    &ix.vectors,
                    ix.graph.n_data(),
                    q.vector,
                    k,
                    &mut self.mat,
                    &mut self.answers,
                    &mut ctr,
                );
                self.answers.finish(ids, dists)
            } else {
                let mut l_run = l;
                if let Terminator::PostingL { scale } = kernel.terminator {
                    let est = self.estimates(plan, ix, exact_stat, &mut ctr);
                    l_run = planned_l(l, &est, scale);
                }
                self.graph(plan, ix, q, l_run, reference, &mut ctr, ids, dists, None)
                    .expect("an unguarded graph search finishes")
            }
        } else {
            let d = self.decide(plan, ix, l, k, exact_stat, &mut ctr);
            if d.exact {
                ctr.route = Route::Exact;
                self.exact_v2(plan, ix, q, k, d.plan, exact_stat, false, &mut ctr);
                self.answers.finish(ids, dists)
            } else {
                let costs = plan.costs.as_ref().expect("validated");
                let mut guard = (kernel.crossover == Crossover::Guarded && reference.is_none())
                    .then(|| {
                        Guard::new(
                            d.est_exact,
                            k,
                            kernel.fallback,
                            costs.graph_dist_ns,
                            costs.graph_pred_ns,
                        )
                    });
                match self.graph(
                    plan,
                    ix,
                    q,
                    d.l,
                    reference,
                    &mut ctr,
                    ids,
                    dists,
                    guard.as_mut(),
                ) {
                    Some(n) => {
                        if guard.is_some() {
                            ctr.graph_ns = costs.charged(ctr.n_dist, ctr.n_pred) as u64;
                        }
                        n
                    }
                    None => {
                        let g = guard.expect("only a guarded trial falls back");
                        ctr.route = Route::Fallback;
                        ctr.fallback = g.outcome as u64;
                        ctr.graph_ns = costs.charged(ctr.n_dist, ctr.n_pred) as u64;
                        // Re-plan with the seeded heap (distance-first prunes with its K-th).
                        let est = self.last_estimates.clone();
                        let (p, _) = PlanInputs {
                            q: &self.compiled,
                            labels: ix.labels,
                            bitsets: exact_stat,
                            k,
                        }
                        .best(costs, kernel.exact_plans, &est, true, false);
                        ctr.plan = p as u64;
                        self.exact_v2(plan, ix, q, k, p, exact_stat, true, &mut ctr);
                        self.answers.finish(ids, dists)
                    }
                }
            }
        };
        self.compiled.clear_bits(&mut self.bits);
        ids[n..].fill(u32::MAX);
        dists[n..].fill(f32::MAX);
        ctr.n_out = n as u64;
        ctr
    }

    /// Estimates of the query (a sketch probe, or the posting bounds); kept for a fallback.
    fn estimates<T: Element>(
        &mut self,
        plan: &Plan,
        ix: &Index<'_, T>,
        exact_stat: Option<&LabelBitsets>,
        ctr: &mut Counters,
    ) -> Estimates {
        let n = ix.graph.n_data();
        let est = match (plan.kernel.estimates, ix.sketches) {
            (EstimateSource::Sketch, Some(sk)) if self.compiled.n_clauses() > 0 => {
                let mut cx = EvalCtx::new(ix.labels, &self.compiled, &self.bits, &[], &[]);
                if plan.kernel.exact_checks == ExactChecks::Residual {
                    cx = cx.cstar_members();
                }
                match exact_stat {
                    Some(b) => {
                        cx = cx.with_static(b);
                        self.probe.run::<StaticBits>(sk, &cx, n, ctr)
                    }
                    None => self.probe.run::<RowScan>(sk, &cx, n, ctr),
                }
            }
            _ => {
                self.probe.witnesses.clear();
                Estimates::from_bounds(&self.compiled, n)
            }
        };
        self.last_estimates = est.clone();
        est
    }

    /// The cost crossover (see [`super::planner`]).
    fn decide<T: Element>(
        &mut self,
        plan: &Plan,
        ix: &Index<'_, T>,
        l: usize,
        k: usize,
        exact_stat: Option<&LabelBitsets>,
        ctr: &mut Counters,
    ) -> Decision {
        let kernel = &plan.kernel;
        let costs = plan.costs.as_ref().expect("validated");
        let n = ix.graph.n_data();
        // Nominal graph cost: the calibrated distances at the list size, the predicate
        // evaluations they lead to under the configured laziness (match rate from the
        // posting bounds before the probe, from the estimates after it), plus the starts.
        let starts = start_count(kernel, ix, &self.compiled) as f64
            * (costs.graph_dist_ns + costs.graph_pred_ns);
        let bounds = Estimates::from_bounds(&self.compiled, n);
        let g0 = costs.graph_cost(l as f64, k, bounds.m / bounds.n, kernel.laziness) + starts;
        let pi = PlanInputs {
            q: &self.compiled,
            labels: ix.labels,
            bitsets: exact_stat,
            k,
        };
        // Obviously cheap exact queries: the exact path's upper cost is below the cheapest
        // graph search; no probe.
        let (_, e_up) = pi.best(
            costs,
            kernel.exact_plans,
            &Estimates::upper(&self.compiled, n),
            false,
            true,
        );
        if e_up <= g0 {
            let est = bounds;
            let (p, e) = pi.best(costs, kernel.exact_plans, &est, false, false);
            record(ctr, p, e.min(e_up), g0, &est);
            self.last_estimates = est;
            self.probe.witnesses.clear();
            return Decision {
                exact: true,
                plan: p,
                est_exact: e.min(e_up),
                est_graph: g0,
                l,
            };
        }
        let est = self.estimates(plan, ix, exact_stat, ctr);
        let pi = PlanInputs {
            q: &self.compiled,
            labels: ix.labels,
            bitsets: exact_stat,
            k,
        };
        let (p, e) = pi.best(costs, kernel.exact_plans, &est, false, false);
        let scale = match kernel.terminator {
            Terminator::PostingL { scale } | Terminator::AdaptiveL { scale, .. } => scale,
            _ => 1.0,
        };
        let l_plan = planned_l(l, &est, scale);
        let rate = if est.n > 0.0 { est.m / est.n } else { 0.0 };
        let g = costs.graph_cost(l_plan as f64, k, rate, kernel.laziness) + starts;
        record(ctr, p, e, g, &est);
        Decision {
            exact: e <= g,
            plan: p,
            est_exact: e,
            est_graph: g,
            l: if matches!(kernel.terminator, Terminator::PostingL { .. }) {
                l_plan
            } else {
                l
            },
        }
    }

    /// The planned exact path for the compiled query (`seeded`: the answer heap holds a
    /// graph search's answers).
    #[expect(clippy::too_many_arguments, reason = "the per-query inputs")]
    fn exact_v2<T: Element>(
        &mut self,
        plan: &Plan,
        ix: &Index<'_, T>,
        q: &Query<'_, T>,
        k: usize,
        p: ExactPlan,
        exact_stat: Option<&LabelBitsets>,
        seeded: bool,
        ctr: &mut Counters,
    ) {
        let mut cx = EvalCtx::new(ix.labels, &self.compiled, &self.bits, &[], &[]);
        if plan.kernel.exact_checks == ExactChecks::Residual {
            cx = cx.cstar_members();
        }
        let eval = match exact_stat {
            Some(b) => {
                cx = cx.with_static(b);
                ExactEval::StaticBits
            }
            None => ExactEval::RowScan,
        };
        debug_assert!(plan.kernel.exact.is_planned());
        exact::run_planned(
            p,
            eval,
            &cx,
            &ix.vectors,
            ix.graph.n_data(),
            q.vector,
            k,
            &mut self.mat,
            &mut self.answers,
            seeded,
            ctr,
        );
    }

    /// The graph search (start selection, then the loop): `Some(answers written)`, or `None`
    /// when a guarded trial falls back (its answers are left in the heap collector).
    #[expect(
        clippy::too_many_arguments,
        reason = "the per-query inputs and outputs"
    )]
    fn graph<T: Element>(
        &mut self,
        plan: &Plan,
        ix: &Index<'_, T>,
        q: &Query<'_, T>,
        l: usize,
        reference: Option<u64>,
        ctr: &mut Counters,
        ids: &mut [u32],
        dists: &mut [f32],
        guard: Option<&mut Guard>,
    ) -> Option<usize> {
        let kernel = &plan.kernel;
        ctr.route = Route::Graph;
        self.visited.clear();
        let predicate = if ix.label_view {
            let cnf = q
                .view
                .expect("the plan's per-label starts need the query's label-view predicate");
            StartPredicate::View {
                cnf,
                postings: ix.view_postings,
            }
        } else {
            StartPredicate::Base
        };
        let inputs = entry::Inputs {
            sources: &kernel.starts,
            sample_start: kernel.sample_start,
            global: ix.global,
            label_starts: ix.label_starts,
            sample: ix.sample,
            predicate,
            n_data: ix.graph.n_data() as u32,
            n_nodes: ix.graph.n_nodes() as u32,
        };
        let cx = EvalCtx::new(ix.labels, &self.compiled, &self.bits, &[], &[]);
        entry::select(
            &inputs,
            &cx,
            &mut self.visited,
            &mut self.starts,
            &mut self.heap,
            &mut self.pos,
            ctr,
        );
        ctr.n_start = self.starts.len() as u64;
        match plan.fast {
            FastPath::Postfilter if reference.is_none() && self.trace.is_none() => {
                Some(self.fast_postfilter(ix, q, l, ctr, ids, dists))
            }
            FastPath::Inline if reference.is_none() && self.trace.is_none() => {
                Some(self.fast_inline(ix, q, l, ctr, ids, dists))
            }
            _ => {
                let witnesses = std::mem::take(&mut self.probe.witnesses);
                let out = dispatch(
                    plan.eval,
                    plan.family,
                    self,
                    plan,
                    ix,
                    q,
                    l,
                    reference,
                    ctr,
                    ids,
                    dists,
                    guard,
                    &witnesses,
                );
                self.probe.witnesses = witnesses;
                out
            }
        }
    }
}

/// Number of starts a graph search of the query will select (before deduplication).
fn start_count<T: Element>(kernel: &Kernel, ix: &Index<'_, T>, q: &Compiled) -> usize {
    let per_label = |l: u32| {
        ix.label_starts
            .filter(|t| !ix.label_view && (l as usize) < t.n_labels())
            .map_or(0, |t| t.label_starts(l).len())
    };
    let mut n = 0;
    for s in &kernel.starts {
        n += match s {
            StartSource::Global => ix.global.len(),
            StartSource::LabelSmallest if q.n_clauses() > 0 => {
                q.clause(0).iter().map(|&l| per_label(l)).sum()
            }
            StartSource::LabelAll => (0..q.n_clauses())
                .flat_map(|j| q.clause(j).iter())
                .map(|&l| per_label(l))
                .sum(),
            StartSource::Sample(k) => *k,
            StartSource::Aux => ix.graph.n_aux(),
            StartSource::LabelSmallest => 0,
        };
    }
    n
}

/// `posting_l`: `L₀ × m(M̂ / n)` (upstream's multiplier, clamped to `scale`).
fn planned_l(l0: usize, est: &Estimates, scale: f64) -> usize {
    let rate = if est.n > 0.0 { est.m / est.n } else { 0.0 };
    ((l0 as f64) * l_multiplier(rate, scale)).round().max(1.0) as usize
}

/// Records the crossover's estimates in the counters.
fn record(ctr: &mut Counters, p: ExactPlan, e: f64, g: f64, est: &Estimates) {
    ctr.plan = p as u64;
    ctr.est_exact_ns = e.round() as u64;
    ctr.est_graph_ns = g.round() as u64;
    ctr.a_hat = est.a.round() as u64;
    ctr.m_hat = est.m.round() as u64;
}

#[expect(clippy::too_many_arguments, reason = "forwards the per-query inputs")]
fn dispatch<T: Element>(
    eval: EvalMode,
    family: Family,
    st: &mut Searcher,
    plan: &Plan,
    ix: &Index<'_, T>,
    q: &Query<'_, T>,
    l: usize,
    reference: Option<u64>,
    ctr: &mut Counters,
    ids: &mut [u32],
    dists: &mut [f32],
    guard: Option<&mut Guard>,
    witnesses: &[u32],
) -> Option<usize> {
    macro_rules! by_family {
        ($p:ty) => {
            match family {
                Family::Distance => run::<T, $p, DistanceScore>(
                    st, plan, ix, q, l, reference, ctr, ids, dists, guard, witnesses,
                ),
                Family::Bonus => run::<T, $p, BonusScore>(
                    st, plan, ix, q, l, reference, ctr, ids, dists, guard, witnesses,
                ),
                Family::Additive => run::<T, $p, AdditiveScore>(
                    st, plan, ix, q, l, reference, ctr, ids, dists, guard, witnesses,
                ),
                Family::Multiplicative => run::<T, $p, MultiplicativeScore>(
                    st, plan, ix, q, l, reference, ctr, ids, dists, guard, witnesses,
                ),
                Family::Lexicographic => run::<T, $p, LexicographicScore>(
                    st, plan, ix, q, l, reference, ctr, ids, dists, guard, witnesses,
                ),
            }
        };
    }
    match eval {
        EvalMode::RowScan => by_family!(RowScan),
        EvalMode::ClauseSearch => by_family!(ClauseSearch),
        EvalMode::Dense => by_family!(DenseBitmap),
        EvalMode::Sparse => by_family!(SparseList),
        EvalMode::StaticBits => by_family!(StaticBits),
    }
}

/// Outcome of offering a node.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Outcome {
    Admitted,
    /// Not admitted; the distance, if computed (NaN otherwise).
    Rejected(f32),
    Skipped,
    Navigated,
}

/// Upstream adaptive-L state.
#[derive(Debug, Clone, Copy)]
struct Adaptive {
    on: bool,
    visited: usize,
    matched: usize,
    next: Option<usize>,
    l0: usize,
    scale: f64,
}

impl Adaptive {
    fn new(t: Terminator, l0: usize, reference: bool) -> Self {
        match t {
            Terminator::AdaptiveL { sample, scale } if !reference => Self {
                on: true,
                visited: 0,
                matched: 0,
                next: Some(sample),
                l0,
                scale,
            },
            _ => Self {
                on: false,
                visited: 0,
                matched: 0,
                next: None,
                l0,
                scale: 1.0,
            },
        }
    }

    /// Whether the sample is still being collected (lazy skipping is then unsafe).
    #[inline(always)]
    fn sampling(&self) -> bool {
        self.on && self.next.is_some()
    }

    /// End of a step: resizes the beam when the sample count is reached (upstream).
    fn end_of_step<K: BeamKey>(&mut self, beam: &mut Beam<K>, view: &mut usize) {
        let Some(next) = self.next else { return };
        if !self.on || self.visited < next {
            return;
        }
        let new_l = adaptive_l(self.l0, self.visited, self.matched, self.scale);
        if new_l > beam.search_l() {
            beam.resize(new_l);
            *view = new_l;
        }
        self.next = if self.matched == 0 {
            next.checked_mul(2)
        } else {
            None
        };
    }
}

/// Upstream's adaptive L: `L₀ × m` with `m` = 1 (rate ≥ 50 %), 2 (10–50 %), `2^{−log10 rate}`
/// below 10 %, clamped to `[1, scale]`; the rate is `1 / visited` without a match.
pub fn adaptive_l(base_l: usize, visited: usize, matched: usize, max_multiplier: f64) -> usize {
    if visited == 0 {
        return (base_l as f64 * max_multiplier) as usize;
    }
    let specificity = if matched == 0 {
        1.0 / visited as f64
    } else {
        matched as f64 / visited as f64
    };
    let multiplier = if specificity >= 0.5 {
        1.0
    } else if specificity >= 0.1 {
        2.0
    } else {
        2.0_f64.powf(-specificity.log10())
    };
    (base_l as f64 * multiplier.clamp(1.0, max_multiplier)) as usize
}

/// The state of one graph search.
struct Run<'r, 'a, T: Element, P: Eval, S: Score> {
    cx: EvalCtx<'r>,
    vectors: NodeVectors<'a, T>,
    graph: &'a Graph,
    qv: &'r [T],
    sc: S,
    laziness: Laziness,
    admission: AdmissionRule,
    start_admission: StartAdmission,
    collector: Collector,
    complete: bool,
    projected: bool,
    aux_g: f32,
    n_data: u32,
    lazy_ok: bool,
    // The beam, the visited set, the answers and the counters are held by value (moved out of
    // the `Searcher` for the query), so that the hot loop, which takes `&mut self`, knows that
    // nothing else aliases them and keeps their state in registers.
    beam: Beam<S::Key>,
    visited: Visited,
    answers: Answers,
    ctr: Counters,
    trace: Option<&'r mut Trace>,
    guard: Option<&'r mut Guard>,
    _p: PhantomData<P>,
}

/// The admission rule with resolved parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdmissionRule {
    All,
    Match,
    Cutoff(u32),
}

/// Laziness codes of the per-expansion neighbour loops.
const EAGER: u8 = 0;
const DEFERRED: u8 = 1;
const PREDICATE_FIRST: u8 = 2;
const DISTANCE_FIRST: u8 = 3;
const HEAP_BOUND: u8 = 4;

/// Records an insertion (trace mode; kept out of the hot loop).
#[cold]
#[inline(never)]
#[expect(clippy::too_many_arguments, reason = "trace columns")]
fn trace_insert<S: Score>(
    t: &mut Trace,
    sc: S,
    res: Inserted,
    key: S::Key,
    d: f32,
    v: u32,
    source: Option<u32>,
    p: Option<PredOut>,
    complete: bool,
    outcome: &'static str,
) {
    let (rank, evicted) = match res {
        Inserted::Dropped => (None, None),
        Inserted::At { rank, evicted } => (Some(rank), evicted),
    };
    t.offer(
        v,
        source,
        Some(d),
        Some(sc.describe(key)),
        p.map(|p| p.m),
        p.filter(|_| complete).map(|p| p.g),
        outcome,
        rank,
        evicted,
    );
}

/// Records a node that did not enter the beam (trace mode).
#[cold]
#[inline(never)]
fn trace_note(
    t: &mut Trace,
    v: u32,
    source: Option<u32>,
    d: Option<f32>,
    p: Option<PredOut>,
    complete: bool,
    outcome: &'static str,
) {
    t.offer(
        v,
        source,
        d,
        None,
        p.map(|p| p.m),
        p.filter(|_| complete).map(|p| p.g),
        outcome,
        None,
        None,
    );
}

#[cold]
#[inline(never)]
fn trace_answer(t: &mut Trace, v: u32, d: f32, evicted: Option<u32>) {
    t.answer(v, d, evicted);
}

#[cold]
#[inline(never)]
fn trace_expand(t: &mut Trace, id: u32, new_step: bool) {
    t.expand(id, new_step);
}

#[cold]
#[inline(never)]
fn trace_two_hop(t: &mut Trace, cands: &[Cand], cap: usize) {
    for (rank, c) in cands.iter().enumerate() {
        t.two_hop(c.id, c.d, rank, rank < cap);
    }
}

impl<T: Element, P: Eval, S: Score> Run<'_, '_, T, P, S> {
    // The per-node methods count into `c`, a local of the caller (merged into `self.ctr` once
    // per expansion), so that the counters live in registers in the hot loop.

    #[inline(always)]
    fn distance(&self, v: u32, cached: Option<f32>, c: &mut Counters) -> f32 {
        match cached {
            Some(d) => d,
            None => {
                c.n_dist += 1;
                T::squared_l2(self.qv, self.vectors.get(v))
            }
        }
    }

    #[inline(always)]
    fn pred(&self, v: u32, c: &mut Counters) -> PredOut {
        c.n_pred += 1;
        P::eval(&self.cx, v, self.complete)
    }

    #[inline(always)]
    fn admits(&self, p: PredOut) -> bool {
        match self.admission {
            AdmissionRule::All => true,
            AdmissionRule::Match => p.m,
            AdmissionRule::Cutoff(t) => p.unit <= t,
        }
    }

    /// Whether `distance_first` may skip a node at distance `d`: it cannot enter the full
    /// beam (strictly), and it cannot be an answer either.
    #[inline(always)]
    fn skip_safe(&self, d: f32) -> bool {
        self.beam.is_full()
            && self
                .beam
                .worst_key()
                .is_some_and(|w| self.sc.key_lower_bound(d) > w)
            && (self.collector == Collector::FinalBeam
                || (self.answers.is_full() && d > self.answers.worst()))
    }

    /// Collects a navigation match (completing the full predicate under a projection) and
    /// returns the beam state of the node.
    #[inline(always)]
    fn collect<const TRACE: bool>(
        &mut self,
        v: u32,
        d: f32,
        p: PredOut,
        c: &mut Counters,
    ) -> State {
        if !p.m {
            return State::NoMatch;
        }
        c.n_match_seen += 1;
        if self.projected {
            if self.collector == Collector::FinalBeam {
                return State::Partial;
            }
            c.n_pred += 1;
            if !P::full(&self.cx, v) {
                return State::NoMatch;
            }
        }
        if self.collector != Collector::FinalBeam {
            let (inserted, evicted) = self.answers.offer(d, v);
            if TRACE && inserted {
                if let Some(t) = self.trace.as_deref_mut() {
                    trace_answer(t, v, d, evicted);
                }
            }
        }
        State::Match
    }

    #[inline(always)]
    #[expect(clippy::too_many_arguments, reason = "trace columns")]
    fn insert<const TRACE: bool>(
        &mut self,
        key: S::Key,
        d: f32,
        v: u32,
        state: State,
        source: Option<u32>,
        p: Option<PredOut>,
        outcome: &'static str,
        c: &mut Counters,
    ) {
        let res = self.beam.insert(key, d, v, state);
        if let Inserted::At {
            evicted: Some(_), ..
        } = res
        {
            c.n_evict += 1;
        }
        if TRACE {
            if let Some(t) = self.trace.as_deref_mut() {
                trace_insert(
                    t,
                    self.sc,
                    res,
                    key,
                    d,
                    v,
                    source,
                    p,
                    self.complete,
                    outcome,
                );
            }
        }
    }

    #[inline(always)]
    fn trace_only<const TRACE: bool>(
        &mut self,
        v: u32,
        source: Option<u32>,
        d: Option<f32>,
        p: Option<PredOut>,
        outcome: &'static str,
    ) {
        if TRACE {
            if let Some(t) = self.trace.as_deref_mut() {
                trace_note(t, v, source, d, p, self.complete, outcome);
            }
        }
    }

    /// Places an evaluated node: collection, then admission to the beam.
    #[inline(always)]
    fn place<const TRACE: bool>(
        &mut self,
        v: u32,
        d: f32,
        p: PredOut,
        source: Option<u32>,
        c: &mut Counters,
    ) -> Outcome {
        let state = self.collect::<TRACE>(v, d, p, c);
        if self.admits(p) {
            let key = self.sc.key(d, p.m, p.g);
            self.insert::<TRACE>(key, d, v, state, source, Some(p), "admitted", c);
            Outcome::Admitted
        } else {
            self.trace_only::<TRACE>(v, source, Some(d), Some(p), "rejected");
            Outcome::Rejected(d)
        }
    }

    #[inline(always)]
    fn navigate_aux<const TRACE: bool>(
        &mut self,
        v: u32,
        source: Option<u32>,
        cached: Option<f32>,
        c: &mut Counters,
    ) {
        if cached.is_none() {
            c.n_aux_dist += 1;
        }
        let d = self.distance(v, cached, c);
        let key = self.sc.key(d, false, self.aux_g);
        self.insert::<TRACE>(key, d, v, State::Aux, source, None, "navigated", c);
    }

    /// Offers a newly visited node with laziness `LZ`; returns the outcome.
    #[inline(always)]
    fn offer<const LZ: u8, const TRACE: bool>(
        &mut self,
        v: u32,
        source: u32,
        cached: Option<f32>,
        c: &mut Counters,
    ) -> Outcome {
        if v >= self.n_data {
            self.navigate_aux::<TRACE>(v, Some(source), cached, c);
            return Outcome::Navigated;
        }
        let src = Some(source);
        match LZ {
            DEFERRED => {
                let d = self.distance(v, cached, c);
                let key = self.sc.key(d, false, 0.0);
                self.insert::<TRACE>(key, d, v, State::Unknown, src, None, "admitted", c);
                Outcome::Admitted
            }
            PREDICATE_FIRST => {
                let p = self.pred(v, c);
                if !self.admits(p) {
                    self.trace_only::<TRACE>(v, src, None, Some(p), "rejected");
                    return Outcome::Rejected(f32::NAN);
                }
                let d = self.distance(v, cached, c);
                self.place::<TRACE>(v, d, p, src, c)
            }
            DISTANCE_FIRST => {
                let d = self.distance(v, cached, c);
                if self.lazy_ok && self.skip_safe(d) {
                    c.n_lazy += 1;
                    self.trace_only::<TRACE>(v, src, Some(d), None, "lazily_skipped");
                    return Outcome::Skipped;
                }
                let p = self.pred(v, c);
                self.place::<TRACE>(v, d, p, src, c)
            }
            HEAP_BOUND => {
                let d = self.distance(v, cached, c);
                // Strictly beyond a full answer heap's K-th: not an answer whatever the
                // predicate says (equal distances may enter by id); the node still navigates.
                if self.lazy_ok && self.answers.is_full() && d > self.answers.worst() {
                    c.n_lazy += 1;
                    let key = self.sc.key(d, false, 0.0);
                    self.insert::<TRACE>(key, d, v, State::Unknown, src, None, "lazy_navigated", c);
                    return Outcome::Admitted;
                }
                let p = self.pred(v, c);
                self.place::<TRACE>(v, d, p, src, c)
            }
            _ => {
                let d = self.distance(v, cached, c);
                let p = self.pred(v, c);
                self.place::<TRACE>(v, d, p, src, c)
            }
        }
    }

    /// [`Self::offer`] with the run's laziness (dispatched per node: for the pilot only).
    fn offer_any(&mut self, v: u32, source: u32, cached: Option<f32>, c: &mut Counters) -> Outcome {
        match self.laziness {
            Laziness::Eager => self.offer::<EAGER, true>(v, source, cached, c),
            Laziness::Deferred => self.offer::<DEFERRED, true>(v, source, cached, c),
            Laziness::PredicateFirst => self.offer::<PREDICATE_FIRST, true>(v, source, cached, c),
            Laziness::DistanceFirst => self.offer::<DISTANCE_FIRST, true>(v, source, cached, c),
            Laziness::HeapBound => self.offer::<HEAP_BOUND, true>(v, source, cached, c),
        }
    }

    /// Offers a start (marked visited by the selection).
    fn offer_start(&mut self, s: u32, cached: Option<f32>) {
        let mut c = Counters::default();
        self.offer_start_into(s, cached, &mut c);
        self.ctr.add(&c);
    }

    #[inline(always)]
    fn offer_start_into(&mut self, s: u32, cached: Option<f32>, c: &mut Counters) {
        if s >= self.n_data {
            self.navigate_aux::<true>(s, None, cached, c);
            return;
        }
        match (self.start_admission, self.laziness) {
            (_, Laziness::Deferred) => {
                let d = self.distance(s, cached, c);
                let key = self.sc.key(d, false, 0.0);
                self.insert::<true>(key, d, s, State::Unknown, None, None, "admitted", c);
            }
            (StartAdmission::Force, _) => {
                let p = self.pred(s, c);
                let d = self.distance(s, cached, c);
                let state = self.collect::<true>(s, d, p, c);
                let key = self.sc.key(d, p.m, p.g);
                let outcome = if self.admits(p) { "admitted" } else { "forced" };
                self.insert::<true>(key, d, s, state, None, Some(p), outcome, c);
            }
            (StartAdmission::Obey, Laziness::PredicateFirst) => {
                let p = self.pred(s, c);
                if !self.admits(p) {
                    self.trace_only::<true>(s, None, cached, Some(p), "rejected");
                    return;
                }
                let d = self.distance(s, cached, c);
                self.place::<true>(s, d, p, None, c);
            }
            (StartAdmission::Obey, _) => {
                let d = self.distance(s, cached, c);
                let p = self.pred(s, c);
                self.place::<true>(s, d, p, None, c);
            }
        }
    }

    #[inline(always)]
    fn prefetch(&self, v: u32, rows: bool) {
        diskann_vector::prefetch_hint_max::<4, _>(self.vectors.get(v));
        if rows && v < self.n_data {
            P::prefetch(&self.cx, v);
        }
    }

    /// Expands node `n`: offers its unvisited neighbours (collecting the rejected ones for a
    /// second hop with `TWO_HOP`).
    #[inline(always)]
    fn expand<const LZ: u8, const TWO_HOP: bool, const TRACE: bool>(
        &mut self,
        n: u32,
        nbrs: &mut Vec<u32>,
        rejected: &mut Vec<Cand>,
        adaptive: &mut Adaptive,
        c: &mut Counters,
    ) {
        let graph = self.graph;
        c.n_hops += 1;
        if n >= self.n_data {
            c.n_aux_expand += 1;
        }
        // Under `heap_bound` with a full heap most neighbours skip their predicate: their rows
        // are not prefetched.
        let rows = P::ROWS
            && LZ != DEFERRED
            && !(LZ == HEAP_BOUND && self.lazy_ok && self.answers.is_full());
        nbrs.clear();
        for &v in graph.neighbors(n) {
            if !self.visited.test_and_set(v) {
                nbrs.push(v);
                if rows && v < self.n_data {
                    P::prefetch_offsets(&self.cx, v);
                }
            }
        }
        let len = nbrs.len();
        let la = LOOKAHEAD.min(len);
        for &v in &nbrs[..la] {
            self.prefetch(v, rows);
        }
        let matched = c.n_match_seen;
        for i in 0..len {
            if i + la < len {
                self.prefetch(nbrs[i + la], rows);
            }
            let v = nbrs[i];
            let outcome = self.offer::<LZ, TRACE>(v, n, None, c);
            if TWO_HOP {
                if let Outcome::Rejected(d) = outcome {
                    rejected.push(Cand { id: v, d });
                }
            }
        }
        if adaptive.on {
            // Upstream counts every evaluated one-hop neighbour and its matches.
            adaptive.visited += len;
            adaptive.matched += (c.n_match_seen - matched) as usize;
        }
    }

    /// The search loop: until the beam has no unexpanded node or the terminator stops, expand
    /// the `W` closest unexpanded nodes (then the second hop, then adaptive resizing). Not
    /// inlined, and monomorphized over the laziness and the expander (dispatched once per
    /// query): as a function of its own, `self` is a non-aliased argument and the whole loop
    /// keeps the search state and the counters in registers.
    #[inline(never)]
    #[expect(
        clippy::too_many_arguments,
        reason = "the loop's scratch and parameters"
    )]
    fn search_loop<const LZ: u8, const TWO_HOP: bool, const TRACE: bool>(
        &mut self,
        terminator: Terminator,
        beam_width: usize,
        two_hop_cap: usize,
        frontier: &mut Vec<u32>,
        nbrs: &mut Vec<u32>,
        rejected: &mut Vec<Cand>,
        adaptive: &mut Adaptive,
        view: &mut usize,
    ) {
        let mut c = Counters::default();
        let base_dist = self.ctr.n_dist;
        loop {
            if !self.beam.has_unexpanded() {
                break;
            }
            match terminator {
                Terminator::Budget(w) => {
                    if base_dist + c.n_dist >= w {
                        break;
                    }
                }
                Terminator::MatchAware => {
                    if self.answers.is_full() {
                        if let Some(kmin) = self.beam.peek_key() {
                            if self.sc.dist_lower_bound(kmin) > self.answers.worst() {
                                break;
                            }
                        }
                    }
                }
                Terminator::Exhaust
                | Terminator::AdaptiveL { .. }
                | Terminator::PostingL { .. } => {}
            }
            frontier.clear();
            while frontier.len() < beam_width {
                match self.beam.pop() {
                    Some(id) => {
                        if TRACE {
                            if let Some(t) = self.trace.as_deref_mut() {
                                trace_expand(t, id, frontier.is_empty());
                            }
                        }
                        frontier.push(id);
                    }
                    None => break,
                }
            }
            c.n_expand += frontier.len() as u64;
            if TWO_HOP {
                rejected.clear();
            }
            for &n in frontier.iter() {
                self.expand::<LZ, TWO_HOP, TRACE>(n, nbrs, rejected, adaptive, &mut c);
            }
            if TWO_HOP {
                self.two_hop::<TRACE>(rejected, two_hop_cap, &mut c);
            }
            if adaptive.on {
                adaptive.end_of_step(&mut self.beam, view);
                self.lazy_ok = !adaptive.sampling();
            }
            if let Some(g) = self.guard.as_deref_mut() {
                let nd = self.ctr.n_dist + c.n_dist;
                if nd >= g.next_check
                    && g.check(
                        nd,
                        self.ctr.n_pred + c.n_pred,
                        self.ctr.n_match_seen + c.n_match_seen,
                        self.answers.len(),
                    )
                {
                    break;
                }
            }
        }
        self.ctr.add(&c);
    }

    /// [`Self::search_loop`] for the run's laziness and expander.
    #[expect(
        clippy::too_many_arguments,
        reason = "the loop's scratch and parameters"
    )]
    fn search(
        &mut self,
        two_hop: bool,
        terminator: Terminator,
        beam_width: usize,
        two_hop_cap: usize,
        frontier: &mut Vec<u32>,
        nbrs: &mut Vec<u32>,
        rejected: &mut Vec<Cand>,
        adaptive: &mut Adaptive,
        view: &mut usize,
    ) {
        let traced = self.trace.is_some();
        macro_rules! go {
            ($lz:expr, $th:expr) => {
                if traced {
                    self.search_loop::<{ $lz }, { $th }, true>(
                        terminator,
                        beam_width,
                        two_hop_cap,
                        frontier,
                        nbrs,
                        rejected,
                        adaptive,
                        view,
                    )
                } else {
                    self.search_loop::<{ $lz }, { $th }, false>(
                        terminator,
                        beam_width,
                        two_hop_cap,
                        frontier,
                        nbrs,
                        rejected,
                        adaptive,
                        view,
                    )
                }
            };
        }
        match (self.laziness, two_hop) {
            (Laziness::Eager, true) => go!(EAGER, true),
            (Laziness::Eager, false) => go!(EAGER, false),
            (Laziness::Deferred, _) => go!(DEFERRED, false),
            (Laziness::PredicateFirst, _) => go!(PREDICATE_FIRST, false),
            (Laziness::DistanceFirst, _) => go!(DISTANCE_FIRST, false),
            (Laziness::HeapBound, _) => go!(HEAP_BOUND, false),
        }
    }

    /// Upstream multihop's second hop over the step's rejected neighbours.
    #[inline(always)]
    fn two_hop<const TRACE: bool>(
        &mut self,
        rejected: &mut Vec<Cand>,
        cap: usize,
        c: &mut Counters,
    ) {
        rejected
            .sort_unstable_by(|a, b| a.d.partial_cmp(&b.d).unwrap_or(std::cmp::Ordering::Equal));
        if TRACE {
            if let Some(t) = self.trace.as_deref_mut() {
                trace_two_hop(t, rejected, cap);
            }
        }
        rejected.truncate(cap);
        let graph = self.graph;
        for cand in rejected.iter() {
            c.n_hops += 1;
            for &u in graph.neighbors(cand.id) {
                if u >= self.n_data || self.visited.contains(u) {
                    continue;
                }
                let p = self.pred(u, c);
                if !self.admits(p) {
                    continue;
                }
                self.visited.insert(u);
                let d = self.distance(u, None, c);
                let state = self.collect::<TRACE>(u, d, p, c);
                let key = self.sc.key(d, p.m, p.g);
                self.insert::<TRACE>(key, d, u, state, Some(cand.id), Some(p), "admitted", c);
                c.n_two_hop += 1;
            }
        }
    }
}

/// The graph search of one query (steps 3–6 of the per-query flow). With a guard: `None`
/// when the trial falls back (the answers, merged with the `witnesses`, stay in the heap
/// collector of `st`).
#[expect(clippy::too_many_arguments, reason = "forwards the per-query inputs")]
fn run<T: Element, P: Eval, S: Score>(
    st: &mut Searcher,
    plan: &Plan,
    ix: &Index<'_, T>,
    q: &Query<'_, T>,
    l: usize,
    reference: Option<u64>,
    ctr: &mut Counters,
    ids: &mut [u32],
    dists: &mut [f32],
    guard: Option<&mut Guard>,
    witnesses: &[u32],
) -> Option<usize> {
    let Searcher {
        beam32,
        beam64,
        visited,
        bits,
        compiled,
        mat,
        answers,
        starts,
        pilot,
        pilot_d,
        sorted,
        nbrs,
        rejected,
        frontier,
        trace,
        ..
    } = st;
    let kernel = &plan.kernel;
    let graph = ix.graph;
    let n_data = graph.n_data() as u32;
    let vectors = ix.vectors;
    let qv = q.vector;
    let k = ids.len();
    if let Some(t) = trace.as_mut() {
        t.clear();
    }

    // 3. Pilot: distances of the starts and of the global start's out-neighbours.
    pilot.clear();
    let mut pilot_starts = 0;
    let mut scales = Scales::default();
    if plan.pilot {
        for &s in starts.iter() {
            let d = T::squared_l2(qv, vectors.get(s));
            ctr.n_dist += 1;
            if s >= n_data {
                ctr.n_aux_dist += 1;
            }
            pilot.push((s, d));
        }
        pilot_starts = pilot.len();
        let g0 = ix.global[0];
        for &v in graph.neighbors(g0) {
            if !visited.test_and_set(v) {
                let d = T::squared_l2(qv, vectors.get(v));
                ctr.n_dist += 1;
                if v >= n_data {
                    ctr.n_aux_dist += 1;
                }
                pilot.push((v, d));
            }
        }
        pilot_d.clear();
        pilot_d.extend(pilot.iter().map(|&(_, d)| d));
        scales = Scales::from_pilot(pilot_d, ix.eps_min, sorted);
    }

    // 5. Materialization.
    match kernel.materializer {
        Materializer::Dense => {
            mat.build_dense(ix.labels, compiled, n_data as usize);
        }
        Materializer::DenseStatic => {
            let b = ix
                .bitsets
                .expect("a dense_static plan needs the static bitsets of the index");
            mat.build_dense_static(ix.labels, compiled, n_data as usize, b);
        }
        Materializer::Sparse => {
            mat.build_sparse(ix.labels, compiled, n_data as usize);
        }
        Materializer::None => {}
    }
    let mut cx = EvalCtx::new(ix.labels, compiled, bits, &mat.dense, &mat.sparse);
    if let (EvalMode::StaticBits, Some(b)) = (plan.eval, ix.bitsets) {
        cx = cx.with_static(b);
    }
    let sc = S::new(&plan.score_params(compiled, &scales));
    let mut beam = std::mem::take(S::beam(beam32, beam64));

    let n_starts = starts.len();
    let (capacity, mut view) = match (reference, kernel.padding) {
        (Some(_), _) => (l.max(n_starts).max(1), usize::MAX),
        (None, Padding::Upstream) => (l + n_starts, l),
        (None, Padding::Cxx) => (l.max(n_starts).max(1), usize::MAX),
        (None, Padding::Exact) => (l.max(1), usize::MAX),
    };
    beam.reset(capacity, capacity, reference.is_some(), S::STORE_D);
    answers.reset(kernel.collector, k, l);
    let terminator = match reference {
        Some(w) => Terminator::Budget(w),
        None => kernel.terminator,
    };
    let mut adaptive = Adaptive::new(terminator, l, reference.is_some());
    let two_hop = matches!(kernel.expander, Expander::TwoHop(_));
    let admission = match kernel.admission {
        Admission::All => AdmissionRule::All,
        Admission::Strict | Admission::Projection(_) => AdmissionRule::Match,
        Admission::Cutoff(t) => AdmissionRule::Cutoff(t),
    };

    let mut r = Run::<T, P, S> {
        cx,
        vectors,
        graph,
        qv,
        sc,
        laziness: kernel.laziness,
        admission,
        start_admission: kernel.start_admission,
        collector: kernel.collector,
        complete: plan.complete,
        projected: compiled.is_projected(),
        aux_g: kernel.aux_g,
        n_data,
        lazy_ok: !adaptive.sampling(),
        beam,
        visited: std::mem::take(visited),
        answers: std::mem::take(answers),
        ctr: *ctr,
        trace: trace.as_mut(),
        guard,
        _p: PhantomData,
    };

    // 4. Starts (with the pilot's distances).
    for (i, &s) in starts.iter().enumerate() {
        let cached = (i < pilot_starts).then(|| pilot[i].1);
        r.offer_start(s, cached);
    }
    // The pilot evaluated the global start's out-neighbours: offer them as its expansion.
    if pilot.len() > pilot_starts {
        let g0 = ix.global[0];
        r.beam.mark_expanded(g0);
        if let Some(t) = r.trace.as_deref_mut() {
            t.expand(g0, true);
        }
        r.ctr.n_expand += 1;
        r.ctr.n_hops += 1;
        if g0 >= n_data {
            r.ctr.n_aux_expand += 1;
        }
        rejected.clear();
        let mut c = Counters::default();
        for &(v, d) in &pilot[pilot_starts..] {
            let outcome = r.offer_any(v, g0, Some(d), &mut c);
            if let (true, Outcome::Rejected(d)) = (two_hop, outcome) {
                rejected.push(Cand { id: v, d });
            }
        }
        if adaptive.on {
            adaptive.visited += pilot.len() - pilot_starts;
            adaptive.matched += c.n_match_seen as usize;
        }
        r.ctr.add(&c);
        if two_hop {
            let mut c = Counters::default();
            r.two_hop::<true>(rejected, plan.two_hop_cap, &mut c);
            r.ctr.add(&c);
        }
        adaptive.end_of_step(&mut r.beam, &mut view);
        r.lazy_ok = !adaptive.sampling();
    }

    if let Some(g) = r.guard.as_deref_mut() {
        g.arm(&r.ctr);
    }
    // The loop.
    r.search(
        two_hop,
        terminator,
        kernel.beam_width,
        plan.two_hop_cap,
        frontier,
        nbrs,
        rejected,
        &mut adaptive,
        &mut view,
    );

    // 6. Collection.
    if kernel.collector == Collector::FinalBeam {
        let mut taken = 0;
        let cx = r.cx;
        for i in 0..r.beam.len() {
            let e = r.beam.entry(i);
            if e.state == State::Aux {
                continue;
            }
            if taken == view {
                break;
            }
            taken += 1;
            let ok = match e.state {
                State::Match => true,
                State::NoMatch | State::Aux => false,
                State::Partial | State::Unknown => {
                    r.ctr.n_pred += 1;
                    P::full(&cx, e.id)
                }
            };
            if ok {
                r.answers.offer(e.d, e.id);
            }
        }
    }
    r.ctr.final_l = r.beam.search_l() as u64;
    // A guarded trial at nominal termination: an incomplete heap, or a held-out sampled match
    // closer than the K-th answer (merged into the heap), falls back; so do the checkpoints.
    let mut fallback = false;
    if let Some(g) = r.guard.as_deref_mut() {
        if g.outcome == Fallback::None && r.answers.len() < k {
            g.outcome = Fallback::Incomplete;
        }
        let check = g.outcome == Fallback::None && g.witness;
        if check || g.outcome != Fallback::None {
            let mut deficit = false;
            for &w in witnesses {
                if r.answers.contains(w) {
                    continue;
                }
                let d = T::squared_l2(qv, vectors.get(w));
                r.ctr.n_dist += 1;
                deficit |= r.answers.offer_unique(d, w);
            }
            if check && deficit {
                g.outcome = Fallback::Witness;
            }
        }
        fallback = g.outcome != Fallback::None;
    }
    *ctr = r.ctr;
    let n = (!fallback).then(|| r.answers.finish(ids, dists));
    let Run {
        beam: b,
        visited: v,
        answers: a,
        ..
    } = r;
    *S::beam(beam32, beam64) = b;
    *visited = v;
    *answers = a;
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_l_matches_upstream() {
        assert_eq!(adaptive_l(100, 1000, 500, 16.0), 100);
        assert_eq!(adaptive_l(100, 1000, 100, 16.0), 200);
        assert_eq!(adaptive_l(100, 1000, 499, 16.0), 200);
        assert!(adaptive_l(100, 1000, 10, 16.0).abs_diff(400) <= 1);
        assert!(adaptive_l(100, 1000, 1, 16.0).abs_diff(800) <= 1);
        assert!(adaptive_l(100, 100, 0, 16.0).abs_diff(400) <= 1);
        assert_eq!(adaptive_l(100, 0, 0, 16.0), 1600);
        assert_eq!(adaptive_l(100, 1000, 1, 4.0), 400);
        assert_eq!(adaptive_l(100, 1000, 10, 1.5), 150);
    }

    #[test]
    fn pilot_scales() {
        let mut sorted = Vec::new();
        let s = Scales::from_pilot(&[4.0, 1.0, 9.0, 16.0], 0.5, &mut sorted);
        assert_eq!(s.s, 6.5);
        assert_eq!(s.eps, 0.5);
        assert_eq!(s.rho, 15.0);
        let z = Scales::from_pilot(&[0.0, 0.0, 0.0], 0.25, &mut sorted);
        assert_eq!((z.s, z.eps, z.rho), (0.25, 0.25, 0.0));
    }
}
