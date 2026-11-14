/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The per-query flow and the search loop (see the module documentation of [`super`]).

use std::marker::PhantomData;

use anyhow::{bail, ensure, Result};

use super::{
    beam::{Beam, Inserted, State},
    collect::{Answers, Collector},
    config::{Admission, Expander, Family, Kernel, Laziness, Padding, PredicateMode, Scorer},
    config::{StartAdmission, StartSource, Terminator},
    entry::{self, SampleIndex, StartPredicate},
    eval::{ClauseSearch, Compiled, DenseBitmap, Eval, EvalCtx, PredOut, RowScan, SparseList},
    exact,
    materialize::{Materialized, Materializer},
    score::{
        self, AdditiveScore, BonusScore, DistanceScore, LexicographicScore, MultiplicativeScore,
        Score,
    },
    trace::Trace,
    visited::Visited,
};
use crate::{
    graph::{Graph, NodeVectors},
    labels::LabelStore,
    predicate::Cnf,
    starts::Starts,
    vectors::Element,
};

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
}

impl Route {
    pub fn name(self) -> &'static str {
        match self {
            Self::Graph => "graph",
            Self::Exact => "exact",
            Self::Empty => "empty",
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
}

impl Counters {
    /// Names of the numeric counters, in [`Self::values`] order.
    pub const NAMES: [&'static str; 14] = [
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
    ];

    /// The numeric counters, in [`Self::NAMES`] order.
    pub fn values(&self) -> [u64; 14] {
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
}

/// A kernel configuration resolved against an index (validated once per run).
#[derive(Debug, Clone)]
pub struct Plan {
    pub kernel: Kernel,
    projection: Option<Vec<u32>>,
    two_hop_cap: usize,
    eval: EvalMode,
    family: Family,
    pilot: bool,
    complete: bool,
    n_aux: usize,
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
            (PredicateMode::Bitmap, Materializer::Dense) => EvalMode::Dense,
            (PredicateMode::Bitmap, _) => EvalMode::Sparse,
        };
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
        Ok(Self {
            kernel: kernel.clone(),
            projection,
            two_hop_cap,
            eval,
            family: kernel.scorer.family(),
            pilot: kernel.scorer.needs_pilot(),
            complete: kernel.needs_g(),
            n_aux: ix.graph.n_aux(),
        })
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
    beam: Beam,
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
            beam: Beam::new(),
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
            trace: None,
        }
    }

    /// Scratch sized for `ix` and the materializers of `plans`.
    pub fn for_index<T: Element>(ix: &Index<'_, T>, plans: &[&Plan]) -> Self {
        let dense = plans
            .iter()
            .any(|p| p.kernel.materializer == Materializer::Dense);
        Self::new(
            ix.graph.n_nodes(),
            ix.graph.n_data(),
            ix.labels.n_labels(),
            dense,
        )
    }

    /// The final beam of the last graph search (diagnostics).
    pub fn beam(&self) -> &Beam {
        &self.beam
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
        let exact_threshold = kernel.exact.threshold();
        let n = if self.compiled.is_empty() {
            ctr.route = Route::Empty;
            0
        } else if exact_threshold.is_some_and(|t| self.compiled.min_upper_bound() <= t) {
            ctr.route = Route::Exact;
            let cx = EvalCtx {
                labels: ix.labels,
                q: &self.compiled,
                bits: &self.bits,
                dense: &[],
                sparse: &[],
            };
            exact::run(
                kernel.exact,
                plan.eval == EvalMode::ClauseSearch,
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
            let cx = EvalCtx {
                labels: ix.labels,
                q: &self.compiled,
                bits: &self.bits,
                dense: &[],
                sparse: &[],
            };
            entry::select(
                &inputs,
                &cx,
                &mut self.visited,
                &mut self.starts,
                &mut self.heap,
                &mut self.pos,
                &mut ctr,
            );
            ctr.n_start = self.starts.len() as u64;
            dispatch(
                plan.eval,
                plan.family,
                self,
                plan,
                ix,
                q,
                l,
                reference,
                &mut ctr,
                ids,
                dists,
            )
        };
        self.compiled.clear_bits(&mut self.bits);
        ids[n..].fill(u32::MAX);
        dists[n..].fill(f32::MAX);
        ctr.n_out = n as u64;
        ctr
    }
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
) -> usize {
    macro_rules! by_family {
        ($p:ty) => {
            match family {
                Family::Distance => {
                    run::<T, $p, DistanceScore>(st, plan, ix, q, l, reference, ctr, ids, dists)
                }
                Family::Bonus => {
                    run::<T, $p, BonusScore>(st, plan, ix, q, l, reference, ctr, ids, dists)
                }
                Family::Additive => {
                    run::<T, $p, AdditiveScore>(st, plan, ix, q, l, reference, ctr, ids, dists)
                }
                Family::Multiplicative => run::<T, $p, MultiplicativeScore>(
                    st, plan, ix, q, l, reference, ctr, ids, dists,
                ),
                Family::Lexicographic => {
                    run::<T, $p, LexicographicScore>(st, plan, ix, q, l, reference, ctr, ids, dists)
                }
            }
        };
    }
    match eval {
        EvalMode::RowScan => by_family!(RowScan),
        EvalMode::ClauseSearch => by_family!(ClauseSearch),
        EvalMode::Dense => by_family!(DenseBitmap),
        EvalMode::Sparse => by_family!(SparseList),
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

    #[inline(always)]
    fn count(&mut self, m: bool) {
        self.visited += 1;
        self.matched += usize::from(m);
    }

    /// End of a step: resizes the beam when the sample count is reached (upstream).
    fn end_of_step(&mut self, beam: &mut Beam, view: &mut usize) {
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
    beam: &'r mut Beam,
    visited: &'r mut Visited,
    answers: &'r mut Answers,
    ctr: &'r mut Counters,
    trace: Option<&'r mut Trace>,
    _p: PhantomData<P>,
}

/// The admission rule with resolved parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdmissionRule {
    All,
    Match,
    Cutoff(u32),
}

impl<T: Element, P: Eval, S: Score> Run<'_, '_, T, P, S> {
    #[inline(always)]
    fn distance(&mut self, v: u32, cached: Option<f32>) -> f32 {
        match cached {
            Some(d) => d,
            None => {
                self.ctr.n_dist += 1;
                T::squared_l2(self.qv, self.vectors.get(v))
            }
        }
    }

    #[inline(always)]
    fn pred(&mut self, v: u32) -> PredOut {
        self.ctr.n_pred += 1;
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
    fn collect(&mut self, v: u32, d: f32, p: PredOut) -> State {
        if !p.m {
            return State::NoMatch;
        }
        self.ctr.n_match_seen += 1;
        if self.projected {
            if self.collector == Collector::FinalBeam {
                return State::Partial;
            }
            self.ctr.n_pred += 1;
            if !P::full(&self.cx, v) {
                return State::NoMatch;
            }
        }
        if self.collector != Collector::FinalBeam {
            let (inserted, evicted) = self.answers.offer(d, v);
            if inserted {
                if let Some(t) = self.trace.as_deref_mut() {
                    t.answer(v, d, evicted);
                }
            }
        }
        State::Match
    }

    #[inline(always)]
    #[expect(clippy::too_many_arguments, reason = "trace columns")]
    fn insert(
        &mut self,
        key: u64,
        d: f32,
        v: u32,
        state: State,
        source: Option<u32>,
        p: Option<PredOut>,
        outcome: &'static str,
    ) {
        let res = self.beam.insert(key, d, v, state);
        if let Inserted::At {
            evicted: Some(_), ..
        } = res
        {
            self.ctr.n_evict += 1;
        }
        if let Some(t) = self.trace.as_deref_mut() {
            let (rank, evicted) = match res {
                Inserted::Dropped => (None, None),
                Inserted::At { rank, evicted } => (Some(rank), evicted),
            };
            t.offer(
                v,
                source,
                Some(d),
                Some(self.sc.describe(key)),
                p.map(|p| p.m),
                p.filter(|_| self.complete).map(|p| p.g),
                outcome,
                rank,
                evicted,
            );
        }
    }

    #[inline(always)]
    fn trace_only(
        &mut self,
        v: u32,
        source: Option<u32>,
        d: Option<f32>,
        p: Option<PredOut>,
        outcome: &'static str,
    ) {
        if let Some(t) = self.trace.as_deref_mut() {
            t.offer(
                v,
                source,
                d,
                None,
                p.map(|p| p.m),
                p.filter(|_| self.complete).map(|p| p.g),
                outcome,
                None,
                None,
            );
        }
    }

    /// Places an evaluated node: collection, then admission to the beam.
    #[inline(always)]
    fn place(&mut self, v: u32, d: f32, p: PredOut, source: Option<u32>) -> Outcome {
        let state = self.collect(v, d, p);
        if self.admits(p) {
            let key = self.sc.key(d, p.m, p.g);
            self.insert(key, d, v, state, source, Some(p), "admitted");
            Outcome::Admitted
        } else {
            self.trace_only(v, source, Some(d), Some(p), "rejected");
            Outcome::Rejected(d)
        }
    }

    #[inline(always)]
    fn navigate_aux(&mut self, v: u32, source: Option<u32>, cached: Option<f32>) {
        let d = match cached {
            Some(d) => d,
            None => {
                self.ctr.n_aux_dist += 1;
                self.distance(v, None)
            }
        };
        let key = self.sc.key(d, false, self.aux_g);
        self.insert(key, d, v, State::Aux, source, None, "navigated");
    }

    /// Offers a newly visited node; returns the outcome and the navigation match bit.
    #[inline(always)]
    fn offer(&mut self, v: u32, source: u32, cached: Option<f32>) -> (Outcome, bool) {
        if v >= self.n_data {
            self.navigate_aux(v, Some(source), cached);
            return (Outcome::Navigated, false);
        }
        let src = Some(source);
        match self.laziness {
            Laziness::Deferred => {
                let d = self.distance(v, cached);
                let key = self.sc.key(d, false, 0.0);
                self.insert(key, d, v, State::Unknown, src, None, "admitted");
                (Outcome::Admitted, false)
            }
            Laziness::PredicateFirst => {
                let p = self.pred(v);
                if !self.admits(p) {
                    self.trace_only(v, src, None, Some(p), "rejected");
                    return (Outcome::Rejected(f32::NAN), p.m);
                }
                let d = self.distance(v, cached);
                (self.place(v, d, p, src), p.m)
            }
            Laziness::DistanceFirst => {
                let d = self.distance(v, cached);
                if self.lazy_ok && self.skip_safe(d) {
                    self.ctr.n_lazy += 1;
                    self.trace_only(v, src, Some(d), None, "lazily_skipped");
                    return (Outcome::Skipped, false);
                }
                let p = self.pred(v);
                (self.place(v, d, p, src), p.m)
            }
            Laziness::Eager => {
                let d = self.distance(v, cached);
                let p = self.pred(v);
                (self.place(v, d, p, src), p.m)
            }
        }
    }

    /// Offers a start (marked visited by the selection).
    fn offer_start(&mut self, s: u32, cached: Option<f32>) {
        if s >= self.n_data {
            self.navigate_aux(s, None, cached);
            return;
        }
        match (self.start_admission, self.laziness) {
            (_, Laziness::Deferred) => {
                let d = self.distance(s, cached);
                let key = self.sc.key(d, false, 0.0);
                self.insert(key, d, s, State::Unknown, None, None, "admitted");
            }
            (StartAdmission::Force, _) => {
                let p = self.pred(s);
                let d = self.distance(s, cached);
                let state = self.collect(s, d, p);
                let key = self.sc.key(d, p.m, p.g);
                let outcome = if self.admits(p) { "admitted" } else { "forced" };
                self.insert(key, d, s, state, None, Some(p), outcome);
            }
            (StartAdmission::Obey, Laziness::PredicateFirst) => {
                let p = self.pred(s);
                if !self.admits(p) {
                    self.trace_only(s, None, cached, Some(p), "rejected");
                    return;
                }
                let d = self.distance(s, cached);
                self.place(s, d, p, None);
            }
            (StartAdmission::Obey, _) => {
                let d = self.distance(s, cached);
                let p = self.pred(s);
                self.place(s, d, p, None);
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

    /// Expands node `n`: offers its unvisited neighbours.
    #[inline(always)]
    fn expand(
        &mut self,
        n: u32,
        nbrs: &mut Vec<u32>,
        rejected: &mut Vec<Cand>,
        two_hop: bool,
        adaptive: &mut Adaptive,
    ) {
        let graph = self.graph;
        self.ctr.n_hops += 1;
        if n >= self.n_data {
            self.ctr.n_aux_expand += 1;
        }
        let rows = P::ROWS && self.laziness != Laziness::Deferred;
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
        for i in 0..len {
            if i + la < len {
                self.prefetch(nbrs[i + la], rows);
            }
            let v = nbrs[i];
            let (outcome, m) = self.offer(v, n, None);
            if adaptive.on {
                adaptive.count(m);
            }
            if two_hop {
                if let Outcome::Rejected(d) = outcome {
                    rejected.push(Cand { id: v, d });
                }
            }
        }
    }

    /// Upstream multihop's second hop over the step's rejected neighbours.
    fn two_hop(&mut self, rejected: &mut Vec<Cand>, cap: usize) {
        rejected
            .sort_unstable_by(|a, b| a.d.partial_cmp(&b.d).unwrap_or(std::cmp::Ordering::Equal));
        if let Some(t) = self.trace.as_deref_mut() {
            for (rank, c) in rejected.iter().enumerate() {
                t.two_hop(c.id, c.d, rank, rank < cap);
            }
        }
        rejected.truncate(cap);
        let graph = self.graph;
        for c in rejected.iter() {
            self.ctr.n_hops += 1;
            for &u in graph.neighbors(c.id) {
                if u >= self.n_data || self.visited.contains(u) {
                    continue;
                }
                let p = self.pred(u);
                if !self.admits(p) {
                    continue;
                }
                self.visited.insert(u);
                let d = self.distance(u, None);
                let state = self.collect(u, d, p);
                let key = self.sc.key(d, p.m, p.g);
                self.insert(key, d, u, state, Some(c.id), Some(p), "admitted");
                self.ctr.n_two_hop += 1;
            }
        }
    }
}

/// The graph search of one query (steps 3–6 of the per-query flow).
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
) -> usize {
    let Searcher {
        beam,
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
        Materializer::Sparse => {
            mat.build_sparse(ix.labels, compiled, n_data as usize);
        }
        Materializer::None => {}
    }
    let cx = EvalCtx {
        labels: ix.labels,
        q: compiled,
        bits,
        dense: &mat.dense,
        sparse: &mat.sparse,
    };
    let sc = S::new(&plan.score_params(compiled, &scales));

    let n_starts = starts.len();
    let (capacity, mut view) = match (reference, kernel.padding) {
        (Some(_), _) => (l.max(n_starts).max(1), usize::MAX),
        (None, Padding::Upstream) => (l + n_starts, l),
        (None, Padding::Cxx) => (l.max(n_starts).max(1), usize::MAX),
    };
    beam.reset(capacity, capacity, reference.is_some());
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
        visited,
        answers,
        ctr,
        trace: trace.as_mut(),
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
        for &(v, d) in &pilot[pilot_starts..] {
            let (outcome, m) = r.offer(v, g0, Some(d));
            if adaptive.on {
                adaptive.count(m);
            }
            if let (true, Outcome::Rejected(d)) = (two_hop, outcome) {
                rejected.push(Cand { id: v, d });
            }
        }
        if two_hop {
            r.two_hop(rejected, plan.two_hop_cap);
        }
        adaptive.end_of_step(r.beam, &mut view);
        r.lazy_ok = !adaptive.sampling();
    }

    // The loop.
    loop {
        if !r.beam.has_unexpanded() {
            break;
        }
        match terminator {
            Terminator::Budget(w) => {
                if r.ctr.n_dist >= w {
                    break;
                }
            }
            Terminator::MatchAware => {
                if r.answers.is_full() {
                    if let Some(kmin) = r.beam.peek_key() {
                        if r.sc.dist_lower_bound(kmin) > r.answers.worst() {
                            break;
                        }
                    }
                }
            }
            Terminator::Exhaust | Terminator::AdaptiveL { .. } => {}
        }
        frontier.clear();
        while frontier.len() < kernel.beam_width {
            match r.beam.pop() {
                Some(e) => {
                    if let Some(t) = r.trace.as_deref_mut() {
                        t.expand(e.id, frontier.is_empty());
                    }
                    frontier.push(e.id);
                }
                None => break,
            }
        }
        r.ctr.n_expand += frontier.len() as u64;
        rejected.clear();
        for &n in frontier.iter() {
            r.expand(n, nbrs, rejected, two_hop, &mut adaptive);
        }
        if two_hop {
            r.two_hop(rejected, plan.two_hop_cap);
        }
        if adaptive.on {
            adaptive.end_of_step(r.beam, &mut view);
            r.lazy_ok = !adaptive.sampling();
        }
    }

    // 6. Collection.
    if kernel.collector == Collector::FinalBeam {
        let mut taken = 0;
        let cx = r.cx;
        for e in r.beam.entries() {
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
    r.answers.finish(ids, dists)
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
