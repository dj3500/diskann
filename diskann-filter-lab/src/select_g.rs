/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Track G's auxiliary structures for the unified selector ([`crate::select`]): a router tree
//! per label and a router tree per label pair ("pair tree", over the pair's joint posting),
//! chosen on the same workload, with the same unit costs, greedy and stop as the hybrid's
//! islands and combination postings — only the routes differ. Track G has no exact path: a
//! query's route without auxiliary structures is the graph router, whose post-filter route at
//! the query's match rate is its cost; a query whose binding clause `c*` (its smallest clause) is
//! a label with a tree takes the tree route, whatever it costs (`tree_choice = always`); a query
//! whose single-label clauses `{a}`, `{b}` have a pair tree takes the pair route.
//!
//! **Workload**: the selector's mixture ([`QueryModel`]): a sample of base-derived queries and,
//! with share `μ`, the independent queries `U` ([`crate::workload::Independent`], shared with the
//! other kinds): conjunctions of one to three single-label clauses of distinct categories (binding
//! clause: the smallest label; `M` = the joint size), with query vectors independent of the
//! labels.
//!
//! **Costs** (ns, the kernel's unit costs):
//!
//! - router: its post-filter route at the query's match rate `M/n` with the list that finds the
//!   `K`-th match at the reference recall `R` — the kernel's effort curve (`effort_list(R, k')`,
//!   the calibrated unfiltered list for recall `R` at `k'` neighbours) at the rank of the `K`-th
//!   match, `k' = K/rate + z·√(K(1 − rate))/rate` (the graph route is not modelled; without an
//!   effort curve, the islands' `postfilter_nominal`);
//! - tree route over `s` members of which `M` match: `E` router expansions, each checking the
//!   router's `b̄` children against the query's other clauses, computing a distance for the
//!   matching ones and one for the router: `E·(b̄·check + b̄·(M/s)·dist + graph_dist)`, the members'
//!   unit costs those of random points (a tree visits its members in geometric, not id, order:
//!   the static-bits member check and the scan's distance at the sparsest calibrated density),
//!   the router's a graph node's. `E = min(s/(b̄ − 1) + 1, F·s/M)`: the reference frontier `F`
//!   (the declared frontier table's at the reference recall `R` = 0.9: 512, the frontier the
//!   route runs with) scaled by the members per match, so that a diluted clause is charged the
//!   search that reaches that recall on it, at most the whole tree. Optionally (`curve`), `E` is
//!   the calibrated **tree effort curve** instead (`flab tree-calibrate`: the router frontier for
//!   recall `R` at `k'` members of a label, per label-size class) at the rank of the `K`-th
//!   match among the members, `k' = K·s/M` with its spread;
//! - pair tree on a pair of joint `j`: the tree route over its `j` members, all matching (its
//!   saving: against the router, as the selector's pair kinds; on its pair context against the
//!   context's current cost, shared with the label trees through [`LabelKind`]).
//!
//! **Kernel prices** (`flab select --pricing kernel`, the default with fixed label trees,
//! [`KernelRoutes`]): instead, every route at the price the kernel's cost-checked pair route
//! (`tree_pairs = "on"`) gives it at the reference level — a tree over `s` members of which `M`
//! match by its frontier at that dilution (as the run's `tree_effort = "fixed(F)"`), a query no
//! tree serves by the graph-only crossover's estimate — so that selection and routing share one
//! cost model: a pair tree saves only where the kernel would route a query to it (cheaper than
//! the query's label tree, or than its graph route when it has none), and a query whose
//! binding label has a tree takes a pair tree only with fewer members than that label.
//!
//! **Calibrated costs** (`flab select --route-curves`, [`RouteCurves`]): instead of the model
//! above, both routes are priced from recall/time curves measured per label-size class on
//! calibration queries drawn from the base (a single label with an independent query point, with
//! a member as the query point, and label pairs by joint size), each at the time its class's
//! Pareto frontier reaches the reference recall `R` (interpolated between classes in log size);
//! a route whose class never reaches `R` costs [`UNREACHABLE_NS`], so a label the router cannot
//! serve gets a must-have tree. A base-derived query is priced by the member curves (the router
//! of a conjunction by the member-placed pair curves `Q` where measured, by its match count; the
//! tree at `K·s/M` members), an independent conjunction by the curves of its clause count (single
//! labels `U`, pairs `P`, triples `T`; the router, and the tree of its smallest label), and a pair
//! tree by the independent tree curve at its joint size diluted to the query's matches on an
//! independent query, by the member curves on base-derived queries.
//!
//! **Bytes**: a tree over `s` members has `⌈(s − 1)/(b̄ − 1)⌉` routers (at least one), each a
//! graph row, a vector and a kind byte (`router_bytes`; the `rows` accounting) or, as a run's
//! bytes are counted (`compact`, [`TreeModel::compact_vector_bytes`]), its adjacency list — 4 B
//! per node and per edge, `s + R − 1` edges for `R` routers — and its vector; a pair tree adds
//! its pair's list and directory entry ([`crate::select::pair_bytes`]) and (`compact`) its root
//! entry; the pairs' co-occurrence matrix is charged by the selector as for combination postings.
//!
//! **Values**: a label tree serves exactly the queries whose `c*` is that label (a clause of
//! several labels needs a tree for each of them, which this kind does not value), so label trees
//! do not interact: a tree's expected saving per query is its queries' signed savings (router −
//! tree route) — the base-derived ones from the sample (or in closed form by size class where the
//! sample is thin), its independent queries' exactly. Pair trees are valued as the selector's
//! pair kinds ([`crate::select::pair_values`]): on base-derived queries in closed form, on every
//! independent query holding both labels against its current cost; a pair tree taken serves those
//! queries (the kernel takes the pair route first, the cheapest of several), which then no longer
//! count for the label tree.

use std::sync::Arc;

use crate::{
    islands::{draw_query, postfilter_nominal, query_matches, QueryModel},
    kernel::{
        config::{Laziness, PostfilterStarts},
        cost::{kth_match_rank, UnitCosts, Z},
        search::{fixed_tree_route_ns, graph_only_route_ns},
    },
    labels::LabelStore,
    select::{context_prob, pair_bytes, LabelKind, PairCand, PairValue},
    workload::{Independent, WQuery},
};

/// Cost and size terms of the router trees (see the module documentation).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct TreeModel {
    /// Answers `K`.
    pub k: usize,
    /// Reference recall `R` of both routes' costs.
    pub recall: f64,
    /// Reference tree frontier `F` (routers expanded at the reference recall).
    pub frontier: f64,
    /// Expansions from the calibration's tree effort curve (`flab tree-calibrate`) instead of
    /// the reference frontier.
    pub curve: bool,
    /// Mean router fan-out `b̄` of the trees (k-means trees of fan-out bound 64: 20–28).
    pub fanout: f64,
    /// Bytes of one router (graph row, vector, kind): the `rows` accounting.
    pub router_bytes: u64,
    /// The `compact` accounting, as a run's bytes are counted (the memory ceiling's base): the
    /// bytes of a router's vector. A tree over `s` members with `R` routers then counts its
    /// adjacency lists, 4 B per node and per edge (`R` nodes; `s + R − 1` edges, one to each
    /// member and to each router below the root), and its routers' vectors; a pair tree also
    /// its root entry. `None`: the `rows` accounting, `R · router_bytes`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compact_vector_bytes: Option<u64>,
}

impl TreeModel {
    /// The graph router's cost (its post-filter route) for `m` matches among `n` points.
    pub fn router_ns(&self, c: &UnitCosts, m: f64, n: usize) -> f64 {
        let rate = (m / n.max(1) as f64).min(1.0);
        let rank = kth_match_rank(self.k, rate, Z);
        if !rank.is_finite() {
            return f64::INFINITY;
        }
        match c.effort_list(self.recall, rank) {
            Some(l) => {
                let l = l.max(self.k as f64).min(n as f64);
                c.postfilter_cost(l, self.k, rate) + c.graph_dist_ns
            }
            None => postfilter_nominal(c, self.k, self.k, rate, n, 1).1,
        }
    }

    /// Routers the tree route expands over `s` members of which `m` match: the reference
    /// frontier scaled by the members per match (with [`Self::curve`], the calibrated tree
    /// effort curve at the reference recall for the rank of the `K`-th match among the
    /// members); at most the whole tree.
    pub fn tree_expansions(&self, c: &UnitCosts, s: f64, m: f64) -> f64 {
        let s = s.max(1.0);
        let m = m.clamp(1.0, s);
        let b = self.fanout.max(2.0);
        let whole = s / (b - 1.0) + 1.0;
        let rank = kth_match_rank(self.k, m / s, Z).min(s);
        let curve = if self.curve && !c.tree_r.is_empty() && rank.is_finite() {
            c.tree_list(self.recall, rank.max(1.0), s)
        } else {
            None
        };
        curve.unwrap_or(self.frontier * s / m).min(whole)
    }

    /// The tree route's cost over `s` members of which `m` match; `checks`: the query has
    /// clauses besides the tree's label, checked on every member.
    pub fn tree_ns(&self, c: &UnitCosts, s: f64, m: f64, checks: bool) -> f64 {
        let s = s.max(1.0);
        let m = m.clamp(1.0, s);
        let b = self.fanout.max(2.0);
        let check = if checks {
            c.pred_member(false, 0.0)
        } else {
            0.0
        };
        self.tree_expansions(c, s, m) * (b * check + b * (m / s) * c.dist(0.0) + c.graph_dist_ns)
    }

    /// Routers of a tree over `s` members.
    pub fn routers(&self, s: usize) -> u64 {
        let b = self.fanout.max(2.0);
        ((s.saturating_sub(1)) as f64 / (b - 1.0)).ceil().max(1.0) as u64
    }

    /// Bytes of a tree over `s` members.
    pub fn tree_bytes(&self, s: usize) -> u64 {
        let r = self.routers(s);
        match self.compact_vector_bytes {
            Some(v) => r * (4 + v) + 4 * (s.max(1) as u64 + r - 1),
            None => r * self.router_bytes,
        }
    }

    /// Bytes of a pair tree over a stored pair of `joint` points: the pair's list and directory
    /// entry ([`pair_bytes`]), its tree and, in the `compact` accounting, its root entry.
    pub fn pair_tree_bytes(&self, joint: u32) -> u64 {
        let root = if self.compact_vector_bytes.is_some() {
            4
        } else {
            0
        };
        pair_bytes(joint) + self.tree_bytes(joint as usize) + root
    }

    /// Bytes of one router of a graph with degree bound `r_max` and vectors of `vector_bytes`.
    pub fn router_bytes_of(r_max: usize, vector_bytes: usize) -> u64 {
        (4 + 4 * r_max + vector_bytes + 1) as u64
    }
}

/// A route's price for queries whose clause holds more than its measured classes allow, or that it
/// cannot answer at the reference recall: a must-have tree (ns).
pub const UNREACHABLE_NS: f64 = 1e9;

/// One label-size class of a route calibration: the classes' representative size (geometric mean
/// of the sampled labels' sizes, or of the pairs' joint sizes) and, per route and `K` (`"10"`,
/// `"100"`), `(mean recall, mean ns)` at each frontier level.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct CurveClass {
    pub size: f64,
    pub router: std::collections::BTreeMap<String, Vec<[f64; 2]>>,
    pub tree: std::collections::BTreeMap<String, Vec<[f64; 2]>>,
}

/// Track G's two routes measured per label-size class on calibration queries (B2i): `U` a single
/// label with an independent query point, `D` a single label with a member as the query point,
/// `P` a cross-category label pair (classes by joint size) with an independent query point, `Q`
/// the same pairs with a member of the joint as the query point; the router alone on the base
/// graph, the tree route on a tree per sampled label (for a pair, its smaller label's).
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct RouteCurves {
    #[serde(rename = "U")]
    pub u: Vec<CurveClass>,
    #[serde(rename = "D")]
    pub d: Vec<CurveClass>,
    #[serde(rename = "P", default)]
    pub p: Vec<CurveClass>,
    #[serde(rename = "Q", default)]
    pub q: Vec<CurveClass>,
    /// Three single-label clauses of distinct categories (classes by joint size), an independent
    /// query point; the tree route on the smallest label's tree.
    #[serde(rename = "T", default)]
    pub t: Vec<CurveClass>,
}

/// Which route of a [`CurveClass`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    Router,
    Tree,
}

/// The time (ns) at which a route's measured `(recall, ns)` points reach mean recall `r`: on their
/// Pareto frontier (cheapest first, recall rising), linearly between the last point below `r` and
/// the first at or above it; `None` if no point reaches `r`.
pub fn curve_price(points: &[[f64; 2]], r: f64) -> Option<f64> {
    let mut pts: Vec<[f64; 2]> = points
        .iter()
        .copied()
        .filter(|p| p[0].is_finite() && p[1].is_finite())
        .collect();
    pts.sort_by(|a, b| a[1].total_cmp(&b[1]));
    let mut frontier: Vec<[f64; 2]> = Vec::new();
    for p in pts {
        if frontier.last().is_none_or(|q| p[0] > q[0]) {
            frontier.push(p);
        }
    }
    let i = frontier.iter().position(|p| p[0] >= r)?;
    if i == 0 {
        return Some(frontier[0][1]);
    }
    let (a, b) = (frontier[i - 1], frontier[i]);
    let t = (r - a[0]) / (b[0] - a[0]);
    Some(a[1] + t * (b[1] - a[1]))
}

impl RouteCurves {
    /// A route's price at recall `r` and `K = k` for a clause of `size` points: log-linear in the
    /// size between the measured classes around it (clamped to the smallest and largest); a class
    /// that cannot reach `r` decides when it is the nearer one.
    pub fn price(classes: &[CurveClass], route: Route, k: usize, size: f64, r: f64) -> Option<f64> {
        Self::price_with(classes, route, k, size, r, false)
    }

    /// [`Self::price`]; with `at_max`, a class that cannot reach `r` is priced at its time at the
    /// highest measured level.
    pub fn price_with(
        classes: &[CurveClass],
        route: Route,
        k: usize,
        size: f64,
        r: f64,
        at_max: bool,
    ) -> Option<f64> {
        if classes.is_empty() {
            return None;
        }
        let key = k.to_string();
        let one = |c: &CurveClass| {
            let m = match route {
                Route::Router => &c.router,
                Route::Tree => &c.tree,
            };
            m.get(&key).and_then(|pts| {
                curve_price(pts, r).or_else(|| {
                    at_max
                        .then(|| {
                            pts.iter()
                                .map(|p| p[1])
                                .filter(|t| t.is_finite())
                                .reduce(f64::max)
                        })
                        .flatten()
                })
            })
        };
        let size = size.max(1.0);
        let i = classes.partition_point(|c| c.size <= size);
        if i == 0 {
            return one(&classes[0]);
        }
        if i == classes.len() {
            return one(&classes[classes.len() - 1]);
        }
        let (lo, hi) = (&classes[i - 1], &classes[i]);
        let t = (size.ln() - lo.size.ln()) / (hi.size.ln() - lo.size.ln());
        match (one(lo), one(hi)) {
            (Some(a), Some(b)) => Some(((1.0 - t) * a.ln() + t * b.ln()).exp()),
            (a, b) => {
                if t < 0.5 {
                    a
                } else {
                    b
                }
            }
        }
    }

    /// The tree route's time over all of its members (the largest frontier's, `K` = 100) for a
    /// label of `size` points, from the nearest class at or above it (the cap of a diluted clause's
    /// price).
    fn whole_tree(classes: &[CurveClass], size: f64) -> Option<f64> {
        if classes.is_empty() {
            return None;
        }
        let i = classes
            .partition_point(|c| c.size <= size.max(1.0))
            .min(classes.len() - 1);
        classes[i]
            .tree
            .get("100")
            .and_then(|v| v.iter().map(|p| p[1]).reduce(f64::max))
    }
}

/// The kernel's route prices (`flab select --track graph --pricing kernel`, the default with
/// fixed label trees): every route priced as the cost-checked pair route (`tree_pairs = "on"`)
/// prices it at the reference level — a tree over `s` members of which `M` match by its
/// frontier at that dilution (`tree_effort = "fixed(F)"`, `F` the reference frontier), a query
/// no tree serves by the graph-only crossover's estimate (the filtered graph search at the list
/// `effort_curve(cap)` plans for the reference recall, or the post-filter route when cheaper).
/// One cost model for selection and routing: a pair tree is worth what it saves on the queries
/// the kernel would route to it, and nothing on the others.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct KernelRoutes {
    /// Target recall of the graph route's planned list, per mille (the reference recall).
    pub effort: usize,
    /// `terminator = "effort_curve(cap)"`.
    pub cap: f64,
    /// Start nodes of the graph route of a query no tree serves (`starts =
    /// "label(c*)+global"`: its binding label's starts and the global ones).
    pub starts: usize,
    /// The global start nodes (a broad query's post-filter route, `postfilter_starts`).
    pub global_starts: usize,
    #[serde(skip)]
    pub laziness: Laziness,
    #[serde(skip)]
    pub postfilter_starts: PostfilterStarts,
}

impl KernelRoutes {
    /// Track G's run keys at the reference recall `recall`: `terminator =
    /// "effort_curve(4)"`, `laziness = "heap_bound"`, `starts = "label(c*)+global"` (a label
    /// without a tree has the start table's 4 k-means starts; one global start) and
    /// `postfilter_starts = "by_rate_both_k"`.
    pub fn recipe(recall: f64) -> Self {
        Self {
            effort: (1000.0 * recall).round() as usize,
            cap: 4.0,
            starts: 5,
            global_starts: 1,
            laziness: Laziness::HeapBound,
            postfilter_starts: PostfilterStarts::ByRateBothK,
        }
    }
}

/// How the selector prices track G's routes: from route calibration curves (B2i) when given, else
/// by the analytic [`TreeModel`] (B2h), or as the kernel's cost-checked routes ([`KernelRoutes`]).
#[derive(Debug, Clone, Copy)]
pub struct Pricing<'a> {
    pub tm: TreeModel,
    pub c: &'a UnitCosts,
    pub curves: Option<&'a RouteCurves>,
    /// Data points.
    pub n: usize,
    /// A route that never reaches the reference recall is priced at its time at the highest
    /// measured level (the effort it spends trying) instead of [`UNREACHABLE_NS`].
    pub at_max: bool,
    /// Every route at the kernel's prices instead (the curves and the analytic model unused).
    pub kernel: Option<KernelRoutes>,
}

impl Pricing<'_> {
    fn priced(&self, classes: &[CurveClass], route: Route, k: usize, size: f64) -> f64 {
        RouteCurves::price_with(classes, route, k, size, self.tm.recall, self.at_max)
            .unwrap_or(UNREACHABLE_NS)
    }

    /// [`KernelRoutes`]: a tree over `s` members of which `m` match.
    fn kernel_tree(&self, s: f64, m: f64) -> f64 {
        fixed_tree_route_ns(self.c, self.tm.fanout, self.tm.frontier, s, m)
    }

    /// [`KernelRoutes`]: a query with `m` matches that no tree serves.
    fn kernel_graph(&self, kr: &KernelRoutes, m: f64) -> f64 {
        let rate = (m / self.n.max(1) as f64).clamp(0.0, 1.0);
        graph_only_route_ns(
            self.c,
            kr.effort,
            self.tm.k,
            rate,
            self.n,
            kr.cap,
            kr.laziness,
            kr.starts,
            kr.global_starts,
            kr.postfilter_starts,
        )
    }

    /// A base-derived (data-like) query with `m` matches: the router — for a conjunction
    /// (`multi`: more than one clause) by the member-placed pair curves where measured, else by
    /// the single labels'.
    pub fn router_derived(&self, m: f64, multi: bool) -> f64 {
        if let Some(kr) = &self.kernel {
            return self.kernel_graph(kr, m);
        }
        match self.curves {
            Some(cv) if multi && !cv.q.is_empty() => {
                self.priced(&cv.q, Route::Router, self.tm.k, m)
            }
            Some(cv) => self.priced(&cv.d, Route::Router, self.tm.k, m),
            None => self.tm.router_ns(self.c, m, self.n),
        }
    }

    /// A base-derived query whose binding clause is a label of `s` points, `m` of them matching:
    /// the label's tree (the clause diluted to `K' = K·s/m` members; `checks`: other clauses).
    pub fn tree_derived(&self, s: f64, m: f64, checks: bool) -> f64 {
        if self.kernel.is_some() {
            return self.kernel_tree(s, m);
        }
        match self.curves {
            Some(cv) => self.diluted(&cv.d, s, m),
            None => self.tm.tree_ns(self.c, s, m, checks),
        }
    }

    /// The single-label query `{ℓ}` (`s` points) placed independently of the label: the router.
    pub fn router_label(&self, s: f64) -> f64 {
        if let Some(kr) = &self.kernel {
            return self.kernel_graph(kr, s);
        }
        match self.curves {
            Some(cv) => self.priced(&cv.u, Route::Router, self.tm.k, s),
            None => self.tm.router_ns(self.c, s, self.n),
        }
    }

    /// The single-label query `{ℓ}` placed independently of the label: its tree.
    pub fn tree_label(&self, s: f64) -> f64 {
        if self.kernel.is_some() {
            return self.kernel_tree(s, s);
        }
        match self.curves {
            Some(cv) => self.priced(&cv.u, Route::Tree, self.tm.k, s),
            None => self.tm.tree_ns(self.c, s, s, false),
        }
    }

    /// The pair query `{a} ∧ {b}` of joint `j` placed independently: the router.
    pub fn router_pair(&self, j: f64) -> f64 {
        if let Some(kr) = &self.kernel {
            return self.kernel_graph(kr, j);
        }
        match self.curves {
            Some(cv) if !cv.p.is_empty() => self.priced(&cv.p, Route::Router, self.tm.k, j),
            _ => self.tm.router_ns(self.c, j, self.n),
        }
    }

    /// The pair query of joint `j` whose smaller label has `s` points: that label's tree.
    pub fn tree_pair_label(&self, s: f64, j: f64) -> f64 {
        if self.kernel.is_some() {
            return self.kernel_tree(s, j);
        }
        match self.curves {
            Some(cv) if !cv.p.is_empty() => self.priced(&cv.p, Route::Tree, self.tm.k, j),
            _ => self.tm.tree_ns(self.c, s, j, true),
        }
    }

    /// An independent conjunction of `c` single-label clauses with `m` matches: the router (the
    /// single labels' curves for one clause, the pairs' for two, the triples' for three or more
    /// where measured, else the pairs').
    pub fn router_conj(&self, c: usize, m: f64) -> f64 {
        if let Some(kr) = &self.kernel {
            return self.kernel_graph(kr, m);
        }
        match (c, self.curves) {
            (0 | 1, _) => self.router_label(m),
            (2, _) | (_, None) => self.router_pair(m),
            (_, Some(cv)) if cv.t.is_empty() => self.router_pair(m),
            (_, Some(cv)) => self.priced(&cv.t, Route::Router, self.tm.k, m),
        }
    }

    /// An independent conjunction of `c` single-label clauses with `m` matches, its smallest
    /// clause a label of `s` points: that label's tree (as [`Self::router_conj`] by clause count).
    pub fn tree_conj(&self, c: usize, s: f64, m: f64) -> f64 {
        if self.kernel.is_some() {
            return self.kernel_tree(s, m);
        }
        match (c, self.curves) {
            (0 | 1, _) => self.tree_label(s),
            (2, _) | (_, None) => self.tree_pair_label(s, m),
            (_, Some(cv)) if cv.t.is_empty() => self.tree_pair_label(s, m),
            (_, Some(cv)) => self.priced(&cv.t, Route::Tree, self.tm.k, m),
        }
    }

    /// An independent conjunction with `m` matches served by a pair tree over `j ≥ m` joint
    /// members (its other clauses checked): the independent tree curve at `j`, diluted to
    /// `K' = K·j/m` members (between the `K` = 10 and 100 curves, beyond ∝ `K'` up to the whole
    /// tree).
    pub fn pair_tree_on(&self, j: f64, m: f64) -> f64 {
        if self.kernel.is_some() {
            return self.kernel_tree(j, m);
        }
        let Some(cv) = self.curves else {
            return self.tm.tree_ns(self.c, j, m, true);
        };
        self.diluted(&cv.u, j, m)
    }

    /// A tree over `s` members of which `m` match, priced by `classes` at `K' = K·s/m` members.
    fn diluted(&self, classes: &[CurveClass], s: f64, m: f64) -> f64 {
        let k = self.tm.k as f64;
        let kp = k * s / m.clamp(1.0, s.max(1.0));
        let at = |kk: usize| self.priced(classes, Route::Tree, kk, s);
        if kp <= k {
            return at(self.tm.k);
        }
        let (lo, hi) = (at(self.tm.k), at(100));
        if lo >= UNREACHABLE_NS || hi >= UNREACHABLE_NS {
            return UNREACHABLE_NS;
        }
        if kp <= 100.0 {
            let t = (kp.ln() - k.ln()) / (100f64.ln() - k.ln());
            return ((1.0 - t) * lo.ln() + t * hi.ln()).exp();
        }
        let whole = RouteCurves::whole_tree(classes, s)
            .unwrap_or(f64::INFINITY)
            .max(hi);
        (hi * kp / 100.0).min(whole)
    }

    /// The pair query of joint `j`: its pair tree (a tree over the `j` joint members).
    pub fn pair_tree(&self, j: f64) -> f64 {
        if self.kernel.is_some() {
            return self.kernel_tree(j, j);
        }
        match self.curves {
            Some(cv) => self.priced(&cv.u, Route::Tree, self.tm.k, j),
            None => self.tm.tree_ns(self.c, j, j, false),
        }
    }

    /// A base-derived query `{a} ∧ {b}` (a member of the joint as the query point) on the pair
    /// tree of joint `j` (the member curves; without curves, [`Self::pair_tree`]).
    pub fn pair_tree_derived(&self, j: f64) -> f64 {
        match self.curves {
            Some(cv) if self.kernel.is_none() => self.priced(&cv.d, Route::Tree, self.tm.k, j),
            _ => self.pair_tree(j),
        }
    }
}

/// The label-tree terms of the workload, as the selector's [`crate::islands::QuerySample`] (the
/// same layout and weights): its `n_data` base-derived queries (a sample, weight `(1 − μ)/n_data`
/// each), then the independent queries `U` ([`Independent`], shared with the other kinds; weight
/// `μ` times each query's probability) — per query its router cost and, when its binding clause
/// is a single label (for a conjunction of single labels, its smallest), that label's tree route
/// cost.
#[derive(Debug, Clone)]
pub struct TreeSample {
    pub model: QueryModel,
    /// Router cost of each query: the base-derived queries, then the independent queries in the
    /// order of [`Independent::queries`].
    pub base_ns: Vec<f64>,
    /// `(label, query, tree route cost)` of the queries whose binding clause is that label, by
    /// label and query.
    recs: Vec<(u32, u32, f32)>,
    /// Base-derived queries (the first `n_data` of `base_ns`).
    pub n_data: usize,
    independent: Option<Arc<Independent>>,
}

impl TreeSample {
    /// Draws the base-derived queries and computes their cost terms (with up to `threads`
    /// threads; the result does not depend on the thread count). Without the independent part
    /// (see [`Self::build_mixed`]).
    pub fn build(labels: &LabelStore, pr: &Pricing, model: QueryModel, threads: usize) -> Self {
        let model = QueryModel {
            uniform: 0.0,
            ..model
        };
        Self::build_mixed(labels, pr, model, None, threads)
    }

    /// [`Self::build`] with the workload's independent queries ([`QueryModel::uniform`]; `U` as
    /// [`crate::workload::independent`] gives it): a query of `c` single-label clauses with `M`
    /// matches costs the router [`Pricing::router_conj`] and, on its smallest label's tree,
    /// [`Pricing::tree_conj`].
    pub fn build_mixed(
        labels: &LabelStore,
        pr: &Pricing,
        model: QueryModel,
        independent: Option<Arc<Independent>>,
        threads: usize,
    ) -> Self {
        let independent = independent.filter(|u| model.uniform > 0.0 && !u.is_empty());
        let u: &[WQuery] = independent.as_deref().map_or(&[], |u| &u.queries);
        let n = labels.n_points();
        let chunk = 1024usize;
        let n_total = model.n_queries + u.len();
        let init = || {
            (
                Vec::<u32>::new(),
                Vec::<(usize, usize)>::new(),
                Vec::<(usize, usize, bool)>::new(),
            )
        };
        let parts = crate::parallel::par_map(
            n_total.div_ceil(chunk),
            threads,
            init,
            |ci, (kl, clauses, cats)| {
                let mut base = Vec::new();
                let mut recs = Vec::new();
                for qi in ci * chunk..((ci + 1) * chunk).min(n_total) {
                    if qi >= model.n_queries {
                        let w = &u[qi - model.n_queries];
                        let m = w.joint.max(1.0);
                        let c = w.labels.len();
                        base.push(pr.router_conj(c, m));
                        let l = w
                            .labels
                            .iter()
                            .copied()
                            .min_by_key(|&l| (labels.posting_len(l), l));
                        if let Some(l) = l {
                            let s = labels.posting_len(l);
                            if s > 0 && s < n {
                                let t = pr.tree_conj(c, s as f64, m);
                                recs.push((l, qi as u32, t as f32));
                            }
                        }
                        continue;
                    }
                    if !draw_query(labels, &model, qi, kl, clauses, cats) {
                        base.push(0.0);
                        continue;
                    }
                    let (jmin, _, m) = query_matches(labels, &model, kl, clauses);
                    base.push(pr.router_derived(m, clauses.len() > 1));
                    let (a, b) = clauses[jmin];
                    let mut nonempty = kl[a..b].iter().filter(|&&l| labels.posting_len(l) > 0);
                    if let (Some(&l), None) = (nonempty.next(), nonempty.next()) {
                        let s = labels.posting_len(l);
                        if s < n {
                            let t = pr.tree_derived(s as f64, m, clauses.len() > 1);
                            recs.push((l, qi as u32, t as f32));
                        }
                    }
                }
                (base, recs)
            },
        );
        let mut base_ns = Vec::with_capacity(n_total);
        let mut recs = Vec::new();
        for (b, r) in parts {
            base_ns.extend(b);
            recs.extend(r);
        }
        recs.sort_unstable_by_key(|&(l, q, _)| (l, q));
        Self {
            model,
            base_ns,
            recs,
            n_data: model.n_queries,
            independent,
        }
    }

    /// Share `μ` of the independent queries (0 without them).
    fn mu(&self) -> f64 {
        if self.independent.is_some() {
            self.model.uniform
        } else {
            0.0
        }
    }

    /// The independent queries `U` and the sample index of the first (`None` without them).
    pub fn independent(&self) -> Option<(&Independent, usize)> {
        self.independent.as_deref().map(|u| (u, self.n_data))
    }

    /// Independent query `q` of the sample (`None` for a base-derived one).
    fn independent_query(&self, q: usize) -> Option<&WQuery> {
        self.independent
            .as_deref()?
            .queries
            .get(q.checked_sub(self.n_data)?)
    }

    /// Workload weight of query `q`.
    pub fn weight(&self, q: usize) -> f64 {
        if q < self.n_data {
            (1.0 - self.mu()) / self.n_data.max(1) as f64
        } else {
            self.mu() * self.independent_query(q).map_or(0.0, |w| w.weight)
        }
    }

    /// The workload's mean of per-query values `x` (its parts weighted).
    pub fn weighted_mean(&self, x: &[f64]) -> f64 {
        x.iter().enumerate().map(|(q, v)| self.weight(q) * v).sum()
    }

    /// Expected cost per query (ns) without auxiliary structures (the router alone).
    pub fn mean_base_ns(&self) -> f64 {
        self.weighted_mean(&self.base_ns)
    }

    /// Means of per-query values `x`: of the base-derived queries, and of the independent
    /// queries of one, two and three labels (weighted; 0 for an empty group).
    pub fn part_means(&self, x: &[f64]) -> (f64, [f64; 3]) {
        let (d, u) = x.split_at(self.n_data.min(x.len()));
        let md = d.iter().sum::<f64>() / self.n_data.max(1) as f64;
        let mu = self
            .independent
            .as_deref()
            .map_or([0.0; 3], |ind| ind.means_by_size(u));
        (md, mu)
    }
}

/// One label of the tree kind.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct TreeTerm {
    pub label: u32,
    pub size: usize,
    /// Sampled base-derived queries whose binding clause is this label.
    pub queries: u32,
    /// Pair contexts whose binding clause is this label.
    pub contexts: u32,
    /// Expected saving per query (ns) on the base-derived queries (signed: queries the router
    /// serves better count against it).
    pub derived_ns: f64,
    /// Expected saving per query (ns) of this tree (signed), on its pair contexts not served by a
    /// pair tree: at the start, and when taken.
    pub saving_ns: f64,
    pub bytes: u64,
    /// Position in the greedy order (`None`: not taken).
    pub order: Option<u32>,
}

/// `f64` with a total order (the greedy's heap).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Key(f64);

impl Eq for Key {}

impl PartialOrd for Key {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

impl Ord for Key {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0)
    }
}

/// The tree kind as the selector's greedy sees it: candidates by expected saving per query per
/// byte. A tree's saving on the base-derived queries is fixed; on a pair context it counts only
/// while no pair tree serves the context (the kernel takes the pair route first), so a pair tree
/// taken lowers the savings of the tree of its smaller label (lazy re-evaluation; a signed saving
/// can also rise, which the lazy order sees only when the candidate comes up again).
#[derive(Debug, Clone)]
pub struct TreeGreedy<'s> {
    sample: &'s TreeSample,
    pr: Pricing<'s>,
    pub terms: Vec<TreeTerm>,
    /// Per candidate, its independent queries' records' range in `ctx_recs`.
    ctx: Vec<(usize, usize)>,
    /// `(query, tree route cost)` of the candidates' independent queries.
    ctx_recs: Vec<(u32, f32)>,
    /// Per independent query: the candidate of its binding label (`u32::MAX`: none) and that
    /// tree's cost on it.
    ctx_cand: Vec<(u32, f32)>,
    /// Per independent query: the cost of the cheapest stored pair tree that serves it (`∞`:
    /// none).
    served: Vec<f64>,
    taken: Vec<bool>,
    heap: std::collections::BinaryHeap<(Key, std::cmp::Reverse<usize>)>,
    n_taken: u32,
}

impl<'s> TreeGreedy<'s> {
    /// Candidates: every label with sampled queries or independent queries, valued on its own.
    pub fn new(sample: &'s TreeSample, labels: &LabelStore, pr: &Pricing<'s>) -> Self {
        Self::build(sample, labels, pr, None)
    }

    /// Candidates: every label, its expected saving per query on the base-derived queries in
    /// closed form where the sample is too thin. A base-derived query holds the single-label
    /// clause `{ℓ}` with probability `f_ℓ/2` ([`crate::select::label_weights`]: category kept,
    /// the clause that label alone); the sample gives, per size class (`⌊log2 |P_ℓ|⌋`), the
    /// share `β` of those clauses that bind (are the query's smallest) and their mean signed
    /// saving `σ`. A label with at least `min_records` sampled queries keeps its own estimate;
    /// another is worth `(f_ℓ/2)·β·σ` of its class (a class without enough records: `β = 1` and
    /// the saving of the clause alone, `M = |P_ℓ|`), times the base-derived queries' share of the
    /// workload. Its pair contexts (exact: every context is in the sample) come on top. Rare
    /// labels, which a sample of base-derived queries almost never draws, are the selective
    /// labels a tree serves best.
    pub fn with_classes(
        sample: &'s TreeSample,
        labels: &LabelStore,
        pr: &Pricing<'s>,
        f: &[f64],
        min_records: u32,
    ) -> Self {
        Self::build(sample, labels, pr, Some((f, min_records)))
    }

    fn build(
        sample: &'s TreeSample,
        labels: &LabelStore,
        pr: &Pricing<'s>,
        closed: Option<(&[f64], u32)>,
    ) -> Self {
        let tm = &pr.tm;
        let nd = sample.n_data;
        let wd = sample.weight(0) * nd as f64; // the base-derived share, 1 − μ
        let recs = &sample.recs;
        let nl = labels.n_labels();
        let n = labels.n_points();
        // Per label: base-derived records and their summed signed saving; context records.
        let mut own: Vec<(u32, f64)> = vec![(0, 0.0); nl];
        let mut ctx_of: Vec<Vec<(u32, f32)>> = vec![Vec::new(); nl];
        for &(l, q, t) in recs {
            if (q as usize) < nd {
                let o = &mut own[l as usize];
                o.0 += 1;
                o.1 += sample.base_ns[q as usize] - f64::from(t);
            } else {
                ctx_of[l as usize].push((q, t));
            }
        }
        let derived: Vec<Option<f64>> = match closed {
            None => (0..nl)
                .map(|l| {
                    let (q, sv) = own[l];
                    (q > 0 || !ctx_of[l].is_empty()).then(|| wd * sv / nd.max(1) as f64)
                })
                .collect(),
            Some((f, min_records)) => {
                let class = |s: usize| 63 - (s.max(1) as u64).leading_zeros() as usize;
                // Per size class: records, their summed saving, and the clause probability mass.
                let mut recs_c = [0u64; 64];
                let mut save_c = [0f64; 64];
                let mut mass_c = [0f64; 64];
                for l in 0..nl {
                    let s = labels.posting_len(l as u32);
                    if s > 0 && s < n {
                        let k = class(s);
                        mass_c[k] += f[l] / 2.0;
                        recs_c[k] += u64::from(own[l].0);
                        save_c[k] += own[l].1;
                    }
                }
                (0..nl)
                    .map(|l| {
                        let s = labels.posting_len(l as u32);
                        if s == 0 || s >= n {
                            return None;
                        }
                        let (q, sv) = own[l];
                        let k = class(s);
                        let per_query = if q >= min_records {
                            sv / nd.max(1) as f64
                        } else if recs_c[k] >= u64::from(min_records) && mass_c[k] > 0.0 {
                            let beta = recs_c[k] as f64 / (nd.max(1) as f64 * mass_c[k]);
                            let sigma = save_c[k] / recs_c[k] as f64;
                            f[l] / 2.0 * beta * sigma
                        } else {
                            let sf = s as f64;
                            f[l] / 2.0
                                * (pr.router_derived(sf, false) - pr.tree_derived(sf, sf, false))
                        };
                        Some(wd * per_query)
                    })
                    .collect()
            }
        };
        let n_ctx = sample.base_ns.len() - nd;
        let mut g = Self {
            sample,
            pr: *pr,
            terms: Vec::new(),
            ctx: Vec::new(),
            ctx_recs: Vec::new(),
            ctx_cand: vec![(u32::MAX, 0.0); n_ctx],
            served: vec![f64::INFINITY; n_ctx],
            taken: Vec::new(),
            heap: std::collections::BinaryHeap::new(),
            n_taken: 0,
        };
        for (l, d) in derived.iter().enumerate() {
            let Some(d) = *d else { continue };
            let x = g.terms.len();
            let a = g.ctx_recs.len();
            for &(q, t) in &ctx_of[l] {
                g.ctx_cand[q as usize - nd] = (x as u32, t);
                g.ctx_recs.push((q, t));
            }
            g.ctx.push((a, g.ctx_recs.len()));
            let s = labels.posting_len(l as u32);
            g.terms.push(TreeTerm {
                label: l as u32,
                size: s,
                queries: own[l].0,
                contexts: ctx_of[l].len() as u32,
                derived_ns: d,
                saving_ns: 0.0,
                bytes: tm.tree_bytes(s),
                order: None,
            });
            g.taken.push(false);
        }
        for x in 0..g.terms.len() {
            let gain = g.gain(x);
            g.terms[x].saving_ns = gain;
            if gain > 0.0 {
                let key = gain / g.terms[x].bytes as f64;
                g.heap.push((Key(key), std::cmp::Reverse(x)));
            }
        }
        g
    }

    /// Fixes the trees of the labels `fixed` holds (built by another rule, e.g. a size cap):
    /// taken before the greedy, valued as the kernel uses them, and every other candidate
    /// withdrawn, so that the greedy selects pair trees only, against the routes these trees
    /// give. Returns the trees fixed, their expected saving per query (ns) and their modelled
    /// bytes.
    pub fn fix(&mut self, fixed: impl Fn(u32) -> bool) -> (u32, f64, u64) {
        let (mut saving, mut bytes) = (0.0, 0);
        for x in 0..self.terms.len() {
            if !self.taken[x] && fixed(self.terms[x].label) {
                let g = self.gain(x);
                self.taken[x] = true;
                self.terms[x].order = Some(self.n_taken);
                self.terms[x].saving_ns = g;
                self.n_taken += 1;
                saving += g;
                bytes += self.terms[x].bytes;
            }
        }
        self.heap.clear();
        (self.n_taken, saving, bytes)
    }

    /// Expected saving per query (ns) of candidate `x` given the items taken.
    fn gain(&self, x: usize) -> f64 {
        let (a, b) = self.ctx[x];
        let nd = self.sample.n_data;
        let ctx: f64 = self.ctx_recs[a..b]
            .iter()
            .filter(|&&(q, _)| self.served[q as usize - nd].is_infinite())
            .map(|&(q, t)| {
                let q = q as usize;
                self.sample.weight(q) * (self.sample.base_ns[q] - f64::from(t))
            })
            .sum();
        self.terms[x].derived_ns + ctx
    }
}

impl LabelKind for TreeGreedy<'_> {
    fn peek(&mut self) -> Option<(f64, usize)> {
        loop {
            let &(Key(key), std::cmp::Reverse(x)) = self.heap.peek()?;
            let g = self.gain(x);
            let now = g / self.terms[x].bytes as f64;
            if (now - key).abs() > 1e-12 * key.abs() {
                self.heap.pop();
                if g > 0.0 {
                    self.heap.push((Key(now), std::cmp::Reverse(x)));
                }
                continue;
            }
            return Some((now, x));
        }
    }

    fn take(&mut self, x: usize) -> f64 {
        let top = self.heap.pop().map(|(_, std::cmp::Reverse(y))| y);
        debug_assert_eq!(top, Some(x), "take the peeked candidate");
        let g = self.gain(x);
        self.taken[x] = true;
        self.terms[x].order = Some(self.n_taken);
        self.terms[x].saving_ns = g;
        self.n_taken += 1;
        g
    }

    fn skip(&mut self, x: usize) {
        let top = self.heap.pop().map(|(_, std::cmp::Reverse(y))| y);
        debug_assert_eq!(top, Some(x), "skip the peeked candidate");
    }

    fn bytes(&self, x: usize) -> u64 {
        self.terms[x].bytes
    }

    fn label(&self, x: usize) -> u32 {
        self.terms[x].label
    }

    fn independent(&self) -> Option<(&Independent, usize)> {
        self.sample.independent()
    }

    fn pair_cost(&self, q: usize, _a: u32, _b: u32, joint: u32) -> Option<f64> {
        let w = self.sample.independent_query(q)?;
        if self.pr.kernel.is_some() {
            // The kernel considers a pair tree for a query whose binding label has a tree only
            // when it has fewer members than that label (`pair_covered`).
            match self.ctx_cand[q - self.sample.n_data] {
                (x, _)
                    if x != u32::MAX
                        && self.taken[x as usize]
                        && joint as usize >= self.terms[x as usize].size =>
                {
                    return None;
                }
                _ => {}
            }
        }
        let j = f64::from(joint).max(1.0);
        Some(self.pr.pair_tree_on(j, w.joint.clamp(1.0, j)))
    }

    fn query_cost(&self, q: usize) -> f64 {
        let nd = self.sample.n_data;
        if q < nd {
            return self.sample.base_ns[q];
        }
        let i = q - nd;
        if self.served[i].is_finite() {
            return self.served[i];
        }
        match self.ctx_cand[i] {
            (x, t) if x != u32::MAX && self.taken[x as usize] => f64::from(t),
            _ => self.sample.base_ns[q],
        }
    }

    fn query_weight(&self, q: usize) -> f64 {
        self.sample.weight(q)
    }

    fn serve(&mut self, q: usize, cost: f64) {
        let nd = self.sample.n_data;
        if q >= nd {
            // The kernel takes a stored pair's route first, whatever the label tree costs; of
            // several stored pairs, the one with the fewest points (here: the cheapest). Under
            // the kernel's prices the route is cost-checked: the pair tree only where it is
            // cheaper than the query's route.
            if self.pr.kernel.is_some() && cost >= self.query_cost(q) {
                return;
            }
            let s = &mut self.served[q - nd];
            *s = s.min(cost);
        }
    }
}

/// The route a pair tree's base-derived queries are valued against, besides the router, when the
/// label trees are fixed ([`pair_tree_values`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PairAlternative {
    /// The pair's smaller label's fixed tree where it has one, in both pricings: a fixed tree
    /// has already taken the router's cost of its queries out of the mean the greedy starts
    /// from, so a pair saves only the rest (each query's saving counted once).
    Trees,
    /// The analytic pricing's router alone (the selection before the fixed trees were priced),
    /// which counts the fixed trees' saving a second time; kept to reproduce those selections.
    Router,
}

/// The pair-tree terms of the candidate pairs, by decreasing value (the selector's pair kind;
/// see the module documentation): `prob` the base-derived probability of the context `{a} ∧ {b}`
/// ([`context_prob`]) and `saving_ns` the pair tree's saving on such a query, whose point is a
/// member of the joint — against its best route without the pair (the router or its smaller
/// label's tree, which the tree kind buys where the router fails; member curves), negative where
/// that route is faster. On the independent queries the selector values each pair through
/// [`LabelKind::pair_cost`] (the pair tree diluted to each query's matches). With fixed label
/// trees (`tree_max`: every label of 1 to `tree_max` points has one; [`TreeGreedy::fix`]) the
/// alternative is the smaller label's tree where that label has one (`alternative`: in both
/// pricings, [`PairAlternative::Trees`]; with the curves only, [`PairAlternative::Router`]).
/// Without fixed trees the curves' alternative is any label's tree and the analytic pricing's the
/// router alone. Under the kernel's prices ([`KernelRoutes`]) the alternative is the kernel's: the
/// smaller label's fixed tree where it has one (a pair with as many points as that label is never
/// routed to, and has no value), else the graph route; a negative saving is none (the
/// cost-checked route does not take the pair tree there).
pub fn pair_tree_values(
    pr: &Pricing,
    labels: &LabelStore,
    model: &QueryModel,
    f: &[f64],
    cands: &[PairCand],
    tree_max: Option<usize>,
    alternative: PairAlternative,
) -> Vec<PairValue> {
    let mut out: Vec<PairValue> = cands
        .iter()
        .filter_map(|&p| {
            let j = f64::from(p.joint).max(1.0);
            let s = labels.posting_len(p.a).min(labels.posting_len(p.b));
            let saving_ns = if pr.kernel.is_some() {
                // The kernel's prices: the context's route is its smaller label's tree where it
                // has one (taken whatever it costs), else the graph route; the pair tree is
                // taken only where cheaper (and, against a label tree, only with fewer members).
                let tree = s < pr.n && tree_max.is_some_and(|m| s <= m);
                if tree && p.joint as usize >= s {
                    return None;
                }
                let alt = if tree {
                    pr.tree_pair_label(s as f64, j)
                } else {
                    pr.router_pair(j)
                };
                (alt - pr.pair_tree(j)).max(0.0)
            } else {
                let tree = s < pr.n
                    && tree_max.is_none_or(|m| s <= m)
                    && (pr.curves.is_some()
                        || (tree_max.is_some() && alternative == PairAlternative::Trees));
                let (mut alt, pair) = match pr.curves {
                    Some(_) => (pr.router_derived(j, true), pr.pair_tree_derived(j)),
                    None => (pr.router_pair(j), pr.pair_tree(j)),
                };
                if tree {
                    alt = alt.min(pr.tree_derived(s as f64, j, true));
                }
                alt - pair
            };
            if !saving_ns.is_finite() {
                return None;
            }
            let prob = context_prob(model, f, &p);
            let bytes = pr.tm.pair_tree_bytes(p.joint);
            Some(PairValue {
                cand: p,
                prob,
                saving_ns,
                bytes,
                value: prob * saving_ns / bytes as f64,
            })
        })
        .collect();
    out.sort_by(|x, y| {
        y.value
            .total_cmp(&x.value)
            .then((x.cand.a, x.cand.b).cmp(&(y.cand.a, y.cand.b)))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{labels::LabelStoreBuilder, select};

    /// 20,000 points: category `c` with one label each among 4 equally frequent ones (broad),
    /// category `t` with a rare label `tr` (1 %) and a common `tx`, category `u` with labels
    /// `u0`…`u9` (10 % each).
    fn store() -> LabelStore {
        let mut b = LabelStoreBuilder::new();
        let names_c = ["c0", "c1", "c2", "c3"];
        let names_u: Vec<String> = (0..10).map(|i| format!("u{i}")).collect();
        for p in 0..20_000u64 {
            let t = if p % 100 == 7 { "tr" } else { "tx" };
            let mut v = vec![
                ("c", names_c[(p % 4) as usize]),
                ("t", t),
                ("u", names_u[((p / 4) % 10) as usize].as_str()),
            ];
            v.sort_unstable();
            b.push_point(v).unwrap();
        }
        b.finish().unwrap()
    }

    /// Graph costs only: a list of `l` evaluates `l` nodes (50 ns a distance, 20 ns a check).
    fn costs() -> UnitCosts {
        UnitCosts {
            graph_dist_ns: 50.0,
            graph_pred_ns: 20.0,
            graph_alpha: 1.0,
            graph_beta: 0.0,
            ..UnitCosts::default()
        }
    }

    #[test]
    fn trees_pay_on_selective_labels_and_cost_on_broad_ones() {
        let c = costs();
        let tm = TreeModel {
            k: 10,
            recall: 0.9,
            frontier: 512.0,
            curve: false,
            fanout: 26.0,
            router_bytes: 400,
            compact_vector_bytes: None,
        };
        let n = 1_000_000;
        // A selective label: the tree searches it whole, below the router's post-filter route
        // at its match rate — far below for 200 points (0.02 %), still below for 2,000.
        assert!(tm.tree_ns(&c, 200.0, 200.0, false) < tm.router_ns(&c, 200.0, n) / 10.0);
        assert!(tm.tree_ns(&c, 2_000.0, 2_000.0, true) < tm.router_ns(&c, 2_000.0, n));
        // A broad label whose matches are dense: the router wins.
        assert!(tm.tree_ns(&c, 400_000.0, 300_000.0, false) > tm.router_ns(&c, 300_000.0, n));
        // Dilution costs the tree route members per match, up to the whole tree.
        let whole = tm.tree_expansions(&c, 500_000.0, 1.0);
        assert_eq!(whole, 500_000.0 / 25.0 + 1.0);
        assert!(
            tm.tree_expansions(&c, 500_000.0, 50_000.0)
                < tm.tree_expansions(&c, 500_000.0, 1_000.0) / 2.0
        );
        assert_eq!(tm.tree_expansions(&c, 500_000.0, 100.0), whole);
        // Member checks cost only when the query has other clauses.
        assert!(tm.tree_ns(&c, 2_000.0, 500.0, true) > tm.tree_ns(&c, 2_000.0, 500.0, false));
        // Bytes: one router per small label, about s/(b̄ − 1) for large ones.
        assert_eq!(tm.routers(10), 1);
        assert_eq!(tm.routers(2_501), 100);
    }

    #[test]
    fn the_tree_kind_selects_within_the_budget_by_saving_per_byte() {
        let s = store();
        let c = costs();
        let tm = TreeModel {
            k: 10,
            recall: 0.9,
            frontier: 512.0,
            curve: false,
            fanout: 26.0,
            router_bytes: 400,
            compact_vector_bytes: None,
        };
        let model = QueryModel {
            n_queries: 20_000,
            seed: 7,
            ..QueryModel::default()
        };
        let pr = Pricing {
            tm,
            c: &c,
            curves: None,
            n: s.n_points(),
            at_max: false,
            kernel: None,
        };
        let sample = TreeSample::build(&s, &pr, model, 2);
        let again = TreeSample::build(&s, &pr, model, 1);
        assert_eq!(
            sample.base_ns, again.base_ns,
            "independent of the thread count"
        );
        let gr = TreeGreedy::new(&sample, &s, &pr);
        let names = s.name_index();
        let id = |name: &str| names[name];
        let term = |l: u32| gr.terms.iter().find(|t| t.label == l).cloned();
        // The rare label is the binding clause of the queries that keep it, and pays.
        let rare = term(id("tr")).expect("queries bind on tr");
        assert!(rare.saving_ns > 0.0, "{rare:?}");
        // The common label of category t pays nothing (its queries are dense).
        if let Some(x) = term(id("tx")) {
            assert!(x.saving_ns <= 0.0, "{x:?}");
        }
        // The greedy through the selector: within every budget, the rare label first.
        let (counts, f) = select::label_weights(&s, model.singleton, 2);
        let cands = select::pair_candidates(&s, &counts, model.singleton, |_| true, 2);
        let values = pair_tree_values(&pr, &s, &model, &f, &cands, None, PairAlternative::Trees);
        assert!(values.windows(2).all(|w| w[0].value >= w[1].value));
        for budget in [0u64, rare.bytes, 10 * rare.bytes, u64::MAX / 4] {
            let params = select::Params {
                k: 10,
                r: 32,
                budget_bytes: budget,
                model,
                stop: select::Stop::Budget,
                ..select::Params::default()
            };
            let sel =
                select::select_with(&s, || TreeGreedy::new(&sample, &s, &pr), &values, &params);
            assert!(sel.best.bytes() <= budget);
            if budget >= rare.bytes && !sel.best.islands.is_empty() {
                assert!(
                    sel.best.islands.contains(&id("tr")),
                    "{:?}",
                    sel.best.islands
                );
            }
            assert!(!sel.best.islands.contains(&id("tx")));
        }
        // Pair trees: the diluted pairs of the rare label (1 % of 10 %) are worth a tree.
        assert!(values
            .iter()
            .any(|v| [v.cand.a, v.cand.b].contains(&id("tr"))));
    }

    #[test]
    fn the_calibrated_tree_curve_charges_larger_trees_more_routers() {
        let mut c = costs();
        // Two size classes (1,000 and 100,000 members): frontiers 4 / 16 (R 0.9) and 32 / 128
        // (R 0.99) at K' = 10, eight times that at K' = 1,000.
        c.tree_r = vec![0.9, 0.99];
        c.tree_k = vec![10.0, 1_000.0];
        c.tree_a = vec![1_000.0, 100_000.0];
        c.tree_l = vec![
            vec![4.0, 32.0],
            vec![32.0, 256.0],
            vec![16.0, 128.0],
            vec![128.0, 1_024.0],
        ];
        let tm = TreeModel {
            k: 10,
            recall: 0.9,
            frontier: 512.0,
            curve: true,
            fanout: 26.0,
            router_bytes: 400,
            compact_vector_bytes: None,
        };
        assert!((tm.tree_expansions(&c, 1_000.0, 1_000.0) - 4.0).abs() < 1e-9);
        assert!((tm.tree_expansions(&c, 100_000.0, 100_000.0) - 16.0).abs() < 1e-9);
        let mid = tm.tree_expansions(&c, 10_000.0, 10_000.0);
        assert!(mid > 4.0 && mid < 16.0, "{mid}");
        // Diluted: the K-th match lies deeper among the members; at most the whole tree.
        assert!(tm.tree_expansions(&c, 100_000.0, 1_000.0) > 16.0);
        assert_eq!(tm.tree_expansions(&c, 100.0, 1.0), 100.0 / 25.0 + 1.0);
        // Without a curve (or not asked to use it), the reference frontier scaled by the
        // members per match.
        let plain = costs();
        assert_eq!(tm.tree_expansions(&plain, 100_000.0, 100_000.0), 512.0);
        let fixed = TreeModel { curve: false, ..tm };
        assert_eq!(fixed.tree_expansions(&c, 100_000.0, 50_000.0), 1_024.0);
    }

    #[test]
    fn independent_queries_value_their_smallest_label_s_tree_until_a_pair_tree_serves_them() {
        let s = store();
        let c = costs();
        let tm = TreeModel {
            k: 10,
            recall: 0.9,
            frontier: 512.0,
            curve: false,
            fanout: 26.0,
            router_bytes: 400,
            compact_vector_bytes: None,
        };
        let (counts, f) = select::label_weights(&s, 0.5, 2);
        let contexts = select::pair_contexts(&s, &counts, 0.5, 2);
        let model = QueryModel {
            n_queries: 4_000,
            seed: 7,
            uniform: 0.5,
            contexts: contexts.len() as u64,
            ..QueryModel::default()
        };
        let pr = Pricing {
            tm,
            c: &c,
            curves: None,
            n: s.n_points(),
            at_max: false,
            kernel: None,
        };
        let u = Arc::new(crate::workload::independent(&s, &contexts, &model, 2));
        assert_eq!(u.k_max, 3);
        let mixed = TreeSample::build_mixed(&s, &pr, model, Some(u.clone()), 2);
        assert_eq!(mixed.base_ns.len(), 4_000 + u.len());
        let plain = TreeSample::build(&s, &pr, model, 2);
        assert_eq!(plain.base_ns[..], mixed.base_ns[..4_000]);
        // Weights: (1 − μ) over the base-derived queries, μ times each independent query's
        // probability.
        let total: f64 = (0..mixed.base_ns.len()).map(|q| mixed.weight(q)).sum();
        assert!((total - 1.0).abs() < 1e-9);
        // tr (1 %) binds every independent query that holds it: its tree is worth their
        // weighted signed savings on top of half its base-derived saving.
        let id = |name: &str| s.name_index()[name];
        let tr = id("tr");
        let mine: Vec<usize> = (0..u.len())
            .filter(|&i| u.queries[i].labels.contains(&tr))
            .collect();
        assert!(mine.iter().any(|&i| u.queries[i].labels.len() == 3));
        let want: f64 = mine
            .iter()
            .map(|&i| {
                let w = &u.queries[i];
                let cl = w.labels.len();
                0.5 * w.weight
                    * (pr.router_conj(cl, w.joint) - pr.tree_conj(cl, 200.0, w.joint) as f32 as f64)
            })
            .sum();
        assert!(want > 0.0);
        let mut g_mixed = TreeGreedy::with_classes(&mixed, &s, &pr, &f, 30);
        let g_plain = TreeGreedy::with_classes(&plain, &s, &pr, &f, 30);
        let x = g_mixed.terms.iter().position(|t| t.label == tr).unwrap();
        let b = g_plain.terms.iter().find(|t| t.label == tr).unwrap();
        let a = &g_mixed.terms[x];
        assert!((a.derived_ns - 0.5 * b.derived_ns).abs() <= 1e-9 * b.derived_ns.abs());
        assert!((a.saving_ns - (a.derived_ns + want)).abs() <= 1e-6 * a.saving_ns.abs());
        // A pair tree over (tr, u1) serves every independent query holding both, at its tree
        // diluted to the query's matches; serving one takes that query's share from tr's tree,
        // and a second, cheaper pair lowers the query's cost again (the cheapest stored pair).
        let u1 = id("u1");
        let pair = (tr.min(u1), tr.max(u1));
        let i = u
            .pair_queries(pair.0, pair.1)
            .next()
            .expect("a query holding the pair");
        let q = 4_000 + i;
        let joint = contexts
            .iter()
            .find(|p| (p.a, p.b) == pair)
            .map(|p| p.joint)
            .unwrap();
        let cost = g_mixed.pair_cost(q, pair.0, pair.1, joint).unwrap();
        let m = u.queries[i].joint.min(f64::from(joint));
        assert_eq!(cost, pr.pair_tree_on(f64::from(joint), m));
        assert_eq!(g_mixed.pair_cost(0, pair.0, pair.1, joint), None);
        let before = g_mixed.gain(x);
        g_mixed.serve(q, cost);
        assert_eq!(g_mixed.query_cost(q), cost);
        let w = &u.queries[i];
        let lost = mixed.weight(q)
            * (mixed.base_ns[q] - pr.tree_conj(w.labels.len(), 200.0, w.joint) as f32 as f64);
        assert!((before - g_mixed.gain(x) - lost).abs() <= 1e-6 * lost.abs());
        g_mixed.serve(q, cost / 2.0);
        assert_eq!(g_mixed.query_cost(q), cost / 2.0);
        g_mixed.serve(q, cost * 2.0);
        assert_eq!(g_mixed.query_cost(q), cost / 2.0);
    }

    #[test]
    fn a_route_is_priced_on_its_pareto_frontier_and_unreachable_above_it() {
        // (recall, ns) at five levels, one dominated (more time, less recall).
        let pts = [
            [0.5, 10.0],
            [0.8, 20.0],
            [0.7, 25.0],
            [0.9, 40.0],
            [0.95, 80.0],
        ];
        assert_eq!(curve_price(&pts, 0.4), Some(10.0));
        assert!((curve_price(&pts, 0.85).unwrap() - 30.0).abs() < 1e-9);
        assert!((curve_price(&pts, 0.95).unwrap() - 80.0).abs() < 1e-9);
        assert_eq!(curve_price(&pts, 0.96), None);
        // Between classes: log-linear in the size; an unreachable class decides when nearer.
        let class = |size: f64, router: Vec<[f64; 2]>| CurveClass {
            size,
            router: [("10".to_string(), router)].into_iter().collect(),
            tree: [("10".to_string(), vec![[1.0, 5.0]])].into_iter().collect(),
        };
        let cl = vec![
            class(100.0, vec![[0.99, 100.0]]),
            class(10_000.0, vec![[0.99, 10_000.0]]),
            class(1_000_000.0, vec![[0.5, 1.0]]),
        ];
        let at = |size| RouteCurves::price(&cl, Route::Router, 10, size, 0.96);
        assert!((at(1_000.0).unwrap() - 1_000.0).abs() < 1e-6);
        assert_eq!(at(10.0), Some(100.0));
        assert_eq!(at(20_000.0), Some(10_000.0));
        assert_eq!(at(500_000.0), None);
        assert_eq!(at(5.0e6), None);
        // Priced at its highest level's time instead, an unreachable class is finite.
        let at_max = |size| RouteCurves::price_with(&cl, Route::Router, 10, size, 0.96, true);
        assert_eq!(at_max(5.0e6), Some(1.0));
        assert!((at_max(100_000.0).unwrap() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn calibrated_pricing_makes_unreachable_labels_must_have_trees_and_splits_the_independent_part()
    {
        let s = store();
        let c = costs();
        let tm = TreeModel {
            k: 10,
            recall: 0.96,
            frontier: 4096.0,
            curve: false,
            fanout: 26.0,
            router_bytes: 400,
            compact_vector_bytes: None,
        };
        // The router reaches 0.96 only on labels of ≥ 2,000 points; trees reach it everywhere.
        let class = |size: f64, router: f64| CurveClass {
            size,
            router: [
                ("10".to_string(), vec![[router, 1_000.0]]),
                ("100".to_string(), vec![[router, 2_000.0]]),
            ]
            .into_iter()
            .collect(),
            tree: [
                ("10".to_string(), vec![[1.0, 50.0]]),
                ("100".to_string(), vec![[1.0, 200.0]]),
            ]
            .into_iter()
            .collect(),
        };
        let classes = vec![
            class(200.0, 0.7),
            class(2_000.0, 0.99),
            class(5_000.0, 0.99),
        ];
        let cv = RouteCurves {
            u: classes.clone(),
            d: classes.clone(),
            p: classes,
            // Conjunctions placed at a member: the router never reaches 0.96.
            q: vec![class(5_000.0, 0.5)],
            t: Vec::new(),
        };
        let pr = Pricing {
            tm,
            c: &c,
            curves: Some(&cv),
            n: s.n_points(),
            at_max: false,
            kernel: None,
        };
        assert_eq!(pr.router_label(200.0), UNREACHABLE_NS);
        assert_eq!(pr.router_label(5_000.0), 1_000.0);
        // A base-derived conjunction is priced by the member-placed pair curves, a single label by
        // the member curves.
        assert_eq!(pr.router_derived(5_000.0, false), 1_000.0);
        assert_eq!(pr.router_derived(5_000.0, true), UNREACHABLE_NS);
        assert!((pr.tree_label(200.0) - 50.0).abs() < 1e-9);
        // A diluted clause: K' = K·s/m between 10 and 100 members, then ∝ K' up to the whole tree.
        assert!((pr.tree_derived(2_000.0, 2_000.0, false) - 50.0).abs() < 1e-9);
        assert!((pr.tree_derived(2_000.0, 200.0, true) - 200.0).abs() < 1e-6);
        assert!((pr.tree_derived(2_000.0, 20.0, true) - 200.0).abs() < 1e-9);
        // Independent conjunctions: the router by the single labels' curves for one clause, the
        // pairs' for two, the triples' for three (the pairs' while no triple curve is measured).
        assert_eq!(pr.router_conj(1, 5_000.0), pr.router_label(5_000.0));
        assert_eq!(pr.router_conj(3, 5_000.0), pr.router_pair(5_000.0));
        let with_t = RouteCurves {
            t: vec![class(1e6, 0.5)],
            ..cv.clone()
        };
        let pr_t = Pricing {
            curves: Some(&with_t),
            ..pr
        };
        assert_eq!(pr_t.router_conj(3, 5_000.0), UNREACHABLE_NS);
        assert_eq!(pr_t.router_conj(2, 5_000.0), 1_000.0);
        // A pair tree on a query with fewer matches than its joint: its tree diluted.
        assert!((pr.pair_tree_on(2_000.0, 2_000.0) - pr.pair_tree(2_000.0)).abs() < 1e-9);
        assert!((pr.pair_tree_on(2_000.0, 200.0) - 200.0).abs() < 1e-6);
        let (counts, f) = select::label_weights(&s, 0.5, 2);
        let contexts = select::pair_contexts(&s, &counts, 0.5, 2);
        let model = QueryModel {
            n_queries: 2_000,
            seed: 7,
            uniform: 0.5,
            contexts: contexts.len() as u64,
            ..QueryModel::default()
        };
        // The selector's layout: base-derived queries, then the independent queries.
        let u = Arc::new(crate::workload::independent(&s, &contexts, &model, 2));
        let sample = TreeSample::build_mixed(&s, &pr, model, Some(u.clone()), 2);
        assert_eq!(sample.base_ns.len(), 2_000 + u.len());
        let total: f64 = (0..sample.base_ns.len()).map(|q| sample.weight(q)).sum();
        assert!((total - 1.0).abs() < 1e-9);
        assert_eq!(sample.independent().map(|(_, first)| first), Some(2_000));
        // The rare label tr (200 points) is a must-have: every independent query holding it binds
        // on it and has at most 200 matches (the router unreachable below 2,000 points; its tree
        // at 50 ns), so each is worth the router's price at its own weight.
        let id = s.name_index()["tr"];
        let g = TreeGreedy::with_classes(&sample, &s, &pr, &f, 30);
        let t = g.terms.iter().find(|t| t.label == id).unwrap();
        let w_tr: f64 = u
            .queries
            .iter()
            .filter(|w| w.labels.contains(&id))
            .map(|w| 0.5 * w.weight)
            .sum();
        assert!(w_tr > 0.0);
        let want = w_tr * (UNREACHABLE_NS - 50.0);
        assert!(
            (t.saving_ns - t.derived_ns - want).abs() <= 1e-6 * want,
            "{t:?} {want}"
        );
    }

    #[test]
    fn calibrated_label_and_pair_trees_never_save_more_than_the_workload_costs() {
        let s = store();
        let c = costs();
        let tm = TreeModel {
            k: 10,
            recall: 0.96,
            frontier: 4096.0,
            curve: false,
            fanout: 26.0,
            router_bytes: 400,
            compact_vector_bytes: None,
        };
        // The router reaches 0.96 only on clauses of ≥ 1,000 points, placed anywhere; trees
        // everywhere at 50 ns, except a label's tree on a pair context (diluted): 500 ns.
        let class = |size: f64, router: f64, tree: f64| CurveClass {
            size,
            router: [
                ("10".to_string(), vec![[router, 100.0]]),
                ("100".to_string(), vec![[router, 100.0]]),
            ]
            .into_iter()
            .collect(),
            tree: [
                ("10".to_string(), vec![[1.0, tree]]),
                ("100".to_string(), vec![[1.0, tree]]),
            ]
            .into_iter()
            .collect(),
        };
        let classes = |tree: f64| {
            vec![
                class(999.0, 0.5, tree),
                class(1_000.0, 1.0, tree),
                class(1e6, 1.0, tree),
            ]
        };
        let cv = RouteCurves {
            u: classes(50.0),
            d: classes(50.0),
            p: classes(500.0),
            q: Vec::new(),
            t: Vec::new(),
        };
        let pr = Pricing {
            tm,
            c: &c,
            curves: Some(&cv),
            n: s.n_points(),
            at_max: false,
            kernel: None,
        };
        let (counts, f) = select::label_weights(&s, 0.5, 2);
        let contexts = select::pair_contexts(&s, &counts, 0.5, 2);
        let model = QueryModel {
            n_queries: 4_000,
            seed: 11,
            uniform: 0.5,
            contexts: contexts.len() as u64,
            ..QueryModel::default()
        };
        let u = Arc::new(crate::workload::independent(&s, &contexts, &model, 2));
        let sample = TreeSample::build_mixed(&s, &pr, model, Some(u), 2);
        let mean = sample.mean_base_ns();
        assert!(mean > 1e6, "unreachable queries dominate the mean");
        let cands = select::pair_candidates(&s, &counts, 0.5, |_| true, 2);
        let values = pair_tree_values(&pr, &s, &model, &f, &cands, None, PairAlternative::Trees);
        assert!(!values.is_empty());
        // On a base-derived `{a} ∧ {b}` (a member as the query point) a pair tree saves at most
        // what the smaller label's tree costs there (50 ns), never the router's unreachable price.
        for v in &values {
            assert!(v.prob * v.saving_ns <= context_prob(&model, &f, &v.cand) * 50.0 + 1e-9);
        }
        let params = select::Params {
            k: 10,
            r: 32,
            budget_bytes: u64::MAX / 4,
            model,
            stop: select::Stop::Budget,
            mean_ns: mean,
            ..select::Params::default()
        };
        let sel = select::select_with(
            &s,
            || TreeGreedy::with_classes(&sample, &s, &pr, &f, 30),
            &values,
            &params,
        );
        // A base-derived query `{a} ∧ {b}` that a label tree makes answerable is not counted
        // again by the pair tree: every tree taken, the expected saving stays within the
        // workload's cost (up to the sampling noise of the rare labels' closed form).
        assert!(sel.best.n_pairs > 0 && !sel.best.islands.is_empty());
        assert!(sel.best.t_ns >= -0.01 * mean, "{} of {mean}", sel.best.t_ns);
    }

    #[test]
    fn compact_bytes_count_a_trees_adjacency_lists_and_vectors() {
        let rows = TreeModel {
            k: 10,
            recall: 0.9,
            frontier: 512.0,
            curve: false,
            fanout: 26.0,
            router_bytes: TreeModel::router_bytes_of(83, 64),
            compact_vector_bytes: None,
        };
        let compact = TreeModel {
            compact_vector_bytes: Some(64),
            ..rows
        };
        for s in [1usize, 2, 26, 27, 1_000, 100_000] {
            let r = rows.routers(s);
            assert_eq!(rows.tree_bytes(s), r * 401);
            // R nodes and s + R − 1 edges of 4 B, R vectors of 64 B.
            assert_eq!(
                compact.tree_bytes(s),
                4 * r + 4 * (s as u64 + r - 1) + 64 * r
            );
            assert!(compact.tree_bytes(s) < rows.tree_bytes(s));
        }
        // A pair tree: its list and directory entry, its tree and (compact) its root entry.
        assert_eq!(
            rows.pair_tree_bytes(60),
            pair_bytes(60) + rows.tree_bytes(60)
        );
        assert_eq!(
            compact.pair_tree_bytes(60),
            pair_bytes(60) + compact.tree_bytes(60) + 4
        );
        // The rows accounting reports as before (no compact key).
        let j = serde_json::to_value(rows).unwrap();
        assert!(j.get("compact_vector_bytes").is_none());
        assert_eq!(
            serde_json::to_value(compact).unwrap()["compact_vector_bytes"],
            64
        );
    }

    #[test]
    fn fixed_trees_are_taken_first_and_only_pair_trees_are_selected() {
        let s = store();
        let c = costs();
        let tm = TreeModel {
            k: 10,
            recall: 0.96,
            frontier: 4096.0,
            curve: false,
            fanout: 26.0,
            router_bytes: 400,
            compact_vector_bytes: None,
        };
        // The router reaches 0.96 only on clauses of ≥ 1,000 points; trees at 50 ns, a label's
        // tree on a pair context (diluted) at 500 ns.
        let class = |size: f64, router: f64, tree: f64| CurveClass {
            size,
            router: [
                ("10".to_string(), vec![[router, 100.0]]),
                ("100".to_string(), vec![[router, 100.0]]),
            ]
            .into_iter()
            .collect(),
            tree: [
                ("10".to_string(), vec![[1.0, tree]]),
                ("100".to_string(), vec![[1.0, tree]]),
            ]
            .into_iter()
            .collect(),
        };
        let classes = |tree: f64| {
            vec![
                class(999.0, 0.5, tree),
                class(1_000.0, 1.0, tree),
                class(1e6, 1.0, tree),
            ]
        };
        let cv = RouteCurves {
            u: classes(50.0),
            d: classes(50.0),
            p: classes(500.0),
            q: Vec::new(),
            t: Vec::new(),
        };
        let pr = Pricing {
            tm,
            c: &c,
            curves: Some(&cv),
            n: s.n_points(),
            at_max: false,
            kernel: None,
        };
        let (counts, f) = select::label_weights(&s, 0.5, 2);
        let contexts = select::pair_contexts(&s, &counts, 0.5, 2);
        let model = QueryModel {
            n_queries: 4_000,
            seed: 11,
            uniform: 0.5,
            contexts: contexts.len() as u64,
            ..QueryModel::default()
        };
        let u = Arc::new(crate::workload::independent(&s, &contexts, &model, 2));
        let sample = TreeSample::build_mixed(&s, &pr, model, Some(u.clone()), 2);
        let mean = sample.mean_base_ns();
        // The size cap: tr (200 points) and u0…u9 (2,000 each) get a tree; tx and c0…c3 do not.
        let cap = 2_000;
        let fixed = |l: u32| (1..=cap).contains(&s.posting_len(l));
        let mut g = TreeGreedy::with_classes(&sample, &s, &pr, &f, 30);
        let free = g.clone();
        let (nt, saving, bytes) = g.fix(fixed);
        assert_eq!(nt, 11);
        // No two trees share a query: their saving is the sum of their values at the start.
        let mine = free.terms.iter().filter(|t| fixed(t.label));
        let want: f64 = mine.clone().map(|t| t.saving_ns).sum();
        assert!(
            (saving - want).abs() <= 1e-9 * want.abs(),
            "{saving} {want}"
        );
        assert_eq!(bytes, mine.map(|t| t.bytes).sum::<u64>());
        assert!(g.terms.iter().all(|t| t.order.is_some() == fixed(t.label)));
        // No other label tree is offered.
        assert_eq!(g.peek(), None);
        // An independent query costs its binding label's tree if that label's tree is fixed, the
        // router otherwise.
        let nd = sample.n_data;
        let mut checked = [0; 2];
        for (i, w) in u.queries.iter().enumerate() {
            let q = nd + i;
            let l = *w
                .labels
                .iter()
                .min_by_key(|&&l| (s.posting_len(l), l))
                .unwrap();
            if fixed(l) {
                let t = pr.tree_conj(w.labels.len(), s.posting_len(l) as f64, w.joint.max(1.0));
                assert_eq!(g.query_cost(q), f64::from(t as f32));
                checked[0] += 1;
            } else {
                assert_eq!(g.query_cost(q), sample.base_ns[q]);
                checked[1] += 1;
            }
        }
        assert!(checked[0] > 0 && checked[1] > 0, "{checked:?}");
        // A base-derived pair context whose smaller label has no tree is valued against the
        // router alone: never less than against the router or a tree.
        let cands = select::pair_candidates(&s, &counts, 0.5, |_| true, 2);
        let values = pair_tree_values(
            &pr,
            &s,
            &model,
            &f,
            &cands,
            Some(cap),
            PairAlternative::Trees,
        );
        let values_free =
            pair_tree_values(&pr, &s, &model, &f, &cands, None, PairAlternative::Trees);
        // With the curves the alternative is the same either way.
        assert_eq!(
            values,
            pair_tree_values(
                &pr,
                &s,
                &model,
                &f,
                &cands,
                Some(cap),
                PairAlternative::Router
            )
        );
        let mut above = 0;
        for v in &values {
            let w = values_free.iter().find(|w| w.cand == v.cand).unwrap();
            let small = s.posting_len(v.cand.a).min(s.posting_len(v.cand.b));
            if small > cap {
                assert!(v.saving_ns >= w.saving_ns);
                above += 1;
            } else {
                assert_eq!(v.saving_ns, w.saving_ns);
            }
        }
        assert!(above > 0);
        // The analytic pricing: without fixed trees against the router alone; with them against
        // the smaller label's fixed tree where it has one (`Trees`, the default), or against the
        // router again (`Router`, counting the fixed trees' saving twice).
        let pr_a = Pricing { curves: None, ..pr };
        let fixed_a = pair_tree_values(
            &pr_a,
            &s,
            &model,
            &f,
            &cands,
            Some(cap),
            PairAlternative::Trees,
        );
        let fixed_r = pair_tree_values(
            &pr_a,
            &s,
            &model,
            &f,
            &cands,
            Some(cap),
            PairAlternative::Router,
        );
        let mut below = 0;
        for v in pair_tree_values(&pr_a, &s, &model, &f, &cands, None, PairAlternative::Trees) {
            let j = f64::from(v.cand.joint).max(1.0);
            assert_eq!(v.saving_ns, pr_a.router_pair(j) - pr_a.pair_tree(j));
            assert_eq!(
                fixed_r.iter().find(|w| w.cand == v.cand).unwrap().saving_ns,
                v.saving_ns
            );
            let w = fixed_a.iter().find(|w| w.cand == v.cand).unwrap();
            let small = s.posting_len(v.cand.a).min(s.posting_len(v.cand.b));
            if small > cap {
                assert_eq!(w.saving_ns, v.saving_ns);
            } else {
                let tree = pr_a.tree_derived(small as f64, j, true);
                assert_eq!(
                    w.saving_ns,
                    pr_a.router_pair(j).min(tree) - pr_a.pair_tree(j)
                );
                below += usize::from(w.saving_ns < v.saving_ns);
            }
        }
        assert!(below > 0);
        // The greedy takes pair trees only, starting from the mean cost with the fixed trees.
        let params = select::Params {
            k: 10,
            r: 32,
            budget_bytes: u64::MAX / 4,
            model,
            stop: select::Stop::Elasticity,
            mean_ns: mean - saving,
            index_bytes: 50_000_000,
            ..select::Params::default()
        };
        let sel = select::select_with(
            &s,
            || {
                let mut g = TreeGreedy::with_classes(&sample, &s, &pr, &f, 30);
                g.fix(fixed);
                g
            },
            &values,
            &params,
        );
        assert!(sel.best.islands.is_empty() && sel.best.n_pairs > 0);
        // Priced against the fixed trees (the default), the analytic selection saves at most what
        // the workload costs after the fixed trees (each query's saving counted once).
        let sample_a = TreeSample::build_mixed(&s, &pr_a, model, Some(u.clone()), 2);
        let mean_a = sample_a.mean_base_ns();
        let (_, saving_a, _) = TreeGreedy::with_classes(&sample_a, &s, &pr_a, &f, 30).fix(fixed);
        let params_a = select::Params {
            mean_ns: mean_a - saving_a,
            ..params
        };
        let sel_a = select::select_with(
            &s,
            || {
                let mut g = TreeGreedy::with_classes(&sample_a, &s, &pr_a, &f, 30);
                g.fix(fixed);
                g
            },
            &fixed_a,
            &params_a,
        );
        assert!(sel_a.best.islands.is_empty());
        assert!(
            sel_a.best.t_ns >= -0.01 * mean_a,
            "{} of {mean_a}",
            sel_a.best.t_ns
        );
        let t = mean - saving - sel.best.pair_saving_ns;
        assert!(
            (sel.best.t_ns - t).abs() <= 1e-9 * mean,
            "{} {t}",
            sel.best.t_ns
        );
    }

    #[test]
    fn kernel_prices_value_a_pair_tree_only_where_the_kernel_routes_to_it() {
        let s = store();
        let n = s.n_points();
        // The kernel's prices need the router's effort curve.
        let c = UnitCosts {
            effort_r: vec![0.5, 0.9],
            effort_k: vec![10.0, 100.0],
            effort_l: vec![vec![10.0, 40.0], vec![40.0, 160.0]],
            ..costs()
        };
        let tm = TreeModel {
            k: 10,
            recall: 0.9,
            frontier: 64.0,
            curve: false,
            fanout: 26.0,
            router_bytes: 400,
            compact_vector_bytes: Some(4),
        };
        let pr = Pricing {
            tm,
            c: &c,
            curves: None,
            n,
            at_max: false,
            kernel: Some(KernelRoutes::recipe(0.9)),
        };
        // Every route at the kernel's price: a tree as the cost-checked route prices it, a query
        // no tree serves as the graph-only crossover estimates it.
        for (a, m) in [(200.0, 200.0), (2_000.0, 50.0), (10_000.0, 9_000.0)] {
            let tree = fixed_tree_route_ns(&c, 26.0, 64.0, a, m);
            assert_eq!(pr.tree_conj(3, a, m), tree);
            assert_eq!(pr.tree_derived(a, m, true), tree);
            assert_eq!(pr.pair_tree_on(a, m), tree);
            let graph = graph_only_route_ns(
                &c,
                900,
                10,
                m / n as f64,
                n,
                4.0,
                Laziness::HeapBound,
                5,
                1,
                PostfilterStarts::ByRateBothK,
            );
            assert_eq!(pr.router_conj(2, m), graph);
            assert_eq!(pr.router_derived(m, true), graph);
        }
        let (counts, f) = select::label_weights(&s, 0.5, 2);
        let contexts = select::pair_contexts(&s, &counts, 0.5, 2);
        let model = QueryModel {
            n_queries: 4_000,
            seed: 11,
            uniform: 0.5,
            contexts: contexts.len() as u64,
            ..QueryModel::default()
        };
        let u = Arc::new(crate::workload::independent(&s, &contexts, &model, 2));
        let sample = TreeSample::build_mixed(&s, &pr, model, Some(u.clone()), 2);
        // The size cap: tr (200 points, all of them in c3) and u0…u9 (2,000 each) have trees.
        let cap = 2_000;
        let fixed = |l: u32| (1..=cap).contains(&s.posting_len(l));
        let cands = select::pair_candidates(&s, &counts, 0.5, |_| true, 2);
        let values = pair_tree_values(
            &pr,
            &s,
            &model,
            &f,
            &cands,
            Some(cap),
            PairAlternative::Trees,
        );
        // A pair context is valued against its smaller label's fixed tree where it has one,
        // else the graph route, and never below zero (the route check refuses a dearer pair
        // tree); a pair as large as its smaller label (tr ∧ c3) is never routed to.
        let id = |name: &str| s.name_index()[name];
        let mut seen = [0; 3];
        for p in &cands {
            let small = s.posting_len(p.a).min(s.posting_len(p.b));
            let v = values.iter().find(|v| v.cand == *p);
            if small <= cap && p.joint as usize >= small {
                assert!(v.is_none(), "{p:?}");
                seen[0] += 1;
                continue;
            }
            let j = f64::from(p.joint).max(1.0);
            let alt = if small <= cap {
                pr.tree_pair_label(small as f64, j)
            } else {
                pr.router_pair(j)
            };
            let want = (alt - pr.pair_tree(j)).max(0.0);
            assert_eq!(v.expect("valued").saving_ns, want, "{p:?}");
            seen[1 + usize::from(want > 0.0)] += 1;
        }
        assert!(seen.iter().all(|&x| x > 0), "{seen:?}");
        assert!(!values
            .iter()
            .any(|v| (v.cand.a, v.cand.b) == (id("c3").min(id("tr")), id("c3").max(id("tr")))));
        // The independent queries: a pair tree serves a query whose binding label has a tree
        // only with fewer members than that label, and lowers a query's cost only where it is
        // cheaper (the route check).
        let mut g = TreeGreedy::with_classes(&sample, &s, &pr, &f, 30);
        g.fix(fixed);
        let nd = sample.n_data;
        let (mut refused, mut cheaper, mut dearer) = (0, 0, 0);
        for (i, w) in u.queries.iter().enumerate() {
            if w.labels.len() < 2 {
                continue;
            }
            let q = nd + i;
            let binding = *w
                .labels
                .iter()
                .min_by_key(|&&l| (s.posting_len(l), l))
                .unwrap();
            let (a, b) = (w.labels[0], w.labels[1]);
            let joint = s
                .postings(a)
                .iter()
                .filter(|p| s.postings(b).binary_search(p).is_ok())
                .count() as u32;
            let cost = g.pair_cost(q, a, b, joint);
            if fixed(binding) && joint as usize >= s.posting_len(binding) {
                assert_eq!(cost, None, "{:?}", w.labels);
                refused += 1;
                continue;
            }
            let cost = cost.expect("a pair tree serves the query");
            let before = g.query_cost(q);
            g.serve(q, cost);
            if cost < before {
                assert_eq!(g.query_cost(q), cost);
                cheaper += 1;
            } else {
                assert_eq!(g.query_cost(q), before);
                dearer += 1;
            }
        }
        assert!(
            refused > 0 && cheaper > 0 && dearer > 0,
            "{refused} {cheaper} {dearer}"
        );
        // The greedy takes only pair trees with a positive saving; the mean cost it reaches is
        // the start minus their savings.
        let sample = TreeSample::build_mixed(&s, &pr, model, Some(u), 2);
        let (_, saving, _) = TreeGreedy::with_classes(&sample, &s, &pr, &f, 30).fix(fixed);
        let mean = sample.mean_base_ns() - saving;
        let params = select::Params {
            k: 10,
            r: 32,
            budget_bytes: u64::MAX / 4,
            model,
            stop: select::Stop::Elasticity,
            mean_ns: mean,
            index_bytes: 5_000_000,
            ..select::Params::default()
        };
        let sel = select::select_with(
            &s,
            || {
                let mut g = TreeGreedy::with_classes(&sample, &s, &pr, &f, 30);
                g.fix(fixed);
                g
            },
            &values,
            &params,
        );
        assert!(sel.best.islands.is_empty());
        assert!(sel.best.pair_gains.iter().all(|&x| x > 0.0));
        assert_eq!(sel.best.pair_gains.len(), sel.best.n_pairs);
        let t = mean - sel.best.pair_saving_ns;
        assert!(
            (sel.best.t_ns - t).abs() <= 1e-9 * mean,
            "{} {t}",
            sel.best.t_ns
        );
    }
}
