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
//! **Workload**: the selector's mixture ([`QueryModel`]): base-derived queries and, with share
//! `μ`, pair contexts `{a} ∧ {b}` drawn uniformly among the co-occurring pairs (binding clause:
//! the smaller label; `M` = the joint size).
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
//! - pair tree on a pair of joint `j`: the tree route over its `j` members, all matching,
//!   against the better of the router and the tree route of the pair's smaller label.
//!
//! **Bytes**: a tree over `s` members has `⌈(s − 1)/(b̄ − 1)⌉` routers (at least one), each a
//! graph row, a vector and a kind byte (`router_bytes`); a pair tree adds its pair's list and
//! directory entry ([`crate::select::pair_bytes`]); the pairs' co-occurrence matrix is charged
//! by the selector as for combination postings.
//!
//! **Values**: a label tree serves exactly the queries whose `c*` is that label (a clause of
//! several labels needs a tree for each of them, which this kind does not value), so label trees
//! do not interact: a tree's expected saving per query is its queries' signed savings (router −
//! tree route) — the base-derived ones from the sample (or in closed form by size class where the
//! sample is thin), the pair contexts' exactly over every context — fixed. Pair trees are valued
//! in closed form as combination postings ([`crate::select::pair_values`]); the kinds'
//! interactions are not modelled.

use crate::{
    islands::{draw_query, pair_context_draw, postfilter_nominal, query_matches, QueryModel},
    kernel::cost::{kth_match_rank, UnitCosts, Z},
    labels::LabelStore,
    select::{context_prob, pair_bytes, LabelKind, PairCand, PairValue},
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
    /// Bytes of one router (graph row, vector, kind).
    pub router_bytes: u64,
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
        self.routers(s) * self.router_bytes
    }

    /// Bytes of one router of a graph with degree bound `r_max` and vectors of `vector_bytes`.
    pub fn router_bytes_of(r_max: usize, vector_bytes: usize) -> u64 {
        (4 + 4 * r_max + vector_bytes + 1) as u64
    }
}

/// The label-tree terms of the workload: a sample of its base-derived queries — per query its
/// router cost and, when its binding clause is a single label, that label's tree route cost —
/// and of its pair contexts (their router costs in the sample, for the workload's mean cost; their
/// tree savings exactly, per label, over every context).
#[derive(Debug, Clone)]
pub struct TreeSample {
    pub model: QueryModel,
    /// Router cost of each query (base-derived, then pair contexts).
    pub base_ns: Vec<f64>,
    /// `(label, query, tree route cost)` of the base-derived queries whose binding clause is
    /// that label.
    recs: Vec<(u32, u32, f32)>,
    /// Per label: the expected saving per workload query (ns, signed) of its tree on the pair
    /// contexts whose binding clause it is (0 without pair contexts).
    pair_part: Vec<f64>,
}

impl TreeSample {
    /// Draws the base-derived queries and computes their cost terms (with up to `threads`
    /// threads; the result does not depend on the thread count). Without pair contexts (see
    /// [`Self::build_mixed`]).
    pub fn build(
        labels: &LabelStore,
        c: &UnitCosts,
        tm: &TreeModel,
        model: QueryModel,
        threads: usize,
    ) -> Self {
        let model = QueryModel {
            uniform: 0.0,
            ..model
        };
        Self::build_mixed(labels, c, tm, model, &[], threads)
    }

    /// [`Self::build`] with the workload's pair contexts ([`QueryModel::uniform`]; as
    /// [`crate::islands::QuerySample::build_mixed`]).
    pub fn build_mixed(
        labels: &LabelStore,
        c: &UnitCosts,
        tm: &TreeModel,
        model: QueryModel,
        contexts: &[PairCand],
        threads: usize,
    ) -> Self {
        assert!(
            model.uniform == 0.0 || model.contexts == contexts.len() as u64,
            "the model's pair contexts must be the given ones"
        );
        let n = labels.n_points();
        let chunk = 1024usize;
        let n_total = model.n_queries + model.n_pair_queries();
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
                        let p = &contexts[pair_context_draw(&model, qi)];
                        base.push(tm.router_ns(c, f64::from(p.joint).max(1.0), n));
                        continue;
                    }
                    if !draw_query(labels, &model, qi, kl, clauses, cats) {
                        base.push(0.0);
                        continue;
                    }
                    let (jmin, _, m) = query_matches(labels, &model, kl, clauses);
                    base.push(tm.router_ns(c, m, n));
                    let (a, b) = clauses[jmin];
                    let mut nonempty = kl[a..b].iter().filter(|&&l| labels.posting_len(l) > 0);
                    if let (Some(&l), None) = (nonempty.next(), nonempty.next()) {
                        let s = labels.posting_len(l);
                        if s < n {
                            let t = tm.tree_ns(c, s as f64, m, clauses.len() > 1);
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
        let mut pair_part = vec![0.0; labels.n_labels()];
        if model.n_pair_queries() > 0 {
            let w = model.uniform / contexts.len() as f64;
            for p in contexts {
                let (sa, sb) = (labels.posting_len(p.a), labels.posting_len(p.b));
                let (l, s) = if (sa, p.a) <= (sb, p.b) {
                    (p.a, sa)
                } else {
                    (p.b, sb)
                };
                if s == 0 || s >= n {
                    continue;
                }
                let m = f64::from(p.joint).max(1.0);
                pair_part[l as usize] +=
                    w * (tm.router_ns(c, m, n) - tm.tree_ns(c, s as f64, m, true));
            }
        }
        Self {
            model,
            base_ns,
            recs,
            pair_part,
        }
    }

    /// Expected cost per query (ns) without auxiliary structures (the router alone).
    pub fn mean_base_ns(&self) -> f64 {
        self.base_ns.iter().sum::<f64>() / self.base_ns.len().max(1) as f64
    }
}

/// One label of the tree kind.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct TreeTerm {
    pub label: u32,
    pub size: usize,
    /// Sampled queries whose binding clause is this label.
    pub queries: u32,
    /// Expected saving per query (ns) of this tree (signed: queries the router serves better
    /// count against it).
    pub saving_ns: f64,
    pub bytes: u64,
    /// Position in the greedy order (`None`: not taken).
    pub order: Option<u32>,
}

/// The tree kind as the selector's greedy sees it: candidates by expected saving per query per
/// byte; the savings do not change as trees are taken (see the module documentation).
#[derive(Debug, Clone)]
pub struct TreeGreedy {
    pub terms: Vec<TreeTerm>,
    /// Candidates with a positive saving, by decreasing saving per byte.
    order: Vec<usize>,
    next: usize,
    taken: u32,
}

impl TreeGreedy {
    /// Candidates: every label with sampled queries or pair contexts, valued on its own.
    pub fn new(sample: &TreeSample, labels: &LabelStore, tm: &TreeModel) -> Self {
        Self::build(sample, labels, tm, None)
    }

    /// Candidates: every label, its expected saving per query on the base-derived queries in
    /// closed form where the sample is too thin. A base-derived query holds the single-label
    /// clause `{ℓ}` with probability `f_ℓ/2` ([`crate::select::label_weights`]: category kept,
    /// the clause that label alone); the sample gives, per size class (`⌊log2 |P_ℓ|⌋`), the
    /// share `β` of those clauses that bind (are the query's smallest) and their mean signed
    /// saving `σ`. A label with at least `min_records` sampled queries keeps its own estimate;
    /// another is worth `(f_ℓ/2)·β·σ` of its class (a class without enough records: `β = 1` and
    /// the saving of the clause alone, `M = |P_ℓ|`), times the base-derived queries' share of the
    /// workload; plus its pair contexts' part (exact). Rare labels, which a sample of
    /// base-derived queries almost never draws, are the selective labels a tree serves best.
    pub fn with_classes(
        sample: &TreeSample,
        labels: &LabelStore,
        tm: &TreeModel,
        c: &UnitCosts,
        f: &[f64],
        min_records: u32,
    ) -> Self {
        Self::build(sample, labels, tm, Some((c, f, min_records)))
    }

    fn build(
        sample: &TreeSample,
        labels: &LabelStore,
        tm: &TreeModel,
        closed: Option<(&UnitCosts, &[f64], u32)>,
    ) -> Self {
        let nq = sample.base_ns.len().max(1) as f64;
        let recs = &sample.recs;
        // Per label: sampled queries and their summed signed saving.
        let mut per: Vec<(u32, u32, f64)> = Vec::new();
        let mut i = 0;
        while i < recs.len() {
            let l = recs[i].0;
            let mut j = i;
            let mut saving = 0.0;
            while j < recs.len() && recs[j].0 == l {
                let (_, q, t) = recs[j];
                saving += sample.base_ns[q as usize] - f64::from(t);
                j += 1;
            }
            per.push((l, (j - i) as u32, saving));
            i = j;
        }
        let n = labels.n_points();
        let mut terms = Vec::new();
        // The base-derived queries' share of the workload (the sample holds the pair contexts'
        // share besides).
        let wd = sample.model.n_queries as f64 / nq;
        let pair = &sample.pair_part;
        match closed {
            None => {
                let own: std::collections::HashMap<u32, (u32, f64)> =
                    per.iter().map(|&(l, q, sv)| (l, (q, sv))).collect();
                for l in 0..labels.n_labels() as u32 {
                    let (q, sv) = own.get(&l).copied().unwrap_or((0, 0.0));
                    if q == 0 && pair[l as usize] == 0.0 {
                        continue;
                    }
                    let s = labels.posting_len(l);
                    terms.push(TreeTerm {
                        label: l,
                        size: s,
                        queries: q,
                        saving_ns: sv / nq + pair[l as usize],
                        bytes: tm.tree_bytes(s),
                        order: None,
                    });
                }
            }
            Some((c, f, min_records)) => {
                let class = |s: usize| 63 - (s.max(1) as u64).leading_zeros() as usize;
                // Per size class: records, their summed saving, and the clause probability mass.
                let mut recs_c = [0u64; 64];
                let mut save_c = [0f64; 64];
                let mut mass_c = [0f64; 64];
                let nl = labels.n_labels();
                for l in 0..nl as u32 {
                    let s = labels.posting_len(l);
                    if s > 0 && s < n {
                        mass_c[class(s)] += f[l as usize] / 2.0;
                    }
                }
                for &(l, q, saving) in &per {
                    let k = class(labels.posting_len(l));
                    recs_c[k] += u64::from(q);
                    save_c[k] += saving;
                }
                let own: std::collections::HashMap<u32, (u32, f64)> =
                    per.iter().map(|&(l, q, sv)| (l, (q, sv))).collect();
                for l in 0..nl as u32 {
                    let s = labels.posting_len(l);
                    if s == 0 || s >= n {
                        continue;
                    }
                    let (q, sv) = own.get(&l).copied().unwrap_or((0, 0.0));
                    let k = class(s);
                    let derived = if q >= min_records {
                        sv / nq
                    } else if recs_c[k] >= u64::from(min_records) && mass_c[k] > 0.0 {
                        let nd = sample.model.n_queries as f64;
                        let beta = recs_c[k] as f64 / (nd * mass_c[k]);
                        let sigma = save_c[k] / recs_c[k] as f64;
                        wd * f[l as usize] / 2.0 * beta * sigma
                    } else {
                        let sf = s as f64;
                        let alone = tm.router_ns(c, sf, n) - tm.tree_ns(c, sf, sf, false);
                        wd * f[l as usize] / 2.0 * alone
                    };
                    let saving_ns = derived + pair[l as usize];
                    terms.push(TreeTerm {
                        label: l,
                        size: s,
                        queries: q,
                        saving_ns,
                        bytes: tm.tree_bytes(s),
                        order: None,
                    });
                }
            }
        }
        let key = |t: &TreeTerm| t.saving_ns / t.bytes as f64;
        let mut order: Vec<usize> = (0..terms.len())
            .filter(|&x| terms[x].saving_ns > 0.0)
            .collect();
        order.sort_by(|&x, &y| {
            key(&terms[y])
                .total_cmp(&key(&terms[x]))
                .then(terms[x].label.cmp(&terms[y].label))
        });
        Self {
            terms,
            order,
            next: 0,
            taken: 0,
        }
    }
}

impl LabelKind for TreeGreedy {
    fn peek(&mut self) -> Option<(f64, usize)> {
        let &x = self.order.get(self.next)?;
        let t = &self.terms[x];
        Some((t.saving_ns / t.bytes as f64, x))
    }

    fn take(&mut self, x: usize) -> f64 {
        debug_assert_eq!(
            self.order.get(self.next),
            Some(&x),
            "take the peeked candidate"
        );
        self.next += 1;
        self.terms[x].order = Some(self.taken);
        self.taken += 1;
        self.terms[x].saving_ns
    }

    fn skip(&mut self, x: usize) {
        debug_assert_eq!(
            self.order.get(self.next),
            Some(&x),
            "skip the peeked candidate"
        );
        self.next += 1;
    }

    fn bytes(&self, x: usize) -> u64 {
        self.terms[x].bytes
    }

    fn label(&self, x: usize) -> u32 {
        self.terms[x].label
    }
}

/// The pair-tree terms of the candidate pairs with a positive saving, by decreasing value (the
/// selector's pair kind; see the module documentation).
pub fn pair_tree_values(
    labels: &LabelStore,
    c: &UnitCosts,
    tm: &TreeModel,
    model: &QueryModel,
    f: &[f64],
    cands: &[PairCand],
) -> Vec<PairValue> {
    let n = labels.n_points();
    let mut out: Vec<PairValue> = cands
        .iter()
        .filter_map(|&p| {
            let s = labels.posting_len(p.a).min(labels.posting_len(p.b)) as f64;
            let j = f64::from(p.joint);
            let alt = tm.router_ns(c, j, n).min(tm.tree_ns(c, s, j, true));
            let saving_ns = alt - tm.tree_ns(c, j, j, false);
            if !saving_ns.is_finite() || saving_ns <= 0.0 {
                return None;
            }
            let prob = context_prob(model, f, &p);
            let bytes = pair_bytes(p.joint) + tm.tree_bytes(p.joint as usize);
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
        };
        let model = QueryModel {
            n_queries: 20_000,
            seed: 7,
            ..QueryModel::default()
        };
        let sample = TreeSample::build(&s, &c, &tm, model, 2);
        let again = TreeSample::build(&s, &c, &tm, model, 1);
        assert_eq!(
            sample.base_ns, again.base_ns,
            "independent of the thread count"
        );
        let gr = TreeGreedy::new(&sample, &s, &tm);
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
        let values = pair_tree_values(&s, &c, &tm, &model, &f, &cands);
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
                select::select_with(&s, || TreeGreedy::new(&sample, &s, &tm), &values, &params);
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
    fn pair_contexts_value_the_tree_of_their_smaller_label_exactly() {
        let s = store();
        let c = costs();
        let tm = TreeModel {
            k: 10,
            recall: 0.9,
            frontier: 512.0,
            curve: false,
            fanout: 26.0,
            router_bytes: 400,
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
        let mixed = TreeSample::build_mixed(&s, &c, &tm, model, &contexts, 2);
        assert_eq!(mixed.base_ns.len(), 4_000 + model.n_pair_queries());
        let plain = TreeSample::build(&s, &c, &tm, model, 2);
        assert_eq!(plain.base_ns.len(), 4_000);
        assert_eq!(plain.base_ns[..], mixed.base_ns[..4_000]);
        let id = |name: &str| s.name_index()[name];
        // tr (1 %) is the binding clause of its contexts with the u labels (0.1 % joint each):
        // their exact share, (μ / N) per context times its signed saving.
        let w = model.uniform / contexts.len() as f64;
        let want: f64 = contexts
            .iter()
            .filter(|p| [p.a, p.b].contains(&id("tr")))
            .filter(|p| s.posting_len(p.a).min(s.posting_len(p.b)) == s.posting_len(id("tr")))
            .map(|p| {
                let m = f64::from(p.joint);
                w * (tm.router_ns(&c, m, s.n_points()) - tm.tree_ns(&c, 200.0, m, true))
            })
            .sum();
        assert!(want > 0.0);
        assert!((mixed.pair_part[id("tr") as usize] - want).abs() <= 1e-9 * want);
        // The greedy adds it to the base-derived part, weighted by the base-derived share.
        let g_mixed = TreeGreedy::with_classes(&mixed, &s, &tm, &c, &f, 30);
        let g_plain = TreeGreedy::with_classes(&plain, &s, &tm, &c, &f, 30);
        let term = |g: &TreeGreedy| g.terms.iter().find(|t| t.label == id("tr")).cloned();
        let (a, b) = (term(&g_mixed).unwrap(), term(&g_plain).unwrap());
        assert!((a.saving_ns - (0.5 * b.saving_ns + want)).abs() <= 1e-6 * a.saving_ns);
    }
}
