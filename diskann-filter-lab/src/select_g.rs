/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Track G's auxiliary structures for the unified selector ([`crate::select`]): a router tree
//! per label and a router tree per label pair ("pair tree", over the pair's joint posting),
//! chosen on the same workload of base-derived queries, with the same unit costs, greedy and
//! budget as the hybrid's islands and combination postings — only the routes differ. Track G
//! has no exact path: a query's route without auxiliary structures is the graph router, whose
//! post-filter route at the query's match rate is its cost; a query whose binding clause `c*`
//! (its smallest clause) is a label with a tree takes the tree route, whatever it costs
//! (`tree_choice = always`); a query whose single-label clauses `{a}`, `{b}` have a pair tree
//! takes the pair route.
//!
//! **Costs** (ns, the kernel's unit costs):
//!
//! - router: its post-filter route at the query's match rate `M/n` with the list that finds the
//!   `K`-th match at the reference recall `R` — the kernel's effort curve (`effort_list(R, k')`,
//!   the calibrated unfiltered list for recall `R` at `k'` neighbours) at the rank of the `K`-th
//!   match, `k' = K/rate + z·√(K(1 − rate))/rate` (the graph route is not modelled; without an
//!   effort curve, the islands' `postfilter_nominal`);
//! - tree route over `s` members of which `M` match: `E` router expansions, each checking the
//!   router's `b̄` children, computing a distance for the matching ones and one for the router,
//!   `E·(b̄·pred(s/n) + b̄·ρ·dist(s/n) + graph_dist)` with `ρ = M/s` — the members' unit costs
//!   those of a scan of the label's members at its density (`pred`, `dist`: the exact path's
//!   calibrated curves; a tree's members are as scattered in memory as a posting's), the
//!   router's a graph node's — where `E = min(s/(b̄ − 1) + 1, F·s/M)`: the reference frontier `F`
//!   (the declared frontier table's at the reference recall `R` = 0.9: 512) scaled by the
//!   members per match, so that a diluted clause is charged the search that reaches that recall
//!   on it, at most the whole tree;
//! - pair tree on a pair of joint `j`: the tree route over its `j` members, all matching,
//!   against the better of the router and the tree route of the pair's smaller label.
//!
//! **Bytes**: a tree over `s` members has `⌈(s − 1)/(b̄ − 1)⌉` routers (at least one), each a
//! graph row, a vector and a kind byte (`router_bytes`); a pair tree adds its pair's list and
//! directory entry ([`crate::select::pair_bytes`]); the pairs' co-occurrence matrix is charged
//! by the selector as for combination postings.
//!
//! **Values**: a label tree serves exactly the sampled queries whose `c*` is that label (a
//! clause of several labels needs a tree for each of them, which this kind does not value), so
//! label trees do not interact: a tree's expected saving per query is the sum of its queries'
//! signed savings (router − tree route) over the sample size, fixed. Pair trees are valued in
//! closed form as combination postings ([`crate::select::pair_values`]); the kinds'
//! interactions are not modelled.

use crate::{
    islands::{draw_query, postfilter_nominal, query_matches, QueryModel},
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

    /// The tree route's cost over `s` members of which `m` match, on `n` points.
    pub fn tree_ns(&self, c: &UnitCosts, s: f64, m: f64, n: usize) -> f64 {
        let s = s.max(1.0);
        let m = m.clamp(1.0, s);
        let b = self.fanout.max(2.0);
        let e = (s / (b - 1.0) + 1.0).min(self.frontier * s / m);
        let rho = (s / n.max(1) as f64).min(1.0);
        e * (b * c.pred(rho) + b * (m / s) * c.dist(rho) + c.graph_dist_ns)
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

/// The label-tree terms of a sample of base-derived queries (the islands' workload and draws):
/// per query its router cost and, when its binding clause is a single label, that label's tree
/// route cost.
#[derive(Debug, Clone)]
pub struct TreeSample {
    pub model: QueryModel,
    /// Router cost of each query.
    pub base_ns: Vec<f64>,
    /// `(label, query, tree route cost)` of the queries whose binding clause is that label.
    recs: Vec<(u32, u32, f32)>,
}

impl TreeSample {
    /// Draws the queries and computes their cost terms (with up to `threads` threads; the result
    /// does not depend on the thread count).
    pub fn build(
        labels: &LabelStore,
        c: &UnitCosts,
        tm: &TreeModel,
        model: QueryModel,
        threads: usize,
    ) -> Self {
        let n = labels.n_points();
        let chunk = 1024usize;
        let init = || {
            (
                Vec::<u32>::new(),
                Vec::<(usize, usize)>::new(),
                Vec::<(usize, usize, bool)>::new(),
            )
        };
        let parts = crate::parallel::par_map(
            model.n_queries.div_ceil(chunk),
            threads,
            init,
            |ci, (kl, clauses, cats)| {
                let mut base = Vec::new();
                let mut recs = Vec::new();
                for qi in ci * chunk..((ci + 1) * chunk).min(model.n_queries) {
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
                            recs.push((l, qi as u32, tm.tree_ns(c, s as f64, m, n) as f32));
                        }
                    }
                }
                (base, recs)
            },
        );
        let mut base_ns = Vec::with_capacity(model.n_queries);
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
    /// Candidates: every label with sampled queries, valued on its own queries.
    pub fn new(sample: &TreeSample, labels: &LabelStore, tm: &TreeModel) -> Self {
        Self::build(sample, labels, tm, None)
    }

    /// Candidates: every label, its expected saving per query in closed form over the same
    /// workload where the sample is too thin. A base-derived query holds the single-label clause
    /// `{ℓ}` with probability `f_ℓ/2` ([`crate::select::label_weights`]: category kept, the
    /// clause that label alone); the sample gives, per size class (`⌊log2 |P_ℓ|⌋`), the share
    /// `β` of those clauses that bind (are the query's smallest) and their mean signed saving
    /// `σ`. A label with at least `min_records` sampled queries keeps its own estimate; another
    /// is worth `(f_ℓ/2)·β·σ` of its class (a class without enough records: `β = 1` and the
    /// saving of the clause alone, `M = |P_ℓ|`). Rare labels, which a sample of base-derived
    /// queries almost never draws, are the selective labels a tree serves best.
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
        match closed {
            None => {
                for &(l, q, saving) in &per {
                    let s = labels.posting_len(l);
                    terms.push(TreeTerm {
                        label: l,
                        size: s,
                        queries: q,
                        saving_ns: saving / nq,
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
                    let saving_ns = if q >= min_records {
                        sv / nq
                    } else if recs_c[k] >= u64::from(min_records) && mass_c[k] > 0.0 {
                        let beta = recs_c[k] as f64 / (nq * mass_c[k]);
                        let sigma = save_c[k] / recs_c[k] as f64;
                        f[l as usize] / 2.0 * beta * sigma
                    } else {
                        let sf = s as f64;
                        f[l as usize] / 2.0 * (tm.router_ns(c, sf, n) - tm.tree_ns(c, sf, sf, n))
                    };
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
            let alt = tm.router_ns(c, j, n).min(tm.tree_ns(c, s, j, n));
            let saving_ns = alt - tm.tree_ns(c, j, j, n);
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
            fanout: 26.0,
            router_bytes: 400,
        };
        let n = 1_000_000;
        // A selective label: the tree searches it whole, below the router's post-filter route
        // at its match rate — far below for 200 points (0.02 %), still below for 2,000.
        assert!(tm.tree_ns(&c, 200.0, 200.0, n) < tm.router_ns(&c, 200.0, n) / 10.0);
        assert!(tm.tree_ns(&c, 2_000.0, 2_000.0, n) < tm.router_ns(&c, 2_000.0, n));
        // A broad label whose matches are dense: the router wins.
        assert!(tm.tree_ns(&c, 400_000.0, 300_000.0, n) > tm.router_ns(&c, 300_000.0, n));
        // Dilution costs the tree route members per match, up to the whole tree.
        let whole = tm.tree_ns(&c, 500_000.0, 1.0, n);
        assert!(
            tm.tree_ns(&c, 500_000.0, 50_000.0, n) < tm.tree_ns(&c, 500_000.0, 1_000.0, n) / 2.0
        );
        assert!((tm.tree_ns(&c, 500_000.0, 100.0, n) - whole).abs() / whole < 0.01);
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
            fanout: 26.0,
            router_bytes: 400,
        };
        let model = QueryModel {
            n_queries: 20_000,
            seed: 7,
            cap: 4096,
            singleton: 0.5,
            fresh: 0.0,
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
                floor: 0.0,
                model,
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
}
