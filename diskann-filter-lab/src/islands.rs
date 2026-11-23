/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Per-label sub-graphs ("islands", sidecar `islands.flab`): a small Vamana graph over the
//! members of each selected label, for the search route `island` of the cost crossover (a
//! search inside one label's island, the rest of the predicate checked on the nodes it
//! evaluates; [`crate::kernel::search`]).
//!
//! Selection, from the label sizes and the unit costs of the cost model only (categories are
//! not read), at `K` answers on a dataset of `n` points. For a label of `s` members:
//! - `exact(s) = s · (dist(s/n) + pred(s/n))`: the exact path over the label (a query whose
//!   other clauses keep most of it),
//! - `base(s)`: the post-filter route's nominal cost on the base graph at the match rate `s/n`
//!   (the label diluted `n/s` times),
//! - `island(s)`: the same route's cost on an island of `s` points at match rate 1,
//! - gain `g(s) = min(exact(s), base(s)) − island(s)`.
//!
//! Labels with `g > 0` and at least [`MIN_SIZE`] members are taken by decreasing key while the
//! budget lasts, `bytes(s) = s · (8 + 3R)` (its rows at the expected mean degree `0.75 R`). The
//! key is the expected saving per byte under a [`Prior`] on which labels queries search in:
//! `g(s) / bytes(s)` when every label is equally likely (`label`), `g(s)` when a label is as
//! likely as it is frequent (`data`: `(s/n) · g(s) / bytes(s) ∝ g(s)`). Under `label` small
//! labels win (a large budget fills with many of them); at 40M the labels that diluted queries
//! search in can be too large to enter.
//!
//! These gains score a label as if queried alone. The `queries` selection ([`QuerySample`])
//! scores it on *base-derived queries* instead — random AND-of-ORs sub-predicates of the label
//! sets of data points drawn uniformly (each of a point's categories kept with probability ½,
//! its clause one of the point's labels in it or a random subset of them, with equal
//! probability), so that a label is seen with the other clauses that dilute it, as often as
//! data points carry it: per query the cost of the best route without islands
//! (`min(exact, base)` at the query's match count `M`) and of the island of each of its
//! single-label clauses (rate `M / s` inside it); labels are taken greedily by expected saving
//! per query per byte given those already taken (a query keeps its cheapest route), while the
//! budget lasts. No query log and no category name is read.
//!
//! Build: the lab's Vamana builder ([`crate::build::build`], label-blind α-RobustPrune) over each
//! selected label's member vectors, one island per thread. Local ids are positions in the
//! label's posting list; the builder's auxiliary start node (a copy of the medoid) is merged
//! into the medoid, the island's start, and members unreachable from it get an edge from it
//! (counted in [`IslandStats::patched`]).
//!
//! Memory: 4 bytes per label (its island index), 16 bytes per island, 8 bytes per member (row
//! offset) and 4 bytes per edge.

use std::{
    io::{Read, Seek, Write},
    path::Path,
    sync::Arc,
    time::Instant,
};

use anyhow::{ensure, Context, Result};

use crate::{
    build::{self, BuildElement, BuildParams},
    container,
    kernel::cost::{kth_match_rank, UnitCosts, Z},
    labels::LabelStore,
    provenance::Provenance,
    select::pair_plan_ns,
    vectors::Matrix,
    workload::{query_point, Independent, Part},
};

/// File magic of `islands.flab`.
pub const MAGIC: &[u8; 8] = b"FLABISL\0";
/// Version of the `islands.flab` format.
pub const VERSION: u32 = 1;
/// Sections of the `islands.flab` format (a provenance section may follow).
pub const N_SECTIONS: usize = 6;

const NONE: u32 = u32::MAX;
/// Smallest label with an island.
pub const MIN_SIZE: usize = 64;

/// The post-filter route's list size and nominal cost (the kernel's crossover formula: a list
/// that holds the `K` nearest matches with high probability at match rate `rate` on a graph of
/// `n` points, at least `l`; `starts` start nodes).
pub fn postfilter_nominal(
    c: &UnitCosts,
    l: usize,
    k: usize,
    rate: f64,
    n: usize,
    starts: usize,
) -> (usize, f64) {
    let rank = kth_match_rank(k, rate, Z);
    if !rank.is_finite() {
        return (l, f64::INFINITY);
    }
    let l_need = c.graph_l_for(rank).ceil().min(n as f64) as usize;
    let l_pf = l_need.max(l).min(n.max(1));
    (
        l_pf,
        c.postfilter_cost(l_pf as f64, k, rate) + starts as f64 * c.graph_dist_ns,
    )
}

/// Selection terms of one label (see the module documentation).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Gain {
    pub label: u32,
    pub size: usize,
    pub exact_ns: f64,
    pub base_ns: f64,
    pub island_ns: f64,
    pub gain_ns: f64,
    pub bytes: u64,
}

/// Expected bytes of an island of `s` members with degree bound `r`.
pub fn island_bytes(s: usize, r: usize) -> u64 {
    (s as u64) * (8 + 3 * r as u64)
}

/// The selection terms of every label with at least [`MIN_SIZE`] members.
pub fn gains(labels: &LabelStore, c: &UnitCosts, k: usize, r: usize) -> Vec<Gain> {
    let n = labels.n_points();
    let nf = n as f64;
    (0..labels.n_labels() as u32)
        .filter_map(|l| {
            let s = labels.posting_len(l);
            if s < MIN_SIZE || s >= n {
                return None;
            }
            let rho = s as f64 / nf;
            let exact_ns = s as f64 * (c.dist(rho) + c.pred(rho));
            let base_ns = postfilter_nominal(c, k, k, rho, n, 1).1;
            let island_ns = postfilter_nominal(c, k, k, 1.0, s, 1).1;
            Some(Gain {
                label: l,
                size: s,
                exact_ns,
                base_ns,
                island_ns,
                gain_ns: exact_ns.min(base_ns) - island_ns,
                bytes: island_bytes(s, r),
            })
        })
        .collect()
}

/// Which labels queries are assumed to search in (the selection's ranking; see the module
/// documentation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Prior {
    /// Every label equally likely: rank by gain per byte.
    #[default]
    Label,
    /// A label as likely as it is frequent: rank by gain.
    Data,
    /// Base-derived queries ([`QuerySample`]): greedy by expected saving per byte over a sample
    /// of predicates made of data points' own labels.
    Queries,
}

impl Prior {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Label => "label",
            Self::Data => "data",
            Self::Queries => "queries",
        }
    }

    fn key(self, g: &Gain) -> f64 {
        match self {
            Self::Label | Self::Queries => g.gain_ns / g.bytes as f64,
            Self::Data => g.gain_ns,
        }
    }
}

impl std::str::FromStr for Prior {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "label" => Ok(Self::Label),
            "data" => Ok(Self::Data),
            "queries" => Ok(Self::Queries),
            _ => anyhow::bail!("unknown island prior {s:?} (label, data, queries)"),
        }
    }
}

/// The labels to build within `budget_bytes` under the `label` prior (in label-id order).
pub fn select(gains: &[Gain], budget_bytes: u64) -> Vec<u32> {
    select_with(gains, budget_bytes, Prior::Label)
}

/// The labels to build within `budget_bytes`: positive gains by decreasing key of `prior`
/// (in label-id order).
pub fn select_with(gains: &[Gain], budget_bytes: u64, prior: Prior) -> Vec<u32> {
    let mut order: Vec<&Gain> = gains.iter().filter(|g| g.gain_ns > 0.0).collect();
    order.sort_by(|a, b| {
        prior
            .key(b)
            .total_cmp(&prior.key(a))
            .then(a.label.cmp(&b.label))
    });
    let mut used = 0u64;
    let mut out = Vec::new();
    for g in order {
        if used + g.bytes > budget_bytes {
            continue;
        }
        used += g.bytes;
        out.push(g.label);
    }
    out.sort_unstable();
    out
}

/// Base-derived queries of the `queries` selection: `n_queries` data points drawn uniformly
/// (seeded); a point's query is a random AND-of-ORs sub-predicate of its own labels: each of
/// its categories kept with probability ½ (one at random if none), the clause of a kept
/// category either one of the point's labels in it (probability `singleton`: a context the
/// point would match, one value per category) or a random non-empty subset of them (each kept
/// with probability ½: an OR list). The match count of a query is counted on its smallest
/// clause's members (every `⌈s / cap⌉`-th one beyond `cap`).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct QueryModel {
    pub n_queries: usize,
    pub seed: u64,
    pub cap: usize,
    /// Probability that a kept category's clause is a single label of the point (else a random
    /// non-empty subset of its labels in the category).
    pub singleton: f64,
    /// Probability that a kept category's clause is drawn from another data point (uniform,
    /// independent of the query's point) instead of the query's point: 0 keeps the categories
    /// of a query as correlated as in the data, 1 makes them independent.
    pub fresh: f64,
    /// Share `μ` of the workload's second part, the *independent* queries `U` (0: base-derived
    /// queries only; [`crate::workload`]): conjunctions of one to three single-label clauses of
    /// distinct categories drawn by a uniform walk on the label co-occurrence structure, each
    /// with a query vector drawn independently. A [`QuerySample`] holds `U` as weighted queries
    /// ([`crate::workload::Independent`]) after its `n_queries` base-derived queries (weight
    /// `(1 − μ) / n_queries` each).
    pub uniform: f64,
    /// Number of co-occurring label pairs of different categories (reported; set from the
    /// counting pass when `uniform > 0`).
    pub contexts: u64,
}

impl Default for QueryModel {
    fn default() -> Self {
        Self {
            n_queries: 200_000,
            seed: 1,
            cap: 4096,
            singleton: 0.5,
            fresh: 0.0,
            uniform: 0.0,
            contexts: 0,
        }
    }
}

#[inline]
pub(crate) fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Whether the sorted `row` holds one of the sorted `clause`'s labels.
#[inline]
fn meets(row: &[u32], clause: &[u32]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < row.len() && j < clause.len() {
        match row[i].cmp(&clause[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

/// The island candidates of a sample of base-derived queries: for every query, the cost of
/// its best route without islands (`min(exact, base)` of the module documentation, with the
/// query's match count `M` and smallest clause `s_min`: `exact = s_min · pred + M · dist`,
/// `base` the post-filter route at rate `M / n`) and, for every single-label clause `ℓ` whose
/// island would be cheaper, that island's cost (the post-filter route on `|P_ℓ|` points at
/// rate `M / |P_ℓ|`). For the combination postings ([`crate::select`]), every base-derived
/// query's pairs of single-label clauses `{a}`, `{b}` whose labels can be in a stored pair
/// ([`crate::select::is_candidate_label`]), and its match count.
#[derive(Debug, Clone)]
pub struct QuerySample {
    pub model: QueryModel,
    /// Cost of each query's best route without islands: the `n_data` base-derived queries,
    /// then the independent queries (if any) in the order of [`Independent::queries`].
    pub base_ns: Vec<f64>,
    /// `(label, query, island cost)` of the cheaper islands, by label.
    recs: Vec<(u32, u32, f32)>,
    /// `(a, b, query)` for every pair `a < b` of candidate labels of two single-label clauses
    /// of a base-derived query, sorted.
    data_pairs: Vec<(u32, u32, u32)>,
    /// Match count `M` of each base-derived query.
    data_m: Vec<f64>,
    /// Base-derived queries (the first `n_data` of `base_ns`).
    pub n_data: usize,
    /// The workload's independent queries `U` (weighted).
    independent: Option<Arc<Independent>>,
    /// The unit costs the sample was priced with.
    costs: UnitCosts,
    n_points: usize,
}

/// One label of the `queries` selection.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct QueryGain {
    pub label: u32,
    pub size: usize,
    /// Sampled queries with this label as a single-label clause and a cheaper island.
    pub queries: u32,
    /// Expected saving per query (ns) of this island alone.
    pub alone_ns: f64,
    /// Expected saving per query added when selected (ns; 0 if not selected).
    pub marginal_ns: f64,
    pub bytes: u64,
    /// Position in the greedy order (`None`: not selected).
    pub order: Option<u32>,
}

/// Draws base-derived query `qi` of `model` (see [`QueryModel`]): its labels into `kl` (clause by
/// clause, each clause's labels increasing as in the source row) and the clauses' ranges of `kl`
/// into `clauses`; `cats` is scratch. Returns `false` (no query) for a point without labels.
pub(crate) fn draw_query(
    labels: &LabelStore,
    model: &QueryModel,
    qi: usize,
    kl: &mut Vec<u32>,
    clauses: &mut Vec<(usize, usize)>,
    cats: &mut Vec<(usize, usize, bool)>,
) -> bool {
    let n = labels.n_points();
    let h = mix(model.seed ^ mix(qi as u64));
    let p = (h % n as u64) as u32;
    let row = labels.row(p);
    if row.is_empty() {
        return false;
    }
    // The point's categories (contiguous in its sorted row: a category's label ids
    // are contiguous); each kept with probability ½ (one at random if none), its
    // clause one of the point's labels in it (probability `singleton`) or a random
    // non-empty subset of them (each kept with probability ½).
    cats.clear();
    let mut z = mix(h);
    let mut any = false;
    let mut a = 0;
    while a < row.len() {
        let cat = labels.category(row[a]);
        let mut b = a + 1;
        while b < row.len() && labels.category(row[b]) == cat {
            b += 1;
        }
        z = mix(z);
        let keep = z & 1 == 1;
        any |= keep;
        cats.push((a, b, keep));
        a = b;
    }
    if !any {
        z = mix(z);
        let f = (z % cats.len() as u64) as usize;
        cats[f].2 = true;
    }
    kl.clear();
    clauses.clear();
    for &(a, b, keep) in cats.iter() {
        if !keep {
            continue;
        }
        // The clause's source: the point's labels in the category, or (probability
        // `fresh`) those of another point drawn uniformly, if it has some.
        let mut src = &row[a..b];
        if model.fresh > 0.0 {
            z = mix(z);
            if ((z >> 11) as f64 / (1u64 << 53) as f64) < model.fresh {
                z = mix(z);
                let other = labels.row((z % n as u64) as u32);
                let r = labels.category_labels(labels.category(row[a]));
                let (x0, x1) = (
                    other.partition_point(|&l| l < r.start),
                    other.partition_point(|&l| l < r.end),
                );
                if x0 < x1 {
                    src = &other[x0..x1];
                }
            }
        }
        let start = kl.len();
        z = mix(z);
        let u = (z >> 11) as f64 / (1u64 << 53) as f64;
        if u < model.singleton {
            z = mix(z);
            kl.push(src[(z % src.len() as u64) as usize]);
        } else {
            for (x, &l) in src.iter().enumerate() {
                if x % 64 == 0 {
                    z = mix(z);
                }
                if (z >> (x % 64)) & 1 == 1 {
                    kl.push(l);
                }
            }
            if kl.len() == start {
                z = mix(z);
                kl.push(src[(z % src.len() as u64) as usize]);
            }
        }
        clauses.push((start, kl.len()));
    }
    true
}

/// The smallest clause of a drawn query (its index and its upper bound `s_min`, the summed
/// postings of its labels) and the query's estimated match count `M`: every `stride`-th member
/// of the smallest clause (each member once: counted at the first of the clause's labels in its
/// row) checked against the other clauses, scaled to `s_min` (at least 1).
pub(crate) fn query_matches(
    labels: &LabelStore,
    model: &QueryModel,
    kl: &[u32],
    clauses: &[(usize, usize)],
) -> (usize, u64, f64) {
    let size = |&(a, b): &(usize, usize)| -> u64 {
        kl[a..b].iter().map(|&l| labels.posting_len(l) as u64).sum()
    };
    let (jmin, s_min) = clauses
        .iter()
        .enumerate()
        .map(|(j, r)| (j, size(r)))
        .min_by_key(|&(_, s)| s)
        .expect("a clause");
    let m = if clauses.len() == 1 {
        s_min as f64
    } else {
        let (a0, b0) = clauses[jmin];
        let cl0 = &kl[a0..b0];
        let stride = (s_min as usize).div_ceil(model.cap).max(1);
        let (mut seen, mut hit, mut pos) = (0u64, 0u64, 0usize);
        for (x, &l) in cl0.iter().enumerate() {
            let post = labels.postings(l);
            let first = (stride - pos % stride) % stride;
            for &v in post.iter().skip(first).step_by(stride) {
                let vr = labels.row(v);
                if cl0[..x].iter().any(|&e| vr.binary_search(&e).is_ok()) {
                    continue;
                }
                seen += 1;
                if clauses
                    .iter()
                    .enumerate()
                    .all(|(j, &(a, b))| j == jmin || meets(vr, &kl[a..b]))
                {
                    hit += 1;
                }
            }
            pos += post.len();
        }
        (hit as f64 * s_min as f64 / seen.max(1) as f64).max(1.0)
    };
    (jmin, s_min, m)
}

impl QuerySample {
    /// Draws the base-derived queries and computes their cost terms (with up to `threads`
    /// threads; the result does not depend on the thread count). Without independent queries
    /// (see [`Self::build_mixed`]).
    pub fn build(
        labels: &LabelStore,
        c: &UnitCosts,
        k: usize,
        model: QueryModel,
        threads: usize,
    ) -> Self {
        let model = QueryModel {
            uniform: 0.0,
            ..model
        };
        Self::build_mixed(labels, c, k, model, None, threads)
    }

    /// [`Self::build`] with the workload's independent queries `U` ([`QueryModel::uniform`]
    /// `μ > 0`; [`crate::workload::independent`]): after the `n_queries` base-derived queries
    /// (weight `(1 − μ) / n_queries` each), `U`'s queries in their order (weight `μ` times their
    /// probability), each of single-label clauses with its match count `M`. Their vectors are
    /// independent of their labels ([`crate::workload::query_point`]); the costs do not depend
    /// on them.
    pub fn build_mixed(
        labels: &LabelStore,
        c: &UnitCosts,
        k: usize,
        model: QueryModel,
        independent: Option<Arc<Independent>>,
        threads: usize,
    ) -> Self {
        let independent = independent.filter(|u| model.uniform > 0.0 && !u.is_empty());
        let uq: &[crate::workload::WQuery] = independent.as_deref().map_or(&[], |u| &u.queries);
        let n = labels.n_points();
        let nf = n as f64;
        let chunk = 1024usize;
        let n_total = model.n_queries + uq.len();
        let n_chunks = n_total.div_ceil(chunk);
        let init = || {
            (
                Vec::<u32>::new(),
                Vec::<(usize, usize)>::new(),
                Vec::<(usize, usize, bool)>::new(),
            )
        };
        let parts = crate::parallel::par_map(n_chunks, threads, init, |ci, scratch| {
            let (kl, clauses, cats) = scratch;
            let mut base = Vec::new();
            let mut recs = Vec::new();
            let mut pairs = Vec::new();
            let mut ms = Vec::new();
            for qi in ci * chunk..((ci + 1) * chunk).min(n_total) {
                let (s_min, m) = if qi < model.n_queries {
                    if !draw_query(labels, &model, qi, kl, clauses, cats) {
                        base.push(0.0);
                        ms.push(0.0);
                        continue;
                    }
                    let (_, s_min, m) = query_matches(labels, &model, kl, clauses);
                    ms.push(m);
                    // The pairs a stored combination posting could serve the query through:
                    // single-label clauses (in distinct categories, by increasing label).
                    let singles = clauses
                        .iter()
                        .filter(|&&(a, b)| b - a == 1)
                        .map(|&(a, _)| kl[a])
                        .filter(|&l| crate::select::is_candidate_label(labels, l));
                    for (x, a) in singles.clone().enumerate() {
                        for b in singles.clone().skip(x + 1) {
                            pairs.push((a.min(b), a.max(b), qi as u32));
                        }
                    }
                    (s_min, m)
                } else {
                    let w = &uq[qi - model.n_queries];
                    kl.clear();
                    kl.extend_from_slice(&w.labels);
                    clauses.clear();
                    clauses.extend((0..w.labels.len()).map(|i| (i, i + 1)));
                    let s_min = w
                        .labels
                        .iter()
                        .map(|&l| labels.posting_len(l) as u64)
                        .min()
                        .unwrap_or(0);
                    (s_min, w.joint.max(1.0))
                };
                let (kl, clauses): (&Vec<u32>, &Vec<(usize, usize)>) = (kl, clauses);
                let rho_min = s_min as f64 / nf;
                let checks = if clauses.len() > 1 {
                    s_min as f64 * c.pred(rho_min)
                } else {
                    0.0
                };
                let exact = checks + m * c.dist(m / nf);
                let graph = postfilter_nominal(c, k, k, m / nf, n, 1).1;
                let cur = exact.min(graph);
                base.push(cur);
                for &(a, b) in clauses {
                    let mut nonempty = kl[a..b].iter().filter(|&&l| labels.posting_len(l) > 0);
                    if let (Some(&l), None) = (nonempty.next(), nonempty.next()) {
                        let s = labels.posting_len(l);
                        if s < MIN_SIZE || s >= n {
                            continue;
                        }
                        let isl = postfilter_nominal(c, k, k, (m / s as f64).min(1.0), s, 1).1;
                        if isl < cur {
                            recs.push((l, qi as u32, isl as f32));
                        }
                    }
                }
            }
            (base, recs, pairs, ms)
        });
        let mut base_ns = Vec::with_capacity(n_total);
        let mut recs = Vec::new();
        let mut data_pairs = Vec::new();
        let mut data_m = Vec::with_capacity(model.n_queries);
        for (b, r, p, m) in parts {
            base_ns.extend(b);
            recs.extend(r);
            data_pairs.extend(p);
            data_m.extend(m);
        }
        recs.sort_unstable_by_key(|&(l, q, _)| (l, q));
        data_pairs.sort_unstable();
        Self {
            model,
            base_ns,
            recs,
            data_pairs,
            data_m,
            n_data: model.n_queries,
            independent,
            costs: c.clone(),
            n_points: n,
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

    /// The workload's mean of per-query values `x` (`(1 − μ)` times the base-derived queries'
    /// mean plus `μ` times the independent queries' weighted mean).
    pub fn weighted_mean(&self, x: &[f64]) -> f64 {
        let (d, u) = x.split_at(self.n_data.min(x.len()));
        let md = d.iter().sum::<f64>() / self.n_data.max(1) as f64;
        match &self.independent {
            None => md,
            Some(ind) => {
                let mu = self.model.uniform;
                let mu_u: f64 = ind.queries.iter().zip(u).map(|(q, v)| q.weight * v).sum();
                (1.0 - mu) * md + mu * mu_u
            }
        }
    }

    /// Workload weight of query `q`.
    pub fn weight(&self, q: usize) -> f64 {
        let mu = self.mu();
        if q < self.n_data {
            (1.0 - mu) / self.n_data.max(1) as f64
        } else {
            let ind = self.independent.as_deref().expect("an independent query");
            mu * ind.queries[q - self.n_data].weight
        }
    }

    /// Part of the workload query `q` comes from.
    pub fn part(&self, q: usize) -> Part {
        if q < self.n_data {
            Part::Data
        } else {
            Part::Independent
        }
    }

    /// The data point whose vector query `q` searches with ([`query_point`]).
    pub fn point(&self, q: usize) -> u32 {
        query_point(&self.model, self.n_points, self.n_data, q)
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

    /// Cost (ns) of query `q` served by a stored pair of `joint` points among the labels of its
    /// single-label clauses: the pair's list read with the other clauses checked, the matches'
    /// distances ([`pair_plan_ns`]).
    pub fn pair_cost(&self, q: usize, joint: u32) -> Option<f64> {
        let m = match q.checked_sub(self.n_data) {
            None => *self.data_m.get(q)?,
            Some(i) => self.independent.as_deref()?.queries.get(i)?.joint,
        };
        let m = m.min(f64::from(joint)).max(1.0);
        Some(pair_plan_ns(&self.costs, self.n_points, joint, m))
    }

    /// The base-derived queries (sample indices) with single-label clauses `{a}` and `{b}`
    /// (`a < b`, both [`crate::select::is_candidate_label`]).
    pub fn data_pair_queries(&self, a: u32, b: u32) -> impl Iterator<Item = usize> + '_ {
        let lo = self
            .data_pairs
            .partition_point(|&(x, y, _)| (x, y) < (a, b));
        self.data_pairs[lo..]
            .iter()
            .take_while(move |&&(x, y, _)| (x, y) == (a, b))
            .map(|&(_, _, q)| q as usize)
    }

    /// Number of `(pair, base-derived query)` records ([`Self::data_pair_queries`]).
    pub fn n_data_pairs(&self) -> usize {
        self.data_pairs.len()
    }

    /// Each query's cost with the islands of `islands` (sorted) and the pairs `(a, b, joint)`
    /// built: its cheapest route (tests).
    #[cfg(test)]
    pub(crate) fn costs_with(&self, islands: &[u32], pairs: &[(u32, u32, u32)]) -> Vec<f64> {
        let mut x = self.base_ns.clone();
        for &(l, q, isl) in &self.recs {
            if islands.binary_search(&l).is_ok() {
                let c = &mut x[q as usize];
                *c = c.min(f64::from(isl));
            }
        }
        for &(a, b, joint) in pairs {
            let ind = self.independent();
            let qs = self.data_pair_queries(a, b).chain(
                ind.into_iter()
                    .flat_map(|(u, first)| u.pair_queries(a, b).map(move |i| first + i)),
            );
            for q in qs {
                let c = self.pair_cost(q, joint).expect("a sample query");
                x[q] = x[q].min(c);
            }
        }
        x
    }

    /// Greedy selection within `budget_bytes` (islands of degree bound `r`): repeatedly the
    /// label with the largest expected saving per byte given the labels already taken (a
    /// query's cost is that of its cheapest route, the selected islands included), skipping
    /// labels that no longer fit; candidates limited by `keep`. Returns the labels in label-id
    /// order and every candidate's terms.
    pub fn select(
        &self,
        labels: &LabelStore,
        budget_bytes: u64,
        r: usize,
        keep: impl Fn(u32) -> bool,
    ) -> (Vec<u32>, Vec<QueryGain>) {
        let mut gr = IslandGreedy::new(self, labels, r, keep);
        let mut used = 0u64;
        let mut out = Vec::new();
        while let Some((_, x)) = gr.peek() {
            let bytes = gr.terms[x].bytes;
            if used + bytes > budget_bytes {
                gr.skip(x);
                continue;
            }
            used += bytes;
            gr.take(x);
            out.push(gr.terms[x].label);
        }
        out.sort_unstable();
        (out, gr.terms)
    }

    /// Expected cost per query (ns) without islands (the workload's weighted mean).
    pub fn mean_base_ns(&self) -> f64 {
        self.weighted_mean(&self.base_ns)
    }
}

/// `f64` with a total order (for the selection heap).
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrdF64(f64);

impl Eq for OrdF64 {}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

impl Ord for OrdF64 {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0)
    }
}

/// Step-wise lazy greedy over the island candidates of a [`QuerySample`]: candidates by
/// expected saving per query per byte given the islands already taken (a query keeps its
/// cheapest route; savings only shrink as islands are taken, so a candidate is re-evaluated
/// only when it reaches the top). Used by [`QuerySample::select`] and by the unified selector
/// ([`crate::select`]).
pub struct IslandGreedy<'s> {
    sample: &'s QuerySample,
    cur: Vec<f64>,
    /// Candidate label and its range of the sample's records.
    cands: Vec<(u32, usize, usize)>,
    /// Every candidate's terms (`marginal_ns`, `order` set when taken).
    pub terms: Vec<QueryGain>,
    heap: std::collections::BinaryHeap<(OrdF64, std::cmp::Reverse<u32>, usize)>,
    taken: u32,
}

impl<'s> IslandGreedy<'s> {
    /// Candidates: the labels with island records accepted by `keep`, islands of degree bound `r`.
    pub fn new(
        sample: &'s QuerySample,
        labels: &LabelStore,
        r: usize,
        keep: impl Fn(u32) -> bool,
    ) -> Self {
        let recs = &sample.recs;
        let mut cands: Vec<(u32, usize, usize)> = Vec::new();
        let mut i = 0;
        while i < recs.len() {
            let l = recs[i].0;
            let mut j = i;
            while j < recs.len() && recs[j].0 == l {
                j += 1;
            }
            if keep(l) {
                cands.push((l, i, j));
            }
            i = j;
        }
        let mut g = Self {
            sample,
            cur: sample.base_ns.clone(),
            cands,
            terms: Vec::new(),
            heap: std::collections::BinaryHeap::new(),
            taken: 0,
        };
        g.terms = (0..g.cands.len())
            .map(|x| {
                let (l, a, b) = g.cands[x];
                let s = labels.posting_len(l);
                QueryGain {
                    label: l,
                    size: s,
                    queries: (b - a) as u32,
                    alone_ns: g.gain(x),
                    marginal_ns: 0.0,
                    bytes: island_bytes(s, r),
                    order: None,
                }
            })
            .collect();
        g.heap = g
            .terms
            .iter()
            .enumerate()
            .filter(|(_, t)| t.alone_ns > 0.0)
            .map(|(x, t)| {
                (
                    OrdF64(t.alone_ns / t.bytes as f64),
                    std::cmp::Reverse(t.label),
                    x,
                )
            })
            .collect();
        g
    }

    /// Expected saving per query (ns) of candidate `x` given the items taken (the workload's
    /// weighted mean of its per-query savings).
    fn gain(&self, x: usize) -> f64 {
        let (_, a, b) = self.cands[x];
        let s = self.sample;
        let nd = s.n_data as u32;
        let recs = &s.recs[a..b];
        // Records by query: the base-derived queries first, then the independent ones.
        let split = recs.partition_point(|&(_, q, _)| q < nd);
        let saving = |r: &[(u32, u32, f32)]| {
            r.iter()
                .map(|&(_, q, isl)| (self.cur[q as usize] - f64::from(isl)).max(0.0))
                .sum::<f64>()
        };
        let sd = saving(&recs[..split]) / s.n_data.max(1) as f64;
        match s.independent.as_deref() {
            None => sd,
            Some(ind) => {
                let mu = s.model.uniform;
                let su: f64 = recs[split..]
                    .iter()
                    .map(|&(_, q, isl)| {
                        ind.queries[q as usize - s.n_data].weight
                            * (self.cur[q as usize] - f64::from(isl)).max(0.0)
                    })
                    .sum();
                (1.0 - mu) * sd + mu * su
            }
        }
    }

    /// The best remaining candidate (its current saving per query per byte and its index),
    /// re-evaluated; candidates whose saving fell to 0 are dropped.
    pub fn peek(&mut self) -> Option<(f64, usize)> {
        loop {
            let &(OrdF64(key), lab, x) = self.heap.peek()?;
            let g = self.gain(x);
            let now = g / self.terms[x].bytes as f64;
            if now < key * (1.0 - 1e-12) {
                self.heap.pop();
                if g > 0.0 {
                    self.heap.push((OrdF64(now), lab, x));
                }
                continue;
            }
            if g <= 0.0 {
                self.heap.pop();
                continue;
            }
            return Some((now, x));
        }
    }

    /// Takes candidate `x` (the one [`Self::peek`] returned): its island serves the queries it
    /// makes cheaper. Returns its marginal saving per query (ns).
    pub fn take(&mut self, x: usize) -> f64 {
        let top = self.heap.pop().map(|e| e.2);
        debug_assert_eq!(top, Some(x), "take the peeked candidate");
        let g = self.gain(x);
        self.terms[x].marginal_ns = g;
        self.terms[x].order = Some(self.taken);
        self.taken += 1;
        let (_, a, b) = self.cands[x];
        for &(_, q, isl) in &self.sample.recs[a..b] {
            let c = &mut self.cur[q as usize];
            *c = c.min(f64::from(isl));
        }
        g
    }

    /// Drops candidate `x` (the one [`Self::peek`] returned) without taking it.
    pub fn skip(&mut self, x: usize) {
        let top = self.heap.pop().map(|e| e.2);
        debug_assert_eq!(top, Some(x), "skip the peeked candidate");
    }

    /// Expected cost per query (ns) with the items taken.
    pub fn mean_cost_ns(&self) -> f64 {
        self.sample.weighted_mean(&self.cur)
    }

    /// The sample's independent queries and the index of the first.
    pub fn independent(&self) -> Option<(&Independent, usize)> {
        self.sample.independent()
    }

    /// The sample (its base-derived queries' pairs of single-label clauses).
    pub fn sample(&self) -> &'s QuerySample {
        self.sample
    }

    /// Cost of sample query `q` when a stored pair of `joint` points among its labels serves it.
    pub fn pair_cost(&self, q: usize, joint: u32) -> Option<f64> {
        self.sample.pair_cost(q, joint)
    }

    /// Current cost of sample query `q` (its cheapest route with the items taken).
    pub fn query_cost(&self, q: usize) -> f64 {
        self.cur[q]
    }

    /// Workload weight of sample query `q`.
    pub fn query_weight(&self, q: usize) -> f64 {
        self.sample.weight(q)
    }

    /// Another kind's item serves sample query `q` at `cost`.
    pub fn serve(&mut self, q: usize, cost: f64) {
        let c = &mut self.cur[q];
        *c = c.min(cost);
    }
}

/// Per-label sub-graphs (see the module documentation).
#[derive(Debug, Clone, PartialEq)]
pub struct IslandIndex {
    n_points: usize,
    dict_hash: u64,
    r: u32,
    l_build: u32,
    /// Island of every label (`NONE` without).
    island_of: Vec<u32>,
    /// Per island: its label and the local id of its start (medoid).
    labels: Vec<u32>,
    medoids: Vec<u32>,
    /// Island `i`'s rows: `offsets[node_base[i]..=node_base[i] + s_i]` (edge offsets).
    node_base: Vec<u64>,
    offsets: Vec<u64>,
    /// Local ids of the neighbors.
    edges: Vec<u32>,
}

/// Size and build statistics.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct IslandStats {
    pub islands: usize,
    pub members: u64,
    pub edges: u64,
    pub bytes: u64,
    pub build_s: f64,
    pub n_dist: u64,
    /// Members unreachable from their island's medoid after the build, given an edge from it.
    pub patched: u64,
}

/// The rows of an island from the builder's graph over its `s` members (local ids; the
/// builder's auxiliary start nodes, copies of the medoid, from `s` on): the auxiliary nodes
/// are merged into the medoid (edges to them point to it, their out-edges are added to its
/// row), then every member still unreachable from the medoid gets an edge from it (the
/// medoid's row may exceed the degree bound). Returns the row offsets, the edges and the
/// number of members patched.
fn merge_start(g: &crate::graph::Graph, s: usize, medoid: u32) -> (Vec<u32>, Vec<u32>, u64) {
    let aux_out: Vec<u32> = (s..g.n_nodes())
        .flat_map(|a| g.neighbors(a as u32).iter().copied())
        .filter(|&u| (u as usize) < s)
        .collect();
    let mut rows: Vec<Vec<u32>> = (0..s as u32)
        .map(|v| {
            let mut row: Vec<u32> = Vec::with_capacity(g.neighbors(v).len());
            let extra = if v == medoid { &aux_out[..] } else { &[][..] };
            for &u in g.neighbors(v).iter().chain(extra) {
                let u = if (u as usize) < s { u } else { medoid };
                if u != v && !row.contains(&u) {
                    row.push(u);
                }
            }
            row
        })
        .collect();
    let mut seen = vec![false; s];
    let mut stack = vec![medoid];
    seen[medoid as usize] = true;
    while let Some(v) = stack.pop() {
        for &u in &rows[v as usize] {
            if !seen[u as usize] {
                seen[u as usize] = true;
                stack.push(u);
            }
        }
    }
    let unreachable: Vec<u32> = (0..s as u32).filter(|&v| !seen[v as usize]).collect();
    let patched = unreachable.len() as u64;
    rows[medoid as usize].extend(unreachable);
    let mut offsets = Vec::with_capacity(s + 1);
    let mut edges = Vec::new();
    for row in rows {
        offsets.push(edges.len() as u32);
        edges.extend(row);
    }
    offsets.push(edges.len() as u32);
    (offsets, edges, patched)
}

impl IslandIndex {
    /// Builds the islands of `selected` (label ids of `labels`, increasing) over `base`: each
    /// island with `params` (its rule must be label-blind; one thread per island), `threads`
    /// islands at a time. The result does not depend on the thread count. Returns the per-island
    /// build seconds (in `selected` order) with the statistics.
    pub fn build<T: BuildElement>(
        base: &Matrix<T>,
        labels: &LabelStore,
        selected: &[u32],
        params: &BuildParams,
        threads: usize,
    ) -> Result<(Self, IslandStats, Vec<f64>)> {
        ensure!(
            base.nrows() == labels.n_points(),
            "the base has {} rows, the labels {} points",
            base.nrows(),
            labels.n_points()
        );
        ensure!(
            !params.rule.needs_labels(),
            "islands use a label-blind rule"
        );
        ensure!(
            selected.windows(2).all(|w| w[0] < w[1]),
            "the selected labels must increase"
        );
        let t0 = Instant::now();
        let d = base.ncols();
        let params = BuildParams {
            threads: 1,
            ..params.clone()
        };
        type Built = Result<(u32, Vec<u32>, Vec<u32>, u64, u64, f64)>;
        let built: Vec<Built> =
            crate::parallel::par_map(selected.len(), threads, Vec::new, |x, buf: &mut Vec<T>| {
                let l = selected[x];
                let members = labels.postings(l);
                let s = members.len();
                ensure!(s >= 2, "label {l} has fewer than two members");
                let t = Instant::now();
                buf.clear();
                for &p in members {
                    buf.extend_from_slice(base.row(p as usize));
                }
                let m = Matrix::from_slice(buf, s, d);
                let out = build::build(&m, None, &params)
                    .with_context(|| format!("building the island of label {l}"))?;
                let (offsets, edges, patched) = merge_start(&out.graph, s, out.stats.medoid);
                let n_dist =
                    out.stats.n_dist_search + out.stats.n_dist_prune + out.stats.n_dist_reprune;
                Ok((
                    out.stats.medoid,
                    offsets,
                    edges,
                    n_dist,
                    patched,
                    t.elapsed().as_secs_f64(),
                ))
            });
        let mut ix = Self {
            n_points: labels.n_points(),
            dict_hash: labels.dict_hash(),
            r: params.rule.degree() as u32,
            l_build: params.l_build as u32,
            island_of: vec![NONE; labels.n_labels()],
            labels: Vec::new(),
            medoids: Vec::new(),
            node_base: vec![0],
            offsets: Vec::new(),
            edges: Vec::new(),
        };
        let mut stats = IslandStats::default();
        let mut secs = Vec::with_capacity(selected.len());
        for (&l, b) in selected.iter().zip(built) {
            let (medoid, offsets, edges, n_dist, patched, t) = b?;
            let base_edge = ix.edges.len() as u64;
            ix.island_of[l as usize] = ix.labels.len() as u32;
            ix.labels.push(l);
            ix.medoids.push(medoid);
            ix.offsets
                .extend(offsets.iter().map(|&o| base_edge + u64::from(o)));
            ix.edges.extend_from_slice(&edges);
            ix.node_base.push(ix.offsets.len() as u64);
            stats.members += (offsets.len() - 1) as u64;
            stats.n_dist += n_dist;
            stats.patched += patched;
            secs.push(t);
        }
        stats.islands = ix.labels.len();
        stats.edges = ix.edges.len() as u64;
        stats.bytes = ix.bytes();
        stats.build_s = t0.elapsed().as_secs_f64();
        Ok((ix, stats, secs))
    }

    /// The islands of the labels in `keep` (others dropped).
    pub fn subset(&self, keep: &[u32]) -> Self {
        let mut out = Self {
            n_points: self.n_points,
            dict_hash: self.dict_hash,
            r: self.r,
            l_build: self.l_build,
            island_of: vec![NONE; self.island_of.len()],
            labels: Vec::new(),
            medoids: Vec::new(),
            node_base: vec![0],
            offsets: Vec::new(),
            edges: Vec::new(),
        };
        for i in 0..self.labels.len() {
            let l = self.labels[i];
            if !keep.contains(&l) {
                continue;
            }
            let (b, e) = (self.node_base[i] as usize, self.node_base[i + 1] as usize);
            let (lo, hi) = (self.offsets[b] as usize, self.offsets[e - 1] as usize);
            let shift = out.edges.len() as u64;
            out.island_of[l as usize] = out.labels.len() as u32;
            out.labels.push(l);
            out.medoids.push(self.medoids[i]);
            out.offsets
                .extend(self.offsets[b..e].iter().map(|&o| o - lo as u64 + shift));
            out.edges.extend_from_slice(&self.edges[lo..hi]);
            out.node_base.push(out.offsets.len() as u64);
        }
        out
    }

    pub fn n_points(&self) -> usize {
        self.n_points
    }

    pub fn n_islands(&self) -> usize {
        self.labels.len()
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn n_members(&self) -> usize {
        self.offsets.len() - self.labels.len()
    }

    pub fn degree_bound(&self) -> usize {
        self.r as usize
    }

    /// Bytes held (every array).
    pub fn bytes(&self) -> u64 {
        (4 * self.island_of.len()
            + 4 * self.labels.len()
            + 4 * self.medoids.len()
            + 8 * self.node_base.len()
            + 8 * self.offsets.len()
            + 4 * self.edges.len()) as u64
    }

    /// The island of label `l`, if any.
    #[inline]
    pub fn island(&self, l: u32) -> Option<usize> {
        let i = self.island_of[l as usize];
        (i != NONE).then_some(i as usize)
    }

    /// Label of island `i`.
    #[inline]
    pub fn label(&self, i: usize) -> u32 {
        self.labels[i]
    }

    /// Members of island `i`.
    #[inline]
    pub fn size(&self, i: usize) -> usize {
        (self.node_base[i + 1] - self.node_base[i]) as usize - 1
    }

    /// Start (local id) of island `i`.
    #[inline]
    pub fn medoid(&self, i: usize) -> u32 {
        self.medoids[i]
    }

    /// Neighbors (local ids) of local node `v` of island `i`.
    #[inline]
    pub fn neighbors(&self, i: usize, v: u32) -> &[u32] {
        let r = self.node_base[i] as usize + v as usize;
        &self.edges[self.offsets[r] as usize..self.offsets[r + 1] as usize]
    }

    /// Prefetches the row of local node `v` of island `i`.
    #[inline]
    pub fn prefetch_row(&self, i: usize, v: u32) {
        let r = self.node_base[i] as usize + v as usize;
        let at = self.offsets[r] as usize;
        crate::kernel::prefetch(self.edges.as_ptr().wrapping_add(at));
    }

    /// Checks that the index belongs to `labels` (dictionary, counts, island sizes).
    pub fn check(&self, labels: &LabelStore) -> Result<()> {
        ensure!(
            self.dict_hash == labels.dict_hash()
                && self.island_of.len() == labels.n_labels()
                && self.n_points == labels.n_points(),
            "the islands refer to another label store"
        );
        for (i, &l) in self.labels.iter().enumerate() {
            ensure!(
                self.size(i) == labels.posting_len(l),
                "island {i} has {} members, label {l} {}",
                self.size(i),
                labels.posting_len(l)
            );
        }
        Ok(())
    }

    /// Writes `islands.flab`.
    pub fn write(&self, path: &Path) -> Result<()> {
        self.write_with(path, None)
    }

    /// [`Self::write`] with a provenance section ([`crate::provenance`]) when given.
    pub fn write_with(&self, path: &Path, prov: Option<&Provenance>) -> Result<()> {
        self.write_to_with(container::create(path)?, prov)
            .with_context(|| format!("writing {}", path.display()))
    }

    pub fn write_to<W: Write>(&self, w: W) -> Result<()> {
        self.write_to_with(w, None)
    }

    /// [`Self::write_to`] with a provenance section when given.
    pub fn write_to_with<W: Write>(&self, w: W, prov: Option<&Provenance>) -> Result<()> {
        let params = [
            self.n_points as u64,
            self.island_of.len() as u64,
            self.dict_hash,
            self.labels.len() as u64,
            self.offsets.len() as u64,
            self.edges.len() as u64,
            u64::from(self.r),
            u64::from(self.l_build),
        ];
        let prov = prov.map(Provenance::section);
        let mut sections: Vec<&[u8]> = vec![
            bytemuck::cast_slice(&self.island_of),
            bytemuck::cast_slice(&self.labels),
            bytemuck::cast_slice(&self.medoids),
            bytemuck::cast_slice(&self.node_base),
            bytemuck::cast_slice(&self.offsets),
            bytemuck::cast_slice(&self.edges),
        ];
        sections.extend(prov.as_deref());
        container::write(w, MAGIC, VERSION, &params, &sections)
    }

    /// Reads and validates `islands.flab` (structure only; [`Self::check`] compares it with a
    /// label store).
    pub fn read(path: &Path) -> Result<Self> {
        container::Reader::open(path, MAGIC)
            .and_then(Self::load)
            .with_context(|| format!("loading islands from {}", path.display()))
    }

    pub fn read_from<R: Read + Seek>(r: R) -> Result<Self> {
        Self::load(container::Reader::new(r, MAGIC)?)
    }

    fn load<R: Read + Seek>(mut r: container::Reader<R>) -> Result<Self> {
        ensure!(
            r.version() == VERSION,
            "unsupported version {}",
            r.version()
        );
        let [n_points, n_labels, dict_hash, n_islands, n_offsets, n_edges, deg, l_build] =
            *r.params();
        let ni = n_islands as usize;
        let island_of: Vec<u32> = r.section(0, n_labels as usize)?;
        let labels: Vec<u32> = r.section(1, ni)?;
        let medoids: Vec<u32> = r.section(2, ni)?;
        let node_base: Vec<u64> = r.section(3, ni + 1)?;
        let offsets: Vec<u64> = r.section(4, n_offsets as usize)?;
        let edges: Vec<u32> = r.section(5, n_edges as usize)?;
        ensure!(
            node_base.first() == Some(&0)
                && node_base.last() == Some(&n_offsets)
                && node_base.windows(2).all(|w| w[1] >= w[0] + 2),
            "inconsistent island rows"
        );
        ensure!(
            offsets.windows(2).all(|w| w[0] <= w[1])
                && offsets.last().is_none_or(|&o| o == n_edges),
            "inconsistent edge offsets"
        );
        for i in 0..ni {
            let (b, e) = (node_base[i] as usize, node_base[i + 1] as usize);
            let s = (e - b - 1) as u32;
            ensure!(
                island_of.get(labels[i] as usize) == Some(&(i as u32)) && medoids[i] < s,
                "inconsistent island {i}"
            );
            let es = &edges[offsets[b] as usize..offsets[e - 1] as usize];
            ensure!(
                es.iter().all(|&u| u < s),
                "island {i} has an edge out of range"
            );
        }
        Ok(Self {
            n_points: n_points as usize,
            dict_hash,
            r: deg as u32,
            l_build: l_build as u32,
            island_of,
            labels,
            medoids,
            node_base,
            offsets,
            edges,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build::PruneRule, labels::LabelStoreBuilder};

    #[test]
    fn islands_hold_their_labels_members_and_read_back() {
        // 600 points on a line (1-d u8 vectors cycling), labels by residue.
        let n = 600usize;
        let mut lb = LabelStoreBuilder::new();
        let mut data = Vec::new();
        for p in 0..n {
            data.push((p % 251) as u8);
            let r = format!("r{}", p % 3);
            let half = if p < 300 { "lo" } else { "hi" };
            lb.push_point([("k", r.as_str()), ("h", half)]).unwrap();
        }
        let labels = lb.finish().unwrap();
        let base = Matrix::from_slice(&data, n, 1);
        let id = labels.name_index();
        let sel = vec![id["r0"].min(id["lo"]), id["r0"].max(id["lo"])];
        let params = BuildParams {
            rule: PruneRule::Alpha { r: 8, alpha: 1.2 },
            l_build: 16,
            threads: 1,
            ..BuildParams::default()
        };
        let (ix, st, secs) = IslandIndex::build(&base, &labels, &sel, &params, 2).unwrap();
        assert_eq!(secs.len(), 2);
        assert_eq!(st.islands, 2);
        ix.check(&labels).unwrap();
        for &l in &sel {
            let i = ix.island(l).unwrap();
            assert_eq!(ix.label(i), l);
            assert_eq!(ix.size(i), labels.posting_len(l));
            for v in 0..ix.size(i) as u32 {
                assert!(ix
                    .neighbors(i, v)
                    .iter()
                    .all(|&u| (u as usize) < ix.size(i)));
            }
        }
        assert!(ix.island(id["r1"]).is_none());
        // A subset keeps the islands it names, identically.
        let one = ix.subset(&[id["lo"]]);
        assert_eq!(one.n_islands(), 1);
        let (a, b) = (ix.island(id["lo"]).unwrap(), one.island(id["lo"]).unwrap());
        for v in 0..ix.size(a) as u32 {
            assert_eq!(ix.neighbors(a, v), one.neighbors(b, v));
        }
        let mut buf = Vec::new();
        ix.write_to(&mut buf).unwrap();
        assert_eq!(
            IslandIndex::read_from(std::io::Cursor::new(buf)).unwrap(),
            ix
        );
    }

    #[test]
    fn selection_prefers_gain_per_byte_within_the_budget() {
        let g = |label, size, gain_ns| Gain {
            label,
            size,
            exact_ns: 0.0,
            base_ns: 0.0,
            island_ns: 0.0,
            gain_ns,
            bytes: island_bytes(size, 32),
        };
        let gains = [
            g(0, 100, 1000.0),
            g(1, 1000, 1000.0),
            g(2, 100, -5.0),
            g(3, 200, 3000.0),
        ];
        // Gain per byte: 3 > 0 > 1; 2 loses.
        assert_eq!(select(&gains, island_bytes(300, 32)), vec![0, 3]);
        assert_eq!(select(&gains, u64::MAX), vec![0, 1, 3]);
        assert!(select(&gains, 10).is_empty());
        // The data prior ranks by gain: 3 > 0 = 1 (ties by label id); with room for 1,200
        // members it takes 3 and then 0 (1 no longer fits).
        assert_eq!(
            select_with(&gains, island_bytes(1200, 32), Prior::Data),
            vec![0, 3]
        );
        let big = [g(0, 100, 1000.0), g(1, 1000, 5000.0)];
        assert_eq!(select(&big, island_bytes(1000, 32)), vec![0]);
        assert_eq!(
            select_with(&big, island_bytes(1000, 32), Prior::Data),
            vec![1]
        );
    }
    #[test]
    fn base_derived_queries_select_greedily_within_the_budget_and_account_their_savings() {
        // 20,000 points in three categories: `m` (m0 on 0..2,000 and m1 on 2,000..4,000),
        // `d` (d0 on even points, d1 on odd ones) and `g` (g0, g1 on two blocks of 200 points
        // inside m0).
        let n = 20_000u32;
        let mut b = LabelStoreBuilder::new();
        for p in 0..n {
            let mut v = vec![("d", if p % 2 == 0 { "d0" } else { "d1" })];
            if p < 2000 {
                v.push(("m", "m0"));
            } else if p < 4000 {
                v.push(("m", "m1"));
            }
            if (100..300).contains(&p) {
                v.push(("g", "g0"));
            } else if (500..700).contains(&p) {
                v.push(("g", "g1"));
            }
            b.push_point(v).unwrap();
        }
        let s = b.finish().unwrap();
        let id = s.name_index();
        let c = UnitCosts::default();
        let model = QueryModel {
            n_queries: 20_000,
            seed: 7,
            cap: 512,
            singleton: 0.5,
            fresh: 0.0,
            ..Default::default()
        };
        let one = QuerySample::build(&s, &c, 10, model, 1);
        let three = QuerySample::build(&s, &c, 10, model, 3);
        assert_eq!(
            one.base_ns, three.base_ns,
            "independent of the thread count"
        );
        assert_eq!(one.recs, three.recs);
        // The first label taken has the largest saving per byte; the budget is respected.
        let budget = island_bytes(2000, 32);
        let (sel, terms) = one.select(&s, budget, 32, |_| true);
        let best = terms
            .iter()
            .max_by(|a, b| (a.alone_ns / a.bytes as f64).total_cmp(&(b.alone_ns / b.bytes as f64)))
            .unwrap();
        assert_eq!(
            terms.iter().find(|t| t.order == Some(0)).map(|t| t.label),
            Some(best.label),
            "{terms:?}"
        );
        let used: u64 = sel
            .iter()
            .map(|&l| island_bytes(s.posting_len(l), 32))
            .sum();
        assert!(!sel.is_empty() && used <= budget, "{terms:?}");
        // Everything fits: the savings add up to the drop in the expected cost per query.
        let (all, terms) = one.select(&s, u64::MAX, 32, |_| true);
        assert!(all.contains(&id["m0"]) && all.contains(&id["m1"]));
        let mut cur = one.base_ns.clone();
        for &(l, q, isl) in &one.recs {
            if all.binary_search(&l).is_ok() {
                cur[q as usize] = cur[q as usize].min(f64::from(isl));
            }
        }
        let drop = (one.base_ns.iter().sum::<f64>() - cur.iter().sum::<f64>()) / 20_000.0;
        let summed: f64 = terms.iter().map(|t| t.marginal_ns).sum();
        assert!(
            (drop - summed).abs() <= 1e-6 * drop.max(1.0),
            "{drop} vs {summed}"
        );
        // Candidates can be limited; the budget is respected.
        let (none, _) = one.select(&s, u64::MAX, 32, |l| l == id["g0"]);
        assert!(none.iter().all(|&l| l == id["g0"]));
        assert!(one.select(&s, 10, 32, |_| true).0.is_empty());
    }
}
