/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! One selector for the auxiliary structures: combination postings (joint postings of label
//! pairs, [`crate::pairs`]) and per-label sub-graphs (islands, [`crate::islands`]), chosen
//! together by expected saving per query per byte on one workload of base-derived queries, under
//! one memory budget, with the unit costs of the cost model. Categories are read as the label
//! store's groups only; no query log is read.
//!
//! **Workload** ([`crate::workload`]; one definition for every kind of candidate). A fixed
//! mixture: with weight `1 − μ`, **data-like** queries (`D`), the base-derived queries of the
//! islands' `queries` selection — a data point drawn uniformly, each of its categories kept with
//! probability ½, a kept category's clause one of the point's labels in it (probability
//! `singleton`) or a random subset of them (optionally, with probability `fresh`, drawn from
//! another data point), the query vector that point's; with weight `μ`, **independent** queries
//! (`U`) — conjunctions of `k` single-label clauses of distinct categories, `k` uniform in
//! `1..=min(3, C)`, drawn by a uniform walk on the label co-occurrence structure (the first label
//! uniformly within a uniformly drawn category, each next one uniformly among the distinct
//! labels of unused categories that co-occur with all labels drawn so far), with a query vector
//! drawn independently of the labels. (`μ = ½` hedges between "queries look like the data" and
//! "any label combination, from anywhere" and is not tuned; `μ = 0` is the base-derived workload
//! alone.) An island serves a query through a single-label clause; a pair `(a, b)` through two
//! single-label clauses `{a}`, `{b}` in different categories (the `pair` plan's exact cover).
//!
//! **Islands** are valued on a sample of this workload ([`QuerySample::build_mixed`]:
//! `n_queries` base-derived queries, weight `(1 − μ)/n_queries` each, and `U` as weighted
//! queries — its one- and two-label parts exactly, its three-label part as `n_queries` walks;
//! [`crate::workload::independent`]; per query the cost of its best route without auxiliary
//! structures and of the island of each single-label clause). The costs do not depend on the
//! query vector; [`crate::workload::Walk::draw`] draws `U` i.i.d. with its vectors (seeded),
//! for measurements on concrete queries.
//!
//! **Pairs** are too many, and each drawn too rarely, for a sample of base-derived queries (a
//! pair whose joint posting holds `j` points is drawn with probability ∝ `j/n`), so their value
//! on `D` is the expectation in closed form: the probability that a query's clauses in the
//! categories of `a` and `b` are `{a}` and `{b}`, `(1 − μ)·P_ab` with
//!
//! ```text
//! P_ab = ¼·[(1 − fresh)²·g_ab + (1 − (1 − fresh)²)·f_a·f_b]
//! g_ab = (1/n)·Σ_{p ∈ P_a ∩ P_b} w(L_p(a))·w(L_p(b)),   f_a = (1/n)·Σ_{p ∈ P_a} w(L_p(a))
//! w(L) = singleton/L + (1 − singleton)·2^(−L)·(1 + 1/L)
//! ```
//!
//! (both categories kept: ¼; `L_p(x)`: the labels of point `p` in the category of `x`; `w(L)`:
//! the probability that a kept category's clause is a given one of its `L` labels alone, drawn
//! as a single label or as a random subset of one; the other categories may be kept too), times
//! the saving of the `pair` plan on `{a} ∧ {b}` (`s = min(|P_a|, |P_b|)`, `M = j`):
//!
//! ```text
//! saving = min(s·pred(s/n) + M·dist(M/n), base(M/n)) − (gallop_cold + j·pred(j/n) + M·dist(M/n))
//! ```
//!
//! (the scan of the smaller clause or the post-filter route `base` at rate `M/n`, against the
//! joint posting read with a residual clause checked), at `4·(j + 3)` bytes; on `U` a pair is
//! valued on every sample query holding both its labels (two or three labels:
//! [`pair_plan_ns`] — its list read with the other clauses checked, the matches' distances),
//! against that query's current cost, which a taken island or pair lowers (one cost per query,
//! shared by the kinds; of several stored pairs the cheapest counts). Only pairs that
//! co-occur are candidates (a query of the workload has a match), of labels of at least
//! [`pairs::T_MIN`] points. On `D`, a pair is valued against the routes without auxiliary
//! structures (not modelled: a base-derived query served by both kinds).
//!
//! **Selection.** One greedy over both kinds by expected saving per query per byte. It stops
//! ([`Stop::Elasticity`]) at the first item whose relative saving of the mean query time is
//! below its relative increase of the index bytes, `Δt / t < ΔB / B` (`t`: the workload's mean
//! cost per query, `B`: the index's compact bytes, [`IndexBytes`], both with the items taken so
//! far; equivalently, an item is taken while it lowers `t · B` to first order); the budget is a
//! ceiling. ([`Stop::Budget`]: the greedy fills the budget, items below a fixed
//! floor `λ` excluded — the previous rule, kept for comparisons.) Pairs need the co-occurrence
//! matrix of their big labels
//! (the `N_T` labels of at least `T` points in a candidate pair with a positive saving:
//! `N_T·⌈N_T/64⌉·8` bytes, plus the directory; a lookup of two big labels that never co-occur
//! answers `Empty`): the greedy is run without pairs and for every power of two `T ≥ 64` (pairs
//! of labels of at least `T` points, their matrix charged to the budget first), and the run with
//! the smallest `t · B` is kept ([`Stop::Budget`]: the largest expected saving, provided the
//! pairs pay for their matrix too — their saving at least `λ` times their bytes and the
//! matrix's). (B3a's rule, every label of at least `T`
//! points big, answers `Empty` for more pairs, but its larger matrix makes the greedy settle on
//! a larger `T`, i.e. fewer pairs of mid-sized labels; measured worse on the severe cells.)

use std::{collections::BTreeSet, path::Path};

use anyhow::{Context, Result};

use crate::{
    bitsets, container, diskann_io,
    graph::{self, AuxKind},
    islands::{IslandGreedy, QueryModel, QuerySample},
    kernel::cost::UnitCosts,
    labels::LabelStore,
    pairs, run,
};

/// The compact bytes of an index without the selector's own structures — `B` of the elasticity
/// rule and of the budget stop's floor — counted as a configuration's bytes are counted from its
/// `run.json` (the base of a memory ceiling given as a fraction of the index): base vectors
/// `n·d·elem`, the graph as adjacency lists, `4·(nodes + edges)` (in either file format), and its
/// router vectors, the label store file (dictionary, rows, postings) and the static bitsets of
/// the planned exact path (`θ = ⌈n/64⌉`). The sketches and start tables, which the selector does
/// not read, are not counted: `flab select --index-bytes` takes the complete value (e.g. measured
/// by a probe run of the configuration).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IndexBytes {
    pub vectors: u64,
    pub graph: u64,
    pub router_vectors: u64,
    pub labels: u64,
    pub bitsets: u64,
}

impl IndexBytes {
    /// The accounting of a dataset directory (`base.*`, `labels.flab`, whose store is `store`)
    /// and the graph the index searches (a directory with `graph.flab`, a `graph.flab`, or a
    /// DiskANN in-memory index file); `elem`: bytes per vector component.
    pub fn of(dataset: &Path, graph_path: &Path, store: &LabelStore, elem: usize) -> Result<Self> {
        let base = run::data::base_file(dataset)?;
        let mut f =
            std::fs::File::open(&base).with_context(|| format!("opening {}", base.display()))?;
        let (rows, dim) = diskann_io::read_bin_header(&mut f)?;
        let labels_path = dataset.join("labels.flab");
        let labels = std::fs::metadata(&labels_path)
            .with_context(|| format!("reading {}", labels_path.display()))?
            .len();
        let g = if graph_path.is_dir() {
            graph_path.join("graph.flab")
        } else {
            graph_path.to_owned()
        };
        let (graph, router_vectors) =
            if container::peek_magic(&g).ok().as_ref() == Some(graph::MAGIC) {
                let mut r = container::Reader::open(&g, graph::MAGIC)?;
                let p = *r.params();
                let (n_data, n_nodes, n_edges) = (p[0], p[1], p[7]);
                let aux = n_nodes.saturating_sub(n_data) as usize;
                let routers = if aux == 0 {
                    0
                } else {
                    r.section::<u8>(3, aux)?
                        .iter()
                        .filter(|&&k| k == AuxKind::Router.code())
                        .count() as u64
                };
                (4 * (n_nodes + n_edges), routers * dim as u64 * elem as u64)
            } else {
                // A DiskANN in-memory index: the header, then per node its degree and neighbours.
                let mut f =
                    std::fs::File::open(&g).with_context(|| format!("opening {}", g.display()))?;
                let h = diskann_io::Header::read(&mut f)?;
                (h.file_size.saturating_sub(diskann_io::HEADER_BYTES), 0)
            };
        Ok(Self {
            vectors: rows as u64 * dim as u64 * elem as u64,
            graph,
            router_vectors,
            labels,
            bitsets: bitsets::footprint(store, store.n_points().div_ceil(64) as u64).1,
        })
    }

    pub fn total(&self) -> u64 {
        self.vectors + self.graph + self.router_vectors + self.labels + self.bitsets
    }
}

/// A per-label kind valued on the query sample, as the greedy sees it: candidates by expected
/// saving per query per byte (islands, [`IslandGreedy`]; track G's router trees,
/// [`crate::select_g::TreeGreedy`]).
pub trait LabelKind {
    /// The best remaining candidate (its saving per query per byte and its index).
    fn peek(&mut self) -> Option<(f64, usize)>;
    /// Takes the peeked candidate; returns its marginal saving per query (ns).
    fn take(&mut self, x: usize) -> f64;
    /// Drops the peeked candidate.
    fn skip(&mut self, x: usize);
    /// Bytes of candidate `x`.
    fn bytes(&self, x: usize) -> u64;
    /// Label of candidate `x`.
    fn label(&self, x: usize) -> u32;
    /// The workload's independent queries `U` this kind's sample holds, and the sample index
    /// of the first (`None` without them; [`crate::workload::Independent`], the same for every
    /// kind). A pair item serves the queries holding both its labels
    /// ([`crate::workload::Independent::pair_queries`]): its saving there is shared with this
    /// kind — valued against each query's current cost ([`Self::query_cost`]) and, when the
    /// pair is taken, lowering it ([`Self::serve`]) — so that no query's saving counts twice.
    fn independent(&self) -> Option<(&crate::workload::Independent, usize)> {
        None
    }
    /// Cost (ns) of sample query `q` when the pair `(a, b)` (`joint` points) is stored and
    /// serves it (`None`: it does not). The kernel takes, of several stored pairs, the one with
    /// the fewest points: [`Self::serve`] keeps the cheapest.
    fn pair_cost(&self, _q: usize, _a: u32, _b: u32, _joint: u32) -> Option<f64> {
        None
    }
    /// Current cost (ns) of sample query `q` with the items taken.
    fn query_cost(&self, _q: usize) -> f64 {
        0.0
    }
    /// Workload weight of sample query `q`.
    fn query_weight(&self, _q: usize) -> f64 {
        0.0
    }
    /// An item of another kind serves sample query `q` at `cost`.
    fn serve(&mut self, _q: usize, _cost: f64) {}
    /// The sample's base-derived queries a pair item of labels `a`, `b` serves when they are
    /// valued one by one (as the independent queries, [`Self::pair_cost`]) rather than by the
    /// closed form of their context ([`Self::derived_closed_form`]).
    fn derived_queries(&self, _a: u32, _b: u32) -> &[u32] {
        &[]
    }
    /// Whether a pair item's saving on the base-derived queries is its context's closed form
    /// (`prob · saving`), or the sum over [`Self::derived_queries`] (each query's saving counted
    /// once, against its current cost).
    fn derived_closed_form(&self) -> bool {
        true
    }
}

impl LabelKind for IslandGreedy<'_> {
    fn peek(&mut self) -> Option<(f64, usize)> {
        IslandGreedy::peek(self)
    }

    fn take(&mut self, x: usize) -> f64 {
        IslandGreedy::take(self, x)
    }

    fn skip(&mut self, x: usize) {
        IslandGreedy::skip(self, x);
    }

    fn bytes(&self, x: usize) -> u64 {
        self.terms[x].bytes
    }

    fn label(&self, x: usize) -> u32 {
        self.terms[x].label
    }

    fn independent(&self) -> Option<(&crate::workload::Independent, usize)> {
        IslandGreedy::independent(self)
    }
    fn pair_cost(&self, q: usize, _a: u32, _b: u32, joint: u32) -> Option<f64> {
        IslandGreedy::pair_cost(self, q, joint)
    }

    fn query_cost(&self, q: usize) -> f64 {
        IslandGreedy::query_cost(self, q)
    }

    fn query_weight(&self, q: usize) -> f64 {
        IslandGreedy::query_weight(self, q)
    }

    fn serve(&mut self, q: usize, cost: f64) {
        IslandGreedy::serve(self, q, cost);
    }
}

/// `f64` with a total order (the pairs' heap).
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

/// A candidate pair: labels `a < b` in different categories, their joint posting size `j` and
/// `g_ab` (see the module documentation).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct PairCand {
    pub a: u32,
    pub b: u32,
    pub joint: u32,
    pub g: f64,
}

/// Bytes of a stored pair (its list and its directory entry).
pub fn pair_bytes(joint: u32) -> u64 {
    4 * (u64::from(joint) + 3)
}

/// Cost (ns) of the `pair` plan on a query served by a stored pair of `joint` points (on `n`
/// points) with `m` matches: the pair's list read with the query's other clauses checked
/// (`gallop_cold + j·pred(j/n)`), the matches' distances (`m·dist(m/n)`).
pub fn pair_plan_ns(c: &UnitCosts, n: usize, joint: u32, m: f64) -> f64 {
    let nf = n as f64;
    let j = f64::from(joint);
    c.gallop_cold() + j * c.pred(j / nf) + m * c.dist(m / nf)
}

/// Fixed bytes of the combination postings with `nb` big labels on a store of `nl` labels:
/// the big index, the matrix, the row and list offsets' first entries.
pub fn pairs_fixed_bytes(nl: usize, nb: usize) -> u64 {
    (4 * nl + nb * nb.div_ceil(64) * 8 + 8 * (nb + 1) + 8) as u64
}

/// `w(L)` of the module documentation: the probability that a kept category's clause of a
/// base-derived query is a given one of the point's `L` labels in it, alone.
pub fn single_prob(l: u16, singleton: f64) -> f64 {
    let lf = f64::from(l.max(1));
    singleton / lf + (1.0 - singleton) * (-lf).exp2() * (1.0 + 1.0 / lf)
}

/// Per-point category counts `L_p(c)` (`n × n_categories`, saturating) and the per-label
/// `f_a` of the module documentation (with up to `threads` threads).
pub fn label_weights(labels: &LabelStore, singleton: f64, threads: usize) -> (Vec<u16>, Vec<f64>) {
    let n = labels.n_points();
    let nc = labels.n_categories().max(1);
    let nl = labels.n_labels();
    let chunk = 1 << 16;
    let parts = crate::parallel::par_map(
        n.div_ceil(chunk),
        threads,
        || (),
        |ci, _| {
            let (lo, hi) = (ci * chunk, ((ci + 1) * chunk).min(n));
            let mut counts = vec![0u16; (hi - lo) * nc];
            let mut f: Vec<(u32, f64)> = Vec::new();
            for p in lo..hi {
                let row = labels.row(p as u32);
                let cnt = &mut counts[(p - lo) * nc..(p - lo + 1) * nc];
                for &l in row {
                    let c = labels.category(l) as usize;
                    cnt[c] = cnt[c].saturating_add(1);
                }
                for &l in row {
                    f.push((l, single_prob(cnt[labels.category(l) as usize], singleton)));
                }
            }
            (counts, f)
        },
    );
    let mut counts = Vec::with_capacity(n * nc);
    let mut f = vec![0.0f64; nl];
    for (c, fs) in parts {
        counts.extend_from_slice(&c);
        for (l, w) in fs {
            f[l as usize] += w;
        }
    }
    for x in &mut f {
        *x /= n.max(1) as f64;
    }
    (counts, f)
}

/// The co-occurring pairs of labels in different categories among those accepted by `keep`
/// (each of at least [`pairs::T_MIN`] and fewer than `n` points), with their joint sizes and
/// `g_ab`: one pass over the postings of every accepted label `a`, counting the labels of later
/// categories in its members' rows (rows in parallel; the result does not depend on the thread
/// count). `counts`: [`label_weights`]'s.
pub fn pair_candidates(
    labels: &LabelStore,
    counts: &[u16],
    singleton: f64,
    keep: impl Fn(u32) -> bool + Sync,
    threads: usize,
) -> Vec<PairCand> {
    let n = labels.n_points();
    count_pairs(
        labels,
        counts,
        singleton,
        |l| {
            let s = labels.posting_len(l);
            s as u64 >= pairs::T_MIN && s < n && keep(l)
        },
        threads,
    )
}

/// Every co-occurring pair of labels in different categories (the pair contexts of the
/// workload, [`QueryModel::uniform`]), in the same pass and order as [`pair_candidates`]; the
/// candidates are the contexts both of whose labels pass [`is_candidate_label`].
pub fn pair_contexts(
    labels: &LabelStore,
    counts: &[u16],
    singleton: f64,
    threads: usize,
) -> Vec<PairCand> {
    count_pairs(
        labels,
        counts,
        singleton,
        |l| labels.posting_len(l) > 0,
        threads,
    )
}

/// Whether label `l` can be in a stored pair: at least [`pairs::T_MIN`] and fewer than `n`
/// points.
pub fn is_candidate_label(labels: &LabelStore, l: u32) -> bool {
    let s = labels.posting_len(l);
    s as u64 >= pairs::T_MIN && s < labels.n_points()
}

/// The co-occurring pairs of labels in different categories among those accepted by `ok`.
fn count_pairs(
    labels: &LabelStore,
    counts: &[u16],
    singleton: f64,
    ok: impl Fn(u32) -> bool + Sync,
    threads: usize,
) -> Vec<PairCand> {
    let n = labels.n_points();
    let nc = labels.n_categories().max(1);
    let nl = labels.n_labels();
    let rows: Vec<u32> = (0..nl as u32)
        .filter(|&a| ok(a) && (labels.category(a) as usize) + 1 < nc)
        .collect();
    let init = || (vec![0u32; nl], vec![0.0f64; nl], Vec::<u32>::new());
    let parts = crate::parallel::par_map(rows.len(), threads, init, |x, s| {
        let (count, w, touched) = s;
        let a = rows[x];
        let ca = labels.category(a) as usize;
        let end = labels.category_labels(ca as u32).end;
        for &p in labels.postings(a) {
            let row = labels.row(p);
            let base = p as usize * nc;
            let wa = single_prob(counts[base + ca], singleton);
            for &b in &row[row.partition_point(|&l| l < end)..] {
                if !ok(b) {
                    continue;
                }
                let wb = single_prob(counts[base + labels.category(b) as usize], singleton);
                if count[b as usize] == 0 {
                    touched.push(b);
                }
                count[b as usize] += 1;
                w[b as usize] += wa * wb;
            }
        }
        touched.sort_unstable();
        let out: Vec<PairCand> = touched
            .iter()
            .map(|&b| PairCand {
                a,
                b,
                joint: count[b as usize],
                g: w[b as usize] / n as f64,
            })
            .collect();
        for &b in touched.iter() {
            count[b as usize] = 0;
            w[b as usize] = 0.0;
        }
        touched.clear();
        out
    });
    parts.into_iter().flatten().collect()
}

/// When the greedy stops (see the module documentation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Stop {
    /// At the budget (and [`Params::floor`]).
    #[default]
    Budget,
    /// At the first item whose relative saving of mean query time is below its relative
    /// increase of index bytes (`Δt / t < ΔB / B`, `t` and `B` with the items taken so far);
    /// the budget is a ceiling.
    Elasticity,
}

/// Selection parameters (see the module documentation).
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize)]
pub struct Params {
    /// Answers `K` of the cost terms.
    pub k: usize,
    /// Degree bound `R` of the islands.
    pub r: usize,
    /// Memory budget (bytes) for both kinds (a ceiling with [`Stop::Elasticity`]).
    pub budget_bytes: u64,
    /// Value floor `λ` (ns of expected saving per query per byte; [`Stop::Budget`] only).
    pub floor: f64,
    /// The workload (its `singleton`, `fresh`, `uniform` and `contexts` enter the pairs' closed
    /// form).
    pub model: QueryModel,
    pub stop: Stop,
    /// Mean cost per query of the workload without auxiliary structures (`t₀`, ns).
    pub mean_ns: f64,
    /// Bytes of the index without auxiliary structures (`B₀`).
    pub index_bytes: u64,
}

/// Value terms of a pair candidate.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct PairValue {
    pub cand: PairCand,
    /// Probability per query of its context `{a} ∧ {b}` among the base-derived part of the
    /// workload ([`context_prob`]).
    pub prob: f64,
    /// Saving of the `pair` plan on the context (ns).
    pub saving_ns: f64,
    pub bytes: u64,
    /// Expected saving per query per byte.
    pub value: f64,
}

/// Probability per query of the context `{a} ∧ {b}` of a candidate pair among the base-derived
/// part of the workload (`(1 − μ)·P_ab`, `P_ab` of the module documentation; `μ =
/// model.uniform`). The independent queries' part is valued on the sample's queries holding the
/// pair, shared with the per-label kind (see [`LabelKind`]).
pub fn context_prob(model: &QueryModel, f: &[f64], p: &PairCand) -> f64 {
    let fr = model.fresh;
    let (own, other) = ((1.0 - fr) * (1.0 - fr), 1.0 - (1.0 - fr) * (1.0 - fr));
    let derived = 0.25 * (own * p.g + other * f[p.a as usize] * f[p.b as usize]);
    if model.uniform > 0.0 {
        (1.0 - model.uniform) * derived
    } else {
        derived
    }
}

/// The value terms of the candidates with a positive saving, by decreasing value.
pub fn pair_values(
    labels: &LabelStore,
    c: &UnitCosts,
    k: usize,
    model: &QueryModel,
    f: &[f64],
    cands: &[PairCand],
) -> Vec<PairValue> {
    let n = labels.n_points();
    let nf = n as f64;
    let mut out: Vec<PairValue> = cands
        .iter()
        .filter_map(|&p| {
            let s = labels.posting_len(p.a).min(labels.posting_len(p.b)) as f64;
            let m = f64::from(p.joint);
            let exact = s * c.pred(s / nf) + m * c.dist(m / nf);
            let base = crate::islands::postfilter_nominal(c, k, k, m / nf, n, 1).1;
            let pair = pair_plan_ns(c, n, p.joint, m);
            let saving_ns = exact.min(base) - pair;
            if !saving_ns.is_finite() || saving_ns <= 0.0 {
                return None;
            }
            let prob = context_prob(model, f, &p);
            let bytes = pair_bytes(p.joint);
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

/// An upper bound on the value of any pair of label `a` (`w ≤ 1`, so `g_ab ≤ j/n`; at least
/// 16 bytes; a saving below the scan of `P_a`):
/// `[(1 − fresh)²/(16n) + (1 − (1 − fresh)²)·f_a·f_max/64]·|P_a|·pred(|P_a|/n)`.
pub fn pair_value_bound(
    labels: &LabelStore,
    c: &UnitCosts,
    model: &QueryModel,
    f: &[f64],
    f_max: f64,
    a: u32,
) -> f64 {
    let nf = labels.n_points() as f64;
    let s = labels.posting_len(a) as f64;
    let fr = model.fresh;
    let (own, other) = ((1.0 - fr) * (1.0 - fr), 1.0 - (1.0 - fr) * (1.0 - fr));
    (own / (16.0 * nf) + other * f[a as usize] * f_max / 64.0) * s * c.pred(s / nf)
}

/// One run of the greedy (see [`select`]).
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Run {
    /// Size threshold of the pairs' big labels (`None`: without pairs).
    pub t_big: Option<u64>,
    pub n_big: usize,
    pub fixed_bytes: u64,
    pub islands: Vec<u32>,
    pub island_bytes: u64,
    pub island_saving_ns: f64,
    /// Indices into the pair values.
    #[serde(skip)]
    pub pairs: Vec<usize>,
    /// The marginal saving per query (ns) of each pair taken, in the order of `pairs`.
    #[serde(skip)]
    pub pair_gains: Vec<f64>,
    pub n_pairs: usize,
    pub pair_bytes: u64,
    pub pair_saving_ns: f64,
    /// Value of the last item taken (the selection's cut).
    pub cut: f64,
    /// The pairs pay for their matrix (or there are none).
    pub pairs_pay: bool,
    /// Mean cost per query (ns) and index bytes with the run's items ([`Stop::Elasticity`]:
    /// the run with the smallest product is kept).
    pub t_ns: f64,
    pub index_bytes: u64,
}

impl Run {
    pub fn saving_ns(&self) -> f64 {
        self.island_saving_ns
            + if self.pairs_pay {
                self.pair_saving_ns
            } else {
                0.0
            }
    }

    pub fn bytes(&self) -> u64 {
        self.island_bytes
            + if self.pairs.is_empty() {
                0
            } else {
                self.pair_bytes + self.fixed_bytes
            }
    }
}

/// The selection: the best [`Run`] and every run tried.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Selection {
    pub best: Run,
    pub runs: Vec<Run>,
    /// The pairs of the best run (label ids) and their big labels.
    pub pairs: Vec<(u32, u32)>,
    pub big_labels: Vec<u32>,
}

/// The greedy over islands (from `sample`) and pairs (`values`, by decreasing value) of labels of
/// at least `t` points (none if `t` is `None`) within `params.budget_bytes`, the matrix of the
/// `big` candidate labels of at least `t` points charged first.
fn run<G: LabelKind>(
    labels: &LabelStore,
    kind: impl Fn() -> G,
    values: &[PairValue],
    cand_labels: &[u32],
    params: &Params,
    t: Option<u64>,
) -> Run {
    let nl = labels.n_labels();
    let (n_big, fixed) = match t {
        Some(t) => {
            let nb = cand_labels
                .iter()
                .filter(|&&l| labels.posting_len(l) as u64 >= t)
                .count();
            (nb, pairs_fixed_bytes(nl, nb))
        }
        None => (0, 0),
    };
    let mut out = Run {
        t_big: t,
        n_big,
        fixed_bytes: fixed,
        ..Default::default()
    };
    if fixed > params.budget_bytes {
        out.pairs_pay = params.stop == Stop::Elasticity;
        out.t_ns = params.mean_ns;
        out.index_bytes = params.index_bytes;
        return out;
    }
    let eligible = |v: &PairValue| match t {
        Some(t) => {
            labels.posting_len(v.cand.a) as u64 >= t && labels.posting_len(v.cand.b) as u64 >= t
        }
        None => false,
    };
    let mut left = params.budget_bytes - fixed;
    let mut gr = kind();
    let mut cut = f64::INFINITY;
    // Mean cost per query and index bytes with the items taken (the matrix charged first).
    let (mut t_cur, mut b_cur) = (params.mean_ns, (params.index_bytes + fixed) as f64);
    // The smallest value an item may have: `Δt / ΔB ≥ t / B` (elasticity at least 1), or the
    // floor.
    let threshold = |t: f64, b: f64| match params.stop {
        Stop::Budget => params.floor,
        Stop::Elasticity => t / b,
    };
    // A pair's marginal saving per query: its base-derived part (`prob · saving`) plus, on each
    // independent query holding both its labels (if the per-label kind holds them), the query's
    // weight times how much the pair plan lowers the query's current cost (`cost`: the pair
    // plan's cost there).
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(values.len());
    let mut served: Vec<(usize, f64, f64)> = Vec::new();
    {
        let ind = gr.independent();
        for v in values {
            let start = served.len();
            if let Some((u, first)) = ind {
                for i in u.pair_queries(v.cand.a, v.cand.b) {
                    let q = first + i;
                    if let Some(cost) = gr.pair_cost(q, v.cand.a, v.cand.b, v.cand.joint) {
                        served.push((q, gr.query_weight(q), cost));
                    }
                }
            }
            for &q in gr.derived_queries(v.cand.a, v.cand.b) {
                let q = q as usize;
                if let Some(cost) = gr.pair_cost(q, v.cand.a, v.cand.b, v.cand.joint) {
                    served.push((q, gr.query_weight(q), cost));
                }
            }
            ranges.push((start, served.len()));
        }
    }
    let closed = gr.derived_closed_form();
    let pair_gain = |gr: &G, i: usize| -> f64 {
        let v = &values[i];
        let (a, b) = ranges[i];
        (if closed { v.prob * v.saving_ns } else { 0.0 })
            + served[a..b]
                .iter()
                .map(|&(q, w, cost)| w * (gr.query_cost(q) - cost).max(0.0))
                .sum::<f64>()
    };
    // The eligible pairs by value (ties: in the order of `values`), re-evaluated lazily: a
    // value only falls as items are taken.
    let mut heap: std::collections::BinaryHeap<(Key, std::cmp::Reverse<usize>)> = (0..values.len())
        .filter(|&i| eligible(&values[i]))
        .map(|i| {
            (
                Key(pair_gain(&gr, i) / values[i].bytes as f64),
                std::cmp::Reverse(i),
            )
        })
        .collect();
    let peek_pair = |heap: &mut std::collections::BinaryHeap<(Key, std::cmp::Reverse<usize>)>,
                     gr: &G|
     -> Option<(f64, usize)> {
        loop {
            let &(Key(key), std::cmp::Reverse(i)) = heap.peek()?;
            let g = pair_gain(gr, i);
            let now = g / values[i].bytes as f64;
            if now < key * (1.0 - 1e-12) {
                heap.pop();
                if g > 0.0 {
                    heap.push((Key(now), std::cmp::Reverse(i)));
                }
                continue;
            }
            if g <= 0.0 {
                heap.pop();
                continue;
            }
            return Some((now, i));
        }
    };
    loop {
        let isl = gr.peek();
        let pv = peek_pair(&mut heap, &gr);
        let take_island = match (isl, pv) {
            (None, None) => break,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (Some((ki, _)), Some((kp, _))) => ki >= kp,
        };
        if take_island {
            let (key, x) = isl.expect("an island");
            if key < threshold(t_cur, b_cur) {
                break;
            }
            let bytes = gr.bytes(x);
            if bytes > left {
                gr.skip(x);
                continue;
            }
            left -= bytes;
            let g = gr.take(x);
            out.island_saving_ns += g;
            out.island_bytes += bytes;
            out.islands.push(gr.label(x));
            t_cur -= g;
            b_cur += bytes as f64;
            cut = key;
        } else {
            let (key, i) = pv.expect("a pair");
            if key < threshold(t_cur, b_cur) {
                // The best item fails: with the elasticity rule the threshold changes only when
                // an item is taken; with the floor no pair left passes it.
                if params.stop == Stop::Elasticity {
                    break;
                }
                heap.clear();
                continue;
            }
            heap.pop();
            let v = &values[i];
            if v.bytes <= left {
                left -= v.bytes;
                let g = pair_gain(&gr, i);
                out.pair_saving_ns += g;
                out.pair_bytes += v.bytes;
                out.pairs.push(i);
                out.pair_gains.push(g);
                t_cur -= g;
                b_cur += v.bytes as f64;
                cut = key;
                let (a, b) = ranges[i];
                for &(q, _, cost) in &served[a..b] {
                    gr.serve(q, cost);
                }
            }
        }
    }
    out.islands.sort_unstable();
    out.n_pairs = out.pairs.len();
    out.cut = if cut.is_finite() { cut } else { 0.0 };
    out.pairs_pay = match params.stop {
        Stop::Budget => {
            out.pairs.is_empty()
                || out.pair_saving_ns >= params.floor * (out.pair_bytes + fixed) as f64
        }
        // Runs are compared by `t · B`, which charges the matrix.
        Stop::Elasticity => true,
    };
    out.t_ns = params.mean_ns - out.saving_ns();
    out.index_bytes = params.index_bytes + out.bytes();
    out
}

/// The selection (see the module documentation): the greedy without pairs and with the pairs
/// of every power-of-two size threshold `T` at which some candidate pair exists; the run with
/// the largest expected saving whose pairs pay for their matrix.
pub fn select(
    labels: &LabelStore,
    sample: &QuerySample,
    values: &[PairValue],
    params: &Params,
) -> Selection {
    select_with(
        labels,
        || IslandGreedy::new(sample, labels, params.r, |_| true),
        values,
        params,
    )
}

/// [`select`] with another per-label kind than the islands (e.g. track G's router trees,
/// [`crate::select_g`]); `values` are the pair kind's terms (combination postings, or pair
/// trees), by decreasing value.
pub fn select_with<G: LabelKind>(
    labels: &LabelStore,
    kind: impl Fn() -> G,
    values: &[PairValue],
    params: &Params,
) -> Selection {
    let cand_labels: Vec<u32> = values
        .iter()
        .flat_map(|v| [v.cand.a, v.cand.b])
        .collect::<BTreeSet<u32>>()
        .into_iter()
        .collect();
    let max_t = values
        .iter()
        .map(|v| {
            labels
                .posting_len(v.cand.a)
                .min(labels.posting_len(v.cand.b)) as u64
        })
        .max()
        .unwrap_or(0);
    let mut ts: Vec<Option<u64>> = vec![None];
    let mut t = pairs::T_MIN;
    while t <= max_t {
        ts.push(Some(t));
        t *= 2;
    }
    let runs: Vec<Run> = ts
        .iter()
        .map(|&t| run(labels, &kind, values, &cand_labels, params, t))
        .collect();
    let best = match params.stop {
        Stop::Budget => runs
            .iter()
            .filter(|r| r.pairs_pay)
            .max_by(|x, y| {
                x.saving_ns()
                    .total_cmp(&y.saving_ns())
                    .then(y.bytes().cmp(&x.bytes()))
            })
            .cloned()
            .unwrap_or_default(),
        // The run whose items lower `t · B` (mean query time × index bytes) the most.
        Stop::Elasticity => runs
            .iter()
            .min_by(|x, y| {
                (x.t_ns * x.index_bytes as f64)
                    .total_cmp(&(y.t_ns * y.index_bytes as f64))
                    .then(x.bytes().cmp(&y.bytes()))
            })
            .cloned()
            .unwrap_or_default(),
    };
    let mut pairs: Vec<(u32, u32)> = best
        .pairs
        .iter()
        .map(|&i| (values[i].cand.a, values[i].cand.b))
        .collect();
    pairs.sort_unstable();
    let big_labels = match best.t_big {
        Some(t) if !pairs.is_empty() => cand_labels
            .iter()
            .copied()
            .filter(|&l| labels.posting_len(l) as u64 >= t)
            .collect(),
        _ => Vec::new(),
    };
    Selection {
        best,
        runs,
        pairs,
        big_labels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{islands, labels::LabelStoreBuilder};

    /// 3,000 points, three categories: `c` (8 labels, one or two per point), `d` (4 labels, one
    /// per point) and `e` (5 labels, one to three per point), drawn from a seeded hash.
    fn store() -> LabelStore {
        let mut b = LabelStoreBuilder::new();
        let h = |p: u64, s: u64| {
            let mut z = p.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ s;
            z = (z ^ (z >> 31)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z ^ (z >> 29)
        };
        let name = |c: &str, i: u64| format!("{c}{i}");
        let names: Vec<(String, String, String)> = (0..8)
            .map(|i| (name("c", i), name("d", i), name("e", i)))
            .collect();
        for p in 0..3000u64 {
            let mut v = vec![("c", names[(h(p, 1) % 8) as usize].0.as_str())];
            if h(p, 2) % 3 == 0 {
                v.push(("c", names[(h(p, 3) % 8) as usize].0.as_str()));
            }
            v.push(("d", names[(h(p, 4) % 4) as usize].1.as_str()));
            for e in 0..1 + h(p, 5) % 3 {
                v.push(("e", names[(h(p, 6 + e) % 5) as usize].2.as_str()));
            }
            v.sort_unstable();
            v.dedup();
            b.push_point(v).unwrap();
        }
        b.finish().unwrap()
    }

    #[test]
    fn the_closed_form_context_probability_matches_the_sampled_queries() {
        let s = store();
        let (counts, f) = label_weights(&s, 0.5, 2);
        let cands = pair_candidates(&s, &counts, 0.5, |_| true, 3);
        assert_eq!(cands, pair_candidates(&s, &counts, 0.5, |_| true, 1));
        // Joints are the postings' intersections.
        for c in &cands {
            let pb = s.postings(c.b);
            let j = s
                .postings(c.a)
                .iter()
                .filter(|p| pb.binary_search(p).is_ok())
                .count();
            assert_eq!(c.joint as usize, j);
            assert_ne!(s.category(c.a), s.category(c.b));
        }
        for fresh in [0.0, 0.5] {
            let model = QueryModel {
                n_queries: 200_000,
                seed: 3,
                cap: 4096,
                singleton: 0.5,
                fresh,
                ..Default::default()
            };
            // Frequency of the singleton contexts {a} ∧ {b} among the drawn queries.
            let mut freq = std::collections::HashMap::<(u32, u32), u64>::new();
            let (mut kl, mut cl, mut cats) = (Vec::new(), Vec::new(), Vec::new());
            for qi in 0..model.n_queries {
                assert!(islands::draw_query(
                    &s, &model, qi, &mut kl, &mut cl, &mut cats
                ));
                let singles: Vec<u32> = cl
                    .iter()
                    .filter(|&&(a, b)| b - a == 1)
                    .map(|&(a, _)| kl[a])
                    .collect();
                for (x, &a) in singles.iter().enumerate() {
                    for &b in &singles[x + 1..] {
                        let key = if a < b { (a, b) } else { (b, a) };
                        *freq.entry(key).or_default() += 1;
                    }
                }
            }
            let fr = fresh;
            let mut checked = 0;
            for c in &cands {
                let p = 0.25
                    * ((1.0 - fr) * (1.0 - fr) * c.g
                        + (1.0 - (1.0 - fr) * (1.0 - fr)) * f[c.a as usize] * f[c.b as usize]);
                let want = p * model.n_queries as f64;
                if want < 400.0 {
                    continue;
                }
                let got = freq.get(&(c.a, c.b)).copied().unwrap_or(0) as f64;
                // Within 5 standard deviations.
                assert!(
                    (got - want).abs() <= 5.0 * want.sqrt(),
                    "fresh {fresh} ({}, {}): sampled {got}, closed form {want:.0}",
                    s.label_name(c.a),
                    s.label_name(c.b)
                );
                checked += 1;
            }
            assert!(checked > 20, "{checked}");
        }
    }

    #[test]
    fn selection_respects_the_budget_and_the_floor_and_builds_its_pairs() {
        let s = store();
        let c = UnitCosts::default();
        let model = QueryModel {
            n_queries: 20_000,
            seed: 5,
            cap: 512,
            singleton: 0.5,
            fresh: 0.5,
            ..Default::default()
        };
        let sample = QuerySample::build(&s, &c, 10, model, 2);
        let (counts, f) = label_weights(&s, 0.5, 2);
        let cands = pair_candidates(&s, &counts, 0.5, |_| true, 2);
        let values = pair_values(&s, &c, 10, &model, &f, &cands);
        assert!(values.windows(2).all(|w| w[0].value >= w[1].value));
        let mut params = Params {
            k: 10,
            r: 8,
            budget_bytes: u64::MAX / 4,
            floor: 0.0,
            model,
            ..Default::default()
        };
        let all = select(&s, &sample, &values, &params);
        assert!(!all.pairs.is_empty(), "every positive pair fits");
        for budget in [0u64, 2_000, 20_000, 200_000] {
            params.budget_bytes = budget;
            let sel = select(&s, &sample, &values, &params);
            assert!(sel.best.bytes() <= budget, "{budget}: {}", sel.best.bytes());
            assert!(sel.runs.iter().all(|r| r.bytes() <= budget));
        }
        // A floor above every value takes nothing.
        params.budget_bytes = u64::MAX / 4;
        params.floor = f64::INFINITY;
        let none = select(&s, &sample, &values, &params);
        assert!(none.pairs.is_empty() && none.best.islands.is_empty());
        // The pairs of a selection build: their lists are the intersections, lookups classify.
        let t = all.best.t_big.expect("pairs");
        let (ix, st) =
            pairs::PairIndex::build_selected(&s, t, &all.big_labels, &all.pairs, 0, 0.0, 3)
                .unwrap();
        let (ix1, _) =
            pairs::PairIndex::build_selected(&s, t, &all.big_labels, &all.pairs, 0, 0.0, 1)
                .unwrap();
        assert_eq!(ix, ix1, "independent of the thread count");
        ix.check_sample(&s, 1).unwrap();
        assert_eq!(st.stored_pairs as usize, all.pairs.len());
        for &(a, b) in &all.pairs {
            assert!(matches!(ix.joint(a, b), pairs::Joint::Stored(..)));
        }
        for &a in &all.big_labels {
            for &b in &all.big_labels {
                if a == b {
                    continue;
                }
                let pb = s.postings(b);
                let joint = s
                    .postings(a)
                    .iter()
                    .filter(|p| pb.binary_search(p).is_ok())
                    .count();
                match ix.joint(a, b) {
                    pairs::Joint::Empty => assert_eq!(joint, 0),
                    pairs::Joint::Stored(x, y) => assert_eq!((y - x) as usize, joint),
                    pairs::Joint::Unknown => assert!(
                        s.category(a) == s.category(b) || joint > 0,
                        "an unstored pair of different categories is empty or co-occurs"
                    ),
                }
            }
        }
        let mut buf = Vec::new();
        ix.write_to(&mut buf).unwrap();
        assert_eq!(
            pairs::PairIndex::read_from(std::io::Cursor::new(buf)).unwrap(),
            ix
        );
    }

    #[test]
    fn pair_contexts_hold_every_cooccurring_pair_and_the_candidates() {
        let s = store();
        let (counts, _) = label_weights(&s, 0.5, 2);
        let contexts = pair_contexts(&s, &counts, 0.5, 3);
        assert_eq!(contexts, pair_contexts(&s, &counts, 0.5, 1));
        // Brute force: every pair of labels of different categories that co-occur.
        let mut want = Vec::new();
        for a in 0..s.n_labels() as u32 {
            for b in a + 1..s.n_labels() as u32 {
                let pb = s.postings(b);
                let j = s
                    .postings(a)
                    .iter()
                    .filter(|p| pb.binary_search(p).is_ok())
                    .count();
                if s.category(a) != s.category(b) && j > 0 {
                    want.push((a, b, j as u32));
                }
            }
        }
        let got: Vec<(u32, u32, u32)> = contexts.iter().map(|c| (c.a, c.b, c.joint)).collect();
        assert_eq!(got, want);
        // The candidates are the contexts of candidate labels, with the same terms.
        let cands = pair_candidates(&s, &counts, 0.5, |_| true, 2);
        let filtered: Vec<PairCand> = contexts
            .iter()
            .filter(|c| is_candidate_label(&s, c.a) && is_candidate_label(&s, c.b))
            .copied()
            .collect();
        assert_eq!(cands, filtered);
    }

    #[test]
    fn the_mixed_workload_weights_the_independent_queries_by_their_probability() {
        let s = store();
        let c = UnitCosts::default();
        let (counts, f) = label_weights(&s, 0.5, 2);
        let contexts = pair_contexts(&s, &counts, 0.5, 2);
        let derived = QueryModel {
            n_queries: 3_000,
            seed: 9,
            cap: 512,
            ..Default::default()
        };
        let mixed = QueryModel {
            uniform: 0.5,
            contexts: contexts.len() as u64,
            ..derived
        };
        let u = std::sync::Arc::new(crate::workload::independent(&s, &contexts, &mixed, 3));
        assert_eq!(
            u.queries,
            crate::workload::independent(&s, &contexts, &mixed, 1).queries
        );
        let one = QuerySample::build(&s, &c, 10, derived, 2);
        let two = QuerySample::build_mixed(&s, &c, 10, mixed, Some(u.clone()), 3);
        assert_eq!(two.base_ns.len(), 3_000 + u.len());
        // The base-derived part is the base-derived sample; an independent query of labels L is
        // priced like a base-derived one with those single-label clauses and its `M`, weighted
        // μ times its probability.
        assert_eq!(&two.base_ns[..3_000], &one.base_ns[..]);
        let nf = s.n_points() as f64;
        for (i, w) in u.queries.iter().enumerate().step_by(7) {
            let q = 3_000 + i;
            let sz = w.labels.iter().map(|&l| s.posting_len(l)).min().unwrap() as f64;
            let m = w.joint;
            let checks = if w.labels.len() > 1 {
                sz * c.pred(sz / nf)
            } else {
                0.0
            };
            let exact = checks + m * c.dist(m / nf);
            let base = islands::postfilter_nominal(&c, 10, 10, m / nf, s.n_points(), 1).1;
            assert!((two.base_ns[q] - exact.min(base)).abs() <= 1e-9 * exact.min(base));
            assert!((two.weight(q) - 0.5 * w.weight).abs() < 1e-18);
        }
        assert!((two.weight(0) - 0.5 / 3_000.0).abs() < 1e-18);
        let total: f64 = (0..two.base_ns.len()).map(|q| two.weight(q)).sum();
        assert!((total - 1.0).abs() < 1e-9);
        let mean_u: f64 = u
            .queries
            .iter()
            .enumerate()
            .map(|(i, w)| w.weight * two.base_ns[3_000 + i])
            .sum();
        assert!((two.mean_base_ns() - 0.5 * (one.mean_base_ns() + mean_u)).abs() < 1e-6);
        // Context probabilities: the base-derived part only, (1 − μ) of it.
        for p in contexts.iter().take(20) {
            let want = 0.5 * context_prob(&derived, &f, p);
            assert!((context_prob(&mixed, &f, p) - want).abs() < 1e-18);
        }
        // A thread count does not change the sample.
        let three = QuerySample::build_mixed(&s, &c, 10, mixed, Some(u), 1);
        assert_eq!(two.base_ns, three.base_ns);
    }

    #[test]
    fn the_independent_queries_are_a_walk_on_the_co_occurrence_structure() {
        let s = store();
        let (counts, _) = label_weights(&s, 0.5, 2);
        let contexts = pair_contexts(&s, &counts, 0.5, 2);
        let model = QueryModel {
            n_queries: 20_000,
            seed: 5,
            cap: 512,
            uniform: 0.5,
            contexts: contexts.len() as u64,
            ..Default::default()
        };
        let u = crate::workload::independent(&s, &contexts, &model, 2);
        let walk = crate::workload::Walk::new(&s, &contexts);
        assert_eq!(u.k_max, 3);
        assert_eq!(
            (u.n_single, u.n_pair, u.n_walk),
            (s.n_labels(), contexts.len(), 20_000)
        );
        let total: f64 = u.queries.iter().map(|q| q.weight).sum();
        assert!((total - 1.0).abs() < 1e-9);
        // Every query: labels sorted, of distinct categories, carried together by some point;
        // the pair index finds it under each of its pairs.
        for (i, q) in u.queries.iter().enumerate() {
            assert!(q.labels.windows(2).all(|w| w[0] < w[1]));
            let mut cats: Vec<u32> = q.labels.iter().map(|&l| s.category(l)).collect();
            cats.dedup();
            assert_eq!(cats.len(), q.labels.len());
            let (p0, rest) = q.labels.split_first().unwrap();
            assert!(s
                .postings(*p0)
                .iter()
                .any(|&p| rest.iter().all(|l| s.row(p).binary_search(l).is_ok())));
            for (x, &a) in q.labels.iter().enumerate() {
                for &b in &q.labels[x + 1..] {
                    assert!(u.pair_queries(a, b).any(|j| j == i));
                }
            }
        }
        // The one- and two-label parts are exact: their weights equal the walk's frequencies
        // (i.i.d. draws; the three-label part holds a third of the mass, every walk of three
        // labels completes on this store).
        let draws = 300_000;
        let mut freq = std::collections::HashMap::<Vec<u32>, f64>::new();
        let mut n_three = 0usize;
        for i in 0..draws {
            let d = walk.draw(&model, i).unwrap();
            let l: Vec<u32> = d.clauses.iter().map(|c| c[0]).collect();
            if l.len() == 3 {
                n_three += 1;
            } else {
                *freq.entry(l).or_default() += 1.0 / draws as f64;
            }
        }
        assert!((n_three as f64 / draws as f64 - 1.0 / 3.0).abs() < 0.01);
        let mut tv = 0.0;
        for q in &u.queries[..u.n_single + u.n_pair] {
            tv += (freq.get(&q.labels).copied().unwrap_or(0.0) - q.weight).abs();
        }
        assert!(tv < 0.03, "total variation {tv}");
        // Vectors: a base-derived query searches from its own point (which matches it); an
        // independent one from a point drawn independently of its labels.
        for i in 0..200 {
            let d = crate::workload::draw_data(&s, &model, i).expect("a labelled point");
            let row = s.row(d.point);
            assert!(d
                .clauses
                .iter()
                .all(|cl| cl.iter().any(|l| row.binary_search(l).is_ok())));
        }
        let n_match = (0..4_000)
            .filter(|&i| {
                let d = walk.draw(&model, i).unwrap();
                let row = s.row(d.point);
                d.clauses
                    .iter()
                    .all(|cl| cl.iter().any(|l| row.binary_search(l).is_ok()))
            })
            .count();
        assert!(
            n_match < 2_000,
            "{n_match} of 4000 points match their predicates"
        );
        assert_eq!(walk.draw(&model, 7), walk.draw(&model, 7));
    }

    #[test]
    fn the_elasticity_rule_takes_items_while_they_lower_time_times_bytes() {
        let s = store();
        let c = UnitCosts::default();
        let (counts, f) = label_weights(&s, 0.5, 2);
        let contexts = pair_contexts(&s, &counts, 0.5, 2);
        let model = QueryModel {
            n_queries: 4_000,
            seed: 11,
            cap: 512,
            uniform: 0.5,
            contexts: contexts.len() as u64,
            ..Default::default()
        };
        let u = crate::workload::independent(&s, &contexts, &model, 2);
        let sample = QuerySample::build_mixed(&s, &c, 10, model, Some(std::sync::Arc::new(u)), 2);
        let cands: Vec<PairCand> = contexts
            .iter()
            .filter(|p| is_candidate_label(&s, p.a) && is_candidate_label(&s, p.b))
            .copied()
            .collect();
        let values = pair_values(&s, &c, 10, &model, &f, &cands);
        let mean = sample.mean_base_ns();
        let params = |index_bytes: u64, stop: Stop| Params {
            k: 10,
            r: 8,
            budget_bytes: u64::MAX / 4,
            model,
            stop,
            mean_ns: mean,
            index_bytes,
            ..Default::default()
        };
        let all = select(&s, &sample, &values, &params(1 << 20, Stop::Budget));
        let mut last = usize::MAX;
        for b0 in [1u64 << 10, 1 << 16, 1 << 20, 1 << 26, 1 << 40] {
            let p = params(b0, Stop::Elasticity);
            let sel = select(&s, &sample, &values, &p);
            let best = &sel.best;
            // Every item taken lowers `t · B`: the run without pairs (no matrix to recover) and
            // the kept run, the smallest product, end at most at `t₀ · B₀`.
            let tb = |r: &Run| r.t_ns * r.index_bytes as f64;
            let no_pairs = sel.runs.iter().find(|r| r.t_big.is_none()).expect("a run");
            assert!(tb(no_pairs) <= mean * b0 as f64 * (1.0 + 1e-12));
            assert!(sel.runs.iter().all(|r| tb(best) <= tb(r)));
            assert_eq!(best.t_ns, mean - best.saving_ns());
            assert_eq!(best.index_bytes, b0 + best.bytes());
            // A larger index makes bytes relatively cheaper: never fewer bytes taken (with
            // the same order of items, a lower threshold).
            let n_items = best.islands.len() + best.pairs.len();
            if last != usize::MAX {
                assert!(n_items >= last, "{b0}: {n_items} < {last}");
            }
            last = n_items;
            assert!(best.bytes() <= all.best.bytes() + best.fixed_bytes);
        }
        // An index of 1 KiB: no item lowers `t · B`; of 1 TiB: the elasticity rule takes what
        // the budget rule takes.
        let none = select(&s, &sample, &values, &params(1 << 10, Stop::Elasticity));
        assert!(none.best.islands.is_empty() && none.pairs.is_empty());
        let big = select(&s, &sample, &values, &params(1 << 40, Stop::Elasticity));
        assert_eq!(big.best.islands, all.best.islands);
    }

    #[test]
    fn pairs_and_islands_share_an_independent_query_s_saving() {
        let s = store();
        let c = UnitCosts::default();
        let (counts, f) = label_weights(&s, 0.5, 2);
        let contexts = pair_contexts(&s, &counts, 0.5, 2);
        let model = QueryModel {
            n_queries: 4_000,
            seed: 13,
            cap: 512,
            uniform: 0.5,
            contexts: contexts.len() as u64,
            ..Default::default()
        };
        let u = std::sync::Arc::new(crate::workload::independent(&s, &contexts, &model, 2));
        let sample = QuerySample::build_mixed(&s, &c, 10, model, Some(u.clone()), 2);
        let cands: Vec<PairCand> = contexts
            .iter()
            .filter(|p| is_candidate_label(&s, p.a) && is_candidate_label(&s, p.b))
            .copied()
            .collect();
        let values = pair_values(&s, &c, 10, &model, &f, &cands);
        let mean = sample.mean_base_ns();
        let params = Params {
            k: 10,
            r: 8,
            budget_bytes: u64::MAX / 4,
            model,
            stop: Stop::Budget,
            mean_ns: mean,
            index_bytes: 1 << 30,
            ..Default::default()
        };
        let sel = select(&s, &sample, &values, &params);
        let best = &sel.best;
        assert!(!best.islands.is_empty() && !best.pairs.is_empty());
        // A lower bound on the cost with everything built: every query at its cheapest island,
        // every independent query holding a taken pair at that pair's plan if cheaper; the
        // pairs' base-derived savings are counted apart (not shared). The selection's model
        // cost is not below it: no saving is counted twice on a query.
        let mut floor = sample.island_floor_ns();
        let mut derived = 0.0;
        let mut served = 0;
        for &i in &best.pairs {
            let v = &values[i];
            derived += v.prob * v.saving_ns;
            for j in u.pair_queries(v.cand.a, v.cand.b) {
                let q = sample.n_data + j;
                let cost = sample
                    .pair_cost(q, v.cand.joint)
                    .expect("an independent query");
                floor[q] = floor[q].min(cost);
                served += usize::from(u.queries[j].labels.len() == 3);
            }
        }
        assert!(served > 0, "a taken pair serves some three-label query");
        let bound = sample.weighted_mean(&floor) - derived;
        assert!(best.t_ns >= bound - 1e-6 * mean, "{} < {bound}", best.t_ns);
        assert!(best.t_ns > 0.0);
    }
}
