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
//! **Workload.** The base-derived queries of the islands' `queries` selection
//! ([`QueryModel`]): a data point drawn uniformly, each of its categories kept with probability
//! ½, a kept category's clause one of the point's labels in it (probability `singleton`) or a
//! random subset of them (optionally, with probability `fresh`, drawn from another data point).
//! An island serves a query through a single-label clause; a pair `(a, b)` through two
//! single-label clauses `{a} ∧ {b}` in different categories (the `pair` plan's exact cover).
//!
//! **Islands** are valued on a sample of these queries ([`QuerySample`]: per query the cost of
//! its best route without auxiliary structures and of the island of each single-label clause).
//!
//! **Pairs** are too many, and each drawn too rarely, for a sample (a pair whose joint posting
//! holds `j` points is drawn with probability ∝ `j/n`), so their value is the expectation over
//! the same workload in closed form: the probability that a query's clauses in the categories
//! of `a` and `b` are `{a}` and `{b}`,
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
//! joint posting read with a residual clause checked), at `4·(j + 3)` bytes. Only pairs that
//! co-occur are candidates (a query of the workload has a match), of labels of at least
//! [`pairs::T_MIN`] points. The kinds' interactions (a query served by both) are not modelled:
//! a pair is valued against the routes without auxiliary structures, an island against those and
//! the islands taken before it.
//!
//! **Selection.** One greedy over both kinds by expected saving per query per byte while the
//! budget lasts; an item is taken only if its value is at least the floor
//! `λ = κ · (mean cost per query) / (index bytes)`: with `κ = 1`, an item must speed the workload
//! up by at least the fraction by which it grows the index (labels of a smaller bound on their
//! pairs' value are not counted at all). Pairs need the co-occurrence matrix of their big labels
//! (every label of at least `T` points, `N_T` of them: `N_T·⌈N_T/64⌉·8` bytes, plus the directory;
//! it also tells the `pair` plan which pairs of big labels never co-occur, which OR-clause covers
//! need where no pair of theirs is stored — a value the model does not count):
//! the greedy is run without pairs and for every power of two `T ≥ 64` (pairs of labels of at
//! least `T` points, their matrix charged to the budget first), and the run with the largest
//! expected saving is kept, provided the pairs pay for their matrix too (their saving at least
//! `λ` times their bytes and the matrix's).

use crate::{
    islands::{IslandGreedy, QueryModel, QuerySample},
    kernel::cost::UnitCosts,
    labels::LabelStore,
    pairs,
};

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
    let nc = labels.n_categories().max(1);
    let nl = labels.n_labels();
    let ok = |l: u32| {
        let s = labels.posting_len(l);
        s as u64 >= pairs::T_MIN && s < n && keep(l)
    };
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

/// Selection parameters (see the module documentation).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Params {
    /// Answers `K` of the cost terms.
    pub k: usize,
    /// Degree bound `R` of the islands.
    pub r: usize,
    /// Memory budget (bytes) for both kinds.
    pub budget_bytes: u64,
    /// Value floor `λ` (ns of expected saving per query per byte).
    pub floor: f64,
    /// The workload (its `singleton` and `fresh` enter the pairs' closed form).
    pub model: QueryModel,
}

/// Value terms of a pair candidate.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct PairValue {
    pub cand: PairCand,
    /// Probability per query of its context `{a} ∧ {b}`.
    pub prob: f64,
    /// Saving of the `pair` plan on the context (ns).
    pub saving_ns: f64,
    pub bytes: u64,
    /// Expected saving per query per byte.
    pub value: f64,
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
    let fr = model.fresh;
    let (own, other) = ((1.0 - fr) * (1.0 - fr), 1.0 - (1.0 - fr) * (1.0 - fr));
    let mut out: Vec<PairValue> = cands
        .iter()
        .filter_map(|&p| {
            let s = labels.posting_len(p.a).min(labels.posting_len(p.b)) as f64;
            let m = f64::from(p.joint);
            let exact = s * c.pred(s / nf) + m * c.dist(m / nf);
            let base = crate::islands::postfilter_nominal(c, k, k, m / nf, n, 1).1;
            let pair = c.gallop_cold() + m * (c.pred(m / nf) + c.dist(m / nf));
            let saving_ns = exact.min(base) - pair;
            if !saving_ns.is_finite() || saving_ns <= 0.0 {
                return None;
            }
            let prob = 0.25 * (own * p.g + other * f[p.a as usize] * f[p.b as usize]);
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
    pub n_pairs: usize,
    pub pair_bytes: u64,
    pub pair_saving_ns: f64,
    /// Value of the last item taken (the selection's cut).
    pub cut: f64,
    /// The pairs pay for their matrix (or there are none).
    pub pairs_pay: bool,
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
fn run(
    labels: &LabelStore,
    sample: &QuerySample,
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
        return out;
    }
    let eligible = |v: &PairValue| match t {
        Some(t) => {
            labels.posting_len(v.cand.a) as u64 >= t && labels.posting_len(v.cand.b) as u64 >= t
        }
        None => false,
    };
    let mut left = params.budget_bytes - fixed;
    let mut gr = IslandGreedy::new(sample, labels, params.r, |_| true);
    let mut pi = 0usize;
    let mut cut = f64::INFINITY;
    loop {
        while pi < values.len() && !eligible(&values[pi]) {
            pi += 1;
        }
        let isl = gr.peek();
        let pv = values.get(pi).map(|v| v.value);
        let take_island = match (isl, pv) {
            (None, None) => break,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (Some((ki, _)), Some(kp)) => ki >= kp,
        };
        if take_island {
            let (key, x) = isl.expect("an island");
            if key < params.floor {
                break;
            }
            let bytes = gr.terms[x].bytes;
            if bytes > left {
                gr.skip(x);
                continue;
            }
            left -= bytes;
            out.island_saving_ns += gr.take(x);
            out.island_bytes += bytes;
            out.islands.push(gr.terms[x].label);
            cut = key;
        } else {
            let v = &values[pi];
            if v.value < params.floor {
                // The pairs are by decreasing value: none of the rest passes the floor.
                pi = values.len();
                continue;
            }
            if v.bytes <= left {
                left -= v.bytes;
                out.pair_saving_ns += v.prob * v.saving_ns;
                out.pair_bytes += v.bytes;
                out.pairs.push(pi);
                cut = v.value;
            }
            pi += 1;
        }
    }
    out.islands.sort_unstable();
    out.n_pairs = out.pairs.len();
    out.cut = if cut.is_finite() { cut } else { 0.0 };
    out.pairs_pay = out.pairs.is_empty()
        || out.pair_saving_ns >= params.floor * (out.pair_bytes + fixed) as f64;
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
    // Big labels at `T`: every label of at least `T` (and fewer than `n`) points, as in B3a's
    // structure — the matrix then tells the `pair` plan which of their pairs never co-occur,
    // which OR-clause covers need even where no pair of theirs is stored.
    let n = labels.n_points();
    let cand_labels: Vec<u32> = (0..labels.n_labels() as u32)
        .filter(|&l| {
            let s = labels.posting_len(l);
            s as u64 >= pairs::T_MIN && s < n
        })
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
        .map(|&t| run(labels, sample, values, &cand_labels, params, t))
        .collect();
    let best = runs
        .iter()
        .filter(|r| r.pairs_pay)
        .max_by(|x, y| {
            x.saving_ns()
                .total_cmp(&y.saving_ns())
                .then(y.bytes().cmp(&x.bytes()))
        })
        .cloned()
        .unwrap_or_default();
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
}
