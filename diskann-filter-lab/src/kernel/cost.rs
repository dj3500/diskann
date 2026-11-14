/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Cost model of the exact path and of graph search, and the per-query estimates it reads.
//!
//! Unit costs ([`UnitCosts`]) are measured once per machine and dataset by `flab calibrate`
//! (nanoseconds; memory-access costs as functions of the candidate density, i.e. the fraction
//! of the data points touched in id order: sparse candidates pay random accesses, dense ones
//! stream). Nothing is fitted on queries.
//!
//! Exact plans ([`ExactPlan`]) for a compiled predicate whose first clause in Boolean order
//! (`c*`, the smallest posting sum `P*`) has the union `A = |union(c*)|` and the match set `M`:
//!
//! ```text
//! scan            merge(c*) + A·pred(A/n) + M·dist(M/n)
//! distance_first  merge(c*) + A·dist(A/n) + E·pred(E/n),   E = predicate evaluations inside
//!                 the answer bound: A if M ≤ K, else (K·A/M)·(1 + ln(M/K)) (2·K·A/M when the
//!                 heap is seeded)
//! gallop          merge(c*) + Σ_{j≥1} Σ_{ℓ∈c_j} probes(S_{j−1}, |post ℓ|) + M·dist(M/n),
//!                 S_j = A·survival_j, probes(s, p) = gallop·s·(1 + log2(1 + p/s))
//! bitmap          Σ_j (n/64 words per bitset label, |post ℓ| scatters otherwise, n/64 per
//!                 clause fill and AND) + n/64 enumerated words + M·dist(M/n)
//! ```
//!
//! with `merge(c*) = P*·merge·log2 k` for a clause of `k ≥ 2` labels (a single posting list is
//! read in place). Graph search at list size `L` costs `(α·L + β)·(dist_g + pred_g)`.

use serde::{Deserialize, Serialize};

use super::eval::Compiled;
use crate::{bitsets::LabelBitsets, labels::LabelStore};

/// Unit costs in nanoseconds (see the module documentation).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnitCosts {
    /// Candidate densities (fractions of the data points, increasing) of the two curves below.
    pub density: Vec<f64>,
    /// Per distance computation over id-ordered candidates of each density (prefetched).
    pub dist_ns: Vec<f64>,
    /// Per exact-path predicate check (Boolean, full predicate) at each density.
    pub pred_ns: Vec<f64>,
    /// Per posting entry and binary level of a k-way merge with deduplication.
    pub merge_ns: f64,
    /// Per galloping search of one candidate in one posting list (per probe step).
    pub gallop_ns: f64,
    /// Per 64-bit bitmap word written, ORed or ANDed.
    pub word_ns: f64,
    /// Per posting entry scattered into a bitmap.
    pub scatter_ns: f64,
    /// Per bitmap word enumerated.
    pub enum_ns: f64,
    /// Per sampled point of a sketch probe (row read and every clause, batch-prefetched).
    pub probe_ns: f64,
    /// Per sketch entry merged by a probe.
    pub sketch_ns: f64,
    /// Per distance computation of a graph search (adjacency, visited set and beam included).
    pub graph_dist_ns: f64,
    /// Per predicate evaluation of a graph search.
    pub graph_pred_ns: f64,
    /// Distance computations of a graph search per unit of list size `L`.
    pub graph_alpha: f64,
    /// Distance computations of a graph search at `L = 0` (starts, first steps).
    pub graph_beta: f64,
}

impl Default for UnitCosts {
    /// Indicative values for 64-dimensional `u8` data at 1M points (for tests; experiments
    /// use calibrated costs).
    fn default() -> Self {
        Self {
            density: vec![1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0],
            dist_ns: vec![35.0, 33.0, 28.0, 15.0, 9.0, 7.0],
            pred_ns: vec![160.0, 150.0, 120.0, 60.0, 35.0, 25.0],
            merge_ns: 4.0,
            gallop_ns: 8.0,
            word_ns: 0.6,
            scatter_ns: 2.0,
            enum_ns: 0.8,
            probe_ns: 120.0,
            sketch_ns: 6.0,
            graph_dist_ns: 60.0,
            graph_pred_ns: 80.0,
            graph_alpha: 10.0,
            graph_beta: 200.0,
        }
    }
}

impl UnitCosts {
    /// Checks the shape and signs of the costs.
    pub fn validate(&self) -> anyhow::Result<()> {
        let n = self.density.len();
        anyhow::ensure!(
            n >= 1 && self.dist_ns.len() == n && self.pred_ns.len() == n,
            "the density curves need equal, non-zero lengths"
        );
        anyhow::ensure!(
            self.density.windows(2).all(|w| w[0] < w[1])
                && self.density.iter().all(|&d| d > 0.0 && d <= 1.0),
            "densities must increase within (0, 1]"
        );
        let scalars = [
            self.merge_ns,
            self.gallop_ns,
            self.word_ns,
            self.scatter_ns,
            self.enum_ns,
            self.probe_ns,
            self.sketch_ns,
            self.graph_dist_ns,
            self.graph_pred_ns,
            self.graph_alpha,
            self.graph_beta,
        ];
        anyhow::ensure!(
            scalars
                .iter()
                .chain(&self.dist_ns)
                .chain(&self.pred_ns)
                .all(|x| x.is_finite() && *x >= 0.0),
            "unit costs must be finite and non-negative"
        );
        Ok(())
    }

    /// Log-linear interpolation of a density curve (clamped at the ends).
    fn curve(&self, v: &[f64], rho: f64) -> f64 {
        let d = &self.density;
        if rho <= d[0] {
            return v[0];
        }
        let last = d.len() - 1;
        if rho >= d[last] {
            return v[last];
        }
        let i = d.partition_point(|&x| x <= rho);
        let (x0, x1) = (d[i - 1].ln(), d[i].ln());
        let t = (rho.ln() - x0) / (x1 - x0);
        v[i - 1] + t * (v[i] - v[i - 1])
    }

    /// Cost of one distance computation over id-ordered candidates of density `rho`.
    pub fn dist(&self, rho: f64) -> f64 {
        self.curve(&self.dist_ns, rho)
    }

    /// Cost of one exact-path predicate check at density `rho`.
    pub fn pred(&self, rho: f64) -> f64 {
        self.curve(&self.pred_ns, rho)
    }

    /// Distance computations of a graph search at list size `l`.
    pub fn graph_dists(&self, l: f64) -> f64 {
        self.graph_alpha * l + self.graph_beta
    }

    /// Nominal cost of a graph search at list size `l` (every distance with a predicate).
    pub fn graph(&self, l: f64) -> f64 {
        self.graph_dists(l) * (self.graph_dist_ns + self.graph_pred_ns)
    }

    /// The model cost of graph work already done.
    #[inline]
    pub fn charged(&self, n_dist: u64, n_pred: u64) -> f64 {
        n_dist as f64 * self.graph_dist_ns + n_pred as f64 * self.graph_pred_ns
    }
}

/// An exact plan (see the module documentation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ExactPlan {
    /// Scan of `c*`'s union, predicate first (prefetched rows), distances of the matches.
    Scan = 1,
    /// Scan of `c*`'s union, distances first; predicates only inside the answer bound.
    DistanceFirst = 2,
    /// Galloping intersection of the clauses' postings, then distances.
    Gallop = 3,
    /// Dense match bitmap (static bitsets for frequent labels), then distances.
    Bitmap = 4,
}

impl ExactPlan {
    pub const ALL: [Self; 4] = [Self::Scan, Self::DistanceFirst, Self::Gallop, Self::Bitmap];

    pub fn name(self) -> &'static str {
        match self {
            Self::Scan => "scan",
            Self::DistanceFirst => "df",
            Self::Gallop => "gallop",
            Self::Bitmap => "bitmap",
        }
    }

    pub fn parse(s: &str) -> anyhow::Result<Self> {
        Ok(match s.trim() {
            "scan" => Self::Scan,
            "df" | "distance_first" => Self::DistanceFirst,
            "gallop" => Self::Gallop,
            "bitmap" => Self::Bitmap,
            _ => anyhow::bail!("unknown exact plan {s:?} (scan, df, gallop, bitmap)"),
        })
    }
}

/// A set of exact plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlanSet(u8);

impl PlanSet {
    pub const ALL: Self = Self(0b1111);

    pub fn contains(self, p: ExactPlan) -> bool {
        self.0 & (1 << (p as u8 - 1)) != 0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Parses `scan+df+gallop+bitmap` (any non-empty subset).
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        let mut m = 0u8;
        for p in s.split('+') {
            m |= 1 << (ExactPlan::parse(p)? as u8 - 1);
        }
        Ok(Self(m))
    }

    pub fn plans(self) -> impl Iterator<Item = ExactPlan> {
        ExactPlan::ALL
            .into_iter()
            .filter(move |&p| self.contains(p))
    }
}

impl std::fmt::Display for PlanSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self.plans().map(ExactPlan::name).collect();
        f.write_str(&names.join("+"))
    }
}

/// Per-query estimates of the exact path's work.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Estimates {
    /// Data points.
    pub n: f64,
    /// `|union(c*)|` and an upper confidence bound.
    pub a: f64,
    pub a_hi: f64,
    /// Matches, with lower and upper confidence bounds.
    pub m: f64,
    pub m_lo: f64,
    pub m_hi: f64,
    /// `survival[j − 1]`: fraction of `union(c*)` satisfying clauses `1..=j` of the Boolean
    /// order (`j = 1..C`).
    pub survival: Vec<f64>,
    /// The union was enumerated completely (`A` and `M` are exact).
    pub exact: bool,
    /// Sampled points of the probe (0 without a probe).
    pub sampled: usize,
}

/// `z` of a two-sided `1 − δ` normal interval (δ = 0.01).
pub const Z: f64 = 2.576;

/// Wilson score interval of a binomial proportion `x / n` at `z`.
pub fn wilson(x: usize, n: usize, z: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let (x, n) = (x as f64, n as f64);
    let p = x / n;
    let z2 = z * z;
    let den = 1.0 + z2 / n;
    let centre = (p + z2 / (2.0 * n)) / den;
    let half = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / den;
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

/// Quantile of the standard gamma distribution with shape `k` at the normal deviate `z`
/// (Wilson–Hilferty).
pub fn gamma_quantile(k: f64, z: f64) -> f64 {
    let c = 1.0 / (9.0 * k);
    (k * (1.0 - c + z * c.sqrt()).powi(3)).max(0.0)
}

impl Estimates {
    /// Posting-only estimates: `A = min(n, P*)` (exact for one label), clause survival from the
    /// clause bounds `min(1, U_j / n)` multiplied (an independence guess), `M = A·Π`.
    pub fn from_bounds(q: &Compiled, n: usize) -> Self {
        let nf = n as f64;
        let c = q.n_clauses();
        let a = if c == 0 {
            nf
        } else {
            (q.upper_bound(0) as f64).min(nf)
        };
        let mut survival = Vec::with_capacity(c.saturating_sub(1));
        let mut s = 1.0;
        for j in 1..c {
            s *= (q.upper_bound(j) as f64 / nf).min(1.0);
            survival.push(s);
        }
        let m = a * s;
        Self {
            n: nf,
            a,
            a_hi: a,
            m,
            m_lo: m,
            m_hi: a,
            survival,
            exact: false,
            sampled: 0,
        }
    }

    /// Estimates from a uniform sample of `union(c*)` of `sampled` points (the whole union if
    /// `complete`) whose last priority fraction is `tau`, of which the first `evaluated` were
    /// checked: `matches` of them satisfy every clause and `surv[j − 1]` satisfy clauses
    /// `1..=j`. `a_min` / `a_max` are the posting bounds of the union (largest label,
    /// `min(n, P*)`).
    #[expect(clippy::too_many_arguments, reason = "the probe's outputs")]
    pub fn from_sample(
        n: usize,
        sampled: usize,
        complete: bool,
        tau: f64,
        evaluated: usize,
        matches: usize,
        surv: &[usize],
        a_min: f64,
        a_max: f64,
    ) -> Self {
        let nf = n as f64;
        if (complete && evaluated == sampled) || sampled == 0 {
            let a = sampled as f64;
            let m = matches as f64;
            let den = (sampled as f64).max(1.0);
            return Self {
                n: nf,
                a,
                a_hi: a,
                m,
                m_lo: m,
                m_hi: m,
                survival: surv.iter().map(|&s| s as f64 / den).collect(),
                exact: true,
                sampled,
            };
        }
        let b = sampled as f64;
        let (a, a_lo, a_hi) = if complete {
            (b, b, b)
        } else {
            let clip = |x: f64| x.clamp(a_min.min(a_max), a_max);
            (
                clip((b - 1.0).max(1.0) / tau),
                clip(gamma_quantile(b, -Z) / tau),
                clip(gamma_quantile(b, Z) / tau),
            )
        };
        let e = evaluated.max(1) as f64;
        let f = matches as f64 / e;
        let (f_lo, f_hi) = wilson(matches, evaluated, Z);
        Self {
            n: nf,
            a,
            a_hi,
            m: a * f,
            m_lo: a_lo * f_lo,
            m_hi: (a_hi * f_hi).min(a_hi),
            survival: surv.iter().map(|&s| s as f64 / e).collect(),
            exact: false,
            sampled: evaluated,
        }
    }

    /// The conservative counts of an upper cost bound: `A = min(n, P*)`, `M = A`, no
    /// filtering by the residual clauses.
    pub fn upper(q: &Compiled, n: usize) -> Self {
        let mut e = Self::from_bounds(q, n);
        e.m = e.a;
        e.m_lo = e.a;
        e.m_hi = e.a;
        for s in &mut e.survival {
            *s = 1.0;
        }
        e
    }
}

/// Inputs of the exact plan costs of one query.
#[derive(Debug, Clone, Copy)]
pub struct PlanInputs<'a> {
    pub q: &'a Compiled,
    pub labels: &'a LabelStore,
    pub bitsets: Option<&'a LabelBitsets>,
    pub k: usize,
}

impl PlanInputs<'_> {
    /// Cost of merging `c*`'s postings into distinct candidates.
    fn merge(&self, c: &UnitCosts) -> f64 {
        if self.q.n_clauses() == 0 {
            return 0.0;
        }
        let k0 = self
            .q
            .clause(0)
            .iter()
            .filter(|&&l| self.labels.posting_len(l) > 0)
            .count();
        if k0 <= 1 {
            0.0
        } else {
            self.q.upper_bound(0) as f64 * c.merge_ns * (k0 as f64).log2()
        }
    }

    /// Cost of `plan` under the estimates `e` (`seeded`: the answer heap starts full;
    /// `upper`: worst case of the evaluations of `distance_first`).
    pub fn cost(
        &self,
        c: &UnitCosts,
        plan: ExactPlan,
        e: &Estimates,
        seeded: bool,
        upper: bool,
    ) -> f64 {
        let n = e.n.max(1.0);
        let (a, m) = (e.a, e.m.min(e.a));
        let checks = self.q.n_clauses() > 1;
        let k = self.k as f64;
        let dist_m = m * c.dist(m / n);
        match plan {
            ExactPlan::Scan => {
                let pred = if checks { a * c.pred(a / n) } else { 0.0 };
                self.merge(c) + pred + dist_m
            }
            ExactPlan::DistanceFirst => {
                let evals = if !checks {
                    0.0
                } else if upper || m <= k {
                    a
                } else if seeded {
                    (2.0 * k * a / m).min(a)
                } else {
                    (k * a / m * (1.0 + (m / k).ln())).min(a)
                };
                self.merge(c) + a * c.dist(a / n) + evals * c.pred(evals / n)
            }
            ExactPlan::Gallop => {
                let mut cost = self.merge(c) + a * c.word_ns;
                let mut s = a;
                for j in 1..self.q.n_clauses() {
                    for &l in self.q.clause(j) {
                        let p = self.labels.posting_len(l) as f64;
                        if s >= 1.0 {
                            cost += c.gallop_ns * s * (1.0 + (1.0 + p / s).log2());
                        }
                    }
                    s = a * e.survival.get(j - 1).copied().unwrap_or(1.0);
                }
                cost + dist_m
            }
            ExactPlan::Bitmap => {
                let words = (n / 64.0).ceil();
                let mut cost = words * (c.enum_ns + c.word_ns);
                for j in 0..self.q.n_clauses() {
                    cost += if j == 0 { 1.0 } else { 2.0 } * words * c.word_ns;
                    for &l in self.q.clause(j) {
                        cost += match self.bitsets.and_then(|b| b.slot(l)) {
                            Some(_) => words * c.word_ns,
                            None => self.labels.posting_len(l) as f64 * c.scatter_ns,
                        };
                    }
                }
                cost + dist_m
            }
        }
    }

    /// The cheapest plan of `plans` and its cost.
    pub fn best(
        &self,
        c: &UnitCosts,
        plans: PlanSet,
        e: &Estimates,
        seeded: bool,
        upper: bool,
    ) -> (ExactPlan, f64) {
        let mut best = (ExactPlan::Scan, f64::INFINITY);
        for p in plans.plans() {
            let x = self.cost(c, p, e, seeded, upper);
            if x < best.1 {
                best = (p, x);
            }
        }
        best
    }
}

/// Upstream DiskANN's adaptive-L multiplier at match rate `rate`: 1 (≥ 50 %), 2 (10–50 %),
/// `2^{−log10 rate}` below 10 %, clamped to `[1, scale]` (`scale` at rate 0).
pub fn l_multiplier(rate: f64, scale: f64) -> f64 {
    if rate.is_nan() || rate <= 0.0 {
        return scale.max(1.0);
    }
    let m = if rate >= 0.5 {
        1.0
    } else if rate >= 0.1 {
        2.0
    } else {
        2.0_f64.powf(-rate.log10())
    };
    m.clamp(1.0, scale.max(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{kernel::eval::Weights, labels::LabelStoreBuilder, predicate::PredicateSetBuilder};

    #[test]
    fn curves_interpolate_and_clamp() {
        let c = UnitCosts::default();
        c.validate().unwrap();
        assert_eq!(c.dist(1e-9), 35.0);
        assert_eq!(c.dist(2.0), 7.0);
        assert_eq!(c.dist(1e-3), 33.0);
        let mid = c.dist(10f64.powf(-3.5));
        assert!((mid - 34.0).abs() < 1e-9, "{mid}");
        let mut bad = c.clone();
        bad.density = vec![0.5, 0.1, 0.2, 0.3, 0.4, 1.0];
        assert!(bad.validate().is_err());
        let text = serde_json::to_string(&c).unwrap();
        assert_eq!(serde_json::from_str::<UnitCosts>(&text).unwrap(), c);
    }

    #[test]
    fn intervals_cover_their_estimates() {
        let (lo, hi) = wilson(10, 100, Z);
        assert!(lo < 0.1 && 0.1 < hi && lo > 0.03 && hi < 0.25, "{lo} {hi}");
        assert_eq!(wilson(0, 50, Z).0, 0.0);
        assert!(wilson(0, 50, Z).1 > 0.05);
        let g = gamma_quantile(128.0, 0.0);
        assert!((g - 127.7).abs() < 0.5, "{g}");
        let (lo, hi) = (gamma_quantile(128.0, -Z), gamma_quantile(128.0, Z));
        assert!(
            (99.0..103.0).contains(&lo) && (155.0..160.0).contains(&hi),
            "{lo} {hi}"
        );
        assert_eq!(l_multiplier(0.7, 16.0), 1.0);
        assert_eq!(l_multiplier(0.2, 16.0), 2.0);
        assert!((l_multiplier(0.01, 16.0) - 4.0).abs() < 1e-9);
        assert_eq!(l_multiplier(0.0, 16.0), 16.0);
        assert_eq!(l_multiplier(1e-9, 8.0), 8.0);
    }

    #[test]
    fn plan_costs_follow_the_estimates() {
        let mut b = LabelStoreBuilder::new();
        for i in 0..1000u32 {
            let a = format!("a={}", i % 10);
            let c = format!("c={}", i % 2);
            b.push_point([("a", a.as_str()), ("c", c.as_str())])
                .unwrap();
        }
        let s = b.finish().unwrap();
        let mut pb = PredicateSetBuilder::new(&s);
        pb.push(&[vec!["a=1", "a=2"], vec!["c=0"]]);
        pb.push(&[vec!["a=1"]]);
        let preds = pb.finish();
        let mut bits = vec![0u64; s.n_labels()];
        let mut q = Compiled::new();
        q.compile(&s, preds.get(0), Weights::Unit, None, &mut bits);
        let e = Estimates::from_bounds(&q, 1000);
        assert_eq!((e.a, e.survival.clone()), (200.0, vec![0.5]));
        assert_eq!(e.m, 100.0);
        let up = Estimates::upper(&q, 1000);
        assert_eq!((up.m, up.survival.clone()), (200.0, vec![1.0]));
        let c = UnitCosts::default();
        let pi = PlanInputs {
            q: &q,
            labels: &s,
            bitsets: None,
            k: 10,
        };
        for p in ExactPlan::ALL {
            let x = pi.cost(&c, p, &e, false, false);
            assert!(x.is_finite() && x > 0.0, "{p:?}");
            assert!(pi.cost(&c, p, &up, false, true) >= x, "{p:?}: upper bound");
        }
        // Distance-first evaluates fewer predicates once seeded.
        assert!(
            pi.cost(&c, ExactPlan::DistanceFirst, &e, true, false)
                <= pi.cost(&c, ExactPlan::DistanceFirst, &e, false, false)
        );
        let (_, best) = pi.best(&c, PlanSet::ALL, &e, false, false);
        let only_scan = PlanSet::parse("scan").unwrap();
        assert!(best <= pi.best(&c, only_scan, &e, false, false).1);
        assert_eq!(only_scan.to_string(), "scan");
        assert_eq!(
            PlanSet::parse("scan+df+gallop+bitmap").unwrap(),
            PlanSet::ALL
        );
        q.clear_bits(&mut bits);
    }
}
