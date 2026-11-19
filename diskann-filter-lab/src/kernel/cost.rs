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
//!                 clause fill and AND) + n/64 enumerated words + M·dist(M/n) (an upper bound
//!                 since A6b: built block by block in L1, empty blocks skip their remaining
//!                 clauses)
//! rowand          words(c*) + Σ_{j≥1} W·reach_j·(base(ρ) + hot_j·word(ρ))
//!                 + Σ_{j≥1} Σ_{ℓ rare in c_j} probes(min(W·reach_j, P_j), |post ℓ|) + M·enum_bit
//!                 + M·dist(M/n)
//! pair            lists·gallop_cold + E·merge·log2(lists) + E·pred_mask(E/n) + min(M, E)·dist(M/n)
//!                 (combination postings, B3a: the `E` entries of the stored joint postings —
//!                 and of the smaller label posting of every pair not stored — of the clause
//!                 pair with the fewest; the checks cover the other clauses, all clauses when a
//!                 label posting is among the lists, at the worst-case work of those clauses
//!                 (`MemberWork::of_mask`: every bitset word, the row if one has row labels);
//!                 `M = E` for two clauses)
//! ```
//!
//! with `merge(c*) = P*·merge·log2 k` for a clause of `k ≥ 2` labels (a single posting list is
//! read in place). The bitmap's enumeration costs `n/64·enum_word + nz·enum_nz + M·enum_bit`
//! (`nz` = the words holding the `M` matches) when those are calibrated (batch A6; older calibrations: `n/64·enum`, measured on a bitmap with 10 % of its
//! bits set, which prices every query as if its bitmap were that dense); A6 calibrations also
//! price a later clause of one bitset label as one AND pass, as built. `rowand` (A6: the AND
//! of the clauses' rows, word by word) visits the `W = n/64·(1 − e^{−64A/n})` words holding
//! members of `union(c*)` (`words(c*)`: `P*` scatters or reads of `c*`'s postings, or the
//! `n/64` words of its bitset rows streamed when it has frequent labels), ANDs each with the
//! residual clauses (`hot_j` bitset words per word reached, priced like a member check's words
//! with the shape-resolved curves; labels without a bitset by galloping cursors over their
//! postings, moved only when a word reaches the clause's next entry: at most
//! `min(W·reach_j, P_j)` times, priced as labels of the clause's mean posting length, plus
//! each cursor's first seek over its list at the largest calibrated gap's step cost) and
//! stops at the first clause that clears it (`reach_j`: the chance that a word of `A/W`
//! members survives clauses `1..j`); it is a candidate only with the A6 enumeration costs.
//! The candidates checked by `scan` and `distance_first` are members of `union(c*)`: `pred`
//! is the cost of checking a member (the residual clauses). Calibrations
//! with the shape-resolved curves (batch A1b) price it from the query's residual clauses:
//! `base(ρ) + words·word(ρ) + prefetch·pf(ρ) + reads·row(ρ)`, with the bitset words probed and
//! the row reads expected from the clauses' labels and the survival estimates
//! ([`MemberWork`]); every member's row is prefetched (`prefetch` = 1) when a residual clause
//! has labels without a bitset, read or not. They are measured cold (a scan touches each
//! candidate once). Batch A1 calibrations use one curve per class
//! (residual clauses reading rows or bitset-only: `pred_row`, `pred_bits`, measured on
//! re-sampled, hence warm, members), batch B6 the miss-dominated `pred` curve; the cost of a
//! galloping step depends on the gap `p/s` (`gallop(p/s)`, cache misses in long lists), B6
//! used a constant. The sketch probe costs `sketch·entries merged + points·pred` (every
//! residual clause per sampled point); with the shape-resolved curves it enters the crossover
//! (the exact path skips the probe when its upper cost is below the cheapest graph search plus
//! the probe, which the graph route needs for its list size).
//!
//! Graph search at list size `L` computes `D(L)` distances (measured at list sizes from 10 to
//! 2,560 and interpolated log-log; older calibrations: the line `α·L + β`) and evaluates `P`
//! predicates: `P = D` when every distance pays one (eager), `P = min(D, K/r + extra(K, L))`
//! under `heap_bound` laziness (the heap fills after about `K/r` evaluations at match rate
//! `r`; afterwards the nodes found inside the K-th distance are checked, a number that grows
//! sublinearly with `L`: measured per list size, older calibrations one `extra(K)`),
//! `P = φ·D` under `distance_first`; it costs `D·dist_g(L) + P·pred_g`. The laziness terms are
//! calibrated; without them `P = D` (batch B6). `dist_g(L)` is the calibrated cost per
//! distance, raised where a predicate-free search at list size `L` costs more per distance
//! (long beams; `graph_dist_ns_curve`); without that curve it is a constant.

use serde::{Deserialize, Serialize};

use super::{config::Laziness, eval::Compiled};
use crate::{bitsets::LabelBitsets, labels::LabelStore, pairs::Cover};

/// Planner caches of [`UnitCosts`], filled by [`UnitCosts::prepare`]: the logarithms of the
/// calibrated grids (empty until then: taken per call, with the same values), the cheapest
/// galloping step, and whether every unit cost is non-negative (the lower bounds that let
/// [`PlanInputs::best`] skip plans hold). Not serialized; ignored by comparisons.
#[derive(Debug, Clone, Default)]
pub struct Prepared {
    ln_density: Vec<f64>,
    ln_gallop_gap: Vec<f64>,
    gallop_min: f64,
    bounds: bool,
}

impl PartialEq for Prepared {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// `(ln g[i − 1], ln g[i])`, from `cache` when it covers `g`.
#[inline]
fn ln_pair(cache: &[f64], g: &[f64], i: usize) -> (f64, f64) {
    if cache.len() == g.len() {
        (cache[i - 1], cache[i])
    } else {
        (g[i - 1].ln(), g[i].ln())
    }
}

/// Position of a density on the calibrated grid: below it, at or above its last point
/// (index), or between points `i − 1` and `i` at log-linear weight `t` (one interpolation
/// for every curve at that density).
#[derive(Debug, Clone, Copy)]
enum Pos {
    Lo,
    Hi(usize),
    At(usize, f64),
}

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
    /// Per bitmap word enumerated (a bitmap with 10 % of its bits set: about 6.4 per word).
    pub enum_ns: f64,
    /// Per bitmap word enumerated without set bits, per set bit enumerated, and per word with
    /// set bits (its loop exit mispredicts; batch A6; 0 in older calibrations: then `enum_ns`
    /// per word, and no `rowand` plan; `enum_nz_ns` 0: the set bits carry it).
    #[serde(default)]
    pub enum_word_ns: f64,
    #[serde(default)]
    pub enum_bit_ns: f64,
    #[serde(default)]
    pub enum_nz_ns: f64,
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
    /// Per check of a member of `union(c*)` (the residual clauses) at each density, for
    /// queries whose residual clauses read rows (labels without a static bitset) and for
    /// bitset-only ones; empty in older calibrations (then `pred_ns`).
    #[serde(default)]
    pub pred_row_ns: Vec<f64>,
    #[serde(default)]
    pub pred_bits_ns: Vec<f64>,
    /// Gaps `p/s` (posting entries per galloped candidate, increasing) and the cost of one
    /// modelled galloping step at each; empty in older calibrations (then `gallop_ns`).
    #[serde(default)]
    pub gallop_gap: Vec<f64>,
    #[serde(default)]
    pub gallop_gap_ns: Vec<f64>,
    /// `heap_bound` graph searches: predicate evaluations beyond `K/r` (`extra(K)`) at each
    /// heap size `K` (increasing); empty in older calibrations (then one per distance).
    #[serde(default)]
    pub heap_bound_k: Vec<f64>,
    #[serde(default)]
    pub heap_bound_extra: Vec<f64>,
    /// `distance_first` graph searches: predicate evaluations per distance computation (0:
    /// not calibrated, then one per distance).
    #[serde(default)]
    pub distance_first_frac: f64,
    /// Distance computations of graph searches measured at the list sizes `graph_l`
    /// (increasing): `graph_l_dists`, interpolated log-log; empty in older calibrations (then
    /// the line `graph_alpha · L + graph_beta`).
    #[serde(default)]
    pub graph_l: Vec<f64>,
    #[serde(default)]
    pub graph_l_dists: Vec<f64>,
    /// `heap_bound` searches: `extra(K, L)` at the heap sizes `heap_bound_k` (rows) and the
    /// list sizes `heap_bound_l` (columns, increasing); empty in older calibrations (then
    /// `heap_bound_extra`, one value per heap size).
    #[serde(default)]
    pub heap_bound_l: Vec<f64>,
    #[serde(default)]
    pub heap_bound_extra_kl: Vec<Vec<f64>>,
    /// Checks of members of `union(c*)` by residual shape, measured cold (each candidate
    /// touched once) at each density: per candidate (`check_base_ns`), per static-bitset
    /// word probed (`check_word_ns`), per row prefetched (`check_prefetch_ns`: the exact path
    /// prefetches every member's row when a residual clause has labels without a bitset, read
    /// or not) and per row read and checked after it (`check_row_ns`); empty in older
    /// calibrations (then `pred_row_ns` / `pred_bits_ns`; without `check_prefetch_ns` the row
    /// cost includes the prefetch). With them the sketch probe costs the same per sampled
    /// point (every residual clause evaluated) and enters the crossover.
    #[serde(default)]
    pub check_base_ns: Vec<f64>,
    #[serde(default)]
    pub check_word_ns: Vec<f64>,
    #[serde(default)]
    pub check_prefetch_ns: Vec<f64>,
    #[serde(default)]
    pub check_row_ns: Vec<f64>,
    /// Per query: compiling the predicate and planning (0 if not calibrated).
    #[serde(default)]
    pub query_ns: f64,
    /// Not calibrated: the kernel's `row_reads = reach` (ablation of A1c's correction): a
    /// member check reads its row whenever it reaches a clause with labels without a bitset.
    #[serde(skip)]
    pub reads_at_reach: bool,
    /// Graph search: list sizes (increasing) and ns per distance computation of a
    /// predicate-free search at each; empty in older calibrations (then `graph_dist_ns` at
    /// every size).
    #[serde(default)]
    pub graph_dist_ns_l: Vec<f64>,
    #[serde(default)]
    pub graph_dist_ns_curve: Vec<f64>,
    /// Unfiltered effort curve of the graph (base-derived queries): `effort_l[i][j]` is the
    /// list size at which the mean unfiltered recall at `effort_k[j]` neighbours reaches
    /// `effort_r[i]` (both increasing); empty when not calibrated.
    #[serde(default)]
    pub effort_r: Vec<f64>,
    #[serde(default)]
    pub effort_k: Vec<f64>,
    #[serde(default)]
    pub effort_l: Vec<Vec<f64>>,
    /// Planner caches ([`Self::prepare`]).
    #[serde(skip)]
    pub prepared: Prepared,
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
            enum_word_ns: 0.0,
            enum_bit_ns: 0.0,
            enum_nz_ns: 0.0,
            probe_ns: 120.0,
            sketch_ns: 6.0,
            graph_dist_ns: 60.0,
            graph_pred_ns: 80.0,
            graph_alpha: 10.0,
            graph_beta: 200.0,
            pred_row_ns: Vec::new(),
            pred_bits_ns: Vec::new(),
            gallop_gap: Vec::new(),
            gallop_gap_ns: Vec::new(),
            heap_bound_k: Vec::new(),
            heap_bound_extra: Vec::new(),
            distance_first_frac: 0.0,
            graph_l: Vec::new(),
            graph_l_dists: Vec::new(),
            heap_bound_l: Vec::new(),
            heap_bound_extra_kl: Vec::new(),
            check_base_ns: Vec::new(),
            check_word_ns: Vec::new(),
            check_prefetch_ns: Vec::new(),
            check_row_ns: Vec::new(),
            query_ns: 0.0,
            reads_at_reach: false,
            graph_dist_ns_l: Vec::new(),
            graph_dist_ns_curve: Vec::new(),
            effort_r: Vec::new(),
            effort_k: Vec::new(),
            effort_l: Vec::new(),
            prepared: Prepared::default(),
        }
    }
}

/// Log-log interpolation of the positive values `ys` over the increasing positive `xs` at `x`,
/// extended beyond the ends with the end segments' slopes (a power law; one point: constant).
fn loglog(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len();
    if n == 1 {
        return ys[0];
    }
    let i = xs.partition_point(|&v| v <= x).clamp(1, n - 1);
    let (x0, x1) = (xs[i - 1].ln(), xs[i].ln());
    let (y0, y1) = (ys[i - 1].ln(), ys[i].ln());
    let t = (x.max(1e-9).ln() - x0) / (x1 - x0);
    (y0 + t * (y1 - y0)).exp()
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
            (self.pred_row_ns.is_empty() && self.pred_bits_ns.is_empty())
                || (self.pred_row_ns.len() == n && self.pred_bits_ns.len() == n),
            "the member check curves need one value per density"
        );
        anyhow::ensure!(
            (self.check_base_ns.is_empty()
                && self.check_word_ns.is_empty()
                && self.check_row_ns.is_empty())
                || (self.check_base_ns.len() == n
                    && self.check_word_ns.len() == n
                    && self.check_row_ns.len() == n),
            "the shape-resolved check curves need one value per density"
        );
        anyhow::ensure!(
            self.check_prefetch_ns.is_empty()
                || (self.check_prefetch_ns.len() == n && self.check_base_ns.len() == n),
            "the row prefetch curve needs the shape-resolved curves, one value per density"
        );
        anyhow::ensure!(
            self.gallop_gap.len() == self.gallop_gap_ns.len()
                && self.gallop_gap.windows(2).all(|w| w[0] < w[1])
                && self.gallop_gap.iter().all(|&g| g >= 1.0),
            "galloping gaps must increase from at least 1, one cost each"
        );
        anyhow::ensure!(
            self.heap_bound_k.len() == self.heap_bound_extra.len()
                && self.heap_bound_k.windows(2).all(|w| w[0] < w[1])
                && self.heap_bound_k.iter().all(|&k| k >= 1.0),
            "heap_bound sizes must increase from at least 1, one value each"
        );
        anyhow::ensure!(
            self.graph_l.len() == self.graph_l_dists.len()
                && self.graph_l.windows(2).all(|w| w[0] < w[1])
                && self.graph_l.iter().all(|&l| l >= 1.0)
                && self.graph_l_dists.iter().all(|&d| d > 0.0),
            "graph list sizes must increase from at least 1, one positive distance count each"
        );
        anyhow::ensure!(
            self.heap_bound_l.windows(2).all(|w| w[0] < w[1])
                && self.heap_bound_l.iter().all(|&l| l >= 1.0)
                && (self.heap_bound_l.is_empty() && self.heap_bound_extra_kl.is_empty()
                    || self.heap_bound_extra_kl.len() == self.heap_bound_k.len()
                        && self
                            .heap_bound_extra_kl
                            .iter()
                            .all(|r| r.len() == self.heap_bound_l.len())),
            "extra(K, L) needs one row per heap size and one value per list size"
        );
        anyhow::ensure!(
            self.graph_dist_ns_l.len() == self.graph_dist_ns_curve.len()
                && self.graph_dist_ns_l.windows(2).all(|w| w[0] < w[1])
                && self.graph_dist_ns_l.iter().all(|&l| l >= 1.0)
                && self
                    .graph_dist_ns_curve
                    .iter()
                    .all(|x| x.is_finite() && *x >= 0.0),
            "the graph cost by list size needs increasing sizes (>= 1), one cost each"
        );
        anyhow::ensure!(
            self.effort_l.len() == self.effort_r.len()
                && self.effort_l.iter().all(|row| row.len() == self.effort_k.len())
                && self.effort_r.windows(2).all(|w| w[0] < w[1])
                && self.effort_k.windows(2).all(|w| w[0] < w[1])
                && self.effort_k.iter().all(|&k| k >= 1.0)
                && self.effort_l.iter().flatten().all(|&l| l.is_finite() && l >= 1.0),
            "the effort curve needs increasing recalls and neighbour counts and one list size (>= 1) each"
        );
        anyhow::ensure!(
            scalars
                .iter()
                .chain(&[self.distance_first_frac])
                .chain(&self.dist_ns)
                .chain(&self.pred_ns)
                .chain(&self.pred_row_ns)
                .chain(&self.pred_bits_ns)
                .chain(&self.gallop_gap_ns)
                .chain(&self.heap_bound_extra)
                .chain(self.heap_bound_extra_kl.iter().flatten())
                .chain(&self.check_base_ns)
                .chain(&self.check_word_ns)
                .chain(&self.check_prefetch_ns)
                .chain(&self.check_row_ns)
                .chain(&[
                    self.query_ns,
                    self.enum_word_ns,
                    self.enum_bit_ns,
                    self.enum_nz_ns,
                ])
                .all(|x| x.is_finite() && *x >= 0.0),
            "unit costs must be finite and non-negative"
        );
        Ok(())
    }

    /// Fills the planner caches ([`Prepared`]): the plan costs and choices are those computed
    /// without them, at a fraction of the work.
    pub fn prepare(&mut self) {
        let ln = |v: &[f64]| v.iter().map(|x| x.ln()).collect();
        let gallop_min = if self.gallop_gap.is_empty() {
            self.gallop_ns
        } else {
            self.gallop_gap_ns
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min)
        };
        let curves = [
            &self.dist_ns,
            &self.pred_ns,
            &self.pred_row_ns,
            &self.pred_bits_ns,
            &self.check_base_ns,
            &self.check_word_ns,
            &self.check_prefetch_ns,
            &self.check_row_ns,
            &self.gallop_gap_ns,
        ];
        let scalars = [
            self.word_ns,
            self.scatter_ns,
            self.merge_ns,
            self.enum_ns,
            self.enum_word_ns,
            self.enum_nz_ns,
            self.enum_bit_ns,
            self.gallop_ns,
        ];
        let bounds = curves
            .iter()
            .flat_map(|v| v.iter())
            .chain(scalars.iter())
            .all(|&x| x >= 0.0 && x.is_finite());
        self.prepared = Prepared {
            ln_density: ln(&self.density),
            ln_gallop_gap: ln(&self.gallop_gap),
            gallop_min,
            bounds,
        };
    }

    /// Position of density `rho` on the grid (clamped at the ends).
    fn pos(&self, rho: f64) -> Pos {
        let d = &self.density;
        if rho <= d[0] {
            return Pos::Lo;
        }
        let last = d.len() - 1;
        if rho >= d[last] {
            return Pos::Hi(last);
        }
        let i = d.partition_point(|&x| x <= rho);
        let (x0, x1) = ln_pair(&self.prepared.ln_density, d, i);
        Pos::At(i, (rho.ln() - x0) / (x1 - x0))
    }

    /// The density curve `v` at position `p`.
    #[inline]
    fn at(v: &[f64], p: Pos) -> f64 {
        match p {
            Pos::Lo => v[0],
            Pos::Hi(last) => v[last],
            Pos::At(i, t) => v[i - 1] + t * (v[i] - v[i - 1]),
        }
    }

    /// Log-linear interpolation of a density curve (clamped at the ends).
    fn curve(&self, v: &[f64], rho: f64) -> f64 {
        Self::at(v, self.pos(rho))
    }

    /// Cost of one distance computation over id-ordered candidates of density `rho`.
    pub fn dist(&self, rho: f64) -> f64 {
        self.curve(&self.dist_ns, rho)
    }

    /// Cost of one exact-path predicate check at density `rho` (random points; batch B6).
    pub fn pred(&self, rho: f64) -> f64 {
        self.curve(&self.pred_ns, rho)
    }

    /// Cost of checking one member of `union(c*)` at density `rho`, for a query whose
    /// residual clauses read rows (`rows`) or are bitset-only; `pred` without the curves.
    pub fn pred_member(&self, rows: bool, rho: f64) -> f64 {
        self.pred_member_at(rows, self.pos(rho))
    }

    fn pred_member_at(&self, rows: bool, p: Pos) -> f64 {
        if self.pred_row_ns.is_empty() {
            return Self::at(&self.pred_ns, p);
        }
        Self::at(
            if rows {
                &self.pred_row_ns
            } else {
                &self.pred_bits_ns
            },
            p,
        )
    }

    /// Whether the shape-resolved check curves are calibrated.
    pub fn has_shape(&self) -> bool {
        !self.check_base_ns.is_empty()
    }

    /// Whether the A6 enumeration costs (per empty word, per set bit) are calibrated: the
    /// bitmap's enumeration is then priced by its set bits, and `rowand` is a candidate plan.
    pub fn has_enum_bits(&self) -> bool {
        self.enum_bit_ns > 0.0
    }

    /// Cost of enumerating a bitmap of `words` words holding `bits` set bits at random
    /// positions (`words` 0: words that each hold some of the bits, at most one per bit;
    /// older calibrations: `enum_ns` per word, whatever the bits).
    pub fn enumerate(&self, words: f64, bits: f64) -> f64 {
        if !self.has_enum_bits() {
            return words * self.enum_ns;
        }
        let nonempty = if words > 0.0 {
            words * -(-bits / words).exp_m1()
        } else {
            bits
        };
        words * self.enum_word_ns + nonempty * self.enum_nz_ns + bits * self.enum_bit_ns
    }

    /// Cost of checking one member of `union(c*)` with the expected work `w` at density
    /// `rho`: the shape-resolved curves, else `pred_member` of `w`'s class.
    pub fn member(&self, w: &MemberWork, rho: f64) -> f64 {
        self.member_at(w, self.pos(rho))
    }

    fn member_at(&self, w: &MemberWork, p: Pos) -> f64 {
        if !self.has_shape() {
            return self.pred_member_at(w.class_rows, p);
        }
        let prefetch = if self.check_prefetch_ns.is_empty() {
            0.0
        } else {
            w.prefetch * Self::at(&self.check_prefetch_ns, p)
        };
        Self::at(&self.check_base_ns, p)
            + w.words * Self::at(&self.check_word_ns, p)
            + prefetch
            + w.rows * Self::at(&self.check_row_ns, p)
    }

    /// Cost of one sampled point of a sketch probe (every residual clause evaluated, work
    /// `w`) at density `rho`: the shape-resolved curves, else `probe_ns`.
    pub fn probe_point(&self, w: &MemberWork, rho: f64) -> f64 {
        if self.has_shape() {
            self.member(w, rho)
        } else {
            self.probe_ns
        }
    }

    /// Cost of one galloping step at the largest calibrated gap (miss-bound steps).
    pub fn gallop_cold(&self) -> f64 {
        self.gallop_gap_ns.last().copied().unwrap_or(self.gallop_ns)
    }

    /// Cost of one modelled galloping step at gap `gap` (posting entries per galloped
    /// candidate): log-linear in the gap, clamped at the calibrated ends.
    pub fn gallop(&self, gap: f64) -> f64 {
        let g = &self.gallop_gap;
        if g.is_empty() {
            return self.gallop_ns;
        }
        let v = &self.gallop_gap_ns;
        if gap <= g[0] {
            return v[0];
        }
        let last = g.len() - 1;
        if gap >= g[last] {
            return v[last];
        }
        let i = g.partition_point(|&x| x <= gap);
        let (x0, x1) = ln_pair(&self.prepared.ln_gallop_gap, g, i);
        let t = (gap.ln() - x0) / (x1 - x0);
        v[i - 1] + t * (v[i] - v[i - 1])
    }

    /// Model cost of a sketch probe that merges `merged` sketch entries and evaluates
    /// `evaluated` sampled points (every residual clause: work `w` each) at density `rho`.
    pub fn probe(&self, merged: f64, evaluated: f64, w: &MemberWork, rho: f64) -> f64 {
        merged * self.sketch_ns + evaluated * self.probe_point(w, rho)
    }

    /// Distance computations of a graph search at list size `l`.
    pub fn graph_dists(&self, l: f64) -> f64 {
        if self.graph_l.is_empty() {
            return self.graph_alpha * l + self.graph_beta;
        }
        loglog(&self.graph_l, &self.graph_l_dists, l)
    }

    /// A graph search with list size `l` that cannot be expected to find a match: even at the
    /// upper confidence bound of the match count, the nodes it evaluates (`D(l)`, at least
    /// `l`) meet fewer than one match when the matches are spread independently of the
    /// distance to the query. Needs the probe's estimates (posting bounds: never).
    pub fn graph_hopeless(&self, l: f64, est: &Estimates) -> bool {
        est.sampled > 0 && est.n > 0.0 && self.graph_dists(l).max(l) * est.m_hi / est.n < 1.0
    }

    /// Cost of one distance computation of a graph search at list size `l`: `graph_dist_ns`,
    /// or more where the cost by list size (log-linear in `L`, clamped at the ends) is higher.
    pub fn graph_dist_at(&self, l: f64) -> f64 {
        let (ls, v) = (&self.graph_dist_ns_l, &self.graph_dist_ns_curve);
        if ls.is_empty() {
            return self.graph_dist_ns;
        }
        let last = ls.len() - 1;
        let at = if l <= ls[0] {
            v[0]
        } else if l >= ls[last] {
            v[last]
        } else {
            let i = ls.partition_point(|&x| x <= l);
            let t = (l.ln() - ls[i - 1].ln()) / (ls[i].ln() - ls[i - 1].ln());
            v[i - 1] + t * (v[i] - v[i - 1])
        };
        self.graph_dist_ns.max(at)
    }

    /// Nominal cost of an eager graph search at list size `l` (every distance with a
    /// predicate).
    pub fn graph(&self, l: f64) -> f64 {
        self.graph_dists(l) * (self.graph_dist_ns + self.graph_pred_ns)
    }

    /// `extra(K, L)` of `heap_bound`: per calibrated heap size, log-log in the list size
    /// (with `extra(K, L)` table) or the size's single value; then power-law interpolation
    /// between the heap sizes, extended with the nearest segment's slope (a single size
    /// scales with `K`).
    fn heap_bound_extra(&self, k: f64, l: f64) -> f64 {
        let ks = &self.heap_bound_k;
        let at = |i: usize| -> f64 {
            if self.heap_bound_l.is_empty() {
                self.heap_bound_extra[i]
            } else {
                // A floor of one evaluation keeps the logarithms finite.
                let row: Vec<f64> = self.heap_bound_extra_kl[i]
                    .iter()
                    .map(|&x| x.max(1.0))
                    .collect();
                loglog(&self.heap_bound_l, &row, l)
            }
        };
        if let Some(i) = ks.iter().position(|&x| x == k) {
            return at(i);
        }
        if ks.len() == 1 {
            return at(0) * k / ks[0];
        }
        let i = ks.partition_point(|&x| x <= k).clamp(1, ks.len() - 1);
        let (k0, k1) = (ks[i - 1], ks[i]);
        let (x0, x1) = (at(i - 1).max(1e-9), at(i).max(1e-9));
        let slope = (x1.ln() - x0.ln()) / (k1.ln() - k0.ln());
        (x0.ln() + slope * (k.max(1.0).ln() - k0.ln())).exp()
    }

    /// Predicate evaluations of a graph search at list size `l` with heap size `k`, match
    /// rate `rate` and laziness `laziness` (see the module documentation).
    pub fn graph_preds(&self, l: f64, k: usize, rate: f64, laziness: Laziness) -> f64 {
        let d = self.graph_dists(l);
        match laziness {
            Laziness::HeapBound if !self.heap_bound_k.is_empty() => {
                if rate > 0.0 {
                    let k = k.max(1) as f64;
                    d.min(k / rate.min(1.0) + self.heap_bound_extra(k, l))
                } else {
                    d
                }
            }
            Laziness::DistanceFirst if self.distance_first_frac > 0.0 => {
                d * self.distance_first_frac.min(1.0)
            }
            _ => d,
        }
    }

    /// Nominal cost of a graph search at list size `l` (heap size `k`, match rate `rate`,
    /// laziness `laziness`): the distances and the predicate evaluations they lead to.
    pub fn graph_cost(&self, l: f64, k: usize, rate: f64, laziness: Laziness) -> f64 {
        self.graph_dists(l) * self.graph_dist_at(l)
            + self.graph_preds(l, k, rate, laziness) * self.graph_pred_ns
    }

    /// The model cost of graph work already done.
    #[inline]
    pub fn charged(&self, n_dist: u64, n_pred: u64) -> f64 {
        n_dist as f64 * self.graph_dist_ns + n_pred as f64 * self.graph_pred_ns
    }

    /// Nominal cost of a post-filter search at list size `l` (heap size `k`, match rate
    /// `rate`): the distances of an unfiltered search, and the checks of the evaluated nodes
    /// in distance order until `k` matches are found (about `k / rate`, at most all of them).
    pub fn postfilter_cost(&self, l: f64, k: usize, rate: f64) -> f64 {
        let d = self.graph_dists(l);
        let checks = if rate > 0.0 {
            d.min(k.max(1) as f64 / rate.min(1.0))
        } else {
            d
        };
        d * self.graph_dist_at(l) + checks * self.graph_pred_ns
    }

    /// The calibrated unfiltered list size for mean recall `r` at `kprime` neighbours:
    /// log-linear interpolation in the recall and log-log in the neighbours, clamped to the
    /// calibrated recalls; beyond the largest calibrated neighbour count it grows in proportion
    /// to it. `None` without a curve.
    pub fn effort_list(&self, r: f64, kprime: f64) -> Option<f64> {
        let (rs, ks) = (&self.effort_r, &self.effort_k);
        if rs.is_empty() || ks.is_empty() {
            return None;
        }
        // Bracket and weight of `x` in the increasing grid `g` (clamped).
        let at = |g: &[f64], x: f64| -> (usize, usize, f64) {
            if x <= g[0] {
                return (0, 0, 0.0);
            }
            let last = g.len() - 1;
            if x >= g[last] {
                return (last, last, 0.0);
            }
            let i = g.partition_point(|&v| v <= x);
            (i - 1, i, (x - g[i - 1]) / (g[i] - g[i - 1]))
        };
        let kp = kprime.max(1.0);
        let lk: Vec<f64> = ks.iter().map(|k| k.ln()).collect();
        let (j0, j1, u) = at(&lk, kp.ln());
        let (i0, i1, t) = at(rs, r);
        let row = |i: usize| {
            let row = &self.effort_l[i];
            (1.0 - u) * row[j0].ln() + u * row[j1].ln()
        };
        let l = ((1.0 - t) * row(i0) + t * row(i1)).exp();
        let top = ks[ks.len() - 1];
        Some(if kp > top { l * kp / top } else { l })
    }

    /// The smallest list size whose graph search computes `nodes` distances (the inverse of
    /// [`Self::graph_dists`]; 0 when the calibration has no growth with `L`).
    pub fn graph_l_for(&self, nodes: f64) -> f64 {
        let ds = &self.graph_l_dists;
        if ds.len() > 1 && ds.windows(2).all(|w| w[0] < w[1]) {
            // The measured curve is piecewise linear in log-log: its inverse swaps the axes.
            return loglog(ds, &self.graph_l, nodes.max(1.0));
        }
        if self.graph_alpha <= 0.0 {
            return 0.0;
        }
        ((nodes - self.graph_beta) / self.graph_alpha).max(0.0)
    }
}

/// Rank of the `k`-th match among points in distance order when each point matches
/// independently with probability `rate`: the negative binomial mean `k / rate` plus `z`
/// standard deviations (`√(k (1 − rate)) / rate`); infinite at rate 0.
pub fn kth_match_rank(k: usize, rate: f64, z: f64) -> f64 {
    if rate.is_nan() || rate <= 0.0 {
        return f64::INFINITY;
    }
    let (k, r) = (k.max(1) as f64, rate.min(1.0));
    k / r + z * (k * (1.0 - r)).sqrt() / r
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
    /// Dense match bitmap (static bitsets for frequent labels), built and enumerated block by
    /// block, then distances.
    Bitmap = 4,
    /// The words of `c*`'s union ANDed with the residual clauses' rows word by word (bitset
    /// words; galloping cursors in the postings of labels without a bitset), then distances.
    RowAnd = 5,
    /// The candidate lists of a clause pair from the combination postings (`pairs.flab`):
    /// merged, checked on the other clauses, then distances.
    Pair = 6,
}

impl ExactPlan {
    pub const ALL: [Self; 6] = [
        Self::Scan,
        Self::DistanceFirst,
        Self::Gallop,
        Self::Bitmap,
        Self::RowAnd,
        Self::Pair,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Scan => "scan",
            Self::DistanceFirst => "df",
            Self::Gallop => "gallop",
            Self::Bitmap => "bitmap",
            Self::RowAnd => "rowand",
            Self::Pair => "pair",
        }
    }

    pub fn parse(s: &str) -> anyhow::Result<Self> {
        Ok(match s.trim() {
            "scan" => Self::Scan,
            "df" | "distance_first" => Self::DistanceFirst,
            "gallop" => Self::Gallop,
            "bitmap" => Self::Bitmap,
            "rowand" => Self::RowAnd,
            "pair" => Self::Pair,
            _ => anyhow::bail!("unknown exact plan {s:?} (scan, df, gallop, bitmap, rowand, pair)"),
        })
    }
}

/// A set of exact plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlanSet(u8);

impl PlanSet {
    pub const ALL: Self = Self(0b11_1111);

    pub fn contains(self, p: ExactPlan) -> bool {
        self.0 & (1 << (p as u8 - 1)) != 0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Parses `scan+df+gallop+bitmap+rowand+pair` (any non-empty subset).
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

/// Expected work of checking one member of `union(c*)` on the residual clauses (positions
/// `1..` of the Boolean order, as [`super::eval`]'s static evaluator does): static-bitset
/// words probed, rows prefetched (the exact path prefetches every member's row when a
/// residual clause has labels without a bitset) and rows read. `class_rows`: some residual
/// clause reads rows (the class of the older per-class curves).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MemberWork {
    pub words: f64,
    pub prefetch: f64,
    pub rows: f64,
    pub class_rows: bool,
}

impl MemberWork {
    /// From the clause shapes (`split`: the query was split against the static bitsets;
    /// otherwise every label is checked in the row) and the survival estimates
    /// (`survival[j − 1]`: the fraction of `union(c*)` satisfying clauses `1..=j`). A clause
    /// probes its bitset words in decreasing label frequency until one is set: a member that
    /// fails it probes every word, one that passes about one; a failed clause without row
    /// labels ends the check; the row is read once if the check gets past those and some
    /// clause with row labels is left unsatisfied by its bitset labels (a member passing a
    /// clause that has some mostly passes by one of them). `full`: no early end (the probe
    /// evaluates every clause); `upper`: the worst case (every word probed, the row read).
    /// `at_reach`: the row is read whenever the check reaches a clause with row labels (the
    /// A1b model, an ablation).
    pub fn of(
        q: &Compiled,
        split: bool,
        survival: &[f64],
        full: bool,
        upper: bool,
        at_reach: bool,
    ) -> Self {
        let c = q.n_clauses();
        let class_rows = q.residual_reads_rows(split);
        if c <= 1 || q.is_empty() {
            return Self {
                class_rows,
                ..Self::default()
            };
        }
        if !split {
            return Self {
                words: 0.0,
                prefetch: 1.0,
                rows: 1.0,
                class_rows,
            };
        }
        let (mut words, mut reach, mut prev) = (0.0, 1.0, 1.0);
        // Row-label clauses seen, and the chance that their bitset labels satisfy them all.
        let (mut row, mut all_static) = (false, 1.0);
        for j in 1..c {
            let s = q.static_slots(j).len() as f64;
            let cur = survival.get(j - 1).copied().unwrap_or(prev).clamp(0.0, 1.0);
            let p = if upper {
                1.0
            } else if prev > 0.0 {
                (cur / prev).min(1.0)
            } else {
                0.0
            };
            prev = cur;
            words += reach
                * if upper {
                    s
                } else {
                    s * (1.0 - p) + s.min(1.0) * p
                };
            if !q.row_labels(j).is_empty() {
                row = true;
                // A member passing a clause with bitset labels mostly passes by one of them (the
                // frequent labels); a clause of row labels only always needs the row.
                if !upper && !at_reach && s > 0.0 {
                    all_static *= p;
                } else {
                    all_static = 0.0;
                }
            } else if !full && !upper {
                reach *= p;
            }
        }
        Self {
            words,
            prefetch: if row { 1.0 } else { 0.0 },
            rows: if row { reach * (1.0 - all_static) } else { 0.0 },
            class_rows,
        }
    }
}

impl MemberWork {
    /// Worst-case work of checking the clauses in `mask` (bit `j`: clause `j`): every bitset
    /// word of those clauses probed, and the row read once if one of them has row labels. The
    /// `pair` plan's checks: its candidates come from joint or single postings, without the
    /// survival estimates of `union(c*)`, and its mask may include `c*`.
    pub fn of_mask(q: &Compiled, split: bool, mask: u64) -> Self {
        if mask == 0 || q.is_empty() {
            return Self::default();
        }
        if !split {
            return Self {
                words: 0.0,
                prefetch: 1.0,
                rows: 1.0,
                class_rows: true,
            };
        }
        let (mut words, mut row) = (0.0, false);
        for j in (0..q.n_clauses()).filter(|&j| mask >> j & 1 == 1) {
            words += q.static_slots(j).len() as f64;
            row |= !q.row_labels(j).is_empty();
        }
        let r = if row { 1.0 } else { 0.0 };
        Self {
            words,
            prefetch: r,
            rows: r,
            class_rows: row,
        }
    }
}

/// Inputs of the exact plan costs of one query.
#[derive(Debug, Clone, Copy)]
pub struct PlanInputs<'a> {
    pub q: &'a Compiled,
    pub labels: &'a LabelStore,
    pub bitsets: Option<&'a LabelBitsets>,
    pub k: usize,
    /// `df` counts its evaluations at the lower confidence bound of the match count
    /// ([`Kernel::df_lower`](super::config::Kernel::df_lower)).
    pub df_lower: bool,
    /// The query's clause pair with the fewest candidates in the combination postings (the
    /// `pair` plan; `None` without them).
    pub pair: Option<&'a Cover>,
}

/// Per-estimate terms shared by the plan costs of a query (computed once per
/// [`PlanInputs::best`]).
struct Ctx {
    n: f64,
    a: f64,
    m: f64,
    /// The match count of `df`'s evaluations: `m`, or its lower confidence bound (`df_lower`,
    /// estimates from a sample).
    m_df: f64,
    k: f64,
    checks: bool,
    dist_m: f64,
    work: MemberWork,
    merge: f64,
    /// `a / n` and its position on the density grid.
    ra: f64,
    pa: Pos,
}

/// Relative margin of the plan skips: a plan is skipped when its lower bound exceeds the best
/// cost by more than the rounding of either evaluation.
const SKIP_MARGIN: f64 = 1e-9;

/// Order in which [`PlanInputs::best`] evaluates the plans: the cheap evaluations first, so
/// that the per-clause sums of `rowand` and `gallop` stop early against their cost.
const EVAL_ORDER: [ExactPlan; 6] = [
    ExactPlan::Pair,
    ExactPlan::Scan,
    ExactPlan::DistanceFirst,
    ExactPlan::Bitmap,
    ExactPlan::RowAnd,
    ExactPlan::Gallop,
];

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
        self.cost_in(
            c,
            plan,
            e,
            seeded,
            upper,
            &self.ctx(c, e, upper),
            f64::INFINITY,
        )
    }

    fn ctx(&self, c: &UnitCosts, e: &Estimates, upper: bool) -> Ctx {
        let n = e.n.max(1.0);
        let (a, m) = (e.a, e.m.min(e.a));
        let ra = a / n;
        let m_df = if self.df_lower && !e.exact {
            e.m_lo.clamp(0.0, m)
        } else {
            m
        };
        Ctx {
            n,
            a,
            m,
            m_df,
            k: self.k as f64,
            checks: self.q.n_clauses() > 1,
            dist_m: m * c.dist(m / n),
            work: MemberWork::of(
                self.q,
                self.bitsets.is_some(),
                &e.survival,
                false,
                upper,
                c.reads_at_reach,
            ),
            merge: self.merge(c),
            ra,
            pa: c.pos(ra),
        }
    }

    /// [`Self::cost`] with the per-estimate terms `x`. `bound`: the sums of `gallop` and
    /// `rowand` (non-negative terms, so their partial sums only grow) stop at the first clause
    /// where the cost so far exceeds it, returning that partial cost (only compared with
    /// `bound` by the caller; `f64::INFINITY`: the full cost).
    #[expect(clippy::too_many_arguments, reason = "the plan cost's inputs")]
    fn cost_in(
        &self,
        c: &UnitCosts,
        plan: ExactPlan,
        e: &Estimates,
        seeded: bool,
        upper: bool,
        x: &Ctx,
        bound: f64,
    ) -> f64 {
        let (n, a, m, k, dist_m) = (x.n, x.a, x.m, x.k, x.dist_m);
        match plan {
            ExactPlan::Scan => {
                let pred = if x.checks {
                    a * c.member_at(&x.work, x.pa)
                } else {
                    0.0
                };
                x.merge + pred + dist_m
            }
            ExactPlan::DistanceFirst => {
                // The answer bound saves checks only once `K` matches are found.
                let m = x.m_df;
                let evals = if !x.checks {
                    0.0
                } else if upper || m <= k {
                    a
                } else if seeded {
                    (2.0 * k * a / m).min(a)
                } else {
                    (k * a / m * (1.0 + (m / k).ln())).min(a)
                };
                let pe = if evals.to_bits() == a.to_bits() {
                    x.pa
                } else {
                    c.pos(evals / n)
                };
                x.merge + a * UnitCosts::at(&c.dist_ns, x.pa) + evals * c.member_at(&x.work, pe)
            }
            ExactPlan::Gallop => {
                let mut cost = x.merge + a * c.word_ns;
                let mut s = a;
                for j in 1..self.q.n_clauses() {
                    for &l in self.q.clause(j) {
                        let p = self.labels.posting_len(l) as f64;
                        if s >= 1.0 {
                            cost += c.gallop(p / s) * s * (1.0 + (1.0 + p / s).log2());
                        }
                    }
                    if cost + dist_m > bound {
                        return cost + dist_m;
                    }
                    s = a * e.survival.get(j - 1).copied().unwrap_or(1.0);
                }
                cost + dist_m
            }
            ExactPlan::Bitmap => {
                let words = (n / 64.0).ceil();
                let mut cost = c.enumerate(words, m) + words * c.word_ns;
                for j in 0..self.q.n_clauses() {
                    // A6 calibrations: a later clause of one bitset label is ANDed in directly
                    // (one pass, as built).
                    let clause = self.q.clause(j);
                    if j > 0
                        && c.has_enum_bits()
                        && clause.len() == 1
                        && self.bitsets.and_then(|b| b.slot(clause[0])).is_some()
                    {
                        cost += words * c.word_ns;
                        continue;
                    }
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
            ExactPlan::RowAnd => {
                if !c.has_enum_bits() {
                    return f64::INFINITY;
                }
                let tail = c.enumerate(0.0, m);
                self.row_and(c, e, upper, x, |cost| cost + tail + dist_m > bound) + tail + dist_m
            }
            ExactPlan::Pair => self.pair_cost(c, x),
        }
    }

    /// Cost of the `pair` plan (see the module documentation); infinite without a cover.
    fn pair_cost(&self, c: &UnitCosts, x: &Ctx) -> f64 {
        let Some(cv) = self.pair.filter(|cv| cv.valid) else {
            return f64::INFINITY;
        };
        let (lists, e) = (cv.lists.len() as f64, cv.entries as f64);
        let mut cost = lists * c.gallop_cold();
        if lists > 1.0 {
            cost += e * c.merge_ns * lists.log2();
        }
        // The check mask of `exact::planned`: every clause but the covered pair's (exact
        // cover), else every clause.
        let full = self.q.full_mask();
        let mask = if cv.exact {
            full & !(1u64 << cv.i) & !(1u64 << cv.j)
        } else {
            full
        };
        if mask != 0 {
            let w = MemberWork::of_mask(self.q, self.bitsets.is_some(), mask);
            cost += e * c.member_at(&w, c.pos(e / x.n));
        }
        let m = if cv.exact && self.q.n_clauses() <= 2 {
            e
        } else {
            x.m.min(e)
        };
        cost + m * c.dist(m / x.n)
    }

    /// A lower bound of the cost of `plan` that is cheaper to evaluate (`None`: none), valid
    /// with non-negative unit costs: distance-first without its predicate evaluations;
    /// galloping at its cheapest step, with `1 + log2(1 + g) ≥ 1 + min(g, 1)` for the search
    /// depth (no logarithms).
    fn lower(&self, c: &UnitCosts, plan: ExactPlan, e: &Estimates, x: &Ctx) -> Option<f64> {
        match plan {
            ExactPlan::DistanceFirst => Some(x.merge + x.a * UnitCosts::at(&c.dist_ns, x.pa)),
            ExactPlan::Gallop => {
                let g = c.prepared.gallop_min;
                let mut cost = x.merge + x.a * c.word_ns;
                let mut s = x.a;
                for j in 1..self.q.n_clauses() {
                    if s >= 1.0 {
                        for &l in self.q.clause(j) {
                            let p = self.labels.posting_len(l) as f64;
                            cost += g * (s + p.min(s));
                        }
                    }
                    s = x.a * e.survival.get(j - 1).copied().unwrap_or(1.0);
                }
                Some(cost + x.dist_m)
            }
            _ => None,
        }
    }

    /// Cost of `rowand` before its enumeration and distances (see the module documentation).
    /// `exceeds`: stop after the first clause whose cost so far it accepts (that partial cost
    /// is returned).
    fn row_and(
        &self,
        c: &UnitCosts,
        e: &Estimates,
        upper: bool,
        x: &Ctx,
        exceeds: impl Fn(f64) -> bool,
    ) -> f64 {
        let q = self.q;
        let n = e.n.max(1.0);
        let words = (n / 64.0).ceil();
        let nc = q.n_clauses();
        if nc == 0 {
            return words * c.word_ns;
        }
        let split = self.bitsets.is_some();
        let hot = |j: usize| if split { q.static_slots(j).len() } else { 0 };
        let rare = |j: usize| if split { q.row_labels(j) } else { q.clause(j) };
        let posts = |j: usize| -> f64 {
            rare(j)
                .iter()
                .map(|&l| self.labels.posting_len(l) as f64)
                .sum()
        };
        let u0 = q.upper_bound(0) as f64;
        let a = if upper { u0 } else { e.a.min(u0) }.max(0.0);
        // Words of c*: its rare postings read in place (one list) or scattered, and its
        // bitset rows streamed (every word).
        let (h0, p0) = (hot(0) as f64, posts(0));
        let mut cost = if h0 > 0.0 {
            let r0 = if p0 > 0.0 { 1.0 } else { 0.0 };
            words * ((h0 + r0) * c.word_ns + c.enumerate(1.0, 0.0)) + p0 * c.scatter_ns
        } else {
            let lists = rare(0)
                .iter()
                .filter(|&&l| self.labels.posting_len(l) > 0)
                .count();
            let summary = if lists > 1 {
                (words / 64.0).ceil() * c.word_ns
            } else {
                0.0
            };
            p0 * c.scatter_ns + summary
        };
        // Distinct words of `a` members spread over the data points.
        let w = (words * -(-a / words).exp_m1()).min(a);
        if w < 1e-9 {
            return cost;
        }
        let per_word = a / w;
        let rho = (a / n).clamp(0.0, 1.0);
        let pos = if rho.to_bits() == x.ra.to_bits() {
            x.pa
        } else {
            c.pos(rho)
        };
        let (base, probe) = if c.has_shape() {
            (
                UnitCosts::at(&c.check_base_ns, pos),
                UnitCosts::at(&c.check_word_ns, pos),
            )
        } else {
            (c.pred_member_at(false, pos), 0.0)
        };
        // `reach`: the fraction of the visited words that reach clause `j` (one pass over the
        // surviving words per residual clause).
        let (mut reach, mut prev) = (1.0, 1.0);
        for j in 1..nc {
            let cur = if upper {
                1.0
            } else {
                let s = e
                    .survival
                    .get(j - 1)
                    .copied()
                    .unwrap_or(1.0)
                    .clamp(0.0, 1.0);
                if per_word > 1.0 + 1e-9 {
                    1.0 - (1.0 - s).powf(per_word)
                } else {
                    s
                }
            };
            let p = if prev > 0.0 {
                (cur / prev).min(1.0)
            } else {
                0.0
            };
            let s = hot(j) as f64;
            let reads = if upper {
                s
            } else {
                s * (1.0 - p) + s.min(1.0) * p
            };
            let at = w * reach;
            cost += at * (base + reads * probe);
            // The cursors of the labels without a bitset move at most once per entry or word
            // reached; priced as `r` labels of the clause's mean length (an upper bound of the
            // per-label sum: the steps are concave in the length; one evaluation per clause),
            // plus each cursor's first seek: a gallop over its list at the step cost of the
            // largest calibrated gap (cold lists).
            let (r, pj) = (rare(j).len() as f64, posts(j));
            let events = at.min(pj);
            if r > 0.0 && events >= 1.0 {
                let gap = pj / (r * events);
                cost += r
                    * (c.gallop(gap) * events * (1.0 + (1.0 + gap).log2())
                        + c.gallop_cold() * (1.0 + pj / r).log2());
            }
            reach *= if upper { 1.0 } else { p };
            prev = cur;
            if exceeds(cost) {
                return cost;
            }
        }
        cost
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
        let x = self.ctx(c, e, upper);
        let mut best = (ExactPlan::Scan, f64::INFINITY);
        // The plans with a lower bound come last (the bound then meets the cheapest of the
        // others); ties go to the earlier plan of `ExactPlan::ALL`, as in that order.
        for p in EVAL_ORDER {
            if !plans.contains(p) {
                continue;
            }
            // A plan whose lower bound already loses is skipped (same choice and cost).
            if c.prepared.bounds
                && best.1.is_finite()
                && self
                    .lower(c, p, e, &x)
                    .is_some_and(|lb| lb > best.1 * (1.0 + SKIP_MARGIN))
            {
                continue;
            }
            let bound = if c.prepared.bounds {
                best.1
            } else {
                f64::INFINITY
            };
            let v = self.cost_in(c, p, e, seeded, upper, &x, bound);
            if v < best.1 || (v == best.1 && (p as u8) < (best.0 as u8)) {
                best = (p, v);
            }
        }
        best
    }
}

/// Match rate from which a predicate is broad: upstream's list multiplier is 1 (at least every
/// second point matches, so the unfiltered neighbourhood holds the matches) and the post-filter
/// route's `by_rate` starts are the global ones.
pub const BROAD_RATE: f64 = 0.5;

/// Upstream DiskANN's adaptive-L multiplier at match rate `rate`: 1 (≥ 50 %), 2 (10–50 %),
/// `2^{−log10 rate}` below 10 %, clamped to `[1, scale]` (`scale` at rate 0).
pub fn l_multiplier(rate: f64, scale: f64) -> f64 {
    if rate.is_nan() || rate <= 0.0 {
        return scale.max(1.0);
    }
    let m = if rate >= BROAD_RATE {
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
    fn calibrations_without_the_new_terms_keep_the_b6_model() {
        let c = UnitCosts::default();
        assert_eq!(c.pred_member(true, 0.01), c.pred(0.01));
        assert_eq!(c.pred_member(false, 0.5), c.pred(0.5));
        assert_eq!(c.gallop(1000.0), c.gallop_ns);
        for lz in [
            Laziness::HeapBound,
            Laziness::DistanceFirst,
            Laziness::Eager,
        ] {
            assert_eq!(c.graph_preds(100.0, 10, 0.5, lz), c.graph_dists(100.0));
            assert!((c.graph_cost(100.0, 10, 0.5, lz) - c.graph(100.0)).abs() < 1e-9);
        }
        // An older calibration file (no new members) reads with empty terms.
        let mut v = serde_json::to_value(&c).unwrap();
        for key in [
            "pred_row_ns",
            "pred_bits_ns",
            "gallop_gap",
            "gallop_gap_ns",
            "heap_bound_k",
            "heap_bound_extra",
            "distance_first_frac",
        ] {
            v.as_object_mut().unwrap().remove(key);
        }
        let old: UnitCosts = serde_json::from_value(v).unwrap();
        old.validate().unwrap();
        assert_eq!(old, c);
    }

    #[test]
    fn member_gallop_and_lazy_graph_terms() {
        let mut c = UnitCosts {
            pred_row_ns: vec![100.0, 90.0, 80.0, 70.0, 60.0, 50.0],
            pred_bits_ns: vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
            gallop_gap: vec![1.0, 100.0],
            gallop_gap_ns: vec![2.0, 20.0],
            heap_bound_k: vec![10.0, 100.0],
            heap_bound_extra: vec![100.0, 400.0],
            distance_first_frac: 0.3,
            ..UnitCosts::default()
        };
        c.validate().unwrap();
        assert_eq!(c.pred_member(true, 1e-4), 100.0);
        assert_eq!(c.pred_member(false, 1e-4), 10.0);
        assert!((c.pred_member(true, 0.03) - c.curve(&c.pred_row_ns, 0.03)).abs() < 1e-12);
        assert!(
            (c.gallop(10.0) - 11.0).abs() < 1e-9,
            "log-linear in the gap"
        );
        assert_eq!(c.gallop(0.5), 2.0);
        assert_eq!(c.gallop(1e6), 20.0);
        // D = 10·100 + 200 = 1,200 distances at L = 100.
        let d = c.graph_dists(100.0);
        assert_eq!(
            c.graph_preds(100.0, 10, 0.5, Laziness::HeapBound),
            20.0 + 100.0
        );
        assert_eq!(
            c.graph_preds(100.0, 100, 1.0, Laziness::HeapBound),
            100.0 + 400.0
        );
        let mid = c.graph_preds(1000.0, 1000, 1.0, Laziness::HeapBound);
        assert!(
            (mid - (1000.0 + 1600.0)).abs() < 1e-6,
            "power law in K: {mid}"
        );
        assert_eq!(
            c.graph_preds(100.0, 10, 0.001, Laziness::HeapBound),
            d,
            "capped by D"
        );
        assert_eq!(c.graph_preds(100.0, 10, 0.0, Laziness::HeapBound), d);
        assert!((c.graph_preds(100.0, 10, 0.5, Laziness::DistanceFirst) - 0.3 * d).abs() < 1e-9);
        assert_eq!(c.graph_preds(100.0, 10, 0.5, Laziness::Eager), d);
        let lazy = c.graph_cost(100.0, 10, 0.5, Laziness::HeapBound);
        assert!((lazy - (d * c.graph_dist_ns + 120.0 * c.graph_pred_ns)).abs() < 1e-6);
        assert!(lazy < c.graph(100.0));
        // Shapes are checked.
        c.pred_bits_ns.pop();
        assert!(c.validate().is_err());
        c.pred_bits_ns.push(5.0);
        c.gallop_gap = vec![100.0, 1.0];
        assert!(c.validate().is_err());
        c.gallop_gap = vec![1.0, 100.0];
        c.heap_bound_extra.pop();
        assert!(c.validate().is_err());
    }

    #[test]
    fn measured_graph_curves_interpolate_log_log() {
        let line = UnitCosts {
            heap_bound_k: vec![10.0, 100.0],
            heap_bound_extra: vec![100.0, 400.0],
            ..UnitCosts::default()
        };
        let mut c = UnitCosts {
            graph_l: vec![10.0, 100.0, 1000.0],
            graph_l_dists: vec![100.0, 500.0, 2000.0],
            heap_bound_l: vec![10.0, 100.0, 1000.0],
            heap_bound_extra_kl: vec![vec![100.0, 200.0, 400.0], vec![300.0, 600.0, 1200.0]],
            ..line.clone()
        };
        c.validate().unwrap();
        // Without the curves: the line and one extra(K) per heap size.
        assert_eq!(line.graph_dists(100.0), 10.0 * 100.0 + 200.0);
        assert_eq!(
            line.graph_preds(1000.0, 100, 1.0, Laziness::HeapBound),
            100.0 + 400.0
        );
        // The measured points; a power law between them and beyond the ends.
        assert!((c.graph_dists(100.0) - 500.0).abs() < 1e-9);
        let mid = c.graph_dists(10f64.powf(1.5));
        assert!((mid - 100.0 * 5f64.sqrt()).abs() < 1e-6, "{mid}");
        assert!((c.graph_dists(10_000.0) - 8000.0).abs() < 1e-6);
        // extra(K, L): the table, log-log in L, power law in K.
        let p = |l: f64, k: usize| c.graph_preds(l, k, 1.0, Laziness::HeapBound) - k as f64;
        assert!((p(100.0, 10) - 200.0).abs() < 1e-9);
        assert!((p(1000.0, 100) - 1200.0).abs() < 1e-6);
        assert!((p(10f64.powf(1.5), 10) - 100.0 * 2f64.sqrt()).abs() < 1e-6);
        let k_mid = c.graph_preds(100.0, 30, 1.0, Laziness::HeapBound) - 30.0;
        let want = 200.0 * 3f64.powf(3f64.ln() / 10f64.ln());
        assert!((k_mid - want).abs() < 1e-6, "power law in K: {k_mid}");
        // Shapes are checked.
        c.heap_bound_extra_kl[1].pop();
        assert!(c.validate().is_err());
        c.heap_bound_extra_kl[1].push(1200.0);
        c.graph_l_dists[0] = 0.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn graph_l_for_inverts_the_distance_count() {
        let line = UnitCosts::default();
        assert!((line.graph_l_for(line.graph_dists(300.0)) - 300.0).abs() < 1e-9);
        let c = UnitCosts {
            graph_l: vec![10.0, 100.0, 1000.0],
            graph_l_dists: vec![100.0, 500.0, 2000.0],
            ..UnitCosts::default()
        };
        for l in [5.0, 10.0, 31.6, 100.0, 640.0, 5000.0] {
            let back = c.graph_l_for(c.graph_dists(l));
            assert!((back - l).abs() < 1e-6 * l, "{l} -> {back}");
        }
    }

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
    fn long_beams_raise_the_graph_distance_cost() {
        let mut c = UnitCosts {
            graph_dist_ns: 70.0,
            graph_dist_ns_l: vec![100.0, 1000.0, 10000.0],
            graph_dist_ns_curve: vec![50.0, 60.0, 120.0],
            ..UnitCosts::default()
        };
        c.validate().unwrap();
        // Never below the constant; log-linear between sizes; clamped at the ends.
        assert_eq!(c.graph_dist_at(10.0), 70.0);
        assert_eq!(c.graph_dist_at(1000.0), 70.0);
        assert!((c.graph_dist_at(10000f64.sqrt() * 1000f64.sqrt()) - 90.0).abs() < 1e-9);
        assert_eq!(c.graph_dist_at(1e6), 120.0);
        let base = c.graph_dists(10000.0) * 120.0
            + c.graph_preds(10000.0, 10, 1.0, Laziness::Eager) * c.graph_pred_ns;
        assert!((c.graph_cost(10000.0, 10, 1.0, Laziness::Eager) - base).abs() < 1e-6);
        // Without the curve: the constant (batch B6 / stage A1 calibrations).
        c.graph_dist_ns_l.clear();
        c.graph_dist_ns_curve.clear();
        assert_eq!(c.graph_dist_at(1e6), 70.0);
        c.graph_dist_ns_l = vec![100.0, 10.0];
        c.graph_dist_ns_curve = vec![1.0, 1.0];
        assert!(c.validate().is_err());
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
    fn member_work_follows_the_residual_shape() {
        let mut b = LabelStoreBuilder::new();
        for i in 0..1000u32 {
            let (a, bb, c, r) = (
                format!("a={}", i % 10),
                format!("b={}", i % 4),
                format!("c={}", i % 2),
                format!("r={}", i % 100),
            );
            b.push_point([
                ("a", a.as_str()),
                ("b", bb.as_str()),
                ("c", c.as_str()),
                ("r", r.as_str()),
            ])
            .unwrap();
        }
        let s = b.finish().unwrap();
        let mut pb = PredicateSetBuilder::new(&s);
        pb.push(&[vec!["a=1"], vec!["b=0", "b=1", "b=2"], vec!["r=1", "c=0"]]);
        let preds = pb.finish();
        let mut bits = vec![0u64; s.n_labels()];
        let mut q = Compiled::new();
        q.compile(&s, preds.get(0), Weights::Unit, None, &mut bits);
        // Bitsets for the labels with at least 200 points (b, c), rows for r.
        let bs = LabelBitsets::build(&s, 200);
        q.split_static(&s, &bs, bs.theta(), false);
        // Boolean order by posting sums: [a=1] (100), [r=1, c=0] (510), [b=0..2] (750).
        assert_eq!(
            (1..3)
                .map(|j| (q.static_slots(j).len(), q.row_labels(j).len()))
                .collect::<Vec<_>>(),
            [(1, 1), (3, 0)]
        );
        // Clause 1 (one word, a row label: no early end) passes half, clause 2 half of them.
        let surv = [0.5, 0.25];
        let w = MemberWork::of(&q, true, &surv, false, false, false);
        // Clause 1 leaves the row unread when its bitset label passes (half the members), and
        // clause 2 (bitset-only) ends half of the checks first: 0.5 · 0.5 reads.
        assert_eq!(
            (w.words, w.prefetch, w.rows, w.class_rows),
            (1.0 + 2.0, 1.0, 0.25, true)
        );
        let full = MemberWork::of(&q, true, &surv, true, false, false);
        assert_eq!((full.words, full.prefetch, full.rows), (3.0, 1.0, 0.5));
        let up = MemberWork::of(&q, true, &surv, false, true, false);
        assert_eq!((up.words, up.rows), (4.0, 1.0));
        let rows = MemberWork::of(&q, false, &surv, false, false, false);
        assert_eq!((rows.words, rows.prefetch, rows.rows), (0.0, 1.0, 1.0));
        // The A1b model (`row_reads = reach`): a read for every check reaching clause 1.
        let reach = MemberWork::of(&q, true, &surv, false, false, true);
        assert_eq!((reach.words, reach.rows), (3.0, 0.5));
        let reach_full = MemberWork::of(&q, true, &surv, true, false, true);
        assert_eq!(reach_full.rows, 1.0);
        // The shape-resolved curves; older calibrations use the class curve.
        let mut c = UnitCosts {
            pred_row_ns: vec![100.0; 6],
            pred_bits_ns: vec![10.0; 6],
            ..UnitCosts::default()
        };
        assert_eq!(c.member(&w, 1e-3), 100.0);
        assert_eq!(c.probe_point(&full, 1e-3), c.probe_ns);
        c.check_base_ns = vec![4.0; 6];
        c.check_word_ns = vec![10.0; 6];
        c.check_row_ns = vec![60.0; 6];
        c.validate().unwrap();
        assert_eq!(c.member(&w, 1e-3), 4.0 + 3.0 * 10.0 + 0.25 * 60.0);
        assert_eq!(c.probe_point(&full, 1e-3), 4.0 + 3.0 * 10.0 + 0.5 * 60.0);
        assert_eq!(
            c.probe(100.0, 32.0, &full, 1e-3),
            100.0 * c.sketch_ns + 32.0 * 64.0
        );
        // Every member's row is prefetched; the reads follow the survival.
        c.check_prefetch_ns = vec![20.0; 6];
        c.validate().unwrap();
        assert_eq!(c.member(&w, 1e-3), 4.0 + 3.0 * 10.0 + 20.0 + 0.25 * 60.0);
        let bits_only = MemberWork {
            prefetch: 0.0,
            rows: 0.0,
            ..w
        };
        assert_eq!(c.member(&bits_only, 1e-3), 4.0 + 3.0 * 10.0);
        c.check_row_ns.pop();
        assert!(c.validate().is_err());
    }

    /// Planning time with and without `rowand` (`cargo test --release -- --ignored --nocapture
    /// planning_time`): a 4-clause query with labels with and without a bitset.
    #[test]
    #[ignore = "timing, run by hand"]
    fn planning_time() {
        let mut b = LabelStoreBuilder::new();
        for i in 0..100_000u32 {
            let a = format!("a={}", i % 7);
            let c = format!("c={}", (i / 7) % 50);
            let d = format!("d={}", (i / 350) % 400);
            b.push_point([("a", a.as_str()), ("c", c.as_str()), ("d", d.as_str())])
                .unwrap();
        }
        let s = b.finish().unwrap();
        let mut pb = PredicateSetBuilder::new(&s);
        pb.push(&[
            vec!["d=1", "d=2"],
            vec!["a=1", "a=2", "c=7"],
            vec!["c=3", "c=4", "d=9", "d=10"],
            vec!["a=3", "d=11", "d=12", "d=13"],
        ]);
        // Eight clauses on a tiny union (the synthetic sets' cheapest cells).
        pb.push(&[
            vec!["d=5"],
            vec!["a=1", "a=2"],
            vec!["c=3", "d=40"],
            vec!["a=4", "c=9", "d=41"],
            vec!["c=11", "c=12"],
            vec!["d=42", "d=43", "a=5"],
            vec!["c=20", "d=44"],
            vec!["a=6", "c=21", "c=22", "d=45"],
        ]);
        let preds = pb.finish();
        let mut bits = vec![0u64; s.n_labels()];
        let bs = LabelBitsets::build(&s, 100_000u64.div_ceil(64));
        let mut q8 = Compiled::new();
        q8.compile(&s, preds.get(1), Weights::Unit, None, &mut bits);
        q8.split_static(&s, &bs, bs.theta(), false);
        q8.clear_bits(&mut bits);
        let mut q = Compiled::new();
        q.compile(&s, preds.get(0), Weights::Unit, None, &mut bits);
        q.split_static(&s, &bs, bs.theta(), false);
        let e = Estimates::from_bounds(&q, 100_000);
        let n = 6;
        let c = UnitCosts {
            enum_word_ns: 0.35,
            enum_bit_ns: 0.1,
            enum_nz_ns: 13.0,
            check_base_ns: vec![12.0; n],
            check_word_ns: vec![3.0; n],
            check_prefetch_ns: vec![40.0; n],
            check_row_ns: vec![60.0; n],
            gallop_gap: vec![1.0, 16.0, 256.0, 4096.0],
            gallop_gap_ns: vec![10.0, 12.0, 30.0, 70.0],
            ..UnitCosts::default()
        };
        let pi = PlanInputs {
            df_lower: false,
            q: &q,
            labels: &s,
            bitsets: Some(&bs),
            k: 10,
            pair: None,
        };
        let mut prepared = c.clone();
        prepared.prepare();
        let four = PlanSet::parse("scan+df+gallop+bitmap").unwrap();
        let up = Estimates::upper(&q, 100_000);
        for (costs, cname) in [(&c, "unprepared"), (&prepared, "prepared")] {
            for (name, set) in [
                ("4 plans", four),
                ("5 plans", PlanSet::ALL),
                ("4 plans", four),
                ("5 plans", PlanSet::ALL),
            ] {
                for (est, upper, ename) in [(&e, false, "bounds"), (&up, true, "upper")] {
                    let t = std::time::Instant::now();
                    let mut acc = 0.0;
                    for _ in 0..200_000 {
                        acc += std::hint::black_box(pi.best(
                            costs,
                            set,
                            std::hint::black_box(est),
                            false,
                            upper,
                        ))
                        .1;
                    }
                    let ns = t.elapsed().as_nanos() as f64 / 200_000.0;
                    println!("{cname}, {name}, {ename}: {ns:.0} ns per best() ({acc:.0})");
                }
            }
            let t = std::time::Instant::now();
            for _ in 0..200_000 {
                std::hint::black_box(pi.cost(
                    costs,
                    ExactPlan::RowAnd,
                    std::hint::black_box(&e),
                    false,
                    false,
                ));
            }
            println!(
                "{cname}, rowand cost: {:.0} ns",
                t.elapsed().as_nanos() as f64 / 200_000.0
            );
            let pi8 = PlanInputs { q: &q8, ..pi };
            let e8 = Estimates::from_bounds(&q8, 100_000);
            let up8 = Estimates::upper(&q8, 100_000);
            for (name, set) in [("4 plans", four), ("5 plans", PlanSet::ALL)] {
                for (est, upper, ename) in [(&e8, false, "bounds"), (&up8, true, "upper")] {
                    let t = std::time::Instant::now();
                    let mut acc = 0.0;
                    for _ in 0..200_000 {
                        acc += std::hint::black_box(pi8.best(
                            costs,
                            set,
                            std::hint::black_box(est),
                            false,
                            upper,
                        ))
                        .1;
                    }
                    let ns = t.elapsed().as_nanos() as f64 / 200_000.0;
                    println!(
                        "{cname}, 8 clauses, {name}, {ename}: {ns:.0} ns per best() ({acc:.0})"
                    );
                }
            }
        }
        q.clear_bits(&mut bits);
    }

    /// The planner caches and plan skips ([`UnitCosts::prepare`]) leave every plan cost and
    /// every choice of [`PlanInputs::best`] bit-identical, over random queries, estimates and
    /// unit costs.
    #[test]
    fn prepared_planner_is_bit_identical() {
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let n = 20_000u32;
        let mut b = LabelStoreBuilder::new();
        for i in 0..n {
            let a = format!("a={}", i % 7);
            let c = format!("c={}", (i / 7) % 50);
            let d = format!("d={}", (i / 350) % 400);
            let e = format!("e={}", (i * 7919) % 2000);
            let f = format!("f={}", i % 3);
            b.push_point([
                ("a", a.as_str()),
                ("c", c.as_str()),
                ("d", d.as_str()),
                ("e", e.as_str()),
                ("f", f.as_str()),
            ])
            .unwrap();
        }
        let s = b.finish().unwrap();
        let bs = LabelBitsets::build(&s, u64::from(n).div_ceil(64));
        // Frequent, mid and rare labels (tiny unions with many clauses: `rowand`'s sum stops
        // early).
        let names: Vec<String> = (0..7)
            .map(|v| format!("a={v}"))
            .chain((0..50).map(|v| format!("c={v}")))
            .chain((0..50).map(|v| format!("d={v}")))
            .chain((0..60).map(|v| format!("e={v}")))
            .chain((0..3).map(|v| format!("f={v}")))
            .collect();
        let mut pb = PredicateSetBuilder::new(&s);
        let n_queries = 60;
        for _ in 0..n_queries {
            let clauses: Vec<Vec<&str>> = (0..1 + next() % 8)
                .map(|_| {
                    (0..1 + next() % 4)
                        .map(|_| names[(next() % names.len() as u64) as usize].as_str())
                        .collect()
                })
                .collect();
            pb.push(&clauses);
        }
        let preds = pb.finish();
        let mut bits = vec![0u64; s.n_labels()];
        let grid = [1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0];
        let mut unit = |lo: f64, hi: f64| lo + (hi - lo) * (next() % 1_000_000) as f64 / 1e6;
        let mut skipped = 0usize;
        let mut decisions = 0usize;
        for t in 0..400 {
            let mut c = UnitCosts {
                word_ns: unit(0.1, 1.0),
                scatter_ns: unit(0.5, 5.0),
                merge_ns: unit(0.5, 5.0),
                gallop_ns: unit(5.0, 50.0),
                dist_ns: grid.iter().map(|_| unit(20.0, 200.0)).collect(),
                ..UnitCosts::default()
            };
            if t % 2 == 0 {
                c.enum_word_ns = unit(0.1, 1.0);
                c.enum_bit_ns = unit(0.05, 2.0);
                c.enum_nz_ns = unit(1.0, 20.0);
            }
            if t % 3 != 0 {
                c.check_base_ns = grid.iter().map(|_| unit(2.0, 30.0)).collect();
                c.check_word_ns = grid.iter().map(|_| unit(1.0, 10.0)).collect();
                c.check_prefetch_ns = grid.iter().map(|_| unit(0.0, 60.0)).collect();
                c.check_row_ns = grid.iter().map(|_| unit(10.0, 100.0)).collect();
            }
            if t % 5 != 0 {
                c.gallop_gap = vec![1.0, 16.0, 256.0, 4096.0];
                c.gallop_gap_ns = (0..4).map(|_| unit(5.0, 80.0)).collect();
            }
            let mut p = c.clone();
            p.prepare();
            assert!(p.prepared.bounds);
            let qi = t % n_queries;
            let mut q = Compiled::new();
            q.compile(&s, preds.get(qi), Weights::Unit, None, &mut bits);
            assert!(!q.is_empty());
            let split = t % 4 != 1;
            if split {
                q.split_static(&s, &bs, bs.theta(), false);
            }
            let pi = PlanInputs {
                df_lower: false,
                q: &q,
                labels: &s,
                bitsets: split.then_some(&bs),
                k: if t % 2 == 0 { 10 } else { 100 },
                pair: None,
            };
            let bounds = Estimates::from_bounds(&q, n as usize);
            let mut est = bounds.clone();
            est.a = (bounds.a * unit(0.0, 1.0)).floor();
            est.m = est.a * unit(0.0, 1.0);
            let mut sv = 1.0;
            for x in &mut est.survival {
                sv *= unit(0.0, 1.0);
                *x = sv;
            }
            let up = Estimates::upper(&q, n as usize);
            for (e, upper) in [(&bounds, false), (&est, false), (&up, true)] {
                for seeded in [false, true] {
                    let mut want = (ExactPlan::Scan, f64::INFINITY);
                    for plan in ExactPlan::ALL {
                        let x = pi.cost(&c, plan, e, seeded, upper);
                        assert_eq!(
                            x.to_bits(),
                            pi.cost(&p, plan, e, seeded, upper).to_bits(),
                            "{plan:?}"
                        );
                        let ctx = pi.ctx(&p, e, upper);
                        if let Some(lb) = pi.lower(&p, plan, e, &ctx) {
                            assert!(lb <= x * (1.0 + 1e-12), "{plan:?}: {lb} > {x}");
                            if lb > want.1 * (1.0 + SKIP_MARGIN) {
                                skipped += 1;
                            }
                        }
                        if x < want.1 {
                            want = (plan, x);
                        }
                    }
                    for set in [
                        PlanSet::ALL,
                        PlanSet::parse("scan+df+gallop+bitmap").unwrap(),
                    ] {
                        let reference =
                            set.plans()
                                .fold((ExactPlan::Scan, f64::INFINITY), |b, plan| {
                                    let x = pi.cost(&c, plan, e, seeded, upper);
                                    if x < b.1 {
                                        (plan, x)
                                    } else {
                                        b
                                    }
                                });
                        for costs in [&c, &p] {
                            let got = pi.best(costs, set, e, seeded, upper);
                            assert_eq!(
                                (got.0, got.1.to_bits()),
                                (reference.0, reference.1.to_bits())
                            );
                        }
                    }
                    assert_eq!(want.0, pi.best(&p, PlanSet::ALL, e, seeded, upper).0);
                    if !c.has_enum_bits() {
                        // Only infinite costs: the first plan of the initial choice stays.
                        let only = PlanSet::parse("rowand").unwrap();
                        let got = pi.best(&p, only, e, seeded, upper);
                        assert_eq!((got.0, got.1), (ExactPlan::Scan, f64::INFINITY));
                    }
                    decisions += 1;
                }
            }
            q.clear_bits(&mut bits);
        }
        assert!(
            decisions > 0 && skipped > 0,
            "{skipped} skips in {decisions} choices"
        );
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
        let old = UnitCosts::default();
        let c = UnitCosts {
            enum_word_ns: 0.3,
            enum_bit_ns: 2.0,
            ..UnitCosts::default()
        };
        let pi = PlanInputs {
            df_lower: false,
            q: &q,
            labels: &s,
            bitsets: None,
            k: 10,
            pair: None,
        };
        // (`pair` needs combination postings: infinite here.)
        for p in ExactPlan::ALL.into_iter().filter(|&p| p != ExactPlan::Pair) {
            let x = pi.cost(&c, p, &e, false, false);
            assert!(x.is_finite() && x > 0.0, "{p:?}");
            assert!(pi.cost(&c, p, &up, false, true) >= x, "{p:?}: upper bound");
        }
        // Without the A6 enumeration costs: no `rowand`, the bitmap enumerates at `enum_ns`
        // per word whatever its matches.
        assert_eq!(
            pi.cost(&old, ExactPlan::RowAnd, &e, false, false),
            f64::INFINITY
        );
        let words = (1000.0f64 / 64.0).ceil();
        assert!(
            (pi.cost(&old, ExactPlan::Bitmap, &e, false, false)
                - pi.cost(&c, ExactPlan::Bitmap, &e, false, false)
                - words * (old.enum_ns - c.enum_word_ns)
                + e.m * c.enum_bit_ns)
                .abs()
                < 1e-6
        );
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
            PlanSet::parse("scan+df+gallop+bitmap+rowand+pair").unwrap(),
            PlanSet::ALL
        );
        q.clear_bits(&mut bits);
    }

    #[test]
    fn df_lower_checks_every_candidate_below_k_matches() {
        // `df_lower`: distance-first counts its evaluations at the lower confidence bound of the
        // match count. With M̂ above K but its lower bound at or below K, every candidate may be
        // checked (the answer bound needs K matches); exact estimates are unaffected.
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
        let preds = pb.finish();
        let mut bits = vec![0u64; s.n_labels()];
        let mut q = Compiled::new();
        q.compile(&s, preds.get(0), Weights::Unit, None, &mut bits);
        let c = UnitCosts::default();
        let mut e = Estimates::from_bounds(&q, 1000);
        (e.m, e.m_lo, e.m_hi) = (28.0, 5.0, 100.0);
        let pi = PlanInputs {
            q: &q,
            labels: &s,
            bitsets: None,
            k: 10,
            df_lower: false,
            pair: None,
        };
        let lower = PlanInputs {
            df_lower: true,
            ..pi
        };
        let df =
            |p: &PlanInputs, e: &Estimates| p.cost(&c, ExactPlan::DistanceFirst, e, false, false);
        // At the estimate, distance-first saves checks; at the lower bound it checks every
        // candidate, as with fewer than K matches.
        let mut few = e.clone();
        few.m = 5.0;
        assert!(df(&pi, &e) < df(&pi, &few));
        assert_eq!(df(&lower, &e), df(&pi, &few));
        assert!(df(&lower, &e) > df(&pi, &e));
        // Lower bound above K: savings at the lower bound, still fewer than at the estimate.
        let mut many = e.clone();
        (many.m, many.m_lo) = (80.0, 40.0);
        assert!(df(&pi, &many) < df(&lower, &many));
        assert!(df(&lower, &many) < df(&pi, &few));
        // Exact estimates (the union enumerated) and the other plans: unchanged.
        let mut exact = e.clone();
        exact.exact = true;
        assert_eq!(df(&lower, &exact), df(&pi, &exact));
        for p in [ExactPlan::Scan, ExactPlan::Gallop, ExactPlan::Bitmap] {
            assert_eq!(
                lower.cost(&c, p, &e, false, false),
                pi.cost(&c, p, &e, false, false)
            );
        }
        q.clear_bits(&mut bits);
    }
}
