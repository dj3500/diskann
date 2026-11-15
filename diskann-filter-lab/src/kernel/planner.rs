/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Per-query planning of the exact/graph crossover (`crossover` role):
//!
//! 1. posting bounds of the compiled predicate (`P_c = U_c`, `c*` = the first clause of the
//!    Boolean order); if the exact path's *upper* cost (`A = min(n, P*)`, `M = A`) is at most
//!    the cheapest graph search (`L = L₀`), the query takes the exact path without a probe;
//! 2. estimates ([`Estimates`]): posting bounds, or a probe of the bottom-`b` sketches (a
//!    uniform sample of `union(c*)`, each sampled point checked against every clause, in
//!    stages: the first [`PROBE_STAGE`] points, then the rest of the sample only if the match
//!    rate estimate is below 1/2), which gives `A`, `M`, the residual-clause survival,
//!    confidence bounds and the sampled matches (held-out witnesses of the `guarded`
//!    crossover);
//! 3. the cheapest exact plan `E`, the planned list size (`posting_l`: `L₀ × m(M̂ / n)`) and
//!    the nominal graph cost `G`; the exact path if `E ≤ G`;
//! 4. `guarded`: a graph trial watched by a [`Guard`] (forecast of the remaining graph cost
//!    from the arrival rate of answers while the heap is incomplete; spend cap `E`; at nominal
//!    termination an incomplete heap or a sampled match closer than the K-th answer), with a
//!    fallback that exhausts the exact path seeded with the graph's answers.

use super::{
    cost::{Estimates, ExactPlan},
    eval::{check, Eval, EvalCtx},
    search::Counters,
};
use crate::sketches::LabelSketches;

/// Why a guarded graph trial fell back to the exact path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum Fallback {
    #[default]
    None = 0,
    /// The forecast remaining graph cost exceeded the exact cost.
    Forecast = 1,
    /// Graph spend reached the estimated exact cost.
    Budget = 2,
    /// Fewer than K answers at nominal termination.
    Incomplete = 3,
    /// A sampled match closer than the K-th answer at nominal termination.
    Witness = 4,
}

/// Points of the first probe stage: the probe stops there when the estimated match rate is at
/// least 1/2 (upstream's multiplier is then 1, and the exact path costs at least `n/2`
/// distances); otherwise it checks the whole sample.
pub const PROBE_STAGE: usize = 32;

/// Probe scratch and outputs (reused per query).
#[derive(Debug, Clone, Default)]
pub struct Probe {
    sample: Vec<u32>,
    heap: Vec<(u64, u32)>,
    pos: Vec<usize>,
    surv: Vec<usize>,
    /// Sampled points that satisfy the full predicate.
    pub witnesses: Vec<u32>,
}

impl Probe {
    pub fn new() -> Self {
        Self::default()
    }

    /// Samples `union(c*)` with the sketches and evaluates every clause on the sampled points
    /// (with the evaluator `P`, rows prefetched ahead); returns the estimates. Counts the
    /// evaluations in `n_pred` and `n_probe`. `cx` holds a compiled, non-empty predicate with
    /// at least one clause.
    pub fn run<P: Eval>(
        &mut self,
        sk: &LabelSketches,
        cx: &EvalCtx<'_>,
        n: usize,
        ctr: &mut Counters,
    ) -> Estimates {
        let q = cx.query();
        let labels = cx.labels;
        let c = q.n_clauses();
        let c0 = q.clause(0);
        let b = sk.b();
        let u = sk.sample_union(
            c0,
            |l| labels.posting_len(l) <= b,
            &mut self.heap,
            &mut self.pos,
            &mut self.sample,
        );
        self.witnesses.clear();
        self.surv.clear();
        self.surv.resize(c.saturating_sub(1), 0);
        let sample = &self.sample;
        let m = sample.len();
        // Union size estimate (for the stage-1 stopping rule).
        let a_stage = if u.complete {
            m as f64
        } else {
            (m as f64 - 1.0).max(1.0) / u.tau
        };
        const AHEAD: usize = 8;
        if c > 1 {
            for &v in &sample[..m.min(2 * AHEAD)] {
                P::prefetch_offsets(cx, v);
            }
            for &v in &sample[..m.min(AHEAD)] {
                P::prefetch(cx, v);
            }
        }
        let mut evaluated = m;
        for i in 0..m {
            if i == PROBE_STAGE
                && !u.complete
                && a_stage * self.witnesses.len() as f64 / i as f64 >= 0.5 * n as f64
            {
                evaluated = i;
                break;
            }
            let v = sample[i];
            if c <= 1 {
                self.witnesses.push(v);
                continue;
            }
            if i + 2 * AHEAD < m {
                P::prefetch_offsets(cx, sample[i + 2 * AHEAD]);
            }
            if i + AHEAD < m {
                P::prefetch(cx, sample[i + AHEAD]);
            }
            let ok = if c <= 64 {
                // Clause 0 holds for every sampled point; survival after clauses 1..=j.
                let failed = P::failed_all(cx, v) & !1;
                for j in 1..c {
                    let mask = if j >= 63 {
                        u64::MAX
                    } else {
                        (1u64 << (j + 1)) - 1
                    };
                    if failed & mask != 0 {
                        break;
                    }
                    self.surv[j - 1] += 1;
                }
                failed == 0
            } else {
                let ok = check::<P>(cx, v);
                if ok {
                    for s in &mut self.surv {
                        *s += 1;
                    }
                }
                ok
            };
            if ok {
                self.witnesses.push(v);
            }
        }
        if c > 1 {
            ctr.n_pred += evaluated as u64;
            ctr.n_probe += evaluated as u64;
        }
        let a_min = c0
            .iter()
            .map(|&l| labels.posting_len(l) as f64)
            .fold(0.0, f64::max);
        let a_max = (q.upper_bound(0) as f64).min(n as f64);
        Estimates::from_sample(
            n,
            m,
            u.complete,
            u.tau,
            evaluated,
            self.witnesses.len(),
            &self.surv,
            a_min,
            a_max,
        )
    }
}

/// Watches a guarded graph trial (see the module documentation). Costs are model costs of
/// the counters ([`super::cost::UnitCosts::charged`]), so a trial's decisions do not depend
/// on timing noise.
#[derive(Debug, Clone, Copy)]
pub struct Guard {
    /// The estimated exact cost (spend cap and the forecast's alternative).
    pub budget: f64,
    pub k: usize,
    pub forecast: bool,
    pub cap: bool,
    pub witness: bool,
    pub dist_ns: f64,
    pub pred_ns: f64,
    /// Counters after the starts (the arrival rate excludes them).
    pub start_dist: u64,
    pub start_pred: u64,
    pub start_matches: u64,
    /// Distance computations at the next checkpoint.
    pub next_check: u64,
    pub outcome: Fallback,
}

impl Guard {
    /// A guard for a trial with exact cost `budget`.
    pub fn new(
        budget: f64,
        k: usize,
        fb: super::config::Fallbacks,
        dist_ns: f64,
        pred_ns: f64,
    ) -> Self {
        Self {
            budget,
            k,
            forecast: fb.forecast,
            cap: fb.budget,
            witness: fb.witness,
            dist_ns,
            pred_ns,
            start_dist: 0,
            start_pred: 0,
            start_matches: 0,
            next_check: u64::MAX,
            outcome: Fallback::None,
        }
    }

    /// Records the counters after the starts and arms the first checkpoint (64 distance
    /// computations later; then at doubling intervals).
    pub fn arm(&mut self, ctr: &Counters) {
        self.start_dist = ctr.n_dist;
        self.start_pred = ctr.n_pred;
        self.start_matches = ctr.n_match_seen;
        self.next_check = ctr.n_dist + 64;
    }

    fn spent(&self, n_dist: u64, n_pred: u64) -> f64 {
        n_dist as f64 * self.dist_ns + n_pred as f64 * self.pred_ns
    }

    /// A checkpoint (called when `n_dist ≥ next_check`): whether to stop the trial.
    #[cold]
    pub fn check(&mut self, n_dist: u64, n_pred: u64, n_matches: u64, collected: usize) -> bool {
        let gap = (self.next_check - self.start_dist).max(1);
        while self.next_check <= n_dist {
            self.next_check = self.start_dist + 2 * (self.next_check - self.start_dist).max(gap);
        }
        let spent = self.spent(n_dist, n_pred);
        if self.cap && spent >= self.budget {
            self.outcome = Fallback::Budget;
            return true;
        }
        if self.forecast && collected < self.k {
            let since = (spent - self.spent(self.start_dist, self.start_pred)).max(1.0);
            let arrivals = n_matches.saturating_sub(self.start_matches) as f64;
            // Optimistic rate: one more arrival than observed.
            let rate = (arrivals + 1.0) / since;
            let more = (self.k - collected) as f64 / rate;
            if more > self.budget {
                self.outcome = Fallback::Forecast;
                return true;
            }
        }
        false
    }
}

/// The decision of the crossover for one query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision {
    pub exact: bool,
    pub plan: ExactPlan,
    pub est_exact: f64,
    pub est_graph: f64,
    /// List size of the graph search.
    pub l: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::config::Fallbacks;

    #[test]
    fn guard_checkpoints_double_and_fall_back() {
        let mut ctr = Counters {
            n_dist: 100,
            n_pred: 100,
            n_match_seen: 3,
            ..Counters::default()
        };
        // Budget 10,000 ns; 10 ns per distance and per predicate.
        let mut g = Guard::new(10_000.0, 10, Fallbacks::ALL, 10.0, 10.0);
        g.arm(&ctr);
        assert_eq!(g.next_check, 164);
        // 64 distances later with 9 answers already collected and 5 arrivals: continue.
        ctr.n_dist = 164;
        ctr.n_pred = 164;
        assert!(!g.check(ctr.n_dist, ctr.n_pred, 8, 9));
        assert_eq!(g.next_check, 228, "the interval doubles");
        // No arrival and an empty heap: the forecast (10 × 1,280 ns) exceeds the budget.
        let mut g2 = Guard::new(10_000.0, 10, Fallbacks::ALL, 10.0, 10.0);
        g2.arm(&ctr);
        assert!(g2.check(228, 228, 3, 0));
        assert_eq!(g2.outcome, Fallback::Forecast);
        // Without the forecast, the spend cap stops the trial at the budget.
        let only_cap = Fallbacks::parse("budget").unwrap();
        let mut g3 = Guard::new(10_000.0, 10, only_cap, 10.0, 10.0);
        g3.arm(&ctr);
        assert!(!g3.check(228, 228, 8, 0));
        assert!(g3.check(600, 500, 8, 0));
        assert_eq!(g3.outcome, Fallback::Budget);
    }
}
