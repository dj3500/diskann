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
//!    confidence bounds and the sampled matches;
//! 3. the cheapest exact plan `E`, the planned list size (`posting_l`: `L₀ × m(M̂ / n)`) and
//!    the nominal graph cost `G`; the exact path if `E ≤ G`.

use super::{
    cost::{Estimates, ExactPlan},
    eval::{Eval, EvalCtx},
    search::Counters,
};
use crate::sketches::LabelSketches;

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
                let ok = P::full(cx, v);
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
