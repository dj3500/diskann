/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Per-query predicate compilation and the predicate evaluators (`PredicateEval`).
//!
//! [`Compiled::compile`] orders the clauses by increasing upper bound `U_c` (the sum of the
//! posting lengths of the clause's labels; Boolean evaluation then meets the clause most likely
//! to fail first), assigns clause `j` of that order the bit `1 << j` in a per-run clause-bit
//! table indexed by label id (queries with at most 64 clauses), and computes clause weights.
//!
//! Evaluators:
//! - [`RowScan`] (mode 1): scans the point's sorted label row and ORs the clause bits; it stops
//!   as soon as all evaluated clauses are satisfied. Rows longer than four times the number of
//!   query labels, and queries with more than 64 clauses, use mode 2;
//! - [`ClauseSearch`] (mode 2): for each clause in Boolean order, a galloping search of its
//!   sorted labels in the point's sorted row with a progressive lower bound; the Boolean form
//!   stops at the first failing clause;
//! - [`DenseBitmap`] / [`SparseList`] (`bitmap`): a lookup in a materialized match set; they
//!   give the match bit only (no failed-clause count);
//! - [`StaticBits`] (`static_bits(θ)`): labels with at least `θ` points are tested in static
//!   per-label bitsets ([`crate::bitsets::LabelBitsets`], O(1) per label, most frequent label
//!   of a clause first); the clauses that no bitset satisfies and that have rarer labels are
//!   then checked on the row (one mode-1 scan, or galloping searches for long rows). A clause
//!   made only of bitset labels never reads the row.
//!
//! The *complete* form returns the failed-clause count `g` (no early exit at a failing
//! clause). Under a clause projection, navigation evaluates only the projected clauses; the
//! full predicate (answer eligibility) is [`Eval::full`].

use crate::{
    bitsets::{self, LabelBitsets},
    labels::LabelStore,
    predicate::Cnf,
};

/// Clause weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Weights {
    /// Every clause weighs 1 (g is the number of failed clauses).
    Unit,
    /// `w_c = −ln p̂_c` with `p̂_c = clamp(U_c / n, 1/n, 1 − 1/n)`, normalized to mean 1 over
    /// the query's clauses (unit weights if they all vanish).
    Surprisal,
}

/// A compiled predicate (per query; buffers are reused).
#[derive(Debug, Clone, Default)]
pub struct Compiled {
    n_clauses: usize,
    empty: bool,
    /// Original index of the clause at each position of the Boolean order.
    order: Vec<u32>,
    /// Labels of the ordered clauses (CSR).
    offsets: Vec<u32>,
    labels: Vec<u32>,
    /// `U_c` per ordered clause.
    u: Vec<u64>,
    /// Normalized weight per ordered clause.
    w: Vec<f32>,
    unit_weights: bool,
    use_bits: bool,
    full_mask: u64,
    nav_mask: u64,
    /// Ordered clauses evaluated during navigation (all, or the projected ones).
    nav: Vec<u32>,
    projected: bool,
    nav_weight: f32,
    max_nav_weight: f32,
    scratch: Vec<(u64, u32)>,
    /// Static-bitset split per ordered clause (CSR): bitset slots of the clause's labels with
    /// a bitset (most points first) and its other labels.
    big_offsets: Vec<u32>,
    big: Vec<u32>,
    small_offsets: Vec<u32>,
    small: Vec<u32>,
    /// Ordered clauses with at least one label without a bitset.
    small_mask: u64,
    /// The first bitset slot of the first navigation clause (`u32::MAX` if none; prefetched).
    static_first: u32,
    /// Whether the static-bitset evaluator prefetches rows: some navigation clause has row
    /// labels (complete evaluation), or the first one has (Boolean evaluation, where a
    /// first clause made of bitset labels decides most nodes without the row).
    static_row_prefetch: bool,
}

impl Compiled {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compiles `cnf` against `labels`. With `projection` (sorted category ids), navigation
    /// evaluates only the clauses whose labels all lie in those categories. Sets the clause
    /// bits of every query label in `bits` (sized `n_labels`, all zero on entry) unless the
    /// query has an empty clause or more than 64 clauses; [`Self::clear_bits`] resets them.
    pub fn compile(
        &mut self,
        labels: &LabelStore,
        cnf: Cnf<'_>,
        weights: Weights,
        projection: Option<&[u32]>,
        bits: &mut [u64],
    ) {
        let c = cnf.n_clauses();
        self.n_clauses = c;
        self.empty = false;
        self.order.clear();
        self.offsets.clear();
        self.offsets.push(0);
        self.labels.clear();
        self.u.clear();
        self.w.clear();
        self.nav.clear();
        self.scratch.clear();
        self.big_offsets.clear();
        self.big.clear();
        self.small_offsets.clear();
        self.small.clear();
        self.small_mask = 0;
        self.static_first = u32::MAX;
        self.static_row_prefetch = false;
        for (i, clause) in cnf.clauses().enumerate() {
            if clause.is_empty() {
                self.empty = true;
            }
            let u: u64 = clause.iter().map(|&l| labels.posting_len(l) as u64).sum();
            self.scratch.push((u, i as u32));
        }
        if self.empty {
            self.use_bits = false;
            return;
        }
        self.scratch.sort_unstable();
        for &(u, i) in &self.scratch {
            self.order.push(i);
            self.labels.extend_from_slice(cnf.clause(i as usize));
            self.offsets.push(self.labels.len() as u32);
            self.u.push(u);
        }

        let n = labels.n_points() as f64;
        self.unit_weights = true;
        self.w.resize(c, 1.0);
        if weights == Weights::Surprisal && n > 2.0 && c > 0 {
            let raw: f64 = self
                .u
                .iter()
                .map(|&u| -(u as f64 / n).clamp(1.0 / n, 1.0 - 1.0 / n).ln())
                .sum();
            let mean = raw / c as f64;
            if mean > 0.0 && mean.is_finite() {
                for (w, &u) in self.w.iter_mut().zip(&self.u) {
                    let p = (u as f64 / n).clamp(1.0 / n, 1.0 - 1.0 / n);
                    *w = (-p.ln() / mean) as f32;
                }
                self.unit_weights = false;
            }
        }

        self.projected = projection.is_some();
        for j in 0..c {
            let keep = match projection {
                None => true,
                Some(cats) => self
                    .clause(j)
                    .iter()
                    .all(|&l| cats.binary_search(&labels.category(l)).is_ok()),
            };
            if keep {
                self.nav.push(j as u32);
            }
        }
        self.nav_weight = self.nav.iter().map(|&j| self.w[j as usize]).sum();
        self.max_nav_weight = self
            .nav
            .iter()
            .map(|&j| self.w[j as usize])
            .fold(0.0, f32::max);

        self.use_bits = c <= 64;
        if self.use_bits {
            self.full_mask = if c == 0 { 0 } else { u64::MAX >> (64 - c) };
            self.nav_mask = self.nav.iter().fold(0, |m, &j| m | 1u64 << j);
            for j in 0..c {
                let bit = 1u64 << j;
                for &l in self.clause(j) {
                    bits[l as usize] |= bit;
                }
            }
        }
    }

    /// Splits every clause for the [`StaticBits`] evaluator: labels with at least `theta`
    /// points that have a bitset in `sb` (ordered by decreasing posting length, so the label
    /// most likely to hit is tested first), and the others (checked on the row). `complete`
    /// tells whether evaluations will count failed clauses (it only changes what is
    /// prefetched). Called after [`Self::compile`] (part of the per-query compilation); a
    /// no-op for queries with an empty clause.
    pub fn split_static(
        &mut self,
        labels: &LabelStore,
        sb: &LabelBitsets,
        theta: u64,
        complete: bool,
    ) {
        self.big_offsets.clear();
        self.big.clear();
        self.small_offsets.clear();
        self.small.clear();
        self.small_mask = 0;
        self.static_first = u32::MAX;
        self.static_row_prefetch = false;
        if self.empty {
            return;
        }
        self.big_offsets.push(0);
        self.small_offsets.push(0);
        for j in 0..self.n_clauses {
            let (a, b) = (self.offsets[j] as usize, self.offsets[j + 1] as usize);
            let start = self.big.len();
            for &l in &self.labels[a..b] {
                match sb.slot(l) {
                    Some(slot) if labels.posting_len(l) as u64 >= theta => self.big.push(slot),
                    _ => self.small.push(l),
                }
            }
            self.big[start..].sort_unstable_by_key(|&slot| {
                (std::cmp::Reverse(labels.posting_len(sb.label(slot))), slot)
            });
            if self.small.len() > *self.small_offsets.last().unwrap_or(&0) as usize && j < 64 {
                self.small_mask |= 1u64 << j;
            }
            self.big_offsets.push(self.big.len() as u32);
            self.small_offsets.push(self.small.len() as u32);
        }
        if let Some(&j) = self.nav.first() {
            if let Some(&slot) = self.static_slots(j as usize).first() {
                self.static_first = slot;
            }
        }
        let has_row = |j: u32| j < 64 && self.small_mask & (1u64 << j) != 0;
        self.static_row_prefetch = if complete {
            self.nav.iter().any(|&j| has_row(j))
        } else {
            self.nav.first().is_some_and(|&j| has_row(j))
        };
    }

    /// Bitset slots of the clause at position `j` (after [`Self::split_static`]).
    #[inline(always)]
    pub fn static_slots(&self, j: usize) -> &[u32] {
        &self.big[self.big_offsets[j] as usize..self.big_offsets[j + 1] as usize]
    }

    /// Labels of the clause at position `j` without a bitset (after [`Self::split_static`]).
    #[inline(always)]
    pub fn row_labels(&self, j: usize) -> &[u32] {
        &self.small[self.small_offsets[j] as usize..self.small_offsets[j + 1] as usize]
    }

    /// Resets the clause bits set by [`Self::compile`].
    pub fn clear_bits(&self, bits: &mut [u64]) {
        if self.use_bits {
            for &l in &self.labels {
                bits[l as usize] = 0;
            }
        }
    }

    pub fn n_clauses(&self) -> usize {
        self.n_clauses
    }

    /// Whether some clause is empty (no answers).
    pub fn is_empty(&self) -> bool {
        self.empty
    }

    /// Labels of the clause at position `j` of the Boolean order.
    #[inline(always)]
    pub fn clause(&self, j: usize) -> &[u32] {
        &self.labels[self.offsets[j] as usize..self.offsets[j + 1] as usize]
    }

    /// Original index of the clause at position `j`.
    pub fn original(&self, j: usize) -> usize {
        self.order[j] as usize
    }

    /// `U_c` of the clause at position `j`.
    pub fn upper_bound(&self, j: usize) -> u64 {
        self.u[j]
    }

    /// `min_c U_c` (`u64::MAX` without clauses).
    pub fn min_upper_bound(&self) -> u64 {
        self.u.first().copied().unwrap_or(u64::MAX)
    }

    /// Normalized weight of the clause at position `j`.
    pub fn weight(&self, j: usize) -> f32 {
        self.w[j]
    }

    /// Total number of query labels.
    pub fn total_labels(&self) -> usize {
        self.labels.len()
    }

    /// Whether navigation evaluates a strict subset of the clauses.
    pub fn is_projected(&self) -> bool {
        self.projected
    }

    /// Positions of the clauses evaluated during navigation.
    pub fn nav(&self) -> &[u32] {
        &self.nav
    }

    /// Sum of the weights of the navigation clauses (the largest possible `g`).
    pub fn nav_weight(&self) -> f32 {
        self.nav_weight
    }

    /// Whether mode 1 (clause bits) is available (at most 64 clauses).
    pub fn uses_bits(&self) -> bool {
        self.use_bits
    }

    /// The largest weight of a navigation clause.
    pub fn max_nav_weight(&self) -> f32 {
        self.max_nav_weight
    }

    #[inline(always)]
    fn failed_weight(&self, failed: u64) -> f32 {
        if self.unit_weights {
            failed.count_ones() as f32
        } else {
            let mut g = 0.0;
            let mut f = failed;
            while f != 0 {
                g += self.w[f.trailing_zeros() as usize];
                f &= f - 1;
            }
            g
        }
    }
}

/// Result of a predicate evaluation.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PredOut {
    /// The node satisfies the evaluated clauses (all of them, or the projection).
    pub m: bool,
    /// Number of failed clauses (complete evaluation only).
    pub unit: u32,
    /// Weighted failed-clause count (complete evaluation only).
    pub g: f32,
}

/// What an evaluator reads.
#[derive(Debug, Clone, Copy)]
pub struct EvalCtx<'a> {
    pub(crate) labels: &'a LabelStore,
    pub(crate) q: &'a Compiled,
    /// Clause bits by label id; covers every label id of `labels` (checked by [`Self::new`]).
    bits: &'a [u64],
    /// Materialized dense match bitmap (over data points).
    pub(crate) dense: &'a [u64],
    /// Materialized sorted match list.
    pub(crate) sparse: &'a [u32],
    /// Static per-label bitsets (the query must be split against them).
    pub(crate) stat: &'a LabelBitsets,
}

impl<'a> EvalCtx<'a> {
    /// # Panics
    ///
    /// Panics if `bits` does not cover every label id of `labels`.
    #[inline]
    pub fn new(
        labels: &'a LabelStore,
        q: &'a Compiled,
        bits: &'a [u64],
        dense: &'a [u64],
        sparse: &'a [u32],
    ) -> Self {
        assert!(
            bits.len() >= labels.n_labels(),
            "the clause-bit table has {} entries for {} labels",
            bits.len(),
            labels.n_labels()
        );
        Self {
            labels,
            q,
            bits,
            dense,
            sparse,
            stat: &bitsets::EMPTY,
        }
    }

    /// The context with static per-label bitsets (for [`StaticBits`]; the query must have
    /// been split against them with [`Compiled::split_static`]).
    #[inline]
    pub fn with_static(self, stat: &'a LabelBitsets) -> Self {
        Self { stat, ..self }
    }

    /// The compiled query.
    pub fn query(&self) -> &'a Compiled {
        self.q
    }
}

/// A predicate evaluator (`PredicateEval` role); a compile-time parameter of the hot loop.
pub trait Eval: Copy + Send + Sync + 'static {
    /// Name in configurations and reports.
    const NAME: &'static str;
    /// Whether the evaluator reads forward rows (which are then prefetched).
    const ROWS: bool;
    /// Navigation evaluation of data point `v`; `complete` asks for the failed-clause count.
    fn eval(cx: &EvalCtx<'_>, v: u32, complete: bool) -> PredOut;
    /// Whether `v` satisfies the full predicate (answer eligibility).
    fn full(cx: &EvalCtx<'_>, v: u32) -> bool;
    /// Prefetches what [`Self::eval`] will read for `v` (second stage: the row itself).
    fn prefetch(cx: &EvalCtx<'_>, v: u32);
    /// First prefetch stage (the row offsets).
    fn prefetch_offsets(cx: &EvalCtx<'_>, v: u32);
}

/// Mode 1: row scan with clause bits (falls back to mode 2, see the module documentation).
#[derive(Debug, Clone, Copy)]
pub struct RowScan;

/// Mode 2: per-clause galloping search in the row.
#[derive(Debug, Clone, Copy)]
pub struct ClauseSearch;

/// Lookup in a materialized dense bitmap.
#[derive(Debug, Clone, Copy)]
pub struct DenseBitmap;

/// Binary search in a materialized sorted match list.
#[derive(Debug, Clone, Copy)]
pub struct SparseList;

/// Static per-label bitsets for frequent labels, the row for the others.
#[derive(Debug, Clone, Copy)]
pub struct StaticBits;

#[inline(always)]
fn from_failed(q: &Compiled, failed: u64, complete: bool) -> PredOut {
    if failed == 0 {
        PredOut {
            m: true,
            unit: 0,
            g: 0.0,
        }
    } else if complete {
        PredOut {
            m: false,
            unit: failed.count_ones(),
            g: q.failed_weight(failed),
        }
    } else {
        PredOut::default()
    }
}

/// ORs the clause bits of `row` (a label row of the store `cx` was built for) until `mask`
/// is covered; returns the uncovered bits.
#[inline(always)]
fn scan(cx: &EvalCtx<'_>, row: &[u32], mask: u64) -> u64 {
    let bits = cx.bits;
    let mut sat = 0u64;
    for &l in row {
        // SAFETY: rows hold label ids `< labels.n_labels()` (validated when the store was
        // loaded or built), and `EvalCtx::new` checked that `bits` covers them.
        sat |= unsafe { *bits.get_unchecked(l as usize) };
        if sat & mask == mask {
            return 0;
        }
    }
    mask & !sat
}

impl Eval for RowScan {
    const NAME: &'static str = "mode1";
    const ROWS: bool = true;

    #[inline(always)]
    fn eval(cx: &EvalCtx<'_>, v: u32, complete: bool) -> PredOut {
        let q = cx.q;
        let row = cx.labels.row(v);
        if !q.use_bits || row.len() > 4 * q.labels.len() {
            return ClauseSearch::eval(cx, v, complete);
        }
        from_failed(q, scan(cx, row, q.nav_mask), complete)
    }

    #[inline(always)]
    fn full(cx: &EvalCtx<'_>, v: u32) -> bool {
        let q = cx.q;
        let row = cx.labels.row(v);
        if !q.use_bits || row.len() > 4 * q.labels.len() {
            return ClauseSearch::full(cx, v);
        }
        scan(cx, row, q.full_mask) == 0
    }

    #[inline(always)]
    fn prefetch(cx: &EvalCtx<'_>, v: u32) {
        let (offsets, rows) = cx.labels.forward();
        super::prefetch(rows.as_ptr().wrapping_add(offsets[v as usize] as usize));
    }

    #[inline(always)]
    fn prefetch_offsets(cx: &EvalCtx<'_>, v: u32) {
        super::prefetch(cx.labels.forward().0.as_ptr().wrapping_add(v as usize));
    }
}

impl Eval for ClauseSearch {
    const NAME: &'static str = "mode2";
    const ROWS: bool = true;

    #[inline(always)]
    fn eval(cx: &EvalCtx<'_>, v: u32, complete: bool) -> PredOut {
        let q = cx.q;
        let row = cx.labels.row(v);
        let mut unit = 0;
        let mut g = 0.0;
        for &j in &q.nav {
            if !gallop_intersects(row, q.clause(j as usize)) {
                if !complete {
                    return PredOut::default();
                }
                unit += 1;
                g += q.w[j as usize];
            }
        }
        PredOut {
            m: unit == 0,
            unit,
            g: if q.unit_weights { unit as f32 } else { g },
        }
    }

    #[inline(always)]
    fn full(cx: &EvalCtx<'_>, v: u32) -> bool {
        let q = cx.q;
        let row = cx.labels.row(v);
        (0..q.n_clauses).all(|j| gallop_intersects(row, q.clause(j)))
    }

    #[inline(always)]
    fn prefetch(cx: &EvalCtx<'_>, v: u32) {
        RowScan::prefetch(cx, v);
    }

    #[inline(always)]
    fn prefetch_offsets(cx: &EvalCtx<'_>, v: u32) {
        RowScan::prefetch_offsets(cx, v);
    }
}

impl Eval for DenseBitmap {
    const NAME: &'static str = "bitmap(dense)";
    const ROWS: bool = false;

    #[inline(always)]
    fn eval(cx: &EvalCtx<'_>, v: u32, _complete: bool) -> PredOut {
        PredOut {
            m: Self::full(cx, v),
            unit: 0,
            g: 0.0,
        }
    }

    #[inline(always)]
    fn full(cx: &EvalCtx<'_>, v: u32) -> bool {
        (cx.dense[(v >> 6) as usize] >> (v & 63)) & 1 == 1
    }

    #[inline(always)]
    fn prefetch(cx: &EvalCtx<'_>, v: u32) {
        super::prefetch(cx.dense.as_ptr().wrapping_add((v >> 6) as usize));
    }

    #[inline(always)]
    fn prefetch_offsets(_cx: &EvalCtx<'_>, _v: u32) {}
}

impl Eval for SparseList {
    const NAME: &'static str = "bitmap(sparse)";
    const ROWS: bool = false;

    #[inline(always)]
    fn eval(cx: &EvalCtx<'_>, v: u32, _complete: bool) -> PredOut {
        PredOut {
            m: Self::full(cx, v),
            unit: 0,
            g: 0.0,
        }
    }

    #[inline(always)]
    fn full(cx: &EvalCtx<'_>, v: u32) -> bool {
        cx.sparse.binary_search(&v).is_ok()
    }

    #[inline(always)]
    fn prefetch(_cx: &EvalCtx<'_>, _v: u32) {}

    #[inline(always)]
    fn prefetch_offsets(_cx: &EvalCtx<'_>, _v: u32) {}
}

/// Failed clauses among `mask` (ordered positions) under [`StaticBits`]: bitset lookups clause
/// by clause in Boolean order, then one row pass for the unsatisfied clauses with labels
/// without a bitset. With `early`, returns as soon as a clause is known to fail (the result
/// is then nonzero but may omit other failed clauses).
#[inline(always)]
fn static_failed(cx: &EvalCtx<'_>, v: u32, mask: u64, early: bool) -> u64 {
    let q = cx.q;
    let sb = cx.stat;
    let (w, b) = ((v >> 6) as usize, v & 63);
    let mut sat = 0u64;
    let mut m = mask;
    while m != 0 {
        let j = m.trailing_zeros() as usize;
        m &= m - 1;
        let bit = 1u64 << j;
        for &s in q.static_slots(j) {
            if (sb.word(s, w) >> b) & 1 != 0 {
                sat |= bit;
                break;
            }
        }
        if early && sat & bit == 0 && q.small_mask & bit == 0 {
            return bit;
        }
    }
    let remaining = mask & !sat;
    let need_row = remaining & q.small_mask;
    let mut failed = remaining & !q.small_mask;
    if need_row != 0 {
        let row = cx.labels.row(v);
        if row.len() > 4 * q.small.len() {
            let mut r = need_row;
            while r != 0 {
                let j = r.trailing_zeros() as usize;
                r &= r - 1;
                if !gallop_intersects(row, q.row_labels(j)) {
                    failed |= 1u64 << j;
                    if early {
                        return failed;
                    }
                }
            }
        } else {
            // The clause-bit table covers every query label; the bitset labels of the
            // clauses in `need_row` are absent from the row (their lookups failed).
            failed |= scan(cx, row, need_row);
        }
    }
    failed
}

impl Eval for StaticBits {
    const NAME: &'static str = "static_bits";
    const ROWS: bool = true;

    #[inline(always)]
    fn eval(cx: &EvalCtx<'_>, v: u32, complete: bool) -> PredOut {
        let q = cx.q;
        if !q.use_bits {
            return ClauseSearch::eval(cx, v, complete);
        }
        from_failed(q, static_failed(cx, v, q.nav_mask, !complete), complete)
    }

    #[inline(always)]
    fn full(cx: &EvalCtx<'_>, v: u32) -> bool {
        let q = cx.q;
        if !q.use_bits {
            return ClauseSearch::full(cx, v);
        }
        static_failed(cx, v, q.full_mask, true) == 0
    }

    /// The row, when it is likely to be read (see [`Compiled::split_static`]).
    #[inline(always)]
    fn prefetch(cx: &EvalCtx<'_>, v: u32) {
        if cx.q.static_row_prefetch {
            RowScan::prefetch(cx, v);
        }
    }

    /// The row offsets (when the row is likely to be read) and the bitset word of the first
    /// navigation clause's most frequent label.
    #[inline(always)]
    fn prefetch_offsets(cx: &EvalCtx<'_>, v: u32) {
        let q = cx.q;
        if q.static_row_prefetch {
            RowScan::prefetch_offsets(cx, v);
        }
        if q.static_first != u32::MAX {
            super::prefetch(cx.stat.word_ptr(q.static_first, (v >> 6) as usize));
        }
    }
}

/// Whether the sorted lists `row` and `clause` share an element: each clause label is searched
/// in `row` by galloping from the position reached by the previous label.
#[inline]
pub fn gallop_intersects(row: &[u32], clause: &[u32]) -> bool {
    let n = row.len();
    let mut lo = 0usize;
    for &l in clause {
        if lo >= n {
            return false;
        }
        if row[lo] >= l {
            if row[lo] == l {
                return true;
            }
            continue;
        }
        // row[lo] < l: gallop to a position whose element is >= l.
        let mut prev = lo;
        let mut step = 1;
        let mut cur = lo + 1;
        while cur < n && row[cur] < l {
            prev = cur;
            step *= 2;
            cur = prev + step;
        }
        let hi = cur.min(n);
        let at = prev + 1 + row[prev + 1..hi].partition_point(|&x| x < l);
        if at < n && row[at] == l {
            return true;
        }
        lo = at;
    }
    false
}

/// Position of the first element `>= x` in `list[from..]` (galloping), as an index of `list`.
#[inline]
pub fn gallop_lower_bound(list: &[u32], from: usize, x: u32) -> usize {
    let n = list.len();
    if from >= n || list[from] >= x {
        return from;
    }
    let mut prev = from;
    let mut step = 1;
    let mut cur = from + 1;
    while cur < n && list[cur] < x {
        prev = cur;
        step *= 2;
        cur = prev + step;
    }
    let hi = cur.min(n);
    prev + 1 + list[prev + 1..hi].partition_point(|&y| y < x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        labels::LabelStoreBuilder,
        predicate::{intersects, PredicateSetBuilder},
    };

    #[test]
    fn galloping_agrees_with_binary_search() {
        let mut state = 12345u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32
        };
        for _ in 0..2000 {
            let mut row: Vec<u32> = (0..next() % 40).map(|_| next() % 100).collect();
            row.sort_unstable();
            row.dedup();
            let mut clause: Vec<u32> = (0..1 + next() % 6).map(|_| next() % 100).collect();
            clause.sort_unstable();
            clause.dedup();
            assert_eq!(
                gallop_intersects(&row, &clause),
                intersects(&row, &clause),
                "{row:?} {clause:?}"
            );
            let x = next() % 110;
            let from = (next() as usize) % (row.len() + 1);
            assert_eq!(
                gallop_lower_bound(&row, from, x),
                from + row[from..].partition_point(|&y| y < x)
            );
        }
    }

    /// Points with labels over categories `a` (a=0..3) and `b` (b=0..1).
    fn store() -> LabelStore {
        let mut b = LabelStoreBuilder::new();
        let rows: [&[&str]; 6] = [
            &["a=0", "b=0"],
            &["a=1", "b=0"],
            &["a=1", "a=2", "b=1"],
            &["a=3"],
            &[],
            &["a=0", "a=1", "a=2", "a=3", "b=0", "b=1"],
        ];
        for row in rows {
            b.push_point(row.iter().map(|&l| (&l[..1], l))).unwrap();
        }
        b.finish().unwrap()
    }

    fn all_modes(cx: &EvalCtx<'_>, v: u32, complete: bool) -> PredOut {
        let a = RowScan::eval(cx, v, complete);
        let b = ClauseSearch::eval(cx, v, complete);
        assert_eq!(a, b, "mode 1 and mode 2 disagree on {v}");
        a
    }

    #[test]
    fn compile_order_bits_and_evaluation() {
        let s = store();
        let mut pb = PredicateSetBuilder::new(&s);
        // Clause 0: a=1|a=3 (U = 3 + 2 = 5); clause 1: b=0 (U = 3).
        pb.push(&[vec!["a=1", "a=3"], vec!["b=0"]]);
        pb.push(&[vec!["a=9"], vec!["b=0"]]);
        pb.push::<&str>(&[]);
        let preds = pb.finish();
        let mut bits = vec![0u64; s.n_labels()];
        let mut q = Compiled::new();

        q.compile(&s, preds.get(0), Weights::Unit, None, &mut bits);
        assert!(!q.is_empty() && q.uses_bits());
        assert_eq!((q.original(0), q.original(1)), (1, 0), "smaller U first");
        assert_eq!((q.upper_bound(0), q.upper_bound(1)), (3, 5));
        let cx = EvalCtx::new(&s, &q, &bits, &[], &[]);
        let expect = [
            (false, 1),
            (true, 0),
            (false, 1),
            (false, 1),
            (false, 2),
            (true, 0),
        ];
        for (v, &(m, unit)) in expect.iter().enumerate() {
            let p = all_modes(&cx, v as u32, true);
            assert_eq!((p.m, p.unit, p.g), (m, unit, unit as f32), "point {v}");
            assert_eq!(all_modes(&cx, v as u32, false).m, m);
            assert_eq!(RowScan::full(&cx, v as u32), m);
            assert_eq!(ClauseSearch::full(&cx, v as u32), m);
        }
        q.clear_bits(&mut bits);
        assert!(bits.iter().all(|&b| b == 0));

        // Unknown labels were dropped: an empty clause.
        q.compile(&s, preds.get(1), Weights::Unit, None, &mut bits);
        assert!(q.is_empty());
        assert!(bits.iter().all(|&b| b == 0));

        // No clause: everything matches.
        q.compile(&s, preds.get(2), Weights::Unit, None, &mut bits);
        let cx = EvalCtx::new(&s, &q, &bits, &[], &[]);
        assert_eq!(q.min_upper_bound(), u64::MAX);
        for v in 0..6 {
            assert!(all_modes(&cx, v, true).m && RowScan::full(&cx, v));
        }
    }

    #[test]
    fn surprisal_weights_and_projection() {
        let s = store();
        let mut pb = PredicateSetBuilder::new(&s);
        pb.push(&[vec!["a=1", "a=3"], vec!["b=0"]]);
        let preds = pb.finish();
        let mut bits = vec![0u64; s.n_labels()];
        let mut q = Compiled::new();
        q.compile(&s, preds.get(0), Weights::Surprisal, None, &mut bits);
        let (w0, w1) = (q.weight(0), q.weight(1));
        assert!((w0 + w1 - 2.0).abs() < 1e-5, "mean 1");
        assert!(w0 > w1, "the rarer clause (U = 3) weighs more");
        let cx = EvalCtx::new(&s, &q, &bits, &[], &[]);
        // Point 3 (a=3) fails only clause b=0, the first in Boolean order.
        let p = all_modes(&cx, 3, true);
        assert_eq!((p.unit, p.g), (1, w0));
        q.clear_bits(&mut bits);

        // Project onto category `b`.
        let b = s.category(s.name_index()["b=0"]);
        q.compile(&s, preds.get(0), Weights::Unit, Some(&[b]), &mut bits);
        assert!(q.is_projected());
        assert_eq!(q.nav(), &[0]);
        let cx = EvalCtx::new(&s, &q, &bits, &[], &[]);
        // Point 0 (a=0, b=0) satisfies the projection but not the full predicate.
        assert!(all_modes(&cx, 0, true).m);
        assert!(!RowScan::full(&cx, 0) && !ClauseSearch::full(&cx, 0));
        assert!(RowScan::full(&cx, 1));
        assert_eq!(all_modes(&cx, 2, true).unit, 1);
    }

    #[test]
    fn static_bits_agree_with_the_row_modes() {
        // 300 points over categories a (40 values, Zipf-like) and b (5 values); every 25th
        // point carries 30 a-labels (a long row: the galloping path).
        let mut state = 99u64;
        let mut next = |n: u32| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as u32) % n
        };
        let mut b = LabelStoreBuilder::new();
        for i in 0..300 {
            let mut row: Vec<String> = Vec::new();
            let na = if i % 25 == 0 { 30 } else { 1 + next(3) };
            for _ in 0..na {
                let r = next(40);
                row.push(format!("a={}", r * r / 40));
            }
            if next(4) != 0 {
                row.push(format!("b={}", next(5)));
            }
            b.push_point(row.iter().map(|l| (&l[..1], l.as_str())))
                .unwrap();
        }
        let s = b.finish().unwrap();
        let mut pb = PredicateSetBuilder::new(&s);
        for _ in 0..60 {
            let nc = 1 + next(4);
            let clauses: Vec<Vec<String>> = (0..nc)
                .map(|c| {
                    let width = 1 + next(5);
                    (0..width)
                        .map(|_| {
                            if c % 2 == 0 {
                                format!("a={}", next(40))
                            } else {
                                format!("b={}", next(5))
                            }
                        })
                        .collect()
                })
                .collect();
            pb.push(&clauses);
        }
        let preds = pb.finish();
        let all = LabelBitsets::build(&s, 1);
        let mut bits = vec![0u64; s.n_labels()];
        let mut q = Compiled::new();
        let mut row_reads_avoided = 0;
        for theta in [1, 5, 20, 60, 1000] {
            let sb = LabelBitsets::build(&s, theta);
            for (bitsets, t) in [(&sb, theta), (&all, theta)] {
                for i in 0..preds.len() {
                    q.compile(&s, preds.get(i), Weights::Surprisal, None, &mut bits);
                    if q.is_empty() {
                        q.clear_bits(&mut bits);
                        continue;
                    }
                    q.split_static(&s, bitsets, t, i % 2 == 0);
                    if (0..q.n_clauses()).all(|j| q.row_labels(j).is_empty()) {
                        row_reads_avoided += 1;
                    }
                    let cx = EvalCtx::new(&s, &q, &bits, &[], &[]).with_static(bitsets);
                    for v in 0..300 {
                        let want = RowScan::eval(&cx, v, true);
                        assert_eq!(StaticBits::eval(&cx, v, true), want, "θ {t} q {i} v {v}");
                        assert_eq!(StaticBits::eval(&cx, v, false).m, want.m);
                        assert_eq!(StaticBits::full(&cx, v), want.m);
                        assert_eq!(ClauseSearch::full(&cx, v), want.m);
                    }
                    q.clear_bits(&mut bits);
                }
            }
        }
        assert!(
            row_reads_avoided > 0,
            "some queries are decided by bitsets alone"
        );
    }

    #[test]
    fn more_than_64_clauses_use_mode_2() {
        let mut b = LabelStoreBuilder::new();
        let names: Vec<String> = (0..70).map(|i| format!("c{i}=x")).collect();
        b.push_point(
            names
                .iter()
                .map(|n| (&n[..n.find('=').unwrap()], n.as_str())),
        )
        .unwrap();
        b.push_point(
            names[..69]
                .iter()
                .map(|n| (&n[..n.find('=').unwrap()], n.as_str())),
        )
        .unwrap();
        let s = b.finish().unwrap();
        let mut pb = PredicateSetBuilder::new(&s);
        let clauses: Vec<Vec<&str>> = names.iter().map(|n| vec![n.as_str()]).collect();
        pb.push(&clauses);
        let preds = pb.finish();
        let mut bits = vec![0u64; s.n_labels()];
        let mut q = Compiled::new();
        q.compile(&s, preds.get(0), Weights::Unit, None, &mut bits);
        assert!(!q.uses_bits() && bits.iter().all(|&b| b == 0));
        let cx = EvalCtx::new(&s, &q, &bits, &[], &[]);
        assert!(all_modes(&cx, 0, true).m && RowScan::full(&cx, 0));
        let p = all_modes(&cx, 1, true);
        assert_eq!((p.m, p.unit), (false, 1));
        assert!(!RowScan::full(&cx, 1));
        let sb = LabelBitsets::build(&s, 1);
        q.split_static(&s, &sb, 1, true);
        let cx = EvalCtx::new(&s, &q, &bits, &[], &[]).with_static(&sb);
        assert_eq!(StaticBits::eval(&cx, 1, true), p);
        assert!(StaticBits::full(&cx, 0) && !StaticBits::full(&cx, 1));
    }
}
