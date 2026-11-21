/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Combination postings (sidecar `pairs.flab`): the joint postings `P_a ∩ P_b` of selected label
//! pairs, for the exact-path plan `pair` ([`crate::kernel::cost::ExactPlan::Pair`]), which
//! enumerates the members of two clauses at once instead of one clause's union.
//!
//! Selection, from the label store only (label ids and posting statistics; categories are not
//! read), under a memory budget `B` (bytes):
//! - *big labels* have postings of at least `T` points. `T` is the smallest power of two
//!   `≥ 64` whose co-occurrence matrix (one bit per ordered pair of big labels) takes at most a
//!   quarter of `B`;
//! - every pair of big labels that co-occurs (`j = |P_a ∩ P_b| > 0`) is a candidate with gain
//!   `g = min(|P_a|, |P_b|) − j` (the candidates a plan saves by enumerating the pair instead of
//!   the smaller label) and cost `j + 3` entries (its list and its directory entry, 4 bytes
//!   each). Pairs are taken by decreasing key while the budget lasts (a threshold on the key,
//!   read from a histogram of 8 bins per octave). The key is the expected saving per stored
//!   entry under a [`Prior`] on which pairs queries use: `g / (j + 3)` when every co-occurring
//!   pair is equally likely (`uniform`), `(min(|P_a|, |P_b|) / n) · g / (j + 3)` when a pair is
//!   as likely as its smaller label is frequent (`rare`: a query's rarest label drawn like a
//!   data point's label). Under `uniform` a larger budget, which lowers `T`, admits many pairs of
//!   small labels with tiny joints (high gain per entry, small saving per query) that can
//!   displace pairs of large labels; `rare` ranks those below. A *nested* selection
//!   ([`PairIndex::build_chain`]) is made along a chain of increasing budgets, each keeping every
//!   pair of the previous one, so that it is monotone in the budget: a budget's chain is the
//!   1-2-5 ladder from 5 % of the label postings below it, then the budget itself
//!   ([`nested_chain`]);
//! - the co-occurrence matrix tells which big pairs are empty: they need no list.
//!
//! A lookup of labels `(a, b)` answers [`Joint::Stored`] (the list), [`Joint::Empty`] (both big,
//! never together) or [`Joint::Unknown`] (a small label, or a co-occurring pair that was not
//! selected). A clause pair `(i, j)` of a query is *covered* when every label pair of its two
//! clauses is stored or empty; the union of the stored lists is then exactly the set of points
//! satisfying both clauses ([`PairIndex::cover`]).
//!
//! Build: two passes over the postings of the big labels; for each big label `a` and member
//! `p`, the labels of `p` after `a` in its row (two parallel passes: counts and histogram, then
//! counts and lists). A chain is built in the same two passes at its last budget's `T` (the
//! histogram is kept by key bin and by the size level of the pair's smaller label, from which
//! every step's threshold follows). Memory: 4 bytes per label (big index),
//! `N_big · ⌈N_big / 64⌉ · 8` bytes of matrix, `8 (N_big + 1)` bytes of row offsets, 12 bytes per
//! stored pair and 4 bytes per stored entry.

use std::{
    io::{Read, Seek, Write},
    path::Path,
};

use anyhow::{ensure, Context, Result};

use crate::{container, labels::LabelStore, parallel};

/// File magic of `pairs.flab`.
pub const MAGIC: &[u8; 8] = b"FLABPRS\0";
/// Version of the `pairs.flab` format.
pub const VERSION: u32 = 1;

/// Not a big label.
const NONE: u32 = u32::MAX;
/// Smallest size threshold of the big labels.
pub const T_MIN: u64 = 64;
/// Directory cost of a stored pair, in entries (partner id and list offset).
const PAIR_OVERHEAD: u64 = 3;
/// Histogram resolution of the selection key: bins per octave over 80 octaves from the
/// prior's [`Prior::key_min_log2`].
const BINS_PER_OCTAVE: f64 = 8.0;
const N_BINS: usize = 8 * 80;

/// Which label pairs queries are assumed to use (the selection's ranking; see the module
/// documentation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Prior {
    /// Every co-occurring pair equally likely: rank by gain per entry.
    #[default]
    Uniform,
    /// A pair as likely as its smaller label is frequent: gain per entry times
    /// `min(|P_a|, |P_b|) / n`.
    Rare,
    /// A pair as likely as a uniformly drawn data point carries both labels (base-derived
    /// queries: sub-predicates of points' own label sets): gain per entry times `j / n` —
    /// the conjunctions points actually have, by the members they save.
    Data,
}

impl Prior {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Rare => "rare",
            Self::Data => "data",
        }
    }

    /// log2 of the key at the top of histogram bin 0.
    fn key_min_log2(self) -> f64 {
        match self {
            Self::Uniform => -10.0,
            Self::Rare | Self::Data => -40.0,
        }
    }

    /// Selection key of a pair with smaller label `min` members, `gain` and `cost` entries on
    /// `n` points.
    fn key(self, min: u64, gain: u64, cost: u64, n: u64) -> f64 {
        let r = gain as f64 / cost as f64;
        match self {
            Self::Uniform => r,
            Self::Rare => r * min as f64 / n.max(1) as f64,
            Self::Data => r * cost.saturating_sub(PAIR_OVERHEAD) as f64 / n.max(1) as f64,
        }
    }
}

impl std::str::FromStr for Prior {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "uniform" => Ok(Self::Uniform),
            "rare" => Ok(Self::Rare),
            "data" => Ok(Self::Data),
            _ => anyhow::bail!("unknown pair prior {s:?} (uniform, rare, data)"),
        }
    }
}

/// Result of a pair lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Joint {
    /// No information: a label is not big, or the pair co-occurs but was not selected.
    Unknown,
    /// Both labels are big and no point carries both.
    Empty,
    /// The joint posting: `points[start..end]`.
    Stored(u64, u64),
}

/// Combination postings of selected label pairs (see the module documentation).
#[derive(Debug, Clone, PartialEq)]
pub struct PairIndex {
    n_points: usize,
    dict_hash: u64,
    /// Size threshold of the big labels.
    t_big: u64,
    /// Selection threshold on the prior's key (pairs at or above it are stored).
    ratio: f64,
    /// Big index of every label (`NONE` for the others); increasing with the label id.
    big: Vec<u32>,
    n_big: usize,
    /// Words per matrix row.
    words: usize,
    /// Co-occurrence of big labels `(r, c)`, `r < c` (bits of row `r`; the other half is unused).
    cooc: Vec<u64>,
    /// Stored pairs `(r, c)`, `r < c`, by row: partners `c` (increasing) of row `r` are
    /// `partners[row_offsets[r]..row_offsets[r + 1]]`; the list of the pair at directory slot `s`
    /// is `points[list_offsets[s]..list_offsets[s + 1]]` (increasing point ids).
    row_offsets: Vec<u64>,
    partners: Vec<u32>,
    list_offsets: Vec<u64>,
    points: Vec<u32>,
}

/// Size statistics of a build (also written to the build report).
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct PairStats {
    pub budget_bytes: u64,
    /// The ranking of a budgeted build (`None`: an explicit selection, [`PairIndex::build_selected`]).
    pub prior: Option<Prior>,
    pub t_big: u64,
    pub n_big: usize,
    pub big_entries: u64,
    pub cooc_pairs: u64,
    pub stored_pairs: u64,
    pub stored_entries: u64,
    pub ratio: f64,
    pub matrix_bytes: u64,
    pub bytes: u64,
    pub postings_bytes: u64,
}

/// One candidate list of a [`Cover`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListRef {
    /// A stored joint posting: [`PairIndex`]'s `points[start..end]`.
    Pair(u64, u64),
    /// A label's own posting (a superset of its joint with the other clause's label).
    Label(u32),
}

/// The candidate lists of a clause pair `(i, j)` (positions in the clause order) of a query:
/// for every label pair of the two clauses its stored joint posting, nothing if it is empty,
/// and otherwise the smaller of the two postings (each label at most once). The union of the
/// lists holds every point satisfying both clauses; with `exact`, it is exactly that set.
#[derive(Debug, Clone, Default)]
pub struct Cover {
    pub valid: bool,
    pub i: usize,
    pub j: usize,
    pub lists: Vec<ListRef>,
    /// Sum of the list lengths (the candidates, with repeats of points in several lists).
    pub entries: u64,
    /// Every list is a stored joint posting (no label posting).
    pub exact: bool,
    /// Directory lookups made (all clause pairs examined).
    pub lookups: u64,
    scratch: Vec<ListRef>,
    seen: Vec<u32>,
}

impl Cover {
    pub fn clear(&mut self) {
        self.valid = false;
        self.lists.clear();
        self.entries = 0;
        self.exact = false;
        self.lookups = 0;
    }

    /// Number of stored joint postings among the lists.
    pub fn n_pair_lists(&self) -> usize {
        self.lists
            .iter()
            .filter(|l| matches!(l, ListRef::Pair(..)))
            .count()
    }
}

/// An exact cover of a query's clause pair `(i, j)` by stored pairs that have a router tree
/// (`PairIndex::tree_cover`): every label pair of the two clauses is empty or such a stored
/// pair, so the trees' members are exactly the points satisfying both clauses.
#[derive(Debug, Clone, Default)]
pub struct TreeCover {
    pub valid: bool,
    pub i: usize,
    pub j: usize,
    /// Directory slots of the stored pairs (their trees' roots are the starts).
    pub slots: Vec<usize>,
    /// Sum of their joint posting sizes.
    pub members: u64,
    /// Directory lookups made.
    pub lookups: u64,
    scratch: Vec<usize>,
}

/// Largest product of clause sizes examined per clause pair (bounds the planning work).
pub const MAX_LOOKUPS: usize = 64;

#[inline]
fn key_bin(prior: Prior, min: u64, gain: u64, cost: u64, n: u64) -> usize {
    if gain == 0 {
        return 0;
    }
    let k = prior.key(min, gain, cost, n);
    let b = ((k.log2() - prior.key_min_log2()) * BINS_PER_OCTAVE).floor();
    if b < 1.0 {
        0
    } else {
        (b as usize).min(N_BINS - 1)
    }
}

/// Lower edge of histogram bin `b` (`b ≥ 1`).
fn bin_floor(prior: Prior, b: usize) -> f64 {
    (prior.key_min_log2() + b as f64 / BINS_PER_OCTAVE).exp2()
}

/// Smallest step of the nested ladder, in per-mille of the label postings' bytes.
pub const LADDER_FIRST_PERMILLE: u64 = 50;

/// The nested chain of the budget fraction `f` (of the label postings' bytes): the steps of the
/// 1-2-5 ladder from [`LADDER_FIRST_PERMILLE`] (5 %, 10 %, 20 %, 50 %, 100 %, 200 %, …) below
/// `f`, then `f`. A budget's nested selection is that of its chain, whatever other budgets are
/// built.
pub fn nested_chain(f: f64) -> Vec<f64> {
    let mut chain = Vec::new();
    let mut decade = 1u64;
    'ladder: loop {
        for m in [1u64, 2, 5] {
            let permille = m * decade;
            if permille < LADDER_FIRST_PERMILLE {
                continue;
            }
            let step = permille as f64 / 1000.0;
            if step >= f || permille > 1 << 40 {
                break 'ladder;
            }
            chain.push(step);
        }
        decade *= 10;
    }
    chain.push(f);
    chain
}

/// Histogram levels: `⌊log2⌋` of the smaller label's size.
const N_LEVELS: usize = 64;

#[inline]
fn level(size: u64) -> usize {
    63 - size.max(1).leading_zeros() as usize
}

impl PairIndex {
    /// Selects and builds the combination postings of `labels` within `budget_bytes` under the
    /// `uniform` prior (with up to `threads` threads; the result does not depend on the thread
    /// count).
    pub fn build(labels: &LabelStore, budget_bytes: u64, threads: usize) -> (Self, PairStats) {
        Self::build_with(labels, budget_bytes, threads, Prior::Uniform)
    }

    /// [`Self::build`] under `prior` (a single selection: no smaller budget's pairs kept).
    pub fn build_with(
        labels: &LabelStore,
        budget_bytes: u64,
        threads: usize,
        prior: Prior,
    ) -> (Self, PairStats) {
        let mut out = Self::build_chain(labels, &[budget_bytes], threads, prior, &[true]);
        out.pop().flatten().expect("the only step is built")
    }

    /// Nested selections along the non-decreasing budgets `budgets` (bytes): step `k` (with its
    /// own `T`, which falls as the budget grows) stores every pair of step `k − 1` first and
    /// gives the rest of its budget to the best other pairs, so that the selections are
    /// monotone in the budget. One build at the last step's `T`: the first pass histograms the
    /// candidate pairs by key bin and by the level `⌊log2 min(|P_a|, |P_b|)⌋` (a pair is a
    /// candidate of step `k` iff its level is at least `log2 T_k`), every step's threshold is
    /// read from it in turn, and the second pass stores the pairs of the last step. Returns the
    /// index and statistics of every step `k` with `want[k]` (the earlier steps' indices are
    /// subsets of the last one: its big labels at their `T`, their stored pairs).
    pub fn build_chain(
        labels: &LabelStore,
        budgets: &[u64],
        threads: usize,
        prior: Prior,
        want: &[bool],
    ) -> Vec<Option<(Self, PairStats)>> {
        assert!(!budgets.is_empty() && want.len() == budgets.len());
        assert!(
            budgets.windows(2).all(|w| w[0] <= w[1]),
            "the budgets of a chain must not decrease"
        );
        let nl = labels.n_labels();
        let n = labels.n_points() as u64;
        let sizes: Vec<u64> = (0..nl as u32)
            .map(|l| labels.posting_len(l) as u64)
            .collect();
        // Size threshold of a budget: the smallest power of two ≥ T_MIN whose matrix fits a
        // quarter of it.
        let matrix_bytes = |nb: usize| (nb * nb.div_ceil(64) * 8) as u64;
        let n_big_at = |t: u64| sizes.iter().filter(|&&s| s >= t).count();
        let t_of = |budget: u64| {
            let mut t = T_MIN;
            while matrix_bytes(n_big_at(t)) > budget / 4 && t < (1 << 40) {
                t *= 2;
            }
            t
        };
        let ts: Vec<u64> = budgets.iter().map(|&b| t_of(b)).collect();
        let t = ts[ts.len() - 1];
        let mut big = vec![NONE; nl];
        let mut big_labels = Vec::new();
        for l in 0..nl {
            if sizes[l] >= t {
                big[l] = big_labels.len() as u32;
                big_labels.push(l as u32);
            }
        }
        let nb = big_labels.len();
        let words = nb.div_ceil(64);

        // Pass 1: co-occurrence rows and the histogram of the pairs' cost by (level, key bin).
        struct Scratch {
            count: Vec<u32>,
            touched: Vec<u32>,
            slot: Vec<u32>,
            hist: Vec<u64>,
            cells: Vec<u32>,
        }
        let init = || Scratch {
            count: vec![0; nb],
            touched: Vec::new(),
            slot: vec![NONE; nb],
            hist: Vec::new(),
            cells: Vec::new(),
        };
        let count_row = |r: usize, s: &mut Scratch| {
            let a = big_labels[r];
            s.touched.clear();
            for &p in labels.postings(a) {
                let row = labels.row(p);
                let from = row.partition_point(|&x| x <= a);
                for &l in &row[from..] {
                    let c = big[l as usize];
                    if c != NONE {
                        let k = &mut s.count[c as usize];
                        if *k == 0 {
                            s.touched.push(c);
                        }
                        *k += 1;
                    }
                }
            }
            s.touched.sort_unstable();
        };
        let size_of = |c: u32| sizes[big_labels[c as usize] as usize];
        // Histogram cell of a pair: its level and key bin.
        let cell = |sa: u64, c: u32, j: u64| {
            let min = sa.min(size_of(c));
            level(min) * N_BINS + key_bin(prior, min, min - j, j + PAIR_OVERHEAD, n)
        };
        let pass1 = parallel::par_map(nb, threads, init, |r, s| {
            count_row(r, s);
            if s.hist.is_empty() {
                s.hist = vec![0; N_LEVELS * N_BINS];
            }
            let sa = size_of(r as u32);
            let mut bits = vec![0u64; words];
            s.cells.clear();
            for &c in &s.touched {
                let j = u64::from(s.count[c as usize]);
                s.count[c as usize] = 0;
                bits[c as usize / 64] |= 1u64 << (c % 64);
                let x = cell(sa, c, j);
                if s.hist[x] == 0 {
                    s.cells.push(x as u32);
                }
                s.hist[x] += j + PAIR_OVERHEAD;
            }
            let hist: Vec<(u32, u64)> = s
                .cells
                .iter()
                .map(|&x| (x, std::mem::take(&mut s.hist[x as usize])))
                .collect();
            (bits, hist, s.touched.len() as u64)
        });
        let mut cooc = vec![0u64; nb * words];
        let mut hist = vec![0u64; N_LEVELS * N_BINS];
        let mut cooc_pairs = 0u64;
        for (r, (bits, h, n)) in pass1.into_iter().enumerate() {
            cooc[r * words..(r + 1) * words].copy_from_slice(&bits);
            for (x, cost) in h {
                hist[x as usize] += cost;
            }
            cooc_pairs += n;
        }

        // The steps' thresholds: step k's candidates are the cells at levels ≥ log2 T_k; the
        // cells of earlier steps are kept (forced), the others are taken by whole key bins from
        // the highest down while the budget lasts. `first[cell]`: the first step storing it.
        let mut first = vec![0u8; N_LEVELS * N_BINS];
        let mut ratios = Vec::with_capacity(budgets.len());
        for (k, (&budget, &tk)) in budgets.iter().zip(&ts).enumerate() {
            let nbk = n_big_at(tk);
            let fixed = 4 * nl as u64 + matrix_bytes(nbk) + 8 * (nbk as u64 + 2);
            let avail = budget.saturating_sub(fixed) / 4;
            let mut free = vec![0u64; N_BINS];
            let mut used = 0u64;
            for lv in level(tk)..N_LEVELS {
                let cells = lv * N_BINS..(lv + 1) * N_BINS;
                for ((f, &h), &s) in free.iter_mut().zip(&hist[cells.clone()]).zip(&first[cells]) {
                    if s != 0 {
                        used += h;
                    } else {
                        *f += h;
                    }
                }
            }
            let mut cut = N_BINS;
            for b in (1..N_BINS).rev() {
                if used + free[b] > avail {
                    break;
                }
                used += free[b];
                cut = b;
            }
            for lv in level(tk)..N_LEVELS {
                for b in cut..N_BINS {
                    let x = lv * N_BINS + b;
                    if first[x] == 0 {
                        first[x] = u8::try_from(k + 1).expect("at most 255 steps");
                    }
                }
            }
            ratios.push(if cut >= N_BINS {
                f64::INFINITY
            } else {
                bin_floor(prior, cut)
            });
        }

        // Pass 2: the lists of the last step's pairs (with the first step storing each).
        let any = first.iter().any(|&f| f != 0);
        let pass2 = parallel::par_map(nb, threads, init, |r, s| {
            let mut partners = Vec::new();
            let mut steps = Vec::new();
            let mut lens = Vec::new();
            let mut points = Vec::new();
            if !any {
                return (partners, steps, lens, points);
            }
            count_row(r, s);
            let sa = size_of(r as u32);
            for &c in &s.touched {
                let j = u64::from(s.count[c as usize]);
                s.count[c as usize] = 0;
                let f = first[cell(sa, c, j)];
                if f != 0 {
                    s.slot[c as usize] = partners.len() as u32;
                    partners.push(c);
                    steps.push(f);
                }
            }
            if partners.is_empty() {
                return (partners, steps, lens, points);
            }
            let mut lists: Vec<Vec<u32>> = vec![Vec::new(); partners.len()];
            let a = big_labels[r];
            for &p in labels.postings(a) {
                let row = labels.row(p);
                let from = row.partition_point(|&x| x <= a);
                for &l in &row[from..] {
                    let c = big[l as usize];
                    if c != NONE {
                        let k = s.slot[c as usize];
                        if k != NONE {
                            lists[k as usize].push(p);
                        }
                    }
                }
            }
            for (&c, list) in partners.iter().zip(&lists) {
                s.slot[c as usize] = NONE;
                lens.push(list.len() as u64);
                points.extend_from_slice(list);
            }
            (partners, steps, lens, points)
        });
        let mut row_offsets = Vec::with_capacity(nb + 1);
        row_offsets.push(0u64);
        let (mut partners, mut steps) = (Vec::new(), Vec::new());
        let (mut list_offsets, mut points) = (vec![0u64], Vec::new());
        for (ps, st, lens, pts) in pass2 {
            partners.extend_from_slice(&ps);
            steps.extend_from_slice(&st);
            row_offsets.push(partners.len() as u64);
            for len in lens {
                list_offsets.push(list_offsets.last().copied().unwrap_or(0) + len);
            }
            points.extend_from_slice(&pts);
        }
        let last = Self {
            n_points: labels.n_points(),
            dict_hash: labels.dict_hash(),
            t_big: t,
            ratio: ratios[ratios.len() - 1],
            big,
            n_big: nb,
            words,
            cooc,
            row_offsets,
            partners,
            list_offsets,
            points,
        };
        let stats_of = |ix: &Self, k: usize, cooc_pairs: u64| PairStats {
            budget_bytes: budgets[k],
            prior: Some(prior),
            t_big: ix.t_big,
            n_big: ix.n_big,
            big_entries: sizes.iter().filter(|&&s| s >= ix.t_big).sum(),
            cooc_pairs,
            stored_pairs: ix.partners.len() as u64,
            stored_entries: ix.points.len() as u64,
            ratio: ix.ratio,
            matrix_bytes: matrix_bytes(ix.n_big),
            bytes: ix.bytes(),
            postings_bytes: 4 * labels.n_entries() as u64,
        };
        let m = budgets.len() - 1;
        let mut out: Vec<Option<(Self, PairStats)>> = (0..m)
            .map(|k| {
                want[k].then(|| {
                    let (ix, cooc_pairs) = last.step(&sizes, ts[k], ratios[k], &steps, k + 1);
                    let st = stats_of(&ix, k, cooc_pairs);
                    (ix, st)
                })
            })
            .collect();
        out.push(want[m].then(|| {
            let st = stats_of(&last, m, cooc_pairs);
            (last, st)
        }));
        out
    }

    /// Combination postings of an explicit selection (the unified selector, [`crate::select`]):
    /// the big labels `big_labels` (increasing ids, each of at least `t` points), the
    /// co-occurrence matrix of their pairs in different categories (pairs of one category are
    /// marked co-occurring, so a lookup answers [`Joint::Unknown`] for them, never
    /// [`Joint::Empty`]) and the joint postings of the `selected` pairs (big labels in different
    /// categories). One pass over the big labels' postings, rows in parallel (the result does not
    /// depend on the thread count). `budget_bytes` and `ratio` (the selection's cut) are recorded
    /// in the statistics.
    pub fn build_selected(
        labels: &LabelStore,
        t: u64,
        big_labels: &[u32],
        selected: &[(u32, u32)],
        budget_bytes: u64,
        ratio: f64,
        threads: usize,
    ) -> Result<(Self, PairStats)> {
        ensure!(
            big_labels.windows(2).all(|w| w[0] < w[1]),
            "big labels must increase"
        );
        let nl = labels.n_labels();
        let mut big = vec![NONE; nl];
        for (i, &l) in big_labels.iter().enumerate() {
            ensure!(
                (l as usize) < nl && labels.posting_len(l) as u64 >= t,
                "label {l} is not a label of at least {t} points"
            );
            big[l as usize] = i as u32;
        }
        let nb = big_labels.len();
        let words = nb.div_ceil(64);
        let mut by_row: Vec<Vec<u32>> = vec![Vec::new(); nb];
        for &(a, b) in selected {
            let (x, y) = (big[a as usize], big[b as usize]);
            ensure!(
                x != NONE && y != NONE && labels.category(a) != labels.category(b),
                "selected pair ({a}, {b}) is not two big labels of different categories"
            );
            let (r, c) = if x < y { (x, y) } else { (y, x) };
            by_row[r as usize].push(c);
        }
        for v in &mut by_row {
            v.sort_unstable();
            v.dedup();
        }
        let rows = parallel::par_map(
            nb,
            threads,
            || vec![NONE; nb],
            |r, slot| {
                let a = big_labels[r];
                let cat = labels.category(a);
                let mut bits = vec![0u64; words];
                for c in r + 1..nb {
                    if labels.category(big_labels[c]) != cat {
                        break;
                    }
                    bits[c / 64] |= 1u64 << (c % 64);
                }
                let partners = &by_row[r];
                for (x, &c) in partners.iter().enumerate() {
                    slot[c as usize] = x as u32;
                }
                let mut lists: Vec<Vec<u32>> = vec![Vec::new(); partners.len()];
                let end = labels.category_labels(cat).end;
                for &p in labels.postings(a) {
                    let row = labels.row(p);
                    for &l in &row[row.partition_point(|&x| x < end)..] {
                        let c = big[l as usize];
                        if c != NONE {
                            bits[c as usize / 64] |= 1u64 << (c % 64);
                            let x = slot[c as usize];
                            if x != NONE {
                                lists[x as usize].push(p);
                            }
                        }
                    }
                }
                for &c in partners {
                    slot[c as usize] = NONE;
                }
                (bits, lists)
            },
        );
        let mut cooc = vec![0u64; nb * words];
        let mut row_offsets = Vec::with_capacity(nb + 1);
        row_offsets.push(0u64);
        let (mut partners, mut list_offsets, mut points) = (Vec::new(), vec![0u64], Vec::new());
        let mut cooc_pairs = 0u64;
        for (r, (bits, lists)) in rows.into_iter().enumerate() {
            cooc_pairs += bits.iter().map(|w| u64::from(w.count_ones())).sum::<u64>();
            cooc[r * words..(r + 1) * words].copy_from_slice(&bits);
            for (&c, list) in by_row[r].iter().zip(lists) {
                partners.push(c);
                points.extend_from_slice(&list);
                list_offsets.push(points.len() as u64);
            }
            row_offsets.push(partners.len() as u64);
        }
        let ix = Self {
            n_points: labels.n_points(),
            dict_hash: labels.dict_hash(),
            t_big: t,
            ratio,
            big,
            n_big: nb,
            words,
            cooc,
            row_offsets,
            partners,
            list_offsets,
            points,
        };
        let stats = PairStats {
            budget_bytes,
            prior: None,
            t_big: t,
            n_big: nb,
            big_entries: big_labels
                .iter()
                .map(|&l| labels.posting_len(l) as u64)
                .sum(),
            cooc_pairs,
            stored_pairs: ix.partners.len() as u64,
            stored_entries: ix.points.len() as u64,
            ratio,
            matrix_bytes: (nb * words * 8) as u64,
            bytes: ix.bytes(),
            postings_bytes: 4 * labels.n_entries() as u64,
        };
        Ok((ix, stats))
    }

    /// The index of an earlier step of a chain built at this index's `T` (see
    /// [`Self::build_chain`]): the labels of at least `t` members are big, the pairs whose
    /// first step (`first`, by stored pair) is at most `step` are stored. Also returns the
    /// number of co-occurring big pairs.
    fn step(&self, sizes: &[u64], t: u64, ratio: f64, first: &[u8], step: usize) -> (Self, u64) {
        let mut map = vec![NONE; self.n_big];
        let mut big = vec![NONE; self.big.len()];
        let mut nb = 0u32;
        for (l, &b) in self.big.iter().enumerate() {
            if b != NONE && sizes[l] >= t {
                map[b as usize] = nb;
                big[l] = nb;
                nb += 1;
            }
        }
        let nb = nb as usize;
        let words = nb.div_ceil(64);
        let mut cooc = vec![0u64; nb * words];
        let mut cooc_pairs = 0u64;
        let mut row_offsets = Vec::with_capacity(nb + 1);
        row_offsets.push(0u64);
        let (mut partners, mut list_offsets, mut points) = (Vec::new(), vec![0u64], Vec::new());
        for (r, &rk) in map.iter().enumerate() {
            if rk == NONE {
                continue;
            }
            let rk = rk as usize;
            for (w, &word) in self.cooc[r * self.words..(r + 1) * self.words]
                .iter()
                .enumerate()
            {
                let mut bits = word;
                while bits != 0 {
                    let c = map[w * 64 + bits.trailing_zeros() as usize];
                    bits &= bits - 1;
                    if c != NONE {
                        cooc[rk * words + c as usize / 64] |= 1u64 << (c % 64);
                        cooc_pairs += 1;
                    }
                }
            }
            for s in self.row_offsets[r] as usize..self.row_offsets[r + 1] as usize {
                if usize::from(first[s]) <= step {
                    let c = map[self.partners[s] as usize];
                    debug_assert!(c != NONE, "a stored pair's labels are big at its step");
                    partners.push(c);
                    points.extend_from_slice(
                        self.list(self.list_offsets[s], self.list_offsets[s + 1]),
                    );
                    list_offsets.push(points.len() as u64);
                }
            }
            row_offsets.push(partners.len() as u64);
        }
        let ix = Self {
            n_points: self.n_points,
            dict_hash: self.dict_hash,
            t_big: t,
            ratio,
            big,
            n_big: nb,
            words,
            cooc,
            row_offsets,
            partners,
            list_offsets,
            points,
        };
        (ix, cooc_pairs)
    }

    pub fn n_points(&self) -> usize {
        self.n_points
    }

    pub fn n_labels(&self) -> usize {
        self.big.len()
    }

    pub fn dict_hash(&self) -> u64 {
        self.dict_hash
    }

    pub fn t_big(&self) -> u64 {
        self.t_big
    }

    pub fn n_big(&self) -> usize {
        self.n_big
    }

    pub fn n_pairs(&self) -> usize {
        self.partners.len()
    }

    pub fn n_entries(&self) -> usize {
        self.points.len()
    }

    /// Bytes held (every array).
    pub fn bytes(&self) -> u64 {
        (4 * self.big.len()
            + 8 * self.cooc.len()
            + 8 * self.row_offsets.len()
            + 4 * self.partners.len()
            + 8 * self.list_offsets.len()
            + 4 * self.points.len()) as u64
    }

    /// Whether label `l` is big.
    #[inline]
    pub fn is_big(&self, l: u32) -> bool {
        self.big[l as usize] != NONE
    }

    /// Lookup of the label pair `(a, b)` (`a ≠ b`).
    #[inline]
    pub fn joint(&self, a: u32, b: u32) -> Joint {
        let (x, y) = (self.big[a as usize], self.big[b as usize]);
        if x == NONE || y == NONE || x == y {
            return Joint::Unknown;
        }
        let (r, c) = if x < y { (x, y) } else { (y, x) };
        let (r, c) = (r as usize, c as usize);
        if (self.cooc[r * self.words + c / 64] >> (c % 64)) & 1 == 0 {
            return Joint::Empty;
        }
        let (lo, hi) = (
            self.row_offsets[r] as usize,
            self.row_offsets[r + 1] as usize,
        );
        match self.partners[lo..hi].binary_search(&(c as u32)) {
            Ok(i) => {
                let s = lo + i;
                Joint::Stored(self.list_offsets[s], self.list_offsets[s + 1])
            }
            Err(_) => Joint::Unknown,
        }
    }

    /// The stored points `[start, end)` of a [`Joint::Stored`] answer.
    #[inline]
    pub fn list(&self, start: u64, end: u64) -> &[u32] {
        &self.points[start as usize..end as usize]
    }

    /// The directory slot of the stored pair whose list starts at `start` (stored lists are
    /// never empty, so the offsets increase strictly).
    #[inline]
    pub fn slot_of(&self, start: u64) -> usize {
        self.list_offsets.partition_point(|&o| o < start)
    }

    /// The stored points of directory slot `s`.
    #[inline]
    pub fn slot_list(&self, s: usize) -> &[u32] {
        self.list(self.list_offsets[s], self.list_offsets[s + 1])
    }

    /// The clause pair of a query with `n_clauses` clauses (`clause(i)`: its labels) whose
    /// label pairs are all empty or stored with a router tree (`has_tree(slot)`), with the
    /// fewest members, into `out`; labels with empty postings are skipped (`posting_len`);
    /// clause pairs with more than [`MAX_LOOKUPS`] label pairs are not examined. Returns
    /// whether such a clause pair exists.
    pub fn tree_cover<'c>(
        &self,
        n_clauses: usize,
        clause: impl Fn(usize) -> &'c [u32],
        posting_len: impl Fn(u32) -> usize,
        has_tree: impl Fn(usize) -> bool,
        out: &mut TreeCover,
    ) -> bool {
        out.valid = false;
        out.slots.clear();
        out.members = 0;
        out.lookups = 0;
        let mut best = u64::MAX;
        for i in 0..n_clauses {
            let ci = clause(i);
            for j in (i + 1)..n_clauses {
                let cj = clause(j);
                if ci.len() * cj.len() > MAX_LOOKUPS {
                    continue;
                }
                out.scratch.clear();
                let (mut sum, mut ok) = (0u64, true);
                'pairs: for &a in ci {
                    if posting_len(a) == 0 {
                        continue;
                    }
                    for &b in cj {
                        if posting_len(b) == 0 {
                            continue;
                        }
                        out.lookups += 1;
                        match self.joint(a, b) {
                            Joint::Empty => {}
                            Joint::Stored(s, e) => {
                                let slot = self.slot_of(s);
                                if !has_tree(slot) {
                                    ok = false;
                                    break 'pairs;
                                }
                                out.scratch.push(slot);
                                sum += e - s;
                            }
                            Joint::Unknown => {
                                ok = false;
                                break 'pairs;
                            }
                        }
                        if sum >= best {
                            ok = false;
                            break 'pairs;
                        }
                    }
                }
                if ok && sum < best {
                    best = sum;
                    out.valid = true;
                    out.i = i;
                    out.j = j;
                    out.members = sum;
                    std::mem::swap(&mut out.slots, &mut out.scratch);
                }
            }
        }
        out.valid
    }

    /// The clause pair with the fewest candidates (see [`Cover`]) of a query with `n_clauses`
    /// clauses (`clause(i)`: its labels) into `out`; labels with empty postings are skipped
    /// (`posting_len`). Clause pairs with more than [`MAX_LOOKUPS`] label pairs are not
    /// examined. Returns whether some clause pair was examined.
    pub fn cover<'c>(
        &self,
        n_clauses: usize,
        clause: impl Fn(usize) -> &'c [u32],
        posting_len: impl Fn(u32) -> usize,
        out: &mut Cover,
    ) -> bool {
        out.clear();
        let mut best = u64::MAX;
        for i in 0..n_clauses {
            let ci = clause(i);
            for j in (i + 1)..n_clauses {
                let cj = clause(j);
                if ci.len() * cj.len() > MAX_LOOKUPS {
                    continue;
                }
                out.scratch.clear();
                out.seen.clear();
                let (mut sum, mut exact, mut ok) = (0u64, true, true);
                'pairs: for &a in ci {
                    let la = posting_len(a);
                    if la == 0 {
                        continue;
                    }
                    for &b in cj {
                        let lb = posting_len(b);
                        if lb == 0 {
                            continue;
                        }
                        out.lookups += 1;
                        match self.joint(a, b) {
                            Joint::Empty => {}
                            Joint::Stored(s, e) => {
                                out.scratch.push(ListRef::Pair(s, e));
                                sum += e - s;
                            }
                            Joint::Unknown => {
                                exact = false;
                                let (x, lx) = if lb < la || (lb == la && b < a) {
                                    (b, lb)
                                } else {
                                    (a, la)
                                };
                                if !out.seen.contains(&x) {
                                    out.seen.push(x);
                                    out.scratch.push(ListRef::Label(x));
                                    sum += lx as u64;
                                }
                            }
                        }
                        if sum >= best {
                            ok = false;
                            break 'pairs;
                        }
                    }
                }
                if ok && sum < best {
                    best = sum;
                    out.valid = true;
                    out.i = i;
                    out.j = j;
                    out.entries = sum;
                    out.exact = exact;
                    std::mem::swap(&mut out.lists, &mut out.scratch);
                }
            }
        }
        out.valid
    }

    /// Checks that the index belongs to `labels` (dictionary and counts) and, on every
    /// `step`-th stored pair, that its list is the intersection of the two postings.
    pub fn check_sample(&self, labels: &LabelStore, step: usize) -> Result<()> {
        ensure!(
            self.dict_hash == labels.dict_hash()
                && self.big.len() == labels.n_labels()
                && self.n_points == labels.n_points(),
            "the combination postings refer to another label store"
        );
        let mut by_big = vec![0u32; self.n_big];
        for (l, &b) in self.big.iter().enumerate() {
            if b != NONE {
                by_big[b as usize] = l as u32;
                ensure!(
                    labels.posting_len(l as u32) as u64 >= self.t_big,
                    "label {l} is not big"
                );
            }
        }
        for r in 0..self.n_big {
            let (lo, hi) = (
                self.row_offsets[r] as usize,
                self.row_offsets[r + 1] as usize,
            );
            for s in (lo..hi).step_by(step.max(1)) {
                let (a, b) = (by_big[r], by_big[self.partners[s] as usize]);
                let (pa, pb) = (labels.postings(a), labels.postings(b));
                let mut want = Vec::new();
                let (mut x, mut y) = (0, 0);
                while x < pa.len() && y < pb.len() {
                    match pa[x].cmp(&pb[y]) {
                        std::cmp::Ordering::Less => x += 1,
                        std::cmp::Ordering::Greater => y += 1,
                        std::cmp::Ordering::Equal => {
                            want.push(pa[x]);
                            x += 1;
                            y += 1;
                        }
                    }
                }
                ensure!(
                    self.list(self.list_offsets[s], self.list_offsets[s + 1]) == &want[..],
                    "the list of labels ({a}, {b}) differs from their intersection"
                );
            }
        }
        Ok(())
    }

    /// Writes `pairs.flab`.
    pub fn write(&self, path: &Path) -> Result<()> {
        self.write_to(container::create(path)?)
            .with_context(|| format!("writing {}", path.display()))
    }

    pub fn write_to<W: Write>(&self, w: W) -> Result<()> {
        let params = [
            self.n_points as u64,
            self.big.len() as u64,
            self.dict_hash,
            self.t_big,
            self.n_big as u64,
            self.partners.len() as u64,
            self.points.len() as u64,
            self.ratio.to_bits(),
        ];
        container::write(
            w,
            MAGIC,
            VERSION,
            &params,
            &[
                bytemuck::cast_slice(&self.big),
                bytemuck::cast_slice(&self.cooc),
                bytemuck::cast_slice(&self.row_offsets),
                bytemuck::cast_slice(&self.partners),
                bytemuck::cast_slice(&self.list_offsets),
                bytemuck::cast_slice(&self.points),
            ],
        )
    }

    /// Reads and validates `pairs.flab` (structure only; [`Self::check_sample`] compares it
    /// with a label store).
    pub fn read(path: &Path) -> Result<Self> {
        container::Reader::open(path, MAGIC)
            .and_then(Self::load)
            .with_context(|| format!("loading combination postings from {}", path.display()))
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
        let [n_points, n_labels, dict_hash, t_big, n_big, n_pairs, n_entries, ratio] = *r.params();
        let (nl, nb) = (n_labels as usize, n_big as usize);
        let words = nb.div_ceil(64);
        let big: Vec<u32> = r.section(0, nl)?;
        let cooc: Vec<u64> = r.section(1, nb * words)?;
        let row_offsets: Vec<u64> = r.section(2, nb + 1)?;
        let partners: Vec<u32> = r.section(3, n_pairs as usize)?;
        let list_offsets: Vec<u64> = r.section(4, n_pairs as usize + 1)?;
        let points: Vec<u32> = r.section(5, n_entries as usize)?;
        let mut next = 0u32;
        for &b in &big {
            if b != NONE {
                ensure!(b == next, "big indices must increase with the label id");
                next += 1;
            }
        }
        ensure!(next as usize == nb, "inconsistent big-label count");
        ensure!(
            row_offsets.first() == Some(&0)
                && row_offsets.last() == Some(&n_pairs)
                && row_offsets.windows(2).all(|w| w[0] <= w[1]),
            "inconsistent row offsets"
        );
        for (rw, w) in row_offsets.windows(2).enumerate() {
            let ps = &partners[w[0] as usize..w[1] as usize];
            ensure!(
                ps.windows(2).all(|x| x[0] < x[1])
                    && ps.iter().all(|&c| (c as usize) > rw && (c as usize) < nb),
                "partners must increase above their row"
            );
        }
        ensure!(
            list_offsets.first() == Some(&0)
                && list_offsets.last() == Some(&n_entries)
                && list_offsets.windows(2).all(|w| w[0] <= w[1]),
            "inconsistent list offsets"
        );
        for w in list_offsets.windows(2) {
            let list = &points[w[0] as usize..w[1] as usize];
            ensure!(
                list.windows(2).all(|x| x[0] < x[1])
                    && list.iter().all(|&p| u64::from(p) < n_points),
                "lists must be increasing data points"
            );
        }
        Ok(Self {
            n_points: n_points as usize,
            dict_hash,
            t_big,
            ratio: f64::from_bits(ratio),
            big,
            n_big: nb,
            words,
            cooc,
            row_offsets,
            partners,
            list_offsets,
            points,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::labels::LabelStoreBuilder;

    /// 400 points; labels: `x` on all, `a` on 0..200, `b` on 150..350, `c` on 0..100 and
    /// 300..400, `d` on 390..400 (small), `e` on the even points below 128 (inside `a`).
    fn store() -> LabelStore {
        let mut b = LabelStoreBuilder::new();
        for p in 0..400u32 {
            let mut v = vec![("k", "x")];
            if p < 200 {
                v.push(("k", "a"));
            }
            if (150..350).contains(&p) {
                v.push(("k", "b"));
            }
            if !(100..300).contains(&p) {
                v.push(("k", "c"));
            }
            if p >= 390 {
                v.push(("k", "d"));
            }
            if p % 2 == 0 && p < 128 {
                v.push(("k", "e"));
            }
            b.push_point(v).unwrap();
        }
        b.finish().unwrap()
    }

    fn intersect(s: &LabelStore, a: u32, b: u32) -> Vec<u32> {
        let pb = s.postings(b);
        s.postings(a)
            .iter()
            .copied()
            .filter(|p| pb.binary_search(p).is_ok())
            .collect()
    }

    #[test]
    fn the_data_prior_prefers_the_diluted_conjunctions_points_carry() {
        // A tiny joint of a small pair (j = 1, min 100) against a diluted joint of larger labels
        // (j = 50, min 1,000): the uniform prior ranks the tiny one first, the data prior the
        // conjunction that 50 points carry.
        let n = 1_000_000;
        let key = |p: Prior, min: u64, j: u64| p.key(min, min - j, j + PAIR_OVERHEAD, n);
        assert!(key(Prior::Uniform, 100, 1) > key(Prior::Uniform, 1000, 50));
        assert!(key(Prior::Data, 100, 1) < key(Prior::Data, 1000, 50));
        assert_eq!("data".parse::<Prior>().unwrap(), Prior::Data);
    }

    #[test]
    fn lists_are_intersections_and_lookups_classify() {
        let s = store();
        let id = s.name_index();
        let (a, b, c, d, e, x) = (id["a"], id["b"], id["c"], id["d"], id["e"], id["x"]);
        let (ix, st) = PairIndex::build(&s, 1 << 20, 2);
        assert_eq!(st.t_big, 64);
        assert!(ix.is_big(a) && ix.is_big(e) && !ix.is_big(d));
        ix.check_sample(&s, 1).unwrap();
        // Every stored pair is an intersection; c and b co-occur on 300..350 (j = 50).
        for (p, q) in [(a, b), (a, c), (b, c), (a, e), (c, e), (b, x)] {
            match ix.joint(p, q) {
                Joint::Stored(lo, hi) => assert_eq!(ix.list(lo, hi), &intersect(&s, p, q)[..]),
                Joint::Unknown => assert!(!intersect(&s, p, q).is_empty()),
                Joint::Empty => panic!("({p}, {q}) co-occur"),
            }
        }
        // e (0..128 even) and b (150..350) never meet: both big, empty.
        assert_eq!(ix.joint(e, b), Joint::Empty);
        assert_eq!(ix.joint(b, e), Joint::Empty);
        // A small label is unknown; a containment (e ⊂ a) has no gain and is not stored.
        assert_eq!(ix.joint(a, d), Joint::Unknown);
        assert_eq!(ix.joint(a, e), Joint::Unknown);
        // Read back identically.
        let mut buf = Vec::new();
        ix.write_to(&mut buf).unwrap();
        assert_eq!(PairIndex::read_from(std::io::Cursor::new(buf)).unwrap(), ix);
    }

    #[test]
    fn the_rare_prior_prefers_the_pair_of_larger_labels() {
        // 2,000 points: s1 = 0..200 and s2 = 199..400 meet once (gain per entry 199/4); L1 =
        // 0..1000 and L2 = 960..2000 meet on 40 points (960/43, but ×10 more frequent).
        let mut b = LabelStoreBuilder::new();
        for p in 0..2000u32 {
            let mut v = vec![];
            if p < 200 {
                v.push(("k", "s1"));
            }
            if (199..400).contains(&p) {
                v.push(("k", "s2"));
            }
            if p < 1000 {
                v.push(("k", "L1"));
            }
            if p >= 960 {
                v.push(("k", "L2"));
            }
            b.push_point(v).unwrap();
        }
        let s = b.finish().unwrap();
        let id = s.name_index();
        let (s1, s2, l1, l2) = (id["s1"], id["s2"], id["L1"], id["L2"]);
        // Room for one of the two lists (4 + 43 entries do not fit).
        let budget = 4 * 4 + 32 + 8 * 6 + 4 * 45;
        let (u, su) = PairIndex::build_with(&s, budget, 1, Prior::Uniform);
        let (r, sr) = PairIndex::build_with(&s, budget, 1, Prior::Rare);
        assert_eq!((su.t_big, sr.t_big), (64, 64));
        assert!(matches!(u.joint(s1, s2), Joint::Stored(..)), "{su:?}");
        assert_eq!(u.joint(l1, l2), Joint::Unknown);
        assert!(matches!(r.joint(l1, l2), Joint::Stored(..)), "{sr:?}");
        assert_eq!(r.joint(s1, s2), Joint::Unknown);
        // Containments (s1 ⊂ L1) have no gain under either prior.
        assert_eq!(r.joint(s1, l1), Joint::Unknown);
        r.check_sample(&s, 1).unwrap();
    }

    /// Stored label pairs of `ix` among `labels`.
    fn stored(ix: &PairIndex, labels: &[u32]) -> Vec<(u32, u32)> {
        let mut v = Vec::new();
        for &p in labels {
            for &q in labels {
                if p < q && matches!(ix.joint(p, q), Joint::Stored(..)) {
                    v.push((p, q));
                }
            }
        }
        v
    }

    #[test]
    fn nested_chains_are_monotone_and_their_steps_match_shorter_chains() {
        // 3,000 points, 150 labels of varied sizes (pseudo-random members), so that `T` falls
        // along the chain and the steps store different pairs.
        let mut rng = 12345u64;
        let mut next = move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 33) as u32
        };
        let n = 3000u32;
        let names: Vec<String> = (0..150).map(|l| format!("l{l}")).collect();
        let mut rows: Vec<Vec<usize>> = vec![Vec::new(); n as usize];
        for l in 0..150usize {
            let size = 40 + (next() % 1500) * (next() % 2 + 1) / ((l as u32 % 7) + 1);
            let start = next() % n;
            for i in 0..size {
                // Clustered members: overlaps between labels with nearby starts.
                let p = (start + i * (1 + l as u32 % 3)) % n;
                rows[p as usize].push(l);
            }
        }
        let mut b = LabelStoreBuilder::new();
        for row in &rows {
            let mut v: Vec<(&str, &str)> = row.iter().map(|&l| ("k", names[l].as_str())).collect();
            v.dedup();
            b.push_point(v).unwrap();
        }
        let s = b.finish().unwrap();
        let labels: Vec<u32> = (0..s.n_labels() as u32).collect();
        let postings = 4 * s.n_entries() as u64;
        let budgets: Vec<u64> = [0.01, 0.03, 0.1, 0.3]
            .iter()
            .map(|f| (f * postings as f64) as u64)
            .collect();
        for prior in [Prior::Uniform, Prior::Rare] {
            let all = PairIndex::build_chain(&s, &budgets, 2, prior, &[true; 4]);
            let mut ts = Vec::new();
            let mut last: Vec<(u32, u32)> = Vec::new();
            for (k, step) in all.iter().enumerate() {
                let (ix, st) = step.as_ref().unwrap();
                ix.check_sample(&s, 1).unwrap();
                assert!(st.bytes <= st.budget_bytes, "{st:?}");
                // A step equals the last step of the shorter chain (built at its own `T`).
                let alone =
                    PairIndex::build_chain(&s, &budgets[..=k], 1, prior, &vec![true; k + 1]);
                let (ix1, st1) = alone[k].as_ref().unwrap();
                assert_eq!(ix, ix1, "step {k} ({prior:?})");
                assert_eq!(st.cooc_pairs, st1.cooc_pairs);
                let now = stored(ix, &labels);
                assert!(last.iter().all(|p| now.contains(p)), "step {k} drops pairs");
                ts.push(st.t_big);
                last = now;
            }
            assert!(ts[0] > ts[3], "T falls along the chain: {ts:?}");
            assert!(!last.is_empty());
            // The first step is a plain selection.
            let (plain, _) = PairIndex::build_with(&s, budgets[0], 3, prior);
            assert_eq!(&plain, &all[0].as_ref().unwrap().0);
        }
        // The ladder: steps below the budget, then the budget.
        assert_eq!(nested_chain(0.2), vec![0.05, 0.1, 0.2]);
        assert_eq!(nested_chain(0.5), vec![0.05, 0.1, 0.2, 0.5]);
        assert_eq!(nested_chain(0.3), vec![0.05, 0.1, 0.2, 0.3]);
        assert_eq!(nested_chain(0.05), vec![0.05]);
        assert_eq!(nested_chain(0.01), vec![0.01]);
        assert_eq!(nested_chain(2.5), vec![0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 2.5]);
    }

    #[test]
    fn budget_limits_the_lists_and_cover_finds_the_cheapest_pair() {
        let s = store();
        let id = s.name_index();
        let (a, b, c, d, e) = (id["a"], id["b"], id["c"], id["d"], id["e"]);
        let (none, st0) = PairIndex::build(&s, 64, 1);
        assert_eq!(st0.stored_pairs, 0, "a tiny budget stores nothing");
        assert_eq!(none.joint(a, b), Joint::Unknown);
        let (ix, st) = PairIndex::build(&s, 1 << 20, 1);
        assert!(st.stored_pairs >= 2, "{st:?}");
        assert!(st.bytes <= st.budget_bytes);
        // Query (a | e) & (b): pairs (a, b) stored, (e, b) empty -> exact, the list of (a, b).
        let clauses: Vec<Vec<u32>> = vec![vec![a.min(e), a.max(e)], vec![b]];
        let mut cov = Cover::default();
        assert!(ix.cover(2, |i| &clauses[i][..], |l| s.posting_len(l), &mut cov));
        assert_eq!((cov.i, cov.j, cov.exact), (0, 1, true));
        assert_eq!(cov.entries as usize, intersect(&s, a, b).len());
        // With a third clause (c), the cheapest pair wins.
        let clauses: Vec<Vec<u32>> = vec![vec![a], vec![b], vec![c]];
        assert!(ix.cover(3, |i| &clauses[i][..], |l| s.posting_len(l), &mut cov));
        let sizes = [
            intersect(&s, a, b).len(),
            intersect(&s, a, c).len(),
            intersect(&s, b, c).len(),
        ];
        assert_eq!(cov.entries as usize, *sizes.iter().min().unwrap());
        // A small label (d) is not big: its own posting is the candidate list (not exact).
        let clauses: Vec<Vec<u32>> = vec![vec![b], vec![d]];
        assert!(ix.cover(2, |i| &clauses[i][..], |l| s.posting_len(l), &mut cov));
        assert!(!cov.exact);
        assert_eq!(cov.lists, vec![ListRef::Label(d)]);
        assert_eq!(cov.entries as usize, s.posting_len(d));
        assert_eq!(cov.n_pair_lists(), 0);
    }
}
