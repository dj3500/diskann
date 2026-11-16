/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Label-aware candidate generation for the builder: which searches produce an inserted
//! point's candidate pool.
//!
//! - [`Candidates::Blind`]: one label-blind greedy search from the frozen start (Vamana).
//! - [`Candidates::Filtered`]: the label-blind search plus, for each label `ℓ` of a subset
//!   `S(p)` of the point's labels, a greedy search that only admits nodes carrying `ℓ`,
//!   started from `ℓ`'s start node (the label's medoid). This is FilteredVamana's filtered
//!   insertion generalized to label sets in several categories: one search per label instead
//!   of one search over the union of the point's labels (whose most frequent label would fill
//!   the list). `S(p)` = the point's labels with at most `θ` points (all labels without a cap),
//!   the rarest first, at most `budget` of them (default: the degree bound `R`, since a point
//!   keeps at most `R` out-neighbors). The pool is the union of every search's expanded nodes.
//! - [`Candidates::Penalized`]: one greedy search from the frozen start whose list is ordered
//!   by `d²(p, c) + λ·f(p, c)`, where `f(p, c)` is the number of categories in which `p` has
//!   labels and `c` shares none of them (the point's own label set read as an AND-of-ORs
//!   predicate, `f` = its failed clauses). `λ` is given in units of a base statistic, the
//!   median squared distance from a point to its nearest non-identical neighbor (estimated on
//!   a seeded sample), so that it is scale-free; `λ = ∞` orders by `f` first (lexicographic).
//!   The pool is the expanded nodes with their squared distance ([`PruneKey::Raw`]) or with the
//!   penalized key ([`PruneKey::Penalized`]: the pruning rule then sees `d² + λ·f` as the
//!   distance from `p`, also in back-edge re-prunes).
//!
//! Every choice depends only on the base vectors, the labels and the parameters (no queries).

use std::{fmt, str::FromStr};

use anyhow::{bail, ensure, Context, Result};
use rayon::prelude::*;
use serde_json::{json, Value};

use super::{prune::Candidate, BuildElement, SplitMix64};
use crate::{label_starts, labels::LabelStore, vectors::Matrix};

/// Which labels of a point get a filtered insertion search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelSubset {
    /// Every label.
    All,
    /// Labels with at most `θ` points (`Cap(θ)`), or at most `n / k` points (`Fraction(k)`).
    Cap(usize),
    Fraction(usize),
}

impl LabelSubset {
    /// The cap in points for a base of `n` points (`None` without a cap).
    pub fn theta(self, n: usize) -> Option<usize> {
        match self {
            Self::All => None,
            Self::Cap(t) => Some(t),
            Self::Fraction(k) => Some(n / k),
        }
    }
}

impl fmt::Display for LabelSubset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::All => write!(f, "all"),
            Self::Cap(t) => write!(f, "{t}"),
            Self::Fraction(k) => write!(f, "n/{k}"),
        }
    }
}

impl FromStr for LabelSubset {
    type Err = anyhow::Error;

    /// `all`, `<θ>` or `n/<k>`.
    fn from_str(s: &str) -> Result<Self> {
        let s = s.trim();
        if s == "all" {
            return Ok(Self::All);
        }
        if let Some(k) = s.strip_prefix("n/") {
            let k: usize = k
                .trim()
                .parse()
                .with_context(|| format!("label subset {s:?}: n/<k> needs an integer k"))?;
            ensure!(k >= 1, "label subset {s:?}: k must be at least 1");
            return Ok(Self::Fraction(k));
        }
        let t: usize = s
            .parse()
            .with_context(|| format!("label subset {s:?}: expected all, <θ> or n/<k>"))?;
        Ok(Self::Cap(t))
    }
}

/// The weight of a failed category in the penalized insertion search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Lambda {
    /// In units of the median squared nearest-neighbor distance.
    Relative(f32),
    /// Lexicographic: fewer failed categories first, then distance.
    Infinite,
}

impl fmt::Display for Lambda {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Relative(x) => write!(f, "{x}"),
            Self::Infinite => write!(f, "inf"),
        }
    }
}

impl FromStr for Lambda {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let s = s.trim();
        if s == "inf" {
            return Ok(Self::Infinite);
        }
        let x: f32 = s
            .parse()
            .with_context(|| format!("λ {s:?}: expected a number or inf"))?;
        ensure!(x.is_finite() && x >= 0.0, "λ must be ≥ 0 (got {x})");
        Ok(Self::Relative(x))
    }
}

/// What the pruning rule sees as the distance from the inserted point in penalized builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruneKey {
    /// The squared distance (the penalty only steers the search).
    Raw,
    /// The penalized key `d² + λ·f` (pool order and occlusion tests; back-edge re-prunes too).
    Penalized,
}

impl fmt::Display for PruneKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Raw => "raw",
            Self::Penalized => "penalized",
        })
    }
}

impl FromStr for PruneKey {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim() {
            "raw" => Ok(Self::Raw),
            "penalized" | "pen" => Ok(Self::Penalized),
            other => bail!("prune key {other:?}: expected raw or penalized"),
        }
    }
}

/// Candidate generation of a build (see the module documentation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Candidates {
    Blind,
    Filtered {
        subset: LabelSubset,
        /// Most filtered searches per point (`None`: the degree bound `R`).
        budget: Option<usize>,
        /// List size of the filtered searches (`None`: `L_build`).
        l_filter: Option<usize>,
    },
    Penalized {
        lambda: Lambda,
        key: PruneKey,
    },
}

impl Candidates {
    pub fn needs_labels(&self) -> bool {
        !matches!(self, Self::Blind)
    }

    /// A short name for file names, e.g. `filtered-n100`, `pen-1-raw`.
    pub fn tag(&self) -> String {
        match *self {
            Self::Blind => "blind".into(),
            Self::Filtered {
                subset,
                budget,
                l_filter,
            } => {
                let mut t = format!("filtered-{}", subset.to_string().replace('/', ""));
                if let Some(b) = budget {
                    t += &format!("-b{b}");
                }
                if let Some(l) = l_filter {
                    t += &format!("-l{l}");
                }
                t
            }
            Self::Penalized { lambda, key } => format!("pen-{lambda}-{key}"),
        }
    }

    pub fn to_json(&self) -> Value {
        match *self {
            Self::Blind => json!({"kind": "blind"}),
            Self::Filtered {
                subset,
                budget,
                l_filter,
            } => json!({
                "kind": "filtered",
                "subset": subset.to_string(),
                "budget": budget,
                "l_filter": l_filter,
            }),
            Self::Penalized { lambda, key } => json!({
                "kind": "penalized",
                "lambda": lambda.to_string(),
                "prune_key": key.to_string(),
            }),
        }
    }
}

impl fmt::Display for Candidates {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Blind => write!(f, "blind"),
            Self::Filtered {
                subset,
                budget,
                l_filter,
            } => {
                write!(f, "filtered({subset}")?;
                match (budget, l_filter) {
                    (None, None) => {}
                    (Some(b), None) => write!(f, ", {b}")?,
                    (b, Some(l)) => {
                        write!(f, ", {}, {l}", b.map_or("R".into(), |b| b.to_string()))?
                    }
                }
                write!(f, ")")
            }
            Self::Penalized { lambda, key } => write!(f, "penalized({lambda}, {key})"),
        }
    }
}

impl FromStr for Candidates {
    type Err = anyhow::Error;

    /// `blind`, `filtered(<subset>[, <budget>|R[, <L_f>]])` with subset `all`, `<θ>` or
    /// `n/<k>`, `penalized(<λ>|inf[, raw|penalized])`.
    fn from_str(s: &str) -> Result<Self> {
        let s = s.trim();
        if s == "blind" {
            return Ok(Self::Blind);
        }
        let (name, rest) = s.split_once('(').with_context(|| {
            format!("candidates {s:?}: expected blind, filtered(...) or penalized(...)")
        })?;
        let args = rest
            .strip_suffix(')')
            .with_context(|| format!("candidates {s:?}: missing ')'"))?;
        let args: Vec<&str> = args.split(',').map(str::trim).collect();
        match name.trim() {
            "filtered" => {
                ensure!(
                    (1..=3).contains(&args.len()),
                    "candidates {s:?}: filtered(<subset>[, <budget>[, <L_f>]])"
                );
                let subset: LabelSubset = args[0].parse()?;
                let budget = match args.get(1) {
                    None | Some(&"R") => None,
                    Some(b) => Some(
                        b.parse::<usize>()
                            .with_context(|| format!("candidates {s:?}: bad budget {b:?}"))?,
                    ),
                };
                let l_filter = match args.get(2) {
                    None => None,
                    Some(l) => Some(
                        l.parse::<usize>()
                            .ok()
                            .filter(|&l| l >= 1)
                            .with_context(|| format!("candidates {s:?}: bad list size {l:?}"))?,
                    ),
                };
                Ok(Self::Filtered {
                    subset,
                    budget,
                    l_filter,
                })
            }
            "penalized" => {
                ensure!(
                    (1..=2).contains(&args.len()),
                    "candidates {s:?}: penalized(<λ>[, raw|penalized])"
                );
                Ok(Self::Penalized {
                    lambda: args[0].parse()?,
                    key: match args.get(1) {
                        None => PruneKey::Raw,
                        Some(k) => k.parse()?,
                    },
                })
            }
            other => bail!("unknown candidate generation {other:?} (blind, filtered, penalized)"),
        }
    }
}

// ------------------------------------------------------------------ filtered insertion

/// Shared, read-only state of filtered insertion: every point's filtered labels `S(p)`, every
/// label's start node, and static member bitsets of the frequent labels searched.
#[derive(Debug)]
pub(crate) struct FilterPlan {
    /// `S(p)` as CSR: labels of point `p` searched at its insertion, rarest first.
    offsets: Vec<u64>,
    labels: Vec<u32>,
    /// Start node (a member) of every label (`u32::MAX` for labels without members).
    pub(crate) starts: Vec<u32>,
    /// Static member bitsets of labels with more than `n / 64` points that occur in some
    /// `S(p)`; other labels mark their members in a per-thread bitset per search.
    static_bits: Vec<Option<Box<[u64]>>>,
    pub(crate) l_filter: usize,
    pub(crate) stats: Value,
}

impl FilterPlan {
    pub(crate) fn new<T: BuildElement>(
        base: &Matrix<T>,
        store: &LabelStore,
        subset: LabelSubset,
        budget: usize,
        l_filter: usize,
        threads: usize,
    ) -> Result<Self> {
        let n = store.n_points();
        ensure!(budget >= 1, "the filtered-search budget must be at least 1");
        let theta = subset.theta(n);
        let in_subset = |l: u32| theta.is_none_or(|t| store.posting_len(l) <= t);
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0u64);
        let mut labels = Vec::new();
        let mut row = Vec::new();
        let (mut truncated_points, mut truncated_labels, mut with_any) = (0u64, 0u64, 0u64);
        let mut used = vec![false; store.n_labels()];
        for p in 0..n as u32 {
            row.clear();
            row.extend(store.row(p).iter().copied().filter(|&l| in_subset(l)));
            row.sort_unstable_by_key(|&l| (store.posting_len(l), l));
            if row.len() > budget {
                truncated_points += 1;
                truncated_labels += (row.len() - budget) as u64;
                row.truncate(budget);
            }
            with_any += u64::from(!row.is_empty());
            for &l in &row {
                used[l as usize] = true;
            }
            labels.extend_from_slice(&row);
            offsets.push(labels.len() as u64);
        }
        let t0 = std::time::Instant::now();
        let medoids = label_starts::build(
            base,
            store,
            label_starts::Strategy::Medoid,
            &label_starts::Params {
                threads,
                ..label_starts::Params::default()
            },
        )?;
        let starts: Vec<u32> = medoids
            .starts
            .iter()
            .map(|s| s.first().copied().unwrap_or(u32::MAX))
            .collect();
        let starts_s = t0.elapsed().as_secs_f64();
        let words = n.div_ceil(64);
        let big = n / 64;
        let mut static_bits: Vec<Option<Box<[u64]>>> = vec![None; store.n_labels()];
        let mut n_static = 0usize;
        for l in 0..store.n_labels() as u32 {
            if used[l as usize] && store.posting_len(l) > big {
                let mut bits = vec![0u64; words].into_boxed_slice();
                for &v in store.postings(l) {
                    bits[v as usize / 64] |= 1 << (v % 64);
                }
                static_bits[l as usize] = Some(bits);
                n_static += 1;
            }
        }
        let searched = labels.len() as u64;
        let stats = json!({
            "theta": theta,
            "budget": budget,
            "l_filter": l_filter,
            "labels_in_subset": (0..store.n_labels() as u32).filter(|&l| in_subset(l)).count(),
            "labels_searched": used.iter().filter(|&&u| u).count(),
            "filtered_searches": searched,
            "filtered_searches_per_point": searched as f64 / n.max(1) as f64,
            "points_with_a_filtered_search": with_any,
            "points_over_budget": truncated_points,
            "label_slots_over_budget": truncated_labels,
            "static_bitsets": n_static,
            "static_bitset_bytes": n_static * words * 8,
            "label_starts": "medoid (flab starts build strategy medoid, seed 0)",
            "label_starts_s": starts_s,
        });
        Ok(Self {
            offsets,
            labels,
            starts,
            static_bits,
            l_filter,
            stats,
        })
    }

    /// The labels searched at the insertion of point `p`.
    #[inline]
    pub(crate) fn labels_of(&self, p: u32) -> &[u32] {
        &self.labels[self.offsets[p as usize] as usize..self.offsets[p as usize + 1] as usize]
    }

    /// The static member bitset of label `l`, if it has one.
    #[inline]
    pub(crate) fn static_bits(&self, l: u32) -> Option<&[u64]> {
        self.static_bits[l as usize].as_deref()
    }
}

/// Per-thread member marks of one label (a bitset over the nodes; auxiliary nodes are never
/// members).
#[derive(Debug, Default)]
pub(crate) struct MemberMarks {
    bits: Vec<u64>,
}

impl MemberMarks {
    pub(crate) fn new(n_nodes: usize) -> Self {
        Self {
            bits: vec![0; n_nodes.div_ceil(64)],
        }
    }

    pub(crate) fn set(&mut self, members: &[u32]) {
        for &v in members {
            self.bits[v as usize / 64] |= 1 << (v % 64);
        }
    }

    pub(crate) fn clear(&mut self, members: &[u32]) {
        for &v in members {
            self.bits[v as usize / 64] = 0;
        }
    }

    pub(crate) fn bits(&self) -> &[u64] {
        &self.bits
    }
}

/// Whether bit `v` is set (`false` beyond the bitset, e.g. for auxiliary nodes).
#[inline]
pub(crate) fn test_bit(bits: &[u64], v: u32) -> bool {
    bits.get(v as usize / 64)
        .is_some_and(|w| w >> (v % 64) & 1 == 1)
}

// ------------------------------------------------------------------ penalized insertion

/// Shared state of penalized insertion: the absolute weight of a failed category.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Penalty<'a> {
    store: &'a LabelStore,
    /// Added per failed category (`d² + weight · f`).
    weight: f32,
    pub(crate) key: PruneKey,
}

impl<'a> Penalty<'a> {
    pub(crate) fn new(store: &'a LabelStore, weight: f32, key: PruneKey) -> Self {
        Self { store, weight, key }
    }

    /// Marks the labels of `p` in `scratch` and records its categories (with the ranges of
    /// its row and the category's label ids).
    pub(crate) fn begin(&self, scratch: &mut PenaltyScratch, p: u32) {
        if scratch.marks.len() * 64 < self.store.n_labels() {
            scratch.marks = vec![0; self.store.n_labels().div_ceil(64)];
        }
        let row = self.row(p);
        scratch.groups.clear();
        let mut last = u32::MAX;
        for (i, &l) in row.iter().enumerate() {
            scratch.marks[l as usize / 64] |= 1 << (l % 64);
            let c = self.store.category(l);
            if c != last {
                let ids = self.store.category_labels(c);
                scratch.groups.push(Group {
                    first: i as u32,
                    end: i as u32 + 1,
                    lo: ids.start,
                    hi: ids.end,
                });
                last = c;
            } else if let Some(g) = scratch.groups.last_mut() {
                g.end = i as u32 + 1;
            }
        }
        scratch.n_categories = scratch.groups.len() as u32;
        scratch.source = p;
    }

    /// Clears the marks of [`Self::begin`].
    pub(crate) fn end(&self, scratch: &mut PenaltyScratch) {
        for &l in self.row(scratch.source) {
            scratch.marks[l as usize / 64] = 0;
        }
    }

    #[inline]
    fn row(&self, v: u32) -> &'a [u32] {
        if (v as usize) < self.store.n_points() {
            self.store.row(v)
        } else {
            &[]
        }
    }

    /// The number of the marked point's categories in which `c` carries none of its labels.
    /// Short rows are scanned against the marks; in long rows each category's segment is
    /// found by binary search and searched for the point's labels (or scanned if it is short).
    #[inline]
    pub(crate) fn failed(&self, scratch: &PenaltyScratch, c: u32) -> u32 {
        let target = scratch.n_categories;
        if target == 0 {
            return 0;
        }
        let row = self.row(c);
        let marked = |l: u32| scratch.marks[l as usize / 64] >> (l % 64) & 1 == 1;
        if row.len() <= LONG_ROW {
            let mut sat = 0;
            let mut last = u32::MAX;
            for &l in row {
                if marked(l) {
                    let cat = self.store.category(l);
                    if cat != last {
                        sat += 1;
                        last = cat;
                        if sat == target {
                            break;
                        }
                    }
                }
            }
            return target - sat;
        }
        let own = self.row(scratch.source);
        let mut failed = 0;
        for g in &scratch.groups {
            let a = row.partition_point(|&l| l < g.lo);
            let b = a + row[a..].partition_point(|&l| l < g.hi);
            let seg = &row[a..b];
            let mine = &own[g.first as usize..g.end as usize];
            let hit = if seg.len() <= 2 * mine.len() + LONG_ROW {
                seg.iter().any(|&l| marked(l))
            } else {
                mine.iter().any(|l| seg.binary_search(l).is_ok())
            };
            failed += u32::from(!hit);
        }
        failed
    }

    /// The key `d + weight · f(p, c)` (with the marks of `p`).
    #[inline]
    pub(crate) fn key(&self, scratch: &PenaltyScratch, c: u32, d: f32) -> f32 {
        if self.weight == 0.0 {
            return d;
        }
        let f = self.failed(scratch, c);
        if f == 0 {
            d
        } else {
            d + self.weight * f as f32
        }
    }

    /// Replaces the distances of `pool` (from the marked point) by their keys.
    pub(crate) fn penalize(&self, scratch: &PenaltyScratch, pool: &mut [Candidate]) {
        for c in pool {
            c.dist = self.key(scratch, c.id, c.dist);
        }
    }

    /// Prefetches the row offsets of node `v` (its row is read when its key is computed).
    #[inline]
    pub(crate) fn prefetch_row(&self, v: u32) {
        let (offsets, _) = self.store.forward();
        if let Some(o) = offsets.get(v as usize) {
            super::prefetch(std::ptr::from_ref(o).cast());
        }
    }
}

/// Rows longer than this are searched per category (binary search) by [`Penalty::failed`].
const LONG_ROW: usize = 64;

/// The labels of one category in the marked point's row: positions `first..end` of its row,
/// label ids `lo..hi` of the category.
#[derive(Debug, Default, Clone, Copy)]
struct Group {
    first: u32,
    end: u32,
    lo: u32,
    hi: u32,
}

/// Per-thread label marks of the point whose penalties are computed.
#[derive(Debug, Default, Clone)]
pub(crate) struct PenaltyScratch {
    marks: Vec<u64>,
    groups: Vec<Group>,
    n_categories: u32,
    source: u32,
}

/// The weight of a failed category for `λ`: `λ · m` with `m` = [`nn_scale`], or for `λ = ∞` a
/// power of two above every squared distance of the base (`4 · max ‖x‖²`), so that keys order
/// by the number of failed categories first.
pub(crate) fn penalty_weight<T: BuildElement>(base: &Matrix<T>, lambda: Lambda, scale: f64) -> f32 {
    match lambda {
        Lambda::Relative(x) => (f64::from(x) * scale) as f32,
        Lambda::Infinite => {
            let max_norm = (0..base.nrows())
                .map(|i| {
                    base.row(i)
                        .iter()
                        .map(|&x| f64::from(x.to_f32()).powi(2))
                        .sum::<f64>()
                })
                .fold(0.0f64, f64::max);
            let bound = (4.0 * max_norm).max(1.0);
            2f64.powi(bound.log2().ceil() as i32 + 1) as f32
        }
    }
}

/// The median over `sample` seeded random points of the squared distance to their nearest
/// non-identical point of the base (exact, brute force; distance 0 = identical vectors are
/// skipped). Returns the median and the sample's quartiles.
pub fn nn_scale<T: BuildElement>(
    base: &Matrix<T>,
    sample: usize,
    seed: u64,
    pool: &diskann_providers::utils::RayonThreadPool,
) -> (f64, [f64; 3]) {
    use diskann_providers::utils::ParallelIteratorInPool;
    let n = base.nrows();
    let mut rng = SplitMix64::new(seed ^ 0x6e6e_5f73_6361_6c65);
    let picks: Vec<usize> = (0..sample.min(n))
        .map(|_| rng.below(n as u64) as usize)
        .collect();
    let mut d: Vec<f64> = picks
        .par_iter()
        .map(|&i| {
            let q = base.row(i);
            let mut best = f32::MAX;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let dist = T::squared_l2(q, base.row(j));
                if dist > 0.0 && dist < best {
                    best = dist;
                }
            }
            f64::from(best)
        })
        .collect_in_pool(pool.as_ref());
    d.retain(|&x| x < f64::from(f32::MAX));
    if d.is_empty() {
        return (1.0, [1.0; 3]);
    }
    d.sort_by(f64::total_cmp);
    let q = |f: f64| d[((d.len() - 1) as f64 * f).round() as usize];
    (q(0.5), [q(0.25), q(0.5), q(0.75)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::labels::LabelStoreBuilder;

    fn store(rows: &[&[(&str, &str)]]) -> LabelStore {
        let mut b = LabelStoreBuilder::new();
        for row in rows {
            b.push_point(row.iter().copied()).unwrap();
        }
        b.finish().unwrap()
    }

    #[test]
    fn failed_categories_of_a_point_read_as_a_predicate() {
        // p = 0 {a: x, y; b: u}; c = 1 {a: y; b: v} fails b; c = 2 {c: z} fails both; c = 3
        // {a: x; b: u; c: z} fails none (its extra category does not count).
        let s = store(&[
            &[("a", "x"), ("a", "y"), ("b", "u")],
            &[("a", "y"), ("b", "v")],
            &[("c", "z")],
            &[("a", "x"), ("b", "u"), ("c", "z")],
        ]);
        let pen = Penalty::new(&s, 10.0, PruneKey::Raw);
        let mut sc = PenaltyScratch::default();
        pen.begin(&mut sc, 0);
        assert_eq!(
            (1..4).map(|c| pen.failed(&sc, c)).collect::<Vec<_>>(),
            vec![1, 2, 0]
        );
        assert_eq!(pen.failed(&sc, 4), 2, "auxiliary nodes fail every category");
        assert_eq!(pen.key(&sc, 1, 3.0), 13.0);
        pen.end(&mut sc);
        assert!(sc.marks.iter().all(|&w| w == 0));
        // Label-rich rows take the per-category path: same counts as a direct check.
        let mut rows: Vec<Vec<(String, String)>> = Vec::new();
        for i in 0..6 {
            let mut row: Vec<(String, String)> = (0..40 * (i % 3 + 1))
                .map(|j| ("g".to_string(), format!("g{}", (j * 7 + i) % 300)))
                .collect();
            row.push(("m".into(), format!("m{}", i % 2)));
            if i != 3 {
                row.push(("d".into(), format!("d{}", i % 3)));
            }
            rows.push(row);
        }
        let mut b = LabelStoreBuilder::new();
        for row in &rows {
            b.push_point(row.iter().map(|(c, l)| (c.as_str(), l.as_str())))
                .unwrap();
        }
        let s3 = b.finish().unwrap();
        let pen3 = Penalty::new(&s3, 1.0, PruneKey::Raw);
        for p in 0..6u32 {
            pen3.begin(&mut sc, p);
            for c in 0..6u32 {
                let (fp, fc) = (s3.row(p), s3.row(c));
                let cats: std::collections::BTreeSet<u32> =
                    fp.iter().map(|&l| s3.category(l)).collect();
                let direct = cats
                    .iter()
                    .filter(|&&k| !fp.iter().any(|&l| s3.category(l) == k && fc.contains(&l)))
                    .count() as u32;
                assert_eq!(pen3.failed(&sc, c), direct, "p {p} c {c}");
            }
            pen3.end(&mut sc);
        }
        assert!(s3.row(2).len() > LONG_ROW);
        // A point without labels has no categories to fail.
        let s2 = store(&[&[], &[("a", "x")]]);
        let pen2 = Penalty::new(&s2, 10.0, PruneKey::Raw);
        pen2.begin(&mut sc, 0);
        assert_eq!(pen2.failed(&sc, 1), 0);
    }

    #[test]
    fn filter_plan_takes_the_rarest_labels_within_the_budget() {
        // 128 points: `f` on all of them (frequent: > n/64 points, a static bitset when
        // searched), `m` on 4, `r` on 2; point 0 carries all three.
        let n = 128;
        let rows: Vec<Vec<(&str, &str)>> = (0..n)
            .map(|i| {
                let mut row = vec![("c", "f")];
                if i < 4 {
                    row.push(("d", "m"));
                }
                if i % 64 == 0 {
                    row.push(("e", "r"));
                }
                row
            })
            .collect();
        let refs: Vec<&[(&str, &str)]> = rows.iter().map(|r| r.as_slice()).collect();
        let s = store(&refs);
        let ix = s.name_index();
        let (f, m, r) = (ix["f"], ix["m"], ix["r"]);
        let base = crate::build::test_util::random_points(n, 4, 1);
        let plan = FilterPlan::new(&base, &s, LabelSubset::All, 2, 10, 1).unwrap();
        assert_eq!(plan.labels_of(0), &[r, m]);
        assert_eq!(plan.labels_of(1), &[m, f]);
        assert_eq!(plan.labels_of(5), &[f]);
        assert_eq!(plan.stats["points_over_budget"], 1);
        // n / 64 = 2: f and m are searched through static bitsets, r through per-thread marks
        assert!(plan.static_bits(f).is_some() && plan.static_bits(m).is_some());
        assert!(plan.static_bits(r).is_none());
        assert!(test_bit(plan.static_bits(f).unwrap(), 127));
        for l in [f, m, r] {
            let st = plan.starts[l as usize];
            assert!(s.postings(l).contains(&st));
        }
        let capped = FilterPlan::new(&base, &s, LabelSubset::Cap(4), 8, 10, 1).unwrap();
        assert_eq!(capped.labels_of(0), &[r, m]);
        assert_eq!(capped.labels_of(5), &[] as &[u32]);
        assert_eq!(capped.stats["filtered_searches"], 4 + 2);
    }

    #[test]
    fn nn_scale_is_the_median_nearest_non_identical_distance() {
        // Points on a line at 0, 1, 3, 3 (duplicate), 7: nearest non-identical distances²
        // 1, 1, 4, 4, 16.
        let xs = [0.0f32, 1.0, 3.0, 3.0, 7.0];
        let m = Matrix::from_slice(&xs, 5, 1);
        let pool = crate::build::thread_pool(2).unwrap();
        let (med, q) = nn_scale(&m, 4000, 3, &pool);
        assert_eq!(med, 4.0);
        assert!(q[0] >= 1.0 && q[2] <= 16.0);
        assert_eq!(penalty_weight(&m, Lambda::Relative(0.5), med), 2.0);
        // ∞: a power of two above 4 · max ‖x‖² = 196
        assert_eq!(penalty_weight(&m, Lambda::Infinite, med), 512.0);
    }

    #[test]
    fn parse_and_display() {
        for s in [
            "blind",
            "filtered(all)",
            "filtered(n/100)",
            "filtered(1000, 16)",
            "filtered(n/1000, R, 50)",
            "penalized(1, raw)",
            "penalized(inf, penalized)",
            "penalized(0.25, raw)",
        ] {
            let c: Candidates = s.parse().unwrap();
            assert_eq!(c.to_string(), s, "{s}");
            assert_eq!(c.to_string().parse::<Candidates>().unwrap(), c);
        }
        assert_eq!(
            "penalized(2)".parse::<Candidates>().unwrap(),
            Candidates::Penalized {
                lambda: Lambda::Relative(2.0),
                key: PruneKey::Raw
            }
        );
        assert!("penalized(-1)".parse::<Candidates>().is_err());
        assert!("filtered(n/0)".parse::<Candidates>().is_err());
        assert!("filtered(x)".parse::<Candidates>().is_err());
        assert!("nope(1)".parse::<Candidates>().is_err());
        assert_eq!(LabelSubset::Fraction(100).theta(1000), Some(10));
        assert_eq!(
            Candidates::Filtered {
                subset: LabelSubset::Fraction(100),
                budget: None,
                l_filter: None
            }
            .tag(),
            "filtered-n100"
        );
    }
}
