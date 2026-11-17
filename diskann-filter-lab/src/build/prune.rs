/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Pruning rules: which candidates of a pool become a node's out-neighbors.
//!
//! A pool is sorted in *pool order* (increasing squared distance from the node `p`, then
//! increasing id). Every rule builds on upstream DiskANN's RobustPrune
//! (`diskann::graph::internal::prune::robust_prune`; the C++ library's `occlude_list`):
//!
//! - a selected candidate `r` *occludes* a later candidate `c` when
//!   `d(p, c) > α' · d(r, c)` (squared L2 distances; `d(r, c) = 0` always occludes);
//! - the scan runs in rounds with `α' = 1`, then `α' ← min(α' · min(α, 1.2), α)` up to `α`;
//!   a candidate is selected in the first round in which no closer selected candidate
//!   occludes it, and selection stops at the degree bound. Neighbors are kept in selection
//!   order.
//!
//! Label-aware rules restrict *which* selected candidates may occlude which (an occluder may
//! only remove a candidate whose labels it can stand in for) or reserve part of the degree for
//! label coverage. Auxiliary nodes carry no labels.

use std::{cmp::Ordering, fmt, str::FromStr};

use anyhow::{bail, ensure, Context, Result};
use bytemuck::{Pod, Zeroable};

use super::{
    candidates::{budget_order, BudgetOrder, Penalty, PenaltyScratch, PruneKey},
    is_subset, signature, LabelInfo, Points,
};
use crate::vectors::Element;

/// A pool entry: a candidate node and its squared distance from the node being pruned.
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct Candidate {
    pub id: u32,
    pub dist: f32,
}

impl Candidate {
    pub fn new(id: u32, dist: f32) -> Self {
        Self { id, dist }
    }

    /// Pool order: increasing distance, then increasing id.
    #[inline]
    pub fn order(a: &Self, b: &Self) -> Ordering {
        a.dist.total_cmp(&b.dist).then(a.id.cmp(&b.id))
    }
}

/// Puts `pool` in pool order without `source` and repeated ids, truncated to `maxc` entries
/// (upstream's maximum occlusion size).
pub fn normalize_pool(pool: &mut Vec<Candidate>, source: u32, maxc: usize) {
    pool.retain(|c| c.id != source);
    pool.sort_unstable_by(Candidate::order);
    pool.dedup_by_key(|c| c.id);
    pool.truncate(maxc);
}

/// A pruning rule with its parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PruneRule {
    /// `alpha(R, α)`: label-blind RobustPrune to degree `R`.
    Alpha { r: usize, alpha: f32 },
    /// `label_containment(R, α)`: FilteredVamana's label-aware occlusion as implemented in the
    /// C++ library (`Index::occlude_list`): a selected `r` may occlude a candidate `c` only if
    /// every label of `c` is a label of `r` (`F_c ⊆ F_r`), whatever the labels of `p`.
    LabelContainment { r: usize, alpha: f32 },
    /// `source_containment(R, α[, θ[, B[, rarest|categories]]])`: the FilteredVamana paper's
    /// rule (Algorithm 3): `r` may occlude `c` only if `F_p ∩ F_c ⊆ F_r`. With `θ`, only
    /// labels with at most `θ` points count (`F_p ∩ F_c ∩ Θ ⊆ F_r`, `Θ` = labels with at most
    /// `θ` points); with `B`, only the first `B` of them in budget order (the rarest, or per
    /// category). The rule then protects exactly the labels a capped (and budgeted) filtered
    /// insertion builds structure for.
    SourceContainment {
        r: usize,
        alpha: f32,
        theta: Option<u32>,
        budget: Option<u32>,
        /// Which labels the budget keeps (as `filtered`'s budget).
        order: BudgetOrder,
    },
    /// `coverage(R, α, B[, k])`: coverage-budgeted pruning. The first `R − B` neighbors of the
    /// label-blind RobustPrune are kept; then, for the labels of `p` from the rarest (fewest
    /// points) to the most frequent, while a label has fewer than `k` neighbors carrying it
    /// and fewer than `B` coverage edges were added, the closest unselected pool candidate
    /// carrying it is added; the remaining slots up to `R` take the rest of the RobustPrune
    /// selection in order.
    Coverage {
        r: usize,
        alpha: f32,
        budget: usize,
        k: usize,
    },
    /// `split(R_geo, R_lab[, α])`: a label-blind RobustPrune to `R_geo` neighbors, then up to
    /// `R_lab` more by RobustPrune with the containment rule of `label_containment`, in which
    /// the label-blind neighbors already act as occluders.
    Split {
        r_geo: usize,
        r_lab: usize,
        alpha: f32,
    },
}

const DEFAULT_ALPHA: f32 = 1.2;
const NEEDS_LABELS: &str = "label-aware rules are checked to have labels";

impl PruneRule {
    /// Total degree bound of the rule.
    pub fn degree(&self) -> usize {
        match *self {
            Self::Alpha { r, .. }
            | Self::LabelContainment { r, .. }
            | Self::SourceContainment { r, .. }
            | Self::Coverage { r, .. } => r,
            Self::Split { r_geo, r_lab, .. } => r_geo + r_lab,
        }
    }

    pub fn alpha(&self) -> f32 {
        match *self {
            Self::Alpha { alpha, .. }
            | Self::LabelContainment { alpha, .. }
            | Self::SourceContainment { alpha, .. }
            | Self::Coverage { alpha, .. }
            | Self::Split { alpha, .. } => alpha,
        }
    }

    /// Whether the rule reads labels.
    pub fn needs_labels(&self) -> bool {
        !matches!(self, Self::Alpha { .. })
    }

    /// A short name for file names, e.g. `alpha-R64-a1.2`.
    pub fn tag(&self) -> String {
        match *self {
            Self::Alpha { r, alpha } => format!("alpha-R{r}-a{alpha}"),
            Self::LabelContainment { r, alpha } => format!("lc-R{r}-a{alpha}"),
            Self::SourceContainment {
                r,
                alpha,
                theta,
                budget,
                order,
            } => {
                let mut t = format!("sc-R{r}-a{alpha}");
                if let Some(theta) = theta {
                    t += &format!("-t{theta}");
                }
                if let Some(b) = budget {
                    t += &format!("-b{b}");
                }
                if order == BudgetOrder::PerCategory {
                    t += "-cat";
                }
                t
            }
            Self::Coverage {
                r,
                alpha,
                budget,
                k,
            } => format!("cov-R{r}-a{alpha}-b{budget}-k{k}"),
            Self::Split {
                r_geo,
                r_lab,
                alpha,
            } => format!("split-G{r_geo}-L{r_lab}-a{alpha}"),
        }
    }

    fn validate(self) -> Result<Self> {
        let alpha = self.alpha();
        ensure!(
            alpha.is_finite() && alpha >= 1.0,
            "α must be a finite number ≥ 1 (got {alpha})"
        );
        ensure!(self.degree() >= 1, "the degree bound must be at least 1");
        ensure!(
            self.degree() <= u16::MAX as usize,
            "degree bound {} too large",
            self.degree()
        );
        if let Self::Coverage { r, budget, k, .. } = self {
            ensure!(budget <= r, "the coverage budget {budget} exceeds R = {r}");
            ensure!(k >= 1, "coverage needs k ≥ 1");
        }
        Ok(self)
    }

    /// Selects the out-neighbors of `source` from `pool` (pool order, without `source`) into
    /// `out`, in selection order. `labels` must be given for label-aware rules.
    pub fn prune<T: Element>(
        &self,
        points: &Points<'_, T>,
        labels: Option<&LabelInfo<'_>>,
        source: u32,
        pool: &[Candidate],
        scratch: &mut PruneScratch,
        out: &mut Vec<Candidate>,
    ) {
        out.clear();
        let PruneScratch {
            states,
            selected,
            base,
            taken,
            masks,
            counts,
            label_order,
            inter,
            inter_offsets,
            inter_sigs,
            protected,
            order_keys,
            n_dist,
            n_label_checks,
            ..
        } = scratch;
        selected.clear();
        if pool.is_empty() {
            return;
        }
        let mut nd = 0u64;
        let mut nl = 0u64;
        let mut dist = |a: usize, b: usize| {
            nd += 1;
            points.dist(pool[a].id, pool[b].id)
        };
        match *self {
            Self::Alpha { r, alpha } => {
                robust_prune(pool, states, selected, r, alpha, |_, _| true, &mut dist);
            }
            Self::LabelContainment { r, alpha } => {
                let labels = labels.expect(NEEDS_LABELS);
                let permit = |a: usize, c: usize| {
                    nl += 1;
                    labels.contains(pool[a].id, pool[c].id)
                };
                robust_prune(pool, states, selected, r, alpha, permit, &mut dist);
            }
            Self::SourceContainment {
                r,
                alpha,
                theta,
                budget,
                order,
            } => {
                let labels = labels.expect(NEEDS_LABELS);
                let mut f_p = labels.row(source);
                if theta.is_some() || budget.is_some() {
                    let store = labels.store();
                    let t = theta.map_or(usize::MAX, |t| t as usize);
                    protected.clear();
                    protected.extend(f_p.iter().copied().filter(|&l| store.posting_len(l) <= t));
                    if let Some(b) = budget.filter(|&b| protected.len() > b as usize) {
                        budget_order(store, protected, order, order_keys);
                        protected.truncate(b as usize);
                        protected.sort_unstable();
                    }
                    f_p = &protected[..];
                }
                if f_p.is_empty() {
                    // nothing to protect: every occlusion is permitted (label-blind RobustPrune)
                    robust_prune(pool, states, selected, r, alpha, |_, _| true, &mut dist);
                    *n_dist += nd;
                    out.extend(selected.iter().map(|&p| pool[p as usize]));
                    return;
                }
                inter.clear();
                inter_offsets.clear();
                inter_sigs.clear();
                inter_offsets.push(0);
                for c in pool {
                    let start = inter.len();
                    intersect(f_p, labels.row(c.id), inter);
                    inter_offsets.push(inter.len() as u32);
                    inter_sigs.push(signature(&inter[start..]));
                }
                let permit = |a: usize, c: usize| {
                    nl += 1;
                    let r_id = pool[a].id;
                    let common = &inter[inter_offsets[c] as usize..inter_offsets[c + 1] as usize];
                    inter_sigs[c] & !labels.signature(r_id) == 0
                        && is_subset(common, labels.row(r_id))
                };
                robust_prune(pool, states, selected, r, alpha, permit, &mut dist);
            }
            Self::Coverage {
                r,
                alpha,
                budget,
                k,
            } => {
                robust_prune(pool, states, selected, r, alpha, |_, _| true, &mut dist);
                let labels = labels.expect(NEEDS_LABELS);
                let f_p = labels.row(source);
                if budget > 0 && !f_p.is_empty() {
                    std::mem::swap(selected, base);
                    selected.clear();
                    let keep = base.len().min(r - budget);
                    selected.extend_from_slice(&base[..keep]);
                    taken.clear();
                    taken.resize(pool.len(), false);
                    for &p in selected.iter() {
                        taken[p as usize] = true;
                    }
                    let m = f_p.len();
                    let words = m.div_ceil(64);
                    masks.clear();
                    masks.resize(pool.len() * words, 0);
                    for (c, cand) in pool.iter().enumerate() {
                        mark_common(
                            f_p,
                            labels.row(cand.id),
                            &mut masks[c * words..(c + 1) * words],
                        );
                    }
                    counts.clear();
                    counts.resize(m, 0);
                    for &p in selected.iter() {
                        add_mask(counts, &masks[p as usize * words..(p as usize + 1) * words]);
                    }
                    label_order.clear();
                    label_order.extend(0..m as u32);
                    let store = labels.store();
                    label_order.sort_unstable_by_key(|&i| {
                        (store.posting_len(f_p[i as usize]), f_p[i as usize])
                    });
                    let mut picks = 0;
                    'labels: for &li in label_order.iter() {
                        let (word, bit) = (li as usize / 64, 1u64 << (li % 64));
                        while counts[li as usize] < k as u32 {
                            if picks == budget {
                                break 'labels;
                            }
                            let Some(c) = (0..pool.len())
                                .find(|&c| !taken[c] && masks[c * words + word] & bit != 0)
                            else {
                                break;
                            };
                            taken[c] = true;
                            selected.push(c as u32);
                            picks += 1;
                            add_mask(counts, &masks[c * words..(c + 1) * words]);
                        }
                    }
                    for &p in &base[keep..] {
                        if selected.len() >= r {
                            break;
                        }
                        if !taken[p as usize] {
                            taken[p as usize] = true;
                            selected.push(p);
                        }
                    }
                }
            }
            Self::Split {
                r_geo,
                r_lab,
                alpha,
            } => {
                robust_prune(pool, states, selected, r_geo, alpha, |_, _| true, &mut dist);
                if r_lab > 0 {
                    let labels = labels.expect(NEEDS_LABELS);
                    let limit = selected.len() + r_lab;
                    let permit = |a: usize, c: usize| {
                        nl += 1;
                        labels.contains(pool[a].id, pool[c].id)
                    };
                    robust_prune(pool, states, selected, limit, alpha, permit, &mut dist);
                }
            }
        }
        *n_dist += nd;
        *n_label_checks += nl;
        out.extend(selected.iter().map(|&p| pool[p as usize]));
    }
}

impl fmt::Display for PruneRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Alpha { r, alpha } => write!(f, "alpha({r}, {alpha})"),
            Self::LabelContainment { r, alpha } => write!(f, "label_containment({r}, {alpha})"),
            Self::SourceContainment {
                r,
                alpha,
                theta,
                budget,
                order,
            } => {
                write!(f, "source_containment({r}, {alpha}")?;
                let t = theta.map_or("inf".into(), |t| t.to_string());
                match (theta, budget, order) {
                    (None, None, BudgetOrder::Rarest) => {}
                    (Some(_), None, BudgetOrder::Rarest) => write!(f, ", {t}")?,
                    (_, Some(b), BudgetOrder::Rarest) => write!(f, ", {t}, {b}")?,
                    (_, b, BudgetOrder::PerCategory) => {
                        let b = b.map_or("R".into(), |b| b.to_string());
                        write!(f, ", {t}, {b}, {order}")?;
                    }
                }
                write!(f, ")")
            }
            Self::Coverage {
                r,
                alpha,
                budget,
                k,
            } => write!(f, "coverage({r}, {alpha}, {budget}, {k})"),
            Self::Split {
                r_geo,
                r_lab,
                alpha,
            } => write!(f, "split({r_geo}, {r_lab}, {alpha})"),
        }
    }
}

impl FromStr for PruneRule {
    type Err = anyhow::Error;

    /// `alpha(R[, α])`, `label_containment(R[, α])`, `source_containment(R[, α[, θ|inf[, B]]])`,
    /// `coverage(R, α, B[, k])`, `split(R_geo, R_lab[, α])`; α defaults to 1.2, k to 1.
    fn from_str(s: &str) -> Result<Self> {
        let s = s.trim();
        let (name, rest) = s
            .split_once('(')
            .with_context(|| format!("pruning rule {s:?}: expected name(arguments)"))?;
        let args = rest
            .strip_suffix(')')
            .with_context(|| format!("pruning rule {s:?}: missing ')'"))?;
        let args: Vec<&str> = args
            .split(',')
            .map(str::trim)
            .filter(|a| !a.is_empty())
            .collect();
        let int = |i: usize| -> Result<usize> {
            args[i].parse().with_context(|| {
                format!("pruning rule {s:?}: argument {} is not an integer", i + 1)
            })
        };
        let real = |i: usize| -> Result<f32> {
            match args.get(i) {
                None => Ok(DEFAULT_ALPHA),
                Some(a) => a.parse().with_context(|| {
                    format!("pruning rule {s:?}: argument {} is not a number", i + 1)
                }),
            }
        };
        let arity = |lo: usize, hi: usize| -> Result<()> {
            ensure!(
                (lo..=hi).contains(&args.len()),
                "pruning rule {s:?}: expected {lo} to {hi} arguments, got {}",
                args.len()
            );
            Ok(())
        };
        let rule = match name.trim() {
            "alpha" => {
                arity(1, 2)?;
                Self::Alpha {
                    r: int(0)?,
                    alpha: real(1)?,
                }
            }
            "label_containment" => {
                arity(1, 2)?;
                Self::LabelContainment {
                    r: int(0)?,
                    alpha: real(1)?,
                }
            }
            "source_containment" => {
                arity(1, 5)?;
                let opt = |i: usize, what: &str| -> Result<Option<u32>> {
                    match args.get(i) {
                        None | Some(&"inf") | Some(&"R") => Ok(None),
                        Some(_) => Ok(Some(u32::try_from(int(i)?).with_context(|| {
                            format!("pruning rule {s:?}: {what} does not fit in 32 bits")
                        })?)),
                    }
                };
                Self::SourceContainment {
                    r: int(0)?,
                    alpha: real(1)?,
                    theta: opt(2, "θ")?,
                    budget: opt(3, "the budget")?,
                    order: match args.get(4) {
                        None => BudgetOrder::Rarest,
                        Some(o) => o.parse()?,
                    },
                }
            }
            "coverage" => {
                arity(3, 4)?;
                Self::Coverage {
                    r: int(0)?,
                    alpha: real(1)?,
                    budget: int(2)?,
                    k: if args.len() > 3 { int(3)? } else { 1 },
                }
            }
            "split" => {
                arity(2, 3)?;
                Self::Split {
                    r_geo: int(0)?,
                    r_lab: int(1)?,
                    alpha: real(2)?,
                }
            }
            other => bail!(
                "unknown pruning rule {other:?} (alpha, label_containment, source_containment, coverage, split)"
            ),
        };
        rule.validate()
    }
}

/// Scratch buffers and counters of pruning (one per thread).
#[derive(Debug, Default)]
pub struct PruneScratch {
    states: Vec<State>,
    selected: Vec<u32>,
    base: Vec<u32>,
    taken: Vec<bool>,
    masks: Vec<u64>,
    counts: Vec<u32>,
    label_order: Vec<u32>,
    inter: Vec<u32>,
    inter_offsets: Vec<u32>,
    inter_sigs: Vec<u64>,
    /// The labels of the pruned node that a capped containment rule protects.
    protected: Vec<u32>,
    order_keys: Vec<(u32, u64, u32)>,
    /// Candidate buffer of back-edge re-prunes.
    pool: Vec<Candidate>,
    /// Back-edge sources not yet in a row.
    pub(crate) fresh: Vec<u32>,
    /// Label marks of penalized re-prunes.
    pub(crate) penalty: PenaltyScratch,
    /// Distance computations between pool candidates (occlusion tests).
    pub n_dist: u64,
    /// Label occlusion-permission tests.
    pub n_label_checks: u64,
    /// Distance computations from a re-pruned node to its candidates.
    pub n_dist_source: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct State {
    /// Largest occlusion factor found so far (`f32::MAX` once selected).
    factor: f32,
    /// Number of entries of the selection list already checked against.
    checked: u32,
}

/// RobustPrune over `pool` continuing from the positions already in `selected` (they occlude
/// but are not re-selected) until `selected` holds `limit` positions or every round has run.
/// `permit(r, c)`: whether selected position `r` may occlude candidate position `c`;
/// `dist(r, c)`: their squared distance. Occlusion tests are lazy as upstream's: a candidate
/// is checked against each selected entry at most once over all rounds, and only while its
/// occlusion factor stays below the round's α.
fn robust_prune<P, D>(
    pool: &[Candidate],
    states: &mut Vec<State>,
    selected: &mut Vec<u32>,
    limit: usize,
    alpha: f32,
    mut permit: P,
    mut dist: D,
) where
    P: FnMut(usize, usize) -> bool,
    D: FnMut(usize, usize) -> f32,
{
    states.clear();
    states.resize(pool.len(), State::default());
    for &s in selected.iter() {
        states[s as usize].factor = f32::MAX;
    }
    let increment = alpha.min(1.2);
    let mut current = 1.0f32;
    while selected.len() < limit {
        for i in 0..pool.len() {
            if selected.len() >= limit {
                break;
            }
            let State {
                mut factor,
                checked,
            } = states[i];
            if factor > current {
                continue;
            }
            let mut checked = checked as usize;
            while checked < selected.len() {
                let r = selected[checked] as usize;
                checked += 1;
                if r >= i || !permit(r, i) {
                    continue;
                }
                let d = dist(r, i);
                factor = if d == 0.0 {
                    f32::MAX
                } else {
                    factor.max(pool[i].dist / d)
                };
                if factor > current {
                    break;
                }
            }
            if factor > current {
                states[i] = State {
                    factor,
                    checked: checked as u32,
                };
                continue;
            }
            states[i] = State {
                factor: f32::MAX,
                checked: checked as u32,
            };
            selected.push(i as u32);
        }
        if current >= alpha {
            break;
        }
        current = (current * increment).min(alpha);
    }
}

/// Appends `a ∩ b` (sorted lists) to `out`.
fn intersect(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
}

/// Sets bit `i` of `mask` for every position `i` of `f_p` whose label is in `f_c`.
fn mark_common(f_p: &[u32], f_c: &[u32], mask: &mut [u64]) {
    let (mut i, mut j) = (0, 0);
    while i < f_p.len() && j < f_c.len() {
        match f_p[i].cmp(&f_c[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                mask[i / 64] |= 1 << (i % 64);
                i += 1;
                j += 1;
            }
        }
    }
}

fn add_mask(counts: &mut [u32], mask: &[u64]) {
    for (w, &word) in mask.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            counts[w * 64 + bits.trailing_zeros() as usize] += 1;
            bits &= bits - 1;
        }
    }
}

/// Collects the ids of `sources` that are not in `row` into `out`.
pub(crate) fn fresh_sources(row: &[u32], sources: impl Iterator<Item = u32>, out: &mut Vec<u32>) {
    out.clear();
    out.extend(sources.filter(|s| !row.contains(s)));
}

/// Re-prunes node `t` over its current neighbors `row` and the new sources `fresh`
/// (upstream's `add_edge_and_prune` when the row would overflow): distances from `t` (with a
/// penalty whose prune key is [`PruneKey::Penalized`]: the penalized keys), pool order,
/// truncated to `maxc`; the result goes to `out`.
#[expect(
    clippy::too_many_arguments,
    reason = "internal helper shared by build and replay"
)]
pub(crate) fn reprune<T: Element>(
    rule: &PruneRule,
    points: &Points<'_, T>,
    labels: Option<&LabelInfo<'_>>,
    penalty: Option<&Penalty<'_>>,
    t: u32,
    row: &[u32],
    fresh: &[u32],
    maxc: usize,
    scratch: &mut PruneScratch,
    out: &mut Vec<Candidate>,
) {
    let mut pool = std::mem::take(&mut scratch.pool);
    pool.clear();
    let tv = points.get(t);
    for &v in row.iter().chain(fresh) {
        pool.push(Candidate::new(v, T::squared_l2(tv, points.get(v))));
    }
    scratch.n_dist_source += pool.len() as u64;
    if let Some(pen) = penalty.filter(|p| p.key == PruneKey::Penalized) {
        pen.begin(&mut scratch.penalty, t);
        pen.penalize(&scratch.penalty, &mut pool);
        pen.end(&mut scratch.penalty);
    }
    normalize_pool(&mut pool, t, maxc);
    rule.prune(points, labels, t, &pool, scratch, out);
    scratch.pool = pool;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{labels::LabelStoreBuilder, vectors::Matrix};

    /// Points on a line (1-d `f32`), id = index.
    fn line(xs: &[f32]) -> Matrix<f32> {
        Matrix::from_slice(xs, xs.len(), 1)
    }

    fn pool_of(points: &Points<'_, f32>, source: u32, ids: &[u32]) -> Vec<Candidate> {
        let mut pool: Vec<Candidate> = ids
            .iter()
            .map(|&i| Candidate::new(i, points.dist(source, i)))
            .collect();
        normalize_pool(&mut pool, source, 750);
        pool
    }

    fn labels(rows: &[&[&str]]) -> crate::labels::LabelStore {
        let mut b = LabelStoreBuilder::new();
        for row in rows {
            b.push_point(row.iter().map(|l| ("c", *l))).unwrap();
        }
        b.finish().unwrap()
    }

    fn ids(out: &[Candidate]) -> Vec<u32> {
        out.iter().map(|c| c.id).collect()
    }

    #[test]
    fn parse_and_display() {
        for s in [
            "alpha(64, 1.2)",
            "label_containment(32, 1.3)",
            "source_containment(48, 1.2)",
            "source_containment(48, 1.2, 10000)",
            "source_containment(48, 1.2, 10000, 2)",
            "coverage(64, 1.2, 16, 2)",
            "split(32, 16, 1.2)",
        ] {
            let rule: PruneRule = s.parse().unwrap();
            assert_eq!(rule.to_string(), s);
            assert_eq!(rule.to_string().parse::<PruneRule>().unwrap(), rule);
        }
        assert_eq!(
            "alpha(8)".parse::<PruneRule>().unwrap(),
            PruneRule::Alpha { r: 8, alpha: 1.2 }
        );
        assert_eq!(
            "coverage(8, 1.2, 2)".parse::<PruneRule>().unwrap(),
            PruneRule::Coverage {
                r: 8,
                alpha: 1.2,
                budget: 2,
                k: 1
            }
        );
        assert!("alpha(8, 0.9)".parse::<PruneRule>().is_err());
        assert!("coverage(8, 1.2, 9)".parse::<PruneRule>().is_err());
        assert!("nope(1)".parse::<PruneRule>().is_err());
        assert!("alpha(1, 2, 3)".parse::<PruneRule>().is_err());
    }

    #[test]
    fn alpha_prune_occludes_and_relaxes_by_rounds() {
        // source 0 at 0; candidates at 1, 2, 3 and -1.05.
        let m = line(&[0.0, 1.0, 2.0, 3.0, -1.05]);
        let p = Points::new(&m, &[]);
        let pool = pool_of(&p, 0, &[1, 2, 3, 4]);
        assert_eq!(ids(&pool), vec![1, 4, 2, 3]);
        let mut s = PruneScratch::default();
        let mut out = Vec::new();
        // α = 1: 2 and 3 are occluded by 1 (4 > 1, 9 > 4); 4 is not (1.1025 < 4.2025).
        PruneRule::Alpha { r: 8, alpha: 1.0 }.prune(&p, None, 0, &pool, &mut s, &mut out);
        assert_eq!(ids(&out), vec![1, 4]);
        // Degree bound.
        PruneRule::Alpha { r: 1, alpha: 1.2 }.prune(&p, None, 0, &pool, &mut s, &mut out);
        assert_eq!(ids(&out), vec![1]);
        // Larger α: the occlusion factors of 3 (9 / 4 = 2.25) and 2 (4 / 1 = 4) are reached in
        // later rounds (α' = 1, 1.2, 1.44, …, α), so 3 is selected before 2.
        PruneRule::Alpha { r: 8, alpha: 2.5 }.prune(&p, None, 0, &pool, &mut s, &mut out);
        assert_eq!(ids(&out), vec![1, 4, 3]);
        PruneRule::Alpha { r: 8, alpha: 4.5 }.prune(&p, None, 0, &pool, &mut s, &mut out);
        assert_eq!(ids(&out), vec![1, 4, 3, 2]);
    }

    #[test]
    fn duplicates_always_occlude_in_label_blind_prune() {
        let m = line(&[0.0, 1.0, 1.0]);
        let p = Points::new(&m, &[]);
        let pool = pool_of(&p, 0, &[1, 2]);
        let mut s = PruneScratch::default();
        let mut out = Vec::new();
        PruneRule::Alpha { r: 8, alpha: 100.0 }.prune(&p, None, 0, &pool, &mut s, &mut out);
        assert_eq!(ids(&out), vec![1]);
    }

    #[test]
    fn containment_rules() {
        // Source 0 at 0; 1 at 1 (labels a, b); 2 at 2 (a); 3 at 2.1 (c); 4 at 1 (a, c), a
        // duplicate of 1.
        let m = line(&[0.0, 1.0, 2.0, 2.1, 1.0]);
        let p = Points::new(&m, &[]);
        let store = labels(&[&["a"], &["a", "b"], &["a"], &["c"], &["a", "c"]]);
        let info = LabelInfo::new(&store);
        let pool = pool_of(&p, 0, &[1, 2, 3, 4]);
        assert_eq!(ids(&pool), vec![1, 4, 2, 3]);
        let mut s = PruneScratch::default();
        let mut out = Vec::new();
        PruneRule::Alpha { r: 8, alpha: 1.2 }.prune(&p, Some(&info), 0, &pool, &mut s, &mut out);
        assert_eq!(ids(&out), vec![1]);
        // F_c ⊆ F_r: 4 {a,c} is not contained in 1 {a,b}; 2 {a} is (occluded by 1); 3 {c} is
        // contained in 4 {a,c} and occluded by it.
        PruneRule::LabelContainment { r: 8, alpha: 1.2 }.prune(
            &p,
            Some(&info),
            0,
            &pool,
            &mut s,
            &mut out,
        );
        assert_eq!(ids(&out), vec![1, 4]);
        // F_p ∩ F_c ⊆ F_r with F_p = {a}: every candidate's common part is ⊆ {a} or empty, so
        // the rule acts like the label-blind one here.
        PruneRule::SourceContainment {
            r: 8,
            alpha: 1.2,
            theta: None,
            budget: None,
            order: BudgetOrder::Rarest,
        }
        .prune(&p, Some(&info), 0, &pool, &mut s, &mut out);
        assert_eq!(ids(&out), vec![1]);
    }

    #[test]
    fn capped_source_containment_protects_only_rare_labels() {
        // Source 0 {a, r} at 0; 1 {x} at 1; 2 {a} at 2 (occluded by 1 for a label-blind rule);
        // 3 {a} far away (not in the pool) makes `a` the frequent label.
        let m = line(&[0.0, 1.0, 2.0, 100.0]);
        let p = Points::new(&m, &[]);
        let store = labels(&[&["a", "r"], &["x"], &["a"], &["a"]]);
        let info = LabelInfo::new(&store);
        let pool = pool_of(&p, 0, &[1, 2]);
        let mut s = PruneScratch::default();
        let mut out = Vec::new();
        let sc = |theta| PruneRule::SourceContainment {
            r: 8,
            alpha: 1.2,
            theta,
            budget: None,
            order: BudgetOrder::Rarest,
        };
        // F_0 ∩ F_2 = {a} is not on 1: the paper's rule keeps 2.
        sc(None).prune(&p, Some(&info), 0, &pool, &mut s, &mut out);
        assert_eq!(ids(&out), vec![1, 2]);
        // With θ = 1 only labels with at most one point (r, x) are protected: 2 shares no
        // protected label with 0, so 1 occludes it as in the label-blind rule.
        sc(Some(1)).prune(&p, Some(&info), 0, &pool, &mut s, &mut out);
        assert_eq!(ids(&out), vec![1]);
        assert_eq!(sc(Some(1)).to_string(), "source_containment(8, 1.2, 1)");
        assert_eq!(
            "source_containment(8, 1.2, 1)"
                .parse::<PruneRule>()
                .unwrap(),
            sc(Some(1))
        );
        assert_eq!(sc(Some(7)).tag(), "sc-R8-a1.2-t7");
        // A budget protects only the rarest labels: source 0 {a, r, s}, r on 1 point, s on 3,
        // a on 4; candidate 2 {a, s}. With B = 1 only r is protected and 1 {x} occludes 2;
        // with B = 2, s is protected too and 2 stays.
        let m = line(&[0.0, 1.0, 2.0, 100.0, 101.0, 102.0]);
        let p = Points::new(&m, &[]);
        let store = labels(&[
            &["a", "r", "s"],
            &["x"],
            &["a", "s"],
            &["a"],
            &["a"],
            &["s"],
        ]);
        let info = LabelInfo::new(&store);
        let pool = pool_of(&p, 0, &[1, 2]);
        let scb = |budget| PruneRule::SourceContainment {
            r: 8,
            alpha: 1.2,
            theta: None,
            budget,
            order: BudgetOrder::Rarest,
        };
        scb(Some(1)).prune(&p, Some(&info), 0, &pool, &mut s, &mut out);
        assert_eq!(ids(&out), vec![1]);
        scb(Some(2)).prune(&p, Some(&info), 0, &pool, &mut s, &mut out);
        assert_eq!(ids(&out), vec![1, 2]);
        assert_eq!(
            scb(Some(2)).to_string(),
            "source_containment(8, 1.2, inf, 2)"
        );
        assert_eq!(
            "source_containment(8, 1.2, inf, 2)"
                .parse::<PruneRule>()
                .unwrap(),
            scb(Some(2))
        );
        assert_eq!(scb(Some(2)).tag(), "sc-R8-a1.2-b2");
    }

    #[test]
    fn coverage_adds_the_closest_candidate_of_an_uncovered_label() {
        // Source 0 {a, r}; 1 {a} at 1 occludes everything farther; 3 {r} at 3, 4 {r} at 3.5;
        // 5 {a} is far (not in the pool) and makes r the rarer label.
        let m = line(&[0.0, 1.0, 2.0, 3.0, 3.5, 100.0]);
        let p = Points::new(&m, &[]);
        let store = labels(&[&["a", "r"], &["a"], &["a"], &["r"], &["r"], &["a"]]);
        let info = LabelInfo::new(&store);
        let pool = pool_of(&p, 0, &[1, 2, 3, 4]);
        let mut s = PruneScratch::default();
        let mut out = Vec::new();
        let cov = |r, budget, k| PruneRule::Coverage {
            r,
            alpha: 1.2,
            budget,
            k,
        };
        cov(4, 1, 1).prune(&p, Some(&info), 0, &pool, &mut s, &mut out);
        assert_eq!(ids(&out), vec![1, 3]);
        cov(4, 2, 2).prune(&p, Some(&info), 0, &pool, &mut s, &mut out);
        assert_eq!(ids(&out), vec![1, 3, 4]);
        // No budget: plain RobustPrune.
        cov(4, 0, 1).prune(&p, Some(&info), 0, &pool, &mut s, &mut out);
        assert_eq!(ids(&out), vec![1]);
        // The budget is reserved: with R = 1 and B = 1 only the coverage edge remains.
        cov(1, 1, 1).prune(&p, Some(&info), 0, &pool, &mut s, &mut out);
        assert_eq!(ids(&out), vec![3]);
    }

    #[test]
    fn split_adds_label_aware_edges_after_the_geometric_part() {
        let m = line(&[0.0, 1.0, 2.0, 2.1, 1.0]);
        let p = Points::new(&m, &[]);
        let store = labels(&[&["a"], &["a", "b"], &["a"], &["c"], &["a", "c"]]);
        let info = LabelInfo::new(&store);
        let pool = pool_of(&p, 0, &[1, 2, 3, 4]);
        let mut s = PruneScratch::default();
        let mut out = Vec::new();
        PruneRule::Split {
            r_geo: 1,
            r_lab: 2,
            alpha: 1.2,
        }
        .prune(&p, Some(&info), 0, &pool, &mut s, &mut out);
        assert_eq!(ids(&out), vec![1, 4]);
        PruneRule::Split {
            r_geo: 1,
            r_lab: 0,
            alpha: 1.2,
        }
        .prune(&p, Some(&info), 0, &pool, &mut s, &mut out);
        assert_eq!(ids(&out), vec![1]);
    }

    #[test]
    fn reprune_merges_the_row_and_new_sources() {
        let m = line(&[0.0, 1.0, 2.0, -1.05, 3.0]);
        let p = Points::new(&m, &[]);
        let mut s = PruneScratch::default();
        let mut out = Vec::new();
        let rule = PruneRule::Alpha { r: 8, alpha: 1.2 };
        reprune(
            &rule,
            &p,
            None,
            None,
            0,
            &[1, 2],
            &[3, 4],
            750,
            &mut s,
            &mut out,
        );
        assert_eq!(ids(&out), vec![1, 3]);
        assert_eq!(s.n_dist_source, 4);
    }
}
