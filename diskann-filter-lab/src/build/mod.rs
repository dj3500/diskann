/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Graph construction for topology experiments (`flab build`).
//!
//! - [`builder`]: a parallel Vamana builder. Points are inserted in batches; within a batch
//!   every point searches the graph as it was before the batch, so the result depends only
//!   on the insertion order and the batch schedule, never on the thread count. It can record
//!   every point's candidate pool ([`pools`]).
//! - [`candidates`]: label-aware candidate generation for the insertions (label-filtered
//!   searches per label, or a search penalized by the number of failed categories).
//! - [`prune`]: pruning rules — label-blind α-RobustPrune, label-aware occlusion
//!   (FilteredVamana's containment rule), coverage budgets and degree splits.
//! - [`replay`]: *prune replay* — the adjacency rebuilt from recorded pools with another
//!   rule, without any search.
//!
//! The node space is the lab's (`graph.flab`): data points `0..n_data`, then one auxiliary
//! frozen start node (a copy of the medoid, as in upstream DiskANN's in-memory index).
//! Auxiliary nodes carry no labels.

pub mod builder;
pub mod candidates;
pub mod cli;
pub mod pools;
pub mod prune;
pub mod replay;

mod adjacency;
mod search;

use std::sync::{Mutex, PoisonError};

use crate::{
    labels::LabelStore,
    vectors::{Element, Matrix},
};

pub use builder::{build, BuildOutput, BuildParams, BuildStats, Order};
pub use candidates::{BudgetOrder, Candidates, LabelSubset, Lambda, PruneKey};
pub use pools::Pools;
pub use prune::{Candidate, PruneRule};
pub use replay::{replay, ReplayMode, ReplayOutput, ReplayParams};

/// Element types the builder supports: the medoid is computed in `f32`.
pub trait BuildElement: Element {
    fn to_f32(self) -> f32;
}

impl BuildElement for u8 {
    #[inline]
    fn to_f32(self) -> f32 {
        f32::from(self)
    }
}

impl BuildElement for f32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
}

/// Index of the row of `base` closest to the mean of all rows (upstream DiskANN's medoid
/// start point: the mean is accumulated in `f64`, distances are squared L2 in `f32`, the
/// first of equally close rows wins).
pub fn medoid_index<T: BuildElement>(base: &Matrix<T>) -> usize {
    let (n, d) = (base.nrows(), base.ncols());
    if n == 0 || d == 0 {
        return 0;
    }
    let mut sum = vec![0.0f64; d];
    for i in 0..n {
        for (s, &x) in sum.iter_mut().zip(base.row(i)) {
            *s += f64::from(x.to_f32());
        }
    }
    let mean: Vec<f32> = sum.iter().map(|s| (s / n as f64) as f32).collect();
    let mut row_f32 = vec![0.0f32; d];
    let (mut best, mut best_d) = (0, f32::MAX);
    for i in 0..n {
        for (o, &x) in row_f32.iter_mut().zip(base.row(i)) {
            *o = x.to_f32();
        }
        let dist = f32::squared_l2(&mean, &row_f32);
        if dist < best_d {
            (best, best_d) = (i, dist);
        }
    }
    best
}

/// Vectors of every node: the data points `0..n_data` (rows of the base), then the
/// auxiliary nodes.
#[derive(Debug, Clone, Copy)]
pub struct Points<'a, T> {
    base: &'a [T],
    aux: &'a [T],
    dim: usize,
    n_data: u32,
}

impl<'a, T: Element> Points<'a, T> {
    /// `aux` holds the auxiliary vectors row-major, in the dimension of `base`.
    pub fn new(base: &'a Matrix<T>, aux: &'a [T]) -> Self {
        let dim = base.ncols();
        assert!(
            dim > 0 && aux.len().is_multiple_of(dim),
            "auxiliary vectors must have the base's dimension"
        );
        Self {
            base: base.as_slice(),
            aux,
            dim,
            n_data: base.nrows() as u32,
        }
    }

    pub fn n_data(&self) -> usize {
        self.n_data as usize
    }

    pub fn n_nodes(&self) -> usize {
        self.n_data as usize + self.aux.len() / self.dim
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The vector of node `id`.
    #[inline]
    pub fn get(&self, id: u32) -> &'a [T] {
        let d = self.dim;
        match id.checked_sub(self.n_data) {
            None => &self.base[id as usize * d..(id as usize + 1) * d],
            Some(a) => &self.aux[a as usize * d..(a as usize + 1) * d],
        }
    }

    /// Squared L2 distance between two nodes.
    #[inline]
    pub fn dist(&self, a: u32, b: u32) -> f32 {
        T::squared_l2(self.get(a), self.get(b))
    }

    /// Prefetches the first cache lines of the vector of node `id`.
    #[inline(always)]
    fn prefetch(&self, id: u32) {
        let v = self.get(id);
        let bytes = std::mem::size_of_val(v).min(256);
        let p = v.as_ptr().cast::<u8>();
        for off in (0..bytes).step_by(64) {
            prefetch(p.wrapping_add(off));
        }
    }
}

#[inline(always)]
fn prefetch(p: *const u8) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: prefetching is a hint that never faults, whatever the address.
    unsafe {
        std::arch::x86_64::_mm_prefetch(p.cast::<i8>(), std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

/// Label rows of the nodes for label-aware rules: data points have their rows in the label
/// store; auxiliary nodes have no labels. Every row also has a 64-bit signature (one bit per
/// label, hashed) for fast rejection of containment tests.
#[derive(Debug)]
pub struct LabelInfo<'a> {
    store: &'a LabelStore,
    signatures: Vec<u64>,
}

impl<'a> LabelInfo<'a> {
    pub fn new(store: &'a LabelStore) -> Self {
        let signatures = (0..store.n_points() as u32)
            .map(|p| signature(store.row(p)))
            .collect();
        Self { store, signatures }
    }

    pub fn store(&self) -> &'a LabelStore {
        self.store
    }

    /// Sorted labels of node `id` (empty for auxiliary nodes).
    #[inline]
    pub fn row(&self, id: u32) -> &'a [u32] {
        if (id as usize) < self.signatures.len() {
            self.store.row(id)
        } else {
            &[]
        }
    }

    #[inline]
    pub fn signature(&self, id: u32) -> u64 {
        self.signatures.get(id as usize).copied().unwrap_or(0)
    }

    /// Whether every label of node `c` is a label of node `r` (`F_c ⊆ F_r`).
    #[inline]
    pub fn contains(&self, r: u32, c: u32) -> bool {
        self.signature(c) & !self.signature(r) == 0 && is_subset(self.row(c), self.row(r))
    }
}

/// One bit per label (hashed) of a sorted label list.
#[inline]
pub(crate) fn signature(labels: &[u32]) -> u64 {
    labels.iter().fold(0, |s, &l| s | label_bit(l))
}

#[inline]
fn label_bit(l: u32) -> u64 {
    1 << (u64::from(l).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 58)
}

/// Whether the sorted list `a` is a subset of the sorted list `b`.
#[inline]
pub(crate) fn is_subset(a: &[u32], b: &[u32]) -> bool {
    if a.len() > b.len() {
        return false;
    }
    let mut j = 0;
    for &x in a {
        while j < b.len() && b[j] < x {
            j += 1;
        }
        if j == b.len() || b[j] != x {
            return false;
        }
        j += 1;
    }
    true
}

/// Per-thread scratch values of a rayon pool, selected by the current thread's index.
pub(crate) struct PerThread<S> {
    slots: Vec<Mutex<S>>,
}

impl<S> PerThread<S> {
    pub(crate) fn new(threads: usize, init: impl Fn() -> S) -> Self {
        Self {
            slots: (0..threads.max(1)).map(|_| Mutex::new(init())).collect(),
        }
    }

    /// Runs `f` with the scratch of the calling pool thread. Each pool thread has its own
    /// slot, so the lock is never contended inside the pool.
    pub(crate) fn with<R>(&self, f: impl FnOnce(&mut S) -> R) -> R {
        let i = rayon::current_thread_index().unwrap_or(0) % self.slots.len();
        let mut guard = self.slots[i].lock().unwrap_or_else(PoisonError::into_inner);
        f(&mut guard)
    }

    pub(crate) fn into_inner(self) -> impl Iterator<Item = S> {
        self.slots
            .into_iter()
            .map(|m| m.into_inner().unwrap_or_else(PoisonError::into_inner))
    }
}

/// SplitMix64: a small seeded generator for insertion orders.
#[derive(Debug, Clone)]
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (`n > 0`).
    pub(crate) fn below(&mut self, n: u64) -> u64 {
        ((u128::from(self.next_u64()) * u128::from(n)) >> 64) as u64
    }
}

/// Creates the rayon pool of a build.
pub(crate) fn thread_pool(
    threads: usize,
) -> anyhow::Result<diskann_providers::utils::RayonThreadPool> {
    diskann_providers::utils::create_thread_pool(threads.max(1))
        .map_err(|e| anyhow::anyhow!("creating a pool of {threads} threads: {e}"))
}

#[cfg(test)]
pub(crate) mod test_util {
    use super::SplitMix64;
    use crate::vectors::Matrix;

    /// Deterministic pseudo-random `u8` points with an exact duplicate every 17 rows.
    pub(crate) fn random_points(n: usize, dim: usize, seed: u64) -> Matrix<u8> {
        let mut rng = SplitMix64::new(seed);
        let mut data: Vec<u8> = (0..n * dim).map(|_| rng.below(256) as u8).collect();
        for i in (0..n).step_by(17).skip(1) {
            let row: Vec<u8> = data[(i - 1) * dim..i * dim].to_vec();
            data[i * dim..(i + 1) * dim].copy_from_slice(&row);
        }
        Matrix::from_slice(&data, n, dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subset_and_signatures() {
        assert!(is_subset(&[], &[1, 2]));
        assert!(is_subset(&[2, 5], &[1, 2, 3, 5]));
        assert!(!is_subset(&[2, 4], &[1, 2, 3, 5]));
        assert!(!is_subset(&[6], &[1, 2, 3, 5]));
        assert!(!is_subset(&[1, 2], &[1]));
        let (a, b) = (signature(&[2, 5]), signature(&[1, 2, 3, 5]));
        assert_eq!(a & !b, 0);
    }

    #[test]
    fn medoid_is_the_row_closest_to_the_mean() {
        let data: Vec<u8> = vec![0, 0, 10, 10, 4, 6, 20, 20];
        let m = Matrix::<u8>::from_slice(&data, 4, 2);
        // mean (8.5, 9): row (10, 10) is at 2.25 + 1, row (4, 6) at 20.25 + 9
        assert_eq!(medoid_index(&m), 1);
    }

    #[test]
    fn split_mix_is_seeded() {
        let a: Vec<u64> = (0..4)
            .map({
                let mut r = SplitMix64::new(7);
                move |_| r.below(10)
            })
            .collect();
        let b: Vec<u64> = (0..4)
            .map({
                let mut r = SplitMix64::new(7);
                move |_| r.below(10)
            })
            .collect();
        assert_eq!(a, b);
        assert!(a.iter().all(|&x| x < 10));
    }
}
