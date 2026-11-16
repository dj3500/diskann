/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The unfiltered effort curve of a graph from base-derived queries (`flab calibrate`; read by
//! `terminator = effort_curve` through [`crate::kernel::cost::UnitCosts::effort_list`]).
//!
//! - **Queries**: midpoints (x_a + x_b) / 2 of seeded uniformly random pairs of base points
//!   (`u8`: rounded half up), or given pairs.
//! - **Ground truth**: the `kmax` smallest squared distances of every query, from one scan of the
//!   base.
//! - **Searches**: one plain unfiltered search per query and list size, from the global start,
//!   collecting the `kmax` nearest evaluated nodes. The traversal does not depend on the
//!   collector, so the `K'` nearest of them are the answers of a heap(K') search at the same list
//!   size, for every `K' ≤ kmax`. List sizes 10·2^i up to `l_max`, stopping after the first size at
//!   which every K' has a mean recall of at least 0.995 (above the largest target recall, 0.99).
//! - **Table**: mean tie-aware recall@K' per list size (made non-decreasing in L); per target
//!   recall R and K', the smallest list size reaching R, log-linear in L between measured sizes.
//!   Above the largest size the last two sizes give a power law, 1 − recall ∝ L^b, at most 8×
//!   the largest size and n.

use std::{cmp::Ordering, collections::BinaryHeap, time::Instant};

use anyhow::{ensure, Result};
use serde_json::{json, Value};

use super::Rng;
use crate::{
    kernel::{
        config::{Collector as Coll, Kernel, Laziness, StartSource},
        search::{Index, Plan, Query, Searcher},
    },
    labels::LabelStore,
    predicate::PredicateSetBuilder,
    vectors::{DType, Element, Matrix},
};

/// Target recalls of the table.
pub const R_GRID: [f64; 11] = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99];
/// Neighbour counts K' of the table (those up to `kmax`).
pub const K_GRID: [usize; 6] = [10, 30, 100, 300, 1000, 3000];
/// Extrapolation above the largest measured list size stops at this multiple of it.
const EXTRAPOLATION_CAP: f64 = 8.0;
/// The list sizes stop growing once every K' reaches this mean recall.
const STOP_RECALL: f64 = 0.995;

/// Options of the effort calibration.
#[derive(Debug, Clone)]
pub struct EffortOptions {
    /// Midpoint queries (0: no effort calibration).
    pub queries: usize,
    pub seed: u64,
    /// Largest measured list size.
    pub l_max: usize,
    /// Given base pairs instead of random ones.
    pub pairs: Option<Vec<(u32, u32)>>,
}

impl Default for EffortOptions {
    fn default() -> Self {
        Self {
            queries: 200,
            seed: 0,
            l_max: 40_960,
            pairs: None,
        }
    }
}

/// The calibrated table (unit-cost fields `effort_r`, `effort_k`, `effort_l`) and its report.
pub struct Effort {
    pub r: Vec<f64>,
    pub k: Vec<f64>,
    pub l: Vec<Vec<f64>>,
    pub doc: Value,
}

#[derive(Clone, Copy, PartialEq)]
struct Dist(f32);

impl Eq for Dist {}

impl PartialOrd for Dist {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Dist {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// The midpoint of two vectors (`u8`: rounded half up; `f32`: in double precision).
fn midpoint<T: Element>(a: &[T], b: &[T]) -> Vec<T> {
    match T::DTYPE {
        DType::U8 => {
            let (a, b): (&[u8], &[u8]) = (bytemuck::cast_slice(a), bytemuck::cast_slice(b));
            let m: Vec<u8> = a
                .iter()
                .zip(b)
                .map(|(&x, &y)| (x as u16 + y as u16).div_ceil(2) as u8)
                .collect();
            bytemuck::cast_slice(&m).to_vec()
        }
        DType::F32 => {
            let (a, b): (&[f32], &[f32]) = (bytemuck::cast_slice(a), bytemuck::cast_slice(b));
            let m: Vec<f32> = a
                .iter()
                .zip(b)
                .map(|(&x, &y)| ((x as f64 + y as f64) / 2.0) as f32)
                .collect();
            bytemuck::cast_slice(&m).to_vec()
        }
    }
}

/// The `kmax` smallest squared distances from each query to the base (ascending), in one scan.
fn ground_truth<T: Element>(base: &Matrix<T>, queries: &[Vec<T>], kmax: usize) -> Vec<Vec<f32>> {
    let mut heaps: Vec<BinaryHeap<Dist>> = queries
        .iter()
        .map(|_| BinaryHeap::with_capacity(kmax + 1))
        .collect();
    for j in 0..base.nrows() {
        let x = base.row(j);
        for (h, q) in heaps.iter_mut().zip(queries) {
            let d = T::squared_l2(q, x);
            if h.len() < kmax {
                h.push(Dist(d));
            } else if let Some(mut top) = h.peek_mut() {
                if d < top.0 {
                    *top = Dist(d);
                }
            }
        }
    }
    heaps
        .into_iter()
        .map(|h| h.into_sorted_vec().into_iter().map(|d| d.0).collect())
        .collect()
}

/// The table: per target recall (rows) and neighbour count (columns), the smallest list size
/// whose mean recall reaches it. `recall[j][i]` is the mean recall@K'_j at `ls[i]`; `cap`
/// bounds the extrapolation.
pub fn fit_table(ls: &[f64], recall: &[Vec<f64>], r_grid: &[f64], cap: f64) -> Vec<Vec<f64>> {
    let monotone: Vec<Vec<f64>> = recall
        .iter()
        .map(|row| {
            let mut best = f64::NEG_INFINITY;
            row.iter()
                .map(|&r| {
                    best = best.max(r);
                    best
                })
                .collect()
        })
        .collect();
    r_grid
        .iter()
        .map(|&target| {
            monotone
                .iter()
                .map(|r| list_for(ls, r, target, cap))
                .collect()
        })
        .collect()
}

/// The smallest list size with recall `target` on the non-decreasing curve `r` over `ls`.
fn list_for(ls: &[f64], r: &[f64], target: f64, cap: f64) -> f64 {
    let last = ls.len() - 1;
    if let Some(i) = r.iter().position(|&x| x >= target) {
        if i == 0 {
            return ls[0];
        }
        let t = (target - r[i - 1]) / (r[i] - r[i - 1]).max(1e-12);
        return (ls[i - 1].ln() + t * (ls[i].ln() - ls[i - 1].ln())).exp();
    }
    // Not reached: the power law of the miss rate through the last two sizes.
    let cap = cap.max(ls[last]);
    if last == 0 || r[last] <= r[last - 1] || r[last] >= 1.0 {
        return cap;
    }
    let (m0, m1) = (1.0 - r[last - 1], 1.0 - r[last]);
    let b = (m1 / m0).ln() / (ls[last] / ls[last - 1]).ln();
    let l = ls[last] * ((1.0 - target) / m1).powf(1.0 / b);
    if l.is_finite() {
        l.min(cap)
    } else {
        cap
    }
}

/// Measures the effort curve of the graph of `ix` (see the module documentation).
pub fn effort_curve<T: Element>(
    base: &Matrix<T>,
    ix: &Index<'_, T>,
    labels: &LabelStore,
    opts: &EffortOptions,
) -> Result<Effort> {
    let n = base.nrows();
    ensure!(n >= 2, "the effort curve needs at least two points");
    let t0 = Instant::now();
    let pairs: Vec<(u32, u32)> = match &opts.pairs {
        Some(p) => {
            ensure!(
                p.iter().all(|&(a, b)| (a as usize) < n && (b as usize) < n),
                "an effort pair is out of range"
            );
            p.clone()
        }
        None => {
            let mut rng = Rng(opts.seed);
            (0..opts.queries)
                .map(|_| (rng.below(n) as u32, rng.below(n) as u32))
                .collect()
        }
    };
    ensure!(!pairs.is_empty(), "the effort curve needs queries");
    let queries: Vec<Vec<T>> = pairs
        .iter()
        .map(|&(a, b)| midpoint(base.row(a as usize), base.row(b as usize)))
        .collect();
    let kmax = K_GRID[K_GRID.len() - 1].min(n);
    let ks: Vec<usize> = K_GRID.iter().copied().filter(|&k| k <= kmax).collect();
    let gt = ground_truth(base, &queries, kmax);
    let t_gt = t0.elapsed().as_secs_f64();

    let kernel = Kernel {
        collector: Coll::Heap,
        laziness: Laziness::HeapBound,
        starts: vec![StartSource::Global],
        ..Kernel::default()
    };
    let plan = Plan::new(&kernel, ix)?.generic();
    let mut st = Searcher::for_index(ix, &[&plan]);
    let none = {
        let mut pb = PredicateSetBuilder::new(labels);
        pb.push::<&str>(&[]);
        pb.finish()
    };
    let mut ls = Vec::new();
    let mut l = 10usize;
    while l <= opts.l_max.max(10) {
        ls.push(l);
        l *= 2;
    }
    let (mut ids, mut ds) = (vec![0u32; kmax], vec![0f32; kmax]);
    let mut recall = vec![vec![0f64; ls.len()]; ks.len()];
    let mut n_dist = vec![0f64; ls.len()];
    let m = queries.len() as f64;
    let mut measured = ls.len();
    for (li, &l) in ls.iter().enumerate() {
        for (q, d_gt) in queries.iter().zip(&gt) {
            let query = Query {
                vector: q.as_slice(),
                cnf: none.get(0),
                view: None,
            };
            let c = st.search(&plan, ix, &query, l, &mut ids, &mut ds, None);
            n_dist[li] += c.n_dist as f64;
            // Tie-aware recall@K': answers within the K'-th true distance.
            for (row, &kp) in recall.iter_mut().zip(&ks) {
                let bound = d_gt[kp - 1];
                let hits = ds[..kp].iter().filter(|&&d| d <= bound).count();
                row[li] += hits as f64 / kp as f64;
            }
        }
        for x in recall
            .iter_mut()
            .map(|row| &mut row[li])
            .chain([&mut n_dist[li]])
        {
            *x /= m;
        }
        if recall.iter().all(|row| row[li] >= STOP_RECALL) {
            measured = li + 1;
            break;
        }
    }
    ls.truncate(measured);
    n_dist.truncate(measured);
    for row in &mut recall {
        row.truncate(measured);
    }
    let lsf: Vec<f64> = ls.iter().map(|&l| l as f64).collect();
    let top = lsf[lsf.len() - 1];
    let cap = (EXTRAPOLATION_CAP * top).min(n as f64);
    let table = fit_table(&lsf, &recall, &R_GRID, cap);
    let extrapolated = table.iter().flatten().filter(|&&l| l > top).count();
    let by_k: serde_json::Map<String, Value> = ks
        .iter()
        .zip(&recall)
        .map(|(k, r)| (k.to_string(), json!(r)))
        .collect();
    let doc = json!({
        "queries": pairs.len(),
        "seed": opts.seed,
        "given_pairs": opts.pairs.is_some(),
        "kmax": kmax,
        "l": ls,
        "n_dist": n_dist,
        "recall": by_k,
        "extrapolated_entries": extrapolated,
        "extrapolation_cap": cap,
        "ground_truth_s": t_gt,
        "elapsed_s": t0.elapsed().as_secs_f64(),
    });
    Ok(Effort {
        r: R_GRID.to_vec(),
        k: ks.iter().map(|&k| k as f64).collect(),
        l: table,
        doc,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_interpolates_and_extrapolates_the_miss_rate() {
        let ls = [10.0, 100.0, 1000.0];
        // Recall@K' rising, with a dip that the running maximum removes.
        let recall = vec![vec![0.5, 0.9, 0.99], vec![0.6, 0.55, 0.9]];
        let t = fit_table(&ls, &recall, &[0.5, 0.7, 0.95, 0.999], 8000.0);
        assert_eq!(t[0][0], 10.0);
        // 0.7 between 0.5 at 10 and 0.9 at 100: halfway in log L.
        assert!((t[1][0] - 10f64.powf(1.5)).abs() < 1e-9);
        // The second curve stays at 0.6 until 100, then 0.9 at 1,000.
        assert!((t[1][1] - 100.0 * 10f64.powf(1.0 / 3.0)).abs() < 1e-6);
        // Above the measured range: the miss rate falls ×10 per ×10, so 0.001 at 10,000 (capped).
        assert_eq!(t[3][0], 8000.0);
        let t = fit_table(&ls, &recall, &[0.999], 1e9);
        assert!((t[0][0] - 10_000.0).abs() < 1e-6, "{}", t[0][0]);
        // No progress at the end: the cap.
        let t = fit_table(&ls, &[vec![0.5, 0.8, 0.8]], &[0.9], 5000.0);
        assert_eq!(t[0][0], 5000.0);
    }

    #[test]
    fn midpoints_round_half_up() {
        assert_eq!(midpoint::<u8>(&[1, 2, 255], &[2, 2, 254]), vec![2, 2, 255]);
        assert_eq!(midpoint::<f32>(&[1.0, -3.0], &[2.0, 1.0]), vec![1.5, -1.0]);
    }

    #[test]
    fn ground_truth_keeps_the_smallest_distances_in_order() {
        let base = Matrix::<u8>::from_slice(&[0, 0, 3, 0, 1, 0, 5, 5, 2, 0], 5, 2);
        let gt = ground_truth(&base, &[vec![0u8, 0], vec![5, 5]], 3);
        assert_eq!(gt[0], vec![0.0, 1.0, 4.0]);
        assert_eq!(gt[1], vec![0.0, 29.0, 34.0]);
    }
}
