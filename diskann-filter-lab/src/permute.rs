/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Isomorphic relabelings of the data points (id layouts).
//!
//! A [`Permutation`] gives every data point a new id; the permuted copies of the id-indexed
//! artifacts (base vectors, labels, graph, start table, sketches, ground truth) describe the
//! same index under the new ids. Adjacency rows keep their order, so a search over the copies
//! expands the same nodes in the same order (the beam breaks key ties by insertion order, not
//! by id); only answers tied in distance with the `K`-th may be chosen differently, as those
//! ties are broken by id. What changes is the memory locality: which nodes share a cache line,
//! a page or a DRAM row.
//!
//! Orders of the data points ([`Order`]):
//! - `bfs`: breadth-first from the graph's first start, neighbors in row order (unreached
//!   points last, by old id);
//! - `kmeans`: a k-means tree: the points are split into `fanout` clusters (centers trained on
//!   a sample), each cluster recursively until at most `leaf` points remain; sibling clusters
//!   follow a nearest-neighbor chain over their centers and a leaf keeps the old id order.
//!   Points that are close in space (and hence graph neighbors) get close ids.
//!
//! Auxiliary nodes (ids from `n_data` on) keep their ids.

use anyhow::{ensure, Result};

use crate::{
    graph::Graph,
    parallel::par_map,
    vectors::{DType, Element, Matrix},
};

/// A permutation of the data-point ids `0..n`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Permutation {
    new_of_old: Vec<u32>,
    old_of_new: Vec<u32>,
}

impl Permutation {
    /// The permutation that lists the old ids in their new order (`order[new] = old`).
    pub fn from_order(order: Vec<u32>) -> Result<Self> {
        let n = order.len();
        ensure!(u32::try_from(n).is_ok(), "too many points for 32-bit ids");
        let mut new_of_old = vec![u32::MAX; n];
        for (new, &old) in order.iter().enumerate() {
            ensure!(
                (old as usize) < n && new_of_old[old as usize] == u32::MAX,
                "the order is not a permutation of 0..{n} (entry {old} at {new})"
            );
            new_of_old[old as usize] = new as u32;
        }
        Ok(Self {
            new_of_old,
            old_of_new: order,
        })
    }

    /// The identity on `0..n`.
    pub fn identity(n: usize) -> Self {
        let ids: Vec<u32> = (0..n as u32).collect();
        Self {
            new_of_old: ids.clone(),
            old_of_new: ids,
        }
    }

    /// Number of data points.
    pub fn len(&self) -> usize {
        self.old_of_new.len()
    }

    pub fn is_empty(&self) -> bool {
        self.old_of_new.is_empty()
    }

    /// New id of node `old` (auxiliary nodes keep their ids).
    #[inline]
    pub fn new_id(&self, old: u32) -> u32 {
        self.new_of_old.get(old as usize).copied().unwrap_or(old)
    }

    /// Old id of node `new` (auxiliary nodes keep their ids).
    #[inline]
    pub fn old_id(&self, new: u32) -> u32 {
        self.old_of_new.get(new as usize).copied().unwrap_or(new)
    }

    /// `new_of_old[old]`.
    pub fn new_of_old(&self) -> &[u32] {
        &self.new_of_old
    }

    /// `old_of_new[new]` (the order).
    pub fn old_of_new(&self) -> &[u32] {
        &self.old_of_new
    }

    /// The inverse permutation.
    pub fn inverse(&self) -> Self {
        Self {
            new_of_old: self.old_of_new.clone(),
            old_of_new: self.new_of_old.clone(),
        }
    }
}

/// Orders of the data points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    Identity,
    Bfs,
    Kmeans {
        fanout: usize,
        leaf: usize,
        seed: u64,
    },
}

impl Order {
    /// Name with parameters (e.g. `kmeans(64,256)`).
    pub fn name(&self) -> String {
        match self {
            Self::Identity => "identity".to_owned(),
            Self::Bfs => "bfs".to_owned(),
            Self::Kmeans { fanout, leaf, .. } => format!("kmeans({fanout},{leaf})"),
        }
    }

    /// The permutation of the data points of `graph` over `base`.
    pub fn permutation<T: Element>(
        &self,
        graph: &Graph,
        base: &Matrix<T>,
        threads: usize,
    ) -> Result<Permutation> {
        ensure!(
            base.nrows() == graph.n_data(),
            "the base has {} vectors, the graph {} data points",
            base.nrows(),
            graph.n_data()
        );
        match *self {
            Self::Identity => Ok(Permutation::identity(base.nrows())),
            Self::Bfs => Permutation::from_order(bfs_order(graph)),
            Self::Kmeans { fanout, leaf, seed } => {
                ensure!(
                    fanout >= 2 && leaf >= 1,
                    "kmeans needs fanout >= 2, leaf >= 1"
                );
                Permutation::from_order(kmeans_order(base, fanout, leaf, seed, threads))
            }
        }
    }
}

/// Breadth-first order of the data points from the graph's first start (auxiliary nodes are
/// traversed but not listed); unreached points follow in id order.
pub fn bfs_order(graph: &Graph) -> Vec<u32> {
    let (n_nodes, n_data) = (graph.n_nodes(), graph.n_data());
    let mut seen = vec![false; n_nodes];
    let mut queue: Vec<u32> = Vec::with_capacity(n_nodes);
    for &s in graph.starts().iter().take(1) {
        seen[s as usize] = true;
        queue.push(s);
    }
    let mut head = 0;
    while head < queue.len() {
        let u = queue[head];
        head += 1;
        for &v in graph.neighbors(u) {
            if !seen[v as usize] {
                seen[v as usize] = true;
                queue.push(v);
            }
        }
    }
    let mut order: Vec<u32> = queue
        .into_iter()
        .filter(|&v| (v as usize) < n_data)
        .collect();
    order.extend((0..n_data as u32).filter(|&v| !seen[v as usize]));
    order
}

/// SplitMix64 step.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn to_f32<T: Element>(x: &[T], out: &mut [f32]) {
    match T::DTYPE {
        DType::U8 => {
            for (o, &b) in out.iter_mut().zip(bytemuck::cast_slice::<T, u8>(x)) {
                *o = f32::from(b);
            }
        }
        DType::F32 => out.copy_from_slice(bytemuck::cast_slice::<T, f32>(x)),
    }
}

/// Points per center trained on (the sample of a split).
const TRAIN_PER_CENTER: usize = 64;
/// Lloyd iterations of a split.
const ITERS: usize = 8;
/// Points per assignment task of a parallel split.
const CHUNK: usize = 1 << 16;

/// The groups of a split of `pts` into at most `k` clusters, in nearest-neighbor chain order
/// of their centers (empty clusters dropped; each group keeps the order of `pts`).
fn split<T: Element>(
    base: &Matrix<T>,
    pts: &[u32],
    k: usize,
    seed: u64,
    threads: usize,
) -> Vec<Vec<u32>> {
    let dim = base.ncols();
    let n = pts.len();
    // Training sample: every point of a small set, else a pseudo-random sample.
    let m = (k * TRAIN_PER_CENTER).min(n);
    let mut rng = seed;
    let sample: Vec<u32> = if m == n {
        pts.to_vec()
    } else {
        let mut idx: Vec<usize> = (0..n).collect();
        for i in 0..m {
            rng = mix(rng);
            let j = i + (rng % (n - i) as u64) as usize;
            idx.swap(i, j);
        }
        idx[..m].iter().map(|&i| pts[i]).collect()
    };
    let mut xs = vec![0f32; m * dim];
    for (i, &p) in sample.iter().enumerate() {
        to_f32(base.row(p as usize), &mut xs[i * dim..(i + 1) * dim]);
    }
    let k = k.min(m);
    // Initial centers: k distinct sample entries (the sample is already shuffled).
    let mut centers = xs[..k * dim].to_vec();
    let mut assign = vec![0usize; m];
    for _ in 0..ITERS {
        for (i, a) in assign.iter_mut().enumerate() {
            *a = nearest(&xs[i * dim..(i + 1) * dim], &centers, dim);
        }
        let mut sums = vec![0f64; k * dim];
        let mut counts = vec![0usize; k];
        for (i, &a) in assign.iter().enumerate() {
            counts[a] += 1;
            for (s, &x) in sums[a * dim..(a + 1) * dim]
                .iter_mut()
                .zip(&xs[i * dim..(i + 1) * dim])
            {
                *s += f64::from(x);
            }
        }
        for j in 0..k {
            if counts[j] > 0 {
                for (c, &s) in centers[j * dim..(j + 1) * dim]
                    .iter_mut()
                    .zip(&sums[j * dim..(j + 1) * dim])
                {
                    *c = (s / counts[j] as f64) as f32;
                }
            }
        }
    }
    // Assign every point.
    let assign_range = |r: std::ops::Range<usize>| -> Vec<u32> {
        let mut x = vec![0f32; dim];
        r.map(|i| {
            to_f32(base.row(pts[i] as usize), &mut x);
            nearest(&x, &centers, dim) as u32
        })
        .collect()
    };
    let labels: Vec<u32> = if threads > 1 && n > CHUNK {
        par_map(
            n.div_ceil(CHUNK),
            threads,
            || (),
            |c, _| assign_range(c * CHUNK..((c + 1) * CHUNK).min(n)),
        )
        .into_iter()
        .flatten()
        .collect()
    } else {
        assign_range(0..n)
    };
    let mut groups: Vec<Vec<u32>> = vec![Vec::new(); k];
    for (&p, &a) in pts.iter().zip(&labels) {
        groups[a as usize].push(p);
    }
    // Nearest-neighbor chain over the non-empty clusters' centers, from the first.
    let live: Vec<usize> = (0..k).filter(|&j| !groups[j].is_empty()).collect();
    let mut chain = Vec::with_capacity(live.len());
    let mut left = live;
    if !left.is_empty() {
        let mut cur = left.remove(0);
        chain.push(cur);
        while !left.is_empty() {
            let c = &centers[cur * dim..(cur + 1) * dim];
            let (pos, _) = left
                .iter()
                .enumerate()
                .map(|(i, &j)| (i, sq(c, &centers[j * dim..(j + 1) * dim])))
                .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
                .expect("non-empty");
            cur = left.remove(pos);
            chain.push(cur);
        }
    }
    chain
        .into_iter()
        .map(|j| std::mem::take(&mut groups[j]))
        .collect()
}

fn nearest(x: &[f32], centers: &[f32], dim: usize) -> usize {
    let mut best = (0, f32::INFINITY);
    for (j, c) in centers.chunks_exact(dim).enumerate() {
        let d = sq(x, c);
        if d < best.1 {
            best = (j, d);
        }
    }
    best.0
}

/// Appends the k-means tree order of `pts` to `out`.
fn order_rec<T: Element>(
    base: &Matrix<T>,
    pts: &[u32],
    fanout: usize,
    leaf: usize,
    seed: u64,
    out: &mut Vec<u32>,
) {
    if pts.len() <= leaf {
        out.extend_from_slice(pts);
        return;
    }
    let groups = split(base, pts, fanout, seed, 1);
    if groups.len() <= 1 {
        // Indistinguishable points: keep their order.
        out.extend_from_slice(pts);
        return;
    }
    for (i, g) in groups.iter().enumerate() {
        order_rec(base, g, fanout, leaf, mix(seed ^ i as u64), out);
    }
}

/// k-means tree order of all data points (see the module documentation): the top split's
/// assignment and the clusters below it run on `threads` threads.
pub fn kmeans_order<T: Element>(
    base: &Matrix<T>,
    fanout: usize,
    leaf: usize,
    seed: u64,
    threads: usize,
) -> Vec<u32> {
    let n = base.nrows();
    let all: Vec<u32> = (0..n as u32).collect();
    if n <= leaf {
        return all;
    }
    let groups = split(base, &all, fanout, mix(seed), threads);
    drop(all);
    par_map(
        groups.len(),
        threads,
        || (),
        |i, _| {
            let mut out = Vec::with_capacity(groups[i].len());
            order_rec(
                base,
                &groups[i],
                fanout,
                leaf,
                mix(seed ^ (i as u64 + 1)),
                &mut out,
            );
            out
        },
    )
    .into_iter()
    .flatten()
    .collect()
}

/// The rows of `m` in the new order.
pub fn permute_rows<T: Element>(m: &Matrix<T>, p: &Permutation) -> Matrix<T> {
    let mut out = Matrix::zeros(m.nrows(), m.ncols());
    for (new, &old) in p.old_of_new().iter().enumerate() {
        out.row_mut(new).copy_from_slice(m.row(old as usize));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::AuxNodes;

    #[test]
    fn permutations_validate_and_invert() {
        let p = Permutation::from_order(vec![2, 0, 1]).unwrap();
        assert_eq!(p.new_of_old(), &[1, 2, 0]);
        assert_eq!((p.new_id(2), p.old_id(0), p.new_id(7)), (0, 2, 7));
        assert_eq!(p.inverse().inverse(), p);
        assert!(Permutation::from_order(vec![0, 0, 1]).is_err());
        assert!(Permutation::from_order(vec![0, 3, 1]).is_err());
    }

    fn line_graph(n: usize) -> (Graph, Matrix<u8>) {
        let data: Vec<u8> = (0..n)
            .flat_map(|i| [(i * 7 % 251) as u8, (i % 13) as u8])
            .collect();
        let base = Matrix::from_slice(&data, n, 2);
        let mut g = Graph::new(
            n,
            2,
            AuxNodes::new(Vec::new(), &Matrix::<u8>::zeros(0, 2)).unwrap(),
        )
        .unwrap();
        for i in 0..n {
            let nb: Vec<u32> = [i.wrapping_sub(1), i + 1]
                .into_iter()
                .filter(|&j| j < n)
                .map(|j| j as u32)
                .collect();
            g.set_neighbors(i as u32, &nb).unwrap();
        }
        g.set_starts(vec![(n / 2) as u32]).unwrap();
        (g, base)
    }

    #[test]
    fn bfs_starts_at_the_start_and_lists_every_point() {
        let (g, base) = line_graph(9);
        let order = bfs_order(&g);
        assert_eq!(order, vec![4, 3, 5, 2, 6, 1, 7, 0, 8]);
        let p = Order::Bfs.permutation(&g, &base, 1).unwrap();
        assert_eq!(p.old_of_new(), order.as_slice());
    }

    #[test]
    fn kmeans_order_is_a_permutation_that_groups_close_points() {
        // Two far-apart blobs, interleaved in id order: the order separates them.
        let n = 400;
        let data: Vec<u8> = (0..n)
            .flat_map(|i| {
                let off = if i % 2 == 0 { 0 } else { 200 };
                [(off + i % 17) as u8, (off + i % 11) as u8]
            })
            .collect();
        let base = Matrix::from_slice(&data, n, 2);
        for threads in [1, 3] {
            let order = kmeans_order(&base, 2, 16, 5, threads);
            let p = Permutation::from_order(order.clone()).unwrap();
            assert_eq!(p.len(), n);
            let first_half: Vec<u32> = order[..n / 2].iter().map(|&v| v % 2).collect();
            assert!(
                first_half.iter().all(|&b| b == first_half[0]),
                "the blobs are contiguous"
            );
            assert_eq!(
                order,
                kmeans_order(&base, 2, 16, 5, threads),
                "deterministic"
            );
        }
    }

    #[test]
    fn permuted_rows_follow_the_order() {
        let (_, base) = line_graph(5);
        let p = Permutation::from_order(vec![4, 0, 3, 1, 2]).unwrap();
        let m = permute_rows(&base, &p);
        for new in 0..5 {
            assert_eq!(m.row(new), base.row(p.old_id(new as u32) as usize));
        }
    }
}
