/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Per-label start nodes computed from the base vectors (`flab starts build`), so that
//! label-derived starts exist for any label store on any graph over the same points (e.g.
//! label-blind graphs and synthetic label sets). Strategies:
//!
//! - `medoid`: the member of the label closest to the label's centroid. Labels with at most
//!   `exact_max` members are processed exactly; larger labels use a seeded sample of `sample`
//!   members for both the centroid and the candidates;
//! - `random(k)`: `k` distinct members drawn uniformly (seeded), in random order;
//! - `kmeans(k)`: up to `k` representatives: k-means++ seeding and `iters` Lloyd iterations on
//!   the members (or on the seeded sample of a larger label), then, for each non-empty cluster,
//!   the member closest to its centroid; largest cluster first. Labels with at most `k`
//!   members get all of them.
//!
//! Distances are squared L2 in `f32` against `f32` centroids (`u8` rows are converted). Every
//! choice depends only on the vectors, the postings, the parameters and the seed (one random
//! stream per label and strategy), not on the number of threads; ties go to the smaller id.

use std::fmt;

use anyhow::{bail, ensure, Context, Result};

use crate::{
    labels::LabelStore,
    parallel,
    vectors::{DType, Element, Matrix},
};

/// How the starts of a label are chosen (see the module documentation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Strategy {
    /// The member closest to the centroid.
    Medoid,
    /// `k` random members.
    Random(usize),
    /// Up to `k` k-means representatives.
    KMeans(usize),
}

impl Strategy {
    /// Parses `medoid`, `random`, `random(k)`, `kmeans` (k = 4) or `kmeans(k)`.
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        let (name, arg) = match s.find('(') {
            None => (s, None),
            Some(i) => {
                ensure!(s.ends_with(')'), "unbalanced parentheses in {s:?}");
                (s[..i].trim(), Some(s[i + 1..s.len() - 1].trim()))
            }
        };
        let k = |default: usize| -> Result<usize> {
            match arg {
                None => Ok(default),
                Some(a) => a
                    .parse()
                    .ok()
                    .filter(|&k| k >= 1)
                    .with_context(|| format!("bad k {a:?} in {s:?} (an integer >= 1)")),
            }
        };
        Ok(match name {
            "medoid" => {
                ensure!(arg.is_none(), "medoid takes no argument");
                Self::Medoid
            }
            "random" => Self::Random(k(1)?),
            "kmeans" => Self::KMeans(k(4)?),
            _ => bail!("unknown strategy {s:?} (medoid, random(k), kmeans(k))"),
        })
    }

    /// File-name tag: `medoid`, `random` (k = 1), `random<k>`, `kmeans<k>`.
    pub fn tag(self) -> String {
        match self {
            Self::Medoid => "medoid".into(),
            Self::Random(1) => "random".into(),
            Self::Random(k) => format!("random{k}"),
            Self::KMeans(k) => format!("kmeans{k}"),
        }
    }

    /// Salt of the per-label random streams.
    fn salt(self) -> u64 {
        match self {
            Self::Medoid => 0x6d65_646f_6964,
            Self::Random(k) => 0x7261_6e64_0000 ^ k as u64,
            Self::KMeans(k) => 0x6b6d_6e73_0000 ^ k as u64,
        }
    }
}

impl fmt::Display for Strategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Medoid => write!(f, "medoid"),
            Self::Random(k) => write!(f, "random({k})"),
            Self::KMeans(k) => write!(f, "kmeans({k})"),
        }
    }
}

/// Parameters of [`build`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Params {
    /// Labels with at most this many members use all of them (`medoid`, `kmeans`).
    pub exact_max: usize,
    /// Members sampled from larger labels.
    pub sample: usize,
    /// Lloyd iterations of `kmeans(k)`.
    pub iters: usize,
    pub seed: u64,
    pub threads: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            exact_max: 10_000,
            sample: 10_000,
            iters: 10,
            seed: 0,
            threads: 1,
        }
    }
}

/// Per-label starts computed by [`build`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelStarts {
    /// Start nodes (data points carrying the label) per label id, in priority order.
    pub starts: Vec<Vec<u32>>,
    /// Labels whose members were sampled (`medoid`, `kmeans`).
    pub sampled: usize,
    /// Distance computations.
    pub n_dist: u64,
}

impl LabelStarts {
    /// `(label, node)` pairs, labels in id order (for [`crate::starts::Starts::set_label_starts`]).
    pub fn pairs(&self) -> Vec<(u32, u32)> {
        self.starts
            .iter()
            .enumerate()
            .flat_map(|(l, s)| s.iter().map(move |&v| (l as u32, v)))
            .collect()
    }
}

/// SplitMix64.
#[derive(Debug, Clone)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64, label: u32, salt: u64) -> Self {
        let mut r = Self(
            seed ^ salt.rotate_left(29) ^ u64::from(label).wrapping_mul(0xd6e8_feb8_6659_fd93),
        );
        r.next();
        r
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (`n > 0`).
    fn below(&mut self, n: usize) -> usize {
        ((u128::from(self.next()) * n as u128) >> 64) as usize
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Selection sampling (Knuth's algorithm S): `s` members of the sorted list `members`,
/// uniformly, in increasing order (all of them if `s >= members.len()`).
fn sample_sorted(members: &[u32], s: usize, rng: &mut Rng, out: &mut Vec<u32>) {
    out.clear();
    let n = members.len();
    if s >= n {
        out.extend_from_slice(members);
        return;
    }
    let mut need = s;
    for (i, &m) in members.iter().enumerate() {
        if rng.below(n - i) < need {
            out.push(m);
            need -= 1;
            if need == 0 {
                break;
            }
        }
    }
}

/// Copies a row into `out` as `f32`.
#[inline]
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

/// Squared L2 distance between a row and an `f32` vector (`buf`: scratch of the dimension).
#[inline]
fn dist_to<T: Element>(x: &[T], c: &[f32], buf: &mut [f32]) -> f32 {
    match T::DTYPE {
        DType::F32 => f32::squared_l2(bytemuck::cast_slice::<T, f32>(x), c),
        DType::U8 => {
            to_f32(x, buf);
            f32::squared_l2(buf, c)
        }
    }
}

/// Adds a row to `acc`.
#[inline]
fn accumulate<T: Element>(x: &[T], acc: &mut [f64]) {
    match T::DTYPE {
        DType::U8 => {
            for (a, &b) in acc.iter_mut().zip(bytemuck::cast_slice::<T, u8>(x)) {
                *a += f64::from(b);
            }
        }
        DType::F32 => {
            for (a, &b) in acc.iter_mut().zip(bytemuck::cast_slice::<T, f32>(x)) {
                *a += f64::from(b);
            }
        }
    }
}

/// Per-thread scratch.
#[derive(Debug, Default)]
struct Scratch {
    cand: Vec<u32>,
    acc: Vec<f64>,
    centroid: Vec<f32>,
    row: Vec<f32>,
    pts: Vec<f32>,
    sums: Vec<f64>,
    centers: Vec<f32>,
    d2: Vec<f32>,
    assign: Vec<u32>,
    counts: Vec<u64>,
    best: Vec<(f32, u32)>,
}

/// The candidate closest to `c` (ties: the first, i.e. the smallest id of a sorted list).
fn nearest<T: Element>(
    base: &Matrix<T>,
    cand: &[u32],
    c: &[f32],
    buf: &mut [f32],
    n_dist: &mut u64,
) -> u32 {
    let mut best = (f32::INFINITY, u32::MAX);
    for &p in cand {
        let d = dist_to(base.row(p as usize), c, buf);
        if d < best.0 || best.1 == u32::MAX {
            best = (d, p);
        }
    }
    *n_dist += cand.len() as u64;
    best.1
}

fn medoid<T: Element>(base: &Matrix<T>, cand: &[u32], s: &mut Scratch, n_dist: &mut u64) -> u32 {
    s.acc.fill(0.0);
    for &p in cand {
        accumulate(base.row(p as usize), &mut s.acc);
    }
    let inv = 1.0 / cand.len() as f64;
    for (c, a) in s.centroid.iter_mut().zip(&s.acc) {
        *c = (*a * inv) as f32;
    }
    nearest(base, cand, &s.centroid, &mut s.row, n_dist)
}

/// Index of the center closest to `x` among the `k` rows of `centers`.
#[inline]
fn closest_center(x: &[f32], centers: &[f32], dim: usize) -> (usize, f32) {
    let mut best = (0, f32::INFINITY);
    for (j, c) in centers.chunks_exact(dim).enumerate() {
        let d = f32::squared_l2(x, c);
        if d < best.1 {
            best = (j, d);
        }
    }
    best
}

/// k-means representatives of `cand` (more than `k` sorted members).
fn kmeans<T: Element>(
    base: &Matrix<T>,
    cand: &[u32],
    k: usize,
    iters: usize,
    rng: &mut Rng,
    s: &mut Scratch,
    n_dist: &mut u64,
) -> Vec<u32> {
    let dim = base.ncols();
    let n = cand.len();
    s.pts.resize(n * dim, 0.0);
    for (i, &p) in cand.iter().enumerate() {
        to_f32(base.row(p as usize), &mut s.pts[i * dim..(i + 1) * dim]);
    }
    let pts = &s.pts[..n * dim];
    let row = |i: usize| &pts[i * dim..(i + 1) * dim];

    // k-means++ seeding (stops early when every point coincides with a center).
    s.centers.clear();
    let first = rng.below(n);
    s.centers.extend_from_slice(row(first));
    s.d2.clear();
    s.d2.extend((0..n).map(|i| f32::squared_l2(row(i), row(first))));
    *n_dist += n as u64;
    while s.centers.len() < k * dim {
        let total: f64 = s.d2.iter().map(|&x| f64::from(x)).sum();
        if total <= 0.0 {
            break;
        }
        let mut r = rng.unit() * total;
        let mut pick = usize::MAX;
        for (i, &x) in s.d2.iter().enumerate() {
            r -= f64::from(x);
            if r < 0.0 && x > 0.0 {
                pick = i;
                break;
            }
        }
        if pick == usize::MAX {
            // Rounding: take the farthest point.
            pick = (0..n)
                .max_by(|&a, &b| s.d2[a].total_cmp(&s.d2[b]).then(b.cmp(&a)))
                .unwrap_or(0);
        }
        let start = s.centers.len();
        s.centers.extend_from_slice(row(pick));
        for i in 0..n {
            let d = f32::squared_l2(row(i), &s.centers[start..start + dim]);
            if d < s.d2[i] {
                s.d2[i] = d;
            }
        }
        *n_dist += n as u64;
    }
    let kk = s.centers.len() / dim;

    // Lloyd iterations; an empty cluster keeps its center.
    s.assign.clear();
    s.assign.resize(n, u32::MAX);
    for _ in 0..iters {
        let mut changed = false;
        for i in 0..n {
            let (j, _) = closest_center(row(i), &s.centers, dim);
            if s.assign[i] != j as u32 {
                s.assign[i] = j as u32;
                changed = true;
            }
        }
        *n_dist += (n * kk) as u64;
        if !changed {
            break;
        }
        s.sums.clear();
        s.sums.resize(kk * dim, 0.0);
        s.counts.clear();
        s.counts.resize(kk, 0);
        for i in 0..n {
            let j = s.assign[i] as usize;
            s.counts[j] += 1;
            for (a, &x) in s.sums[j * dim..(j + 1) * dim].iter_mut().zip(row(i)) {
                *a += f64::from(x);
            }
        }
        for j in 0..kk {
            if s.counts[j] > 0 {
                let inv = 1.0 / s.counts[j] as f64;
                for (c, &a) in s.centers[j * dim..(j + 1) * dim]
                    .iter_mut()
                    .zip(&s.sums[j * dim..(j + 1) * dim])
                {
                    *c = (a * inv) as f32;
                }
            }
        }
    }

    // Final assignment: per cluster its size and the member closest to its center.
    s.counts.clear();
    s.counts.resize(kk, 0);
    s.best.clear();
    s.best.resize(kk, (f32::INFINITY, u32::MAX));
    for (i, &p) in cand.iter().enumerate() {
        let (j, d) = closest_center(row(i), &s.centers, dim);
        s.counts[j] += 1;
        if d < s.best[j].0 || s.best[j].1 == u32::MAX {
            s.best[j] = (d, p);
        }
    }
    *n_dist += (n * kk) as u64;
    let mut clusters: Vec<(u64, u32)> = (0..kk)
        .filter(|&j| s.counts[j] > 0)
        .map(|j| (s.counts[j], s.best[j].1))
        .collect();
    clusters.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut out: Vec<u32> = Vec::with_capacity(clusters.len());
    for (_, p) in clusters {
        if !out.contains(&p) {
            out.push(p);
        }
    }
    out
}

/// Starts of one label.
fn one<T: Element>(
    base: &Matrix<T>,
    members: &[u32],
    label: u32,
    strategy: Strategy,
    p: &Params,
    s: &mut Scratch,
) -> (Vec<u32>, bool, u64) {
    let mut n_dist = 0;
    if members.is_empty() {
        return (Vec::new(), false, 0);
    }
    let mut rng = Rng::new(p.seed, label, strategy.salt());
    let sampled = members.len() > p.exact_max;
    let mut cand = std::mem::take(&mut s.cand);
    let take = if sampled { p.sample } else { members.len() };
    let out = match strategy {
        Strategy::Medoid => {
            sample_sorted(members, take, &mut rng, &mut cand);
            vec![medoid(base, &cand, s, &mut n_dist)]
        }
        Strategy::Random(k) => {
            sample_sorted(members, k, &mut rng, &mut cand);
            for i in (1..cand.len()).rev() {
                cand.swap(i, rng.below(i + 1));
            }
            cand.clone()
        }
        Strategy::KMeans(k) => {
            sample_sorted(members, take, &mut rng, &mut cand);
            if cand.len() <= k {
                cand.clone()
            } else {
                kmeans(base, &cand, k, p.iters, &mut rng, s, &mut n_dist)
            }
        }
    };
    s.cand = cand;
    let sampled = sampled && !matches!(strategy, Strategy::Random(_));
    (out, sampled, n_dist)
}

/// Computes the per-label starts of every label of `labels` from the base vectors.
pub fn build<T: Element>(
    base: &Matrix<T>,
    labels: &LabelStore,
    strategy: Strategy,
    p: &Params,
) -> Result<LabelStarts> {
    ensure!(
        base.nrows() == labels.n_points(),
        "{} base vectors for {} label rows",
        base.nrows(),
        labels.n_points()
    );
    ensure!(
        p.exact_max >= 1 && p.sample >= 1,
        "exact_max and sample must be at least 1"
    );
    let dim = base.ncols();
    let per_label = parallel::par_map(
        labels.n_labels(),
        p.threads,
        || Scratch {
            acc: vec![0.0; dim],
            centroid: vec![0.0; dim],
            row: vec![0.0; dim],
            ..Scratch::default()
        },
        |l, s| {
            let l = l as u32;
            one(base, labels.postings(l), l, strategy, p, s)
        },
    );
    let mut out = LabelStarts {
        starts: Vec::with_capacity(per_label.len()),
        sampled: 0,
        n_dist: 0,
    };
    for (starts, sampled, n_dist) in per_label {
        out.starts.push(starts);
        out.sampled += usize::from(sampled);
        out.n_dist += n_dist;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::labels::LabelStoreBuilder;

    /// 2-d points: label `c=a` on a cross around (10, 10) plus a far outlier, `c=b` on two
    /// separated clusters, `c=z` on one point.
    fn data() -> (Matrix<u8>, LabelStore) {
        let pts: [([u8; 2], &[&str]); 12] = [
            ([10, 10], &["c=a"]),
            ([9, 10], &["c=a"]),
            ([11, 10], &["c=a"]),
            ([10, 9], &["c=a"]),
            ([10, 11], &["c=a"]),
            ([60, 60], &["c=a"]),
            ([0, 0], &["c=b"]),
            ([1, 0], &["c=b"]),
            ([0, 1], &["c=b"]),
            ([200, 200], &["c=b"]),
            ([201, 200], &["c=b", "c=z"]),
            ([200, 201], &["c=b"]),
        ];
        let mut m = Matrix::<u8>::zeros(pts.len(), 2);
        let mut b = LabelStoreBuilder::new();
        for (i, (x, row)) in pts.iter().enumerate() {
            m.row_mut(i).copy_from_slice(x);
            b.push_point(row.iter().map(|&l| ("c", l))).unwrap();
        }
        (m, b.finish().unwrap())
    }

    #[test]
    fn strategies_parse_and_tag() {
        for (s, tag) in [
            ("medoid", "medoid"),
            ("random", "random"),
            ("random(3)", "random3"),
            ("kmeans", "kmeans4"),
            (" kmeans( 2 ) ", "kmeans2"),
        ] {
            let st = Strategy::parse(s).unwrap();
            assert_eq!(st.tag(), tag);
            assert_eq!(Strategy::parse(&st.to_string()).unwrap(), st);
        }
        for bad in [
            "medoid(2)",
            "random(0)",
            "kmeans(x)",
            "centroid",
            "kmeans(2",
        ] {
            assert!(Strategy::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn medoids_are_members_closest_to_the_centroid() {
        let (base, labels) = data();
        let id = |n: &str| labels.name_index()[n] as usize;
        let p = Params::default();
        let s = build(&base, &labels, Strategy::Medoid, &p).unwrap();
        // c=a: centroid (18.3, 18.3) is closest to (11, 10) / (10, 11) (tie: smaller id 2).
        assert_eq!(s.starts[id("c=a")], [2]);
        assert_eq!(s.starts[id("c=z")], [10]);
        assert_eq!(s.sampled, 0);
        // Every start carries its label, whatever the thread count.
        let t = build(
            &base,
            &labels,
            Strategy::Medoid,
            &Params { threads: 3, ..p },
        )
        .unwrap();
        assert_eq!(s, t);
        for (l, starts) in s.starts.iter().enumerate() {
            for &v in starts {
                assert!(labels.row(v).contains(&(l as u32)));
            }
        }
    }

    #[test]
    fn sampled_medoids_use_members_only() {
        let (base, labels) = data();
        let p = Params {
            exact_max: 2,
            sample: 3,
            ..Params::default()
        };
        let s = build(&base, &labels, Strategy::Medoid, &p).unwrap();
        assert_eq!(s.sampled, 2, "c=a and c=b are sampled");
        for (l, starts) in s.starts.iter().enumerate() {
            assert_eq!(starts.len(), 1);
            assert!(labels.row(starts[0]).contains(&(l as u32)));
        }
        let again = build(&base, &labels, Strategy::Medoid, &p).unwrap();
        assert_eq!(s, again, "seeded");
    }

    #[test]
    fn random_members_are_distinct_and_seeded() {
        let (base, labels) = data();
        let id = |n: &str| labels.name_index()[n] as usize;
        let p = Params::default();
        let s = build(&base, &labels, Strategy::Random(4), &p).unwrap();
        let a = &s.starts[id("c=a")];
        assert_eq!(a.len(), 4);
        let mut sorted = a.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 4);
        assert!(sorted.iter().all(|&v| v <= 5));
        assert_eq!(s.starts[id("c=z")], [10], "fewer members than k: all");
        let other = build(
            &base,
            &labels,
            Strategy::Random(4),
            &Params { seed: 7, ..p },
        )
        .unwrap();
        let t = build(
            &base,
            &labels,
            Strategy::Random(4),
            &Params { seed: 7, ..p },
        )
        .unwrap();
        assert_eq!(other, t);
    }

    #[test]
    fn kmeans_finds_one_representative_per_cluster() {
        let (base, labels) = data();
        let id = |n: &str| labels.name_index()[n] as usize;
        let s = build(&base, &labels, Strategy::KMeans(2), &Params::default()).unwrap();
        let mut b = s.starts[id("c=b")].clone();
        assert_eq!(b.len(), 2);
        b.sort_unstable();
        assert!(b[0] <= 8 && (9..=11).contains(&b[1]), "{b:?}");
        assert_eq!(s.starts[id("c=z")], [10]);
        // Equal cluster sizes: the representative with the smaller id comes first.
        assert_eq!(s.starts[id("c=b")][0], b[0]);
        // c=a: the large cluster (5 points around (10, 10)) comes first.
        let a = &s.starts[id("c=a")];
        assert_eq!(a.len(), 2);
        assert!(a[0] <= 4 && a[1] == 5, "{a:?}");
    }

    #[test]
    fn pairs_follow_label_order() {
        let ls = LabelStarts {
            starts: vec![vec![3, 1], vec![], vec![7]],
            sampled: 0,
            n_dist: 0,
        };
        assert_eq!(ls.pairs(), [(0, 3), (0, 1), (2, 7)]);
    }

    #[test]
    fn selection_sampling_is_uniform_enough() {
        let members: Vec<u32> = (0..10).collect();
        let mut counts = [0u32; 10];
        let mut rng = Rng::new(1, 2, 3);
        let mut out = Vec::new();
        for _ in 0..20_000 {
            sample_sorted(&members, 3, &mut rng, &mut out);
            assert_eq!(out.len(), 3);
            assert!(out.windows(2).all(|w| w[0] < w[1]));
            for &m in &out {
                counts[m as usize] += 1;
            }
        }
        // Expected 6000 each.
        assert!(
            counts.iter().all(|&c| (5600..6400).contains(&c)),
            "{counts:?}"
        );
    }
}
