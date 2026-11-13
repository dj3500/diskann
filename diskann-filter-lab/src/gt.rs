/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Exact filtered k-nearest-neighbor ground truth with complete distance-tie bands.
//!
//! For a query `q` with predicate `P`, let `M` be the set of points matching `P` and `d_k` the
//! k-th smallest squared L2 distance from `q` to `M`. The *band* of `q` lists every point of
//! `M` within distance `d_K` (all of `M` when `|M| < K`), sorted by (distance, id). Its first
//! `min(K, |M|)` entries are the exact top-K with ties broken by id, and for every `k ≤ K` it
//! contains all points tied at `d_k`.
//!
//! Candidates are the points of the predicate's smallest clause (least total posting length).
//! Before its distance is computed, each candidate is checked against the other clauses,
//! either in the forward store or with per-query bitsets built from postings
//! ([`Evaluation`]); both checks are exact.

use std::{
    cmp::Ordering,
    io::{Read, Seek, Write},
    path::Path,
};

use anyhow::{ensure, Context, Result};
use hashbrown::{HashMap, HashSet};

use crate::{
    container,
    labels::LabelStore,
    parallel::par_map,
    predicate::{intersects, Cnf, PredicateSet},
    vectors::{Element, Matrix},
};

/// File magic of extended ground truth files.
pub const MAGIC: &[u8; 8] = b"FLABGT\0\0";
/// Version of the extended ground truth format.
pub const VERSION: u32 = 1;
/// Stored as the smallest clause of a predicate without clauses.
pub const NO_CLAUSE: u32 = u32::MAX;
/// Metric code in the header: squared Euclidean distance.
pub const METRIC_SQUARED_L2: u64 = 0;

/// Ground truth of a query set (the extended format, see the crate `README.md`).
#[derive(Debug, Clone, PartialEq)]
pub struct GroundTruth {
    k: u32,
    n_points: u32,
    dict_hash: u64,
    n_matches: Vec<u32>,
    min_clause: Vec<u32>,
    min_clause_sum: Vec<u64>,
    min_clause_union: Vec<u32>,
    band_offsets: Vec<u64>,
    band_ids: Vec<u32>,
    band_dists: Vec<f32>,
}

struct QueryTruth {
    n_matches: u32,
    min_clause: u32,
    min_clause_sum: u64,
    min_clause_union: u32,
    band: Vec<(f32, u32)>,
}

fn by_distance_then_id(a: &(f32, u32), b: &(f32, u32)) -> Ordering {
    a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
}

/// The clause with the least total posting length (the first one on ties) and that length;
/// `None` for a predicate without clauses.
pub fn smallest_clause(labels: &LabelStore, cnf: Cnf<'_>) -> Option<(usize, u64)> {
    cnf.clauses()
        .map(|c| c.iter().map(|&l| labels.posting_len(l) as u64).sum::<u64>())
        .enumerate()
        .min_by_key(|&(i, sum)| (sum, i))
}

/// How the candidates of the smallest clause are checked against the other clauses. Both
/// checks are exact and give identical results; they differ in speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Evaluation {
    /// Per query, the check that touches fewer label entries: forward rows (candidates × mean
    /// labels per point) or postings (total posting length of all clauses plus one bitset word
    /// per 64 points and clause).
    #[default]
    Auto,
    /// Search each candidate's forward row for the labels of the other clauses.
    Forward,
    /// Intersect per-query bitsets of the points satisfying each clause (unions of postings).
    Postings,
}

/// Parameters of [`compute`].
#[derive(Debug, Clone, Copy)]
pub struct Params {
    /// Number of neighbors K.
    pub k: usize,
    pub threads: usize,
    pub evaluation: Evaluation,
}

#[derive(Default)]
struct Scratch {
    found: Vec<(f32, u32)>,
    acc: Vec<u64>,
    tmp: Vec<u64>,
}

fn use_postings(labels: &LabelStore, cnf: Cnf<'_>, min_clause_sum: u64, e: Evaluation) -> bool {
    match e {
        Evaluation::Forward => false,
        Evaluation::Postings => true,
        Evaluation::Auto => {
            let postings: u64 = cnf
                .clauses()
                .flatten()
                .map(|&l| labels.posting_len(l) as u64)
                .sum();
            let words = (cnf.n_clauses() * labels.n_points().div_ceil(64)) as u64;
            let mean_row = labels.n_entries() as f64 / labels.n_points().max(1) as f64;
            ((postings + words) as f64) < min_clause_sum as f64 * mean_row
        }
    }
}

/// Visits the points of clause `c` (from its postings) and checks the other clauses in the
/// forward store. Returns the number of points satisfying clause `c`.
fn forward_check(
    labels: &LabelStore,
    cnf: Cnf<'_>,
    c: usize,
    mut on_match: impl FnMut(usize),
) -> u32 {
    let clause = cnf.clause(c);
    let mut union = 0;
    for (j, &l) in clause.iter().enumerate() {
        for &p in labels.postings(l) {
            let row = labels.row(p);
            // Visit each point once: under the first label of the clause that it carries.
            if intersects(row, &clause[..j]) {
                continue;
            }
            union += 1;
            if cnf.matches_except(row, c) {
                on_match(p as usize);
            }
        }
    }
    union
}

/// Sets `bits` to the points carrying any label of `clause`.
fn clause_bitset(labels: &LabelStore, clause: &[u32], bits: &mut Vec<u64>) {
    bits.clear();
    bits.resize(labels.n_points().div_ceil(64), 0);
    for &l in clause {
        for &p in labels.postings(l) {
            bits[p as usize / 64] |= 1 << (p % 64);
        }
    }
}

/// Intersects the bitsets of all clauses and visits the result. Returns the number of points
/// satisfying clause `c`.
fn postings_check(
    labels: &LabelStore,
    cnf: Cnf<'_>,
    c: usize,
    (acc, tmp): (&mut Vec<u64>, &mut Vec<u64>),
    mut on_match: impl FnMut(usize),
) -> u32 {
    clause_bitset(labels, cnf.clause(c), acc);
    let union = acc.iter().map(|w| w.count_ones()).sum();
    for (i, clause) in cnf.clauses().enumerate() {
        if i != c {
            clause_bitset(labels, clause, tmp);
            for (a, t) in acc.iter_mut().zip(tmp.iter()) {
                *a &= t;
            }
        }
    }
    for (w, &word) in acc.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            on_match(64 * w + bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
    }
    union
}

fn query_truth<T: Element>(
    base: &Matrix<T>,
    labels: &LabelStore,
    query: &[T],
    cnf: Cnf<'_>,
    params: &Params,
    s: &mut Scratch,
) -> QueryTruth {
    let found = &mut s.found;
    found.clear();
    let mut push = |p: usize| found.push((T::squared_l2(query, base.row(p)), p as u32));
    let n = base.nrows();
    let (min_clause, min_clause_sum, min_clause_union) = match smallest_clause(labels, cnf) {
        None => {
            (0..n).for_each(&mut push);
            (NO_CLAUSE, n as u64, n as u32)
        }
        Some((c, sum)) => {
            let union = if use_postings(labels, cnf, sum, params.evaluation) {
                postings_check(labels, cnf, c, (&mut s.acc, &mut s.tmp), &mut push)
            } else {
                forward_check(labels, cnf, c, &mut push)
            };
            (c as u32, sum, union)
        }
    };
    let k = params.k;
    let n_matches = found.len() as u32;
    if found.len() > k {
        let kth = found
            .select_nth_unstable_by(k - 1, by_distance_then_id)
            .1
             .0;
        found.retain(|e| e.0 <= kth);
    }
    found.sort_unstable_by(by_distance_then_id);
    QueryTruth {
        n_matches,
        min_clause,
        min_clause_sum,
        min_clause_union,
        band: found.clone(),
    }
}

/// Computes the ground truth of all queries. `progress` is called with the number of finished
/// queries after each query.
pub fn compute<T: Element>(
    base: &Matrix<T>,
    queries: &Matrix<T>,
    labels: &LabelStore,
    predicates: &PredicateSet,
    params: &Params,
    progress: &(dyn Fn(usize) + Sync),
) -> Result<GroundTruth> {
    let k = params.k;
    ensure!(
        k >= 1 && u32::try_from(k).is_ok(),
        "k must be a positive 32-bit integer"
    );
    ensure!(
        base.ncols() == queries.ncols(),
        "base and query dimensions differ ({} vs {})",
        base.ncols(),
        queries.ncols()
    );
    ensure!(
        labels.n_points() == base.nrows(),
        "{} labeled points but {} base vectors",
        labels.n_points(),
        base.nrows()
    );
    ensure!(
        predicates.len() == queries.nrows(),
        "{} predicates but {} query vectors",
        predicates.len(),
        queries.nrows()
    );
    predicates.check_dictionary(labels)?;
    let done = std::sync::atomic::AtomicUsize::new(0);
    let results = par_map(queries.nrows(), params.threads, Scratch::default, |q, s| {
        let truth = query_truth(base, labels, queries.row(q), predicates.get(q), params, s);
        progress(done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1);
        truth
    });

    let nq = results.len();
    let mut gt = GroundTruth {
        k: k as u32,
        n_points: base.nrows() as u32,
        dict_hash: labels.dict_hash(),
        n_matches: Vec::with_capacity(nq),
        min_clause: Vec::with_capacity(nq),
        min_clause_sum: Vec::with_capacity(nq),
        min_clause_union: Vec::with_capacity(nq),
        band_offsets: vec![0],
        band_ids: Vec::new(),
        band_dists: Vec::new(),
    };
    for r in results {
        gt.n_matches.push(r.n_matches);
        gt.min_clause.push(r.min_clause);
        gt.min_clause_sum.push(r.min_clause_sum);
        gt.min_clause_union.push(r.min_clause_union);
        gt.band_ids.extend(r.band.iter().map(|e| e.1));
        gt.band_dists.extend(r.band.iter().map(|e| e.0));
        gt.band_offsets.push(gt.band_ids.len() as u64);
    }
    Ok(gt)
}

impl GroundTruth {
    pub fn n_queries(&self) -> usize {
        self.n_matches.len()
    }

    /// The `K` of the bands.
    pub fn k(&self) -> usize {
        self.k as usize
    }

    pub fn n_points(&self) -> usize {
        self.n_points as usize
    }

    /// Fingerprint of the label dictionary used.
    pub fn dict_hash(&self) -> u64 {
        self.dict_hash
    }

    /// Number of points matching the predicate of query `q`.
    pub fn n_matches(&self, q: usize) -> u32 {
        self.n_matches[q]
    }

    /// Index of the smallest clause (least total posting length), if the predicate has clauses.
    pub fn min_clause(&self, q: usize) -> Option<u32> {
        Some(self.min_clause[q]).filter(|&c| c != NO_CLAUSE)
    }

    /// Total posting length of the smallest clause (number of points without clauses).
    pub fn min_clause_sum(&self, q: usize) -> u64 {
        self.min_clause_sum[q]
    }

    /// Number of points satisfying the smallest clause (size of the union of its postings).
    pub fn min_clause_union(&self, q: usize) -> u32 {
        self.min_clause_union[q]
    }

    /// Ids and distances of the band of query `q`, sorted by (distance, id).
    pub fn band(&self, q: usize) -> (&[u32], &[f32]) {
        let (a, b) = (
            self.band_offsets[q] as usize,
            self.band_offsets[q + 1] as usize,
        );
        (&self.band_ids[a..b], &self.band_dists[a..b])
    }

    /// The exact top-`k` of query `q` with ties broken by id (`min(k, |M|)` entries).
    ///
    /// # Panics
    ///
    /// Panics if `k` exceeds [`Self::k`].
    pub fn top(&self, q: usize, k: usize) -> (&[u32], &[f32]) {
        assert!(
            k <= self.k(),
            "k = {k} exceeds the ground truth's K = {}",
            self.k
        );
        let (ids, dists) = self.band(q);
        let n = k.min(ids.len());
        (&ids[..n], &dists[..n])
    }

    /// Every match within the `k`-th smallest distance (all matches if there are fewer than
    /// `k`), sorted by (distance, id).
    ///
    /// # Panics
    ///
    /// Panics if `k` is 0 or exceeds [`Self::k`].
    pub fn band_at(&self, q: usize, k: usize) -> (&[u32], &[f32]) {
        assert!(k >= 1 && k <= self.k(), "k = {k} outside 1..={}", self.k);
        let (ids, dists) = self.band(q);
        if ids.len() <= k {
            return (ids, dists);
        }
        let kth = dists[k - 1];
        let end = dists.partition_point(|&d| d <= kth);
        (&ids[..end], &dists[..end])
    }

    /// Writes the extended format.
    pub fn write(&self, path: &Path) -> Result<()> {
        self.write_to(container::create(path)?)
            .with_context(|| format!("writing {}", path.display()))
    }

    /// Writes the extended format to any writer.
    pub fn write_to<W: Write>(&self, w: W) -> Result<()> {
        let params = [
            self.n_queries() as u64,
            u64::from(self.k),
            u64::from(self.n_points),
            self.band_ids.len() as u64,
            self.dict_hash,
            METRIC_SQUARED_L2,
            0,
            0,
        ];
        container::write(
            w,
            MAGIC,
            VERSION,
            &params,
            &[
                bytemuck::cast_slice(&self.n_matches),
                bytemuck::cast_slice(&self.min_clause),
                bytemuck::cast_slice(&self.min_clause_sum),
                bytemuck::cast_slice(&self.min_clause_union),
                bytemuck::cast_slice(&self.band_offsets),
                bytemuck::cast_slice(&self.band_ids),
                bytemuck::cast_slice(&self.band_dists),
            ],
        )
    }

    /// Reads and validates the extended format.
    pub fn read(path: &Path) -> Result<Self> {
        container::Reader::open(path, MAGIC)
            .and_then(Self::load)
            .with_context(|| format!("loading ground truth from {}", path.display()))
    }

    /// Reads and validates the extended format from any reader.
    pub fn read_from<R: Read + Seek>(r: R) -> Result<Self> {
        Self::load(container::Reader::new(r, MAGIC)?)
    }

    fn load<R: Read + Seek>(mut r: container::Reader<R>) -> Result<Self> {
        ensure!(
            r.version() == VERSION,
            "unsupported version {}",
            r.version()
        );
        let [nq, k, n_points, nb, dict_hash, metric, ..] = *r.params();
        ensure!(
            metric == METRIC_SQUARED_L2,
            "unsupported metric code {metric}"
        );
        let (nq, nb) = (nq as usize, nb as usize);
        let gt = Self {
            k: u32::try_from(k).context("invalid k")?,
            n_points: u32::try_from(n_points).context("invalid point count")?,
            dict_hash,
            n_matches: r.section(0, nq)?,
            min_clause: r.section(1, nq)?,
            min_clause_sum: r.section(2, nq)?,
            min_clause_union: r.section(3, nq)?,
            band_offsets: r.section(4, nq + 1)?,
            band_ids: r.section(5, nb)?,
            band_dists: r.section(6, nb)?,
        };
        gt.validate()?;
        Ok(gt)
    }

    fn validate(&self) -> Result<()> {
        let offsets = &self.band_offsets;
        ensure!(
            offsets[0] == 0
                && offsets[self.n_queries()] == self.band_ids.len() as u64
                && offsets.windows(2).all(|w| w[0] <= w[1]),
            "inconsistent band offsets"
        );
        for q in 0..self.n_queries() {
            let (ids, dists) = self.band(q);
            let matches = self.n_matches[q] as usize;
            let complete = if matches <= self.k() {
                ids.len() == matches
            } else {
                ids.len() >= self.k() && ids.len() <= matches
            };
            let sorted = ids
                .windows(2)
                .zip(dists.windows(2))
                .all(|(i, d)| by_distance_then_id(&(d[0], i[0]), &(d[1], i[1])) == Ordering::Less);
            let in_range = ids.iter().all(|&p| p < self.n_points);
            ensure!(
                complete && sorted && in_range,
                "query {q}: invalid band ({} entries, {matches} matches, sorted: {sorted}, ids in range: {in_range})",
                ids.len()
            );
        }
        Ok(())
    }

    /// Writes the standard DiskANN ground truth file: `u32` query count, `u32` K, the ids
    /// (row-major, query by query), then the distances. Queries with fewer than K matches are
    /// padded with id `u32::MAX` and distance `f32::MAX`.
    pub fn write_diskann(&self, path: &Path) -> Result<()> {
        self.write_diskann_to(container::create(path)?)
            .with_context(|| format!("writing {}", path.display()))
    }

    /// Writes the DiskANN ground truth format to any writer.
    pub fn write_diskann_to<W: Write>(&self, mut w: W) -> Result<()> {
        let (nq, k) = (self.n_queries(), self.k());
        let mut ids = vec![u32::MAX; nq * k];
        let mut dists = vec![f32::MAX; nq * k];
        for (q, (id_row, dist_row)) in ids.chunks_mut(k).zip(dists.chunks_mut(k)).enumerate() {
            let (top_ids, top_dists) = self.top(q, k);
            id_row[..top_ids.len()].copy_from_slice(top_ids);
            dist_row[..top_dists.len()].copy_from_slice(top_dists);
        }
        w.write_all(&u32::try_from(nq)?.to_le_bytes())?;
        w.write_all(&self.k.to_le_bytes())?;
        w.write_all(bytemuck::cast_slice(&ids))?;
        w.write_all(bytemuck::cast_slice(&dists))?;
        w.flush()?;
        Ok(())
    }
}

/// A standard DiskANN ground truth file (ids, optionally distances).
#[derive(Debug, Clone)]
pub struct DiskannTruth {
    pub n_queries: usize,
    pub k: usize,
    pub ids: Vec<u32>,
    pub dists: Option<Vec<f32>>,
}

impl DiskannTruth {
    pub fn read(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_bytes(&bytes).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        ensure!(bytes.len() >= 8, "truncated header");
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let (nq, k) = (words[0] as usize, words[1] as usize);
        let cells = nq * k;
        let payload = bytes.len() - 8;
        let dists = if payload == 8 * cells {
            Some(
                words[2 + cells..]
                    .iter()
                    .map(|&w| f32::from_bits(w))
                    .collect(),
            )
        } else {
            ensure!(
                payload == 4 * cells,
                "{payload} payload bytes do not fit {nq} × {k} ids (with or without distances)"
            );
            None
        };
        Ok(Self {
            n_queries: nq,
            k,
            ids: words[2..2 + cells].to_vec(),
            dists,
        })
    }
}

/// Why a reference ground truth disagrees with ours on a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MismatchKind {
    /// Different number of valid (non-padding) entries than `min(k, |M|)`.
    Count,
    /// The reference repeats an id.
    Duplicate,
    /// A reference id is not in our band, or has a different distance there.
    Membership,
    /// The sorted distance lists differ.
    Distances,
}

#[derive(Debug, Clone)]
pub struct Mismatch {
    pub query: usize,
    pub kind: MismatchKind,
    pub detail: String,
}

/// Result of [`compare`].
#[derive(Debug, Clone, Default)]
pub struct Comparison {
    pub n_queries: usize,
    pub k: usize,
    /// Same ids in the same order.
    pub identical: usize,
    /// Same distances and a valid choice of ids within every tie band, in another order.
    pub equal_up_to_ties: usize,
    /// Among `equal_up_to_ties`: the same set of ids (only the order within ties differs).
    pub same_id_set: usize,
    pub mismatches: Vec<Mismatch>,
}

/// Compares the top-`k` of a DiskANN ground truth file (with distances) with ours, up to the
/// order and choice of points within distance ties.
///
/// A reference row agrees when it has `min(k, |M|)` distinct valid ids, each of them lies in
/// our band with exactly the reference distance, and its sorted distances equal our top-`k`
/// distances. Since our band holds every match within the k-th distance, this means the
/// reference is a correct top-`k` up to tie-breaking.
pub fn compare(ours: &GroundTruth, reference: &DiskannTruth, k: usize) -> Result<Comparison> {
    ensure!(
        reference.n_queries == ours.n_queries(),
        "query counts differ ({} vs {})",
        reference.n_queries,
        ours.n_queries()
    );
    ensure!(
        k >= 1 && k <= ours.k() && k <= reference.k,
        "k = {k} must be in 1..=min({}, {})",
        ours.k(),
        reference.k
    );
    let ref_dists = reference
        .dists
        .as_ref()
        .context("the reference has no distances")?;
    let mut out = Comparison {
        n_queries: ours.n_queries(),
        k,
        ..Comparison::default()
    };
    let rk = reference.k;
    for q in 0..ours.n_queries() {
        let rids = &reference.ids[q * rk..q * rk + k];
        let rdists = &ref_dists[q * rk..q * rk + k];
        let n = rids.iter().position(|&id| id == u32::MAX).unwrap_or(k);
        let (ids, dists) = ours.top(q, k);
        let mut fail = |kind, detail: String| {
            out.mismatches.push(Mismatch {
                query: q,
                kind,
                detail,
            })
        };
        if n != ids.len() || rids[n..].iter().any(|&id| id != u32::MAX) {
            fail(
                MismatchKind::Count,
                format!(
                    "reference has {n} valid ids, ours {} ({} matches)",
                    ids.len(),
                    ours.n_matches(q)
                ),
            );
            continue;
        }
        let (rids, rdists) = (&rids[..n], &rdists[..n]);
        if rids == ids && rdists == dists {
            out.identical += 1;
            continue;
        }
        if rids.iter().collect::<HashSet<_>>().len() != n {
            fail(MismatchKind::Duplicate, "repeated id".into());
            continue;
        }
        let (band_ids, band_dists) = ours.band(q);
        let band: HashMap<u32, f32> = band_ids
            .iter()
            .copied()
            .zip(band_dists.iter().copied())
            .collect();
        if let Some((&id, &d)) = rids.iter().zip(rdists).find(|&(id, d)| {
            band.get(id)
                .is_none_or(|ours| ours.to_bits() != d.to_bits())
        }) {
            let detail = match band.get(&id) {
                Some(ours) => format!("id {id}: reference distance {d}, ours {ours}"),
                None => format!(
                    "id {id} (reference distance {d}) is not within our k-th distance {}",
                    dists[n - 1]
                ),
            };
            fail(MismatchKind::Membership, detail);
            continue;
        }
        let mut sorted = rdists.to_vec();
        sorted.sort_by(f32::total_cmp);
        if sorted != dists {
            fail(MismatchKind::Distances, "sorted distances differ".into());
            continue;
        }
        out.equal_up_to_ties += 1;
        let ours_set: HashSet<u32> = ids.iter().copied().collect();
        out.same_id_set += usize::from(rids.iter().all(|id| ours_set.contains(id)));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{labels::LabelStoreBuilder, predicate::PredicateSetBuilder};

    /// Deterministic xorshift generator.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self, bound: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % bound
        }
    }

    fn brute_force(
        base: &Matrix<u8>,
        labels: &LabelStore,
        query: &[u8],
        cnf: Cnf<'_>,
        k: usize,
    ) -> (u32, Vec<(f32, u32)>) {
        let mut all: Vec<(f32, u32)> = (0..base.nrows() as u32)
            .filter(|&p| cnf.matches(labels.row(p)))
            .map(|p| (u8::squared_l2(query, base.row(p as usize)), p))
            .collect();
        all.sort_by(by_distance_then_id);
        let n = all.len() as u32;
        if all.len() > k {
            let kth = all[k - 1].0;
            all.retain(|e| e.0 <= kth);
        }
        (n, all)
    }

    #[test]
    fn matches_brute_force_with_ties() {
        // Few distinct coordinates, so that distance ties and duplicate vectors are common.
        let (n, dim, k) = (300, 4, 5);
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let data: Vec<u8> = (0..n * dim).map(|_| rng.next(3) as u8).collect();
        let base = Matrix::from_slice(&data, n, dim);
        let names = ["a=0", "a=1", "a=2", "a=3", "b=0", "b=1", "b=2"];
        let mut lb = LabelStoreBuilder::new();
        for _ in 0..n {
            let row: Vec<&str> = names.iter().copied().filter(|_| rng.next(3) == 0).collect();
            lb.push_point(row.iter().map(|&l| (&l[..1], l))).unwrap();
        }
        let labels = lb.finish().unwrap();
        let clause_sets: [&[&[&str]]; 6] = [
            &[&["a=0", "a=1", "a=3"], &["b=1", "b=2"]],
            &[&["a=2"], &["b=0", "a=1"], &["b=2"]],
            &[&["a=0", "a=nowhere"]],
            &[&["a=nowhere"], &["b=1"]],
            &[],
            &[&["a=3"], &["b=0"], &["a=1", "b=2"], &["a=0"]],
        ];
        let mut pb = PredicateSetBuilder::new(&labels);
        let mut qdata = Vec::new();
        for clauses in clause_sets {
            let clauses: Vec<Vec<&str>> = clauses.iter().map(|c| c.to_vec()).collect();
            pb.push(&clauses);
            qdata.extend((0..dim).map(|_| rng.next(3) as u8));
        }
        let preds = pb.finish();
        let queries = Matrix::from_slice(&qdata, preds.len(), dim);

        let params = |evaluation| Params {
            k,
            threads: 3,
            evaluation,
        };
        let gt = compute(
            &base,
            &queries,
            &labels,
            &preds,
            &params(Evaluation::Forward),
            &|_| {},
        )
        .unwrap();
        for e in [Evaluation::Postings, Evaluation::Auto] {
            let other = compute(&base, &queries, &labels, &preds, &params(e), &|_| {}).unwrap();
            assert_eq!(other, gt, "{e:?}");
        }
        let mut saw_long_band = false;
        for q in 0..preds.len() {
            let cnf = preds.get(q);
            let (n_matches, expected) = brute_force(&base, &labels, queries.row(q), cnf, k);
            let (ids, dists) = gt.band(q);
            let got: Vec<(f32, u32)> = dists.iter().copied().zip(ids.iter().copied()).collect();
            assert_eq!((gt.n_matches(q), &got), (n_matches, &expected), "query {q}");
            saw_long_band |= got.len() > k;
            if let Some(c) = gt.min_clause(q) {
                let clause = cnf.clause(c as usize);
                let union = (0..n as u32)
                    .filter(|&p| intersects(labels.row(p), clause))
                    .count();
                assert_eq!(gt.min_clause_union(q) as usize, union, "query {q}");
            }
        }
        assert!(saw_long_band, "the test data should produce ties beyond k");
        assert_eq!(
            (gt.n_matches(2), gt.n_matches(3)),
            (labels.postings(0).len() as u32, 0)
        );
        assert_eq!((gt.n_matches(4), gt.min_clause(4)), (n as u32, None));

        let mut buf = Vec::new();
        gt.write_to(&mut buf).unwrap();
        assert_eq!(
            GroundTruth::read_from(std::io::Cursor::new(&buf)).unwrap(),
            gt
        );
    }

    fn tiny() -> GroundTruth {
        // Query 0: four matches at distances 1, 2, 2, 2 (K = 2: band of 4 entries).
        // Query 1: one match (padded in the DiskANN format).
        GroundTruth {
            k: 2,
            n_points: 10,
            dict_hash: 0,
            n_matches: vec![4, 1],
            min_clause: vec![0, 0],
            min_clause_sum: vec![4, 1],
            min_clause_union: vec![4, 1],
            band_offsets: vec![0, 4, 5],
            band_ids: vec![7, 1, 3, 5, 2],
            band_dists: vec![1.0, 2.0, 2.0, 2.0, 0.0],
        }
    }

    #[test]
    fn diskann_output_is_padded() {
        let gt = tiny();
        assert_eq!(gt.band_at(0, 1), (&[7u32][..], &[1.0f32][..]));
        assert_eq!(gt.band_at(0, 2).0, &[7, 1, 3, 5]);
        let mut buf = Vec::new();
        gt.write_diskann_to(&mut buf).unwrap();
        let d = DiskannTruth::from_bytes(&buf).unwrap();
        assert_eq!((d.n_queries, d.k), (2, 2));
        assert_eq!(d.ids, [7, 1, 2, u32::MAX]);
        assert_eq!(d.dists.unwrap(), [1.0, 2.0, 0.0, f32::MAX]);
    }

    #[test]
    fn comparison_accepts_other_tie_choices_only() {
        let gt = tiny();
        let reference = |ids: [u32; 4], dists: [f32; 4]| DiskannTruth {
            n_queries: 2,
            k: 2,
            ids: ids.to_vec(),
            dists: Some(dists.to_vec()),
        };
        let max = f32::MAX;
        let ok = compare(
            &gt,
            &reference([7, 1, 2, u32::MAX], [1.0, 2.0, 0.0, max]),
            2,
        )
        .unwrap();
        assert_eq!((ok.identical, ok.equal_up_to_ties), (2, 0));
        let tie = compare(
            &gt,
            &reference([7, 5, 2, u32::MAX], [1.0, 2.0, 0.0, max]),
            2,
        )
        .unwrap();
        assert_eq!(
            (tie.identical, tie.equal_up_to_ties, tie.same_id_set),
            (1, 1, 0)
        );
        let order = compare(
            &gt,
            &reference([7, 1, 2, u32::MAX], [1.0, 2.0, 0.0, max]),
            1,
        )
        .unwrap();
        assert_eq!((order.identical, order.equal_up_to_ties), (2, 0));
        let kinds = |r: DiskannTruth| {
            compare(&gt, &r, 2)
                .unwrap()
                .mismatches
                .iter()
                .map(|m| m.kind)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            kinds(reference([7, 9, 2, 4], [1.0, 2.0, 0.0, 3.0])),
            [MismatchKind::Membership, MismatchKind::Count]
        );
        assert_eq!(
            kinds(reference([1, 3, 2, u32::MAX], [2.0, 2.0, 0.0, max])),
            [MismatchKind::Distances]
        );
    }
}
