/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Bottom-`b` priority sketches of the label postings (sidecar `sketches.flab`).
//!
//! Every data point gets a fixed pseudo-random priority (a hash of its id and a seed); every
//! label keeps the `b` members with the lowest `(priority, id)` (all of its members if it has
//! at most `b`), sorted. For any set of labels, merging their lists in that order with
//! deduplication yields the `b` lowest-priority members of the labels' union, i.e. a uniform
//! sample without replacement of the union: a label's list holds every member whose priority
//! is below the union's `b`-th, and the shared priorities keep points with several of the
//! labels from being oversampled. When every label of the set has at most `b` members the
//! merge enumerates the whole union.
//!
//! Memory: `8 × Σ_ℓ min(b, |post(ℓ)|)` bytes of entries plus `8 × (n_labels + 1)` bytes of
//! offsets; the build reads every posting once.

use std::{
    io::{Read, Seek, Write},
    path::Path,
};

use anyhow::{ensure, Context, Result};

use crate::{container, labels::LabelStore, parallel};

/// File magic of `sketches.flab`.
pub const MAGIC: &[u8; 8] = b"FLABSKT\0";
/// Version of the `sketches.flab` format.
pub const VERSION: u32 = 1;

/// Priority of data point `p` under `seed` (SplitMix64 of the id, high 32 bits).
#[inline(always)]
pub fn priority(seed: u64, p: u32) -> u32 {
    let mut z = seed ^ u64::from(p).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((z ^ (z >> 31)) >> 32) as u32
}

/// The sketch entry of point `p`: `(priority << 32) | p`, ordered by priority, then id.
#[inline(always)]
pub fn entry(seed: u64, p: u32) -> u64 {
    (u64::from(priority(seed, p)) << 32) | u64::from(p)
}

/// Point id of a sketch entry.
#[inline(always)]
pub fn entry_id(e: u64) -> u32 {
    e as u32
}

/// Priority of a sketch entry as a fraction of the hash range, in `(0, 1]`.
#[inline]
pub fn entry_fraction(e: u64) -> f64 {
    ((e >> 32) as f64 + 1.0) / 4_294_967_296.0
}

/// Per-label bottom-`b` sketches (see the module documentation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelSketches {
    n_points: usize,
    b: usize,
    seed: u64,
    dict_hash: u64,
    /// Label `ℓ`'s entries are `entries[offsets[ℓ]..offsets[ℓ + 1]]`.
    offsets: Vec<u64>,
    entries: Vec<u64>,
}

/// The sample of a union of labels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnionSample {
    /// Distinct points sampled (`≤ b`).
    pub n: usize,
    /// Whether the sample is the whole union.
    pub complete: bool,
    /// Priority fraction of the last (`b`-th) sampled entry (1 if complete).
    pub tau: f64,
}

impl LabelSketches {
    /// Sketches of every label of `labels` with `b ≥ 1` entries at most (built with up to
    /// `threads` threads; the result does not depend on the thread count).
    pub fn build(labels: &LabelStore, b: usize, seed: u64, threads: usize) -> Self {
        let b = b.max(1);
        let n_labels = labels.n_labels();
        let chunk = 1024;
        let parts = parallel::par_map(
            n_labels.div_ceil(chunk),
            threads,
            Vec::new,
            |c, tmp: &mut Vec<u64>| {
                let mut out = Vec::new();
                let mut lens = Vec::new();
                for l in (c * chunk)..((c + 1) * chunk).min(n_labels) {
                    tmp.clear();
                    tmp.extend(labels.postings(l as u32).iter().map(|&p| entry(seed, p)));
                    if tmp.len() > b {
                        tmp.select_nth_unstable(b - 1);
                        tmp.truncate(b);
                    }
                    tmp.sort_unstable();
                    lens.push(tmp.len() as u64);
                    out.extend_from_slice(tmp);
                }
                (lens, out)
            },
        );
        let mut offsets = Vec::with_capacity(n_labels + 1);
        offsets.push(0u64);
        let mut entries = Vec::new();
        for (lens, part) in parts {
            for len in lens {
                offsets.push(offsets.last().copied().unwrap_or(0) + len);
            }
            entries.extend_from_slice(&part);
        }
        Self {
            n_points: labels.n_points(),
            b,
            seed,
            dict_hash: labels.dict_hash(),
            offsets,
            entries,
        }
    }

    /// Entries per label at most.
    pub fn b(&self) -> usize {
        self.b
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn n_points(&self) -> usize {
        self.n_points
    }

    pub fn n_labels(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Fingerprint of the dictionary of the label store.
    pub fn dict_hash(&self) -> u64 {
        self.dict_hash
    }

    /// Number of stored entries.
    pub fn n_entries(&self) -> usize {
        self.entries.len()
    }

    /// Bytes held (entries and offsets).
    pub fn bytes(&self) -> u64 {
        8 * (self.entries.len() + self.offsets.len()) as u64
    }

    /// The sorted entries of label `l`.
    #[inline]
    pub fn list(&self, l: u32) -> &[u64] {
        let l = l as usize;
        &self.entries[self.offsets[l] as usize..self.offsets[l + 1] as usize]
    }

    /// Samples the union of `labels` (label ids of the store the sketches were built from):
    /// writes the point ids of its `b` lowest-priority members, in priority order, into `out`.
    /// `full(ℓ)` tells whether label `ℓ` has at most `b` members (its list is complete).
    /// `heap` is reusable scratch.
    pub fn sample_union(
        &self,
        labels: &[u32],
        full: impl Fn(u32) -> bool,
        heap: &mut Vec<(u64, u32)>,
        pos: &mut Vec<usize>,
        out: &mut Vec<u32>,
    ) -> UnionSample {
        out.clear();
        heap.clear();
        pos.clear();
        pos.resize(labels.len(), 1);
        let mut all_full = true;
        for (i, &l) in labels.iter().enumerate() {
            all_full &= full(l);
            if let Some(&e) = self.list(l).first() {
                heap.push((e, i as u32));
            }
        }
        for i in (0..heap.len() / 2).rev() {
            sift_down(heap, i);
        }
        let mut last = u64::MAX;
        let mut tau = 1.0;
        while let Some(&(e, i)) = heap.first() {
            if e != last {
                if out.len() == self.b {
                    break;
                }
                last = e;
                out.push(entry_id(e));
                tau = entry_fraction(e);
            }
            let list = self.list(labels[i as usize]);
            let p = &mut pos[i as usize];
            if *p < list.len() {
                heap[0] = (list[*p], i);
                *p += 1;
            } else {
                let tail = heap.pop().unwrap_or((0, 0));
                if heap.is_empty() {
                    break;
                }
                heap[0] = tail;
            }
            sift_down(heap, 0);
        }
        // Complete: every list was merged to its end and holds all of its label's members.
        let complete = all_full && heap.is_empty();
        UnionSample {
            n: out.len(),
            complete,
            tau: if complete { 1.0 } else { tau },
        }
    }

    /// Checks that the sketches belong to `labels` (dictionary, counts, and every list against
    /// the label's postings).
    pub fn check(&self, labels: &LabelStore) -> Result<()> {
        self.check_sample(labels, 1)
    }

    /// [`Self::check`] on every `step`-th label only (the dictionary and counts always).
    pub fn check_sample(&self, labels: &LabelStore, step: usize) -> Result<()> {
        ensure!(
            self.dict_hash == labels.dict_hash()
                && self.n_labels() == labels.n_labels()
                && self.n_points == labels.n_points(),
            "the sketches refer to another label store"
        );
        let mut tmp = Vec::new();
        for l in (0..labels.n_labels() as u32).step_by(step.max(1)) {
            tmp.clear();
            tmp.extend(labels.postings(l).iter().map(|&p| entry(self.seed, p)));
            tmp.sort_unstable();
            tmp.truncate(self.b);
            ensure!(
                self.list(l) == &tmp[..],
                "the sketch of label {l} differs from its postings"
            );
        }
        Ok(())
    }

    /// Writes `sketches.flab` (see the crate `README.md` for the layout).
    pub fn write(&self, path: &Path) -> Result<()> {
        self.write_to(container::create(path)?)
            .with_context(|| format!("writing {}", path.display()))
    }

    /// Writes the `sketches.flab` format to any writer.
    pub fn write_to<W: Write>(&self, w: W) -> Result<()> {
        let params = [
            self.n_points as u64,
            self.n_labels() as u64,
            self.dict_hash,
            self.b as u64,
            self.seed,
            self.entries.len() as u64,
            0,
            0,
        ];
        container::write(
            w,
            MAGIC,
            VERSION,
            &params,
            &[
                bytemuck::cast_slice(&self.offsets),
                bytemuck::cast_slice(&self.entries),
            ],
        )
    }

    /// Reads and validates `sketches.flab` (structure only; [`Self::check`] compares it with a
    /// label store).
    pub fn read(path: &Path) -> Result<Self> {
        container::Reader::open(path, MAGIC)
            .and_then(Self::load)
            .with_context(|| format!("loading sketches from {}", path.display()))
    }

    /// Reads the `sketches.flab` format from any reader.
    pub fn read_from<R: Read + Seek>(r: R) -> Result<Self> {
        Self::load(container::Reader::new(r, MAGIC)?)
    }

    fn load<R: Read + Seek>(mut r: container::Reader<R>) -> Result<Self> {
        ensure!(
            r.version() == VERSION,
            "unsupported version {}",
            r.version()
        );
        let [n_points, n_labels, dict_hash, b, seed, n_entries, ..] = *r.params();
        ensure!(b >= 1, "inconsistent parameters");
        let offsets: Vec<u64> = r.section(0, n_labels as usize + 1)?;
        let entries: Vec<u64> = r.section(1, n_entries as usize)?;
        ensure!(
            offsets.first() == Some(&0)
                && offsets.last() == Some(&n_entries)
                && offsets.windows(2).all(|w| w[0] <= w[1] && w[1] - w[0] <= b),
            "inconsistent offsets"
        );
        for w in offsets.windows(2) {
            let list = &entries[w[0] as usize..w[1] as usize];
            ensure!(
                list.windows(2).all(|p| p[0] < p[1])
                    && list.iter().all(|&e| u64::from(entry_id(e)) < n_points),
                "sketch lists must be increasing entries of data points"
            );
        }
        Ok(Self {
            n_points: n_points as usize,
            b: b as usize,
            seed,
            dict_hash,
            offsets,
            entries,
        })
    }
}

#[inline]
fn sift_down(h: &mut [(u64, u32)], mut i: usize) {
    let n = h.len();
    loop {
        let l = 2 * i + 1;
        if l >= n {
            break;
        }
        let r = l + 1;
        let c = if r < n && h[r] < h[l] { r } else { l };
        if h[i] <= h[c] {
            break;
        }
        h.swap(i, c);
        i = c;
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::labels::LabelStoreBuilder;

    /// 2,000 points over category `a` (Zipf-like, 1–3 labels per point) and `b` (4 values).
    fn store() -> LabelStore {
        let mut state = 17u64;
        let mut next = |n: u32| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as u32) % n
        };
        let mut b = LabelStoreBuilder::new();
        for _ in 0..2000 {
            let mut row: Vec<String> = Vec::new();
            for _ in 0..1 + next(3) {
                let r = next(30);
                row.push(format!("a={}", r * r / 30));
            }
            row.push(format!("b={}", next(4)));
            b.push_point(row.iter().map(|l| (&l[..1], l.as_str())))
                .unwrap();
        }
        b.finish().unwrap()
    }

    #[test]
    fn union_samples_are_the_lowest_priorities_of_the_union() {
        let s = store();
        for (b, seed) in [(8, 1), (32, 2), (5000, 3)] {
            let sk = LabelSketches::build(&s, b, seed, 3);
            sk.check(&s).unwrap();
            assert_eq!(sk, LabelSketches::build(&s, b, seed, 1), "thread count");
            let mut buf = Vec::new();
            sk.write_to(&mut buf).unwrap();
            assert_eq!(LabelSketches::read_from(Cursor::new(&buf)).unwrap(), sk);

            let (mut heap, mut pos, mut out) = (Vec::new(), Vec::new(), Vec::new());
            let n_labels = s.n_labels() as u32;
            for set in [
                vec![0u32],
                vec![0, 1],
                vec![1, 2, 5],
                (0..n_labels).collect(),
            ] {
                let set: Vec<u32> = set.into_iter().filter(|&l| l < n_labels).collect();
                let full = |l: u32| s.posting_len(l) <= b;
                let u = sk.sample_union(&set, full, &mut heap, &mut pos, &mut out);
                // The union's members in priority order.
                let mut union: Vec<u64> = set
                    .iter()
                    .flat_map(|&l| s.postings(l).iter().map(|&p| entry(seed, p)))
                    .collect();
                union.sort_unstable();
                union.dedup();
                let want: Vec<u32> = union.iter().take(b).map(|&e| entry_id(e)).collect();
                assert_eq!(out, want, "b {b} labels {set:?}");
                assert_eq!(u.n, want.len());
                assert_eq!(u.complete, union.len() <= b && set.iter().all(|&l| full(l)));
                if !u.complete {
                    assert_eq!(u.tau, entry_fraction(union[b - 1]));
                }
            }
        }
    }

    #[test]
    fn priorities_are_spread_and_bytes_are_counted() {
        let s = store();
        let sk = LabelSketches::build(&s, 16, 9, 2);
        let expect: usize = (0..s.n_labels() as u32)
            .map(|l| s.posting_len(l).min(16))
            .sum();
        assert_eq!(sk.n_entries(), expect);
        assert_eq!(sk.bytes(), 8 * (expect + s.n_labels() + 1) as u64);
        // Priorities of 10,000 ids are roughly uniform: each quarter of the range gets 20-30 %.
        let mut quarters = [0usize; 4];
        for p in 0..10_000 {
            quarters[(priority(9, p) >> 30) as usize] += 1;
        }
        assert!(
            quarters.iter().all(|&q| (2000..3000).contains(&q)),
            "{quarters:?}"
        );
    }
}
