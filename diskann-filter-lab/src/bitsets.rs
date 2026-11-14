/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Static per-label bitsets (format `bitsets.flab`): a dense bitset over the data points for
//! every label with at least `θ` points, built once from the postings (a sidecar of a label
//! store). They let a predicate evaluator test "point `p` carries label `ℓ`" in O(1) for
//! frequent labels (`static_bits(θ)` evaluation mode of the lab kernel); the other labels are
//! checked on the point's label row.
//!
//! Memory: `⌈n/64⌉ × 8` bytes per label with a bitset. A bitset is smaller than the label's
//! postings (`4 |post|` bytes) when `|post| > n/32`, so `θ = n/32` never stores more than the
//! postings of the labels it covers.

use std::{
    io::{Read, Seek, Write},
    path::Path,
};

use anyhow::{ensure, Context, Result};

use crate::{container, labels::LabelStore};

/// File magic of `bitsets.flab`.
pub const MAGIC: &[u8; 8] = b"FLABBIT\0";
/// Version of the `bitsets.flab` format.
pub const VERSION: u32 = 1;

/// No bitset for this label.
const NONE: u32 = u32::MAX;

/// Dense bitsets of the labels with at least `θ` points (see the module documentation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelBitsets {
    n_points: usize,
    theta: u64,
    dict_hash: u64,
    /// `u64` words per bitset.
    words: usize,
    /// Bitset slot of every label id (`NONE` without one).
    slot: Vec<u32>,
    /// Label of every slot, in increasing id order.
    labels: Vec<u32>,
    /// Slot `s` occupies `bits[s * words..(s + 1) * words]`.
    bits: Vec<u64>,
}

/// Static bitsets of an empty store (the evaluator context's default).
pub static EMPTY: LabelBitsets = LabelBitsets {
    n_points: 0,
    theta: u64::MAX,
    dict_hash: 0,
    words: 0,
    slot: Vec::new(),
    labels: Vec::new(),
    bits: Vec::new(),
};

/// Size of the static bitsets of `labels` at threshold `theta` without building them:
/// `(labels with a bitset, bytes, posting entries covered)`.
pub fn footprint(labels: &LabelStore, theta: u64) -> (usize, u64, u64) {
    let words = labels.n_points().div_ceil(64) as u64;
    let (mut n, mut covered) = (0usize, 0u64);
    for l in 0..labels.n_labels() as u32 {
        let len = labels.posting_len(l) as u64;
        if len >= theta.max(1) {
            n += 1;
            covered += len;
        }
    }
    (n, n as u64 * words * 8, covered)
}

impl LabelBitsets {
    /// Bitsets of every label of `labels` with at least `theta` points (`theta ≥ 1`).
    pub fn build(labels: &LabelStore, theta: u64) -> Self {
        let theta = theta.max(1);
        let n_points = labels.n_points();
        let words = n_points.div_ceil(64);
        let mut slot = vec![NONE; labels.n_labels()];
        let mut chosen = Vec::new();
        for l in 0..labels.n_labels() as u32 {
            if labels.posting_len(l) as u64 >= theta {
                slot[l as usize] = chosen.len() as u32;
                chosen.push(l);
            }
        }
        let mut bits = vec![0u64; chosen.len() * words];
        for (s, &l) in chosen.iter().enumerate() {
            let b = &mut bits[s * words..(s + 1) * words];
            for &p in labels.postings(l) {
                b[(p >> 6) as usize] |= 1u64 << (p & 63);
            }
        }
        Self {
            n_points,
            theta,
            dict_hash: labels.dict_hash(),
            words,
            slot,
            labels: chosen,
            bits,
        }
    }

    /// The threshold `θ`: every label with at least `θ` points has a bitset.
    pub fn theta(&self) -> u64 {
        self.theta
    }

    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// Number of labels with a bitset.
    pub fn n_bitsets(&self) -> usize {
        self.labels.len()
    }

    /// Labels with a bitset, in increasing id order.
    pub fn labels(&self) -> &[u32] {
        &self.labels
    }

    /// Bytes of the bitsets.
    pub fn bytes(&self) -> u64 {
        8 * self.bits.len() as u64
    }

    /// Fingerprint of the dictionary of the label store.
    pub fn dict_hash(&self) -> u64 {
        self.dict_hash
    }

    /// Slot of label `l`'s bitset, if it has one.
    #[inline]
    pub fn slot(&self, l: u32) -> Option<u32> {
        self.slot.get(l as usize).copied().filter(|&s| s != NONE)
    }

    /// Label of slot `s`.
    #[inline]
    pub fn label(&self, s: u32) -> u32 {
        self.labels[s as usize]
    }

    /// Word `w` (points `64w..64w + 64`) of the bitset in slot `s`.
    #[inline(always)]
    pub fn word(&self, s: u32, w: usize) -> u64 {
        self.bits[s as usize * self.words + w]
    }

    /// Address of word `w` of slot `s` (for prefetching; never dereferenced here).
    #[inline(always)]
    pub fn word_ptr(&self, s: u32, w: usize) -> *const u64 {
        self.bits.as_ptr().wrapping_add(s as usize * self.words + w)
    }

    /// Whether point `p` carries the label of slot `s`.
    #[inline(always)]
    pub fn test(&self, s: u32, p: u32) -> bool {
        (self.word(s, (p >> 6) as usize) >> (p & 63)) & 1 == 1
    }

    /// Checks that the bitsets belong to `labels` (dictionary, point count, threshold
    /// coverage and every bit against the postings).
    pub fn check(&self, labels: &LabelStore) -> Result<()> {
        ensure!(
            self.dict_hash == labels.dict_hash()
                && self.slot.len() == labels.n_labels()
                && self.n_points == labels.n_points(),
            "the bitsets refer to another label store"
        );
        for l in 0..labels.n_labels() as u32 {
            let len = labels.posting_len(l) as u64;
            match self.slot(l) {
                None => ensure!(len < self.theta, "label {l} ({len} points) has no bitset"),
                Some(s) => {
                    let ones: u64 = (0..self.words)
                        .map(|w| u64::from(self.word(s, w).count_ones()))
                        .sum();
                    ensure!(
                        ones == len && labels.postings(l).iter().all(|&p| self.test(s, p)),
                        "the bitset of label {l} differs from its postings"
                    );
                }
            }
        }
        Ok(())
    }

    /// Writes `bitsets.flab` (see the crate `README.md` for the layout).
    pub fn write(&self, path: &Path) -> Result<()> {
        self.write_to(container::create(path)?)
            .with_context(|| format!("writing {}", path.display()))
    }

    /// Writes the `bitsets.flab` format to any writer.
    pub fn write_to<W: Write>(&self, w: W) -> Result<()> {
        let params = [
            self.n_points as u64,
            self.slot.len() as u64,
            self.dict_hash,
            self.theta,
            self.labels.len() as u64,
            self.words as u64,
            0,
            0,
        ];
        container::write(
            w,
            MAGIC,
            VERSION,
            &params,
            &[
                bytemuck::cast_slice(&self.labels),
                bytemuck::cast_slice(&self.bits),
            ],
        )
    }

    /// Reads and validates `bitsets.flab` (structure only; [`Self::check`] compares it with
    /// a label store).
    pub fn read(path: &Path) -> Result<Self> {
        container::Reader::open(path, MAGIC)
            .and_then(Self::load)
            .with_context(|| format!("loading bitsets from {}", path.display()))
    }

    /// Reads the `bitsets.flab` format from any reader.
    pub fn read_from<R: Read + Seek>(r: R) -> Result<Self> {
        Self::load(container::Reader::new(r, MAGIC)?)
    }

    fn load<R: Read + Seek>(mut r: container::Reader<R>) -> Result<Self> {
        ensure!(
            r.version() == VERSION,
            "unsupported version {}",
            r.version()
        );
        let [n_points, n_labels, dict_hash, theta, n_bitsets, words, ..] = *r.params();
        ensure!(
            words == n_points.div_ceil(64) && theta >= 1 && n_bitsets <= n_labels,
            "inconsistent parameters"
        );
        let labels: Vec<u32> = r.section(0, n_bitsets as usize)?;
        let bits = r.section(1, (n_bitsets * words) as usize)?;
        ensure!(
            labels.windows(2).all(|w| w[0] < w[1])
                && labels.last().is_none_or(|&l| u64::from(l) < n_labels),
            "bitset labels must be increasing label ids"
        );
        let mut slot = vec![NONE; n_labels as usize];
        for (s, &l) in labels.iter().enumerate() {
            slot[l as usize] = s as u32;
        }
        Ok(Self {
            n_points: n_points as usize,
            theta,
            dict_hash,
            words: words as usize,
            slot,
            labels,
            bits,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::labels::LabelStoreBuilder;

    fn store() -> LabelStore {
        let mut b = LabelStoreBuilder::new();
        for i in 0..130u32 {
            let mut row = vec![format!("c={}", i % 3)];
            if i % 10 == 0 {
                row.push("c=rare".into());
            }
            b.push_point(row.iter().map(|l| ("c", l.as_str()))).unwrap();
        }
        b.finish().unwrap()
    }

    #[test]
    fn bitsets_cover_frequent_labels() {
        let s = store();
        let id = |n: &str| s.name_index()[n];
        let b = LabelBitsets::build(&s, 20);
        assert_eq!(
            b.n_bitsets(),
            3,
            "c=0, c=1, c=2 (43-44 points); c=rare has 13"
        );
        assert_eq!(b.slot(id("c=rare")), None);
        assert_eq!(b.bytes(), 3 * 3 * 8);
        assert_eq!(footprint(&s, 20), (3, 72, 130));
        let s1 = b.slot(id("c=1")).unwrap();
        assert_eq!(b.label(s1), id("c=1"));
        for p in 0..130 {
            assert_eq!(b.test(s1, p), p % 3 == 1, "point {p}");
        }
        b.check(&s).unwrap();

        let mut buf = Vec::new();
        b.write_to(&mut buf).unwrap();
        let c = LabelBitsets::read_from(Cursor::new(&buf)).unwrap();
        assert_eq!(c, b);
        c.check(&s).unwrap();

        // Every label has a bitset at θ = 1; none at a θ above n.
        assert_eq!(LabelBitsets::build(&s, 0).n_bitsets(), 4);
        assert_eq!(LabelBitsets::build(&s, 131).n_bitsets(), 0);
        let mut bad = b.clone();
        bad.bits[0] ^= 1 << 5;
        assert!(bad.check(&s).is_err());
        let mut high = b;
        high.theta = 1;
        assert!(high.check(&s).is_err(), "c=rare would need a bitset");
    }
}
