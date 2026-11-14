/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Boolean point labels: dictionary, forward store and postings (format `labels.flab`).
//!
//! Points have dense `u32` ids in input order. Every label has a dense `u32` id and belongs to
//! one anonymous category, also a dense `u32`. Categories are numbered in the natural order of
//! their names, and label ids are contiguous per category ([`LabelStore::category_labels`]),
//! following the natural order of the label names inside a category ([`natural_cmp`]). Every
//! forward row is therefore sorted and grouped by category.

use std::{
    cmp::Ordering,
    io::{Read, Seek, Write},
    ops::Range,
    path::Path,
};

use anyhow::{ensure, Context, Result};
use hashbrown::HashMap;

use crate::container;

/// File magic of `labels.flab`.
pub const MAGIC: &[u8; 8] = b"FLABLBL\0";
/// Version of the `labels.flab` format.
pub const VERSION: u32 = 1;

/// Concatenated UTF-8 strings with `u64` offsets (`offsets[0] == 0`).
#[derive(Debug, Clone)]
struct Strings {
    offsets: Vec<u64>,
    bytes: String,
}

impl Strings {
    fn new<'a>(items: impl IntoIterator<Item = &'a str>) -> Self {
        let mut offsets = vec![0];
        let mut bytes = String::new();
        for s in items {
            bytes.push_str(s);
            offsets.push(bytes.len() as u64);
        }
        Self { offsets, bytes }
    }

    fn from_parts(offsets: Vec<u64>, bytes: Vec<u8>) -> Result<Self> {
        let bytes = String::from_utf8(bytes).context("names are not valid UTF-8")?;
        ensure!(
            offsets.first() == Some(&0) && offsets.last() == Some(&(bytes.len() as u64)),
            "inconsistent name offsets"
        );
        for w in offsets.windows(2) {
            ensure!(
                w[0] <= w[1] && bytes.is_char_boundary(w[1] as usize),
                "inconsistent name offsets"
            );
        }
        Ok(Self { offsets, bytes })
    }

    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    fn get(&self, i: usize) -> &str {
        &self.bytes[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }

    fn iter(&self) -> impl Iterator<Item = &str> {
        (0..self.len()).map(|i| self.get(i))
    }
}

/// Natural order of names: maximal digit runs compare by numeric value, all other bytes
/// compare bytewise; names that are equal under this rule fall back to byte order, so the
/// order is total. Examples: `a2 < a10`, `id=19 < id=407`, `x=ab-c < x=ab-d`.
pub fn natural_cmp(a: &str, b: &str) -> Ordering {
    let (x, y) = (a.as_bytes(), b.as_bytes());
    let (mut i, mut j) = (0, 0);
    while i < x.len() && j < y.len() {
        if x[i].is_ascii_digit() && y[j].is_ascii_digit() {
            let (si, sj) = (i, j);
            while i < x.len() && x[i].is_ascii_digit() {
                i += 1;
            }
            while j < y.len() && y[j].is_ascii_digit() {
                j += 1;
            }
            let (u, v) = (strip_zeros(&x[si..i]), strip_zeros(&y[sj..j]));
            let ord = u.len().cmp(&v.len()).then_with(|| u.cmp(v));
            if ord != Ordering::Equal {
                return ord;
            }
        } else {
            if x[i] != y[j] {
                return x[i].cmp(&y[j]);
            }
            i += 1;
            j += 1;
        }
    }
    (x.len() - i).cmp(&(y.len() - j)).then_with(|| x.cmp(y))
}

fn strip_zeros(digits: &[u8]) -> &[u8] {
    let first = digits
        .iter()
        .position(|&d| d != b'0')
        .unwrap_or(digits.len());
    &digits[first..]
}

/// Immutable label store.
#[derive(Debug, Clone)]
pub struct LabelStore {
    n_points: usize,
    label_category: Vec<u32>,
    label_names: Strings,
    category_names: Strings,
    category_start: Vec<u32>,
    row_offsets: Vec<u64>,
    row_labels: Vec<u32>,
    posting_offsets: Vec<u64>,
    posting_points: Vec<u32>,
    dict_hash: u64,
}

impl LabelStore {
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    pub fn n_labels(&self) -> usize {
        self.label_category.len()
    }

    pub fn n_categories(&self) -> usize {
        self.category_start.len() - 1
    }

    /// Number of (point, label) incidences.
    pub fn n_entries(&self) -> usize {
        self.row_labels.len()
    }

    /// Sorted label ids of point `p`.
    #[inline]
    pub fn row(&self, p: u32) -> &[u32] {
        let p = p as usize;
        &self.row_labels[self.row_offsets[p] as usize..self.row_offsets[p + 1] as usize]
    }

    /// The forward store as raw arrays: `n_points + 1` row offsets and the concatenated rows
    /// (e.g. to prefetch a row before reading it).
    #[inline]
    pub fn forward(&self) -> (&[u64], &[u32]) {
        (&self.row_offsets, &self.row_labels)
    }

    /// Sorted point ids carrying label `l`.
    #[inline]
    pub fn postings(&self, l: u32) -> &[u32] {
        let l = l as usize;
        &self.posting_points[self.posting_offsets[l] as usize..self.posting_offsets[l + 1] as usize]
    }

    #[inline]
    pub fn posting_len(&self, l: u32) -> usize {
        let l = l as usize;
        (self.posting_offsets[l + 1] - self.posting_offsets[l]) as usize
    }

    /// Category of label `l`.
    #[inline]
    pub fn category(&self, l: u32) -> u32 {
        self.label_category[l as usize]
    }

    /// The contiguous id range of the labels of category `c`.
    pub fn category_labels(&self, c: u32) -> Range<u32> {
        self.category_start[c as usize]..self.category_start[c as usize + 1]
    }

    pub fn label_name(&self, l: u32) -> &str {
        self.label_names.get(l as usize)
    }

    pub fn category_name(&self, c: u32) -> &str {
        self.category_names.get(c as usize)
    }

    /// Fingerprint of the dictionary (names and categories). Predicate and ground truth files
    /// record it so that they are only used with the dictionary they were compiled against.
    pub fn dict_hash(&self) -> u64 {
        self.dict_hash
    }

    /// Map from label name to id.
    pub fn name_index(&self) -> HashMap<&str, u32> {
        self.label_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name, i as u32))
            .collect()
    }

    /// Writes `labels.flab` (see the crate `README.md` for the layout).
    pub fn write(&self, path: &Path) -> Result<()> {
        self.write_to(container::create(path)?)
            .with_context(|| format!("writing {}", path.display()))
    }

    /// Writes the `labels.flab` format to any writer.
    pub fn write_to<W: Write>(&self, w: W) -> Result<()> {
        let params = [
            self.n_points as u64,
            self.n_labels() as u64,
            self.n_categories() as u64,
            self.n_entries() as u64,
            self.dict_hash,
            0,
            0,
            0,
        ];
        container::write(
            w,
            MAGIC,
            VERSION,
            &params,
            &[
                bytemuck::cast_slice(&self.label_category),
                bytemuck::cast_slice(&self.label_names.offsets),
                self.label_names.bytes.as_bytes(),
                bytemuck::cast_slice(&self.category_names.offsets),
                self.category_names.bytes.as_bytes(),
                bytemuck::cast_slice(&self.category_start),
                bytemuck::cast_slice(&self.row_offsets),
                bytemuck::cast_slice(&self.row_labels),
                bytemuck::cast_slice(&self.posting_offsets),
                bytemuck::cast_slice(&self.posting_points),
            ],
        )
    }

    /// Reads and validates `labels.flab`.
    pub fn read(path: &Path) -> Result<Self> {
        container::Reader::open(path, MAGIC)
            .and_then(Self::load)
            .with_context(|| format!("loading labels from {}", path.display()))
    }

    /// Reads and validates the `labels.flab` format from any reader.
    pub fn read_from<R: Read + Seek>(r: R) -> Result<Self> {
        Self::load(container::Reader::new(r, MAGIC)?)
    }

    fn load<R: Read + Seek>(mut r: container::Reader<R>) -> Result<Self> {
        ensure!(
            r.version() == VERSION,
            "unsupported version {}",
            r.version()
        );
        let [n_points, n_labels, n_categories, n_entries, dict_hash, ..] = *r.params();
        let (np, nl, nc, ne) = (
            n_points as usize,
            n_labels as usize,
            n_categories as usize,
            n_entries as usize,
        );
        let label_category = r.section::<u32>(0, nl)?;
        let name_offsets = r.section::<u64>(1, nl + 1)?;
        let name_len = name_offsets.last().copied().unwrap_or(0) as usize;
        let label_names = Strings::from_parts(name_offsets, r.section::<u8>(2, name_len)?)?;
        let cat_offsets = r.section::<u64>(3, nc + 1)?;
        let cat_len = cat_offsets.last().copied().unwrap_or(0) as usize;
        let category_names = Strings::from_parts(cat_offsets, r.section::<u8>(4, cat_len)?)?;
        let store = Self {
            n_points: np,
            label_category,
            label_names,
            category_names,
            category_start: r.section(5, nc + 1)?,
            row_offsets: r.section(6, np + 1)?,
            row_labels: r.section(7, ne)?,
            posting_offsets: r.section(8, nl + 1)?,
            posting_points: r.section(9, ne)?,
            dict_hash,
        };
        store.validate()?;
        Ok(store)
    }

    /// Checks all structural invariants (offsets, sortedness, ranges, forward/posting counts,
    /// dictionary fingerprint).
    fn validate(&self) -> Result<()> {
        let (np, nl, nc) = (self.n_points, self.n_labels(), self.n_categories());
        ensure!(
            u32::try_from(np).is_ok() && u32::try_from(nl).is_ok(),
            "too many points or labels"
        );
        ensure!(
            self.label_names.len() == nl && self.category_names.len() == nc,
            "dictionary sizes disagree"
        );
        ensure!(
            self.category_start.first() == Some(&0)
                && self.category_start.last() == Some(&(nl as u32))
                && self.category_start.windows(2).all(|w| w[0] <= w[1]),
            "inconsistent category ranges"
        );
        for (l, &c) in self.label_category.iter().enumerate() {
            ensure!(
                (c as usize) < nc && self.category_labels(c).contains(&(l as u32)),
                "label {l} lies outside the id range of its category {c}"
            );
        }
        check_csr(&self.row_offsets, &self.row_labels, np, nl, "point")?;
        check_csr(&self.posting_offsets, &self.posting_points, nl, np, "label")?;
        let mut counts = vec![0u64; nl];
        for &l in &self.row_labels {
            counts[l as usize] += 1;
        }
        for (l, &count) in counts.iter().enumerate() {
            ensure!(
                count == self.posting_len(l as u32) as u64,
                "forward store and postings disagree on label {l}"
            );
        }
        ensure!(
            dictionary_hash(
                &self.category_names,
                &self.label_names,
                &self.label_category
            ) == self.dict_hash,
            "dictionary fingerprint mismatch"
        );
        Ok(())
    }
}

/// Checks a CSR: `n_rows + 1` monotone offsets covering `values`, every row strictly
/// increasing and below `bound`.
fn check_csr(
    offsets: &[u64],
    values: &[u32],
    n_rows: usize,
    bound: usize,
    what: &str,
) -> Result<()> {
    ensure!(
        offsets.len() == n_rows + 1 && offsets[0] == 0 && offsets[n_rows] == values.len() as u64,
        "inconsistent {what} offsets"
    );
    for (i, w) in offsets.windows(2).enumerate() {
        ensure!(w[0] <= w[1], "inconsistent {what} offsets at {i}");
        let row = &values[w[0] as usize..w[1] as usize];
        ensure!(
            row.windows(2).all(|p| p[0] < p[1]),
            "{what} {i}: entries are not strictly increasing"
        );
        if let Some(&last) = row.last() {
            ensure!(
                (last as usize) < bound,
                "{what} {i}: entry {last} out of range"
            );
        }
    }
    Ok(())
}

/// FNV-1a 64 over the category names and, per label, its category and name.
fn dictionary_hash(categories: &Strings, labels: &Strings, label_category: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    };
    feed(&(categories.len() as u64).to_le_bytes());
    for name in categories.iter() {
        feed(&(name.len() as u64).to_le_bytes());
        feed(name.as_bytes());
    }
    feed(&(labels.len() as u64).to_le_bytes());
    for (name, &c) in labels.iter().zip(label_category) {
        feed(&c.to_le_bytes());
        feed(&(name.len() as u64).to_le_bytes());
        feed(name.as_bytes());
    }
    h
}

/// Incremental construction of a [`LabelStore`], one point at a time.
#[derive(Debug, Default)]
pub struct LabelStoreBuilder {
    ids: HashMap<String, u32>,
    names: Vec<String>,
    label_category: Vec<u32>,
    categories: HashMap<String, u32>,
    category_names: Vec<String>,
    row_offsets: Vec<u64>,
    row_labels: Vec<u32>,
}

impl LabelStoreBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of points added so far.
    pub fn n_points(&self) -> usize {
        self.row_offsets.len()
    }

    /// Adds the next point, given its labels as `(category, label)` name pairs. Repeated labels
    /// are merged.
    ///
    /// # Errors
    ///
    /// Fails if a label name was seen before with a different category.
    pub fn push_point<'a>(
        &mut self,
        labels: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<()> {
        for (category, label) in labels {
            let id = match self.ids.get(label) {
                Some(&id) => {
                    ensure!(
                        self.category_names[self.label_category[id as usize] as usize] == category,
                        "label {label:?} appears in categories {:?} and {category:?}",
                        self.category_names[self.label_category[id as usize] as usize]
                    );
                    id
                }
                None => {
                    let c = match self.categories.get(category) {
                        Some(&c) => c,
                        None => {
                            let c = self.category_names.len() as u32;
                            self.categories.insert(category.to_owned(), c);
                            self.category_names.push(category.to_owned());
                            c
                        }
                    };
                    let id = u32::try_from(self.names.len()).context("too many labels")?;
                    self.ids.insert(label.to_owned(), id);
                    self.names.push(label.to_owned());
                    self.label_category.push(c);
                    id
                }
            };
            self.row_labels.push(id);
        }
        self.row_offsets.push(self.row_labels.len() as u64);
        Ok(())
    }

    /// Assigns the final ids (see the module docs) and builds forward rows and postings.
    pub fn finish(self) -> Result<LabelStore> {
        let n_points = self.row_offsets.len();
        ensure!(u32::try_from(n_points).is_ok(), "too many points");
        let nc = self.category_names.len();
        let nl = self.names.len();

        let mut cat_order: Vec<usize> = (0..nc).collect();
        cat_order.sort_by(|&a, &b| natural_cmp(&self.category_names[a], &self.category_names[b]));
        let mut cat_rank = vec![0u32; nc];
        for (rank, &c) in cat_order.iter().enumerate() {
            cat_rank[c] = rank as u32;
        }

        let mut order: Vec<usize> = (0..nl).collect();
        order.sort_by(|&a, &b| {
            cat_rank[self.label_category[a] as usize]
                .cmp(&cat_rank[self.label_category[b] as usize])
                .then_with(|| natural_cmp(&self.names[a], &self.names[b]))
        });
        let mut remap = vec![0u32; nl];
        for (new, &old) in order.iter().enumerate() {
            remap[old] = new as u32;
        }
        let label_category: Vec<u32> = order
            .iter()
            .map(|&old| cat_rank[self.label_category[old] as usize])
            .collect();
        let mut category_start = vec![0u32; nc + 1];
        for &c in &label_category {
            category_start[c as usize + 1] += 1;
        }
        for c in 0..nc {
            category_start[c + 1] += category_start[c];
        }

        let mut row_offsets = Vec::with_capacity(n_points + 1);
        row_offsets.push(0u64);
        let mut row_labels = Vec::with_capacity(self.row_labels.len());
        let mut scratch = Vec::new();
        let mut start = 0usize;
        for &end in &self.row_offsets {
            scratch.clear();
            scratch.extend(
                self.row_labels[start..end as usize]
                    .iter()
                    .map(|&l| remap[l as usize]),
            );
            scratch.sort_unstable();
            scratch.dedup();
            row_labels.extend_from_slice(&scratch);
            row_offsets.push(row_labels.len() as u64);
            start = end as usize;
        }

        let mut posting_offsets = vec![0u64; nl + 1];
        for &l in &row_labels {
            posting_offsets[l as usize + 1] += 1;
        }
        for l in 0..nl {
            posting_offsets[l + 1] += posting_offsets[l];
        }
        let mut cursor = posting_offsets[..nl].to_vec();
        let mut posting_points = vec![0u32; row_labels.len()];
        for (p, w) in row_offsets.windows(2).enumerate() {
            for &l in &row_labels[w[0] as usize..w[1] as usize] {
                let at = &mut cursor[l as usize];
                posting_points[*at as usize] = p as u32;
                *at += 1;
            }
        }

        let label_names = Strings::new(order.iter().map(|&old| self.names[old].as_str()));
        let category_names =
            Strings::new(cat_order.iter().map(|&c| self.category_names[c].as_str()));
        let dict_hash = dictionary_hash(&category_names, &label_names, &label_category);
        Ok(LabelStore {
            n_points,
            label_category,
            label_names,
            category_names,
            category_start,
            row_offsets,
            row_labels,
            posting_offsets,
            posting_points,
            dict_hash,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> LabelStore {
        let rows: [&[(&str, &str)]; 4] = [
            &[
                ("C2", "C2=4"),
                ("C10", "C10=x"),
                ("C2", "C2=10"),
                ("C2", "C2=4"),
            ],
            &[],
            &[("C10", "C10=x"), ("C2", "C2=9")],
            &[("C2", "C2=10")],
        ];
        let mut b = LabelStoreBuilder::new();
        for row in rows {
            b.push_point(row.iter().copied()).unwrap();
        }
        b.finish().unwrap()
    }

    #[test]
    fn natural_order() {
        let mut v = [
            "c10", "c2", "c1=407", "c1=19", "c1=019", "b", "a1b", "a01", "a",
        ];
        v.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(
            v,
            ["a", "a01", "a1b", "b", "c1=019", "c1=19", "c1=407", "c2", "c10"]
        );
    }

    #[test]
    fn ids_are_grouped_by_category_in_natural_order() {
        let s = sample();
        assert_eq!(
            (s.n_points(), s.n_labels(), s.n_categories(), s.n_entries()),
            (4, 4, 2, 6)
        );
        assert_eq!((s.category_name(0), s.category_name(1)), ("C2", "C10"));
        let names: Vec<&str> = (0..4).map(|l| s.label_name(l)).collect();
        assert_eq!(names, ["C2=4", "C2=9", "C2=10", "C10=x"]);
        assert_eq!((s.category_labels(0), s.category_labels(1)), (0..3, 3..4));
        assert_eq!(s.row(0), &[0, 2, 3]);
        assert!(s.row(1).is_empty());
        assert_eq!(s.row(2), &[1, 3]);
        assert_eq!(s.postings(2), &[0, 3]);
        assert_eq!(s.postings(3), &[0, 2]);
        assert_eq!(s.name_index()["C2=9"], 1);
    }

    #[test]
    fn rejects_label_in_two_categories() {
        let mut b = LabelStoreBuilder::new();
        b.push_point([("a", "x")]).unwrap();
        assert!(b.push_point([("b", "x")]).is_err());
    }

    #[test]
    fn file_roundtrip() {
        let s = sample();
        let mut buf = Vec::new();
        s.write_to(&mut buf).unwrap();
        let t = LabelStore::read_from(std::io::Cursor::new(&buf)).unwrap();
        assert_eq!(t.dict_hash(), s.dict_hash());
        for p in 0..4 {
            assert_eq!(t.row(p), s.row(p));
        }
        for l in 0..4 {
            assert_eq!(
                (t.postings(l), t.label_name(l)),
                (s.postings(l), s.label_name(l))
            );
        }
    }
}
