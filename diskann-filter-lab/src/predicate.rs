/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Predicates in conjunctive normal form (AND of ORs) over label ids.
//!
//! A predicate is a list of clauses; a clause is a sorted, duplicate-free list of label ids and
//! is satisfied by a point that carries at least one of them; a point matches when it
//! satisfies every clause. Clauses may mix categories, and several clauses may use the same
//! category. Labels that are absent from the dictionary are dropped when a predicate is
//! compiled (no point carries them): a clause made only of unknown labels becomes empty, and a
//! predicate with an empty clause matches nothing. A predicate without clauses matches every
//! point.

use std::{
    io::{Read, Seek, Write},
    path::Path,
};

use anyhow::{ensure, Context, Result};
use hashbrown::HashMap;

use crate::{container, labels::LabelStore};

/// File magic of predicate files.
pub const MAGIC: &[u8; 8] = b"FLABPRD\0";
/// Version of the predicate file format.
pub const VERSION: u32 = 1;

/// Whether two sorted lists share an element (binary search of each `clause` label in the
/// remaining part of `row`).
#[inline]
pub fn intersects(row: &[u32], clause: &[u32]) -> bool {
    let mut rest = row;
    for &l in clause {
        match rest.binary_search(&l) {
            Ok(_) => return true,
            Err(pos) => rest = &rest[pos..],
        }
    }
    false
}

/// One predicate: a view of its clauses.
#[derive(Debug, Clone, Copy)]
pub struct Cnf<'a> {
    bounds: &'a [u64],
    labels: &'a [u32],
}

impl<'a> Cnf<'a> {
    pub fn n_clauses(&self) -> usize {
        self.bounds.len() - 1
    }

    /// Sorted label ids of clause `i`.
    #[inline]
    pub fn clause(&self, i: usize) -> &'a [u32] {
        &self.labels[self.bounds[i] as usize..self.bounds[i + 1] as usize]
    }

    pub fn clauses(&self) -> impl ExactSizeIterator<Item = &'a [u32]> + 'a {
        let this = *self;
        (0..this.n_clauses()).map(move |i| this.clause(i))
    }

    /// Whether some clause is empty, so that no point can match.
    pub fn is_unsatisfiable(&self) -> bool {
        self.clauses().any(<[u32]>::is_empty)
    }

    /// Whether a point with the sorted label ids `row` matches.
    #[inline]
    pub fn matches(&self, row: &[u32]) -> bool {
        self.clauses().all(|c| intersects(row, c))
    }

    /// Like [`Self::matches`], skipping clause `skip` (e.g. one already known to hold).
    #[inline]
    pub fn matches_except(&self, row: &[u32], skip: usize) -> bool {
        self.clauses()
            .enumerate()
            .all(|(i, c)| i == skip || intersects(row, c))
    }

    /// Number of clauses that `row` does not satisfy.
    #[inline]
    pub fn failed_clauses(&self, row: &[u32]) -> usize {
        self.clauses().filter(|c| !intersects(row, c)).count()
    }
}

/// A list of predicates compiled against one label dictionary.
#[derive(Debug, Clone)]
pub struct PredicateSet {
    n_labels: u32,
    dict_hash: u64,
    query_offsets: Vec<u64>,
    clause_offsets: Vec<u64>,
    labels: Vec<u32>,
    unknown: Vec<u32>,
}

impl PredicateSet {
    /// Number of predicates.
    pub fn len(&self) -> usize {
        self.query_offsets.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn get(&self, q: usize) -> Cnf<'_> {
        let (a, b) = (
            self.query_offsets[q] as usize,
            self.query_offsets[q + 1] as usize,
        );
        Cnf {
            bounds: &self.clause_offsets[a..=b],
            labels: &self.labels,
        }
    }

    /// Number of label occurrences of predicate `q` that were dropped because the dictionary
    /// does not contain them.
    pub fn unknown_labels(&self, q: usize) -> u32 {
        self.unknown[q]
    }

    /// Fingerprint of the dictionary the predicates were compiled against.
    pub fn dict_hash(&self) -> u64 {
        self.dict_hash
    }

    /// Checks that the predicates were compiled against the dictionary of `store`.
    pub fn check_dictionary(&self, store: &LabelStore) -> Result<()> {
        ensure!(
            self.dict_hash == store.dict_hash() && self.n_labels as usize == store.n_labels(),
            "predicates were compiled against a different label dictionary"
        );
        Ok(())
    }

    /// Writes a predicate file (see the crate `README.md` for the layout).
    pub fn write(&self, path: &Path) -> Result<()> {
        self.write_to(container::create(path)?)
            .with_context(|| format!("writing {}", path.display()))
    }

    /// Writes the predicate file format to any writer.
    pub fn write_to<W: Write>(&self, w: W) -> Result<()> {
        let params = [
            self.len() as u64,
            (self.clause_offsets.len() - 1) as u64,
            self.labels.len() as u64,
            u64::from(self.n_labels),
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
                bytemuck::cast_slice(&self.query_offsets),
                bytemuck::cast_slice(&self.clause_offsets),
                bytemuck::cast_slice(&self.labels),
                bytemuck::cast_slice(&self.unknown),
            ],
        )
    }

    /// Reads and validates a predicate file.
    pub fn read(path: &Path) -> Result<Self> {
        container::Reader::open(path, MAGIC)
            .and_then(Self::load)
            .with_context(|| format!("loading predicates from {}", path.display()))
    }

    /// Reads and validates the predicate file format from any reader.
    pub fn read_from<R: Read + Seek>(r: R) -> Result<Self> {
        Self::load(container::Reader::new(r, MAGIC)?)
    }

    fn load<R: Read + Seek>(mut r: container::Reader<R>) -> Result<Self> {
        ensure!(
            r.version() == VERSION,
            "unsupported version {}",
            r.version()
        );
        let [nq, nc, nr, n_labels, dict_hash, ..] = *r.params();
        let set = Self {
            n_labels: u32::try_from(n_labels).context("too many labels")?,
            dict_hash,
            query_offsets: r.section(0, nq as usize + 1)?,
            clause_offsets: r.section(1, nc as usize + 1)?,
            labels: r.section(2, nr as usize)?,
            unknown: r.section(3, nq as usize)?,
        };
        set.validate()?;
        Ok(set)
    }

    fn validate(&self) -> Result<()> {
        for (offsets, len, what) in [
            (&self.query_offsets, self.clause_offsets.len() - 1, "query"),
            (&self.clause_offsets, self.labels.len(), "clause"),
        ] {
            ensure!(
                offsets.first() == Some(&0)
                    && offsets.last() == Some(&(len as u64))
                    && offsets.windows(2).all(|w| w[0] <= w[1]),
                "inconsistent {what} offsets"
            );
        }
        for w in self.clause_offsets.windows(2) {
            let clause = &self.labels[w[0] as usize..w[1] as usize];
            ensure!(
                clause.windows(2).all(|p| p[0] < p[1])
                    && clause.last().is_none_or(|&l| l < self.n_labels),
                "clause labels must be sorted, distinct and within the dictionary"
            );
        }
        Ok(())
    }
}

/// Compiles predicates given by label names against a dictionary.
#[derive(Debug)]
pub struct PredicateSetBuilder<'a> {
    index: HashMap<&'a str, u32>,
    set: PredicateSet,
    scratch: Vec<u32>,
    unknown_names: HashMap<String, u32>,
}

impl<'a> PredicateSetBuilder<'a> {
    pub fn new(store: &'a LabelStore) -> Self {
        Self {
            index: store.name_index(),
            set: PredicateSet {
                n_labels: store.n_labels() as u32,
                dict_hash: store.dict_hash(),
                query_offsets: vec![0],
                clause_offsets: vec![0],
                labels: Vec::new(),
                unknown: Vec::new(),
            },
            scratch: Vec::new(),
            unknown_names: HashMap::new(),
        }
    }

    /// Appends one predicate, given as clauses of label names.
    pub fn push<S: AsRef<str>>(&mut self, clauses: &[Vec<S>]) {
        let set = &mut self.set;
        let mut unknown = 0;
        for clause in clauses {
            self.scratch.clear();
            for name in clause {
                match self.index.get(name.as_ref()) {
                    Some(&id) => self.scratch.push(id),
                    None => {
                        unknown += 1;
                        *self.unknown_names.entry_ref(name.as_ref()).or_insert(0) += 1;
                    }
                }
            }
            self.scratch.sort_unstable();
            self.scratch.dedup();
            set.labels.extend_from_slice(&self.scratch);
            set.clause_offsets.push(set.labels.len() as u64);
        }
        set.query_offsets
            .push((set.clause_offsets.len() - 1) as u64);
        set.unknown.push(unknown);
    }

    /// Label names pushed so far that are absent from the dictionary, with their numbers of
    /// occurrences (most frequent first).
    pub fn unknown_names(&self) -> Vec<(&str, u32)> {
        let mut names: Vec<(&str, u32)> = self
            .unknown_names
            .iter()
            .map(|(name, &count)| (name.as_str(), count))
            .collect();
        names.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        names
    }

    pub fn finish(self) -> PredicateSet {
        self.set
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::labels::LabelStoreBuilder;

    #[test]
    fn intersection_of_sorted_lists() {
        assert!(intersects(&[1, 4, 9], &[0, 9]));
        assert!(intersects(&[1, 4, 9], &[4]));
        assert!(!intersects(&[1, 4, 9], &[0, 2, 5, 10]));
        assert!(!intersects(&[], &[1]));
        assert!(!intersects(&[1], &[]));
    }

    #[test]
    fn evaluation_and_unknown_labels() {
        let mut b = LabelStoreBuilder::new();
        b.push_point([("g", "g=0"), ("d", "d=1"), ("m", "m=us")])
            .unwrap();
        b.push_point([("g", "g=7"), ("d", "d=2"), ("m", "m=us")])
            .unwrap();
        let store = b.finish().unwrap();
        let id = |name: &str| store.name_index()[name];

        let mut pb = PredicateSetBuilder::new(&store);
        pb.push(&[vec!["g=7", "g=0", "g=0"], vec!["d=1"]]);
        pb.push(&[vec!["g=7", "g=nowhere"], vec!["m=us"]]);
        pb.push(&[vec!["g=nowhere"], vec!["m=us"]]);
        pb.push::<&str>(&[]);
        assert_eq!(pb.unknown_names(), [("g=nowhere", 2)]);
        let set = pb.finish();
        let mut buf = Vec::new();
        set.write_to(&mut buf).unwrap();
        let set = PredicateSet::read_from(std::io::Cursor::new(&buf)).unwrap();
        set.check_dictionary(&store).unwrap();

        let (p0, p1) = (store.row(0), store.row(1));
        let q0 = set.get(0);
        let mut g = [id("g=0"), id("g=7")];
        g.sort_unstable();
        assert_eq!((q0.n_clauses(), q0.clause(0)), (2, &g[..]));
        assert!(q0.matches(p0) && !q0.matches(p1));
        assert_eq!((q0.failed_clauses(p0), q0.failed_clauses(p1)), (0, 1));
        assert!(q0.matches_except(p1, 1));

        let q1 = set.get(1);
        assert_eq!((set.unknown_labels(1), q1.clause(0)), (1, &[id("g=7")][..]));
        assert!(!q1.matches(p0) && q1.matches(p1));

        let q2 = set.get(2);
        assert!(q2.is_unsatisfiable() && q2.clause(0).is_empty());
        assert!(!q2.matches(p0) && !q2.matches(p1));
        assert_eq!(q2.failed_clauses(p1), 1);

        let q3 = set.get(3);
        assert_eq!(q3.n_clauses(), 0);
        assert!(q3.matches(p0) && !q3.is_unsatisfiable());
    }
}
