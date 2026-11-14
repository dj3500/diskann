/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Start selection (`Starts` role) and the per-label sample index of the `sample(k)` source.
//!
//! The selected starts are deduplicated in order (the first occurrence wins) and marked
//! visited. Per-label starts come from a start table ([`crate::starts::Starts`]) over the
//! dictionary of the *start predicate*: the query's label-view predicate when the run has a
//! label view (e.g. a market-label or product-label graph), else the query's predicate.

use anyhow::{ensure, Result};

use super::{
    config::{SampleStart, StartSource},
    eval::{Compiled, Eval, EvalCtx, RowScan},
    materialize::merge_distinct,
    search::Counters,
    visited::Visited,
};
use crate::{labels::LabelStore, predicate::Cnf, starts::Starts};

/// Per-label postings over the positions of a sample (the `sample(k)` start source): for each
/// label, the sorted positions of the sample entries whose point carries it.
#[derive(Debug, Clone)]
pub struct SampleIndex {
    offsets: Vec<u64>,
    positions: Vec<u32>,
    sample: Vec<u32>,
    start: Vec<u32>,
}

impl SampleIndex {
    /// Builds the index of the sample of `starts` from the label rows of its points.
    pub fn build(starts: &Starts, labels: &LabelStore) -> Result<Self> {
        let sample = starts.sample().to_vec();
        ensure!(!sample.is_empty(), "the start table has no sample");
        ensure!(
            starts.n_data() == labels.n_points(),
            "the start table refers to {} data points, the labels have {}",
            starts.n_data(),
            labels.n_points()
        );
        let nl = labels.n_labels();
        let mut offsets = vec![0u64; nl + 1];
        for &p in &sample {
            for &l in labels.row(p) {
                offsets[l as usize + 1] += 1;
            }
        }
        for l in 0..nl {
            offsets[l + 1] += offsets[l];
        }
        let mut cursor = offsets[..nl].to_vec();
        let mut positions = vec![0u32; offsets[nl] as usize];
        for (i, &p) in sample.iter().enumerate() {
            for &l in labels.row(p) {
                let at = &mut cursor[l as usize];
                positions[*at as usize] = i as u32;
                *at += 1;
            }
        }
        Ok(Self {
            offsets,
            positions,
            sample,
            start: starts.sample_start().to_vec(),
        })
    }

    /// Number of sample entries.
    pub fn len(&self) -> usize {
        self.sample.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sample.is_empty()
    }

    /// Bytes held by the index.
    pub fn bytes(&self) -> usize {
        8 * self.offsets.len() + 4 * (self.positions.len() + self.sample.len() + self.start.len())
    }

    /// Sorted sample positions carrying label `l`.
    #[inline]
    pub fn positions(&self, l: u32) -> &[u32] {
        let l = l as usize;
        &self.positions[self.offsets[l] as usize..self.offsets[l + 1] as usize]
    }
}

/// The predicate whose labels select per-label starts.
#[derive(Debug, Clone, Copy)]
pub enum StartPredicate<'a> {
    /// The query's (compiled) predicate.
    Base,
    /// A label-view predicate, with the posting offsets of the view's labels (to find the
    /// clause with the smallest upper bound).
    View {
        cnf: Cnf<'a>,
        postings: Option<&'a [u64]>,
    },
}

/// Inputs of start selection.
pub(crate) struct Inputs<'a> {
    pub sources: &'a [StartSource],
    pub sample_start: SampleStart,
    pub global: &'a [u32],
    pub label_starts: Option<&'a Starts>,
    pub sample: Option<&'a SampleIndex>,
    pub predicate: StartPredicate<'a>,
    pub n_data: u32,
    pub n_nodes: u32,
}

#[inline]
fn push(v: u32, visited: &mut Visited, out: &mut Vec<u32>) {
    if !visited.test_and_set(v) {
        out.push(v);
    }
}

fn label_starts(table: Option<&Starts>, l: u32, visited: &mut Visited, out: &mut Vec<u32>) {
    if let Some(t) = table {
        if (l as usize) < t.n_labels() {
            for &v in t.label_starts(l) {
                push(v, visited, out);
            }
        }
    }
}

/// Selects the starts of a query into `out` (deduplicated, marked visited).
pub(crate) fn select(
    inp: &Inputs<'_>,
    cx: &EvalCtx<'_>,
    visited: &mut Visited,
    out: &mut Vec<u32>,
    heap: &mut Vec<(u32, u32)>,
    pos: &mut Vec<usize>,
    ctr: &mut Counters,
) {
    out.clear();
    let q: &Compiled = cx.q;
    for &source in inp.sources {
        match source {
            StartSource::Global => {
                for &v in inp.global {
                    push(v, visited, out);
                }
            }
            StartSource::Aux => {
                for v in inp.n_data..inp.n_nodes {
                    push(v, visited, out);
                }
            }
            StartSource::LabelSmallest => match inp.predicate {
                StartPredicate::Base => {
                    if q.n_clauses() > 0 {
                        for &l in q.clause(0) {
                            label_starts(inp.label_starts, l, visited, out);
                        }
                    }
                }
                StartPredicate::View { cnf, postings } => {
                    let best = (0..cnf.n_clauses()).min_by_key(|&i| {
                        let u: u64 = match postings {
                            Some(p) => cnf
                                .clause(i)
                                .iter()
                                .map(|&l| p[l as usize + 1] - p[l as usize])
                                .sum(),
                            None => cnf.clause(i).len() as u64,
                        };
                        (u, i)
                    });
                    if let Some(i) = best {
                        for &l in cnf.clause(i) {
                            label_starts(inp.label_starts, l, visited, out);
                        }
                    }
                }
            },
            StartSource::LabelAll => match inp.predicate {
                StartPredicate::Base => {
                    for j in 0..q.n_clauses() {
                        for &l in q.clause(j) {
                            label_starts(inp.label_starts, l, visited, out);
                        }
                    }
                }
                StartPredicate::View { cnf, .. } => {
                    for clause in cnf.clauses() {
                        for &l in clause {
                            label_starts(inp.label_starts, l, visited, out);
                        }
                    }
                }
            },
            StartSource::Sample(k) => {
                let Some(si) = inp.sample else { continue };
                let mut taken = 0;
                let mut take = |i: u32, visited: &mut Visited, out: &mut Vec<u32>| {
                    let i = i as usize;
                    let v = match inp.sample_start {
                        SampleStart::Intended => si.sample[i],
                        SampleStart::Recorded => si.start[i],
                    };
                    push(v, visited, out);
                    taken += 1;
                    taken < k
                };
                if q.n_clauses() == 0 {
                    for i in 0..si.len().min(k) {
                        take(i as u32, visited, out);
                    }
                    continue;
                }
                let c0 = q.clause(0);
                let others = q.n_clauses() > 1;
                merge_distinct(
                    c0.len(),
                    |i| si.positions(c0[i]),
                    heap,
                    pos,
                    |i| {
                        let ok = !others || {
                            ctr.n_pred += 1;
                            RowScan::full(cx, si.sample[i as usize])
                        };
                        !ok || take(i, visited, out)
                    },
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph::{AuxNodes, Graph},
        kernel::eval::Weights,
        labels::LabelStoreBuilder,
        predicate::PredicateSetBuilder,
    };

    #[test]
    fn sample_starts_are_the_first_full_matches_in_sample_order() {
        let mut b = LabelStoreBuilder::new();
        let rows: [&[&str]; 6] = [
            &["a=0", "b=0"],
            &["a=1", "b=0"],
            &["a=1", "b=1"],
            &["a=1", "b=0"],
            &["a=0"],
            &["a=1", "b=0"],
        ];
        for row in rows {
            b.push_point(row.iter().map(|&l| (&l[..1], l))).unwrap();
        }
        let labels = b.finish().unwrap();
        let g = Graph::new(6, 2, AuxNodes::untyped()).unwrap();
        let mut table = Starts::new(&g);
        // Sample order 5, 0, 3, 1 with recorded starts 4, 4, 2, 2.
        table
            .set_sample(vec![5, 0, 3, 1], Some(vec![4, 4, 2, 2]))
            .unwrap();
        let si = SampleIndex::build(&table, &labels).unwrap();

        let mut pb = PredicateSetBuilder::new(&labels);
        pb.push(&[vec!["a=1"], vec!["b=0"]]);
        let preds = pb.finish();
        let mut bits = vec![0u64; labels.n_labels()];
        let mut q = Compiled::new();
        q.compile(&labels, preds.get(0), Weights::Unit, None, &mut bits);
        let cx = EvalCtx {
            labels: &labels,
            q: &q,
            bits: &bits,
            dense: &[],
            sparse: &[],
        };
        let run = |k: usize, mode: SampleStart| {
            let inp = Inputs {
                sources: &[StartSource::Sample(k), StartSource::Global],
                sample_start: mode,
                global: &[5],
                label_starts: None,
                sample: Some(&si),
                predicate: StartPredicate::Base,
                n_data: 6,
                n_nodes: 6,
            };
            let mut visited = Visited::new(6);
            let mut out = Vec::new();
            let mut ctr = Counters::default();
            select(
                &inp,
                &cx,
                &mut visited,
                &mut out,
                &mut Vec::new(),
                &mut Vec::new(),
                &mut ctr,
            );
            out
        };
        // Matching sample positions: 0 (point 5), 2 (point 3), 3 (point 1); the global start 5
        // is a duplicate.
        assert_eq!(run(2, SampleStart::Intended), [5, 3]);
        assert_eq!(run(10, SampleStart::Intended), [5, 3, 1]);
        assert_eq!(run(2, SampleStart::Recorded), [4, 2, 5]);
    }
}
