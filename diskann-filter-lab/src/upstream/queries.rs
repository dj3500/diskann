/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Query sets for timed runs: all queries of one query file, or the rows of a query panel.
//!
//! A dataset directory holds `labels.flab` and, per query set `S`, the vectors `qS.u8bin` or
//! `qS.fbin` and the predicates `qS.fpred`. A **panel** is a TSV file with a header line and
//! one row per query; the columns `qset` (query set name `S`) and `qidx` (row in that set)
//! are required, `split` is optional, other columns are ignored. Rows keep panel order.

use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use hashbrown::HashMap;

use crate::{
    labels::LabelStore,
    predicate::{Cnf, PredicateSet},
    vectors::{DType, Element, Matrix},
};

/// Where the queries of a run come from.
#[derive(Debug, Clone)]
pub enum QuerySource {
    /// Every query of query set `S` of the dataset.
    Set(String),
    /// Explicit vector and predicate files (every query).
    Files {
        vectors: PathBuf,
        predicates: PathBuf,
    },
    /// The rows of a panel file.
    Panel(PathBuf),
}

/// A panel row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PanelRow {
    pub qset: String,
    pub qidx: usize,
    pub split: Option<String>,
}

/// Reads a panel file.
pub fn read_panel(path: &Path) -> Result<Vec<PanelRow>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    parse_panel(&text).with_context(|| format!("parsing {}", path.display()))
}

fn parse_panel(text: &str) -> Result<Vec<PanelRow>> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines.next().context("empty panel")?.split('\t').collect();
    let col = |name: &str| header.iter().position(|h| h.trim() == name);
    let (Some(qset), Some(qidx)) = (col("qset"), col("qidx")) else {
        bail!("a panel needs the columns `qset` and `qidx`");
    };
    let split = col("split");
    lines
        .enumerate()
        .map(|(i, line)| {
            let fields: Vec<&str> = line.split('\t').map(str::trim).collect();
            let get = |c: usize| {
                fields
                    .get(c)
                    .copied()
                    .with_context(|| format!("row {i}: missing column {c}"))
            };
            Ok(PanelRow {
                qset: get(qset)?.to_owned(),
                qidx: get(qidx)?
                    .parse()
                    .with_context(|| format!("row {i}: bad qidx"))?,
                split: match split {
                    Some(c) => Some(get(c)?.to_owned()),
                    None => None,
                },
            })
        })
        .collect()
}

/// Query vectors and predicates of a run, in run order.
#[derive(Debug)]
pub struct Queries<T> {
    sets: Vec<(String, Matrix<T>, PredicateSet)>,
    /// Per row: (index into `sets`, row in that set).
    rows: Vec<(usize, usize)>,
    /// Per row: panel split, if the source is a panel with a `split` column.
    pub splits: Vec<Option<String>>,
}

impl<T: Element> Queries<T> {
    /// Loads the queries of `source`, checking dimensions and the predicates' dictionary.
    pub fn load(
        dataset: &Path,
        source: &QuerySource,
        labels: &LabelStore,
        dim: usize,
        max_rows: Option<usize>,
    ) -> Result<Self> {
        let ext = match T::NAME {
            "u8" => "u8bin",
            _ => "fbin",
        };
        let open = |vectors: &Path, predicates: &Path| -> Result<(Matrix<T>, PredicateSet)> {
            let m = Matrix::<T>::read_bin(vectors)?;
            let p = PredicateSet::read(predicates)?;
            p.check_dictionary(labels)?;
            ensure!(
                m.ncols() == dim,
                "{}: dimension {} differs from the index dimension {dim}",
                vectors.display(),
                m.ncols()
            );
            ensure!(
                m.nrows() == p.len(),
                "{} has {} vectors but {} has {} predicates",
                vectors.display(),
                m.nrows(),
                predicates.display(),
                p.len()
            );
            Ok((m, p))
        };
        let set_files = |name: &str| {
            (
                dataset.join(format!("q{name}.{ext}")),
                dataset.join(format!("q{name}.fpred")),
            )
        };
        let mut queries = match source {
            QuerySource::Set(name) => {
                let (v, p) = set_files(name);
                let (m, p) = open(&v, &p)?;
                let n = m.nrows();
                Self {
                    sets: vec![(name.clone(), m, p)],
                    rows: (0..n).map(|i| (0, i)).collect(),
                    splits: vec![None; n],
                }
            }
            QuerySource::Files {
                vectors,
                predicates,
            } => {
                let (m, p) = open(vectors, predicates)?;
                let n = m.nrows();
                Self {
                    sets: vec![(vectors.display().to_string(), m, p)],
                    rows: (0..n).map(|i| (0, i)).collect(),
                    splits: vec![None; n],
                }
            }
            QuerySource::Panel(path) => {
                let panel = read_panel(path)?;
                let mut index: HashMap<String, usize> = HashMap::new();
                let mut sets = Vec::new();
                let mut rows = Vec::with_capacity(panel.len());
                for row in &panel {
                    let s = match index.get(&row.qset) {
                        Some(&s) => s,
                        None => {
                            let (v, p) = set_files(&row.qset);
                            let (m, p) = open(&v, &p)?;
                            sets.push((row.qset.clone(), m, p));
                            index.insert(row.qset.clone(), sets.len() - 1);
                            sets.len() - 1
                        }
                    };
                    ensure!(
                        row.qidx < sets[s].1.nrows(),
                        "panel row qset {} qidx {} is out of range",
                        row.qset,
                        row.qidx
                    );
                    rows.push((s, row.qidx));
                }
                Self {
                    sets,
                    rows,
                    splits: panel.into_iter().map(|r| r.split).collect(),
                }
            }
        };
        if let Some(max) = max_rows {
            queries.rows.truncate(max);
            queries.splits.truncate(max);
        }
        Ok(queries)
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    #[inline]
    pub fn vector(&self, row: usize) -> &[T] {
        let (s, i) = self.rows[row];
        self.sets[s].1.row(i)
    }

    #[inline]
    pub fn predicate(&self, row: usize) -> Cnf<'_> {
        let (s, i) = self.rows[row];
        self.sets[s].2.get(i)
    }

    /// Names of the query sets used, in first-use order.
    pub fn set_names(&self) -> Vec<&str> {
        self.sets.iter().map(|s| s.0.as_str()).collect()
    }
}

/// Element type of the queries of a dataset, from the extension of its base vectors.
pub fn dataset_dtype(dataset: &Path) -> Result<DType> {
    match (
        dataset.join("base.u8bin").exists(),
        dataset.join("base.fbin").exists(),
    ) {
        (true, false) => Ok(DType::U8),
        (false, true) => Ok(DType::F32),
        _ => bail!(
            "cannot infer the element type of {} (expected exactly one of base.u8bin, base.fbin)",
            dataset.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panels() {
        let rows = parse_panel("qset\tqidx\tband\tsplit\n10K\t5\t3\tdiscovery\n2K\t0\t0\taudit\n")
            .unwrap();
        assert_eq!(
            rows,
            [
                PanelRow {
                    qset: "10K".into(),
                    qidx: 5,
                    split: Some("discovery".into())
                },
                PanelRow {
                    qset: "2K".into(),
                    qidx: 0,
                    split: Some("audit".into())
                }
            ]
        );
        assert!(parse_panel("qset\tband\n10K\t1\n").is_err());
        let rows = parse_panel("qidx\tqset\n7\ta\n").unwrap();
        assert_eq!((rows[0].qidx, rows[0].split.as_deref()), (7, None));
    }
}
