/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Provenance of derived files: cheap fingerprints of the label store, the graph and the base
//! vectors a file was built from, recorded in the file and compared with the inputs it is
//! loaded with. A mismatch is an error unless the command is given
//! `--allow-provenance-mismatch` (then a warning); a file written before the fingerprints
//! existed loads with a warning.
//!
//! **Fingerprints** (JSON; hashes as `"0x"` and 16 hexadecimal digits):
//! - `labels`: `n_points`, `n_labels`, `n_entries` (point-label incidences), `dict_hash` (the
//!   dictionary fingerprint of `labels.flab`) and `postings_hash` = `H(|P_0|, …, |P_{L−1}|,
//!   p[s] for s in samples(E))`: every label's posting size, then sampled entries of the
//!   concatenated postings `p` (labels in id order, `E` entries).
//! - `graph`: `n_data`, `n_nodes`, `data_edges` (out-edges of the data points), `edges` (of all
//!   nodes), `data_hash` = `H(rows of the data points v in samples(n_data))` and `hash` = the
//!   same over `samples(n_nodes)`, where a row contributes its degree and then its neighbour ids
//!   in stored order. `flab steiner` appends routers without changing the rows of the base
//!   graph's nodes, so a Steiner graph and its base share the data part (`n_data`,
//!   `data_edges`, `data_hash`).
//! - `samples(m)`: every position `0..m` when `m ≤ 65,536`, else 1,024 runs of 64 consecutive
//!   positions starting at `⌊i·m/1,024⌋` (`i = 0..1,024`), in increasing order: few random
//!   reads of a file.
//! - `base`: `n`, `dim`, `dtype` of the base vectors; `cpu_model`: `model name` of
//!   `/proc/cpuinfo` (calibrations).
//!
//! `H(w_1, …, w_m) = mix(acc ⊕ mix(m))` with `acc = Σ_i mix(w_i + i·φ)` (all arithmetic
//! modulo 2⁶⁴, `i` from 1, `φ = 0x9e3779b97f4a7c15`) and `mix` the SplitMix64 finalizer
//! (`z ^= z >> 30; z *= 0xbf58476d1ce4e5b9; z ^= z >> 27; z *= 0x94d049bb133111eb;
//! z ^= z >> 31`): vectorizable, e.g. with NumPy `uint64` arrays.
//!
//! **Where it is recorded**: container files (`pairs.flab`, `islands.flab`, `bitsets.flab`,
//! `sketches.flab`, `starts.flab`, `pair_roots.flab`) carry it as one extra section after their
//! own, `FLABPRV\0` followed by the JSON object; the pair contexts of `flab select`
//! (`SELCAND3`) after their records; JSON outputs (calibrations, selections, `flab steiner`'s
//! `meta.json`) under the key `provenance`; label lists (`trees.txt`) in a first comment line
//! `# provenance {…}`. `flab provenance` prints the fingerprints of a dataset and a graph, and
//! with `--file` what a file records.
//!
//! **What is compared**: the label store for every file; the graph's data part for
//! calibrations (unit costs and effort curves are those of the data points' graph, shared by its
//! Steiner graphs) and the whole graph for pair roots and for start tables of the graph's node
//! count (they may name routers; a table of another graph holds data-point starts only and is
//! not bound to a graph); the base shape and the CPU model for calibrations.

use std::{
    io::{Read, Seek},
    path::Path,
};

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    container, diskann_io,
    graph::{self, Graph},
    labels::LabelStore,
    vectors::{Element, Matrix},
};

/// Magic of the provenance section of a container file.
pub const SECTION_MAGIC: &[u8; 8] = b"FLABPRV\0";
/// Sampled positions (postings, graph rows): runs of [`RUN`] consecutive positions at [`RUNS`]
/// evenly spaced starts, every position of a shorter sequence.
pub const RUNS: usize = 1 << 10;
pub const RUN: usize = 64;
const PHI: u64 = 0x9e37_79b9_7f4a_7c15;

/// The SplitMix64 finalizer.
#[inline]
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The hash `H` of a word sequence (see the module documentation).
#[derive(Debug, Default)]
struct Hasher {
    acc: u64,
    n: u64,
}

impl Hasher {
    #[inline]
    fn push(&mut self, w: u64) {
        self.n += 1;
        self.acc = self
            .acc
            .wrapping_add(mix(w.wrapping_add(self.n.wrapping_mul(PHI))));
    }

    fn finish(&self) -> u64 {
        mix(self.acc ^ mix(self.n))
    }
}

/// The sampled positions of `m` items (see the module documentation), increasing.
fn samples(m: usize) -> impl Iterator<Item = usize> {
    let all = m <= RUNS * RUN;
    let (runs, len) = if all { (1, m) } else { (RUNS, RUN) };
    (0..runs).flat_map(move |i| {
        let start = if all {
            0
        } else {
            (i as u128 * m as u128 / RUNS as u128) as usize
        };
        start..start + len
    })
}

mod hex {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(x: &u64, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&format!("{x:#018x}"))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<u64, D::Error> {
        let s = String::deserialize(d)?;
        u64::from_str_radix(s.trim_start_matches("0x"), 16).map_err(serde::de::Error::custom)
    }
}

/// Fingerprint of a label store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LabelsFp {
    pub n_points: u64,
    pub n_labels: u64,
    pub n_entries: u64,
    #[serde(with = "hex")]
    pub dict_hash: u64,
    #[serde(with = "hex")]
    pub postings_hash: u64,
}

impl LabelsFp {
    pub fn of(store: &LabelStore) -> Self {
        let mut h = Hasher::default();
        for l in 0..store.n_labels() as u32 {
            h.push(store.posting_len(l) as u64);
        }
        let all = store.postings_flat();
        for s in samples(all.len()) {
            h.push(u64::from(all[s]));
        }
        Self {
            n_points: store.n_points() as u64,
            n_labels: store.n_labels() as u64,
            n_entries: store.n_entries() as u64,
            dict_hash: store.dict_hash(),
            postings_hash: h.finish(),
        }
    }
}

/// Fingerprint of a graph: its data part and the whole graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphFp {
    pub n_data: u64,
    pub n_nodes: u64,
    pub data_edges: u64,
    pub edges: u64,
    #[serde(with = "hex")]
    pub data_hash: u64,
    #[serde(with = "hex")]
    pub hash: u64,
}

impl GraphFp {
    pub fn of(g: &Graph) -> Self {
        Self::from_rows(
            g.n_data(),
            g.n_nodes(),
            |v| g.degree(v as u32) as u64,
            |v, out| {
                out.clear();
                out.extend_from_slice(g.neighbors(v as u32));
                Ok(())
            },
        )
        .expect("in-memory rows")
    }

    /// The fingerprint of a graph file: of `graph.flab` (or a directory holding one) from its
    /// degrees and the sampled rows only (unbuffered reads: a buffered reader would refill its
    /// buffer at every seek); of a DiskANN graph file by loading it.
    pub fn of_path(path: &Path) -> Result<Self> {
        let p = if path.is_dir() {
            path.join("graph.flab")
        } else {
            path.to_owned()
        };
        if container::peek_magic(&p)? != *graph::MAGIC {
            return Ok(Self::of(&diskann_io::open(&p, None)?));
        }
        let file = std::fs::File::open(&p).with_context(|| format!("opening {}", p.display()))?;
        let mut r = container::Reader::new(file, graph::MAGIC)?;
        let [n_data, n_nodes, r_max, ..] = *r.params();
        let (n_data, n_nodes, r_max) = (n_data as usize, n_nodes as usize, r_max as usize);
        let degrees: Vec<u32> = r.section(0, n_nodes)?;
        ensure!(
            degrees.iter().all(|&d| d as usize <= r_max),
            "{}: a degree exceeds the row size",
            p.display()
        );
        Self::from_rows(
            n_data,
            n_nodes,
            |v| u64::from(degrees[v]),
            |v, out| {
                out.resize(degrees[v] as usize, 0);
                r.read_range(1, n_nodes * r_max, v * r_max, out.as_mut_slice())
            },
        )
        .with_context(|| format!("fingerprinting {}", p.display()))
    }

    fn from_rows(
        n_data: usize,
        n_nodes: usize,
        degree: impl Fn(usize) -> u64,
        mut row: impl FnMut(usize, &mut Vec<u32>) -> Result<()>,
    ) -> Result<Self> {
        let data_edges: u64 = (0..n_data).map(&degree).sum();
        let edges = data_edges + (n_data..n_nodes).map(&degree).sum::<u64>();
        let mut buf = Vec::new();
        let mut hash = |n: usize| -> Result<u64> {
            let mut h = Hasher::default();
            for v in samples(n) {
                h.push(degree(v));
                row(v, &mut buf)?;
                for &u in &buf {
                    h.push(u64::from(u));
                }
            }
            Ok(h.finish())
        };
        let data_hash = hash(n_data)?;
        let all_hash = hash(n_nodes)?;
        Ok(Self {
            n_data: n_data as u64,
            n_nodes: n_nodes as u64,
            data_edges,
            edges,
            data_hash,
            hash: all_hash,
        })
    }

    /// Whether the data parts agree (`n_data`, `data_edges`, `data_hash`).
    pub fn same_data(&self, o: &Self) -> bool {
        (self.n_data, self.data_edges, self.data_hash) == (o.n_data, o.data_edges, o.data_hash)
    }
}

/// Shape of the base vectors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaseShape {
    pub n: u64,
    pub dim: u64,
    pub dtype: String,
}

impl BaseShape {
    pub fn of<T: Element>(base: &Matrix<T>) -> Self {
        Self {
            n: base.nrows() as u64,
            dim: base.ncols() as u64,
            dtype: T::NAME.to_owned(),
        }
    }
}

/// The provenance of a file: the fingerprints of what it was built from (each optional).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub labels: Option<LabelsFp>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph: Option<GraphFp>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<BaseShape>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_model: Option<String>,
}

/// Which part of a graph fingerprint a file is bound to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphPart {
    /// The data points' rows (shared by a graph and its Steiner graphs).
    Data,
    /// Every node.
    Full,
}

/// The fingerprints a loaded file is compared with (each optional).
#[derive(Debug, Clone, Copy, Default)]
pub struct Expect<'a> {
    pub labels: Option<&'a LabelsFp>,
    pub graph: Option<(&'a GraphFp, GraphPart)>,
    pub base: Option<&'a BaseShape>,
    pub cpu_model: Option<&'a str>,
}

impl Provenance {
    /// Of a label store alone.
    pub fn of_labels(store: &LabelStore) -> Self {
        Self {
            labels: Some(LabelsFp::of(store)),
            ..Self::default()
        }
    }

    /// The recorded fingerprints that differ from the expected ones.
    pub fn mismatches(&self, e: &Expect<'_>) -> Vec<String> {
        let mut out = Vec::new();
        if let (Some(r), Some(x)) = (&self.labels, e.labels) {
            if r != x {
                out.push(format!(
                    "label store: built from {}, loaded with {}",
                    json(r),
                    json(x)
                ));
            }
        }
        if let (Some(r), Some((x, part))) = (&self.graph, e.graph) {
            let same = match part {
                GraphPart::Data => r.same_data(x),
                GraphPart::Full => r == x,
            };
            if !same {
                out.push(format!(
                    "graph ({}): built from {}, loaded with {}",
                    match part {
                        GraphPart::Data => "data points' rows",
                        GraphPart::Full => "every node",
                    },
                    json(r),
                    json(x)
                ));
            }
        }
        if let (Some(r), Some(x)) = (&self.base, e.base) {
            if r != x {
                out.push(format!(
                    "base vectors: built from {}, loaded with {}",
                    json(r),
                    json(x)
                ));
            }
        }
        if let (Some(r), Some(x)) = (&self.cpu_model, e.cpu_model) {
            if r != x {
                out.push(format!("CPU model: measured on {r:?}, running on {x:?}"));
            }
        }
        out
    }

    /// The JSON value (as recorded under `provenance`).
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).expect("serializable")
    }

    /// The `provenance` member of a JSON document (`None` without one).
    pub fn from_doc(doc: &Value) -> Result<Option<Self>> {
        doc.get("provenance")
            .filter(|v| !v.is_null())
            .map(|v| serde_json::from_value(v.clone()).context("parsing the provenance"))
            .transpose()
    }

    /// The bytes of the provenance section of a container file.
    pub fn section(&self) -> Vec<u8> {
        let mut b = SECTION_MAGIC.to_vec();
        b.extend_from_slice(self.to_json().to_string().as_bytes());
        b
    }

    /// Parses a provenance section.
    pub fn from_section(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() >= 8 && &bytes[..8] == SECTION_MAGIC,
            "not a provenance section"
        );
        serde_json::from_slice(&bytes[8..]).context("parsing the provenance section")
    }

    /// The comment line of a label list.
    pub fn comment_line(&self) -> String {
        format!("# provenance {}\n", self.to_json())
    }

    /// The provenance of a label list (its `# provenance {…}` comment line, if any).
    pub fn from_text(text: &str) -> Result<Option<Self>> {
        text.lines()
            .find_map(|l| l.strip_prefix("# provenance "))
            .map(|j| serde_json::from_str(j).context("parsing the provenance line"))
            .transpose()
    }
}

fn json<T: Serialize>(x: &T) -> String {
    serde_json::to_string(x).unwrap_or_default()
}

/// The provenance section of a container read by `r` whose format has `n_own` sections
/// (`None`: written before provenance was recorded).
pub fn of_container<R: Read + Seek>(
    r: &mut container::Reader<R>,
    n_own: usize,
) -> Result<Option<Provenance>> {
    if r.n_sections() <= n_own {
        return Ok(None);
    }
    let bytes: Vec<u8> = r.section_bytes(n_own)?;
    Provenance::from_section(&bytes).map(Some)
}

/// The provenance of the container file `path` (magic `magic`, `n_own` sections of its own).
pub fn of_file(path: &Path, magic: &[u8; 8], n_own: usize) -> Result<Option<Provenance>> {
    let mut r = container::Reader::open(path, magic)?;
    of_container(&mut r, n_own).with_context(|| format!("reading {}", path.display()))
}

/// Checks the provenance `recorded` of `what` (a file) against `e`: a mismatch is an error
/// unless `allow` (then a warning); a file without provenance gets a warning. Returns the
/// warnings, also printed to standard error.
pub fn check(
    what: &str,
    recorded: Option<&Provenance>,
    e: &Expect<'_>,
    allow: bool,
) -> Result<Vec<String>> {
    let warnings = match recorded {
        None => vec![format!(
            "{what}: no provenance (written before it was recorded): not checked"
        )],
        Some(p) => {
            let m = p.mismatches(e);
            if !m.is_empty() && !allow {
                anyhow::bail!(
                    "{what} was built from other inputs ({}); rebuild it, or pass --allow-provenance-mismatch",
                    m.join("; ")
                );
            }
            m.into_iter()
                .map(|x| format!("{what}: provenance mismatch allowed: {x}"))
                .collect()
        }
    };
    for w in &warnings {
        eprintln!("warning: {w}");
    }
    Ok(warnings)
}

/// The checks of the files one command loads: each check once, the warnings collected.
#[derive(Debug, Default)]
pub struct Checker {
    pub allow: bool,
    seen: std::collections::HashSet<String>,
    pub warnings: Vec<String>,
}

impl Checker {
    /// `allow`: mismatches are warnings (`--allow-provenance-mismatch`).
    pub fn new(allow: bool) -> Self {
        Self {
            allow,
            ..Self::default()
        }
    }

    /// [`check`] of `what`, once per `what`.
    pub fn check(
        &mut self,
        what: &str,
        recorded: Option<&Provenance>,
        e: &Expect<'_>,
    ) -> Result<()> {
        if self.seen.insert(what.to_owned()) {
            let w = check(what, recorded, e, self.allow)?;
            self.warnings.extend(w);
        }
        Ok(())
    }

    /// Checks the container file `path` (magic `magic`, `n_own` sections of its own).
    pub fn file(
        &mut self,
        what: &str,
        path: &Path,
        magic: &[u8; 8],
        n_own: usize,
        e: &Expect<'_>,
    ) -> Result<()> {
        if self.seen.contains(what) {
            return Ok(());
        }
        let recorded = of_file(path, magic, n_own)?;
        self.check(what, recorded.as_ref(), e)
    }

    /// Checks a calibration document: its `provenance`, or for an older document the base
    /// shape it records (`n_points`, `dim`, `dtype`), which is compared too.
    pub fn calibration(&mut self, what: &str, doc: &Value, e: &Expect<'_>) -> Result<()> {
        if self.seen.contains(what) {
            return Ok(());
        }
        if let Some(p) = Provenance::from_doc(doc)? {
            return self.check(what, Some(&p), e);
        }
        let shape = match (
            doc.get("n_points").and_then(Value::as_u64),
            doc.get("dim").and_then(Value::as_u64),
            doc.get("dtype").and_then(Value::as_str),
        ) {
            (Some(n), Some(dim), Some(dtype)) => Some(BaseShape {
                n,
                dim,
                dtype: dtype.to_owned(),
            }),
            _ => None,
        };
        if let Some(base) = shape {
            let recorded = Provenance {
                base: Some(base),
                ..Provenance::default()
            };
            let shape_only = Expect {
                base: e.base,
                ..Expect::default()
            };
            let m = recorded.mismatches(&shape_only);
            if !m.is_empty() && !self.allow {
                anyhow::bail!(
                    "{what} was measured on other data ({}); recalibrate, or pass --allow-provenance-mismatch",
                    m.join("; ")
                );
            }
            for x in m {
                let w = format!("{what}: provenance mismatch allowed: {x}");
                eprintln!("warning: {w}");
                self.warnings.push(w);
            }
        }
        self.check(what, None, e)
    }

    /// The record of the checks (for `run.json`).
    pub fn to_json(&self) -> Value {
        serde_json::json!({"allow_mismatch": self.allow, "warnings": self.warnings})
    }
}

/// The CPU model (`model name` of `/proc/cpuinfo`).
pub fn cpu_model() -> Option<String> {
    let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    text.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        (k.trim() == "model name").then(|| v.trim().to_owned())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{graph::AuxNodes, labels::LabelStoreBuilder};

    fn store(rows: &[&[(&str, &str)]]) -> LabelStore {
        let mut b = LabelStoreBuilder::new();
        for r in rows {
            b.push_point(r.iter().copied()).unwrap();
        }
        b.finish().unwrap()
    }

    #[test]
    fn the_hash_is_the_documented_sum() {
        // H(5, 7) = mix(mix(5 + φ) + mix(7 + 2φ) ⊕ mix(2)).
        let mut h = Hasher::default();
        h.push(5);
        h.push(7);
        let acc =
            mix(5u64.wrapping_add(PHI)).wrapping_add(mix(7u64.wrapping_add(PHI.wrapping_mul(2))));
        assert_eq!(h.finish(), mix(acc ^ mix(2)));
        assert_eq!(mix(0), 0);
        assert_eq!(samples(3).collect::<Vec<_>>(), [0, 1, 2]);
        let s: Vec<usize> = samples(1 << 20).collect();
        assert_eq!(s.len(), RUNS * RUN);
        assert_eq!((s[0], s[63], s[64], s[65]), (0, 63, 1024, 1025));
        assert!(s.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn a_same_shape_label_change_changes_the_fingerprint() {
        // Points 0 and 1 swap their `a` labels: same sizes, same dictionary.
        let s1 = store(&[&[("a", "a=1")], &[("a", "a=2")], &[("a", "a=1")]]);
        let s2 = store(&[&[("a", "a=2")], &[("a", "a=1")], &[("a", "a=1")]]);
        let (f1, f2) = (LabelsFp::of(&s1), LabelsFp::of(&s2));
        assert_eq!(
            (f1.n_points, f1.n_labels, f1.n_entries, f1.dict_hash),
            (f2.n_points, f2.n_labels, f2.n_entries, f2.dict_hash)
        );
        assert_ne!(f1.postings_hash, f2.postings_hash);
        assert_eq!(f1, LabelsFp::of(&s1.clone()));
        let p = Provenance::of_labels(&s1);
        let e2 = Expect {
            labels: Some(&f2),
            ..Expect::default()
        };
        assert_eq!(p.mismatches(&e2).len(), 1);
        assert!(check("x", Some(&p), &e2, false).is_err());
        assert_eq!(check("x", Some(&p), &e2, true).unwrap().len(), 1);
        assert_eq!(check("x", None, &e2, false).unwrap().len(), 1);
        let e1 = Expect {
            labels: Some(&f1),
            ..Expect::default()
        };
        assert!(check("x", Some(&p), &e1, false).unwrap().is_empty());
    }

    #[test]
    fn records_round_trip_in_every_form() {
        let s = store(&[&[("a", "a=1"), ("b", "b=1")], &[("a", "a=2")]]);
        let mut g = Graph::new(2, 2, AuxNodes::untyped()).unwrap();
        g.set_neighbors(0, &[1]).unwrap();
        g.set_neighbors(1, &[0]).unwrap();
        let p = Provenance {
            labels: Some(LabelsFp::of(&s)),
            graph: Some(GraphFp::of(&g)),
            base: Some(BaseShape {
                n: 2,
                dim: 3,
                dtype: "u8".into(),
            }),
            cpu_model: Some("cpu".into()),
        };
        assert_eq!(Provenance::from_section(&p.section()).unwrap(), p);
        let doc = serde_json::json!({ "provenance": p.to_json() });
        assert_eq!(Provenance::from_doc(&doc).unwrap(), Some(p.clone()));
        assert_eq!(Provenance::from_doc(&serde_json::json!({})).unwrap(), None);
        let text = format!("{}0\n1\n", p.comment_line());
        assert_eq!(Provenance::from_text(&text).unwrap(), Some(p.clone()));
        assert!(p.to_json()["labels"]["dict_hash"]
            .as_str()
            .unwrap()
            .starts_with("0x"));
    }

    #[test]
    fn a_steiner_graph_keeps_the_data_part() {
        let mut g = Graph::new(3, 2, AuxNodes::untyped()).unwrap();
        g.set_neighbors(0, &[1]).unwrap();
        g.set_neighbors(1, &[2, 0]).unwrap();
        g.set_neighbors(2, &[0]).unwrap();
        let aux = AuxNodes::new(
            vec![crate::graph::AuxKind::Router],
            &Matrix::<u8>::from_slice(&[1, 2], 1, 2),
        )
        .unwrap();
        let mut h = Graph::new(3, 2, aux).unwrap();
        for v in 0..3 {
            h.set_neighbors(v, g.neighbors(v)).unwrap();
        }
        h.set_neighbors(3, &[0, 2]).unwrap();
        let (a, b) = (GraphFp::of(&g), GraphFp::of(&h));
        assert!(a.same_data(&b) && a != b);
        assert_eq!((b.data_edges, b.edges), (4, 6));
        let mut k = g.clone();
        k.set_neighbors(2, &[1]).unwrap();
        assert!(!GraphFp::of(&k).same_data(&a));
    }
}
