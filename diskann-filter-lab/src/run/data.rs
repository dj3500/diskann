/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Inputs of a run: the dataset (base vectors, labels, query sets and their predicates, label
//! views), the query rows (a panel or a whole query set), graphs and their start tables.

use std::{
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{bail, ensure, Context, Result};

use crate::{
    container, diskann_io,
    graph::{self, Graph},
    kernel::entry::SampleIndex,
    labels::{self, LabelStore},
    predicate::PredicateSet,
    starts::Starts,
    vectors::{DType, Element, Matrix},
};

/// FNV-1a 64 of bytes.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// SHA-256 of a file, as lowercase hex.
pub fn sha256_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(sha256_hex(&bytes))
}

/// SHA-256 (FIPS 180-4) as lowercase hex.
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bits.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for (a, b) in h.iter_mut().zip(v) {
            *a = a.wrapping_add(b);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

/// Finds `<dir>/<stem>.u8bin` or `<dir>/<stem>.fbin`.
fn vector_file(dir: &Path, stem: &str) -> Result<(PathBuf, DType)> {
    for (ext, dtype) in [("u8bin", DType::U8), ("fbin", DType::F32)] {
        let p = dir.join(format!("{stem}.{ext}"));
        if p.exists() {
            return Ok((p, dtype));
        }
    }
    bail!("no {stem}.u8bin or {stem}.fbin in {}", dir.display())
}

/// The base vectors of a dataset directory (`base.u8bin` or `base.fbin`).
pub fn base_file(dir: &Path) -> Result<PathBuf> {
    Ok(vector_file(dir, "base")?.0)
}

/// Reads the dictionary fingerprint, label count and posting offsets of a `labels.flab`
/// without its rows and postings.
pub fn read_view_postings(path: &Path) -> Result<(u64, usize, Vec<u64>)> {
    let mut r = container::Reader::open(path, labels::MAGIC)?;
    let [_, n_labels, _, _, dict_hash, ..] = *r.params();
    let offsets = r.section::<u64>(8, n_labels as usize + 1)?;
    Ok((dict_hash, n_labels as usize, offsets))
}

/// One query set: vectors, predicates and, for a label view, the view's predicates.
#[derive(Debug)]
pub struct QuerySet<T> {
    pub name: String,
    pub vectors: Matrix<T>,
    pub preds: PredicateSet,
    pub views: BTreeMap<String, PredicateSet>,
}

/// The dataset of a run.
#[derive(Debug)]
pub struct Dataset<T> {
    pub dir: PathBuf,
    pub base: Matrix<T>,
    pub base_file: PathBuf,
    pub labels: LabelStore,
    pub qsets: BTreeMap<String, QuerySet<T>>,
    /// Per view: dictionary fingerprint, label count and posting offsets.
    pub views: BTreeMap<String, (u64, usize, Vec<u64>)>,
}

/// Element type of a dataset's base vectors.
pub fn dataset_dtype(dir: &Path) -> Result<DType> {
    Ok(vector_file(dir, "base")?.1)
}

impl<T: Element> Dataset<T> {
    /// Loads the base, the labels, the query sets `qsets` and the label views `views`.
    pub fn load(dir: &Path, qsets: &[String], views: &[String]) -> Result<Self> {
        let (base_file, dtype) = vector_file(dir, "base")?;
        ensure!(dtype == T::DTYPE, "the base is {}", dtype.name());
        let base = Matrix::<T>::read_bin(&base_file)?;
        let labels = LabelStore::read(&dir.join("labels.flab"))?;
        ensure!(
            labels.n_points() == base.nrows(),
            "{} label rows for {} base vectors",
            labels.n_points(),
            base.nrows()
        );
        let mut view_meta = BTreeMap::new();
        for v in views {
            let p = dir.join(format!("labels.{v}.flab"));
            view_meta.insert(v.clone(), read_view_postings(&p)?);
        }
        let mut sets = BTreeMap::new();
        for q in qsets {
            let (vpath, vtype) = vector_file(dir, &format!("q{q}"))?;
            ensure!(vtype == T::DTYPE, "{} is {}", vpath.display(), vtype.name());
            let vectors = Matrix::<T>::read_bin(&vpath)?;
            ensure!(
                vectors.ncols() == base.ncols(),
                "{}: dimension {} vs {}",
                vpath.display(),
                vectors.ncols(),
                base.ncols()
            );
            let preds = PredicateSet::read(&dir.join(format!("q{q}.fpred")))?;
            preds.check_dictionary(&labels)?;
            ensure!(
                preds.len() == vectors.nrows(),
                "q{q}: {} predicates for {} vectors",
                preds.len(),
                vectors.nrows()
            );
            let mut vp = BTreeMap::new();
            for v in views {
                let p = dir.join(format!("q{q}.{v}.fpred"));
                let set = PredicateSet::read(&p)?;
                let (hash, n_labels, _) = &view_meta[v];
                ensure!(
                    set.dict_hash() == *hash && set.len() == vectors.nrows(),
                    "{} does not belong to labels.{v}.flab or to q{q} ({n_labels} labels)",
                    p.display()
                );
                vp.insert(v.clone(), set);
            }
            sets.insert(
                q.clone(),
                QuerySet {
                    name: q.clone(),
                    vectors,
                    preds,
                    views: vp,
                },
            );
        }
        Ok(Self {
            dir: dir.to_owned(),
            base,
            base_file,
            labels,
            qsets: sets,
            views: view_meta,
        })
    }
}

/// The query rows of a run.
#[derive(Debug, Clone)]
pub struct Rows {
    /// `(query set, row in that set)` in run order.
    pub rows: Vec<(String, usize)>,
    /// Header and lines of the panel rows kept (for the run's own panel file).
    pub panel_header: Option<String>,
    pub panel_lines: Vec<String>,
    /// The panel file and its SHA-256.
    pub source: Option<(PathBuf, String)>,
}

impl Rows {
    /// Rows of a panel TSV (columns `qset`, `qidx`, optional `split`), filtered.
    pub fn from_panel(
        path: &Path,
        split: Option<&str>,
        qsets: Option<&[String]>,
        limit: Option<usize>,
    ) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut rows = Self::parse_panel(&text, split, qsets, limit)?;
        rows.source = Some((path.to_owned(), sha256_hex(text.as_bytes())));
        Ok(rows)
    }

    /// [`Self::from_panel`] on the panel's text.
    pub fn parse_panel(
        text: &str,
        split: Option<&str>,
        qsets: Option<&[String]>,
        limit: Option<usize>,
    ) -> Result<Self> {
        let mut lines = text.lines();
        let header = lines.next().context("empty panel")?.to_owned();
        let cols: Vec<&str> = header.split('\t').collect();
        let col = |name: &str| cols.iter().position(|&c| c == name);
        let (qc, ic) = (
            col("qset").context("panel without a qset column")?,
            col("qidx").context("panel without a qidx column")?,
        );
        let sc = col("split");
        ensure!(
            split.is_none() || sc.is_some(),
            "the panel has no split column"
        );
        let mut rows = Vec::new();
        let mut kept = Vec::new();
        for line in lines.filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split('\t').collect();
            ensure!(f.len() == cols.len(), "ragged panel line {line:?}");
            if let (Some(s), Some(c)) = (split, sc) {
                if f[c] != s {
                    continue;
                }
            }
            if let Some(q) = qsets {
                if !q.iter().any(|x| x == f[qc]) {
                    continue;
                }
            }
            if limit.is_some_and(|n| rows.len() >= n) {
                break;
            }
            rows.push((
                f[qc].to_owned(),
                f[ic]
                    .parse()
                    .with_context(|| format!("bad qidx in {line:?}"))?,
            ));
            kept.push(line.to_owned());
        }
        ensure!(!rows.is_empty(), "no panel rows left after filtering");
        Ok(Self {
            rows,
            panel_header: Some(header),
            panel_lines: kept,
            source: None,
        })
    }

    /// Every row of a query set (the first `limit`).
    pub fn from_qset(qset: &str, n: usize, limit: Option<usize>) -> Self {
        let n = limit.map_or(n, |l| l.min(n));
        Self {
            rows: (0..n).map(|i| (qset.to_owned(), i)).collect(),
            panel_header: None,
            panel_lines: Vec::new(),
            source: None,
        }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Distinct query sets in first-appearance order.
    pub fn qsets(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for (q, _) in &self.rows {
            if !out.contains(q) {
                out.push(q.clone());
            }
        }
        out
    }
}

/// A loaded graph with its start table (the graph is shared by the tables of one graph file).
#[derive(Debug)]
pub struct LoadedGraph {
    pub path: PathBuf,
    pub graph: Arc<Graph>,
    pub starts_path: Option<PathBuf>,
    pub starts: Option<Starts>,
    /// Global starts used by searches.
    pub global: Vec<u32>,
    pub sample: Option<SampleIndex>,
}

/// Loads a graph: a directory with `graph.flab` (and `starts.flab`), a `graph.flab`, or a
/// DiskANN index prefix (vectors in `<prefix>.data`).
pub fn load_graph(path: &Path, sidecar: Option<&Path>, dtype: DType) -> Result<LoadedGraph> {
    load_graph_shared(path, sidecar, dtype, &mut HashMap::new())
}

/// [`load_graph`], reading each graph file once: `shared` holds the graphs loaded so far by path,
/// so that start tables of one graph (several `starts_sidecar`s) share its memory.
pub fn load_graph_shared(
    path: &Path,
    sidecar: Option<&Path>,
    dtype: DType,
    shared: &mut HashMap<PathBuf, Arc<Graph>>,
) -> Result<LoadedGraph> {
    let (graph_path, default_sidecar) = if path.is_dir() {
        let s = path.join("starts.flab");
        (path.join("graph.flab"), s.exists().then_some(s))
    } else {
        let s = path.with_file_name("starts.flab");
        let flab = container::peek_magic(path).ok().as_ref() == Some(graph::MAGIC);
        (path.to_owned(), (flab && s.exists()).then_some(s))
    };
    let graph = match shared.get(&graph_path) {
        Some(g) => Arc::clone(g),
        None => {
            let g = if graph_path.exists()
                && container::peek_magic(&graph_path).ok().as_ref() == Some(graph::MAGIC)
            {
                Graph::read(&graph_path)?
            } else {
                let mut data = graph_path.clone().into_os_string();
                data.push(".data");
                let data = PathBuf::from(data);
                diskann_io::open(&graph_path, Some((&data, dtype)))?
            };
            let g = Arc::new(g);
            shared.insert(graph_path.clone(), Arc::clone(&g));
            g
        }
    };
    let starts_path = sidecar.map(Path::to_owned).or(default_sidecar);
    let starts = starts_path.as_deref().map(Starts::read).transpose()?;
    let mut global = graph.starts().to_vec();
    if let Some(s) = &starts {
        if s.n_nodes() == graph.n_nodes() {
            s.check_graph(&graph)?;
            if !s.global().is_empty() {
                global = s.global().to_vec();
            }
        } else {
            // A table of another graph over the same data points: only data-point starts are
            // meaningful here.
            ensure!(
                s.n_data() == graph.n_data(),
                "the start table refers to {} data points, the graph has {}",
                s.n_data(),
                graph.n_data()
            );
            let n = graph.n_data() as u32;
            for l in 0..s.n_labels() as u32 {
                ensure!(
                    s.label_starts(l).iter().all(|&v| v < n),
                    "per-label starts of another graph must be data points"
                );
            }
            ensure!(
                s.sample_start().iter().all(|&v| v < n),
                "sample starts of another graph must be data points"
            );
        }
    }
    ensure!(!global.is_empty(), "{}: no global start", path.display());
    Ok(LoadedGraph {
        path: path.to_owned(),
        graph,
        starts_path,
        starts,
        global,
        sample: None,
    })
}

/// Reads one byte of every page of `bytes` (pre-faults memory before timing).
pub fn touch<T: bytemuck::Pod>(data: &[T]) -> u64 {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    let mut acc = 0u64;
    for chunk in bytes.chunks(4096) {
        acc = acc.wrapping_add(u64::from(std::hint::black_box(chunk[0])));
    }
    acc
}

/// Pre-faults the arrays of a dataset and of graphs.
pub fn prefault<'g, T: Element>(
    ds: &Dataset<T>,
    graphs: impl IntoIterator<Item = &'g LoadedGraph>,
) -> u64 {
    let mut acc = touch(ds.base.as_slice());
    let (off, rows) = ds.labels.forward();
    acc = acc.wrapping_add(touch(off)).wrapping_add(touch(rows));
    for l in (0..ds.labels.n_labels() as u32).step_by(64) {
        acc = acc.wrapping_add(touch(ds.labels.postings(l)));
    }
    for q in ds.qsets.values() {
        acc = acc.wrapping_add(touch(q.vectors.as_slice()));
    }
    for g in graphs {
        let r = g.graph.r_max().max(1);
        let step = (1024 / r).max(1);
        for id in (0..g.graph.n_nodes() as u32).step_by(step) {
            acc = acc.wrapping_add(touch(g.graph.row(id)));
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let long = vec![b'a'; 1000];
        assert_eq!(
            sha256_hex(&long),
            "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"
        );
    }

    #[test]
    fn panel_filters() {
        let text = "qset\tqidx\tsplit\n10K\t3\tdiscovery\n40K\t7\tdiscovery\n2Khard\t1\taudit\n10K\t9\tdiscovery\n";
        let r = Rows::parse_panel(text, Some("discovery"), None, None).unwrap();
        assert_eq!(r.rows.len(), 3);
        let keep = ["10K".to_owned(), "2Khard".to_owned()];
        let r = Rows::parse_panel(text, None, Some(&keep), Some(2)).unwrap();
        assert_eq!(
            r.rows,
            vec![("10K".to_owned(), 3), ("2Khard".to_owned(), 1)]
        );
        assert_eq!(r.qsets(), vec!["10K".to_owned(), "2Khard".to_owned()]);
        assert_eq!(r.panel_lines, ["10K\t3\tdiscovery", "2Khard\t1\taudit"]);
        assert!(Rows::parse_panel(text, Some("none"), None, None).is_err());
        assert!(Rows::parse_panel("qset\tqidx\n10K\n", None, None, None).is_err());
    }
}
