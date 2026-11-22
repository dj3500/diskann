/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Start-node tables of a graph (format `starts.flab`).
//!
//! A sidecar lists, for one graph, the nodes that search strategies may start from:
//! - **global starts**: the graph's own entry points (e.g. a medoid or frozen points);
//! - **per-label starts**: for each label of one label store, a list of nodes (e.g. the
//!   per-label medoids of filtered Vamana builds). The table records the fingerprint of the
//!   store's dictionary;
//! - a **sample**: an ordered list of data points, each paired with the node to start from when
//!   the entry's labels match (the entry itself, unless the source paired the entries with
//!   other ids). A search may start from the first matching entries of the sample.
//!
//! Readers for the C++ build outputs: `<prefix>_labels_to_medoids.txt` (lines `label, point`
//! with integer labels), `<prefix>_labels_map.txt` (lines `name<TAB>label`) and the sample id
//! files `<prefix>_sample_ids.bin` / `<prefix>_sample_labels.txt_map.bin` (`u32` count,
//! `u32` 1, then the ids).

use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom, Write},
    path::Path,
};

use anyhow::{bail, ensure, Context, Result};
use hashbrown::HashMap;

use crate::{
    container, diskann_io::read_bin_header, graph::Graph, labels::LabelStore, permute::Permutation,
    provenance::Provenance, text,
};

/// File magic of `starts.flab`.
pub const MAGIC: &[u8; 8] = b"FLABSTR\0";
/// Version of the `starts.flab` format.
pub const VERSION: u32 = 1;
/// Sections of the `starts.flab` format (a provenance section may follow).
pub const N_SECTIONS: usize = 5;

/// Start nodes of one graph (see the module documentation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Starts {
    n_nodes: u32,
    n_data: u32,
    dict_hash: u64,
    global: Vec<u32>,
    label_offsets: Vec<u64>,
    label_starts: Vec<u32>,
    sample: Vec<u32>,
    sample_start: Vec<u32>,
}

impl Starts {
    /// The global starts of `graph`, without per-label starts or sample.
    pub fn new(graph: &Graph) -> Self {
        Self {
            n_nodes: graph.n_nodes() as u32,
            n_data: graph.n_data() as u32,
            dict_hash: 0,
            global: graph.starts().to_vec(),
            label_offsets: vec![0],
            label_starts: Vec::new(),
            sample: Vec::new(),
            sample_start: Vec::new(),
        }
    }

    /// Sets the per-label starts over the labels of `store` from `(label, node)` pairs; the
    /// nodes of each label keep their order in `pairs`.
    pub fn set_label_starts(&mut self, store: &LabelStore, pairs: &[(u32, u32)]) -> Result<()> {
        let nl = store.n_labels();
        for &(l, v) in pairs {
            ensure!((l as usize) < nl, "label {l} out of range ({nl} labels)");
            ensure!(
                v < self.n_nodes,
                "start {v} of label {l} out of range ({} nodes)",
                self.n_nodes
            );
        }
        let mut offsets = vec![0u64; nl + 1];
        for &(l, _) in pairs {
            offsets[l as usize + 1] += 1;
        }
        for l in 0..nl {
            offsets[l + 1] += offsets[l];
        }
        let mut cursor = offsets[..nl].to_vec();
        let mut starts = vec![0u32; pairs.len()];
        for &(l, v) in pairs {
            let at = &mut cursor[l as usize];
            starts[*at as usize] = v;
            *at += 1;
        }
        self.label_offsets = offsets;
        self.label_starts = starts;
        self.dict_hash = store.dict_hash();
        Ok(())
    }

    /// Sets the sample: ordered data points and, optionally, the start node of each entry
    /// (by default the entry itself).
    pub fn set_sample(&mut self, sample: Vec<u32>, start: Option<Vec<u32>>) -> Result<()> {
        if let Some(&p) = sample.iter().find(|&&p| p >= self.n_data) {
            bail!(
                "sample entry {p} is not a data point ({} points)",
                self.n_data
            );
        }
        let start = start.unwrap_or_else(|| sample.clone());
        ensure!(
            start.len() == sample.len(),
            "{} sample entries but {} start nodes",
            sample.len(),
            start.len()
        );
        if let Some(&v) = start.iter().find(|&&v| v >= self.n_nodes) {
            bail!("sample start {v} out of range ({} nodes)", self.n_nodes);
        }
        self.sample = sample;
        self.sample_start = start;
        Ok(())
    }

    /// Number of nodes of the graph.
    pub fn n_nodes(&self) -> usize {
        self.n_nodes as usize
    }

    /// Number of data points of the graph.
    pub fn n_data(&self) -> usize {
        self.n_data as usize
    }

    /// Checks that the table belongs to a graph with the node space of `graph`.
    pub fn check_graph(&self, graph: &Graph) -> Result<()> {
        ensure!(
            self.n_nodes() == graph.n_nodes() && self.n_data() == graph.n_data(),
            "the starts refer to {} nodes ({} data points), the graph has {} ({})",
            self.n_nodes,
            self.n_data,
            graph.n_nodes(),
            graph.n_data()
        );
        Ok(())
    }

    /// Global starts in priority order.
    pub fn global(&self) -> &[u32] {
        &self.global
    }

    /// Size of the per-label table (0 without per-label starts).
    pub fn n_labels(&self) -> usize {
        self.label_offsets.len() - 1
    }

    /// Total number of per-label start entries.
    pub fn n_label_entries(&self) -> usize {
        self.label_starts.len()
    }

    /// Start nodes of label `l`.
    #[inline]
    pub fn label_starts(&self, l: u32) -> &[u32] {
        let l = l as usize;
        &self.label_starts[self.label_offsets[l] as usize..self.label_offsets[l + 1] as usize]
    }

    /// Fingerprint of the dictionary of the per-label table (0 without one).
    pub fn dict_hash(&self) -> u64 {
        self.dict_hash
    }

    /// Checks that the per-label table refers to the dictionary of `store`.
    pub fn check_dictionary(&self, store: &LabelStore) -> Result<()> {
        ensure!(
            self.dict_hash == store.dict_hash() && self.n_labels() == store.n_labels(),
            "the per-label starts refer to a different label dictionary"
        );
        Ok(())
    }

    /// Per-label start entries that are data points without their label (none for per-label
    /// medoids).
    pub fn label_mismatches(&self, store: &LabelStore) -> Result<usize> {
        self.check_dictionary(store)?;
        let mut bad = 0;
        for l in 0..self.n_labels() as u32 {
            bad += self
                .label_starts(l)
                .iter()
                .filter(|&&v| v < self.n_data && store.row(v).binary_search(&l).is_err())
                .count();
        }
        Ok(bad)
    }

    /// Sample entries in order.
    pub fn sample(&self) -> &[u32] {
        &self.sample
    }

    /// Start node of each sample entry.
    pub fn sample_start(&self) -> &[u32] {
        &self.sample_start
    }

    /// The table under the new ids of `p` (every list keeps its order; auxiliary nodes keep
    /// their ids).
    pub fn permuted(&self, p: &Permutation) -> Result<Self> {
        ensure!(
            p.len() == self.n_data as usize,
            "the permutation covers {} points, the table {} data points",
            p.len(),
            self.n_data
        );
        let map = |v: &Vec<u32>| v.iter().map(|&x| p.new_id(x)).collect::<Vec<u32>>();
        Ok(Self {
            n_nodes: self.n_nodes,
            n_data: self.n_data,
            dict_hash: self.dict_hash,
            global: map(&self.global),
            label_offsets: self.label_offsets.clone(),
            label_starts: map(&self.label_starts),
            sample: map(&self.sample),
            sample_start: map(&self.sample_start),
        })
    }

    /// Writes `starts.flab` (see the crate `README.md` for the layout).
    pub fn write(&self, path: &Path) -> Result<()> {
        self.write_with(path, None)
    }

    /// [`Self::write`] with a provenance section ([`crate::provenance`]) when given.
    pub fn write_with(&self, path: &Path, prov: Option<&Provenance>) -> Result<()> {
        self.write_to_with(container::create(path)?, prov)
            .with_context(|| format!("writing {}", path.display()))
    }

    /// Writes the `starts.flab` format to any writer.
    pub fn write_to<W: Write>(&self, w: W) -> Result<()> {
        self.write_to_with(w, None)
    }

    /// [`Self::write_to`] with a provenance section when given.
    pub fn write_to_with<W: Write>(&self, w: W, prov: Option<&Provenance>) -> Result<()> {
        let params = [
            u64::from(self.n_nodes),
            u64::from(self.n_data),
            self.n_labels() as u64,
            self.dict_hash,
            self.global.len() as u64,
            self.label_starts.len() as u64,
            self.sample.len() as u64,
            0,
        ];
        let prov = prov.map(Provenance::section);
        let mut sections: Vec<&[u8]> = vec![
            bytemuck::cast_slice(&self.global),
            bytemuck::cast_slice(&self.label_offsets),
            bytemuck::cast_slice(&self.label_starts),
            bytemuck::cast_slice(&self.sample),
            bytemuck::cast_slice(&self.sample_start),
        ];
        sections.extend(prov.as_deref());
        container::write(w, MAGIC, VERSION, &params, &sections)
    }

    /// Reads and validates `starts.flab`.
    pub fn read(path: &Path) -> Result<Self> {
        container::Reader::open(path, MAGIC)
            .and_then(Self::load)
            .with_context(|| format!("loading starts from {}", path.display()))
    }

    /// Reads and validates the `starts.flab` format from any reader.
    pub fn read_from<R: Read + Seek>(r: R) -> Result<Self> {
        Self::load(container::Reader::new(r, MAGIC)?)
    }

    fn load<R: Read + Seek>(mut r: container::Reader<R>) -> Result<Self> {
        ensure!(
            r.version() == VERSION,
            "unsupported version {}",
            r.version()
        );
        let [n_nodes, n_data, n_labels, dict_hash, n_global, n_entries, n_sample, _] = *r.params();
        ensure!(
            n_data <= n_nodes && n_nodes < u64::from(u32::MAX),
            "inconsistent node counts {n_data} / {n_nodes}"
        );
        let starts = Self {
            n_nodes: n_nodes as u32,
            n_data: n_data as u32,
            dict_hash,
            global: r.section(0, n_global as usize)?,
            label_offsets: r.section(1, n_labels as usize + 1)?,
            label_starts: r.section(2, n_entries as usize)?,
            sample: r.section(3, n_sample as usize)?,
            sample_start: r.section(4, n_sample as usize)?,
        };
        starts.validate()?;
        Ok(starts)
    }

    fn validate(&self) -> Result<()> {
        let offsets = &self.label_offsets;
        ensure!(
            offsets.first() == Some(&0)
                && offsets.last() == Some(&(self.label_starts.len() as u64))
                && offsets.windows(2).all(|w| w[0] <= w[1]),
            "inconsistent per-label offsets"
        );
        for (what, ids, bound) in [
            ("global start", &self.global, self.n_nodes),
            ("per-label start", &self.label_starts, self.n_nodes),
            ("sample entry", &self.sample, self.n_data),
            ("sample start", &self.sample_start, self.n_nodes),
        ] {
            if let Some(&v) = ids.iter().find(|&&v| v >= bound) {
                bail!("{what} {v} out of range ({bound})");
            }
        }
        Ok(())
    }
}

/// Parses a line `label, point` of a C++ `_labels_to_medoids.txt` file.
pub fn parse_medoid_line(line: &str) -> Result<(u32, u32)> {
    let (label, point) = line
        .split_once(',')
        .with_context(|| format!("expected `label, point`, got {line:?}"))?;
    Ok((
        label.trim().parse().context("bad label")?,
        point.trim().parse().context("bad point id")?,
    ))
}

/// Parses a line `name<TAB>label` of a C++ `_labels_map.txt` file.
pub fn parse_label_map_line(line: &str) -> Result<(&str, u32)> {
    let (name, label) = line
        .rsplit_once('\t')
        .with_context(|| format!("expected `name<TAB>label`, got {line:?}"))?;
    Ok((name, label.trim().parse().context("bad label")?))
}

/// Reads a C++ `_labels_to_medoids.txt` file: `(label, point)` pairs in file order.
pub fn read_cxx_medoids(path: &Path) -> Result<Vec<(u32, u32)>> {
    let mut pairs = Vec::new();
    text::for_each_line(path, |_, line| {
        if !line.trim().is_empty() {
            pairs.push(parse_medoid_line(line)?);
        }
        Ok(())
    })?;
    Ok(pairs)
}

/// Reads a C++ `_labels_map.txt` file: the name of every integer label.
pub fn read_cxx_label_map(path: &Path) -> Result<HashMap<u32, String>> {
    let mut names = HashMap::new();
    text::for_each_line(path, |_, line| {
        if !line.trim().is_empty() {
            let (name, label) = parse_label_map_line(line)?;
            ensure!(
                names.insert(label, name.to_owned()).is_none(),
                "label {label} appears twice"
            );
        }
        Ok(())
    })?;
    Ok(names)
}

/// C++ per-label starts translated to the label ids of a store (see [`join_medoids`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinedMedoids {
    /// `(label id, node)` pairs in input order.
    pub pairs: Vec<(u32, u32)>,
    /// Names that the store does not contain; their pairs are dropped.
    pub absent: Vec<String>,
}

/// Translates C++ `(integer label, point)` pairs to `(label id, point)` pairs of `store` via
/// the label names.
pub fn join_medoids(
    medoids: &[(u32, u32)],
    names: &HashMap<u32, String>,
    store: &LabelStore,
) -> Result<JoinedMedoids> {
    let index = store.name_index();
    let mut joined = JoinedMedoids {
        pairs: Vec::with_capacity(medoids.len()),
        absent: Vec::new(),
    };
    for &(label, point) in medoids {
        let name = names
            .get(&label)
            .with_context(|| format!("label {label} is missing from the label map"))?;
        match index.get(name.as_str()) {
            Some(&l) => joined.pairs.push((l, point)),
            None => joined.absent.push(name.clone()),
        }
    }
    Ok(joined)
}

/// Reads a `u32` id list in the `.bin` layout (`u32` count, `u32` 1, then the ids).
pub fn read_ids<R: Read + Seek>(mut r: R) -> Result<Vec<u32>> {
    let len = r.seek(SeekFrom::End(0))?;
    r.seek(SeekFrom::Start(0))?;
    let (n, dim) = read_bin_header(&mut r)?;
    ensure!(dim == 1, "an id list has dimension 1, not {dim}");
    ensure!(
        len == 8 + 4 * n as u64,
        "{len} bytes for {n} ids (expected {})",
        8 + 4 * n as u64
    );
    let mut ids = vec![0u32; n];
    r.read_exact(bytemuck::cast_slice_mut(&mut ids))?;
    Ok(ids)
}

/// [`read_ids`] from a file.
pub fn read_id_file(path: &Path) -> Result<Vec<u32>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    read_ids(BufReader::new(file)).with_context(|| format!("reading {}", path.display()))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::{
        graph::{AuxKind, AuxNodes},
        labels::LabelStoreBuilder,
        vectors::Matrix,
    };

    fn graph() -> Graph {
        let aux = AuxNodes::new(vec![AuxKind::Frozen], &Matrix::<u8>::zeros(1, 2)).unwrap();
        let mut g = Graph::new(4, 2, aux).unwrap();
        g.set_starts(vec![4]).unwrap();
        g
    }

    fn store() -> LabelStore {
        let mut b = LabelStoreBuilder::new();
        for row in [&["c=a"][..], &["c=a", "c=b"], &["c=b"], &[]] {
            b.push_point(row.iter().map(|&l| ("c", l))).unwrap();
        }
        b.finish().unwrap()
    }

    #[test]
    fn label_starts_sample_and_file_roundtrip() {
        let (g, store) = (graph(), store());
        let id = |name: &str| store.name_index()[name];
        let mut s = Starts::new(&g);
        assert_eq!((s.global(), s.n_labels()), (&[4][..], 0));
        s.set_label_starts(&store, &[(id("c=b"), 2), (id("c=a"), 0), (id("c=b"), 1)])
            .unwrap();
        s.set_sample(vec![0, 2, 3], Some(vec![1, 2, 3])).unwrap();
        assert_eq!(s.label_starts(id("c=a")), &[0]);
        assert_eq!(s.label_starts(id("c=b")), &[2, 1]);
        assert_eq!((s.n_labels(), s.n_label_entries()), (2, 3));
        assert_eq!(s.label_mismatches(&store).unwrap(), 0);
        assert_eq!(
            (s.sample(), s.sample_start()),
            (&[0, 2, 3][..], &[1, 2, 3][..])
        );

        let mut buf = Vec::new();
        s.write_to(&mut buf).unwrap();
        let t = Starts::read_from(Cursor::new(&buf)).unwrap();
        assert_eq!(t, s);
        t.check_graph(&g).unwrap();
        t.check_dictionary(&store).unwrap();

        assert!(s.set_label_starts(&store, &[(id("c=a"), 5)]).is_err());
        assert!(s.set_label_starts(&store, &[(7, 0)]).is_err());
        assert!(s.set_sample(vec![4], None).is_err());
        assert!(s.set_sample(vec![0, 1], Some(vec![0])).is_err());
        s.set_label_starts(&store, &[(id("c=a"), 2), (id("c=b"), 4)])
            .unwrap();
        assert_eq!(s.label_mismatches(&store).unwrap(), 1);
        s.set_sample(vec![1], None).unwrap();
        assert_eq!(s.sample_start(), &[1]);
    }

    #[test]
    fn cxx_medoids_join_by_name() {
        let store = store();
        assert_eq!(parse_medoid_line("12, 947631").unwrap(), (12, 947631));
        assert!(parse_medoid_line("12 947631").is_err());
        assert_eq!(
            parse_label_map_line("CP=1+2+x\t7").unwrap(),
            ("CP=1+2+x", 7)
        );
        let names: HashMap<u32, String> = [(1, "c=b"), (2, "c=a"), (3, "c=z")]
            .into_iter()
            .map(|(l, n)| (l, n.to_owned()))
            .collect();
        let joined = join_medoids(&[(2, 0), (3, 1), (1, 2)], &names, &store).unwrap();
        let id = |name: &str| store.name_index()[name];
        assert_eq!(joined.pairs, vec![(id("c=a"), 0), (id("c=b"), 2)]);
        assert_eq!(joined.absent, vec!["c=z".to_owned()]);
        assert!(join_medoids(&[(9, 0)], &names, &store).is_err());
    }

    #[test]
    fn id_lists() {
        let file = |n: u32, dim: u32, ids: &[u32]| {
            let mut f = n.to_le_bytes().to_vec();
            f.extend_from_slice(&dim.to_le_bytes());
            f.extend_from_slice(bytemuck::cast_slice(ids));
            Cursor::new(f)
        };
        assert_eq!(read_ids(file(3, 1, &[2, 5, 9])).unwrap(), vec![2, 5, 9]);
        assert!(read_ids(file(3, 2, &[2, 5, 9])).is_err());
        assert!(read_ids(file(4, 1, &[2, 5, 9])).is_err());
    }
}
