/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Per-label sub-graphs ("islands", sidecar `islands.flab`): a small Vamana graph over the
//! members of each selected label, for the search route `island` of the cost crossover (a
//! search inside one label's island, the rest of the predicate checked on the nodes it
//! evaluates; [`crate::kernel::search`]).
//!
//! Selection, from the label sizes and the unit costs of the cost model only (categories are
//! not read), at `K` answers on a dataset of `n` points. For a label of `s` members:
//! - `exact(s) = s · (dist(s/n) + pred(s/n))`: the exact path over the label (a query whose
//!   other clauses keep most of it),
//! - `base(s)`: the post-filter route's nominal cost on the base graph at the match rate `s/n`
//!   (the label diluted `n/s` times),
//! - `island(s)`: the same route's cost on an island of `s` points at match rate 1,
//! - gain `g(s) = min(exact(s), base(s)) − island(s)`.
//!
//! Labels with `g > 0` and at least [`MIN_SIZE`] members are taken by decreasing
//! `g(s) / bytes(s)` while the budget lasts, `bytes(s) = s · (8 + 3R)` (its rows at the
//! expected mean degree `0.75 R`).
//!
//! Build: the lab's Vamana builder ([`crate::build::build`], label-blind α-RobustPrune) over each
//! selected label's member vectors, one island per thread. Local ids are positions in the
//! label's posting list; the builder's auxiliary start node (a copy of the medoid) is merged
//! into the medoid, the island's start, and members unreachable from it get an edge from it
//! (counted in [`IslandStats::patched`]).
//!
//! Memory: 4 bytes per label (its island index), 16 bytes per island, 8 bytes per member (row
//! offset) and 4 bytes per edge.

use std::{
    io::{Read, Seek, Write},
    path::Path,
    time::Instant,
};

use anyhow::{ensure, Context, Result};

use crate::{
    build::{self, BuildElement, BuildParams},
    container,
    kernel::cost::{kth_match_rank, UnitCosts, Z},
    labels::LabelStore,
    vectors::Matrix,
};

/// File magic of `islands.flab`.
pub const MAGIC: &[u8; 8] = b"FLABISL\0";
/// Version of the `islands.flab` format.
pub const VERSION: u32 = 1;

const NONE: u32 = u32::MAX;
/// Smallest label with an island.
pub const MIN_SIZE: usize = 64;

/// The post-filter route's list size and nominal cost (the kernel's crossover formula: a list
/// that holds the `K` nearest matches with high probability at match rate `rate` on a graph of
/// `n` points, at least `l`; `starts` start nodes).
pub fn postfilter_nominal(
    c: &UnitCosts,
    l: usize,
    k: usize,
    rate: f64,
    n: usize,
    starts: usize,
) -> (usize, f64) {
    let rank = kth_match_rank(k, rate, Z);
    if !rank.is_finite() {
        return (l, f64::INFINITY);
    }
    let l_need = c.graph_l_for(rank).ceil().min(n as f64) as usize;
    let l_pf = l_need.max(l).min(n.max(1));
    (
        l_pf,
        c.postfilter_cost(l_pf as f64, k, rate) + starts as f64 * c.graph_dist_ns,
    )
}

/// Selection terms of one label (see the module documentation).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Gain {
    pub label: u32,
    pub size: usize,
    pub exact_ns: f64,
    pub base_ns: f64,
    pub island_ns: f64,
    pub gain_ns: f64,
    pub bytes: u64,
}

/// Expected bytes of an island of `s` members with degree bound `r`.
pub fn island_bytes(s: usize, r: usize) -> u64 {
    (s as u64) * (8 + 3 * r as u64)
}

/// The selection terms of every label with at least [`MIN_SIZE`] members.
pub fn gains(labels: &LabelStore, c: &UnitCosts, k: usize, r: usize) -> Vec<Gain> {
    let n = labels.n_points();
    let nf = n as f64;
    (0..labels.n_labels() as u32)
        .filter_map(|l| {
            let s = labels.posting_len(l);
            if s < MIN_SIZE || s >= n {
                return None;
            }
            let rho = s as f64 / nf;
            let exact_ns = s as f64 * (c.dist(rho) + c.pred(rho));
            let base_ns = postfilter_nominal(c, k, k, rho, n, 1).1;
            let island_ns = postfilter_nominal(c, k, k, 1.0, s, 1).1;
            Some(Gain {
                label: l,
                size: s,
                exact_ns,
                base_ns,
                island_ns,
                gain_ns: exact_ns.min(base_ns) - island_ns,
                bytes: island_bytes(s, r),
            })
        })
        .collect()
}

/// The labels to build within `budget_bytes`: positive gains by decreasing gain per byte (in
/// label-id order).
pub fn select(gains: &[Gain], budget_bytes: u64) -> Vec<u32> {
    let mut order: Vec<&Gain> = gains.iter().filter(|g| g.gain_ns > 0.0).collect();
    order.sort_by(|a, b| {
        (b.gain_ns / b.bytes as f64)
            .total_cmp(&(a.gain_ns / a.bytes as f64))
            .then(a.label.cmp(&b.label))
    });
    let mut used = 0u64;
    let mut out = Vec::new();
    for g in order {
        if used + g.bytes > budget_bytes {
            continue;
        }
        used += g.bytes;
        out.push(g.label);
    }
    out.sort_unstable();
    out
}

/// Per-label sub-graphs (see the module documentation).
#[derive(Debug, Clone, PartialEq)]
pub struct IslandIndex {
    n_points: usize,
    dict_hash: u64,
    r: u32,
    l_build: u32,
    /// Island of every label (`NONE` without).
    island_of: Vec<u32>,
    /// Per island: its label and the local id of its start (medoid).
    labels: Vec<u32>,
    medoids: Vec<u32>,
    /// Island `i`'s rows: `offsets[node_base[i]..=node_base[i] + s_i]` (edge offsets).
    node_base: Vec<u64>,
    offsets: Vec<u64>,
    /// Local ids of the neighbors.
    edges: Vec<u32>,
}

/// Size and build statistics.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct IslandStats {
    pub islands: usize,
    pub members: u64,
    pub edges: u64,
    pub bytes: u64,
    pub build_s: f64,
    pub n_dist: u64,
    /// Members unreachable from their island's medoid after the build, given an edge from it.
    pub patched: u64,
}

/// The rows of an island from the builder's graph over its `s` members (local ids; the
/// builder's auxiliary start nodes, copies of the medoid, from `s` on): the auxiliary nodes
/// are merged into the medoid (edges to them point to it, their out-edges are added to its
/// row), then every member still unreachable from the medoid gets an edge from it (the
/// medoid's row may exceed the degree bound). Returns the row offsets, the edges and the
/// number of members patched.
fn merge_start(g: &crate::graph::Graph, s: usize, medoid: u32) -> (Vec<u32>, Vec<u32>, u64) {
    let aux_out: Vec<u32> = (s..g.n_nodes())
        .flat_map(|a| g.neighbors(a as u32).iter().copied())
        .filter(|&u| (u as usize) < s)
        .collect();
    let mut rows: Vec<Vec<u32>> = (0..s as u32)
        .map(|v| {
            let mut row: Vec<u32> = Vec::with_capacity(g.neighbors(v).len());
            let extra = if v == medoid { &aux_out[..] } else { &[][..] };
            for &u in g.neighbors(v).iter().chain(extra) {
                let u = if (u as usize) < s { u } else { medoid };
                if u != v && !row.contains(&u) {
                    row.push(u);
                }
            }
            row
        })
        .collect();
    let mut seen = vec![false; s];
    let mut stack = vec![medoid];
    seen[medoid as usize] = true;
    while let Some(v) = stack.pop() {
        for &u in &rows[v as usize] {
            if !seen[u as usize] {
                seen[u as usize] = true;
                stack.push(u);
            }
        }
    }
    let unreachable: Vec<u32> = (0..s as u32).filter(|&v| !seen[v as usize]).collect();
    let patched = unreachable.len() as u64;
    rows[medoid as usize].extend(unreachable);
    let mut offsets = Vec::with_capacity(s + 1);
    let mut edges = Vec::new();
    for row in rows {
        offsets.push(edges.len() as u32);
        edges.extend(row);
    }
    offsets.push(edges.len() as u32);
    (offsets, edges, patched)
}

impl IslandIndex {
    /// Builds the islands of `selected` (label ids of `labels`, increasing) over `base`: each
    /// island with `params` (its rule must be label-blind; one thread per island), `threads`
    /// islands at a time. The result does not depend on the thread count. Returns the per-island
    /// build seconds (in `selected` order) with the statistics.
    pub fn build<T: BuildElement>(
        base: &Matrix<T>,
        labels: &LabelStore,
        selected: &[u32],
        params: &BuildParams,
        threads: usize,
    ) -> Result<(Self, IslandStats, Vec<f64>)> {
        ensure!(
            base.nrows() == labels.n_points(),
            "the base has {} rows, the labels {} points",
            base.nrows(),
            labels.n_points()
        );
        ensure!(
            !params.rule.needs_labels(),
            "islands use a label-blind rule"
        );
        ensure!(
            selected.windows(2).all(|w| w[0] < w[1]),
            "the selected labels must increase"
        );
        let t0 = Instant::now();
        let d = base.ncols();
        let params = BuildParams {
            threads: 1,
            ..params.clone()
        };
        type Built = Result<(u32, Vec<u32>, Vec<u32>, u64, u64, f64)>;
        let built: Vec<Built> =
            crate::parallel::par_map(selected.len(), threads, Vec::new, |x, buf: &mut Vec<T>| {
                let l = selected[x];
                let members = labels.postings(l);
                let s = members.len();
                ensure!(s >= 2, "label {l} has fewer than two members");
                let t = Instant::now();
                buf.clear();
                for &p in members {
                    buf.extend_from_slice(base.row(p as usize));
                }
                let m = Matrix::from_slice(buf, s, d);
                let out = build::build(&m, None, &params)
                    .with_context(|| format!("building the island of label {l}"))?;
                let (offsets, edges, patched) = merge_start(&out.graph, s, out.stats.medoid);
                let n_dist =
                    out.stats.n_dist_search + out.stats.n_dist_prune + out.stats.n_dist_reprune;
                Ok((
                    out.stats.medoid,
                    offsets,
                    edges,
                    n_dist,
                    patched,
                    t.elapsed().as_secs_f64(),
                ))
            });
        let mut ix = Self {
            n_points: labels.n_points(),
            dict_hash: labels.dict_hash(),
            r: params.rule.degree() as u32,
            l_build: params.l_build as u32,
            island_of: vec![NONE; labels.n_labels()],
            labels: Vec::new(),
            medoids: Vec::new(),
            node_base: vec![0],
            offsets: Vec::new(),
            edges: Vec::new(),
        };
        let mut stats = IslandStats::default();
        let mut secs = Vec::with_capacity(selected.len());
        for (&l, b) in selected.iter().zip(built) {
            let (medoid, offsets, edges, n_dist, patched, t) = b?;
            let base_edge = ix.edges.len() as u64;
            ix.island_of[l as usize] = ix.labels.len() as u32;
            ix.labels.push(l);
            ix.medoids.push(medoid);
            ix.offsets
                .extend(offsets.iter().map(|&o| base_edge + u64::from(o)));
            ix.edges.extend_from_slice(&edges);
            ix.node_base.push(ix.offsets.len() as u64);
            stats.members += (offsets.len() - 1) as u64;
            stats.n_dist += n_dist;
            stats.patched += patched;
            secs.push(t);
        }
        stats.islands = ix.labels.len();
        stats.edges = ix.edges.len() as u64;
        stats.bytes = ix.bytes();
        stats.build_s = t0.elapsed().as_secs_f64();
        Ok((ix, stats, secs))
    }

    /// The islands of the labels in `keep` (others dropped).
    pub fn subset(&self, keep: &[u32]) -> Self {
        let mut out = Self {
            n_points: self.n_points,
            dict_hash: self.dict_hash,
            r: self.r,
            l_build: self.l_build,
            island_of: vec![NONE; self.island_of.len()],
            labels: Vec::new(),
            medoids: Vec::new(),
            node_base: vec![0],
            offsets: Vec::new(),
            edges: Vec::new(),
        };
        for i in 0..self.labels.len() {
            let l = self.labels[i];
            if !keep.contains(&l) {
                continue;
            }
            let (b, e) = (self.node_base[i] as usize, self.node_base[i + 1] as usize);
            let (lo, hi) = (self.offsets[b] as usize, self.offsets[e - 1] as usize);
            let shift = out.edges.len() as u64;
            out.island_of[l as usize] = out.labels.len() as u32;
            out.labels.push(l);
            out.medoids.push(self.medoids[i]);
            out.offsets
                .extend(self.offsets[b..e].iter().map(|&o| o - lo as u64 + shift));
            out.edges.extend_from_slice(&self.edges[lo..hi]);
            out.node_base.push(out.offsets.len() as u64);
        }
        out
    }

    pub fn n_points(&self) -> usize {
        self.n_points
    }

    pub fn n_islands(&self) -> usize {
        self.labels.len()
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn n_members(&self) -> usize {
        self.offsets.len() - self.labels.len()
    }

    pub fn degree_bound(&self) -> usize {
        self.r as usize
    }

    /// Bytes held (every array).
    pub fn bytes(&self) -> u64 {
        (4 * self.island_of.len()
            + 4 * self.labels.len()
            + 4 * self.medoids.len()
            + 8 * self.node_base.len()
            + 8 * self.offsets.len()
            + 4 * self.edges.len()) as u64
    }

    /// The island of label `l`, if any.
    #[inline]
    pub fn island(&self, l: u32) -> Option<usize> {
        let i = self.island_of[l as usize];
        (i != NONE).then_some(i as usize)
    }

    /// Label of island `i`.
    #[inline]
    pub fn label(&self, i: usize) -> u32 {
        self.labels[i]
    }

    /// Members of island `i`.
    #[inline]
    pub fn size(&self, i: usize) -> usize {
        (self.node_base[i + 1] - self.node_base[i]) as usize - 1
    }

    /// Start (local id) of island `i`.
    #[inline]
    pub fn medoid(&self, i: usize) -> u32 {
        self.medoids[i]
    }

    /// Neighbors (local ids) of local node `v` of island `i`.
    #[inline]
    pub fn neighbors(&self, i: usize, v: u32) -> &[u32] {
        let r = self.node_base[i] as usize + v as usize;
        &self.edges[self.offsets[r] as usize..self.offsets[r + 1] as usize]
    }

    /// Prefetches the row of local node `v` of island `i`.
    #[inline]
    pub fn prefetch_row(&self, i: usize, v: u32) {
        let r = self.node_base[i] as usize + v as usize;
        let at = self.offsets[r] as usize;
        crate::kernel::prefetch(self.edges.as_ptr().wrapping_add(at));
    }

    /// Checks that the index belongs to `labels` (dictionary, counts, island sizes).
    pub fn check(&self, labels: &LabelStore) -> Result<()> {
        ensure!(
            self.dict_hash == labels.dict_hash()
                && self.island_of.len() == labels.n_labels()
                && self.n_points == labels.n_points(),
            "the islands refer to another label store"
        );
        for (i, &l) in self.labels.iter().enumerate() {
            ensure!(
                self.size(i) == labels.posting_len(l),
                "island {i} has {} members, label {l} {}",
                self.size(i),
                labels.posting_len(l)
            );
        }
        Ok(())
    }

    /// Writes `islands.flab`.
    pub fn write(&self, path: &Path) -> Result<()> {
        self.write_to(container::create(path)?)
            .with_context(|| format!("writing {}", path.display()))
    }

    pub fn write_to<W: Write>(&self, w: W) -> Result<()> {
        let params = [
            self.n_points as u64,
            self.island_of.len() as u64,
            self.dict_hash,
            self.labels.len() as u64,
            self.offsets.len() as u64,
            self.edges.len() as u64,
            u64::from(self.r),
            u64::from(self.l_build),
        ];
        container::write(
            w,
            MAGIC,
            VERSION,
            &params,
            &[
                bytemuck::cast_slice(&self.island_of),
                bytemuck::cast_slice(&self.labels),
                bytemuck::cast_slice(&self.medoids),
                bytemuck::cast_slice(&self.node_base),
                bytemuck::cast_slice(&self.offsets),
                bytemuck::cast_slice(&self.edges),
            ],
        )
    }

    /// Reads and validates `islands.flab` (structure only; [`Self::check`] compares it with a
    /// label store).
    pub fn read(path: &Path) -> Result<Self> {
        container::Reader::open(path, MAGIC)
            .and_then(Self::load)
            .with_context(|| format!("loading islands from {}", path.display()))
    }

    pub fn read_from<R: Read + Seek>(r: R) -> Result<Self> {
        Self::load(container::Reader::new(r, MAGIC)?)
    }

    fn load<R: Read + Seek>(mut r: container::Reader<R>) -> Result<Self> {
        ensure!(
            r.version() == VERSION,
            "unsupported version {}",
            r.version()
        );
        let [n_points, n_labels, dict_hash, n_islands, n_offsets, n_edges, deg, l_build] =
            *r.params();
        let ni = n_islands as usize;
        let island_of: Vec<u32> = r.section(0, n_labels as usize)?;
        let labels: Vec<u32> = r.section(1, ni)?;
        let medoids: Vec<u32> = r.section(2, ni)?;
        let node_base: Vec<u64> = r.section(3, ni + 1)?;
        let offsets: Vec<u64> = r.section(4, n_offsets as usize)?;
        let edges: Vec<u32> = r.section(5, n_edges as usize)?;
        ensure!(
            node_base.first() == Some(&0)
                && node_base.last() == Some(&n_offsets)
                && node_base.windows(2).all(|w| w[1] >= w[0] + 2),
            "inconsistent island rows"
        );
        ensure!(
            offsets.windows(2).all(|w| w[0] <= w[1])
                && offsets.last().is_none_or(|&o| o == n_edges),
            "inconsistent edge offsets"
        );
        for i in 0..ni {
            let (b, e) = (node_base[i] as usize, node_base[i + 1] as usize);
            let s = (e - b - 1) as u32;
            ensure!(
                island_of.get(labels[i] as usize) == Some(&(i as u32)) && medoids[i] < s,
                "inconsistent island {i}"
            );
            let es = &edges[offsets[b] as usize..offsets[e - 1] as usize];
            ensure!(
                es.iter().all(|&u| u < s),
                "island {i} has an edge out of range"
            );
        }
        Ok(Self {
            n_points: n_points as usize,
            dict_hash,
            r: deg as u32,
            l_build: l_build as u32,
            island_of,
            labels,
            medoids,
            node_base,
            offsets,
            edges,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build::PruneRule, labels::LabelStoreBuilder};

    #[test]
    fn islands_hold_their_labels_members_and_read_back() {
        // 600 points on a line (1-d u8 vectors cycling), labels by residue.
        let n = 600usize;
        let mut lb = LabelStoreBuilder::new();
        let mut data = Vec::new();
        for p in 0..n {
            data.push((p % 251) as u8);
            let r = format!("r{}", p % 3);
            let half = if p < 300 { "lo" } else { "hi" };
            lb.push_point([("k", r.as_str()), ("h", half)]).unwrap();
        }
        let labels = lb.finish().unwrap();
        let base = Matrix::from_slice(&data, n, 1);
        let id = labels.name_index();
        let sel = vec![id["r0"].min(id["lo"]), id["r0"].max(id["lo"])];
        let params = BuildParams {
            rule: PruneRule::Alpha { r: 8, alpha: 1.2 },
            l_build: 16,
            threads: 1,
            ..BuildParams::default()
        };
        let (ix, st, secs) = IslandIndex::build(&base, &labels, &sel, &params, 2).unwrap();
        assert_eq!(secs.len(), 2);
        assert_eq!(st.islands, 2);
        ix.check(&labels).unwrap();
        for &l in &sel {
            let i = ix.island(l).unwrap();
            assert_eq!(ix.label(i), l);
            assert_eq!(ix.size(i), labels.posting_len(l));
            for v in 0..ix.size(i) as u32 {
                assert!(ix
                    .neighbors(i, v)
                    .iter()
                    .all(|&u| (u as usize) < ix.size(i)));
            }
        }
        assert!(ix.island(id["r1"]).is_none());
        // A subset keeps the islands it names, identically.
        let one = ix.subset(&[id["lo"]]);
        assert_eq!(one.n_islands(), 1);
        let (a, b) = (ix.island(id["lo"]).unwrap(), one.island(id["lo"]).unwrap());
        for v in 0..ix.size(a) as u32 {
            assert_eq!(ix.neighbors(a, v), one.neighbors(b, v));
        }
        let mut buf = Vec::new();
        ix.write_to(&mut buf).unwrap();
        assert_eq!(
            IslandIndex::read_from(std::io::Cursor::new(buf)).unwrap(),
            ix
        );
    }

    #[test]
    fn selection_prefers_gain_per_byte_within_the_budget() {
        let g = |label, size, gain_ns| Gain {
            label,
            size,
            exact_ns: 0.0,
            base_ns: 0.0,
            island_ns: 0.0,
            gain_ns,
            bytes: island_bytes(size, 32),
        };
        let gains = [
            g(0, 100, 1000.0),
            g(1, 1000, 1000.0),
            g(2, 100, -5.0),
            g(3, 200, 3000.0),
        ];
        // Gain per byte: 3 > 0 > 1; 2 loses.
        assert_eq!(select(&gains, island_bytes(300, 32)), vec![0, 3]);
        assert_eq!(select(&gains, u64::MAX), vec![0, 1, 3]);
        assert!(select(&gains, 10).is_empty());
    }
}
