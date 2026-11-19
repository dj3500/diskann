/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The effort curve of a Steiner graph's router trees (`flab tree-calibrate`; read by
//! `tree_effort = curve` through [`crate::kernel::cost::UnitCosts::tree_list`]).
//!
//! - **Labels**: the labels whose per-label starts are all auxiliary nodes (router-tree roots),
//!   in size classes by decade (≤ 10 points, 11–100, 101–1,000, ...; a tree's depth and leaf
//!   count grow with its label's size); per class, `queries` labels sampled with probability
//!   proportional to their size (seeded).
//! - **Queries**: per sampled label, the midpoint of two random members (`label`, queries
//!   inside the label's region) or of two random data points (`base`, queries independent of
//!   the label, as the graph's effort curve; `u8`: rounded half up).
//! - **Ground truth**: the label's `kmax` nearest members (one scan of its postings).
//! - **Searches**: the tree route's search with a frontier of routers only (`aux_frontier`),
//!   the label as the predicate, a heap of `kmax`. The traversal does not depend on the
//!   collector, so the `K'` nearest collected members are the answers of a heap(K') search with
//!   the same frontier, for every `K' ≤ kmax`; a query counts for the `K'` its label has members
//!   for. Frontiers 2^i up to `l_max`, stopping after the first at which every `K'` has a mean
//!   recall of at least 0.995 (per class).
//! - **Table**: per class as the graph's effort curve ([`super::effort::fit_table`]): per target
//!   recall and `K'`, the smallest frontier whose mean recall reaches it; a class without labels
//!   of `K'` members extends its largest measured `K'` in proportion (as the lookup does). The
//!   class's size (`tree_a`) is the geometric mean of its sampled labels' sizes.

use std::{collections::BinaryHeap, time::Instant};

use anyhow::{ensure, Result};
use serde_json::{json, Value};

use super::{
    effort::{fit_table, midpoint, Dist, K_GRID, R_GRID},
    Rng,
};
use crate::{
    kernel::{
        config::{
            Admission, Collector as Coll, Expander, Kernel, Laziness, StartAdmission, StartSource,
        },
        search::{Index, Plan, Query, Searcher},
    },
    labels::LabelStore,
    predicate::PredicateSetBuilder,
    vectors::{Element, Matrix},
};

/// The frontiers stop growing once every K' reaches this mean recall.
const STOP_RECALL: f64 = 0.995;
/// Extrapolation above the largest measured frontier stops at this multiple of it.
const EXTRAPOLATION_CAP: f64 = 8.0;

/// Where the calibration queries come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeQueries {
    /// Midpoints of two random members of the sampled label.
    Label,
    /// Midpoints of two random data points (independent of the label).
    Base,
}

impl TreeQueries {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim() {
            "label" => Self::Label,
            "base" => Self::Base,
            v => anyhow::bail!("unknown tree calibration queries {v:?} (label, base)"),
        })
    }
}

/// Options of the tree effort calibration.
#[derive(Debug, Clone)]
pub struct TreeOptions {
    /// Queries per size class (one sampled label each).
    pub queries: usize,
    pub seed: u64,
    /// Largest measured frontier.
    pub l_max: usize,
    pub queries_from: TreeQueries,
}

impl Default for TreeOptions {
    fn default() -> Self {
        Self {
            queries: 200,
            seed: 0,
            l_max: 65_536,
            queries_from: TreeQueries::Label,
        }
    }
}

/// The labels whose per-label starts are all auxiliary nodes (roots of router trees).
pub fn tree_labels<T: Element>(ix: &Index<'_, T>, labels: &LabelStore) -> Vec<u32> {
    let n = ix.graph.n_data() as u32;
    let Some(t) = ix.label_starts else {
        return Vec::new();
    };
    (0..t.n_labels().min(labels.n_labels()) as u32)
        .filter(|&l| {
            let s = t.label_starts(l);
            labels.posting_len(l) > 0 && !s.is_empty() && s.iter().all(|&v| v >= n)
        })
        .collect()
}

/// The size class of a label of `s` points (decades: ≤ 10, 11–100, 101–1,000, ...).
fn size_class(s: usize) -> usize {
    let mut c = 0;
    let mut top = 10usize;
    while s > top {
        c += 1;
        top = top.saturating_mul(10);
    }
    c
}

/// One size class's measurement: its frontiers, per-`K'` mean recalls, work and report.
struct ClassCurve {
    size: f64,
    ks: Vec<usize>,
    ls: Vec<f64>,
    recall: Vec<Vec<f64>>,
    doc: Value,
}

/// Measures the recall of tree searches over the labels `trees` (one class; see the module
/// documentation).
fn class_curve<T: Element>(
    base: &Matrix<T>,
    ix: &Index<'_, T>,
    labels: &LabelStore,
    trees: &[u32],
    opts: &TreeOptions,
    seed: u64,
) -> Result<ClassCurve> {
    let t0 = Instant::now();
    // Labels sampled in proportion to their size.
    let mut cum = Vec::with_capacity(trees.len());
    let mut total = 0u64;
    for &l in trees {
        total += labels.posting_len(l) as u64;
        cum.push(total);
    }
    let mut rng = Rng(seed);
    let mut chosen = Vec::with_capacity(opts.queries);
    let mut queries: Vec<Vec<T>> = Vec::with_capacity(opts.queries);
    for _ in 0..opts.queries {
        let x = rng.next() % total;
        let l = trees[cum.partition_point(|&c| c <= x)];
        let (a, b) = match opts.queries_from {
            TreeQueries::Label => {
                let p = labels.postings(l);
                (p[rng.below(p.len())], p[rng.below(p.len())])
            }
            TreeQueries::Base => (
                rng.below(base.nrows()) as u32,
                rng.below(base.nrows()) as u32,
            ),
        };
        queries.push(midpoint(base.row(a as usize), base.row(b as usize)));
        chosen.push(l);
    }
    let kmax = K_GRID[K_GRID.len() - 1];
    // Ground truth: the kmax nearest members of each query's label.
    let gt: Vec<Vec<f32>> = queries
        .iter()
        .zip(&chosen)
        .map(|(q, &l)| {
            let mut h: BinaryHeap<Dist> = BinaryHeap::with_capacity(kmax + 1);
            for &x in labels.postings(l) {
                let d = T::squared_l2(q, base.row(x as usize));
                if h.len() < kmax {
                    h.push(Dist(d));
                } else if let Some(mut top) = h.peek_mut() {
                    if d < top.0 {
                        *top = Dist(d);
                    }
                }
            }
            h.into_sorted_vec().into_iter().map(|d| d.0).collect()
        })
        .collect();
    let t_gt = t0.elapsed().as_secs_f64();
    let ks: Vec<usize> = K_GRID
        .iter()
        .copied()
        .filter(|&k| gt.iter().any(|g| g.len() >= k))
        .collect();
    ensure!(!ks.is_empty(), "the sampled labels are too small");

    let kernel = Kernel {
        starts: vec![StartSource::LabelSmallest],
        start_admission: StartAdmission::Force,
        admission: Admission::Strict,
        laziness: Laziness::PredicateFirst,
        collector: Coll::Heap,
        expander: Expander::AuxFrontier,
        ..Kernel::default()
    };
    let plan = Plan::new(&kernel, ix)?;
    let mut st = Searcher::for_index(ix, &[&plan]);
    let preds = {
        let mut pb = PredicateSetBuilder::new(labels);
        for &l in &chosen {
            pb.push(&[vec![labels.label_name(l)]]);
        }
        pb.finish()
    };
    let mut ls = Vec::new();
    let mut l = 2usize;
    while l <= opts.l_max.max(2) {
        ls.push(l);
        l *= 2;
    }
    let (mut ids, mut ds) = (vec![0u32; kmax], vec![0f32; kmax]);
    let mut recall = vec![vec![0f64; ls.len()]; ks.len()];
    let count: Vec<f64> = ks
        .iter()
        .map(|&k| gt.iter().filter(|g| g.len() >= k).count() as f64)
        .collect();
    let (mut n_dist, mut n_aux) = (vec![0f64; ls.len()], vec![0f64; ls.len()]);
    let m = queries.len() as f64;
    let mut measured = ls.len();
    for (li, &l) in ls.iter().enumerate() {
        for (qi, (q, d_gt)) in queries.iter().zip(&gt).enumerate() {
            let query = Query {
                vector: q.as_slice(),
                cnf: preds.get(qi),
                view: None,
            };
            let c = st.search(&plan, ix, &query, l, &mut ids, &mut ds, None);
            n_dist[li] += c.n_dist as f64;
            n_aux[li] += c.n_aux_expand as f64;
            // Tie-aware recall@K': answers within the K'-th true distance.
            for (row, &kp) in recall.iter_mut().zip(&ks) {
                if d_gt.len() < kp {
                    continue;
                }
                let bound = d_gt[kp - 1];
                let hits = ds[..kp].iter().filter(|&&d| d <= bound).count();
                row[li] += hits as f64 / kp as f64;
            }
        }
        for (row, &c) in recall.iter_mut().zip(&count) {
            row[li] /= c.max(1.0);
        }
        n_dist[li] /= m;
        n_aux[li] /= m;
        if recall.iter().all(|row| row[li] >= STOP_RECALL) {
            measured = li + 1;
            break;
        }
    }
    ls.truncate(measured);
    n_dist.truncate(measured);
    n_aux.truncate(measured);
    for row in &mut recall {
        row.truncate(measured);
    }
    let mut sizes: Vec<usize> = chosen.iter().map(|&l| labels.posting_len(l)).collect();
    sizes.sort_unstable();
    let size = (sizes.iter().map(|&s| (s as f64).ln()).sum::<f64>() / sizes.len() as f64).exp();
    let by_k: serde_json::Map<String, Value> = ks
        .iter()
        .zip(&recall)
        .map(|(k, r)| (k.to_string(), json!(r)))
        .collect();
    let doc = json!({
        "labels": trees.len(),
        "queries": queries.len(),
        "size": size,
        "label_sizes_sampled_min_median_max": [sizes[0], sizes[sizes.len() / 2], sizes[sizes.len() - 1]],
        "queries_per_k": ks.iter().zip(&count).map(|(k, c)| json!([k, c])).collect::<Vec<_>>(),
        "l": ls,
        "n_dist": n_dist,
        "n_router_expansions": n_aux,
        "recall": by_k,
        "ground_truth_s": t_gt,
        "elapsed_s": t0.elapsed().as_secs_f64(),
    });
    Ok(ClassCurve {
        size,
        ks,
        ls: ls.iter().map(|&l| l as f64).collect(),
        recall,
        doc,
    })
}

/// The tree effort curve of a graph of `ix`: one table per label-size class (see the module
/// documentation), with the classes' sizes.
pub struct TreeCurve {
    pub r: Vec<f64>,
    pub k: Vec<f64>,
    /// `a.len()` blocks of `r.len()` rows (class by class).
    pub l: Vec<Vec<f64>>,
    pub a: Vec<f64>,
    pub doc: Value,
}

/// Measures the tree effort curve of the graph of `ix` (see the module documentation).
pub fn tree_curve<T: Element>(
    base: &Matrix<T>,
    ix: &Index<'_, T>,
    labels: &LabelStore,
    opts: &TreeOptions,
) -> Result<TreeCurve> {
    let t0 = Instant::now();
    let trees = tree_labels(ix, labels);
    ensure!(
        !trees.is_empty(),
        "no label has router-tree roots as its only starts (a Steiner graph's start table?)"
    );
    ensure!(opts.queries > 0, "the tree effort curve needs queries");
    let mut by_class: Vec<Vec<u32>> = Vec::new();
    for &l in &trees {
        let c = size_class(labels.posting_len(l));
        if by_class.len() <= c {
            by_class.resize(c + 1, Vec::new());
        }
        by_class[c].push(l);
    }
    let mut classes = Vec::new();
    for (c, members) in by_class.iter().enumerate() {
        if !members.is_empty() {
            let seed = opts.seed.wrapping_add(c as u64);
            classes.push(class_curve(base, ix, labels, members, opts, seed)?);
        }
    }
    // Shared K' columns: a class without labels of K' members extends its largest measured K'
    // in proportion (the lookup's rule beyond the table).
    let ks: Vec<usize> = K_GRID
        .iter()
        .copied()
        .filter(|k| classes.iter().any(|c| c.ks.contains(k)))
        .collect();
    let (mut l, mut a, mut extrapolated, mut docs) = (Vec::new(), Vec::new(), 0, Vec::new());
    for c in &classes {
        let top = c.ls[c.ls.len() - 1];
        let cap = EXTRAPOLATION_CAP * top;
        let table = fit_table(&c.ls, &c.recall, &R_GRID, cap);
        extrapolated += table.iter().flatten().filter(|&&x| x > top).count();
        let k_top = *c.ks.last().expect("measured") as f64;
        for row in table {
            l.push(
                ks.iter()
                    .map(|&k| match c.ks.iter().position(|&x| x == k) {
                        Some(j) => row[j],
                        None => row[row.len() - 1] * k as f64 / k_top,
                    })
                    .collect(),
            );
        }
        a.push(c.size);
        docs.push(c.doc.clone());
    }
    let doc = json!({
        "queries_from": format!("{:?}", opts.queries_from).to_lowercase(),
        "queries_per_class": opts.queries,
        "seed": opts.seed,
        "tree_labels": trees.len(),
        "class_sizes": a,
        "classes": docs,
        "extrapolated_entries": extrapolated,
        "extrapolation_cap": EXTRAPOLATION_CAP,
        "elapsed_s": t0.elapsed().as_secs_f64(),
    });
    // Equal class sizes cannot happen (decades), but guard the increasing order.
    ensure!(
        a.windows(2).all(|w| w[0] < w[1]),
        "size classes out of order"
    );
    Ok(TreeCurve {
        r: R_GRID.to_vec(),
        k: ks.iter().map(|&k| k as f64).collect(),
        l,
        a,
        doc,
    })
}

/// `flab tree-calibrate`: the calibration document `calibration` (a `flab calibrate` output
/// for the base graph, whose searches the router trees leave unchanged) with the tree effort
/// curve of the Steiner graph `graph` (and its start table `starts`) added to its unit costs
/// (`tree_r`, `tree_k`, `tree_l`) and its report under `details.tree`.
pub fn tree_calibrate<T: Element>(
    dataset: &std::path::Path,
    graph: &std::path::Path,
    starts: Option<&std::path::Path>,
    calibration: &Value,
    opts: &TreeOptions,
) -> Result<Value> {
    let ds = crate::run::data::Dataset::<T>::load(dataset, &[], &[])?;
    let g = crate::run::data::load_graph(graph, starts, T::DTYPE)?;
    ensure!(
        g.graph.n_data() == ds.base.nrows(),
        "the graph has another point count"
    );
    let label_starts = g.starts.as_ref().filter(|s| s.n_labels() > 0);
    ensure!(
        label_starts.is_some(),
        "the tree effort curve needs per-label starts"
    );
    let ix = Index {
        graph: &g.graph,
        vectors: g.graph.vectors(&ds.base)?,
        labels: &ds.labels,
        global: &g.global,
        label_starts,
        sample: None,
        label_view: false,
        view_postings: None,
        eps_min: 0.0,
        bitsets: None,
        costs: None,
        sketches: None,
        pairs: None,
    };
    let e = tree_curve(&ds.base, &ix, &ds.labels, opts)?;
    let mut doc = calibration.clone();
    let mut costs: crate::kernel::cost::UnitCosts = serde_json::from_value(
        doc.get("costs")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("the calibration has no unit costs"))?,
    )?;
    (costs.tree_r, costs.tree_k, costs.tree_l, costs.tree_a) = (e.r, e.k, e.l, e.a);
    costs.validate()?;
    doc["costs"] = serde_json::to_value(&costs)?;
    let mut tree = e.doc;
    tree["graph"] = json!(graph);
    tree["starts"] = json!(starts);
    tree["host"] = json!(crate::run::sys::host_info());
    tree["cpu"] = json!(crate::run::sys::current_cpu());
    tree["time"] = json!(crate::run::sys::utc_timestamp(false));
    if !doc["details"].is_object() {
        doc["details"] = json!({});
    }
    doc["details"]["tree"] = tree;
    Ok(doc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph::{AuxKind, AuxNodes, Graph},
        labels::LabelStoreBuilder,
        starts::Starts,
    };

    /// 64 points on a line (x = i), label `a=1` on the even points, one router per 8 of its
    /// members (in order) under a root (the label's only start); `a=2` on the odd points
    /// with a data-point start.
    fn toy() -> (Matrix<f32>, Graph, LabelStore, Starts) {
        let n = 64;
        let base = Matrix::from_slice(&(0..n).map(|i| i as f32).collect::<Vec<_>>(), n, 1);
        let mut b = LabelStoreBuilder::new();
        for i in 0..n {
            b.push_point([("a", if i % 2 == 0 { "a=1" } else { "a=2" })])
                .unwrap();
        }
        let labels = b.finish().unwrap();
        // Routers 64..68 over the even points 16 at a time (centroids), root 68.
        let mut vecs = Vec::new();
        for r in 0..4 {
            vecs.push((r * 16 + 7) as f32);
        }
        vecs.push(31.0);
        let aux = AuxNodes::new(vec![AuxKind::Router; 5], &Matrix::from_slice(&vecs, 5, 1));
        let mut g = Graph::new(n, 8, aux.unwrap()).unwrap();
        for i in 0..n as u32 {
            let nb: Vec<u32> = [i.checked_sub(1), (i + 1 < n as u32).then_some(i + 1)]
                .into_iter()
                .flatten()
                .collect();
            g.set_neighbors(i, &nb).unwrap();
        }
        for r in 0..4u32 {
            let members: Vec<u32> = (0..8).map(|j| r * 16 + 2 * j).collect();
            g.set_neighbors(64 + r, &members).unwrap();
        }
        g.set_neighbors(68, &[64, 65, 66, 67]).unwrap();
        g.set_starts(vec![0]).unwrap();
        let id = |s: &str| labels.name_index()[s];
        let mut st = Starts::new(&g);
        st.set_label_starts(&labels, &[(id("a=1"), 68), (id("a=2"), 1)])
            .unwrap();
        (base, g, labels, st)
    }

    #[test]
    fn the_tree_curve_samples_tree_labels_and_reaches_full_recall() {
        let (base, g, labels, st) = toy();
        let global = [0u32];
        let ix = Index {
            graph: &g,
            vectors: g.vectors(&base).unwrap(),
            labels: &labels,
            global: &global,
            label_starts: Some(&st),
            sample: None,
            label_view: false,
            view_postings: None,
            eps_min: 0.0,
            bitsets: None,
            costs: None,
            sketches: None,
            pairs: None,
        };
        assert_eq!(tree_labels(&ix, &labels), [labels.name_index()["a=1"]]);
        let e = tree_curve(
            &base,
            &ix,
            &labels,
            &TreeOptions {
                queries: 20,
                seed: 3,
                l_max: 64,
                queries_from: TreeQueries::Label,
            },
        )
        .unwrap();
        // a=1 has 32 members: one size class (11-100 points), K' = 10 and 30 are measured.
        assert_eq!(e.k, [10.0, 30.0]);
        assert_eq!(e.a, [32.0]);
        assert_eq!(e.l.len(), R_GRID.len());
        assert_eq!(
            (
                size_class(10),
                size_class(11),
                size_class(100),
                size_class(101)
            ),
            (0, 1, 1, 2)
        );
        // A frontier of 8 routers holds the whole tree (5 routers): every recall reached.
        assert!(
            e.l.iter().flatten().all(|&l| (1.0..=8.0).contains(&l)),
            "{:?}",
            e.l
        );
        // The frontier needed grows with the recall and with K'.
        let last = e.l.len() - 1;
        assert!(e.l[last][1] >= e.l[0][0]);
        // Queries anywhere on the line (base midpoints) reach full recall within the tree too.
        let b = tree_curve(
            &base,
            &ix,
            &labels,
            &TreeOptions {
                queries: 20,
                seed: 3,
                l_max: 64,
                queries_from: TreeQueries::Base,
            },
        )
        .unwrap();
        assert!(b.l.iter().flatten().all(|&l| (1.0..=8.0).contains(&l)));
    }
}
