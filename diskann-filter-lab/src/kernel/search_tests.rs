/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Search-level tests: properties of every role combination on random graphs, tiny
//! adversarial graphs, exact paths and the reference executor.

use super::{
    config::{Kernel, Preset},
    entry::SampleIndex,
    search::{Counters, Index, Plan, Query, Route, Searcher},
};
use crate::{
    graph::{AuxKind, AuxNodes, Graph},
    labels::{LabelStore, LabelStoreBuilder},
    predicate::{PredicateSet, PredicateSetBuilder},
    starts::Starts,
    vectors::{Element, Matrix},
};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

/// A random dataset: `n` points in 6 dimensions (`u8`), a k-NN graph plus random edges, one
/// frozen auxiliary start (a copy of point 0), labels over three categories, predicates, and
/// a start table with per-label starts and a sample.
struct Fixture {
    base: Matrix<u8>,
    queries: Matrix<u8>,
    graph: Graph,
    labels: LabelStore,
    preds: PredicateSet,
    starts: Starts,
    sample: SampleIndex,
}

const DIM: usize = 6;

fn fixture(n: usize, seed: u64, aux: bool) -> Fixture {
    let mut rng = Lcg(seed);
    let mut data: Vec<u8> = (0..n * DIM).map(|_| rng.below(32) as u8).collect();
    // A few exact duplicates (with different labels below).
    for i in (0..n).step_by(17).skip(1) {
        let (src, dst) = (i - 1, i);
        for d in 0..DIM {
            data[dst * DIM + d] = data[src * DIM + d];
        }
    }
    let base = Matrix::from_slice(&data, n, DIM);
    let queries = Matrix::from_slice(
        &(0..40 * DIM)
            .map(|_| rng.below(32) as u8)
            .collect::<Vec<_>>(),
        40,
        DIM,
    );

    let n_aux = usize::from(aux);
    let auxn = if aux {
        AuxNodes::new(
            vec![AuxKind::Frozen],
            &Matrix::from_slice(base.row(0), 1, DIM),
        )
        .unwrap()
    } else {
        AuxNodes::new(Vec::new(), &Matrix::<u8>::zeros(0, DIM)).unwrap()
    };
    let r = 12;
    let mut graph = Graph::new(n, r, auxn).unwrap();
    let vec_of = |v: usize| -> &[u8] {
        if v < n {
            base.row(v)
        } else {
            base.row(0)
        }
    };
    for v in 0..n + n_aux {
        let mut cand: Vec<(f32, u32)> = (0..n)
            .filter(|&u| u != v)
            .map(|u| (u8::squared_l2(vec_of(v), base.row(u)), u as u32))
            .collect();
        cand.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut nbrs: Vec<u32> = cand.iter().take(8).map(|c| c.1).collect();
        // A cycle through all data points keeps the graph strongly connected.
        let next = ((v + 1) % n) as u32;
        if !nbrs.contains(&next) {
            nbrs.push(next);
        }
        while nbrs.len() < 10 {
            let u = rng.below(n as u32);
            if u as usize != v && !nbrs.contains(&u) {
                nbrs.push(u);
            }
        }
        if aux && v < n && v % 50 == 0 {
            nbrs.push(n as u32);
        }
        graph.set_neighbors(v as u32, &nbrs).unwrap();
    }
    graph
        .set_starts(vec![if aux { n as u32 } else { 0 }])
        .unwrap();

    let mut b = LabelStoreBuilder::new();
    for i in 0..n {
        let mut row = Vec::new();
        let a = format!("a={}", rng.below(6));
        let a2 = format!("a={}", rng.below(6));
        let bb = format!("b={}", rng.below(3));
        let c = format!("c={}", i % 2);
        row.push(("a", a.as_str()));
        if rng.below(3) == 0 {
            row.push(("a", a2.as_str()));
        }
        if rng.below(5) != 0 {
            row.push(("b", bb.as_str()));
        }
        row.push(("c", c.as_str()));
        b.push_point(row).unwrap();
    }
    let labels = b.finish().unwrap();

    let mut pb = PredicateSetBuilder::new(&labels);
    for q in 0..40 {
        let mut clauses: Vec<Vec<String>> = Vec::new();
        let na = 1 + rng.below(3);
        clauses.push((0..na).map(|_| format!("a={}", rng.below(6))).collect());
        if q % 3 != 0 {
            clauses.push(vec![format!("b={}", rng.below(3))]);
        }
        if q % 4 == 0 {
            clauses.push(vec![format!("c={}", rng.below(2))]);
        }
        if q == 7 {
            clauses.push(vec!["a=missing".to_owned()]);
        }
        pb.push(&clauses);
    }
    let preds = pb.finish();

    let mut starts = Starts::new(&graph);
    let pairs: Vec<(u32, u32)> = (0..labels.n_labels() as u32)
        .filter_map(|l| labels.postings(l).first().map(|&p| (l, p)))
        .collect();
    starts.set_label_starts(&labels, &pairs).unwrap();
    let sample: Vec<u32> = (0..n as u32).filter(|p| p % 3 != 1).collect();
    starts.set_sample(sample, None).unwrap();
    let sample = SampleIndex::build(&starts, &labels).unwrap();
    Fixture {
        base,
        queries,
        graph,
        labels,
        preds,
        starts,
        sample,
    }
}

impl Fixture {
    fn index(&self) -> Index<'_, u8> {
        Index {
            graph: &self.graph,
            vectors: self.graph.vectors(&self.base).unwrap(),
            labels: &self.labels,
            global: self.graph.starts(),
            label_starts: Some(&self.starts),
            sample: Some(&self.sample),
            label_view: false,
            view_postings: None,
            eps_min: 1.0,
        }
    }

    fn query(&self, q: usize) -> Query<'_, u8> {
        Query {
            vector: self.queries.row(q),
            cnf: self.preds.get(q),
            view: None,
        }
    }

    /// Exact top-k matches by (distance, id).
    fn truth(&self, q: usize, k: usize) -> Vec<(f32, u32)> {
        let cnf = self.preds.get(q);
        let mut all: Vec<(f32, u32)> = (0..self.base.nrows() as u32)
            .filter(|&p| cnf.matches(self.labels.row(p)))
            .map(|p| {
                (
                    u8::squared_l2(self.queries.row(q), self.base.row(p as usize)),
                    p,
                )
            })
            .collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        all.truncate(k);
        all
    }
}

fn search(
    f: &Fixture,
    st: &mut Searcher,
    plan: &Plan,
    q: usize,
    l: usize,
    k: usize,
    reference: Option<u64>,
) -> (Vec<(f32, u32)>, Counters) {
    let ix = f.index();
    let mut ids = vec![0u32; k];
    let mut d = vec![0f32; k];
    let ctr = st.search(plan, &ix, &f.query(q), l, &mut ids, &mut d, reference);
    let out: Vec<(f32, u32)> = d
        .iter()
        .zip(&ids)
        .take(ctr.n_out as usize)
        .map(|(&d, &i)| (d, i))
        .collect();
    assert!(
        ids[ctr.n_out as usize..].iter().all(|&i| i == u32::MAX),
        "padding"
    );
    (out, ctr)
}

/// Answers satisfy the predicate, are data points, are distinct, carry their true distances
/// and are sorted by (distance, id).
fn check_properties(f: &Fixture, q: usize, out: &[(f32, u32)], what: &str) {
    let cnf = f.preds.get(q);
    for &(d, id) in out {
        assert!(
            (id as usize) < f.base.nrows(),
            "{what}: {id} is not a data point"
        );
        assert!(
            cnf.matches(f.labels.row(id)),
            "{what}: {id} violates the predicate"
        );
        let true_d = u8::squared_l2(f.queries.row(q), f.base.row(id as usize));
        assert_eq!(d, true_d, "{what}: distance of {id}");
    }
    for w in out.windows(2) {
        assert!(
            (w[0].0, w[0].1) < (w[1].0, w[1].1),
            "{what}: not sorted by (distance, id) or not distinct: {w:?}"
        );
    }
}

fn configs() -> Vec<(String, Kernel)> {
    let mut out = Vec::new();
    for p in [
        "postfilter",
        "upstream-inline",
        "upstream-inline-adaptive(10, 4)",
        "upstream-multihop",
        "upstream-beta(0.5)",
        "bf(40)",
        "filt(10, 1)",
        "filt(10, 0)",
        "cp",
    ] {
        out.push((p.to_owned(), Preset::parse(p).unwrap().kernel()));
    }
    out.push((
        "sfmarket(b)".to_owned(),
        Preset::parse("sfmarket(b)").unwrap().kernel(),
    ));
    let overrides: &[&[(&str, &str)]] = &[
        &[("collector", "heap")],
        &[("collector", "heap"), ("terminator", "match_aware")],
        &[("laziness", "distance_first")],
        &[("laziness", "distance_first"), ("collector", "heap")],
        &[("laziness", "predicate_first"), ("admission", "strict")],
        &[("admission", "cutoff(1)"), ("scorer", "additive(5)")],
        &[("scorer", "normalized_additive(0.1)")],
        &[("scorer", "multiplicative(1)"), ("collector", "heap")],
        &[("scorer", "shifted(1)"), ("weights", "surprisal")],
        &[("scorer", "shell_adaptive(2)")],
        &[("scorer", "lexicographic"), ("collector", "heap")],
        &[("predicate", "mode2"), ("scorer", "bonus(0.3)")],
        &[
            ("predicate", "bitmap"),
            ("materializer", "dense"),
            ("scorer", "bonus(0.5)"),
        ],
        &[
            ("predicate", "bitmap"),
            ("materializer", "sparse"),
            ("admission", "strict"),
        ],
        &[("starts", "label(c*)+global"), ("collector", "heap")],
        &[
            ("starts", "label(all)"),
            ("start_admission", "obey"),
            ("admission", "strict"),
        ],
        &[("starts", "sample(3)+global"), ("sample_start", "recorded")],
        &[("starts", "aux+global")],
        &[("beam_width", "4"), ("collector", "all")],
        &[
            ("beam_width", "2"),
            ("admission", "strict"),
            ("expander", "two_hop(3)"),
        ],
        &[("exact", "intersect_then_scan(60)")],
        &[
            ("exact", "scan_smallest_clause(60)"),
            ("predicate", "mode2"),
        ],
        &[("padding", "cxx"), ("terminator", "budget(50)")],
        &[("admission", "projection(a)"), ("collector", "heap")],
        &[
            ("admission", "projection(b)"),
            ("laziness", "predicate_first"),
        ],
    ];
    for (i, edits) in overrides.iter().enumerate() {
        let mut k = Kernel::default();
        for (key, value) in edits.iter() {
            k.set(key, value).unwrap();
        }
        k.validate().unwrap();
        out.push((format!("override-{i}"), k));
    }
    out
}

#[test]
fn every_configuration_returns_valid_answers() {
    for aux in [false, true] {
        let f = fixture(300, 7 + u64::from(aux), aux);
        let ix = f.index();
        for (name, kernel) in configs() {
            let plan = Plan::new(&kernel, &ix).unwrap_or_else(|e| panic!("{name}: {e}"));
            let mut st = Searcher::for_index(&ix, &[&plan]);
            for q in 0..f.preds.len() {
                for l in [10, 25, 80] {
                    let (out, ctr) = search(&f, &mut st, &plan, q, l, 10, None);
                    let what = format!("{name} aux={aux} q={q} L={l}");
                    check_properties(&f, q, &out, &what);
                    assert!(out.len() <= 10);
                    if q == 7 {
                        assert_eq!(ctr.route, Route::Empty, "{what}");
                        assert!(out.is_empty());
                    }
                    if aux && ctr.route == Route::Graph {
                        assert!(ctr.n_aux_dist <= ctr.n_dist);
                    }
                }
            }
        }
    }
}

#[test]
fn lazy_and_predicate_first_evaluation_do_not_change_answers() {
    let f = fixture(300, 3, true);
    let ix = f.index();
    type Edits<'a> = &'a [(&'a str, &'a str)];
    let pairs: &[(Edits<'_>, Edits<'_>)] = &[
        (&[], &[("laziness", "distance_first")]),
        (
            &[("collector", "heap")],
            &[("collector", "heap"), ("laziness", "distance_first")],
        ),
        (
            &[("admission", "strict")],
            &[("admission", "strict"), ("laziness", "predicate_first")],
        ),
        (
            &[("scorer", "bonus(0.5)")],
            &[("scorer", "bonus(0.5)"), ("laziness", "distance_first")],
        ),
        (
            &[("scorer", "multiplicative(1)"), ("collector", "heap")],
            &[
                ("scorer", "multiplicative(1)"),
                ("collector", "heap"),
                ("laziness", "distance_first"),
            ],
        ),
    ];
    for (a, b) in pairs {
        let mk = |edits: &[(&str, &str)]| {
            let mut k = Kernel::default();
            for (key, value) in edits {
                k.set(key, value).unwrap();
            }
            Plan::new(&k, &ix).unwrap()
        };
        let (pa, pb) = (mk(a), mk(b));
        let mut st = Searcher::for_index(&ix, &[&pa]);
        let mut lazy_total = 0;
        for q in 0..f.preds.len() {
            for l in [10, 30] {
                let (oa, ca) = search(&f, &mut st, &pa, q, l, 10, None);
                let (ob, cb) = search(&f, &mut st, &pb, q, l, 10, None);
                assert_eq!(oa, ob, "{a:?} vs {b:?}, q={q} L={l}");
                assert_eq!(ca.n_expand, cb.n_expand, "same trajectory");
                lazy_total += cb.n_lazy;
                assert!(cb.n_dist <= ca.n_dist && cb.n_pred <= ca.n_pred + cb.n_out);
            }
        }
        if b.iter().any(|&(_, v)| v == "distance_first") {
            assert!(lazy_total > 0, "{b:?} never skipped");
        }
    }
}

#[test]
fn reference_mode_finds_the_exact_answers() {
    let f = fixture(300, 11, true);
    let ix = f.index();
    for p in ["postfilter", "upstream-inline", "upstream-beta(0.5)"] {
        let mut kernel = Preset::parse(p).unwrap().kernel();
        for collector in ["final_beam", "heap"] {
            if p == "postfilter" && collector == "heap" {
                continue;
            }
            kernel.set("collector", collector).unwrap();
            let plan = Plan::new(&kernel, &ix).unwrap();
            let mut st = Searcher::for_index(&ix, &[&plan]);
            for q in 0..f.preds.len() {
                let (out, ctr) = search(&f, &mut st, &plan, q, 10, 10, Some(1_000_000));
                let truth = f.truth(q, 10);
                assert_eq!(out, truth, "{p} {collector} q={q}");
                if ctr.route == Route::Graph {
                    // Every node was evaluated (one aux node included).
                    assert_eq!(ctr.n_dist, 301, "{p} q={q}");
                }
            }
        }
    }
}

#[test]
fn exact_paths_are_exact() {
    let f = fixture(300, 5, false);
    let ix = f.index();
    for exact in [
        "scan_smallest_clause(1000000)",
        "intersect_then_scan(1000000)",
    ] {
        for mode in ["mode1", "mode2"] {
            let mut k = Kernel::default();
            k.set("exact", exact).unwrap();
            k.set("predicate", mode).unwrap();
            let plan = Plan::new(&k, &ix).unwrap();
            let mut st = Searcher::for_index(&ix, &[&plan]);
            for q in 0..f.preds.len() {
                let (out, ctr) = search(&f, &mut st, &plan, q, 10, 7, None);
                assert_eq!(out, f.truth(q, 7), "{exact} {mode} q={q}");
                if q != 7 {
                    assert_eq!(ctr.route, Route::Exact);
                    assert!(ctr.n_exact_cand >= ctr.n_dist);
                }
            }
        }
    }
}

/// A line of points 0..6 at x = 0..6 (dim 1, `f32`); edges i → i+1 and i → i−1.
fn line(labels_of: &[&[&str]], extra_aux: bool) -> (Matrix<f32>, Graph, LabelStore) {
    let n = labels_of.len();
    let base = Matrix::from_slice(&(0..n).map(|i| i as f32).collect::<Vec<_>>(), n, 1);
    let aux = if extra_aux {
        AuxNodes::new(vec![AuxKind::Frozen], &Matrix::from_slice(&[2.5f32], 1, 1)).unwrap()
    } else {
        AuxNodes::new(Vec::new(), &Matrix::<f32>::zeros(0, 1)).unwrap()
    };
    let mut g = Graph::new(n, 4, aux).unwrap();
    for i in 0..n {
        let mut nb = Vec::new();
        if i + 1 < n {
            nb.push(i as u32 + 1);
        }
        if i > 0 {
            nb.push(i as u32 - 1);
        }
        g.set_neighbors(i as u32, &nb).unwrap();
    }
    g.set_starts(vec![0]).unwrap();
    let mut b = LabelStoreBuilder::new();
    for row in labels_of {
        b.push_point(row.iter().map(|&l| (&l[..1], l))).unwrap();
    }
    (base, g, b.finish().unwrap())
}

#[expect(clippy::too_many_arguments, reason = "test helper")]
fn run_line(
    base: &Matrix<f32>,
    g: &Graph,
    labels: &LabelStore,
    global: &[u32],
    kernel: &Kernel,
    pred: &[Vec<&str>],
    qx: f32,
    l: usize,
    k: usize,
) -> (Vec<u32>, Counters) {
    let mut pb = PredicateSetBuilder::new(labels);
    pb.push(pred);
    let preds = pb.finish();
    let ix = Index {
        graph: g,
        vectors: g.vectors(base).unwrap(),
        labels,
        global,
        label_starts: None,
        sample: None,
        label_view: false,
        view_postings: None,
        eps_min: 0.5,
    };
    let plan = Plan::new(kernel, &ix).unwrap();
    let mut st = Searcher::for_index(&ix, &[&plan]);
    let q = [qx];
    let mut ids = vec![0u32; k];
    let mut d = vec![0f32; k];
    let ctr = st.search(
        &plan,
        &ix,
        &Query {
            vector: &q,
            cnf: preds.get(0),
            view: None,
        },
        l,
        &mut ids,
        &mut d,
        None,
    );
    ids.truncate(ctr.n_out as usize);
    (ids, ctr)
}

fn kernel(edits: &[(&str, &str)]) -> Kernel {
    let mut k = Kernel::default();
    for (key, value) in edits {
        k.set(key, value).unwrap();
    }
    k.validate().unwrap();
    k
}

#[test]
fn rejected_starts_under_force_and_obey() {
    // Point 0 (the start) does not match; points 3 and 4 match.
    let rows: &[&[&str]] = &[&["x=0"], &["x=0"], &["x=0"], &["x=1"], &["x=1"]];
    let (base, g, labels) = line(rows, false);
    let pred = [vec!["x=1"]];
    let force = kernel(&[("admission", "strict"), ("expander", "two_hop")]);
    let (ids, ctr) = run_line(&base, &g, &labels, &[0], &force, &pred, 4.0, 4, 2);
    // The forced start navigates (its neighbour 1 is evaluated) but is never returned; no
    // match lies within two hops of it.
    assert!(ids.is_empty());
    assert_eq!((ctr.n_start, ctr.n_expand, ctr.n_dist), (1, 1, 2));
    let obey = kernel(&[
        ("admission", "strict"),
        ("start_admission", "obey"),
        ("laziness", "predicate_first"),
    ]);
    let (ids, ctr) = run_line(&base, &g, &labels, &[0], &obey, &pred, 4.0, 4, 2);
    assert!(ids.is_empty(), "the only start is rejected");
    assert_eq!((ctr.n_dist, ctr.n_expand), (0, 0));
    // A forced, rejected start still navigates under admission `all`.
    let all = kernel(&[]);
    let (ids, _) = run_line(&base, &g, &labels, &[0], &all, &pred, 4.0, 10, 2);
    assert_eq!(ids, [4, 3]);
}

#[test]
fn starts_beyond_the_capacity() {
    // Starts 0, 1, 4 for a query at 0 with L = 2.
    let rows: &[&[&str]] = &[&["x=1"], &["x=1"], &["x=1"], &["x=1"], &["x=1"]];
    let (base, g, labels) = line(rows, false);
    let pred = [vec!["x=1"]];
    // Capacity max(L, |starts|) = 3: start 4 enters, then 2 evicts it and is expanded too.
    let cxx = kernel(&[("padding", "cxx")]);
    let (ids, ctr) = run_line(&base, &g, &labels, &[0, 1, 4], &cxx, &pred, 0.0, 2, 2);
    assert_eq!(ids, [0, 1]);
    assert_eq!((ctr.n_expand, ctr.n_evict, ctr.n_dist), (3, 1, 5));
    // Capacity L = 2: start 4 is dropped on arrival (still visited), so is 2.
    let exact = kernel(&[("padding", "exact")]);
    let (ids, ctr) = run_line(&base, &g, &labels, &[0, 1, 4], &exact, &pred, 0.0, 2, 2);
    assert_eq!(ids, [0, 1]);
    assert_eq!((ctr.n_expand, ctr.n_evict, ctr.n_dist), (2, 0, 4));
}

#[test]
fn duplicate_starts_and_a_start_that_is_a_neighbour() {
    let rows: &[&[&str]] = &[&["x=1"], &["x=1"], &["x=1"], &["x=1"], &["x=1"]];
    let (base, g, labels) = line(rows, false);
    let pred = [vec!["x=1"]];
    let k = kernel(&[]);
    let (ids, ctr) = run_line(&base, &g, &labels, &[0, 1, 0, 1], &k, &pred, 0.0, 5, 5);
    assert_eq!(ids, [0, 1, 2, 3, 4]);
    assert_eq!(ctr.n_start, 2);
    // Every node is evaluated exactly once.
    assert_eq!(ctr.n_dist, 5);
}

#[test]
fn two_hop_crosses_non_matching_bridges() {
    // 0 matches, 1 does not, 2 and 3 match: only a second hop through 1 reaches them.
    let rows: &[&[&str]] = &[&["x=1"], &["x=0"], &["x=1"], &["x=1"], &["x=0"]];
    let (base, g, labels) = line(rows, false);
    let pred = [vec!["x=1"]];
    let one = kernel(&[("admission", "strict")]);
    let (ids, _) = run_line(&base, &g, &labels, &[0], &one, &pred, 3.0, 5, 3);
    assert_eq!(ids, [0], "one hop cannot cross the bridge");
    let two = kernel(&[("admission", "strict"), ("expander", "two_hop")]);
    let (ids, ctr) = run_line(&base, &g, &labels, &[0], &two, &pred, 3.0, 5, 3);
    assert_eq!(ids, [3, 2, 0]);
    assert!(ctr.n_two_hop >= 1);
}

#[test]
fn exact_duplicates_follow_the_upstream_tie_rule() {
    // Points 1 and 2 are both at x = 1 (2 is a copy of 1's vector with another label).
    let n = 4;
    let base = Matrix::from_slice(&[0.0f32, 1.0, 1.0, 3.0], n, 1);
    let mut g = Graph::new(
        n,
        3,
        AuxNodes::new(Vec::new(), &Matrix::<f32>::zeros(0, 1)).unwrap(),
    )
    .unwrap();
    g.set_neighbors(0, &[1, 2, 3]).unwrap();
    g.set_starts(vec![0]).unwrap();
    let mut b = LabelStoreBuilder::new();
    for row in [&["x=0"][..], &["x=0"], &["x=1"], &["x=1"]] {
        b.push_point(row.iter().map(|&l| (&l[..1], l))).unwrap();
    }
    let labels = b.finish().unwrap();
    // L = 2 with one start: capacity 3. The query at 1: start 0 (d=1), then 1 (d=0), 2 (d=0)
    // inserted before 1, then 3 (d=4) dropped... the beam is [2, 1, 0].
    let k = kernel(&[("laziness", "deferred")]);
    let (ids, _) = run_line(&base, &g, &labels, &[0], &k, &[vec!["x=1"]], 1.0, 2, 2);
    assert_eq!(ids, [2]);
    let ix = Index {
        graph: &g,
        vectors: g.vectors(&base).unwrap(),
        labels: &labels,
        global: &[0],
        label_starts: None,
        sample: None,
        label_view: false,
        view_postings: None,
        eps_min: 0.5,
    };
    let plan = Plan::new(&k, &ix).unwrap();
    let mut st = Searcher::for_index(&ix, &[&plan]);
    let mut pb = PredicateSetBuilder::new(&labels);
    pb.push(&[vec!["x=1"]]);
    let preds = pb.finish();
    let (mut ids, mut d) = (vec![0u32; 2], vec![0f32; 2]);
    st.search(
        &plan,
        &ix,
        &Query {
            vector: &[1.0],
            cnf: preds.get(0),
            view: None,
        },
        2,
        &mut ids,
        &mut d,
        None,
    );
    let order: Vec<u32> = st.beam().entries().map(|e| e.id).collect();
    assert_eq!(order, [2, 1, 0], "the later duplicate goes first");
}

#[test]
fn auxiliary_nodes_navigate_but_are_never_answers() {
    // A frozen node (id 5, x = 2.5) is the only start; it links to 2 and 3.
    let rows: &[&[&str]] = &[&["x=1"], &["x=1"], &["x=1"], &["x=1"], &["x=1"]];
    let (base, mut g, labels) = line(rows, true);
    g.set_neighbors(5, &[2, 3]).unwrap();
    g.set_neighbors(4, &[3, 5]).unwrap();
    g.set_starts(vec![5]).unwrap();
    let pred = [vec!["x=1"]];
    for edits in [
        &[][..],
        &[("collector", "heap")][..],
        &[("admission", "strict"), ("expander", "two_hop")][..],
        &[("scorer", "multiplicative(1)")][..],
    ] {
        let k = kernel(edits);
        let (ids, ctr) = run_line(&base, &g, &labels, &[5], &k, &pred, 2.5, 6, 6);
        assert_eq!(ids.len(), 5, "{edits:?}");
        assert!(!ids.contains(&5));
        assert_eq!(ctr.n_aux_dist, 1, "{edits:?}");
        assert_eq!(ctr.n_aux_expand, 1, "{edits:?}");
    }
}

#[test]
fn many_clauses_use_mode_2() {
    let mut b = LabelStoreBuilder::new();
    let names: Vec<String> = (0..70).map(|i| format!("c{i}=1")).collect();
    let cat = |n: &str| n[..n.find('=').unwrap()].to_owned();
    for p in 0..4 {
        let row: Vec<(String, &str)> = names
            .iter()
            .take(if p % 2 == 0 { 70 } else { 69 })
            .map(|n| (cat(n), n.as_str()))
            .collect();
        b.push_point(row.iter().map(|(c, n)| (c.as_str(), *n)))
            .unwrap();
    }
    let labels = b.finish().unwrap();
    let base = Matrix::from_slice(&[0.0f32, 1.0, 2.0, 3.0], 4, 1);
    let mut g = Graph::new(
        4,
        3,
        AuxNodes::new(Vec::new(), &Matrix::<f32>::zeros(0, 1)).unwrap(),
    )
    .unwrap();
    for i in 0..4u32 {
        let nb: Vec<u32> = (0..4).filter(|&j| j != i).collect();
        g.set_neighbors(i, &nb).unwrap();
    }
    let pred: Vec<Vec<&str>> = names.iter().map(|n| vec![n.as_str()]).collect();
    for edits in [
        &[][..],
        &[("scorer", "lexicographic"), ("collector", "heap")][..],
    ] {
        let k = kernel(edits);
        let (ids, _) = run_line(&base, &g, &labels, &[1], &k, &pred, 0.0, 4, 4);
        assert_eq!(ids, [0, 2], "{edits:?}");
    }
}

#[test]
fn adaptive_l_grows_the_beam_for_rare_matches() {
    let f = fixture(300, 9, false);
    let ix = f.index();
    let plan = Plan::new(
        &Preset::parse("upstream-inline-adaptive(10, 8)")
            .unwrap()
            .kernel(),
        &ix,
    )
    .unwrap();
    let fixed = Plan::new(&Preset::parse("upstream-inline").unwrap().kernel(), &ix).unwrap();
    let mut st = Searcher::for_index(&ix, &[&plan]);
    let mut grew = 0;
    for q in 0..f.preds.len() {
        let (_, c) = search(&f, &mut st, &plan, q, 10, 10, None);
        let (_, c0) = search(&f, &mut st, &fixed, q, 10, 10, None);
        if c0.route == Route::Graph {
            assert_eq!(c0.final_l, 11);
        }
        if c.route == Route::Graph && c.final_l > 11 {
            grew += 1;
        }
    }
    assert!(grew > 0);
}

#[test]
fn fast_paths_agree_with_the_configurable_loop() {
    for aux in [false, true] {
        let f = fixture(300, 21 + u64::from(aux), aux);
        let ix = f.index();
        for (preset, fast) in [
            ("postfilter", super::search::FastPath::Postfilter),
            ("upstream-inline", super::search::FastPath::Inline),
        ] {
            let plan = Plan::new(&Preset::parse(preset).unwrap().kernel(), &ix).unwrap();
            assert_eq!(plan.fast_path(), fast);
            let generic = plan.clone().generic();
            let mut st = Searcher::for_index(&ix, &[&plan]);
            for q in 0..f.preds.len() {
                for l in [10, 40] {
                    let (a, ca) = search(&f, &mut st, &plan, q, l, 10, None);
                    let (b, cb) = search(&f, &mut st, &generic, q, l, 10, None);
                    assert_eq!(a, b, "{preset} q={q} L={l}");
                    assert_eq!(ca, cb, "{preset} counters q={q} L={l}");
                }
            }
        }
    }
}
