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
    bitsets::LabelBitsets,
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
/// frozen auxiliary start (a copy of point 0), labels over three categories, predicates, a
/// start table with per-label starts and a sample, and static bitsets of every label.
struct Fixture {
    base: Matrix<u8>,
    queries: Matrix<u8>,
    graph: Graph,
    labels: LabelStore,
    preds: PredicateSet,
    starts: Starts,
    sample: SampleIndex,
    bitsets: LabelBitsets,
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
    let bitsets = LabelBitsets::build(&labels, 1);
    Fixture {
        base,
        queries,
        graph,
        labels,
        preds,
        starts,
        sample,
        bitsets,
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
            bitsets: Some(&self.bitsets),
            costs: None,
            sketches: None,
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
        &[
            ("predicate", "static_bits(60)"),
            ("scorer", "multiplicative(1)"),
            ("collector", "heap"),
        ],
        &[
            ("predicate", "static_bits(1)"),
            ("admission", "cutoff(1)"),
            ("scorer", "additive(5)"),
        ],
        &[
            ("predicate", "static_bits(100)"),
            ("admission", "projection(b)"),
            ("collector", "heap"),
        ],
        &[
            ("predicate", "static_bits(60)"),
            ("exact", "scan_smallest_clause(60)"),
        ],
        &[
            ("predicate", "bitmap"),
            ("materializer", "dense_static"),
            ("collector", "heap"),
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
fn static_bits_do_not_change_answers_or_counters() {
    let f = fixture(300, 13, true);
    let ix = f.index();
    let edits: &[&[(&str, &str)]] = &[
        &[],
        &[("collector", "heap"), ("laziness", "distance_first")],
        &[("admission", "strict"), ("laziness", "predicate_first")],
        &[("admission", "cutoff(1)"), ("scorer", "additive(5)")],
        &[("scorer", "shifted(1)"), ("weights", "surprisal")],
        &[("admission", "projection(a)"), ("collector", "heap")],
        &[("admission", "strict"), ("expander", "two_hop")],
        &[("starts", "sample(3)+global"), ("collector", "all")],
        &[("exact", "scan_smallest_clause(70)")],
    ];
    let mut static_preds = 0;
    for e in edits {
        let mut base = Kernel::default();
        for (key, value) in e.iter() {
            base.set(key, value).unwrap();
        }
        let pa = Plan::new(&base, &ix).unwrap();
        let mut st = Searcher::for_index(&ix, &[&pa]);
        for theta in ["1", "60", "100", "1000"] {
            let mut k = base.clone();
            k.set("predicate", &format!("static_bits({theta})"))
                .unwrap();
            let pb = Plan::new(&k, &ix).unwrap();
            for q in 0..f.preds.len() {
                for l in [10, 30] {
                    let (oa, mut ca) = search(&f, &mut st, &pa, q, l, 10, None);
                    let (ob, mut cb) = search(&f, &mut st, &pb, q, l, 10, None);
                    // The residual shape counts labels by bitset, i.e. follows the split's θ.
                    for c in [&mut ca, &mut cb] {
                        [c.res_words, c.res_rows, c.c1_words, c.c1_rows] = [0; 4];
                    }
                    assert_eq!((oa, ca), (ob, cb), "{e:?} static_bits({theta}) q={q} L={l}");
                    static_preds += cb.n_pred;
                }
            }
        }
    }
    assert!(static_preds > 0);
    // Bitsets built above the requested threshold are refused.
    let high = LabelBitsets::build(&f.labels, 100);
    let ix_high = Index {
        bitsets: Some(&high),
        costs: None,
        sketches: None,
        ..f.index()
    };
    let mut k = Kernel::default();
    k.set("predicate", "static_bits(60)").unwrap();
    assert!(Plan::new(&k, &ix_high).is_err());
    assert!(Plan::new(
        &k,
        &Index {
            bitsets: None,
            costs: None,
            sketches: None,
            ..f.index()
        }
    )
    .is_err());
    k.set("predicate", "static_bits(100)").unwrap();
    assert!(Plan::new(&k, &ix_high).is_ok());
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
        for mode in ["mode1", "mode2", "static_bits(60)"] {
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
        bitsets: None,
        costs: None,
        sketches: None,
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
        bitsets: None,
        costs: None,
        sketches: None,
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

// ------------------------------------------------- exact path v2, laziness, crossovers (B6)

fn search_ix(
    f: &Fixture,
    ix: &Index<'_, u8>,
    st: &mut Searcher,
    plan: &Plan,
    q: usize,
    l: usize,
    k: usize,
) -> (Vec<(f32, u32)>, Counters) {
    let mut ids = vec![0u32; k];
    let mut d = vec![0f32; k];
    let ctr = st.search(plan, ix, &f.query(q), l, &mut ids, &mut d, None);
    let out: Vec<(f32, u32)> = d
        .iter()
        .zip(&ids)
        .take(ctr.n_out as usize)
        .map(|(&d, &i)| (d, i))
        .collect();
    (out, ctr)
}

/// The k-hp-ad policy of the label-blind graph with role overrides.
fn khpad(edits: &[(&str, &str)]) -> Kernel {
    let mut k = kernel(&[
        ("starts", "label(c*)+global"),
        ("collector", "heap"),
        ("terminator", "adaptive_l(20, 4)"),
        ("laziness", "distance_first"),
    ]);
    for (key, value) in edits {
        k.set(key, value).unwrap();
    }
    k.validate().unwrap();
    k
}

#[test]
fn planned_exact_plans_are_exact() {
    use super::cost::{ExactPlan, UnitCosts};
    let f = fixture(300, 5, false);
    let costs = UnitCosts::default();
    let sketches = crate::sketches::LabelSketches::build(&f.labels, 16, 3, 1);
    for with_bits in [false, true] {
        let ix = Index {
            bitsets: with_bits.then_some(&f.bitsets),
            costs: Some(&costs),
            sketches: Some(&sketches),
            ..f.index()
        };
        for plans in ["scan", "df", "gallop", "bitmap", "scan+df+gallop+bitmap"] {
            for estimates in ["bounds", "sketch"] {
                let k = kernel(&[
                    ("exact", "planned(18446744073709551615)"),
                    ("exact_plans", plans),
                    ("estimates", estimates),
                ]);
                let plan = Plan::new(&k, &ix).unwrap();
                let mut st = Searcher::for_index(&ix, &[&plan]);
                for q in 0..f.preds.len() {
                    for kk in [1, 7, 10] {
                        let (out, ctr) = search_ix(&f, &ix, &mut st, &plan, q, 10, kk);
                        let what = format!("{plans} {estimates} bits={with_bits} q={q} K={kk}");
                        assert_eq!(out, f.truth(q, kk), "{what}");
                        if q == 7 {
                            continue;
                        }
                        assert_eq!(ctr.route, Route::Exact, "{what}");
                        if !plans.contains('+') {
                            assert_eq!(ctr.plan, ExactPlan::parse(plans).unwrap() as u64);
                        }
                        if estimates == "sketch" && f.preds.get(q).n_clauses() > 1 {
                            assert!(ctr.n_probe > 0, "{what}: no probe");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn residual_checks_keep_answers_routes_and_estimates() {
    use super::cost::UnitCosts;
    let f = fixture(300, 5, false);
    let costs = UnitCosts::default();
    let sketches = crate::sketches::LabelSketches::build(&f.labels, 16, 3, 1);
    for with_bits in [false, true] {
        let ix = Index {
            bitsets: with_bits.then_some(&f.bitsets),
            costs: Some(&costs),
            sketches: Some(&sketches),
            ..f.index()
        };
        let exact_v2 = [
            ("exact", "planned(18446744073709551615)"),
            ("estimates", "sketch"),
        ];
        let s2 = [
            ("laziness", "heap_bound"),
            ("exact", "planned"),
            ("crossover", "cost"),
            ("estimates", "sketch"),
            ("terminator", "posting_l(16)"),
        ];
        for (name, edits) in [("exact-v2", &exact_v2[..]), ("s2", &s2[..])] {
            let plans: Vec<Plan> = ["residual", "full"]
                .iter()
                .map(|m| {
                    let mut e = edits.to_vec();
                    e.push(("exact_checks", m));
                    Plan::new(&khpad(&e), &ix).unwrap()
                })
                .collect();
            let mut st = Searcher::for_index(&ix, &[&plans[0], &plans[1]]);
            for q in 0..f.preds.len() {
                for kk in [1, 7, 10] {
                    let (a, ca) = search_ix(&f, &ix, &mut st, &plans[0], q, 10, kk);
                    let (b, cb) = search_ix(&f, &ix, &mut st, &plans[1], q, 10, kk);
                    let what = format!("{name} bits={with_bits} q={q} K={kk}");
                    assert_eq!(a, b, "{what}");
                    assert_eq!(ca, cb, "{what}: counters");
                    if ca.route == Route::Exact {
                        assert_eq!(a, f.truth(q, kk), "{what}");
                    }
                }
            }
        }
    }
}

#[test]
fn seeded_exact_paths_exhaust_every_candidate() {
    use super::{
        collect::{Answers, Collector},
        cost::ExactPlan,
        eval::{Compiled, EvalCtx, Weights},
        exact::{run_planned, ExactEval},
        materialize::Materialized,
    };
    let f = fixture(300, 9, false);
    let vectors = f.graph.vectors(&f.base).unwrap();
    let mut bits = vec![0u64; f.labels.n_labels()];
    let mut q = Compiled::new();
    let mut mat = Materialized::new(300, true);
    let mut answers = Answers::new();
    for qi in 0..f.preds.len() {
        q.compile(&f.labels, f.preds.get(qi), Weights::Unit, None, &mut bits);
        if q.is_empty() {
            q.clear_bits(&mut bits);
            continue;
        }
        q.split_static(&f.labels, &f.bitsets, 1, false);
        let all = f.truth(qi, 300);
        for k in [1usize, 5, 10] {
            // Seeds: the farthest matches (valid but wrong answers), and a mix with some of the
            // true answers (which the exact path finds again and must not duplicate).
            let far: Vec<(f32, u32)> = all.iter().rev().take(k).copied().collect();
            let mixed: Vec<(f32, u32)> = all
                .iter()
                .step_by(2)
                .take(k)
                .chain(all.iter().rev().take(k))
                .copied()
                .collect();
            for seed in [&far, &mixed] {
                for plan in ExactPlan::ALL {
                    for eval in [ExactEval::RowScan, ExactEval::StaticBits] {
                        let mut cx = EvalCtx::new(&f.labels, &q, &bits, &[], &[]);
                        if eval == ExactEval::StaticBits {
                            cx = cx.with_static(&f.bitsets);
                        }
                        answers.reset(Collector::Heap, k, k);
                        for &(d, id) in seed.iter() {
                            answers.offer_unique(d, id);
                        }
                        let mut ctr = Counters::default();
                        run_planned(
                            plan,
                            eval,
                            &cx,
                            &vectors,
                            300,
                            f.queries.row(qi),
                            k,
                            &mut mat,
                            &mut answers,
                            true,
                            &mut ctr,
                        );
                        let mut ids = vec![0u32; k];
                        let mut ds = vec![0f32; k];
                        let n = answers.finish(&mut ids, &mut ds);
                        let got: Vec<(f32, u32)> = ds[..n]
                            .iter()
                            .copied()
                            .zip(ids[..n].iter().copied())
                            .collect();
                        assert_eq!(got, f.truth(qi, k), "q={qi} k={k} {plan:?} {eval:?}");
                        assert!(ctr.n_exact_cand as usize >= all.len().min(1));
                    }
                }
            }
        }
        q.clear_bits(&mut bits);
    }
}

#[test]
fn heap_bound_laziness_keeps_answers_and_skips_checks() {
    let f = fixture(300, 17, true);
    let ix = f.index();
    for terminator in ["adaptive_l(20, 4)", "exhaust"] {
        let base = Plan::new(&khpad(&[("terminator", terminator)]), &ix).unwrap();
        let lazy = Plan::new(
            &khpad(&[("terminator", terminator), ("laziness", "heap_bound")]),
            &ix,
        )
        .unwrap();
        let eager = Plan::new(
            &khpad(&[("terminator", terminator), ("laziness", "eager")]),
            &ix,
        )
        .unwrap();
        let mut st = Searcher::for_index(&ix, &[&base]);
        let (mut skipped, mut saved) = (0, 0i64);
        for q in 0..f.preds.len() {
            for l in [10, 40] {
                for k in [3, 10] {
                    let (a, ca) = search(&f, &mut st, &base, q, l, k, None);
                    let (b, cb) = search(&f, &mut st, &lazy, q, l, k, None);
                    let (c, cc) = search(&f, &mut st, &eager, q, l, k, None);
                    let what = format!("{terminator} q={q} L={l} K={k}");
                    assert_eq!(a, b, "{what}");
                    assert_eq!(a, c, "{what}");
                    assert_eq!(ca.n_expand, cb.n_expand, "{what}: same trajectory");
                    assert_eq!(ca.n_dist, cb.n_dist, "{what}");
                    assert!(cb.n_pred <= ca.n_pred && cb.n_pred <= cc.n_pred, "{what}");
                    skipped += cb.n_lazy;
                    saved += ca.n_pred as i64 - cb.n_pred as i64;
                }
            }
        }
        assert!(skipped > 0 && saved > 0, "{terminator}: nothing skipped");
    }
}

/// Costs whose nominal graph search costs only its starts: the cost crossovers try the graph
/// unless the exact path is cheaper than evaluating the starts.
fn free_graph_costs() -> super::cost::UnitCosts {
    super::cost::UnitCosts {
        graph_alpha: 0.0,
        graph_beta: 0.0,
        ..super::cost::UnitCosts::default()
    }
}

#[test]
fn crossovers_give_valid_answers_and_exact_fallbacks() {
    use super::cost::UnitCosts;
    let f = fixture(300, 23, true);
    let sketches = crate::sketches::LabelSketches::build(&f.labels, 12, 5, 1);
    let calibrated = UnitCosts::default();
    // A nominal graph search of a few starts, exact checks costing 0.1 ms, and a graph spend
    // of 1 ms per distance: the graph is tried and overspends at the first checkpoint.
    let overspend = UnitCosts {
        graph_dist_ns: 1e6,
        pred_ns: vec![1e5; 6],
        ..free_graph_costs()
    };
    let mut fallbacks = [0usize; 5];
    for costs in [&calibrated, &overspend] {
        let ix = Index {
            costs: Some(costs),
            sketches: Some(&sketches),
            ..f.index()
        };
        let configs: Vec<Kernel> = vec![
            khpad(&[("exact", "planned(60)"), ("laziness", "heap_bound")]),
            khpad(&[("exact", "planned"), ("crossover", "cost")]),
            khpad(&[
                ("exact", "planned"),
                ("crossover", "cost"),
                ("estimates", "sketch"),
                ("terminator", "posting_l(4)"),
                ("laziness", "heap_bound"),
            ]),
            khpad(&[
                ("exact", "planned"),
                ("crossover", "guarded"),
                ("estimates", "sketch"),
                ("terminator", "posting_l(4)"),
                ("laziness", "heap_bound"),
            ]),
            khpad(&[
                ("exact", "planned"),
                ("crossover", "guarded"),
                ("estimates", "sketch"),
                ("fallback", "budget+witness"),
                ("exact_plans", "scan+df"),
            ]),
        ];
        for k in configs {
            let plan = Plan::new(&k, &ix).unwrap();
            let mut st = Searcher::for_index(&ix, &[&plan]);
            for q in 0..f.preds.len() {
                for (l, kk) in [(10, 10), (40, 5)] {
                    let (out, ctr) = search_ix(&f, &ix, &mut st, &plan, q, l, kk);
                    let what = format!("{} q={q} L={l} K={kk}", k.to_json());
                    check_properties(&f, q, &out, &what);
                    if matches!(ctr.route, Route::Exact | Route::Fallback) {
                        assert_eq!(out, f.truth(q, kk), "{what}");
                    }
                    if ctr.route == Route::Fallback {
                        assert!((1..=4).contains(&ctr.fallback), "{what}");
                        fallbacks[ctr.fallback as usize] += 1;
                        assert!(ctr.n_exact_cand > 0 && ctr.graph_ns > 0, "{what}");
                    } else {
                        assert_eq!(ctr.fallback, 0, "{what}");
                    }
                }
            }
        }
    }
    assert!(fallbacks[2] > 0, "the spend cap never fired: {fallbacks:?}");
}

#[test]
fn complete_witnesses_make_the_guarded_trial_exact() {
    // With sketches holding every label completely, the probe enumerates all matches; at
    // nominal termination any missed answer is a held-out witness, so every answer is exact.
    let f = fixture(300, 29, false);
    let sketches = crate::sketches::LabelSketches::build(&f.labels, 1000, 1, 1);
    let costs = free_graph_costs();
    let ix = Index {
        costs: Some(&costs),
        sketches: Some(&sketches),
        ..f.index()
    };
    let k = khpad(&[
        ("exact", "planned"),
        ("crossover", "guarded"),
        ("estimates", "sketch"),
        ("fallback", "witness"),
        ("laziness", "heap_bound"),
    ]);
    let plan = Plan::new(&k, &ix).unwrap();
    let mut st = Searcher::for_index(&ix, &[&plan]);
    let mut witnessed = 0;
    for q in 0..f.preds.len() {
        for (l, kk) in [(10, 10), (10, 3), (30, 10)] {
            let (out, ctr) = search_ix(&f, &ix, &mut st, &plan, q, l, kk);
            assert_eq!(out, f.truth(q, kk), "q={q} L={l} K={kk} {ctr:?}");
            if ctr.fallback == super::planner::Fallback::Witness as u64 {
                witnessed += 1;
            }
            if q != 7 {
                let truth = (0..300u32)
                    .filter(|&p| f.preds.get(q).matches(f.labels.row(p)))
                    .count();
                assert_eq!(ctr.m_hat as usize, truth, "exact estimates q={q}");
            }
        }
    }
    assert!(witnessed > 0, "no discovery deficit was caught");
}

#[test]
fn posting_l_fixes_the_list_size_before_navigation() {
    use super::cost::l_multiplier;
    let f = fixture(300, 31, false);
    let sketches = crate::sketches::LabelSketches::build(&f.labels, 1000, 1, 1);
    let ix = Index {
        sketches: Some(&sketches),
        ..f.index()
    };
    let k = khpad(&[
        ("estimates", "sketch"),
        ("terminator", "posting_l(4)"),
        ("laziness", "heap_bound"),
    ]);
    let plan = Plan::new(&k, &ix).unwrap();
    let mut st = Searcher::for_index(&ix, &[&plan]);
    for q in 0..f.preds.len() {
        if q == 7 {
            continue;
        }
        let m = (0..300u32)
            .filter(|&p| f.preds.get(q).matches(f.labels.row(p)))
            .count();
        let want = (20.0 * l_multiplier(m as f64 / 300.0, 4.0)).round() as u64;
        let (out, ctr) = search_ix(&f, &ix, &mut st, &plan, q, 20, 10);
        check_properties(&f, q, &out, "posting_l");
        assert_eq!(ctr.route, Route::Graph);
        // The expandable prefix is L plus the starts (upstream padding).
        assert_eq!(ctr.final_l, want + ctr.n_start, "q={q}: m={m}");
    }
}

#[test]
fn the_postfilter_route_returns_the_heap_answers_of_a_global_search() {
    use super::cost::kth_match_rank;
    // Every graph-routed query takes the post-filter route (`always`); exact routes only when
    // the exact path is cheaper than the starts. Its answers are those of a heap collector
    // over the same (unfiltered) traversal from the global start, with fewer checks.
    let f = fixture(300, 37, true);
    let sketches = crate::sketches::LabelSketches::build(&f.labels, 1000, 1, 1);
    // An expensive exact path: every query is graph-routed.
    let costs = super::cost::UnitCosts {
        graph_alpha: 2.0,
        graph_beta: 10.0,
        pred_ns: vec![1e9; 6],
        dist_ns: vec![1e9; 6],
        ..super::cost::UnitCosts::default()
    };
    let mut logged = 0;
    for with_bits in [false, true] {
        let ix = Index {
            bitsets: with_bits.then_some(&f.bitsets),
            costs: Some(&costs),
            sketches: Some(&sketches),
            ..f.index()
        };
        let route = khpad(&[
            ("exact", "planned"),
            ("crossover", "cost"),
            ("estimates", "sketch"),
            ("terminator", "posting_l(4)"),
            ("laziness", "heap_bound"),
            ("postfilter", "always"),
        ]);
        let heap = kernel(&[
            ("collector", "heap"),
            ("laziness", "heap_bound"),
            ("starts", "global"),
        ]);
        let (rp, hp) = (
            Plan::new(&route, &ix).unwrap(),
            Plan::new(&heap, &ix).unwrap(),
        );
        let mut st = Searcher::for_index(&ix, &[&rp, &hp]);
        let mut routed = 0;
        for q in 0..f.preds.len() {
            for kk in [1, 5, 10, 40] {
                let (out, ctr) = search_ix(&f, &ix, &mut st, &rp, q, 8, kk);
                let what = format!("bits={with_bits} q={q} K={kk}");
                check_properties(&f, q, &out, &what);
                if ctr.route != Route::Postfilter {
                    continue;
                }
                routed += 1;
                // The list size evaluates the K-th match's rank at the estimated rate, and is at
                // least the planned list size of the graph search.
                let rate = ctr.m_hat as f64 / 300.0;
                let l_plan = (8.0 * super::cost::l_multiplier(rate, 4.0)).round() as u64;
                let rank = kth_match_rank(kk, rate, super::cost::Z);
                let need = costs.graph_l_for(rank).ceil() as u64;
                let want = need.max(l_plan).min(300);
                assert_eq!(ctr.final_l, want + ctr.n_start, "{what}");
                let (base, bctr) = search_ix(&f, &ix, &mut st, &hp, q, want as usize, kk);
                assert_eq!(out, base, "{what}");
                assert_eq!(
                    (ctr.n_dist, ctr.n_expand),
                    (bctr.n_dist, bctr.n_expand),
                    "{what}"
                );
                if ctr.fallback == super::planner::Fallback::Log as u64 {
                    logged += 1;
                } else {
                    assert_eq!(ctr.fallback, 0, "{what}");
                }
            }
        }
        assert!(routed > 0, "no query took the post-filter route");
    }
    assert!(logged > 0, "the log of evaluated nodes was never needed");
}

#[test]
fn the_cost_rule_prefers_postfilters_for_frequent_matches() {
    use super::cost::UnitCosts;
    // Expensive checks: a post-filter search wins when the K-th match is near (frequent
    // matches) and loses when its list size would have to grow a lot.
    let f = fixture(300, 41, false);
    let sketches = crate::sketches::LabelSketches::build(&f.labels, 1000, 1, 1);
    let costs = UnitCosts {
        graph_alpha: 1.0,
        graph_beta: 10.0,
        graph_dist_ns: 10.0,
        graph_pred_ns: 1000.0,
        pred_ns: vec![1e9; 6],
        dist_ns: vec![1e9; 6],
        ..UnitCosts::default()
    };
    let ix = Index {
        costs: Some(&costs),
        sketches: Some(&sketches),
        ..f.index()
    };
    // The guarded crossover routes the same way (its trial guards only inline graph searches).
    for crossover in ["cost", "guarded"] {
        let k = khpad(&[
            ("exact", "planned"),
            ("crossover", crossover),
            ("estimates", "sketch"),
            ("terminator", "posting_l(4)"),
            ("laziness", "heap_bound"),
            ("postfilter", "cost"),
        ]);
        let plan = Plan::new(&k, &ix).unwrap();
        let mut st = Searcher::for_index(&ix, &[&plan]);
        let (mut pf, mut inline) = (0, 0);
        for q in 0..f.preds.len() {
            let (out, ctr) = search_ix(&f, &ix, &mut st, &plan, q, 10, 3);
            check_properties(&f, q, &out, "cost rule");
            let m = ctr.m_hat as f64;
            if m == 0.0 {
                continue;
            }
            // The post-filter alternative: a list size that evaluates the K-th match, about K / r
            // checks, the global starts. The recorded graph estimate is the cheaper search.
            let rate = m / 300.0;
            let l_plan = (10.0 * super::cost::l_multiplier(rate, 4.0)).round();
            let rank = super::cost::kth_match_rank(3, rate, super::cost::Z);
            let l_pf = costs.graph_l_for(rank).ceil().max(l_plan).min(300.0);
            let g_pf = costs.postfilter_cost(l_pf, 3, rate) + ix.global.len() as f64 * 10.0;
            match ctr.route {
                Route::Postfilter => {
                    pf += 1;
                    assert_eq!(ctr.est_graph_ns, g_pf.round() as u64, "q={q}");
                    assert_eq!(ctr.final_l, l_pf as u64 + ctr.n_start, "q={q}");
                }
                // A guarded inline trial may fall back to the exact path.
                Route::Graph | Route::Fallback
                    if ctr.route == Route::Graph || crossover == "guarded" =>
                {
                    inline += 1;
                    assert!(
                        (ctr.est_graph_ns as f64) < g_pf,
                        "q={q}: {} >= {g_pf}",
                        ctr.est_graph_ns
                    );
                }
                r => panic!("q={q}: unexpected route {r:?}"),
            }
        }
        assert!(
            pf > 0 && inline > 0,
            "{crossover}: both routes are used: {pf} / {inline}"
        );
    }
}

#[test]
fn hopeless_graph_searches_take_the_exact_path() {
    use super::cost::UnitCosts;
    // A free graph search evaluating L nodes (D(L) = L) and a prohibitive exact path: the cost
    // crossover always prefers the graph, except where the gate finds the planned search
    // hopeless, i.e. at the upper bound of the match count (exact here: complete sketches) it
    // meets fewer than one match, L·M/n < 1.
    let f = fixture(300, 47, false);
    let sketches = crate::sketches::LabelSketches::build(&f.labels, 1000, 1, 1);
    let costs = UnitCosts {
        graph_alpha: 1.0,
        graph_beta: 0.0,
        graph_dist_ns: 1.0,
        graph_pred_ns: 1.0,
        pred_ns: vec![1e9; 6],
        dist_ns: vec![1e9; 6],
        ..UnitCosts::default()
    };
    let ix = Index {
        costs: Some(&costs),
        sketches: Some(&sketches),
        ..f.index()
    };
    let s2 = [
        ("exact", "planned"),
        ("crossover", "cost"),
        ("estimates", "sketch"),
        ("terminator", "posting_l(4)"),
        ("laziness", "heap_bound"),
    ];
    let mut off = s2.to_vec();
    off.push(("hopeless_gate", "off"));
    // The graph-only crossover keeps hopeless searches on the graph; with the post-filter route
    // the gate does not force it (its list would reach n): gate on and off decide alike.
    let mut graph_only = s2.to_vec();
    graph_only[1] = ("crossover", "graph");
    let mut graph_pf = graph_only.clone();
    graph_pf.push(("postfilter", "cost"));
    let mut graph_pf_off = graph_pf.clone();
    graph_pf_off.push(("hopeless_gate", "off"));
    let (gated, ungated) = (khpad(&s2), khpad(&off));
    assert_eq!(gated.to_json().get("hopeless_gate"), None);
    assert_eq!(ungated.to_json()["hopeless_gate"], "off");
    let (pg, pu, po) = (
        Plan::new(&gated, &ix).unwrap(),
        Plan::new(&ungated, &ix).unwrap(),
        Plan::new(&khpad(&graph_only), &ix).unwrap(),
    );
    let (ppf, ppf_off) = (
        Plan::new(&khpad(&graph_pf), &ix).unwrap(),
        Plan::new(&khpad(&graph_pf_off), &ix).unwrap(),
    );
    let mut st = Searcher::for_index(&ix, &[&pg, &pu, &po, &ppf, &ppf_off]);
    let (mut exact, mut graph) = (0, 0);
    for q in 0..f.preds.len() {
        let m = (0..300u32)
            .filter(|&p| f.preds.get(q).matches(f.labels.row(p)))
            .count() as f64;
        for l0 in [5, 20] {
            let l = (l0 as f64 * super::cost::l_multiplier(m / 300.0, 4.0)).round();
            let (out, ctr) = search_ix(&f, &ix, &mut st, &pg, q, l0, 3);
            check_properties(&f, q, &out, "gated");
            let (_, cu) = search_ix(&f, &ix, &mut st, &pu, q, l0, 3);
            let (oo, co) = search_ix(&f, &ix, &mut st, &po, q, l0, 3);
            check_properties(&f, q, &oo, "graph only");
            let (op, cp) = search_ix(&f, &ix, &mut st, &ppf, q, l0, 3);
            let (op_off, cp_off) = search_ix(&f, &ix, &mut st, &ppf_off, q, l0, 3);
            assert_eq!(
                (cp.route, cp.final_l, &op),
                (cp_off.route, cp_off.final_l, &op_off),
                "q={q} L0={l0}: the gate under the graph crossover"
            );
            if q == 7 {
                continue;
            }
            assert_eq!(co.route, Route::Graph, "q={q} L0={l0}: graph only");
            assert_eq!(
                cu.route,
                Route::Graph,
                "q={q} L0={l0}: the ungated crossover"
            );
            assert_eq!(
                ctr.m_hi as f64, m,
                "q={q}: complete sketches give exact counts"
            );
            if l * m / 300.0 < 1.0 {
                exact += 1;
                assert_eq!(ctr.route, Route::Exact, "q={q} L0={l0}: m={m}, L={l}");
                assert_eq!(out, f.truth(q, 3), "q={q} L0={l0}");
                assert!(
                    ctr.est_graph_ns < ctr.est_exact_ns,
                    "q={q}: the graph was cheaper"
                );
            } else {
                graph += 1;
                assert_eq!(ctr.route, Route::Graph, "q={q} L0={l0}: m={m}, L={l}");
            }
        }
    }
    assert!(exact > 0 && graph > 0, "both outcomes: {exact} / {graph}");
}

#[test]
fn the_effort_curve_plans_the_list_size_from_the_match_rate() {
    use super::cost::UnitCosts;
    // A curve over K' = 10 / 100 neighbours and recalls 0.5 / 0.9; an expensive exact path.
    let f = fixture(300, 43, false);
    let sketches = crate::sketches::LabelSketches::build(&f.labels, 1000, 1, 1);
    let costs = UnitCosts {
        effort_r: vec![0.5, 0.9],
        effort_k: vec![10.0, 100.0],
        effort_l: vec![vec![10.0, 40.0], vec![40.0, 160.0]],
        pred_ns: vec![1e9; 6],
        dist_ns: vec![1e9; 6],
        ..UnitCosts::default()
    };
    // Interpolation: log-linear in R, log-log in K'; clamped in R, proportional beyond K'.
    let near = |r: f64, kp: f64, want: f64| {
        let got = costs.effort_list(r, kp).unwrap();
        assert!(
            (got - want).abs() < 1e-6 * want,
            "R={r} K'={kp}: {got} vs {want}"
        );
    };
    near(0.5, 10.0, 10.0);
    near(0.7, 10.0, 20.0);
    near(0.9, 31.622_776_601_683_79, 80.0);
    near(0.99, 400.0, 640.0);
    near(0.3, 5.0, 10.0);
    assert_eq!(UnitCosts::default().effort_list(0.9, 10.0), None);
    let ix = Index {
        costs: Some(&costs),
        sketches: Some(&sketches),
        ..f.index()
    };
    let k = khpad(&[
        ("exact", "planned"),
        ("crossover", "cost"),
        ("estimates", "sketch"),
        ("terminator", "effort_curve"),
        ("laziness", "heap_bound"),
    ]);
    let plan = Plan::new(&k, &ix).unwrap();
    let mut st = Searcher::for_index(&ix, &[&plan]);
    for q in 0..f.preds.len() {
        let (out, ctr) = search_ix(&f, &ix, &mut st, &plan, q, 700, 10);
        check_properties(&f, q, &out, "effort_curve");
        if ctr.m_hat == 0 {
            continue;
        }
        assert_eq!(ctr.route, Route::Graph, "q={q}");
        let kprime = 10.0 * 300.0 / ctr.m_hat as f64;
        let want = costs.effort_list(0.7, kprime).unwrap().ceil().min(300.0) as u64;
        assert_eq!(ctr.final_l, want + ctr.n_start, "q={q}");
    }
    // Without a curve, the plan is rejected.
    let ix0 = Index {
        costs: Some(&free_graph_costs()),
        ..ix
    };
    assert!(Plan::new(&k, &ix0).is_err());
}

#[test]
fn the_postfilter_route_with_the_configured_starts_keeps_the_heap_answers() {
    // `postfilter_starts = configured`: the same traversal as the configured graph search
    // (label starts, heap(K), heap-bound checks), so the same answers, with fewer checks.
    let f = fixture(300, 47, true);
    let sketches = crate::sketches::LabelSketches::build(&f.labels, 1000, 1, 1);
    let costs = super::cost::UnitCosts {
        graph_alpha: 2.0,
        graph_beta: 10.0,
        pred_ns: vec![1e9; 6],
        dist_ns: vec![1e9; 6],
        ..super::cost::UnitCosts::default()
    };
    let ix = Index {
        costs: Some(&costs),
        sketches: Some(&sketches),
        ..f.index()
    };
    let base = [
        ("exact", "planned"),
        ("crossover", "cost"),
        ("estimates", "sketch"),
        ("terminator", "posting_l(4)"),
        ("laziness", "heap_bound"),
    ];
    let inline = khpad(&base);
    let mut edits = base.to_vec();
    edits.extend([
        ("postfilter", "always"),
        ("postfilter_starts", "configured"),
    ]);
    let route = khpad(&edits);
    let (ip, rp) = (
        Plan::new(&inline, &ix).unwrap(),
        Plan::new(&route, &ix).unwrap(),
    );
    let mut st = Searcher::for_index(&ix, &[&ip, &rp]);
    let (mut routed, mut fewer) = (0, 0);
    for q in 0..f.preds.len() {
        for kk in [1, 5, 10] {
            let (out, ctr) = search_ix(&f, &ix, &mut st, &rp, q, 40, kk);
            if ctr.route != Route::Postfilter {
                continue;
            }
            routed += 1;
            // The inline search at the route's list size: identical answers and traversal.
            let l = (ctr.final_l - ctr.n_start) as usize;
            let k2 = khpad(&[("laziness", "heap_bound"), ("terminator", "exhaust")]);
            let p2 = Plan::new(&k2, &ix).unwrap();
            let (want, wctr) = search_ix(&f, &ix, &mut st, &p2, q, l, kk);
            assert_eq!(out, want, "q={q} K={kk}");
            assert_eq!(
                (ctr.n_dist, ctr.n_start),
                (wctr.n_dist, wctr.n_start),
                "q={q} K={kk}"
            );
            if ctr.n_pred - ctr.n_probe < wctr.n_pred {
                fewer += 1;
            }
        }
    }
    let _ = ip;
    assert!(
        routed > 0 && fewer > 0,
        "routed {routed}, fewer checks {fewer}"
    );
}

#[test]
fn the_graph_crossover_never_takes_the_exact_path() {
    use super::cost::UnitCosts;
    // Expensive graph distances: the cost crossover sends many queries to the exact path; the
    // graph crossover keeps every query on the graph (inline or post-filter), with valid answers.
    let f = fixture(300, 47, false);
    let sketches = crate::sketches::LabelSketches::build(&f.labels, 1000, 1, 1);
    let costs = UnitCosts {
        graph_dist_ns: 1e6,
        effort_r: vec![0.5, 0.9],
        effort_k: vec![10.0, 100.0],
        effort_l: vec![vec![10.0, 40.0], vec![40.0, 160.0]],
        ..UnitCosts::default()
    };
    let ix = Index {
        costs: Some(&costs),
        sketches: Some(&sketches),
        ..f.index()
    };
    for (crossover, terminator) in [("cost", "effort_curve"), ("graph", "effort_curve(2)")] {
        let k = khpad(&[
            ("exact", "planned"),
            ("crossover", crossover),
            ("estimates", "sketch"),
            ("terminator", terminator),
            ("laziness", "heap_bound"),
            ("postfilter", "cost"),
            ("postfilter_starts", "configured"),
        ]);
        let plan = Plan::new(&k, &ix).unwrap();
        let mut st = Searcher::for_index(&ix, &[&plan]);
        let mut exact = 0;
        for q in 0..f.preds.len() {
            let (out, ctr) = search_ix(&f, &ix, &mut st, &plan, q, 700, 10);
            check_properties(&f, q, &out, crossover);
            match ctr.route {
                Route::Exact => exact += 1,
                Route::Graph if crossover == "graph" => {
                    // effort_curve(2): at most twice the list size of an unselective query.
                    let cap = (2.0 * costs.effort_list(0.7, 10.0).unwrap()).ceil() as u64;
                    assert!(ctr.final_l <= cap + ctr.n_start, "q={q}: {}", ctr.final_l);
                }
                _ => {}
            }
        }
        if crossover == "graph" {
            assert_eq!(exact, 0, "the graph crossover took the exact path");
        } else {
            assert!(exact > 0, "the cost crossover never took the exact path");
        }
    }
}
