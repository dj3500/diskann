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
            pairs: None,
            islands: None,
            pair_roots: None,
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
            ("admission", "projection(c*)"),
            ("collector", "heap"),
            ("expander", "two_hop(3)"),
        ],
        &[("admission", "projection(c*)"), ("terminator", "exhaust")],
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
        &[
            ("predicate", "bitmap"),
            ("materializer", "dense_nav"),
            ("admission", "projection(c*)"),
            ("collector", "heap"),
            ("expander", "two_hop(3)"),
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

/// A relabeled index (an isomorphic id layout, `crate::permute`) answers with the same points:
/// for every configuration the same answer distances and, below the K-th distance (where ties
/// are broken by id), the same points; the configurations without id-dependent rules expand
/// the same nodes.
#[test]
fn permuted_layouts_return_the_same_answers() {
    use crate::permute::{permute_rows, Permutation};
    for aux in [false, true] {
        let f = fixture(300, 41 + u64::from(aux), aux);
        let n = f.base.nrows();
        let mut order: Vec<u32> = (0..n as u32).collect();
        let mut rng = Lcg(5);
        for i in (1..n).rev() {
            order.swap(i, rng.below(i as u32 + 1) as usize);
        }
        let p = Permutation::from_order(order).unwrap();
        let labels = f.labels.clone().permute(&p, 3).unwrap();
        let starts = f.starts.permuted(&p).unwrap();
        let g = Fixture {
            base: permute_rows(&f.base, &p),
            queries: f.queries.clone(),
            graph: f.graph.permuted(&p).unwrap(),
            preds: f.preds.clone(),
            sample: SampleIndex::build(&starts, &labels).unwrap(),
            bitsets: LabelBitsets::build(&labels, 1),
            starts,
            labels,
        };
        for v in 0..n as u32 {
            assert_eq!(g.labels.row(p.new_id(v)), f.labels.row(v));
        }
        let (ia, ib) = (f.index(), g.index());
        for (name, kernel) in configs() {
            let pa = Plan::new(&kernel, &ia).unwrap();
            let pb = Plan::new(&kernel, &ib).unwrap();
            let mut sa = Searcher::for_index(&ia, &[&pa]);
            let mut sb = Searcher::for_index(&ib, &[&pb]);
            let same_nodes = ["postfilter", "upstream-inline", "cp"].contains(&name.as_str());
            for q in 0..f.preds.len() {
                for l in [10, 40] {
                    let (ra, ca) = search(&f, &mut sa, &pa, q, l, 10, None);
                    let (rb, cb) = search(&g, &mut sb, &pb, q, l, 10, None);
                    let what = format!("{name} aux={aux} q={q} L={l}");
                    let (da, db): (Vec<f32>, Vec<f32>) = (
                        ra.iter().map(|x| x.0).collect(),
                        rb.iter().map(|x| x.0).collect(),
                    );
                    assert_eq!(da, db, "{what}: answer distances");
                    if let Some(&kth) = da.last() {
                        let below = |r: &[(f32, u32)], map: bool| {
                            let mut ids: Vec<u32> = r
                                .iter()
                                .filter(|x| x.0 < kth)
                                .map(|x| if map { p.new_id(x.1) } else { x.1 })
                                .collect();
                            ids.sort_unstable();
                            ids
                        };
                        assert_eq!(below(&ra, true), below(&rb, false), "{what}: answers");
                    }
                    if same_nodes {
                        assert_eq!(
                            (ca.n_dist, ca.n_expand, ca.n_pred),
                            (cb.n_dist, cb.n_expand, cb.n_pred),
                            "{what}: counters"
                        );
                    }
                }
            }
        }
    }
}

/// Profiling builds count the post-filter loop's phases: every distance computation after the
/// starts is one insertion attempt, and one in `SAMPLE_EVERY` of them is timed.
#[cfg(feature = "phase-profile")]
#[test]
fn the_phase_profile_counts_the_post_filter_loop() {
    use super::profile;
    let f = fixture(300, 9, false);
    let ix = f.index();
    let plan = Plan::new(&Preset::parse("postfilter").unwrap().kernel(), &ix).unwrap();
    let mut st = Searcher::for_index(&ix, &[&plan]);
    profile::take();
    let (mut dists, mut sampled) = (0, 0);
    for q in 0..f.preds.len() {
        let (_, ctr) = search(&f, &mut st, &plan, q, 40, 10, None);
        let d = ctr.n_dist - ctr.n_start;
        dists += d;
        sampled += d.div_ceil(profile::SAMPLE_EVERY);
    }
    let p = profile::take();
    assert_eq!((p[profile::INSERTS], p[profile::SAMPLED]), (dists, sampled));
    assert!(p[profile::EXPAND] > 0 && p[profile::NEIGHBOURS] > p[profile::INSERT_SAMPLED]);
    assert_eq!(profile::take(), [0; profile::SLOTS]);
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
        &[("admission", "projection(c*)"), ("collector", "heap")],
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
        pairs: None,
        islands: None,
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
            pairs: None,
            islands: None,
            ..f.index()
        }
    )
    .is_err());
    k.set("predicate", "static_bits(100)").unwrap();
    assert!(Plan::new(&k, &ix_high).is_ok());
}

#[test]
fn nav_bitmaps_do_not_change_answers_or_counters() {
    // `bitmap` + `dense_nav` (the navigation clauses materialized, the others on the row) gives
    // the default evaluator's answers and counters, with or without static bitsets, also
    // under clause projections (where `dense` / `dense_static` are not allowed).
    let f = fixture(300, 17, true);
    let ix_bits = f.index();
    let ix_none = Index {
        bitsets: None,
        costs: None,
        sketches: None,
        ..f.index()
    };
    let edits: &[&[(&str, &str)]] = &[
        &[],
        &[("admission", "strict"), ("collector", "final_beam")],
        &[
            ("admission", "strict"),
            ("expander", "two_hop(3)"),
            ("start_admission", "force"),
        ],
        &[("admission", "projection(c*)"), ("collector", "heap")],
        &[
            ("admission", "projection(c*)"),
            ("collector", "heap"),
            ("expander", "two_hop(3)"),
            ("terminator", "exhaust"),
        ],
        &[("admission", "projection(a)"), ("collector", "heap")],
        &[
            ("admission", "projection(a, c)"),
            ("collector", "final_beam"),
        ],
        &[("collector", "heap"), ("laziness", "distance_first")],
    ];
    let mut checked = 0;
    for e in edits {
        let mut base = Kernel::default();
        for (key, value) in e.iter() {
            base.set(key, value).unwrap();
        }
        let mut k = base.clone();
        k.set("predicate", "bitmap").unwrap();
        k.set("materializer", "dense_nav").unwrap();
        for ix in [&ix_bits, &ix_none] {
            let pa = Plan::new(&base, ix).unwrap();
            let pb = Plan::new(&k, ix).unwrap_or_else(|err| panic!("{e:?}: {err}"));
            let mut st = Searcher::for_index(ix, &[&pa, &pb]);
            for q in 0..f.preds.len() {
                for l in [10, 30] {
                    let (oa, mut ca) = search_ix(&f, ix, &mut st, &pa, q, l, 10);
                    let (ob, mut cb) = search_ix(&f, ix, &mut st, &pb, q, l, 10);
                    for c in [&mut ca, &mut cb] {
                        [c.res_words, c.res_rows, c.c1_words, c.c1_rows] = [0; 4];
                    }
                    assert_eq!((oa, ca), (ob, cb), "{e:?} q={q} L={l}");
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 100, "{checked}");
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
        pairs: None,
        islands: None,
        pair_roots: None,
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
        pairs: None,
        islands: None,
        pair_roots: None,
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
fn aux_only_expands_routers_only() {
    // A router (id 5) over points 0, 2 and 4 is the only start; the matching point 3 is
    // reachable only through data points, which `aux_only` never expands.
    let rows: &[&[&str]] = &[&["x=1"], &["x=0"], &["x=1"], &["x=1"], &["x=1"]];
    let (base, mut g, labels) = line(rows, true);
    g.set_neighbors(5, &[0, 2, 4]).unwrap();
    g.set_starts(vec![5]).unwrap();
    let pred = [vec!["x=1"]];
    let edits = [
        ("admission", "strict"),
        ("laziness", "predicate_first"),
        ("collector", "heap"),
    ];
    let k = kernel(&[&edits[..], &[("expander", "aux_only")]].concat());
    let (ids, ctr) = run_line(&base, &g, &labels, &[5], &k, &pred, 3.2, 6, 6);
    assert_eq!(ids, [4, 2, 0]);
    assert_eq!((ctr.n_expand, ctr.n_aux_expand), (1, 1));
    assert_eq!((ctr.n_dist, ctr.n_aux_dist), (4, 1));
    let (ids, _) = run_line(&base, &g, &labels, &[5], &kernel(&edits), &pred, 3.2, 6, 6);
    assert_eq!(ids, [3, 4, 2, 0]);
}

#[test]
fn the_tree_route_serves_tree_eligible_queries() {
    // A router (id 5) over the members 0, 2 and 4 of `x=1` is that label's only start;
    // `y=1` (members 1 and 3) starts at its member 1; the global start is point 0.
    let rows: &[&[&str]] = &[&["x=1"], &["y=1"], &["x=1"], &["y=1"], &["x=1"]];
    let (base, mut g, labels) = line(rows, true);
    g.set_neighbors(5, &[0, 2, 4]).unwrap();
    g.set_starts(vec![0]).unwrap();
    let id = |s: &str| labels.name_index()[s];
    let mut st = Starts::new(&g);
    st.set_label_starts(&labels, &[(id("x=1"), 5), (id("y=1"), 1)])
        .unwrap();
    let ix = Index {
        graph: &g,
        vectors: g.vectors(&base).unwrap(),
        labels: &labels,
        global: &[0],
        label_starts: Some(&st),
        sample: None,
        label_view: false,
        view_postings: None,
        eps_min: 0.5,
        bitsets: None,
        costs: None,
        sketches: None,
        pairs: None,
        islands: None,
        pair_roots: None,
    };
    let mut pb = PredicateSetBuilder::new(&labels);
    pb.push(&[vec!["x=1"]]);
    pb.push(&[vec!["y=1"]]);
    let preds = pb.finish();
    for (route, edits) in [
        (Route::Tree, &[("tree_route", "row_scan")][..]),
        (Route::Graph, &[][..]),
    ] {
        let k = kernel(
            &[
                &[("starts", "label(c*)+global"), ("collector", "heap")][..],
                edits,
            ]
            .concat(),
        );
        let plan = Plan::new(&k, &ix).unwrap();
        let mut s = Searcher::for_index(&ix, &[&plan]);
        let (mut ids, mut d) = (vec![0u32; 3], vec![0f32; 3]);
        let mut run = |qi: usize, ids: &mut [u32], d: &mut [f32]| {
            let q = Query {
                vector: &[3.2f32],
                cnf: preds.get(qi),
                view: None,
            };
            s.search(&plan, &ix, &q, 4, ids, d, None)
        };
        let ctr = run(0, &mut ids, &mut d);
        assert_eq!(ids, [4, 2, 0], "{route:?}");
        assert_eq!(ctr.route, route);
        if route == Route::Tree {
            // Only the router is expanded; its members are checked, then measured.
            assert_eq!((ctr.n_expand, ctr.n_aux_expand, ctr.n_dist), (1, 1, 4));
        }
        // `y=1` has a data-point start: not tree-eligible, the configured search runs.
        let ctr = run(1, &mut ids[..2], &mut d[..2]);
        assert_eq!(ids[..2], [3, 1]);
        assert_eq!(ctr.route, Route::Graph);
    }
}

#[test]
fn a_label_whose_starts_are_its_members_takes_the_tree_route() {
    // `flab steiner --leaf-starts`: `x=2` (members 1 and 3) has no router, its starts are its
    // members; `x=1` (members 0, 2, 4) has a router (id 5); `w=1` (members 0 and 4) has only
    // one of its members as a start (as a base start table may): not eligible.
    let rows: &[&[&str]] = &[
        &["x=1", "w=1", "v=1"],
        &["x=2"],
        &["x=1"],
        &["x=2", "v=1"],
        &["x=1", "w=1", "v=1"],
    ];
    let (base, mut g, labels) = line(rows, true);
    g.set_neighbors(5, &[0, 2, 4]).unwrap();
    let id = |s: &str| labels.name_index()[s];
    let mut st = Starts::new(&g);
    st.set_label_starts(
        &labels,
        &[
            (id("x=1"), 5),
            (id("x=2"), 1),
            (id("x=2"), 3),
            (id("w=1"), 4),
        ],
    )
    .unwrap();
    let ix = Index {
        graph: &g,
        vectors: g.vectors(&base).unwrap(),
        labels: &labels,
        global: &[0],
        label_starts: Some(&st),
        sample: None,
        label_view: false,
        view_postings: None,
        eps_min: 0.5,
        bitsets: None,
        costs: None,
        sketches: None,
        pairs: None,
        islands: None,
        pair_roots: None,
    };
    let mut pb = PredicateSetBuilder::new(&labels);
    pb.push(&[vec!["x=2"]]);
    pb.push(&[vec!["x=2"], vec!["v=1"]]);
    pb.push(&[vec!["x=1", "x=2"]]);
    pb.push(&[vec!["w=1"]]);
    let preds = pb.finish();
    // A router frontier (fixed effort): member starts are checked and only matches measured.
    let k = kernel(&[
        ("starts", "label(c*)+global"),
        ("collector", "heap"),
        ("tree_route", "row_scan"),
        ("tree_effort", "fixed(1)"),
    ]);
    let plan = Plan::new(&k, &ix).unwrap();
    let mut s = Searcher::for_index(&ix, &[&plan]);
    let mut run = |qi: usize, k: usize| {
        let (mut ids, mut d) = (vec![0u32; k], vec![0f32; k]);
        let q = Query {
            vector: &[3.2f32],
            cnf: preds.get(qi),
            view: None,
        };
        let ctr = s.search(&plan, &ix, &q, 4, &mut ids, &mut d, None);
        (ids, ctr)
    };
    let (ids, ctr) = run(0, 2);
    assert_eq!(ids, [3, 1]);
    assert_eq!(ctr.route, Route::Tree);
    assert_eq!((ctr.n_expand, ctr.n_aux_expand, ctr.n_dist), (0, 0, 2));
    // With a second clause only the matching member (3) is measured.
    let (ids, ctr) = run(1, 2);
    assert_eq!(ids, [3, u32::MAX]);
    assert_eq!(ctr.route, Route::Tree);
    assert_eq!((ctr.n_pred, ctr.n_dist), (2, 1));
    // A clause with a tree and member starts: the router frontier (list 1 plus its one router
    // start) expands the router; the member starts are collected besides; every point found.
    let (ids, ctr) = run(2, 5);
    assert_eq!(ids, [3, 4, 2, 1, 0]);
    assert_eq!(ctr.route, Route::Tree);
    assert_eq!((ctr.n_expand, ctr.n_aux_expand), (1, 1));
    // Starts that are not all of the label's members: not eligible.
    let (ids, ctr) = run(3, 2);
    assert_eq!(ids, [4, 0]);
    assert_eq!(ctr.route, Route::Graph);
}

#[test]
fn the_static_bits_tree_route_checks_only_the_other_clauses() {
    // A router (id 5) over the members 0, 2 and 4 of `x=1` (c*: 3 points vs `y=1`'s 4) is
    // that label's only start; the members 0 and 4 also carry `y=1`.
    let rows: &[&[&str]] = &[
        &["x=1", "y=1"],
        &["y=1"],
        &["x=1", "y=2"],
        &["y=1"],
        &["x=1", "y=1"],
    ];
    let (base, mut g, labels) = line(rows, true);
    g.set_neighbors(5, &[0, 2, 4]).unwrap();
    let id = |s: &str| labels.name_index()[s];
    let mut st = Starts::new(&g);
    st.set_label_starts(&labels, &[(id("x=1"), 5), (id("y=1"), 1), (id("y=2"), 2)])
        .unwrap();
    let bitsets = LabelBitsets::build(&labels, 1);
    let ix = Index {
        graph: &g,
        vectors: g.vectors(&base).unwrap(),
        labels: &labels,
        global: &[0],
        label_starts: Some(&st),
        sample: None,
        label_view: false,
        view_postings: None,
        eps_min: 0.5,
        bitsets: Some(&bitsets),
        costs: None,
        sketches: None,
        pairs: None,
        islands: None,
        pair_roots: None,
    };
    let mut pb = PredicateSetBuilder::new(&labels);
    pb.push(&[vec!["x=1"], vec!["y=1"]]);
    pb.push(&[vec!["x=1"]]);
    let preds = pb.finish();
    let mut out = Vec::new();
    for route in ["row_scan", "static_bits"] {
        let k = kernel(&[
            ("starts", "label(c*)+global"),
            ("collector", "heap"),
            ("tree_route", route),
        ]);
        let plan = Plan::new(&k, &ix).unwrap();
        let mut s = Searcher::for_index(&ix, &[&plan]);
        for (qi, want) in [(0, &[4u32, 0, u32::MAX][..]), (1, &[4, 2, 0][..])] {
            let (mut ids, mut d) = (vec![0u32; 3], vec![0f32; 3]);
            let q = Query {
                vector: &[3.2f32],
                cnf: preds.get(qi),
                view: None,
            };
            let ctr = s.search(&plan, &ix, &q, 4, &mut ids, &mut d, None);
            assert_eq!(ids, want, "{route} {qi}");
            assert_eq!(ctr.route, Route::Tree);
            out.push((ctr.n_expand, ctr.n_aux_expand, ctr.n_dist, ctr.n_pred));
        }
    }
    // The same members checked and measured either way (the router, then the matches).
    assert_eq!(out[..2], out[2..]);
    assert_eq!(out[0], (1, 1, 3, 3));
    // Under `cstar_members` a node is judged by the clauses other than c* alone: point 1
    // (`y=1`, not `x=1`) passes, which only members of c*'s trees ever meet in the search.
    let mut compiled = super::eval::Compiled::new();
    let mut bits = vec![0u64; labels.n_labels()];
    compiled.compile(
        &labels,
        preds.get(0),
        super::eval::Weights::Unit,
        None,
        &mut bits,
    );
    compiled.residual_nav();
    compiled.split_static(&labels, &bitsets, 1, false);
    let cx = super::eval::EvalCtx::new(&labels, &compiled, &bits, &[], &[])
        .with_static(&bitsets)
        .cstar_members();
    let pass = |v: u32| {
        use super::eval::Eval;
        super::eval::StaticBits::eval(&cx, v, false).m
    };
    assert_eq!(
        (0..5).map(pass).collect::<Vec<_>>(),
        [true, true, false, true, true]
    );
    // The route's checks need the static bitsets and a tree search from c*.
    let mut k = kernel(&[("tree_route", "static_bits")]);
    k.set("static_bitsets", "off").unwrap();
    assert!(k.validate().is_err());
    let mut k = kernel(&[]);
    k.set("cstar_members", "on").unwrap();
    assert!(k.validate().is_err());
    kernel(&[
        ("starts", "label(c*)"),
        ("start_admission", "force"),
        ("admission", "strict"),
        ("expander", "aux_frontier"),
        ("collector", "heap"),
        ("terminator", "exhaust"),
        ("laziness", "predicate_first"),
        ("predicate", "static_bits(1)"),
        ("cstar_members", "on"),
    ]);
}

#[test]
fn the_cost_tree_route_picks_the_cheapest_member_check() {
    use super::cost::UnitCosts;
    // As above: a router (id 5) over the members 0, 2 and 4 of `x=1` (c*), `y=1` on 0, 1, 3, 4.
    let rows: &[&[&str]] = &[
        &["x=1", "y=1"],
        &["y=1"],
        &["x=1", "y=2"],
        &["y=1"],
        &["x=1", "y=1"],
    ];
    let (base, mut g, labels) = line(rows, true);
    g.set_neighbors(5, &[0, 2, 4]).unwrap();
    let id = |s: &str| labels.name_index()[s];
    let mut st = Starts::new(&g);
    st.set_label_starts(&labels, &[(id("x=1"), 5), (id("y=1"), 1), (id("y=2"), 2)])
        .unwrap();
    let bitsets = LabelBitsets::build(&labels, 1);
    let sketches = crate::sketches::LabelSketches::build(&labels, 1000, 1, 1);
    let mut pb = PredicateSetBuilder::new(&labels);
    pb.push(&[vec!["x=1"], vec!["y=1"]]);
    pb.push(&[vec!["x=1"], vec!["y=1", "y=2"]]);
    let preds = pb.finish();
    let shaped = |base: f64, word: f64, row: f64, word_ns: f64| UnitCosts {
        check_base_ns: vec![base; 6],
        check_word_ns: vec![word; 6],
        check_prefetch_ns: vec![0.0; 6],
        check_row_ns: vec![row; 6],
        word_ns,
        scatter_ns: 0.0,
        ..UnitCosts::default()
    };
    let run = |costs: &UnitCosts, qi: usize| {
        let ix = Index {
            graph: &g,
            vectors: g.vectors(&base).unwrap(),
            labels: &labels,
            global: &[0],
            label_starts: Some(&st),
            sample: None,
            label_view: false,
            view_postings: None,
            eps_min: 0.5,
            bitsets: Some(&bitsets),
            costs: Some(costs),
            sketches: Some(&sketches),
            pairs: None,
            islands: None,
            pair_roots: None,
        };
        let k = kernel(&[
            ("starts", "label(c*)+global"),
            ("collector", "heap"),
            ("crossover", "graph"),
            ("exact", "planned"),
            ("estimates", "sketch"),
            ("tree_route", "cost"),
        ]);
        let plan = Plan::new(&k, &ix).unwrap();
        let mut s = Searcher::for_index(&ix, &[&plan]);
        let (mut ids, mut d) = (vec![0u32; 3], vec![0f32; 3]);
        let q = Query {
            vector: &[3.2f32],
            cnf: preds.get(qi),
            view: None,
        };
        let ctr = s.search(&plan, &ix, &q, 4, &mut ids, &mut d, None);
        assert_eq!(ctr.route, Route::Tree);
        (ctr.plan, ids)
    };
    let want = [4u32, 0, u32::MAX];
    // Expensive rows, cheap words, a costly bitmap build: the static check (102).
    assert_eq!(run(&shaped(1.0, 1.0, 100.0, 1e6), 0), (102, want.to_vec()));
    // Cheap rows, expensive words: the row scan (101).
    assert_eq!(run(&shaped(1.0, 100.0, 1.0, 1e6), 0), (101, want.to_vec()));
    // A free bitmap build against two residual words per member: the bitmap (103).
    assert_eq!(run(&shaped(1.0, 10.0, 100.0, 0.0), 1), (103, vec![4, 2, 0]));
    // The cost route needs a planned exact path (its unit costs) and tree_choice = always.
    let mut k = Kernel::default();
    k.set("tree_route", "cost").unwrap();
    assert!(k.validate().is_err());
}

#[test]
fn the_pair_tree_route_searches_the_conjunctions_tree() {
    use crate::pairs::PairIndex;
    // 200 points on a line: `x=1` on 0..100, `x=2` on 100..200, `y=1` on 80..200, `y=2` on
    // 0..80; the conjunction `x=1 & y=1` holds on 80..100 only (20 points).
    let n = 200usize;
    let base = Matrix::from_slice(&(0..n).map(|i| i as f32).collect::<Vec<_>>(), n, 1);
    let mut b = LabelStoreBuilder::new();
    for i in 0..n {
        let x = if i < 100 { "x=1" } else { "x=2" };
        let y = if i >= 80 { "y=1" } else { "y=2" };
        b.push_point([("x", x), ("y", y)]).unwrap();
    }
    let labels = b.finish().unwrap();
    let id = |s: &str| labels.name_index()[s];
    let (px, _) = PairIndex::build(&labels, 1 << 20, 1);
    let crate::pairs::Joint::Stored(js, je) = px.joint(id("x=1"), id("y=1")) else {
        panic!("the pair x=1 & y=1 is stored");
    };
    assert_eq!(px.list(js, je), (80..100).collect::<Vec<u32>>());
    // Routers: 200 over `x=1`'s members (its label tree), 201 over the conjunction's.
    let aux = AuxNodes::new(
        vec![AuxKind::Router; 2],
        &Matrix::from_slice(&[50.0f32, 90.0], 2, 1),
    )
    .unwrap();
    let mut g = Graph::new(n, 100, aux).unwrap();
    for i in 0..n as u32 {
        let nb: Vec<u32> = [i.checked_sub(1), (i + 1 < n as u32).then_some(i + 1)]
            .into_iter()
            .flatten()
            .collect();
        g.set_neighbors(i, &nb).unwrap();
    }
    g.set_neighbors(200, &(0..100).collect::<Vec<u32>>())
        .unwrap();
    g.set_neighbors(201, &(80..100).collect::<Vec<u32>>())
        .unwrap();
    g.set_starts(vec![0]).unwrap();
    let mut st = Starts::new(&g);
    st.set_label_starts(
        &labels,
        &[
            (id("x=1"), 200),
            (id("x=2"), 150),
            (id("y=1"), 150),
            (id("y=2"), 40),
        ],
    )
    .unwrap();
    let mut roots = vec![u32::MAX; px.n_pairs()];
    roots[px.slot_of(js)] = 201;
    let no_roots = vec![u32::MAX; px.n_pairs()];
    let bitsets = LabelBitsets::build(&labels, 1);
    let sketches = crate::sketches::LabelSketches::build(&labels, 1000, 1, 1);
    let costs = super::cost::UnitCosts::default();
    let mut pb = PredicateSetBuilder::new(&labels);
    pb.push(&[vec!["x=1"], vec!["y=1"]]);
    let preds = pb.finish();
    let run = |route: &str, pairs: &str, roots: &[u32]| {
        let ix = Index {
            graph: &g,
            vectors: g.vectors(&base).unwrap(),
            labels: &labels,
            global: &[0],
            label_starts: Some(&st),
            sample: None,
            label_view: false,
            view_postings: None,
            eps_min: 0.5,
            bitsets: Some(&bitsets),
            costs: Some(&costs),
            sketches: Some(&sketches),
            pairs: Some(&px),
            islands: None,
            pair_roots: Some(roots),
        };
        let mut edits = vec![
            ("starts", "label(c*)+global"),
            ("collector", "heap"),
            ("crossover", "graph"),
            ("exact", "planned"),
            ("estimates", "sketch"),
            ("tree_route", route),
        ];
        if pairs != "off" {
            edits.push(("tree_pairs", pairs));
        }
        let plan = Plan::new(&kernel(&edits), &ix).unwrap();
        if pairs == "on" {
            // The model's prices at the query's dilution: a tree's frontier scaled to its
            // members per estimated match, at most the whole tree, its list at least the
            // configured one; a diluted tree costs more than an undiluted one of its size.
            let est = super::cost::Estimates {
                n: 200.0,
                m: 20.0,
                ..Default::default()
            };
            let (l_pair, c_pair) = super::search::tree_route_option(&plan, &est, 8, 3, 20.0, 202);
            let (l_label, c_label) =
                super::search::tree_route_option(&plan, &est, 8, 3, 100.0, 202);
            assert_eq!((l_pair, l_label), (8, 8));
            assert!(c_pair < c_label, "{c_pair} {c_label}");
            let thin = super::cost::Estimates { m: 2.0, ..est };
            let (l_thin, c_thin) = super::search::tree_route_option(&plan, &thin, 8, 3, 1e6, 202);
            assert!(l_thin > 8 && c_thin > c_label, "{l_thin} {c_thin}");
        }
        let mut s = Searcher::for_index(&ix, &[&plan]);
        let (mut ids, mut d) = (vec![0u32; 3], vec![0f32; 3]);
        let q = Query {
            vector: &[90.3f32],
            cnf: preds.get(0),
            view: None,
        };
        let ctr = s.search(&plan, &ix, &q, 8, &mut ids, &mut d, None);
        assert_eq!(ids, [90, 91, 89], "{route} {pairs}");
        assert_eq!(ctr.route, Route::Tree);
        (ctr.plan, ctr.n_pred, ctr.n_aux_expand)
    };
    // The label tree checks x=1's 100 members; the pair tree only the conjunction's 20 — taken
    // at any cost (`always`), and by the cost check (`on`) since it is cheaper than the label
    // tree, the query's other route.
    let (plan, label_checks, _) = run("row_scan", "off", &roots);
    assert_eq!((plan, label_checks), (0, 100));
    let (plan, pair_checks, expanded) = run("row_scan", "always", &roots);
    assert_eq!((plan, pair_checks, expanded), (110, 20, 1));
    let (plan, _, expanded) = run("row_scan", "on", &roots);
    assert_eq!((plan, expanded), (110, 1));
    // Without a tree for the pair, the label tree route.
    assert_eq!(run("row_scan", "always", &no_roots).0, 0);
    assert_eq!(run("row_scan", "on", &no_roots).0, 0);
    // Under the cost route the pair route picks its check too (111–113).
    assert!((111..=113).contains(&run("cost", "always", &roots).0));
    assert!((111..=113).contains(&run("cost", "on", &roots).0));
    // tree_pairs needs the tree route; `on` also the unit costs and a crossover that prices its
    // routes (not the threshold router).
    let mut k = Kernel::default();
    k.set("tree_pairs", "on").unwrap();
    assert!(k.validate().is_err());
    let mut k = Kernel::default();
    for (key, value) in [
        ("starts", "label(c*)+global"),
        ("collector", "heap"),
        ("crossover", "threshold"),
        ("exact", "planned"),
        ("estimates", "sketch"),
        ("tree_route", "row_scan"),
        ("tree_pairs", "on"),
    ] {
        k.set(key, value).unwrap();
    }
    let e = k.validate().unwrap_err().to_string();
    assert!(e.contains("tree_pairs"), "{e}");
    k.set("tree_pairs", "always").unwrap();
    assert!(k.validate().is_ok());
    let mut k = Kernel::default();
    assert!(k.set("tree_pairs", "sometimes").is_err());
}

/// Six points on a line (x = i, all labelled `x=1`), leaf routers 6 (over 0, 1, 2; at 1) and
/// 7 (over 3, 4, 5; at 4) under a root 8 (at 2.5), the root as the label's only start.
fn router_tree() -> (Matrix<f32>, Graph, LabelStore, Starts) {
    let n = 6;
    let base = Matrix::from_slice(&(0..n).map(|i| i as f32).collect::<Vec<_>>(), n, 1);
    let aux = AuxNodes::new(
        vec![AuxKind::Router; 3],
        &Matrix::from_slice(&[1.0f32, 4.0, 2.5], 3, 1),
    )
    .unwrap();
    let mut g = Graph::new(n, 4, aux).unwrap();
    for i in 0..n as u32 {
        let nb: Vec<u32> = [i.checked_sub(1), (i + 1 < n as u32).then_some(i + 1)]
            .into_iter()
            .flatten()
            .collect();
        g.set_neighbors(i, &nb).unwrap();
    }
    g.set_neighbors(6, &[0, 1, 2]).unwrap();
    g.set_neighbors(7, &[3, 4, 5]).unwrap();
    g.set_neighbors(8, &[6, 7]).unwrap();
    g.set_starts(vec![0]).unwrap();
    let mut b = LabelStoreBuilder::new();
    for _ in 0..n {
        b.push_point([("x", "x=1")]).unwrap();
    }
    let labels = b.finish().unwrap();
    let mut st = Starts::new(&g);
    st.set_label_starts(&labels, &[(labels.name_index()["x=1"], 8)])
        .unwrap();
    (base, g, labels, st)
}

fn tree_index<'a>(
    base: &'a Matrix<f32>,
    g: &'a Graph,
    labels: &'a LabelStore,
    st: &'a Starts,
) -> Index<'a, f32> {
    Index {
        graph: g,
        vectors: g.vectors(base).unwrap(),
        labels,
        global: &[0],
        label_starts: Some(st),
        sample: None,
        label_view: false,
        view_postings: None,
        eps_min: 0.5,
        bitsets: None,
        costs: None,
        sketches: None,
        pairs: None,
        islands: None,
        pair_roots: None,
    }
}

#[test]
fn aux_frontier_keeps_only_routers_in_the_list() {
    let (base, g, labels, st) = router_tree();
    let ix = tree_index(&base, &g, &labels, &st);
    let mut pb = PredicateSetBuilder::new(&labels);
    pb.push(&[vec!["x=1"]]);
    let preds = pb.finish();
    let run = |expander: &str, l: usize| {
        let k = kernel(&[
            ("starts", "label(c*)"),
            ("admission", "strict"),
            ("laziness", "predicate_first"),
            ("collector", "heap"),
            ("expander", expander),
        ]);
        let plan = Plan::new(&k, &ix).unwrap();
        let mut s = Searcher::for_index(&ix, &[&plan]);
        let (mut ids, mut d) = (vec![0u32; 6], vec![0f32; 6]);
        let q = Query {
            vector: &[5.0f32],
            cnf: preds.get(0),
            view: None,
        };
        let ctr = s.search(&plan, &ix, &q, l, &mut ids, &mut d, None);
        ids.truncate(ctr.n_out as usize);
        (ids, ctr)
    };
    // A frontier of two routers: both leaves are expanded, every member is collected.
    let (ids, ctr) = run("aux_frontier", 2);
    assert_eq!(ids, [5, 4, 3, 2, 1, 0]);
    assert_eq!(ctr.n_aux_expand, 3);
    // aux_only: the near leaf's members enter the list of two and evict the far leaf.
    let (ids, ctr) = run("aux_only", 2);
    assert_eq!(ids, [5, 4, 3]);
    assert_eq!(ctr.n_aux_expand, 2);
    // A frontier of one router keeps only the nearer leaf.
    let (ids, _) = run("aux_frontier", 1);
    assert_eq!(ids, [5, 4, 3]);
    // aux_frontier needs the heap collector.
    let mut k = Kernel::default();
    k.set("starts", "label(c*)").unwrap();
    k.set("expander", "aux_frontier").unwrap();
    k.collector = super::collect::Collector::FinalBeam;
    assert!(k.validate().is_err());
}

#[test]
fn the_tree_effort_curve_sets_the_tree_list_size() {
    use super::cost::UnitCosts;
    let (base, g, labels, st) = router_tree();
    let sketches = crate::sketches::LabelSketches::build(&labels, 1000, 1, 1);
    let costs = UnitCosts {
        effort_r: vec![0.5, 0.9],
        effort_k: vec![10.0, 100.0],
        effort_l: vec![vec![10.0, 40.0], vec![40.0, 160.0]],
        // Measured far beyond the table: no list is cut.
        effort_l_top: 1e9,
        tree_r: vec![0.5, 0.9],
        tree_k: vec![1.0, 100.0],
        tree_l: vec![vec![1.0, 1.0], vec![2.0, 2.0]],
        ..UnitCosts::default()
    };
    assert_eq!(costs.tree_list(0.9, 6.0, 6.0), Some(2.0));
    // Size classes: log-linear in the label size between the classes, clamped outside them.
    let classes = UnitCosts {
        tree_a: vec![10.0, 1000.0],
        tree_l: vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![100.0, 100.0],
            vec![200.0, 200.0],
        ],
        ..costs.clone()
    };
    classes.validate().unwrap();
    let at = |a: f64| classes.tree_list(0.9, 6.0, a).unwrap();
    for (a, want) in [(5.0, 2.0), (100.0, 20.0), (5000.0, 200.0)] {
        assert!((at(a) - want).abs() < 1e-9, "a = {a}: {}", at(a));
    }
    let ix = Index {
        costs: Some(&costs),
        sketches: Some(&sketches),
        ..tree_index(&base, &g, &labels, &st)
    };
    let mut pb = PredicateSetBuilder::new(&labels);
    pb.push(&[vec!["x=1"]]);
    let preds = pb.finish();
    let k = kernel(&[
        ("starts", "label(c*)+global"),
        ("collector", "heap"),
        ("terminator", "effort_curve"),
        ("laziness", "heap_bound"),
        ("crossover", "graph"),
        ("exact", "planned"),
        ("estimates", "sketch"),
        ("tree_route", "row_scan"),
        ("tree_effort", "curve"),
    ]);
    let plan = Plan::new(&k, &ix).unwrap();
    let mut s = Searcher::for_index(&ix, &[&plan]);
    for (effort, want) in [(900, vec![5, 4, 3, 2, 1, 0]), (500, vec![5, 4, 3])] {
        let (mut ids, mut d) = (vec![0u32; 6], vec![0f32; 6]);
        let q = Query {
            vector: &[5.0f32],
            cnf: preds.get(0),
            view: None,
        };
        let ctr = s.search(&plan, &ix, &q, effort, &mut ids, &mut d, None);
        ids.truncate(ctr.n_out as usize);
        assert_eq!(ctr.route, Route::Tree);
        // Every member matches (rho = 1): the frontier for R at K' = K.
        assert_eq!(ids, want, "effort {effort}");
        assert_eq!(ctr.m_hat, 6);
    }
    // Without a tree curve the plan is rejected.
    let c0 = UnitCosts {
        tree_r: Vec::new(),
        tree_k: Vec::new(),
        tree_l: Vec::new(),
        ..costs.clone()
    };
    let ix0 = Index {
        costs: Some(&c0),
        ..ix
    };
    assert!(Plan::new(&k, &ix0).is_err());
}

#[test]
fn a_fixed_tree_effort_uses_one_frontier_for_every_effort() {
    use super::config::TreeEffort;
    assert_eq!(
        TreeEffort::parse("fixed(16)").unwrap(),
        TreeEffort::Fixed(16)
    );
    assert_eq!(TreeEffort::Fixed(16).to_string(), "fixed(16)");
    assert!(TreeEffort::parse("fixed(0)").is_err());
    assert!(TreeEffort::parse("fixed(x)").is_err());
    let (base, g, labels, st) = router_tree();
    // No unit costs: a fixed frontier needs no tree effort curve.
    let ix = tree_index(&base, &g, &labels, &st);
    let mut pb = PredicateSetBuilder::new(&labels);
    pb.push(&[vec!["x=1"]]);
    let preds = pb.finish();
    // Frontier 1: the root, then the router nearer the query; frontier 2: both routers.
    for (frontier, want) in [(1, vec![5u32, 4, 3]), (2, vec![5, 4, 3, 2, 1, 0])] {
        let fixed = format!("fixed({frontier})");
        let k = kernel(&[
            ("starts", "label(c*)+global"),
            ("collector", "heap"),
            ("terminator", "exhaust"),
            ("tree_route", "row_scan"),
            ("tree_effort", fixed.as_str()),
        ]);
        let plan = Plan::new(&k, &ix).unwrap();
        let mut s = Searcher::for_index(&ix, &[&plan]);
        for effort in [1, 6, 100] {
            let (mut ids, mut d) = (vec![0u32; 6], vec![0f32; 6]);
            let q = Query {
                vector: &[5.0f32],
                cnf: preds.get(0),
                view: None,
            };
            let ctr = s.search(&plan, &ix, &q, effort, &mut ids, &mut d, None);
            ids.truncate(ctr.n_out as usize);
            assert_eq!(ctr.route, Route::Tree);
            assert_eq!(ids, want, "frontier {frontier}, effort {effort}");
        }
    }
    // The tree route is required.
    let mut k = kernel(&[("collector", "heap"), ("terminator", "exhaust")]);
    k.tree_effort = TreeEffort::Fixed(4);
    assert!(k.validate().is_err());
    // The bitmap tree route checks members with the static bitsets: not under `static_bitsets = off`.
    let mut k = kernel(&[("tree_route", "bitmap"), ("tree_effort", "fixed(4)")]);
    k.set("static_bitsets", "off").unwrap();
    assert!(k.validate().is_err());
    k.set("tree_route", "row_scan").unwrap();
    k.validate().unwrap();
}

/// Ten points on a line; label x on 0..=5 under routers 10 = {0, 1, 2}, 11 = {3, 4, 5} and
/// the root 12; label y on {1, 3, 5, 6, ..., 9}: x AND y keeps half of x (rho = 0.5).
fn diluted_tree() -> (Matrix<f32>, Graph, LabelStore, Starts) {
    let n = 10;
    let base = Matrix::from_slice(&(0..n).map(|i| i as f32).collect::<Vec<_>>(), n, 1);
    let aux = AuxNodes::new(
        vec![AuxKind::Router; 3],
        &Matrix::from_slice(&[1.0f32, 4.0, 2.5], 3, 1),
    )
    .unwrap();
    let mut g = Graph::new(n, 4, aux).unwrap();
    for i in 0..n as u32 {
        let nb: Vec<u32> = [i.checked_sub(1), (i + 1 < n as u32).then_some(i + 1)]
            .into_iter()
            .flatten()
            .collect();
        g.set_neighbors(i, &nb).unwrap();
    }
    g.set_neighbors(10, &[0, 1, 2]).unwrap();
    g.set_neighbors(11, &[3, 4, 5]).unwrap();
    g.set_neighbors(12, &[10, 11]).unwrap();
    g.set_starts(vec![0]).unwrap();
    let mut b = LabelStoreBuilder::new();
    for i in 0..n {
        let x = (i <= 5).then_some(("x", "x=1"));
        let y = (i >= 6 || i % 2 == 1).then_some(("y", "y=1"));
        b.push_point(x.into_iter().chain(y)).unwrap();
    }
    let labels = b.finish().unwrap();
    let mut st = Starts::new(&g);
    st.set_label_starts(&labels, &[(labels.name_index()["x=1"], 12)])
        .unwrap();
    (base, g, labels, st)
}

#[test]
fn the_capped_tree_effort_bounds_diluted_queries() {
    use super::cost::UnitCosts;
    let (base, g, labels, st) = diluted_tree();
    let sketches = crate::sketches::LabelSketches::build(&labels, 1000, 1, 1);
    // The tree curve's frontier is 1.2 times the number of members (L(K') = 1.2 K').
    let costs = UnitCosts {
        effort_r: vec![0.5, 0.9],
        effort_k: vec![10.0, 100.0],
        effort_l: vec![vec![10.0, 40.0], vec![40.0, 160.0]],
        // Measured far beyond the table: no list is cut.
        effort_l_top: 1e9,
        tree_r: vec![0.5, 0.9],
        tree_k: vec![1.0, 100.0],
        tree_l: vec![vec![1.2, 120.0], vec![1.2, 120.0]],
        ..UnitCosts::default()
    };
    let ix = Index {
        costs: Some(&costs),
        sketches: Some(&sketches),
        ..tree_index(&base, &g, &labels, &st)
    };
    let mut pb = PredicateSetBuilder::new(&labels);
    pb.push(&[vec!["x=1"], vec!["y=1"]]);
    let preds = pb.finish();
    let run = |effort: &str| {
        let k = kernel(&[
            ("starts", "label(c*)+global"),
            ("collector", "heap"),
            ("terminator", "effort_curve(1.5)"),
            ("laziness", "heap_bound"),
            ("crossover", "graph"),
            ("exact", "planned"),
            ("estimates", "sketch"),
            ("tree_route", "row_scan"),
            ("tree_effort", effort),
        ]);
        let plan = Plan::new(&k, &ix).unwrap();
        let mut s = Searcher::for_index(&ix, &[&plan]);
        let (mut ids, mut d) = (vec![0u32; 2], vec![0f32; 2]);
        let q = Query {
            vector: &[5.0f32],
            cnf: preds.get(0),
            view: None,
        };
        let ctr = s.search(&plan, &ix, &q, 900, &mut ids, &mut d, None);
        assert_eq!(ctr.route, Route::Tree);
        assert_eq!((ctr.a_hat, ctr.m_hat), (6, 3));
        ids.truncate(ctr.n_out as usize);
        (ids, ctr.final_l)
    };
    // K = 2 at rho <= 0.5 (sketch estimates): the curve's frontier for K' >= 4 members (the
    // list's capacity is the frontier plus the start, upstream's padding) ...
    let (ids, l_curve) = run("curve");
    assert_eq!(ids, [5, 3]);
    assert!(l_curve >= 6, "{l_curve}");
    // ... capped at 1.5 times the frontier for K = 2 (3.6, rounded up).
    assert_eq!(run("capped"), (vec![5, 3], 4 + 1));
}

#[test]
fn tree_choice_cost_takes_the_tree_route_only_when_it_is_cheaper() {
    use super::cost::UnitCosts;
    let (base, g, labels, st) = diluted_tree();
    let sketches = crate::sketches::LabelSketches::build(&labels, 1000, 1, 1);
    let curves = UnitCosts {
        effort_r: vec![0.5, 0.9],
        effort_k: vec![10.0, 100.0],
        effort_l: vec![vec![10.0, 40.0], vec![40.0, 160.0]],
        // Measured far beyond the table: no list is cut.
        effort_l_top: 1e9,
        tree_r: vec![0.5, 0.9],
        tree_k: vec![1.0, 100.0],
        tree_l: vec![vec![1.2, 120.0], vec![1.2, 120.0]],
        ..UnitCosts::default()
    };
    // Graph searches of 10 L + 200 distances, or of about one.
    let cheap_graph = UnitCosts {
        graph_alpha: 0.001,
        graph_beta: 1.0,
        ..curves.clone()
    };
    let mut pb = PredicateSetBuilder::new(&labels);
    pb.push(&[vec!["x=1"], vec!["y=1"]]);
    let preds = pb.finish();
    let run = |costs: &UnitCosts| {
        let ix = Index {
            costs: Some(costs),
            sketches: Some(&sketches),
            ..tree_index(&base, &g, &labels, &st)
        };
        let k = kernel(&[
            ("starts", "label(c*)+global"),
            ("collector", "heap"),
            ("terminator", "effort_curve"),
            ("laziness", "heap_bound"),
            ("crossover", "graph"),
            ("exact", "planned"),
            ("estimates", "sketch"),
            ("tree_route", "row_scan"),
            ("tree_effort", "curve"),
            ("tree_choice", "cost"),
        ]);
        let plan = Plan::new(&k, &ix).unwrap();
        let mut s = Searcher::for_index(&ix, &[&plan]);
        let (mut ids, mut d) = (vec![0u32; 2], vec![0f32; 2]);
        let q = Query {
            vector: &[5.0f32],
            cnf: preds.get(0),
            view: None,
        };
        let ctr = s.search(&plan, &ix, &q, 900, &mut ids, &mut d, None);
        ids.truncate(ctr.n_out as usize);
        (ctr.route, ids, ctr.est_graph_ns)
    };
    // A few router expansions against a graph search of hundreds of distances: the tree.
    let (route, ids, g_ns) = run(&curves);
    assert_eq!((route, ids), (Route::Tree, vec![5, 3]), "graph {g_ns} ns");
    // Against a search of about one distance: the planned route.
    let (route, _, g_ns) = run(&cheap_graph);
    assert_ne!(route, Route::Tree, "graph {g_ns} ns");
    // tree_choice = cost needs a tree effort curve.
    let mut k = Kernel::default();
    k.set("tree_route", "row_scan").unwrap();
    k.set("tree_choice", "cost").unwrap();
    assert!(k.validate().is_err());
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

/// `packed_beam` changes no answer or counter of the post-filter preset (fast path kept) and
/// of the hybrid's post-filter route.
#[test]
fn the_packed_beam_changes_no_answer() {
    let f = fixture(300, 29, true);
    let ix = f.index();
    let off = Preset::parse("postfilter").unwrap().kernel();
    let mut on = off.clone();
    on.set("packed_beam", "on").unwrap();
    let (a, b) = (Plan::new(&off, &ix).unwrap(), Plan::new(&on, &ix).unwrap());
    assert_eq!(b.fast_path(), super::search::FastPath::Postfilter);
    let mut st = Searcher::for_index(&ix, &[&a, &b]);
    for q in 0..f.preds.len() {
        for l in [5, 40] {
            let (x, cx) = search(&f, &mut st, &a, q, l, 10, None);
            let (y, cy) = search(&f, &mut st, &b, q, l, 10, None);
            assert_eq!((x, cx), (y, cy), "postfilter q={q} L={l}");
        }
    }
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
        pairs: None,
        islands: None,
        ..f.index()
    };
    let route = khpad(&[
        ("exact", "planned"),
        ("crossover", "cost"),
        ("estimates", "sketch"),
        ("terminator", "posting_l(4)"),
        ("laziness", "heap_bound"),
        ("postfilter", "always"),
        ("postfilter_starts", "by_rate"),
    ]);
    let mut on = route.clone();
    on.set("packed_beam", "on").unwrap();
    let (a, b) = (
        Plan::new(&route, &ix).unwrap(),
        Plan::new(&on, &ix).unwrap(),
    );
    let mut st = Searcher::for_index(&ix, &[&a, &b]);
    let mut routed = 0;
    for q in 0..f.preds.len() {
        for kk in [1, 10] {
            let (x, cx) = search_ix(&f, &ix, &mut st, &a, q, 20, kk);
            let (y, cy) = search_ix(&f, &ix, &mut st, &b, q, 20, kk);
            assert_eq!((x, cx), (y, cy), "route q={q} K={kk}");
            routed += usize::from(cx.route == Route::Postfilter);
        }
    }
    assert!(routed > 0);
}

#[test]
fn row_prefetch_changes_no_answer_and_keeps_the_fast_paths() {
    let f = fixture(300, 23, true);
    let ix = f.index();
    let presets = [
        ("postfilter", super::search::FastPath::Postfilter),
        ("upstream-inline", super::search::FastPath::Inline),
    ];
    for (preset, fast) in presets {
        let off = Preset::parse(preset).unwrap().kernel();
        let mut on = off.clone();
        on.set("row_prefetch", "on").unwrap();
        let (a, b) = (Plan::new(&off, &ix).unwrap(), Plan::new(&on, &ix).unwrap());
        assert_eq!(
            b.fast_path(),
            fast,
            "{preset}: row_prefetch keeps the fast path"
        );
        let mut st = Searcher::for_index(&ix, &[&a, &b]);
        for q in 0..f.preds.len() {
            let (x, cx) = search(&f, &mut st, &a, q, 40, 10, None);
            let (y, cy) = search(&f, &mut st, &b, q, 40, 10, None);
            assert_eq!(x, y, "{preset} q={q}");
            assert_eq!(cx, cy, "{preset} counters q={q}");
        }
    }
    // The configurable loop (k-hp-ad) and the hybrid's post-filter route (every graph-routed
    // query takes it; an expensive exact path routes every query to the graph).
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
        pairs: None,
        islands: None,
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
    let mut routed = 0;
    for kernel in [khpad(&[]), route] {
        let mut on = kernel.clone();
        on.set("row_prefetch", "on").unwrap();
        let (a, b) = (
            Plan::new(&kernel, &ix).unwrap(),
            Plan::new(&on, &ix).unwrap(),
        );
        let mut st = Searcher::for_index(&ix, &[&a, &b]);
        for q in 0..f.preds.len() {
            let (x, cx) = search_ix(&f, &ix, &mut st, &a, q, 20, 10);
            let (y, cy) = search_ix(&f, &ix, &mut st, &b, q, 20, 10);
            assert_eq!(x, y, "q={q}");
            assert_eq!(cx, cy, "counters q={q}");
            routed += usize::from(cx.route == Route::Postfilter);
        }
    }
    assert!(routed > 0, "the post-filter route was taken");
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
    // With the A6 enumeration costs (`rowand` is a candidate).
    let costs = UnitCosts {
        enum_word_ns: 0.3,
        enum_bit_ns: 2.0,
        ..UnitCosts::default()
    };
    let sketches = crate::sketches::LabelSketches::build(&f.labels, 16, 3, 1);
    for with_bits in [false, true] {
        let ix = Index {
            bitsets: with_bits.then_some(&f.bitsets),
            costs: Some(&costs),
            sketches: Some(&sketches),
            pairs: None,
            islands: None,
            ..f.index()
        };
        for plans in [
            "scan",
            "df",
            "gallop",
            "bitmap",
            "rowand",
            "scan+df+gallop+bitmap",
            "scan+df+gallop+bitmap+rowand",
        ] {
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
            pairs: None,
            islands: None,
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
        exact::{run_planned, DfPipeline, ExactEval},
        materialize::Materialized,
    };
    let f = fixture(300, 9, false);
    let vectors = f.graph.vectors(&f.base).unwrap();
    let mut bits = vec![0u64; f.labels.n_labels()];
    let mut q = Compiled::new();
    let mut mat = Materialized::new(300, true);
    let mut answers = Answers::new();
    // Combination postings of the fixture (every useful pair fits), for the `pair` plan.
    let (px, _) = crate::pairs::PairIndex::build(&f.labels, 1 << 20, 1);
    let mut cover = crate::pairs::Cover::default();
    let mut pair_runs = 0usize;
    for qi in 0..f.preds.len() {
        q.compile(&f.labels, f.preds.get(qi), Weights::Unit, None, &mut bits);
        if q.is_empty() {
            q.clear_bits(&mut bits);
            continue;
        }
        q.split_static(&f.labels, &f.bitsets, 1, false);
        px.cover(
            q.n_clauses(),
            |j| q.clause(j),
            |l| f.labels.posting_len(l),
            &mut cover,
        );
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
                for (plan, pipelined) in ExactPlan::ALL
                    .into_iter()
                    .map(|p| (p, false))
                    .chain([(ExactPlan::DistanceFirst, true)])
                {
                    if plan == ExactPlan::Pair {
                        if !cover.valid {
                            continue;
                        }
                        pair_runs += 1;
                    }
                    for eval in [ExactEval::RowScan, ExactEval::StaticBits] {
                        let mut cx = EvalCtx::new(&f.labels, &q, &bits, &[], &[]);
                        if eval == ExactEval::StaticBits {
                            cx = cx.with_static(&f.bitsets);
                        }
                        cx = cx.with_pairs(&px, &cover);
                        answers.reset(Collector::Heap, k, k);
                        for &(d, id) in seed.iter() {
                            answers.offer_unique(d, id);
                        }
                        let mut ctr = Counters::default();
                        let mut df = DfPipeline::new();
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
                            pipelined.then_some(&mut df),
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
    assert!(pair_runs > 0, "the fixture exercises the pair plan");
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
            pairs: None,
            islands: None,
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
        pairs: None,
        islands: None,
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
        pairs: None,
        islands: None,
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
            pairs: None,
            islands: None,
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
        pairs: None,
        islands: None,
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
        pairs: None,
        islands: None,
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
        // Measured far beyond the table: no list is cut.
        effort_l_top: 1e9,
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
        pairs: None,
        islands: None,
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
        assert_eq!(ctr.list_capped, 0, "q={q}");
    }
    // Measured up to L = 50 only: the lists of selective queries (their growth beyond K' = 100
    // passes it) are cut there, and counted.
    let capped = UnitCosts {
        effort_l_top: 50.0,
        ..costs.clone()
    };
    let ix50 = Index {
        costs: Some(&capped),
        ..ix
    };
    let plan50 = Plan::new(&k, &ix50).unwrap();
    let mut st = Searcher::for_index(&ix50, &[&plan50]);
    let mut hits = 0;
    for q in 0..f.preds.len() {
        let (_, ctr) = search_ix(&f, &ix50, &mut st, &plan50, q, 700, 10);
        if ctr.m_hat == 0 {
            continue;
        }
        let kprime = 10.0 * 300.0 / ctr.m_hat as f64;
        let (l, raw) = capped.effort_list_capped(0.7, kprime).unwrap();
        let (l, raw) = ((l.ceil() as u64).min(300), (raw.ceil() as u64).min(300));
        assert_eq!(ctr.final_l, l + ctr.n_start, "q={q}");
        assert!(l <= 50, "q={q}");
        assert_eq!(ctr.list_capped, u64::from(raw > l), "q={q}");
        hits += usize::from(raw > l);
    }
    assert!(hits > 0);
    // A calibration without its largest measured list size plans nothing from the curve.
    let unknown = UnitCosts {
        effort_l_top: 0.0,
        ..costs.clone()
    };
    let ix0 = Index {
        costs: Some(&unknown),
        ..ix
    };
    let Err(e) = Plan::new(&k, &ix0) else {
        panic!("a calibration without effort_l_top is refused")
    };
    assert!(e.to_string().contains("effort_l_top"), "{e}");
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
        pairs: None,
        islands: None,
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
fn by_rate_starts_are_the_global_ones_for_broad_predicates() {
    // `postfilter_starts = by_rate`: the route of a broad predicate (estimated match rate at
    // least 1/2) is the `global` route, that of a narrow one the `configured` route.
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
        pairs: None,
        islands: None,
        ..f.index()
    };
    let route = |starts: &'static str| {
        khpad(&[
            ("exact", "planned"),
            ("crossover", "cost"),
            ("estimates", "sketch"),
            ("terminator", "posting_l(4)"),
            ("laziness", "heap_bound"),
            ("postfilter", "always"),
            ("postfilter_starts", starts),
        ])
    };
    let plans: Vec<Plan> = ["by_rate", "global", "configured"]
        .into_iter()
        .map(|s| Plan::new(&route(s), &ix).unwrap())
        .collect();
    let mut st = Searcher::for_index(&ix, &plans.iter().collect::<Vec<_>>());
    let (mut broad, mut narrow) = (0, 0);
    for q in 0..f.preds.len() {
        for kk in [1, 10] {
            let (out, ctr) = search_ix(&f, &ix, &mut st, &plans[0], q, 20, kk);
            if ctr.route != Route::Postfilter {
                continue;
            }
            let is_broad = ctr.m_hat as f64 / 300.0 >= super::cost::BROAD_RATE;
            let want = &plans[if is_broad { 1 } else { 2 }];
            let (wout, wctr) = search_ix(&f, &ix, &mut st, want, q, 20, kk);
            assert_eq!(out, wout, "q={q} K={kk} broad={is_broad}");
            assert_eq!(
                (ctr.n_dist, ctr.n_start, ctr.final_l),
                (wctr.n_dist, wctr.n_start, wctr.final_l),
                "q={q} K={kk} broad={is_broad}"
            );
            if is_broad {
                broad += 1;
            } else {
                narrow += 1;
            }
        }
    }
    assert!(broad > 0 && narrow > 0, "broad {broad}, narrow {narrow}");
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
        // Measured far beyond the table: no list is cut.
        effort_l_top: 1e9,
        ..UnitCosts::default()
    };
    let ix = Index {
        costs: Some(&costs),
        sketches: Some(&sketches),
        pairs: None,
        islands: None,
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

#[test]
fn by_rate_late_continues_from_closer_starts_and_never_loses() {
    // `by_rate_late`: the search of a broad predicate starts from the global starts (as
    // `by_rate`) and, once it ends, offers the configured starts. With none admitted (this
    // connected fixture), the answers and expansions are `by_rate`'s; any admitted start makes
    // it evaluate a superset of `by_rate`'s nodes, so its answers (the K nearest matches among
    // the evaluated nodes) are at least as close.
    use super::cost::UnitCosts;
    let f = fixture(300, 47, true);
    let sketches = crate::sketches::LabelSketches::build(&f.labels, 1000, 1, 1);
    let costs = UnitCosts {
        graph_alpha: 2.0,
        graph_beta: 10.0,
        pred_ns: vec![1e9; 6],
        dist_ns: vec![1e9; 6],
        ..UnitCosts::default()
    };
    let ix = Index {
        costs: Some(&costs),
        sketches: Some(&sketches),
        ..f.index()
    };
    let plans: Vec<Plan> = ["by_rate_late", "by_rate", "by_rate_both", "by_rate_both_k"]
        .into_iter()
        .map(|s| Plan::new(&postfilter_route(s), &ix).unwrap())
        .collect();
    let mut st = Searcher::for_index(&ix, &plans.iter().collect::<Vec<_>>());
    let (mut broad, mut second) = (0, 0);
    for q in 0..f.preds.len() {
        for kk in [1, 10] {
            for l in [1, 2, 20] {
                let (out, ctr) = search_ix(&f, &ix, &mut st, &plans[0], q, l, kk);
                let (want, wctr) = search_ix(&f, &ix, &mut st, &plans[1], q, l, kk);
                assert_eq!(ctr.route, wctr.route, "q={q} K={kk} L={l}");
                if ctr.route != Route::Postfilter {
                    continue;
                }
                assert!(out.len() >= want.len(), "q={q} K={kk} L={l}");
                for (i, (a, b)) in out.iter().zip(&want).enumerate() {
                    assert!(a.0 <= b.0, "q={q} K={kk} L={l}: answer {i} {a:?} vs {b:?}");
                }
                if ctr.n_late == 0 {
                    assert_eq!(ctr.n_expand, wctr.n_expand, "q={q} K={kk} L={l}");
                    if ctr.fallback == 0 {
                        assert_eq!(out, want, "q={q} K={kk} L={l}");
                    }
                }
                if ctr.n_dist > wctr.n_dist {
                    broad += 1;
                }
                // `by_rate_both(_k)`: the K nearest matches of `by_rate`'s search and a second
                // one (list size L, or K).
                for p in &plans[2..] {
                    let (both, bctr) = search_ix(&f, &ix, &mut st, p, q, l, kk);
                    assert!(both.len() >= want.len(), "q={q} K={kk} L={l}");
                    for (i, (a, b)) in both.iter().zip(&want).enumerate() {
                        assert!(a.0 <= b.0, "q={q} K={kk} L={l}: answer {i} {a:?} vs {b:?}");
                    }
                    if bctr.n_late == 0 {
                        assert_eq!(both, want, "q={q} K={kk} L={l}");
                    } else {
                        assert!(bctr.n_expand > wctr.n_expand, "q={q} K={kk} L={l}");
                        second += 1;
                    }
                }
            }
        }
    }
    assert!(broad > 0, "no broad predicate offered its starts");
    assert!(second > 0, "no second search");
}

fn postfilter_route(starts: &str) -> Kernel {
    khpad(&[
        ("exact", "planned"),
        ("crossover", "cost"),
        ("estimates", "sketch"),
        ("terminator", "posting_l(4)"),
        ("laziness", "heap_bound"),
        ("postfilter", "always"),
        ("postfilter_starts", starts),
    ])
}

#[test]
fn by_rate_late_rescues_a_search_that_missed_the_query() {
    // Two clusters without edges between them; the global start in the first. A query in the
    // second: the search from the global start never leaves the first cluster; the configured
    // (label) start in the second cluster is closer than its whole list, so `by_rate_late`
    // continues from it and returns the second cluster's points. A query in the first cluster:
    // no start admitted, the `by_rate` answers.
    use super::cost::UnitCosts;
    let (n, half) = (200usize, 100usize);
    let mut rng = Lcg(5);
    let data: Vec<u8> = (0..n * DIM)
        .map(|i| {
            let lo = if i / DIM < half { 0 } else { 24 };
            lo + rng.below(8) as u8
        })
        .collect();
    let base = Matrix::from_slice(&data, n, DIM);
    let queries = Matrix::from_slice(&[[28u8; DIM], [4u8; DIM]].concat(), 2, DIM);
    let mut graph = Graph::new(
        n,
        12,
        AuxNodes::new(Vec::new(), &Matrix::<u8>::zeros(0, DIM)).unwrap(),
    )
    .unwrap();
    for v in 0..n {
        let range = if v < half { 0..half } else { half..n };
        let mut cand: Vec<(f32, u32)> = range
            .clone()
            .filter(|&u| u != v)
            .map(|u| (u8::squared_l2(base.row(v), base.row(u)), u as u32))
            .collect();
        cand.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut nbrs: Vec<u32> = cand.iter().take(8).map(|c| c.1).collect();
        let next = (if v + 1 == range.end {
            range.start
        } else {
            v + 1
        }) as u32;
        if !nbrs.contains(&next) {
            nbrs.push(next);
        }
        graph.set_neighbors(v as u32, &nbrs).unwrap();
    }
    graph.set_starts(vec![0]).unwrap();
    let mut b = LabelStoreBuilder::new();
    for i in 0..n {
        let c = format!("c={}", i % 2);
        b.push_point([("x", "x=1"), ("c", c.as_str())]).unwrap();
    }
    let labels = b.finish().unwrap();
    let mut pb = PredicateSetBuilder::new(&labels);
    pb.push(&[vec!["x=1"]]);
    pb.push(&[vec!["x=1"]]);
    let preds = pb.finish();
    let mut starts = Starts::new(&graph);
    let x = labels.name_index()["x=1"];
    starts.set_label_starts(&labels, &[(x, 150)]).unwrap();
    let bitsets = LabelBitsets::build(&labels, 1);
    let sketches = crate::sketches::LabelSketches::build(&labels, 1000, 1, 1);
    let costs = UnitCosts {
        graph_alpha: 2.0,
        graph_beta: 10.0,
        pred_ns: vec![1e9; 6],
        dist_ns: vec![1e9; 6],
        ..UnitCosts::default()
    };
    let ix = Index {
        graph: &graph,
        vectors: graph.vectors(&base).unwrap(),
        labels: &labels,
        global: graph.starts(),
        label_starts: Some(&starts),
        sample: None,
        label_view: false,
        view_postings: None,
        eps_min: 1.0,
        bitsets: Some(&bitsets),
        costs: Some(&costs),
        sketches: Some(&sketches),
        pairs: None,
        islands: None,
        pair_roots: None,
    };
    let plans: Vec<Plan> = ["by_rate_late", "by_rate", "by_rate_both", "by_rate_both_k"]
        .into_iter()
        .map(|s| Plan::new(&postfilter_route(s), &ix).unwrap())
        .collect();
    let mut st = Searcher::for_index(&ix, &plans.iter().collect::<Vec<_>>());
    let run = |st: &mut Searcher, p: &Plan, qi: usize| {
        let mut ids = vec![0u32; 10];
        let mut d = vec![0f32; 10];
        let q = Query {
            vector: queries.row(qi),
            cnf: preds.get(qi),
            view: None,
        };
        let ctr = st.search(p, &ix, &q, 10, &mut ids, &mut d, None);
        (ids[..ctr.n_out as usize].to_vec(), ctr)
    };
    // The query in the second cluster.
    let (late, lctr) = run(&mut st, &plans[0], 0);
    let (plain, pctr) = run(&mut st, &plans[1], 0);
    assert_eq!(
        (lctr.route, pctr.route),
        (Route::Postfilter, Route::Postfilter)
    );
    assert!(plain.iter().all(|&v| (v as usize) < half), "{plain:?}");
    assert_eq!(late.len(), 10);
    assert!(late.iter().all(|&v| (v as usize) >= half), "{late:?}");
    assert!(lctr.n_late >= 1 && lctr.n_expand > pctr.n_expand);
    for p in &plans[2..] {
        let (both, bctr) = run(&mut st, p, 0);
        assert!(both.iter().all(|&v| (v as usize) >= half), "{both:?}");
        assert!(bctr.n_late >= 1);
    }
    // The query in the first cluster: no start admitted; the second search finds only farther
    // points of the second cluster, so the answers are `by_rate`'s.
    let (late, lctr) = run(&mut st, &plans[0], 1);
    let (plain, _) = run(&mut st, &plans[1], 1);
    assert_eq!(lctr.n_late, 0);
    assert_eq!(late, plain);
    for p in &plans[2..] {
        let (both, _) = run(&mut st, p, 1);
        assert_eq!(both, plain);
    }
}

#[test]
fn by_rate_both_k_second_search_has_list_size_k() {
    use super::config::PostfilterStarts;
    let p = PostfilterStarts::parse("by_rate_both_k").unwrap();
    assert_eq!(p.to_string(), "by_rate_both_k");
    assert!(p.both_at(0.9) && !p.both_at(0.1));
    assert_eq!(p.second_list(640, 10), 10);
    assert_eq!(p.second_list(5, 10), 5);
    assert_eq!(PostfilterStarts::ByRateBoth.second_list(640, 10), 640);
}

#[test]
fn inline_rank_plans_the_inline_search_at_the_kth_match_rank() {
    // `inline_rank`: an inline graph search (`posting_l`) is planned at a list size whose
    // evaluated nodes reach the K-th match's rank at the estimated match rate (as the post-filter
    // route); without it, `L₀ × m(M̂ / n)` only. The exact path is priced out here, so every query
    // takes the inline search.
    use super::cost::{kth_match_rank, UnitCosts, Z};
    let f = fixture(300, 47, true);
    let sketches = crate::sketches::LabelSketches::build(&f.labels, 1000, 1, 1);
    let costs = UnitCosts {
        graph_alpha: 2.0,
        graph_beta: 10.0,
        pred_ns: vec![1e9; 6],
        dist_ns: vec![1e9; 6],
        ..UnitCosts::default()
    };
    let ix = Index {
        costs: Some(&costs),
        sketches: Some(&sketches),
        ..f.index()
    };
    let route = |rank: &'static str| {
        khpad(&[
            ("exact", "planned"),
            ("crossover", "cost"),
            ("estimates", "sketch"),
            ("terminator", "posting_l(4)"),
            ("laziness", "heap_bound"),
            ("inline_rank", rank),
        ])
    };
    let plans: Vec<Plan> = ["on", "off"]
        .into_iter()
        .map(|s| Plan::new(&route(s), &ix).unwrap())
        .collect();
    let mut st = Searcher::for_index(&ix, &plans.iter().collect::<Vec<_>>());
    let mut raised = 0;
    for q in 0..f.preds.len() {
        for kk in [1, 10] {
            let (_, ctr) = search_ix(&f, &ix, &mut st, &plans[0], q, 4, kk);
            let (_, off) = search_ix(&f, &ix, &mut st, &plans[1], q, 4, kk);
            if ctr.route != Route::Graph || off.route != Route::Graph || ctr.m_hat == 0 {
                continue;
            }
            let rate = ctr.m_hat as f64 / 300.0;
            let need = costs
                .graph_l_for(kth_match_rank(kk, rate, Z))
                .ceil()
                .min(300.0) as u64;
            assert!(ctr.final_l + 1 >= need, "q={q} K={kk}");
            assert!(ctr.final_l >= off.final_l, "q={q} K={kk}");
            if ctr.final_l > off.final_l {
                raised += 1;
            }
        }
    }
    assert!(raised > 0, "no list raised");
}

#[test]
fn df_pipeline_checks_and_answers_as_the_direct_loop() {
    use super::{
        collect::{Answers, Collector},
        cost::ExactPlan,
        eval::{Compiled, EvalCtx, Weights},
        exact::{run_planned, DfPipeline, ExactEval},
        materialize::Materialized,
    };
    // More candidates than a pipeline block, so that checks wait across blocks.
    let f = fixture(1000, 5, false);
    let vectors = f.graph.vectors(&f.base).unwrap();
    let mut bits = vec![0u64; f.labels.n_labels()];
    let mut q = Compiled::new();
    let mut mat = Materialized::new(1000, true);
    let mut answers = Answers::new();
    let mut df = DfPipeline::new();
    let mut multi = 0;
    for qi in 0..f.preds.len() {
        q.compile(&f.labels, f.preds.get(qi), Weights::Unit, None, &mut bits);
        if q.is_empty() {
            q.clear_bits(&mut bits);
            continue;
        }
        q.split_static(&f.labels, &f.bitsets, 1, false);
        multi += usize::from(q.n_clauses() > 1);
        for k in [1usize, 10, 50] {
            for eval in [ExactEval::RowScan, ExactEval::StaticBits] {
                let mut cx = EvalCtx::new(&f.labels, &q, &bits, &[], &[]);
                if eval == ExactEval::StaticBits {
                    cx = cx.with_static(&f.bitsets);
                }
                let mut run = |pipelined: bool| {
                    let mut ctr = Counters::default();
                    answers.reset(Collector::Heap, k, k);
                    run_planned(
                        ExactPlan::DistanceFirst,
                        eval,
                        &cx,
                        &vectors,
                        1000,
                        f.queries.row(qi),
                        k,
                        &mut mat,
                        &mut answers,
                        false,
                        pipelined.then_some(&mut df),
                        &mut ctr,
                    );
                    let mut ids = vec![0u32; k];
                    let mut ds = vec![0f32; k];
                    let n = answers.finish(&mut ids, &mut ds);
                    (ids[..n].to_vec(), ds[..n].to_vec(), ctr.n_pred, ctr.n_dist)
                };
                let direct = run(false);
                assert_eq!(run(true), direct, "q={qi} k={k} {eval:?}");
                let truth: Vec<u32> = f.truth(qi, k).iter().map(|&(_, id)| id).collect();
                assert_eq!(direct.0, truth, "q={qi} k={k} {eval:?}");
            }
        }
        q.clear_bits(&mut bits);
    }
    assert!(multi > 0, "the fixture has queries with residual clauses");
}

#[test]
fn threshold_on_estimate_routes_on_the_sketch_estimate() {
    use super::cost::UnitCosts;
    // Complete sketches make the estimate exact: the graph route iff M ≥ T, else the exact path.
    let f = fixture(300, 59, false);
    let sketches = crate::sketches::LabelSketches::build(&f.labels, 1000, 1, 1);
    let costs = UnitCosts::default();
    let ix = Index {
        costs: Some(&costs),
        sketches: Some(&sketches),
        pairs: None,
        ..f.index()
    };
    let counts: Vec<usize> = (0..f.preds.len())
        .map(|q| {
            (0..300u32)
                .filter(|&p| f.preds.get(q).matches(f.labels.row(p)))
                .count()
        })
        .collect();
    let mut sorted = counts.clone();
    sorted.sort_unstable();
    let t = sorted[sorted.len() / 2].max(1);
    let base = [
        ("exact", "planned(0)"),
        ("estimates", "sketch"),
        ("terminator", "posting_l(4)"),
        ("laziness", "heap_bound"),
    ];
    let mut est = base.to_vec();
    let threshold = format!("planned({t})");
    est[0] = ("exact", threshold.as_str());
    est.push(("threshold_on", "estimate"));
    let mut bound = est.clone();
    bound.pop();
    let mut bound_explicit = bound.clone();
    bound_explicit.push(("threshold_on", "bound"));
    let (ke, kb, kx) = (khpad(&est), khpad(&bound), khpad(&bound_explicit));
    assert_eq!(ke.to_json()["threshold_on"], "estimate");
    assert_eq!(kb.to_json(), kx.to_json());
    assert_eq!(kb.to_json().get("threshold_on"), None);
    let (pe, pb, px) = (
        Plan::new(&ke, &ix).unwrap(),
        Plan::new(&kb, &ix).unwrap(),
        Plan::new(&kx, &ix).unwrap(),
    );
    let mut st = Searcher::for_index(&ix, &[&pe, &pb, &px]);
    let (mut exact, mut graph) = (0, 0);
    for (q, &m) in counts.iter().enumerate() {
        for l in [10, 30] {
            // Off (the default or explicit `bound`): answers and counters are identical.
            let (ob, cb) = search_ix(&f, &ix, &mut st, &pb, q, l, 10);
            let (ox, cx) = search_ix(&f, &ix, &mut st, &px, q, l, 10);
            assert_eq!((ob, cb), (ox, cx), "q={q} L={l}");
            let (oe, ce) = search_ix(&f, &ix, &mut st, &pe, q, l, 10);
            if m == 0 {
                continue;
            }
            assert_eq!(
                ce.m_hat, m as u64,
                "q={q}: the estimate of a complete sketch"
            );
            if m < t {
                assert_eq!(ce.route, Route::Exact, "q={q}: M = {m} < T = {t}");
                assert_eq!(oe, f.truth(q, 10), "q={q}");
                exact += 1;
            } else {
                assert_eq!(ce.route, Route::Graph, "q={q}: M = {m} >= T = {t}");
                check_properties(&f, q, &oe, "threshold_on = estimate");
                graph += 1;
            }
        }
    }
    assert!(exact > 0 && graph > 0, "both routes: {exact} / {graph}");
    // It needs the threshold crossover, sketch estimates and a threshold.
    for bad in [
        &[("threshold_on", "estimate")][..],
        &[
            ("threshold_on", "estimate"),
            ("estimates", "sketch"),
            ("exact", "planned"),
            ("crossover", "cost"),
        ][..],
        &[("threshold_on", "estimate"), ("exact", "planned(10)")][..],
    ] {
        let mut k = Kernel::default();
        for (key, value) in bad {
            k.set(key, value).unwrap();
        }
        assert!(k.validate().is_err(), "{bad:?}");
    }
    assert!(Kernel::default().set("threshold_on", "sum").is_err());
}

#[test]
fn postfilter_always_under_threshold_is_the_route_at_the_fixed_list() {
    use super::cost::UnitCosts;
    // The threshold router with `postfilter = "always"`: the same exact side, and the graph side
    // is the post-filter route at the effort's list (heap(K) over the evaluated nodes).
    let f = fixture(300, 61, false);
    let sketches = crate::sketches::LabelSketches::build(&f.labels, 1000, 1, 1);
    let costs = UnitCosts::default();
    let ix = Index {
        costs: Some(&costs),
        sketches: Some(&sketches),
        pairs: None,
        ..f.index()
    };
    let inline = [("exact", "planned(150)"), ("laziness", "heap_bound")];
    let mut pf = inline.to_vec();
    pf.push(("postfilter", "always"));
    let mut conf = pf.clone();
    conf.push(("postfilter_starts", "configured"));
    let mut est = pf.clone();
    est.extend([
        ("threshold_on", "estimate"),
        ("estimates", "sketch"),
        ("postfilter_starts", "by_rate_both_k"),
    ]);
    let kernels = [khpad(&inline), khpad(&pf), khpad(&conf), khpad(&est)];
    let plans: Vec<Plan> = kernels.iter().map(|k| Plan::new(k, &ix).unwrap()).collect();
    let refs: Vec<&Plan> = plans.iter().collect();
    let mut st = Searcher::for_index(&ix, &refs);
    let (mut exact, mut routed, mut second) = (0, 0, 0);
    for q in 0..f.preds.len() {
        for l in [20, 300] {
            let (oi, ci) = search_ix(&f, &ix, &mut st, &plans[0], q, l, 10);
            let (op, cp) = search_ix(&f, &ix, &mut st, &plans[1], q, l, 10);
            if ci.route != Route::Graph {
                assert_eq!((&op, &cp), (&oi, &ci), "q={q} L={l}: the exact side");
                exact += usize::from(ci.route == Route::Exact);
                continue;
            }
            assert_eq!(ci.route, Route::Graph, "q={q}");
            assert_eq!(cp.route, Route::Postfilter, "q={q}");
            check_properties(&f, q, &op, "threshold + postfilter");
            if l == 300 {
                // The whole graph evaluated: the K nearest matches.
                assert_eq!(op, f.truth(q, 10), "q={q}");
            }
            routed += 1;
            let (oc, cc) = search_ix(&f, &ix, &mut st, &plans[2], q, l, 10);
            assert_eq!(cc.route, Route::Postfilter, "q={q}");
            check_properties(&f, q, &oc, "configured starts");
            let (oe, ce) = search_ix(&f, &ix, &mut st, &plans[3], q, l, 10);
            check_properties(&f, q, &oe, "estimate + by_rate_both_k");
            second += usize::from(ce.route == Route::Postfilter && ce.n_late > 0);
        }
    }
    assert!(exact > 0 && routed > 0, "both routes: {exact} / {routed}");
    assert!(second > 0, "no second search");
    // Only `always`; rate-dependent starts need the estimate.
    for bad in [
        &[("exact", "planned(10)"), ("postfilter", "cost")][..],
        &[
            ("exact", "planned(10)"),
            ("postfilter", "always"),
            ("postfilter_starts", "by_rate"),
        ][..],
    ] {
        let mut k = Kernel::default();
        for (key, value) in bad {
            k.set(key, value).unwrap();
        }
        assert!(k.validate().is_err(), "{bad:?}");
    }
}

#[test]
fn static_bitsets_off_is_an_index_without_them() {
    use super::cost::UnitCosts;
    // The hybrid (exact path, probe, post-filter route) with `static_bitsets = "off"` on an
    // index with the static bitsets answers and counts as on an index without them.
    let f = fixture(300, 67, false);
    let sketches = crate::sketches::LabelSketches::build(&f.labels, 1000, 1, 1);
    let costs = UnitCosts::default();
    let with = Index {
        costs: Some(&costs),
        sketches: Some(&sketches),
        pairs: None,
        ..f.index()
    };
    let without = Index {
        bitsets: None,
        ..with
    };
    let base = [
        ("exact", "planned"),
        ("crossover", "cost"),
        ("estimates", "sketch"),
        ("terminator", "posting_l(4)"),
        ("laziness", "heap_bound"),
        ("postfilter", "cost"),
    ];
    let mut off = base.to_vec();
    off.push(("static_bitsets", "off"));
    let (kb, ko) = (khpad(&base), khpad(&off));
    assert_eq!(ko.to_json()["static_bitsets"], "off");
    assert_eq!(kb.to_json().get("static_bitsets"), None);
    let (pb, po, pw) = (
        Plan::new(&kb, &with).unwrap(),
        Plan::new(&ko, &with).unwrap(),
        Plan::new(&kb, &without).unwrap(),
    );
    let mut st = Searcher::for_index(&with, &[&pb, &po, &pw]);
    let mut differ = 0;
    for q in 0..f.preds.len() {
        for l in [20, 60] {
            let (oo, co) = search_ix(&f, &with, &mut st, &po, q, l, 10);
            let (ow, cw) = search_ix(&f, &without, &mut st, &pw, q, l, 10);
            assert_eq!((&oo, &co), (&ow, &cw), "q={q} L={l}");
            let (_, cb) = search_ix(&f, &with, &mut st, &pb, q, l, 10);
            differ += usize::from(cb != co);
        }
    }
    assert!(differ > 0, "the bitsets were used without the flag");
    let mut k = Kernel::default();
    k.set("static_bitsets", "off").unwrap();
    k.validate().unwrap();
    k.set("predicate", "static_bits(1)").unwrap();
    assert!(k.validate().is_err());
    assert!(Kernel::default().set("static_bitsets", "no").is_err());
}

#[test]
fn projection_smallest_navigates_on_the_binding_clause() {
    // `projection(c*)` equals `projection(<category>)` of the clause with the smallest U_c when
    // that clause is the only one of its category (every non-empty query of the fixture).
    let f = fixture(300, 21, false);
    let ix = f.index();
    let plan = |admission: &str| {
        let mut k = Kernel::default();
        for (key, value) in [
            ("admission", admission),
            ("collector", "heap"),
            ("starts", "label(c*)+global"),
            ("expander", "two_hop(3)"),
        ] {
            k.set(key, value).unwrap();
        }
        Plan::new(&k, &ix).unwrap()
    };
    let star = plan("projection(c*)");
    let by_category: Vec<(String, Plan)> = ["a", "b", "c"]
        .iter()
        .map(|c| ((*c).to_owned(), plan(&format!("projection({c})"))))
        .collect();
    let mut plans = vec![&star];
    plans.extend(by_category.iter().map(|(_, p)| p));
    let mut st = Searcher::for_index(&ix, &plans);
    let (mut checked, mut multi) = (0, 0);
    for q in 0..f.preds.len() {
        let cnf = f.preds.get(q);
        if cnf.n_clauses() == 0 || cnf.clauses().any(|c| c.is_empty()) {
            continue;
        }
        let mut best = (u64::MAX, 0);
        for (i, clause) in cnf.clauses().enumerate() {
            let u: u64 = clause.iter().map(|&l| f.labels.posting_len(l) as u64).sum();
            if u < best.0 {
                best = (u, i);
            }
        }
        let cat = f
            .labels
            .category_name(f.labels.category(cnf.clause(best.1)[0]));
        let same_category = cnf
            .clauses()
            .filter(|c| f.labels.category_name(f.labels.category(c[0])) == cat)
            .count();
        assert_eq!(same_category, 1);
        let pk = &by_category.iter().find(|(c, _)| c == cat).unwrap().1;
        multi += usize::from(cnf.n_clauses() > 1);
        for l in [10, 30] {
            let (oa, ca) = search(&f, &mut st, &star, q, l, 10, None);
            let (ob, mut cb) = search(&f, &mut st, pk, q, l, 10, None);
            if cnf.n_clauses() == 1 {
                // A single clause is the whole predicate: no projection, so no re-check of the
                // navigation matches (the category projection re-checks them).
                assert!(ca.n_pred <= cb.n_pred);
                cb.n_pred = ca.n_pred;
            }
            assert_eq!((&oa, &ca), (&ob, &cb), "q={q} L={l} c*={cat}");
            for &(_, v) in &oa {
                let row = f.labels.row(v);
                assert!(
                    cnf.clauses().all(|c| c.iter().any(|l| row.contains(l))),
                    "q={q}: {v} is no match"
                );
            }
            checked += 1;
        }
    }
    assert!(checked > 40 && multi > 10, "{checked} {multi}");
}

#[test]
fn island_search_with_a_list_of_the_whole_island_is_exact() {
    use crate::{
        build::{BuildParams, PruneRule},
        islands::IslandIndex,
    };
    let f = fixture(300, 13, false);
    let all: Vec<u32> = (0..f.labels.n_labels() as u32)
        .filter(|&l| f.labels.posting_len(l) >= 2)
        .collect();
    let params = BuildParams {
        rule: PruneRule::Alpha { r: 8, alpha: 1.2 },
        l_build: 16,
        ..BuildParams::default()
    };
    let (isl, _, _) = IslandIndex::build(&f.base, &f.labels, &all, &params, 2).unwrap();
    let ix = Index {
        islands: Some(&isl),
        ..f.index()
    };
    let mut st = Searcher::for_index(&ix, &[]);
    let mut checked = 0;
    for qi in 0..f.preds.len() {
        for k in [1usize, 5, 10] {
            let (mut ids, mut ds) = (vec![0u32; k], vec![0f32; k]);
            if let Some((n, ctr)) = st.island_for_test(&ix, &f.query(qi), |s| s, &mut ids, &mut ds)
            {
                let got: Vec<(f32, u32)> = ds[..n]
                    .iter()
                    .copied()
                    .zip(ids[..n].iter().copied())
                    .collect();
                assert_eq!(got, f.truth(qi, k), "q={qi} k={k}");
                assert_eq!(ctr.route, Route::Island);
                checked += 1;
            }
        }
    }
    assert!(
        checked > 0,
        "some query has a single-label clause with an island"
    );
}

/// The fixture's vectors and graph with labels in 72 categories: `s` (2 values) and 71
/// categories `kJ` whose points carry `kJ=0` (9 in 10) and `kJ=1` (3 in 10; every point has
/// one of them): every label is big enough for combination postings (64 points). Queries alternate narrow ones (2 to 4 clauses) and wide ones (more than 64
/// clauses: every `kJ` category, most clauses `[kJ=0, kJ=1]`, a few `[kJ=0]` or `[kJ=1]`).
fn wide_fixture(n: usize, seed: u64) -> Fixture {
    let f = fixture(n, seed, false);
    let mut rng = Lcg(seed ^ 0x5eed);
    let cats = 71;
    let cat: Vec<String> = (0..cats).map(|j| format!("k{j}")).collect();
    let val: Vec<[String; 2]> = (0..cats)
        .map(|j| [format!("k{j}=0"), format!("k{j}=1")])
        .collect();
    let s_names: Vec<String> = (0..2).map(|v| format!("s={v}")).collect();
    let mut b = LabelStoreBuilder::new();
    for _ in 0..n {
        let mut row: Vec<(&str, &str)> = vec![("s", s_names[rng.below(2) as usize].as_str())];
        for (c, v) in cat.iter().zip(&val) {
            let r = rng.below(10);
            if r < 9 {
                row.push((c.as_str(), v[0].as_str()));
            }
            if r >= 7 {
                row.push((c.as_str(), v[1].as_str()));
            }
        }
        b.push_point(row).unwrap();
    }
    let labels = b.finish().unwrap();
    let mut pb = PredicateSetBuilder::new(&labels);
    for q in 0..40 {
        let mut clauses: Vec<Vec<String>> = vec![vec![s_names[rng.below(2) as usize].clone()]];
        if q % 2 == 0 {
            for _ in 0..1 + rng.below(3) {
                let j = rng.below(cats as u32) as usize;
                clauses.push(vec![val[j][rng.below(2) as usize].clone()]);
            }
        } else {
            let picked: Vec<usize> = (0..2 + rng.below(3))
                .map(|_| rng.below(cats as u32) as usize)
                .collect();
            for (j, v) in val.iter().enumerate() {
                if !picked.contains(&j) {
                    clauses.push(vec![v[0].clone(), v[1].clone()]);
                } else if q % 3 == 0 {
                    clauses.push(vec![v[1].clone()]);
                } else {
                    clauses.push(vec![v[0].clone()]);
                }
            }
        }
        pb.push(&clauses);
    }
    let preds = pb.finish();
    let mut starts = Starts::new(&f.graph);
    let pairs: Vec<(u32, u32)> = (0..labels.n_labels() as u32)
        .filter_map(|l| labels.postings(l).first().map(|&p| (l, p)))
        .collect();
    starts.set_label_starts(&labels, &pairs).unwrap();
    starts
        .set_sample((0..n as u32).filter(|p| p % 3 != 1).collect(), None)
        .unwrap();
    let sample = SampleIndex::build(&starts, &labels).unwrap();
    let bitsets = LabelBitsets::build(&labels, 1);
    Fixture {
        labels,
        preds,
        starts,
        sample,
        bitsets,
        ..f
    }
}

/// Queries of more than 64 clauses have no clause bits: on a searcher that alternates them
/// with narrow queries they inherit nothing (answers and counters as on a fresh searcher),
/// every check and price takes the mask-free path, and the answers of exhaustive searches are
/// the brute-force ones.
#[test]
fn wide_predicates_alternating_with_narrow_ones_are_exact() {
    use super::cost::UnitCosts;
    let n = 300;
    let f = wide_fixture(n, 61);
    assert!((0..f.preds.len()).any(|q| f.preds.get(q).n_clauses() > 64));
    let costs = UnitCosts {
        enum_word_ns: 0.3,
        enum_bit_ns: 2.0,
        ..UnitCosts::default()
    };
    let sketches = crate::sketches::LabelSketches::build(&f.labels, 16, 3, 1);
    // Every pair of labels stored.
    let (px, _) = crate::pairs::PairIndex::build(&f.labels, 1 << 26, 1);
    let ix = Index {
        costs: Some(&costs),
        sketches: Some(&sketches),
        pairs: Some(&px),
        ..f.index()
    };
    let hybrid = [
        ("starts", "label(c*)+global"),
        ("collector", "heap"),
        ("laziness", "heap_bound"),
        ("exact", "planned"),
        ("crossover", "cost"),
        ("estimates", "sketch"),
        ("postfilter", "cost"),
        ("postfilter_starts", "by_rate_both_k"),
        ("df_evals", "lower"),
    ];
    let kernels: Vec<(&str, Kernel)> = vec![
        (
            "exact pair",
            kernel(&[
                ("exact", "planned(18446744073709551615)"),
                ("exact_plans", "scan+pair"),
                ("collector", "heap"),
            ]),
        ),
        (
            "exact all plans",
            kernel(&[
                ("exact", "planned(18446744073709551615)"),
                ("estimates", "sketch"),
                ("collector", "heap"),
            ]),
        ),
        ("hybrid", kernel(&hybrid)),
        (
            "hybrid static bits",
            kernel(&[&hybrid[..], &[("predicate", "static_bits(1)")]].concat()),
        ),
        (
            "graph static bits",
            kernel(&[
                ("starts", "label(c*)+global"),
                ("collector", "heap"),
                ("predicate", "static_bits(1)"),
            ]),
        ),
    ];
    let (mut wide_pair, mut wide_runs) = (0, 0);
    for (name, k) in &kernels {
        let plan = Plan::new(k, &ix).unwrap();
        let mut shared = Searcher::for_index(&ix, &[&plan]);
        for q in 0..f.preds.len() {
            for kk in [1, 10] {
                let what = format!("{name} q={q} K={kk}");
                let mut fresh = Searcher::for_index(&ix, &[&plan]);
                let (a, ca) = search_ix(&f, &ix, &mut shared, &plan, q, n, kk);
                let (b, cb) = search_ix(&f, &ix, &mut fresh, &plan, q, n, kk);
                assert_eq!((&a, &ca), (&b, &cb), "{what}: state of an earlier query");
                assert_eq!(a, f.truth(q, kk), "{what}");
                if ca.n_clauses > 64 {
                    wide_runs += 1;
                    wide_pair += usize::from(ca.plan == super::cost::ExactPlan::Pair as u64);
                }
            }
        }
    }
    assert!(wide_runs > 0 && wide_pair > 0, "{wide_runs} {wide_pair}");
}

/// The cost-chosen member check of the tree route prices a query of more than 64 clauses on
/// the row (no clause bits), after and before a narrow query.
#[test]
fn the_cost_tree_route_takes_wide_predicates() {
    use super::cost::UnitCosts;
    // As there: a router (id 5) over the members 0, 2 and 4 of `x=1`; every point also
    // carries the 70 labels `zJ=1` (one category).
    let z: Vec<String> = (0..70).map(|j| format!("z{j}=1")).collect();
    let short: [&[&str]; 5] = [
        &["x=1", "y=1"],
        &["y=1"],
        &["x=1", "y=2"],
        &["y=1"],
        &["x=1", "y=1"],
    ];
    let row_refs: Vec<Vec<&str>> = short
        .iter()
        .map(|r| {
            r.iter()
                .copied()
                .chain(z.iter().map(String::as_str))
                .collect()
        })
        .collect();
    let row_slices: Vec<&[&str]> = row_refs.iter().map(Vec::as_slice).collect();
    let (base, mut g, labels) = line(&row_slices, true);
    g.set_neighbors(5, &[0, 2, 4]).unwrap();
    let id = |s: &str| labels.name_index()[s];
    let mut st = Starts::new(&g);
    st.set_label_starts(&labels, &[(id("x=1"), 5), (id("y=1"), 1), (id("y=2"), 2)])
        .unwrap();
    let bitsets = LabelBitsets::build(&labels, 1);
    let sketches = crate::sketches::LabelSketches::build(&labels, 1000, 1, 1);
    let mut pb = PredicateSetBuilder::new(&labels);
    pb.push(&[vec!["x=1"], vec!["y=1"]]);
    let wide: Vec<Vec<&str>> = std::iter::once(vec!["x=1"])
        .chain(z.iter().map(|l| vec![l.as_str()]))
        .collect();
    pb.push(&wide);
    let preds = pb.finish();
    let costs = UnitCosts {
        check_base_ns: vec![1.0; 6],
        check_word_ns: vec![1.0; 6],
        check_prefetch_ns: vec![0.0; 6],
        check_row_ns: vec![100.0; 6],
        word_ns: 1e6,
        scatter_ns: 0.0,
        ..UnitCosts::default()
    };
    let ix = Index {
        graph: &g,
        vectors: g.vectors(&base).unwrap(),
        labels: &labels,
        global: &[0],
        label_starts: Some(&st),
        sample: None,
        label_view: false,
        view_postings: None,
        eps_min: 0.5,
        bitsets: Some(&bitsets),
        costs: Some(&costs),
        sketches: Some(&sketches),
        pairs: None,
        islands: None,
        pair_roots: None,
    };
    let k = kernel(&[
        ("starts", "label(c*)+global"),
        ("collector", "heap"),
        ("crossover", "graph"),
        ("exact", "planned"),
        ("estimates", "sketch"),
        ("tree_route", "cost"),
    ]);
    let plan = Plan::new(&k, &ix).unwrap();
    let mut s = Searcher::for_index(&ix, &[&plan]);
    let mut answers = Vec::new();
    for qi in [0, 1, 0, 1] {
        let (mut ids, mut d) = (vec![0u32; 3], vec![0f32; 3]);
        let q = Query {
            vector: &[3.2f32],
            cnf: preds.get(qi),
            view: None,
        };
        let ctr = s.search(&plan, &ix, &q, 4, &mut ids, &mut d, None);
        assert_eq!(ctr.route, Route::Tree, "q={qi}");
        answers.push((ctr.plan, ids));
    }
    // The narrow query: the static check (cheap words, expensive rows); the wide one: the row
    // scan, whose price equals the static check's (both read the row) and comes first.
    assert_eq!(answers[0], (102, vec![4, 0, u32::MAX]));
    assert_eq!(answers[1], (101, vec![4, 2, 0]));
    assert_eq!(answers[2], answers[0]);
    assert_eq!(answers[3], answers[1]);
}

/// `price_second_search = "on"` adds the nominal cost of `by_rate_both_k`'s second search (a
/// list of `K` from the configured starts) to the post-filter price of broad predicates only,
/// and changes nothing else.
#[test]
fn price_second_search_prices_the_broad_post_filter_route_only() {
    use super::cost::UnitCosts;
    let f = fixture(300, 17, false);
    // An expensive exact path: the crossover prices the graph routes of both predicates.
    let costs = UnitCosts {
        dist_ns: vec![1e4; 6],
        pred_ns: vec![1e4; 6],
        ..UnitCosts::default()
    };
    let ix = Index {
        costs: Some(&costs),
        ..f.index()
    };
    let mut pb = PredicateSetBuilder::new(&f.labels);
    // Broad (half of the points: global starts, then the second search) and selective.
    pb.push(&[vec!["c=0"]]);
    pb.push(&[vec!["a=0"], vec!["b=1"]]);
    let preds = pb.finish();
    let edits = [
        ("starts", "label(c*)+global"),
        ("collector", "heap"),
        ("laziness", "heap_bound"),
        ("exact", "planned"),
        ("crossover", "cost"),
        ("postfilter", "always"),
        ("postfilter_starts", "by_rate_both_k"),
    ];
    let off = kernel(&edits);
    let on = kernel(&[&edits[..], &[("price_second_search", "on")]].concat());
    assert_eq!(on.to_json()["price_second_search"], "on");
    assert!(off.to_json().get("price_second_search").is_none());
    let mut wrong = kernel(&[]);
    wrong.set("price_second_search", "on").unwrap();
    assert!(wrong.validate().is_err());
    let run = |k: &Kernel, qi: usize| {
        let plan = Plan::new(k, &ix).unwrap();
        let mut st = Searcher::for_index(&ix, &[&plan]);
        let (mut ids, mut d) = (vec![0u32; 10], vec![0f32; 10]);
        let q = Query {
            vector: f.queries.row(0),
            cnf: preds.get(qi),
            view: None,
        };
        let ctr = st.search(&plan, &ix, &q, 10, &mut ids, &mut d, None);
        (ids, ctr)
    };
    let (a, ca) = run(&off, 0);
    let (b, cb) = run(&on, 0);
    let rate = f.labels.posting_len(f.labels.name_index()["c=0"]) as f64 / 300.0;
    assert!(rate >= super::cost::BROAD_RATE);
    // Two configured starts: c=0's label start and the global one.
    let second = costs.postfilter_cost(10.0, 10, rate) + 2.0 * costs.graph_dist_ns;
    let extra = cb.est_graph_ns as f64 - ca.est_graph_ns as f64;
    assert!((extra - second).abs() <= 1.0, "{extra} vs {second}");
    if ca.route == cb.route {
        assert_eq!(a, b);
    }
    // The selective predicate searches from its label starts: no second search, no change.
    assert_eq!(run(&off, 1), run(&on, 1));
}
