/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The kernel's upstream presets against upstream's own search code on the same graph: build a
//! small index with `diskann-providers`, import it, and run `Knn` (post-filtered),
//! `InlineFilterSearch` (± adaptive L), `MultihopFilterSearch` and `Knn` under `BetaFilter`
//! next to the presets `postfilter`, `upstream-inline(-adaptive)`, `upstream-multihop` and
//! `upstream-beta`. Results must agree on every slot strictly closer than the K-th distance
//! and in their distances (ties may be ordered differently), and the distance and expansion
//! counts must be equal.

use std::{
    io::{Cursor, Read},
    sync::Arc,
};

use diskann::{
    graph::{
        config,
        ext::labeled::{Filtered, QueryLabelProvider},
        search::{AdaptiveL, InlineFilterSearch, Knn, MultihopFilterSearch},
        IdDistance, SearchOutputBuffer, StartPointStrategy,
    },
    neighbor::{BackInserter, Neighbor},
    provider::DefaultContext,
    utils::ONE,
};
use diskann_filter_lab::{
    diskann_io,
    kernel::{
        search::{Index, Plan, Query, Searcher},
        Preset,
    },
    labels::{LabelStore, LabelStoreBuilder},
    predicate::{PredicateSet, PredicateSetBuilder},
    vectors::{DType, Element, Matrix},
};
use diskann_providers::{
    index::diskann_async,
    model::graph::provider::{
        async_::{
            common::{FullPrecision, NoDeletes},
            inmem::{DefaultProviderParameters, SetStartPoints},
        },
        layers::BetaFilter,
    },
    storage::{AsyncIndexMetadata, SaveWith, StorageReadProvider, VirtualStorageProvider},
};
use diskann_utils::views::rowmajor::{self, Matrix as _};
use diskann_vector::distance::Metric;

/// Upstream's view of a predicate: the clauses, evaluated on our label rows.
#[derive(Debug)]
struct Pred {
    labels: Arc<LabelStore>,
    clauses: Vec<Vec<u32>>,
}

impl QueryLabelProvider<u32> for Pred {
    fn is_match(&self, i: u32) -> bool {
        (i as usize) < self.labels.n_points()
            && self
                .clauses
                .iter()
                .all(|c| diskann_filter_lab::predicate::intersects(self.labels.row(i), c))
    }
}

/// Checks two answer lists: same distances, same ids strictly below the K-th distance.
fn same_answers(ours: &[(f32, u32)], theirs: &[(f32, u32)], what: &str) {
    let mut t = theirs.to_vec();
    t.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let dk = |v: &[(f32, u32)]| v.last().map_or(f32::INFINITY, |x| x.0);
    let dist = |v: &[(f32, u32)]| v.iter().map(|x| x.0).collect::<Vec<_>>();
    assert_eq!(dist(ours), dist(&t), "{what}: distances");
    let below = |v: &[(f32, u32)]| {
        let mut ids: Vec<u32> = v.iter().filter(|x| x.0 < dk(v)).map(|x| x.1).collect();
        ids.sort_unstable();
        ids
    };
    assert_eq!(
        below(ours),
        below(&t),
        "{what}: ids below the K-th distance"
    );
}

#[test]
fn presets_reproduce_upstream_search() {
    let (n, dim, degree, max_degree) = (2000, 10, 16, 20);
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let data = rowmajor::Owned::from_fn(n, dim, |_| (next() >> 56) as u8);
    let config = config::Builder::new_with(
        degree,
        config::MaxDegree::new(max_degree),
        32,
        Metric::L2.into(),
        |b| {
            b.alpha(1.2);
        },
    )
    .build()
    .unwrap();
    let params = DefaultProviderParameters {
        max_points: n,
        frozen_points: ONE,
        metric: Metric::L2,
        dim,
        prefetch_lookahead: None,
        prefetch_cache_line_level: None,
        max_degree: max_degree as u32,
    };
    let index = diskann_async::new_index::<u8, _>(config, params, NoDeletes).unwrap();
    let medoid = StartPointStrategy::Medoid.compute(data.as_view()).unwrap();
    index.provider().set_start_points(medoid.rows()).unwrap();
    let storage = VirtualStorageProvider::new_memory();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    rt.block_on(async {
        for (i, row) in data.rows().enumerate() {
            index
                .insert(&FullPrecision, &DefaultContext, &(i as u32), row)
                .await
                .unwrap();
        }
        index
            .save_with(&storage, &AsyncIndexMetadata::new("/index"))
            .await
            .unwrap();
    });
    let file = |path: &str| {
        let mut bytes = Vec::new();
        storage
            .open_reader(path)
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        Cursor::new(bytes)
    };
    let (graph, _) =
        diskann_io::read_index(file("/index"), Some((file("/index.data"), DType::U8))).unwrap();
    let base = Matrix::<u8>::from_slice(data.as_slice(), n, dim);

    // Labels: category a (12 values, 1-2 per point), b (3 values), c (parity).
    let mut b = LabelStoreBuilder::new();
    for i in 0..n {
        let a1 = format!("a={}", next() % 12);
        let a2 = format!("a={}", next() % 12);
        let bb = format!("b={}", next() % 3);
        let c = format!("c={}", i % 2);
        let mut row = vec![("a", a1.as_str()), ("b", bb.as_str()), ("c", c.as_str())];
        if next() % 2 == 0 {
            row.push(("a", a2.as_str()));
        }
        b.push_point(row).unwrap();
    }
    let labels = Arc::new(b.finish().unwrap());
    let mut pb = PredicateSetBuilder::new(&labels);
    let n_queries = 30;
    for q in 0..n_queries {
        let mut clauses = vec![vec![
            format!("a={}", q % 12),
            format!("a={}", (q * 5 + 1) % 12),
        ]];
        if q % 2 == 0 {
            clauses.push(vec![format!("b={}", q % 3)]);
        }
        if q % 3 == 0 {
            clauses.push(vec![format!("c={}", q % 2)]);
        }
        pb.push(&clauses);
    }
    let preds: PredicateSet = pb.finish();
    let queries: Vec<Vec<u8>> = (0..n_queries)
        .map(|_| (0..dim).map(|_| (next() >> 56) as u8).collect())
        .collect();

    let ix = Index {
        graph: &graph,
        vectors: graph.vectors(&base).unwrap(),
        labels: &labels,
        global: graph.starts(),
        label_starts: None,
        sample: None,
        label_view: false,
        view_postings: None,
        eps_min: 1.0,
        bitsets: None,
        costs: None,
        sketches: None,
    };
    let plan = |p: &str| Plan::new(&Preset::parse(p).unwrap().kernel(), &ix).unwrap();
    let postfilter = plan("postfilter");
    let inline = plan("upstream-inline");
    let adaptive = plan("upstream-inline-adaptive(20, 8)");
    let multihop = plan("upstream-multihop");
    let beta = plan("upstream-beta(0.5)");
    assert_eq!(multihop.two_hop_cap(), max_degree / 2);
    let mut st = Searcher::for_index(&ix, &[&postfilter]);
    let k = 10;

    let mut checked = 0;
    for (q, qv) in queries.iter().enumerate() {
        let cnf = preds.get(q);
        let pred = Arc::new(Pred {
            labels: labels.clone(),
            clauses: cnf.clauses().map(<[u32]>::to_vec).collect(),
        });
        let query = Query {
            vector: &qv[..],
            cnf,
            view: None,
        };
        let true_d = |id: u32| u8::squared_l2(qv, base.row(id as usize));
        for l in [10, 20, 50] {
            let ours = |st: &mut Searcher, p: &Plan| {
                let (mut ids, mut d) = (vec![0u32; k], vec![0f32; k]);
                let c = st.search(p, &ix, &query, l, &mut ids, &mut d, None);
                let out: Vec<(f32, u32)> = d
                    .iter()
                    .zip(&ids)
                    .take(c.n_out as usize)
                    .map(|(&d, &i)| (d, i))
                    .collect();
                (out, c)
            };
            let what = |name: &str| format!("{name} q={q} L={l}");
            let knn = Knn::new(l, None).unwrap();

            // Post-filter.
            let mut cand = vec![Neighbor::<u32>::default(); l];
            let mut out = BackInserter::new(&mut cand[..]);
            let s = rt
                .block_on(index.search(knn, &FullPrecision, &DefaultContext, &qv[..], &mut out))
                .unwrap();
            let got = out.current_len();
            let theirs: Vec<(f32, u32)> = cand[..got]
                .iter()
                .filter(|nb| pred.is_match(*nb.id()))
                .take(k)
                .map(|nb| (*nb.distance(), *nb.id()))
                .collect();
            let (mine, c) = ours(&mut st, &postfilter);
            same_answers(&mine, &theirs, &what("postfilter"));
            assert_eq!(c.n_dist, u64::from(s.cmps), "{}", what("postfilter cmps"));
            assert_eq!(c.n_expand, u64::from(s.hops), "{}", what("postfilter hops"));

            // Inline, without and with adaptive L.
            let filtered = Filtered::new(FullPrecision, &*pred);
            for (p, adaptive_l) in [
                (&inline, None),
                (&adaptive, Some(AdaptiveL::new(20, 8.0).unwrap())),
            ] {
                let (mut ids, mut ds) = (vec![0u32; k], vec![0f32; k]);
                let mut out = IdDistance::new(&mut ids, &mut ds);
                let s = rt
                    .block_on(index.search(
                        InlineFilterSearch::new(knn, adaptive_l),
                        &filtered,
                        &DefaultContext,
                        &qv[..],
                        &mut out,
                    ))
                    .unwrap();
                let got = out.current_len();
                let theirs: Vec<(f32, u32)> = ds[..got]
                    .iter()
                    .copied()
                    .zip(ids[..got].iter().copied())
                    .collect();
                let (mine, c) = ours(&mut st, p);
                let name = if p.kernel == adaptive.kernel {
                    "adaptive"
                } else {
                    "inline"
                };
                same_answers(&mine, &theirs, &what(name));
                assert_eq!(c.n_dist, u64::from(s.cmps) + 1, "{}", what(name));
                assert_eq!(c.n_expand, u64::from(s.hops), "{}", what(name));
            }

            // Multihop.
            let (mut ids, mut ds) = (vec![0u32; k], vec![0f32; k]);
            let mut out = IdDistance::new(&mut ids, &mut ds);
            let s = rt
                .block_on(index.search(
                    MultihopFilterSearch::new(knn),
                    &filtered,
                    &DefaultContext,
                    &qv[..],
                    &mut out,
                ))
                .unwrap();
            let got = out.current_len();
            let theirs: Vec<(f32, u32)> = ds[..got]
                .iter()
                .copied()
                .zip(ids[..got].iter().copied())
                .collect();
            let (mine, c) = ours(&mut st, &multihop);
            same_answers(&mine, &theirs, &what("multihop"));
            assert_eq!(c.n_dist, u64::from(s.cmps) + 1, "{}", what("multihop cmps"));
            assert_eq!(c.n_hops, u64::from(s.hops), "{}", what("multihop hops"));

            // Beta.
            let shared: Arc<dyn QueryLabelProvider<u32>> = pred.clone();
            let strategy = BetaFilter::new(FullPrecision, shared, 0.5);
            let mut cand = vec![Neighbor::<u32>::default(); l];
            let mut out = BackInserter::new(&mut cand[..]);
            let s = rt
                .block_on(index.search(knn, &strategy, &DefaultContext, &qv[..], &mut out))
                .unwrap();
            let got = out.current_len();
            let theirs: Vec<(f32, u32)> = cand[..got]
                .iter()
                .filter(|nb| pred.is_match(*nb.id()))
                .take(k)
                .map(|nb| (true_d(*nb.id()), *nb.id()))
                .collect();
            let (mine, c) = ours(&mut st, &beta);
            same_answers(&mine, &theirs, &what("beta"));
            assert_eq!(c.n_dist, u64::from(s.cmps), "{}", what("beta cmps"));
            assert_eq!(c.n_expand, u64::from(s.hops), "{}", what("beta hops"));
            checked += 1;
        }
    }
    assert_eq!(checked, 3 * n_queries);
}
