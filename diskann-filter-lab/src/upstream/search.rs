/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! One query through upstream's unmodified search algorithms.
//!
//! | [`Algorithm`] | upstream code | strategy | output |
//! |---|---|---|---|
//! | `KnnPostfilter` | [`Knn`] (unfiltered) | `FullPrecision` | the `L` candidates, post-filtered here to the best `K` matches |
//! | `Inline` | [`InlineFilterSearch`] without adaptive L | `labeled::Filtered<FullPrecision>` | best `K` matches |
//! | `InlineAdaptive` | [`InlineFilterSearch`] with [`AdaptiveL`] | `labeled::Filtered<FullPrecision>` | best `K` matches |
//! | `Multihop` | [`MultihopFilterSearch`] | `labeled::Filtered<FullPrecision>` | best `K` matches |
//! | `Beta` | [`Knn`] | `BetaFilter<FullPrecision>` | the `L` candidates (matches with distances scaled by β), post-filtered here to the best `K` matches with recomputed true distances |
//!
//! Every search uses beam width 1 and the strategy's default post-processor (start points and
//! deleted ids removed). Searches run on the calling thread: the in-memory provider's futures
//! complete without suspending, so they are polled with a no-op waker.

use std::{
    future::Future,
    sync::Arc,
    task::{Context, Poll, Waker},
};

use anyhow::Result;
use diskann::{
    graph::{
        ext::labeled::{Filtered, QueryLabelProvider},
        index::SearchStats,
        search::{AdaptiveL, InlineFilterSearch, Knn, MultihopFilterSearch},
        strategy::FullPrecision,
        DiskANNIndex, IdDistance, SearchOutputBuffer,
    },
    neighbor::{BackInserter, Neighbor},
    provider::DefaultContext,
    utils::VectorRepr,
};
use diskann_providers::model::graph::provider::layers::BetaFilter;
use diskann_vector::{distance::Metric, PreprocessedDistanceFunction};

use super::{filter::PredicateFilter, index::Provider};

/// Upstream filtered-search algorithm run by the adapter.
#[derive(Debug, Clone)]
pub enum Algorithm {
    KnnPostfilter,
    Inline,
    InlineAdaptive {
        sample_count: usize,
        scale_factor: f64,
        params: AdaptiveL,
    },
    Multihop,
    Beta {
        beta: f32,
    },
}

impl Algorithm {
    /// Inline search with upstream's adaptive L (validated by upstream).
    pub fn inline_adaptive(sample_count: usize, scale_factor: f64) -> Result<Self> {
        Ok(Self::InlineAdaptive {
            sample_count,
            scale_factor,
            params: AdaptiveL::new(sample_count, scale_factor)?,
        })
    }

    /// Beta search; upstream's benchmark accepts β in (0, 1].
    pub fn beta(beta: f32) -> Result<Self> {
        anyhow::ensure!(beta > 0.0 && beta <= 1.0, "beta must be in (0, 1]");
        Ok(Self::Beta { beta })
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::KnnPostfilter => "knn-postfilter",
            Self::Inline => "inline",
            Self::InlineAdaptive { .. } => "inline-adaptive",
            Self::Multihop => "multihop",
            Self::Beta { .. } => "beta",
        }
    }

    /// Whether the adapter post-filters the `L` candidates of an unfiltered-style search.
    pub fn post_filters(&self) -> bool {
        matches!(self, Self::KnnPostfilter | Self::Beta { .. })
    }
}

/// Counters of one query.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QueryStats {
    /// Results written (`<= K`).
    pub n_out: u32,
    /// Upstream `SearchStats::cmps`: distance computations of the search. Includes the start
    /// points for `Knn` (also under beta) but not for inline and multihop search.
    pub up_cmps: u32,
    /// Upstream `SearchStats::hops`: expanded nodes, plus second-hop adjacency fetches for
    /// multihop.
    pub up_hops: u32,
    /// Upstream `SearchStats::result_count`: results written by the post-processor.
    pub up_results: u32,
    /// Distance computations made by the adapter (beta: true distances of the returned matches).
    pub extra_dist: u32,
    /// Predicate evaluations made by the adapter's post-filter.
    pub post_pred: u32,
}

/// Polls a future that never suspends (the in-memory provider's futures are ready
/// immediately); spins with `yield_now` if it ever returns `Pending`.
pub fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut cx) {
            return output;
        }
        std::thread::yield_now();
    }
}

/// Runs queries against one index, reusing an output buffer of the largest `L`.
pub struct Searcher<'a, T: VectorRepr> {
    index: &'a DiskANNIndex<Provider<T>>,
    candidates: Vec<Neighbor<u32>>,
}

impl<'a, T: VectorRepr> Searcher<'a, T> {
    pub fn new(index: &'a DiskANNIndex<Provider<T>>, max_l: usize) -> Self {
        Self {
            index,
            candidates: vec![Neighbor::default(); max_l],
        }
    }

    /// Searches `query` with list size `l` for the `ids.len()` best matches of `filter`'s
    /// compiled predicate. Writes ids and true squared-L2 distances in increasing distance,
    /// padding unused slots with `u32::MAX` / `f32::MAX`.
    ///
    /// # Panics
    ///
    /// Panics if `l` exceeds the `max_l` given to [`Self::new`] or the output slices differ in
    /// length.
    pub fn search<F: PredicateFilter>(
        &mut self,
        algorithm: &Algorithm,
        l: usize,
        query: &[T],
        filter: &Arc<F>,
        ids: &mut [u32],
        distances: &mut [f32],
    ) -> Result<QueryStats> {
        assert_eq!(ids.len(), distances.len());
        let index = self.index;
        let context = DefaultContext;
        let knn = Knn::new(l, None)?;
        let labels: &dyn QueryLabelProvider<u32> = &**filter;
        let mut stats = QueryStats::default();
        let record = |stats: &mut QueryStats, s: SearchStats| {
            stats.up_cmps = s.cmps;
            stats.up_hops = s.hops;
            stats.up_results = s.result_count;
        };

        let n_out = match algorithm {
            Algorithm::KnnPostfilter => {
                let mut out = BackInserter::new(&mut self.candidates[..l]);
                let s = block_on(index.search(knn, &FullPrecision, &context, query, &mut out))?;
                let n = out.current_len();
                record(&mut stats, s);
                post_filter(
                    &self.candidates[..n],
                    labels,
                    &mut stats,
                    ids,
                    distances,
                    |_, d| d,
                )
            }
            Algorithm::Inline | Algorithm::InlineAdaptive { .. } => {
                let adaptive = match algorithm {
                    Algorithm::InlineAdaptive { params, .. } => Some(params.clone()),
                    _ => None,
                };
                let strategy = Filtered::new(FullPrecision, labels);
                let mut out = IdDistance::new(ids, distances);
                let s = block_on(index.search(
                    InlineFilterSearch::new(knn, adaptive),
                    &strategy,
                    &context,
                    query,
                    &mut out,
                ))?;
                record(&mut stats, s);
                out.current_len()
            }
            Algorithm::Multihop => {
                let strategy = Filtered::new(FullPrecision, labels);
                let mut out = IdDistance::new(ids, distances);
                let s = block_on(index.search(
                    MultihopFilterSearch::new(knn),
                    &strategy,
                    &context,
                    query,
                    &mut out,
                ))?;
                record(&mut stats, s);
                out.current_len()
            }
            Algorithm::Beta { beta } => {
                let shared: Arc<dyn QueryLabelProvider<u32>> = filter.clone();
                let strategy = BetaFilter::new(FullPrecision, shared, *beta);
                let mut out = BackInserter::new(&mut self.candidates[..l]);
                let s = block_on(index.search(knn, &strategy, &context, query, &mut out))?;
                let n = out.current_len();
                drop(strategy);
                record(&mut stats, s);
                let computer = T::query_distance(query, Metric::L2);
                let vectors = &index.provider().base_vectors;
                let mut extra = 0;
                let n_out = post_filter(
                    &self.candidates[..n],
                    labels,
                    &mut stats,
                    ids,
                    distances,
                    |id, _| {
                        extra += 1;
                        // SAFETY: the index is not mutated while searching, and `id` was
                        // returned by the search, so it is a valid slot.
                        let v = unsafe { vectors.get_vector_sync(id as usize) };
                        computer.evaluate_similarity(v)
                    },
                );
                stats.extra_dist = extra;
                n_out
            }
        };
        ids[n_out..].fill(u32::MAX);
        distances[n_out..].fill(f32::MAX);
        stats.n_out = n_out as u32;
        Ok(stats)
    }
}

/// Keeps the first `ids.len()` candidates that match, in candidate order.
fn post_filter(
    candidates: &[Neighbor<u32>],
    labels: &dyn QueryLabelProvider<u32>,
    stats: &mut QueryStats,
    ids: &mut [u32],
    distances: &mut [f32],
    mut true_distance: impl FnMut(u32, f32) -> f32,
) -> usize {
    let mut n = 0;
    for &nb in candidates {
        if n == ids.len() {
            break;
        }
        let (id, distance) = nb.as_tuple();
        stats.post_pred += 1;
        if labels.is_match(id) {
            ids[n] = id;
            distances[n] = true_distance(id, distance);
            n += 1;
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        labels::{LabelStore, LabelStoreBuilder},
        predicate::{PredicateSet, PredicateSetBuilder},
        upstream::filter::{BitmapFilter, ForwardFilter},
        vectors::Element,
    };
    use diskann::graph::config;
    use diskann_providers::{
        index::diskann_async::new_index,
        model::graph::provider::async_::{
            common::NoDeletes,
            inmem::{DefaultProviderParameters, SetStartPoints},
        },
    };

    const N: usize = 400;
    const DIM: usize = 8;

    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*state >> 40) as f32 / (1u64 << 24) as f32
    }

    /// A small index over random points with frozen start point `N` (a copy of point 0).
    fn fixture() -> (
        DiskANNIndex<Provider<f32>>,
        Vec<f32>,
        Arc<LabelStore>,
        PredicateSet,
    ) {
        let mut state = 7;
        let data: Vec<f32> = (0..N * DIM).map(|_| lcg(&mut state)).collect();
        let config =
            config::Builder::new_with(12, config::MaxDegree::new(16), 32, Metric::L2.into(), |b| {
                b.alpha(1.2);
            })
            .build()
            .unwrap();
        let params = DefaultProviderParameters::simple(N, DIM, Metric::L2, 16);
        let index = new_index::<f32, _>(config, params, NoDeletes).unwrap();
        index
            .provider()
            .set_start_points(std::iter::once(&data[..DIM]))
            .unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        for (i, v) in data.chunks_exact(DIM).enumerate() {
            runtime
                .block_on(index.insert(&FullPrecision, &DefaultContext, &(i as u32), v))
                .unwrap();
        }
        let index = Arc::try_unwrap(index).ok().unwrap();

        let mut b = LabelStoreBuilder::new();
        let names = ["c=0", "c=1", "c=2", "c=3"];
        for i in 0..N {
            let mut labels = vec![("c", names[i % 4]), ("d", ["d=0", "d=1"][(i / 7) % 2])];
            if i % 5 == 0 {
                labels.push(("c", names[(i + 1) % 4]));
            }
            b.push_point(labels).unwrap();
        }
        let labels = Arc::new(b.finish().unwrap());
        let mut pb = PredicateSetBuilder::new(&labels);
        pb.push(&[vec!["c=0", "c=1"], vec!["d=0"]]);
        pb.push(&[vec!["c=3"], vec!["d=1"]]);
        pb.push::<&str>(&[]);
        pb.push(&[vec!["c=nowhere"]]);
        let predicates = pb.finish();
        (index, data, labels, predicates)
    }

    fn algorithms() -> Vec<Algorithm> {
        vec![
            Algorithm::KnnPostfilter,
            Algorithm::Inline,
            Algorithm::inline_adaptive(10, 16.0).unwrap(),
            Algorithm::Multihop,
            Algorithm::beta(0.5).unwrap(),
            Algorithm::beta(0.3).unwrap(),
        ]
    }

    fn search_all<F: PredicateFilter>(
        searcher: &mut Searcher<'_, f32>,
        algorithm: &Algorithm,
        query: &[f32],
        filter: &mut Arc<F>,
        cnf: crate::predicate::Cnf<'_>,
    ) -> (Vec<u32>, Vec<f32>, QueryStats) {
        Arc::get_mut(filter).unwrap().compile(cnf);
        let (mut ids, mut distances) = (vec![0; 10], vec![0.0; 10]);
        let stats = searcher
            .search(algorithm, 40, query, filter, &mut ids, &mut distances)
            .unwrap();
        (ids, distances, stats)
    }

    #[test]
    fn upstream_algorithms_return_true_matches() {
        let (index, data, labels, predicates) = fixture();
        let mut searcher = Searcher::new(&index, 64);
        let mut forward = Arc::new(ForwardFilter::new(labels.clone()));
        let mut bitmap = Arc::new(BitmapFilter::new(labels.clone()));
        let mut state = 99;
        for _ in 0..5 {
            let query: Vec<f32> = (0..DIM).map(|_| lcg(&mut state)).collect();
            for q in 0..predicates.len() {
                let cnf = predicates.get(q);
                let mut all_match = Vec::new();
                for algorithm in algorithms() {
                    let (ids, distances, stats) =
                        search_all(&mut searcher, &algorithm, &query, &mut forward, cnf);
                    let other = search_all(&mut searcher, &algorithm, &query, &mut bitmap, cnf);
                    assert_eq!((&ids, &distances), (&other.0, &other.1), "providers differ");
                    let n = stats.n_out as usize;
                    assert!(ids[n..].iter().all(|&i| i == u32::MAX));
                    for (&id, &d) in ids[..n].iter().zip(&distances[..n]) {
                        assert!(
                            (id as usize) < N,
                            "{}: auxiliary id returned",
                            algorithm.name()
                        );
                        assert!(
                            cnf.matches(labels.row(id)),
                            "{}: non-match",
                            algorithm.name()
                        );
                        let exact = f32::squared_l2(&query, &data[id as usize * DIM..][..DIM]);
                        assert!(
                            (d - exact).abs() <= 1e-5 * exact.max(1.0),
                            "{}",
                            algorithm.name()
                        );
                    }
                    assert!(distances[..n].windows(2).all(|w| w[0] <= w[1]));
                    if q == 3 {
                        assert_eq!(n, 0);
                    }
                    if q == 2 {
                        assert_eq!(n, 10);
                        all_match.push(ids);
                    }
                }
                // Every point matches: all algorithms reduce to the same unfiltered search.
                assert!(all_match.windows(2).all(|w| w[0] == w[1]));
            }
        }
    }
}
