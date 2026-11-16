/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A thin timed adapter around upstream DiskANN's unmodified filtered searches.
//!
//! Upstream's algorithms ([`Knn`](diskann::graph::search::Knn) with post-filtering,
//! [`InlineFilterSearch`](diskann::graph::search::InlineFilterSearch) with or without
//! [`AdaptiveL`](diskann::graph::search::AdaptiveL),
//! [`MultihopFilterSearch`](diskann::graph::search::MultihopFilterSearch) and the
//! [`BetaFilter`](diskann_providers::model::graph::provider::layers::BetaFilter) layer) run on
//! an index saved by upstream, with the lab's predicate providers ([`filter`]), a per-query
//! timer and the lab's run outputs ([`run`]). Used by the `flab-upstream` binary.

pub mod encoded;
pub mod filter;
pub mod index;
pub mod queries;
pub mod run;
pub mod search;
pub mod sys;
