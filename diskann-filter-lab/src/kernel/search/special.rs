/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Hand-specialized fast paths of the `postfilter` and `upstream-inline` presets, with every
//! role fixed in the code (no role branches, no traces). `Searcher::search` uses them for
//! plans that are exactly these presets (unless the plan was made generic), and they return
//! exactly the answers and counters of the configurable loop (tested). They exist because the
//! configurable loop costs a few percent more than these loops (`examples/kernel_overhead.rs`).

use super::{Counters, Index, Query, Searcher, LOOKAHEAD};
use crate::kernel::planner::Fallback;
use crate::{
    bitsets::LabelBitsets,
    kernel::{
        beam::{Inserted, State},
        collect::Collector,
        eval::{Eval, EvalCtx, RowScan},
        fkey, pack, unpack,
    },
    vectors::Element,
};

/// A hand-specialized fast path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FastPath {
    None,
    Postfilter,
    Inline,
}

impl FastPath {
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Postfilter => "postfilter",
            Self::Inline => "upstream-inline",
        }
    }
}

impl Searcher {
    /// The `postfilter` preset after start selection: unfiltered search with list size `l`,
    /// then the matches among the first `l` data entries of the final beam.
    pub(super) fn fast_postfilter<T: Element>(
        &mut self,
        ix: &Index<'_, T>,
        q: &Query<'_, T>,
        l: usize,
        ctr: &mut Counters,
        ids: &mut [u32],
        dists: &mut [f32],
    ) -> usize {
        self.postfilter_loop::<T, RowScan>(ix, q, l, None, false, ctr, ids, dists)
    }

    /// Unfiltered search with list size `l` from the selected starts, then collection with the
    /// evaluator `P` (static bitsets `bits` if given):
    /// - `deferred = false` (the `postfilter` preset): every one of the first `l` data
    ///   entries of the final beam is checked, and the matches are the answers;
    /// - `deferred = true`: the answers of a `heap(K)` collector over every evaluated node
    ///   (the `K` nearest matches by (distance, id)), found by checking the final beam in
    ///   distance order until `K` matches and the entries at the `K`-th match's distance are
    ///   checked; nodes outside the final beam are farther than all of its entries, so they
    ///   are checked (in (distance, id) order, from a log of every evaluated node) only when
    ///   the whole beam was checked.
    #[expect(
        clippy::too_many_arguments,
        reason = "the per-query inputs and outputs"
    )]
    pub(super) fn postfilter_loop<T: Element, P: Eval>(
        &mut self,
        ix: &Index<'_, T>,
        q: &Query<'_, T>,
        l: usize,
        bits: Option<&LabelBitsets>,
        deferred: bool,
        ctr: &mut Counters,
        ids: &mut [u32],
        dists: &mut [f32],
    ) -> usize {
        let k = ids.len();
        let log = &mut self.log;
        log.clear();
        let n_data = ix.graph.n_data() as u32;
        let (graph, vectors, qv) = (ix.graph, ix.vectors, q.vector);
        let beam = &mut self.beam32;
        let cap = l + self.starts.len();
        beam.reset(cap, cap, false, false);
        let mut c = Counters::default();
        for &s in &self.starts {
            let d = T::squared_l2(qv, vectors.get(s));
            c.n_dist += 1;
            let state = if s < n_data {
                if deferred {
                    log.push(pack(d, s));
                }
                State::Unknown
            } else {
                c.n_aux_dist += 1;
                State::Aux
            };
            if let Inserted::At {
                evicted: Some(_), ..
            } = beam.insert(fkey(d), d, s, state)
            {
                c.n_evict += 1;
            }
        }
        let nbrs = &mut self.nbrs;
        while let Some(n) = beam.pop() {
            c.n_expand += 1;
            if n >= n_data {
                c.n_aux_expand += 1;
            }
            if self.row_prefetch {
                // The next expansion's row (speculative: a closer neighbour of `n` may come first)
                // loads while the distances of `n`'s neighbours are computed.
                if let Some(next) = beam.peek_id() {
                    graph.prefetch_row(next);
                }
            }
            nbrs.clear();
            for &v in graph.neighbors(n) {
                if !self.visited.test_and_set(v) {
                    nbrs.push(v);
                }
            }
            let len = nbrs.len();
            c.n_dist += len as u64;
            let la = LOOKAHEAD.min(len);
            for &v in &nbrs[..la] {
                diskann_vector::prefetch_hint_max::<4, _>(vectors.get(v));
            }
            for i in 0..len {
                if i + la < len {
                    diskann_vector::prefetch_hint_max::<4, _>(vectors.get(nbrs[i + la]));
                }
                let v = nbrs[i];
                let d = T::squared_l2(qv, vectors.get(v));
                let state = if v < n_data {
                    if deferred {
                        log.push(pack(d, v));
                    }
                    State::Unknown
                } else {
                    c.n_aux_dist += 1;
                    State::Aux
                };
                if let Inserted::At {
                    evicted: Some(_), ..
                } = beam.insert(fkey(d), d, v, state)
                {
                    c.n_evict += 1;
                }
            }
        }
        c.n_hops = c.n_expand;
        let mut cx = EvalCtx::new(ix.labels, &self.compiled, &self.bits, &[], &[]);
        if let Some(b) = bits {
            cx = cx.with_static(b);
        }
        self.answers.reset(Collector::FinalBeam, k, l);
        let limit = if deferred { usize::MAX } else { l };
        let (mut taken, mut found, mut bound) = (0, 0, f32::INFINITY);
        let mut stopped = false;
        for i in 0..beam.len() {
            let e = beam.entry(i);
            if e.state == State::Aux {
                continue;
            }
            if taken == limit || e.d > bound {
                stopped = true;
                break;
            }
            taken += 1;
            c.n_pred += 1;
            if P::full(&cx, e.id) {
                self.answers.offer(e.d, e.id);
                found += 1;
                if deferred && found == k {
                    bound = e.d;
                }
            }
        }
        if deferred && !stopped && k > 0 {
            // The whole beam was checked: the K nearest matches of every evaluated node, in
            // (distance, id) order.
            ctr.fallback = Fallback::Log as u64;
            self.answers.reset(Collector::Heap, k, l);
            log.sort_unstable();
            for &p in log.iter() {
                let (d, id) = unpack(p);
                if self.answers.is_full() && d > self.answers.worst() {
                    break;
                }
                c.n_pred += 1;
                if P::full(&cx, id) {
                    self.answers.offer(d, id);
                }
            }
        }
        ctr.add(&c);
        ctr.final_l = beam.search_l() as u64;
        self.answers.finish(ids, dists)
    }

    /// The `upstream-inline` preset after start selection: every evaluated node navigates,
    /// every evaluated match is collected; the best `K` of the sorted matches (truncated to
    /// `l`).
    pub(super) fn fast_inline<T: Element>(
        &mut self,
        ix: &Index<'_, T>,
        q: &Query<'_, T>,
        l: usize,
        ctr: &mut Counters,
        ids: &mut [u32],
        dists: &mut [f32],
    ) -> usize {
        let k = ids.len();
        let n_data = ix.graph.n_data() as u32;
        let (graph, vectors, qv) = (ix.graph, ix.vectors, q.vector);
        let cx = EvalCtx::new(ix.labels, &self.compiled, &self.bits, &[], &[]);
        let beam = &mut self.beam32;
        let answers = &mut self.answers;
        answers.reset(Collector::All, k, l);
        let cap = l + self.starts.len();
        beam.reset(cap, cap, false, false);
        let mut c = Counters::default();
        for &s in &self.starts {
            let d = T::squared_l2(qv, vectors.get(s));
            c.n_dist += 1;
            let state = if s >= n_data {
                c.n_aux_dist += 1;
                State::Aux
            } else {
                c.n_pred += 1;
                if RowScan::eval(&cx, s, false).m {
                    c.n_match_seen += 1;
                    answers.offer(d, s);
                    State::Match
                } else {
                    State::NoMatch
                }
            };
            if let Inserted::At {
                evicted: Some(_), ..
            } = beam.insert(fkey(d), d, s, state)
            {
                c.n_evict += 1;
            }
        }
        let nbrs = &mut self.nbrs;
        while let Some(n) = beam.pop() {
            c.n_expand += 1;
            if n >= n_data {
                c.n_aux_expand += 1;
            }
            if self.row_prefetch {
                if let Some(next) = beam.peek_id() {
                    graph.prefetch_row(next);
                }
            }
            nbrs.clear();
            for &v in graph.neighbors(n) {
                if !self.visited.test_and_set(v) {
                    nbrs.push(v);
                    if v < n_data {
                        RowScan::prefetch_offsets(&cx, v);
                    }
                }
            }
            let len = nbrs.len();
            c.n_dist += len as u64;
            let la = LOOKAHEAD.min(len);
            let prefetch = |v: u32| {
                diskann_vector::prefetch_hint_max::<4, _>(vectors.get(v));
                if v < n_data {
                    RowScan::prefetch(&cx, v);
                }
            };
            for &v in &nbrs[..la] {
                prefetch(v);
            }
            for i in 0..len {
                if i + la < len {
                    prefetch(nbrs[i + la]);
                }
                let v = nbrs[i];
                let d = T::squared_l2(qv, vectors.get(v));
                let state = if v >= n_data {
                    c.n_aux_dist += 1;
                    State::Aux
                } else {
                    c.n_pred += 1;
                    if RowScan::eval(&cx, v, false).m {
                        c.n_match_seen += 1;
                        answers.offer(d, v);
                        State::Match
                    } else {
                        State::NoMatch
                    }
                };
                if let Inserted::At {
                    evicted: Some(_), ..
                } = beam.insert(fkey(d), d, v, state)
                {
                    c.n_evict += 1;
                }
            }
        }
        c.n_hops = c.n_expand;
        ctr.add(&c);
        ctr.final_l = beam.search_l() as u64;
        answers.finish(ids, dists)
    }
}
