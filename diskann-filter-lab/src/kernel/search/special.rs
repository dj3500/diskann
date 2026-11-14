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
use crate::{
    kernel::{
        beam::{Inserted, State},
        collect::Collector,
        eval::{Eval, EvalCtx, RowScan},
        fkey,
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
        let k = ids.len();
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
        let cx = EvalCtx::new(ix.labels, &self.compiled, &self.bits, &[], &[]);
        self.answers.reset(Collector::FinalBeam, k, l);
        let mut taken = 0;
        for i in 0..beam.len() {
            let e = beam.entry(i);
            if e.state == State::Aux {
                continue;
            }
            if taken == l {
                break;
            }
            taken += 1;
            c.n_pred += 1;
            if RowScan::full(&cx, e.id) {
                self.answers.offer(e.d, e.id);
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
