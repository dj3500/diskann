/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The lab search kernel: one best-first graph search whose behaviour is fixed by a
//! configuration with one implementation per role.
//!
//! Per query ([`search::Searcher::search`]):
//! 1. the predicate is compiled ([`eval::Compiled`]): clause upper bounds `U_c` (sums of posting
//!    lengths), the Boolean clause order (increasing `U_c`), a clause-bit table and clause
//!    weights. A query with an empty clause has no answers (route `empty`);
//! 2. exact-path routing: when `min_c U_c ≤ T_exact` the query is answered exactly from the
//!    postings ([`exact`]) and nothing else is computed;
//! 3. start selection ([`entry`]), the pilot (for scorers with per-query scales), the
//!    materialization of a match set ([`materialize`]) if configured;
//! 4. the loop: the `W` closest unexpanded beam nodes are expanded, every unvisited neighbour
//!    is *offered* (distance and predicate in the order fixed by the laziness role, admission
//!    to the beam, collection of matches), optionally followed by a two-hop expansion through
//!    rejected neighbours; a terminator decides when to stop;
//! 5. the collector returns at most `K` matching data points sorted by (true distance, id).
//!
//! Roles ([`config::Kernel`]): `PredicateEval` (mode 1 row scan, mode 2 per-clause search, a
//! materialized bitmap), `Materializer`, `Admission`, `Laziness`, `Scorer`, `Collector`,
//! `Expander`, `Terminator`, `Starts` (with a start-admission policy) and `ExactPath`.
//! The element type, the predicate mode and the scorer family are compile-time parameters
//! (dispatched once per plan), and so are the laziness, the expander and tracing (dispatched
//! once per query); the other roles are enums matched in the loop. The `postfilter` and
//! `upstream-inline` presets also have hand-specialized loops ([`search::FastPath`]).
//!
//! The beam keeps upstream DiskANN's tie rule (a new entry goes before existing entries with
//! an equal key) and stores `(key, true distance, id)`. Auxiliary nodes (ids `≥ n_data`) are
//! navigated but never evaluated against the predicate and never returned.

pub mod beam;
pub mod collect;
pub mod config;
pub mod entry;
pub mod eval;
pub mod exact;
pub mod materialize;
pub mod score;
pub mod search;
#[cfg(test)]
mod search_tests;
pub mod trace;
pub mod visited;

pub use config::{Kernel, Preset};
pub use search::{Counters, Index, Query, Route, Searcher};

/// Maps an `f32` to a `u32` whose unsigned order is the total order of the floats (for
/// non-NaN values: the numeric order, with `-0.0 < +0.0`).
#[inline(always)]
pub fn fkey(x: f32) -> u32 {
    let b = x.to_bits();
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    }
}

/// Inverse of [`fkey`].
#[inline(always)]
pub fn funkey(k: u32) -> f32 {
    f32::from_bits(if k & 0x8000_0000 != 0 {
        k & 0x7fff_ffff
    } else {
        !k
    })
}

/// Packs `(distance, id)` into a `u64` whose order is the order by distance, then id.
#[inline(always)]
pub fn pack(d: f32, id: u32) -> u64 {
    (u64::from(fkey(d)) << 32) | u64::from(id)
}

/// Inverse of [`pack`].
#[inline(always)]
pub fn unpack(p: u64) -> (f32, u32) {
    (funkey((p >> 32) as u32), p as u32)
}

/// Prefetches the cache line holding `*p` into all cache levels (a hint; no effect on other
/// architectures).
#[inline(always)]
pub(crate) fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: prefetching is a hint that never faults, whatever the address.
    unsafe {
        std::arch::x86_64::_mm_prefetch(p.cast::<i8>(), std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_keys_preserve_order() {
        let xs = [
            f32::NEG_INFINITY,
            -3.5,
            -0.0,
            0.0,
            1e-30,
            1.0,
            1.5,
            7e20,
            f32::INFINITY,
        ];
        for w in xs.windows(2) {
            assert!(fkey(w[0]) < fkey(w[1]), "{} < {}", w[0], w[1]);
        }
        for &x in &xs {
            assert_eq!(funkey(fkey(x)).to_bits(), x.to_bits());
        }
        assert!(pack(1.0, 9) < pack(1.0, 10));
        assert!(pack(1.0, 10) < pack(1.5, 0));
        assert_eq!(unpack(pack(2.5, 77)), (2.5, 77));
    }
}
