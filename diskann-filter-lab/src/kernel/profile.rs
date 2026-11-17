/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Cycle counts of the post-filter loop's phases (feature `phase-profile`, for profiling
//! builds only; without it every function is a no-op, `now` returns 0 and the counting code
//! is removed by constant folding).
//!
//! Per expansion the loop reads the time-stamp counter twice: [`EXPAND`] (pop, next-row
//! prefetch, adjacency row and visited checks) and [`NEIGHBOURS`] (vector prefetches, distance
//! computations, the evaluation log and the beam insertions of the new neighbours). One
//! insertion in [`SAMPLE_EVERY`] is timed on its own ([`INSERT_SAMPLED`], [`SAMPLED`] of
//! [`INSERTS`]), which estimates the insertions' share of [`NEIGHBOURS`] without timing each.
//! [`CHECK`]: the final predicate checks and collection. The loop accumulates its counts and
//! adds them to process-wide totals once per search; [`take`] reads and resets the totals.

pub const EXPAND: usize = 0;
pub const NEIGHBOURS: usize = 1;
pub const INSERT_SAMPLED: usize = 2;
pub const CHECK: usize = 3;
pub const INSERTS: usize = 4;
pub const SAMPLED: usize = 5;
pub const SLOTS: usize = 6;
pub const NAMES: [&str; SLOTS] = [
    "expand",
    "neighbours",
    "insert_sampled",
    "check",
    "inserts",
    "sampled",
];
/// One beam insertion in this many is timed.
pub const SAMPLE_EVERY: u64 = 8;

/// Whether this build counts cycles.
pub const ENABLED: bool = cfg!(feature = "phase-profile");

#[cfg(feature = "phase-profile")]
mod imp {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    use super::SLOTS;

    static TOTALS: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];

    /// The time-stamp counter.
    #[inline(always)]
    pub fn now() -> u64 {
        // SAFETY: `rdtsc` has no preconditions on x86-64.
        unsafe { core::arch::x86_64::_rdtsc() }
    }

    /// Adds one search's counts to the totals.
    pub fn add(counts: &[u64; SLOTS]) {
        for (t, &c) in TOTALS.iter().zip(counts) {
            t.fetch_add(c, Relaxed);
        }
    }

    /// Reads and resets the totals.
    pub fn take() -> [u64; SLOTS] {
        std::array::from_fn(|i| TOTALS[i].swap(0, Relaxed))
    }
}

#[cfg(not(feature = "phase-profile"))]
mod imp {
    use super::SLOTS;

    #[inline(always)]
    pub fn now() -> u64 {
        0
    }

    #[inline(always)]
    pub fn add(_: &[u64; SLOTS]) {}

    pub fn take() -> [u64; SLOTS] {
        [0; SLOTS]
    }
}

pub use imp::{add, now, take};
