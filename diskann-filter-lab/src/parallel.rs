/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Minimal scoped parallelism with an explicit thread count.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Computes `f(i, state)` for every `i in 0..n` on `threads` scoped threads and returns the
/// results in index order.
///
/// Items are handed out one at a time, so uneven per-item costs balance well. Every thread
/// creates its own `state` with `init` (e.g. reusable scratch buffers).
///
/// # Panics
///
/// Re-raises the panic of a worker thread.
pub fn par_map<S, R, I, F>(n: usize, threads: usize, init: I, f: F) -> Vec<R>
where
    R: Send,
    I: Fn() -> S + Sync,
    F: Fn(usize, &mut S) -> R + Sync,
{
    let threads = threads.clamp(1, n.max(1));
    let next = AtomicUsize::new(0);
    let worker = || {
        let mut state = init();
        let mut out = Vec::new();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n {
                break out;
            }
            out.push((i, f(i, &mut state)));
        }
    };
    let mut items: Vec<(usize, R)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads).map(|_| s.spawn(worker)).collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    });
    items.sort_unstable_by_key(|item| item.0);
    items.into_iter().map(|item| item.1).collect()
}
