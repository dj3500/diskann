/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Transparent huge pages for the large read-mostly arrays (graph rows, vectors, labels,
//! postings, bitsets, sketches).
//!
//! A search touches these arrays at random. With 4 KiB pages, nearly every such touch in a
//! multi-gigabyte array also misses the TLB, and the page walk adds a memory access to the
//! cache miss. With 2 MiB pages the TLB covers the vectors and much of the graph of a 40M
//! index (A7). The arrays are advised with `MADV_HUGEPAGE` before their first write, so the
//! kernel backs them with huge pages at the page faults of loading, where the system enables
//! transparent huge pages for advised memory (`/sys/kernel/mm/transparent_hugepage/enabled`
//! = `madvise` or `always`). Only memory placement changes: no result does.
//!
//! On by default; `FLAB_HUGEPAGES=off` (or `0`) disables the advice (for A/B timing).

use std::sync::OnceLock;

/// Size of a transparent huge page on x86-64 and most aarch64 kernels.
pub const HUGE_PAGE: usize = 2 << 20;

/// Whether large allocations are advised for transparent huge pages (`FLAB_HUGEPAGES`).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("FLAB_HUGEPAGES")
            .map(|v| !matches!(v.trim(), "off" | "0" | "false"))
            .unwrap_or(true)
    })
}

/// The 2 MiB-aligned interior `[start, end)` of `[addr, addr + len)`, if not empty.
fn interior(addr: usize, len: usize) -> Option<(usize, usize)> {
    let start = addr.checked_next_multiple_of(HUGE_PAGE)?;
    let end = (addr.checked_add(len)? / HUGE_PAGE) * HUGE_PAGE;
    (end > start).then_some((start, end))
}

/// Advises the 2 MiB-aligned interior of `[ptr, ptr + len)` for transparent huge pages. Call
/// it before the memory is first written: pages already faulted in stay small. A no-op when
/// disabled, for ranges below one huge page, and off Linux.
pub fn advise(ptr: *const u8, len: usize) {
    if !enabled() {
        return;
    }
    let Some((start, end)) = interior(ptr as usize, len) else {
        return;
    };
    #[cfg(target_os = "linux")]
    // SAFETY: madvise only changes the kernel's paging advice for this mapped range of the
    // caller's allocation; an error (e.g. no THP support) leaves the memory as it was.
    unsafe {
        libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (start, end);
}

/// Advises the memory of `v` (before its first write) for transparent huge pages.
pub fn advise_slice<T>(v: &mut [T]) {
    advise(v.as_ptr().cast(), std::mem::size_of_val(v));
}

/// A vector of `n` copies of `value` whose allocation was advised for transparent huge pages
/// before it was written.
pub fn filled<T: Clone>(n: usize, value: T) -> Vec<T> {
    let mut v: Vec<T> = Vec::with_capacity(n);
    advise(
        v.as_ptr().cast(),
        n.saturating_mul(std::mem::size_of::<T>()),
    );
    v.resize(n, value);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_interior_is_aligned_and_inside() {
        assert_eq!(interior(0, HUGE_PAGE), Some((0, HUGE_PAGE)));
        assert_eq!(interior(1, HUGE_PAGE), None);
        assert_eq!(interior(1, 3 * HUGE_PAGE), Some((HUGE_PAGE, 3 * HUGE_PAGE)));
        assert_eq!(
            interior(HUGE_PAGE - 1, 2 * HUGE_PAGE),
            Some((HUGE_PAGE, 2 * HUGE_PAGE))
        );
        assert_eq!(interior(usize::MAX - 10, 100), None);
    }

    #[test]
    fn advised_vectors_hold_their_values() {
        for n in [0, 1, 1000, (3 * HUGE_PAGE) / 4 + 17] {
            let v = filled(n, 7u32);
            assert_eq!(v.len(), n);
            assert!(v.iter().all(|&x| x == 7));
        }
        let mut z = vec![0u64; 5 * HUGE_PAGE / 8];
        advise_slice(&mut z);
        z[12345] = 3;
        assert_eq!(z.iter().sum::<u64>(), 3);
    }
}
