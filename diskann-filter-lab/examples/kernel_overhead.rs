/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Overhead of the configurable lab kernel against its hand-specialized fast paths of the
//! `postfilter` and `upstream-inline` presets, on a dataset and a graph (both through
//! `Searcher::search`, so the per-query flow outside the loop is identical).
//!
//! Usage (pin the process to one CPU):
//! `kernel_overhead <dataset dir> <graph> <query set> <queries> <L1,L2,...> <reps>`
//!
//! For each repetition and L, the four variants (generic and specialized post-filter and
//! inline search) run interleaved over blocks of 25 queries (rotated so that every variant
//! covers every block once per repetition); the report gives the median over repetitions of
//! the mean time per query and the generic variants' overhead. An untimed pass first checks
//! that each generic variant returns exactly what its specialized twin returns.

use std::{path::Path, time::Instant};

use anyhow::{ensure, Context, Result};
use diskann_filter_lab::{
    kernel::{
        search::{Index, Plan, Query, Searcher},
        Preset,
    },
    run::{data, sys},
    vectors::{DType, Element},
};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(
        args.len() == 7,
        "usage: kernel_overhead <dataset dir> <graph> <query set> <queries> <L1,L2,...> <reps>"
    );
    let dir = Path::new(&args[1]);
    let graph = Path::new(&args[2]);
    let qset = args[3].trim_start_matches('q').to_owned();
    let n: usize = args[4].parse().context("queries")?;
    let efforts: Vec<usize> = args[5]
        .split(',')
        .map(|s| s.parse().context("efforts"))
        .collect::<Result<_>>()?;
    let reps: usize = args[6].parse().context("reps")?;
    match data::dataset_dtype(dir)? {
        DType::U8 => bench::<u8>(dir, graph, &qset, n, &efforts, reps),
        DType::F32 => bench::<f32>(dir, graph, &qset, n, &efforts, reps),
    }
}

const NAMES: [&str; 4] = [
    "postfilter (generic)",
    "postfilter (special)",
    "inline (generic)",
    "inline (special)",
];

fn bench<T: Element>(
    dir: &Path,
    graph: &Path,
    qset: &str,
    n: usize,
    efforts: &[usize],
    reps: usize,
) -> Result<()> {
    let affinity = sys::affinity();
    if affinity.as_ref().is_none_or(|a| a.len() != 1) {
        eprintln!("warning: not pinned to one CPU (affinity {affinity:?})");
    }
    let ds = data::Dataset::<T>::load(dir, &[qset.to_owned()], &[])?;
    let g = data::load_graph(graph, None, T::DTYPE)?;
    let ix = Index {
        graph: &g.graph,
        vectors: g.graph.vectors(&ds.base)?,
        labels: &ds.labels,
        global: &g.global,
        label_starts: None,
        sample: None,
        label_view: false,
        view_postings: None,
        eps_min: 0.0,
        bitsets: None,
        costs: None,
        sketches: None,
        pairs: None,
        islands: None,
        pair_roots: None,
    };
    let post_fast = Plan::new(&Preset::parse("postfilter")?.kernel(), &ix)?;
    let inline_fast = Plan::new(&Preset::parse("upstream-inline")?.kernel(), &ix)?;
    let post = post_fast.clone().generic();
    let inline = inline_fast.clone().generic();
    let mut st = Searcher::for_index(&ix, &[&post]);
    let set = &ds.qsets[qset];
    let n = n.min(set.vectors.nrows());
    let queries: Vec<Query<'_, T>> = (0..n)
        .map(|i| Query {
            vector: set.vectors.row(i),
            cnf: set.preds.get(i),
            view: None,
        })
        .collect();
    std::hint::black_box(data::prefault(&ds, [&g]));
    let k = 10;
    let (mut ids, mut d) = (vec![0u32; k], vec![0f32; k]);
    let run = |st: &mut Searcher,
               v: usize,
               q: &Query<'_, T>,
               l: usize,
               ids: &mut [u32],
               d: &mut [f32]| match v {
        0 => {
            st.search(&post, &ix, q, l, ids, d, None);
        }
        1 => {
            st.search(&post_fast, &ix, q, l, ids, d, None);
        }
        2 => {
            st.search(&inline, &ix, q, l, ids, d, None);
        }
        _ => {
            st.search(&inline_fast, &ix, q, l, ids, d, None);
        }
    };

    // Identity check (untimed), which also warms up every variant.
    let (mut ids2, mut d2) = (vec![0u32; k], vec![0f32; k]);
    for &l in efforts {
        for q in &queries {
            for pair in [(0, 1), (2, 3)] {
                run(&mut st, pair.0, q, l, &mut ids, &mut d);
                run(&mut st, pair.1, q, l, &mut ids2, &mut d2);
                ensure!(
                    ids == ids2 && d == d2,
                    "{} differs from its specialization",
                    NAMES[pair.0]
                );
            }
        }
    }

    // Fine-grained interleaving: blocks of BLOCK queries; in rotation r, block b runs variant
    // (b + r) mod 4, so every variant covers every block once per repetition and consecutive
    // blocks (different queries, cold caches) run different variants.
    const BLOCK: usize = 25;
    let n_blocks = n.div_ceil(BLOCK);
    let mut times = vec![vec![Vec::new(); 4]; efforts.len()];
    let mut cpu_times = vec![vec![Vec::new(); 4]; efforts.len()];
    for rep in 0..reps {
        for (e, &l) in efforts.iter().enumerate() {
            let mut sums = [0u128; 4];
            let mut cpu_sums = [0u64; 4];
            for r in 0..4 {
                for b in 0..n_blocks {
                    let v = (b + r + rep) % 4;
                    let block = &queries[b * BLOCK..((b + 1) * BLOCK).min(n)];
                    let c0 = sys::thread_cpu_ns();
                    let t = Instant::now();
                    for q in block {
                        run(&mut st, v, q, l, &mut ids, &mut d);
                    }
                    sums[v] += t.elapsed().as_nanos();
                    cpu_sums[v] += sys::thread_cpu_ns() - c0;
                }
            }
            for v in 0..4 {
                times[e][v].push(sums[v] as f64 / n as f64);
                cpu_times[e][v].push(cpu_sums[v] as f64 / n as f64);
            }
        }
    }
    let median = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_unstable_by(f64::total_cmp);
        s[s.len() / 2]
    };
    println!(
        "cpu {:?}, load {:?}, {} queries of q{qset}, {reps} repetitions; median over repetitions of the mean µs per query",
        sys::current_cpu(),
        sys::load_average(),
        n
    );
    for (what, t) in [("thread CPU time", &cpu_times), ("wall time", &times)] {
        println!("{what}:");
        println!(
            "{:>6} {:>12} {:>12} {:>9} {:>12} {:>12} {:>9}",
            "L", "post gen", "post spec", "overhead", "inline gen", "inline spec", "overhead"
        );
        for (e, &l) in efforts.iter().enumerate() {
            let m: Vec<f64> = (0..4).map(|v| median(&t[e][v]) / 1000.0).collect();
            println!(
                "{l:>6} {:>12.2} {:>12.2} {:>8.2}% {:>12.2} {:>12.2} {:>8.2}%",
                m[0],
                m[1],
                100.0 * (m[0] / m[1] - 1.0),
                m[2],
                m[3],
                100.0 * (m[2] / m[3] - 1.0)
            );
        }
    }
    Ok(())
}
