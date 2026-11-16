/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! `flab calibrate` end to end on a small dataset written to cargo's target directory: the
//! terms of the cost model that the B6 calibration lacked (member checks by class, galloping
//! steps by gap, the laziness of graph search) are measured and read back.

use std::path::Path;

use diskann_filter_lab::{
    calibrate,
    graph::{AuxNodes, Graph},
    labels::LabelStoreBuilder,
    run,
    vectors::{Element, Matrix},
};

const N: usize = 2000;
const DIM: usize = 8;

fn rng(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// 2,000 points; category `g`: g=0 on about 70 % of the points, then Zipf-like values (rare
/// ones below the bitset threshold ⌈n/64⌉ = 32); category `m`: m=0 on about 60 %, m=1, m=2;
/// a 10-NN graph with a ring, global start 0. `single`: the same labels in one category `w`.
fn write_dataset(dir: &Path, single: bool) {
    std::fs::create_dir_all(dir).unwrap();
    let mut s = 7u64;
    let base = Matrix::<u8>::from_slice(
        &(0..N * DIM)
            .map(|_| (rng(&mut s) % 64) as u8)
            .collect::<Vec<_>>(),
        N,
        DIM,
    );
    base.write_bin(&dir.join("base.u8bin")).unwrap();
    let mut b = LabelStoreBuilder::new();
    for _ in 0..N {
        let mut row = Vec::new();
        if rng(&mut s) % 10 < 7 {
            row.push("g=0".to_owned());
        }
        let r = rng(&mut s) % 40;
        row.push(format!("g={}", 1 + r * r / 40));
        let m = match rng(&mut s) % 10 {
            0..=5 => 0,
            6..=8 => 1,
            _ => 2,
        };
        row.push(format!("m={m}"));
        if single {
            row = row.into_iter().map(|l| format!("w{l}")).collect();
        }
        b.push_point(row.iter().map(|l| (&l[..1], l.as_str())))
            .unwrap();
    }
    b.finish().unwrap().write(&dir.join("labels.flab")).unwrap();
    let mut g = Graph::new(
        N,
        12,
        AuxNodes::new(Vec::new(), &Matrix::<u8>::zeros(0, DIM)).unwrap(),
    )
    .unwrap();
    for v in 0..N {
        let mut c: Vec<(f32, u32)> = (0..N)
            .filter(|&u| u != v)
            .map(|u| (u8::squared_l2(base.row(v), base.row(u)), u as u32))
            .collect();
        c.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut nb: Vec<u32> = c.iter().take(10).map(|x| x.1).collect();
        let next = ((v + 1) % N) as u32;
        if !nb.contains(&next) {
            nb.push(next);
        }
        g.set_neighbors(v as u32, &nb).unwrap();
    }
    g.set_starts(vec![0]).unwrap();
    std::fs::create_dir_all(dir.join("graph")).unwrap();
    g.write(&dir.join("graph/graph.flab")).unwrap();
}

#[test]
fn calibration_measures_member_checks_galloping_and_laziness() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("flab-calibrate-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    write_dataset(&dir, false);
    let opts = calibrate::Options {
        reps: 1,
        queries: 16,
        seed: 3,
        theta: None,
    };
    let report = calibrate::calibrate::<u8>(&dir, Some(&dir.join("graph")), None, &opts).unwrap();
    let c = &report["costs"];
    let curve = |key: &str| -> Vec<f64> {
        c[key]
            .as_array()
            .unwrap_or_else(|| panic!("{key}: {c}"))
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    };
    let densities = curve("density").len();
    for key in ["pred_row_ns", "pred_bits_ns"] {
        let v = curve(key);
        assert_eq!(v.len(), densities, "{key}");
        assert!(v.iter().all(|x| x.is_finite() && *x > 0.0), "{key}: {v:?}");
    }
    let members = &report["details"]["member_checks"];
    let measured = |class: &str| {
        members[class]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["ns"].is_number())
            .count()
    };
    assert!(
        measured("rows") + measured("bits") > 0,
        "no member check measured: {members}"
    );
    let gaps = curve("gallop_gap");
    assert!(gaps.len() >= 2, "galloping gaps: {gaps:?}");
    assert!(gaps.windows(2).all(|w| w[0] < w[1]));
    assert!(curve("gallop_gap_ns").iter().all(|x| *x > 0.0));
    assert_eq!(curve("heap_bound_k"), [10.0, 100.0]);
    assert!(curve("heap_bound_extra")
        .iter()
        .all(|x| x.is_finite() && *x >= 0.0));
    // The measured distance counts per list size and extra(K, L) per list size.
    let ls = curve("graph_l");
    assert_eq!(ls.len(), 9, "{ls:?}");
    assert!(curve("graph_l_dists").iter().all(|x| *x > 0.0));
    let hl = curve("heap_bound_l");
    assert!(hl.len() >= 2, "{hl:?}");
    let rows = c["heap_bound_extra_kl"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|r| r.as_array().unwrap().len() == hl.len()));
    // Cold member checks by residual shape and the per-query cost.
    for key in [
        "check_base_ns",
        "check_word_ns",
        "check_prefetch_ns",
        "check_row_ns",
    ] {
        let v = curve(key);
        assert_eq!(v.len(), densities, "{key}");
        assert!(v.iter().all(|x| x.is_finite() && *x >= 0.0), "{key}: {v:?}");
    }
    assert!(
        curve("check_word_ns").iter().any(|x| *x > 0.0)
            || curve("check_row_ns").iter().any(|x| *x > 0.0),
        "no word or row cost measured"
    );
    assert!(c["query_ns"].as_f64().unwrap() > 0.0);
    let frac = c["distance_first_frac"].as_f64().unwrap();
    assert!(frac > 0.0 && frac <= 1.0, "{frac}");
    // The report reads back as unit costs.
    let path = dir.join("costs.json");
    std::fs::write(&path, report.to_string()).unwrap();
    let costs = run::read_costs(&path).unwrap();
    assert_eq!(costs.pred_row_ns.len(), densities);
    assert_eq!(costs.heap_bound_k, [10.0, 100.0]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Single-category label sets have no AND across categories: the member checks are measured
/// on ANDs of two labels instead of falling back to the random-point curve.
#[test]
fn single_category_calibration_measures_member_checks() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("flab-calibrate-single-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    write_dataset(&dir, true);
    let opts = calibrate::Options {
        reps: 1,
        queries: 16,
        seed: 3,
        theta: None,
    };
    let report = calibrate::calibrate::<u8>(&dir, None, None, &opts).unwrap();
    let members = &report["details"]["member_checks"];
    let measured: usize = ["rows", "bits"]
        .iter()
        .map(|class| {
            members[class]
                .as_array()
                .unwrap()
                .iter()
                .filter(|r| r["ns"].is_number())
                .count()
        })
        .sum();
    assert!(measured > 0, "no member check measured: {members}");
    let _ = std::fs::remove_dir_all(&dir);
}
