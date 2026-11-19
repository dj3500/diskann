/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! `flab run` end to end on a tiny dataset written to cargo's target directory: trace,
//! reference and timed modes, the run-output format and the ledger.

use std::path::{Path, PathBuf};

use diskann_filter_lab::{
    bitsets::LabelBitsets,
    graph::{AuxNodes, Graph},
    kernel::cost::UnitCosts,
    labels::{LabelStore, LabelStoreBuilder},
    predicate::{PredicateSet, PredicateSetBuilder},
    run,
    sketches::LabelSketches,
    starts::Starts,
    vectors::{Element, Matrix},
};
use serde_json::Value;

const N: usize = 400;
const DIM: usize = 8;
const NQ: usize = 24;

fn rng(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

struct Data {
    dir: PathBuf,
    base: Matrix<u8>,
    queries: Matrix<u8>,
    labels: LabelStore,
    preds: PredicateSet,
}

fn write_dataset(dir: &Path) -> Data {
    std::fs::create_dir_all(dir).unwrap();
    let mut s = 99u64;
    let base = Matrix::<u8>::from_slice(
        &(0..N * DIM)
            .map(|_| (rng(&mut s) % 64) as u8)
            .collect::<Vec<_>>(),
        N,
        DIM,
    );
    base.write_bin(&dir.join("base.u8bin")).unwrap();
    let queries = Matrix::<u8>::from_slice(
        &(0..NQ * DIM)
            .map(|_| (rng(&mut s) % 64) as u8)
            .collect::<Vec<_>>(),
        NQ,
        DIM,
    );
    queries.write_bin(&dir.join("qa.u8bin")).unwrap();

    let mut b = LabelStoreBuilder::new();
    for i in 0..N {
        let g = format!("g={}", rng(&mut s) % 8);
        let m = format!("m={}", i % 3);
        b.push_point([("g", g.as_str()), ("m", m.as_str())])
            .unwrap();
    }
    let labels = b.finish().unwrap();
    labels.write(&dir.join("labels.flab")).unwrap();
    let mut pb = PredicateSetBuilder::new(&labels);
    for q in 0..NQ {
        pb.push(&[
            vec![format!("g={}", q % 8), format!("g={}", (q + 3) % 8)],
            vec![format!("m={}", q % 3)],
        ]);
    }
    let preds = pb.finish();
    preds.write(&dir.join("qa.fpred")).unwrap();

    // A k-NN graph with a cycle; the global start is point 0.
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
    let gdir = dir.join("graph");
    std::fs::create_dir_all(&gdir).unwrap();
    g.write(&gdir.join("graph.flab")).unwrap();
    let mut st = Starts::new(&g);
    let pairs: Vec<(u32, u32)> = (0..labels.n_labels() as u32)
        .map(|l| (l, labels.postings(l)[0]))
        .collect();
    st.set_label_starts(&labels, &pairs).unwrap();
    st.set_sample((0..N as u32).step_by(2).collect(), None)
        .unwrap();
    st.write(&gdir.join("starts.flab")).unwrap();

    let mut panel = String::from("qset\tqidx\tsplit\tband\n");
    for q in 0..NQ {
        let split = if q % 4 == 3 { "audit" } else { "discovery" };
        panel.push_str(&format!("a\t{q}\t{split}\t5\n"));
    }
    std::fs::create_dir_all(dir.join("panels")).unwrap();
    std::fs::write(dir.join("panels/p.tsv"), panel).unwrap();
    Data {
        dir: dir.to_owned(),
        base,
        queries,
        labels,
        preds,
    }
}

fn experiment(d: &Data, mode: &str, extra: &str) -> PathBuf {
    let text = format!(
        r#"
[experiment]
name = "driver-test"
dataset = "{ds}"
graph = "{ds}/graph"
panel = "panels/p.tsv"
split = "discovery"
k = [5]
efforts = [3, 10, 30]
reps = 2
mode = "{mode}"
runs_dir = "{ds}/runs"
ledger = "{ds}/ledger.jsonl"
trace_rows = [0, 2]
{extra}

[[config]]
name = "post"
preset = "postfilter"

[[config]]
name = "inline"
preset = "upstream-inline"

[[config]]
name = "filt"
preset = "filt(10, 1)"

[[config]]
name = "heap+labels"
preset = "upstream-beta(0.5)"
starts = "label(c*)+global"
collector = "heap"
"#,
        ds = d.dir.display()
    );
    let p = d.dir.join(format!("{mode}.toml"));
    std::fs::write(&p, text).unwrap();
    p
}

fn read_res(path: &Path) -> (usize, usize, Vec<u32>, Vec<f32>) {
    let b = std::fs::read(path).unwrap();
    let nq = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let k = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    assert_eq!(b.len(), 8 + nq * k * 8);
    let ids = bytemuck::cast_slice::<u8, u32>(&b[8..8 + nq * k * 4]).to_vec();
    let d = b[8 + nq * k * 4..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (nq, k, ids, d)
}

fn check_dirs(d: &Data, dirs: &[PathBuf], mode: &str, timed: bool) {
    assert_eq!(dirs.len(), 4);
    let rows: Vec<usize> = (0..NQ).filter(|q| q % 4 != 3).collect();
    for dir in dirs {
        let run: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("run.json")).unwrap()).unwrap();
        assert_eq!(run["mode"], mode);
        assert_eq!(run["K"], 5);
        assert_eq!(run["splits"], serde_json::json!(["discovery"]));
        let efforts: Vec<u64> = run["efforts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap())
            .collect();
        let name = run["config"]["name"].as_str().unwrap();
        if name == "heap+labels" {
            assert_eq!(efforts, [3, 10, 30], "heap answers do not need L >= K");
        } else {
            assert_eq!(efforts, [10, 30], "{name}: L = 3 < K is not applicable");
            assert_eq!(run["skipped_efforts"], serde_json::json!([3]));
        }
        let panel = std::fs::read_to_string(dir.join("panel.tsv")).unwrap();
        assert_eq!(panel.lines().count(), 1 + rows.len());
        for &l in &efforts {
            let (nq, k, ids, dist) = read_res(&dir.join(format!("res_{l}.bin")));
            assert_eq!((nq, k), (rows.len(), 5));
            for (r, &q) in rows.iter().enumerate() {
                let cnf = d.preds.get(q);
                for j in 0..k {
                    let id = ids[r * k + j];
                    if id == u32::MAX {
                        assert_eq!(dist[r * k + j], f32::MAX);
                        continue;
                    }
                    assert!(cnf.matches(d.labels.row(id)), "{name}: violation");
                    assert_eq!(
                        dist[r * k + j],
                        u8::squared_l2(d.queries.row(q), d.base.row(id as usize)),
                        "true distances"
                    );
                }
            }
        }
        let pq = std::fs::read_to_string(dir.join("perquery.tsv")).unwrap();
        let mut lines = pq.lines();
        let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
        assert_eq!(
            &header[..12],
            [
                "effort",
                "rep",
                "row",
                "wall_ns",
                "cpu_ns",
                "n_dist",
                "n_pred",
                "n_expand",
                "n_hops",
                "n_match_seen",
                "n_out",
                "route"
            ]
        );
        let reps = if timed { 2 } else { 1 };
        let body: Vec<&str> = lines.collect();
        assert_eq!(body.len(), reps * efforts.len() * rows.len());
        let wall = body[0].split('\t').nth(3).unwrap();
        assert_eq!(!wall.is_empty(), timed);
        if timed {
            for l in &efforts {
                assert_eq!(run["results"][l.to_string()]["nondeterministic_rows"], 0);
            }
        }
        if mode == "trace" {
            for &l in &efforts {
                for r in [0, 2] {
                    let t =
                        std::fs::read_to_string(dir.join(format!("trace/{l}_{r}.tsv"))).unwrap();
                    assert!(t.lines().count() > 2, "trace of row {r}");
                }
            }
        }
    }
}

#[test]
fn driver_writes_the_run_format() {
    let dir =
        Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("flab-run-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let d = write_dataset(&dir);
    let opts = run::Options::default();

    let dirs = run::run(&experiment(&d, "trace", ""), &opts).unwrap();
    check_dirs(&d, &dirs, "trace", false);

    let dirs = run::run(
        &experiment(&d, "reference", "reference_min = 1000000"),
        &opts,
    )
    .unwrap();
    check_dirs(&d, &dirs, "reference", false);
    // With an unbounded budget, the reference executor of the post-filter is exact.
    let post = dirs
        .iter()
        .find(|p| p.to_string_lossy().contains("-post-"))
        .unwrap();
    let (_, k, ids, _) = read_res(&post.join("res_10.bin"));
    let rows: Vec<usize> = (0..NQ).filter(|q| q % 4 != 3).collect();
    for (r, &q) in rows.iter().enumerate() {
        let cnf = d.preds.get(q);
        let mut all: Vec<(f32, u32)> = (0..N as u32)
            .filter(|&p| cnf.matches(d.labels.row(p)))
            .map(|p| (u8::squared_l2(d.queries.row(q), d.base.row(p as usize)), p))
            .collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let truth: Vec<u32> = all.iter().take(k).map(|x| x.1).collect();
        assert_eq!(&ids[r * k..r * k + truth.len()], &truth[..], "row {r}");
    }

    // Timed mode refuses to run unless pinned to one CPU.
    let timed = experiment(&d, "timed", "");
    #[cfg(target_os = "linux")]
    {
        // SAFETY: an all-zero `cpu_set_t` is a valid (empty) set; the calls get and set the
        // affinity of this thread with a set of the right size.
        unsafe {
            let mut old: libc::cpu_set_t = std::mem::zeroed();
            libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut old);
            let n_allowed = (0..libc::CPU_SETSIZE as usize)
                .filter(|&c| libc::CPU_ISSET(c, &old))
                .count();
            if n_allowed > 1 {
                assert!(run::run(&timed, &opts).is_err(), "unpinned timed run");
            }
            let cpu = libc::sched_getcpu() as usize;
            let mut one: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_SET(cpu, &mut one);
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &one);
            let dirs = run::run(&timed, &opts);
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &old);
            check_dirs(&d, &dirs.unwrap(), "timed", true);
        }
    }

    // A prefix of a query set is described by the rows written to `panel.tsv`.
    let prefix = d.dir.join("prefix.toml");
    let text = format!(
        r#"
[experiment]
name = "driver-prefix"
dataset = "{ds}"
graph = "{ds}/graph"
queries = "qa"
limit = 3
k = 5
efforts = [10]
mode = "trace"
runs_dir = "{ds}/runs"
ledger = "{ds}/ledger.jsonl"

[[config]]
name = "post"
preset = "postfilter"
"#,
        ds = d.dir.display()
    );
    std::fs::write(&prefix, text).unwrap();
    let dirs = run::run(&prefix, &opts).unwrap();
    let run: Value =
        serde_json::from_str(&std::fs::read_to_string(dirs[0].join("run.json")).unwrap()).unwrap();
    let file = run["panel"]["file"].as_str().unwrap();
    assert_eq!(
        std::fs::read_to_string(file).unwrap().lines().count(),
        1 + 3
    );
    assert_eq!(read_res(&dirs[0].join("res_10.bin")).0, 3);

    // Static per-label bitsets, built in memory or loaded from a sidecar: mode 1's answers.
    let sidecar = d.dir.join("bitsets.flab");
    LabelBitsets::build(&d.labels, 40).write(&sidecar).unwrap();
    for (i, extra) in [
        String::new(),
        format!("bitsets_sidecar = \"{}\"", sidecar.display()),
    ]
    .into_iter()
    .enumerate()
    {
        let p = d.dir.join("static.toml");
        let text = format!(
            r#"
[experiment]
name = "driver-static-{i}"
dataset = "{ds}"
graph = "{ds}/graph"
queries = "qa"
k = 5
efforts = [10]
mode = "trace"
runs_dir = "{ds}/runs"
ledger = "{ds}/ledger.jsonl"
{extra}

[[config]]
name = "mode1"
preset = "upstream-beta(0.5)"
collector = "heap"

[[config]]
name = "static60"
preset = "upstream-beta(0.5)"
collector = "heap"
predicate = "static_bits(60)"

[[config]]
name = "static1000"
preset = "upstream-beta(0.5)"
collector = "heap"
predicate = "static_bits(1000)"

[[config]]
name = "bitmap-static"
preset = "upstream-beta(0.5)"
collector = "heap"
predicate = "bitmap"
materializer = "dense_static"
"#,
            ds = d.dir.display()
        );
        std::fs::write(&p, text).unwrap();
        let dirs = run::run(&p, &opts).unwrap();
        let res: Vec<_> = dirs
            .iter()
            .map(|dir| read_res(&dir.join("res_10.bin")))
            .collect();
        assert_eq!(res[0], res[1]);
        assert_eq!(res[0], res[2]);
        assert_eq!(res[0], res[3]);
        let run: Value =
            serde_json::from_str(&std::fs::read_to_string(dirs[1].join("run.json")).unwrap())
                .unwrap();
        let bits = &run["index"]["bitsets"];
        assert_eq!(bits["theta"], if extra.is_empty() { 60 } else { 40 });
        assert!(bits["n_bitsets"].as_u64().unwrap() >= 3, "{bits}");
    }

    // B6: unit costs and sketches sidecars, per-configuration efforts, the crossover
    // counters; the planned exact path returns the version-1 scan's answers.
    let costs = d.dir.join("costs.json");
    std::fs::write(
        &costs,
        serde_json::json!({"costs": UnitCosts::default()}).to_string(),
    )
    .unwrap();
    let other = d.dir.join("costs-other.json");
    std::fs::write(
        &other,
        serde_json::json!({"costs": UnitCosts { graph_alpha: 99.0, ..UnitCosts::default() }})
            .to_string(),
    )
    .unwrap();
    let sk = d.dir.join("sketches.flab");
    LabelSketches::build(&d.labels, 16, 1, 2)
        .write(&sk)
        .unwrap();
    let sk2 = d.dir.join("sketches-seed7.flab");
    LabelSketches::build(&d.labels, 16, 7, 2)
        .write(&sk2)
        .unwrap();
    let p = d.dir.join("b6.toml");
    let text = format!(
        r#"
[experiment]
name = "driver-b6"
dataset = "{ds}"
graph = "{ds}/graph"
queries = "qa"
k = 5
efforts = [10, 30]
mode = "trace"
runs_dir = "{ds}/runs"
ledger = "{ds}/ledger.jsonl"
calibration = "{costs}"
sketches_sidecar = "{sk}"

[[config]]
name = "exact-v1"
exact = "scan_smallest_clause(1000000)"
efforts = [10]

[[config]]
name = "exact-v2"
exact = "planned(1000000)"
estimates = "sketch"
efforts = [10]

[[config]]
name = "s3"
starts = "label(c*)+global"
collector = "heap"
laziness = "heap_bound"
exact = "planned"
crossover = "guarded"
estimates = "sketch"
terminator = "posting_l(16)"

[[config]]
name = "s3-other-costs"
starts = "label(c*)+global"
collector = "heap"
laziness = "heap_bound"
exact = "planned"
crossover = "guarded"
estimates = "sketch"
terminator = "posting_l(16)"
exact_checks = "full"
calibration = "{other}"
sketches_sidecar = "{sk2}"
efforts = [10]
"#,
        ds = d.dir.display(),
        costs = costs.display(),
        other = other.display(),
        sk = sk.display(),
        sk2 = sk2.display(),
    );
    std::fs::write(&p, text).unwrap();
    let dirs = run::run(&p, &opts).unwrap();
    assert_eq!(dirs.len(), 4);
    assert_eq!(
        read_res(&dirs[0].join("res_10.bin")),
        read_res(&dirs[1].join("res_10.bin"))
    );
    assert!(!dirs[1].join("res_30.bin").exists(), "own efforts only");
    assert!(dirs[2].join("res_30.bin").exists());
    let run: Value =
        serde_json::from_str(&std::fs::read_to_string(dirs[2].join("run.json")).unwrap()).unwrap();
    assert_eq!(run["index"]["sketches"]["b"], 16);
    assert!(run["index"]["calibration"]["costs"]["graph_alpha"].is_number());
    assert_eq!(
        run["index"]["calibration"]["source"],
        costs.display().to_string()
    );
    assert_eq!(run["config"]["kernel"]["exact_checks"], "residual");
    let other_run: Value =
        serde_json::from_str(&std::fs::read_to_string(dirs[3].join("run.json")).unwrap()).unwrap();
    assert_eq!(
        other_run["index"]["calibration"]["source"],
        other.display().to_string()
    );
    assert_eq!(
        other_run["index"]["calibration"]["costs"]["graph_alpha"],
        99.0
    );
    assert_eq!(other_run["config"]["kernel"]["exact_checks"], "full");
    // Per-configuration sketches (another priority seed).
    assert_eq!(run["index"]["sketches"]["seed"], 1);
    assert_eq!(other_run["index"]["sketches"]["seed"], 7);
    assert_eq!(
        other_run["index"]["sketches"]["source"],
        sk2.display().to_string()
    );
    assert_eq!(run["skipped_efforts"], serde_json::json!([]));
    let pq = std::fs::read_to_string(dirs[2].join("perquery.tsv")).unwrap();
    let header: Vec<&str> = pq.lines().next().unwrap().split('\t').collect();
    for col in [
        "n_probe",
        "plan",
        "fallback",
        "est_exact_ns",
        "a_hat",
        "m_hat",
    ] {
        assert!(header.contains(&col), "{col} in {header:?}");
    }

    let ledger = std::fs::read_to_string(dir.join("ledger.jsonl")).unwrap();
    let lines: Vec<Value> = ledger
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(lines.iter().filter(|l| l["status"] == "ok").count() >= 8);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn start_tables_of_one_graph_share_it() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("flab-run-shared-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let _ = write_dataset(&dir);
    let gdir = dir.join("graph");
    // A second table of the same graph with an entry layer: more global starts.
    let mut g = Graph::read(&gdir.join("graph.flab")).unwrap();
    g.set_starts(vec![0, 7, 42]).unwrap();
    Starts::new(&g).write(&dir.join("entry.flab")).unwrap();
    let mut shared = std::collections::HashMap::new();
    let a = run::data::load_graph_shared(&gdir, None, u8::DTYPE, &mut shared).unwrap();
    let b =
        run::data::load_graph_shared(&gdir, Some(&dir.join("entry.flab")), u8::DTYPE, &mut shared)
            .unwrap();
    assert!(std::sync::Arc::ptr_eq(&a.graph, &b.graph));
    assert_eq!(a.global, vec![0]);
    assert_eq!(b.global, vec![0, 7, 42]);
    std::fs::remove_dir_all(&dir).unwrap();
}
