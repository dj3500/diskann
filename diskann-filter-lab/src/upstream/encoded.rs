/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The encoded label index of open upstream PRs #1319 and #1477 (crate `diskann-label-index`,
//! not in upstream `main`) and the two searches built on it:
//!
//! - **#1319**: [`MultihopFilterSearch`] whose per-node label test is the index's flat CNF plan
//!   over dense per-label bit rows ("bitslice"; one 8-byte load per evaluated term).
//! - **#1477**: [`HybridFilterSearch`]: a conservative match bound `U` from exact per-label
//!   counts (CNF: the smallest clause's sum of counts) routes the query to an exhaustive scan of
//!   the enumerated candidates (`U < T`) or to multihop (`U >= T`); `U = 0` returns no results.
//!   Its index keeps exact rows for the `E` most frequent labels and maps the others into `B`
//!   shared Bloom rows (`k` hashes), which can admit false positives.
//!
//! **Exact admission (ours).** Neither PR re-checks Bloom positives. When a query's plan reads a
//! Bloom row, every point its evaluator accepts is re-checked on the point's label row (the
//! adapter's forward provider), in the graph route and in the exhaustive route, so answers are
//! exact. Queries whose labels all have exact rows skip the re-check.
//!
//! Per query, everything is inside the caller's timer: the clause strings (the library's query
//! interface takes label strings; the index's label names are our label ids in decimal), the
//! plan compilation, the routing bound, the candidate enumeration and the search.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    sync::Arc,
    time::Instant,
};

use anyhow::{bail, ensure, Context, Result};
use diskann::{
    graph::{
        ext::labeled::{CandidateLabelProvider, Filtered, QueryLabelProvider},
        search::{HybridFilterSearch, Knn, MultihopFilterSearch},
        strategy::FullPrecision,
        DiskANNIndex, IdDistance, SearchOutputBuffer,
    },
    provider::DefaultContext,
    utils::VectorRepr,
};
use diskann_label_index::{
    encode_unified_label_index_jsonl, BloomFilterConfig, EncodedLabelIndex,
    EncodedLabelPointEvaluator, EncodedLabelQuery, FilterExpressionType,
};
use serde_json::{json, Value};

use super::{
    filter::{Counter, ForwardFilter, PredicateFilter},
    index::Provider,
    search::block_on,
};
use crate::{labels::LabelStore, predicate::Cnf};

/// Layout of an encoded label index to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// One exact bit row per label (#1319's bitslice).
    ExactOnly,
    /// Exact rows for the `exact` most frequent labels, the rest in `bits` Bloom rows with
    /// `hashes` hash functions (#1477).
    Bloom { exact: u32, bits: u32, hashes: u32 },
}

/// Builds the encoded label index of `labels` at `out` with the PR's builder
/// (`encode_unified_label_index_jsonl`), through a JSONL file `<out>.jsonl` (one
/// `{"doc_id": i, "labels": [...]}` row per point; label names are our label ids in decimal),
/// which is deleted afterwards. Returns a JSON report (sizes, times).
pub fn build(labels: &LabelStore, out: &Path, layout: Layout) -> Result<Value> {
    let t0 = Instant::now();
    let jsonl = out.with_extension("jsonl");
    {
        let mut w = BufWriter::new(
            File::create(&jsonl).with_context(|| format!("create {}", jsonl.display()))?,
        );
        let mut line = String::new();
        for p in 0..labels.n_points() as u32 {
            line.clear();
            line.push_str("{\"doc_id\":");
            line.push_str(&p.to_string());
            line.push_str(",\"labels\":[");
            for (i, l) in labels.row(p).iter().enumerate() {
                if i > 0 {
                    line.push(',');
                }
                line.push('"');
                line.push_str(&l.to_string());
                line.push('"');
            }
            line.push_str("]}\n");
            w.write_all(line.as_bytes())?;
        }
        w.flush()?;
    }
    let jsonl_bytes = std::fs::metadata(&jsonl)?.len();
    let t_jsonl = t0.elapsed().as_secs_f64();
    let config = match layout {
        Layout::ExactOnly => BloomFilterConfig::exact_only(),
        Layout::Bloom {
            exact,
            bits,
            hashes,
        } => BloomFilterConfig::new(bits, hashes)?.with_exact_label_count(exact)?,
    };
    let t1 = Instant::now();
    encode_unified_label_index_jsonl(&jsonl, out, config)
        .map_err(|e| anyhow::anyhow!("encoding the label index: {e}"))?;
    let t_encode = t1.elapsed().as_secs_f64();
    std::fs::remove_file(&jsonl)?;
    let index = EncodedLabelIndex::load(out).map_err(|e| anyhow::anyhow!("{e}"))?;
    ensure!(
        index.num_vectors() as usize == labels.n_points(),
        "the index covers {} points, the labels {}",
        index.num_vectors(),
        labels.n_points()
    );
    Ok(json!({
        "layout": format!("{layout:?}"),
        "file": out,
        "file_bytes": std::fs::metadata(out)?.len(),
        "num_vectors": index.num_vectors(),
        "num_labels": index.num_labels(),
        "num_exact_labels": index.num_exact_labels(),
        "physical_rows": index.physical_row_valid_counts().len(),
        "payload_bytes": payload_bytes(&index),
        "jsonl_bytes": jsonl_bytes,
        "jsonl_s": t_jsonl,
        "encode_s": t_encode,
    }))
}

/// Bytes of the bit rows (exact + Bloom) and their word summaries.
fn payload_bytes(index: &EncodedLabelIndex) -> u64 {
    let rows = index.physical_row_valid_counts().len() as u64;
    let words = u64::from(index.num_vectors()).div_ceil(64);
    // one summary bit per payload word
    rows * (words * 8 + words.div_ceil(64) * 8)
}

/// Search of an encoded-index provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodedAlgorithm {
    /// #1319: multihop with the plan as the per-node test.
    Multihop,
    /// #1477: count-routed exhaustive scan (`U < threshold`) or multihop.
    Hybrid { threshold: u64 },
}

impl EncodedAlgorithm {
    pub fn name(&self) -> String {
        match self {
            Self::Multihop => "encoded-multihop".into(),
            Self::Hybrid { threshold } => format!("hybrid({threshold})"),
        }
    }
}

/// Route of the last query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodedRoute {
    Graph,
    Exhaustive,
    /// No point can match (`U = 0` or an empty clause).
    Empty,
}

/// Counters of one query.
#[derive(Debug, Clone, Copy, Default)]
pub struct EncodedStats {
    pub n_out: u32,
    /// Upstream `SearchStats::cmps` (exhaustive route: candidates scored).
    pub up_cmps: u32,
    pub up_hops: u32,
    /// Plan evaluations (graph route; the exhaustive route enumerates instead).
    pub calls: u64,
    /// Evaluations and enumerated candidates accepted by the plan.
    pub plan_pos: u64,
    /// Plan positives rejected by exact admission (Bloom false positives).
    pub false_pos: u64,
    /// Exact admission checks.
    pub exact_checks: u64,
    /// Routing bound `U` (`u64::MAX` when not computed: #1319).
    pub bound: u64,
    /// Exhaustive route: candidates scored (after admission).
    pub candidates: u64,
    /// Exhaustive route: candidates enumerated by the index (before admission).
    pub candidates_enumerated: u64,
    pub route: Option<EncodedRoute>,
    /// Whether the plan read a Bloom row (exact admission on).
    pub admitted: bool,
}

/// The plan's point evaluator, then exact admission when the plan reads Bloom rows.
#[derive(Debug)]
struct Admit<'a, E> {
    eval: E,
    exact: Option<&'a ForwardFilter>,
    calls: &'a Counter,
    pos: &'a Counter,
    fps: &'a Counter,
}

impl<E: QueryLabelProvider<u32>> QueryLabelProvider<u32> for Admit<'_, E> {
    #[inline]
    fn is_match(&self, id: u32) -> bool {
        self.calls.bump();
        if !self.eval.is_match(id) {
            return false;
        }
        self.pos.bump();
        match self.exact {
            Some(x) if !x.is_match(id) => {
                self.fps.bump();
                false
            }
            _ => true,
        }
    }

    #[inline]
    fn prefetch(&self, id: u32) {
        self.eval.prefetch(id);
    }

    #[inline]
    fn prefetch_lookahead(&self) -> usize {
        self.eval.prefetch_lookahead()
    }
}

/// [`Admit`] plus the query's routing bound and candidate enumeration (#1477's provider).
#[derive(Debug)]
struct Candidates<'a, E> {
    point: Admit<'a, E>,
    query: &'a EncodedLabelQuery<'static>,
    enumerated: &'a Counter,
}

impl<E: QueryLabelProvider<u32>> QueryLabelProvider<u32> for Candidates<'_, E> {
    #[inline]
    fn is_match(&self, id: u32) -> bool {
        self.point.is_match(id)
    }

    #[inline]
    fn prefetch(&self, id: u32) {
        self.point.prefetch(id);
    }

    #[inline]
    fn prefetch_lookahead(&self) -> usize {
        self.point.prefetch_lookahead()
    }
}

impl<E: QueryLabelProvider<u32>> CandidateLabelProvider<u32> for Candidates<'_, E> {
    fn match_upper_bound(&self) -> Option<u64> {
        self.query.match_upper_bound()
    }

    fn visit_candidates(&self, mut visit: impl FnMut(u32)) {
        let p = &self.point;
        self.query.visit_matches(|id| {
            self.enumerated.bump();
            match p.exact {
                Some(x) if !x.is_match(id) => p.fps.bump(),
                _ => visit(id),
            }
        });
    }
}

/// Every data point matches (a predicate without clauses; the library cannot express it).
#[derive(Debug)]
struct AllMatch {
    n_data: u32,
}

impl QueryLabelProvider<u32> for AllMatch {
    #[inline]
    fn is_match(&self, id: u32) -> bool {
        id < self.n_data
    }
}

impl CandidateLabelProvider<u32> for AllMatch {
    fn match_upper_bound(&self) -> Option<u64> {
        Some(u64::from(self.n_data))
    }

    fn visit_candidates(&self, visit: impl FnMut(u32)) {
        (0..self.n_data).for_each(visit);
    }
}

/// A loaded encoded label index with the per-query state of the adapter.
pub struct EncodedProvider {
    index: Arc<EncodedLabelIndex>,
    /// Per label id of the label store: whether the index keeps an exact row for it.
    exact_row: Vec<bool>,
    exact: ForwardFilter,
    clauses: Vec<String>,
    n_data: u32,
    calls: Counter,
    pos: Counter,
    fps: Counter,
    enumerated: Counter,
    desc: Value,
}

impl EncodedProvider {
    /// Loads the index at `path`, built by [`build`] from `labels`.
    pub fn open(path: &Path, labels: Arc<LabelStore>) -> Result<Self> {
        let t0 = Instant::now();
        let index = EncodedLabelIndex::load(path)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        let load_s = t0.elapsed().as_secs_f64();
        ensure!(
            index.num_vectors() as usize == labels.n_points(),
            "{}: {} points, the labels have {}",
            path.display(),
            index.num_vectors(),
            labels.n_points()
        );
        // The builder orders its dictionary by decreasing count, then name bytes, and keeps exact
        // rows for the first `num_exact_labels` labels; labels without points are not in it.
        let mut order: Vec<(usize, String, u32)> = (0..labels.n_labels() as u32)
            .filter(|&l| labels.posting_len(l) > 0)
            .map(|l| (labels.posting_len(l), l.to_string(), l))
            .collect();
        ensure!(
            order.len() == index.num_labels(),
            "{}: {} labels, the label store has {} with points",
            path.display(),
            index.num_labels(),
            order.len()
        );
        order.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.as_bytes().cmp(b.1.as_bytes())));
        let mut exact_row = vec![false; labels.n_labels()];
        for (_, name, l) in order.iter().take(index.num_exact_labels()) {
            ensure!(index.contains_label(name), "label {name} missing from the index");
            exact_row[*l as usize] = true;
        }
        let desc = json!({
            "label_index": path,
            "label_index_file_bytes": std::fs::metadata(path)?.len(),
            "label_index_payload_bytes": payload_bytes(&index),
            "num_labels": index.num_labels(),
            "num_exact_labels": index.num_exact_labels(),
            "physical_rows": index.physical_row_valid_counts().len(),
            "load_s": load_s,
        });
        let n_data = labels.n_points() as u32;
        Ok(Self {
            index: Arc::new(index),
            exact_row,
            exact: ForwardFilter::new(labels),
            clauses: Vec::new(),
            n_data,
            calls: Counter::default(),
            pos: Counter::default(),
            fps: Counter::default(),
            enumerated: Counter::default(),
            desc,
        })
    }

    pub fn describe(&self) -> Value {
        self.desc.clone()
    }

    /// Answers one query: the best `ids.len()` matches of `cnf` with list size `l`.
    #[allow(clippy::too_many_arguments)]
    pub fn search<T: VectorRepr>(
        &mut self,
        index: &DiskANNIndex<Provider<T>>,
        algorithm: EncodedAlgorithm,
        l: usize,
        query: &[T],
        cnf: Cnf<'_>,
        ids: &mut [u32],
        distances: &mut [f32],
    ) -> Result<EncodedStats> {
        self.calls.reset();
        self.pos.reset();
        self.fps.reset();
        self.enumerated.reset();
        let mut stats = EncodedStats {
            bound: u64::MAX,
            ..EncodedStats::default()
        };
        let n_out = if cnf.is_unsatisfiable() {
            stats.route = Some(EncodedRoute::Empty);
            stats.bound = 0;
            0
        } else if cnf.n_clauses() == 0 {
            let all = AllMatch {
                n_data: self.n_data,
            };
            run(index, algorithm, l, query, &all, &mut stats, ids, distances)?
        } else {
            self.clauses.clear();
            let mut admitted = false;
            for clause in cnf.clauses() {
                let mut s = String::new();
                for (i, &label) in clause.iter().enumerate() {
                    if i > 0 {
                        s.push('|');
                    }
                    s.push_str(&label.to_string());
                    admitted |= !self.exact_row[label as usize];
                }
                self.clauses.push(s);
            }
            let plan = self
                .index
                .query(&self.clauses, FilterExpressionType::CNF)
                .map_err(|e| anyhow::anyhow!("compiling {:?}: {e}", self.clauses))?;
            stats.admitted = admitted;
            let exact = if admitted {
                self.exact.compile(cnf);
                Some(&self.exact)
            } else {
                None
            };
            let (calls, pos, fps, enumerated) =
                (&self.calls, &self.pos, &self.fps, &self.enumerated);
            macro_rules! with_evaluator {
                ($e:expr) => {{
                    let provider = Candidates {
                        point: Admit {
                            eval: $e,
                            exact,
                            calls,
                            pos,
                            fps,
                        },
                        query: &plan,
                        enumerated,
                    };
                    run(index, algorithm, l, query, &provider, &mut stats, ids, distances)?
                }};
            }
            match plan.point_evaluator() {
                EncodedLabelPointEvaluator::CnfPayload(e) => with_evaluator!(e),
                EncodedLabelPointEvaluator::CnfSummary(e) => with_evaluator!(e),
                EncodedLabelPointEvaluator::DnfPayload(e) => with_evaluator!(e),
                EncodedLabelPointEvaluator::DnfSummary(e) => with_evaluator!(e),
            }
        };
        ids[n_out..].fill(u32::MAX);
        distances[n_out..].fill(f32::MAX);
        stats.n_out = n_out as u32;
        stats.calls = self.calls.get();
        stats.plan_pos = self.pos.get() + self.enumerated.get();
        stats.candidates_enumerated = self.enumerated.get();
        stats.false_pos = self.fps.get();
        stats.exact_checks = self.exact.counts().calls;
        Ok(stats)
    }
}

/// One search with `provider` (static dispatch); returns the number of results written.
#[allow(clippy::too_many_arguments)]
fn run<T, P>(
    index: &DiskANNIndex<Provider<T>>,
    algorithm: EncodedAlgorithm,
    l: usize,
    query: &[T],
    provider: &P,
    stats: &mut EncodedStats,
    ids: &mut [u32],
    distances: &mut [f32],
) -> Result<usize>
where
    T: VectorRepr,
    P: CandidateLabelProvider<u32>,
{
    let context = DefaultContext;
    let knn = Knn::new(l, None)?;
    let strategy = Filtered::new(FullPrecision, provider);
    let k = ids.len();
    let mut out = IdDistance::new(ids, distances);
    match algorithm {
        EncodedAlgorithm::Multihop => {
            let s = block_on(index.search(
                MultihopFilterSearch::new(knn),
                &strategy,
                &context,
                query,
                &mut out,
            ))?;
            stats.up_cmps = s.cmps;
            stats.up_hops = s.hops;
            stats.route = Some(EncodedRoute::Graph);
        }
        EncodedAlgorithm::Hybrid { threshold } => {
            ensure!(k > 0, "hybrid search needs k > 0");
            let bound = provider.match_upper_bound().unwrap_or(u64::MAX);
            let search = HybridFilterSearch::new(knn, k, threshold, provider)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let s = block_on(index.search(search, &strategy, &context, query, &mut out))?;
            stats.up_cmps = s.stats.cmps;
            stats.up_hops = s.stats.hops;
            stats.bound = bound;
            stats.route = Some(match s.timings {
                None => EncodedRoute::Graph,
                Some(_) if bound == 0 => EncodedRoute::Empty,
                Some(t) => {
                    stats.candidates = u64::from(t.candidate_count);
                    EncodedRoute::Exhaustive
                }
            });
        }
    }
    Ok(out.current_len())
}

/// Parses `encoded-multihop` / `hybrid(T)`.
pub fn parse_algorithm(text: &str) -> Result<EncodedAlgorithm> {
    let t: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    if t == "multihop" || t == "encoded-multihop" {
        return Ok(EncodedAlgorithm::Multihop);
    }
    if let Some(a) = t.strip_prefix("hybrid(").and_then(|r| r.strip_suffix(')')) {
        let threshold = if a == "max" { u64::MAX } else { a.parse()? };
        ensure!(threshold > 0, "hybrid threshold must be positive");
        return Ok(EncodedAlgorithm::Hybrid { threshold });
    }
    bail!("unknown algorithm {text:?} for the encoded provider (multihop, hybrid(T), hybrid(max))")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        labels::LabelStoreBuilder,
        predicate::PredicateSetBuilder,
    };

    /// The query's own (enum-dispatched) point test, by reference.
    #[derive(Debug)]
    struct ByRef<'a>(&'a EncodedLabelQuery<'static>);

    impl QueryLabelProvider<u32> for ByRef<'_> {
        fn is_match(&self, id: u32) -> bool {
            self.0.is_match(id)
        }
    }

    fn store(n: usize) -> Arc<LabelStore> {
        let mut b = LabelStoreBuilder::new();
        let names: Vec<String> = (0..40).map(|i| format!("g={i}")).collect();
        for i in 0..n {
            let mut labels = vec![("c", ["c=0", "c=1", "c=2"][i % 3]), ("d", names[i % 40].as_str())];
            if i % 7 == 0 {
                labels.push(("d", names[(i / 7) % 40].as_str()));
            }
            b.push_point(labels).unwrap();
        }
        Arc::new(b.finish().unwrap())
    }

    /// The candidate enumeration with exact admission yields exactly the matching points, for an
    /// exact-only index and for a Bloom index whose tail labels share few rows.
    #[test]
    fn enumeration_and_point_test_are_exact() {
        let n = 3000;
        let labels = store(n);
        let dir = std::env::temp_dir().join(format!("flab-encoded-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut pb = PredicateSetBuilder::new(&labels);
        pb.push(&[vec!["c=0", "c=2"], vec!["g=3", "g=17", "g=39"]]);
        pb.push(&[vec!["g=5"]]);
        pb.push(&[vec!["c=1"], vec!["g=1"], vec!["g=2", "g=9"]]);
        let set = pb.finish();
        for layout in [
            Layout::ExactOnly,
            Layout::Bloom {
                exact: 4,
                bits: 8,
                hashes: 2,
            },
        ] {
            let path = dir.join(format!("{layout:?}.danlbl").replace(' ', ""));
            build(&labels, &path, layout).unwrap();
            let mut p = EncodedProvider::open(&path, labels.clone()).unwrap();
            for q in 0..set.len() {
                let cnf = set.get(q);
                let want: Vec<u32> = (0..n as u32)
                    .filter(|&i| cnf.matches(labels.row(i)))
                    .collect();
                p.clauses = cnf
                    .clauses()
                    .map(|c| c.iter().map(u32::to_string).collect::<Vec<_>>().join("|"))
                    .collect();
                let plan = p.index.query(&p.clauses, FilterExpressionType::CNF).unwrap();
                let admitted = cnf
                    .clauses()
                    .flatten()
                    .any(|&l| !p.exact_row[l as usize]);
                p.exact.compile(cnf);
                let exact = admitted.then_some(&p.exact);
                let c = Candidates {
                    point: Admit {
                        eval: ByRef(&plan),
                        exact,
                        calls: &p.calls,
                        pos: &p.pos,
                        fps: &p.fps,
                    },
                    query: &plan,
                    enumerated: &p.enumerated,
                };
                let mut got = Vec::new();
                c.visit_candidates(|id| got.push(id));
                got.sort_unstable();
                assert_eq!(got, want, "{layout:?} q{q}: enumeration");
                let tested: Vec<u32> = (0..n as u32 + 2).filter(|&i| c.is_match(i)).collect();
                assert_eq!(tested, want, "{layout:?} q{q}: point test");
                assert!(c.match_upper_bound().unwrap() >= want.len() as u64);
                if layout == Layout::ExactOnly {
                    assert!(!admitted);
                }
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
