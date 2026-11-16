/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! External engines: configurations of `flab run` whose queries are answered by code outside
//! the lab kernel, inside the same timed loop as the kernel's configurations (same per-query
//! timer scope, warm-up, interleaving, repetitions and outputs), so that both can be compared
//! within one invocation.
//!
//! - `engine = "upstream"`: upstream DiskANN's search algorithms through the adapter
//!   ([`crate::upstream`]) on the DiskANN index saved at the configuration's `graph` prefix.
//!   `algorithm` = `knn-postfilter` | `inline` | `inline-adaptive` / `inline-adaptive(s, f)` |
//!   `multihop` | `beta(β)` | `pathseer`; `provider` = `forward` (default) | `bitmap`. The
//!   provider is compiled for every query inside the timer.
//!   With `provider = "encoded(<label index>)"` (the encoded label index of open upstream PRs
//!   #1319 / #1477, built by `flab label-index`), `algorithm` = `multihop` (#1319) |
//!   `hybrid(T)` / `hybrid(max)` (#1477's count-routed search; [`crate::upstream::encoded`]).
//! - `engine = "dylib"`: a shared library (`library`) implementing the C ABI below, opened on
//!   the dataset directory with its own `index` path and a free-form `params` string.
//!
//! C ABI, version 1 (all functions `extern "C"`):
//!
//! ```c
//! typedef struct {
//!     uint64_t n_dist, n_pred, n_expand, n_hops, n_match_seen, n_out;
//!     uint64_t route;    /* 0 graph, 1 exact, 2 empty (no point can match) */
//!     uint64_t prep_ns;  /* time spent before the search proper (part of the query's time) */
//!     uint64_t x[4];     /* engine-specific counters, named in the description */
//! } flab_ext_counters;
//!
//! uint32_t flab_ext_abi_version(void);
//! void* flab_ext_open(const char* dataset_dir, const char* index, const char* params,
//!                     uint32_t elem /* 1 = u8, 2 = f32 */, uint32_t dim,
//!                     char* err, size_t err_len);              /* NULL on error */
//! const char* flab_ext_describe(void* engine);                /* JSON object */
//! int flab_ext_search(void* engine, const void* vector, const uint64_t* clause_bounds,
//!                     const uint32_t* labels, uint32_t n_clauses, uint32_t effort,
//!                     uint32_t k, uint32_t* ids, float* dists,
//!                     flab_ext_counters* counters);         /* 0 on success */
//! void flab_ext_close(void* engine);
//! ```
//!
//! Clause `c` of a query is `labels[clause_bounds[c]..clause_bounds[c + 1]]`, sorted label ids
//! of the dataset's `labels.flab` (the engine reads the same file). `flab_ext_search` writes at
//! most `k` matching data points in increasing true squared-L2 distance and pads the remaining
//! slots with `u32::MAX` / `f32::MAX`. The library is never unloaded.

use std::{
    ffi::{c_char, c_void, CStr, CString},
    os::unix::ffi::OsStrExt,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{bail, ensure, Context, Result};
use diskann::utils::VectorRepr;
use serde_json::{json, Value};

use super::data::Dataset;
use crate::{
    kernel::search::{Counters, Query, Route},
    labels::LabelStore,
    upstream::{
        encoded::{self, EncodedAlgorithm, EncodedProvider, EncodedRoute},
        filter::{BitmapFilter, ForwardFilter, PredicateFilter},
        index::{self, LoadedIndex},
        search::{Algorithm, QueryStats, Searcher},
    },
    vectors::{DType, Element},
};

/// The engine of a configuration (`engine = ...` in the experiment file).
#[derive(Debug, Clone, PartialEq)]
pub enum ExternalSpec {
    /// Upstream DiskANN code through the adapter, on the DiskANN index at the configuration's
    /// `graph` prefix.
    Upstream { algorithm: String, provider: String },
    /// A shared library implementing the C ABI of this module.
    Dylib {
        library: PathBuf,
        index: PathBuf,
        params: String,
    },
}

impl ExternalSpec {
    pub fn to_json(&self) -> Value {
        match self {
            Self::Upstream {
                algorithm,
                provider,
            } => json!({"engine": "upstream", "algorithm": algorithm, "provider": provider}),
            Self::Dylib {
                library,
                index,
                params,
            } => json!({"engine": "dylib", "library": library, "index": index, "params": params}),
        }
    }
}

/// Engine-specific outputs of one query besides the kernel's counters.
pub const EXTRA_NAMES: [&str; 5] = ["ext_prep_ns", "ext_x0", "ext_x1", "ext_x2", "ext_x3"];

/// An external engine answering queries of one configuration.
pub trait External<T> {
    /// Answers one query with effort `l`: the best `ids.len()` matches (ids and true squared-L2
    /// distances in increasing distance, padded with `u32::MAX` / `f32::MAX`).
    fn search(
        &mut self,
        q: &Query<'_, T>,
        l: usize,
        ids: &mut [u32],
        dists: &mut [f32],
    ) -> Result<(Counters, [u64; 5])>;

    /// What the engine is (recorded in `run.json` and hashed into the run id).
    fn describe(&self) -> Value;
}

/// Opens the engine of `spec` for a configuration on `graph` over the dataset `ds`; `max_l` is
/// the largest effort it will be asked for.
pub fn open<T: Element>(
    spec: &ExternalSpec,
    graph: &Path,
    ds: &Dataset<T>,
    max_l: usize,
) -> Result<Box<dyn External<T>>> {
    match spec {
        ExternalSpec::Upstream {
            algorithm,
            provider,
        } if provider.starts_with("encoded(") => {
            let path = provider
                .strip_prefix("encoded(")
                .and_then(|r| r.strip_suffix(')'))
                .context("provider = \"encoded(<label index path>)\"")?;
            let algorithm = encoded::parse_algorithm(algorithm)?;
            match T::DTYPE {
                DType::U8 => Ok(Box::new(Encoded::<u8>::open(
                    graph,
                    algorithm,
                    Path::new(path),
                    ds.labels.clone(),
                )?)),
                DType::F32 => Ok(Box::new(Encoded::<f32>::open(
                    graph,
                    algorithm,
                    Path::new(path),
                    ds.labels.clone(),
                )?)),
            }
        }
        ExternalSpec::Upstream {
            algorithm,
            provider,
        } => {
            let algorithm = parse_algorithm(algorithm)?;
            match T::DTYPE {
                DType::U8 => Ok(Box::new(Upstream::<u8>::open(
                    graph,
                    algorithm,
                    provider,
                    ds.labels.clone(),
                    max_l,
                )?)),
                DType::F32 => Ok(Box::new(Upstream::<f32>::open(
                    graph,
                    algorithm,
                    provider,
                    ds.labels.clone(),
                    max_l,
                )?)),
            }
        }
        ExternalSpec::Dylib {
            library,
            index,
            params,
        } => Ok(Box::new(Dylib::open(
            library,
            index,
            params,
            &ds.dir,
            T::DTYPE,
            ds.base.ncols(),
        )?)),
    }
}

/// Parses an upstream algorithm: `knn-postfilter`, `inline`, `inline-adaptive`,
/// `inline-adaptive(s, f)`, `multihop`, `beta(β)`, `pathseer`.
pub fn parse_algorithm(text: &str) -> Result<Algorithm> {
    let t: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    let args = |prefix: &str| -> Option<Vec<String>> {
        t.strip_prefix(prefix)
            .and_then(|r| r.strip_prefix('('))
            .and_then(|r| r.strip_suffix(')'))
            .map(|r| r.split(',').map(str::to_owned).collect())
    };
    Ok(match t.as_str() {
        "knn-postfilter" => Algorithm::KnnPostfilter,
        "inline" => Algorithm::Inline,
        "inline-adaptive" => Algorithm::inline_adaptive(1000, 16.0)?,
        "multihop" => Algorithm::Multihop,
        "pathseer" => Algorithm::PathSeer,
        _ => {
            if let Some(a) = args("inline-adaptive") {
                ensure!(a.len() == 2, "inline-adaptive(sample, scale)");
                Algorithm::inline_adaptive(a[0].parse()?, a[1].parse()?)?
            } else if let Some(a) = args("beta") {
                ensure!(a.len() == 1, "beta(β)");
                Algorithm::beta(a[0].parse()?)?
            } else {
                bail!("unknown upstream algorithm {text:?} (knn-postfilter, inline, inline-adaptive(s, f), multihop, beta(β), pathseer)")
            }
        }
    })
}

/// The adapter's predicate providers.
enum Provider {
    Forward(Arc<ForwardFilter>),
    Bitmap(Arc<BitmapFilter>),
}

/// Upstream DiskANN code through the adapter ([`crate::upstream::search::Searcher`]).
struct Upstream<U: VectorRepr + 'static> {
    searcher: Searcher<'static, U>,
    algorithm: Algorithm,
    provider: Provider,
    n_start: u64,
    desc: Value,
}

impl<U: VectorRepr + Element + 'static> Upstream<U> {
    fn open(
        graph: &Path,
        algorithm: Algorithm,
        provider: &str,
        labels: Arc<LabelStore>,
        max_l: usize,
    ) -> Result<Self> {
        let prefix = graph
            .to_str()
            .with_context(|| format!("{}: not UTF-8", graph.display()))?;
        // The index lives as long as the process: the searcher borrows it.
        let loaded: &'static LoadedIndex<U> = Box::leak(Box::new(
            index::load::<U>(prefix).with_context(|| format!("upstream index {prefix}"))?,
        ));
        ensure!(
            loaded.n_data == labels.n_points(),
            "{prefix}: {} data points, the labels have {}",
            loaded.n_data,
            labels.n_points()
        );
        let provider = match provider {
            "forward" => Provider::Forward(Arc::new(ForwardFilter::new(labels))),
            "bitmap" => Provider::Bitmap(Arc::new(BitmapFilter::new(labels))),
            p => bail!("unknown upstream provider {p:?} (forward, bitmap)"),
        };
        let desc = json!({
            "engine": "upstream",
            "algorithm": algorithm.name(),
            "algorithm_params": format!("{algorithm:?}"),
            "provider": match &provider { Provider::Forward(_) => "forward", Provider::Bitmap(_) => "bitmap" },
            "provider_doc": "the adapter's provider (crate::upstream::filter), compiled per query inside the timer; clauses in ascending total posting length",
            "index": prefix,
            "n_data": loaded.n_data,
            "start_ids": loaded.start_ids,
            "max_observed_degree": loaded.header.max_observed_degree,
            "search_api": "DiskANNIndex::search, beam width 1, strategy default post-processor; futures polled on the calling thread",
            "n_dist": "upstream cmps (+ the start-point distances for inline, multihop and pathseer, which upstream does not count) + the adapter's true-distance recomputations (beta)",
        });
        Ok(Self {
            searcher: Searcher::new(&loaded.index, max_l),
            algorithm,
            provider,
            n_start: loaded.start_ids.len() as u64,
            desc,
        })
    }

    fn counters(&self, s: &QueryStats, calls: u64, matches: u64) -> Counters {
        let start = match self.algorithm {
            Algorithm::Inline
            | Algorithm::InlineAdaptive { .. }
            | Algorithm::Multihop
            | Algorithm::PathSeer => self.n_start,
            _ => 0,
        };
        // Multihop and PathSeer count second-hop adjacency reads in `hops`.
        let mixed = matches!(self.algorithm, Algorithm::Multihop | Algorithm::PathSeer);
        Counters {
            n_dist: u64::from(s.up_cmps) + start + u64::from(s.extra_dist),
            n_pred: calls + u64::from(s.post_pred),
            n_expand: if mixed { 0 } else { u64::from(s.up_hops) },
            n_hops: u64::from(s.up_hops),
            n_match_seen: matches,
            n_out: u64::from(s.n_out),
            n_start: self.n_start,
            route: Route::Graph,
            ..Counters::default()
        }
    }
}

impl<T: Element, U: VectorRepr + Element + 'static> External<T> for Upstream<U> {
    fn search(
        &mut self,
        q: &Query<'_, T>,
        l: usize,
        ids: &mut [u32],
        dists: &mut [f32],
    ) -> Result<(Counters, [u64; 5])> {
        let vector: &[U] = bytemuck::cast_slice(q.vector);
        let (stats, calls, matches) = match &mut self.provider {
            Provider::Forward(f) => {
                Arc::get_mut(f)
                    .context("the provider is still shared")?
                    .compile(q.cnf);
                let s = self
                    .searcher
                    .search(&self.algorithm, l, vector, f, ids, dists)?;
                let c = f.counts();
                (s, c.calls, c.matches)
            }
            Provider::Bitmap(f) => {
                Arc::get_mut(f)
                    .context("the provider is still shared")?
                    .compile(q.cnf);
                let s = self
                    .searcher
                    .search(&self.algorithm, l, vector, f, ids, dists)?;
                let c = f.counts();
                (s, c.calls, c.matches)
            }
        };
        Ok((self.counters(&stats, calls, matches), [0; 5]))
    }

    fn describe(&self) -> Value {
        self.desc.clone()
    }
}

/// Upstream PRs #1319 / #1477 through the adapter ([`crate::upstream::encoded`]).
struct Encoded<U: VectorRepr + 'static> {
    loaded: &'static LoadedIndex<U>,
    algorithm: EncodedAlgorithm,
    provider: EncodedProvider,
    n_start: u64,
    desc: Value,
}

impl<U: VectorRepr + Element + 'static> Encoded<U> {
    fn open(
        graph: &Path,
        algorithm: EncodedAlgorithm,
        label_index: &Path,
        labels: Arc<LabelStore>,
    ) -> Result<Self> {
        let prefix = graph
            .to_str()
            .with_context(|| format!("{}: not UTF-8", graph.display()))?;
        let loaded: &'static LoadedIndex<U> = Box::leak(Box::new(
            index::load::<U>(prefix).with_context(|| format!("upstream index {prefix}"))?,
        ));
        ensure!(
            loaded.n_data == labels.n_points(),
            "{prefix}: {} data points, the labels have {}",
            loaded.n_data,
            labels.n_points()
        );
        let provider = EncodedProvider::open(label_index, labels)?;
        let desc = json!({
            "engine": "upstream",
            "algorithm": algorithm.name(),
            "algorithm_doc": match algorithm {
                EncodedAlgorithm::Multihop => "PR #1319: MultihopFilterSearch with the encoded index's CNF plan as the per-node test (static dispatch on the plan's point evaluator)",
                EncodedAlgorithm::Hybrid { .. } => "PR #1477: HybridFilterSearch (U = min over clauses of the sum of label counts; U = 0: empty; U < T: exhaustive scan of the enumerated candidates; else multihop)",
            },
            "provider": "encoded",
            "provider_doc": "diskann-label-index (PR #1477 head 17a8266a, incl. #1319): CNF plan compiled per query inside the timer from clause strings; exact admission (forward-row re-check) of plan positives when the plan reads a Bloom row",
            "label_index": provider.describe(),
            "index": prefix,
            "n_data": loaded.n_data,
            "start_ids": loaded.start_ids,
            "max_observed_degree": loaded.header.max_observed_degree,
            "search_api": "DiskANNIndex::search, beam width 1, strategy default post-processor; futures polled on the calling thread",
            "n_dist": "upstream cmps (+ the start-point distances on the graph route, which upstream does not count); exhaustive route: candidates scored",
            "n_pred": "plan evaluations (graph route) + exact admission checks",
            "extras": "ext_x0 = routing bound U (u64::MAX: not computed), ext_x1 = candidates enumerated by the exhaustive route (before admission), ext_x2 = plan positives rejected by exact admission (Bloom false positives), ext_x3 = exact admission checks",
        });
        Ok(Self {
            loaded,
            algorithm,
            provider,
            n_start: loaded.start_ids.len() as u64,
            desc,
        })
    }
}

impl<T: Element, U: VectorRepr + Element + 'static> External<T> for Encoded<U> {
    fn search(
        &mut self,
        q: &Query<'_, T>,
        l: usize,
        ids: &mut [u32],
        dists: &mut [f32],
    ) -> Result<(Counters, [u64; 5])> {
        let vector: &[U] = bytemuck::cast_slice(q.vector);
        let s = self.provider.search(
            &self.loaded.index,
            self.algorithm,
            l,
            vector,
            q.cnf,
            ids,
            dists,
        )?;
        let route = match s.route {
            Some(EncodedRoute::Graph) | None => Route::Graph,
            Some(EncodedRoute::Exhaustive) => Route::Exact,
            Some(EncodedRoute::Empty) => Route::Empty,
        };
        let graph = route == Route::Graph;
        let counters = Counters {
            n_dist: u64::from(s.up_cmps) + if graph { self.n_start } else { 0 },
            n_pred: s.calls + s.exact_checks,
            n_expand: 0,
            n_hops: u64::from(s.up_hops),
            n_match_seen: s.plan_pos - s.false_pos,
            n_exact_cand: if graph { 0 } else { u64::from(s.up_cmps) },
            n_out: u64::from(s.n_out),
            n_start: if graph { self.n_start } else { 0 },
            route,
            ..Counters::default()
        };
        Ok((
            counters,
            [0, s.bound, s.candidates_enumerated, s.false_pos, s.exact_checks],
        ))
    }

    fn describe(&self) -> Value {
        self.desc.clone()
    }
}

/// Counters of the C ABI.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct ExtCounters {
    n_dist: u64,
    n_pred: u64,
    n_expand: u64,
    n_hops: u64,
    n_match_seen: u64,
    n_out: u64,
    route: u64,
    prep_ns: u64,
    x: [u64; 4],
}

type VersionFn = unsafe extern "C" fn() -> u32;
type OpenFn = unsafe extern "C" fn(
    *const c_char,
    *const c_char,
    *const c_char,
    u32,
    u32,
    *mut c_char,
    usize,
) -> *mut c_void;
type DescribeFn = unsafe extern "C" fn(*mut c_void) -> *const c_char;
type SearchFn = unsafe extern "C" fn(
    *mut c_void,
    *const c_void,
    *const u64,
    *const u32,
    u32,
    u32,
    u32,
    *mut u32,
    *mut f32,
    *mut ExtCounters,
) -> i32;
type CloseFn = unsafe extern "C" fn(*mut c_void);

/// A shared library implementing the C ABI.
struct Dylib {
    engine: *mut c_void,
    search: SearchFn,
    close: CloseFn,
    desc: Value,
}

fn cstring(p: &Path) -> Result<CString> {
    CString::new(p.as_os_str().as_bytes()).context("path with a NUL byte")
}

/// Resolves `name` in the library `lib` (a `dlopen` handle).
///
/// # Safety
///
/// `lib` must be a live handle and `F` the function type of the symbol.
unsafe fn symbol<F: Copy>(lib: *mut c_void, name: &str) -> Result<F> {
    let c = CString::new(name)?;
    // SAFETY: `lib` is a live handle; the name is NUL-terminated.
    let p = unsafe { libc::dlsym(lib, c.as_ptr()) };
    ensure!(!p.is_null(), "symbol {name} not found");
    // SAFETY: the caller guarantees that `F` is a function pointer type of this symbol.
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&p) })
}

impl Dylib {
    fn open(
        library: &Path,
        index: &Path,
        params: &str,
        dataset: &Path,
        dtype: DType,
        dim: usize,
    ) -> Result<Self> {
        let lib_c = cstring(library)?;
        // SAFETY: a NUL-terminated path; the handle is never closed.
        let lib = unsafe { libc::dlopen(lib_c.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if lib.is_null() {
            // SAFETY: dlerror returns a NUL-terminated message (or NULL).
            let msg = unsafe {
                let e = libc::dlerror();
                if e.is_null() {
                    String::from("unknown error")
                } else {
                    CStr::from_ptr(e).to_string_lossy().into_owned()
                }
            };
            bail!("dlopen {}: {msg}", library.display());
        }
        // SAFETY: the symbol types are those of the C ABI documented above.
        let (version, open, describe, search, close) = unsafe {
            (
                symbol::<VersionFn>(lib, "flab_ext_abi_version")?,
                symbol::<OpenFn>(lib, "flab_ext_open")?,
                symbol::<DescribeFn>(lib, "flab_ext_describe")?,
                symbol::<SearchFn>(lib, "flab_ext_search")?,
                symbol::<CloseFn>(lib, "flab_ext_close")?,
            )
        };
        // SAFETY: a function without arguments.
        let v = unsafe { version() };
        ensure!(
            v == 1,
            "{}: ABI version {v}, expected 1",
            library.display()
        );
        let (ds_c, ix_c, pa_c) = (cstring(dataset)?, cstring(index)?, CString::new(params)?);
        let mut err = vec![0u8; 4096];
        let elem = match dtype {
            DType::U8 => 1,
            DType::F32 => 2,
        };
        // SAFETY: NUL-terminated strings and an error buffer of the given length.
        let engine = unsafe {
            open(
                ds_c.as_ptr(),
                ix_c.as_ptr(),
                pa_c.as_ptr(),
                elem,
                u32::try_from(dim)?,
                err.as_mut_ptr().cast(),
                err.len(),
            )
        };
        if engine.is_null() {
            let msg = CStr::from_bytes_until_nul(&err)
                .map(|c| c.to_string_lossy().into_owned())
                .unwrap_or_default();
            bail!("{}: opening the engine failed: {msg}", library.display());
        }
        // SAFETY: a live engine; the string is valid until `close`.
        let text = unsafe {
            let p = describe(engine);
            ensure!(!p.is_null(), "flab_ext_describe returned NULL");
            CStr::from_ptr(p).to_string_lossy().into_owned()
        };
        let mut desc: Value = serde_json::from_str(&text)
            .with_context(|| format!("the engine's description is not JSON: {text}"))?;
        if let Value::Object(m) = &mut desc {
            m.insert("library".into(), json!(library));
            m.insert("library_sha256".into(), json!(super::data::sha256_file(library)?));
            m.insert("abi_version".into(), json!(v));
        }
        Ok(Self {
            engine,
            search,
            close,
            desc,
        })
    }
}

impl Drop for Dylib {
    fn drop(&mut self) {
        // SAFETY: the engine is live and closed once.
        unsafe { (self.close)(self.engine) };
    }
}

impl<T: Element> External<T> for Dylib {
    fn search(
        &mut self,
        q: &Query<'_, T>,
        l: usize,
        ids: &mut [u32],
        dists: &mut [f32],
    ) -> Result<(Counters, [u64; 5])> {
        let (bounds, labels) = q.cnf.raw();
        let mut c = ExtCounters::default();
        // SAFETY: the vector has the dataset's dimension and type (checked at open), the clause
        // bounds index `labels`, and the output slices hold `k` entries.
        let rc = unsafe {
            (self.search)(
                self.engine,
                q.vector.as_ptr().cast(),
                bounds.as_ptr(),
                labels.as_ptr(),
                (bounds.len() - 1) as u32,
                l as u32,
                ids.len() as u32,
                ids.as_mut_ptr(),
                dists.as_mut_ptr(),
                &raw mut c,
            )
        };
        ensure!(rc == 0, "flab_ext_search failed with {rc}");
        let route = match c.route {
            0 => Route::Graph,
            1 => Route::Exact,
            _ => Route::Empty,
        };
        Ok((
            Counters {
                n_dist: c.n_dist,
                n_pred: c.n_pred,
                n_expand: c.n_expand,
                n_hops: c.n_hops,
                n_match_seen: c.n_match_seen,
                n_out: c.n_out,
                route,
                ..Counters::default()
            },
            [c.prep_ns, c.x[0], c.x[1], c.x[2], c.x[3]],
        ))
    }

    fn describe(&self) -> Value {
        self.desc.clone()
    }
}
