# DiskANN Rust — In-Memory Index CLI Tools

## Overview

Two CLI tools for building and searching in-memory DiskANN indices,
analogous to the C++ `apps/build_memory_index` and `apps/search_memory_index`.

- **`build_memory_index`** — Build an in-memory Vamana graph index from `.fbin`
  vector data and save it to disk.
- **`search_memory_index`** — Load a saved index, search it with varying L
  values, and report recall@K and QPS. Supports unfiltered search and three
  filtered-search strategies with an optional brute-force fallback.

Both binaries live in the `diskann-benchmark` crate and are built with:

```bash
cargo build --release -p diskann-benchmark \
  --bin build_memory_index --bin search_memory_index
```

---

## build_memory_index

### Synopsis

```
build_memory_index [OPTIONS] --data_path <PATH> --index_path_prefix <PATH>
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--data_type` | `float` | Element type: `float`, `fp16`, `uint8`, `int8` |
| `--dist_fn` | `l2` | Distance metric: `l2`, `mips`, `cosine` |
| `--data_path` | *(required)* | Input vectors in `.fbin` / `.bin` format |
| `--index_path_prefix` | *(required)* | Output path prefix for saved index |
| `-R, --max_degree` | `64` | Max graph degree |
| `-L, --Lbuild` | `100` | Search list size during build |
| `--alpha` | `1.2` | Pruning diameter parameter (1.0–1.5) |
| `-T, --num_threads` | `0` (auto) | Number of build threads |

### What it does

1. Loads vectors from `--data_path` (reads the standard `[npoints: u32][ndim:
   u32][data...]` binary format).
2. Creates a `DiskANNIndex` via `diskann_async::new_index()`.
3. Computes medoid start point(s).
4. Inserts all vectors sequentially (each triggering Vamana graph search +
   pruning).
5. Saves the graph and data to `--index_path_prefix` (produces two files: the
   graph file and a `.data` companion).

### What it does NOT do

**The build is entirely filter-unaware.** No label files are read; no bitmaps
or inverted indexes are computed during build. The graph is a standard
(unfiltered) Vamana graph. All filtering is applied at search time only.

This is a fundamental difference from the C++ DiskANN, which supports
**filtered-vamana** and **stitched-vamana** build modes that create
filter-aware graphs. See the [Algorithms](#algorithms) section for details.

### Example

```bash
build_memory_index \
  --data_type float --dist_fn l2 \
  --data_path siftsmall/siftsmall_base.fbin \
  --index_path_prefix siftsmall/index_R32_L50 \
  -R 32 -L 50 --alpha 1.2 -T 8
```

---

## search_memory_index

### Synopsis

```
search_memory_index [OPTIONS] \
  --index_path_prefix <PATH> --query_file <PATH> --gt_file <PATH> \
  -L <L1> [L2 ...]
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--data_type` | `float` | Element type |
| `--dist_fn` | `l2` | Distance metric |
| `--index_path_prefix` | *(required)* | Saved index path prefix |
| `--query_file` | *(required)* | Query vectors (`.fbin` / `.bin`) |
| `--gt_file` | *(required)* | Ground truth file (`"null"` to skip recall) |
| `-K` | `10` | Number of neighbors |
| `-L, --search_list` | *(required)* | One or more search list sizes |
| `-T, --num_threads` | `1` | Search threads |
| `--search_reps` | `1` | Repeat each search N times for stable QPS |
| `--result_path` | *(none)* | Prefix for result output files |
| `--data_labels` | *(none)* | Base vector labels JSONL file |
| `--query_labels` | *(none)* | Query predicates JSONL file |
| `--filter_strategy` | `none` | `none`, `beta`, or `multihop` |
| `--beta` | `0.5` | Beta factor for `beta` strategy (0, 1] |
| `--data_path` | *(none)* | Base vectors for brute-force fallback |
| `--brute_force_threshold` | `0` | Match count below which brute-force is used |

### Output

A header block reports the strategy, beta (if applicable), and the bitmap
precomputation time:

```
Strategy: beta-filter
Beta: 0.5
Bitmap precomputation: 12.34 ms
```

Then a table with one row per L value:

```
      Ls         QPS  Mean Lat(us)   p99 Lat(us)   Recall@10
================================================================
      10    51813.47         19.30         31.00       99.00
      20    40000.00         25.00         37.00      100.00
```

When `--brute_force_threshold > 0`, an extra **BF Qrys** column shows how many
queries used brute-force.

### Filter strategies

#### `--filter_strategy none` (default)

If no labels are provided, runs a standard unfiltered search.

If labels are provided (via `--data_labels` and `--query_labels`), runs an
unfiltered graph search requesting **L candidates** (the full search list, not
just K), then post-filters those L candidates against the query's filter
bitmap, and finally returns the best K matching results. Requesting L instead
of K gives the post-filter more candidates to work with, significantly
improving filtered recall at the cost of a larger output buffer.

This is the simplest filtered strategy but can still yield poor recall for
very selective filters because the graph search does not bias traversal toward
matching vectors — it may spend the entire L-sized budget on non-matching
nodes.

#### `--filter_strategy beta`

Uses **BetaFilter**: wraps the search strategy so that distances to matching
vectors are multiplied by `beta` (< 1), biasing the beam search toward them.
Non-matching vectors can still appear in results (soft filter); the post-filter
step removes them.

**Parameters:**
- `--beta`: distance multiplier for matching vectors. Lower = stronger bias.
  Must be in (0, 1]. Default 0.5.

#### `--filter_strategy multihop`

Uses **MultihopSearch**: extends the beam search with two-hop expansion.
When a candidate doesn't match the filter, its neighbors are expanded (one
extra hop) to discover matching vectors reachable through non-matching
intermediaries. Only matching vectors enter the result set (hard filter).

**Parameters:** None beyond the standard search parameters.

### Brute-force fallback

When `--brute_force_threshold N` is set (N > 0) and `--data_path` is provided:

1. Before searching each query, the tool inspects its filter bitmap and counts
   the number of matching points.
2. If `matching_count < N`, a brute-force linear scan is performed over *only*
   the matching points instead of using the graph index.
3. Otherwise, the graph search proceeds normally with the selected filter
   strategy.

This is useful for highly selective queries where the matching set is tiny. For
such queries, brute-force is both faster (no graph navigation overhead) and
gives perfect recall.

The bitmap statistics (min/max/mean match counts and the number of brute-force
queries) are printed at startup.

### Examples

**Unfiltered:**
```bash
search_memory_index \
  --data_type float --dist_fn l2 \
  --index_path_prefix siftsmall/index_R32_L50 \
  --query_file siftsmall/siftsmall_query.fbin \
  --gt_file siftsmall/gt.bin \
  -K 10 -L 10 20 30 40 50 100
```

**Beta filter with brute-force fallback:**
```bash
search_memory_index \
  --data_type float --dist_fn l2 \
  --index_path_prefix siftsmall/index_R32_L50 \
  --query_file siftsmall/siftsmall_query.fbin \
  --gt_file siftsmall/gt_filtered.bin \
  --data_labels siftsmall/base_labels.jsonl \
  --query_labels siftsmall/query_labels.jsonl \
  --filter_strategy beta --beta 0.3 \
  --data_path siftsmall/siftsmall_base.fbin \
  --brute_force_threshold 500 \
  -K 10 -L 10 20 50 100
```

**Multihop:**
```bash
search_memory_index \
  --data_type float --dist_fn l2 \
  --index_path_prefix siftsmall/index_R32_L50 \
  --query_file siftsmall/siftsmall_query.fbin \
  --gt_file null \
  --data_labels siftsmall/base_labels.jsonl \
  --query_labels siftsmall/query_labels.jsonl \
  --filter_strategy multihop \
  -K 10 -L 10 20 50 100
```

---

## Label and Filter File Formats

### Base vector labels (JSONL)

One JSON object per line. Each must have a `doc_id` field matching the
zero-based vector index. All other fields are arbitrary attributes:

```json
{"doc_id": 0, "year": "2010", "month": "May", "camera": "Panasonic", "country": "US"}
{"doc_id": 1, "year": "2011", "month": "July", "camera": "Canon"}
{"doc_id": 2, "year": "2011", "month": "July", "camera": "Canon"}
```

### Query predicates (JSONL)

One JSON object per line with `query_id` and `filter`:

```json
{"query_id": 0, "filter": {"camera": {"$eq": "NIKON"}}}
{"query_id": 1, "filter": {"$and": [{"year": {"$gt": 2009}}, {"camera": {"$eq": "Canon"}}]}}
{"query_id": 2, "filter": {"$or": [{"camera": {"$eq": "Canon"}}, {"camera": {"$eq": "NIKON"}}]}}
```

### Supported filter operators

| Operator | JSON key | Value type | Semantics |
|----------|----------|------------|-----------|
| Equals | `$eq` | any JSON value | field == value |
| Not equal | `$ne` | any JSON value | field != value |
| Less than | `$lt` | number | field < value |
| Less or equal | `$lte` | number | field ≤ value |
| Greater than | `$gt` | number | field > value |
| Greater or equal | `$gte` | number | field ≥ value |
| In | `$in` | array | field ∈ values |
| Not in | `$nin` | array | field ∉ values |

### Logical operators

- `$and`: `{"$and": [expr1, expr2, ...]}`
- `$or`: `{"$or": [expr1, expr2, ...]}`
- `$not`: `{"$not": expr}`

Multiple fields at the top level without `$` prefix are implicitly AND'd.
Dot notation supports nested fields (e.g., `"specs.cpu": {"$eq": "i7"}`).
Maximum nesting depth is 2.

---

## Algorithms

### Graph construction (Vamana)

The Rust codebase builds a **standard, unfiltered Vamana** graph:

1. Initialize with a random graph or medoid start points.
2. For each vector, perform a greedy search on the current graph to find
   approximate nearest neighbors.
3. Prune the candidate set using the alpha-pruning rule (RobustPrune) to
   select the final edges, trading off between short-range accuracy and
   long-range graph connectivity.
4. Insert backedges to maintain graph quality.

Key parameters:
- **R (max_degree)**: Target graph degree. Higher R = better recall, more
  memory, slower build.
- **L (Lbuild)**: Search list size during build. Higher L = better graph
  quality, slower build.
- **alpha**: Pruning factor (1.0–1.5). Controls graph diameter; lower alpha
  yields sparser graphs.

### Search-time filter strategies

Since the graph is built without filter awareness, all filtering happens at
query time. Three strategies are available:

#### 1. BetaFilter (soft filter + post-filter)

**Crate:** `diskann-providers`
**Key type:** `BetaFilter<Strategy, I>`

For each candidate during beam search, the distance is multiplied by `beta` if
the candidate matches the query filter. This makes matching vectors appear
closer, biasing the greedy search toward them. After search completes, a
post-filter step removes any remaining non-matching results.

**Parameters:**
- `beta` ∈ (0, 1]: Lower values bias more aggressively. Default: 0.5.

**Characteristics:**
- Soft filter during search; hard filter in post-processing.
- Same algorithmic cost as unfiltered search (no extra graph traversal).
- Works well when a significant fraction of vectors match the filter.
- Degrades for very selective filters — most of the search budget is spent
  visiting non-matching vectors even with the distance bias.

#### 2. MultihopSearch (hard filter)

**Crate:** `diskann` (core)
**Key type:** `MultihopSearch<InternalId>`

Extends the beam search with two-hop expansion:

1. Standard beam expansion: visit the closest unexpanded nodes.
2. For each visited neighbor, call `on_visit()`:
   - **Accept** (matches filter) → insert into best-candidates list.
   - **Reject** (doesn't match) → add to two-hop expansion queue.
   - **Terminate** → stop search immediately.
3. Expand rejected neighbors' adjacency lists (second hop), applying both the
   visited-set check and the label match check.
4. Matching two-hop neighbors are inserted into the best-candidates list.
5. Repeat until convergence.

**Characteristics:**
- Hard filter: only matching vectors enter results. No post-filter needed.
- More expensive than BetaFilter: each iteration does an additional expansion
  round through non-matching neighbors.
- Better than BetaFilter for medium-selectivity filters where matching vectors
  are reachable within 2 hops.
- Effectiveness degrades when matching vectors are >2 hops away in the graph.

#### 3. InlineBetaStrategy (rich predicate evaluation)

**Crate:** `diskann-label-filter`
**Key type:** `InlineBetaStrategy<Strategy>`

Combines beta-weighted scoring with per-document predicate evaluation against a
`RoaringAttributeStore`. Unlike BetaFilter (which uses precomputed bitmaps),
this evaluates the AST filter expression inline during search by reading each
candidate's encoded attributes. In post-processing, non-matching candidates are
removed (hard filter).

This strategy is used internally by the benchmark framework's
`DocumentProvider` path and is **not exposed via the CLI tool**. The reason is
architectural: InlineBetaStrategy requires the index to be wrapped in a
`DocumentProvider<DP, RoaringAttributeStore<...>>`, which is a different
index type from the `DiskANNIndex<FullPrecisionProvider<T>>` that the CLI
creates/loads. Using InlineBetaStrategy would require:

1. Creating a `RoaringAttributeStore` and populating it from the JSONL labels
   (iterating all documents, converting JSON attributes to `Attribute` structs,
   calling `set_element()` for each).
2. Wrapping the loaded `FullPrecisionProvider` in a `DocumentProvider`.
3. Rebuilding or re-loading the `DiskANNIndex` with this composite provider.
4. Using `FilteredQuery<[T]>` as the query type instead of raw `&[T]`.

This is a substantial integration effort for marginal benefit over BetaFilter
with precomputed bitmaps — both apply the same beta-weighted distance
adjustment. The main advantage of InlineBetaStrategy is that it doesn't need
bitmap precomputation (it evaluates predicates inline per-candidate during
search), which could matter for very large datasets where precomputing bitmaps
for all queries is expensive. But for the CLI tool's use case, bitmap
precomputation is fast enough (the timing is reported in the output).

### Bitmap computation

When `--data_labels` and `--query_labels` are provided, the search tool
precomputes per-query filter bitmaps:

1. Read all base-vector labels from the JSONL file.
2. Parse each query's filter expression into an AST.
3. For each query, evaluate its AST against every base-vector label to produce
   a `BitSet` of matching vector IDs.

This is a brute-force O(num_queries × num_vectors) computation. For each
(query, vector) pair, the AST is evaluated recursively:
- `AND` → `all()` (short-circuit)
- `OR` → `any()` (short-circuit)
- `NOT` → negate
- `Compare` → field lookup + operator check

For AND-of-OR queries, the evaluator short-circuits: if any AND sub-expression
is false, it stops evaluating the remaining sub-expressions. Similarly, OR
sub-expressions short-circuit on the first true match.

**Alternative (not used by CLI):** The crate also contains a `GenericIndex`
backed by `BfTreeStore` which maintains an inverted index (attribute-value →
posting list of matching doc IDs). The `evaluate_query()` method on this index
computes AND as posting-list intersection and OR as union, which is
asymptotically more efficient. However, using it requires building and
populating the inverted index first, which the CLI tool does not do. The
bitmap approach is simpler and sufficient for datasets up to a few million
points.

### Brute-force fallback

When a query's filter matches very few points (below `--brute_force_threshold`),
no graph navigation is needed. The tool performs a linear scan over only the
matching points (identified by the bitmap), computing exact distances to the
query and maintaining a size-K max-heap of nearest neighbors.

This gives **perfect recall** for those queries and is faster than graph search
when the matching set is small (typically < a few hundred points, depending on
dimensionality).

---

## What's missing compared to C++ DiskANN

| Feature | C++ | Rust | Impact |
|---------|-----|------|--------|
| **Filtered-Vamana build** | `--label_file --FilteredLbuild` | Not implemented | No filter-aware graph construction; edges don't preferentially connect same-label vectors |
| **Stitched-Vamana build** | `build_stitched_index --Stitched_R` | Not implemented | No per-label subgraph construction or graph merging |
| **Per-label medoid start points** | Computed during filtered build | Not implemented | All searches start from the global medoid regardless of filter |
| **Universal label** | `--universal_label` during build | N/A | Concept applies only to filtered build |
| **FilteredLbuild** | Separate L for filtered construction | N/A | No filtered build |
| **Single-label query** | `--filter_label 35` (same label for all queries) | Different model | Rust uses per-query AST expressions in JSONL |
| **Query routing / selectivity** | Not in open-source C++ | Not present | No automatic algorithm selection based on selectivity |

### Impact of missing filtered build

Without filtered-vamana or stitched-vamana, the graph is built purely on
distance. For datasets with highly selective filters:

1. Matching vectors may be many hops apart, degrading recall for both
   BetaFilter and MultihopSearch.
2. MultihopSearch partially compensates with two-hop expansion but is limited
   to 2 hops.
3. BetaFilter only biases distance scoring; with very selective filters, most
   search budget is wasted on non-matching vectors.
4. The **brute-force fallback** fully compensates for these cases — when few
   points match, linear scan is both faster and gives perfect recall.

---

## Architecture reference

### Crate map

| Crate | Role |
|-------|------|
| `diskann` | Core graph, search algorithms (`Knn`, `MultihopSearch`), `QueryLabelProvider` trait |
| `diskann-providers` | Storage, index creation (`new_index`), `BetaFilter` search strategy, `FullPrecisionProvider` |
| `diskann-label-filter` | AST parser, evaluator, JSONL reader, `RoaringAttributeStore`, `InlineBetaStrategy`, `GenericIndex` |
| `diskann-benchmark` | CLI tools (`build_memory_index`, `search_memory_index`), benchmark framework |
| `diskann-benchmark-core` | Benchmark search harness (KNN, MultiHop, linear search) |
| `diskann-tools` | Utilities (`compute_groundtruth`, `generate_synthetic_labels`) |
| `diskann-vector` | Distance functions, SIMD acceleration |

### Key types

| Type | Location | Purpose |
|------|----------|---------|
| `DiskANNIndex<DP>` | `diskann::graph` | The core index holding graph + data provider |
| `Knn` | `diskann::graph::search` | Standard k-NN beam search parameters |
| `MultihopSearch` | `diskann::graph::search` | Two-hop filtered search parameters |
| `BetaFilter<S, I>` | `diskann_providers::model::graph::provider::layers` | Beta-weighted distance filter strategy |
| `QueryLabelProvider<V>` | `diskann::graph::index` | Trait for per-query label matching |
| `ASTExpr` | `diskann_label_filter::parser::ast` | Parsed filter expression tree |
| `FilteredQuery<V>` | `diskann_label_filter::query` | Query vector + filter expression wrapper |
| `RoaringAttributeStore` | `diskann_label_filter::encoded_attribute_provider` | Forward + inverted index for attributes |

### Benchmark JSON configuration

The benchmark framework (separate from these CLI tools) supports filtered
search via JSON config files:

**topk-beta-filter:**
```json
{
  "search-type": "topk-beta-filter",
  "queries": "queries.fbin",
  "groundtruth": "gt.bin",
  "beta": 0.5,
  "query_predicates": "query_labels.jsonl",
  "data_labels": "data_labels.jsonl",
  "reps": 5,
  "num_threads": [1],
  "runs": [{"search_n": 20, "search_l": [20, 30, 40], "recall_k": 10}]
}
```

**topk-multihop-filter:**
```json
{
  "search-type": "topk-multihop-filter",
  "queries": "queries.fbin",
  "groundtruth": "gt.bin",
  "query_predicates": "query_labels.jsonl",
  "data_labels": "data_labels.jsonl",
  "reps": 5,
  "num_threads": [1],
  "runs": [{"search_n": 20, "search_l": [20, 30, 40, 50, 100], "recall_k": 10}]
}
```

### Default constants

| Constant | Value | Location |
|----------|-------|----------|
| `FILTER_BETA` | 0.5 | `diskann/src/graph/config/defaults.rs` |
| `ALPHA` | 1.2 | `diskann/src/graph/config/defaults.rs` |
| `GRAPH_SLACK_FACTOR` | 1.3 | `diskann/src/graph/config/defaults.rs` |
| `MAX_OCCLUSION_SIZE` | 750 | `diskann/src/graph/config/defaults.rs` |
| `ALLOWED_DEPTH_LIMIT` | 2 | `diskann-label-filter` parser |

---

## Design note: why not use the benchmark framework?

The `diskann-benchmark-core` crate provides a search framework with threading,
result aggregation, metrics collection, and parameter sweeps
(`search::search()` and `search::search_all()` APIs). The existing benchmark
harness in `diskann-benchmark` uses this framework with JSON configuration
files.

The CLI tools **do not use this framework**, for several reasons:

1. **Per-query brute-force fallback.** The framework's `Search` trait runs the
   same search strategy for all queries. The CLI tool inspects each query's
   bitmap cardinality and routes low-selectivity queries to brute-force — a
   per-query decision the framework doesn't support.

2. **Post-filter with L candidates.** In `filter_strategy=none` mode, the tool
   searches for L (not K) candidates and post-filters. This changes `Knn::new`
   parameters per-strategy, which is straightforward in a direct loop but
   would require a custom `Search` implementation in the framework.

3. **Simplicity.** The CLI is a thin loop: load data → for each L → for each
   query → search → collect stats → print table. The framework would add
   indirection (trait impls, aggregator types, `Run` wrappers) without
   reducing complexity for this use case.

4. **JSON-driven design.** The BENCHMARK framework is designed around JSON
   config files (`SearchPhase` enum deserialized via serde). The CLI tools are
   designed around command-line flags. Bridging the two would require either
   synthesizing JSON configs from CLI args or bypassing the JSON layer entirely.

The framework's search primitives (`KNN`, `MultiHop`) *could* technically be
called directly from Rust code without JSON, but the integration work would
exceed the effort saved for the current feature set.
