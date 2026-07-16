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
| `--filter_strategy` | `none` | `none`, `beta`, `multihop`, `inline_beta`, `inline_beta_bf`, or `inline_beta_bfx` |
| `--beta` | `0.5` | Beta factor for `beta`/`inline_beta*` strategies (0, 1] |
| `--data_path` | *(none)* | Base vectors for brute-force; required whenever `--brute_force_threshold > 0` uses a BF path |
| `--brute_force_threshold` | `0` | Exact-match-count fallback threshold for non-inline strategies; routing/augmentation budget for `inline_beta_bf` / `inline_beta_bfx` |

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

For non-inline strategies, when `--brute_force_threshold > 0`, an extra
**BF Qrys** column shows how many queries used brute-force. The inline hybrids
report their BF decisions in the routing summary instead.

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

#### `--filter_strategy inline_beta`

Same as `inline_beta` documented in the [Filter strategies](#filter-strategies-detail)
section below: BetaFilter graph search using the compact, vector-backed forward
index in `RoaringAttributeStore` for per-node predicate evaluation (no
per-query bitmap precomputation).

#### `--filter_strategy inline_beta_bf` (V1: bf-or-graph hybrid)

Hybrid of `inline_beta` and pure brute-force, dispatched per query based on a
static analysis of the query's encoded predicate AST. Requires `--data_path`
and `--brute_force_threshold T` (T > 0).

**Per-query routing:**

1. **AST shape check.** Walk the encoded AST. Accept only *AND-of-ORs* form
   (top-level `AND` of OR sub-expressions, each OR being a flat disjunction of
   terminal label literals; bare `OR` and single terminals also count). Any
   `NOT` or non-flat shape disqualifies the query — it falls back to plain
   graph search.
2. **Upper-bound (UB) on the result set.** For each AND-conjunct
   *Cᵢ = (lᵢ,₁ OR lᵢ,₂ OR …)*, compute the *literal sum*
   *mᵢ = Σⱼ |posting_list(lᵢ,ⱼ)|* (cheap O(num_categories_in_OR) lookup —
   no bitmap union). The intersection size is bounded by *UB = minᵢ mᵢ*.
   * **UB = 0** → query is *unsatisfiable*: skip search, return sentinel IDs.
   * **UB ≤ T** → take the minimising conjunct *C\**, union its posting
     lists into a candidate set, and brute-force top-K over that set,
     **re-checking the full filter** for each candidate so other conjuncts
     are honoured. Path: **BfV1**.
   * **UB > T** → run normal `inline_beta` graph search. Path: **Graph**.
3. Latency is dominated by either graph search or by the candidate-set scan,
   never both.

**Parameters:**
- `--beta` ∈ (0, 1]: passed through to the graph-path BetaFilter.
- `--data_path`: full base-vector matrix; loaded once.
- `--brute_force_threshold T`: routing threshold (in candidate count).

**When it helps:** queries whose minimising category is small (e.g. a rare
geolocation in conjunction with a common date range). For those queries
brute-force is both faster than the graph search and gives perfect recall.
Non-canonical AST shapes are silently routed through the graph path, so this
mode is safe to enable on mixed workloads.

#### `--filter_strategy inline_beta_bfx` (V2: bfx-augmented graph)

Graph + brute-force augmentation. Same prelude as V1 (AND-of-ORs analysis,
UB, unsatisfiable detection). When UB > T, instead of giving up on
brute-force, V2 *augments* the graph search with a partial brute-force scan
over a carefully chosen subset of *rare* labels.

**Per-query routing:**

1. **Unsatisfiable / BfV1 / non-canonical:** same as V1.
2. **UB > T (graph path):** sort all OR-literals across all AND-conjuncts by
   ascending posting-list size *m*. Greedily accumulate the rarest labels
   into a set *R* while the running literal sum stays ≤ *T*. If at least one
   label fits, take their roaring union *S = ⋃_{l∈R} posting_list(l)* and:
   * Run the normal `inline_beta` graph search → top-K.
   * Brute-force top-K over *S* (re-checking the full filter).
   * Merge the two top-K lists (dedup, sort, take K). Path: **BfxAugment**.
3. If even the rarest label exceeds *T*, fall through to plain graph search.

**Why:** the brute-force pass guarantees that any true neighbour matching one
of the rare labels is found, plugging the recall gap that graph-only search
leaves on highly selective conjuncts. The graph pass still covers the bulk of
the predicate. *T* directly bounds the extra BF work per query.

**Parameters:** same as `inline_beta_bf`.

**Routing summary.** Both V1 and V2 print a one-shot summary after parsing
queries, e.g.:

```
Routing summary (9976 queries):
  Unsatisfiable:                        1 (  0.0%)
  Non-canonical (fallback->graph):      0 (  0.0%)
  Brute force V1 (UB<=T):             428 (  4.3%)  mean UB=989  mean |C|=989
  Graph + BFX rare-label aug:        7084 ( 71.0%)  mean UB=678516  mean rare_sum=486  mean |S|=478
  Graph only:                        2463 ( 24.7%)
```

### Brute-force fallback for non-inline strategies

When `--brute_force_threshold N` is set (N > 0) and `--data_path` is provided:

1. Before searching each query, the tool inspects its filter bitmap and counts
   the number of matching points.
2. If `matching_count < N`, a brute-force linear scan is performed over *only*
   the matching points instead of using the graph index.
3. Otherwise, the graph search proceeds normally with the selected filter
   strategy.

This check happens before dispatching the graph strategy, so it applies to all
three non-inline filtered modes: `none`, `beta`, and `multihop`. In filtered
`none` mode it replaces the usual unfiltered graph search plus post-filtering
for sufficiently small matching sets. It does not apply to unfiltered `none`
(there is no filter bitmap to count) or plain `inline_beta`.

The benchmark name `beta_bf` refers to regular `beta` invoked with this generic
fallback enabled. `inline_beta_bf` and `inline_beta_bfx` instead use the
upper-bound routing and augmentation algorithms described in their own
sections above; they do not use this exact-bitmap fallback.

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

**Inline beta (no bitmap precomputation):**
```bash
search_memory_index \
  --data_type uint8 --dist_fn l2 \
  --index_path_prefix siftsmall/index_R32_L50 \
  --query_file siftsmall/siftsmall_query.fbin \
  --gt_file siftsmall/gt_filtered.bin \
  --data_labels siftsmall/base_labels.jsonl \
  --query_labels siftsmall/query_labels.jsonl \
  --filter_strategy inline_beta --beta 0.5 \
  -K 10 -L 10 20 50 100
```

**Inline beta + brute-force hybrid (V1):**
```bash
search_memory_index \
  --data_type uint8 --dist_fn l2 \
  --index_path_prefix siftsmall/index_R32_L50 \
  --query_file siftsmall/siftsmall_query.fbin \
  --gt_file siftsmall/gt_filtered.bin \
  --data_labels siftsmall/base_labels.jsonl \
  --query_labels siftsmall/query_labels.jsonl \
  --filter_strategy inline_beta_bf --beta 0.5 \
  --data_path siftsmall/siftsmall_base.fbin \
  --brute_force_threshold 2000 \
  -K 10 -L 10 20 50 100
```

**Inline beta + bfx graph-augmenting brute-force (V2):**
```bash
search_memory_index \
  --data_type uint8 --dist_fn l2 \
  --index_path_prefix siftsmall/index_R32_L50 \
  --query_file siftsmall/siftsmall_query.fbin \
  --gt_file siftsmall/gt_filtered.bin \
  --data_labels siftsmall/base_labels.jsonl \
  --query_labels siftsmall/query_labels.jsonl \
  --filter_strategy inline_beta_bfx --beta 0.5 \
  --data_path siftsmall/siftsmall_base.fbin \
  --brute_force_threshold 2000 \
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

#### 3. InlineBeta (inline encoded predicate evaluation)

**Crate:** `diskann-label-filter` (attribute store + encoded filter evaluation)
**CLI:** `--filter_strategy inline_beta --beta 0.5`

Combines beta-weighted scoring with per-node predicate evaluation against a
`RoaringAttributeStore`. Unlike BetaFilter (which uses precomputed per-query
bitmaps), InlineBeta evaluates the query's encoded filter expression inline
during graph traversal by looking up each candidate's encoded attributes in the
store's compact, vector-backed forward index. Despite the store's name, Roaring
is used only for its inverse posting lists, not for this per-node lookup.

**How it works:**

1. **Precomputation (before search):** Base labels are loaded from the JSONL
   file and inserted into a `RoaringAttributeStore`, which stores each point's
   encoded attribute IDs as a sorted `Box<[u32]>` in a vector-backed forward
   index. It also maintains Roaring inverse posting lists for bulk set
   operations. Each query's `ASTExpr` is then encoded into an
   `EncodedFilterExpr` using the store's attribute map — this converts string
   field+value comparisons into integer lookups.
2. **Per query:** An `InlineLabelProvider` is created, wrapping a reference to
   the `RoaringAttributeStore` and the query's `EncodedFilterExpr`. This
   implements `QueryLabelProvider<u32>` and is passed to `BetaFilter::new()`.
3. **During search:** For each candidate node visited, `is_match(vec_id)`
   calls `RoaringAttributeStore::matches_filter()`, which reads the point's
   encoded attribute slice by directly indexing the forward `Vec`, then
   evaluates the predicate with `SlicePredicateEvaluator`. Terminal checks use
   a linear scan for small slices and binary search for larger slices — no
   Roaring operation or JSON parsing occurs during search.
4. **Post-filter:** Same as BetaFilter — non-matching candidates removed.

**Parameters:**
- `--beta` ∈ (0, 1]: Same as BetaFilter. Default: 0.5.

**Characteristics:**
- Same recall behavior as BetaFilter (both apply beta-weighted distance).
- No per-query bitmap precomputation — the reported QPS includes the full
  label-evaluation cost per node visit.
- Attribute store construction is a one-time cost reported separately.
- Uses the same `FPIndex<T>` (plain `DiskANNIndex<FullPrecisionProvider<T>>`)
  as all other strategies — no `DocumentProvider` wrapper needed.
- Slightly higher per-query cost than bitmap BetaFilter (predicate evaluation
   over a small sorted slice vs BitSet membership test), but avoids the
   O(queries × points) bitmap
  precomputation.

#### 4. InlineBetaBf (V1: inline_beta + brute-force routing)

**Crate:** `diskann-label-filter` + `diskann-benchmark` (routing logic lives in
the binary)
**CLI:** `--filter_strategy inline_beta_bf --beta 0.5 --brute_force_threshold T --data_path ...`

A static, per-query dispatcher: each query is routed *either* to brute-force
*or* to the `inline_beta` graph search, based on a cheap upper-bound estimate
of its result-set size derived from the inverse index.

**How it works:**

1. **AST canonicalisation.** Walk the encoded predicate AST (`ASTIdExpr<u64>`).
   Accept only AND-of-OR-of-literal expressions (i.e. CNF where every clause
   is a disjunction of label terminals). Nested `OR` is flattened. `NOT` or
   any other shape disqualifies the query from BF routing — it goes to the
   graph path unchanged.
2. **Cheap UB.** For each AND-conjunct *Cᵢ*, look up posting-list lengths of
   its literals on `RoaringAttributeStore::posting_list_len()` (an O(1)
   `len()` on a treemap) and sum them: *mᵢ = Σⱼ |posting_list(lᵢ,ⱼ)|*. The
   *literal sum* `mᵢ` is an upper bound on `|Cᵢ|` (union ≤ sum) and therefore
   `min_i mᵢ` is an upper bound on the intersection. No actual bitmap union
   is computed.
3. **Dispatch.**
   * `UB == 0`: query is unsatisfiable — return sentinel IDs, skip all work.
   * `UB ≤ T`: take the minimising conjunct *C\**, union its posting lists
     into a `RoaringTreemap` (`union_posting_lists()`), and brute-force
     top-K over that set with `matches_filter()` re-checking the full
     predicate. **Path: BfV1.**
   * `UB > T`: standard `inline_beta` graph search. **Path: Graph.**
4. **Brute force.** `brute_force_topk` iterates the treemap, computes raw
   distances via `T::distance_comparer` against `Matrix<T>::row()`, and
   maintains a size-K max-heap.

**Characteristics:**
- BF gives perfect recall on the routed queries — strictly improves recall vs
  `inline_beta` whenever the routed queries had non-trivial recall loss.
- At low *L*, BF often *also* improves latency: skipping graph navigation for
  a small candidate set is faster than running the full graph search.
- *T* controls the recall/QPS trade-off: larger *T* routes more queries to BF
  (more recall gain, but each BF query is more expensive).
- The UB is loose (sum, not union), so some BF-eligible queries are sent to
  graph search. That's acceptable: the goal is a cheap gate, not a tight
  one.
- Routing decisions and means are printed once at startup.

#### 5. InlineBetaBfx (V2: inline_beta with rare-label augmentation)

**Crate:** `diskann-label-filter` + `diskann-benchmark`
**CLI:** `--filter_strategy inline_beta_bfx --beta 0.5 --brute_force_threshold T --data_path ...`

Same prelude as V1 — AND-of-OR canonicalisation, UB, unsatisfiable
detection, BfV1 routing for `UB ≤ T`. The difference is on the `UB > T`
branch: instead of giving up entirely on brute-force, V2 *augments* the
graph search with a partial BF scan over the **rarest labels in the
predicate**, then merges the two top-K result lists.

**How it works (for `UB > T` queries):**

1. **Rare-label selection.** Collect every literal across every AND-conjunct
   into a list of `(m_l, label_id)` pairs (m_l = `posting_list_len(label_id)`).
   Sort ascending by *m_l*. Greedily accumulate the rarest labels into *R*
   while the running literal sum stays ≤ *T*. If at least one label fits,
   take their roaring union *S* via `union_posting_lists(R)`.
2. **Graph search.** Run `inline_beta` exactly as `--filter_strategy
   inline_beta` would.
3. **Augmenting brute-force.** Run `brute_force_topk` over *S* (re-checking
   the full filter on each member). This guarantees every true neighbour
   matching one of the rare labels is found.
4. **Merge.** Dedup by ID, sort by distance, take the top-K from the union of
   the two result lists. Latency = graph search + BF over *S* (bounded by
   *T*).

> **Note on distance scaling.** The graph traversal uses `BetaFilter`, which
> multiplies the inner L2 distance by `beta` for documents matching the
> predicate (and leaves non-matching docs at raw distance — soft filtering).
> To merge safely against BF results (raw L2 over strictly-matching docs),
> V2 (a) re-checks each graph candidate against the filter and drops
> non-matches, and (b) recomputes raw L2 distances for the surviving
> candidates so both sides of the merge are on the same scale. Without
> this, the beta-scaled graph distances would always sort ahead of the
> raw BF distances, defeating the merge.

**Trade-off:**
- Strictly ≥ `inline_beta` recall on routed queries (BF is exhaustive over
  *S*). Adds extra recall over V1 by also catching the rare-label intersections
  that V1's UB skipped past.
- Costs an extra BF pass per `UB > T` query — typically most of the workload —
  so QPS is *lower* than plain `inline_beta` and lower than V1 at the same *T*.
- Pay this when recall is the priority and you can afford the cost.

**Helpers introduced for V1/V2:**

```rust
// On RoaringAttributeStore
pub fn posting_list_len(&self, label_id: u64) -> usize;
pub fn union_posting_lists(&self, label_ids: &[u64]) -> RoaringTreemap;

// In search_memory_index.rs
fn analyze_and_of_ors(expr: &ASTIdExpr<u64>) -> Option<Vec<Vec<u64>>>;
fn brute_force_topk<T, F>(query, base_data, candidates, keep, k, metric)
    -> Vec<(u32, f32)>;
fn merge_topk(graph: &[(u32, f32)], bf: &[(u32, f32)], k: usize) -> Vec<u32>;
```

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

### Non-inline fallback implementation

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
| `RoaringAttributeStore` | `diskann_label_filter::encoded_attribute_provider` | Vector-backed forward index + Roaring inverted index for attributes |

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
