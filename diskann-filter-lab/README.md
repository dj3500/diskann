# diskann-filter-lab

Research tooling for filtered approximate nearest neighbor search with predicates in conjunctive
normal form (an AND of ORs over Boolean labels). The crate provides:

- compact stores for vectors, labels and predicates with allocation-free accessors
  (`Matrix::row`, `LabelStore::row` / `postings`, `PredicateSet::get(q).clause(i)`), meant to be
  used by search kernels;
- binary file formats for labels, predicates and ground truth, and converters from text formats;
- exact filtered k-NN ground truth with complete distance-tie bands;
- graphs over data points and auxiliary nodes (`Graph::neighbors`, `NodeLabels`, `NodeVectors`)
  with start-node tables, and an importer of saved DiskANN in-memory indices (C++ library and
  Rust `diskann-providers`);
- a configurable filtered graph-search kernel (`kernel`) with reference presets of known
  filtered-search algorithms, and `flab run`, a timed experiment driver (`run`);
- the `flab` command-line tool.

## Concepts

- **Points** have dense `u32` ids in input order. **Labels** are Boolean attributes with dense
  `u32` ids; every label belongs to one anonymous **category** (dense `u32`). In the text format the
  category of `name=value` is `name` (labels without `=` share one unnamed category); in JSONL the
  category of the key `prefix_rest` is `prefix`. Category names are kept only in the dictionary.
- Label ids are **contiguous per category**. Categories are numbered in the natural order of their
  names, and labels within a category follow the natural order of their names (digit runs compare
  numerically: `a2 < a10`, `id=19 < id=407`). Every forward row is therefore sorted and grouped by
  category.
- A **predicate** is a list of clauses; a clause is a sorted list of label ids and is satisfied by a
  point carrying at least one of them. A point matches when it satisfies every clause. Clauses may
  mix categories and several clauses may use one category.
- **Unknown labels** (absent from the base dictionary) cannot match any point and are dropped when
  a predicate is compiled; the number dropped is kept per query. A clause made only of unknown
  labels becomes empty, so the predicate matches nothing. A predicate without clauses (an empty
  text line) matches every point.
- A **graph** has `n_nodes ≥ n_data` nodes. Ids `0..n_data` are the data points (rows of the
  base vectors and of the label store); ids `n_data..n_nodes` are **auxiliary nodes** (frozen
  start points, centroids, routers) whose vectors are stored with the graph, in the element type
  of the data. Auxiliary nodes have no labels and are never answers: `Graph::labels(&store)`
  gives bounds-safe access that returns `Node::Auxiliary` for them instead of indexing a label
  row (and never matches a predicate), and `Graph::vectors(&base)` serves base rows and
  auxiliary vectors by node id. Adjacency rows have a fixed stride of `r_max` slots, so
  `Graph::neighbors(id)` is a slice of one 64-byte-aligned array.

## Command-line tool

```sh
# Labels and predicates
flab convert labels     --format text  -i base_labels.txt   -o labels.flab
flab convert predicates --format text  --labels labels.flab -i query_labels.txt -o queries.fpred
flab convert labels     --format jsonl -i base_labels.jsonl -o labels.flab

# Vectors: copy a .bin file, optionally scaling rows to unit L2 norm (f32 only)
flab convert vectors --dtype f32 --normalize -i query.bin -o queries.fbin

# Ground truth: DiskANN file and/or the extended file
flab gt --base base.u8bin --queries queries.u8bin --labels labels.flab \
        --predicates queries.fpred -k 100 -t 8 --out-bin gt100.bin --out-ext gt100.fgt

# Compare with another DiskANN ground truth file (with distances), up to tie order
flab gt-compare --ext gt100.fgt --reference other_gt.bin -k 100

# Graphs: import a saved DiskANN in-memory index (graph file <prefix>, vectors <prefix>.data)
flab import-graph --graph <prefix> --data <prefix>.data --base base.u8bin \
        -o graph.flab --report import.json

# Start-node tables: the graph's starts, C++ per-label start points (joined by label name with
# the label store of the graph's label view) and the C++ build-time sample
flab import-starts --graph graph.flab --labels labels.flab \
        --medoids <prefix>_labels_to_medoids.txt --label-map <prefix>_labels_map.txt \
        --sample <prefix>_sample_ids.bin --sample-map <prefix>_sample_labels.txt_map.bin \
        -o starts.flab --report starts.json

# Experiments with the lab search kernel (pinned to one CPU; see below)
taskset -c 7 flab run experiment.toml --commit <sha>

# Summaries
flab info labels.flab
flab info queries.fpred
flab info gt100.fgt
flab info graph.flab    # node space, degree distributions, starts, reachability from the starts
flab info starts.flab
```

The element type of vector files is inferred from the extension (`.u8bin` = `u8`, `.fbin` =
`f32`) or given with `--dtype`.

### Text formats

- Base labels: one line per point, labels separated by `,`
  (`color=red,size=s,size=m` or `12,58,301`). Only the part before the first tab is read;
  surrounding whitespace and `\r` are ignored; an empty line is a point without labels.
- Predicates: one line per query, clauses separated by `&`, alternatives by `|`
  (`color=red|color=blue&size=m`, `12&58`). Empty clauses or alternatives are errors.
- JSONL base labels: `{"doc_id": i, "<label>": true, ...}` with `doc_id` equal to the line index;
  `false` and `null` values are skipped.
- JSONL predicates: `{"query_id": i, "filter": F}` with `F` = `{"$and": [C, ...]}` or `C`, `C` =
  `{"$or": [A, ...]}` or `A`, `A` = `{"<label>": {"$eq": true}}`.

## Lab search kernel

`kernel::search::Searcher::search` answers one query (a vector and a predicate) on a graph
with a configuration (`kernel::config::Kernel`) that fixes one implementation per role. Per
query: the predicate is compiled (clause upper bounds `U_c` = sums of posting lengths, clause
order, a clause-bit table, weights); a query with an empty clause has no answers; optional
exact-path routing answers queries with `min_c U_c ≤ T` from the postings; then start
selection, a pilot (scorers with per-query scales), an optional materialization of the match
set, the loop (expand the `W` closest unexpanded beam nodes, *offer* every unvisited
neighbour: distance and predicate in the order of the laziness role, admission to the beam,
collection of matches), and collection of at most `K` matching data points sorted by (true
squared-L2 distance, id). The beam keeps upstream DiskANN's tie rule (a new entry goes before
existing entries with an equal key) and entries `(key, true distance, id)`. Auxiliary nodes
are navigated, never evaluated against the predicate and never returned.

| Role key | Values (default first) |
|---|---|
| `starts` | `+`-joined sources: `global`, `label(c*)` (the smallest clause's per-label starts), `label(all)` (alias `product`), `sample(k)`, `aux` |
| `start_admission` | `force`, `obey` (starts must pass admission) |
| `sample_start` | `intended` (sample points that match fully, via per-label sample postings), `recorded` (the sample ids stored in the start table) |
| `admission` | `all`, `strict`, `projection(<category>, ...)` (navigate on the clauses of these categories), `cutoff(τ)` (at most `τ` failed clauses) |
| `laziness` | `eager`, `deferred` (predicate only at collection), `predicate_first`, `distance_first` (skips nodes that can enter neither the beam nor the answers) |
| `scorer` | `distance`, `bonus(β)` (matches keyed `β·d`), `additive(λ)`, `normalized_additive(η)`, `multiplicative(η)`, `shifted(η)`, `shell_adaptive(κ)`, `lexicographic` |
| `collector` | `final_beam`, `heap` (best `K` matches seen), `all` (every match seen; the sorted list is truncated to `L`) |
| `expander` | `one_hop`, `two_hop` / `two_hop(cap)` (second hop through the closest rejected neighbours; default cap `r_max / 2`) |
| `terminator` | `exhaust`, `adaptive_l(sample, scale)` (default 1000, 16), `budget(W)` (distance computations), `match_aware` (heap collector) |
| `predicate` | `mode1` (row scan with clause bits), `mode2` (per-clause search in the row), `bitmap` (needs a materializer) |
| `materializer` | `none`, `dense` (match bitmap), `sparse` (sorted match list) |
| `exact` | `none`, `scan_smallest_clause(T)`, `intersect_then_scan(T)` |
| `weights` | `unit`, `surprisal` (clause weights of the graded scorers) |
| `beam_width` | `W` (1) |
| `padding` | `upstream` (beam capacity `L + #starts`, answers from the first `L` entries), `cxx` (capacity `max(L, #starts)`) |
| `aux_g` | penalty of auxiliary nodes in graded keys (0) |

Illegal combinations are rejected when the configuration is validated (for example `deferred`
needs `all` / `distance` / `final_beam`; `match_aware` needs `heap`; `bitmap` needs a
materializer and no projection).

Presets (`kernel::Preset`): `postfilter`, `upstream-inline`, `upstream-inline-adaptive` /
`upstream-inline-adaptive(sample, scale)`, `upstream-multihop`, `upstream-beta(β)` (alias
`inline-beta`), `bf(T)` / `inline-beta-bf(β, T)` (beta search with an exact posting scan),
`filt(λ, τ)` (C++ multi-filter search: sample and global starts, `cutoff(τ)`, additive
penalty), `sfmarket(<categories>)` (C++ OR-filtered search with navigation on a projection)
and `cp` (C++ OR-filtered search on a product-label graph). The `upstream-*` presets reproduce
upstream DiskANN's answers on unambiguous slots and its counters
(`tests/kernel_vs_upstream.rs`).

Compile-time parameters: the element type, the predicate mode and the scorer family (once per
plan), and the laziness, the expander and tracing (once per query); the other roles are enums
matched in the loop. The `postfilter` and `upstream-inline` presets also have
hand-specialized loops (`search::FastPath`, same answers and counters), used unless the plan
is made generic (`Plan::generic`). `examples/kernel_overhead.rs` measures the configurable
loop against them.

Counters per query: `n_dist`, `n_pred`, `n_expand`, `n_hops`, `n_match_seen`, `n_out`, the
route, `n_aux_dist`, `n_lazy`, `n_aux_expand`, `n_two_hop`, `n_evict`, `n_exact_cand`,
`n_start` and the final list size. A trace mode records per-node events (offers with keys,
predicate results and outcomes, expansions, answers), and a reference mode runs with a
resizable candidate list until a distance budget is spent.

## Experiments: `flab run`

`flab run experiment.toml` runs the configurations of an experiment over a query panel and an
effort grid (`L`), one run per `K`, and writes one run directory per (configuration, `K`).

```toml
[experiment]
name = "smoke"
dataset = "/path/to/dataset"     # base.<u8bin|fbin>, labels.flab, q<set>.<ext>, q<set>.fpred
graph = "/path/to/graph"         # default graph: a directory with graph.flab, a graph.flab, or a DiskANN index prefix
# starts_sidecar = "/path/to/starts.flab"  # default: starts.flab next to graph.flab
# view = "<view>"                # label view: labels.<view>.flab, q<set>.<view>.fpred
queries = "q10K"                 # a whole query set, or:
# panel = "panels/screen.tsv"    # TSV with columns qset, qidx (optional split); relative to the dataset
# split = "discovery"            # panel filters
# qsets = ["10K"]
# limit = 500
k = [10, 100]                    # or a single integer
efforts = [10, 50, 200, 1000]    # L
reps = 3
mode = "timed"                   # timed | trace | reference
seed = 0                         # configuration order per (repetition, effort)
runs_dir = "/path/to/runs"
ledger = "/path/to/ledger.jsonl"
# warmup = true
# fast_paths = true              # false: no hand-specialized preset loops
# trace_rows = [0, 5]            # trace mode (or every trace_sample-th row)
# reference_factor = 20.0        # reference mode: budget = max(factor × n_dist, reference_min)
# reference_min = 1000

[[config]]
name = "inline"
preset = "upstream-inline"

[[config]]
name = "inline-w4"
preset = "upstream-inline"
beam_width = 4                   # any role key of the table above
# graph / starts_sidecar / view override the experiment's
```

Timed mode refuses to run unless the process is pinned to one CPU. It pre-faults the inputs,
records the CPU, NUMA node, memory policy, niceness, load and a calibration at the start and
the end, warms every configuration up at the median effort, then for each repetition and effort
runs the configurations in a seeded shuffled order; each query is timed with the wall clock
nested inside `CLOCK_THREAD_CPUTIME_ID` reads, including every per-query cost (predicate
compilation, routing, starts, pilot, materialization, search, collection, resets).
Repetitions must return identical results. Outputs per run directory: `run.json`,
`res_<L>.bin` (DiskANN truthset layout: `u32 nq`, `u32 K`, ids padded with `u32::MAX`, true
squared-L2 distances; rows in panel order), `perquery.tsv` (`effort rep row wall_ns cpu_ns`,
then the counters), `panel.tsv` (the rows used), `trace/` (trace mode); one ledger line per
run, failures included.

## Importing DiskANN indices

The C++ library (`Index::save`) and the Rust in-memory index of `diskann-providers`
(`SaveWith<AsyncIndexMetadata>`) save the same layout:

- the graph file `<prefix>`: a 24-byte header (`u64` file size, `u32` maximum degree, `u32`
  start id, `u64` number of frozen points `F`), then for every node in id order a `u32` degree
  `k` and `k` `u32` neighbor ids. The C++ library records the largest observed degree, Rust its
  configured maximum degree;
- the vectors `<prefix>.data` in the `.bin` layout, one row per node; the `F` frozen points are
  the last nodes and rows.

The Rust provider allocates one adjacency list more than it has nodes, so its graph files end
with an empty list beyond the `N + F` rows of the vector file; it belongs to no node.

`import-graph` (and `diskann_io::read_index`) takes the node count from the vector file
(without it, from the number of lists; then `F` must be 0). The first `n_nodes − F` nodes are the
data points, the frozen points become auxiliary nodes of kind *frozen* with their vectors,
`r_max` is the header's maximum degree, and the starts are the header's start id followed by
the frozen points (the start set of both libraries). It checks the file size, every degree
against the maximum degree, every id against the node count, and that the extra list is empty.
With `--base` it also checks that the first `n_data` rows of the vector file equal the base
vectors, so that point ids agree, and fails otherwise (unless `--allow-base-mismatch`). The
written file is read back and compared. `diskann_io::open` loads a graph either from
`graph.flab` or directly from a DiskANN graph file, without conversion.

`import-starts` builds a start-node table: the global starts are the graph's starts; C++
per-label start points (`label, point` lines with integer labels) are translated to label names
with the label map (`name<TAB>label` lines) and to the label ids of `--labels` by name (names
absent from the store are dropped and reported, and so are starts that lack their label); the
sample keeps the order of the id file, and `--sample-map` gives the node that the C++ search
starts from for each sample entry (entries beyond the sample are ignored).

The example `upstream_check` compares an import with the adjacency lists that the upstream
loader reads from the same files:
`cargo run --release -p diskann-filter-lab --example upstream_check -- <prefix> [--flab graph.flab]`.
The test `tests/upstream_roundtrip.rs` builds and saves an index with `diskann-providers` and
checks that the import holds the same lists, starts and frozen vector.

## Ground truth semantics

Distances are squared L2, computed with `diskann-vector`'s SIMD kernels. For `u8` vectors they are
accumulated in integers and are exact (as `f32`) up to 2^24, i.e. for any dimension up to 258, so
distance ties are exact.

For query `q` with matching set `M` and `k`-th smallest distance `d_k`, the **band** is every point
of `M` within `d_K` (all of `M` when `|M| < K`), sorted by (distance, id). Its first
`min(K, |M|)` entries are the exact top-K with ties broken by id, and for every `k ≤ K` the band
contains all points tied at `d_k`, so tie-aware recall can be computed at any `k ≤ K`.

Candidates are the points of the **smallest clause** (least total posting length, first on ties).
Each candidate is checked against the other clauses either in the forward store or with
per-query bitsets built from postings (`--eval auto|forward|postings`; `auto` picks whichever
touches fewer label entries). Both are exact and produce identical files.

The DiskANN output is the usual truthset: `u32` query count, `u32` K, `nq × K` `u32` ids, then
`nq × K` `f32` distances; queries with fewer than K matches are padded with id `u32::MAX` and
distance `f32::MAX`.

`gt-compare` accepts a reference row when it has `min(k, |M|)` distinct valid ids, each of them is
in our band with exactly the reference distance, and its sorted distances equal ours.

## Binary formats

All lab formats share one container. Integers are little-endian.

| Offset | Type | Field |
|---|---|---|
| 0 | `[u8; 8]` | magic |
| 8 | `u32` | format version (currently 1 for all formats) |
| 12 | `u32` | number of sections `S` |
| 16 | `[u64; 8]` | format-specific parameters |
| 80 | `[(u64, u64); S]` | per section: byte offset from the file start, byte length |

Each section starts at a 64-byte-aligned offset, so a memory-mapped file can be viewed as typed
arrays. Offset arrays are `u64`; ids are `u32`; distances are `f32`. A CSR `(offsets, values)`
stores row `i` as `values[offsets[i]..offsets[i + 1]]`.

### Labels: `labels.flab` (magic `FLABLBL\0`)

Parameters: `n_points`, `n_labels`, `n_categories`, `n_entries` (Σ labels per point),
`dict_hash` (FNV-1a fingerprint of the dictionary).

| Section | Type | Length | Content |
|---|---|---|---|
| 0 | `u32` | `n_labels` | category of each label |
| 1 | `u64` | `n_labels + 1` | label name offsets into section 2 |
| 2 | `u8` | | label names, UTF-8, concatenated |
| 3 | `u64` | `n_categories + 1` | category name offsets into section 4 |
| 4 | `u8` | | category names, UTF-8, concatenated |
| 5 | `u32` | `n_categories + 1` | first label id of each category (labels of category `c` are `[start[c], start[c+1])`) |
| 6 | `u64` | `n_points + 1` | forward offsets |
| 7 | `u32` | `n_entries` | forward rows: sorted label ids per point |
| 8 | `u64` | `n_labels + 1` | posting offsets |
| 9 | `u32` | `n_entries` | postings: sorted point ids per label |

### Predicates: `*.fpred` (magic `FLABPRD\0`)

Parameters: `n_queries`, `n_clauses`, `n_refs` (Σ clause lengths), `n_labels` and `dict_hash` of
the dictionary the predicates were compiled against (checked when used with a label store).

| Section | Type | Length | Content |
|---|---|---|---|
| 0 | `u64` | `n_queries + 1` | clause offsets of each query into the clause list |
| 1 | `u64` | `n_clauses + 1` | label offsets of each clause into section 2 |
| 2 | `u32` | `n_refs` | sorted, distinct label ids of each clause |
| 3 | `u32` | `n_queries` | unknown label occurrences dropped per query |

### Extended ground truth: `*.fgt` (magic `FLABGT\0\0`)

Parameters: `n_queries`, `K`, `n_points`, `n_band` (Σ band lengths), `dict_hash`, metric
(0 = squared L2).

| Section | Type | Length | Content |
|---|---|---|---|
| 0 | `u32` | `n_queries` | number of matching points |
| 1 | `u32` | `n_queries` | index of the smallest clause (`u32::MAX` without clauses) |
| 2 | `u64` | `n_queries` | total posting length of the smallest clause (`n_points` without clauses) |
| 3 | `u32` | `n_queries` | points satisfying the smallest clause (size of its postings union) |
| 4 | `u64` | `n_queries + 1` | band offsets |
| 5 | `u32` | `n_band` | band ids, per query sorted by (distance, id) |
| 6 | `f32` | `n_band` | band distances |

### Graphs: `graph.flab` (magic `FLABGRF\0`)

Parameters: `n_data`, `n_nodes`, `r_max`, `n_starts`, element type of the vectors (0 = not
recorded, 1 = `u8`, 2 = `f32`), dimension (0 if not recorded), flags (0; none are defined in
version 1), `n_edges` (sum of the degrees). Auxiliary nodes require a recorded element type.

| Section | Type | Length | Content |
|---|---|---|---|
| 0 | `u32` | `n_nodes` | out-degree of each node |
| 1 | `u32` | `n_nodes × r_max` | adjacency rows; the slots after the degree hold `u32::MAX` |
| 2 | `u32` | `n_starts` | start nodes in priority order |
| 3 | `u8` | `n_nodes − n_data` | kind of each auxiliary node (0 = frozen, 1 = centroid, 2 = router) |
| 4 | element type | `(n_nodes − n_data) × dim` | vectors of the auxiliary nodes, row-major |

### Start-node tables: `starts.flab` (magic `FLABSTR\0`)

Parameters: `n_nodes` and `n_data` of the graph; `n_labels` and `dict_hash` of the label store
that the per-label table refers to (0 without a table); `n_global`; `n_label_entries`;
`n_sample`.

| Section | Type | Length | Content |
|---|---|---|---|
| 0 | `u32` | `n_global` | global starts in priority order |
| 1 | `u64` | `n_labels + 1` | per-label offsets into section 2 |
| 2 | `u32` | `n_label_entries` | start nodes of each label |
| 3 | `u32` | `n_sample` | sample entries (data points) in order |
| 4 | `u32` | `n_sample` | start node of each sample entry |

Reading a section with NumPy:

```python
import numpy as np
hdr = np.fromfile(path, dtype="<u8", count=10 + 2 * n_sections)  # magic+version+S, params, table
def section(i, dtype):
    offset, nbytes = hdr[10 + 2 * i], hdr[11 + 2 * i]
    return np.fromfile(path, dtype=dtype, count=nbytes // np.dtype(dtype).itemsize, offset=offset)
```

(`hdr[0]` is the magic, `hdr[1]` packs the version and `S`, `hdr[2:10]` are the parameters.)
