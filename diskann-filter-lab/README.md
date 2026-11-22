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

# Ground truth: DiskANN file and/or the extended file (queries without clauses — the filter-less
# problem — are computed in blocks: each base chunk read once per block of queries)
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

# Per-label start tables computed from the data, for any label store and any graph over the
# same points (one starts-<tag>.flab + starts-<tag>.json report per strategy)
flab starts build --dataset <dir> --graph <graph> --strategy medoid,random,kmeans(4) \
        -t 4 --out-dir <dir>

# Static per-label bitsets (labels with at least θ points) and their memory per θ
flab bitsets build --labels labels.flab --theta 31250 -o bitsets.flab
flab bitsets stats --labels labels.flab --theta 1000,10000,31250

# Unit costs of the exact path's cost model on this machine and dataset (pinned like the runs)
taskset -c 7 flab calibrate --dataset <dir> --graph <graph> -o costs.json

# Bottom-b priority sketches of the label postings (one sketches-b<b>.flab per b) and their memory
flab sketches build --labels labels.flab -b 64,128,256 -t 4 --out-dir <dir>
flab sketches stats --labels labels.flab -b 64,128,256

# Combination postings (joint postings of label pairs; budgets as fractions of the label postings'
# bytes; one pairs-nested-b<per-mille>.flab per budget, with the coverage of the --report predicate
# sets; nested selections by default, --plain for independent ones)
flab pairs build --labels labels.flab --budget 0.2,0.5 -t 2 --out-dir <dir> --report q10K.fpred

# One selector for the auxiliary structures (see "Selected auxiliary structures" below): combination
# postings and per-label sub-graphs (islands) for the hybrid, or pair trees (and router trees) for
# graph-only search (--track graph); --budget is a memory ceiling in bytes per data point
flab select --dataset <dir> --calibration costs.json --graph <graph file> --budget 40 -t 4 \
        --out-dir <dir>
flab select --track graph --dataset <dir> --calibration costs.json --graph <graph file> \
        --budget 40 -t 2 --out-dir <dir>

# Router trees per label (auxiliary router nodes reached only from the start table) and pair trees
# over selected label pairs, on any graph over the same points
flab steiner --dataset <dir> --graph <graph> --starts starts.flab --construction tree --block 64 \
        --min-size 0 --max-size n/10 --router centroid --partition kmeans -t 2 --out-dir <trees>
flab steiner --dataset <dir> --graph <trees> --starts <trees>/starts.flab --construction pairs \
        --pairs pairs.flab --block 64 --min-size 0 --max-size n/1 --router centroid \
        --partition kmeans -t 2 --out-dir <pair trees>

# Experiments with the lab search kernel (pinned to one CPU; see below)
taskset -c 7 flab run experiment.toml --commit <sha>

# An isomorphic id layout: permuted copies of a dataset (base, labels, ground truth, sketches;
# links to the query files) and of its graph and start table (see below)
flab permute --dataset <dir> --graph <graph> --order kmeans -b 128 -t 4 \
        --out-dataset <dir2> --out-graph <graph2>

# Graph construction: the lab's parallel Vamana builder, recording the candidate pools, and
# prune replay of the pools with another pruning rule (see below)
flab build --dataset <dir> --prune 'alpha(64, 1.2)' --l-build 100 -t 3 \
        --pools pools/pools.flab -o graphs/vamana
flab build --dataset <dir> --replay pools/pools.flab --prune 'label_containment(64, 1.2)' \
        -t 3 --compare graphs/vamana/graph.flab -o graphs/lc

# Provenance fingerprints (JSON) of a dataset, a graph, and what derived files record (see
# "Provenance" below)
flab provenance --dataset <dir> --graph <graph> --file pairs.flab --file costs.json

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
| `admission` | `all`, `strict`, `projection(<category>, ...)` (navigate on the clauses of these categories), `projection(c*)` (navigate on the query's clause with the smallest summed posting size; answers are full matches), `cutoff(τ)` (at most `τ` failed clauses) |
| `laziness` | `eager`, `deferred` (predicate only at collection), `predicate_first`, `distance_first` (skips nodes that can enter neither the beam nor the answers), `heap_bound` (skips the predicate of every node that cannot be an answer: `heap` collector full and distance beyond its K-th; the node still navigates; needs admission `all` and the `distance` scorer) |
| `scorer` | `distance`, `bonus(β)` (matches keyed `β·d`), `additive(λ)`, `normalized_additive(η)`, `multiplicative(η)`, `shifted(η)`, `shell_adaptive(κ)`, `lexicographic` |
| `collector` | `final_beam`, `heap` (best `K` matches seen), `all` (every match seen; the sorted list is truncated to `L`) |
| `expander` | `one_hop`, `two_hop` / `two_hop(cap)` (second hop through the closest rejected neighbours; default cap `r_max / 2`) |
| `terminator` | `exhaust`, `adaptive_l(sample, scale)` (default 1000, 16), `budget(W)` (distance computations), `match_aware` (heap collector), `posting_l(scale)` (posting-based adaptive L: `L₀ ×` upstream's multiplier of the estimated match rate `M̂/n`, fixed before navigation; default scale 16), `effort_curve` (the effort is a target recall `R` in per mille; the list size is the graph's calibrated unfiltered effort for recall `R` at `K/r̂` neighbours, `r̂ = M̂/n`, fixed before navigation; needs a cost crossover and the unit costs' effort curve; no planned list exceeds the curve's largest measured list size `effort_l_top`, after the proportional growth beyond the largest calibrated `K'` (a target recall beyond the measured curve plans that maximum: `flab run` warns, and `run.json` lists those efforts under `effort_curve_capped`; every query whose list was cut has `list_capped = 1`, counted per effort in `run.json` under `list_limit_hits`); a calibration that records neither `effort_l_top` nor `details.effort.l` is refused; `effort_curve(cap)`: at most `cap` × the list size of an unselective query) |
| `predicate` | `mode1` (row scan with clause bits), `mode2` (per-clause search in the row), `bitmap` (needs a materializer), `static_bits(θ)` (static bitsets for labels with at least `θ` points, the row for the others) |
| `materializer` | `none`, `dense` (match bitmap), `sparse` (sorted match list), `dense_static` (match bitmap built with the static bitsets), `dense_nav` (bitmap of the navigation clauses only — all clauses, or the projected ones such as `c*` — built with the static bitsets when the index has them; the other clauses of a projected query are checked on the row of the bitmap's members) |
| `exact` | `none`, `scan_smallest_clause(T)`, `intersect_then_scan(T)`, `planned(T)` (the exact path v2 below) |
| `exact_plans` | plans `planned` may choose from: `scan+df+gallop+bitmap+rowand+pair` (default) or any subset (`rowand` needs a calibration with the A6 enumeration costs; `pair` needs a `pairs_sidecar`) |
| `crossover` | `threshold` (the exact path when `min_c U_c ≤ T` of `exact`), `cost` (the planned exact path when its estimated cost is at most the nominal graph cost), `guarded` (`cost`, then a graph trial with online fallback; heap collector), `graph` (graph only, track G: the estimates set the list size and the post-filter route, the exact path is never taken; the hopeless gate does not apply) |
| `estimates` | `bounds` (posting bounds), `sketch` (a probe of the bottom-b sketches) |
| `fallback` | signals of `guarded`: `forecast+budget+witness` (default), any subset, or `none` |
| `postfilter` | `off`, `cost` (a graph-routed query of the cost or guarded crossover takes the post-filter route when its nominal cost is at most the configured graph search's; under `guarded` the route itself is not guarded), `always` (every graph-routed query; a control; under the `threshold` crossover: the router's graph side is the post-filter route at the fixed list `L`, no list planning, from the `global` or `configured` starts, or the rate-dependent ones with `threshold_on = "estimate"` — a threshold router without a cost model): see below |
| `postfilter_starts` | starts of the post-filter route: `global` (the graph's global starts), `configured` (the `starts` role, e.g. per-label starts), `by_rate` (the configured starts for a narrow predicate, estimated match rate below 1/2; the global starts for a broad one), `by_rate_late` (`by_rate`, and a broad predicate's search, once it ends, offers the configured starts to its list and continues from those it admits, i.e. those closer to the query than its last entry; counter `n_late`), `by_rate_both` (`by_rate`, and a broad predicate's search is followed by a second search from the configured starts with the same list size, sharing the visited nodes; the answers are the `K` nearest matches of both; `n_late` counts its starts), `by_rate_both_k` (`by_rate_both` with the second search's list size `K`, at most the first one's) |
| `price_second_search` | `off` (default: the post-filter route is priced by its first search), `on`: wherever `by_rate_both*` runs a second search (a broad predicate), its nominal cost — a post-filter search at its list size (`K` for `by_rate_both_k`) from the configured starts, an upper bound since it shares the first search's visited nodes — is added to the route's price (the crossover's comparison with the inline search and the exact path); needs the post-filter route and `postfilter_starts = by_rate_both` or `by_rate_both_k` |
| `exact_checks` | what the planned exact path and the probe check on their points, all members of `union(c*)`: `residual` (default: the clauses after `c*`, with prefetches for them) or `full` (every clause with the graph search's prefetch policy, as in batch B6) |
| `weights` | `unit`, `surprisal` (clause weights of the graded scorers) |
| `beam_width` | `W` (1) |
| `padding` | `upstream` (beam capacity `L + #starts`, answers from the first `L` entries), `cxx` (capacity `max(L, #starts)`, C++ OR-filtered search), `exact` (capacity `L`, C++ multi-filter search) |
| `aux_g` | penalty of auxiliary nodes in graded keys (0) |
| `row_prefetch` | `off` (default), `on`: prefetch the adjacency row of the closest unexpanded node (the next expansion, unless a closer node arrives first) while the current node is expanded; changes no answer or counter |
| `packed_beam` | `off` (default), `on`: the post-filter loop (preset and route) keeps its beam in one array of `u64` entries (key, expanded flag, state, 28-bit node id) instead of a key array and a slot array: the same beam semantics, one array to search and move per insertion; changes no answer or counter; graphs with more than 2^28 nodes keep the ordinary beam |
| `df_evals` | `estimate` (default), `lower`: the exact plan `df` counts its predicate evaluations at the lower confidence bound of the match count instead of the estimate (its answer bound saves checks only once `K` matches are found, so an over-estimated match count must not make it look cheap); exact estimates are unaffected |
| `inline_rank` | `off` (default), `on`: under a cost or guarded crossover, an inline graph search with `posting_l` is planned (and priced) at a list size whose evaluated nodes reach the rank of the `K`-th match at the estimated match rate, as the post-filter route's list (`effort_curve` plans at `K / r` neighbours already) |
| `static_bitsets` | `on` (default: the planned exact path, the probe and the post-filter route check clauses with the static bitsets when the index has them), `off`: on the label rows (the plans keep their forms without bitsets: `bitmap` scatters postings, `rowand` gallops; the bitsets are not built for such configurations alone): an ablation of the stored dense rows; excludes `predicate = "static_bits(θ)"` and `materializer = "dense_static"` |
| `threshold_on` | `bound` (default: the `threshold` crossover takes the exact path when `min_c U_c ≤ T` of `exact`, `U_c` = the sum of clause `c`'s posting sizes), `estimate`: the graph route when the sketch estimate `M̂` of the match count is `≥ T`, else the exact path (needs `crossover = "threshold"`, `estimates = "sketch"` and an exact path with a threshold; the estimates are made once, before routing, and `m_hat` is recorded); an ablation that separates better match-count estimates from pricing by the cost model |
| `df_pipeline` | `off` (default), `on`: the exact plan `df` checks a candidate inside the answer bound one block of 32 candidates after its distance, with its row offsets and row prefetched meanwhile (as `scan` prefetches its checks; the cost model prices `df`'s checks with the prefetched check curves); the same checks, answers and counters as the direct loop, only memory timing |
| `tree_route` | `off` (default), `row_scan`, `bitmap`, `static_bits`, `cost`: a query whose binding clause c\* has router-tree roots (`flab steiner`) as starts searches the trees (`aux_only`, strict admission, a heap of the `K` best matches) with that member check; `cost` chooses per query the check of the lowest model cost (calibrated unit costs) |
| `tree_effort` | `list` (default: the configuration's list), `curve` (the calibrated tree effort curve, `flab tree-calibrate`), `capped`, `fixed(F)` (one router frontier `F` for every query and effort) |
| `tree_pairs` | `off` (default), `on`: a query whose clause pair is exactly covered by pair trees (`flab steiner --construction pairs`; the run's `pairs_sidecar` and `pair_roots`) with fewer members than c\* searches those trees |

Illegal combinations are rejected when the configuration is validated (for example `deferred`
needs `all` / `distance` / `final_beam`; `match_aware` needs `heap`; `bitmap` needs a
materializer, and no projection unless the materializer is `dense_nav`).

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

**Static per-label bitsets** (`predicate = "static_bits(θ)"`, module `bitsets`): a dense
bitset over the data points for every label with at least `θ` points (a sidecar built once:
`⌈n/64⌉ × 8` bytes per label, smaller than the label's postings when `θ > n/32`). Per query the
clauses are split into bitset labels (tested in O(1), most frequent first) and row labels. A
node is evaluated clause by clause in Boolean order with the bitsets; the clauses that remain
unsatisfied and have row labels are then checked in one pass over the node's row (a mode-1
scan, or galloping searches for long rows); a clause made only of bitset labels never reads the
row, and Boolean evaluation stops at the first such clause that fails. It returns the
failed-clause count like modes 1 and 2 (so it works with every scorer and admission rule), and
the exact path uses it too. The materializer `dense_static` builds the dense match bitmap
with the same bitsets (a label's bitset is ORed in word by word instead of scattering its
postings). The run driver loads `bitsets_sidecar` or builds the bitsets in memory at the
smallest `θ` of the experiment (untimed; recorded in `run.json`); a configuration's `θ` may
exceed the sidecar's.

`examples/kernel_microbench.rs` measures the kernel's components in isolation (ns per
predicate check per evaluator, row length and access pattern; per-query materializations;
exact paths per candidate; distances).

**Exact path v2** (`exact = "planned(T)"`, module `kernel::exact`): for a query routed to the
exact path, a cost model (`kernel::cost`) chooses the cheapest of five plans from the posting
sizes, the estimated union of the smallest clause `A` and match count `M`, and calibrated unit
costs: `scan` (candidates of the smallest clause, predicate first), `df` (distances first; the
predicate only within the answer heap's K-th distance), `gallop` (galloping intersection of
the clauses' postings), `bitmap` (a dense match bitmap built with the static bitsets, one
L1-sized block of 1,024 words at a time, enumerated per block; a block skips its remaining
clauses once it is empty; batch A6b: full-width scratch bitmaps, 10 MB of written lines per
query at 40M, slowed the next queries) and
`rowand` (batch A6: only the bitmap words that hold members of the smallest clause, ANDed
clause by clause with the other clauses' bitset words, prefetched, and with the postings of
their labels without a bitset through galloping cursors; no memory beyond one bitmap of
scratch). The bitmap's enumeration is priced per empty word, non-empty word and set bit (A6; older
calibrations price every word as if 10 % of its bits were set, which made `bitmap` look 5 to
20 times dearer than it is on selective queries), and `rowand` is a candidate only with those
costs, so older calibrations reproduce the earlier choices. The planner (`PlanInputs::best`)
computes the per-query terms once for all plans, interpolates with cached grid logarithms
(`UnitCosts::prepare`) and skips `df` / `gallop` when a cheap lower bound already loses: the
same costs and choices, ≈ 0.39 µs per call for a 4-clause query with five plans (0.87 µs with
four plans before A6b; ignored test `planning_time`).
A sixth plan, `pair` (batch B3a), needs combination postings (`pairs.flab`, module `pairs`):
the joint postings `P_a ∩ P_b` of label pairs selected from the label store alone under a
budget (big labels of at least `T` points, `T` from the budget; pairs by decreasing saved
candidates per stored entry, `(min(|P_a|, |P_b|) − |P_ab|) / (|P_ab| + 3)`; `--prior rare`
weights that by `min(|P_a|, |P_b|) / n`). Selections are nested by default: a budget's
selection keeps every pair of the smaller budgets of its chain (the 1-2-5 ladder from 5 % of the
postings below it: 5 %, 10 %, 20 %, …), so that a larger budget stores a superset although its
`T` is lower; a chain is built in the passes of a single build at its last `T`
(`--plain`: one independent selection per budget). Per query the cover takes the clause pair whose
lists (stored joint postings, nothing for empty pairs, the smaller posting otherwise) hold the
fewest entries; the plan merges them, checks the other clauses (all of them when a label
posting is among the lists; priced at that mask's worst-case work) and computes distances.
Without the sidecar its cost is infinite and nothing changes.
Candidate rows and vectors are prefetched in a software pipeline, and checks use the static
bitsets when the run has them. Every candidate of `scan` and `df` (and every point of the probe
below) is a member of `union(c*)`, so with `exact_checks = "residual"` only the residual
clauses are checked, and the pipeline prefetches what they read: the row when a residual
clause has labels without a bitset, and the first residual clause's bitset word (the graph
search's policy, which looks at the first clause, would prefetch nothing when `c*` is made of
bitset labels). A *seeded* exact path keeps the answers of another search and still processes
every candidate. The unit costs come from `flab calibrate` (module `calibrate`; base data
only): memory-access costs as functions of the candidate density on random points (`pred_ns`,
mostly misses of the first clause) and on members of the calibration predicates' `union(c0)`
checked like the exact path's candidates, separately for predicates whose residual clauses
read rows and bitset-only ones (`pred_row_ns`, `pred_bits_ns`; used for `scan` and `df`);
merge and bitmap costs (enumeration per empty word, non-empty word and set bit: bitmaps with 0 to 1/16 of
their bits set); galloping steps by gap, measured with the gallop plan on label pairs
whose posting lengths have that ratio (`gallop_gap_ns`: cache misses grow with the gap); graph-search costs per distance and per predicate (and per distance by list size up to
L = 40,960: long beams cost more per distance), and the predicate evaluations of `heap_bound`
and `distance_first` searches (below). With a graph it also measures the effort curve of
`effort_curve` (`--effort-queries`, default 200; 0 skips it): midpoints of seeded random base
pairs, their exact unfiltered ground truth (`K' ≤ 3,000`) from one scan of the base, one
unfiltered search per query and list size `10·2^i ≤ --effort-lmax` (default 40,960; it stops at
the first size where every `K'` reaches a mean recall of 0.995) from the global start (the `K'`
nearest evaluated nodes answer every `K'` at once), and per target recall
the smallest list size reaching it (log-linear between sizes; above the largest, the power law
of the miss rate through the last two sizes, at most 8× it; the largest measured size is kept as
`effort_l_top`, and `effort_curve` plans at most that list — calibrations written without it take
the largest of `details.effort.l`). Recall@`K'` is counted with tie credit capped as a
tie-aware evaluator caps it: the answers strictly closer than the `K'`-th true distance plus
those at that distance up to the number of true points tied there (distinct answers tied beyond those slots, e.g. exact
duplicates, earn nothing; earlier calibrations counted every answer within the distance);
`flab tree-calibrate`'s tree curve counts recall the same way. The report records its provenance (the label
store, the graph's data points, the base and the CPU model; see "Provenance"). It takes a few
seconds (1M points) to a few minutes (40M) per dataset; the costs are loaded with the experiment
or configuration key `calibration`. Calibrations without the newer terms reproduce the batch-B6
cost model.

**Estimates and the cost crossover** (`kernel::planner`, module `sketches`): every data point
has a fixed hash priority and every label keeps its `b` lowest-priority members (sidecar
`sketches.flab`, `flab sketches build`); merging the lists of the smallest clause's labels
gives a uniform sample of its union (the whole union when every label has at most `b`
members). The probe checks the sampled points against every clause (the first 32, then the
rest only if the estimated match rate is below 1/2) and estimates `A` (bottom-k estimator),
`M`, the residual clauses' survival and confidence bounds. `crossover = "cost"` takes the
exact path when the cheapest plan's estimated cost is at most the nominal graph cost of the
planned list size: the calibrated distances per `L`, the predicate evaluations they lead to
under the configuration's laziness (every distance when eager; under `heap_bound` about `K/r`
until the heap is full plus a calibrated number `extra(K)` inside the K-th distance, at the
estimated match rate `r`; a calibrated share under `distance_first`), plus the starts; queries
whose exact path is cheaper than the cheapest graph search even with posting-bound counts skip
the probe.
`crossover = "guarded"` runs the graph search as a bounded trial: at checkpoints (64 distance
computations after the starts, then at doubling intervals) it falls back when, with the heap
incomplete, the remaining graph cost forecast from the arrival rate of answers exceeds the
exact cost (`forecast`), or when the graph's model cost reaches the estimated exact cost
(`budget`); at nominal termination it falls back on an incomplete heap or when a held-out
sampled match is closer than the K-th answer (`witness`). The fallback exhausts the exact path
seeded with the graph's answers (route `fallback`). Costs are model costs of the counters,
so trials are deterministic.

**Post-filter route** (`postfilter = "cost"`, route `postfilter`): an unfiltered search from
the graph's global starts (or the configured starts, `postfilter_starts = "configured"`; or
those for a narrow predicate only, `"by_rate"`: a broad predicate's per-label starts say
nothing about where its matches are, and best-first search from them can settle in a local
minimum that the graph's own entry point avoids) with no predicate checks during the traversal; at the end the
evaluated nodes are checked in distance order until `K` matches are found. Its answers are
those of a `heap(K)` collector over the same traversal (the `K` nearest evaluated matches by
(distance, id)): the final beam holds the nearest evaluated nodes, so it is checked first and
the checks stop after the `K`-th match (and its distance ties); only when the whole beam
holds fewer than `K` matches are the other evaluated nodes checked, from a log of every
evaluated node (counter `fallback = 5`). The route's list size is the planned list size, raised
until the search evaluates as many nodes as the rank of the `K`-th match at the estimated
match rate `r` (negative binomial mean `K/r` plus `z = 2.576` standard deviations), and its
nominal cost is the distances of that search plus about `K/r` checks. The cost crossover takes
it when that is at most the configured graph search's cost (label starts, inline checks), and
the exact path when that is cheaper still. `UnitCosts::effort_l` holds the optional effort
curve of `effort_curve` (list sizes by target recall `effort_r` and neighbours `effort_k`,
from unfiltered searches of base-derived queries; measured by `flab calibrate` with a graph, see
below).

Counters per query: `n_dist`, `n_pred`, `n_expand`, `n_hops`, `n_match_seen`, `n_out`, the
route, `n_aux_dist`, `n_lazy`, `n_aux_expand`, `n_two_hop`, `n_evict`, `n_exact_cand`,
`n_start`, `n_late` (starts admitted after a post-filter search ended, `by_rate_late`), the final list size, and for the planned exact path and the crossover the plan, the
probe's predicate evaluations, the estimated exact and nominal graph costs, the estimates of
`A` and `M`, a guarded trial's fallback reason and its graph work (model ns), `list_capped`
(`effort_curve`: the planned list was cut at the effort curve's largest measured list size). A trace mode records per-node events (offers with keys,
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
# bitsets_sidecar = "/path/to/bitsets.flab" # static_bits(θ) configs; default: built in memory
#                                           # (untimed) at the smallest θ of the experiment
#                                           # (θ = ⌈n/64⌉ for the planned exact path alone)
# calibration = "/path/to/costs.json"       # unit costs (flab calibrate) of the planned exact path
# sketches_sidecar = "/path/to/sketches.flab" # bottom-b sketches (estimates = "sketch")
# pairs_sidecar = "/path/to/pairs.flab"     # combination postings (the exact path's `pair` plan)
# islands_sidecar = "/path/to/islands.flab" # per-label sub-graphs (the cost crossover's `island` route)
# pair_roots = "/path/to/pair_roots.flab"   # roots of the pair trees over pairs_sidecar (tree_pairs)
# view = "<view>"                # label view: labels.<view>.flab, q<set>.<view>.fpred (its
#                                # predicates feed the presets that use a view: sfmarket, cp)
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
# hot = false                    # profiling: each timed search right after an untimed search of
#                                # the same row and effort (its working set cached)
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
# graph / starts_sidecar / view / calibration override the experiment's (configurations on one
#                                # graph with different start tables share one copy of the graph)
# predicates = "view"            # the view's predicates replace the queries' own (default "own"),
#                                # e.g. an all-match view: the filter-less problem, same invocation
# efforts = [100]                # a subset of the experiment's efforts (e.g. an L-independent
#                                # exact path)

[[config]]                       # timed mode: a per-query mixture (time sharing) of two earlier
name = "mix"                     # configurations on one graph: row r runs the first at its
mix = "inline@200 inline-w4@50 0.25 7" # effort when `mix_draw(seed, r)` (splitmix64 of seed ^ r·φ,
efforts = [200]                  # top 53 bits / 2^53) < p, else the second; `efforts`: one label
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

Large arrays (base vectors, label store, graph rows, the visited bitset) are allocated
2 MiB-aligned and advised for transparent huge pages (`madvise(MADV_HUGEPAGE)`; with the
system's THP mode `madvise` or `always`), which removes most TLB misses of a search over a
large index; `FLAB_HUGEPAGES=off` keeps base pages (`run.json` records whether the advice was
given and the process's huge-page memory).

**Id layouts** (`flab permute`): relabels the data points by an order, `bfs` (breadth-first
from the graph's first start) or `kmeans` (a k-means tree: `--fanout` clusters per split,
centers trained on a sample, recursively down to leaves of at most `--leaf` points; sibling
clusters in nearest-neighbour chain order), so that points close in space get close ids. It
writes the permuted base, `labels.flab` (rows reordered, postings mapped), every `*.fgt` (and
the DiskANN ground truth next to it), sketches rebuilt for `-b` (their priorities are functions
of the new ids), links to the query files, `new_of_old.u32` and `permutation.json` (with the
share of graph edges whose ends lie within 2^6 … 2^15 ids before and after), and the permuted
`graph.flab` and `starts.flab`. Rows keep their neighbour order, so searches expand the same
nodes in the same order; answers tied at the K-th distance may differ (ties are broken by id).

## Selected auxiliary structures (`flab select`)

One selector (module `select`; track G's kinds in `select_g`) chooses the index's auxiliary
structures by **expected saving per query per byte** on one fixed workload, priced with the
calibrated unit costs (`flab calibrate` of the graph the index searches); no query log is read.

- **Workload** (module `workload`): a ½ / ½ mixture of *data-like* queries (a data point drawn
  uniformly, each of its categories kept with probability ½, a kept category's clause one of the
  point's labels or a random subset of them; the point is the query vector) and *independent*
  queries (conjunctions of 1 to `min(3, C)` single-label clauses of distinct categories, drawn by
  a uniform walk on the label co-occurrence structure; the query vector an independent data
  point). `--uniform` sets the independent share (0: data-like only).
- **Stop** (`--stop elasticity`, default): an item is taken only if it lowers the workload's mean
  query time by at least the fraction by which it grows the index (`Δt/t ≥ ΔB/B`); `--budget`
  (bytes per data point) is then only a ceiling. `--stop budget` fills the budget instead (items
  below the `--kappa` floor excluded).
- **Index bytes** `B` (of both stops): the index's compact bytes without the selected
  structures, counted as the reports count a configuration's bytes — base vectors `n·d·elem`,
  the graph as adjacency lists `4·(nodes + edges)` (whatever its file format) and its router
  vectors, the label store, the static bitsets (`θ = ⌈n/64⌉`), the sketches and the start
  tables. `--index-bytes` takes this value (e.g. measured by a probe `flab run` of the
  configuration, whose ceiling `--budget` is then a fraction of the same number); without it the
  selector counts what it reads (all but the sketches and start tables: `select::IndexBytes`).
- **Kinds.** `--track hybrid` (default): combination postings (pairs of labels, the exact path's
  `pair` plan) and per-label sub-graphs (islands, the cost crossover's `island` route), written as
  `sel-b<B>.pairs.flab` / `sel-b<B>.islands.flab` (a kind with nothing selected writes no file) and
  built. `--track graph`: a router tree per label and a pair tree per label pair, valued against
  the graph router alone; writes the selected labels (`sel-b<B>.trees.txt`, for `flab steiner
  --labels-file`) and pairs (`sel-b<B>.pairs.flab`, for `flab steiner --construction pairs`).
  With `--fixed-trees-max-size SIZE` the label trees are fixed by a size cap instead (every label
  of 1 to SIZE points, as `flab steiner --construction tree --min-size 0 --max-size SIZE` builds
  them; their modelled bytes count in `B` unless `--index-bytes` is given) and only pair trees
  are selected, each valued against the route its queries have with those trees. A router tree's
  bytes are counted as a run's bytes are (`--tree-bytes compact`, default: 4 B per node and per
  edge, the routers' vectors, a pair tree's root entry); `--tree-bytes rows` (a fixed row per
  router) reproduces selections made before that accounting.
- Each run writes a JSON report (the runs per threshold, bytes, the model's mean costs, coverage
  of `--report` predicate sets, `provenance`). `--save-candidates` / `--load-candidates` keep the
  counting pass over every co-occurring label pair (the costly part on large label stores; the
  file, magic `SELCAND3`, records the label store's fingerprint after its records: a `u64`
  length and the provenance JSON; `SELCAND2` / `SELCAND1` files of earlier versions load with a
  warning); results do not depend on `--threads`. The calibration must belong to the dataset, the
  graph's data points and the machine, the pair contexts to the label store (see "Provenance").
- Track G's router pricing keeps the effort list as it was before that cap (`UnitCosts::effort_list_grown`:
  capped at the curve's largest measured list before the growth with the rank of the `K`-th
  match), so its selections do not change with the kernel's list cap (`effort_curve` above).

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

## Computed start tables (`flab starts build`)

Per-label starts for any label store on any graph over the same points (for example a
label-blind graph, or a synthetic label set on an existing graph): for every label, from its
members (data points carrying it):

- `medoid`: the member closest to the label's centroid;
- `random(k)`: `k` distinct members drawn uniformly, in random order (`random` = 1);
- `kmeans(k)`: up to `k` representatives: k-means++ seeding and `--iters` Lloyd iterations,
  then, per non-empty cluster, the member closest to its centroid, largest cluster first
  (labels with at most `k` members get all of them).

Labels with more than `--exact-max` members (default 10,000) use a seeded sample of `--sample`
members (default 10,000) for the centroid, the k-means and the candidates. Distances are
squared L2 in `f32` against `f32` centroids. Every choice depends only on the data, the
parameters and `--seed` (one random stream per label), not on the thread count. The table
carries the graph's global starts and node space, and the label store's dictionary
fingerprint (`label(c*)` / `label(all)` starts of the kernel); the JSON report gives the
counts (labels with a start, entries, distinct nodes, sampled labels, distance computations)
and the build cost (wall and CPU time, threads, peak RSS).

## Graph construction and prune replay (`flab build`)

`flab build` writes `graph.flab` and `meta.json` (parameters, wall and CPU time, threads, peak
RSS, load, counters, degree statistics, bytes, and with `--compare` the edge overlap with
another graph) into the output directory (module `build`).

**Builder.** Vamana with batch-parallel insertion, close to upstream DiskANN's `multi_insert`
(with `IntraBatchCandidates::None`):

- the start is one auxiliary *frozen* node holding a copy of the medoid (the data row closest to
  the mean, as upstream's `StartPointStrategy::Medoid`); it is a regular node of the graph;
- points are inserted in a seeded random order (`--order random --seed s`, or `identity`), in
  batches whose sizes double from 1 up to `--batch` (default 128) and then stay there (prefix
  doubling, instead of upstream's multi-insert bootstrap);
- per batch, in parallel, each point runs a greedy search from the start with list size
  `--l-build` (upstream's semantics: a list of `L + 1` entries, a new entry goes before entries
  at an equal distance) over the graph as it was before the batch; the expanded nodes, sorted by
  (distance, id) and truncated to `--maxc` (750), are its *search pool*, which the pruning rule
  reduces to its out-neighbors;
- then every new edge `p → t` proposes the back-edge `t → p`: per target, the batch's sources
  (by id) are appended while the row stays within `⌊slack · R⌋` entries (`--slack`, 1.3), else
  the row and the new sources are re-pruned to `R` with the same rule (upstream's
  `add_edge_and_prune`). `--final-prune` re-prunes every row longer than `R` at the end.

A batch depends only on the graph before it, so the graph depends on the order and the batch
schedule but not on the thread count (`-t`). Parallelism uses a rayon pool of `-t` threads.

**Pruning rules** (`--prune`; module `build::prune`). Every rule scans the pool in (distance,
id) order and builds on upstream's RobustPrune: a selected `r` occludes a later candidate `c`
when `d(p, c) > α' · d(r, c)` (squared distances; `d(r, c) = 0` always occludes), in rounds
`α' = 1, ×min(α, 1.2), …, α`; neighbors are kept in selection order. Auxiliary nodes carry no
labels.

| Rule | Meaning |
|---|---|
| `alpha(R, α)` | label-blind RobustPrune to degree `R` (α defaults to 1.2) |
| `label_containment(R, α)` | FilteredVamana's label-aware occlusion as the C++ library implements it (`Index::occlude_list`): `r` may occlude `c` only if every label of `c` is a label of `r` (`F_c ⊆ F_r`), whatever the labels of `p` |
| `source_containment(R, α[, θ\|inf[, B[, rarest\|categories]]])` | the FilteredVamana paper's rule: only if `F_p ∩ F_c ⊆ F_r`; with `θ`, only `p`'s labels with at most `θ` points count, with `B` only the first `B` of them in budget order (the labels a capped, budgeted filtered insertion searches for) |
| `coverage(R, α, B[, k])` | the first `R − B` neighbors of `alpha(R, α)` are kept; then, for the labels of `p` from the rarest (fewest points) to the most frequent, while a label has fewer than `k` neighbors carrying it and fewer than `B` coverage edges were added, the closest unselected pool candidate carrying it is added; the remaining slots up to `R` take the rest of the RobustPrune selection |
| `split(R_geo, R_lab[, α])` | `alpha(R_geo, α)`, then up to `R_lab` more neighbors by RobustPrune with the containment rule of `label_containment`, in which the first part's neighbors already occlude |

Label-aware rules read `--labels` (default: the dataset's `labels.flab`).

**Label-aware candidate generation** (`--candidates`; module `build::candidates`):

| Value | The insertion of point `p` |
|---|---|
| `blind` (default) | one greedy search from the frozen start (the builder above) |
| `filtered(<subset>[, B\|R[, L_f\|L[, rarest\|categories]]])` | the blind search plus, for each label `ℓ` of `S(p)`, a greedy search from `ℓ`'s medoid that admits only members of `ℓ` (list size `L_f`, default `--l-build`); `S(p)` = `p`'s labels in the subset (`all`, `<θ>` or `n/<k>`: labels with at most `θ` / `n/k` points), the rarest first (or each category's rarest first), at most `B` (default the degree bound `R`); the pool is the union of every search's expanded nodes. A node that received back-edges before its insertion (a label start) prunes them together with its pool |
| `penalized(<λ>\|inf[, raw\|penalized])` | one greedy search ordered by `d²(p, c) + λ·m·f(p, c)`: `f` = the number of `p`'s categories in which `c` shares none of `p`'s labels, `m` = the median squared distance to the nearest non-identical point (`--nn-sample` seeded points, exact); `inf` orders by `f` first. The pool carries the raw distances or the keys (`penalized`: the rule sees the key as the distance from `p`, also in back-edge re-prunes) |

**Stitching** (`--stitch <subset> --stitch-into <graph.flab>`; module `build::stitch`): for every label of the
subset with at least 2 points, a Vamana graph over the label's points alone (`--stitch-r` 32, `--stitch-l` 100,
α of `--prune`, the label's medoid as start, seeded random order); then the row of every point in some per-label
graph is re-pruned with `--prune` from its row in the given label-blind graph together with its per-label
out-neighbors (pool truncated to `--maxc`); other rows are unchanged; no back-edge pass.

**Candidate pools** (`--pools <file>`): the build records, for every node, its search pool and
every back-edge proposal it received (each with its distance), the insertion order and the
batch schedule (format below).

**Prune replay** (`--replay pools.flab --prune <rule>`): the adjacency rebuilt from recorded
pools, without searching (module `build::replay`):

- `--replay-mode incremental` (default) replays the build's insertion process: every point's
  out-neighbors are the rule applied to its recorded search pool, and the back-edges they propose
  reach their targets batch by batch, appended or re-pruned with the rule as in the builder.
  With the build's rule the replay reproduces the build's graph exactly; with another rule it is
  a proxy for an incremental build with that rule (the pools stay those of the recorded build).
  A target's row evolves only through its own back-edges, so all prunes run in parallel;
- `--replay-mode union` prunes every node once over its whole recorded pool (search pool and
  proposals, repeated ids removed), with no back-edges afterwards (a stitch-then-prune
  construction over the pools). A point can then lose every in-edge (e.g. an exact duplicate
  occluded by its twin in every pool).

`--slack` and `--maxc` default to the recorded build's.

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

## Provenance

Derived files record cheap fingerprints of what they were built from, and the commands that load
them compare those with their inputs (module `provenance`). A mismatch is an
error unless the command is given `--allow-provenance-mismatch` (then a warning: for experiments
only); a file written before the fingerprints existed loads with a warning (older calibrations:
their recorded `n_points`, `dim` and `dtype` are still compared). `flab run` records the checks'
warnings and the flag in `run.json` (`provenance`) and the run's fingerprints in its `index`
(`fingerprints`). `flab provenance --dataset D --graph G` prints the current fingerprints (the
JSON objects below, with `cpu_model`); `--file F` prints what a file records (`null`: nothing).

| file | written by | records | compared with |
|---|---|---|---|
| `pairs.flab`, `islands.flab` (also `flab select`'s), `bitsets.flab`, `sketches.flab` | `flab pairs build`, `islands build`, `select`, `bitsets build`, `sketches build` | `labels` | the dataset's label store |
| pair contexts (`SELCAND3`) | `flab select --save-candidates` | `labels` | the label store (`--load-candidates`) |
| label lists (`sel-b<B>.trees.txt`) | `flab select --track graph` | `labels` (first line `# provenance {…}`) | the label store (`flab steiner --labels-file`) |
| `starts.flab` | `flab starts build`, `flab steiner` | `labels`, `graph` (all nodes) | the label store; the graph it is loaded with when the table has the graph's node count (it may name routers or frozen points; a table of another graph holds data-point starts only and is not bound to a graph) |
| `pair_roots.flab` | `flab steiner --construction pairs` | `labels`, `graph` (the pair-tree graph) | the label store and every node of the run's graph |
| calibrations (`provenance` member) | `flab calibrate` (kept by `flab tree-calibrate`) | `labels`, `graph` (when measured with one), `base`, `cpu_model` | the label store, the data points' rows of the graph (a Steiner graph keeps its base graph's), the base shape and the CPU model (`flab run`, `flab select`, `flab islands build`, `flab tree-calibrate`) |
| selection reports, `flab steiner`'s `meta.json` | `flab select`, `flab steiner` | `labels`, `graph` (and the base graph's fingerprint) | (records) |

Fingerprints (JSON; 64-bit hashes as `"0x"` and 16 hexadecimal digits):

- `labels`: `n_points`, `n_labels`, `n_entries`, `dict_hash` (that of `labels.flab`) and
  `postings_hash` = `H(|P_0|, …, |P_{L−1}|, p[s] for s in samples(E))`: every label's posting
  size in label id order, then the sampled entries of the concatenated postings `p` (section 9
  of `labels.flab`, `E` entries).
- `graph`: `n_data`, `n_nodes`, `data_edges` (the out-degrees of the data points summed),
  `edges` (of all nodes), `data_hash` and `hash`: `H` over the rows of the nodes
  `samples(m)` with `m = n_data` resp. `n_nodes`, a row contributing its degree, then its
  neighbour ids in stored order. Graphs are fingerprinted as loaded (`graph.flab`, or a
  DiskANN index read as `flab import-graph` reads it). `flab steiner` appends routers without
  changing the rows of the base graph's nodes, so a Steiner graph and its base graph share the
  data part (`n_data`, `data_edges`, `data_hash`).
- `base`: `n`, `dim`, `dtype` (`"u8"` or `"f32"`); `cpu_model`: `model name` of
  `/proc/cpuinfo`.
- `samples(m)`: every position `0..m` when `m ≤ 65,536`, else 1,024 runs of 64 consecutive
  positions starting at `⌊i·m/1,024⌋`, `i = 0..1,024`, in increasing order (few random reads of
  a file).
- `H(w_1, …, w_m) = mix(acc ⊕ mix(m))`, `acc = Σ_i mix(w_i + i·φ)` with `i` from 1,
  `φ = 0x9e3779b97f4a7c15`, all arithmetic modulo 2⁶⁴, and `mix(z)`: `z ^= z >> 30;
  z *= 0xbf58476d1ce4e5b9; z ^= z >> 27; z *= 0x94d049bb133111eb; z ^= z >> 31` (SplitMix64's
  finalizer; vectorizable with NumPy `uint64`).

They take a fraction of a second at 40M points (the posting sizes, 65,536 sampled entries, the
degrees and 2 × 65,536 sampled rows; `flab provenance` reports the times). A same-shape change
(the same counts and sizes) is found when a sampled entry or row changes; a new data version
changes many.

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
stores row `i` as `values[offsets[i]..offsets[i + 1]]`. The sidecars of the provenance table
(`pairs.flab`, `islands.flab`, `bitsets.flab`, `sketches.flab`, `starts.flab`,
`pair_roots.flab`) may hold one more section after their own (their sections below, 1 for
`pair_roots.flab`): `FLABPRV\0` followed by the provenance JSON; readers of a format ignore it.

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

### Static per-label bitsets: `bitsets.flab` (magic `FLABBIT\0`)

Parameters: `n_points`, `n_labels` and `dict_hash` of the label store, `θ`, the number of labels
with a bitset `B` (every label with at least `θ` points), `words` = `⌈n_points / 64⌉`.

| Section | Type | Length | Content |
|---|---|---|---|
| 0 | `u32` | `B` | labels with a bitset, increasing |
| 1 | `u64` | `B × words` | the bitsets in the order of section 0; bit `p % 64` of word `p / 64` is set when point `p` carries the label |

### Bottom-b sketches: `sketches.flab` (magic `FLABSKT\0`)

Parameters: `n_points`, `n_labels` and `dict_hash` of the label store, `b`, the priority seed,
the number of entries `E` (`Σ_ℓ min(b, |post(ℓ)|)`). An entry is `(priority << 32) | point` with
`priority` the high 32 bits of SplitMix64(`seed ^ point × 0x9e3779b97f4a7c15`).

| Section | Type | Length | Content |
|---|---|---|---|
| 0 | `u64` | `n_labels + 1` | per-label offsets into section 1 |
| 1 | `u64` | `E` | each label's `min(b, |post|)` lowest entries, increasing |

### Combination postings: `pairs.flab` (magic `FLABPRS\0`)

Parameters: `n_points`, `n_labels`, `dict_hash` of the label store, the size threshold `T` of the
big labels, their number `N`, the stored pairs `S`, their entries `E`, the selection threshold
(`f64` bits). Big labels are indexed in increasing label id; `w = ⌈N / 64⌉`.

| Section | Type | Length | Content |
|---|---|---|---|
| 0 | `u32` | `n_labels` | big index of every label (`u32::MAX`: not big) |
| 1 | `u64` | `N × w` | co-occurrence bits: row `r`, bit `c` (`r < c`) set when big labels `r` and `c` share a point |
| 2 | `u64` | `N + 1` | per-row offsets into section 3 |
| 3 | `u32` | `S` | partners `c > r` of each row with a stored joint posting, increasing |
| 4 | `u64` | `S + 1` | offsets of the joint postings into section 5, in the order of section 3 |
| 5 | `u32` | `E` | the joint postings (increasing point ids) |

### Candidate pools: `pools.flab` (magic `FLABPOL\0`)

Parameters: `n_data`, `n_nodes`, `n_entries`, `n_batches`, `slots` (row capacity of the build),
`maxc` (pool truncation of the build), element type code (1 = `u8`, 2 = `f32`) `| dim << 8`,
length of the metadata in bytes.

| Section | Type | Length | Content |
|---|---|---|---|
| 0 | `u32` | `n_data` | insertion order of the data points |
| 1 | `u64` | `n_batches + 1` | batch boundaries into section 0 |
| 2 | `u64` | `n_nodes + 1` | row offsets per node into sections 4 and 5 |
| 3 | `u32` | `n_nodes` | length of each row's search pool |
| 4 | `u32` | `n_entries` | rows: the node's search pool (sorted by (distance, id), truncated to `maxc`: the list handed to the pruning rule), then the back-edge proposals it received, in arrival order (batch by batch, by source id within a batch) |
| 5 | `f32` | `n_entries` | squared distances from the row's node |
| 6 | element type | `(n_nodes − n_data) × dim` | vectors of the auxiliary nodes |
| 7 | `u8` | metadata length | the build's parameters (JSON text) |

Reading a section with NumPy:

```python
import numpy as np
hdr = np.fromfile(path, dtype="<u8", count=10 + 2 * n_sections)  # magic+version+S, params, table
def section(i, dtype):
    offset, nbytes = hdr[10 + 2 * i], hdr[11 + 2 * i]
    return np.fromfile(path, dtype=dtype, count=nbytes // np.dtype(dtype).itemsize, offset=offset)
```

(`hdr[0]` is the magic, `hdr[1]` packs the version and `S`, `hdr[2:10]` are the parameters.)

## Upstream adapter (`flab-upstream`)

Runs upstream DiskANN's unmodified filtered searches on an index saved by upstream (for example
by the `diskann-benchmark` `graph-index-build` job with `save_path`), with the lab's predicate
providers, a per-query timer and the lab's run outputs (module `upstream`).

| `--algorithms` | upstream code | output |
|---|---|---|
| `knn-postfilter` | `Knn` (unfiltered), list size L | best K matches of the L candidates (post-filter) |
| `inline` | `InlineFilterSearch`, fixed L | best K matches |
| `inline-adaptive` | `InlineFilterSearch` + `AdaptiveL` (`--adaptive-samples`, `--adaptive-scale`) | best K matches |
| `multihop` | `MultihopFilterSearch` | best K matches |
| `beta` | `BetaFilter` layer over `Knn` (`--beta`) | L candidates post-filtered to the best K matches, with recomputed true distances |

Predicate providers (`--providers`) implement upstream's `QueryLabelProvider`:

- `forward`: evaluates the CNF on the point's sorted label row, clauses in ascending order of
  total posting length (most selective first), with early exit;
- `bitmap`: builds a dense match bitmap over the data points from the postings of every query
  (per clause the union, across clauses the intersection), then tests one bit.

Ids at or above the number of data points (frozen start points) never match, and they are never
used to index label rows or bitmaps.

**Timing.** Single-threaded on the calling thread (pin it with `taskset`, and bind memory with
`numactl` if needed). Every configuration (algorithm × provider × K) and effort L gets one untimed
warm-up pass (its results are the reference), then `--reps` timed passes interleaved over all
configurations of the invocation. Each query is timed end to end: predicate compilation (and
the bitmap construction), the upstream search with the strategy's default post-processing, the
adapter's post-filter and the result copy. Wall time (monotonic clock) and thread CPU time are
recorded per query.

**Outputs.** One directory per configuration under `--runs-dir`, plus one line appended to
`--ledger` (JSON lines, also on failure):

- `run.json`: tool, commit, dataset, query source (query set, files or panel with its hash),
  index, resolved configuration and its hash, host (CPU, NUMA node, affinity, memory policy),
  load averages and a CPU calibration at start and end, per-effort summaries;
- `res_<L>.bin`: DiskANN truthset layout (`u32` nq, `u32` K, ids padded with `u32::MAX`, true
  squared-L2 distances padded with `f32::MAX`), rows in query order;
- `perquery.tsv`: `effort, rep, row, wall_ns, cpu_ns, n_dist, n_pred, n_expand, n_hops,
  n_match_seen, n_out, route`, then upstream's raw `up_cmps, up_hops, up_results`, `n_start`,
  `post_pred`, `extra_dist`, `prep_ns`, `prep_entries` (definitions in `run.json`).

```sh
taskset -c 8 numactl --membind=0 flab-upstream run --index graphs/ds/index --dataset ds \
    --qset 10K --algorithms inline,multihop --providers forward,bitmap -k 10,100 \
    --efforts 10,20,50,100,200,500 --runs-dir runs --ledger ledger.jsonl
flab-upstream info --index graphs/ds/index --dtype u8
```

Queries come from `--qset S` (`qS.{u8bin,fbin}` and `qS.fpred` in the dataset directory),
explicit `--queries`/`--predicates`, or `--panel` (TSV with columns `qset`, `qidx` and optionally
`split`; rows in panel order). Efforts below K are skipped for that K; `-k K:L1/L2/...` gives K
its own efforts (e.g. `-k 10:10/20/50,100:100/200/500`).
