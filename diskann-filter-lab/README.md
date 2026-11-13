# diskann-filter-lab

Research tooling for filtered approximate nearest neighbor search with predicates in conjunctive
normal form (an AND of ORs over Boolean labels). The crate provides:

- compact stores for vectors, labels and predicates with allocation-free accessors
  (`Matrix::row`, `LabelStore::row` / `postings`, `PredicateSet::get(q).clause(i)`), meant to be
  used by search kernels;
- binary file formats for labels, predicates and ground truth, and converters from text formats;
- exact filtered k-NN ground truth with complete distance-tie bands;
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

# Summaries
flab info labels.flab
flab info queries.fpred
flab info gt100.fgt
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

Reading a section with NumPy:

```python
import numpy as np
hdr = np.fromfile(path, dtype="<u8", count=10 + 2 * n_sections)  # magic+version+S, params, table
def section(i, dtype):
    offset, nbytes = hdr[10 + 2 * i], hdr[11 + 2 * i]
    return np.fromfile(path, dtype=dtype, count=nbytes // np.dtype(dtype).itemsize, offset=offset)
```

(`hdr[0]` is the magic, `hdr[1]` packs the version and `S`, `hdr[2:10]` are the parameters.)
