/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The selector's workload, one definition for every kind of candidate (islands, combination
//! postings, track G's trees): a fixed mixture of two parts, `U`'s share `μ`
//! ([`QueryModel::uniform`]):
//!
//! - **D** (data-like): a data point drawn uniformly, its label set turned into an AND-of-ORs
//!   predicate ([`crate::islands::QueryModel`]); the query vector is that point's.
//! - **U** (independent): a conjunction of `k` single-label clauses of distinct categories, `k`
//!   uniform in `1..=min(3, C)` (`C`: the categories with labels), drawn by a uniform walk on the
//!   label co-occurrence structure — the first label uniformly among the labels of a uniformly
//!   drawn category; each next label uniformly among the *distinct* labels of the categories not
//!   used yet that co-occur with every label drawn so far (the labels present on the
//!   conjunction's member points, each counted once); if there is none, the conjunction keeps
//!   the labels drawn so far. The query vector is a data point drawn uniformly, independently
//!   of the labels. (Close to, not exactly, uniform over the satisfiable conjunctions.)
//!
//! [`independent`] gives `U` as weighted queries ([`Independent`]): its one- and two-label parts
//! exactly — every label that occurs with its probability, every co-occurring pair of labels of
//! different categories with its probability (a two-label walk that stops after one label
//! counted for that label) — and its three-label part as `n_queries` walks drawn i.i.d.
//! (seeded). [`Walk::draw`] draws `U` i.i.d. as concrete queries (for measurements).

use crate::{
    islands::{draw_query, mix, query_matches, QueryModel},
    labels::LabelStore,
    select::PairCand,
};

/// Salt of the independent queries' vector draws.
const POINT_SALT: u64 = 0x5eed_0f0b_ec70_45a1;
/// Salt of the walks.
const WALK_SALT: u64 = 0x3c6e_f372_fe94_f82b;
/// Proposals tried before the third label's candidates are enumerated.
const PROPOSALS: usize = 64;

/// The data point whose vector is query `q`'s in a sample of `n_data` base-derived queries
/// followed by the independent ones: a base-derived query's own point (the one it was drawn
/// from); for an independent query a point drawn uniformly, independently of its labels (seeded
/// by `model.seed` and `q`).
pub fn query_point(model: &QueryModel, n_points: usize, n_data: usize, q: usize) -> u32 {
    let h = if q < n_data {
        mix(model.seed ^ mix(q as u64))
    } else {
        mix(mix(model.seed ^ POINT_SALT) ^ mix(q as u64))
    };
    (h % n_points.max(1) as u64) as u32
}

/// Part of the workload a query comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Part {
    /// `D`: a data point's sub-predicate, the point's own vector.
    Data,
    /// `U`: single-label clauses of distinct categories, an independent vector.
    Independent,
}

/// One query drawn from the workload: its clauses (sorted label ids each) and the data point
/// whose vector it searches with.
#[derive(Debug, Clone, PartialEq)]
pub struct Drawn {
    pub part: Part,
    pub clauses: Vec<Vec<u32>>,
    pub point: u32,
}

/// Base-derived query `i` of the workload (the `i`-th of a sample's base-derived queries;
/// `None` for a point without labels).
pub fn draw_data(labels: &LabelStore, model: &QueryModel, i: usize) -> Option<Drawn> {
    let (mut kl, mut clauses, mut cats) = (Vec::new(), Vec::new(), Vec::new());
    if !draw_query(labels, model, i, &mut kl, &mut clauses, &mut cats) {
        return None;
    }
    Some(Drawn {
        part: Part::Data,
        clauses: clauses
            .iter()
            .map(|&(a, b)| {
                let mut c = kl[a..b].to_vec();
                c.sort_unstable();
                c
            })
            .collect(),
        point: query_point(model, labels.n_points(), usize::MAX, i),
    })
}

/// One independent query with its probability.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct WQuery {
    /// Its labels, sorted, of distinct categories: one single-label clause each.
    pub labels: Vec<u32>,
    /// Its match count `M` (exact for one or two labels; for three estimated as the base-derived
    /// queries' are).
    pub joint: f64,
    /// Its probability within `U` (the weights of all of `U`'s queries sum to 1).
    pub weight: f64,
}

/// `U` as weighted queries: the labels that occur (one-label queries), then the co-occurring
/// pairs (two-label queries, in the order of the pair contexts), then the sampled walks of the
/// three-label part; and, for the combination postings, the queries holding each pair of
/// labels.
#[derive(Debug, Clone, Default)]
pub struct Independent {
    pub queries: Vec<WQuery>,
    /// `(a, b, query)` for every pair of labels `a < b` of a query, sorted.
    pairs: Vec<(u32, u32, u32)>,
    /// Queries of the exact one-label part, of the exact two-label part, sampled walks.
    pub n_single: usize,
    pub n_pair: usize,
    pub n_walk: usize,
    /// Largest number of clauses, `min(3, C)`.
    pub k_max: usize,
}

impl Independent {
    pub fn len(&self) -> usize {
        self.queries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }

    /// The queries (indices into [`Self::queries`]) whose labels include both `a` and `b`
    /// (`a < b`).
    pub fn pair_queries(&self, a: u32, b: u32) -> impl Iterator<Item = usize> + '_ {
        let lo = self.pairs.partition_point(|&(x, y, _)| (x, y) < (a, b));
        self.pairs[lo..]
            .iter()
            .take_while(move |&&(x, y, _)| (x, y) == (a, b))
            .map(|&(_, _, q)| q as usize)
    }

    /// Weighted means of per-query values `x` (indexed like [`Self::queries`]) within the
    /// queries of one, two and three labels (0 for an empty group).
    pub fn means_by_size(&self, x: &[f64]) -> [f64; 3] {
        let mut s = [0.0; 3];
        let mut w = [0.0; 3];
        for (q, v) in self.queries.iter().zip(x) {
            let i = q.labels.len().clamp(1, 3) - 1;
            s[i] += q.weight * v;
            w[i] += q.weight;
        }
        [0, 1, 2].map(|i| if w[i] > 0.0 { s[i] / w[i] } else { 0.0 })
    }
}

/// The walk's structure: per category its labels that occur, and per label the labels of other
/// categories it co-occurs with.
pub struct Walk<'a> {
    labels: &'a LabelStore,
    contexts: &'a [PairCand],
    /// The labels that occur, per category with labels.
    cats: Vec<Vec<u32>>,
    /// Co-occurring labels of other categories, per label (rows sorted).
    off: Vec<usize>,
    adj: Vec<u32>,
    /// Largest number of clauses, `min(3, C)`.
    pub k_max: usize,
}

impl<'a> Walk<'a> {
    /// `contexts`: every co-occurring pair of labels of different categories, sorted
    /// ([`crate::select::pair_contexts`]).
    pub fn new(labels: &'a LabelStore, contexts: &'a [PairCand]) -> Self {
        let nl = labels.n_labels();
        let mut cats = vec![Vec::new(); labels.n_categories()];
        for l in 0..nl as u32 {
            if labels.posting_len(l) > 0 {
                cats[labels.category(l) as usize].push(l);
            }
        }
        cats.retain(|c| !c.is_empty());
        let mut deg = vec![0usize; nl + 1];
        for p in contexts {
            deg[p.a as usize] += 1;
            deg[p.b as usize] += 1;
        }
        let mut off = vec![0usize; nl + 1];
        for l in 0..nl {
            off[l + 1] = off[l] + deg[l];
        }
        let mut fill = off.clone();
        let mut adj = vec![0u32; off[nl]];
        for p in contexts {
            adj[fill[p.a as usize]] = p.b;
            fill[p.a as usize] += 1;
            adj[fill[p.b as usize]] = p.a;
            fill[p.b as usize] += 1;
        }
        for l in 0..nl {
            adj[off[l]..off[l + 1]].sort_unstable();
        }
        let k_max = cats.len().min(3);
        Self {
            labels,
            contexts,
            cats,
            off,
            adj,
            k_max,
        }
    }

    /// The labels of other categories that co-occur with `l` (sorted).
    pub fn neighbors(&self, l: u32) -> &[u32] {
        &self.adj[self.off[l as usize]..self.off[l as usize + 1]]
    }

    /// Probability that the walk's first label is `l` (`1/C · 1/L_c`, `L_c` the labels of
    /// `l`'s category that occur).
    pub fn first_prob(&self, l: u32) -> f64 {
        let c = self.cats.iter().find(|c| c.binary_search(&l).is_ok());
        c.map_or(0.0, |c| 1.0 / (self.cats.len() as f64 * c.len() as f64))
    }

    fn first(&self, z: &mut u64) -> u32 {
        *z = mix(*z);
        let c = &self.cats[(*z % self.cats.len() as u64) as usize];
        *z = mix(*z);
        c[(*z % c.len() as u64) as usize]
    }

    /// Whether some point carries `x`, `y` and `w`.
    fn cooccur3(&self, x: u32, y: u32, w: u32) -> bool {
        let mut t = [x, y, w];
        t.sort_by_key(|&l| self.labels.posting_len(l));
        self.labels.postings(t[0]).iter().any(|&p| {
            let r = self.labels.row(p);
            r.binary_search(&t[1]).is_ok() && r.binary_search(&t[2]).is_ok()
        })
    }

    /// A third label for `x`, `y`: uniformly among the distinct labels of the other categories
    /// on the points carrying both (proposals drawn uniformly from the smaller of the two
    /// neighbour lists outside `x`'s and `y`'s categories and accepted if they qualify — so the
    /// accepted one is uniform among those that do; after [`PROPOSALS`] rejections the
    /// qualifying ones are enumerated). `None` if there is none.
    fn third(&self, x: u32, y: u32, z: &mut u64) -> Option<u32> {
        let (nx, ny) = (self.neighbors(x), self.neighbors(y));
        let (prop, other) = if nx.len() <= ny.len() {
            (nx, ny)
        } else {
            (ny, nx)
        };
        let mut cut: Vec<(usize, usize)> = [x, y]
            .iter()
            .map(|&l| {
                let r = self.labels.category_labels(self.labels.category(l));
                (
                    prop.partition_point(|&v| v < r.start),
                    prop.partition_point(|&v| v < r.end),
                )
            })
            .collect();
        cut.sort_unstable();
        let segs: Vec<&[u32]> = [
            &prop[..cut[0].0],
            &prop[cut[0].1.max(cut[0].0)..cut[1].0.max(cut[0].1)],
            &prop[cut[1].1.max(cut[0].1)..],
        ]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect();
        let total: usize = segs.iter().map(|s| s.len()).sum();
        if total == 0 {
            return None;
        }
        let ok = |w: u32| other.binary_search(&w).is_ok() && self.cooccur3(x, y, w);
        let pick = |mut i: usize| -> u32 {
            for s in &segs {
                if i < s.len() {
                    return s[i];
                }
                i -= s.len();
            }
            unreachable!("an index below the total")
        };
        for _ in 0..PROPOSALS {
            *z = mix(*z);
            let w = pick((*z % total as u64) as usize);
            if ok(w) {
                return Some(w);
            }
        }
        let all: Vec<u32> = segs
            .iter()
            .flat_map(|s| s.iter().copied())
            .filter(|&w| ok(w))
            .collect();
        if all.is_empty() {
            return None;
        }
        *z = mix(*z);
        Some(all[(*z % all.len() as u64) as usize])
    }

    /// A walk of (at most) `k` labels from state `z` (sorted).
    fn walk(&self, k: usize, z: &mut u64) -> Vec<u32> {
        let x = self.first(z);
        let mut out = vec![x];
        if k >= 2 {
            let nx = self.neighbors(x);
            if !nx.is_empty() {
                *z = mix(*z);
                let y = nx[(*z % nx.len() as u64) as usize];
                out.push(y);
                if k >= 3 {
                    if let Some(w) = self.third(x, y, z) {
                        out.push(w);
                    }
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// Match count of a conjunction of `labels` (sorted, one clause each).
    fn joint(&self, model: &QueryModel, labels: &[u32]) -> f64 {
        match labels {
            [a] => self.labels.posting_len(*a) as f64,
            [a, b] => {
                let i = self
                    .contexts
                    .binary_search_by(|p| (p.a, p.b).cmp(&(*a, *b)))
                    .expect("co-occurring labels");
                f64::from(self.contexts[i].joint)
            }
            _ => {
                let clauses: Vec<(usize, usize)> = (0..labels.len()).map(|i| (i, i + 1)).collect();
                query_matches(self.labels, model, labels, &clauses).2
            }
        }
        .max(1.0)
    }

    /// Independent query `i` drawn i.i.d. (seeded by `model.seed` and `i`): `k` uniform in
    /// `1..=min(3, C)`, the walk, and a vector point drawn independently.
    pub fn draw(&self, model: &QueryModel, i: usize) -> Option<Drawn> {
        if self.cats.is_empty() {
            return None;
        }
        let mut z = mix(mix(model.seed ^ WALK_SALT) ^ mix(i as u64));
        let k = 1 + (z % self.k_max as u64) as usize;
        let labels = self.walk(k, &mut z);
        z = mix(z);
        Some(Drawn {
            part: Part::Independent,
            clauses: labels.iter().map(|&l| vec![l]).collect(),
            point: (z % self.labels.n_points().max(1) as u64) as u32,
        })
    }
}

/// `U` as weighted queries (see the module documentation): with `K = min(3, C)`, each number of
/// clauses has probability `1/K`; the one-label part is every label `ℓ` that occurs with weight
/// `(1/K)·P₁(ℓ)` (plus `(1/K)·P₁(ℓ)` when `ℓ` co-occurs with no label of another category: a
/// two-label walk stops there), the two-label part every co-occurring pair `{a, b}` with weight
/// `(1/K)·(P₁(a)/deg(a) + P₁(b)/deg(b))` (`P₁`: [`Walk::first_prob`], `deg`: the number of
/// co-occurring labels of other categories), the three-label part `model.n_queries` walks
/// drawn i.i.d. (seeded), weight `(1/K)/n_queries` each (a walk that stops early keeps one or
/// two labels). Walks in parallel; the result does not depend on the thread count.
pub fn independent(
    labels: &LabelStore,
    contexts: &[PairCand],
    model: &QueryModel,
    threads: usize,
) -> Independent {
    let walk = Walk::new(labels, contexts);
    let k = walk.k_max;
    if k == 0 {
        return Independent::default();
    }
    let kf = k as f64;
    let mut queries = Vec::new();
    for c in &walk.cats {
        let p1 = 1.0 / (walk.cats.len() as f64 * c.len() as f64);
        for &l in c {
            let stops = k >= 2 && walk.neighbors(l).is_empty();
            queries.push(WQuery {
                labels: vec![l],
                joint: labels.posting_len(l) as f64,
                weight: p1 / kf * if stops { 2.0 } else { 1.0 },
            });
        }
    }
    queries.sort_by_key(|q| q.labels[0]);
    let n_single = queries.len();
    if k >= 2 {
        for p in contexts {
            let w = walk.first_prob(p.a) / walk.neighbors(p.a).len() as f64
                + walk.first_prob(p.b) / walk.neighbors(p.b).len() as f64;
            queries.push(WQuery {
                labels: vec![p.a, p.b],
                joint: f64::from(p.joint).max(1.0),
                weight: w / kf,
            });
        }
    }
    let n_pair = queries.len() - n_single;
    let n_walk = if k >= 3 { model.n_queries } else { 0 };
    if n_walk > 0 {
        let chunk = 256usize;
        let parts = crate::parallel::par_map(
            n_walk.div_ceil(chunk),
            threads,
            || (),
            |ci, _| {
                (ci * chunk..((ci + 1) * chunk).min(n_walk))
                    .map(|i| {
                        let mut z = mix(mix(model.seed ^ WALK_SALT ^ 3) ^ mix(i as u64));
                        let l = walk.walk(3, &mut z);
                        WQuery {
                            joint: walk.joint(model, &l),
                            labels: l,
                            weight: 1.0 / kf / n_walk as f64,
                        }
                    })
                    .collect::<Vec<_>>()
            },
        );
        queries.extend(parts.into_iter().flatten());
    }
    let mut pairs = Vec::new();
    for (q, wq) in queries.iter().enumerate() {
        for (i, &a) in wq.labels.iter().enumerate() {
            for &b in &wq.labels[i + 1..] {
                pairs.push((a, b, q as u32));
            }
        }
    }
    pairs.sort_unstable();
    Independent {
        queries,
        pairs,
        n_single,
        n_pair,
        n_walk,
        k_max: k,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::labels::LabelStoreBuilder;

    /// 2,000 points, three categories `c` (6 labels), `d` (3 labels), `e` (4 labels); label
    /// `c5` never meets `e3`.
    fn store() -> LabelStore {
        let mut b = LabelStoreBuilder::new();
        let h = |p: u64, s: u64| {
            let mut z = p.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ s;
            z = (z ^ (z >> 31)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z ^ (z >> 29)
        };
        for p in 0..2000u64 {
            let c = format!("c{}", h(p, 1) % 6);
            let d = format!("d{}", h(p, 2) % 3);
            let mut e = format!("e{}", h(p, 3) % 4);
            if c == "c5" && e == "e3" {
                e = "e0".into();
            }
            let mut v = vec![("c", c.as_str()), ("d", d.as_str()), ("e", e.as_str())];
            v.sort_unstable();
            b.push_point(v).unwrap();
        }
        b.finish().unwrap()
    }

    #[test]
    fn the_third_label_is_uniform_among_those_on_the_pairs_points() {
        let s = store();
        let (counts, _) = crate::select::label_weights(&s, 0.5, 1);
        let contexts = crate::select::pair_contexts(&s, &counts, 0.5, 1);
        let walk = Walk::new(&s, &contexts);
        for p in contexts.iter().step_by(5) {
            let (x, y) = (p.a, p.b);
            // The distinct labels of the third category on the points carrying both.
            let mut want: Vec<u32> = s
                .postings(x)
                .iter()
                .filter(|&&q| s.row(q).binary_search(&y).is_ok())
                .flat_map(|&q| s.row(q).iter().copied())
                .filter(|&l| s.category(l) != s.category(x) && s.category(l) != s.category(y))
                .collect();
            want.sort_unstable();
            want.dedup();
            let mut z = mix(u64::from(x) << 32 | u64::from(y));
            let draws = 6_000;
            let mut got = std::collections::BTreeMap::<u32, usize>::new();
            for _ in 0..draws {
                let w = walk.third(x, y, &mut z).expect("a third label");
                *got.entry(w).or_default() += 1;
            }
            assert_eq!(got.keys().copied().collect::<Vec<_>>(), want, "({x}, {y})");
            let e = draws as f64 / want.len() as f64;
            for (&w, &n) in &got {
                assert!(
                    (n as f64 - e).abs() < 5.0 * e.sqrt(),
                    "({x}, {y}) {w}: {n} vs {e}"
                );
            }
        }
        // `c5` and `e3` never meet: no walk holds both, and a pair with `c5` never takes `e3`.
        let idx = s.name_index();
        let (c5, e3) = (idx["c5"], idx["e3"]);
        assert!(walk.neighbors(c5).binary_search(&e3).is_err());
    }

    #[test]
    fn the_first_label_is_uniform_within_a_uniform_category() {
        let s = store();
        let (counts, _) = crate::select::label_weights(&s, 0.5, 1);
        let contexts = crate::select::pair_contexts(&s, &counts, 0.5, 1);
        let walk = Walk::new(&s, &contexts);
        let idx = s.name_index();
        let want: f64 = (0..s.n_labels() as u32).map(|l| walk.first_prob(l)).sum();
        assert!((want - 1.0).abs() < 1e-12);
        assert!((walk.first_prob(idx["d0"]) - 1.0 / 9.0).abs() < 1e-12);
        assert!((walk.first_prob(idx["c0"]) - 1.0 / 18.0).abs() < 1e-12);
        let mut z = 17u64;
        let mut n_d = 0usize;
        let cd = s.category(idx["d0"]);
        for _ in 0..30_000 {
            n_d += usize::from(s.category(walk.first(&mut z)) == cd);
        }
        assert!((n_d as f64 / 30_000.0 - 1.0 / 3.0).abs() < 0.015);
    }
}
