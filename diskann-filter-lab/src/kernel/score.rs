/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Scorers: the beam key of an evaluated node from its true distance `d`, match bit `m` and
//! (weighted) failed-clause count `g`.
//!
//! | Scorer | Key | Family |
//! |---|---|---|
//! | `distance` | `d` | [`DistanceScore`] |
//! | `bonus(β)` | `β·d` if `m`, else `d` | [`BonusScore`] |
//! | `additive(λ)` | `d + λ·g` (λ in squared-distance units) | [`AdditiveScore`] |
//! | `normalized_additive(η)` | `d + η·s·g` | [`AdditiveScore`] |
//! | `multiplicative(η)` | `(d + ε_q)·e^{η·g}` | [`MultiplicativeScore`] |
//! | `shifted(η)` | `log(1 + d/s) + η·g`, ranked as `(d + s)·e^{η·g}` | [`MultiplicativeScore`] |
//! | `shell_adaptive(κ)` | `(d + ε_q)·(1 + ρ_q)^{κ·F}`, `F = g / Σw` | [`MultiplicativeScore`] |
//! | `lexicographic` | `(g, d)` | [`LexicographicScore`] |
//!
//! Keys are `u64` whose unsigned order is the order of the scores ([`super::fkey`] of the
//! `f32` score; `(fkey(g), fkey(d))` for the lexicographic key), so ties are exact `f32` ties.
//! `s`, `ε_q` and `ρ_q` come from the pilot.

use super::{fkey, funkey};

/// Per-query parameters of a scorer.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Params {
    /// `β` (bonus).
    pub beta: f32,
    /// Additive coefficient `c` of `d + c·g` (`λ`, or `η·s`).
    pub c: f32,
    /// Exponent coefficient of `e^{η·g}`.
    pub eta: f32,
    /// Distance offset (`ε_q`, or `s` for `shifted`).
    pub eps: f32,
    /// Largest `g` of a node that may be in the beam (for distance lower bounds).
    pub g_max: f32,
    /// Whether non-matching nodes may be in the beam (for `bonus` distance lower bounds).
    pub nonmatch_in_beam: bool,
}

/// A scorer family; a compile-time parameter of the hot loop.
pub trait Score: Copy + Send + Sync + 'static {
    /// Whether keys depend on `g`.
    const USES_G: bool;
    /// Whether keys depend on the match bit.
    const USES_MATCH: bool;
    fn new(p: &Params) -> Self;
    /// Key of a node.
    fn key(&self, d: f32, m: bool, g: f32) -> u64;
    /// The smallest key a node at distance `d` can get (its most favourable outcome).
    fn key_lower_bound(&self, d: f32) -> u64;
    /// `inf {d ≥ 0 : key(d, m, g) ≥ k for some (m, g) that may be in the beam}`: a lower bound
    /// on the true distance of any node with key `≥ k`.
    fn dist_lower_bound(&self, k: u64) -> f32;
    /// Human-readable value of a key (traces).
    fn describe(&self, k: u64) -> String;
}

#[inline(always)]
fn f(k: u64) -> f32 {
    funkey(k as u32)
}

/// `d`.
#[derive(Debug, Clone, Copy)]
pub struct DistanceScore;

impl Score for DistanceScore {
    const USES_G: bool = false;
    const USES_MATCH: bool = false;

    fn new(_: &Params) -> Self {
        Self
    }

    #[inline(always)]
    fn key(&self, d: f32, _m: bool, _g: f32) -> u64 {
        u64::from(fkey(d))
    }

    #[inline(always)]
    fn key_lower_bound(&self, d: f32) -> u64 {
        u64::from(fkey(d))
    }

    #[inline(always)]
    fn dist_lower_bound(&self, k: u64) -> f32 {
        f(k).max(0.0)
    }

    fn describe(&self, k: u64) -> String {
        f(k).to_string()
    }
}

/// `β·d` for matches, `d` otherwise (upstream's beta filter).
#[derive(Debug, Clone, Copy)]
pub struct BonusScore {
    beta: f32,
    nonmatch_in_beam: bool,
}

impl Score for BonusScore {
    const USES_G: bool = false;
    const USES_MATCH: bool = true;

    fn new(p: &Params) -> Self {
        Self {
            beta: p.beta,
            nonmatch_in_beam: p.nonmatch_in_beam,
        }
    }

    #[inline(always)]
    fn key(&self, d: f32, m: bool, _g: f32) -> u64 {
        u64::from(fkey(if m { d * self.beta } else { d }))
    }

    #[inline(always)]
    fn key_lower_bound(&self, d: f32) -> u64 {
        u64::from(fkey(d * self.beta.min(1.0)))
    }

    #[inline(always)]
    fn dist_lower_bound(&self, k: u64) -> f32 {
        let k = f(k).max(0.0);
        let as_match = k / self.beta;
        if self.nonmatch_in_beam {
            k.min(as_match)
        } else {
            as_match
        }
    }

    fn describe(&self, k: u64) -> String {
        f(k).to_string()
    }
}

/// `d + c·g` (`additive(λ)`: `c = λ`; `normalized_additive(η)`: `c = η·s`).
#[derive(Debug, Clone, Copy)]
pub struct AdditiveScore {
    c: f32,
    g_max: f32,
}

impl Score for AdditiveScore {
    const USES_G: bool = true;
    const USES_MATCH: bool = false;

    fn new(p: &Params) -> Self {
        Self {
            c: p.c,
            g_max: p.g_max,
        }
    }

    #[inline(always)]
    fn key(&self, d: f32, _m: bool, g: f32) -> u64 {
        u64::from(fkey(d + self.c * g))
    }

    #[inline(always)]
    fn key_lower_bound(&self, d: f32) -> u64 {
        u64::from(fkey(d + self.c.min(0.0) * self.g_max))
    }

    #[inline(always)]
    fn dist_lower_bound(&self, k: u64) -> f32 {
        (f(k) - self.c.max(0.0) * self.g_max).max(0.0)
    }

    fn describe(&self, k: u64) -> String {
        f(k).to_string()
    }
}

/// `(d + ε)·e^{η·g}` (`multiplicative(η)`, `shifted(η)` with `ε = s`, `shell_adaptive(κ)`
/// with `η = κ·ln(1 + ρ_q) / Σw`).
#[derive(Debug, Clone, Copy)]
pub struct MultiplicativeScore {
    eps: f32,
    eta: f32,
    g_max: f32,
}

impl Score for MultiplicativeScore {
    const USES_G: bool = true;
    const USES_MATCH: bool = false;

    fn new(p: &Params) -> Self {
        Self {
            eps: p.eps,
            eta: p.eta,
            g_max: p.g_max,
        }
    }

    #[inline(always)]
    fn key(&self, d: f32, _m: bool, g: f32) -> u64 {
        let base = d + self.eps;
        u64::from(fkey(if g == 0.0 {
            base
        } else {
            base * (self.eta * g).exp()
        }))
    }

    #[inline(always)]
    fn key_lower_bound(&self, d: f32) -> u64 {
        let base = d + self.eps;
        u64::from(fkey(if self.eta >= 0.0 {
            base
        } else {
            base * (self.eta * self.g_max).exp()
        }))
    }

    #[inline(always)]
    fn dist_lower_bound(&self, k: u64) -> f32 {
        let worst = (-self.eta.max(0.0) * self.g_max).exp();
        (f(k) * worst - self.eps).max(0.0)
    }

    fn describe(&self, k: u64) -> String {
        f(k).to_string()
    }
}

/// `(g, d)` compared lexicographically.
#[derive(Debug, Clone, Copy)]
pub struct LexicographicScore {
    g_max: f32,
}

impl Score for LexicographicScore {
    const USES_G: bool = true;
    const USES_MATCH: bool = false;

    fn new(p: &Params) -> Self {
        Self { g_max: p.g_max }
    }

    #[inline(always)]
    fn key(&self, d: f32, _m: bool, g: f32) -> u64 {
        (u64::from(fkey(g)) << 32) | u64::from(fkey(d))
    }

    #[inline(always)]
    fn key_lower_bound(&self, d: f32) -> u64 {
        (u64::from(fkey(0.0)) << 32) | u64::from(fkey(d))
    }

    #[inline(always)]
    fn dist_lower_bound(&self, k: u64) -> f32 {
        let g = funkey((k >> 32) as u32);
        if self.g_max > g {
            0.0
        } else {
            f(k).max(0.0)
        }
    }

    fn describe(&self, k: u64) -> String {
        format!("{}|{}", funkey((k >> 32) as u32), f(k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Params {
        Params {
            beta: 0.5,
            c: 10.0,
            eta: 0.7,
            eps: 2.0,
            g_max: 2.0,
            nonmatch_in_beam: true,
        }
    }

    /// `dist_lower_bound(key(d, m, g)) ≤ d` for every outcome, and `key_lower_bound(d) ≤
    /// key(d, m, g)`.
    fn check_bounds<S: Score>(s: S, outcomes: &[(bool, f32)]) {
        for d in [0.0f32, 0.5, 3.0, 100.0, 12345.0] {
            for &(m, g) in outcomes {
                let k = s.key(d, m, g);
                assert!(
                    s.key_lower_bound(d) <= k,
                    "key lower bound at d={d} m={m} g={g}"
                );
                let lb = s.dist_lower_bound(k);
                assert!(
                    lb <= d * (1.0 + 1e-6) + 1e-6,
                    "distance lower bound {lb} > {d}"
                );
            }
        }
    }

    #[test]
    fn keys_and_bounds() {
        let p = params();
        let outcomes = [(true, 0.0), (false, 1.0), (false, 2.0)];
        check_bounds(DistanceScore::new(&p), &outcomes);
        check_bounds(BonusScore::new(&p), &outcomes);
        check_bounds(AdditiveScore::new(&p), &outcomes);
        check_bounds(MultiplicativeScore::new(&p), &outcomes);
        check_bounds(LexicographicScore::new(&p), &outcomes);

        let b = BonusScore::new(&p);
        assert_eq!(b.key(4.0, true, 0.0), b.key(2.0, false, 0.0));
        // Only matches in the beam: a key k means a distance of at least k / β.
        let strict = BonusScore::new(&Params {
            nonmatch_in_beam: false,
            ..p
        });
        assert_eq!(strict.dist_lower_bound(strict.key(4.0, true, 0.0)), 4.0);

        let a = AdditiveScore::new(&p);
        assert!(a.key(5.0, false, 1.0) == a.key(15.0, true, 0.0));
        let l = LexicographicScore::new(&p);
        assert!(l.key(1000.0, true, 0.0) < l.key(1.0, false, 1.0));
        assert_eq!(l.describe(l.key(3.0, false, 1.0)), "1|3");
        // The lexicographic bound is 0 when a larger g is admissible.
        assert_eq!(l.dist_lower_bound(l.key(9.0, false, 1.0)), 0.0);
        assert_eq!(l.dist_lower_bound(l.key(9.0, false, 2.0)), 9.0);

        let m = MultiplicativeScore::new(&p);
        assert_eq!(m.key(1.0, true, 0.0), u64::from(fkey(3.0)));
        assert!(
            m.key(0.0, false, 1.0) > m.key(0.0, true, 0.0),
            "ε penalizes duplicates"
        );
    }
}
