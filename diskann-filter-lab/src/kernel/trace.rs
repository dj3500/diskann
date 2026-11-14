/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Event traces of single queries (trace mode; untimed).
//!
//! One event per offered node (`offer`, or `start` for starts) with its discovery source, hop,
//! true distance, key, match bit, failed-clause count, outcome (`admitted`, `rejected`,
//! `lazily_skipped`, `navigated`), beam rank and evicted node, and the step at which it was
//! expanded; `answer` events for answer-heap insertions (with the evicted answer), and
//! `two_hop` / `two_hop_cut` events for the kept and truncated second-hop candidates.

use std::io::Write;

use hashbrown::HashMap;

/// One trace event.
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    pub seq: u32,
    pub kind: &'static str,
    pub node: u32,
    /// Node whose expansion discovered this one (`None` for starts).
    pub source: Option<u32>,
    pub hop: u32,
    pub d: Option<f32>,
    pub key: Option<String>,
    pub m: Option<bool>,
    pub g: Option<f32>,
    pub outcome: &'static str,
    pub rank: Option<usize>,
    pub evicted: Option<u32>,
    pub expanded_at: Option<u32>,
}

/// The trace of one query.
#[derive(Debug, Clone, Default)]
pub struct Trace {
    pub events: Vec<Event>,
    offered: HashMap<u32, usize>,
    hop: HashMap<u32, u32>,
    step: u32,
}

impl Trace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.events.clear();
        self.offered.clear();
        self.hop.clear();
        self.step = 0;
    }

    fn hop_of(&self, source: Option<u32>) -> u32 {
        source.map_or(0, |s| self.hop.get(&s).copied().unwrap_or(0) + 1)
    }

    /// Records an offered node.
    #[expect(clippy::too_many_arguments, reason = "one argument per trace column")]
    pub fn offer(
        &mut self,
        node: u32,
        source: Option<u32>,
        d: Option<f32>,
        key: Option<String>,
        m: Option<bool>,
        g: Option<f32>,
        outcome: &'static str,
        rank: Option<usize>,
        evicted: Option<u32>,
    ) {
        let hop = self.hop_of(source);
        self.hop.insert(node, hop);
        self.offered.insert(node, self.events.len());
        self.events.push(Event {
            seq: self.events.len() as u32,
            kind: if source.is_none() { "start" } else { "offer" },
            node,
            source,
            hop,
            d,
            key,
            m,
            g,
            outcome,
            rank,
            evicted,
            expanded_at: None,
        });
    }

    /// Records the expansion of `node` (starts a new step when `new_step`).
    pub fn expand(&mut self, node: u32, new_step: bool) {
        if new_step {
            self.step += 1;
        }
        if let Some(&i) = self.offered.get(&node) {
            self.events[i].expanded_at = Some(self.step);
        }
    }

    /// Records an answer-heap insertion.
    pub fn answer(&mut self, node: u32, d: f32, evicted: Option<u32>) {
        self.events.push(Event {
            seq: self.events.len() as u32,
            kind: "answer",
            node,
            source: None,
            hop: self.hop.get(&node).copied().unwrap_or(0),
            d: Some(d),
            key: None,
            m: Some(true),
            g: None,
            outcome: "collected",
            rank: None,
            evicted,
            expanded_at: None,
        });
    }

    /// Records a second-hop candidate (kept at `rank`, or cut by the truncation).
    pub fn two_hop(&mut self, node: u32, d: f32, rank: usize, kept: bool) {
        self.events.push(Event {
            seq: self.events.len() as u32,
            kind: if kept { "two_hop" } else { "two_hop_cut" },
            node,
            source: None,
            hop: self.hop.get(&node).copied().unwrap_or(0),
            d: Some(d),
            key: None,
            m: Some(false),
            g: None,
            outcome: if kept { "expanded" } else { "truncated" },
            rank: Some(rank),
            evicted: None,
            expanded_at: Some(self.step),
        });
    }

    /// Writes the events as a TSV table.
    pub fn write_tsv<W: Write>(&self, mut w: W) -> std::io::Result<()> {
        writeln!(
            w,
            "seq\tkind\tnode\tsource\thop\td\tkey\tmatch\tg\toutcome\tinserted_rank\tevicted_node\texpanded_at"
        )?;
        fn opt<T: ToString>(x: &Option<T>) -> String {
            x.as_ref().map_or_else(String::new, ToString::to_string)
        }
        for e in &self.events {
            writeln!(
                w,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                e.seq,
                e.kind,
                e.node,
                opt(&e.source),
                e.hop,
                opt(&e.d),
                opt(&e.key),
                opt(&e.m.map(u8::from)),
                opt(&e.g),
                e.outcome,
                opt(&e.rank),
                opt(&e.evicted),
                opt(&e.expanded_at),
            )?;
        }
        Ok(())
    }
}
