/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Text formats for base labels and query predicates.
//!
//! **Base labels**: one line per point with comma-separated labels, e.g.
//! `color=red,size=s,size=m` or `12,58,301`. A label `name=value` belongs to category
//! `name`; labels without `=` belong to a single unnamed category. Only the part of a line
//! before the first tab is read, surrounding whitespace is ignored, and an empty line is a
//! point without labels.
//!
//! **Predicates**: one line per query in conjunctive normal form, clauses separated by `&` and
//! alternatives within a clause by `|`, e.g. `color=red|color=blue&size=m` or `12&58`. An empty
//! line is the predicate without clauses (matches every point); empty clauses or alternatives
//! are errors.

use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use anyhow::{ensure, Context, Result};

use crate::{
    labels::{LabelStore, LabelStoreBuilder},
    predicate::PredicateSetBuilder,
};

/// Calls `f(line_index, line)` for every line of a UTF-8 file (line terminators removed) and
/// returns the number of lines.
pub fn for_each_line(path: &Path, mut f: impl FnMut(usize, &str) -> Result<()>) -> Result<usize> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = BufReader::with_capacity(1 << 22, file);
    let mut line = String::new();
    let mut n = 0;
    loop {
        line.clear();
        let read = reader
            .read_line(&mut line)
            .with_context(|| format!("{}: line {}", path.display(), n + 1))?;
        if read == 0 {
            return Ok(n);
        }
        let text = line.strip_suffix('\n').unwrap_or(&line);
        let text = text.strip_suffix('\r').unwrap_or(text);
        f(n, text).with_context(|| format!("{}: line {}", path.display(), n + 1))?;
        n += 1;
    }
}

/// Category of a text label: the part before the first `=`, or `""` when there is none.
pub fn category_of(label: &str) -> &str {
    label.split_once('=').map_or("", |(category, _)| category)
}

fn first_field(line: &str) -> &str {
    line.split('\t').next().unwrap_or_default()
}

/// Labels of one base line.
pub fn parse_base_line(line: &str) -> impl Iterator<Item = &str> {
    first_field(line)
        .split(',')
        .map(str::trim)
        .filter(|label| !label.is_empty())
}

/// Clauses of one predicate line, as label names.
pub fn parse_predicate_line(line: &str) -> Result<Vec<Vec<&str>>> {
    let field = first_field(line).trim();
    if field.is_empty() {
        return Ok(Vec::new());
    }
    field
        .split('&')
        .map(|clause| {
            clause
                .split('|')
                .map(|label| {
                    let label = label.trim();
                    ensure!(!label.is_empty(), "empty label or clause in {field:?}");
                    Ok(label)
                })
                .collect()
        })
        .collect()
}

/// Reads a base label file.
pub fn read_base_labels(path: &Path) -> Result<LabelStore> {
    let mut builder = LabelStoreBuilder::new();
    for_each_line(path, |_, line| {
        builder.push_point(parse_base_line(line).map(|label| (category_of(label), label)))
    })?;
    builder.finish()
}

/// Compiles every line of a predicate file into `builder`; returns the number of lines.
pub fn push_predicates(path: &Path, builder: &mut PredicateSetBuilder<'_>) -> Result<usize> {
    for_each_line(path, |_, line| {
        builder.push(&parse_predicate_line(line)?);
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_lines() {
        let labels: Vec<&str> = parse_base_line(" color=red,size=s,,shape=x-y \tignored").collect();
        assert_eq!(labels, ["color=red", "size=s", "shape=x-y"]);
        assert_eq!(
            labels.iter().map(|l| category_of(l)).collect::<Vec<_>>(),
            ["color", "size", "shape"]
        );
        assert_eq!(parse_base_line("").count(), 0);
        assert_eq!(category_of("3626"), "");
    }

    #[test]
    fn predicate_lines() {
        assert_eq!(
            parse_predicate_line("color=red|color=blue&size=m&shape=x").unwrap(),
            vec![
                vec!["color=red", "color=blue"],
                vec!["size=m"],
                vec!["shape=x"]
            ]
        );
        assert_eq!(
            parse_predicate_line("12&58").unwrap(),
            vec![vec!["12"], vec!["58"]]
        );
        assert!(parse_predicate_line("").unwrap().is_empty());
        assert!(parse_predicate_line("a&&b").is_err());
        assert!(parse_predicate_line("a|&b").is_err());
    }
}
