/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! JSON-lines formats for base labels and query predicates (one object per line).
//!
//! **Base labels**: `{"doc_id": 0, "color_red": true, "size_m": true, ...}`. `doc_id` must equal
//! the line index (from 0). Every other key with value `true` is a label; `false` and `null`
//! are skipped and other values are errors. The key `prefix_rest` belongs to category `prefix`
//! (a key without `_` is its own category).
//!
//! **Predicates**: `{"query_id": 0, "filter": F}` with `query_id` equal to the line index, where
//! `F` is `{"$and": [C, ...]}` or a single `C`, `C` is `{"$or": [A, ...]}` or a single `A`, and
//! `A` is `{"<label>": {"$eq": true}}`.

use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;

use crate::{
    labels::{LabelStore, LabelStoreBuilder},
    predicate::PredicateSetBuilder,
    text::for_each_line,
};

/// Category of a JSON label key: the part before the first `_`, or the whole key.
pub fn category_of(key: &str) -> &str {
    key.split_once('_').map_or(key, |(category, _)| category)
}

fn check_id(object: &Value, key: &str, line: usize) -> Result<()> {
    let id = object
        .get(key)
        .and_then(Value::as_u64)
        .with_context(|| format!("missing or invalid {key:?}"))?;
    ensure!(
        id == line as u64,
        "{key} {id} does not match the line index {line}"
    );
    Ok(())
}

/// Labels of one base line.
pub fn parse_base_line(line: &str, index: usize) -> Result<Vec<String>> {
    let value: Value = serde_json::from_str(line)?;
    check_id(&value, "doc_id", index)?;
    let mut labels = Vec::new();
    for (key, v) in value.as_object().context("expected a JSON object")? {
        match v {
            _ if key == "doc_id" => {}
            Value::Bool(true) => labels.push(key.clone()),
            Value::Bool(false) | Value::Null => {}
            other => bail!("unsupported value {other} for {key:?} (only Boolean labels)"),
        }
    }
    Ok(labels)
}

/// Clauses of one predicate line, as label names.
pub fn parse_predicate_line(line: &str, index: usize) -> Result<Vec<Vec<String>>> {
    let value: Value = serde_json::from_str(line)?;
    check_id(&value, "query_id", index)?;
    let filter = value.get("filter").context("missing \"filter\"")?;
    match filter.get("$and") {
        Some(Value::Array(clauses)) => clauses.iter().map(parse_clause).collect(),
        Some(other) => bail!("\"$and\" must be an array, found {other}"),
        None => Ok(vec![parse_clause(filter)?]),
    }
}

fn parse_clause(value: &Value) -> Result<Vec<String>> {
    match value.get("$or") {
        Some(Value::Array(atoms)) => {
            ensure!(!atoms.is_empty(), "empty \"$or\"");
            atoms.iter().map(parse_atom).collect()
        }
        Some(other) => bail!("\"$or\" must be an array, found {other}"),
        None => Ok(vec![parse_atom(value)?]),
    }
}

fn parse_atom(value: &Value) -> Result<String> {
    let object = value
        .as_object()
        .filter(|o| o.len() == 1)
        .with_context(|| format!("expected {{\"<label>\": {{\"$eq\": true}}}}, found {value}"))?;
    let (key, condition) = object.iter().next().context("empty atom")?;
    let is_eq_true = condition
        .as_object()
        .is_some_and(|c| c.len() == 1 && c.get("$eq") == Some(&Value::Bool(true)));
    ensure!(
        !key.starts_with('$') && is_eq_true,
        "unsupported condition {value} (only {{\"<label>\": {{\"$eq\": true}}}})"
    );
    Ok(key.clone())
}

/// Reads a JSONL base label file.
pub fn read_base_labels(path: &Path) -> Result<LabelStore> {
    let mut builder = LabelStoreBuilder::new();
    for_each_line(path, |i, line| {
        let labels = parse_base_line(line, i)?;
        builder.push_point(labels.iter().map(|l| (category_of(l), l.as_str())))
    })?;
    builder.finish()
}

/// Compiles every line of a JSONL predicate file into `builder`; returns the number of lines.
pub fn push_predicates(path: &Path, builder: &mut PredicateSetBuilder<'_>) -> Result<usize> {
    for_each_line(path, |i, line| {
        builder.push(&parse_predicate_line(line, i)?);
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_line() {
        let mut labels = parse_base_line(
            r#"{"doc_id": 3, "size_m": true, "color_red": true, "x": false}"#,
            3,
        )
        .unwrap();
        labels.sort();
        assert_eq!(labels, ["color_red", "size_m"]);
        assert_eq!(category_of("shape_x-y"), "shape");
        assert!(parse_base_line(r#"{"doc_id": 2, "size_m": true}"#, 3).is_err());
        assert!(parse_base_line(r#"{"doc_id": 3, "geo": "32"}"#, 3).is_err());
    }

    #[test]
    fn predicate_line() {
        let line = r#"{"query_id": 0, "filter": {"$and": [{"$or": [{"color_red": {"$eq": true}}, {"color_blue": {"$eq": true}}]}, {"$or": [{"size_m": {"$eq": true}}]}, {"shape_x": {"$eq": true}}]}}"#;
        assert_eq!(
            parse_predicate_line(line, 0).unwrap(),
            vec![
                vec!["color_red", "color_blue"],
                vec!["size_m"],
                vec!["shape_x"]
            ]
        );
        let single = r#"{"query_id": 1, "filter": {"$or": [{"a": {"$eq": true}}]}}"#;
        assert_eq!(parse_predicate_line(single, 1).unwrap(), vec![vec!["a"]]);
        let negated = r#"{"query_id": 0, "filter": {"$not": {"a": {"$eq": true}}}}"#;
        assert!(parse_predicate_line(negated, 0).is_err());
        let not_true = r#"{"query_id": 0, "filter": {"a": {"$eq": false}}}"#;
        assert!(parse_predicate_line(not_true, 0).is_err());
    }
}
