/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Building blocks for filtered approximate nearest neighbor experiments.
//!
//! Points carry sets of Boolean labels grouped into anonymous categories; queries carry a
//! vector and a predicate in conjunctive normal form (an AND of ORs over labels). The crate
//! provides compact stores for vectors, labels and predicates, their binary file formats,
//! converters from text formats, and exact filtered ground truth with complete distance-tie
//! bands. File formats and the `flab` command-line tool are described in `README.md`.

pub mod bitsets;
pub mod build;
pub mod calibrate;
pub mod container;
pub mod diag;
pub mod diskann_io;
pub mod graph;
pub mod gt;
pub mod hugepages;
pub mod jsonl;
pub mod kernel;
pub mod label_starts;
pub mod labels;
pub mod parallel;
pub mod permute;
pub mod predicate;
pub mod run;
pub mod sketches;
pub mod starts;
pub mod text;
pub mod upstream;
pub mod vectors;
