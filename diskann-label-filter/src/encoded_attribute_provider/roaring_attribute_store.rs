/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Attribute store backed by:
//!   * a **compact forward index** (`Vec<Box<[u32]>>` indexed by vec_id) for
//!     per-point label lookups in the search hot path, and
//!   * a roaring-bitmap **inverse index** (label_id -> set of doc_ids) for
//!     bulk set operations used by bitmap-based filter strategies.
//!
//! The forward index intentionally does not use roaring bitmaps: per-point
//! label sets are tiny (typically <= 20 labels), and a sorted `Box<[u32]>` is
//! much faster to probe (one cache line, no `BTreeMap`/`HashMap` indirection)
//! than a per-point `RoaringTreemap`. See `matches_filter` for the hot path.
//!
//! The inverse index keeps roaring because per-label posting lists can be
//! large and dense, where roaring's compression + fast set ops pay off.

use crate::{
    attribute::Attribute,
    encoded_attribute_provider::{
        ast_id_expr::{ASTIdExpr, ASTIdExprVisitor},
        attribute_encoder::AttributeEncoder,
        encoded_filter_expr::EncodedFilterExpr,
    },
    set::{roaring_set_provider::RoaringTreemapSetProvider, SetProvider},
    traits::attribute_store::AttributeStore,
};
use diskann::{utils::VectorId, ANNError, ANNErrorKind, ANNResult};
use diskann_utils::future::AsyncFriendly;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

pub struct RoaringAttributeStore<IT>
where
    IT: VectorId + AsyncFriendly,
{
    attribute_map: Arc<RwLock<AttributeEncoder>>,
    /// Compact forward index: per-point sorted slice of u32 attr_ids,
    /// indexed directly by `(*vec_id).into() as usize`. Built by
    /// `bulk_insert_encoded` and kept in sync by `set_element` / `delete`.
    index: Arc<RwLock<Vec<Box<[u32]>>>>,
    /// Inverse index: attr_id -> set of doc_ids. Roaring is well-suited
    /// here because individual posting lists can be large.
    inv_index: Arc<RwLock<RoaringTreemapSetProvider<u64>>>,
    _phantom: PhantomData<IT>,
}

impl<IT> RoaringAttributeStore<IT>
where
    IT: VectorId,
{
    pub fn new() -> Self {
        Self {
            attribute_map: Arc::new(RwLock::new(AttributeEncoder::new())),
            index: Arc::new(RwLock::new(Vec::new())),
            inv_index: Arc::new(RwLock::new(RoaringTreemapSetProvider::<u64>::new())),
            _phantom: PhantomData,
        }
    }

    pub fn attribute_map(&self) -> Arc<RwLock<AttributeEncoder>> {
        self.attribute_map.clone()
    }

    /// Number of points carrying `label_id` (i.e. the size of the inverse-index
    /// posting list for that label). Returns 0 if the label is absent. O(1)
    /// (well, O(number of roaring containers), which is small in practice).
    pub fn posting_list_len(&self, label_id: u64) -> usize {
        let inv = self.inv_index.read().unwrap_or_else(|e| e.into_inner());
        match inv.get(&label_id) {
            Ok(Some(set)) => set.len() as usize,
            _ => 0,
        }
    }

    /// Union of the inverse-index posting lists for the given label_ids,
    /// materialised as a `RoaringTreemap` of u64 vec_ids.
    ///
    /// Used by the inline-beta brute-force hybrids to materialise the candidate
    /// set for a category (V1) or the rare-label set (V2) without having to
    /// scan the whole base.
    pub fn union_posting_lists(&self, label_ids: &[u64]) -> roaring::RoaringTreemap {
        use roaring::RoaringTreemap;
        let inv = self.inv_index.read().unwrap_or_else(|e| e.into_inner());
        let mut out = RoaringTreemap::new();
        for &lid in label_ids {
            if let Ok(Some(set)) = inv.get(&lid) {
                out |= set.as_ref();
            }
        }
        out
    }

    /// Check if a point's encoded attributes satisfy the given encoded filter.
    /// Returns `true` if the point matches, `false` if `vec_id` is out of
    /// range, has no attributes, or the filter does not match.
    ///
    /// Fast path: direct `Vec` index into the compact forward index, then a
    /// tight `SlicePredicateEvaluator` walk over the AST. Each terminal does
    /// one `contains` probe (linear scan for small sets, binary search above
    /// `SlicePredicateEvaluator::LINEAR_SCAN_THRESHOLD`).
    pub fn matches_filter(&self, vec_id: &IT, filter: &EncodedFilterExpr) -> bool {
        let idx = self.index.read().unwrap_or_else(|e| e.into_inner());
        let i = (*vec_id).into() as usize;
        if i >= idx.len() {
            return false;
        }
        let slice: &[u32] = &idx[i];
        if slice.is_empty() {
            return false;
        }
        let evaluator = SlicePredicateEvaluator::new(slice);
        filter
            .encoded_filter_expr()
            .accept(&evaluator)
            .unwrap_or(false)
    }

    /// Bulk-load pre-encoded attributes in one shot. Intended for benchmark
    /// harnesses that have already extracted attribute strings + per-doc
    /// attr_id lists from a label file.
    ///
    /// `attribute_strings` lists the attribute objects in encoder-id order:
    /// `attribute_strings[i]` becomes encoder id `i`. The store must be empty
    /// when this is called (the caller has just `new()`'d it).
    ///
    /// `docs_iter` yields `(vec_id, attr_ids)` pairs where each `attr_id` is
    /// an index into `attribute_strings`. Docs with no attributes are silently
    /// skipped. Documents are expected to arrive in ascending `vec_id` order;
    /// per-doc attr_id lists need not be sorted.
    ///
    /// Build strategy: stream once, bucketing each (vec_id, attr_id) pair into
    /// a per-attr_id posting list while building per-doc compact slices
    /// inline. At the end, each per-attr_id bucket is consumed in one shot via
    /// `RoaringTreemap::from_sorted_iter`.
    pub fn bulk_insert_encoded<I>(
        &self,
        attribute_strings: &[Attribute],
        docs_iter: I,
    ) -> ANNResult<()>
    where
        I: IntoIterator<Item = (IT, Vec<u64>)>,
    {
        use roaring::RoaringTreemap;

        let mut attr_map = self.attribute_map.write().map_err(|_| {
            ANNError::message(
                ANNErrorKind::LockPoisonError,
                "Failed to acquire write lock on attribute_map",
            )
        })?;
        let mut index = self.index.write().map_err(|_| {
            ANNError::message(
                ANNErrorKind::LockPoisonError,
                "Failed to acquire write lock on index",
            )
        })?;
        let mut inv_index = self.inv_index.write().map_err(|_| {
            ANNError::message(
                ANNErrorKind::LockPoisonError,
                "Failed to acquire write lock on inv_index",
            )
        })?;

        // Populate the encoder with all attribute strings; the encoder assigns
        // ids in insertion order starting at 0, which must match the caller's
        // attr_id convention.
        for (expected_id, attr) in attribute_strings.iter().enumerate() {
            let id = attr_map.insert(attr);
            debug_assert_eq!(id, expected_id as u64);
        }

        let num_attrs = attribute_strings.len();
        // Inverse-index buckets: posting list of doc_ids per attr_id.
        let mut inv_buckets: Vec<Vec<u64>> = (0..num_attrs).map(|_| Vec::new()).collect();

        // Single pass: for each doc, build the forward compact slice
        // immediately and append the doc_id to each touched attr_id's bucket.
        let mut sort_buf: Vec<u32> = Vec::with_capacity(16);
        for (doc_id, attr_ids) in docs_iter {
            if attr_ids.is_empty() {
                continue;
            }
            let doc_id_u64: u64 = doc_id.into();

            // Inverse: push doc_id onto each attr's bucket. Since docs arrive
            // in ascending vec_id order, each bucket ends up sorted.
            for &aid in &attr_ids {
                debug_assert!((aid as usize) < num_attrs);
                inv_buckets[aid as usize].push(doc_id_u64);
            }

            // Forward (compact): build a sorted Box<[u32]> of attr_ids.
            sort_buf.clear();
            sort_buf.extend(attr_ids.iter().map(|&v| v as u32));
            sort_buf.sort_unstable();
            let slot = doc_id_u64 as usize;
            if slot >= index.len() {
                index.resize(slot + 1, Box::<[u32]>::default());
            }
            index[slot] = sort_buf.as_slice().into();
        }

        // Build each inverse-index treemap from its (already-sorted) bucket.
        for (attr_id, doc_ids) in inv_buckets.into_iter().enumerate() {
            if doc_ids.is_empty() {
                continue;
            }
            let tm = RoaringTreemap::from_sorted_iter(doc_ids.iter().copied()).map_err(|e| {
                ANNError::message(
                    ANNErrorKind::Opaque,
                    format!("Inverse treemap for attr_id {} failed: {}", attr_id, e),
                )
            })?;
            inv_index.install_for_key(attr_id as u64, tm);
        }

        Ok(())
    }
}

impl<IT> AttributeStore<IT> for RoaringAttributeStore<IT>
where
    IT: VectorId,
{
    type AT = u64;
    type StoreError = ANNError;

    /// Delete the attributes of a vector from the store.
    fn delete(&self, vec_id: &IT) -> ANNResult<bool>
    where
        IT: VectorId,
    {
        let vec_id_u64: u64 = (*vec_id).into();

        // Acquire locks in consistent order: index first, then inv_index.
        let mut index_guard = self.index.write().map_err(|_| {
            ANNError::message(
                ANNErrorKind::LockPoisonError,
                "Failed to acquire write lock on index",
            )
        })?;
        let mut inv_index_guard = self.inv_index.write().map_err(|_| {
            ANNError::message(
                ANNErrorKind::LockPoisonError,
                "Failed to acquire write lock on inv_index",
            )
        })?;

        let slot = vec_id_u64 as usize;
        if slot >= index_guard.len() || index_guard[slot].is_empty() {
            return Ok(false);
        }

        // Remove the id from each per-attr posting list in the inverse index.
        for &attr_id in index_guard[slot].iter() {
            // Ignore the "value not present" return: the doc may already
            // have been removed from a bucket if state was inconsistent.
            let _ = inv_index_guard.delete_from_set(&(attr_id as u64), &vec_id_u64)?;
        }

        // Clear the forward slot.
        index_guard[slot] = Box::<[u32]>::default();
        Ok(true)
    }

    fn id_exists(&self, vec_id: &IT) -> ANNResult<bool> {
        let index_guard = self.index.read().map_err(|_| {
            ANNError::message(
                ANNErrorKind::LockPoisonError,
                "Failed to acquire read lock on the label index.",
            )
        })?;
        let slot = (*vec_id).into() as usize;
        Ok(slot < index_guard.len() && !index_guard[slot].is_empty())
    }

    fn set_element(&self, vec_id: &IT, attributes: &[Attribute]) -> ANNResult<bool>
    where
        IT: VectorId,
    {
        let id_u64: u64 = (*vec_id).into();

        // For now, we assume that it is an error if a point has zero attributes.
        if attributes.is_empty() {
            return Err(ANNError::message(
                ANNErrorKind::Opaque,
                "A vector must have atleast one attribute.",
            ));
        }

        // Acquire locks in consistent order: attribute_map, index, inv_index.
        let mut attr_map_guard = self.attribute_map.write().map_err(|_| {
            ANNError::message(
                ANNErrorKind::LockPoisonError,
                "Failed to acquire write lock on attribute_map",
            )
        })?;
        let mut index_guard = self.index.write().map_err(|_| {
            ANNError::message(
                ANNErrorKind::LockPoisonError,
                "Failed to acquire write lock on index",
            )
        })?;
        let mut inv_index_guard = self.inv_index.write().map_err(|_| {
            ANNError::message(
                ANNErrorKind::LockPoisonError,
                "Failed to acquire write lock on inv_index",
            )
        })?;

        // Decrement inverse index for any previously-set attrs on this id.
        let slot = id_u64 as usize;
        if slot < index_guard.len() {
            for &attr_id in index_guard[slot].iter() {
                let _ = inv_index_guard.delete_from_set(&(attr_id as u64), &id_u64)?;
            }
        }

        // Encode new attrs, insert into inverse index, and build sorted u32 slice.
        let mut new_attr_ids: Vec<u32> = Vec::with_capacity(attributes.len());
        for attr in attributes {
            let attr_id = attr_map_guard.insert(attr);
            inv_index_guard.insert(&attr_id, &id_u64)?;
            new_attr_ids.push(attr_id as u32);
        }
        new_attr_ids.sort_unstable();

        if slot >= index_guard.len() {
            index_guard.resize(slot + 1, Box::<[u32]>::default());
        }
        index_guard[slot] = new_attr_ids.as_slice().into();

        Ok(true)
    }
}

/// Predicate evaluator that operates on a **sorted** `&[u32]` of attr_ids
/// (the compact forward-index representation).
///
/// `contains` uses a tight linear scan for small slices (cache-friendly,
/// branch-predictor-friendly, autovectorizable) and switches to binary
/// search above a small threshold. Threshold chosen so that linear scan
/// stays within roughly one cache line of work.
struct SlicePredicateEvaluator<'a> {
    labels: &'a [u32],
}

impl<'a> SlicePredicateEvaluator<'a> {
    const LINEAR_SCAN_THRESHOLD: usize = 32;

    fn new(labels: &'a [u32]) -> Self {
        Self { labels }
    }

    #[inline(always)]
    fn contains_id(&self, id: u32) -> bool {
        let s = self.labels;
        if s.len() <= Self::LINEAR_SCAN_THRESHOLD {
            for &v in s {
                if v == id {
                    return true;
                }
            }
            false
        } else {
            s.binary_search(&id).is_ok()
        }
    }
}

impl<'a> ASTIdExprVisitor<u64> for SlicePredicateEvaluator<'a> {
    type Output = ANNResult<bool>;

    fn visit_and(&self, exprs: &[ASTIdExpr<u64>]) -> Self::Output {
        for expr in exprs {
            if !self.visit(expr)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn visit_or(&self, exprs: &[ASTIdExpr<u64>]) -> Self::Output {
        for expr in exprs {
            if self.visit(expr)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn visit_not(&self, expr: &ASTIdExpr<u64>) -> Self::Output {
        Ok(!self.visit(expr)?)
    }

    fn visit_terminal(&self, label_id: &u64) -> Self::Output {
        // attr_ids are assigned sequentially from 0 by the encoder and fit in
        // u32 in practice. Out-of-range ids cannot be present in the slice.
        if *label_id > u32::MAX as u64 {
            return Ok(false);
        }
        Ok(self.contains_id(*label_id as u32))
    }
}
