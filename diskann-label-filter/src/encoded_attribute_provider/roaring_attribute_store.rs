/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{
    attribute::Attribute,
    encoded_attribute_provider::{
        attribute_encoder::AttributeEncoder, encoded_attribute_accessor::EncodedAttributeAccessor,
        encoded_filter_expr::EncodedFilterExpr,
    },
    inline_beta_search::predicate_evaluator::PredicateEvaluator,
    set::{roaring_set_provider::RoaringTreemapSetProvider, SetProvider},
    traits::attribute_store::AttributeStore,
};
use diskann::{utils::VectorId, ANNError, ANNErrorKind, ANNResult};
use diskann_utils::future::AsyncFriendly;
use std::sync::{Arc, RwLock};

pub struct RoaringAttributeStore<IT>
where
    IT: VectorId + AsyncFriendly,
{
    attribute_map: Arc<RwLock<AttributeEncoder>>,
    index: Arc<RwLock<RoaringTreemapSetProvider<IT>>>,
    inv_index: Arc<RwLock<RoaringTreemapSetProvider<u64>>>,
}

impl<IT> RoaringAttributeStore<IT>
where
    IT: VectorId,
{
    #[allow(
        dead_code,
        reason = "This will be invoked by callers when they create a document provider."
    )]
    pub fn new() -> Self {
        Self {
            attribute_map: Arc::new(RwLock::new(AttributeEncoder::new())),
            index: Arc::new(RwLock::new(RoaringTreemapSetProvider::<IT>::new())),
            inv_index: Arc::new(RwLock::new(RoaringTreemapSetProvider::<u64>::new())),
        }
    }

    #[cfg(test)]
    pub fn get_index(&self) -> Arc<RwLock<RoaringTreemapSetProvider<IT>>> {
        self.index.clone()
    }

    pub fn attribute_map(&self) -> Arc<RwLock<AttributeEncoder>> {
        self.attribute_map.clone()
    }

    /// Check if a point's encoded attributes satisfy the given encoded filter.
    /// Returns `true` if the point matches, `false` if it doesn't match or has no attributes.
    /// This performs an efficient roaring bitmap lookup + integer predicate evaluation.
    pub fn matches_filter(&self, vec_id: &IT, filter: &EncodedFilterExpr) -> bool {
        let index = self.index.read().unwrap_or_else(|e| e.into_inner());
        match index.get(vec_id) {
            Ok(Some(set)) => {
                let evaluator = PredicateEvaluator::new(set.as_ref());
                filter
                    .encoded_filter_expr()
                    .accept(&evaluator)
                    .unwrap_or(false)
            }
            _ => false,
        }
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
    /// a per-attr_id posting list while building per-doc treemaps inline. At
    /// the end, each per-attr_id bucket is consumed in one shot via
    /// `RoaringTreemap::from_sorted_iter`. This avoids the per-pair
    /// `RoaringTreemap::insert` overhead that dominates the naive loop.
    ///
    /// Transient memory cost is `O(num_attribute_pairs)` u64s for the inverse
    /// buckets; the final structures replace the empty placeholders held by
    /// the store. All three internal write locks are taken once for the
    /// duration of the load.
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

        // Single pass: for each doc, build the forward treemap immediately and
        // append the doc_id to each touched attr_id's bucket.
        let mut sort_buf: Vec<u64> = Vec::with_capacity(16);
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

            // Forward: build a sorted RoaringTreemap of attr_ids for this doc.
            sort_buf.clear();
            sort_buf.extend_from_slice(&attr_ids);
            sort_buf.sort_unstable();
            let tm = RoaringTreemap::from_sorted_iter(sort_buf.iter().copied()).map_err(|e| {
                ANNError::message(
                    ANNErrorKind::Opaque,
                    format!("Forward treemap construction failed: {}", e),
                )
            })?;
            index.install_for_key(doc_id, tm);
        }

        // Build each inverse-index treemap from its (already-sorted) bucket.
        for (attr_id, doc_ids) in inv_buckets.into_iter().enumerate() {
            if doc_ids.is_empty() {
                continue;
            }
            let tm = RoaringTreemap::from_sorted_iter(doc_ids.iter().copied()).map_err(|e| {
                ANNError::message(
                    ANNErrorKind::Opaque,
                    format!(
                        "Inverse treemap for attr_id {} failed: {}",
                        attr_id, e
                    ),
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
    type Accessor = EncodedAttributeAccessor<RoaringTreemapSetProvider<IT>>;
    type StoreError = ANNError;

    fn attribute_accessor(&self) -> Result<Self::Accessor, Self::StoreError> {
        Ok(EncodedAttributeAccessor::new(self.index.clone()))
    }

    /// Delete the attributes of a vector represented by the vec_id from the store.
    /// Returns "Result" because we may make this a trait going forward, so even
    /// though this implementation will simply return Ok().
    ///
    fn delete(&self, vec_id: &IT) -> ANNResult<bool>
    where
        IT: VectorId,
    {
        let vec_id_u64 = (*vec_id).into();
        let mut deleted = true;

        // Acquire locks in consistent order: index first, then inv_index
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

        let existing_set = match index_guard.get(vec_id)? {
            Some(set) => set,
            None => {
                return Ok(false);
            } //we are in good shape even if the vector id doesn't exist.
        };

        // At this point we have already checked that the id exists in the index.
        // Therefore any failures in delete_from_set() or delete() are logical errors.
        // So we will flag them as such.

        // delete the id from the inverted index.
        for attr_id in existing_set.iter() {
            deleted = deleted && inv_index_guard.delete_from_set(&attr_id, &vec_id_u64)?;
        }
        if !deleted {
            return Err(ANNError::message(
                ANNErrorKind::IndexError,
                "Failed to delete id from the inverted index.",
            ));
        }

        //delete the id from the index.
        deleted = index_guard.delete(vec_id)?; //we know deleted is true so far.
        if deleted {
            Ok(true)
        } else {
            Err(ANNError::message(
                ANNErrorKind::IndexError,
                "Failed to delete id from the index.",
            ))
        }
    }

    fn id_exists(&self, vec_id: &IT) -> ANNResult<bool> {
        let index_guard = self.index.read().map_err(|_| {
            ANNError::message(
                ANNErrorKind::LockPoisonError,
                "Failed to acquire read lock on the label index.",
            )
        })?;
        index_guard.exists(vec_id)
    }

    fn set_element(&self, vec_id: &IT, attributes: &[Attribute]) -> ANNResult<bool>
    where
        IT: VectorId,
    {
        let id_u64: u64 = (*vec_id).into();

        //For now, we assume that it is an error if a point has zero attributes.
        if attributes.is_empty() {
            return Err(ANNError::message(
                ANNErrorKind::Opaque,
                "A vector must have atleast one attribute.",
            ));
        }

        // Acquire locks in consistent order: attribute_map, index, inv_index
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

        // Update the inverted index.
        // Delete all instances of id from the inv_index for the old labels.
        if let Some(set) = index_guard.get(vec_id)? {
            for attr_id in set.iter() {
                //delete_from_set() returns false if the attr_id or id_u64 don't exist. It
                //doesn't make a difference, so we ignore the return value.
                let _ = inv_index_guard.delete_from_set(&attr_id, &id_u64)?;
            }
        };

        // Delete existing entries in the label index
        index_guard.delete(vec_id)?; //returns false if vec_id doesn't exist, but that don't matter to us.

        // Insert entries for the new attributes in the inv_index and index
        for attr in attributes {
            let attr_id = attr_map_guard.insert(attr);
            inv_index_guard.insert(&attr_id, &id_u64)?;
            index_guard.insert(vec_id, &attr_id)?;
        }
        Ok(true)
    }
}
