/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{error::StandardError, utils::VectorId};

use crate::attribute::{Attribute, AttributeType};

pub trait AttributeStore<IdType>: Send + Sync
where
    IdType: VectorId,
{
    type AT: AttributeType;
    type StoreError: StandardError;

    /// Delete the attributes of a vector represented by the vec_id from the store.
    /// Returns true if the vector was deleted, false if it didn't exist.
    /// Returns ANNError if the operation failed
    fn delete(&self, vec_id: &IdType) -> Result<bool, Self::StoreError>;

    /// Check if a vector ID exists in the store. Returns true if it exists,
    /// false if it doesn't. It doesn't matter if the vector id has no labels
    /// associated with it.
    /// Returns ANNError if the operation failed
    fn id_exists(&self, vec_id: &IdType) -> Result<bool, Self::StoreError>;

    /// Set the attributes for a vector, replacing any existing attributes
    /// Returns true if the vec_id was inserted, false if it exists and
    /// was overwritten. For most clients, this distinction is unnecessary.
    ///
    /// Returns ANNError if the operation failed and the vector was not
    /// inserted.
    fn set_element(
        &self,
        vec_id: &IdType,
        attributes: &[Attribute],
    ) -> Result<bool, Self::StoreError>;
}
