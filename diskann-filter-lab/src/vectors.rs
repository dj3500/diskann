/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Dense vectors in the DiskANN `.bin` layout: `u32` row count `n`, `u32` dimension `d`, then
//! `n × d` elements in row-major order (little-endian). Rows are stored contiguously in
//! 64-byte-aligned memory.

use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    marker::PhantomData,
    path::Path,
};

use anyhow::{bail, ensure, Context, Result};
use bytemuck::{Pod, Zeroable};
use diskann_vector::{distance::SquaredL2, PureDistanceFunction};

/// Vector element types supported by the lab.
pub trait Element: Pod + Send + Sync {
    /// Short type name used in messages.
    const NAME: &'static str;
    /// The element type as a value.
    const DTYPE: DType;

    /// Squared Euclidean distance computed with `diskann-vector`'s SIMD kernels.
    ///
    /// For `u8` the sum is accumulated in integers, so the result is exact as long as it is
    /// below 2^24 (any dimension up to 258).
    fn squared_l2(a: &[Self], b: &[Self]) -> f32;
}

impl Element for u8 {
    const NAME: &'static str = "u8";
    const DTYPE: DType = DType::U8;

    #[inline(always)]
    fn squared_l2(a: &[u8], b: &[u8]) -> f32 {
        <SquaredL2 as PureDistanceFunction<&[u8], &[u8], f32>>::evaluate(a, b)
    }
}

impl Element for f32 {
    const NAME: &'static str = "f32";
    const DTYPE: DType = DType::F32;

    #[inline(always)]
    fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
        <SquaredL2 as PureDistanceFunction<&[f32], &[f32], f32>>::evaluate(a, b)
    }
}

/// Element type of a vector file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    U8,
    F32,
}

impl DType {
    /// Size of one element in bytes.
    pub fn size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::F32 => 4,
        }
    }

    /// Short type name used in messages.
    pub fn name(self) -> &'static str {
        match self {
            Self::U8 => u8::NAME,
            Self::F32 => f32::NAME,
        }
    }

    /// Infers the element type from the file extension (`.u8bin` or `.fbin`).
    pub fn from_path(path: &Path) -> Result<Self> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("u8bin") => Ok(Self::U8),
            Some("fbin") => Ok(Self::F32),
            _ => bail!(
                "cannot infer the element type of {} (expected .u8bin or .fbin); specify it explicitly",
                path.display()
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C, align(64))]
struct CacheLine([u8; 64]);

/// Row-major matrix in 64-byte-aligned contiguous storage.
#[derive(Debug, Clone)]
pub struct Matrix<T> {
    lines: Vec<CacheLine>,
    nrows: usize,
    ncols: usize,
    _element: PhantomData<T>,
}

impl<T: Element> Matrix<T> {
    /// A zero-filled `nrows × ncols` matrix.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        let bytes = nrows * ncols * std::mem::size_of::<T>();
        Self {
            lines: crate::hugepages::filled(bytes.div_ceil(64), CacheLine::zeroed()),
            nrows,
            ncols,
            _element: PhantomData,
        }
    }

    /// Builds a matrix from row-major data.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != nrows * ncols`.
    pub fn from_slice(data: &[T], nrows: usize, ncols: usize) -> Self {
        assert_eq!(
            data.len(),
            nrows * ncols,
            "data length must equal nrows * ncols"
        );
        let mut m = Self::zeros(nrows, ncols);
        m.as_mut_slice().copy_from_slice(data);
        m
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// All elements, row-major; starts at a 64-byte boundary.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &bytemuck::cast_slice(&self.lines)[..self.nrows * self.ncols]
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut bytemuck::cast_slice_mut(&mut self.lines)[..self.nrows * self.ncols]
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[T] {
        &self.as_slice()[i * self.ncols..(i + 1) * self.ncols]
    }

    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [T] {
        let d = self.ncols;
        &mut self.as_mut_slice()[i * d..(i + 1) * d]
    }

    /// Reads a `.bin` file; its size must match the header exactly.
    pub fn read_bin(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let file_len = file.metadata()?.len();
        let mut r = BufReader::with_capacity(1 << 22, file);
        let mut header = [0u8; 8];
        r.read_exact(&mut header)
            .with_context(|| format!("reading the header of {}", path.display()))?;
        let n = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let d = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
        let expected = 8 + (n * d * std::mem::size_of::<T>()) as u64;
        ensure!(
            file_len == expected,
            "{}: header says {n} × {d} {} ({expected} bytes) but the file has {file_len} bytes",
            path.display(),
            T::NAME
        );
        let mut m = Self::zeros(n, d);
        r.read_exact(bytemuck::cast_slice_mut(m.as_mut_slice()))
            .with_context(|| format!("reading {}", path.display()))?;
        Ok(m)
    }

    /// Writes a `.bin` file.
    pub fn write_bin(&self, path: &Path) -> Result<()> {
        let n = u32::try_from(self.nrows).context("too many rows for the .bin format")?;
        let d = u32::try_from(self.ncols).context("dimension too large for the .bin format")?;
        let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let mut w = BufWriter::with_capacity(1 << 22, file);
        w.write_all(&n.to_le_bytes())?;
        w.write_all(&d.to_le_bytes())?;
        w.write_all(bytemuck::cast_slice(self.as_slice()))?;
        w.flush()
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }
}

/// Scales every row to unit L2 norm (computed in `f64`) and returns the original norms.
/// Zero rows are left unchanged.
pub fn normalize_rows(m: &mut Matrix<f32>) -> Vec<f64> {
    (0..m.nrows())
        .map(|i| {
            let row = m.row_mut(i);
            let norm = row
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                .sqrt();
            if norm > 0.0 {
                for x in row.iter_mut() {
                    *x = (f64::from(*x) / norm) as f32;
                }
            }
            norm
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_aligned_and_distances_exact() {
        let data: Vec<u8> = (0..3 * 70).map(|i| (i * 37 % 256) as u8).collect();
        let m = Matrix::<u8>::from_slice(&data, 3, 70);
        assert_eq!(m.as_slice().as_ptr() as usize % 64, 0);
        assert_eq!(m.row(2), &data[140..210]);
        let exact: i64 = m
            .row(0)
            .iter()
            .zip(m.row(2))
            .map(|(&a, &b)| (i64::from(a) - i64::from(b)).pow(2))
            .sum();
        assert_eq!(u8::squared_l2(m.row(0), m.row(2)), exact as f32);
    }

    #[test]
    fn normalize_handles_zero_rows() {
        let mut m = Matrix::<f32>::from_slice(&[3.0, 4.0, 0.0, 0.0], 2, 2);
        let norms = normalize_rows(&mut m);
        assert_eq!(norms, vec![5.0, 0.0]);
        assert_eq!(m.as_slice(), &[0.6, 0.8, 0.0, 0.0]);
    }
}
