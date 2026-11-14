/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Sectioned binary container shared by the lab's file formats.
//!
//! All integers are little-endian. The header is
//!
//! | offset | type | field |
//! |---|---|---|
//! | 0 | `[u8; 8]` | magic (identifies the format) |
//! | 8 | `u32` | format version |
//! | 12 | `u32` | number of sections `S` |
//! | 16 | `[u64; 8]` | format-specific parameters |
//! | 80 | `[(u64, u64); S]` | per section: byte offset from the start of the file, byte length |
//!
//! Every section starts at a 64-byte-aligned offset (zero padding in between), so a
//! memory-mapped file can be viewed as typed slices without copying.

use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
};

use anyhow::{ensure, Context, Result};
use bytemuck::Pod;

#[cfg(target_endian = "big")]
compile_error!("the lab's file formats are little-endian; big-endian targets are not supported");

/// Number of format-specific `u64` parameters in every header.
pub const N_PARAMS: usize = 8;

const ALIGN: u64 = 64;
const FIXED_HEADER: usize = 16 + 8 * N_PARAMS;
const MAX_SECTIONS: usize = 1024;

fn align_up(x: u64) -> u64 {
    x.div_ceil(ALIGN) * ALIGN
}

/// Writes a container whose sections are given as raw bytes.
pub fn write<W: Write>(
    mut w: W,
    magic: &[u8; 8],
    version: u32,
    params: &[u64; N_PARAMS],
    sections: &[&[u8]],
) -> Result<()> {
    let table_end = (FIXED_HEADER + 16 * sections.len()) as u64;
    let mut header = Vec::with_capacity(align_up(table_end) as usize);
    header.extend_from_slice(magic);
    header.extend_from_slice(&version.to_le_bytes());
    header.extend_from_slice(&(sections.len() as u32).to_le_bytes());
    for p in params {
        header.extend_from_slice(&p.to_le_bytes());
    }
    let mut offset = align_up(table_end);
    for s in sections {
        header.extend_from_slice(&offset.to_le_bytes());
        header.extend_from_slice(&(s.len() as u64).to_le_bytes());
        offset = align_up(offset + s.len() as u64);
    }
    header.resize(align_up(table_end) as usize, 0);
    w.write_all(&header)?;

    let zeros = [0u8; ALIGN as usize];
    for s in sections {
        w.write_all(s)?;
        let pad = align_up(s.len() as u64) - s.len() as u64;
        w.write_all(&zeros[..pad as usize])?;
    }
    w.flush()?;
    Ok(())
}

/// Creates `path` for buffered writing.
pub fn create(path: &Path) -> Result<BufWriter<File>> {
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    Ok(BufWriter::with_capacity(1 << 22, file))
}

/// Reads the first 8 bytes of a file.
pub fn peek_magic(path: &Path) -> Result<[u8; 8]> {
    let mut magic = [0u8; 8];
    File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .with_context(|| format!("reading {}", path.display()))?;
    Ok(magic)
}

/// Reader of a container; sections are read on demand into typed vectors.
pub struct Reader<R> {
    inner: R,
    version: u32,
    params: [u64; N_PARAMS],
    sections: Vec<(u64, u64)>,
}

impl Reader<BufReader<File>> {
    /// Opens `path` and checks its magic.
    pub fn open(path: &Path, magic: &[u8; 8]) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        Self::new(BufReader::with_capacity(1 << 22, file), magic)
            .with_context(|| format!("reading {}", path.display()))
    }
}

impl<R: Read + Seek> Reader<R> {
    /// Parses the header and the section table.
    pub fn new(mut inner: R, magic: &[u8; 8]) -> Result<Self> {
        let file_len = inner.seek(SeekFrom::End(0))?;
        inner.seek(SeekFrom::Start(0))?;
        let mut fixed = [0u8; FIXED_HEADER];
        inner.read_exact(&mut fixed).context("truncated header")?;
        ensure!(
            &fixed[..8] == magic,
            "bad magic {:?} (expected {:?})",
            String::from_utf8_lossy(&fixed[..8]),
            String::from_utf8_lossy(magic)
        );
        let word = |at: usize| {
            u32::from_le_bytes([fixed[at], fixed[at + 1], fixed[at + 2], fixed[at + 3]])
        };
        let version = word(8);
        let n_sections = word(12) as usize;
        ensure!(
            n_sections <= MAX_SECTIONS,
            "implausible section count {n_sections}"
        );
        let mut params = [0u64; N_PARAMS];
        for (i, p) in params.iter_mut().enumerate() {
            *p = u64_at(&fixed, 16 + 8 * i);
        }
        let mut table = vec![0u8; 16 * n_sections];
        inner
            .read_exact(&mut table)
            .context("truncated section table")?;
        let sections: Vec<(u64, u64)> = (0..n_sections)
            .map(|i| (u64_at(&table, 16 * i), u64_at(&table, 16 * i + 8)))
            .collect();
        for (i, &(offset, len)) in sections.iter().enumerate() {
            ensure!(
                offset.is_multiple_of(ALIGN) && offset.checked_add(len).is_some_and(|end| end <= file_len),
                "section {i} (offset {offset}, {len} bytes) is misaligned or beyond the end of the file ({file_len} bytes)"
            );
        }
        Ok(Self {
            inner,
            version,
            params,
            sections,
        })
    }

    /// Format version stored in the header.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Format-specific parameters.
    pub fn params(&self) -> &[u64; N_PARAMS] {
        &self.params
    }

    /// Number of sections.
    pub fn n_sections(&self) -> usize {
        self.sections.len()
    }

    /// Reads section `i`, which must hold exactly `len` elements of `T`.
    pub fn section<T: Pod>(&mut self, i: usize, len: usize) -> Result<Vec<T>> {
        self.locate::<T>(i, len)?;
        let mut out = vec![T::zeroed(); len];
        self.section_into(i, &mut out)?;
        Ok(out)
    }

    /// Reads section `i` into `out`; the section must hold exactly `out.len()` elements.
    pub fn section_into<T: Pod>(&mut self, i: usize, out: &mut [T]) -> Result<()> {
        let offset = self.locate::<T>(i, out.len())?;
        self.inner.seek(SeekFrom::Start(offset))?;
        self.inner
            .read_exact(bytemuck::cast_slice_mut(out))
            .with_context(|| format!("reading section {i}"))
    }

    /// Byte offset of section `i` after checking that it holds exactly `len` elements of `T`.
    fn locate<T>(&self, i: usize, len: usize) -> Result<u64> {
        let &(offset, bytes) = self
            .sections
            .get(i)
            .with_context(|| format!("missing section {i}"))?;
        let size = std::mem::size_of::<T>();
        ensure!(
            bytes == (len * size) as u64,
            "section {i} has {bytes} bytes, expected {len} elements of {size} bytes"
        );
        Ok(offset)
    }
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[at..at + 8]);
    u64::from_le_bytes(b)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn roundtrip_aligns_sections() {
        let a: Vec<u32> = vec![1, 2, 3];
        let b: Vec<u8> = b"hello".to_vec();
        let c: Vec<u64> = vec![];
        let params = [7, 0, 0, 0, 0, 0, 0, 9];
        let mut buf = Vec::new();
        write(
            &mut buf,
            b"TESTFMT\0",
            3,
            &params,
            &[bytemuck::cast_slice(&a), &b, bytemuck::cast_slice(&c)],
        )
        .unwrap();
        assert_eq!(buf.len() % 64, 0);

        let mut r = Reader::new(Cursor::new(&buf), b"TESTFMT\0").unwrap();
        assert_eq!((r.version(), r.n_sections(), *r.params()), (3, 3, params));
        assert_eq!(r.section::<u32>(0, 3).unwrap(), a);
        assert_eq!(r.section::<u8>(1, 5).unwrap(), b);
        assert!(r.section::<u64>(2, 0).unwrap().is_empty());
        assert!(r.section::<u32>(0, 2).is_err());
        assert!(Reader::new(Cursor::new(&buf), b"OTHERFMT").is_err());
    }
}
