//! Paged KV cache.
//!
//! K and V live in fixed pools of pages, `[max_pages, kv_heads, PAGE_SIZE,
//! head_dim]` each. Every sequence in the batch owns a row of the page table
//! mapping its logical pages to physical ones; pages are handed out from a
//! free list as the sequence grows. One `KvCache` per decoder layer.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::{Device, Tensor};

pub const PAGE_SIZE: usize = 32;

pub struct KvCache {
    num_kv_heads: usize,
    head_dim: usize,
    batch_size: usize,
    max_pages: usize,
    max_pages_per_seq: usize,
    k_pages: Tensor,
    v_pages: Tensor,
    /// free physical pages; popped from the back so page 0 is used first
    free_pages: Vec<usize>,
    /// `[batch_size * max_pages_per_seq]`, -1 = unmapped
    page_table: Vec<i32>,
    seq_lens: Vec<usize>,
    /// device copies of `page_table` and of the pre-append lengths
    #[cfg(feature = "cuda")]
    device_tables: Option<(crate::cuda::DeviceBuffer, crate::cuda::DeviceBuffer)>,
}

impl KvCache {
    pub fn new(
        num_kv_heads: usize,
        head_dim: usize,
        max_pages: usize,
        batch_size: usize,
        max_pages_per_seq: usize,
        dtype: DType,
        device: Device,
    ) -> Self {
        let shape = [max_pages, num_kv_heads, PAGE_SIZE, head_dim];
        KvCache {
            num_kv_heads,
            head_dim,
            batch_size,
            max_pages,
            max_pages_per_seq,
            k_pages: Tensor::zeros(&shape, dtype, device),
            v_pages: Tensor::zeros(&shape, dtype, device),
            free_pages: (0..max_pages).rev().collect(),
            page_table: vec![-1; batch_size * max_pages_per_seq],
            seq_lens: vec![0; batch_size],
            #[cfg(feature = "cuda")]
            device_tables: (device == Device::Cuda).then(|| {
                let table = crate::cuda::DeviceBuffer::alloc(batch_size * max_pages_per_seq * 4);
                let lens = crate::cuda::DeviceBuffer::alloc(batch_size * 4);
                (table.expect("device allocation failed"), lens.expect("device allocation failed"))
            }),
        }
    }

    /// Device page table and pre-append lengths (`int32`), CUDA caches only.
    #[cfg(feature = "cuda")]
    pub fn device_tables(&self) -> Option<(u64, u64)> {
        self.device_tables.as_ref().map(|(t, l)| (t.ptr(), l.ptr()))
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
    pub fn max_pages_per_seq(&self) -> usize {
        self.max_pages_per_seq
    }
    pub fn seq_len(&self, b: usize) -> usize {
        self.seq_lens[b]
    }
    pub fn seq_lens(&self) -> &[usize] {
        &self.seq_lens
    }
    pub fn page_table(&self) -> &[i32] {
        &self.page_table
    }
    pub fn k_pages(&self) -> &Tensor {
        &self.k_pages
    }
    pub fn v_pages(&self) -> &Tensor {
        &self.v_pages
    }
    pub fn free_page_count(&self) -> usize {
        self.free_pages.len()
    }
    /// Longest sequence a single batch entry can hold.
    pub fn max_seq_len(&self) -> usize {
        self.max_pages_per_seq * PAGE_SIZE
    }

    /// Map physical pages for the next `new_tokens` positions of every
    /// sequence (the C++ `prepare_table`).
    pub fn prepare(&mut self, new_tokens: usize) -> Result<()> {
        for b in 0..self.batch_size {
            let end = self.seq_lens[b] + new_tokens;
            if end > self.max_seq_len() {
                return Err(Error::Config(format!(
                    "sequence {b} would reach {end} tokens, KV cache holds {} per sequence",
                    self.max_seq_len()
                )));
            }
            for logical in self.seq_lens[b] / PAGE_SIZE..end.div_ceil(PAGE_SIZE) {
                let slot = &mut self.page_table[b * self.max_pages_per_seq + logical];
                if *slot < 0 {
                    let page = self
                        .free_pages
                        .pop()
                        .ok_or_else(|| Error::Config(format!("KV cache out of pages ({} total)", self.max_pages)))?;
                    *slot = page as i32;
                }
            }
        }
        #[cfg(feature = "cuda")]
        if let Some((table, lens)) = &self.device_tables {
            // pageable sources: the driver stages them before returning
            let t: Vec<u8> = self.page_table.iter().flat_map(|v| v.to_le_bytes()).collect();
            let l: Vec<u8> = self.seq_lens.iter().flat_map(|&v| (v as i32).to_le_bytes()).collect();
            table.upload(0, &t)?;
            lens.upload(0, &l)?;
        }
        Ok(())
    }

    fn physical_page(&self, b: usize, pos: usize) -> usize {
        let p = self.page_table[b * self.max_pages_per_seq + pos / PAGE_SIZE];
        debug_assert!(p >= 0, "position {pos} of sequence {b} has no page");
        p as usize
    }

    /// Element offset of `(page, head, slot)` inside the page pool.
    #[inline]
    pub fn row_offset(&self, page: usize, head: usize, slot: usize) -> usize {
        ((page * self.num_kv_heads + head) * PAGE_SIZE + slot) * self.head_dim
    }

    /// Write `k`, `v` (`[B, S, kv_heads, head_dim]`) at the end of every
    /// sequence and advance the lengths.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let expected_tail = [self.num_kv_heads, self.head_dim];
        if k.ndim() != 4 || k.dim(0) != self.batch_size || k.shape()[2..] != expected_tail || k.shape() != v.shape() {
            return Err(Error::Shape(format!(
                "kv append: k {:?} v {:?}, cache expects [{}, S, {}, {}]",
                k.shape(),
                v.shape(),
                self.batch_size,
                self.num_kv_heads,
                self.head_dim
            )));
        }
        let s = k.dim(1);
        self.prepare(s)?;
        #[cfg(feature = "cuda")]
        if self.device_tables.is_some() {
            crate::cuda::ops::kv_append(self, k, v)?;
            for len in &mut self.seq_lens {
                *len += s;
            }
            return Ok(());
        }
        let (hkv, dh) = (self.num_kv_heads, self.head_dim);
        let offsets: Vec<(usize, usize)> = (0..self.batch_size)
            .flat_map(|b| (0..s).map(move |t| (b, t)))
            .flat_map(|(b, t)| {
                let pos = self.seq_lens[b] + t;
                let page = self.physical_page(b, pos);
                let src_tok = (b * s + t) * hkv * dh;
                (0..hkv).map(move |h| (src_tok + h * dh, ((page * hkv + h) * PAGE_SIZE + pos % PAGE_SIZE) * dh))
            })
            .collect();
        copy_rows(&mut self.k_pages, k, &offsets, dh)?;
        copy_rows(&mut self.v_pages, v, &offsets, dh)?;
        for len in &mut self.seq_lens {
            *len += s;
        }
        Ok(())
    }

    /// Drop all sequences and return every page to the free list.
    pub fn reset(&mut self) {
        self.page_table.fill(-1);
        self.seq_lens.fill(0);
        self.free_pages = (0..self.max_pages).rev().collect();
    }
}

fn copy_rows(dst: &mut Tensor, src: &Tensor, offsets: &[(usize, usize)], row: usize) -> Result<()> {
    if dst.dtype() != src.dtype() {
        return Err(Error::Shape(format!("kv cache is {}, got {}", dst.dtype(), src.dtype())));
    }
    let es = src.dtype().size();
    let s = src.bytes();
    let d = dst.bytes_mut()?;
    for &(from, to) in offsets {
        d[to * es..(to + row) * es].copy_from_slice(&s[from * es..(from + row) * es]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_are_allocated_lazily() {
        let mut kv = KvCache::new(1, 2, 4, 1, 3, DType::F32, Device::Cpu);
        kv.prepare(1).unwrap();
        assert_eq!(kv.page_table(), &[0, -1, -1]);
        kv.prepare(PAGE_SIZE + 1).unwrap();
        assert_eq!(kv.page_table(), &[0, 1, -1]);
        assert_eq!(kv.free_page_count(), 2);
        assert!(kv.prepare(3 * PAGE_SIZE + 1).is_err());
        kv.reset();
        assert_eq!(kv.free_page_count(), 4);
    }

    #[test]
    fn append_writes_at_sequence_end() {
        let (hkv, dh) = (2, 3);
        let mut kv = KvCache::new(hkv, dh, 4, 1, 4, DType::F32, Device::Cpu);
        let n = 40; // crosses a page boundary
        let vals: Vec<f32> = (0..n * hkv * dh).map(|i| i as f32).collect();
        let k = Tensor::from_f32(&vals, &[1, n, hkv, dh]).unwrap();
        kv.append(&k, &k).unwrap();
        assert_eq!(kv.seq_len(0), n);
        let pool = kv.k_pages().as_slice::<f32>();
        for pos in [0, 31, 32, 39] {
            for h in 0..hkv {
                let page = kv.page_table()[pos / PAGE_SIZE] as usize;
                let off = kv.row_offset(page, h, pos % PAGE_SIZE);
                let src = (pos * hkv + h) * dh;
                assert_eq!(&pool[off..off + dh], &vals[src..src + dh]);
            }
        }
    }
}
