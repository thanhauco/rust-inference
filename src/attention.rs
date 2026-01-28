//! Causal attention over the paged KV cache.
//!
//! Replaces the cuDNN-frontend paged SDPA graph of the C++ engine. Queries
//! are `[B, S, H, Dh]`; query `i` of a sequence whose cache now holds `L`
//! tokens sits at position `L - S + i` and attends to positions `0..=that`
//! (the "bottom-right" causal mask). Grouped-query attention maps query head
//! `h` to KV head `h / (H / H_kv)`. Softmax is computed online one page at a
//! time, so memory use does not grow with sequence length.

use rayon::prelude::*;

use crate::cpu::simd;
use crate::error::{Error, Result};
use crate::kvcache::{KvCache, PAGE_SIZE};
use crate::ops::expect_f32;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct PagedAttention {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
}

impl PagedAttention {
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        PagedAttention { num_heads, num_kv_heads, head_dim, scale: 1.0 / (head_dim as f32).sqrt() }
    }

    /// Attention output for `q: [B, S, H, Dh]`, after the new keys/values
    /// were appended to `cache`.
    pub fn forward(&self, q: &Tensor, cache: &KvCache) -> Result<Tensor> {
        expect_f32(q, "attention")?;
        let (h, hkv, dh) = (self.num_heads, self.num_kv_heads, self.head_dim);
        if q.ndim() != 4 || q.dim(2) != h || q.dim(3) != dh || q.dim(0) != cache.batch_size() {
            return Err(Error::Shape(format!("attention: q {:?} for {h} heads of {dh}", q.shape())));
        }
        if cache.num_kv_heads() != hkv || cache.head_dim() != dh || h % hkv != 0 {
            return Err(Error::Shape("attention: cache / head configuration mismatch".into()));
        }
        let (b, s) = (q.dim(0), q.dim(1));
        for bi in 0..b {
            if cache.seq_len(bi) < s {
                return Err(Error::Shape("attention: append k/v to the cache before attending".into()));
            }
        }
        let group = h / hkv;
        let qv = q.as_slice::<f32>();
        let kp = cache.k_pages().as_slice::<f32>();
        let vp = cache.v_pages().as_slice::<f32>();

        // One task per (sequence, token, kv head): the `group` query heads
        // sharing a KV head are contiguous, so each K/V row is read once for
        // all of them.
        let mut out = vec![0f32; qv.len()];
        out.par_chunks_mut(group * dh).enumerate().for_each(|(task, o)| {
            let kvh = task % hkv;
            let tok = (task / hkv) % s;
            let bi = task / (hkv * s);
            let q0 = ((bi * s + tok) * h + kvh * group) * dh;
            let ctx = Ctx {
                kp,
                vp,
                cache,
                kvh,
                dh,
                scale: self.scale,
                qpos: cache.seq_len(bi) - s + tok,
                table: &cache.page_table()[bi * cache.max_pages_per_seq()..],
            };
            let qs = &qv[q0..q0 + group * dh];
            match group {
                1 => attend::<1>(&ctx, qs, o),
                2 => attend::<2>(&ctx, qs, o),
                3 => attend::<3>(&ctx, qs, o),
                4 => attend::<4>(&ctx, qs, o),
                5 => attend::<5>(&ctx, qs, o),
                6 => attend::<6>(&ctx, qs, o),
                7 => attend::<7>(&ctx, qs, o),
                8 => attend::<8>(&ctx, qs, o),
                _ => {
                    for (qh, oh) in qs.chunks_exact(dh).zip(o.chunks_exact_mut(dh)) {
                        attend::<1>(&ctx, qh, oh);
                    }
                }
            }
        });
        Tensor::from_f32(&out, q.shape())
    }
}

struct Ctx<'a> {
    kp: &'a [f32],
    vp: &'a [f32],
    cache: &'a KvCache,
    table: &'a [i32],
    kvh: usize,
    dh: usize,
    qpos: usize,
    scale: f32,
}

/// Online-softmax attention of `G` query heads (`qs: [G, Dh]`) against one
/// KV head, a page at a time.
#[allow(clippy::needless_range_loop)]
fn attend<const G: usize>(c: &Ctx<'_>, qs: &[f32], o: &mut [f32]) {
    let dh = c.dh;
    let qrows: [&[f32]; G] = std::array::from_fn(|g| &qs[g * dh..(g + 1) * dh]);
    let mut m = [f32::NEG_INFINITY; G];
    let mut l = [0f32; G];
    let mut scores = [[0f32; PAGE_SIZE]; G];
    for (logical, &page) in c.table[..=c.qpos / PAGE_SIZE].iter().enumerate() {
        let n = (c.qpos + 1 - logical * PAGE_SIZE).min(PAGE_SIZE);
        let base = c.cache.row_offset(page as usize, c.kvh, 0);
        let mut block_max = [f32::NEG_INFINITY; G];
        for j in 0..n {
            let d = simd::dot_mr::<G>(&c.kp[base + j * dh..base + (j + 1) * dh], &qrows);
            for g in 0..G {
                scores[g][j] = d[g] * c.scale;
                block_max[g] = block_max[g].max(scores[g][j]);
            }
        }
        let mut m_new = [0f32; G];
        for (g, og) in o.chunks_exact_mut(dh).enumerate() {
            m_new[g] = m[g].max(block_max[g]);
            let correction = (m[g] - m_new[g]).exp();
            if correction != 1.0 {
                simd::scale(og, correction);
            }
            l[g] *= correction;
            m[g] = m_new[g];
        }
        for j in 0..n {
            let vrow = &c.vp[base + j * dh..base + (j + 1) * dh];
            for (g, og) in o.chunks_exact_mut(dh).enumerate() {
                let p = (scores[g][j] - m_new[g]).exp();
                l[g] += p;
                simd::axpy(og, p, vrow);
            }
        }
    }
    for (g, og) in o.chunks_exact_mut(dh).enumerate() {
        simd::scale(og, 1.0 / l[g]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::tensor::Device;

    /// Dense reference: softmax(q k^T * scale + causal mask) v
    #[allow(clippy::too_many_arguments)]
    fn reference(q: &[f32], k: &[f32], v: &[f32], s: usize, l: usize, h: usize, hkv: usize, dh: usize) -> Vec<f32> {
        let scale = 1.0 / (dh as f32).sqrt();
        let mut out = vec![0f32; s * h * dh];
        for t in 0..s {
            let qpos = l - s + t;
            for hh in 0..h {
                let kvh = hh / (h / hkv);
                let qr = &q[(t * h + hh) * dh..][..dh];
                let scores: Vec<f32> = (0..=qpos)
                    .map(|p| qr.iter().zip(&k[(p * hkv + kvh) * dh..][..dh]).map(|(a, b)| a * b).sum::<f32>() * scale)
                    .collect();
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = scores.iter().map(|x| (x - mx).exp()).collect();
                let z: f32 = e.iter().sum();
                for (p, w) in e.iter().enumerate() {
                    for d in 0..dh {
                        out[(t * h + hh) * dh + d] += w / z * v[(p * hkv + kvh) * dh + d];
                    }
                }
            }
        }
        out
    }

    fn data(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.37 + seed).sin()).collect()
    }

    #[test]
    fn prefill_then_decode_matches_dense() {
        // GQA groups of 2, 8, 3 and 9 (the last takes the per-head fallback)
        for (h, hkv) in [(4, 2), (8, 1), (6, 2), (9, 1)] {
            check_against_dense(h, hkv, 8);
        }
    }

    fn check_against_dense(h: usize, hkv: usize, dh: usize) {
        let total = 70; // spans three pages
        let k = data(total * hkv * dh, 0.1);
        let v = data(total * hkv * dh, 0.7);
        let q = data(total * h * dh, 1.3);
        let attn = PagedAttention::new(h, hkv, dh);
        let mut cache = KvCache::new(hkv, dh, 8, 1, 4, DType::F32, Device::Cpu);

        // prefill 65 tokens, then decode 5 one at a time
        let mut got = Vec::new();
        let mut pos = 0;
        for step in std::iter::once(65).chain(std::iter::repeat_n(1, 5)) {
            let kt = Tensor::from_f32(&k[pos * hkv * dh..(pos + step) * hkv * dh], &[1, step, hkv, dh]).unwrap();
            let vt = Tensor::from_f32(&v[pos * hkv * dh..(pos + step) * hkv * dh], &[1, step, hkv, dh]).unwrap();
            cache.append(&kt, &vt).unwrap();
            let qt = Tensor::from_f32(&q[pos * h * dh..(pos + step) * h * dh], &[1, step, h, dh]).unwrap();
            got.extend_from_slice(attn.forward(&qt, &cache).unwrap().as_slice::<f32>());
            pos += step;
        }
        let want = reference(&q, &k, &v, total, total, h, hkv, dh);
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!((a - b).abs() < 1e-5, "h={h} hkv={hkv} elem {i}: {a} vs {b}");
        }
    }
}
