//! `y = x W^T` on the CPU for F32 / BF16 / F16 / block-scaled FP8 weights.
//!
//! Work is split over output features (rows of W) so every weight row is
//! streamed from memory once per call.
//!
//! * Few activation rows (decode, M <= 4): each weight row is decoded in
//!   registers and dotted against all M rows at once. FP8 partial sums are
//!   kept per 128-wide k-group and scaled at the end of the group, as in the
//!   C++ GEMV kernels.
//! * Many rows (prefill): a task decodes its block of weight rows to f32
//!   once and runs a register-tiled microkernel over 4-row tiles of x.

use rayon::prelude::*;

use super::simd::{self, Isa};
use crate::dtype::{self, DType};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Rows of W per rayon task.
const ROWS_PER_TASK: usize = 16;
/// Default FP8 quantization block (Qwen3 FP8 uses `weight_block_size: [128, 128]`).
pub const FP8_BLOCK: usize = 128;
/// Up to this many activation rows take the fused per-row path.
const FUSED_MAX_M: usize = 4;

/// How a `[sN, sK]` block scale maps onto an `[N, K]` weight.
#[derive(Debug, Clone, Copy)]
pub struct BlockLayout {
    pub block_n: usize,
    pub block_k: usize,
    pub scale_cols: usize,
}

impl BlockLayout {
    pub fn new(n: usize, k: usize, scale_shape: &[usize]) -> Result<Self> {
        let (sn, sk) = (scale_shape[0], scale_shape[1]);
        if sn == 0 || sk == 0 {
            return Err(Error::Shape(format!("empty block scale {scale_shape:?}")));
        }
        // Square power-of-two blocks with partial last blocks: 128x128 is
        // what Qwen ships, other sizes come from `quantize --block`.
        for block in std::iter::once(FP8_BLOCK).chain((4..=10).map(|p| 1usize << p)) {
            if sn == n.div_ceil(block) && sk == k.div_ceil(block) {
                return Ok(BlockLayout { block_n: block, block_k: block, scale_cols: sk });
            }
        }
        // Otherwise assume uniform blocks (covers per-tensor [1, 1] scales).
        Ok(BlockLayout { block_n: n.div_ceil(sn), block_k: k.div_ceil(sk), scale_cols: sk })
    }
}

/// One weight row in its stored format.
#[derive(Clone, Copy)]
pub(crate) enum Row<'a> {
    F32(&'a [f32]),
    Bf16(&'a [u16]),
    F16(&'a [u16]),
    /// `scales[g]` applies to elements `g*group..(g+1)*group`
    Fp8 {
        data: &'a [u8],
        scales: &'a [f32],
        group: usize,
    },
}

/// A weight matrix prepared for row access.
pub(crate) struct Weight<'a> {
    w: &'a Tensor,
    k: usize,
    layout: Option<(BlockLayout, &'a [f32])>,
}

impl<'a> Weight<'a> {
    pub(crate) fn new(w: &'a Tensor) -> Result<Self> {
        if w.ndim() != 2 {
            return Err(Error::Shape(format!("linear weight must be 2-D, got {:?}", w.shape())));
        }
        if !matches!(w.dtype(), DType::F32 | DType::BF16 | DType::F16 | DType::F8E4M3) {
            return Err(Error::Unsupported(format!("linear weight dtype {}", w.dtype())));
        }
        let (n, k) = (w.dim(0), w.dim(1));
        let layout = match (w.dtype(), w.scale()) {
            (DType::F8E4M3, Some(s)) => Some((BlockLayout::new(n, k, s.shape())?, s.as_slice::<f32>())),
            _ => None,
        };
        Ok(Weight { w, k, layout })
    }

    pub(crate) fn rows(&self) -> usize {
        self.w.dim(0)
    }

    pub(crate) fn row(&self, n: usize) -> Row<'a> {
        let k = self.k;
        let r = n * k..(n + 1) * k;
        match self.w.dtype() {
            DType::F32 => Row::F32(&self.w.as_slice::<f32>()[r]),
            DType::BF16 => Row::Bf16(&self.w.as_slice::<u16>()[r]),
            DType::F16 => Row::F16(&self.w.as_slice::<u16>()[r]),
            DType::F8E4M3 => {
                const ONE: [f32; 1] = [1.0];
                let data = &self.w.as_slice::<u8>()[r];
                match &self.layout {
                    Some((l, s)) => Row::Fp8 {
                        data,
                        scales: &s[(n / l.block_n) * l.scale_cols..][..l.scale_cols],
                        group: l.block_k,
                    },
                    None => Row::Fp8 { data, scales: &ONE, group: k.max(1) },
                }
            }
            _ => unreachable!("checked in Weight::new"),
        }
    }
}

/// Decode a row to f32 (scales applied).
pub(crate) fn decode_row(row: Row<'_>, dst: &mut [f32]) {
    match row {
        Row::F32(s) => dst.copy_from_slice(s),
        Row::Bf16(s) => simd::decode_bf16(s, dst),
        Row::F16(s) => simd::decode_f16(s, dst),
        Row::Fp8 { data, scales, group } => {
            for (g, (src, out)) in data.chunks(group).zip(dst.chunks_mut(group)).enumerate() {
                simd::decode_fp8(src, out, scales[g]);
            }
        }
    }
}

/// `out[t] = row . xs[t]` with the row decoded on the fly.
pub(crate) fn row_dot<const MR: usize>(row: Row<'_>, xs: &[&[f32]; MR]) -> [f32; MR] {
    match simd::isa() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the ISA was detected at runtime; all slices have length K.
        Isa::Avx512 => unsafe {
            match row {
                Row::F32(s) => x86::dot_f32_512(s.as_ptr(), xs, s.len(), s.len().max(1), &[1.0]),
                Row::Bf16(s) => x86::dot_bf16_512(s.as_ptr(), xs, s.len(), s.len().max(1), &[1.0]),
                Row::F16(s) => x86::dot_f16_512(s.as_ptr(), xs, s.len(), s.len().max(1), &[1.0]),
                Row::Fp8 { data, scales, group } => x86::dot_fp8_512(data.as_ptr(), xs, data.len(), group, scales),
            }
        },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe {
            match row {
                Row::F32(s) => x86::dot_f32_256(s.as_ptr(), xs, s.len(), s.len().max(1), &[1.0]),
                Row::Bf16(s) => x86::dot_bf16_256(s.as_ptr(), xs, s.len(), s.len().max(1), &[1.0]),
                Row::F16(s) => x86::dot_f16_256(s.as_ptr(), xs, s.len(), s.len().max(1), &[1.0]),
                Row::Fp8 { data, scales, group } => x86::dot_fp8_256(data.as_ptr(), xs, data.len(), group, scales),
            }
        },
        _ => row_dot_scalar(row, xs),
    }
}

fn row_dot_scalar<const MR: usize>(row: Row<'_>, xs: &[&[f32]; MR]) -> [f32; MR] {
    let mut buf = [0f32; 256];
    let (len, group, scales): (usize, usize, &[f32]) = match row {
        Row::F32(s) => (s.len(), 256, &[]),
        Row::Bf16(s) | Row::F16(s) => (s.len(), 256, &[]),
        Row::Fp8 { data, scales, group } => (data.len(), group, scales),
    };
    let mut out = [0f32; MR];
    let mut g0 = 0;
    let mut gi = 0;
    while g0 < len {
        let g1 = (g0 + group).min(len);
        let mut part = [0f32; MR];
        let mut c0 = g0;
        while c0 < g1 {
            let c1 = (c0 + buf.len()).min(g1);
            let chunk = &mut buf[..c1 - c0];
            match row {
                Row::F32(s) => chunk.copy_from_slice(&s[c0..c1]),
                Row::Bf16(s) => chunk.iter_mut().zip(&s[c0..c1]).for_each(|(d, &b)| *d = dtype::bf16_to_f32(b)),
                Row::F16(s) => chunk.iter_mut().zip(&s[c0..c1]).for_each(|(d, &b)| *d = dtype::f16_to_f32(b)),
                Row::Fp8 { data, .. } => {
                    chunk.iter_mut().zip(&data[c0..c1]).for_each(|(d, &b)| *d = dtype::fp8_e4m3_to_f32(b))
                }
            }
            let sub: [&[f32]; MR] = std::array::from_fn(|t| &xs[t][c0..c1]);
            let d = simd::dot_mr::<MR>(chunk, &sub);
            for t in 0..MR {
                part[t] += d[t];
            }
            c0 = c1;
        }
        let s = scales.get(gi).copied().unwrap_or(1.0);
        for t in 0..MR {
            out[t] += part[t] * s;
        }
        g0 = g1;
        gi += 1;
    }
    out
}

/// `x: [M, K]` (row-major f32), `w: [N, K]` -> `y: [M, N]`.
pub fn linear_f32(x: &[f32], m: usize, w: &Tensor) -> Result<Vec<f32>> {
    let weight = Weight::new(w)?;
    let (n, k) = (weight.rows(), weight.k);
    if x.len() != m * k {
        return Err(Error::Shape(format!("linear: x has {} elements, expected {m}x{k}", x.len())));
    }
    if m == 0 || n == 0 {
        return Ok(vec![0f32; m * n]);
    }

    // yt[n][m]: each task owns a contiguous block of output features.
    let mut yt = vec![0f32; n * m];
    let xrow = |t: usize| &x[t * k..(t + 1) * k];
    yt.par_chunks_mut(ROWS_PER_TASK * m).enumerate().for_each(|(task, out)| {
        let n0 = task * ROWS_PER_TASK;
        let rows = out.len() / m;
        if m <= FUSED_MAX_M {
            for r in 0..rows {
                let row = weight.row(n0 + r);
                let o = &mut out[r * m..(r + 1) * m];
                match m {
                    1 => o.copy_from_slice(&row_dot::<1>(row, &[xrow(0)])),
                    2 => o.copy_from_slice(&row_dot::<2>(row, &[xrow(0), xrow(1)])),
                    3 => o.copy_from_slice(&row_dot::<3>(row, &[xrow(0), xrow(1), xrow(2)])),
                    _ => o.copy_from_slice(&row_dot::<4>(row, &[xrow(0), xrow(1), xrow(2), xrow(3)])),
                }
            }
            return;
        }
        // prefill: decode this task's rows once, then tile over x
        let mut wdec = vec![0f32; rows * k];
        for (r, dst) in wdec.chunks_mut(k).enumerate() {
            decode_row(weight.row(n0 + r), dst);
        }
        let wrow = |r: usize| &wdec[r * k..(r + 1) * k];
        let mut t = 0;
        while t + 4 <= m {
            let xs = [xrow(t), xrow(t + 1), xrow(t + 2), xrow(t + 3)];
            let mut r = 0;
            while r + 4 <= rows {
                let tile = tile_4x4([wrow(r), wrow(r + 1), wrow(r + 2), wrow(r + 3)], &xs);
                for (i, vals) in tile.iter().enumerate() {
                    out[(r + i) * m + t..][..4].copy_from_slice(vals);
                }
                r += 4;
            }
            for r in r..rows {
                out[r * m + t..][..4].copy_from_slice(&simd::dot_mr::<4>(wrow(r), &xs));
            }
            t += 4;
        }
        for t in t..m {
            for r in 0..rows {
                out[r * m + t] = simd::dot(wrow(r), xrow(t));
            }
        }
    });

    if m == 1 {
        return Ok(yt);
    }
    let mut y = vec![0f32; m * n];
    y.par_chunks_mut(n).enumerate().for_each(|(row, out)| {
        for (col, v) in out.iter_mut().enumerate() {
            *v = yt[col * m + row];
        }
    });
    Ok(y)
}

/// `out[i][j] = w[i] . x[j]` for a 4x4 tile of decoded rows.
fn tile_4x4(w: [&[f32]; 4], x: &[&[f32]; 4]) -> [[f32; 4]; 4] {
    match simd::isa() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the ISA was detected at runtime; all slices have length K.
        Isa::Avx512 => unsafe { x86::tile_4x4_512(w, x) },
        _ => std::array::from_fn(|i| simd::dot_mr::<4>(w[i], x)),
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use crate::dtype;

    // --- 16-wide loads (AVX-512) ---------------------------------------------

    #[target_feature(enable = "avx512f,avx2,fma,f16c")]
    #[inline]
    unsafe fn ld16_f32(p: *const f32) -> __m512 {
        unsafe { _mm512_loadu_ps(p) }
    }

    #[target_feature(enable = "avx512f,avx2,fma,f16c")]
    #[inline]
    unsafe fn ld16_bf16(p: *const u16) -> __m512 {
        unsafe {
            let h = _mm256_loadu_si256(p as *const __m256i);
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16))
        }
    }

    #[target_feature(enable = "avx512f,avx2,fma,f16c")]
    #[inline]
    unsafe fn ld16_f16(p: *const u16) -> __m512 {
        unsafe { _mm512_cvtph_ps(_mm256_loadu_si256(p as *const __m256i)) }
    }

    /// 16 E4M3 bytes -> f32 (integer exponent rebias, see
    /// `simd::x86::decode8_fp8`).
    #[target_feature(enable = "avx512f,avx2,fma,f16c")]
    #[inline]
    unsafe fn ld16_fp8(p: *const u8) -> __m512 {
        unsafe {
            let w32 = _mm512_cvtepu8_epi32(_mm_loadu_si128(p as *const __m128i));
            let sgn = _mm512_slli_epi32(_mm512_and_si512(w32, _mm512_set1_epi32(0x80)), 24);
            let mag = _mm512_slli_epi32(_mm512_and_si512(w32, _mm512_set1_epi32(0x7F)), 20);
            let norm = _mm512_add_epi32(mag, _mm512_set1_epi32(120 << 23));
            let sub = _mm512_cmpeq_epi32_mask(_mm512_and_si512(w32, _mm512_set1_epi32(0x78)), _mm512_setzero_si512());
            let subv = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_and_si512(w32, _mm512_set1_epi32(0x7))),
                _mm512_set1_ps(1.0 / 512.0),
            );
            let bits = _mm512_mask_blend_epi32(sub, norm, _mm512_castps_si512(subv));
            _mm512_castsi512_ps(_mm512_or_si512(bits, sgn))
        }
    }

    // --- 8-wide loads (AVX2) -------------------------------------------------

    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    unsafe fn ld8_f32(p: *const f32) -> __m256 {
        unsafe { _mm256_loadu_ps(p) }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    unsafe fn ld8_bf16(p: *const u16) -> __m256 {
        unsafe {
            let h = _mm_loadu_si128(p as *const __m128i);
            _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16))
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    unsafe fn ld8_f16(p: *const u16) -> __m256 {
        unsafe { _mm256_cvtph_ps(_mm_loadu_si128(p as *const __m128i)) }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    unsafe fn ld8_fp8(p: *const u8) -> __m256 {
        unsafe { crate::cpu::simd::x86::decode8_fp8(p) }
    }

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn hsum256(v: __m256) -> f32 {
        unsafe { crate::cpu::simd::x86::hsum256(v) }
    }

    #[inline(always)]
    fn sc_f32(p: *const f32, i: usize) -> f32 {
        unsafe { *p.add(i) }
    }
    #[inline(always)]
    fn sc_bf16(p: *const u16, i: usize) -> f32 {
        dtype::bf16_to_f32(unsafe { *p.add(i) })
    }
    #[inline(always)]
    fn sc_f16(p: *const u16, i: usize) -> f32 {
        dtype::f16_to_f32(unsafe { *p.add(i) })
    }
    #[inline(always)]
    fn sc_fp8(p: *const u8, i: usize) -> f32 {
        dtype::fp8_e4m3_to_f32(unsafe { *p.add(i) })
    }

    /// Fused decode + dot of one weight row against MR activation rows, with
    /// per-group scales: `out[t] = sum_g scale[g] * sum_{k in g} w[k] x_t[k]`.
    macro_rules! dot_kernel {
        ($name:ident, $t:ty, $feat:literal, $vec:ty, $lanes:literal, $zero:ident, $fma:ident, $add:ident,
         $set1:ident, $loadx:ident, $ld:ident, $sc:ident, $reduce:ident) => {
            #[target_feature(enable = $feat)]
            pub unsafe fn $name<const MR: usize>(
                w: *const $t,
                xs: &[&[f32]; MR],
                k: usize,
                group: usize,
                scales: &[f32],
            ) -> [f32; MR] {
                unsafe {
                    let mut acc: [$vec; MR] = [$zero(); MR];
                    let mut tail = [0f32; MR];
                    let mut g0 = 0;
                    let mut gi = 0;
                    while g0 < k {
                        let g1 = (g0 + group).min(k);
                        let mut a0: [$vec; MR] = [$zero(); MR];
                        let mut a1: [$vec; MR] = [$zero(); MR];
                        let mut i = g0;
                        while i + 2 * $lanes <= g1 {
                            let w0 = $ld(w.add(i));
                            let w1 = $ld(w.add(i + $lanes));
                            for t in 0..MR {
                                let xp = xs[t].as_ptr();
                                a0[t] = $fma(w0, $loadx(xp.add(i)), a0[t]);
                                a1[t] = $fma(w1, $loadx(xp.add(i + $lanes)), a1[t]);
                            }
                            i += 2 * $lanes;
                        }
                        while i + $lanes <= g1 {
                            let w0 = $ld(w.add(i));
                            for t in 0..MR {
                                a0[t] = $fma(w0, $loadx(xs[t].as_ptr().add(i)), a0[t]);
                            }
                            i += $lanes;
                        }
                        let mut gt = [0f32; MR];
                        while i < g1 {
                            let wv = $sc(w, i);
                            for t in 0..MR {
                                gt[t] += wv * xs[t][i];
                            }
                            i += 1;
                        }
                        let s = *scales.get_unchecked(gi);
                        let vs = $set1(s);
                        for t in 0..MR {
                            acc[t] = $fma($add(a0[t], a1[t]), vs, acc[t]);
                            tail[t] += gt[t] * s;
                        }
                        g0 = g1;
                        gi += 1;
                    }
                    let mut out = [0f32; MR];
                    for t in 0..MR {
                        out[t] = $reduce(acc[t]) + tail[t];
                    }
                    out
                }
            }
        };
    }

    dot_kernel!(
        dot_f32_512,
        f32,
        "avx512f,avx2,fma,f16c",
        __m512,
        16,
        _mm512_setzero_ps,
        _mm512_fmadd_ps,
        _mm512_add_ps,
        _mm512_set1_ps,
        _mm512_loadu_ps,
        ld16_f32,
        sc_f32,
        _mm512_reduce_add_ps
    );
    dot_kernel!(
        dot_bf16_512,
        u16,
        "avx512f,avx2,fma,f16c",
        __m512,
        16,
        _mm512_setzero_ps,
        _mm512_fmadd_ps,
        _mm512_add_ps,
        _mm512_set1_ps,
        _mm512_loadu_ps,
        ld16_bf16,
        sc_bf16,
        _mm512_reduce_add_ps
    );
    dot_kernel!(
        dot_f16_512,
        u16,
        "avx512f,avx2,fma,f16c",
        __m512,
        16,
        _mm512_setzero_ps,
        _mm512_fmadd_ps,
        _mm512_add_ps,
        _mm512_set1_ps,
        _mm512_loadu_ps,
        ld16_f16,
        sc_f16,
        _mm512_reduce_add_ps
    );
    dot_kernel!(
        dot_fp8_512,
        u8,
        "avx512f,avx2,fma,f16c",
        __m512,
        16,
        _mm512_setzero_ps,
        _mm512_fmadd_ps,
        _mm512_add_ps,
        _mm512_set1_ps,
        _mm512_loadu_ps,
        ld16_fp8,
        sc_fp8,
        _mm512_reduce_add_ps
    );

    dot_kernel!(
        dot_f32_256,
        f32,
        "avx2,fma,f16c",
        __m256,
        8,
        _mm256_setzero_ps,
        _mm256_fmadd_ps,
        _mm256_add_ps,
        _mm256_set1_ps,
        _mm256_loadu_ps,
        ld8_f32,
        sc_f32,
        hsum256
    );
    dot_kernel!(
        dot_bf16_256,
        u16,
        "avx2,fma,f16c",
        __m256,
        8,
        _mm256_setzero_ps,
        _mm256_fmadd_ps,
        _mm256_add_ps,
        _mm256_set1_ps,
        _mm256_loadu_ps,
        ld8_bf16,
        sc_bf16,
        hsum256
    );
    dot_kernel!(
        dot_f16_256,
        u16,
        "avx2,fma,f16c",
        __m256,
        8,
        _mm256_setzero_ps,
        _mm256_fmadd_ps,
        _mm256_add_ps,
        _mm256_set1_ps,
        _mm256_loadu_ps,
        ld8_f16,
        sc_f16,
        hsum256
    );
    dot_kernel!(
        dot_fp8_256,
        u8,
        "avx2,fma,f16c",
        __m256,
        8,
        _mm256_setzero_ps,
        _mm256_fmadd_ps,
        _mm256_add_ps,
        _mm256_set1_ps,
        _mm256_loadu_ps,
        ld8_fp8,
        sc_fp8,
        hsum256
    );

    /// 4 weight rows x 4 activation rows: 16 accumulators, 8 loads per 16
    /// FMAs (vs 5 per 4 for a row-at-a-time dot).
    #[target_feature(enable = "avx512f,avx2,fma")]
    pub unsafe fn tile_4x4_512(w: [&[f32]; 4], x: &[&[f32]; 4]) -> [[f32; 4]; 4] {
        let k = w[0].len();
        unsafe {
            let mut acc = [[_mm512_setzero_ps(); 4]; 4];
            let mut i = 0;
            while i + 16 <= k {
                let xv = [
                    _mm512_loadu_ps(x[0].as_ptr().add(i)),
                    _mm512_loadu_ps(x[1].as_ptr().add(i)),
                    _mm512_loadu_ps(x[2].as_ptr().add(i)),
                    _mm512_loadu_ps(x[3].as_ptr().add(i)),
                ];
                for r in 0..4 {
                    let wv = _mm512_loadu_ps(w[r].as_ptr().add(i));
                    for c in 0..4 {
                        acc[r][c] = _mm512_fmadd_ps(wv, xv[c], acc[r][c]);
                    }
                }
                i += 16;
            }
            let mut out = [[0f32; 4]; 4];
            for r in 0..4 {
                for c in 0..4 {
                    let mut s = _mm512_reduce_add_ps(acc[r][c]);
                    for j in i..k {
                        s += w[r][j] * x[c][j];
                    }
                    out[r][c] = s;
                }
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(x: &[f32], m: usize, w: &[f32], n: usize, k: usize) -> Vec<f32> {
        let mut y = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                y[i * n + j] = (0..k).map(|kk| x[i * k + kk] * w[j * k + kk]).sum();
            }
        }
        y
    }

    fn data(len: usize, seed: f32) -> Vec<f32> {
        (0..len).map(|i| ((i as f32 + seed) * 0.618).sin()).collect()
    }

    fn check(got: &[f32], want: &[f32], tol: f32) {
        for (a, b) in got.iter().zip(want) {
            assert!((a - b).abs() < tol * (1.0 + b.abs()), "{a} vs {b}");
        }
    }

    #[test]
    fn all_dtypes_all_m() {
        let (n, k) = (37, 70);
        let wv = data(n * k, 2.0);
        let base = Tensor::from_f32(&wv, &[n, k]).unwrap();
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let w = base.to_dtype(dtype).unwrap();
            let wq = w.to_f32_vec().unwrap();
            for m in [1, 2, 3, 4, 5, 9] {
                let x = data(m * k, 1.0 + m as f32);
                check(&linear_f32(&x, m, &w).unwrap(), &naive(&x, m, &wq, n, k), 1e-4);
            }
        }
    }

    #[test]
    fn fp8_block_scaled() {
        let (n, k) = (130, 260); // partial blocks in both dims
        let raw = data(n * k, 4.0);
        let mut w = Tensor::from_f32(&raw, &[n, k]).unwrap().to_dtype(DType::F8E4M3).unwrap();
        let (sn, sk) = (n.div_ceil(128), k.div_ceil(128));
        let scales: Vec<f32> = (0..sn * sk).map(|i| 0.5 + i as f32 * 0.25).collect();
        w.set_scale(Tensor::from_f32(&scales, &[sn, sk]).unwrap()).unwrap();

        let decoded = w.to_f32_vec().unwrap();
        let mut deq = vec![0f32; n * k];
        for r in 0..n {
            for c in 0..k {
                deq[r * k + c] = decoded[r * k + c] * scales[(r / 128) * sk + c / 128];
            }
        }
        for m in [1, 3, 4, 7] {
            let x = data(m * k, 3.0);
            check(&linear_f32(&x, m, &w).unwrap(), &naive(&x, m, &deq, n, k), 1e-3);
        }
    }
}
