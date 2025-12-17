//! `y = x W^T` on the CPU for F32 / BF16 / F16 / block-scaled FP8 weights.
//!
//! Work is split over output features (rows of W) so every weight row is
//! streamed from memory once per call: each row is decoded to f32 into a
//! small scratch buffer and then dotted against up to four activation rows at
//! a time. Decode is memory-bound, so the extra pass through L1 is cheap.

use rayon::prelude::*;

use super::simd;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Rows of W per rayon task.
const ROWS_PER_TASK: usize = 16;
/// Default FP8 quantization block (Qwen3 FP8 uses `weight_block_size: [128, 128]`).
pub const FP8_BLOCK: usize = 128;

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
        // The usual case: 128x128 blocks with a partial last block.
        if sn == n.div_ceil(FP8_BLOCK) && sk == k.div_ceil(FP8_BLOCK) {
            return Ok(BlockLayout { block_n: FP8_BLOCK, block_k: FP8_BLOCK, scale_cols: sk });
        }
        // Otherwise assume uniform blocks (covers per-tensor [1, 1] scales).
        Ok(BlockLayout { block_n: n.div_ceil(sn), block_k: k.div_ceil(sk), scale_cols: sk })
    }
}

/// Decode row `n` of `w` (row length `k`) into `dst`, applying FP8 scales.
fn decode_row(w: &Tensor, layout: Option<(&BlockLayout, &[f32])>, n: usize, k: usize, dst: &mut [f32]) {
    match w.dtype() {
        DType::F32 => dst.copy_from_slice(&w.as_slice::<f32>()[n * k..(n + 1) * k]),
        DType::BF16 => simd::decode_bf16(&w.as_slice::<u16>()[n * k..(n + 1) * k], dst),
        DType::F16 => simd::decode_f16(&w.as_slice::<u16>()[n * k..(n + 1) * k], dst),
        DType::F8E4M3 => {
            let row = &w.as_slice::<u8>()[n * k..(n + 1) * k];
            match layout {
                Some((l, scales)) => {
                    let srow = &scales[(n / l.block_n) * l.scale_cols..];
                    for (g, (src, out)) in row.chunks(l.block_k).zip(dst.chunks_mut(l.block_k)).enumerate() {
                        simd::decode_fp8(src, out, srow[g]);
                    }
                }
                None => simd::decode_fp8(row, dst, 1.0),
            }
        }
        _ => unreachable!("checked by caller"),
    }
}

/// `x: [M, K]` (row-major f32), `w: [N, K]` -> `y: [M, N]`.
pub fn linear_f32(x: &[f32], m: usize, w: &Tensor) -> Result<Vec<f32>> {
    if w.ndim() != 2 {
        return Err(Error::Shape(format!("linear weight must be 2-D, got {:?}", w.shape())));
    }
    let (n, k) = (w.dim(0), w.dim(1));
    if x.len() != m * k {
        return Err(Error::Shape(format!("linear: x has {} elements, expected {m}x{k}", x.len())));
    }
    if !matches!(w.dtype(), DType::F32 | DType::BF16 | DType::F16 | DType::F8E4M3) {
        return Err(Error::Unsupported(format!("linear weight dtype {}", w.dtype())));
    }
    let layout = match (w.dtype(), w.scale()) {
        (DType::F8E4M3, Some(s)) => Some((BlockLayout::new(n, k, s.shape())?, s.as_slice::<f32>())),
        _ => None,
    };
    let layout_ref = layout.as_ref().map(|(l, s)| (l, *s));

    // yt[n][m]: each task owns a contiguous block of output features.
    let mut yt = vec![0f32; n * m];
    yt.par_chunks_mut(ROWS_PER_TASK * m).enumerate().for_each(|(task, out)| {
        let n0 = task * ROWS_PER_TASK;
        let rows = out.len() / m;
        let mut scratch = vec![0f32; k];
        for r in 0..rows {
            let row = n0 + r;
            let wrow: &[f32] = if w.dtype() == DType::F32 {
                &w.as_slice::<f32>()[row * k..(row + 1) * k]
            } else {
                decode_row(w, layout_ref, row, k, &mut scratch);
                &scratch
            };
            let o = &mut out[r * m..(r + 1) * m];
            let mut t = 0;
            while t + 4 <= m {
                let xs = [
                    &x[t * k..(t + 1) * k],
                    &x[(t + 1) * k..(t + 2) * k],
                    &x[(t + 2) * k..(t + 3) * k],
                    &x[(t + 3) * k..(t + 4) * k],
                ];
                o[t..t + 4].copy_from_slice(&simd::dot_mr::<4>(wrow, &xs));
                t += 4;
            }
            while t < m {
                o[t] = simd::dot(wrow, &x[t * k..(t + 1) * k]);
                t += 1;
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

    #[test]
    fn f32_and_bf16_weights() {
        let (m, n, k) = (5, 37, 70);
        let x = data(m * k, 1.0);
        let wv = data(n * k, 2.0);
        let w = Tensor::from_f32(&wv, &[n, k]).unwrap();
        let want = naive(&x, m, &wv, n, k);
        let got = linear_f32(&x, m, &w).unwrap();
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-4);
        }
        let wb = w.to_dtype(DType::BF16).unwrap();
        let wb_vals = wb.to_f32_vec().unwrap();
        let want = naive(&x, m, &wb_vals, n, k);
        let got = linear_f32(&x, m, &wb).unwrap();
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn fp8_block_scaled() {
        let (m, n, k) = (3, 130, 260); // partial blocks in both dims
        let x = data(m * k, 3.0);
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
        let want = naive(&x, m, &deq, n, k);
        let got = linear_f32(&x, m, &w).unwrap();
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-3 * (1.0 + b.abs()), "{a} vs {b}");
        }
    }
}
