//! Block-wise FP8 (E4M3) weight quantization in the format of the Qwen3 FP8
//! checkpoints: every `block x block` tile of a `[N, K]` matrix gets one f32
//! scale (stored as `{name}_scale_inv`, i.e. the factor to multiply by when
//! dequantizing), chosen so the tile's largest magnitude maps to 448.

use rayon::prelude::*;

use crate::dtype::{self, DType, FP8_E4M3_MAX};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Whether a checkpoint tensor should be quantized: the linear projections
/// of attention, MLPs and experts. Embeddings, the LM head, norms and the
/// MoE router stay in their original precision (as in Qwen's FP8 releases).
pub fn should_quantize(name: &str, shape: &[usize]) -> bool {
    shape.len() == 2 && name.ends_with("_proj.weight") && !name.contains("embed_tokens") && !name.starts_with("lm_head")
}

/// Quantize a 2-D float tensor. Returns the E4M3 weight (with its block
/// scale attached) and the `[ceil(N/block), ceil(K/block)]` scale tensor.
pub fn quantize_fp8(w: &Tensor, block: usize) -> Result<(Tensor, Tensor)> {
    if w.ndim() != 2 || !w.dtype().is_float() || w.dtype() == DType::F8E4M3 {
        return Err(Error::Unsupported(format!("quantize {:?} {}", w.shape(), w.dtype())));
    }
    let (n, k) = (w.dim(0), w.dim(1));
    let vals = w.to_f32_vec()?;
    let (sn, sk) = (n.div_ceil(block), k.div_ceil(block));

    let mut scales = vec![0f32; sn * sk];
    scales.par_chunks_mut(sk).enumerate().for_each(|(bi, srow)| {
        for (bj, s) in srow.iter_mut().enumerate() {
            let mut amax = 0f32;
            for r in bi * block..((bi + 1) * block).min(n) {
                for &v in &vals[r * k + bj * block..r * k + ((bj + 1) * block).min(k)] {
                    amax = amax.max(v.abs());
                }
            }
            *s = if amax > 0.0 { amax / FP8_E4M3_MAX } else { 1.0 };
        }
    });

    let mut q = vec![0u8; n * k];
    q.par_chunks_mut(k).enumerate().for_each(|(r, row)| {
        let srow = &scales[(r / block) * sk..(r / block + 1) * sk];
        for (c, out) in row.iter_mut().enumerate() {
            *out = dtype::f32_to_fp8_e4m3(vals[r * k + c] / srow[c / block]);
        }
    });

    let scale = Tensor::from_f32(&scales, &[sn, sk])?;
    let mut qt = Tensor::from_slice(&q, &[n, k], DType::F8E4M3)?;
    qt.set_scale(scale.clone())?;
    Ok((qt, scale))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::linear::linear_f32;

    #[test]
    fn selection_rules() {
        assert!(should_quantize("model.layers.0.self_attn.q_proj.weight", &[8, 8]));
        assert!(should_quantize("model.layers.3.mlp.experts.7.down_proj.weight", &[8, 8]));
        assert!(!should_quantize("model.layers.3.mlp.gate.weight", &[8, 8]));
        assert!(!should_quantize("model.embed_tokens.weight", &[8, 8]));
        assert!(!should_quantize("lm_head.weight", &[8, 8]));
        assert!(!should_quantize("model.layers.0.input_layernorm.weight", &[8]));
    }

    #[test]
    fn roundtrip_error_is_small() {
        let (n, k) = (200, 300); // partial blocks in both dimensions
        let vals: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.37).sin() * (1.0 + (i % 7) as f32)).collect();
        let w = Tensor::from_f32(&vals, &[n, k]).unwrap();
        let (q, scale) = quantize_fp8(&w, 128).unwrap();
        assert_eq!(scale.shape(), &[2, 3]);

        // dequantize by running the identity through the FP8 linear kernel
        let mut err = 0f64;
        let mut norm = 0f64;
        for c in [0, 1, 127, 128, 299] {
            let mut e = vec![0f32; k];
            e[c] = 1.0;
            let col = linear_f32(&e, 1, &q).unwrap();
            for r in 0..n {
                err += ((col[r] - vals[r * k + c]) as f64).powi(2);
                norm += (vals[r * k + c] as f64).powi(2);
            }
        }
        // a 3-bit mantissa gives a few percent RMS error
        let rel = (err / norm).sqrt();
        assert!(rel < 0.04, "relative error {rel}");
    }
}
