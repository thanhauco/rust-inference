use rayon::prelude::*;

use super::expect_f32;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::{Device, Tensor};

/// Rotary position embedding tables (HF "default" rope type).
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    pub head_dim: usize,
    pub inv_freq: Vec<f32>,
    pub attention_scaling: f32,
}

impl RotaryEmbedding {
    /// `inv_freq[i] = 1 / theta^(2i / head_dim)`, computed in f32 like
    /// `transformers` does so positions line up bit-for-bit.
    pub fn new(theta: f32, head_dim: usize) -> Self {
        let inv_freq = (0..head_dim / 2).map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32)).collect();
        RotaryEmbedding { head_dim, inv_freq, attention_scaling: 1.0 }
    }

    /// cos/sin tables `[*positions.shape, head_dim]` (F32). The second half
    /// repeats the first (`cat(freqs, freqs)`).
    pub fn cos_sin(&self, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        if positions.dtype() != DType::U32 {
            return Err(Error::Unsupported(format!("positions must be U32, got {}", positions.dtype())));
        }
        let mut shape = positions.shape().to_vec();
        shape.push(self.head_dim);
        let mut cos = Tensor::zeros(&shape, DType::F32, Device::Cpu);
        let mut sin = Tensor::zeros(&shape, DType::F32, Device::Cpu);
        let half = self.head_dim / 2;
        let pos = positions.as_slice::<u32>();
        let (cv, sv) = (cos.as_mut_slice::<f32>()?, sin.as_mut_slice::<f32>()?);
        for ((&p, c), s) in pos.iter().zip(cv.chunks_mut(self.head_dim)).zip(sv.chunks_mut(self.head_dim)) {
            for i in 0..half {
                let (sn, cs) = (p as f32 * self.inv_freq[i]).sin_cos();
                c[i] = cs * self.attention_scaling;
                c[i + half] = c[i];
                s[i] = sn * self.attention_scaling;
                s[i + half] = s[i];
            }
        }
        Ok((cos, sin))
    }
}

/// Rotate `x: [B, S, H, Dh]` in place with `cos`/`sin: [B, S, Dh]`
/// (rotate-half convention).
pub fn apply_rope(x: &mut Tensor, cos: &Tensor, sin: &Tensor) -> Result<()> {
    #[cfg(feature = "cuda")]
    if x.device() == Device::Cuda {
        return crate::cuda::ops::apply_rope(x, cos, sin);
    }
    expect_f32(x, "apply_rope")?;
    if x.ndim() != 4 {
        return Err(Error::Shape(format!("apply_rope expects [B,S,H,Dh], got {:?}", x.shape())));
    }
    let (b, s, h, dh) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3));
    if cos.shape() != [b, s, dh] || sin.shape() != [b, s, dh] {
        return Err(Error::Shape(format!("apply_rope: cos/sin {:?} for x {:?}", cos.shape(), x.shape())));
    }
    let half = dh / 2;
    let (cv, sv) = (cos.as_slice::<f32>(), sin.as_slice::<f32>());
    x.as_mut_slice::<f32>()?.par_chunks_mut(h * dh).enumerate().for_each(|(tok, heads)| {
        let c = &cv[tok * dh..(tok + 1) * dh];
        let sn = &sv[tok * dh..(tok + 1) * dh];
        for head in heads.chunks_mut(dh) {
            for d in 0..half {
                let (x1, x2) = (head[d], head[d + half]);
                head[d] = x1 * c[d] - x2 * sn[d];
                head[d + half] = x2 * c[d + half] + x1 * sn[d + half];
            }
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_zero_is_identity() {
        let rope = RotaryEmbedding::new(10000.0, 4);
        let pos = Tensor::from_u32(&[0, 3], &[1, 2]).unwrap();
        let (cos, sin) = rope.cos_sin(&pos).unwrap();
        let mut x = Tensor::from_f32(&[1., 2., 3., 4., 1., 2., 3., 4.], &[1, 2, 1, 4]).unwrap();
        apply_rope(&mut x, &cos, &sin).unwrap();
        let v = x.as_slice::<f32>();
        assert_eq!(&v[..4], &[1., 2., 3., 4.]);
        // position 3, frequency 1: rotate (1, 3) by 3 radians
        let (s3, c3) = 3f32.sin_cos();
        assert!((v[4] - (1.0 * c3 - 3.0 * s3)).abs() < 1e-6);
        assert!((v[6] - (3.0 * c3 + 1.0 * s3)).abs() < 1e-6);
    }

    #[test]
    fn preserves_norm() {
        let rope = RotaryEmbedding::new(1e6, 8);
        let pos = Tensor::from_u32(&[12345], &[1, 1]).unwrap();
        let (cos, sin) = rope.cos_sin(&pos).unwrap();
        let vals: Vec<f32> = (0..16).map(|i| i as f32 - 7.5).collect();
        let mut x = Tensor::from_f32(&vals, &[1, 1, 2, 8]).unwrap();
        apply_rope(&mut x, &cos, &sin).unwrap();
        let n0: f32 = vals.iter().map(|v| v * v).sum();
        let n1: f32 = x.as_slice::<f32>().iter().map(|v| v * v).sum();
        assert!((n0 - n1).abs() < 1e-3);
    }
}
