use rayon::prelude::*;

use super::{expect_f32, weight_f32};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

fn rms_scale(row: &[f32], eps: f32) -> f32 {
    let ss: f32 = row.iter().map(|v| v * v).sum();
    1.0 / (ss / row.len() as f32 + eps).sqrt()
}

/// In-place RMSNorm over consecutive groups of `group` elements (the last
/// dimension, or `head_dim` for the per-head q/k norms).
pub fn rms_norm(x: &mut Tensor, weight: &Tensor, eps: f32, group: Option<usize>) -> Result<()> {
    #[cfg(feature = "cuda")]
    if x.device() == crate::tensor::Device::Cuda {
        return crate::cuda::ops::rms_norm(x, weight, eps, group.unwrap_or(*x.shape().last().unwrap()));
    }
    expect_f32(x, "rms_norm")?;
    let group = group.unwrap_or_else(|| *x.shape().last().unwrap());
    if weight.numel() != group || !x.numel().is_multiple_of(group) {
        return Err(Error::Shape(format!("rms_norm: x {:?}, weight {:?}, group {group}", x.shape(), weight.shape())));
    }
    let w = weight_f32(weight)?;
    x.as_mut_slice::<f32>()?.par_chunks_mut(group).for_each(|row| {
        let inv = rms_scale(row, eps);
        for (v, g) in row.iter_mut().zip(w.iter()) {
            *v = *v * inv * g;
        }
    });
    Ok(())
}

/// Fused residual add + norm: `residual += x; x = rmsnorm(residual) * w`.
pub fn add_rms_norm(x: &mut Tensor, residual: &mut Tensor, weight: &Tensor, eps: f32) -> Result<()> {
    #[cfg(feature = "cuda")]
    if x.device() == crate::tensor::Device::Cuda {
        return crate::cuda::ops::add_rms_norm(x, residual, weight, eps);
    }
    expect_f32(x, "add_rms_norm")?;
    expect_f32(residual, "add_rms_norm")?;
    if x.shape() != residual.shape() {
        return Err(Error::Shape(format!("add_rms_norm: {:?} vs {:?}", x.shape(), residual.shape())));
    }
    let dim = *x.shape().last().unwrap();
    if weight.numel() != dim {
        return Err(Error::Shape(format!("add_rms_norm: weight {:?} for dim {dim}", weight.shape())));
    }
    let w = weight_f32(weight)?;
    let xs = x.as_mut_slice::<f32>()?;
    let rs = residual.as_mut_slice::<f32>()?;
    xs.par_chunks_mut(dim).zip(rs.par_chunks_mut(dim)).for_each(|(xr, rr)| {
        for (r, v) in rr.iter_mut().zip(xr.iter()) {
            *r += v;
        }
        let inv = rms_scale(rr, eps);
        for ((v, r), g) in xr.iter_mut().zip(rr.iter()).zip(w.iter()) {
            *v = r * inv * g;
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalises_rows_and_groups() {
        let mut x = Tensor::from_f32(&[3.0, 4.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], &[2, 4]).unwrap();
        let w = Tensor::from_f32(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
        rms_norm(&mut x, &w, 0.0, None).unwrap();
        let rms = (25.0f32 / 4.0).sqrt();
        let v = x.as_slice::<f32>();
        assert!((v[0] - 3.0 / rms).abs() < 1e-6 && (v[4] - 1.0).abs() < 1e-6);

        let mut h = Tensor::from_f32(&[2.0, 2.0, 5.0, 5.0], &[1, 4]).unwrap();
        let w2 = Tensor::from_f32(&[1.0, 2.0], &[2]).unwrap();
        rms_norm(&mut h, &w2, 0.0, Some(2)).unwrap();
        assert_eq!(h.as_slice::<f32>(), &[1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn fused_matches_unfused() {
        let a = [0.5f32, -1.0, 2.0, 0.25];
        let r = [1.0f32, 1.0, -1.0, 0.0];
        let w = Tensor::from_f32(&[1.0, 0.5, 2.0, 1.0], &[4]).unwrap();
        let mut x = Tensor::from_f32(&a, &[1, 4]).unwrap();
        let mut res = Tensor::from_f32(&r, &[1, 4]).unwrap();
        add_rms_norm(&mut x, &mut res, &w, 1e-6).unwrap();

        let sum: Vec<f32> = a.iter().zip(&r).map(|(p, q)| p + q).collect();
        let mut want = Tensor::from_f32(&sum, &[1, 4]).unwrap();
        rms_norm(&mut want, &w, 1e-6, None).unwrap();
        assert_eq!(res.as_slice::<f32>(), sum.as_slice());
        assert_eq!(x.as_slice::<f32>(), want.as_slice::<f32>());
    }
}
