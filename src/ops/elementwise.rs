use rayon::prelude::*;

use super::expect_f32;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

const CHUNK: usize = 4096;

#[inline]
pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `x += y` (same shape).
pub fn add_inplace(x: &mut Tensor, y: &Tensor) -> Result<()> {
    expect_f32(x, "add_inplace")?;
    expect_f32(y, "add_inplace")?;
    if x.numel() != y.numel() {
        return Err(Error::Shape(format!("add_inplace: {:?} vs {:?}", x.shape(), y.shape())));
    }
    let ys = y.as_slice::<f32>();
    x.as_mut_slice::<f32>()?.par_chunks_mut(CHUNK).enumerate().for_each(|(c, xs)| {
        for (a, b) in xs.iter_mut().zip(&ys[c * CHUNK..]) {
            *a += b;
        }
    });
    Ok(())
}

/// SwiGLU: `gate = silu(gate) * up`.
pub fn swiglu(gate: &mut Tensor, up: &Tensor) -> Result<()> {
    expect_f32(gate, "swiglu")?;
    expect_f32(up, "swiglu")?;
    if gate.numel() != up.numel() {
        return Err(Error::Shape(format!("swiglu: {:?} vs {:?}", gate.shape(), up.shape())));
    }
    let us = up.as_slice::<f32>();
    gate.as_mut_slice::<f32>()?.par_chunks_mut(CHUNK).enumerate().for_each(|(c, gs)| {
        for (g, u) in gs.iter_mut().zip(&us[c * CHUNK..]) {
            *g = silu(*g) * u;
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_swiglu() {
        let mut a = Tensor::from_f32(&[1.0, 2.0], &[2]).unwrap();
        add_inplace(&mut a, &Tensor::from_f32(&[0.5, -2.0], &[2]).unwrap()).unwrap();
        assert_eq!(a.as_slice::<f32>(), &[1.5, 0.0]);

        let mut g = Tensor::from_f32(&[0.0, 2.0], &[2]).unwrap();
        swiglu(&mut g, &Tensor::from_f32(&[3.0, 3.0], &[2]).unwrap()).unwrap();
        let v = g.as_slice::<f32>();
        assert_eq!(v[0], 0.0);
        assert!((v[1] - 3.0 * 2.0 / (1.0 + (-2f32).exp())).abs() < 1e-6);
    }
}
