use super::expect_f32;
use crate::cpu;
use crate::error::Result;
use crate::tensor::{Device, Tensor};

/// `y = x W^T` with `x: [..., K]` and `w: [N, K]`; returns `[..., N]`.
pub fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if x.device() == Device::Cuda {
        return crate::cuda::ops::linear(x, w);
    }
    expect_f32(x, "linear")?;
    let k = *x.shape().last().unwrap();
    let m = x.numel() / k.max(1);
    let y = cpu::linear::linear_f32(x.as_slice::<f32>(), m, w)?;
    let mut shape = x.shape().to_vec();
    *shape.last_mut().unwrap() = w.dim(0);
    debug_assert_eq!(x.device(), Device::Cpu);
    Tensor::from_f32(&y, &shape)
}
