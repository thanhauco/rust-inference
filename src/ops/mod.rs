//! Tensor operations used by the models.
//!
//! These are device-dispatching free functions in the spirit of the C++
//! `matmul(out, a, b)` / `rmsnorm(x, w)` API. Activations are F32 on the CPU;
//! weights may be any float dtype the kernels understand.

mod elementwise;
mod embedding;
mod linear;
mod moe;
mod norm;
mod reduce;
mod rope;

pub use elementwise::{add_inplace, swiglu};
pub use embedding::embedding;
pub use linear::linear;
pub use moe::MoeRouting;
pub use norm::{add_rms_norm, rms_norm};
pub use reduce::{argmax_last, topk_softmax};
pub use rope::{RotaryEmbedding, apply_rope};

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Decode a small weight vector (norm gains etc.) to f32.
pub(crate) fn weight_f32(w: &Tensor) -> Result<std::borrow::Cow<'_, [f32]>> {
    if w.dtype() == DType::F32 {
        Ok(std::borrow::Cow::Borrowed(w.as_slice::<f32>()))
    } else {
        Ok(std::borrow::Cow::Owned(w.to_f32_vec()?))
    }
}

pub(crate) fn expect_f32(t: &Tensor, what: &str) -> Result<()> {
    if t.dtype() != DType::F32 {
        return Err(Error::Unsupported(format!("{what}: CPU activations must be F32, got {}", t.dtype())));
    }
    Ok(())
}
