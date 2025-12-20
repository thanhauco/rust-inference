//! garlic: a small LLM inference engine for Qwen3-family models.

pub mod cpu;
pub mod dtype;
pub mod error;
pub mod ops;
pub mod safetensors;
pub mod tensor;
pub mod tokenizer;

pub use dtype::DType;
pub use error::{Error, Result};
pub use tensor::{Device, Tensor};
