//! garlic: a small LLM inference engine for Qwen3-family models.

pub mod dtype;
pub mod error;

pub use dtype::DType;
pub use error::{Error, Result};
