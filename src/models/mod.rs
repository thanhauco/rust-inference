//! Model definitions.

pub mod config;
pub mod moe;
pub mod qwen3;

use std::path::Path;

pub use config::{Architecture, ModelConfig, Qwen3Config};
pub use qwen3::Qwen3;

use crate::error::Result;
use crate::kvcache::KvCache;
use crate::tensor::Tensor;

/// Which positions to produce logits for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogitsMode {
    /// Every input position, `[B, S, vocab]`.
    All,
    /// Only the last position of each sequence, `[B, 1, vocab]`.
    Last,
}

pub trait LanguageModel: Send + Sync {
    fn config(&self) -> &ModelConfig;

    /// One KV cache per decoder layer.
    fn new_kv_caches(&self, batch_size: usize, max_pages: usize, max_pages_per_seq: usize) -> Vec<KvCache>;

    /// Run `input_ids` / `positions` (both `[B, S]`, U32) through the model,
    /// appending to `caches`, and return logits.
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        caches: &mut [KvCache],
        mode: LogitsMode,
    ) -> Result<Tensor>;
}

/// Load whichever supported architecture `dir/config.json` describes.
pub fn from_pretrained(dir: impl AsRef<Path>) -> Result<Box<dyn LanguageModel>> {
    let config = ModelConfig::from_pretrained(&dir)?;
    Ok(match config.architecture {
        Architecture::Qwen3 | Architecture::Qwen3Moe => Box::new(Qwen3::load(config, dir)?),
    })
}
