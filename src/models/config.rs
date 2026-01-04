//! `config.json` parsing for the supported architectures.

use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    #[serde(default)]
    pub quant_method: Option<String>,
    #[serde(default)]
    pub weight_block_size: Option<Vec<usize>>,
}

/// Fields shared by Qwen3 and Qwen3-MoE. The MoE-only fields default to a
/// dense model.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,

    // --- MoE ---
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_one")]
    pub decoder_sparse_step: usize,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,

    #[serde(default)]
    pub quantization_config: Option<QuantizationConfig>,
}

fn default_max_pos() -> usize {
    40960
}
fn default_eps() -> f32 {
    1e-6
}
fn default_theta() -> f32 {
    1_000_000.0
}
fn default_true() -> bool {
    true
}
fn default_one() -> usize {
    1
}

impl Qwen3Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Whether decoder layer `i` uses the sparse MoE block (HF semantics:
    /// not listed in `mlp_only_layers` and on a `decoder_sparse_step` stride).
    pub fn is_sparse_layer(&self, i: usize) -> bool {
        self.num_experts > 0
            && !self.mlp_only_layers.contains(&i)
            && (i + 1).is_multiple_of(self.decoder_sparse_step.max(1))
    }

    fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(Error::Config(m));
        if self.num_attention_heads == 0 || self.num_key_value_heads == 0 {
            return bad("attention heads must be non-zero".into());
        }
        if !self.num_attention_heads.is_multiple_of(self.num_key_value_heads) {
            return bad(format!(
                "{} attention heads not divisible by {} kv heads",
                self.num_attention_heads, self.num_key_value_heads
            ));
        }
        if !self.head_dim().is_multiple_of(2) {
            return bad(format!("head_dim {} must be even for rotary embeddings", self.head_dim()));
        }
        if self.num_experts > 0 && (self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_experts) {
            return bad(format!("top-{} of {} experts", self.num_experts_per_tok, self.num_experts));
        }
        if let Some(rs) = &self.rope_scaling {
            let kind = rs.get("rope_type").or_else(|| rs.get("type")).and_then(|v| v.as_str());
            if !matches!(kind, None | Some("default")) {
                return bad(format!("rope_scaling {rs} is not supported"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Qwen3,
    Qwen3Moe,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: Architecture,
    pub config: Qwen3Config,
}

#[derive(Deserialize)]
struct Header {
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    model_type: Option<String>,
}

impl ModelConfig {
    /// Read `config.json` from a checkpoint directory (the C++ `AutoConfig`).
    pub fn from_pretrained(dir: impl AsRef<Path>) -> Result<Self> {
        let path = dir.as_ref().join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| Error::io(&path, e))?;
        let header: Header = serde_json::from_str(&text).map_err(|e| Error::json(&path, e))?;
        let arch = header.architectures.first().map(String::as_str).or(header.model_type.as_deref());
        let architecture = match arch {
            Some(a) if a.contains("Qwen3MoeForCausalLM") || a == "qwen3_moe" => Architecture::Qwen3Moe,
            Some(a) if a.contains("Qwen3ForCausalLM") || a == "qwen3" => Architecture::Qwen3,
            other => return Err(Error::Unsupported(format!("architecture {other:?}"))),
        };
        let config: Qwen3Config = serde_json::from_str(&text).map_err(|e| Error::json(&path, e))?;
        if architecture == Architecture::Qwen3Moe && config.num_experts == 0 {
            return Err(Error::Config("Qwen3Moe config without num_experts".into()));
        }
        config.validate()?;
        Ok(ModelConfig { architecture, config })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_layer_selection() {
        let mut c: Qwen3Config = serde_json::from_str(
            r#"{"vocab_size": 10, "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 4,
                "num_attention_heads": 2, "num_key_value_heads": 1, "num_experts": 4,
                "num_experts_per_tok": 2, "moe_intermediate_size": 4, "mlp_only_layers": [0]}"#,
        )
        .unwrap();
        assert_eq!(c.head_dim(), 4);
        assert_eq!((0..4).map(|i| c.is_sparse_layer(i)).collect::<Vec<_>>(), vec![false, true, true, true]);
        c.decoder_sparse_step = 2;
        assert_eq!((0..4).map(|i| c.is_sparse_layer(i)).collect::<Vec<_>>(), vec![false, true, false, true]);
        assert!(c.validate().is_ok());
        c.rope_scaling = Some(serde_json::json!({"rope_type": "yarn", "factor": 4.0}));
        assert!(c.validate().is_err());
    }
}
