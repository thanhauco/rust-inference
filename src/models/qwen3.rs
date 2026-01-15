//! Qwen3 decoder (dense and MoE variants share everything but the FFN).

use std::path::Path;

use crate::attention::PagedAttention;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::kvcache::KvCache;
use crate::models::config::{Architecture, ModelConfig, Qwen3Config};
use crate::models::{LanguageModel, LogitsMode};
use crate::ops::{self, RotaryEmbedding};
use crate::profile;
use crate::profiler::Section;
use crate::safetensors::ModelLoader;
use crate::tensor::{Device, Tensor};

pub struct Attention {
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub o_proj: Tensor,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
}

pub struct Mlp {
    pub gate_proj: Tensor,
    pub up_proj: Tensor,
    pub down_proj: Tensor,
}

pub enum Ffn {
    Dense(Mlp),
}

pub struct DecoderLayer {
    pub idx: usize,
    pub input_layernorm: Tensor,
    pub post_attention_layernorm: Tensor,
    pub self_attn: Attention,
    pub ffn: Ffn,
}

pub struct Qwen3 {
    config: ModelConfig,
    pub embed_tokens: Tensor,
    pub layers: Vec<DecoderLayer>,
    pub norm: Tensor,
    pub lm_head: Tensor,
    rope: RotaryEmbedding,
    attn: PagedAttention,
}

impl Attention {
    fn load(wl: &mut ModelLoader, p: &str) -> Result<Self> {
        Ok(Attention {
            q_proj: wl.load(&format!("{p}q_proj.weight"))?,
            k_proj: wl.load(&format!("{p}k_proj.weight"))?,
            v_proj: wl.load(&format!("{p}v_proj.weight"))?,
            o_proj: wl.load(&format!("{p}o_proj.weight"))?,
            q_norm: wl.load_as(&format!("{p}q_norm.weight"), DType::F32)?,
            k_norm: wl.load_as(&format!("{p}k_norm.weight"), DType::F32)?,
        })
    }

    fn forward(
        &self,
        h: &Tensor,
        cfg: &Qwen3Config,
        cos: &Tensor,
        sin: &Tensor,
        cache: &mut KvCache,
        attn: &PagedAttention,
    ) -> Result<Tensor> {
        let (b, s) = (h.dim(0), h.dim(1));
        let (nh, nkv, dh) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim());
        let (mut q, mut k, v) = profile!(
            Section::QkvProjection,
            (ops::linear(h, &self.q_proj)?, ops::linear(h, &self.k_proj)?, ops::linear(h, &self.v_proj)?)
        );
        profile!(Section::QkNorm, {
            ops::rms_norm(&mut q, &self.q_norm, cfg.rms_norm_eps, Some(dh))?;
            ops::rms_norm(&mut k, &self.k_norm, cfg.rms_norm_eps, Some(dh))?;
        });
        q.reshape(&[b, s, nh, dh])?;
        k.reshape(&[b, s, nkv, dh])?;
        let v = v.reshaped(&[b, s, nkv, dh])?;
        profile!(Section::PosEmbedding, {
            ops::apply_rope(&mut q, cos, sin)?;
            ops::apply_rope(&mut k, cos, sin)?;
        });
        profile!(Section::KvAppend, cache.append(&k, &v)?);
        let o = profile!(Section::Attention, attn.forward(&q, cache)?).reshaped(&[b, s, nh * dh])?;
        profile!(Section::OutProjection, ops::linear(&o, &self.o_proj))
    }
}

impl Mlp {
    fn load(wl: &mut ModelLoader, p: &str) -> Result<Self> {
        Ok(Mlp {
            gate_proj: wl.load(&format!("{p}gate_proj.weight"))?,
            up_proj: wl.load(&format!("{p}up_proj.weight"))?,
            down_proj: wl.load(&format!("{p}down_proj.weight"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut gate = ops::linear(x, &self.gate_proj)?;
        let up = ops::linear(x, &self.up_proj)?;
        ops::swiglu(&mut gate, &up)?;
        ops::linear(&gate, &self.down_proj)
    }
}

impl DecoderLayer {
    fn forward(
        &self,
        hidden: Tensor,
        cfg: &Qwen3Config,
        cos: &Tensor,
        sin: &Tensor,
        cache: &mut KvCache,
        attn: &PagedAttention,
    ) -> Result<Tensor> {
        let mut residual = hidden;
        let mut h = residual.deep_clone();
        profile!(Section::Norm, ops::rms_norm(&mut h, &self.input_layernorm, cfg.rms_norm_eps, None)?);
        let mut h = self.self_attn.forward(&h, cfg, cos, sin, cache, attn)?;
        // residual += attn_out; h = post_attention_layernorm(residual)
        profile!(
            Section::Norm,
            ops::add_rms_norm(&mut h, &mut residual, &self.post_attention_layernorm, cfg.rms_norm_eps)?
        );
        let mut out = match &self.ffn {
            Ffn::Dense(mlp) => profile!(Section::Mlp, mlp.forward(&h)?),
        };
        ops::add_inplace(&mut out, &residual)?;
        Ok(out)
    }
}

impl Qwen3 {
    pub fn from_pretrained(dir: impl AsRef<Path>) -> Result<Self> {
        let config = ModelConfig::from_pretrained(&dir)?;
        Self::load(config, dir)
    }

    pub fn load(config: ModelConfig, dir: impl AsRef<Path>) -> Result<Self> {
        let mut wl = ModelLoader::new(dir)?;
        let cfg = &config.config;
        let embed_tokens = wl.load("model.embed_tokens.weight")?;
        if embed_tokens.shape() != [cfg.vocab_size, cfg.hidden_size] {
            return Err(Error::Shape(format!(
                "embed_tokens is {:?}, config says [{}, {}]",
                embed_tokens.shape(),
                cfg.vocab_size,
                cfg.hidden_size
            )));
        }
        // Small Qwen3 models tie the output projection to the embeddings.
        let lm_head = if cfg.tie_word_embeddings || !wl.contains("lm_head.weight") {
            embed_tokens.clone()
        } else {
            wl.load("lm_head.weight")?
        };
        let norm = wl.load_as("model.norm.weight", DType::F32)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}.");
            if config.architecture == Architecture::Qwen3Moe && cfg.is_sparse_layer(i) {
                return Err(Error::Unsupported("Qwen3-MoE sparse layers are not implemented yet".into()));
            }
            let ffn = Ffn::Dense(Mlp::load(&mut wl, &format!("{p}mlp."))?);
            layers.push(DecoderLayer {
                idx: i,
                input_layernorm: wl.load_as(&format!("{p}input_layernorm.weight"), DType::F32)?,
                post_attention_layernorm: wl.load_as(&format!("{p}post_attention_layernorm.weight"), DType::F32)?,
                self_attn: Attention::load(&mut wl, &format!("{p}self_attn."))?,
                ffn,
            });
        }

        let rope = RotaryEmbedding::new(cfg.rope_theta, cfg.head_dim());
        let attn = PagedAttention::new(cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim());
        Ok(Qwen3 { config, embed_tokens, layers, norm, lm_head, rope, attn })
    }
}

impl LanguageModel for Qwen3 {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn new_kv_caches(&self, batch_size: usize, max_pages: usize, max_pages_per_seq: usize) -> Vec<KvCache> {
        let cfg = &self.config.config;
        (0..cfg.num_hidden_layers)
            .map(|_| {
                KvCache::new(
                    cfg.num_key_value_heads,
                    cfg.head_dim(),
                    max_pages,
                    batch_size,
                    max_pages_per_seq,
                    DType::F32,
                    Device::Cpu,
                )
            })
            .collect()
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        caches: &mut [KvCache],
        mode: LogitsMode,
    ) -> Result<Tensor> {
        let cfg = &self.config.config;
        if input_ids.ndim() != 2 || positions.shape() != input_ids.shape() {
            return Err(Error::Shape(format!(
                "input_ids {:?} / positions {:?} must both be [B, S]",
                input_ids.shape(),
                positions.shape()
            )));
        }
        if caches.len() != self.layers.len() {
            return Err(Error::Shape(format!("{} kv caches for {} layers", caches.len(), self.layers.len())));
        }
        let (b, s) = (input_ids.dim(0), input_ids.dim(1));
        let mut hidden = profile!(Section::Embedding, ops::embedding(input_ids, &self.embed_tokens)?);
        let (cos, sin) = profile!(Section::Rope, self.rope.cos_sin(positions)?);
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            hidden = layer.forward(hidden, cfg, &cos, &sin, cache, &self.attn)?;
        }
        if mode == LogitsMode::Last && s > 1 {
            let d = cfg.hidden_size;
            let all = hidden.as_slice::<f32>();
            let last: Vec<f32> = (0..b).flat_map(|bi| all[((bi * s) + s - 1) * d..][..d].iter().copied()).collect();
            hidden = Tensor::from_f32(&last, &[b, 1, d])?;
        }
        profile!(Section::Norm, ops::rms_norm(&mut hidden, &self.norm, cfg.rms_norm_eps, None)?);
        profile!(Section::LmHead, ops::linear(&hidden, &self.lm_head))
    }
}
