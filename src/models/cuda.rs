//! Moving a loaded model onto the GPU.

use crate::cuda::gpu;
use crate::cuda::moe::{DeviceExpert, GpuExperts, upload_weight};
use crate::dtype::DType;
use crate::error::Result;
use crate::models::qwen3::{Ffn, Qwen3};
use crate::tensor::{Device, Tensor};

impl Qwen3 {
    /// Upload the weights and switch activations to BF16 (or F16 for F16
    /// checkpoints). FP8 weights stay FP8.
    pub fn to_cuda(mut self) -> Result<Self> {
        let g = gpu()?;
        let act = match self.embed_tokens.dtype() {
            DType::F16 => DType::F16,
            _ => DType::BF16,
        };
        let f32_dev = |t: &Tensor| t.to_dtype(DType::F32)?.to_device(Device::Cuda);
        let tied = self.config.config.tie_word_embeddings
            || self.lm_head.shape() == self.embed_tokens.shape() && {
                // same storage => tied (the loader shares the tensor)
                std::ptr::eq(self.lm_head.bytes().as_ptr(), self.embed_tokens.bytes().as_ptr())
            };
        self.embed_tokens = self.embed_tokens.to_dtype(act)?.to_device(Device::Cuda)?;
        self.lm_head = if tied { self.embed_tokens.clone() } else { upload_weight(&self.lm_head, act)? };
        self.norm = f32_dev(&self.norm)?;

        for layer in &mut self.layers {
            layer.input_layernorm = f32_dev(&layer.input_layernorm)?;
            layer.post_attention_layernorm = f32_dev(&layer.post_attention_layernorm)?;
            let a = &mut layer.self_attn;
            a.q_proj = upload_weight(&a.q_proj, act)?;
            a.k_proj = upload_weight(&a.k_proj, act)?;
            a.v_proj = upload_weight(&a.v_proj, act)?;
            a.o_proj = upload_weight(&a.o_proj, act)?;
            a.q_norm = f32_dev(&a.q_norm)?;
            a.k_norm = f32_dev(&a.k_norm)?;
            if let Ffn::Dense(m) = &mut layer.ffn {
                m.gate_proj = upload_weight(&m.gate_proj, act)?;
                m.up_proj = upload_weight(&m.up_proj, act)?;
                m.down_proj = upload_weight(&m.down_proj, act)?;
            }
        }

        // MoE layers: router and every expert on the device
        for layer in &mut self.layers {
            if let Ffn::Sparse(moe) = &mut layer.ffn {
                let router = upload_weight(&moe.gate, act)?;
                let experts = moe.experts.iter().map(|e| DeviceExpert::upload(e, act)).collect::<Result<_>>()?;
                moe.gpu = Some((router, GpuExperts::Resident(experts)));
            }
        }

        self.device = Device::Cuda;
        self.act_dtype = act;
        g.synchronize()?;
        Ok(self)
    }
}
