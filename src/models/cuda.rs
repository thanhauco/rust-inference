//! Moving a loaded model onto the GPU.

use std::sync::Arc;

use crate::cuda::moe::{DeviceExpert, GpuExperts, OffloadOptions, OffloadRuntime, upload_weight};
use crate::cuda::{err, gpu};
use crate::dtype::DType;
use crate::error::Result;
use crate::models::qwen3::{Ffn, Qwen3};
use crate::moe_cache::Policy;
use crate::tensor::{Device, Tensor};

/// Headroom left for activations, the KV cache and cuBLAS workspaces when
/// deciding whether all experts fit.
const RESERVE_BYTES: usize = 3 << 30;

#[derive(Debug, Clone, Default)]
pub struct CudaOptions {
    /// Expert offloading; `None` decides from free device memory (all
    /// experts resident if they fit, otherwise offload with as many slots as
    /// fit).
    pub offload: Option<OffloadOptions>,
}

impl Qwen3 {
    /// Upload the weights and switch activations to BF16 (or F16 for F16
    /// checkpoints). FP8 weights stay FP8.
    pub fn to_cuda(mut self, opts: &CudaOptions) -> Result<Self> {
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

        // MoE placement
        let sparse: Vec<usize> =
            (0..self.layers.len()).filter(|&i| matches!(self.layers[i].ffn, Ffn::Sparse(_))).collect();
        let mut expert_bytes = 0usize;
        let mut max_expert = 0usize;
        for &i in &sparse {
            if let Ffn::Sparse(moe) = &self.layers[i].ffn {
                for e in &moe.experts {
                    let b = [&e.gate, &e.up, &e.down]
                        .iter()
                        .map(|t| if t.dtype() == DType::F8E4M3 { t.numel() } else { t.numel() * act.size() })
                        .sum::<usize>();
                    expert_bytes += b;
                    max_expert = max_expert.max(b);
                }
            }
        }
        let offload = match &opts.offload {
            Some(o) => Some(o.clone()),
            None if sparse.is_empty() => None,
            None => {
                let (free, _) = g.ctx.mem_get_info().map_err(err)?;
                if expert_bytes + RESERVE_BYTES <= free {
                    None
                } else {
                    let slots = free.saturating_sub(RESERVE_BYTES) / max_expert.max(1);
                    Some(OffloadOptions {
                        slots: slots.max(2 * self.config.config.num_experts_per_tok),
                        resident_layers: 0,
                        cpu_threshold: 8,
                        policy: Policy::LayerDistance { num_layers: self.layers.len() },
                        prefetch: true,
                    })
                }
            }
        };

        let mut routers: Vec<Option<Tensor>> = vec![None; self.layers.len()];
        for &i in &sparse {
            if let Ffn::Sparse(moe) = &self.layers[i].ffn {
                routers[i] = Some(upload_weight(&moe.gate, act)?);
            }
        }
        match offload {
            None => {
                for &i in &sparse {
                    if let Ffn::Sparse(moe) = &mut self.layers[i].ffn {
                        let experts =
                            moe.experts.iter().map(|e| DeviceExpert::upload(e, act)).collect::<Result<_>>()?;
                        moe.gpu = Some((routers[i].clone().unwrap(), GpuExperts::Resident(experts)));
                    }
                }
            }
            Some(o) => {
                let (resident, offloaded) = sparse.split_at(o.resident_layers.min(sparse.len()));
                let groups: Vec<(usize, &[crate::cpu::expert::ExpertWeights])> = offloaded
                    .iter()
                    .filter_map(|&i| match &self.layers[i].ffn {
                        Ffn::Sparse(moe) => Some((i, moe.experts.as_slice())),
                        Ffn::Dense(_) => None,
                    })
                    .collect();
                let rt: Arc<OffloadRuntime> = OffloadRuntime::new(&groups, routers.clone(), act, o)?;
                for &i in resident {
                    if let Ffn::Sparse(moe) = &mut self.layers[i].ffn {
                        let experts =
                            moe.experts.iter().map(|e| DeviceExpert::upload(e, act)).collect::<Result<_>>()?;
                        moe.gpu = Some((routers[i].clone().unwrap(), GpuExperts::Resident(experts)));
                    }
                }
                for &i in offloaded {
                    if let Ffn::Sparse(moe) = &mut self.layers[i].ffn {
                        moe.gpu = Some((
                            routers[i].clone().unwrap(),
                            GpuExperts::Offloaded { layer: i, runtime: rt.clone() },
                        ));
                    }
                }
            }
        }

        self.device = Device::Cuda;
        self.act_dtype = act;
        g.synchronize()?;
        Ok(self)
    }
}
