//! Qwen3-MoE sparse block: router + top-k experts.

use crate::cpu::expert::{self, ExpertWeights};
use crate::error::{Error, Result};
use crate::models::config::Qwen3Config;
use crate::ops::{self, MoeRouting};
use crate::profile;
use crate::profiler::Section;
use crate::safetensors::ModelLoader;
use crate::tensor::Tensor;

pub struct SparseMoe {
    /// router, `[num_experts, hidden]`
    pub gate: Tensor,
    pub experts: Vec<ExpertWeights>,
    pub top_k: usize,
    pub norm_topk_prob: bool,
    /// device router and experts, once the model is moved to the GPU
    #[cfg(feature = "cuda")]
    pub gpu: Option<(Tensor, crate::cuda::moe::GpuExperts)>,
}

impl SparseMoe {
    pub fn load(wl: &mut ModelLoader, p: &str, cfg: &Qwen3Config) -> Result<Self> {
        let gate = wl.load(&format!("{p}gate.weight"))?;
        if gate.shape() != [cfg.num_experts, cfg.hidden_size] {
            return Err(Error::Shape(format!("{p}gate.weight is {:?}", gate.shape())));
        }
        let experts = (0..cfg.num_experts)
            .map(|e| {
                let ep = format!("{p}experts.{e}.");
                ExpertWeights::new(
                    wl.load(&format!("{ep}gate_proj.weight"))?,
                    wl.load(&format!("{ep}up_proj.weight"))?,
                    wl.load(&format!("{ep}down_proj.weight"))?,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(SparseMoe {
            gate,
            experts,
            top_k: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
            #[cfg(feature = "cuda")]
            gpu: None,
        })
    }

    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }

    /// `h: [B, S, D]` -> `[B, S, D]`.
    pub fn forward(&self, h: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if h.device() == crate::tensor::Device::Cuda {
            let (router, experts) =
                self.gpu.as_ref().ok_or_else(|| Error::Cuda("MoE layer has not been moved to the GPU".into()))?;
            return profile!(
                Section::Experts,
                crate::cuda::moe::forward(h, router, self.top_k, self.norm_topk_prob, self.num_experts(), experts)
            );
        }
        let d = *h.shape().last().unwrap();
        let (ids, weights) = profile!(Section::Router, {
            let logits = ops::linear(h, &self.gate)?;
            ops::topk_softmax(&logits, self.top_k, self.norm_topk_prob)?
        });
        let routing = MoeRouting::new(&ids, self.top_k, self.num_experts())?;
        let gathered = profile!(Section::ExpertScatter, routing.gather(h.as_slice::<f32>(), d));

        // every active expert goes to the executor; workers steal row
        // chunks across all of them
        let expert_out = profile!(Section::Experts, {
            let exec = expert::global();
            let jobs = routing
                .active_experts()
                .map(|e| {
                    let rows = routing.expert_offsets[e] * d..routing.expert_offsets[e + 1] * d;
                    let x = gathered[rows.clone()].to_vec();
                    Ok((rows, exec.submit(self.experts[e].clone(), x, routing.tokens_for(e), false)?))
                })
                .collect::<Result<Vec<_>>>()?;
            exec.wait_all();
            let mut out = vec![0f32; gathered.len()];
            for (rows, job) in jobs {
                out[rows].copy_from_slice(&job.take()?);
            }
            out
        });

        let y = profile!(Section::ExpertScatter, routing.scatter(&expert_out, &weights, d));
        Tensor::from_f32(&y, h.shape())
    }
}
