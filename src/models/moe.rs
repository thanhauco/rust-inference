//! Qwen3-MoE sparse block: router + top-k experts.

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::models::config::Qwen3Config;
use crate::models::qwen3::Mlp;
use crate::ops::{self, MoeRouting};
use crate::profile;
use crate::profiler::Section;
use crate::safetensors::ModelLoader;
use crate::tensor::Tensor;

pub struct SparseMoe {
    /// router, `[num_experts, hidden]`
    pub gate: Tensor,
    pub experts: Vec<Mlp>,
    pub top_k: usize,
    pub norm_topk_prob: bool,
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
                Ok(Mlp {
                    gate_proj: wl.load(&format!("{ep}gate_proj.weight"))?,
                    up_proj: wl.load(&format!("{ep}up_proj.weight"))?,
                    down_proj: wl.load(&format!("{ep}down_proj.weight"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(SparseMoe { gate, experts, top_k: cfg.num_experts_per_tok, norm_topk_prob: cfg.norm_topk_prob })
    }

    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }

    /// `h: [B, S, D]` -> `[B, S, D]`.
    pub fn forward(&self, h: &Tensor) -> Result<Tensor> {
        let d = *h.shape().last().unwrap();
        let (ids, weights) = profile!(Section::Router, {
            let logits = ops::linear(h, &self.gate)?;
            ops::topk_softmax(&logits, self.top_k, self.norm_topk_prob)?
        });
        let routing = MoeRouting::new(&ids, self.top_k, self.num_experts())?;
        let gathered = profile!(Section::ExpertScatter, routing.gather(h.as_slice::<f32>(), d));

        // carve the output into one disjoint block per active expert
        let mut expert_out = vec![0f32; gathered.len()];
        let mut jobs = Vec::new();
        let mut rest = expert_out.as_mut_slice();
        for e in 0..self.num_experts() {
            let (mine, tail) = rest.split_at_mut(routing.tokens_for(e) * d);
            rest = tail;
            if !mine.is_empty() {
                let x = &gathered[routing.expert_offsets[e] * d..routing.expert_offsets[e + 1] * d];
                jobs.push((e, x, mine));
            }
        }
        profile!(
            Section::Experts,
            jobs.into_par_iter().try_for_each(|(e, x, out)| -> Result<()> {
                let n = x.len() / d;
                let y = self.experts[e].forward(&Tensor::from_f32(x, &[n, d])?)?;
                out.copy_from_slice(y.as_slice::<f32>());
                Ok(())
            })?
        );

        let y = profile!(Section::ExpertScatter, routing.scatter(&expert_out, &weights, d));
        Tensor::from_f32(&y, h.shape())
    }
}
