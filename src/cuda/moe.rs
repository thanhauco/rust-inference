//! MoE layers on the GPU (all experts device-resident for now).

use super::ops;
use crate::cpu::expert::ExpertWeights;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::MoeRouting;
use crate::tensor::{Device, Tensor};

/// One expert with all three projections on the device.
#[derive(Clone)]
pub struct DeviceExpert {
    pub gate: Tensor,
    pub up: Tensor,
    pub down: Tensor,
}

/// Device copy of a weight: FP8 stays FP8 (scale uploaded), other float
/// types become the activation dtype so cuBLAS sees matching types.
pub fn upload_weight(w: &Tensor, act: DType) -> Result<Tensor> {
    if w.dtype() == DType::F8E4M3 { w.to_device(Device::Cuda) } else { w.to_dtype(act)?.to_device(Device::Cuda) }
}

impl DeviceExpert {
    pub fn upload(w: &ExpertWeights, act: DType) -> Result<Self> {
        Ok(DeviceExpert {
            gate: upload_weight(&w.gate, act)?,
            up: upload_weight(&w.up, act)?,
            down: upload_weight(&w.down, act)?,
        })
    }

    /// `y = down(up(x) * silu(gate(x)))` for `x: [n, D]` into `y: [n, D]`
    /// (both may be views into larger buffers).
    pub fn forward_out(&self, y: &Tensor, x: &Tensor) -> Result<()> {
        let n = x.dim(0);
        let inter = if self.gate.dtype() == DType::F8E4M3 && self.up.dtype() == DType::F8E4M3 {
            let inter = Tensor::empty(&[n, self.gate.dim(0)], x.dtype(), Device::Cuda)?;
            ops::fp8_gate_up_out(&inter, x, &self.gate, &self.up)?;
            inter
        } else {
            let mut g = ops::linear(x, &self.gate)?;
            ops::swiglu(&mut g, &ops::linear(x, &self.up)?)?;
            g
        };
        ops::linear_out(y, &inter, &self.down)
    }
}

pub enum GpuExperts {
    Resident(Vec<DeviceExpert>),
}

/// `h: [B, S, D]` (device) -> `[B, S, D]`.
pub fn forward(
    h: &Tensor,
    router: &Tensor,
    top_k: usize,
    normalize: bool,
    num_experts: usize,
    experts: &GpuExperts,
) -> Result<Tensor> {
    let d = *h.shape().last().unwrap();
    let t = h.numel() / d;
    let logits = ops::linear(h, router)?;
    let (ids, weights) = ops::topk_softmax(&logits, top_k, normalize)?;
    let routing = MoeRouting::new(&ids, top_k, num_experts)?;
    let gathered = ops::moe_gather(h, &routing.token_of_slot)?;
    let expert_out = Tensor::empty(gathered.shape(), h.dtype(), Device::Cuda)?;
    let rows = |e: usize, t: &Tensor| -> Result<Tensor> {
        let (a, b) = (routing.expert_offsets[e], routing.expert_offsets[e + 1]);
        t.view(&[b - a, d], a * d * t.dtype().size())
    };

    match experts {
        GpuExperts::Resident(list) => {
            for e in routing.active_experts() {
                list[e].forward_out(&rows(e, &expert_out)?, &rows(e, &gathered)?)?;
            }
        }
    }

    let y = ops::moe_scatter(&expert_out, &weights, &routing.slot_of, t, top_k)?;
    y.reshaped(h.shape())
}
