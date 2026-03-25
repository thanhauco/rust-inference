//! MoE layers on the GPU, with optional expert offloading.
//!
//! Experts are either all device-resident, or (when they do not fit) kept
//! in pinned host memory and paged into a fixed pool of device slots by an
//! [`ExpertCache`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaEvent, result};

use super::{DeviceBuffer, PinnedBuffer, err, gpu, ops};
use crate::cpu::expert::ExpertWeights;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::moe_cache::{ExpertCache, ExpertKey, Policy, Residency};
use crate::ops::MoeRouting;
use crate::tensor::{Device, Tensor};

/// Alignment of each projection inside a packed expert.
const PACK_ALIGN: usize = 256;

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

/// An offloaded expert: gate|up|down packed in one pinned allocation (one
/// async copy moves it), scales permanently on the device.
struct HostExpert {
    packed: PinnedBuffer,
    parts: [(Vec<usize>, usize); 3],
    dtype: DType,
    dev_scales: [Option<Tensor>; 3],
}

fn align_up(v: usize) -> usize {
    v.div_ceil(PACK_ALIGN) * PACK_ALIGN
}

impl HostExpert {
    fn pack(w: &ExpertWeights, act: DType) -> Result<Self> {
        // non-FP8 experts are stored in the activation dtype
        let convert = |t: &Tensor| if t.dtype() == DType::F8E4M3 { Ok(t.clone()) } else { t.to_dtype(act) };
        let ts = [convert(&w.gate)?, convert(&w.up)?, convert(&w.down)?];
        let mut offsets = [0usize; 3];
        let mut total = 0;
        for (o, t) in offsets.iter_mut().zip(&ts) {
            *o = total;
            total = align_up(total + t.size_in_bytes());
        }
        let mut pinned = PinnedBuffer::alloc(total)?;
        for (o, t) in offsets.iter().zip(&ts) {
            pinned.as_mut_slice()[*o..*o + t.size_in_bytes()].copy_from_slice(t.bytes());
        }
        let mut dev_scales = [None, None, None];
        for (i, t) in ts.iter().enumerate() {
            if let Some(s) = t.scale() {
                dev_scales[i] = Some(s.to_device(Device::Cuda)?);
            }
        }
        Ok(HostExpert {
            packed: pinned,
            parts: [
                (ts[0].shape().to_vec(), offsets[0]),
                (ts[1].shape().to_vec(), offsets[1]),
                (ts[2].shape().to_vec(), offsets[2]),
            ],
            dtype: ts[0].dtype(),
            dev_scales,
        })
    }

    fn bytes(&self) -> &[u8] {
        self.packed.as_slice()
    }
}

/// Host<->device mover for the expert cache.
pub struct CudaResidency {
    experts: HashMap<ExpertKey, HostExpert>,
    slot_bytes: usize,
    slots: Vec<Arc<DeviceBuffer>>,
    free: Vec<usize>,
    slot_of: HashMap<ExpertKey, usize>,
    loaded: HashMap<ExpertKey, CudaEvent>,
}

impl CudaResidency {
    fn new() -> Self {
        CudaResidency {
            experts: HashMap::new(),
            slot_bytes: 0,
            slots: Vec::new(),
            free: Vec::new(),
            slot_of: HashMap::new(),
            loaded: HashMap::new(),
        }
    }

    fn add(&mut self, key: ExpertKey, e: HostExpert) {
        self.slot_bytes = self.slot_bytes.max(e.bytes().len());
        self.experts.insert(key, e);
    }

    /// Views of a resident expert; makes the compute stream wait for its
    /// upload.
    fn device_expert(&self, key: ExpertKey) -> Result<DeviceExpert> {
        let g = gpu()?;
        let slot = self.slot_of.get(&key).ok_or_else(|| Error::Cuda(format!("expert {key:?} is not resident")))?;
        if let Some(ev) = self.loaded.get(&key) {
            g.compute.wait(ev).map_err(err)?;
        }
        let e = &self.experts[&key];
        let buf = self.slots[*slot].clone();
        let view = |i: usize| -> Result<Tensor> {
            let (shape, off) = &e.parts[i];
            let mut t = Tensor::from_device_buffer(buf.clone(), shape, e.dtype, *off)?;
            if let Some(s) = &e.dev_scales[i] {
                t.set_scale(s.clone())?;
            }
            Ok(t)
        };
        Ok(DeviceExpert { gate: view(0)?, up: view(1)?, down: view(2)? })
    }
}

impl Residency for CudaResidency {
    fn load(&mut self, key: ExpertKey, reuse: Option<ExpertKey>) -> Result<()> {
        let g = gpu()?;
        let slot = match reuse.and_then(|r| {
            self.loaded.remove(&r);
            self.slot_of.remove(&r)
        }) {
            Some(s) => s,
            None => match self.free.pop() {
                Some(s) => s,
                None => {
                    self.slots.push(Arc::new(DeviceBuffer::alloc(self.slot_bytes)?));
                    self.slots.len() - 1
                }
            },
        };
        let host = self.experts.get(&key).ok_or_else(|| Error::Cuda(format!("unknown expert {key:?}")))?;
        // the slot may still be read by kernels queued on the compute stream
        let compute_done = g.ctx.new_event(None).map_err(err)?;
        compute_done.record(&g.compute).map_err(err)?;
        g.transfer.wait(&compute_done).map_err(err)?;
        // SAFETY: pinned source outlives the copy (owned by self.experts,
        // never freed while the residency lives); slot holds slot_bytes.
        unsafe { result::memcpy_htod_async(self.slots[slot].ptr(), host.bytes(), g.transfer.cu_stream()) }
            .map_err(err)?;
        let done = g.ctx.new_event(None).map_err(err)?;
        done.record(&g.transfer).map_err(err)?;
        self.loaded.insert(key, done);
        self.slot_of.insert(key, slot);
        Ok(())
    }

    fn is_ready(&self, key: ExpertKey) -> bool {
        self.loaded.get(&key).is_some_and(|e| e.is_complete())
    }
}

#[derive(Debug, Clone)]
pub struct OffloadOptions {
    /// Device slots for offloaded experts.
    pub slots: usize,
    /// The first `resident_layers` MoE layers keep all experts on the GPU.
    pub resident_layers: usize,
    pub policy: Policy,
}

/// State shared by all offloaded MoE layers.
pub struct OffloadRuntime {
    cache: Mutex<ExpertCache<CudaResidency>>,
    /// device routers by layer index
    routers: Vec<Option<Tensor>>,
    opts: OffloadOptions,
}

impl OffloadRuntime {
    /// Pack the experts of `(layer, experts)` into pinned memory and set up
    /// the cache. `routers[l]` is layer `l`'s device router (if sparse).
    pub fn new(
        layers: &[(usize, &[ExpertWeights])],
        routers: Vec<Option<Tensor>>,
        act: DType,
        opts: OffloadOptions,
    ) -> Result<Arc<Self>> {
        let mut res = CudaResidency::new();
        let mut keys = Vec::new();
        for &(layer, experts) in layers {
            for (e, w) in experts.iter().enumerate() {
                res.add((layer, e), HostExpert::pack(w, act)?);
                keys.push((layer, e));
            }
        }
        let mut cache = ExpertCache::new(res, opts.slots, opts.policy);
        for k in keys {
            cache.register(k, false);
        }
        Ok(Arc::new(OffloadRuntime { cache: Mutex::new(cache), routers, opts }))
    }

    pub fn stats(&self) -> crate::moe_cache::CacheStats {
        self.cache.lock().unwrap().stats()
    }
}

pub enum GpuExperts {
    Resident(Vec<DeviceExpert>),
    Offloaded { layer: usize, runtime: Arc<OffloadRuntime> },
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
        GpuExperts::Offloaded { layer, runtime } => {
            run_offloaded(*layer, runtime, h, t, &routing, &gathered, &expert_out, &rows)?;
        }
    }

    let y = ops::moe_scatter(&expert_out, &weights, &routing.slot_of, t, top_k)?;
    y.reshaped(h.shape())
}

#[allow(clippy::too_many_arguments)]
fn run_offloaded(
    layer: usize,
    rt: &OffloadRuntime,
    _h: &Tensor,
    _tokens: usize,
    routing: &MoeRouting,
    gathered: &Tensor,
    expert_out: &Tensor,
    rows: &dyn Fn(usize, &Tensor) -> Result<Tensor>,
) -> Result<()> {
    let num_layers = rt.routers.len().max(1);
    let protect: Vec<usize> = (0..3).map(|i| (layer + i) % num_layers).collect();
    let mut cache = rt.cache.lock().unwrap();
    let experts: Vec<usize> = routing.active_experts().collect();

    // Batches of at most half the pool, so loading one batch never evicts an
    // expert of the same batch; experts run as soon as their upload lands.
    for batch in experts.chunks((rt.opts.slots / 2).max(1)) {
        for &e in batch {
            cache.prepare((layer, e), &protect)?;
        }
        let mut pending = batch.to_vec();
        while !pending.is_empty() {
            let mut i = 0;
            while i < pending.len() {
                let key = (layer, pending[i]);
                if cache.is_ready(key) {
                    let dev = cache.residency().device_expert(key)?;
                    dev.forward_out(&rows(key.1, expert_out)?, &rows(key.1, gathered)?)?;
                    pending.swap_remove(i);
                } else {
                    i += 1;
                }
            }
            if !pending.is_empty() {
                std::thread::yield_now();
            }
        }
    }
    Ok(())
}
