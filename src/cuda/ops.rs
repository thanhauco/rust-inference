//! Device implementations of the ops in `crate::ops` (and the MoE / KV-cache
//! helpers they need). Activations are BF16 or F16; norm gains, rope tables
//! and FP8 block scales are F32 device tensors.

use std::sync::Arc;

use cudarc::driver::{LaunchConfig, result};

use super::{DeviceBuffer, err, gpu, grid1, launch};
use crate::cpu::linear::BlockLayout;
use crate::dtype::{self, DType};
use crate::error::{Error, Result};
use crate::kvcache::KvCache;
use crate::tensor::{Device, Tensor};

const BLOCK: u32 = 256;

fn cfg(grid: (u32, u32, u32), block: (u32, u32, u32)) -> LaunchConfig {
    LaunchConfig { grid_dim: grid, block_dim: block, shared_mem_bytes: 0 }
}

fn expect_act(t: &Tensor, what: &str) -> Result<DType> {
    if t.device() != Device::Cuda {
        return Err(Error::Unsupported(format!("{what}: expected a CUDA tensor")));
    }
    match t.dtype() {
        DType::BF16 | DType::F16 => Ok(t.dtype()),
        other => Err(Error::Unsupported(format!("{what}: CUDA activations must be BF16/F16, got {other}"))),
    }
}

fn expect_f32_dev(t: &Tensor, what: &str) -> Result<()> {
    if t.device() != Device::Cuda || t.dtype() != DType::F32 {
        return Err(Error::Unsupported(format!(
            "{what}: expected an F32 CUDA tensor, got {} {:?}",
            t.dtype(),
            t.device()
        )));
    }
    Ok(())
}

/// Upload a small host array (indices, routing weights) for one kernel.
fn upload<T: crate::tensor::Pod>(v: &[T]) -> Result<Arc<DeviceBuffer>> {
    // SAFETY: Pod values are plain bytes.
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    let buf = DeviceBuffer::alloc(bytes.len())?;
    buf.upload(0, bytes)?;
    Ok(Arc::new(buf))
}

/// Device-to-device copy into fresh storage.
pub fn copy(x: &Tensor) -> Result<Tensor> {
    let out = Tensor::empty(x.shape(), x.dtype(), Device::Cuda)?;
    let g = gpu()?;
    if x.size_in_bytes() > 0 {
        // SAFETY: both allocations hold size_in_bytes() bytes.
        unsafe {
            result::memcpy_dtod_async(out.device_ptr(), x.device_ptr(), x.size_in_bytes(), g.compute.cu_stream())
        }
        .map_err(err)?;
    }
    Ok(out)
}

/// Copy `src` into the (pre-allocated, same-size) device tensor `dst`.
pub fn copy_into(dst: &Tensor, src: &Tensor) -> Result<()> {
    if dst.size_in_bytes() != src.size_in_bytes() {
        return Err(Error::Shape(format!("copy_into: {:?} vs {:?}", dst.shape(), src.shape())));
    }
    let g = gpu()?;
    // SAFETY: sizes checked.
    unsafe { result::memcpy_dtod_async(dst.device_ptr(), src.device_ptr(), src.size_in_bytes(), g.compute.cu_stream()) }
        .map_err(err)
}

pub fn embedding(ids: &Tensor, table: &Tensor) -> Result<Tensor> {
    let dt = expect_act(table, "embedding")?;
    let ids = ids.to_device(Device::Cuda)?;
    let (n, dim) = (ids.numel(), table.dim(1));
    let mut shape = ids.shape().to_vec();
    shape.push(dim);
    let out = Tensor::empty(&shape, dt, Device::Cuda)?;
    let g = gpu()?;
    let (pi, pt, po, d) = (ids.device_ptr(), table.device_ptr(), out.device_ptr(), dim as i32);
    launch!(g.compute, g.func("embedding", dt)?, cfg((n as u32, 1, 1), (BLOCK, 1, 1)), &pi, &pt, &po, &d)?;
    Ok(out)
}

pub fn rms_norm(x: &mut Tensor, w: &Tensor, eps: f32, group: usize) -> Result<()> {
    let dt = expect_act(x, "rms_norm")?;
    expect_f32_dev(w, "rms_norm weight")?;
    if w.numel() != group || !x.numel().is_multiple_of(group) {
        return Err(Error::Shape(format!("rms_norm: x {:?} weight {:?} group {group}", x.shape(), w.shape())));
    }
    let rows = (x.numel() / group) as u32;
    let g = gpu()?;
    let (px, pw, d) = (x.device_ptr(), w.device_ptr(), group as i32);
    launch!(g.compute, g.func("rmsnorm", dt)?, cfg((rows, 1, 1), (BLOCK, 1, 1)), &px, &pw, &d, &eps)
}

pub fn add_rms_norm(x: &mut Tensor, residual: &mut Tensor, w: &Tensor, eps: f32) -> Result<()> {
    let dt = expect_act(x, "add_rms_norm")?;
    expect_f32_dev(w, "add_rms_norm weight")?;
    let dim = *x.shape().last().unwrap();
    if x.shape() != residual.shape() || w.numel() != dim {
        return Err(Error::Shape(format!("add_rms_norm: {:?} / {:?} / {:?}", x.shape(), residual.shape(), w.shape())));
    }
    let rows = (x.numel() / dim) as u32;
    let g = gpu()?;
    let (px, pr, pw, d) = (x.device_ptr(), residual.device_ptr(), w.device_ptr(), dim as i32);
    launch!(g.compute, g.func("add_rmsnorm", dt)?, cfg((rows, 1, 1), (BLOCK, 1, 1)), &px, &pr, &pw, &d, &eps)
}

/// `x: [B, S, H, Dh]`; `cos`/`sin`: `[B, S, Dh]` (host or device F32).
pub fn apply_rope(x: &mut Tensor, cos: &Tensor, sin: &Tensor) -> Result<()> {
    let dt = expect_act(x, "apply_rope")?;
    let (b, s, h, dh) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3));
    let (cos, sin) = (cos.to_device(Device::Cuda)?, sin.to_device(Device::Cuda)?);
    if cos.shape() != [b, s, dh] || sin.shape() != [b, s, dh] {
        return Err(Error::Shape(format!("apply_rope: tables {:?} for x {:?}", cos.shape(), x.shape())));
    }
    let total = b * s * h * dh / 2;
    let g = gpu()?;
    let (px, pc, ps) = (x.device_ptr(), cos.device_ptr(), sin.device_ptr());
    let (tokens, heads, d) = ((b * s) as i32, h as i32, dh as i32);
    launch!(g.compute, g.func("rope", dt)?, grid1(total, BLOCK), &px, &pc, &ps, &tokens, &heads, &d)
}

pub fn swiglu(gate: &mut Tensor, up: &Tensor) -> Result<()> {
    let dt = expect_act(gate, "swiglu")?;
    if gate.numel() != up.numel() || up.dtype() != dt {
        return Err(Error::Shape(format!("swiglu: {:?} vs {:?}", gate.shape(), up.shape())));
    }
    let g = gpu()?;
    let (pg, pu, n) = (gate.device_ptr(), up.device_ptr(), gate.numel() as i64);
    launch!(g.compute, g.func("swiglu", dt)?, grid1(gate.numel(), BLOCK), &pg, &pu, &n)
}

pub fn add_inplace(x: &mut Tensor, y: &Tensor) -> Result<()> {
    let dt = expect_act(x, "add_inplace")?;
    if x.numel() != y.numel() || y.dtype() != dt {
        return Err(Error::Shape(format!("add_inplace: {:?} vs {:?}", x.shape(), y.shape())));
    }
    let g = gpu()?;
    let (px, py, n) = (x.device_ptr(), y.device_ptr(), x.numel() as i64);
    launch!(g.compute, g.func("add", dt)?, grid1(x.numel(), BLOCK), &px, &py, &n)
}

/// `y = x W^T` into a freshly allocated tensor.
pub fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let mut shape = x.shape().to_vec();
    *shape.last_mut().unwrap() = w.dim(0);
    let y = Tensor::empty(&shape, x.dtype(), Device::Cuda)?;
    linear_out(&y, x, w)?;
    Ok(y)
}

/// `y = x W^T` into `y` (which may be a view into a larger buffer).
pub fn linear_out(y: &Tensor, x: &Tensor, w: &Tensor) -> Result<()> {
    let dt = expect_act(x, "linear")?;
    let k = *x.shape().last().unwrap();
    let m = x.numel() / k.max(1);
    let n = w.dim(0);
    if w.ndim() != 2 || w.dim(1) != k || y.numel() != m * n || y.dtype() != dt {
        return Err(Error::Shape(format!("linear: x {:?} w {:?} y {:?}", x.shape(), w.shape(), y.shape())));
    }
    if m == 0 {
        return Ok(());
    }
    let g = gpu()?;
    match w.dtype() {
        DType::F8E4M3 => {
            let (bn, bk, sc, ps) = fp8_layout(w)?;
            let (py, pw, px) = (y.device_ptr(), w.device_ptr(), x.device_ptr());
            let (ni, ki, mi) = (n as i32, k as i32, m as i32);
            let grid = (n.div_ceil(8) as u32, m as u32, 1);
            launch!(
                g.compute,
                g.func("fp8_gemv", dt)?,
                cfg(grid, (32, 8, 1)),
                &py,
                &pw,
                &ps,
                &px,
                &ni,
                &ki,
                &mi,
                &bn,
                &bk,
                &sc
            )
        }
        wd if wd == dt => g.gemm(y.device_ptr(), x.device_ptr(), w.device_ptr(), m, n, k, dt, &g.compute),
        other => Err(Error::Unsupported(format!("CUDA linear with {other} weights and {dt} activations"))),
    }
}

/// `(block_n, block_k, scale_cols, scale ptr)` for an FP8 weight.
fn fp8_layout(w: &Tensor) -> Result<(i32, i32, i32, u64)> {
    let s = w.scale().ok_or_else(|| Error::Unsupported("FP8 weight without block scale".into()))?;
    expect_f32_dev(s, "fp8 scale")?;
    let (n, k) = (w.dim(0), w.dim(1));
    let l = BlockLayout::new(n, k, s.shape())?;
    if !k.is_multiple_of(4) || !l.block_k.is_multiple_of(4) {
        return Err(Error::Unsupported(format!(
            "FP8 GEMV needs K and block_k divisible by 4 (K={k}, block {})",
            l.block_k
        )));
    }
    Ok((l.block_n as i32, l.block_k as i32, l.scale_cols as i32, s.device_ptr()))
}

/// Fused expert front half for FP8 experts:
/// `y = bf16(up(x)) * bf16(silu(gate(x)))` into `y: [M, I]`.
pub fn fp8_gate_up_out(y: &Tensor, x: &Tensor, gate: &Tensor, up: &Tensor) -> Result<()> {
    let dt = expect_act(x, "fp8_gate_up")?;
    let k = *x.shape().last().unwrap();
    let m = x.numel() / k.max(1);
    let n = gate.dim(0);
    if gate.dtype() != DType::F8E4M3 || up.shape() != gate.shape() || gate.dim(1) != k || y.numel() != m * n {
        return Err(Error::Shape(format!("fp8_gate_up: x {:?} gate {:?} y {:?}", x.shape(), gate.shape(), y.shape())));
    }
    if m == 0 {
        return Ok(());
    }
    let (bn, bk, sc, psg) = fp8_layout(gate)?;
    let (_, _, _, psu) = fp8_layout(up)?;
    let g = gpu()?;
    let (py, pg, pu, px) = (y.device_ptr(), gate.device_ptr(), up.device_ptr(), x.device_ptr());
    let (ni, ki, mi) = (n as i32, k as i32, m as i32);
    launch!(
        g.compute,
        g.func("fp8_gemv_swiglu", dt)?,
        cfg((n.div_ceil(8) as u32, m as u32, 1), (32, 8, 1)),
        &py,
        &pg,
        &psg,
        &pu,
        &psu,
        &px,
        &ni,
        &ki,
        &mi,
        &bn,
        &bk,
        &sc
    )
}

pub fn argmax_last(x: &Tensor) -> Result<Vec<u32>> {
    let dt = expect_act(x, "argmax")?;
    let n = *x.shape().last().unwrap();
    let rows = x.numel() / n;
    let out = Tensor::empty(&[rows], DType::U32, Device::Cuda)?;
    let g = gpu()?;
    let (px, po, ni) = (x.device_ptr(), out.device_ptr(), n as i32);
    launch!(g.compute, g.func("argmax", dt)?, cfg((rows as u32, 1, 1), (BLOCK, 1, 1)), &px, &po, &ni)?;
    Ok(out.to_cpu()?.as_slice::<u32>().to_vec())
}

/// Router top-k on the device; results come back to the host because the
/// expert dispatch is decided there.
pub fn topk_softmax(logits: &Tensor, k: usize, normalize: bool) -> Result<(Vec<u32>, Vec<f32>)> {
    let dt = expect_act(logits, "topk_softmax")?;
    let e = *logits.shape().last().unwrap();
    if k == 0 || k > 32 || k > e {
        return Err(Error::Shape(format!("CUDA top-{k} of {e} experts (k must be <= 32)")));
    }
    let t = logits.numel() / e;
    let ids = Tensor::empty(&[t, k], DType::U32, Device::Cuda)?;
    let w = Tensor::empty(&[t, k], DType::F32, Device::Cuda)?;
    let g = gpu()?;
    let (pl, pi, pw) = (logits.device_ptr(), ids.device_ptr(), w.device_ptr());
    let (ei, ki, norm) = (e as i32, k as i32, normalize as i32);
    launch!(g.compute, g.func("topk_softmax", dt)?, cfg((t as u32, 1, 1), (32, 1, 1)), &pl, &pi, &pw, &ei, &ki, &norm)?;
    Ok((ids.to_cpu()?.as_slice::<u32>().to_vec(), w.to_cpu()?.as_slice::<f32>().to_vec()))
}

/// `x: [T, D]` -> `[slots, D]` with `out[s] = x[token_of_slot[s]]`.
pub fn moe_gather(x: &Tensor, token_of_slot: &[usize]) -> Result<Tensor> {
    let dt = expect_act(x, "moe_gather")?;
    let d = *x.shape().last().unwrap();
    let out = Tensor::empty(&[token_of_slot.len(), d], dt, Device::Cuda)?;
    if token_of_slot.is_empty() {
        return Ok(out);
    }
    let idx = upload(&token_of_slot.iter().map(|&t| t as u32).collect::<Vec<u32>>())?;
    let g = gpu()?;
    let (px, pi, po, di) = (x.device_ptr(), idx.ptr(), out.device_ptr(), d as i32);
    launch!(
        g.compute,
        g.func("moe_gather", dt)?,
        cfg((token_of_slot.len() as u32, 1, 1), (BLOCK, 1, 1)),
        &px,
        &pi,
        &po,
        &di
    )?;
    Ok(out)
}

/// `out[t] = sum_j weights[t, j] * expert_out[slot_of[t, j]]`.
pub fn moe_scatter(expert_out: &Tensor, weights: &[f32], slot_of: &[usize], tokens: usize, k: usize) -> Result<Tensor> {
    let dt = expect_act(expert_out, "moe_scatter")?;
    let d = *expert_out.shape().last().unwrap();
    let out = Tensor::empty(&[tokens, d], dt, Device::Cuda)?;
    let w = upload(weights)?;
    let s = upload(&slot_of.iter().map(|&v| v as u32).collect::<Vec<u32>>())?;
    let g = gpu()?;
    let (pe, pw, ps, po) = (expert_out.device_ptr(), w.ptr(), s.ptr(), out.device_ptr());
    let (di, ki) = (d as i32, k as i32);
    launch!(
        g.compute,
        g.func("moe_scatter", dt)?,
        cfg((tokens as u32, 1, 1), (BLOCK, 1, 1)),
        &pe,
        &pw,
        &ps,
        &po,
        &di,
        &ki
    )?;
    Ok(out)
}

/// Write `k`, `v` (`[B, S, kv_heads, Dh]`) into the cache pages. The cache
/// has already uploaded its page table and pre-append lengths.
pub fn kv_append(cache: &KvCache, k: &Tensor, v: &Tensor) -> Result<()> {
    let dt = expect_act(k, "kv_append")?;
    let (b, s) = (k.dim(0), k.dim(1));
    let (pt, lens) = cache.device_tables().ok_or_else(|| Error::Cuda("KV cache has no device tables".into()))?;
    let g = gpu()?;
    let (pk, pv, kp, vp) = (k.device_ptr(), v.device_ptr(), cache.k_pages().device_ptr(), cache.v_pages().device_ptr());
    let (si, hkv, dh, mpps) =
        (s as i32, cache.num_kv_heads() as i32, cache.head_dim() as i32, cache.max_pages_per_seq() as i32);
    launch!(
        g.compute,
        g.func("kv_append", dt)?,
        cfg(((b * s) as u32, cache.num_kv_heads() as u32, 1), (128, 1, 1)),
        &pk,
        &pv,
        &kp,
        &vp,
        &pt,
        &lens,
        &si,
        &hkv,
        &dh,
        &mpps
    )
}

/// Causal paged attention for `q: [B, S, H, Dh]` after `kv_append`.
pub fn paged_attention(q: &Tensor, cache: &KvCache, heads: usize, scale: f32) -> Result<Tensor> {
    let dt = expect_act(q, "attention")?;
    let (b, s, dh) = (q.dim(0), q.dim(1), q.dim(3));
    if dh > 256 {
        return Err(Error::Unsupported(format!("CUDA attention supports head_dim <= 256, got {dh}")));
    }
    let (pt, lens) = cache.device_tables().ok_or_else(|| Error::Cuda("KV cache has no device tables".into()))?;
    let out = Tensor::empty(q.shape(), dt, Device::Cuda)?;
    let g = gpu()?;
    let (pq, kp, vp, po) =
        (q.device_ptr(), cache.k_pages().device_ptr(), cache.v_pages().device_ptr(), out.device_ptr());
    let (si, hi, hkv, di, mpps) =
        (s as i32, heads as i32, cache.num_kv_heads() as i32, dh as i32, cache.max_pages_per_seq() as i32);
    launch!(
        g.compute,
        g.func("paged_attention", dt)?,
        cfg((heads as u32, (b * s) as u32, 1), (128, 1, 1)),
        &pq,
        &kp,
        &vp,
        &po,
        &pt,
        &lens,
        &si,
        &hi,
        &hkv,
        &di,
        &mpps,
        &scale
    )?;
    Ok(out)
}

/// Device activations -> host f32 (for logits).
pub fn to_host_f32(x: &Tensor) -> Result<Vec<f32>> {
    let host = x.to_cpu()?;
    Ok(match host.dtype() {
        DType::BF16 => host.as_slice::<u16>().iter().map(|&b| dtype::bf16_to_f32(b)).collect(),
        DType::F16 => host.as_slice::<u16>().iter().map(|&h| dtype::f16_to_f32(h)).collect(),
        DType::F32 => host.as_slice::<f32>().to_vec(),
        other => return Err(Error::Unsupported(format!("to_host_f32 on {other}"))),
    })
}

/// Host f32 values -> device activations of `dtype`.
pub fn from_host_f32(vals: &[f32], shape: &[usize], dtype: DType) -> Result<Tensor> {
    Tensor::from_f32(vals, shape)?.to_dtype(dtype)?.to_device(Device::Cuda)
}
