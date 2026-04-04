//! GPU vs CPU agreement. Needs `--features cuda` and a CUDA device; every
//! test skips (and says so) when no device is available.

#![cfg(feature = "cuda")]

use garlic::attention::PagedAttention;
use garlic::cuda::moe::OffloadOptions;
use garlic::kvcache::KvCache;
use garlic::models::cuda::CudaOptions;
use garlic::models::{self, LanguageModel, LogitsMode, ModelConfig, Qwen3};
use garlic::moe_cache::Policy;
use garlic::{DType, Device, Tensor, cuda, ops};

fn have_gpu() -> bool {
    let ok = cuda::is_available();
    if !ok {
        eprintln!("skipping: no usable CUDA device");
    }
    ok
}

fn data(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + seed) * 0.618).sin()).collect()
}

fn gpu_bf16(v: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(v, shape).unwrap().to_dtype(DType::BF16).unwrap().to_device(Device::Cuda).unwrap()
}

fn host(t: &Tensor) -> Vec<f32> {
    cuda::ops::to_host_f32(t).unwrap()
}

/// bf16 activations: compare relative to the overall magnitude.
fn assert_close(got: &[f32], want: &[f32], rel: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(worst <= rel * scale, "{what}: max diff {worst} vs scale {scale}");
}

#[test]
fn elementwise_and_norm_ops() {
    if !have_gpu() {
        return;
    }
    let (rows, dim) = (5, 96);
    let x = data(rows * dim, 1.0);
    let w = data(dim, 2.0);
    let wt = Tensor::from_f32(&w, &[dim]).unwrap();

    let mut cpu = Tensor::from_f32(&x, &[rows, dim]).unwrap();
    ops::rms_norm(&mut cpu, &wt, 1e-6, None).unwrap();
    let mut gpu = gpu_bf16(&x, &[rows, dim]);
    ops::rms_norm(&mut gpu, &wt.to_device(Device::Cuda).unwrap(), 1e-6, None).unwrap();
    assert_close(&host(&gpu), cpu.as_slice::<f32>(), 2e-2, "rms_norm");

    let r = data(rows * dim, 3.0);
    let mut cx = Tensor::from_f32(&x, &[rows, dim]).unwrap();
    let mut cr = Tensor::from_f32(&r, &[rows, dim]).unwrap();
    ops::add_rms_norm(&mut cx, &mut cr, &wt, 1e-6).unwrap();
    let (mut gx, mut gr) = (gpu_bf16(&x, &[rows, dim]), gpu_bf16(&r, &[rows, dim]));
    ops::add_rms_norm(&mut gx, &mut gr, &wt.to_device(Device::Cuda).unwrap(), 1e-6).unwrap();
    assert_close(&host(&gx), cx.as_slice::<f32>(), 2e-2, "add_rms_norm x");
    assert_close(&host(&gr), cr.as_slice::<f32>(), 1e-2, "add_rms_norm residual");

    let mut cg = Tensor::from_f32(&x, &[rows, dim]).unwrap();
    ops::swiglu(&mut cg, &Tensor::from_f32(&r, &[rows, dim]).unwrap()).unwrap();
    let mut gg = gpu_bf16(&x, &[rows, dim]);
    ops::swiglu(&mut gg, &gpu_bf16(&r, &[rows, dim])).unwrap();
    assert_close(&host(&gg), cg.as_slice::<f32>(), 2e-2, "swiglu");

    let logits = data(3 * 40, 4.0);
    let (ci, cw) = ops::topk_softmax(&Tensor::from_f32(&logits, &[3, 40]).unwrap(), 4, true).unwrap();
    let (gi, gw) = ops::topk_softmax(&gpu_bf16(&logits, &[3, 40]), 4, true).unwrap();
    assert_eq!(ci, gi, "topk ids");
    assert_close(&gw, &cw, 2e-2, "topk weights");
    assert_eq!(
        ops::argmax_last(&gpu_bf16(&logits, &[3, 40])).unwrap(),
        ops::argmax_last(&Tensor::from_f32(&logits, &[3, 40]).unwrap()).unwrap()
    );
}

#[test]
fn linear_bf16_and_fp8() {
    if !have_gpu() {
        return;
    }
    let (m, n, k) = (3, 300, 260);
    let x = data(m * k, 1.0);
    let w = Tensor::from_f32(&data(n * k, 5.0), &[n, k]).unwrap();
    for wt in [w.to_dtype(DType::BF16).unwrap(), garlic::quantize::quantize_fp8(&w, 128).unwrap().0] {
        let cpu = ops::linear(&Tensor::from_f32(&x, &[m, k]).unwrap(), &wt).unwrap();
        let gpu = ops::linear(&gpu_bf16(&x, &[m, k]), &wt.to_device(Device::Cuda).unwrap()).unwrap();
        assert_close(&host(&gpu), cpu.as_slice::<f32>(), 2e-2, &format!("linear {}", wt.dtype()));
    }
}

#[test]
fn paged_attention_prefill_and_decode() {
    if !have_gpu() {
        return;
    }
    let (h, hkv, dh, total) = (8, 2, 64, 70);
    let k = data(total * hkv * dh, 0.1);
    let v = data(total * hkv * dh, 0.7);
    let q = data(total * h * dh, 1.3);
    let attn = PagedAttention::new(h, hkv, dh);
    let mut cc = KvCache::new(hkv, dh, 8, 1, 4, DType::F32, Device::Cpu);
    let mut gc = KvCache::new(hkv, dh, 8, 1, 4, DType::BF16, Device::Cuda);
    let mut pos = 0;
    for step in [65, 1, 1, 1, 1, 1] {
        let ks = &k[pos * hkv * dh..(pos + step) * hkv * dh];
        let vs = &v[pos * hkv * dh..(pos + step) * hkv * dh];
        let qs = &q[pos * h * dh..(pos + step) * h * dh];
        cc.append(
            &Tensor::from_f32(ks, &[1, step, hkv, dh]).unwrap(),
            &Tensor::from_f32(vs, &[1, step, hkv, dh]).unwrap(),
        )
        .unwrap();
        gc.append(&gpu_bf16(ks, &[1, step, hkv, dh]), &gpu_bf16(vs, &[1, step, hkv, dh])).unwrap();
        let co = attn.forward(&Tensor::from_f32(qs, &[1, step, h, dh]).unwrap(), &cc).unwrap();
        let go = attn.forward(&gpu_bf16(qs, &[1, step, h, dh]), &gc).unwrap();
        assert_close(&host(&go), co.as_slice::<f32>(), 3e-2, &format!("attention step of {step}"));
        pos += step;
    }
}

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn prefill_logits(model: &dyn LanguageModel, ids: &[u32]) -> Vec<f32> {
    let n = ids.len();
    let pos: Vec<u32> = (0..n as u32).collect();
    let mut caches = model.new_kv_caches(1, 8, 4);
    let out = model
        .forward(
            &Tensor::from_u32(ids, &[1, n]).unwrap(),
            &Tensor::from_u32(&pos, &[1, n]).unwrap(),
            &mut caches,
            LogitsMode::All,
        )
        .unwrap();
    if out.device() == Device::Cuda { host(&out) } else { out.as_slice::<f32>().to_vec() }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    dot / (a.iter().map(|x| x * x).sum::<f32>().sqrt() * b.iter().map(|x| x * x).sum::<f32>().sqrt())
}

#[test]
fn models_match_cpu() {
    if !have_gpu() {
        return;
    }
    let ids = [1u32, 50, 300, 7, 42, 99, 3, 11, 250];
    for name in ["tiny-qwen3", "tiny-qwen3-moe", "tiny-qwen3-moe-fp8"] {
        let cpu = models::from_pretrained(fixture(name)).unwrap();
        let gpu = models::from_pretrained_on(fixture(name), Device::Cuda).unwrap();
        assert_eq!(gpu.device(), Device::Cuda);
        let c = cosine(&prefill_logits(&*gpu, &ids), &prefill_logits(&*cpu, &ids));
        assert!(c > 0.995, "{name}: cosine {c}");
    }
}

#[test]
fn offloaded_moe_matches_resident() {
    if !have_gpu() {
        return;
    }
    let dir = fixture("tiny-qwen3-moe-fp8");
    let load = |offload: Option<OffloadOptions>| {
        let cfg = ModelConfig::from_pretrained(&dir).unwrap();
        Qwen3::load(cfg, &dir).unwrap().to_cuda(&CudaOptions { offload }).unwrap()
    };
    let ids = [5u32, 17, 300, 42, 8, 9, 100];
    let resident = prefill_logits(&load(None), &ids);
    for (cpu_threshold, policy) in [(0, Policy::Lru), (8, Policy::Lru), (2, Policy::LayerDistance { num_layers: 2 })] {
        let opts = OffloadOptions { slots: 3, resident_layers: 0, cpu_threshold, policy, prefetch: true };
        let got = prefill_logits(&load(Some(opts)), &ids);
        let c = cosine(&got, &resident);
        assert!(c > 0.999, "cpu_threshold {cpu_threshold}: cosine {c}");
    }
}
