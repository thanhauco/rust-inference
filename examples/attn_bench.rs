//! Prefill attention throughput over the paged KV cache.
//!
//!     cargo run --release --example attn_bench [SEQ] [HEADS] [KV_HEADS]

use std::time::Instant;

use garlic::attention::PagedAttention;
use garlic::kvcache::{KvCache, PAGE_SIZE};
use garlic::{DType, Device, Tensor};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let s = args.first().copied().unwrap_or(200);
    let h = args.get(1).copied().unwrap_or(16);
    let hkv = args.get(2).copied().unwrap_or(8);
    let dh = 128;
    let pages = s.div_ceil(PAGE_SIZE);

    let mut cache = KvCache::new(hkv, dh, pages, 1, pages, DType::F32, Device::Cpu);
    let kv: Vec<f32> = (0..s * hkv * dh).map(|i| (i as f32 * 0.37).sin()).collect();
    let kv = Tensor::from_f32(&kv, &[1, s, hkv, dh]).unwrap();
    cache.append(&kv, &kv).unwrap();
    let q: Vec<f32> = (0..s * h * dh).map(|i| (i as f32 * 0.11).cos()).collect();
    let q = Tensor::from_f32(&q, &[1, s, h, dh]).unwrap();
    let attn = PagedAttention::new(h, hkv, dh);

    attn.forward(&q, &cache).unwrap();
    let iters = 20;
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(attn.forward(&q, &cache).unwrap());
    }
    let secs = t0.elapsed().as_secs_f64() / iters as f64;
    // q.k and p.v: 2 * 2 * dh flops per (query head, key) pair, causal
    let pairs = h as f64 * (s * (s + 1) / 2) as f64;
    println!("S={s} H={h} Hkv={hkv}: {:.2} ms  {:.1} GFLOP/s", secs * 1e3, 4.0 * dh as f64 * pairs / secs / 1e9);
}
