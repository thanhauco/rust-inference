//! Throughput of the CPU linear kernel for each weight dtype.
//!
//!     cargo run --release --example linear_bench [N] [K] [M]
//!
//! Reports effective weight bandwidth (GB/s), which is what bounds decode.

use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use garlic::cpu::linear::linear_f32;
use garlic::{DType, Tensor};

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let n = args.first().copied().unwrap_or(3072);
    let k = args.get(1).copied().unwrap_or(1024);
    let m = args.get(2).copied().unwrap_or(1);
    println!("y[{m}, {n}] = x[{m}, {k}] W[{n}, {k}]^T, isa={}", garlic::cpu::simd::isa().name());

    let w: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.618).sin() * 0.05).collect();
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.31).cos()).collect();
    let base = Tensor::from_f32(&w, &[n, k]).unwrap();

    for dtype in [DType::F32, DType::BF16, DType::F16, DType::F8E4M3] {
        let mut wt = base.to_dtype(dtype).unwrap();
        if dtype == DType::F8E4M3 {
            let (sn, sk) = (n.div_ceil(128), k.div_ceil(128));
            wt.set_scale(Tensor::from_f32(&vec![1.0; sn * sk], &[sn, sk]).unwrap()).unwrap();
        }
        // warm up, then time enough iterations to be stable
        linear_f32(&x, m, &wt).unwrap();
        let iters = (2_000_000_000 / wt.size_in_bytes().max(1)).clamp(5, 2000);
        let t0 = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(linear_f32(&x, m, &wt).unwrap());
        }
        let secs = t0.elapsed().as_secs_f64() / iters as f64;
        let gbps = wt.size_in_bytes() as f64 / secs / 1e9;
        let gflops = 2.0 * (m * n * k) as f64 / secs / 1e9;
        println!("{:>8}: {:>8.1} us  {:>6.1} GB/s  {:>6.1} GFLOP/s", dtype.to_string(), secs * 1e6, gbps, gflops);
    }
}
