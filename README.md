# rust-inference

A small LLM inference engine for Qwen3 and Qwen3-MoE models, written in Rust.
It is a rewrite of [garlic-inference](https://github.com/NikolayBlagoev/garlic-inference)
(C++/CUDA, by Nikolay Blagoev) and keeps its design: paged KV cache, FP8
block-scaled weights, MoE expert offloading with a CPU path for cold experts,
and energy profiling. Like the original it is a playground for inference
ideas rather than a production server.

| | |
|---|---|
| Models | Qwen3 dense, Qwen3-MoE (`Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`) |
| Weights | BF16 / F16 / F32, FP8 E4M3 with 128x128 block scales (Qwen FP8 format) |
| Backends | CPU (AVX-512 / AVX2+FMA / scalar, runtime-dispatched), CUDA (feature `cuda`) |
| Attention | paged KV cache (32-token pages), causal GQA, online softmax |
| MoE | top-k router (`norm_topk_prob` both ways), gather/scatter, `mlp_only_layers` |
| Offload | expert cache with LRU or layer-distance eviction, CPU cold-expert path, speculative prefetch |
| Extras | byte-level BPE tokenizer (HF-compatible), sampling, FP8 quantizer, section timers, RAPL/NVML energy |

## Build and run

```bash
cargo build --release
./target/release/runner <MODEL_DIR>                       # asks for a prompt
./target/release/runner <MODEL_DIR> -p "Why is the sky blue?" --chat -n 256
```

`MODEL_DIR` is a HuggingFace checkpoint directory (`config.json`,
`*.safetensors`, `tokenizer.json`), e.g.
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) or
[Qwen3-30B-A3B-Instruct-2507-FP8](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8).
Weights are memory-mapped, so loading is instant and only touched pages are read.

GPU build (no CUDA toolkit needed at build time; `libcuda`, `libnvrtc` and
`libcublas` are loaded at runtime and the kernels are compiled with NVRTC for
the installed GPU):

```bash
cargo build --release --features cuda
./target/release/runner <MODEL_DIR> --device cuda
```

Useful runner flags (`runner --help` for all of them):

| flag | |
|---|---|
| `--chat`, `--think` | wrap the prompt in the Qwen chat template (thinking off unless `--think`) |
| `-n N`, `--ignore-eos` | number of tokens; keep going past EOS for benchmarking |
| `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`, `--seed` | sampling (greedy by default, like the original) |
| `--profile` | per-section timing breakdown |
| `--power` | CPU (RAPL) and GPU (NVML) energy for idle / load / inference, following the original protocol |
| `--expert-slots N` | GPU: offload MoE experts to pinned host memory with N device slots (default: only if they do not fit) |
| `--cpu-expert-threshold N` | GPU offload: non-resident experts with at most N tokens run on the CPU (default 8, 0 = off) |
| `--lru`, `--resident-moe-layers N`, `--no-prefetch` | GPU offload policy knobs |
| `-t N` | CPU threads (`GARLIC_CPU_THREADS` sizes the expert executor, `GARLIC_CPU_ISA=avx2/scalar` caps SIMD) |

Converting a BF16 checkpoint to FP8 (same layout as Qwen's FP8 releases):

```bash
./target/release/quantize <BF16_DIR> <OUT_DIR> [--block 128]
```

## Performance

CPU numbers on a 4-vCPU Xeon VM (AVX-512) with random weights in the real
model shapes (`scripts/synthetic_checkpoint.py`), greedy, 32 generated tokens:

| model | weights | prefill | decode |
|---|---|---|---|
| Qwen3-0.6B shapes | BF16 | 125 tok/s (200-token prompt) | 12.1 tok/s |
| Qwen3-0.6B shapes | FP8 | 135 tok/s | 14.9 tok/s |
| Qwen3-30B-A3B shapes, 4 of 48 layers | FP8 | 68 tok/s (25-token prompt) | 23.5 tok/s |

Decode on this machine is bound by memory bandwidth (~26 GB/s): the BF16
GEMV streams weights at ~28 GB/s and the FP8 one decodes ~15 G weights/s.
`examples/linear_bench.rs` and `examples/attn_bench.rs` measure the kernels.

For reference, the C++ engine reports 53.3 tok/s (RTX 5060 Ti) and 81.3 tok/s
(RTX 4090) for Qwen3-30B-A3B FP8. The Rust CUDA path has not been benchmarked
against those numbers yet (see Status).

## Status

* **CPU**: complete and verified. Every architecture variant is checked
  against a float64 transcription of the `transformers` Qwen3 / Qwen3-MoE
  code (prefill logits at every position plus incremental decode through the
  paged cache), and the tokenizer against HuggingFace `tokenizers` on 324
  strings.
* **CUDA**: implemented and built in CI, with every kernel compiled by
  NVRTC there, but CI runners have no GPU. `tests/cuda_parity.rs` compares
  each op, the three fixture models and offloaded-vs-resident MoE against
  the CPU path; it skips when no device is present, so run
  `cargo test --release --features cuda --test cuda_parity` on a GPU machine
  before relying on the GPU path.

## Testing

```bash
cargo test --release                      # unit + parity tests (CPU)
cargo test --release --features cuda      # + CUDA build; GPU tests run if a GPU is present
LD_LIBRARY_PATH=<dir with libnvrtc.so.12> \
  cargo test --release --features cuda --lib kernels_compile -- --ignored
```

The fixtures under `tests/fixtures/` are regenerated with

```bash
python scripts/tokenizer_fixture.py tests/fixtures/tokenizer   # needs `tokenizers`
python scripts/reference_qwen3.py tests/fixtures               # needs `numpy`
```

## Layout

```
src/
  tensor.rs, dtype.rs      shared-storage tensors (CPU, mmap, pinned, CUDA); bf16/f16/fp8 conversions
  safetensors.rs           mmap reader, sharded loader, streaming writer
  tokenizer/               byte-level BPE + exact Qwen pre-tokenizer
  ops/                     device-dispatching ops (embedding, norms, rope, linear, router, MoE routing)
  cpu/                     SIMD kernels, fused GEMV / tiled GEMM, expert executor
  kvcache.rs, attention.rs paged KV cache and causal paged attention
  models/                  config, Qwen3 decoder (dense + MoE), GPU conversion
  moe_cache.rs             expert residency policies (LRU, layer distance)
  cuda/                    context, buffers, kernels.cu, ops, MoE offload, NVML
  profiler.rs, sampling.rs, quantize.rs
  bin/runner.rs, bin/quantize.rs
```

## Differences from the C++ engine

Beyond the language change:

* The tokenizer matches the Qwen split regex exactly (Unicode classes and
  the `\s+(?!\S)` lookahead) and recognises special tokens in the text; the
  original approximated the regex with ASCII classes.
* Untied checkpoints load their own `lm_head`; `mlp_only_layers` /
  `decoder_sparse_step` and `norm_topk_prob = false` are supported.
* Attention is a custom paged kernel (CPU and CUDA) instead of the cuDNN
  frontend graph, so cuDNN is not needed.
* Offloading waits for queued compute before overwriting an evicted slot and
  loads a layer's experts in batches that cannot evict each other.
* A CPU backend that runs the whole model, not just cold experts.

## License

MIT, see [LICENSE](LICENSE). The original garlic-inference is MIT licensed by
Nikolay Blagoev; this port keeps that notice.
