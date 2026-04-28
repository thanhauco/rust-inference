# Changelog

## 0.1.0

First release: a Rust port of garlic-inference.

* Qwen3 and Qwen3-MoE on the CPU (AVX-512 / AVX2+FMA / scalar), with BF16,
  F16, F32 and block-scaled FP8 weights, memory-mapped from safetensors.
* Paged KV cache and causal GQA paged attention.
* MoE routing, a work-stealing CPU expert executor, and expert cache policies
  (LRU, layer distance).
* CUDA backend behind the `cuda` feature: NVRTC-compiled kernels, cuBLAS,
  expert offloading to pinned memory with a CPU path for cold experts and
  speculative prefetch.
* Byte-level BPE tokenizer matching HuggingFace `tokenizers`.
* `runner` (generation, sampling, profiling, RAPL / NVML energy) and
  `quantize` (BF16 to FP8) binaries.
* Parity tests against a float64 reference of the `transformers` models.
  The GPU parity tests have not been run on hardware yet.
