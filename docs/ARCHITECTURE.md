# Architecture

This is a map of how a token flows through the engine and where each piece
lives. Names in parentheses are the C++ garlic-inference counterparts.

## Overview

```mermaid
flowchart LR
    runner["bin/runner"] --> tok["tokenizer"]
    runner --> models
    subgraph garlic["garlic (library)"]
        st["safetensors + tensor"] --> models["models: Qwen3 dense / MoE"]
        models --> ops["ops: device-dispatching"]
        models --> kv["kvcache + attention"]
        models --> mc["moe_cache: residency policies"]
        ops --> cpu["cpu: SIMD kernels, expert executor"]
        kv --> cpu
        ops --> cuda["cuda: kernels, cuBLAS, offload"]
        kv --> cuda
        mc --> cuda
    end
```

## Tensors and devices

`Tensor` (`Tensor` + `DataView`) is a shape, a dtype and a byte offset into
reference-counted storage. Cloning is shallow, and several tensors may view
one allocation. Storage is one of:

| storage | used for |
|---|---|
| `CpuBuffer::Owned` | activations, converted weights (64-byte aligned) |
| `CpuBuffer::Mapped` | weights straight out of the safetensors mmap |
| `CpuBuffer::Pinned` | offloaded experts (CUDA builds), readable by CPU kernels and uploadable asynchronously |
| `Storage::Cuda` | device memory, allocated stream-ordered on the compute stream |

FP8 weights carry a 2-D F32 block scale (`*_scale_inv`, normally one per
128x128 tile; `BlockLayout` works out the tiling from the scale shape).

On the CPU, activations are F32. On the GPU they are BF16 (F16 for F16
checkpoints) and the KV cache uses the same type.

## Concepts

**Paged KV cache.** Keys and values are stored in fixed pools of 32-token
pages (`[max_pages, kv_heads, 32, head_dim]` per layer) instead of one
contiguous buffer per sequence. A per-sequence page table maps token
positions to pages, and pages come from a free list as the sequence grows.
`--max-pages` sizes the pool and `--pages-per-seq` caps a sequence, so the
default of 128 pages allows 4096 tokens of context.

```mermaid
flowchart LR
    pt["page table of sequence 0, 70 tokens: 3, 0, 5"]
    subgraph pool["page pool of one layer, max_pages pages of 32 tokens"]
        p3["page 3: tokens 0-31"]
        p0["page 0: tokens 32-63"]
        p5["page 5: tokens 64-69"]
        fl["pages 1, 2, 4, 6 ...: free list"]
    end
    pt -->|"logical page 0"| p3
    pt -->|"logical page 1"| p0
    pt -->|"logical page 2"| p5
```

**Block-scaled FP8.** An FP8 (E4M3) weight is one byte per value plus a
scale for each 128x128 tile (`<name>_scale_inv`, an F32 matrix). A value is
`decode(byte) * scale`. Weights stay in FP8 in memory and are decoded inside
the matmul kernels, which halves the bytes streamed per token compared with
BF16. That is what bounds decode speed.

**Expert offloading.** A MoE layer has many experts (128 in Qwen3-30B-A3B)
but each token uses only a few. When all experts do not fit in GPU memory,
they live in pinned host memory and only a fixed number of device slots hold
copies. Experts that a batch needs are uploaded on demand, experts that got
only a few tokens run on the CPU instead of being uploaded, and the next
layers' likely experts are uploaded early. See "GPU expert offloading" below.

**Device dispatch.** Every op takes tensors and looks at where they live, so
the model code is the same for CPU and GPU. Moving a model to the GPU swaps
the weights' storage (`Qwen3::to_cuda`); nothing else changes.

## Using the library

```rust
use garlic::models::{self, LogitsMode};
use garlic::tokenizer::BpeTokenizer;
use garlic::{Tensor, ops};

let tokenizer = BpeTokenizer::load(dir)?;
let model = models::from_pretrained(dir)?;          // from_pretrained_on(dir, Device::Cuda) for the GPU
let mut caches = model.new_kv_caches(1, 64, 32);    // batch, pages in the pool, pages per sequence

let ids = tokenizer.encode("Hello");
let n = ids.len();
let positions: Vec<u32> = (0..n as u32).collect();
let logits = model.forward(
    &Tensor::from_u32(&ids, &[1, n])?,
    &Tensor::from_u32(&positions, &[1, n])?,
    &mut caches,
    LogitsMode::Last,                               // only the last position's logits
)?;
let next = ops::argmax_last(&logits)?[0];
```

Prefill passes the whole prompt in one call. Decoding then feeds one token at
a time with its absolute position; the caches remember everything before it.
`sampling::Sampler` replaces `argmax_last` when you want temperature, top-k,
top-p or a repetition penalty, and `StreamDecoder` turns tokens into text
without splitting multi-byte characters. `examples/generate.rs` is the full
loop and `src/bin/runner.rs` adds streaming, sampling and profiling.

## One forward pass

```
ids --embedding--> hidden [B,S,D]
for each decoder layer:
    residual = hidden
    h = rmsnorm(hidden)
    q,k,v = h Wq, h Wk, h Wv           per-head rmsnorm on q,k; rope on q,k
    KvCache::append(k, v)              pages mapped lazily from the free list
    o = PagedAttention(q, cache)       causal, GQA, online softmax
    h = o Wo
    residual += h; h = rmsnorm(residual)        (fused: add_rms_norm)
    h = MLP(h) | SparseMoe(h)
    hidden = h + residual
logits = rmsnorm(hidden[last]) W_lm   (LogitsMode::Last skips the other rows)
```

```mermaid
flowchart TD
    ids["token ids"] --> emb["embedding"]
    emb --> n1
    subgraph layer["repeated for every decoder layer"]
        n1["rms_norm"] --> qkv["q, k, v projections"]
        qkv --> qk["per-head rms_norm and rope on q, k"]
        qk --> app["KvCache append"]
        app --> att["paged attention"]
        att --> oproj["o projection"]
        oproj --> an["add residual, rms_norm"]
        an --> ffn{"dense layer?"}
        ffn -->|yes| mlp["SwiGLU MLP"]
        ffn -->|no| moe["router, top-k, experts"]
        mlp --> res["add residual"]
        moe --> res
    end
    res --> fn["final rms_norm"]
    fn --> lm["lm_head on the last position"]
    lm --> logits["logits"]
```

`ops::*` are free functions that dispatch on the tensor's device
(`linear`, `rms_norm`, ...), so `models/qwen3.rs` has a single code path.

## CPU kernels (`src/cpu`)

* `simd`: AVX-512 / AVX2+FMA / scalar, chosen once at runtime
  (`GARLIC_CPU_ISA` caps it).
* `linear`: `y = x W^T` split over rows of W. For up to 4 activation rows
  each weight row is decoded in registers and dotted against all of them
  (FP8 partial sums are scaled once per 128-wide group, like `fp8_gemv`).
  For more rows a task decodes its 16 weight rows once and runs a 4x4
  register-tiled microkernel.
* FP8 decode rebiases the exponent with an integer add. The classic "place
  the bits, multiply by 2^120" trick lets the compiler multiply before the
  subnormal lanes are masked, which costs ~100 cycles per denormal lane.
* `expert` (`fp8_expert_cpu`): one expert's SwiGLU FFN over T tokens plus a
  persistent executor. Each expert splits into 64-row chunks over two
  dependent stages (gate/up rows, then down rows); workers steal chunks
  across every in-flight expert, and submission is non-blocking.

```mermaid
flowchart LR
    job["expert job: x, W gate, W up, W down"] --> s1["stage 1: gate/up rows in 64-row chunks, then SwiGLU"]
    s1 -->|"all chunks done"| s2["stage 2: down rows in 64-row chunks"]
    s2 --> done["job done, output taken"]
    pool["worker threads"] -.->|"steal chunks from any in-flight job"| s1
    pool -.-> s2
```

The runner uses mimalloc: with glibc's allocator every op's output buffer
was returned to the kernel and page-faulted again, and parallel first-touch
faults serialise in the kernel.

## MoE

The router is a linear plus top-k softmax (`norm_topk_prob` renormalises
over the selected experts). `MoeRouting` lays the (token, choice) pairs out
contiguously per expert (`moe_gather`), each expert runs on its block, and
`scatter` adds the outputs back per token, weighted.

```mermaid
flowchart LR
    h["hidden states, T tokens"] --> router["router linear, top-k softmax"]
    router --> gather["gather: group the token-choice pairs by expert"]
    gather --> e0["expert 3: tokens 0, 4"]
    gather --> e1["expert 7: token 1"]
    gather --> e2["expert 9: tokens 2, 3, 5"]
    e0 --> scatter["scatter: weighted add back per token"]
    e1 --> scatter
    e2 --> scatter
    scatter --> out["MoE output, T tokens"]
```

### GPU expert offloading (`src/cuda/moe.rs`, `src/moe_cache.rs`)

If the experts fit in VRAM they all stay resident. Otherwise
(`MoEManager`, `LRU_MoEManager`, `MRU_MoEManager`):

* each expert is packed gate|up|down into one pinned buffer, and its scales
  stay on the device;
* `ExpertCache` owns a fixed number of device slots and decides what to
  evict: LRU, or *layer distance* (evict the expert whose layer comes up
  last in the cyclic layer order, skipping protected layers);
* `CudaResidency` uploads on the transfer stream and records an event per
  expert. The compute stream waits on that event before using the slot, and
  the transfer stream waits for queued compute before overwriting an
  evicted slot;
* a layer loads its GPU experts in batches of at most half the pool, so a
  batch never evicts one of its own experts;
* **cold path**: an expert that is not resident and got at most
  `cpu_threshold` tokens runs on the CPU executor, straight from its pinned
  buffer with bf16 rounding that matches the GPU kernel. The result rows
  are uploaded while the GPU experts run;
* **prefetch**: during decode, layer `l` runs layer `l+2`'s router on its
  own hidden state and starts uploading the experts it picks.

How one layer decides where each of its experts runs:

```mermaid
flowchart TD
    a["experts with tokens in this layer"] --> b{"resident on the GPU?"}
    b -->|yes| g["run on the GPU"]
    b -->|no| c{"tokens at most cpu_threshold?"}
    c -->|yes| d["run on the CPU executor, straight from the pinned buffer"]
    d --> u["upload the output rows"]
    c -->|no| e["ExpertCache prepare: take a slot, evict by LRU or layer distance"]
    e --> f["transfer stream uploads the pinned expert and records an event"]
    f --> w["compute stream waits on the event"]
    w --> g
    g --> m["scatter into the layer output"]
    u --> m
```

Why the two streams cannot overwrite each other's data:

```mermaid
sequenceDiagram
    participant C as compute stream
    participant T as transfer stream
    participant H as pinned host memory
    Note over C,T: expert e must be loaded into slot s
    C->>T: event after the last kernel that reads slot s (the evicted expert)
    T->>T: wait for that event before overwriting the slot
    H->>T: copy the packed expert into slot s
    T->>C: record the ready event for e
    C->>C: wait for the ready event, run e on the GPU
    C->>T: event after e's kernels, slot s may be reused
```

Speculative prefetch during decode:

```mermaid
flowchart LR
    l["hidden state at layer l"] --> r["router of layer l+2"]
    r --> p["experts it would pick"]
    p --> up["transfer stream uploads them early"]
    l --> run["layer l runs as usual"]
    up -.->|"already resident when layer l+2 starts"| run2["layer l+2"]
    run --> run2
```

## CUDA backend (`src/cuda`)

cudarc's raw driver API, with `libcuda`, `libnvrtc`, `libcublas` (and
`libnvidia-ml` for energy) loaded at runtime. `kernels.cu` is compiled with
NVRTC for the device's compute capability on first use and includes no CUDA
headers: bf16 and f16 are raw 16-bit values. Kernels are instantiated per
activation type (`*_bf16`, `*_f16`). Dense linears use cuBLAS GemmEx and FP8
linears a warp-per-row GEMV. Attention is a paged kernel (one block per
query row and head, four warps over the keys, merged in shared memory) that
replaces the cuDNN frontend SDPA graph. There are three streams: compute,
transfer (expert uploads) and secondary.

## Tokenizer

Byte-level BPE from `tokenizer.json`. Added tokens are split out first,
then NFC normalisation, then the Qwen pre-tokenizer regex (matched by hand
because `regex` has no lookahead), the GPT-2 byte alphabet, and merges by
rank on a heap over a linked symbol list. `StreamDecoder` buffers partial
UTF-8 while streaming.

```mermaid
flowchart LR
    text["text"] --> sp["split out added tokens"]
    sp --> nfc["NFC normalisation"]
    nfc --> pre["Qwen pre-tokenizer regex"]
    pre --> by["GPT-2 byte alphabet"]
    by --> bpe["BPE merges by rank, heap over a linked list"]
    bpe --> ids["token ids"]
```

## Verification

* `tests/model_parity.rs`: three tiny checkpoints against a float64 numpy
  transcription of `transformers`' Qwen3 / Qwen3-MoE code (dense with tied
  embeddings; MoE with a dense first layer; FP8 MoE with partial blocks).
* `tests/tokenizer_parity.rs`: 324 strings against HuggingFace
  `tokenizers`, using the exact Qwen pre-tokenizer configuration.
* `tests/cuda_parity.rs`: GPU vs CPU for each op, each model, and offloaded
  vs resident MoE. Needs a GPU.
* `cuda::tests::kernels_compile_for_all_archs`: NVRTC only, runs in CI.
