#!/usr/bin/env python3
"""Tiny random Qwen3 / Qwen3-MoE checkpoints plus float64 reference outputs.

The forward pass below follows `transformers`' modeling_qwen3.py and
modeling_qwen3_moe.py step by step (RMSNorm, per-head q/k norm, rotate-half
RoPE, GQA, causal softmax attention, SwiGLU, softmax top-k router with
norm_topk_prob), evaluated in float64 on the exact (bf16- or fp8-rounded)
weights written to disk. The Rust engine computes in f32, so it should agree
to ~1e-4 relative.

    pip install numpy
    python scripts/reference_qwen3.py tests/fixtures
"""
import json
import os
import struct
import sys

import numpy as np

# ---------------------------------------------------------------------------
# number formats
# ---------------------------------------------------------------------------


def to_bf16_bits(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype(np.uint16)


def from_bf16_bits(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


def fp8_table():
    vals = np.zeros(256, dtype=np.float64)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 0xF, b & 7
        if e == 0:
            v = m * 2.0**-9
        elif e == 15 and m == 7:
            v = np.nan
        else:
            v = (1 + m / 8) * 2.0 ** (e - 7)
        vals[b] = -v if s else v
    return vals


FP8 = fp8_table()


def to_fp8(x):
    """Nearest finite E4M3 byte for every element (values pre-scaled into range)."""
    finite = [b for b in range(256) if not np.isnan(FP8[b])]
    vals = FP8[finite]
    order = np.argsort(vals)
    sv, sb = vals[order], np.array(finite)[order]
    x = np.clip(x, -448, 448)
    idx = np.clip(np.searchsorted(sv, x), 1, len(sv) - 1)
    left, right = sv[idx - 1], sv[idx]
    pick = np.where(np.abs(x - left) <= np.abs(right - x), idx - 1, idx)
    return sb[pick].astype(np.uint8)


def quantize_fp8(w, block=128):
    n, k = w.shape
    sn, sk = -(-n // block), -(-k // block)
    q = np.zeros((n, k), dtype=np.uint8)
    scale = np.zeros((sn, sk), dtype=np.float32)
    for i in range(sn):
        for j in range(sk):
            blk = w[i * block:(i + 1) * block, j * block:(j + 1) * block]
            s = max(float(np.abs(blk).max()) / 448.0, 1e-8)
            scale[i, j] = s
            q[i * block:(i + 1) * block, j * block:(j + 1) * block] = to_fp8(blk / s)
    return q, scale


def dequant_fp8(q, scale, block=128):
    w = FP8[q]
    rows = np.arange(q.shape[0]) // block
    cols = np.arange(q.shape[1]) // block
    return w * scale.astype(np.float64)[rows][:, cols]


# ---------------------------------------------------------------------------
# safetensors
# ---------------------------------------------------------------------------


def save_safetensors(path, tensors):
    header, blobs, off = {}, [], 0
    for name in sorted(tensors):
        dtype, arr = tensors[name]
        raw = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - (8 + len(h)) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


# ---------------------------------------------------------------------------
# reference model
# ---------------------------------------------------------------------------


def rms_norm(x, w, eps):
    var = np.mean(x * x, axis=-1, keepdims=True)
    return w * (x / np.sqrt(var + eps))


def rope_tables(positions, head_dim, theta):
    exps = np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim)
    inv_freq = (np.float32(1.0) / (np.float32(theta) ** exps)).astype(np.float32)
    freqs = positions.astype(np.float32)[:, None] * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1).astype(np.float64)
    return np.cos(emb), np.sin(emb)


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def silu(x):
    return x / (1 + np.exp(-x))


def mlp(x, p):
    return (silu(x @ p["gate_proj"].T) * (x @ p["up_proj"].T)) @ p["down_proj"].T


def moe(x, p, cfg):
    logits = x @ p["gate"].T
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    k = cfg["num_experts_per_tok"]
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        # stable descending order, lower index first on ties (torch.topk)
        top = sorted(range(probs.shape[1]), key=lambda e: (-probs[t, e], e))[:k]
        w = probs[t, top]
        if cfg.get("norm_topk_prob", True):
            w = w / w.sum()
        for e, we in zip(top, w):
            out[t] += we * mlp(x[t:t + 1], p["experts"][e])[0]
    return out


def forward(cfg, W, ids):
    """Full-sequence forward, returns logits [S, V] (float64)."""
    S = len(ids)
    H, KV, D = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps = cfg["rms_norm_eps"]
    h = W["embed"][ids]
    cos, sin = rope_tables(np.arange(S), D, cfg["rope_theta"])
    mask = np.triu(np.full((S, S), -np.inf), 1)
    for L in W["layers"]:
        res = h
        x = rms_norm(h, L["input_layernorm"], eps)
        q = (x @ L["q_proj"].T).reshape(S, H, D)
        k = (x @ L["k_proj"].T).reshape(S, KV, D)
        v = (x @ L["v_proj"].T).reshape(S, KV, D)
        q = rms_norm(q, L["q_norm"], eps)
        k = rms_norm(k, L["k_norm"], eps)
        q = q * cos[:, None, :] + rotate_half(q) * sin[:, None, :]
        k = k * cos[:, None, :] + rotate_half(k) * sin[:, None, :]
        rep = H // KV
        k = np.repeat(k, rep, axis=1)
        v = np.repeat(v, rep, axis=1)
        att = np.einsum("shd,thd->hst", q, k) / np.sqrt(D) + mask
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        o = np.einsum("hst,thd->shd", att, v).reshape(S, H * D)
        h = res + o @ L["o_proj"].T
        res = h
        x = rms_norm(h, L["post_attention_layernorm"], eps)
        h = res + (moe(x, L["mlp"], cfg) if "gate" in L["mlp"] else mlp(x, L["mlp"]))
    h = rms_norm(h, W["norm"], eps)
    return h @ W["lm_head"].T


# ---------------------------------------------------------------------------
# checkpoint generation
# ---------------------------------------------------------------------------


class Builder:
    """Creates weights, stores them in the chosen format and returns the
    exact values the file decodes to."""

    def __init__(self, rng, fp8):
        self.rng, self.fp8, self.tensors = rng, fp8, {}

    def bf16(self, name, arr):
        bits = to_bf16_bits(arr)
        self.tensors[name] = ("BF16", bits)
        return from_bf16_bits(bits).astype(np.float64)

    def linear(self, name, n, k, std, quantize=True):
        w = self.rng.normal(0, std, size=(n, k)).astype(np.float32)
        if self.fp8 and quantize:
            q, s = quantize_fp8(w)
            self.tensors[name] = ("F8_E4M3", q)
            self.tensors[name + "_scale_inv"] = ("F32", s)
            return dequant_fp8(q, s)
        return self.bf16(name, w)

    def norm(self, name, n):
        return self.bf16(name, 1.0 + 0.1 * self.rng.normal(size=n).astype(np.float32))


def build(cfg, seed, fp8, std=0.15, embed_std=1.0):
    rng = np.random.default_rng(seed)
    b = Builder(rng, fp8)
    hd, D, V = cfg["hidden_size"], cfg["head_dim"], cfg["vocab_size"]
    H, KV = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    W = {"embed": b.bf16("model.embed_tokens.weight", rng.normal(0, embed_std, size=(V, hd)).astype(np.float32))}
    W["layers"] = []
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        L = {
            "input_layernorm": b.norm(p + "input_layernorm.weight", hd),
            "post_attention_layernorm": b.norm(p + "post_attention_layernorm.weight", hd),
            "q_proj": b.linear(p + "self_attn.q_proj.weight", H * D, hd, std),
            "k_proj": b.linear(p + "self_attn.k_proj.weight", KV * D, hd, std),
            "v_proj": b.linear(p + "self_attn.v_proj.weight", KV * D, hd, std),
            "o_proj": b.linear(p + "self_attn.o_proj.weight", hd, H * D, std),
            "q_norm": b.norm(p + "self_attn.q_norm.weight", D),
            "k_norm": b.norm(p + "self_attn.k_norm.weight", D),
        }
        sparse = cfg.get("num_experts", 0) > 0 and i not in cfg.get("mlp_only_layers", [])
        if sparse:
            E, I = cfg["num_experts"], cfg["moe_intermediate_size"]
            m = {"gate": b.linear(p + "mlp.gate.weight", E, hd, 0.3, quantize=False), "experts": []}
            for e in range(E):
                ep = f"{p}mlp.experts.{e}."
                m["experts"].append(
                    {
                        "gate_proj": b.linear(ep + "gate_proj.weight", I, hd, std),
                        "up_proj": b.linear(ep + "up_proj.weight", I, hd, std),
                        "down_proj": b.linear(ep + "down_proj.weight", hd, I, std),
                    }
                )
        else:
            I = cfg["intermediate_size"]
            m = {
                "gate_proj": b.linear(p + "mlp.gate_proj.weight", I, hd, std),
                "up_proj": b.linear(p + "mlp.up_proj.weight", I, hd, std),
                "down_proj": b.linear(p + "mlp.down_proj.weight", hd, I, std),
            }
        L["mlp"] = m
        W["layers"].append(L)
    W["norm"] = b.norm("model.norm.weight", hd)
    if cfg["tie_word_embeddings"]:
        W["lm_head"] = W["embed"]
    else:
        W["lm_head"] = b.linear("lm_head.weight", V, hd, 0.3, quantize=False)
    return W, b.tensors


def base_config(**kw):
    cfg = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": 512,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }
    cfg.update(kw)
    return cfg


VARIANTS = {
    # dense, tied embeddings, vocab covering the tokenizer fixture
    "tiny-qwen3": dict(
        seed=1, fp8=False, std=0.35, embed_std=0.5, cfg=base_config(vocab_size=3072, tie_word_embeddings=True)
    ),
    # MoE with a dense first layer (mlp_only_layers) and an untied lm_head
    "tiny-qwen3-moe": dict(
        seed=2,
        fp8=False,
        cfg=base_config(
            architectures=["Qwen3MoeForCausalLM"],
            model_type="qwen3_moe",
            num_hidden_layers=3,
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            norm_topk_prob=True,
            decoder_sparse_step=1,
            mlp_only_layers=[0],
        ),
    ),
    # FP8 block-scaled linear layers, partial 128-blocks everywhere
    "tiny-qwen3-moe-fp8": dict(
        seed=3,
        fp8=True,
        cfg=base_config(
            architectures=["Qwen3MoeForCausalLM"],
            model_type="qwen3_moe",
            hidden_size=160,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=40,
            num_experts=6,
            num_experts_per_tok=2,
            moe_intermediate_size=136,
            norm_topk_prob=True,
            quantization_config={"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]},
        ),
    ),
}

PROMPT_LEN, NEW_TOKENS = 11, 6


def main(root):
    for name, v in VARIANTS.items():
        out = os.path.join(root, name)
        os.makedirs(out, exist_ok=True)
        cfg = v["cfg"]
        W, tensors = build(cfg, v["seed"], v["fp8"], v.get("std", 0.15), v.get("embed_std", 1.0))
        save_safetensors(os.path.join(out, "model.safetensors"), tensors)
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

        rng = np.random.default_rng(100 + v["seed"])
        ids = [int(t) for t in rng.integers(0, cfg["vocab_size"], size=PROMPT_LEN)]
        prefill = forward(cfg, W, ids)
        seq, steps, margins = list(ids), [], []
        nxt = int(np.argmax(prefill[-1]))
        for _ in range(NEW_TOKENS):
            seq.append(nxt)
            logits = forward(cfg, W, seq)[-1]
            steps.append(logits)
            top2 = np.sort(logits)[-2:]
            margins.append(float(top2[1] - top2[0]))
            nxt = int(np.argmax(logits))
        generated = seq[PROMPT_LEN:] + [nxt]

        np.asarray(prefill, dtype="<f4").tofile(os.path.join(out, "expected_prefill.f32"))
        np.asarray(steps, dtype="<f4").tofile(os.path.join(out, "expected_decode.f32"))
        with open(os.path.join(out, "expected.json"), "w") as f:
            json.dump({"prompt": ids, "generated": generated, "min_margin": min(margins)}, f, indent=1)
        print(f"{name}: generated {generated}, min top-2 margin {min(margins):.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures")
