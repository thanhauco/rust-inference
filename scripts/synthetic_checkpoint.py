#!/usr/bin/env python3
"""Write a random checkpoint with the dimensions of a real Qwen3 model, for
benchmarking without downloading weights. Output is a normal HF-style
directory (config.json + model.safetensors[.index.json]).

    python scripts/synthetic_checkpoint.py qwen3-0.6b /tmp/q06
    python scripts/synthetic_checkpoint.py qwen3-30b-a3b /tmp/q30 --fp8 --layers 4

Weights are random, so generated text is noise; throughput is what matters.
"""
import argparse
import json
import os
import struct

import numpy as np

PRESETS = {
    "qwen3-0.6b": dict(
        architectures=["Qwen3ForCausalLM"], vocab_size=151936, hidden_size=1024, intermediate_size=3072,
        num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, head_dim=128,
        tie_word_embeddings=True,
    ),
    "qwen3-4b": dict(
        architectures=["Qwen3ForCausalLM"], vocab_size=151936, hidden_size=2560, intermediate_size=9728,
        num_hidden_layers=36, num_attention_heads=32, num_key_value_heads=8, head_dim=128,
        tie_word_embeddings=True,
    ),
    "qwen3-30b-a3b": dict(
        architectures=["Qwen3MoeForCausalLM"], vocab_size=151936, hidden_size=2048, intermediate_size=6144,
        num_hidden_layers=48, num_attention_heads=32, num_key_value_heads=4, head_dim=128,
        num_experts=128, num_experts_per_tok=8, moe_intermediate_size=768, norm_topk_prob=True,
        tie_word_embeddings=False,
    ),
}

SHARD_BYTES = 2 << 30


def bf16_bits(x):
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def fp8_encoder():
    vals = []
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 0xF, b & 7
        if e == 15 and m == 7:
            continue
        v = m * 2.0**-9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
        vals.append((-v if s else v, b))
    vals.sort()
    sv = np.array([v for v, _ in vals])
    sb = np.array([b for _, b in vals], dtype=np.uint8)

    def enc(x):
        idx = np.clip(np.searchsorted(sv, x), 1, len(sv) - 1)
        left = sv[idx - 1]
        pick = np.where(np.abs(x - left) <= np.abs(sv[idx] - x), idx - 1, idx)
        return sb[pick]

    return enc


ENC = fp8_encoder()


def quantize(w, block=128):
    n, k = w.shape
    sn, sk = -(-n // block), -(-k // block)
    pad = np.zeros((sn * block, sk * block), dtype=np.float32)
    pad[:n, :k] = w
    blocks = pad.reshape(sn, block, sk, block)
    scale = np.maximum(np.abs(blocks).max(axis=(1, 3)) / 448.0, 1e-8).astype(np.float32)
    q = ENC(blocks / scale[:, None, :, None]).reshape(sn * block, sk * block)[:n, :k]
    return np.ascontiguousarray(q), scale


class Writer:
    def __init__(self, out):
        self.out, self.pending, self.size, self.shards, self.weight_map = out, {}, 0, 0, {}

    def add(self, name, dtype, arr):
        self.pending[name] = (dtype, arr)
        self.size += arr.nbytes
        if self.size >= SHARD_BYTES:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        self.shards += 1
        fname = f"model-{self.shards:05d}.safetensors"
        header, off = {}, 0
        for name in sorted(self.pending):
            dtype, arr = self.pending[name]
            header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [off, off + arr.nbytes]}
            off += arr.nbytes
            self.weight_map[name] = fname
        h = json.dumps(header, separators=(",", ":")).encode()
        h += b" " * ((8 - (8 + len(h)) % 8) % 8)
        with open(os.path.join(self.out, fname), "wb") as f:
            f.write(struct.pack("<Q", len(h)))
            f.write(h)
            for name in sorted(self.pending):
                f.write(np.ascontiguousarray(self.pending[name][1]).tobytes())
        self.pending, self.size = {}, 0

    def close(self):
        self.flush()
        with open(os.path.join(self.out, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": self.weight_map}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", choices=sorted(PRESETS))
    ap.add_argument("out")
    ap.add_argument("--fp8", action="store_true", help="FP8 E4M3 weights with 128x128 block scales")
    ap.add_argument("--layers", type=int, help="override num_hidden_layers")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = dict(PRESETS[args.preset])
    if args.layers:
        cfg["num_hidden_layers"] = args.layers
    cfg.update(rms_norm_eps=1e-6, rope_theta=1000000.0, max_position_embeddings=40960, torch_dtype="bfloat16")
    if args.fp8:
        cfg["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    rng = np.random.default_rng(args.seed)
    w = Writer(args.out)
    hd, V = cfg["hidden_size"], cfg["vocab_size"]

    def bf16(name, shape, std):
        w.add(name, "BF16", bf16_bits(rng.normal(0, std, size=shape).astype(np.float32)))

    def linear(name, n, k, quantized=True):
        x = rng.normal(0, 0.02, size=(n, k)).astype(np.float32)
        if args.fp8 and quantized:
            q, s = quantize(x)
            w.add(name, "F8_E4M3", q)
            w.add(name + "_scale_inv", "F32", s)
        else:
            w.add(name, "BF16", bf16_bits(x))

    def norm(name, n):
        w.add(name, "BF16", bf16_bits(np.ones(n, dtype=np.float32)))

    H, KV, D = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    bf16("model.embed_tokens.weight", (V, hd), 0.02)
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        norm(p + "input_layernorm.weight", hd)
        norm(p + "post_attention_layernorm.weight", hd)
        linear(p + "self_attn.q_proj.weight", H * D, hd)
        linear(p + "self_attn.k_proj.weight", KV * D, hd)
        linear(p + "self_attn.v_proj.weight", KV * D, hd)
        linear(p + "self_attn.o_proj.weight", hd, H * D)
        norm(p + "self_attn.q_norm.weight", D)
        norm(p + "self_attn.k_norm.weight", D)
        if cfg.get("num_experts"):
            linear(p + "mlp.gate.weight", cfg["num_experts"], hd, quantized=False)
            I = cfg["moe_intermediate_size"]
            for e in range(cfg["num_experts"]):
                ep = f"{p}mlp.experts.{e}."
                linear(ep + "gate_proj.weight", I, hd)
                linear(ep + "up_proj.weight", I, hd)
                linear(ep + "down_proj.weight", hd, I)
        else:
            I = cfg["intermediate_size"]
            linear(p + "mlp.gate_proj.weight", I, hd)
            linear(p + "mlp.up_proj.weight", I, hd)
            linear(p + "mlp.down_proj.weight", hd, I)
        print(f"layer {i + 1}/{cfg['num_hidden_layers']}", flush=True)
    norm("model.norm.weight", hd)
    if not cfg["tie_word_embeddings"]:
        linear("lm_head.weight", V, hd, quantized=False)
    w.close()


if __name__ == "__main__":
    main()
