#!/usr/bin/env python3
"""Build a small Qwen-style byte-level BPE tokenizer with HuggingFace
`tokenizers` and record reference encodings for the Rust parity test.

The tokenizer uses exactly the Qwen2/Qwen3 normalizer + pre-tokenizer
configuration, so any disagreement in the test points at our implementation
rather than at a different setup.

    pip install tokenizers
    python scripts/tokenizer_fixture.py tests/fixtures/tokenizer
"""
import glob
import json
import os
import sys

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

QWEN_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

MULTILINGUAL = [
    "The quick brown fox jumps over the lazy dog. It's 12:45 and we've got 3 apples.",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.",
    "Ich möchte gerne ein Stück Kuchen, danke schön! Größe: 42.",
    "Le café est très bon, n'est-ce pas ? Voilà l'été.",
    "你好，世界！今天天气很好。我们一起去公园吧。",
    "こんにちは世界。カタカナとひらがな、漢字。",
    "안녕하세요 세계! 반갑습니다.",
    "Привет, мир! Как дела? Всё хорошо.",
    "مرحبا بالعالم! كيف حالك؟",
    "नमस्ते दुनिया! आप कैसे हैं?",
    "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
    "for (int i = 0; i < n; ++i) { sum += a[i] * b[i]; }\n\n\n",
    "Emoji test 🙂🚀🔥 and ✓ ✗ → ← ∑ ∫ √π ≈ 3.14159",
    "  leading spaces\ttabs\t\tand\r\nwindows newlines  \n",
]

CASES = [
    "",
    "Hello world",
    "hello   world  ",
    "I'm sure they'LL say it's FINE, isn't it?",
    "Numbers: 1234567890 and ١٢٣ and １２３",
    "Ask the model something: what is 2+2?",
    "tabs\tand\nnewlines\r\n\r\nmixed \n \n end",
    "trailing whitespace   ",
    "   ",
    "\n",
    "á combining é marks (decomposed é)",
    "Precomposed é vs decomposed é normalise the same",
    "你好，世界！今天天气很好。",
    "नमस्ते दुनिया",
    "Привет мир",
    "🙂🙂 emojis 🚀",
    "x = foo(bar[1], {'k': v}) # comment",
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
    "<think>\n\n</think>\n\nThe answer is 42.",
    "mixed<|im_end|>special<|endoftext|>tokens",
    "URL: https://example.com/path?q=1&r=2#frag",
    "!!!??? ... --- ***",
    "Ünïcödé ÀÉÎÕÜ ß ſ K",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
]


def random_cases(n, seed=1234):
    """Random strings biased towards the regex's tricky boundaries."""
    import random

    rng = random.Random(seed)
    pools = [
        "abcxyzABCXYZ", "0123456789", " ", "  ", "\t", "\n", "\r\n", "'", "'s", "'LL", "'ve",
        ".,;:!?()[]{}<>-_=+*/\\\"#@$%^&|~`", "éßñøüÉ", "你好世界", "жыщЮ", "١٢٣", "２３", "½¾",
        "\u0301\u0308", "🙂🚀", "\u00a0", "\u3000", "<|im_start|>", "<|im_end|>", "<think>",
    ]
    out = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(1, 12)):
            pool = rng.choice(pools)
            if pool.startswith("<") and len(pool) > 3:
                parts.append(pool)
            else:
                parts.append("".join(rng.choice(pool) for _ in range(rng.randint(1, 4))) if len(pool) > 2 else pool)
        out.append("".join(parts))
    return out


def corpus():
    texts = list(MULTILINGUAL) * 20
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for path in sorted(glob.glob(os.path.join(root, "src", "**", "*.rs"), recursive=True)):
        with open(path, encoding="utf-8") as f:
            texts.append(f.read())
    import tokenizers  # its own sources/docs are a convenient English corpus

    for path in sorted(glob.glob(os.path.join(os.path.dirname(tokenizers.__file__), "**", "*.pyi"), recursive=True)):
        with open(path, encoding="utf-8") as f:
            texts.append(f.read())
    return texts


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(QWEN_PATTERN), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=3000,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(corpus(), trainer)
    tok.add_special_tokens(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    tok.add_tokens(["<think>", "</think>"])
    tok.save(os.path.join(out_dir, "tokenizer.json"))

    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
        json.dump({"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "bos_token": None}, f, indent=2)

    cases = []
    for text in CASES + random_cases(300):
        ids = tok.encode(text, add_special_tokens=False).ids
        cases.append({"text": text, "ids": ids, "decoded": tok.decode(ids, skip_special_tokens=True)})
    with open(os.path.join(out_dir, "cases.json"), "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)
    print(f"vocab={tok.get_vocab_size()} cases={len(cases)} -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/tokenizer")
