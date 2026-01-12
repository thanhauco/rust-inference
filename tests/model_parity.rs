//! End-to-end check of the Qwen3 forward pass against the
//! float64 numpy reference in scripts/reference_qwen3.py: prefill logits for
//! every prompt position, then incremental decode through the paged KV cache.

use garlic::Tensor;
use garlic::models::{self, LogitsMode};
use serde::Deserialize;

#[derive(Deserialize)]
struct Expected {
    prompt: Vec<u32>,
    generated: Vec<u32>,
}

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn read_f32(path: &str) -> Vec<f32> {
    std::fs::read(path).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(worst <= 2e-4 * scale + 1e-5, "{what}: max abs diff {worst} (logit scale {scale})");
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

fn check(name: &str) {
    let dir = fixture(name);
    let model = models::from_pretrained(&dir).unwrap();
    let vocab = model.config().config.vocab_size;
    let exp: Expected =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/expected.json")).unwrap()).unwrap();
    let prefill_ref = read_f32(&format!("{dir}/expected_prefill.f32"));
    let decode_ref = read_f32(&format!("{dir}/expected_decode.f32"));

    let mut caches = model.new_kv_caches(1, 8, 4);
    let s = exp.prompt.len();
    let ids = Tensor::from_u32(&exp.prompt, &[1, s]).unwrap();
    let pos: Vec<u32> = (0..s as u32).collect();
    let pos = Tensor::from_u32(&pos, &[1, s]).unwrap();
    let logits = model.forward(&ids, &pos, &mut caches, LogitsMode::All).unwrap();
    assert_eq!(logits.shape(), &[1, s, vocab]);
    assert_close(logits.as_slice::<f32>(), &prefill_ref, &format!("{name} prefill"));

    let mut next = argmax(&logits.as_slice::<f32>()[(s - 1) * vocab..]);
    assert_eq!(next, exp.generated[0], "{name}: first generated token");
    for step in 0..decode_ref.len() / vocab {
        let ids = Tensor::from_u32(&[next], &[1, 1]).unwrap();
        let pos = Tensor::from_u32(&[(s + step) as u32], &[1, 1]).unwrap();
        let logits = model.forward(&ids, &pos, &mut caches, LogitsMode::Last).unwrap();
        let want = &decode_ref[step * vocab..(step + 1) * vocab];
        assert_close(logits.as_slice::<f32>(), want, &format!("{name} decode step {step}"));
        next = argmax(logits.as_slice::<f32>());
        assert_eq!(next, exp.generated[step + 1], "{name}: token {}", step + 1);
    }
}

#[test]
fn qwen3_dense_tied() {
    check("tiny-qwen3");
}

#[test]
fn last_mode_matches_all_mode() {
    let model = models::from_pretrained(fixture("tiny-qwen3")).unwrap();
    let vocab = model.config().config.vocab_size;
    let ids = Tensor::from_u32(&[5, 17, 300, 42], &[1, 4]).unwrap();
    let pos = Tensor::from_u32(&[0, 1, 2, 3], &[1, 4]).unwrap();
    let mut c1 = model.new_kv_caches(1, 4, 4);
    let mut c2 = model.new_kv_caches(1, 4, 4);
    let all = model.forward(&ids, &pos, &mut c1, LogitsMode::All).unwrap();
    let last = model.forward(&ids, &pos, &mut c2, LogitsMode::Last).unwrap();
    assert_eq!(last.shape(), &[1, 1, vocab]);
    assert_eq!(&all.as_slice::<f32>()[3 * vocab..], last.as_slice::<f32>());
}
