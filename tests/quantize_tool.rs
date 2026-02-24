//! Round trip through the `quantize` binary: the FP8 model must load and
//! stay close to the original.

use garlic::models::{self, LogitsMode};
use garlic::{DType, Tensor};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[test]
fn quantized_tiny_model_tracks_original() {
    let src = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tiny-qwen3-moe");
    let out = std::env::temp_dir().join(format!("garlic-quantize-{}", std::process::id()));
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_quantize"))
        .args([src, out.to_str().unwrap(), "--block", "32"])
        .status()
        .unwrap();
    assert!(status.success());

    let q = models::from_pretrained(&out).unwrap();
    let r = models::from_pretrained(src).unwrap();
    let cfg = q.config().config.clone();
    assert!(cfg.quantization_config.is_some());

    let ids = Tensor::from_u32(&[1, 50, 300, 7, 42, 99], &[1, 6]).unwrap();
    let pos = Tensor::from_u32(&[0, 1, 2, 3, 4, 5], &[1, 6]).unwrap();
    let lq = q.forward(&ids, &pos, &mut q.new_kv_caches(1, 4, 4), LogitsMode::All).unwrap();
    let lr = r.forward(&ids, &pos, &mut r.new_kv_caches(1, 4, 4), LogitsMode::All).unwrap();
    let c = cosine(lq.as_slice::<f32>(), lr.as_slice::<f32>());
    assert!(c > 0.99, "cosine similarity {c}");

    // the projections really are FP8 now, the router is untouched
    let st = garlic::safetensors::SafeTensorReader::open(out.join("model.safetensors")).unwrap();
    assert_eq!(st.info("model.layers.1.mlp.experts.0.up_proj.weight").unwrap().dtype, DType::F8E4M3);
    assert_eq!(st.info("model.layers.1.mlp.gate.weight").unwrap().dtype, DType::BF16);
    assert_eq!(st.info("model.layers.1.mlp.experts.0.up_proj.weight_scale_inv").unwrap().shape, vec![1, 2]);
}
