//! Compare our BPE tokenizer against reference encodings produced by
//! HuggingFace `tokenizers` (see scripts/tokenizer_fixture.py).

use garlic::tokenizer::BpeTokenizer;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    text: String,
    ids: Vec<u32>,
    decoded: String,
}

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tokenizer");

#[test]
fn encodings_match_huggingface() {
    let tok = BpeTokenizer::load(FIXTURE).unwrap();
    let cases: Vec<Case> =
        serde_json::from_str(&std::fs::read_to_string(format!("{FIXTURE}/cases.json")).unwrap()).unwrap();
    assert!(!cases.is_empty());
    let mut failures = Vec::new();
    for case in &cases {
        let ids = tok.encode(&case.text);
        if ids != case.ids {
            failures.push(format!("{:?}\n  ours: {:?}\n  hf:   {:?}", case.text, ids, case.ids));
        }
        let decoded = tok.decode(&case.ids);
        if decoded != case.decoded {
            failures.push(format!("decode {:?}: ours {:?} hf {:?}", case.text, decoded, case.decoded));
        }
    }
    assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn config_overrides_special_ids() {
    let tok = BpeTokenizer::load(FIXTURE).unwrap();
    // tokenizer_config.json sets eos to <|im_end|>
    assert_eq!(tok.eos_id(), tok.token_id("<|im_end|>"));
    assert_eq!(tok.pad_id(), tok.token_id("<|endoftext|>"));
    assert!(tok.is_stop_token(tok.token_id("<|im_end|>").unwrap()));
}
