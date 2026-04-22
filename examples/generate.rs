//! Using garlic as a library: greedy generation into a `String`.
//!
//!     cargo run --release --example generate -- <MODEL_DIR> "Why is the sky blue?" [max_new_tokens]

use garlic::models::{self, LogitsMode};
use garlic::tokenizer::BpeTokenizer;
use garlic::{Result, Tensor, ops};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: generate <MODEL_DIR> <PROMPT> [MAX_NEW_TOKENS]");
    let prompt = args.next().unwrap_or_else(|| "Hello".to_string());
    let max_new: usize = args.next().and_then(|n| n.parse().ok()).unwrap_or(64);

    let tokenizer = BpeTokenizer::load(&dir)?;
    let model = models::from_pretrained(&dir)?;
    let mut caches = model.new_kv_caches(1, 64, 32);

    let mut input = tokenizer.encode(&BpeTokenizer::chat_prompt(&prompt, false));
    let mut generated = Vec::new();
    let mut position = 0u32;
    for _ in 0..max_new {
        let n = input.len();
        let positions: Vec<u32> = (position..position + n as u32).collect();
        let logits = model.forward(
            &Tensor::from_u32(&input, &[1, n])?,
            &Tensor::from_u32(&positions, &[1, n])?,
            &mut caches,
            LogitsMode::Last,
        )?;
        let next = ops::argmax_last(&logits)?[0];
        if tokenizer.is_stop_token(next) {
            break;
        }
        position += n as u32;
        generated.push(next);
        input = vec![next];
    }
    println!("{}", tokenizer.decode(&generated));
    Ok(())
}
