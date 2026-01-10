//! Interactive greedy generation, the Rust counterpart of `inference.cpp`.
//!
//!     runner <MODEL_DIR> [--prompt "..."] [--max-new-tokens 600] [--chat]

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use garlic::models::{self, LogitsMode};
use garlic::ops;
use garlic::tokenizer::{BpeTokenizer, StreamDecoder};
use garlic::Tensor;

#[derive(Parser, Debug)]
#[command(version, about = "Greedy text generation with a Qwen3 / Qwen3-MoE checkpoint")]
struct Args {
    /// HuggingFace checkpoint directory (config.json, *.safetensors, tokenizer.json)
    model_dir: PathBuf,

    /// Prompt text; read from stdin when omitted
    #[arg(short, long)]
    prompt: Option<String>,

    /// Number of tokens to generate
    #[arg(short = 'n', long, default_value_t = 600)]
    max_new_tokens: usize,

    /// Wrap the prompt in the Qwen chat template
    #[arg(long)]
    chat: bool,

    /// With --chat, leave thinking mode enabled
    #[arg(long, requires = "chat")]
    think: bool,

    /// Keep generating past end-of-sequence tokens (for benchmarking)
    #[arg(long)]
    ignore_eos: bool,

    /// Directory holding tokenizer.json, if not the model directory
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// KV cache pages per layer (32 tokens each)
    #[arg(long, default_value_t = 256)]
    max_pages: usize,

    /// Maximum KV pages per sequence
    #[arg(long, default_value_t = 128)]
    pages_per_seq: usize,

    /// Worker threads (defaults to all cores)
    #[arg(short = 't', long)]
    threads: Option<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if let Some(n) = args.threads {
        rayon::ThreadPoolBuilder::new().num_threads(n).build_global()?;
    }

    let tokenizer = BpeTokenizer::load(args.tokenizer.as_ref().unwrap_or(&args.model_dir))?;
    let t0 = Instant::now();
    let model = models::from_pretrained(&args.model_dir)?;
    eprintln!(
        "loaded {:?} ({} layers) in {:.2}s",
        model.config().architecture,
        model.config().config.num_hidden_layers,
        t0.elapsed().as_secs_f64()
    );
    let mut caches = model.new_kv_caches(1, args.max_pages, args.pages_per_seq);

    let prompt = match &args.prompt {
        Some(p) => p.clone(),
        None => {
            print!("Ask the model something: ");
            std::io::stdout().flush()?;
            let mut line = String::new();
            std::io::stdin().lock().read_line(&mut line)?;
            line.trim_end_matches(['\n', '\r']).to_string()
        }
    };
    let text = if args.chat { BpeTokenizer::chat_prompt(&prompt, args.think) } else { prompt };
    let prompt_ids = tokenizer.encode(&text);
    if prompt_ids.is_empty() {
        return Err("empty prompt".into());
    }

    let mut stdout = std::io::stdout().lock();
    let mut stream = StreamDecoder::new();
    let n = prompt_ids.len();

    let t_prefill = Instant::now();
    let ids = Tensor::from_u32(&prompt_ids, &[1, n])?;
    let pos: Vec<u32> = (0..n as u32).collect();
    let logits = model.forward(&ids, &Tensor::from_u32(&pos, &[1, n])?, &mut caches, LogitsMode::Last)?;
    let mut next = ops::argmax_last(&logits)?[0];
    let prefill_s = t_prefill.elapsed().as_secs_f64();

    let t_decode = Instant::now();
    let mut generated = 1;
    loop {
        if !args.ignore_eos && tokenizer.is_stop_token(next) {
            break;
        }
        write!(stdout, "{}", stream.push(&tokenizer, next))?;
        stdout.flush()?;
        if generated >= args.max_new_tokens {
            break;
        }
        let pos = (n + generated - 1) as u32;
        let logits = model.forward(
            &Tensor::from_u32(&[next], &[1, 1])?,
            &Tensor::from_u32(&[pos], &[1, 1])?,
            &mut caches,
            LogitsMode::Last,
        )?;
        next = ops::argmax_last(&logits)?[0];
        generated += 1;
    }
    writeln!(stdout, "{}", stream.finish())?;
    let decode_s = t_decode.elapsed().as_secs_f64();

    eprintln!();
    eprintln!("prefill: {n} tokens in {:.3}s ({:.1} tok/s)", prefill_s, n as f64 / prefill_s);
    let steps = generated - 1;
    if steps > 0 {
        eprintln!("decode:  {steps} tokens in {:.3}s ({:.2} tok/s)", decode_s, steps as f64 / decode_s);
    }
    Ok(())
}
