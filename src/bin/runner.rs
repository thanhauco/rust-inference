//! Interactive greedy generation, the Rust counterpart of `inference.cpp`.
//!
//!     runner <MODEL_DIR> [--prompt "..."] [--max-new-tokens 600] [--chat]

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;

// glibc hands large freed blocks back to the kernel, so every op's output
// buffer would page-fault again on first touch, and the faults from parallel
// writers contend in the kernel. mimalloc keeps the memory mapped.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
use garlic::kvcache::KvCache;
use garlic::models::{self, LanguageModel, LogitsMode};
use garlic::profiler::Section;
use garlic::profiler::{self, PowerProfiler};
use garlic::sampling::{Sampler, SamplingParams};
use garlic::tokenizer::{BpeTokenizer, StreamDecoder};
use garlic::{Tensor, profile};

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

    /// Print a per-section timing breakdown at the end
    #[arg(long)]
    profile: bool,

    /// Measure energy (RAPL / NVML) for idle, load and inference phases
    #[arg(long)]
    power: bool,

    /// Length of the idle power measurements, in seconds
    #[arg(long, default_value_t = 10)]
    power_idle_secs: u64,

    /// Sampling temperature; 0 = greedy
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// Keep only the k most likely tokens (0 = off)
    #[arg(long, default_value_t = 0)]
    top_k: usize,

    /// Nucleus sampling threshold (1.0 = off)
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,

    /// Penalty for tokens already in the context (1.0 = off)
    #[arg(long, default_value_t = 1.0)]
    repetition_penalty: f32,

    /// Sampling seed
    #[arg(long, default_value_t = 0x5EED)]
    seed: u64,

    /// Run on the GPU (needs the `cuda` feature)
    #[arg(long, value_enum, default_value_t = DeviceArg::Cpu)]
    device: DeviceArg,

    /// GPU: offload MoE experts to pinned host memory with this many device
    /// slots (default: offload only if the experts do not fit)
    #[arg(long)]
    expert_slots: Option<usize>,

    /// GPU offload: MoE layers whose experts always stay on the device
    #[arg(long, default_value_t = 0)]
    resident_moe_layers: usize,

    /// GPU offload: evict by recency instead of by upcoming layer
    #[arg(long)]
    lru: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum DeviceArg {
    Cpu,
    Cuda,
}

fn load_model(args: &Args) -> Result<Box<dyn LanguageModel>, Box<dyn std::error::Error>> {
    match args.device {
        DeviceArg::Cpu => Ok(models::from_pretrained(&args.model_dir)?),
        #[cfg(feature = "cuda")]
        DeviceArg::Cuda => {
            use garlic::cuda::moe::OffloadOptions;
            use garlic::models::cuda::CudaOptions;
            use garlic::moe_cache::Policy;
            let config = models::ModelConfig::from_pretrained(&args.model_dir)?;
            let layers = config.config.num_hidden_layers;
            let offload = args.expert_slots.map(|slots| OffloadOptions {
                slots,
                resident_layers: args.resident_moe_layers,
                policy: if args.lru { Policy::Lru } else { Policy::LayerDistance { num_layers: layers } },
            });
            let model = models::Qwen3::load(config, &args.model_dir)?;
            Ok(Box::new(model.to_cuda(&CudaOptions { offload })?))
        }
        #[cfg(not(feature = "cuda"))]
        DeviceArg::Cuda => Err("this build has no CUDA support (rebuild with --features cuda)".into()),
    }
}

/// Energy measurement following the C++ runner: let the previous phase's
/// draw settle for a few seconds, then measure.
struct Power {
    counters: Option<Vec<Box<dyn profiler::EnergyCounter>>>,
}

impl Power {
    const SETTLE: Duration = Duration::from_secs(5);

    fn new(enabled: bool) -> Self {
        let counters = enabled.then(profiler::available_counters);
        if counters.as_ref().is_some_and(Vec::is_empty) {
            eprintln!("--power: no energy counters available (RAPL/NVML)");
        }
        Power { counters }
    }

    fn measure<T>(&mut self, label: &str, f: impl FnOnce() -> T) -> T {
        let Some(counters) = self.counters.take() else {
            return f();
        };
        let p = PowerProfiler::start(counters, Self::SETTLE);
        let out = f();
        let (reading, counters) = p.finish();
        eprintln!("{label} {reading}");
        self.counters = Some(counters);
        out
    }

    fn idle(&mut self, label: &str, secs: u64) {
        if self.counters.is_some() {
            self.measure(label, || std::thread::sleep(Duration::from_secs(secs)));
        }
    }
}

fn main() -> std::process::ExitCode {
    match run(Args::parse()) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(n) = args.threads {
        rayon::ThreadPoolBuilder::new().num_threads(n).build_global()?;
    }

    profiler::enable(args.profile);
    let mut power = Power::new(args.power);
    power.idle("IDLE", args.power_idle_secs);

    let tokenizer = BpeTokenizer::load(args.tokenizer.as_ref().unwrap_or(&args.model_dir))?;
    let t0 = Instant::now();
    let model = power.measure("MODEL LOADING", || load_model(&args))?;
    eprintln!(
        "loaded {:?} ({} layers) in {:.2}s",
        model.config().architecture,
        model.config().config.num_hidden_layers,
        t0.elapsed().as_secs_f64()
    );
    power.idle("IDLE WITH MODEL", args.power_idle_secs);
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

    let n = prompt_ids.len();
    let stats = power.measure("INFERENCE", || generate(&args, &*model, &tokenizer, &prompt_ids, &mut caches))?;

    eprintln!();
    eprintln!("prefill: {n} tokens in {:.3}s ({:.1} tok/s)", stats.prefill_s, n as f64 / stats.prefill_s);
    if stats.steps > 0 {
        eprintln!(
            "decode:  {} tokens in {:.3}s ({:.2} tok/s)",
            stats.steps,
            stats.decode_s,
            stats.steps as f64 / stats.decode_s
        );
    }
    if args.profile {
        eprintln!("\n{}", profiler::report());
    }
    Ok(())
}

struct Stats {
    prefill_s: f64,
    decode_s: f64,
    steps: usize,
}

fn generate(
    args: &Args,
    model: &dyn LanguageModel,
    tokenizer: &BpeTokenizer,
    prompt_ids: &[u32],
    caches: &mut [KvCache],
) -> Result<Stats, Box<dyn std::error::Error>> {
    let mut stdout = std::io::stdout().lock();
    let mut stream = StreamDecoder::new();
    let n = prompt_ids.len();

    let t_prefill = Instant::now();
    let ids = Tensor::from_u32(prompt_ids, &[1, n])?;
    let pos: Vec<u32> = (0..n as u32).collect();
    let logits = model.forward(&ids, &Tensor::from_u32(&pos, &[1, n])?, caches, LogitsMode::Last)?;
    let mut sampler = Sampler::new(SamplingParams {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        seed: args.seed,
    });
    let mut history = prompt_ids.to_vec();
    let mut next = profile!(Section::Sampling, sampler.sample(&logits.to_f32_vec()?, &history));
    let prefill_s = t_prefill.elapsed().as_secs_f64();

    let t_decode = Instant::now();
    let mut generated = 1;
    loop {
        if !args.ignore_eos && tokenizer.is_stop_token(next) {
            break;
        }
        write!(stdout, "{}", stream.push(tokenizer, next))?;
        stdout.flush()?;
        if generated >= args.max_new_tokens {
            break;
        }
        let pos = (n + generated - 1) as u32;
        let logits = model.forward(
            &Tensor::from_u32(&[next], &[1, 1])?,
            &Tensor::from_u32(&[pos], &[1, 1])?,
            caches,
            LogitsMode::Last,
        )?;
        history.push(next);
        next = profile!(Section::Sampling, sampler.sample(&logits.to_f32_vec()?, &history));
        generated += 1;
    }
    writeln!(stdout, "{}", stream.finish())?;
    Ok(Stats { prefill_s, decode_s: t_decode.elapsed().as_secs_f64(), steps: generated - 1 })
}
