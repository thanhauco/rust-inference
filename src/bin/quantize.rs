//! Convert a BF16/F16/F32 checkpoint to block-scaled FP8 (E4M3).
//!
//!     quantize <IN_DIR> <OUT_DIR> [--block 128]
//!
//! Linear projections are quantized; embeddings, LM head, norms and MoE
//! routers are copied unchanged. Shards are converted one tensor at a time,
//! and tokenizer / generation files are copied over.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use clap::Parser;
use garlic::DType;
use garlic::quantize::{quantize_fp8, should_quantize};
use garlic::safetensors::{SCALE_SUFFIX, SafeTensorReader, StreamWriter, TensorEntry};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser, Debug)]
#[command(version, about = "Quantize a checkpoint's linear layers to FP8 E4M3 with block scales")]
struct Args {
    input: PathBuf,
    output: PathBuf,
    /// Quantization block size (rows and columns)
    #[arg(long, default_value_t = 128)]
    block: usize,
}

const COPY_FILES: &[&str] =
    &["tokenizer.json", "tokenizer_config.json", "generation_config.json", "vocab.json", "merges.txt"];

fn shards(dir: &Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(index)?)?;
        let mut files: Vec<String> = v["weight_map"]
            .as_object()
            .ok_or("index has no weight_map")?
            .values()
            .filter_map(|f| f.as_str().map(String::from))
            .collect();
        files.sort();
        files.dedup();
        Ok(files)
    } else {
        Ok(vec!["model.safetensors".into()])
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.output)?;

    let mut weight_map = BTreeMap::new();
    let mut quantized = 0usize;
    for file in shards(&args.input)? {
        let reader = SafeTensorReader::open(args.input.join(&file))?;
        let mut names: Vec<String> = reader.names().map(String::from).collect();
        names.sort();

        // header first: quantized weights gain a scale tensor
        let mut entries = Vec::new();
        for name in &names {
            let info = reader.info(name)?;
            if should_quantize(name, &info.shape) && info.dtype.is_float() && info.dtype != DType::F8E4M3 {
                let (n, k) = (info.shape[0], info.shape[1]);
                entries.push(TensorEntry { name: name.clone(), dtype: DType::F8E4M3, shape: info.shape.clone() });
                entries.push(TensorEntry {
                    name: format!("{name}{SCALE_SUFFIX}"),
                    dtype: DType::F32,
                    shape: vec![n.div_ceil(args.block), k.div_ceil(args.block)],
                });
            } else {
                entries.push(TensorEntry { name: name.clone(), dtype: info.dtype, shape: info.shape.clone() });
            }
        }
        for e in &entries {
            weight_map.insert(e.name.clone(), file.clone());
        }

        let mut writer = StreamWriter::create(args.output.join(&file), entries.clone())?;
        let mut i = 0;
        while i < entries.len() {
            let name = &entries[i].name;
            let next_is_scale = entries.get(i + 1).is_some_and(|e| e.name == format!("{name}{SCALE_SUFFIX}"));
            if entries[i].dtype == DType::F8E4M3 && next_is_scale && reader.info(name)?.dtype != DType::F8E4M3 {
                let (q, scale) = quantize_fp8(&reader.load(name)?, args.block)?;
                writer.write(name, q.bytes())?;
                writer.write(&entries[i + 1].name, scale.bytes())?;
                quantized += 1;
                i += 2;
            } else {
                writer.write(name, reader.bytes(name)?)?;
                i += 1;
            }
        }
        writer.finish()?;
        eprintln!("{file}: done ({quantized} tensors quantized so far)");
    }

    let index = serde_json::json!({ "metadata": { "quantized_by": "garlic quantize" }, "weight_map": weight_map });
    std::fs::write(args.output.join("model.safetensors.index.json"), serde_json::to_string_pretty(&index)?)?;

    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(args.input.join("config.json"))?)?;
    config["quantization_config"] = serde_json::json!({
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [args.block, args.block],
    });
    std::fs::write(args.output.join("config.json"), serde_json::to_string_pretty(&config)?)?;
    for f in COPY_FILES {
        let src = args.input.join(f);
        if src.exists() {
            std::fs::copy(&src, args.output.join(f))?;
        }
    }
    eprintln!("quantized {quantized} tensors -> {}", args.output.display());
    Ok(())
}
