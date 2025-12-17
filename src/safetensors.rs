//! safetensors reading (memory-mapped, zero-copy) and a minimal writer.
//!
//! File layout: `u64 little-endian header length`, a JSON header mapping
//! tensor names to `{dtype, shape, data_offsets: [start, end]}`, then the raw
//! data. Sharded checkpoints add `model.safetensors.index.json` with a
//! `weight_map` from tensor name to shard file.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;
use serde::Deserialize;

use crate::dtype::{self, DType};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Suffix the FP8 checkpoints use for per-block inverse scales.
pub const SCALE_SUFFIX: &str = "_scale_inv";

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Absolute byte range inside the file.
    pub start: usize,
    pub end: usize,
}

#[derive(Deserialize)]
struct RawInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

pub struct SafeTensorReader {
    path: PathBuf,
    map: Arc<Mmap>,
    tensors: HashMap<String, TensorInfo>,
}

impl SafeTensorReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|e| Error::io(&path, e))?;
        // SAFETY: the file is opened read-only; checkpoints are not expected
        // to be modified while the engine runs.
        let map = unsafe { Mmap::map(&file) }.map_err(|e| Error::io(&path, e))?;
        if map.len() < 8 {
            return Err(Error::Safetensors(format!("{} is too small", path.display())));
        }
        let hlen = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
        if 8 + hlen > map.len() {
            return Err(Error::Safetensors(format!("{}: header past end of file", path.display())));
        }
        let header: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(&map[8..8 + hlen]).map_err(|e| Error::json(&path, e))?;
        let base = 8 + hlen;
        let mut tensors = HashMap::with_capacity(header.len());
        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            let raw: RawInfo = serde_json::from_value(value).map_err(|e| Error::json(&path, e))?;
            let dtype = DType::from_safetensors(&raw.dtype)?;
            let info = TensorInfo {
                dtype,
                start: base + raw.data_offsets[0],
                end: base + raw.data_offsets[1],
                shape: raw.shape,
            };
            let expected = info.shape.iter().product::<usize>() * dtype.size();
            if info.end - info.start != expected || info.end > map.len() {
                return Err(Error::Safetensors(format!(
                    "{}: bad data offsets for {name}",
                    path.display()
                )));
            }
            tensors.insert(name, info);
        }
        Ok(SafeTensorReader { path, map: Arc::new(map), tensors })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::TensorNotFound(format!("{name} in {}", self.path.display())))
    }

    /// Raw bytes of a tensor, straight out of the mapping.
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self.info(name)?;
        Ok(&self.map[info.start..info.end])
    }

    /// Load a tensor without copying. If it is FP8 and a `{name}_scale_inv`
    /// tensor exists, it is attached as the block scale (converted to F32).
    pub fn load(&self, name: &str) -> Result<Tensor> {
        let info = self.info(name)?;
        let mut t = Tensor::from_mmap(self.map.clone(), info.start, &info.shape, info.dtype)?;
        if info.dtype == DType::F8E4M3
            && let Some(scale) = self.load_scale(name)?
        {
            t.set_scale(scale)?;
        }
        Ok(t)
    }

    /// The F32 block scale for `name`, if the checkpoint has one.
    pub fn load_scale(&self, name: &str) -> Result<Option<Tensor>> {
        let scale_name = format!("{name}{SCALE_SUFFIX}");
        let Some(info) = self.tensors.get(&scale_name) else {
            return Ok(None);
        };
        let raw = &self.map[info.start..info.end];
        let vals: Vec<f32> = match info.dtype {
            DType::F32 => raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            DType::BF16 => raw
                .chunks_exact(2)
                .map(|c| dtype::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            DType::F16 => raw
                .chunks_exact(2)
                .map(|c| dtype::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            other => return Err(Error::Unsupported(format!("scale dtype {other} for {name}"))),
        };
        // Some checkpoints store a per-tensor scalar scale; normalise to 2-D.
        let shape = match info.shape.len() {
            0 => vec![1, 1],
            1 => vec![1, info.shape[0]],
            2 => info.shape.clone(),
            _ => return Err(Error::Shape(format!("scale for {name} has shape {:?}", info.shape))),
        };
        Ok(Some(Tensor::from_f32(&vals, &shape)?))
    }
}

/// Resolves tensor names across one or many safetensors shards.
pub struct ModelLoader {
    dir: PathBuf,
    weight_to_file: HashMap<String, String>,
    shards: HashMap<String, SafeTensorReader>,
}

#[derive(Deserialize)]
struct ShardIndex {
    weight_map: HashMap<String, String>,
}

impl ModelLoader {
    pub fn new(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let index_path = dir.join("model.safetensors.index.json");
        let mut loader = ModelLoader { dir: dir.clone(), weight_to_file: HashMap::new(), shards: HashMap::new() };
        if index_path.exists() {
            let text = std::fs::read_to_string(&index_path).map_err(|e| Error::io(&index_path, e))?;
            let index: ShardIndex = serde_json::from_str(&text).map_err(|e| Error::json(&index_path, e))?;
            loader.weight_to_file = index.weight_map;
        } else {
            let file = "model.safetensors".to_string();
            let reader = SafeTensorReader::open(dir.join(&file))?;
            for name in reader.names() {
                loader.weight_to_file.insert(name.to_string(), file.clone());
            }
            loader.shards.insert(file, reader);
        }
        Ok(loader)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn contains(&self, name: &str) -> bool {
        self.weight_to_file.contains_key(name)
    }

    fn shard(&mut self, name: &str) -> Result<&SafeTensorReader> {
        let file = self
            .weight_to_file
            .get(name)
            .ok_or_else(|| Error::TensorNotFound(name.to_string()))?
            .clone();
        if !self.shards.contains_key(&file) {
            let reader = SafeTensorReader::open(self.dir.join(&file))?;
            self.shards.insert(file.clone(), reader);
        }
        Ok(&self.shards[&file])
    }

    pub fn info(&mut self, name: &str) -> Result<TensorInfo> {
        Ok(self.shard(name)?.info(name)?.clone())
    }

    pub fn load(&mut self, name: &str) -> Result<Tensor> {
        self.shard(name)?.load(name)
    }

    /// Load a tensor and convert it to an owned tensor of `dtype`.
    pub fn load_as(&mut self, name: &str, dtype: DType) -> Result<Tensor> {
        let t = self.load(name)?;
        if t.dtype() == dtype { Ok(t) } else { t.to_dtype(dtype) }
    }
}

/// Write tensors to a single safetensors file (used for test fixtures and the
/// quantization tool). Tensors are stored in name order.
pub fn save(path: impl AsRef<Path>, tensors: &[(String, Tensor)]) -> Result<()> {
    let path = path.as_ref();
    let mut sorted: Vec<&(String, Tensor)> = tensors.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));

    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, t) in &sorted {
        let len = t.size_in_bytes();
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": t.dtype().safetensors_name(),
                "shape": t.shape(),
                "data_offsets": [offset, offset + len],
            }),
        );
        offset += len;
    }
    let mut header_bytes = serde_json::to_vec(&serde_json::Value::Object(header))
        .map_err(|e| Error::json(path, e))?;
    // keep the data section 8-byte aligned
    while (8 + header_bytes.len()) % 8 != 0 {
        header_bytes.push(b' ');
    }

    let mut f = std::io::BufWriter::new(File::create(path).map_err(|e| Error::io(path, e))?);
    let io = |e| Error::io(path, e);
    f.write_all(&(header_bytes.len() as u64).to_le_bytes()).map_err(io)?;
    f.write_all(&header_bytes).map_err(io)?;
    for (_, t) in &sorted {
        f.write_all(t.bytes()).map_err(io)?;
    }
    f.flush().map_err(io)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("garlic-st-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn roundtrip_single_file() {
        let dir = tmpdir("single");
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = a.to_dtype(DType::BF16).unwrap();
        save(dir.join("model.safetensors"), &[("a".into(), a), ("b.weight".into(), b)]).unwrap();

        let mut loader = ModelLoader::new(&dir).unwrap();
        let a2 = loader.load("a").unwrap();
        assert_eq!(a2.shape(), &[2, 3]);
        assert_eq!(a2.as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b2 = loader.load_as("b.weight", DType::F32).unwrap();
        assert_eq!(b2.as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(matches!(loader.load("missing"), Err(Error::TensorNotFound(_))));
    }

    #[test]
    fn fp8_scale_is_attached() {
        let dir = tmpdir("fp8");
        let w = Tensor::from_f32(&[0.5; 8], &[2, 4]).unwrap().to_dtype(DType::F8E4M3).unwrap();
        let s = Tensor::from_f32(&[2.0], &[1, 1]).unwrap().to_dtype(DType::BF16).unwrap();
        save(dir.join("model.safetensors"), &[("w.weight".into(), w), ("w.weight_scale_inv".into(), s)])
            .unwrap();
        let mut loader = ModelLoader::new(&dir).unwrap();
        let w2 = loader.load("w.weight").unwrap();
        assert_eq!(w2.dtype(), DType::F8E4M3);
        let scale = w2.scale().expect("scale attached");
        assert_eq!(scale.shape(), &[1, 1]);
        assert_eq!(scale.as_slice::<f32>(), &[2.0]);
    }

    #[test]
    fn sharded_index() {
        let dir = tmpdir("sharded");
        let a = Tensor::from_f32(&[1.0, 2.0], &[2]).unwrap();
        let b = Tensor::from_f32(&[3.0], &[1]).unwrap();
        save(dir.join("part-1.safetensors"), &[("a".into(), a)]).unwrap();
        save(dir.join("part-2.safetensors"), &[("b".into(), b)]).unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"metadata": {}, "weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}}"#,
        )
        .unwrap();
        let mut loader = ModelLoader::new(&dir).unwrap();
        assert_eq!(loader.load("b").unwrap().as_slice::<f32>(), &[3.0]);
        assert_eq!(loader.load("a").unwrap().as_slice::<f32>(), &[1.0, 2.0]);
        assert_eq!(loader.info("a").unwrap().shape, vec![2]);
    }
}
