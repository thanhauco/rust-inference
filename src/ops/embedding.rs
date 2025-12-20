use rayon::prelude::*;

use crate::cpu::simd;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::{Device, Tensor};

/// Look up rows of `table: [vocab, dim]` for `ids` (U32, any shape).
/// Returns `[*ids.shape, dim]` in F32.
pub fn embedding(ids: &Tensor, table: &Tensor) -> Result<Tensor> {
    if ids.dtype() != DType::U32 {
        return Err(Error::Unsupported(format!("embedding ids must be U32, got {}", ids.dtype())));
    }
    let (vocab, dim) = (table.dim(0), table.dim(1));
    let ids_v = ids.as_slice::<u32>();
    if let Some(&bad) = ids_v.iter().find(|&&i| i as usize >= vocab) {
        return Err(Error::Shape(format!("token id {bad} out of range for vocab {vocab}")));
    }
    let mut shape = ids.shape().to_vec();
    shape.push(dim);
    let mut out = Tensor::zeros(&shape, DType::F32, Device::Cpu);
    let dst = out.as_mut_slice::<f32>()?;
    dst.par_chunks_mut(dim).zip(ids_v.par_iter()).for_each(|(row, &id)| {
        let r = id as usize * dim..(id as usize + 1) * dim;
        match table.dtype() {
            DType::F32 => row.copy_from_slice(&table.as_slice::<f32>()[r]),
            DType::BF16 => simd::decode_bf16(&table.as_slice::<u16>()[r], row),
            DType::F16 => simd::decode_f16(&table.as_slice::<u16>()[r], row),
            DType::F8E4M3 => {
                // per-tensor scale only, as in the C++ kernel
                let scale = table.scale().map_or(1.0, |s| s.as_slice::<f32>()[0]);
                simd::decode_fp8(&table.as_slice::<u8>()[r], row, scale)
            }
            _ => unreachable!(),
        }
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gathers_rows() {
        let table = Tensor::from_f32(&[0., 1., 10., 11., 20., 21.], &[3, 2])
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let ids = Tensor::from_u32(&[2, 0, 2], &[1, 3]).unwrap();
        let out = embedding(&ids, &table).unwrap();
        assert_eq!(out.shape(), &[1, 3, 2]);
        assert_eq!(out.as_slice::<f32>(), &[20., 21., 0., 1., 20., 21.]);
        let bad = Tensor::from_u32(&[3], &[1]).unwrap();
        assert!(embedding(&bad, &table).is_err());
    }
}
