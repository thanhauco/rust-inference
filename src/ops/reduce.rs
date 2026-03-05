use rayon::prelude::*;

use super::expect_f32;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Index of the maximum along the last dimension, for every row. Ties go to
/// the first occurrence (like `torch.argmax`).
pub fn argmax_last(x: &Tensor) -> Result<Vec<u32>> {
    #[cfg(feature = "cuda")]
    if x.device() == crate::tensor::Device::Cuda {
        return crate::cuda::ops::argmax_last(x);
    }
    expect_f32(x, "argmax_last")?;
    let n = *x.shape().last().unwrap();
    Ok(x.as_slice::<f32>()
        .par_chunks(n)
        .map(|row| {
            let mut best = 0usize;
            for (i, &v) in row.iter().enumerate() {
                if v > row[best] {
                    best = i;
                }
            }
            best as u32
        })
        .collect())
}

/// MoE router: pick the top-`k` experts of each row of `logits: [T, E]`.
///
/// Returns `(ids, weights)`, both `[T * k]`, ordered by descending score.
/// With `normalize` (Qwen3's `norm_topk_prob`) the weights are a softmax over
/// the selected logits only, which equals renormalising the full softmax;
/// without it they are the full-softmax probabilities.
pub fn topk_softmax(logits: &Tensor, k: usize, normalize: bool) -> Result<(Vec<u32>, Vec<f32>)> {
    #[cfg(feature = "cuda")]
    if logits.device() == crate::tensor::Device::Cuda {
        return crate::cuda::ops::topk_softmax(logits, k, normalize);
    }
    expect_f32(logits, "topk_softmax")?;
    let e = *logits.shape().last().unwrap();
    if k == 0 || k > e {
        return Err(Error::Shape(format!("top-{k} of {e} experts")));
    }
    let rows: Vec<(Vec<u32>, Vec<f32>)> = logits
        .as_slice::<f32>()
        .par_chunks(e)
        .map(|row| {
            let mut idx: Vec<u32> = (0..e as u32).collect();
            // stable: higher score first, lower index on ties
            idx.sort_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]).then(a.cmp(&b)));
            idx.truncate(k);
            let max = row[idx[0] as usize];
            let mut w: Vec<f32> = idx.iter().map(|&i| (row[i as usize] - max).exp()).collect();
            let denom: f32 = if normalize { w.iter().sum() } else { row.iter().map(|&v| (v - max).exp()).sum() };
            w.iter_mut().for_each(|v| *v /= denom);
            (idx, w)
        })
        .collect();
    let mut ids = Vec::with_capacity(rows.len() * k);
    let mut weights = Vec::with_capacity(rows.len() * k);
    for (i, w) in rows {
        ids.extend(i);
        weights.extend(w);
    }
    Ok((ids, weights))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_rows() {
        let x = Tensor::from_f32(&[1.0, 3.0, 3.0, -1.0, 0.0, -5.0], &[2, 3]).unwrap();
        assert_eq!(argmax_last(&x).unwrap(), vec![1, 1]);
    }

    #[test]
    fn topk_matches_reference() {
        // same input as the C++ top_k_test / top_k_python.py
        let base = [0.23f32, 0.1, 0.109, 0.0003, -0.4, 0.25, 0.9, 9.7, 0.109, 0.9];
        let row: Vec<f32> = base.iter().cycle().take(40).copied().collect();
        let x = Tensor::from_f32(&row, &[1, 40]).unwrap();
        let (ids, w) = topk_softmax(&x, 4, true).unwrap();
        assert_eq!(ids, vec![7, 17, 27, 37]);
        for v in &w {
            assert!((v - 0.25).abs() < 1e-6);
        }
        let (_, w_full) = topk_softmax(&x, 4, false).unwrap();
        let total: f32 = row.iter().map(|v| (v - 9.7).exp()).sum();
        assert!((w_full[0] - 1.0 / total).abs() < 1e-6);
    }
}
