//! Token <-> expert reordering for MoE layers (the C++ `moe_gather` /
//! `moe_scatter`).
//!
//! Routing produces `ids: [T, K]`. Gathering lays the `T*K` (token, expert)
//! pairs out contiguously per expert so each expert sees a dense `[n_e, D]`
//! block; slots are ordered by expert, then by token. Scattering combines the
//! expert outputs back per token with the routing weights.

use rayon::prelude::*;

use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct MoeRouting {
    pub num_tokens: usize,
    pub top_k: usize,
    /// `[E + 1]`: expert `e` owns slots `offsets[e]..offsets[e + 1]`
    pub expert_offsets: Vec<usize>,
    /// `[T * K]`: slot holding (token t, choice k)
    pub slot_of: Vec<usize>,
    /// `[T * K]`: token that slot s came from
    pub token_of_slot: Vec<usize>,
}

impl MoeRouting {
    pub fn new(ids: &[u32], top_k: usize, num_experts: usize) -> Result<Self> {
        if top_k == 0 || !ids.len().is_multiple_of(top_k) {
            return Err(Error::Shape(format!("{} routing ids for top-{top_k}", ids.len())));
        }
        if let Some(&e) = ids.iter().find(|&&e| e as usize >= num_experts) {
            return Err(Error::Shape(format!("expert id {e} >= {num_experts}")));
        }
        let mut counts = vec![0usize; num_experts];
        for &e in ids {
            counts[e as usize] += 1;
        }
        let mut expert_offsets = vec![0usize; num_experts + 1];
        for e in 0..num_experts {
            expert_offsets[e + 1] = expert_offsets[e] + counts[e];
        }
        let mut next = expert_offsets.clone();
        let mut slot_of = vec![0usize; ids.len()];
        let mut token_of_slot = vec![0usize; ids.len()];
        for (pair, &e) in ids.iter().enumerate() {
            let slot = next[e as usize];
            next[e as usize] += 1;
            slot_of[pair] = slot;
            token_of_slot[slot] = pair / top_k;
        }
        Ok(MoeRouting { num_tokens: ids.len() / top_k, top_k, expert_offsets, slot_of, token_of_slot })
    }

    pub fn num_experts(&self) -> usize {
        self.expert_offsets.len() - 1
    }

    pub fn tokens_for(&self, e: usize) -> usize {
        self.expert_offsets[e + 1] - self.expert_offsets[e]
    }

    /// Experts that received at least one token.
    pub fn active_experts(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.num_experts()).filter(|&e| self.tokens_for(e) > 0)
    }

    /// `x: [T, D]` -> `[T*K, D]` in slot order.
    pub fn gather(&self, x: &[f32], dim: usize) -> Vec<f32> {
        let mut out = vec![0f32; self.slot_of.len() * dim];
        out.par_chunks_mut(dim).enumerate().for_each(|(slot, row)| {
            let t = self.token_of_slot[slot];
            row.copy_from_slice(&x[t * dim..(t + 1) * dim]);
        });
        out
    }

    /// `out[t] = sum_k weights[t, k] * expert_out[slot(t, k)]`.
    pub fn scatter(&self, expert_out: &[f32], weights: &[f32], dim: usize) -> Vec<f32> {
        let k = self.top_k;
        let mut out = vec![0f32; self.num_tokens * dim];
        out.par_chunks_mut(dim).enumerate().for_each(|(t, row)| {
            for j in 0..k {
                let w = weights[t * k + j];
                let src = &expert_out[self.slot_of[t * k + j] * dim..][..dim];
                for (o, v) in row.iter_mut().zip(src) {
                    *o += w * v;
                }
            }
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_groups_by_expert_then_token() {
        // 3 tokens, top-2, 4 experts
        let ids = [2, 0, 0, 3, 2, 1];
        let r = MoeRouting::new(&ids, 2, 4).unwrap();
        assert_eq!(r.expert_offsets, vec![0, 2, 3, 5, 6]);
        assert_eq!(r.token_of_slot, vec![0, 1, 2, 0, 2, 1]);
        assert_eq!(r.active_experts().collect::<Vec<_>>(), vec![0, 1, 2, 3]);

        let x = [10.0, 20.0, 30.0]; // dim 1
        let g = r.gather(&x, 1);
        assert_eq!(g, vec![10.0, 20.0, 30.0, 10.0, 30.0, 20.0]);

        // identity experts: scatter returns the weighted sum of copies
        let w = [0.75, 0.25, 0.5, 0.5, 1.0, 0.0];
        let y = r.scatter(&g, &w, 1);
        assert_eq!(y, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn rejects_bad_ids() {
        assert!(MoeRouting::new(&[0, 5], 2, 4).is_err());
        assert!(MoeRouting::new(&[0, 1, 2], 2, 4).is_err());
    }
}
