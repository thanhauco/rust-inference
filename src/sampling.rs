//! Next-token selection.
//!
//! Greedy (argmax) by default, matching the C++ runner. With a positive
//! temperature the logits are optionally penalised for repetition, filtered by
//! top-k and nucleus (top-p), and sampled with a small seeded PRNG so runs are
//! reproducible.

#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// 0 means greedy.
    pub temperature: f32,
    /// 0 disables top-k filtering.
    pub top_k: usize,
    /// 1.0 disables nucleus filtering.
    pub top_p: f32,
    /// 1.0 disables the penalty (CTRL-style: positive logits are divided,
    /// negative ones multiplied).
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams { temperature: 0.0, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, seed: 0x5EED }
    }
}

/// xorshift64* — plenty for sampling, no dependency.
#[derive(Debug, Clone)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // splitmix64 the seed so small seeds still give well-mixed state
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Rng((z ^ (z >> 31)) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub struct Sampler {
    params: SamplingParams,
    rng: Rng,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        let rng = Rng::new(params.seed);
        Sampler { params, rng }
    }

    pub fn is_greedy(&self) -> bool {
        self.params.temperature <= 0.0
    }

    /// Pick the next token from one row of logits. `history` is the prompt
    /// plus everything generated so far (used by the repetition penalty).
    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> u32 {
        let p = &self.params;
        let mut logits = logits.to_vec();
        if p.repetition_penalty != 1.0 {
            let mut seen = vec![false; logits.len()];
            for &t in history {
                if let Some(s) = seen.get_mut(t as usize) {
                    *s = true;
                }
            }
            for (l, s) in logits.iter_mut().zip(seen) {
                if s {
                    *l = if *l > 0.0 { *l / p.repetition_penalty } else { *l * p.repetition_penalty };
                }
            }
        }
        if self.is_greedy() {
            return argmax(&logits);
        }

        let mut cand: Vec<(u32, f32)> =
            logits.iter().enumerate().map(|(i, &l)| (i as u32, l / p.temperature)).collect();
        if p.top_k > 0 && p.top_k < cand.len() {
            cand.select_nth_unstable_by(p.top_k - 1, |a, b| b.1.total_cmp(&a.1));
            cand.truncate(p.top_k);
        }
        cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

        let max = cand[0].1;
        let mut probs: Vec<f32> = cand.iter().map(|c| (c.1 - max).exp()).collect();
        let total: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|v| *v /= total);
        if p.top_p < 1.0 {
            let mut acc = 0.0;
            let keep = probs
                .iter()
                .position(|&q| {
                    acc += q;
                    acc >= p.top_p
                })
                .map_or(probs.len(), |i| i + 1);
            probs.truncate(keep);
            let total: f32 = probs.iter().sum();
            probs.iter_mut().for_each(|v| *v /= total);
        }

        let r = self.rng.next_f32();
        let mut acc = 0.0;
        for (c, q) in cand.iter().zip(&probs) {
            acc += q;
            if r < acc {
                return c.0;
            }
        }
        cand[probs.len() - 1].0
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_by_default() {
        let mut s = Sampler::new(SamplingParams::default());
        assert_eq!(s.sample(&[0.1, 3.0, 2.0], &[]), 1);
    }

    #[test]
    fn repetition_penalty_changes_argmax() {
        let mut s = Sampler::new(SamplingParams { repetition_penalty: 2.0, ..Default::default() });
        assert_eq!(s.sample(&[0.1, 3.0, 2.0], &[1]), 2);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let params = SamplingParams { temperature: 1.0, top_k: 1, ..Default::default() };
        let mut s = Sampler::new(params);
        for _ in 0..20 {
            assert_eq!(s.sample(&[0.5, 0.4, 0.6, 0.1], &[]), 2);
        }
    }

    #[test]
    fn sampling_follows_distribution_and_seed() {
        let logits = [0.0f32, (3.0f32).ln()]; // p = [0.25, 0.75]
        let params = SamplingParams { temperature: 1.0, seed: 7, ..Default::default() };
        let mut s = Sampler::new(params.clone());
        let picks: Vec<u32> = (0..4000).map(|_| s.sample(&logits, &[])).collect();
        let ones = picks.iter().filter(|&&t| t == 1).count() as f32 / picks.len() as f32;
        assert!((ones - 0.75).abs() < 0.03, "{ones}");
        let mut again = Sampler::new(params);
        let picks2: Vec<u32> = (0..4000).map(|_| again.sample(&logits, &[])).collect();
        assert_eq!(picks, picks2);
    }

    #[test]
    fn top_p_drops_the_tail() {
        let params = SamplingParams { temperature: 1.0, top_p: 0.5, ..Default::default() };
        let mut s = Sampler::new(params);
        // first token alone has ~0.84 of the mass
        for _ in 0..50 {
            assert_eq!(s.sample(&[3.0, 1.0, 0.5, 0.0], &[]), 0);
        }
    }
}
