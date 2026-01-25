//! Runtime-dispatched SIMD helpers for the CPU path.
//!
//! Everything here has a scalar fallback; on x86-64 the AVX2+FMA and
//! AVX-512F versions are selected once at startup. `GARLIC_CPU_ISA` can cap
//! the level (`scalar`, `avx2`, `avx512`) for debugging and benchmarks.

use std::sync::OnceLock;

use crate::dtype;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Isa {
    Scalar,
    Avx2,
    Avx512,
}

impl Isa {
    pub fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            Isa::Avx2 => "avx2",
            Isa::Avx512 => "avx512",
        }
    }
}

static ISA: OnceLock<Isa> = OnceLock::new();

pub fn isa() -> Isa {
    *ISA.get_or_init(detect)
}

fn detect() -> Isa {
    let cap = std::env::var("GARLIC_CPU_ISA").ok();
    let cap = cap.as_deref();
    #[cfg(target_arch = "x86_64")]
    {
        let allow512 = cap.is_none_or(|c| c == "avx512");
        let allow2 = cap.is_none_or(|c| c == "avx512" || c == "avx2");
        let avx2 =
            is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c");
        if allow512 && avx2 && is_x86_feature_detected!("avx512f") {
            return Isa::Avx512;
        }
        if allow2 && avx2 {
            return Isa::Avx2;
        }
    }
    let _ = cap;
    Isa::Scalar
}

// ---------------------------------------------------------------------------
// dot products: one weight row against MR activation rows
// ---------------------------------------------------------------------------

/// `out[t] = sum_i w[i] * xs[t][i]` for `t < MR`. All slices must have the
/// same length.
#[inline]
pub fn dot_mr<const MR: usize>(w: &[f32], xs: &[&[f32]; MR]) -> [f32; MR] {
    for x in xs {
        debug_assert_eq!(x.len(), w.len());
    }
    match isa() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the ISA was detected at runtime.
        Isa::Avx512 => unsafe { x86::dot_mr_avx512(w, xs) },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { x86::dot_mr_avx2(w, xs) },
        _ => dot_mr_scalar(w, xs),
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    dot_mr::<1>(a, &[b])[0]
}

fn dot_mr_scalar<const MR: usize>(w: &[f32], xs: &[&[f32]; MR]) -> [f32; MR] {
    // 8 independent lanes per row keep the dependency chains short and
    // let LLVM vectorise the fallback where it can.
    let mut acc = [[0f32; 8]; MR];
    let chunks = w.len() / 8;
    for c in 0..chunks {
        let base = c * 8;
        for t in 0..MR {
            for l in 0..8 {
                acc[t][l] += w[base + l] * xs[t][base + l];
            }
        }
    }
    let mut out = [0f32; MR];
    for t in 0..MR {
        out[t] = acc[t].iter().sum();
        for i in chunks * 8..w.len() {
            out[t] += w[i] * xs[t][i];
        }
    }
    out
}

/// `y += a * x`
#[inline]
pub fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    debug_assert_eq!(y.len(), x.len());
    match isa() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the ISA was detected at runtime.
        Isa::Avx512 | Isa::Avx2 => unsafe { x86::axpy_avx2(y, a, x) },
        _ => {
            for (yi, xi) in y.iter_mut().zip(x) {
                *yi += a * xi;
            }
        }
    }
}

/// `y *= a`
#[inline]
pub fn scale(y: &mut [f32], a: f32) {
    for v in y {
        *v *= a;
    }
}

// ---------------------------------------------------------------------------
// decoding weight rows to f32
// ---------------------------------------------------------------------------

pub fn decode_bf16(src: &[u16], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    match isa() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the ISA was detected at runtime.
        Isa::Avx512 | Isa::Avx2 => unsafe { x86::decode_bf16_avx2(src, dst) },
        _ => {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = dtype::bf16_to_f32(s);
            }
        }
    }
}

pub fn decode_f16(src: &[u16], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    match isa() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the ISA was detected at runtime.
        Isa::Avx512 | Isa::Avx2 => unsafe { x86::decode_f16_avx2(src, dst) },
        _ => {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = dtype::f16_to_f32(s);
            }
        }
    }
}

/// Decode E4M3 bytes and multiply by `scale`.
pub fn decode_fp8(src: &[u8], dst: &mut [f32], scale: f32) {
    debug_assert_eq!(src.len(), dst.len());
    match isa() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the ISA was detected at runtime.
        Isa::Avx512 | Isa::Avx2 => unsafe { x86::decode_fp8_avx2(src, dst, scale) },
        _ => {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = dtype::fp8_e4m3_to_f32(s) * scale;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86 {
    use std::arch::x86_64::*;

    use crate::dtype;

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    pub unsafe fn hsum256(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
        _mm_cvtss_f32(s)
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_mr_avx2<const MR: usize>(w: &[f32], xs: &[&[f32]; MR]) -> [f32; MR] {
        let n = w.len();
        let wp = w.as_ptr();
        let mut acc0 = [_mm256_setzero_ps(); MR];
        let mut acc1 = [_mm256_setzero_ps(); MR];
        let mut i = 0;
        unsafe {
            while i + 16 <= n {
                let w0 = _mm256_loadu_ps(wp.add(i));
                let w1 = _mm256_loadu_ps(wp.add(i + 8));
                for t in 0..MR {
                    let xp = xs[t].as_ptr();
                    acc0[t] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xp.add(i)), acc0[t]);
                    acc1[t] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xp.add(i + 8)), acc1[t]);
                }
                i += 16;
            }
            while i + 8 <= n {
                let w0 = _mm256_loadu_ps(wp.add(i));
                for t in 0..MR {
                    acc0[t] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xs[t].as_ptr().add(i)), acc0[t]);
                }
                i += 8;
            }
            let mut out = [0f32; MR];
            for t in 0..MR {
                out[t] = hsum256(_mm256_add_ps(acc0[t], acc1[t]));
                for j in i..n {
                    out[t] += w[j] * xs[t][j];
                }
            }
            out
        }
    }

    #[target_feature(enable = "avx512f,avx2,fma")]
    pub unsafe fn dot_mr_avx512<const MR: usize>(w: &[f32], xs: &[&[f32]; MR]) -> [f32; MR] {
        let n = w.len();
        let wp = w.as_ptr();
        let mut acc0 = [_mm512_setzero_ps(); MR];
        let mut acc1 = [_mm512_setzero_ps(); MR];
        let mut i = 0;
        unsafe {
            while i + 32 <= n {
                let w0 = _mm512_loadu_ps(wp.add(i));
                let w1 = _mm512_loadu_ps(wp.add(i + 16));
                for t in 0..MR {
                    let xp = xs[t].as_ptr();
                    acc0[t] = _mm512_fmadd_ps(w0, _mm512_loadu_ps(xp.add(i)), acc0[t]);
                    acc1[t] = _mm512_fmadd_ps(w1, _mm512_loadu_ps(xp.add(i + 16)), acc1[t]);
                }
                i += 32;
            }
            while i + 16 <= n {
                let w0 = _mm512_loadu_ps(wp.add(i));
                for t in 0..MR {
                    acc0[t] = _mm512_fmadd_ps(w0, _mm512_loadu_ps(xs[t].as_ptr().add(i)), acc0[t]);
                }
                i += 16;
            }
            let mut out = [0f32; MR];
            for t in 0..MR {
                out[t] = _mm512_reduce_add_ps(_mm512_add_ps(acc0[t], acc1[t]));
                for j in i..n {
                    out[t] += w[j] * xs[t][j];
                }
            }
            out
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn axpy_avx2(y: &mut [f32], a: f32, x: &[f32]) {
        let n = y.len();
        let va = _mm256_set1_ps(a);
        let mut i = 0;
        unsafe {
            while i + 8 <= n {
                let yp = y.as_mut_ptr().add(i);
                let v = _mm256_fmadd_ps(va, _mm256_loadu_ps(x.as_ptr().add(i)), _mm256_loadu_ps(yp));
                _mm256_storeu_ps(yp, v);
                i += 8;
            }
        }
        for j in i..n {
            y[j] += a * x[j];
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn decode_bf16_avx2(src: &[u16], dst: &mut [f32]) {
        let n = src.len();
        let mut i = 0;
        unsafe {
            while i + 8 <= n {
                let h = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
                let w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_castsi256_ps(w));
                i += 8;
            }
        }
        for j in i..n {
            dst[j] = dtype::bf16_to_f32(src[j]);
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn decode_f16_avx2(src: &[u16], dst: &mut [f32]) {
        let n = src.len();
        let mut i = 0;
        unsafe {
            while i + 8 <= n {
                let h = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_cvtph_ps(h));
                i += 8;
            }
        }
        for j in i..n {
            dst[j] = dtype::f16_to_f32(src[j]);
        }
    }

    /// Branch-free E4M3 -> f32 for 8 bytes.
    ///
    /// For normal values (e != 0), `(1 + m/8) * 2^(e-7)` is an f32 with
    /// biased exponent `e + 120` and mantissa `m << 20`, built with integer
    /// ops only. Subnormal lanes (e == 0) are `m * 2^-9`, exact via
    /// int->float. Nothing here ever does float arithmetic on a denormal: the
    /// classic "place bits and multiply by 2^120" trick does, as soon as the
    /// compiler moves the multiply ahead of the lane masking (LLVM does), and
    /// every denormal lane costs ~100 cycles of microcode. NaN bytes
    /// (0x7F/0xFF) decode to +-480; they never occur in valid checkpoints.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    pub unsafe fn decode8_fp8(p: *const u8) -> __m256 {
        unsafe {
            let w32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(p as *const __m128i));
            let sgn = _mm256_slli_epi32(_mm256_and_si256(w32, _mm256_set1_epi32(0x80)), 24);
            let mag = _mm256_slli_epi32(_mm256_and_si256(w32, _mm256_set1_epi32(0x7F)), 20);
            let norm = _mm256_add_epi32(mag, _mm256_set1_epi32(120 << 23));
            let sub = _mm256_cmpeq_epi32(_mm256_and_si256(w32, _mm256_set1_epi32(0x78)), _mm256_setzero_si256());
            let subv = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_and_si256(w32, _mm256_set1_epi32(0x7))),
                _mm256_set1_ps(1.0 / 512.0),
            );
            let bits = _mm256_blendv_epi8(norm, _mm256_castps_si256(subv), sub);
            _mm256_castsi256_ps(_mm256_or_si256(bits, sgn))
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn decode_fp8_avx2(src: &[u8], dst: &mut [f32], scale: f32) {
        let n = src.len();
        let vs = _mm256_set1_ps(scale);
        let mut i = 0;
        unsafe {
            while i + 8 <= n {
                let v = _mm256_mul_ps(decode8_fp8(src.as_ptr().add(i)), vs);
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), v);
                i += 8;
            }
        }
        for j in i..n {
            dst[j] = dtype::fp8_e4m3_to_f32(src[j]) * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize, k: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * k).sin()).collect()
    }

    #[test]
    fn dot_matches_scalar() {
        for n in [0, 1, 7, 8, 15, 16, 31, 33, 128, 1000] {
            let w = ramp(n, 0.37);
            let x0 = ramp(n, 0.11);
            let x1 = ramp(n, 0.73);
            let got = dot_mr::<2>(&w, &[&x0, &x1]);
            let want = dot_mr_scalar::<2>(&w, &[&x0, &x1]);
            for t in 0..2 {
                assert!((got[t] - want[t]).abs() < 1e-4 * (1.0 + want[t].abs()), "n={n}");
            }
        }
    }

    #[test]
    fn decoders_match_scalar() {
        let vals = ramp(37, 1.3);
        let bf: Vec<u16> = vals.iter().map(|&v| dtype::f32_to_bf16(v * 100.0)).collect();
        let mut out = vec![0.0; bf.len()];
        decode_bf16(&bf, &mut out);
        for (o, &b) in out.iter().zip(&bf) {
            assert_eq!(*o, dtype::bf16_to_f32(b));
        }
        let hf: Vec<u16> = vals.iter().map(|&v| dtype::f32_to_f16(v)).collect();
        decode_f16(&hf, &mut out);
        for (o, &h) in out.iter().zip(&hf) {
            assert_eq!(*o, dtype::f16_to_f32(h));
        }
    }

    #[test]
    fn fp8_decode_all_bytes() {
        // every non-NaN byte, including subnormals and signed zero
        let bytes: Vec<u8> = (0u8..=255).filter(|&b| b & 0x7F != 0x7F).collect();
        let mut out = vec![0.0; bytes.len()];
        decode_fp8(&bytes, &mut out, 0.5);
        for (o, &b) in out.iter().zip(&bytes) {
            assert_eq!(*o, dtype::fp8_e4m3_to_f32(b) * 0.5, "byte {b:#04x}");
        }
    }
}
