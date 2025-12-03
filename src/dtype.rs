//! Element types and the scalar conversions between them.
//!
//! Weights come out of safetensors as F32 / BF16 / F16 / F8_E4M3; activations
//! on the CPU path are always F32. The FP8 helpers here follow the OCP
//! "E4M3FN" encoding used by the Qwen3 FP8 checkpoints (no infinities, a
//! single NaN mantissa pattern, max finite value 448).

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    F8E4M3,
    U32,
    I32,
    I64,
    U8,
}

impl DType {
    pub const fn size(self) -> usize {
        match self {
            DType::F8E4M3 | DType::U8 => 1,
            DType::F16 | DType::BF16 => 2,
            DType::F32 | DType::U32 | DType::I32 => 4,
            DType::I64 => 8,
        }
    }

    pub fn from_safetensors(s: &str) -> Result<Self> {
        Ok(match s {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "F8_E4M3" => DType::F8E4M3,
            "U32" => DType::U32,
            "I32" => DType::I32,
            "I64" => DType::I64,
            "U8" => DType::U8,
            other => return Err(Error::Unsupported(format!("safetensors dtype {other}"))),
        })
    }

    pub const fn safetensors_name(self) -> &'static str {
        match self {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::F8E4M3 => "F8_E4M3",
            DType::U32 => "U32",
            DType::I32 => "I32",
            DType::I64 => "I64",
            DType::U8 => "U8",
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F16 | DType::BF16 | DType::F8E4M3)
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.safetensors_name())
    }
}

// ---------------------------------------------------------------------------
// bf16 / f16
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Round-to-nearest-even, NaNs stay (quiet) NaNs. Bit-identical to
/// `__float2bfloat16_rn`.
#[inline(always)]
pub fn f32_to_bf16(f: f32) -> u16 {
    let u = f.to_bits();
    if (u & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((u >> 16) | 0x0040) as u16;
    }
    let rounded = u.wrapping_add(0x7FFF + ((u >> 16) & 1));
    (rounded >> 16) as u16
}

/// Round an f32 through bf16 and back.
#[inline(always)]
pub fn round_bf16(f: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(f))
}

#[inline(always)]
pub fn f16_to_f32(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}

#[inline(always)]
pub fn f32_to_f16(f: f32) -> u16 {
    half::f16::from_f32(f).to_bits()
}

// ---------------------------------------------------------------------------
// FP8 E4M3FN
// ---------------------------------------------------------------------------

pub const FP8_E4M3_MAX: f32 = 448.0;

const fn fp8_e4m3_decode(b: u8) -> f32 {
    let sign = (b >> 7) & 1;
    let e = ((b >> 3) & 0xF) as u32;
    let m = (b & 0x7) as u32;
    let v = if e == 0 {
        // subnormal: m * 2^-3 * 2^-6
        (m as f32) * (1.0 / 512.0)
    } else if e == 15 && m == 7 {
        f32::NAN
    } else {
        // (1 + m/8) * 2^(e-7): place directly into the f32 bit pattern.
        f32::from_bits(((e + 120) << 23) | (m << 20))
    };
    if sign == 1 { -v } else { v }
}

const fn build_fp8_lut() -> [f32; 256] {
    let mut t = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = fp8_e4m3_decode(i as u8);
        i += 1;
    }
    t
}

/// Every E4M3 byte decoded to f32.
pub static FP8_E4M3_LUT: [f32; 256] = build_fp8_lut();

#[inline(always)]
pub fn fp8_e4m3_to_f32(b: u8) -> f32 {
    FP8_E4M3_LUT[b as usize]
}

/// f32 -> E4M3FN with round-to-nearest-even. Values beyond the finite range
/// saturate to +-448 (the "satfinite" behaviour quantizers expect); NaN maps
/// to the canonical NaN byte.
pub fn f32_to_fp8_e4m3(x: f32) -> u8 {
    let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0 };
    let a = x.abs();
    if a.is_nan() {
        return sign | 0x7F;
    }
    if a >= FP8_E4M3_MAX {
        return sign | 0x7E;
    }
    // smallest normal is 2^-6
    if a < 1.0 / 64.0 {
        let m = (a * 512.0).round_ties_even() as u8; // units of 2^-9, 0..=8
        // m == 8 rolls over into the smallest normal (0x08), same bit pattern
        return sign | m;
    }
    let bits = a.to_bits();
    let mut exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let mant23 = bits & 0x7F_FFFF;
    let mut mant3 = mant23 >> 20;
    let rem = mant23 & 0xF_FFFF;
    const HALF: u32 = 0x8_0000;
    if rem > HALF || (rem == HALF && (mant3 & 1) == 1) {
        mant3 += 1;
    }
    if mant3 == 8 {
        mant3 = 0;
        exp += 1;
    }
    let biased = exp + 7;
    if biased > 15 || (biased == 15 && mant3 == 7) {
        return sign | 0x7E;
    }
    sign | ((biased as u8) << 3) | (mant3 as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_known_values() {
        assert_eq!(fp8_e4m3_to_f32(0x00), 0.0);
        assert_eq!(fp8_e4m3_to_f32(0x38), 1.0);
        assert_eq!(fp8_e4m3_to_f32(0xB8), -1.0);
        assert_eq!(fp8_e4m3_to_f32(0x7E), 448.0);
        assert_eq!(fp8_e4m3_to_f32(0x01), 2f32.powi(-9));
        assert_eq!(fp8_e4m3_to_f32(0x08), 2f32.powi(-6));
        assert!(fp8_e4m3_to_f32(0x7F).is_nan());
        assert!(fp8_e4m3_to_f32(0xFF).is_nan());
    }

    #[test]
    fn fp8_roundtrip_all_finite() {
        for b in 0u8..=255 {
            let v = fp8_e4m3_to_f32(b);
            if v.is_nan() {
                continue;
            }
            let back = f32_to_fp8_e4m3(v);
            // +0 / -0 both decode to zero; everything else must be exact
            if v == 0.0 {
                assert_eq!(back & 0x7F, 0);
            } else {
                assert_eq!(back, b, "byte {b:#04x} value {v}");
            }
        }
    }

    #[test]
    fn fp8_rounding_and_saturation() {
        assert_eq!(f32_to_fp8_e4m3(1000.0), 0x7E);
        assert_eq!(f32_to_fp8_e4m3(-1000.0), 0xFE);
        // 1.0625 is exactly halfway between 1.0 (0x38) and 1.125 (0x39): ties to even
        assert_eq!(f32_to_fp8_e4m3(1.0625), 0x38);
        assert_eq!(f32_to_fp8_e4m3(1.07), 0x39);
        // halfway between the two smallest subnormals rounds to even (0x02)
        assert_eq!(f32_to_fp8_e4m3(1.5 * 2f32.powi(-9)), 0x02);
    }

    #[test]
    fn bf16_rne() {
        assert_eq!(f32_to_bf16(1.0), 0x3F80);
        // 1 + 2^-8 is a tie between 1.0 and 1 + 2^-7 -> even (1.0)
        assert_eq!(f32_to_bf16(1.0 + 2f32.powi(-8)), 0x3F80);
        assert_eq!(f32_to_bf16(1.0 + 3.0 * 2f32.powi(-8)), 0x3F82);
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
        for v in [0.1f32, -3.25, 65504.0, 1e-20] {
            assert_eq!(f32_to_bf16(v), half::bf16::from_f32(v).to_bits());
        }
    }
}
