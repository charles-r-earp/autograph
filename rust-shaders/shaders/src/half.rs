use spirv_std::{
    //glam::{Vec2, Vec4, Vec4Swizzles},
    memory::{Scope, Semantics},
};
use crate::{util::{Load, Store}, atomic::atomic_compare_exchange};
use core::{mem::size_of, /*sync::atomic::{AtomicU32, Ordering}*/};

/* Original: https://github.com/starkat99/half-rs/releases/tag/v1.7.1/src/bfloat/conver.rs
pub(crate) fn f32_to_bf16(value: f32) -> u16 {
    // Convert to raw bytes
    let x = value.to_bits();

    // check for NaN
    if x & 0x7FFF_FFFFu32 > 0x7F80_0000u32 {
        // Keep high part of current mantissa but also set most significiant mantissa bit
        return ((x >> 16) | 0x0040u32) as u16;
    }

    // round and shift
    let round_bit = 0x0000_8000u32;
    if (x & round_bit) != 0 && (x & (3 * round_bit - 1)) != 0 {
        (x >> 16) as u16 + 1
    } else {
        (x >> 16) as u16
    }
}
*/
fn f32_to_bf16(value: f32) -> u32 {
    // Convert to raw bytes
    let x = value.to_bits();
    // check for NaN
    if x & 0x7FFF_FFFFu32 > 0x7F80_0000u32 {
        // Keep high part of current mantissa but also set most significiant mantissa bit
        return (x >> 16) | 0x0040u32;
    }

    // round and shift
    let round_bit = 0x0000_8000u32;
    if (x & round_bit) != 0 /*&& (x & (3 * round_bit - 1)) != 0*/ {
        (x >> 16) + 1
    } else {
        x >> 16
    }
}

/* Original: https://github.com/starkat99/half-rs/releases/tag/v1.7.1/src/bfloat/conver.rs
pub(crate) fn bf16_to_f32(i: u16) -> f32 {
    // If NaN, keep current mantissa but also set most significiant mantissa bit
    if i & 0x7FFFu16 > 0x7F80u16 {
        f32::from_bits((i as u32 | 0x0040u32) << 16)
    } else {
        f32::from_bits((i as u32) << 16)
    }
}
*/

pub(crate) fn bf16_to_f32(i: u32) -> f32 {
    // If NaN, keep current mantissa but also set most significiant mantissa bit
    if i & 0x7FFFu32 > 0x7F80u32 {
        f32::from_bits((i | 0x0040u32) << 16)
    } else {
        f32::from_bits(i << 16)
    }
}

/*fn bf16x2_to_vec2(x: u32) -> Vec2 {
    Vec2::new(bf16_to_f32(x), bf16_to_f32(x >> 16))
}*/

/*
pub(crate) fn vec2_to_bf16x2(x: Vec2) -> u32 {
    f32_to_bf16(x.x) | f32_to_bf16(x.y) << 16
}

pub(crate) fn vec4_to_bf16x4(x: Vec4) -> (u32, u32) {
    (vec2_to_bf16x2(x.xy()), vec2_to_bf16x2(x.zw()))
}*/

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct bf16x2(u32);

impl bf16x2 {
    fn as_bits_mut(&mut self) -> &mut u32 { &mut self.0 }
    fn to_bits(self) -> u32 { self.0 }
    fn from_bits(v: u32) -> Self { Self(v) }
    pub(crate) fn from_f32x2(x: [f32; 2]) -> Self {
        Self::from_bits(f32_to_bf16(x[0]) | f32_to_bf16(x[1]) << 16)
    }
    pub(crate) fn to_f32x2(self) -> [f32; 2] {
        let bits = self.to_bits();
        [bf16_to_f32(bits), bf16_to_f32(bits >> 16)]
    }
}

impl Load<f32> for [bf16x2] {
    fn load(&self, index: usize) -> f32 {
        let bits = self[index / size_of::<u16>()].0;
        if index & 1 == 0 {
            f32::from_bits(bits << 16)
        } else {
            f32::from_bits(bits >> 16 << 16)
        }
    }
}

impl Store<f32> for [bf16x2] {
    fn store(&mut self, index: usize, value: f32) {
        let mut previous: u32;
        let idx = index / 2;
        loop {
            previous = self[idx].to_bits();
            let mut values = self[idx].to_f32x2();
            values[index & 1] = value;
            let value = bf16x2::from_f32x2(values).to_bits();
            if unsafe {
                atomic_compare_exchange::<u32, {Scope::Device as u32}, {Semantics::NONE.bits()}, {Semantics::NONE.bits()}>(self[idx].as_bits_mut(), value, previous)
            } == previous {
                break;
            }
        }
    }
}
