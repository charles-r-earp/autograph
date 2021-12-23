//! Adapted from https://github.com/EmbarkStudios/rust-gpu/blob/atomics/crates/spirv-std/src/arch/atomics.rs

use crate::autobind;
use spirv_std::{
    // scalar::Scalar,
    memory::{Scope, Semantics},
    integer::Integer,
    glam::UVec3,
};

/// Atomically load through `ptr` using the given `SEMANTICS`. All subparts of
/// the value that is loaded are read atomically with respect to all other
/// atomic accesses to it within `SCOPE`.
#[cfg(feature = "false")]
#[doc(alias = "OpAtomicLoad")]
#[inline]
pub(crate) unsafe fn atomic_load<T: Scalar, const SCOPE: u32, const SEMANTICS: u32>(ptr: &T) -> T {
    let mut result = T::default();

    asm! {
        "%u32 = OpTypeInt 32 0",
        "%scope = OpConstant %u32 {scope}",
        "%semantics = OpConstant %u32 {semantics}",
        "%result = OpAtomicLoad _ {ptr} %scope %semantics",
        "OpStore {result} %result",
        scope = const SCOPE,
        semantics = const SEMANTICS,
        ptr = in(reg) ptr,
        result = in(reg) &mut result
    }

    result
}


/// Atomically store through `ptr` using the given `SEMANTICS`. All subparts of
/// `value` are written atomically with respect to all other atomic accesses to
/// it within `SCOPE`.
#[cfg(feature = "false")]
#[doc(alias = "OpAtomicStore")]
#[inline]
pub(crate) unsafe fn atomic_store<T: Scalar, const SCOPE: u32, const SEMANTICS: u32>(
    ptr: &mut T,
    value: T,
) {
    asm! {
        "%u32 = OpTypeInt 32 0",
        "%scope = OpConstant %u32 {scope}",
        "%semantics = OpConstant %u32 {semantics}",
        "%value = OpLoad _ {value}",
        "OpAtomicStore {ptr} %scope %semantics %value",
        scope = const SCOPE,
        semantics = const SEMANTICS,
        ptr = in(reg) ptr,
        value = in(reg) &value,
    }
}
/*
/// Perform the following steps atomically with respect to any other atomic
/// accesses within `SCOPE` to the same location:
///
/// 1. Load through `ptr` to get the original value,
/// 2. Get a new value from copying `value`, and
/// 3. Store the new value back through `ptr`.
///
/// The result is the original value.
#[cfg(feature = "false")]
#[doc(alias = "OpAtomicExchange")]
#[inline]
pub(crate) unsafe fn atomic_exchange<T: Scalar, const SCOPE: u32, const SEMANTICS: u32>(
    ptr: &mut T,
    value: T,
) -> T {
    let mut old = T::default();

    asm! {
        "%u32 = OpTypeInt 32 0",
        "%scope = OpConstant %u32 {scope}",
        "%semantics = OpConstant %u32 {semantics}",
        "%value = OpLoad _ {value}",
        "%old = OpAtomicExchange _ {ptr} %scope %semantics %value",
        "OpStore {old} %old",
        scope = const SCOPE,
        semantics = const SEMANTICS,
        ptr = in(reg) ptr,
        value = in(reg) &value,
        old = in(reg) &mut old,
    }

    old
}*/


/// Perform the following steps atomically with respect to any other atomic
/// accesses within `SCOPE` to the same location:
///
/// 1. Load through `ptr` to get the original value
/// 2. Get a new value from `value` only if the original value equals
///    `comparator`, and
/// 3. Store the new value back through `ptr`, only if the original value
///    equaled `comparator`.
///
/// The result is the original value.
//#[cfg(feature = "false")]
#[doc(alias = "OpAtomicCompareExchange")]
#[inline]
pub(crate) unsafe fn atomic_compare_exchange<
    I: Integer,
    const SCOPE: u32,
    const EQUAL: u32,
    const UNEQUAL: u32,
>(
    ptr: &mut I,
    value: I,
    comparator: I,
) -> I {
    let mut old = I::default();

    asm! {
        "%u32 = OpTypeInt 32 0",
        "%scope = OpConstant %u32 {scope}",
        "%equal = OpConstant %u32 {equal}",
        "%unequal = OpConstant %u32 {unequal}",
        "%value = OpLoad _ {value}",
        "%comparator = OpLoad _ {comparator}",
        "%old = OpAtomicCompareExchange _ {ptr} %scope %equal %unequal %value %comparator",
        "OpStore {old} %old",
        scope = const SCOPE,
        equal = const EQUAL,
        unequal = const UNEQUAL,
        ptr = in(reg) ptr,
        value = in(reg) &value,
        comparator = in(reg) &comparator,
        old = in(reg) &mut old,
    }

    old
}

/*
/// Perform the following steps atomically with respect to any other atomic
/// accesses within `SCOPE` to the same location:
///
/// 1. Load through `ptr` to get the original value
/// 2. Get a new value from `value` only if the original value equals
///    `comparator`, and
/// 3. Store the new value back through `ptr`, only if the original value
///    equaled `comparator`.
///
/// The result is the original value.
#[inline]
pub(crate) unsafe fn atomic_compare_exchange_f32<
    const SCOPE: u32,
    const EQUAL: u32,
    const UNEQUAL: u32,
>(
    ptr: &mut f32,
    value: f32,
    comparator: f32,
) -> f32 {
    let mut old = 0u32;

    asm! {
        "%u32 = OpTypeInt 32 0",
        "%scope = OpConstant %u32 {scope}",
        "%equal = OpConstant %u32 {equal}",
        "%unequal = OpConstant %u32 {unequal}",
        "%value = OpLoad _ {value}",
        "%comparator = OpLoad _ {comparator}",
        "%ty_ptr_u32 = OpTypePointer Generic %u32",
        "%ptr_u32 = OpBitcast %ty_ptr_u32 {ptr}",
        "%old = OpAtomicCompareExchange %ty_ptr_u32 %ptr_u32 %scope %equal %unequal %value %comparator",
        "OpStore {old} %old",
        scope = const SCOPE,
        equal = const EQUAL,
        unequal = const UNEQUAL,
        ptr = in(reg) ptr,
        value = in(reg) &value.to_bits(),
        comparator = in(reg) &comparator.to_bits(),
        old = in(reg) &mut old,
    }

    f32::from_bits(old)
}*/

/// Perform the following steps atomically with respect to any other atomic
/// accesses within `SCOPE` to the same location:
///
/// 1) load through `ptr` to get an original value,
/// 2) get a new value by integer addition of original value and `value`, and
/// 3) store the new value back through `ptr`.
///
/// The result is the Original Value.
#[cfg(feature = "false")]
#[doc(alias = "OpAtomicIAdd")]
#[inline]
pub(crate) unsafe fn atomic_i_add<I: Integer, const SCOPE: u32, const SEMANTICS: u32>(
    ptr: &mut I,
    value: I,
) -> I {
    let mut old = I::default();

    asm! {
        "%u32 = OpTypeInt 32 0",
        "%scope = OpConstant %u32 {scope}",
        "%semantics = OpConstant %u32 {semantics}",
        "%value = OpLoad _ {value}",
        "%old = OpAtomicIAdd _ {ptr} %scope %semantics %value",
        "OpStore {old} %old",
        scope = const SCOPE,
        semantics = const SEMANTICS,
        ptr = in(reg) ptr,
        old = in(reg) &mut old,
        value = in(reg) &value,
    }

    old
}


// Adapted from https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/blob/master/demokernels/tC0_tA0_tB0_colMaj1_m1000_n2000_k3000_lda1100_ldb3200_ldc1300_ws100000000_f32/A_MIC8_PAD1_PLU0_LIW0_MIW1_WOS1__B_MIC6_PAD1_PLU1_LIW0_MIW1_WOS1__C_UNR8_GAL3_PUN1_ICE2_NAW16_UFO0_MAC256_SKW10/cw_alpha.cl
unsafe fn atomic_f32_add<const SCOPE: u32, const SEMANTICS: u32>(
    ptr: &mut u32,
    value: f32,
) -> f32 {
    let mut previous: u32;
    loop {
        previous = *ptr;
        let value = (f32::from_bits(previous) + value).to_bits();
        if atomic_compare_exchange::<u32, SCOPE, SEMANTICS, SEMANTICS>(ptr, value, previous) == previous {
            return f32::from_bits(previous);
        }
    }
}

#[repr(transparent)]
pub struct AtomicF32(u32);

impl core::ops::AddAssign<f32> for AtomicF32 {
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        unsafe {
            atomic_f32_add::<{Scope::Device as u32}, {Semantics::NONE.bits()}>(&mut self.0, rhs);
        }
    }
}

pub mod tests {
    use super::*;

    #[autobind]
    #[spirv(compute(threads(1)))]
    pub fn atomic_add_f32(
        #[spirv(workgroup_id)]
        group_id: UVec3,
        #[spirv(storage_buffer)]
        x: &[f32],
        #[spirv(storage_buffer)]
        y: &mut [AtomicF32],
    ) {
        y[0] += x[group_id.x as usize];
    }
}
