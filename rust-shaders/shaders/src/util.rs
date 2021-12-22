//use spirv_std::glam::UVec4;

/*
pub(crate) fn u8x4_to_uvec4(x: u32) -> UVec4 {
    UVec4::new(
        x & 0xFF,
        (x >> 8) & 0xFF,
        (x >> 16) & 0xFF,
        x >> 24,
    )
}
*/
/*
pub(crate) fn bf16x2_to_vec2(x: u32) -> Vec2 {
    Vec2::new(
        f32::from_bits(x << 16),
        f32::from_bits(x >> 16 << 16)
    )
}

pub(crate) fn vec2_to_bf16x2(x: Vec2) -> u32 {
    (x.x.to_bits() >> 16) | ((x.y.to_bits() >> 16) << 16)
}

pub(crate) fn vec4_to_bf16x4(x: Vec4) -> (u32, u32) {
    (vec2_to_bf16x2(x.xy()), vec2_to_bf16x2(x.zw()))
}
*/

pub(crate) trait Load<T> {
    fn load(&self, index: usize) -> T;
}

pub(crate) trait Store<T>: Load<T> {
    fn store(&mut self, index: usize, value: T);
}

impl<T: Copy> Load<T> for [T] {
    fn load(&self, index: usize) -> T {
        self[index]
    }
}

impl<T: Copy> Store<T> for [T] {
    fn store(&mut self, index: usize, value: T) {
        self[index] = value;
    }
}
/*
pub(crate) trait AtomicAdd<T> {
    unsafe fn atomic_add<const SCOPE: u32, const SEMANTICS: u32>(&mut self, index: usize, value: T) -> T;
}

impl AtomicAdd<u32> for [u32] {
    unsafe fn atomic_add<const SCOPE: u32, const SEMANTICS: u32>(&mut self, index: usize, value: u32) -> u32 {
        atomic_i_add::<u32, SCOPE, SEMANTICS>(&mut self[index], value)
    }
}

impl AtomicAdd<i32> for [i32] {
    unsafe fn atomic_add<const SCOPE: u32, const SEMANTICS: u32>(&mut self, index: usize, value: i32) -> i32 {
        atomic_i_add::<i32, SCOPE, SEMANTICS>(&mut self[index], value)
    }
}
*/
/*
impl AtomicAdd<f32> for [f32] {
    unsafe fn atomic_add<const SCOPE: u32, const SEMANTICS: u32>(&mut self, index: usize, value: f32) -> f32 {
        atomic_f32_add::<SCOPE, SEMANTICS>(&mut self[index], value)
    }
}*/
/*
pub mod tests {
    use super::*;
    use spirv_std::{
        memory::{Scope, Semantics},
        glam::UVec3,
    };

    const SCOPE: u32 = Scope::Workgroup as u32;
    const SEMANTICS: u32 = Semantics::NONE.bits();

    #[allow(unused_attributes)]
    #[spirv(compute(threads(1)))]
    pub fn atomic_add_u32(
        #[spirv(workgroup_id)]
        group_id: UVec3,
        #[spirv(storage_buffer, descriptor_set=0, binding=0)]
        x: &[u32],
        #[spirv(storage_buffer, descriptor_set=0, binding=1)]
        y: &mut [u32],
    ) {
        unsafe {
            y.atomic_add::<SCOPE, SEMANTICS>(0, x[group_id.x as usize]);
        }
    }

    #[allow(unused_attributes)]
    #[spirv(compute(threads(1)))]
    pub fn atomic_add_i32(
        #[spirv(workgroup_id)]
        group_id: UVec3,
        #[spirv(storage_buffer, descriptor_set=0, binding=0)]
        x: &[i32],
        #[spirv(storage_buffer, descriptor_set=0, binding=1)]
        y: &mut [i32],
    ) {
        unsafe {
            y.atomic_add::<SCOPE, SEMANTICS>(0, x[group_id.x as usize]);
        }
    }
/*
    #[allow(unused_attributes)]
    #[spirv(compute(threads(1)))]
    pub fn atomic_add_f32(
        #[spirv(workgroup_id)]
        group_id: UVec3,
        #[spirv(storage_buffer, descriptor_set=0, binding=0)]
        x: &[f32],
        #[spirv(storage_buffer, descriptor_set=0, binding=1)]
        y: &mut [f32],
    ) {
        unsafe {
            y.atomic_add::<SCOPE, SEMANTICS>(0, x[group_id.x as usize]);
        }
    }*/
}
*/
