use spirv_std::glam::UVec4;

pub(crate) fn u8x4_to_uvec4(x: u32) -> UVec4 {
    UVec4::new(
        x & 0xFF,
        (x >> 8) & 0xFF,
        (x >> 16) & 0xFF,
        x >> 24,
    )
}

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
