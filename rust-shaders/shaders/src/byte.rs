use crate::util::Load;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct u8x4(u32);

impl u8x4 {
    fn to_bits(self) -> u32 { self.0 }
    //fn from_bits(v: u32) -> Self { Self(v) }
    fn to_u32x4(self) -> [u32; 4] {
        let x = self.to_bits();
        [
            x & 0xFF,
            (x >> 8) & 0xFF,
            (x >> 16) & 0xFF,
            x >> 24,
        ]
    }
}

impl Load<u32> for [u8x4] {
    fn load(&self, index: usize) -> u32 {
        let idx = index / 4;
        let rem = index % 4;
        self[idx].to_u32x4()[rem]
    }
}
