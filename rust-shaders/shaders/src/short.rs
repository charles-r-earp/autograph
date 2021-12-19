use crate::util::Load;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct u16x2(u32);

impl u16x2 {
    fn to_bits(self) -> u32 { self.0 }
    //fn from_bits(v: u32) -> Self { Self(v) }
    fn to_u32x2(&self) -> [u32; 2] {
        let x = self.to_bits();
        [
            x & 0xFFFF,
            (x >> 16) & 0xFFFF,
        ]
    }
}

impl Load<u32> for [u16x2] {
    fn load(&self, index: usize) -> u32 {
        let idx = index / 2;
        let rem = index % 2;
        self[idx].to_u32x2()[rem]
    }
}
