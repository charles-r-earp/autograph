use super::{BufferSliceMut, Device, Scalar};
use crate::util::size_eq;
use crate::Result;
use bytemuck::{Pod, Zeroable};

#[derive(Copy)]
#[repr(C, packed)]
struct FillPushConsts<T> {
    x: T,
    n: u32,
}

impl<T: Copy> Clone for FillPushConsts<T> {
    fn clone(&self) -> Self {
        *self
    }
}

unsafe impl<T: Zeroable> Zeroable for FillPushConsts<T> {}

unsafe impl<T: Pod> Pod for FillPushConsts<T> {}

pub(super) fn fill<T: Scalar>(device: &Device, slice: BufferSliceMut<T>, x: T) -> Result<()>
where
    T: Scalar,
{
    let src = if size_eq::<T, u64>() {
        include_shader!("glsl/fill_u64.spv")
    } else {
        include_shader!("glsl/fill_u32.spv")
    };

    let n = slice.len as u32;

    let builder = device.compute_pass(src, "main")?.buffer_slice_mut(slice)?;

    let builder = if size_eq::<T, u64>() {
        builder.push_constants(bytemuck::cast_slice(&[FillPushConsts {
            x: x.to_bits_u64().unwrap(),
            n,
        }]))?
    } else {
        let (x, n) = if let Some(x_u8) = x.to_bits_u8() {
            let n = if n % 4 == 0 { n / 4 } else { n / 4 + 1 };
            (u32::from_ne_bytes([x_u8; 4]), n)
        } else if let Some(x_u16) = x.to_bits_u16() {
            let n = if n % 2 == 0 { n / 2 } else { n / 2 + 1 };
            let [a, b] = x_u16.to_ne_bytes();
            (u32::from_ne_bytes([a, b, a, b]), n)
        } else {
            (x.to_bits_u32().unwrap(), n)
        };
        builder.push_constants(bytemuck::cast_slice(&[FillPushConsts { x, n }]))?
    };

    builder.global_size([n, 1, 1]).enqueue()
}
