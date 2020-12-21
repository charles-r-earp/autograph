use super::{BufferSliceMut, Device, Scalar};
use crate::util::size_eq;
use crate::Result;
use bytemuck::{Pod, Zeroable};

#[derive(Copy)]
#[repr(C, packed)]
struct FillPushConsts<T: Scalar> {
    x: T,
    n: u32,
}

impl<T: Scalar> Clone for FillPushConsts<T> {
    fn clone(&self) -> Self {
        *self
    }
}

unsafe impl<T: Scalar> Zeroable for FillPushConsts<T> {}

unsafe impl<T: Scalar> Pod for FillPushConsts<T> {}

pub(super) fn fill<T: Scalar>(device: &Device, slice: BufferSliceMut<T>, x: T) -> Result<()>
where
    T: Scalar,
{
    let src = if size_eq::<T, u8>() {
        include_shader!("glsl/fill_u8.spv")
    } else if size_eq::<T, u16>() {
        include_shader!("glsl/fill_u16.spv")
    } else if size_eq::<T, u32>() {
        include_shader!("glsl/fill_u32.spv")
    } else if size_eq::<T, u64>() {
        include_shader!("glsl/fill_u64.spv")
    } else {
        unreachable!()
    };

    let n = slice.len as u32;

    let builder = device
        .compute_pass(src, "main")?
        .buffer_slice_mut(slice)?
        .global_size([n, 1, 1]);

    if size_eq::<T, u64>() {
        let push_consts = FillPushConsts {
            x: x.to_bits_u64().unwrap(),
            n,
        };
        builder.push_constants(push_consts)?.enqueue()
    } else {
        let push_consts = FillPushConsts {
            x: x.to_bits_u32().unwrap(),
            n,
        };
        builder.push_constants(push_consts)?.enqueue()
    }
}
