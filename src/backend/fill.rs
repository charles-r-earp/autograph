use super::{BufferSliceMut, Device, Num};
use crate::Result;
use bytemuck::{Pod, Zeroable};
use std::mem::size_of;

#[derive(Clone, Copy)]
#[repr(C)]
struct FillPushConsts<T> {
    n: u32,
    x: T,
}

unsafe impl<T: Zeroable> Zeroable for FillPushConsts<T> {}

unsafe impl<T: Pod> Pod for FillPushConsts<T> {}

pub(super) fn fill<T: Num>(device: &Device, slice: BufferSliceMut<T>, x: T) -> Result<()>
where
    T: Pod,
{
    let src = if size_of::<T>() == size_of::<f32>() {
        include_shader!("glsl/fill_f32.spv")
    } else {
        unreachable!()
    };

    let n = slice.len as u32;

    device
        .compute_pass(src, "main")?
        .buffer_slice_mut(slice)?
        .push_constants(FillPushConsts { n, x })?
        .global_size([n, 1, 1])
        .enqueue()
}
