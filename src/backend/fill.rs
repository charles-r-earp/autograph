use super::{BufferSliceMut, Device, Scalar};
use crate::Result;
use crate::util::size_eq;
use bytemuck::{Pod, Zeroable};

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
struct FillU32PushConsts {
    n: u32,
    x: u32
}


pub(super) fn fill<T: Scalar>(device: &Device, slice: BufferSliceMut<T>, x: T) -> Result<()>
where
    T: Pod,
{
    let src = if size_eq::<T, u8>() {
        include_shader!("glsl/fill_u8.spv")
    } else if size_eq::<T, u16>() {
        include_shader!("glsl/fill_u16.spv")
    } else if size_eq::<T, u32>() {
        include_shader!("glsl/fill_u32.spv")
    } else {
        unreachable!()
    };
    
    eprintln!("{:?}", x);
    
    let n = slice.len as u32;
    
    let push_consts = FillU32PushConsts {
        n,
        x: x.to_bits_u32()
    };

    device
        .compute_pass(src, "main")?
        .buffer_slice_mut(slice)?
        .push_constants(push_consts)?
        .global_size([n, 1, 1])
        .enqueue()
}
