use spirv_std::glam::UVec3;
use crate::autobind;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct u32x2 {
    _x: u32,
    _y: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct u32x4 {
    _x: u32,
    _y: u32,
    _z: u32,
    _w: u32,
}

#[repr(C)]
pub struct FillPushConsts<T> {
    n: u32,
    x: T,
}

macro_rules! impl_fill {
    ($($func:ident<$t:ty>),* $(,)?) => (
        $(
            #[autobind]
            #[spirv(compute(threads(256)))]
            pub fn $func(
                #[spirv(workgroup_id)]
                group_id: UVec3,
                #[spirv(local_invocation_id)]
                local_id: UVec3,
                #[spirv(storage_buffer)] y: &mut [$t],
                #[spirv(push_constant)]
                push_consts: &FillPushConsts<$t>,
            ) {
                let gid = (group_id.x * 256 + local_id.x) as usize;
                let n = push_consts.n as usize;
                let x = push_consts.x;
                if gid < n {
                    y[gid] = x;
                }
            }
        )*
    );
}

impl_fill!{
    fill_u32<u32>,
    fill_u32x2<u32x2>,
}
