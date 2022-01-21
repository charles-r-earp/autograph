use crate::{
    autobind,
};
use spirv_std::glam::UVec3;

#[repr(C)]
pub struct CopyPushConsts {
    n: u32,
}

#[autobind]
#[spirv(compute(threads(256)))]
pub fn copy_u32(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer)] x: &[u32],
    #[spirv(storage_buffer)] y: &mut [u32],
    #[spirv(push_constant)]
    push_consts: &CopyPushConsts,
) {
    let gid = (group_id.x * 256 + local_id.x) as usize;
    let n = push_consts.n as usize;
    if gid < n {
        y[gid] = x[gid];
    }
}
