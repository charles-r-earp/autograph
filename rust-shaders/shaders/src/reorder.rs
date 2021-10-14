use spirv_std::glam::UVec3;

#[repr(C)]
pub struct AsStandardLayoutPushConsts6 {
    d0: u32,
    s0: i32,
    d1: u32,
    s1: i32,
    d2: u32,
    s2: i32,
    d3: u32,
    s3: i32,
    d4: u32,
    s4: i32,
    d5: u32,
    s5: i32,
}


#[allow(unused_attributes)]
#[spirv(compute(threads(64)))]
pub fn as_standard_layout_6d_u32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] x: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] y: &mut [u32],
    #[spirv(push_constant)]
    push_consts: &AsStandardLayoutPushConsts6,
) {
    let d0 = push_consts.d0 as usize;
    let s0 = push_consts.s0 as isize;
    let d1 = push_consts.d1 as usize;
    let s1 = push_consts.s1 as isize;
    let d2 = push_consts.d2 as usize;
    let s2 = push_consts.s2 as isize;
    let d3 = push_consts.d3 as usize;
    let s3 = push_consts.s3 as isize;
    let d4 = push_consts.d4 as usize;
    let s4 = push_consts.s4 as isize;
    let d5 = push_consts.d5 as usize;
    let s5 = push_consts.s5 as isize;
    let gid = global_id.x as usize;
    let n = d0 * d1 * d2 * d3 * d4 * d5;
    let i0 = (gid / (d1 * d2 * d3 * d4 * d5)) as isize;
    let r0 = gid % (d1 * d2 * d3 * d4 * d5);
    let i1 = (r0 / (d2 * d3 * d4 * d5)) as isize;
    let r1 = r0 % (d2 * d3 * d4 * d5);
    let i2 = (r1 / (d3 * d4 * d5)) as isize;
    let r2 = r1 % (d3 * d4 * d5);
    let i3 = (r2 / (d4 * d5)) as isize;
    let r3 = r2 % (d4 * d5);
    let i4 = (r3 / d5) as isize;
    let i5 = (r3 % d5) as isize;
    let xid = (i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3 + i4 * s4 + i5 * s5) as usize;
    if gid < n {
        y[gid] = x[xid];
    }
}
