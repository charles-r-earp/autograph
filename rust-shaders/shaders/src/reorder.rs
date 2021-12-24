use crate::{
    autobind,
    util::{Load, Store},
};
use spirv_std::glam::UVec3;
use num_traits::Num;

#[repr(C)]
pub struct ReorderPushConsts2<T> {
    bs: u32,
    ih: u32,
    iw: u32,
    rsx: u32,
    csx: u32,
    beta: T,
    oh: u32,
    ow: u32,
    rsy: u32,
    csy: u32,
}

fn reorder_2d<T, X, Y>(
    group_id: UVec3,
    local_id: UVec3,
    x: &[X],
    y: &mut [Y],
    push_consts: &ReorderPushConsts2<T>,
) where T: Num + Copy, [X]: Load<T>, [Y]: Store<T> {
    let bs = push_consts.bs as usize;
    let ih = push_consts.ih as usize;
    let iw = push_consts.iw as usize;
    let rsx = push_consts.rsx as usize;
    let csx = push_consts.csx as usize;
    let beta = push_consts.beta;
    let oh = push_consts.oh as usize;
    let ow = push_consts.ow as usize;
    let rsy = push_consts.rsy as usize;
    let csy = push_consts.csy as usize;
    let group_id = group_id.x as usize;
    let groups_h = oh / 16 + if oh % 16 != 0 { 1 } else { 0 };
    let groups_w = ow / 16 + if ow % 16 != 0 { 1 } else { 0 };
    let groups_hw = groups_h * groups_w;
    let bid = group_id / groups_hw;
    let group_hw = group_id % groups_hw;
    let group_h = group_hw / groups_w;
    let group_w = group_hw % groups_w;
    let local_id = local_id.x as usize;
    let local_h = local_id / 16;
    let local_w = local_id % 16;
    let hid = group_h * 16 + local_h;
    let wid = group_w * 16 + local_w;

    if bid < bs { if hid < oh { if wid < ow {
        let y_idx = bid * oh * ow + hid * rsy + wid * csy;
        let x = if hid < ih {
            if wid < iw {
                let x_idx = bid * ih * iw + hid * rsx + wid * csx;
                x.load(x_idx)
            } else {
                T::zero()
            }
        } else {
            T::zero()
        };
        y.store(y_idx, beta * y.load(y_idx) + x);
    }}}
}

#[autobind]
#[spirv(compute(threads(256)))]
pub fn reorder_2d_f32_f32(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer)] x: &[f32],
    #[spirv(storage_buffer)] y: &mut [f32],
    #[spirv(push_constant)] push_consts: &ReorderPushConsts2<f32>,
) {
    reorder_2d(
        group_id,
        local_id,
        x,
        y,
        push_consts,
    );
}


#[repr(C)]
pub struct AsStandardLayoutPushConsts4 {
    d0: u32,
    s0: i32,
    d1: u32,
    s1: i32,
    d2: u32,
    s2: i32,
    d3: u32,
    s3: i32,
}

#[allow(unused_attributes)]
#[spirv(compute(threads(256)))]
pub fn as_standard_layout_4d_u32(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] x: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] y: &mut [u32],
    #[spirv(push_constant)]
    push_consts: &AsStandardLayoutPushConsts4,
) {
    let d0 = push_consts.d0 as usize;
    let s0 = push_consts.s0 as isize;
    let d1 = push_consts.d1 as usize;
    let s1 = push_consts.s1 as isize;
    let d2 = push_consts.d2 as usize;
    let s2 = push_consts.s2 as isize;
    let d3 = push_consts.d3 as usize;
    let s3 = push_consts.s3 as isize;
    let gid = group_id.x as usize * 256 + local_id.x as usize;
    let n = d0 * d1 * d2 * d3;
    let i0 = (gid / (d1 * d2 * d3)) as isize;
    let r0 = gid % (d1 * d2 * d3);
    let i1 = (r0 / (d2 * d3)) as isize;
    let r1 = r0 % (d2 * d3);
    let i2 = (r1 / d3) as isize;
    let i3 = (r1 % d3) as isize;
    let xid = (i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3) as usize;
    if gid < n {
        y[gid] = x[xid];
    }
}

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
#[spirv(compute(threads(256)))]
pub fn as_standard_layout_6d_u32(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
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
    let gid = group_id.x as usize * 256 + local_id.x as usize;
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
