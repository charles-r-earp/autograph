use spirv_std::{
    memory::{Scope, Semantics},
    glam::UVec3,
};
use crate::{
    autobind,
    atomic::atomic_compare_exchange,
};

#[repr(C)]
pub struct PoolPushConsts2 {
    bs: u32,
    ih: u32,
    iw: u32,
    oh: u32,
    ow: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32
}

#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(u32)]
enum PoolKind {
    Max,
    Mean,
}

fn pool_2d_f32_impl(kind: PoolKind, global_id: UVec3, x: &[f32], y: &mut [f32], push_consts: &PoolPushConsts2) {
    let bs = push_consts.bs as usize;
    let ih = push_consts.ih as usize;
    let iw = push_consts.iw as usize;
    let oh = push_consts.oh as usize;
    let ow = push_consts.ow as usize;
    let kh = push_consts.kh as usize;
    let kw = push_consts.kw as usize;
    let sh = push_consts.sh as usize;
    let sw = push_consts.sw as usize;
    let ph = push_consts.ph as usize;
    let pw = push_consts.pw as usize;
    let global_x = global_id.x as usize;
    let global_y = global_id.y as usize;
    let bid = global_x / oh;
    let hid = global_x % oh;
    let wid = global_y;
    let mut acc = match kind {
        PoolKind::Max => f32::NEG_INFINITY,
        PoolKind::Mean => 0.,
    };
    if bid < bs { if hid < oh { if wid < ow {
        for ki in 0 .. kh {
            let i = (hid * sh + ki) as isize - ph as isize;
            if i >= 0 {
                let i = i as usize;
                if i < ih {
                    for kj in 0 .. kw {
                        let j = (wid * sw + kj) as isize - pw as isize;
                        if j >= 0 {
                            let j = j as usize;
                            if j < iw {
                                let val = x[bid * ih * iw + i as usize * iw + j as usize];
                                match kind {
                                    PoolKind::Max => {
                                        acc = f32::max(val, acc);
                                    }
                                    PoolKind::Mean => {
                                        acc += val / (kh * kw) as f32;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        y[bid * oh * ow + hid * ow + wid] = acc;
    }}}
}

#[allow(unused_attributes)]
#[spirv(compute(threads(16, 16)))]
pub fn max_pool_2d_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] x: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] y: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &PoolPushConsts2,
) {
    pool_2d_f32_impl(PoolKind::Max, global_id, x, y, push_consts);
}

#[allow(unused_attributes)]
#[spirv(compute(threads(16, 16)))]
pub fn max_pool_indices_2d_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] x: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] ix: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] y: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &PoolPushConsts2,
) {
    let bs = push_consts.bs as usize;
    let ih = push_consts.ih as usize;
    let iw = push_consts.iw as usize;
    let oh = push_consts.oh as usize;
    let ow = push_consts.ow as usize;
    let kh = push_consts.kh as usize;
    let kw = push_consts.kw as usize;
    let sh = push_consts.sh as usize;
    let sw = push_consts.sw as usize;
    let ph = push_consts.ph as usize;
    let pw = push_consts.pw as usize;
    let global_x = global_id.x as usize;
    let global_y = global_id.y as usize;
    let bid = global_x / oh;
    let hid = global_x % oh;
    let wid = global_y;
    let mut idx = 0;
    let mut acc = f32::NEG_INFINITY;
    if bid < bs { if hid < oh { if wid < ow {
        for ki in 0 .. kh {
            let i = (hid * sh + ki) as isize - ph as isize;
            if i >= 0 {
                let i = i as usize;
                if i < ih {
                    for kj in 0 .. kw {
                        let j = (wid * sw + kj) as isize - pw as isize;
                        if j >= 0 {
                            let j = j as usize;
                            if j < iw {
                                let val = x[bid * ih * iw + i as usize * iw + j as usize];
                                if val > acc {
                                    idx = i * iw + j;
                                    acc = val;
                                }
                            }
                        }
                    }
                }
            }
        }
        ix[bid * oh * ow + hid * ow + wid] = idx as u32;
        y[bid * oh * ow + hid * ow + wid] = acc;
    }}}
}

#[allow(unused_attributes)]
#[spirv(compute(threads(8, 8)))]
pub fn mean_pool_2d_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] x: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] y: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &PoolPushConsts2,
) {
    pool_2d_f32_impl(PoolKind::Mean, global_id, x, y, push_consts);
}

/*
#[repr(C)]
pub struct PoolBackwardPushConsts2 {
    bs: u32,
    ic: u32,
    ih: u32,
    iw: u32,
    oh: u32,
    ow: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
    s_bez_h: u32,
    s_bez_w: u32,
    d_bez_h: u32,
    d_bez_w: u32,
    gcd_h: u32,
    gcd_w: u32,
}

fn grid_ceil(x: i32, step: i32) -> i32 {
    if x > 0 {
        (x - 1) / step + 1
    } else {
        x / step * step
    }
}

fn max_i32(a: i32, b: i32) -> i32 {
    if a > b {
        a
    } else {
        b
    }
}

fn min_usize(a: usize, b: usize) -> usize {
    if a < b {
        a
    } else {
        b
    }
}


#[allow(unused_attributes)]
#[spirv(compute(threads(16, 16)))]
pub fn max_pool_2d_backward_f32(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] ix: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dx: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] dy: &[f32],
    #[spirv(push_constant)]
    push_consts: &PoolBackwardPushConsts2
) {
    let bs = push_consts.bs as usize;
    let ic = push_consts.ic as usize;
    let ih = push_consts.ih as usize;
    let iw = push_consts.iw as usize;
    let oh = push_consts.oh as usize;
    let ow = push_consts.ow as usize;
    let kh = push_consts.kh as usize;
    let kw = push_consts.kw as usize;
    let sh = push_consts.sh as usize;
    let sw = push_consts.sw as usize;
    let ph = push_consts.ph as usize;
    let pw = push_consts.pw as usize;
    let dh = push_consts.dh as usize;
    let dw = push_consts.dw as usize;
    let s_bez_h = push_consts.s_bez_h as usize;
    let s_bez_w = push_consts.s_bez_w as usize;
    let d_bez_h = push_consts.d_bez_h as usize;
    let d_bez_w = push_consts.d_bez_w as usize;
    let gcd_h = push_consts.gcd_h as usize;
    let gcd_w = push_consts.gcd_w as usize;
    let group_x = group_id.x as usize;
    let group_y = group_id.y as usize;
    let local_x = local_id.x as usize;
    let local_y = local_id.y as usize;
    let global_x = group_x * 16 + local_x;
    let global_y = group_y * 16 + local_y;
    let bid = global_x / ic;
    let cid = global_x % ic;
    let bidx = bid * ic * ih * iw;
    let bidy = bid * ic * oh * ow;
    let cidx = cid * ih * iw;
    let cidy = cid * oh * ow;
    let iw_scaled = (iw - 1) / gcd_w + 1;
    let gcd_scale_h = global_y / iw_scaled + (ph - 1) / gcd_h + 1;
    let gcd_scale_w = global_y % iw_scaled + (pw - 1) / gcd_w + 1;
    let hidx = gcd_scale_h * gcd_h - ph;
    let widx = gcd_scale_w * gcd_w - pw;
    let th_step = sh * dh / gcd_h;
    let tw_step = sw * dw / gcd_w;

    let th_begin = grid_ceil(max_i32(-((s_bez_h * gcd_scale_h * sh) as i32), ((d_bez_h * gcd_scale_h - kh + 1) * dh) as i32), th_step as i32) as usize;
    let th_end = min_usize((oh - s_bez_h * gcd_scale_h) * sh, (d_bez_h * gcd_scale_h + 1) * dh);
    let tw_begin = grid_ceil(max_i32(-((s_bez_w * gcd_scale_w * sw) as i32), ((d_bez_w * gcd_scale_w - kw + 1) * dw) as i32), tw_step as i32) as usize;
    let tw_end = min_usize((ow - s_bez_w * gcd_scale_w) * sw, (d_bez_w * gcd_scale_w + 1) * dw);

    if bid < bs { if cid < ic { if hidx < ih { if widx < iw {
        let mut acc = 0.;
        let mut th = th_begin;
        while th < th_end {
            let mut tw = tw_begin;
            while tw < tw_end {
                let hid = th / sh + s_bez_h * gcd_scale_h;
                let wid = tw / sw + s_bez_w * gcd_scale_w;
                if hid < oh { if wid < ow {
                    if ix[bidy + cidy + hid * ow + wid] as usize == hidx * iw + widx {
                        acc += dy[bidy + cidy + hid * ow + wid];
                    }
                }}
                tw += tw_step;
            }
            th += th_step;
        }
        dx[bidx + cidx + hidx * iw + widx] += acc;
    }}}}
}*/


#[repr(C)]
pub struct MaxPoolBackwardPushConsts2 {
    bs: u32,
    ih: u32,
    iw: u32,
    oh: u32,
    ow: u32,
}

// TOOD: flatten to threads(256) and move atomic impl to type / trait / function abstraction
#[autobind]
#[spirv(compute(threads(1, 16, 16)))]
pub fn max_pool_2d_backward_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer)] ix: &[u32],
    #[spirv(storage_buffer)] dx: &mut [u32],
    #[spirv(storage_buffer)] dy: &[f32],
    #[spirv(push_constant)]
    push_consts: &MaxPoolBackwardPushConsts2
) {
    let bs = push_consts.bs as usize;
    let ih = push_consts.ih as usize;
    let iw = push_consts.iw as usize;
    let oh = push_consts.oh as usize;
    let ow = push_consts.ow as usize;
    let bid = global_id.x as usize;
    let hid = global_id.y as usize;
    let wid = global_id.z as usize;
    if bid < bs { if hid < oh { if wid < ow {
        let y_idx = bid * oh * ow + hid * ow + wid;
        let x_idx = bid * ih * iw + ix[y_idx] as usize;
        // dx[x_idx] += dy[y_idx];
        let mut previous: u32;
        loop {
            previous = dx[x_idx];
            let value = (f32::from_bits(previous) + dy[y_idx]).to_bits();
            if unsafe {
                atomic_compare_exchange::<u32, {Scope::Device as u32}, {Semantics::NONE.bits()}, {Semantics::NONE.bits()}>(&mut dx[x_idx], value, previous)
            } == previous {
                break;
            }
        }
    }}}
}

#[repr(C)]
pub struct MeanPoolBackwardPushConsts2 {
    bs: u32,
    ic: u32,
    ih: u32,
    iw: u32,
    oh: u32,
    ow: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
}

#[allow(unused_attributes)]
#[spirv(compute(threads(8, 8)))]
pub fn mean_pool_2d_backward_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] dx: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dy: &[f32],
    #[spirv(push_constant)]
    push_consts: &MeanPoolBackwardPushConsts2
) {
    let bs = push_consts.bs as usize;
    let ic = push_consts.ic as usize;
    let ih = push_consts.ih as usize;
    let iw = push_consts.iw as usize;
    let oh = push_consts.oh as usize;
    let ow = push_consts.ow as usize;
    let kh = push_consts.kh as usize;
    let kw = push_consts.kw as usize;
    let sh = push_consts.sh as usize;
    let sw = push_consts.sw as usize;
    let ph = push_consts.ph as usize;
    let pw = push_consts.pw as usize;
    let dh = push_consts.dh as usize;
    let dw = push_consts.dw as usize;
    let global_x = global_id.x as usize;
    let global_y = global_id.y as usize;
    let bid = global_x / ic;
    let cid = global_x % ic;
    let bidx = bid * ic * ih * iw;
    let bidy = bid * ic * oh * ow;
    let cidx = cid * ih * iw;
    let cidy = cid * oh * ow;
    let hidy = global_y / ow;
    let widy = global_y % ow;

    let dy = dy[bidy + cidy + hidy * ow + widy];

    if bid < bs { if cid < ic { if hidy < oh { if widy < ow {
        for ki in 0 .. kh {
            for kj in 0 .. kw {
                let hidx = (hidy * sh + ki * dh) as i32 - ph as i32;
                let widx = (widy * sw + kj * dw) as i32 - pw as i32;
                if hidx >= 0 { if hidx < ih as i32 { if widx >= 0 { if widx < iw as i32 {
                    // should be atomic add
                    dx[bidx + cidx + hidx as usize * iw + widx as usize] += dy;
                }}}}
            }
        }
    }}}}
}
