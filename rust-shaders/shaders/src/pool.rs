use spirv_std::glam::UVec3;

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
#[spirv(compute(threads(8, 8)))]
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
#[spirv(compute(threads(8, 8)))]
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

/*
fn pool_2d_backward_f32_impl(
    kind: PoolKind, global_id: UVec3, x: &[f32], dx: &mut [f32], dy: &[f32], push_consts: &PoolBackwardPushConsts2
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
    let global_x = global_id.x as usize;
    let global_y = global_id.y as usize;
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

    let th_begin = grid_ceil(max_i32(-((s_bez_h * gcd_scale_h * sh) as i32), ((d_bez_h * gcd_scale_h - kh + 1) * dh) as i32), th_step as i32) as usize;
    let th_end = min_usize((oh - s_bez_h * gcd_scale_h) * sh, (d_bez_h * gcd_scale_h + 1) * dh);
    let tw_begin = grid_ceil(max_i32(-((s_bez_w * gcd_scale_w * sw) as i32), ((d_bez_w * gcd_scale_w - kw + 1) * dw) as i32), tw_step as i32) as usize;
    let tw_end = min_usize((ow - s_bez_w * gcd_scale_w) * sw, (d_bez_w * gcd_scale_w + 1) * dw);

    if bid < bs { if cid < ic { if hidx < ih { if widx < iw {
        let mut max = f32::NEG_INFINITY;
        let mut acc = 0.;
        let mut th = th_begin;
        while th < th_end {
            let mut tw = tw_begin;
            while tw < tw_end {
                let hid = th / sh + s_bez_h * gcd_scale_h;
                let wid = tw / sw + s_bez_w * gcd_scale_w;
                if hid < oh { if wid < ow {
                    match kind {
                        PoolKind::Max => {
                            let val = x[bidy + cidy + hid * ow + wid];
                            if val > max {
                                max = val;
                                acc = dy[bidy + cidy + hid * ow + wid];
                            }
                        }
                        _ => {
                            acc += dy[bidy + cidy + hid * ow + wid];
                        }
                    }
                }}
                tw += tw_step;
            }
            th += th_step;
        }
        dx[bidx + cidx + hidx * iw + widx] = acc;
    }}}}
}
*/

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
#[spirv(compute(threads(8, 8)))]
pub fn max_pool_2d_backward_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
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
    let global_x = global_id.x as usize;
    let global_y = global_id.y as usize;
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
        dx[bidx + cidx + hidx * iw + widx] = acc;
    }}}}
}

#[allow(unused_attributes)]
#[spirv(compute(threads(8, 8)))]
pub fn mean_pool_2d_backward_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] dx: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dy: &[f32],
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
    let global_x = global_id.x as usize;
    let global_y = global_id.y as usize;
    let bid = global_x / ic;
    let cid = global_x % ic;
    let bidx = bid * ic * ih * iw;
    let bidy = bid * ic * oh * ow;
    let cidx = cid * ih * iw;
    let cidy = cid * oh * ow;
    let iw_scaled = (iw - 1) / gcd_w + 1;
    let gcd_scale_h = (global_y / iw_scaled) + ((ph as i32 - 1) / gcd_h as i32 + 1) as usize;
    let gcd_scale_w = (global_y % iw_scaled) + ((pw as i32 - 1) / gcd_w as i32 + 1) as usize;
    let hidx = gcd_scale_h * gcd_h - ph;
    let widx = gcd_scale_w * gcd_w - pw;
    let th_step = sh * dh / gcd_h;
    let tw_step = sw * dw / gcd_w;

    let th_begin = grid_ceil(max_i32(-((s_bez_h * gcd_scale_h * sh) as i32), ((d_bez_h * gcd_scale_h) as i32 - kh as i32 + 1) * dh as i32), th_step as i32) as usize;
    let th_end = th_begin + 1;
    //let th_end = min_usize((oh - s_bez_h * gcd_scale_h) * sh, (d_bez_h * gcd_scale_h + 1) * dh);
    let tw_begin = grid_ceil(max_i32(-((s_bez_w * gcd_scale_w * sw) as i32), ((d_bez_w * gcd_scale_w) as i32 - kw as i32 + 1) * dw as i32), tw_step as i32) as usize;
    let tw_end = tw_begin + 1;
    //let tw_end = min_usize((ow - s_bez_w * gcd_scale_w) * sw, (d_bez_w * gcd_scale_w + 1) * dw);

    if bid < bs { if cid < ic { if hidx < ih { if widx < iw {
        let mut acc = 0.;
        let mut th = th_begin;
        while th <= th_end {
            let mut tw = tw_begin;
            while tw <= tw_end {
                let hid = th / sh + s_bez_h * gcd_scale_h;
                let wid = tw / sw + s_bez_w * gcd_scale_w;
                if hid < oh { if wid < ow {
                    acc += 1. + 0. * dy[bidy + cidy + hid * ow + wid];
                }}
                tw += tw_step;
            }
            th += th_step;
        }
        dx[bidx + cidx + hidx * iw + widx] = acc;
    }}}}
}
