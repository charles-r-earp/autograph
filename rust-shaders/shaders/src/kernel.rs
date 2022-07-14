use spirv_std::glam::UVec3;
use crate::{
    autobind,
    half::bf16x2,
    util::{Load, Store, group_barrier},
};
use crunchy::unroll;

#[repr(C)]
pub struct Im2ColPushConsts2 {
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

/*
// Adapted from https://github.com/CNugteren/CLBlast/blob/master/src/kernels/levelx/im2col.opencl
fn im2col_2d<T>(kernel_flip: bool, group_id: UVec3, local_id: UVec3, x: &[T], y: &mut [T], push_consts: &Im2ColPushConsts2)
    where T: Copy, [T]: Store<f32> {
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
    let group_id = group_id.x as usize;
    let groups_h = oh / 16 + if oh % 16 != 0 { 1 } else { 0 };
    let groups_w = ow / 16 + if ow % 16 != 0 { 1 } else { 0 };
    let groups_hw = groups_h * groups_w;
    let group_bc = group_id / groups_hw;
    let group_hw = group_id % groups_hw;
    let group_h = group_hw / groups_w;
    let group_w = group_hw % groups_w;
    let local_id = local_id.x as usize;
    let local_h = local_id / 16;
    let local_w = local_id % 16;
    let bid = group_bc / ic;
    let cid = group_bc % ic;
    let bidx = bid * ic * ih * iw;
    let bidy = bid * oh * ow * ic * kh * kw;
    let cidx = cid * ih * iw;
    let cidy = cid * kh * kw;
    let hid = group_h * 16 + local_h;
    let wid = group_w * 16 + local_w;
    let patch_idx = (hid * ow + wid) * ic * kh * kw;
    if bid < bs { if cid < ic { if hid < oh { if wid < ow {
        for ki in 0 .. kh {
            for kj in 0 .. kw {
                let hidx = -(ph as isize) + (ki * dh + sh * hid) as isize;
                let widx = -(pw as isize) + (kj * dw + sw * wid) as isize;
                let mut val = 0.;
                if hidx >= 0 { if hidx < ih as isize { if widx >= 0 { if widx < iw as isize {
                    val = x.load(bidx + cidx + (hidx as usize) * iw + widx as usize);
                }}}}
                let mut kidx = ki * kw + kj;
                if kernel_flip {
                    kidx = kh * kw - (kidx + 1);
                }
                y.store(bidy + patch_idx + cidy + kidx, val);
            }
        }
    }}}}
}*/

// Adapted from https://github.com/ROCmSoftwarePlatform/MIOpen/blob/develop/src/kernels/MIOpenIm2d2Col.cl
fn im2col_2d<T>(global_id: UVec3, x: &[T], y: &mut [T], push_consts: &Im2ColPushConsts2)
    where T: Copy, [T]: Store<f32> {
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
    let global_id = global_id.x as usize;
    if global_id < bs * oh * ow * ic * kh * kw {
        let bcid = global_id / (oh * ow * kh * kw);
        let hwkid = global_id % (oh * ow * kh * kw);
        let bid = bcid / ic;
        let cid = bcid % ic;
        let hwid = hwkid / (kh * kw);
        let kid = hwkid % (kh * kw);
        let hid = hwid / ow;
        let wid = hwid % ow;
        let ki = kid / kw;
        let kj = kid % kw;
        let hidx = -(ph as i32) + (ki * dh + sh * hid) as i32;
        let widx = -(pw as i32) + (kj * dw + sw * wid) as i32;
        let val = if hidx >= 0 && hidx < ih as i32 && widx >= 0 && widx < iw as i32 {
            x.load(bcid * ih * iw + hidx as usize * iw + widx as usize)
        } else {
            0.
        };
        let y_idx = bid * oh * ow * ic * kh * kw + hid * ow * ic * kh * kw + wid * ic * kh * kw + cid * kh * kw + kid;
        y.store(y_idx, val);
    }
}


#[autobind]
#[spirv(compute(threads(1024)))]
pub fn im2col_2d_convolution_f32_1024(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer)] x: &[f32],
    #[spirv(storage_buffer)] y: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &Im2ColPushConsts2,
) {
    im2col_2d(global_id, x, y, push_consts);
}

#[autobind]
#[spirv(compute(threads(1024)))]
pub fn im2col_2d_convolution_f32_64(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer)] x: &[f32],
    #[spirv(storage_buffer)] y: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &Im2ColPushConsts2,
) {
    im2col_2d(global_id, x, y, push_consts);
}

#[autobind]
#[spirv(compute(threads(256)))]
pub fn im2col_2d_convolution_bf16(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer)] x: &[bf16x2],
    #[spirv(storage_buffer)] y: &mut [bf16x2],
    #[spirv(push_constant)]
    push_consts: &Im2ColPushConsts2,
) {
    im2col_2d(global_id, x, y, push_consts);
}

#[autobind]
#[spirv(compute(threads(256)))]
pub fn im2col_2d_convolution_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer)] x: &[f32],
    #[spirv(storage_buffer)] y: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &Im2ColPushConsts2,
) {
    im2col_2d(global_id, x, y, push_consts);
}
/*
// Adapted from https://github.com/ROCmSoftwarePlatform/MIOpen/blob/develop/src/kernels/MIOpenIm2d2Col.cl
fn im2col_2d_v2<T, const N: usize>(group_id: UVec3, local_id: UVec3, x: &[T], x_tile: &mut [f32; N], y: &mut [T], push_consts: &Im2ColPushConsts2)
    where T: Copy + Default, [T]: Store<f32> {
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
    let group_id = group_id.x as usize;
    let local_id = local_id.x as usize;
    let global_id = group_id * 256 + local_id;

    let start = {
        let get_x_index = |global_index| {
            let bcid = global_index / (oh * ow * kh * kw);
            let hwkid = global_index % (oh * ow * kh * kw);
            //let bid = bcid / ic;
            //let cid = bcid % ic;
            let hwid = hwkid / (kh * kw);
            let kid = hwkid % (kh * kw);
            let hid = hwid / ow;
            let wid = hwid % ow;
            let ki = kid / kw;
            let kj = kid % kw;

            let hidx = -(ph as i32) + (ki * dh + sh * hid) as i32;
            let hidx = if hidx < 0 {
                0
            } else if hidx as usize >= oh {
                oh - 1
            } else {
                hidx as usize
            };
            let widx = -(pw as i32) + (kj * dw + sw * wid) as i32;
            let widx = if widx < 0 {
                0
            } else if widx as usize >= oh {
                ow - 1
            } else {
                widx as usize
            };
            bcid * ih * iw + hidx * ow + widx
        };
        let start = get_x_index(group_id * 256);
        let end = get_x_index(group_id * 256 + 255);
        let mut index = 0;
        while start + index <= end {
            x_tile[index + local_id] = x.load(start + index + local_id);
            index += 256;
        }
        start
    };

    if global_id < bs * oh * ow * ic * kh * kw {
        let bcid = global_id / (oh * ow * kh * kw);
        let hwkid = global_id % (oh * ow * kh * kw);
        let bid = bcid / ic;
        let cid = bcid % ic;
        let hwid = hwkid / (kh * kw);
        let kid = hwkid % (kh * kw);
        let hid = hwid / ow;
        let wid = hwid % ow;
        let ki = kid / kw;
        let kj = kid % kw;
        let hidx = -(ph as i32) + (ki * dh + sh * hid) as i32;
        let widx = -(pw as i32) + (kj * dw + sw * wid) as i32;
        let val = if hidx >= 0 && hidx < ih as i32 && widx >= 0 && widx < iw as i32 {
            let index = bcid * ih * iw + hidx as usize * iw + widx as usize;
            x_tile[index - start]
        } else {
            0.
        };
        let y_idx = bid * oh * ow * ic * kh * kw + hid * ow * ic * kh * kw + wid * ic * kh * kw + cid * kh * kw + kid;
        y.store(y_idx, val);
    }
}

#[autobind]
#[spirv(compute(threads(256)))]
pub fn im2col_2d_convolution_f32_v2(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer)] x: &[f32],
    #[spirv(workgroup)] x_tile: &mut [f32; 32*32],
    #[spirv(storage_buffer)] y: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &Im2ColPushConsts2,
) {
    im2col_2d_v2(group_id, local_id, x, x_tile, y, push_consts);
}
*/
#[repr(C)]
pub struct Col2ImPushConsts2 {
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

// adapted from https://github.com/CNugteren/CLBlast/blob/master/src/kernels/levelx/col2im.opencl
fn col2im_2d_f32_impl(kernel_flip: bool, global_id: UVec3, x: &[f32], y: &mut [f32], push_consts: &Col2ImPushConsts2) {
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
    let bidx = bid * ih * iw * ic * kh * kw;
    let bidy = bid * ic * oh * ow;
    let cidx = cid * kh * kw;
    let cidy = cid * oh * ow;
    let ow_scaled = (ow - 1) / gcd_w + 1;
    let gcd_scale_h = global_y / ow_scaled + (ph - 1) / gcd_h + 1;
    let gcd_scale_w = global_y % ow_scaled + (pw - 1) / gcd_w + 1;
    let hidy = gcd_scale_h * gcd_h - ph;
    let widy = gcd_scale_w * gcd_w - pw;
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
    let th_end = min_usize((ih - s_bez_h * gcd_scale_h) * sh, (d_bez_h * gcd_scale_h + 1) * dh);
    let tw_begin = grid_ceil(max_i32(-((s_bez_w * gcd_scale_w * sw) as i32), ((d_bez_w * gcd_scale_w - kw + 1) * dw) as i32), tw_step as i32) as usize;
    let tw_end = min_usize((iw - s_bez_w * gcd_scale_w) * sw, (d_bez_w * gcd_scale_w + 1) * dw);

    if bid < bs { if cid < ic { if hidy < oh { if widy < ow {
        let mut acc = 0.;
        let mut th = th_begin;
        while th < th_end {
            let mut tw = tw_begin;
            while tw < tw_end {
                let khid = d_bez_h * gcd_scale_h - th / dh;
                let kwid = d_bez_w * gcd_scale_w - tw / dw;
                let hid = th / sh + s_bez_h * gcd_scale_h;
                let wid = tw / sw + s_bez_w * gcd_scale_w;
                let mut kidx = khid * kw + kwid;
                if kernel_flip {
                    kidx = kh * kw - (kidx + 1);
                }
                let patch_idx = (hid * iw + wid) * ic * kh * kw;
                acc += x[bidx + patch_idx + cidx + kidx];
                tw += tw_step;
            }
            th += th_step;
        }
        y[bidy + cidy + hidy * ow + widy] = acc;
    }}}}
}

#[allow(unused_attributes)]
#[spirv(compute(threads(16, 16)))]
pub fn col2im_2d_convolution_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] x: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] y: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &Col2ImPushConsts2,
) {
    col2im_2d_f32_impl(false, global_id, x, y, push_consts);
}


#[repr(C)]
pub struct ConvDirect2dPushConsts {
    bs: u32,
    ic: u32,
    ih: u32,
    iw: u32,
    oc: u32,
    oh: u32,
    ow: u32,
    /*kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,*/
}

// TOOD impl strides / padding / dilation
// limited to known shared memory size 16 x 16, ie kernel must be < 16 x 16 as well
// computes 1 output per thread
#[autobind]
#[spirv(compute(threads(256)))]
pub fn conv_direct_5x5_f32(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer)] x: &[f32],
    #[spirv(workgroup)] x_tile: &mut [[f32; 20 + 1]; 20],
    #[spirv(storage_buffer)] w: &[f32],
    #[spirv(workgroup)] w_tile: &mut [[f32; 16 + 1]; 16],
    #[spirv(storage_buffer)] y: &mut [f32],
    #[spirv(push_constant)] push_consts: &ConvDirect2dPushConsts,
) {
    let ic = push_consts.ic as usize;
    let ih = push_consts.ih as usize;
    let iw = push_consts.iw as usize;
    let oc = push_consts.oc as usize;
    let oh = push_consts.oh as usize;
    let ow = push_consts.ow as usize;
    let kh = 5;
    let kw = 5;

    let group_id = group_id.x as usize;
    let group_rows = oh / 16 + if oh % 16 != 0 { 1 } else { 0 };
    let group_cols = ow / 16 + if ow % 16 != 0 { 1 } else { 0 };
    let groups_xy = group_rows * group_cols;
    let group_bc = group_id / groups_xy;
    let group_b = group_bc / oc;
    let group_oc = group_bc % oc;
    let group_xy = group_id % groups_xy;
    let group_row = group_xy / group_cols;
    let group_col = group_xy % group_cols;

    let local_id = local_id.x as usize;
    let local_row = local_id / 16;
    let local_col = local_id % 16;

    let x_idx = group_b * ic * ih * iw;
    let w_idx = group_oc * ic * kh * kw;

    let mut x_micro = <[[f32; 5]; 5]>::default();
    let mut w_micro = <[[f32; 5]; 5]>::default();
    let mut y_micro = 0.;

    for ic_idx in 0 .. ic {
        unroll! { for i in 0 .. 2 {
            unroll! { for j in 0 .. 2 {
                let row_start = i * 16;
                let col_start = j * 16;
                let tile_row = row_start + local_row;
                let tile_col = col_start + local_col;
                let row = group_row * 16 + tile_row;
                let col = group_col * 16 + tile_col;
                if tile_row < 20 { if tile_col < 20 {
                    x_tile[tile_row][tile_col] = if row < ih {
                        if col < iw {
                            x[x_idx + ic_idx * ih * iw + row * iw + col]
                        } else {
                            0.
                        }
                    } else {
                        0.
                    };
                }}
            }}
        }}
        w_tile[local_row][local_col] = if local_row < kh {
            if local_col < kw {
                w[w_idx + ic_idx * kh * kw + local_row * kw + local_col]
            } else {
                0.
            }
        } else {
            0.
        };
        group_barrier();
        unroll! { for ki in 0 .. 5 {
            unroll! { for kj in 0 .. 5 {
                x_micro[ki][kj] = x_tile[ki + local_row][kj + local_col];
            }}
        }}
        unroll! { for ki in 0 .. 5 {
            unroll! { for kj in 0 .. 5 {
                w_micro[ki][kj] = w_tile[ki][kj];
            }}
        }}
        unroll! { for ki in 0 .. 5 {
            unroll! { for kj in 0 .. 5 {
                y_micro += x_micro[ki][kj] * w_micro[ki][kj];
            }}
        }}
        group_barrier();
    }

    let global_row = group_row * 16 + local_row;
    let global_col = group_col * 16 + local_col;
    let y_idx = group_b * oc * oh * ow + group_oc * oh * ow + global_row * ow + global_col;
    if global_row < oh { if global_col < ow {
        y[y_idx] = y_micro;
    }}
}

/*
#[autobind]
#[spirv(compute(threads(1, 256)))]
pub fn conv_direct_backward_5x5_f32(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer)] dx: &mut [AtomicF32],
    #[spirv(storage_buffer)] w: &[[[f32; 5]; 5]],
    #[spirv(workgroup)] w_tile: &mut [[f32; 5]; 5],
    #[spirv(storage_buffer)] dy: &[f32],
    #[spirv(push_constant)] push_consts: &ConvDirect2dPushConsts,
) {
    let ic = push_consts.ic as usize;
    let ih = push_consts.ih as usize;
    let iw = push_consts.iw as usize;
    let oc = push_consts.oc as usize;
    let oh = push_consts.oh as usize;
    let ow = push_consts.ow as usize;
    let kh = 5;
    let kw = 5;

    let group_id = group_id.x as usize;
    let group_rows = oh / 16 + if oh % 16 != 0 { 1 } else { 0 };
    let group_cols = ow / 16 + if ow % 16 != 0 { 1 } else { 0 };
    let groups_xy = group_rows * group_cols;
    let group_bc = group_id / groups_xy;
    let group_b = group_bc / (ic * oc);
    let group_c = group_bc % (ic * oc);
    let group_ic = group_c / oc;
    let group_oc = group_c % oc;
    let group_xy = group_id % groups_xy;
    let group_row = group_xy / group_cols;
    let group_col = group_xy % group_cols;

    let local_id = local_id.x as usize;
    let local_row = local_id / 16;
    let local_col = local_id % 16;

    let x_idx = group_b * ic * ih * iw + group_ic * ih * iw;
    let w_idx = group_oc * ic + group_ic;

    let mut dx_micro = <[[f32; 5]; 5]>::default();
    let mut w_micro = <[[f32; 5]; 5]>::default();
    let mut dy_micro = 0.;

    w_tile[local_row][local_col] = if local_row < kh && local_col < kw {
        w[w_idx][local_row][local_col]
    } else {
        0.
    };
    let global_row = group_row * 16 + local_row;
    let global_col = group_col * 16 + local_col;
    let y_idx = group_b * oc * oh * ow + group_oc * oh * ow + global_row * ow + global_col;
    if global_row < oh { if global_col < ow {
        dy_micro = dy[y_idx];
    }}
    group_barrier();
    unroll! { for ki in 0 .. 5 {
        unroll! { for kj in 0 .. 5 {
            w_micro[ki][kj] = w_tile[ki][kj];
        }}
    }}

    unroll! { for ki in 0 .. 5 {
        unroll! { for kj in 0 .. 5 {
            dx_micro[ki][kj] += dy_micro * w_micro[ki][kj];
        }}
    }}
    unroll! { for ki in 0 .. 5 {
        unroll! { for kj in 0 .. 5 {
            let row = global_row + ki;
            let col = global_col + kj;
            if row < ih && row < iw {
                let x_idx = x_idx + row * iw + col;
                dx[x_idx] += dx_micro[ki][kj];
            }
        }}
    }}
}
*/

#[autobind]
#[spirv(compute(threads(1, 256)))]
pub fn conv_direct_backward_5x5_f32(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer)] dx: &mut [[[f32; 5]; 5]],
    #[spirv(storage_buffer)] w: &[[[f32; 5]; 5]],
    #[spirv(workgroup)] dy_tile: &mut [[f32; 32 + 1]; 8],
    #[spirv(storage_buffer)] dy: &[f32],
    #[spirv(push_constant)] push_consts: &ConvDirect2dPushConsts,
) {
    let ic = push_consts.ic as usize;
    let ih = push_consts.ih as usize;
    let iw = push_consts.iw as usize;
    let oc = push_consts.oc as usize;
    let oh = push_consts.oh as usize;
    let ow = push_consts.ow as usize;
    //let kh = 5;
    //let kw = 5;

    let group_id = group_id.x as usize;
    let group_b = group_id / ic;
    let group_ic = group_id % ic;

    let local_id = local_id.y as usize;
    let local_row = local_id / 32;
    let local_col = local_id % 32;

    let x_idx = group_b * ic * ih * iw + group_ic * ih * iw;

    let mut dx_micro = <[[f32; 5]; 5]>::default();
    let mut w_micro = <[[f32; 5]; 5]>::default();
    let mut dy_micro: f32;

    let mut row = 0;
    while row < oh {
        let mut col = 0;
        while col < ow {
            let global_row = row + local_row;
            let global_col = col + local_col;
            for oc_idx in 0 .. oc {
                unroll! { for ki in 0 .. 5 {
                    unroll! { for kj in 0 .. 5 {
                        w_micro[ki][kj] = w[oc_idx * ic + group_ic][ki][kj];
                    }}
                }}
                group_barrier();
                dy_tile[local_row][local_col] = if global_row < oh && global_col < ow {
                    dy[group_b * oc * oh * ow + oc_idx * oh * ow + global_row * ow + global_col]
                } else {
                    0.
                };
                group_barrier();
                dy_micro = dy_tile[local_row][local_col];
                unroll! { for ki in 0 .. 5 {
                    unroll! { for kj in 0 .. 5 {
                        dx_micro[ki][kj] += dy_micro * w_micro[ki][kj];
                    }}
                }}
            }
            unroll! { for ki in 0 .. 5 {
                unroll! { for kj in 0 .. 5 {
                    let x_row = global_row + ki;
                    let x_col = global_col + kj;
                    if x_row < ih && x_col < iw {
                        dx[x_idx + x_row * iw + x_col][ki][kj] = dx_micro[ki][kj];
                    }
                }}
            }}
            col += 32;
        }
        row += 8;
    }
}

#[autobind]
#[spirv(compute(threads(16, 16, 1)))]
pub fn conv_direct_backward_weight_5x5_f32(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer)] x: &[f32],
    #[spirv(workgroup)] x_tile: &mut [[f32; 32 + 5]; 16],
    #[spirv(storage_buffer)] dw: &mut [f32],
    #[spirv(storage_buffer)] dy: &[f32],
    #[spirv(push_constant)] push_consts: &ConvDirect2dPushConsts,
) {
    let bs = push_consts.bs as usize;
    let ic = push_consts.ic as usize;
    let ih = push_consts.ih as usize;
    let iw = push_consts.iw as usize;
    let oc = push_consts.oc as usize;
    let oh = push_consts.oh as usize;
    let ow = push_consts.ow as usize;
    let group_b = group_id.x as usize;
    let local_b = local_id.x as usize;
    let group_oc = group_id.y as usize;
    let group_ic = group_id.z as usize;
    let local_oc = local_id.y as usize;
    let global_b = group_b * 16 + local_b;
    let global_oc = group_oc * 16 + local_oc;
    let global_ic = group_ic;

    let mut x_micro = <[f32; 5]>::default();
    let mut dw_micro = <[[f32; 5]; 5]>::default();
    let mut dy_micro = <[f32; 5]>::default();

    x_tile[local_b][local_oc + 5] = 0f32;

    for row in 0 .. ih {
        let mut col = 0;
        while col < iw {
            group_barrier();
            unroll! { for u in 0 .. 2 {
                let local_col = local_oc + u * 16;
                let global_col = col + local_col;
                x_tile[local_b][local_col] = if global_b < bs && global_col < iw {
                    x[global_b * ic * ih * iw + global_ic * ih * iw + row * iw + global_col]
                } else {
                    0f32
                };
            }}
            group_barrier();
            if global_b < bs && global_oc < oc {
                let ncols = if col + 32 <= iw { 32 } else { ih - col };
                for j in 0 .. ncols {
                    unroll! { for kj in 0 .. 5 {
                        x_micro[kj] = x_tile[local_b][j + kj];
                    }}
                    unroll! { for ki in 0 .. 5 {
                        dy_micro[ki] = if ki <= row && row - ki < oh && col + j < ow {
                            dy[group_b * oc * oh * ow + global_oc * oh * ow + (row - ki) * ow + (col + j)]
                        } else {
                            0f32
                        };
                    }}
                    unroll! { for ki in 0 .. 5 {
                        unroll! { for kj in 0 .. 5 {
                            dw_micro[ki][kj] += dy_micro[ki] * x_micro[kj];
                        }}
                    }}
                }
            }
            col += 32;
        }
    }
    if global_b < bs && global_oc < oc {
        unroll! { for ki in 0 .. 5 {
            unroll! { for kj in 0 .. 5 {
                dw[global_oc * ic * 25 * bs + global_ic * 25 * bs + ki * 5 * bs + kj * bs + global_b] = dw_micro[ki][kj];
            }}
        }}
    }
}

#[repr(C)]
pub struct ReducePushConsts {
    bs: u32,
    n: u32,
    alpha: f32,
}

#[autobind]
#[spirv(compute(threads(128)))]
pub fn reduce_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer)] x: &[f32],
    #[spirv(storage_buffer)] y: &mut [f32],
    #[spirv(push_constant)] push_consts: &ReducePushConsts,
) {
    let bs = push_consts.bs as usize;
    let n = push_consts.n as usize;
    let alpha = push_consts.alpha;
    let global_id = global_id.x as usize;
    if global_id < bs {
        let mut acc = 0f32;
        for i in 0 .. n {
            acc += alpha * x[global_id * n + i];
        }
        y[global_id] += acc;
    }
}
