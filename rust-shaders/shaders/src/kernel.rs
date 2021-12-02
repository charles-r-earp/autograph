use spirv_std::glam::UVec3;

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

// Adapted from https://github.com/CNugteren/CLBlast/blob/master/src/kernels/levelx/im2col.opencl
fn im2col_2d_f32<const TX: usize, const TY: usize>(kernel_flip: bool, group_id: UVec3, local_id: UVec3, x: &[f32], y: &mut [f32], push_consts: &Im2ColPushConsts2) {
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
    let group_x = group_id.x as usize;
    let group_y = group_id.y as usize;
    let local_x = local_id.x as usize;
    let local_y = local_id.y as usize;
    let global_x = group_x * TX + local_x;
    let global_y = group_y * TY + local_y;
    let bid = global_x / ic;
    let cid = global_x % ic;
    let bidx = bid * ic * ih * iw;
    let bidy = bid * oh * ow * ic * kh * kw;
    let cidx = cid * ih * iw;
    let cidy = cid * kh * kw;
    let hid = global_y / ow;
    let wid = global_y % ow;
    let patch_idx = (hid * ow + wid) * ic * kh * kw;
    if bid < bs { if cid < ic { if hid < oh { if wid < ow {
        for ki in 0 .. kh {
            for kj in 0 .. kw {
                let hidx = -(ph as isize) + (ki * dh + sh * hid) as isize;
                let widx = -(pw as isize) + (kj * dw + sw * wid) as isize;
                let mut val = 0.;
                if hidx >= 0 { if hidx < ih as isize { if widx >= 0 { if widx < iw as isize {
                    val = x[bidx + cidx + (hidx as usize) * iw + widx as usize];
                }}}}
                let mut kidx = ki * kw + kj;
                if kernel_flip {
                    kidx = kh * kw - (kidx + 1);
                }
                y[bidy + patch_idx + cidy + kidx] = val;
            }}
        }}
    }}
}

#[allow(unused_attributes)]
#[spirv(compute(threads(8, 8)))]
pub fn im2col_2d_convolution_f32(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] x: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] y: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &Im2ColPushConsts2,
) {
    im2col_2d_f32::<8, 8>(false, group_id, local_id, x, y, push_consts);
}

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
