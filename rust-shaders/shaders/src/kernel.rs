use spirv_std::glam::UVec3;

#[repr(C)]
pub struct KernelPushConsts2 {
    channels: u32,
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
#[allow(unused)]
fn im2col_2d_f32_impl(kernel_flip: bool, global_id: UVec3, x: &[f32], y: &mut [f32], push_consts: &KernelPushConsts2) {
    let channels = push_consts.channels as usize;
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
    let wid = global_id.x as usize;
    let global_y = global_id.y as usize;
    let hid = global_y % ih;
    let cid = global_y / ih;
    if hid < oh { if wid < ow { if cid < channels {
        for ki in 0 .. kh {
            for kj in 0 .. kw {
                let hidx = -(ph as isize) + (ki * dh + sh * hid) as isize;
                let widx = -(pw as isize) + (kj * dw + sw * wid) as isize;
                let mut val = 0.;
                if hidx >= 0 { if hidx < ih as isize { if widx >= 0 { if widx > iw as isize {
                    val = x[widx as usize + iw * (hidx as usize + ih * cid)];
                }}}}
                let kidx = if kernel_flip {
                    kh * kw - kj - kw * ki - 1
                } else {
                    kj + kw * ki
                };
                let patch_idx = wid + ow * hid;
                let output_idx = patch_idx + kidx * ow * oh + cid * ow * oh * kh * kw;
                y[output_idx] = val;
            }
        }
    }}}
}

#[allow(unused_attributes)]
#[spirv(compute(threads(8, 8)))]
pub fn im2col_2d_convolution_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] x: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] y: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &KernelPushConsts2,
) {
    im2col_2d_f32_impl(true, global_id, x, y, push_consts);
}
