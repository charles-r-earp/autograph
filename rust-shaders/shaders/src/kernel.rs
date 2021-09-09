use spirv_std::glam::UVec3;

#[repr(C)]
pub struct KernelPushConsts<A> {
    channels: u32,
    input_dim: A,
    output_dim: A,
    kernel: A,
    strides: A,
    padding: A,
    dilation: A,
}

type KernelPushConsts2 = KernelPushConsts<[u32; 2]>;

// Adapted from https://github.com/CNugteren/CLBlast/blob/master/src/kernels/levelx/im2col.opencl
#[allow(unused)]
fn im2col_2d_f32_impl(kernel_flip: bool, global_id: UVec3, x: &[f32], y: &mut [f32], push_consts: &KernelPushConsts2) {
    let KernelPushConsts {
        channels,
        input_dim,
        output_dim,
        kernel,
        strides,
        padding,
        dilation
    } = push_consts;
    let channels = *channels as usize;
    let ih = input_dim[0] as usize;
    let iw = input_dim[1] as usize;
    let oh = output_dim[0] as usize;
    let ow = output_dim[1] as usize;
    let kh = kernel[0] as usize;
    let kw = kernel[1] as usize;
    let sh = strides[0] as usize;
    let sw = strides[1] as usize;
    let ph = padding[0] as usize;
    let pw = padding[1] as usize;
    let dh = dilation[0] as usize;
    let dw = dilation[1] as usize;
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
