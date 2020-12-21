use super::{Num, Scalar, TensorView1, TensorView2, TensorViewMut2};
use crate::error::ShapeError;
use crate::util::{size_eq, type_eq};
use crate::Result;
use bytemuck::{Pod, Zeroable};
use half::bf16;
use std::convert::TryInto;

enum PostOp {
    Identity,
    #[allow(unused)]
    Relu,
}

#[derive(Copy)]
#[repr(C, packed)]
struct GemmPushConsts<T: Num> {
    alpha: T,
    beta: T,
    a0: T, // ie relu negative slope
    m: u32,
    k: u32,
    n: u32,
    rsa: i32,
    csa: i32,
    rsb: i32,
    csb: i32,
    rsc: i32,
    csc: i32,
}

impl<T: Num> Clone for GemmPushConsts<T> {
    fn clone(&self) -> Self {
        *self
    }
}

unsafe impl<T: Num> Zeroable for GemmPushConsts<T> {}

unsafe impl<T: Num> Pod for GemmPushConsts<T> {}

pub fn gemm<T: Num>(
    alpha: T,
    a: &TensorView2<T>,
    b: &TensorView2<T>,
    beta: T,
    c: &mut TensorViewMut2<T>,
) -> Result<()> {
    gemm_impl(alpha, a, b, None, PostOp::Identity, T::zero(), beta, c)
}

#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
fn gemm_impl<T: Num>(
    alpha: T,
    a: &TensorView2<T>,
    b: &TensorView2<T>,
    bias: Option<TensorView1<T>>,
    post_op: PostOp,
    a0: T,
    beta: T,
    c: &mut TensorViewMut2<T>,
) -> Result<()> {
    let device = a.device();

    let src = if type_eq::<T, bf16>() {
        match post_op {
            PostOp::Identity => {
                if bias.is_some() {
                    include_shader!("glsl/gemm_bias_bf16_as_f32.spv")
                } else {
                    include_shader!("glsl/gemm_bf16_as_f32.spv")
                }
            }
            PostOp::Relu => {
                if bias.is_some() {
                    include_shader!("glsl/gemm_bias_relu_bf16_as_f32.spv")
                } else {
                    include_shader!("glsl/gemm_relu_bf16_as_f32.spv")
                }
            }
        }
    } else if type_eq::<T, f32>() {
        match post_op {
            PostOp::Identity => {
                if bias.is_some() {
                    include_shader!("glsl/gemm_bias_f32.spv")
                } else {
                    include_shader!("glsl/gemm_f32.spv")
                }
            }
            PostOp::Relu => {
                if bias.is_some() {
                    include_shader!("glsl/gemm_bias_relu_f32.spv")
                } else {
                    include_shader!("glsl/gemm_relu_f32.spv")
                }
            }
        }
    } else if type_eq::<T, u32>() {
        include_shader!("glsl/gemm_u32.spv")
    } else if type_eq::<T, i32>() {
        include_shader!("glsl/gemm_i32.spv")
    } else if type_eq::<T, u64>() {
        include_shader!("glsl/gemm_u64.spv")
    } else if type_eq::<T, i64>() {
        include_shader!("glsl/gemm_i64.spv")
    } else if type_eq::<T, f64>() {
        include_shader!("glsl/gemm_f64.spv")
    } else {
        unreachable!()
    };

    let (m, k) = a.dim();
    let (k2, n) = b.dim();
    let (m2, n2) = c.dim();

    if m != m2 || k != k2 || n != n2 {
        return Err(ShapeError::IncompatibleShape.into());
    }

    let m = m as u32;
    let k = k as u32;
    let n = n as u32;

    let [rsa, csa]: [isize; 2] = a.strides().try_into().unwrap();
    let [rsa, csa] = [rsa as i32, csa as i32];

    let [rsb, csb]: [isize; 2] = b.strides().try_into().unwrap();
    let [rsb, csb] = [rsb as i32, csb as i32];

    let [rsc, csc]: [isize; 2] = c.strides().try_into().unwrap();
    let [rsc, csc] = [rsc as i32, csc as i32];

    let builder = device
        .compute_pass(src, "main")?
        .buffer_slice(a.as_buffer_slice())?
        .buffer_slice(b.as_buffer_slice())?
        .option_buffer_slice(bias.as_ref().map(|bias| bias.as_buffer_slice()))?
        .buffer_slice_mut(c.as_buffer_slice_mut())?
        .global_size([m, n, 1]);

    if size_eq::<T, u64>() {
        let push_consts = GemmPushConsts {
            alpha: alpha.to_bits_u64().unwrap(),
            beta: beta.to_bits_u64().unwrap(),
            a0: a0.to_bits_u64().unwrap(),
            m,
            k,
            n,
            rsa,
            csa,
            rsb,
            csb,
            rsc,
            csc,
        };
        builder.push_constants(push_consts)?.enqueue()
    } else {
        let (alpha, beta, a0) = if type_eq::<T, bf16>() {
            (
                alpha.to_f32().unwrap().to_bits_u32().unwrap(),
                beta.to_f32().unwrap().to_bits_u32().unwrap(),
                a0.to_f32().unwrap().to_bits_u32().unwrap(),
            )
        } else {
            (
                alpha.to_bits_u32().unwrap(),
                beta.to_bits_u32().unwrap(),
                a0.to_bits_u32().unwrap(),
            )
        };
        let push_consts = GemmPushConsts {
            alpha,
            beta,
            a0,
            m,
            k,
            n,
            rsa,
            csa,
            rsb,
            csb,
            rsc,
            csc,
        };
        builder.push_constants(push_consts)?.enqueue()
    }
}
