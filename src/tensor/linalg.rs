use super::{TensorView1, TensorView2, TensorViewMut2};
use crate::error::ShapeError;
use crate::Result;
use bytemuck::{Pod, Zeroable};
use num_traits::{One, Zero};
use std::any::TypeId;
use std::convert::TryInto;

mod sealed {
    pub trait Sealed {}
}

pub trait Scalar: sealed::Sealed + Copy + Zeroable + Pod + Zero + One + 'static {}

impl sealed::Sealed for f32 {}

impl Scalar for f32 {}

impl sealed::Sealed for f64 {}

impl Scalar for f64 {}

impl sealed::Sealed for i32 {}

impl Scalar for i32 {}

enum PostOp {
    Identity,
    #[allow(unused)]
    Relu,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct GemmPushConsts<T: Scalar> {
    m: u32,
    k: u32,
    n: u32,
    rsa: i32,
    csa: i32,
    rsb: i32,
    csb: i32,
    rsc: i32,
    csc: i32,
    pad: u32,
    alpha: T,
    beta: T,
    a0: T, // ie relu negative slope
}

unsafe impl<T: Scalar> Zeroable for GemmPushConsts<T> {}

unsafe impl<T: Scalar> Pod for GemmPushConsts<T> {}

// For neural networks this should support f32 and eventually bf16
// For general purpose there's no reason not to support f64, u32, i32, u64, i64 as well, potentially f16
pub fn gemm<T: Scalar>(
    alpha: T,
    a: &TensorView2<T>,
    b: &TensorView2<T>,
    beta: T,
    c: &mut TensorViewMut2<T>,
) -> Result<()> {
    gemm_impl(alpha, a, b, None, PostOp::Identity, T::zero(), beta, c)
}

#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
fn gemm_impl<T: Scalar>(
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
    
    let src = if TypeId::of::<T>() == TypeId::of::<f32>() {
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
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        include_shader!("glsl/gemm_f64.spv")
    } else if TypeId::of::<T>() == TypeId::of::<i32>() {
        include_shader!("glsl/gemm_i32.spv")
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

    let push_consts = GemmPushConsts {
        m,
        k,
        n,
        rsa,
        csa,
        rsb,
        csb,
        rsc,
        csc,
        pad: 0,
        alpha,
        beta,
        a0,
    };

    device
        .compute_pass(src, "main")?
        .buffer_slice(a.as_buffer_slice())?
        .buffer_slice(b.as_buffer_slice())?
        .option_buffer_slice(bias.as_ref().map(|bias| bias.as_buffer_slice()))?
        .buffer_slice_mut(c.as_buffer_slice_mut())?
        .push_constants(push_consts)?
        .global_size([m, n, 1])
        .enqueue()
}
