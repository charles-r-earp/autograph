use crate::autobind;
use half::bf16;
use spirv_std::glam::UVec3;
use num_traits::Float;
use paste::paste;
use core::ops::AddAssign;

#[repr(C)]
pub struct ReluPushConsts {
    n: u32,
}

fn relu<T: Float>(
    gid: usize,
    x: &[T],
    y: &mut [T],
    push_consts: &ReluPushConsts,
) {
    let n = push_consts.n as usize;
    if gid < n {
        let x = x[gid];
        y[gid] = if x > T::zero() { x } else { T::zero() };
    }
}

fn relu_backward<T: Float + AddAssign<T>>(
    gid: usize,
    x: &[T],
    dx: &mut [T],
    dy: &[T],
    push_consts: &ReluPushConsts,
) {
    let n = push_consts.n as usize;
    if gid < n {
        dx[gid] += if x[gid] > T::zero() {
            dy[gid]
        } else {
            T::zero()
        };
    }
}

macro_rules! impl_relu {
    ($($t:ty),*) => (
        paste! {
            $(
                #[autobind]
                #[spirv(compute(threads(256)))]
                pub fn [<relu_ $t>](
                    #[spirv(global_invocation_id)]
                    global_id: UVec3,
                    #[spirv(storage_buffer)] x: &[$t],
                    #[spirv(storage_buffer)] y: &mut [$t],
                    #[spirv(push_constant)]
                    push_consts: &ReluPushConsts,
                ) {
                    relu(global_id.x as usize, x, y, push_consts);
                }

                #[autobind]
                #[spirv(compute(threads(256)))]
                pub fn [<relu_backward_ $t>](
                    #[spirv(global_invocation_id)]
                    global_id: UVec3,
                    #[spirv(storage_buffer)] x: &[$t],
                    #[spirv(storage_buffer)] dx: &mut [$t],
                    #[spirv(storage_buffer)] dy: &[$t],
                    #[spirv(push_constant)]
                    push_consts: &ReluPushConsts,
                ) {
                    relu_backward(global_id.x as usize, x, dx, dy, push_consts);
                }
            )*
        }
    );
}

impl_relu! {
    bf16, f32
}
