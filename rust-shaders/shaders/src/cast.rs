use crate::{
    util::{Load, Store},
    byte::u8x4,
    short::u16x2,
    half::bf16x2,
    autobind,
};
use spirv_std::glam::UVec3;
use num_traits::cast::AsPrimitive;
use core::ops::Mul;

#[repr(C)]
pub struct ScalePushConsts<A> {
    n: u32,
    alpha: A
}

fn scale<X, T1, Y, T2>(
    global_id: UVec3,
    x: &[X],
    y: &mut [Y],
    push_consts: &ScalePushConsts<T2>,
) where [X]: Load<T1>, T1: AsPrimitive<T2>, [Y]: Store<T2>, T2: Copy + Mul<T2, Output=T2> + 'static {
    let gid = global_id.x as usize;
    let n = push_consts.n as usize;
    let alpha = push_consts.alpha;
    if gid < n {
        y.store(gid, alpha * x.load(gid).as_());
    }
}

macro_rules! impl_scale {
    ($($func:ident<$X:ty, $T1:ty, $Y:ty, $T2:ty>),* $(,)?) => (
        $(
            #[autobind]
            #[spirv(compute(threads(64)))]
            pub fn $func(
                #[spirv(global_invocation_id)]
                global_id: UVec3,
                #[spirv(storage_buffer)] x: &[$X],
                #[spirv(storage_buffer)] y: &mut [$Y],
                #[spirv(push_constant)]
                push_consts: &ScalePushConsts<$T2>,
            ) {
                scale::<$X, $T1, $Y, $T2>(global_id, x, y, push_consts);
            }
        )*
    );
}

include!(concat!(env!("OUT_DIR"), "/scale_impls.rs"));
