use crate::{util::{Load, Store}, half::bf16x2, autobind};
use spirv_std::glam::UVec3;

#[repr(C)]
pub struct ReluPushConsts {
    n: u32,
}

fn relu<T>(
    gid: usize,
    x: &[T],
    y: &mut [T],
    push_consts: &ReluPushConsts,
) where [T]: Store<f32> {
    let n = push_consts.n as usize;
    if gid < n {
        let x = x.load(gid);
        y.store(gid, if x > 0. { x } else { 0. });
    }
}

fn relu_backward<T>(
    gid: usize,
    x: &[T],
    dx: &mut [T],
    dy: &[T],
    push_consts: &ReluPushConsts,
) where [T]: Store<f32> {
    let n = push_consts.n as usize;
    if gid < n {
        let x = x.load(gid);
        let dy = dy.load(gid);
        let dy = if x > 0. {
            dy
        } else {
            0.
        };
        dx.store(gid, dx.load(gid) + dy);
    }
}

macro_rules! impl_relu {
    ($($fw:ident | $bw:ident <$t:ty>),* $(,)?) => (
        $(
            #[autobind]
            #[spirv(compute(threads(256)))]
            pub fn $fw(
                #[spirv(workgroup_id)]
                group_id: UVec3,
                #[spirv(local_invocation_id)]
                local_id: UVec3,
                #[spirv(storage_buffer)] x: &[$t],
                #[spirv(storage_buffer)] y: &mut [$t],
                #[spirv(push_constant)]
                push_consts: &ReluPushConsts,
            ) {
                let gid = (group_id.x * 256 + local_id.x) as usize;
                relu(gid, x, y, push_consts);
            }

            #[autobind]
            #[spirv(compute(threads(256)))]
            pub fn $bw(
                #[spirv(workgroup_id)]
                group_id: UVec3,
                #[spirv(local_invocation_id)]
                local_id: UVec3,
                #[spirv(storage_buffer)] x: &[$t],
                #[spirv(storage_buffer)] dx: &mut [$t],
                #[spirv(storage_buffer)] dy: &[$t],
                #[spirv(push_constant)]
                push_consts: &ReluPushConsts,
            ) {
                let gid = (group_id.x * 256 + local_id.x) as usize;
                relu_backward(gid, x, dx, dy, push_consts);
            }
        )*
    );
}

impl_relu!{
    relu_bf16 | relu_backward_bf16 <bf16x2>,
    relu_f32 | relu_backward_f32 <f32>,
}
