use spirv_std::glam::UVec3;
use num_traits::Float;

#[repr(C)]
pub struct CrossEntropyLossPushConsts {
    n: u32,
    nclasses: u32,
}

#[allow(unused_attributes)]
#[spirv(compute(threads(64)))]
pub fn cross_entropy_loss_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)]
    x: &[f32],
    #[spirv(storage_buffer, descriptor_set=0, binding=1)]
    t: &[f32],
    #[spirv(storage_buffer, descriptor_set=0, binding=2)]
    y: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &CrossEntropyLossPushConsts,
) {
    let global_x = global_id.x as usize;
    let n = push_consts.n as usize;
    let nclasses = push_consts.nclasses as usize;
    let idx = global_x * nclasses;

    let mut m = f32::MIN;
    let mut s = 0.;
    if global_x < n {
        for i in 0 .. nclasses {
            m = m.max(x[idx + i]);
        }
        for i in 0 .. nclasses {
            s += (x[idx + i] - m).exp();
        }
    }

    for i in 0 .. nclasses {
        if global_x < n {
            y[idx + i] = (s.ln() - (x[idx + i] - m)) * t[idx + i];
        }
    }
}

#[allow(unused_attributes)]
#[spirv(compute(threads(64)))]
pub fn cross_entropy_loss_backward_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)]
    x: &[f32],
    #[spirv(storage_buffer, descriptor_set=0, binding=1)]
    dx: &mut [f32],
    #[spirv(storage_buffer, descriptor_set=0, binding=2)]
    t: &[f32],
    #[spirv(storage_buffer, descriptor_set=0, binding=3)]
    dy: &[f32],
    #[spirv(push_constant)]
    push_consts: &CrossEntropyLossPushConsts,
) {
    let global_x = global_id.x as usize;
    let n = push_consts.n as usize;
    let nclasses = push_consts.nclasses as usize;

    let idx = global_x * nclasses;
/*
    if global_x < n {
        let mut m = x[idx];
        for i in 1 .. nclasses {
            m = m.max(x[idx + i]);
        }
        let mut s = 0.;
        for i in 0 .. nclasses {
            s += (x[idx + i] - m).exp();
        }
        for i in 0 .. nclasses {
            let p = (x[idx + i] - m).exp() / s;
            dx[idx + i] += dy * (p - t[idx + i]);
        }
    }
*/

    let mut m = f32::MIN;
    let mut s = 0.;
    if global_x < n {
        for i in 0 .. nclasses {
            m = m.max(x[idx + i]);
        }
        for i in 0 .. nclasses {
            s += (x[idx + i] - m).exp();
        }
    }

    for i in 0 .. nclasses {
        if global_x < n {
            let p = (x[idx + i] - m).exp() / s;
            let dy = dy[0] / (n as f32);
            dx[idx + i] += dy * (p - t[idx + i]);
        }
    }
}
