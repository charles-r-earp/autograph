use spirv_std::{
    //memory::{Scope, Semantics},
    //arch::{control_barrier, /* memory_barrier */},
    glam::UVec3,
};
use num_traits::NumAssign;

trait Load<T> {
    fn load(&self, index: usize) -> T;
}

trait Store<T> {
    fn store(&mut self, index: usize, value: T);
}

impl Load<f32> for [f32] {
    fn load(&self, index: usize) -> f32 {
        self[index]
    }
}

impl Store<f32> for [f32] {
    fn store(&mut self, index: usize, value: f32) {
        self[index] = value;
    }
}

#[repr(C)]
pub struct GemmPushConsts<T> {
    alpha: T,
    beta: T,
    a0: T,
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

#[allow(unused)]
#[repr(u32)]
#[derive(Copy, Clone)]
enum Activation {
    Identity,
    Relu,
}

#[allow(unused)]
fn gemm_impl<X, T: Copy + NumAssign + PartialOrd, const M_TILE: usize, const K_TILE: usize, const N_TILE: usize>(
    global_id: UVec3,
    local_id: UVec3,
    a: &[X],
    a_tile: &mut [[T; M_TILE];  K_TILE],
    b: &[X],
    b_tile: &mut [[T; N_TILE]; K_TILE],
    use_bias: u32,
    bias: &[X],
    c: &mut [X],
    activation: Activation,
    push_consts: &GemmPushConsts<T>,
) where [X]: Load<T> + Store<T> {
    let global_x = global_id.x as usize;
    let global_y = global_id.y as usize;
    let local_x = local_id.x as usize;
    let local_y = local_id.y as usize;
    let m = push_consts.m as usize;
    let k = push_consts.k as usize;
    let n = push_consts.n as usize;
    let rsa = push_consts.rsa as usize;
    let csa = push_consts.csa as usize;
    let rsb = push_consts.rsb as usize;
    let csb = push_consts.csb as usize;
    let rsc = push_consts.rsc as usize;
    let csc = push_consts.csc as usize;

    let mut valid = 0;
    if global_x < m { if global_y < n {
        valid = 1;
    }}

    let mut acc = T::zero();
    for z in 0 .. k {
        let tile_k = z % K_TILE;
        if tile_k == 0 {
            /*
            unsafe {
                memory_barrier::<{Scope::Workgroup as u32}, {Semantics::ACQUIRE_RELEASE.bits() | Semantics::WORKGROUP_MEMORY.bits()}>();
            }
            */
            if global_x < m {
                a_tile[local_y][local_x] = a.load(global_x * rsa + (z + local_y) * csa);
            }

            if global_y < n {
                b_tile[local_x][local_y] = b.load((z + local_x) * rsb + global_y * csb);
            }

            /*unsafe {
                control_barrier::<{Scope::Workgroup as u32}, {Scope::Workgroup as u32}, {Semantics::NONE.bits()}>;
            }*/
        }

        /*
        unsafe {
            memory_barrier::<{Scope::Workgroup as u32}, {Semantics::ACQUIRE_RELEASE.bits() | Semantics::WORKGROUP_MEMORY.bits()}>();
        }
        */
        if valid == 1 {
            acc += a_tile[tile_k][local_x] * b_tile[tile_k][local_y];
        }

        /*unsafe {
            control_barrier::<{Scope::Workgroup as u32}, {Scope::Workgroup as u32}, {Semantics::ACQUIRE_RELEASE.bits() | Semantics::WORKGROUP_MEMORY.bits()}>();
        }*/
    }

    let mut y = push_consts.alpha * acc;

    let cidx = global_x * rsc + global_y * csc;
    if valid == 1 { if use_bias == 1 {
        y += push_consts.beta * c.load(cidx);
    }}

    match activation {
        Activation::Identity => (),
        Activation::Relu => {
            if y < T::zero() {
                y *= push_consts.a0;
            }
        }
    }

    if valid == 1 {
        c.store(cidx, y);
    }
}

#[allow(unused_attributes)]
#[spirv(compute(threads(8, 8)))]
pub fn gemm_f32(
    #[spirv(global_invocation_id)]
    global_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &[f32],
    #[spirv(workgroup)] a_tile: &mut [[f32; 8]; 8],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] b: &[f32],
    #[spirv(workgroup)] b_tile: &mut [[f32; 8]; 8],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] c: &mut [f32],
    #[spirv(push_constant)]
    push_consts: &GemmPushConsts<f32>,
) {
    gemm_impl::<f32, f32, 8, 8, 8>(global_id, local_id, a, a_tile, b, b_tile, 0, a, c, Activation::Identity, push_consts)
}
