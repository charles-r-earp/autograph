use spirv_std::glam::UVec3;
use crate::autobind;

trait FillPushConsts {
    type Elem;
    fn elem(&self) -> Self::Elem;
}


#[repr(C)]
pub struct FillPushConstsU32 {
    n: u32,
    x: u32,
}

impl FillPushConsts for FillPushConstsU32 {
    type Elem = u32;
    fn elem(&self) -> u32 {
        self.x
    }
}

#[repr(C)]
pub struct FillPushConstsU32x2 {
    n: u32,
    x: u32,
    y: u32,
}

impl FillPushConsts for FillPushConstsU32x2 {
    type Elem = [u32; 2];
    fn elem(&self) -> [u32; 2] {
        [self.x, self.y]
    }
}


macro_rules! impl_fill {
    ($($func:ident<$T:ty, $P:ty>),* $(,)?) => (
        $(
            #[autobind]
            #[spirv(compute(threads(256)))]
            pub fn $func(
                #[spirv(workgroup_id)]
                group_id: UVec3,
                #[spirv(local_invocation_id)]
                local_id: UVec3,
                #[spirv(storage_buffer)] y: &mut [$T],
                #[spirv(push_constant)]
                push_consts: &$P,
            ) {
                let gid = (group_id.x * 256 + local_id.x) as usize;
                let n = push_consts.n as usize;
                if gid < n {
                    y[gid] = push_consts.elem();
                }
            }
        )*
    );
}

impl_fill!{
    fill_u32<u32, FillPushConstsU32>,
    //fill_u32x2<[u32; 2], FillPushConstsU32x2>,
}

#[autobind]
#[spirv(compute(threads(256)))]
pub fn fill_u32x2(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer)] y: &mut [u32],
    #[spirv(push_constant)]
    push_consts: &FillPushConstsU32x2,
) {
    let gid = (group_id.x * 256 + local_id.x) as usize;
    let n = push_consts.n as usize;
    if gid < n {
        y[gid * 2] = push_consts.x;
        y[gid * 2 + 1] = push_consts.y;
    }
}
