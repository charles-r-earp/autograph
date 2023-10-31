use super::*;
#[cfg(feature = "device")]
use krnl::macros::module;

impl<T: Scalar, S: Data<Elem = T>, D: Dimension> TensorBase<S, D> {
    /// Sums the tensor.
    pub fn sum(&self) -> Result<T> {
        if let Some(input) = self.as_array() {
            Ok(input.sum())
        } else {
            let mut output = unsafe { Tensor::uninit(self.device(), ())? };
            self.sum_with(T::default(), &mut output)?;
            Ok(output.into_array()?.into_scalar())
        }
    }
    /// Sums the tensor with `output`.
    pub fn sum_with<S2: DataMut<Elem = T>>(
        &self,
        beta: T,
        output: &mut TensorBase<S2, Ix0>,
    ) -> Result<()> {
        if let Some((input, mut output)) = self.as_array().zip(output.as_array_mut()) {
            if beta == T::default() {
                output[()] = input.sum();
            } else {
                output[()] = input.sum() + beta * output[()];
            }
            return Ok(());
        }
        #[cfg(not(feature = "device"))]
        {
            unreachable!()
        }
        #[cfg(feature = "device")]
        {
            sum(
                self.view().into_dyn().into(),
                beta.into(),
                output.view_mut().into_dyn().into(),
            )
        }
    }
}

impl<T: Scalar, S: Data<Elem = T>, D: RemoveAxis> TensorBase<S, D> {
    /// Sums the tensor along `axis`.
    pub fn sum_axis(&self, axis: Axis) -> Result<Tensor<T, D::Smaller>> {
        if let Some(input) = self.as_array() {
            return Ok(input.sum_axis(axis).into());
        }
        #[cfg(not(feature = "device"))]
        {
            unreachable!()
        }
        #[cfg(feature = "device")]
        {
            let mut output =
                unsafe { Tensor::uninit(self.device(), self.raw_dim().remove_axis(axis))? };
            self.sum_axis_with(axis, T::default(), &mut output)?;
            Ok(output)
        }
    }
    /// Sums the tensor along `axis` with `output`.
    pub fn sum_axis_with<S2: DataMut<Elem = T>>(
        &self,
        axis: Axis,
        beta: T,
        output: &mut TensorBase<S2, D::Smaller>,
    ) -> Result<()> {
        if let Some((x, mut y)) = self.as_array().zip(output.as_array_mut()) {
            if beta == T::default() {
                y.fill(T::default());
            } else {
                for y in y.iter_mut() {
                    *y *= beta;
                }
            }
            for x in x.axis_iter(axis) {
                y.zip_mut_with(&x, |y, x| *y += *x);
            }
            return Ok(());
        }
        #[cfg(not(feature = "device"))]
        {
            unreachable!()
        }
        #[cfg(feature = "device")]
        {
            sum_axis(
                self.view().into_dyn().into(),
                axis,
                beta.into(),
                output.view_mut().into_dyn().into(),
            )
        }
    }
}

#[cfg(feature = "device")]
fn sum(x: ScalarTensorViewD, beta: ScalarElem, mut y: ScalarTensorViewMutD) -> Result<()> {
    if x.scalar_type() != y.scalar_type() {
        todo!();
    }
    if x.device() != y.device() {
        todo!();
    }
    let device = y.device();
    let info = device.info().unwrap();

    let groups: u32 = y.len() as u32;
    let threads = info.subgroup_threads();

    let x = if x.is_contiguous() {
        x.into()
    } else {
        x.as_standard_layout()?
    };
    let x = x.as_scalar_slice().unwrap();
    let y = y.as_scalar_slice_mut().unwrap();

    macro_for!($T in [bf16, u32, i32, f32, u64, i64, f64] {
        if x.scalar_type() == $T::scalar_type() {
            let x = Slice::try_from(x).unwrap();
            let y = SliceMut::try_from(y).unwrap();
            let kernel = paste! {
                kernels::[<sum_ $T>]::builder()?.with_threads(threads).build(device)?
            };
            kernel.with_groups(groups).dispatch(
                x,
                beta.cast(),
                y,
            )?;
            return Ok(());
        }
    });
    todo!()
}

#[cfg(feature = "device")]
fn sum_axis(
    x: ScalarTensorViewD,
    axis: Axis,
    beta: ScalarElem,
    mut y: ScalarTensorViewMutD,
) -> Result<()> {
    if x.scalar_type() != y.scalar_type() {
        todo!();
    }
    if x.device() != y.device() {
        todo!();
    }
    let device = y.device();
    let info = device.info().unwrap();

    let groups = y.len().to_u32().unwrap();
    let threads = info.subgroup_threads();
    let axis = axis.0.to_u32().unwrap();

    let ndim = x.ndim();
    assert!(x.ndim() == y.ndim() + 1, "{:?} {:?}", x.shape(), y.shape());
    if ndim <= 2 {
        let axis = if ndim == 1 { axis + 1 } else { axis };
        let (d0, d1) = match x.shape() {
            [d0, d1] => (d0.to_u32().unwrap(), d1.to_u32().unwrap()),
            [d1] => (1, d1.to_u32().unwrap()),
            [] => (1, 1),
            _ => unreachable!(),
        };
        let (sx0, sx1) = match x.strides() {
            [sx0, sx1] => (sx0.to_i32().unwrap(), sx1.to_i32().unwrap()),
            [sx1] => (1, sx1.to_i32().unwrap()),
            [] => (1, 1),
            _ => unreachable!(),
        };
        let sy0 = y.strides().first().copied().unwrap_or(1).to_i32().unwrap();
        let (x, offset_x) = x.as_raw_scalar_slice_offset();
        let offset_x = offset_x.to_u32().unwrap();
        let (y, offset_y) = y.as_raw_scalar_slice_offset_mut();
        let offset_y = offset_y.to_u32().unwrap();

        macro_for!($T in [bf16, u32, i32, f32, u64, i64, f64] {
            if x.scalar_type() == $T::scalar_type() {
                let x = Slice::try_from(x).unwrap();
                let y = SliceMut::try_from(y).unwrap();
                let kernel = paste! {
                    kernels::[<sum_axis2_ $T>]::builder()?
                        .with_threads(threads)
                        .specialize(axis)
                        .build(device)?
                };
                kernel.with_groups(groups).dispatch(
                    d0,
                    d1,
                    x,
                    sx0,
                    sx1,
                    offset_x,
                    beta.cast(),
                    y,
                    sy0,
                    offset_y,
                )?;
                return Ok(());
            }
        });
    }
    if ndim <= 4 {
        let axis = if ndim == 3 { axis + 1 } else { axis };
        let [d0, d1, d2, d3] = match x.shape() {
            [d0, d1, d2, d3] => [
                d0.to_u32().unwrap(),
                d1.to_u32().unwrap(),
                d2.to_u32().unwrap(),
                d3.to_u32().unwrap(),
            ],
            [d1, d2, d3] => [
                1,
                d1.to_u32().unwrap(),
                d2.to_u32().unwrap(),
                d3.to_u32().unwrap(),
            ],
            _ => unreachable!(),
        };
        let [sx0, sx1, sx2, sx3] = match x.strides() {
            [sx0, sx1, sx2, sx3] => [
                sx0.to_i32().unwrap(),
                sx1.to_i32().unwrap(),
                sx2.to_i32().unwrap(),
                sx3.to_i32().unwrap(),
            ],
            [sx1, sx2, sx3] => [
                1,
                sx1.to_i32().unwrap(),
                sx2.to_i32().unwrap(),
                sx3.to_i32().unwrap(),
            ],
            _ => unreachable!(),
        };
        let [sy0, sy1, sy2] = match y.strides() {
            [sy0, sy1, sy2] => [
                sy0.to_i32().unwrap(),
                sy1.to_i32().unwrap(),
                sy2.to_i32().unwrap(),
            ],
            [sy1, sy2] => [1, sy1.to_i32().unwrap(), sy2.to_i32().unwrap()],
            _ => unreachable!(),
        };
        let (x, offset_x) = x.as_raw_scalar_slice_offset();
        let offset_x = offset_x.to_u32().unwrap();
        let (y, offset_y) = y.as_raw_scalar_slice_offset_mut();
        let offset_y = offset_y.to_u32().unwrap();

        macro_for!($T in [bf16, u32, i32, f32, u64, i64, f64] {
            if x.scalar_type() == $T::scalar_type() {
                let x = Slice::try_from(x).unwrap();
                let y = SliceMut::try_from(y).unwrap();
                let kernel = paste! {
                    kernels::[<sum_axis4_ $T>]::builder()?
                        .with_threads(threads)
                        .specialize(axis)
                        .build(device)?
                };
                kernel.with_groups(groups).dispatch(
                    d0,
                    d1,
                    d2,
                    d3,
                    x,
                    sx0,
                    sx1,
                    sx2,
                    sx3,
                    offset_x,
                    beta.cast(),
                    y,
                    sy0,
                    sy1,
                    sy2,
                    offset_y,
                )?;
                return Ok(());
            }
        });
    }
    bail!(
        "sum_axis{ndim}<{}>(axis={axis}) unimplemented!",
        x.scalar_type().name()
    )
}

#[cfg(feature = "device")]
#[module]
mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;
    #[cfg(target_arch = "spirv")]
    use krnl_core::{buffer::UnsafeIndex, half::bf16, scalar::Scalar};
    use paste::paste;

    #[cfg(target_arch = "spirv")]
    unsafe fn subgroup_add_u32(x: u32) -> u32 {
        use core::arch::asm;

        let mut y = 0u32;
        asm! {
            "%u32 = OpTypeInt 32 0",
            "%subgroup = OpConstant %u32 3",
            "%y = OpGroupNonUniformIAdd _ %subgroup Reduce {x}",
            "OpStore {y} %y",
            x = in(reg) x,
            y = in(reg) &mut y,
        }
        y
    }

    #[cfg(target_arch = "spirv")]
    unsafe fn subgroup_add_i32(x: i32) -> i32 {
        use core::arch::asm;

        let mut y = 0i32;
        asm! {
            "%u32 = OpTypeInt 32 0",
            "%subgroup = OpConstant %u32 3",
            "%y = OpGroupNonUniformIAdd _ %subgroup Reduce {x}",
            "OpStore {y} %y",
            x = in(reg) x,
            y = in(reg) &mut y,
        }
        y
    }

    #[cfg(target_arch = "spirv")]
    unsafe fn subgroup_add_f32(x: f32) -> f32 {
        use core::arch::asm;

        let mut y = 0f32;
        asm! {
            "%u32 = OpTypeInt 32 0",
            "%subgroup = OpConstant %u32 3",
            "%y = OpGroupNonUniformFAdd _ %subgroup Reduce {x}",
            "OpStore {y} %y",
            x = in(reg) x,
            y = in(reg) &mut y,
        }
        y
    }

    #[cfg(target_arch = "spirv")]
    unsafe fn subgroup_add_u64(x: u64) -> u64 {
        use core::arch::asm;

        let mut y = 0u64;
        asm! {
            "%u32 = OpTypeInt 32 0",
            "%subgroup = OpConstant %u32 3",
            "%y = OpGroupNonUniformIAdd _ %subgroup Reduce {x}",
            "OpStore {y} %y",
            x = in(reg) x,
            y = in(reg) &mut y,
        }
        y
    }

    #[cfg(target_arch = "spirv")]
    unsafe fn subgroup_add_i64(x: i64) -> i64 {
        use core::arch::asm;

        let mut y = 0i64;
        asm! {
            "%u32 = OpTypeInt 32 0",
            "%subgroup = OpConstant %u32 3",
            "%y = OpGroupNonUniformIAdd _ %subgroup Reduce {x}",
            "OpStore {y} %y",
            x = in(reg) x,
            y = in(reg) &mut y,
        }
        y
    }

    #[cfg(target_arch = "spirv")]
    unsafe fn subgroup_add_f64(x: f64) -> f64 {
        use core::arch::asm;

        let mut y = 0f64;
        asm! {
            "%u32 = OpTypeInt 32 0",
            "%subgroup = OpConstant %u32 3",
            "%y = OpGroupNonUniformFAdd _ %subgroup Reduce {x}",
            "OpStore {y} %y",
            x = in(reg) x,
            y = in(reg) &mut y,
        }
        y
    }

    #[cfg(target_arch = "spirv")]
    trait SubgroupAdd: Sized {
        unsafe fn subgroup_add(self) -> Self;
    }

    macro_rules! impl_subgroup_add {
        ($($t:ty),*) => {
            $(
                #[cfg(target_arch = "spirv")]
                impl SubgroupAdd for $t {
                    unsafe fn subgroup_add(self) -> Self {
                        unsafe {
                            paste! {
                                [<subgroup_add_ $t>](self)
                            }
                        }
                    }
                }
            )*
        };
    }

    impl_subgroup_add!(u32, i32, f32, u64, i64, f64);

    macro_rules! impl_sum {
        ($t:ty => $a:ty) => {
            paste! {
                #[kernel]
                pub fn [<sum_ $t>](
                    #[global] x: Slice<$t>,
                    beta: $a,
                    #[global] y: UnsafeSlice<$t>,
                ) {
                    type T = $t;
                    type A = $a;
                    let thread_id = kernel.thread_id as usize;
                    let subgroup_id = kernel.subgroup_id as usize;
                    if subgroup_id > 0 {
                        return;
                    }
                    let subgroup_threads = (kernel.threads / kernel.subgroups) as usize;
                    let mut y_thread = A::default();
                    let mut idx = 0;
                    let n = x.len() / y.len();
                    while idx < n {
                        let x_idx = idx + thread_id;
                        if x_idx < n {
                            y_thread += x[x_idx].cast::<A>();
                        }
                        idx += subgroup_threads;
                    }
                    unsafe {
                        y_thread = y_thread.subgroup_add();
                    };
                    if thread_id == 0 {
                        if beta != A::default() {
                            unsafe {
                                y_thread += beta * y.unsafe_index(0).cast::<A>();
                            }
                        }
                        unsafe {
                            *y.unsafe_index_mut(0) = y_thread.cast::<T>();
                        }
                    }
                }

                #[kernel]
                pub fn [<sum_axis2_ $t>]<const AXIS: u32>(
                    d0: u32,
                    d1: u32,
                    #[global] x: Slice<$t>,
                    sx0: i32,
                    sx1: i32,
                    offset_x: u32,
                    beta: $a,
                    #[global] y: UnsafeSlice<$t>,
                    sy0: i32,
                    offset_y: u32,
                ) {
                    type T = $t;
                    type A = $a;
                    let group_id = kernel.group_id as usize;
                    let thread_id = kernel.thread_id as usize;
                    let subgroup_id = kernel.subgroup_id as usize;
                    if subgroup_id > 0 {
                        return;
                    }
                    let axis = AXIS as usize;
                    let n = [d0, d1][axis] as usize;
                    let stride_group = [sx1, sx0][axis];
                    let stride_axis = [sx0, sx1][axis];
                    let mut x_start = group_id as i32 * stride_group + offset_x as i32;
                    let subgroup_threads = (kernel.threads / kernel.subgroups) as usize;
                    let mut y_thread = A::default();
                    let mut idx = 0;
                    while idx < n {
                        let x_idx = idx + thread_id;
                        if x_idx < n {
                            y_thread += x[(x_start + x_idx as i32 * stride_axis) as usize].cast::<A>();
                        }
                        idx += subgroup_threads;
                    }
                    unsafe {
                        y_thread = y_thread.subgroup_add();
                    };
                    let y_idx = (group_id as i32 * sy0 + offset_y as i32) as usize;
                    if thread_id == 0 {
                        if beta != A::default() {
                            unsafe {
                                y_thread += beta * y.unsafe_index(y_idx).cast::<A>();
                            }
                        }
                        unsafe {
                            *y.unsafe_index_mut(y_idx) = y_thread.cast::<T>();
                        }
                    }
                }

                #[kernel]
                pub fn [<sum_axis4_ $t>]<const AXIS: u32>(
                    d0: u32,
                    d1: u32,
                    d2: u32,
                    d3: u32,
                    #[global] x: Slice<$t>,
                    sx0: i32,
                    sx1: i32,
                    sx2: i32,
                    sx3: i32,
                    offset_x: u32,
                    beta: $a,
                    #[global] y: UnsafeSlice<$t>,
                    sy0: i32,
                    sy1: i32,
                    sy2: i32,
                    offset_y: u32,
                ) {
                    type T = $t;
                    type A = $a;
                    let group_id = kernel.group_id;
                    let thread_id = kernel.thread_id as usize;
                    let subgroup_id = kernel.subgroup_id as usize;
                    if subgroup_id > 0 {
                        return;
                    }
                    let axis = AXIS as usize;
                    let n = [d0, d1, d2, d3][axis] as usize;
                    let stride_axis = [sx0, sx1, sx2, sx3][axis];
                    let [gd0, gd1, gd2] = match axis {
                        0 => [d1, d2, d3],
                        1 => [d0, d2, d3],
                        2 => [d0, d1, d3],
                        3 => [d0, d1, d2],
                        _ => unreachable!(),
                    };
                    let [sg0, sg1, sg2] = match axis {
                        0 => [sx1, sx2, sx3],
                        1 => [sx0, sx2, sx3],
                        2 => [sx0, sx1, sx3],
                        3 => [sx0, sx1, sx2],
                        _ => unreachable!(),
                    };
                    let [i0, i1, i2] = {
                        let i0 = group_id / (gd1 * gd2);
                        let r0 = group_id % (gd1 * gd2);
                        let i1 = r0 / gd2;
                        let i2 = r0 % gd2;
                        [i0, i1, i2]
                    };
                    let mut x_start = i0 as i32 * sg0 + i1 as i32 * sg1 + i2 as i32 * sg2 + offset_x as i32;
                    let subgroup_threads = (kernel.threads / kernel.subgroups) as usize;
                    let mut y_thread = A::default();
                    let mut idx = 0;
                    while idx < n {
                        let x_idx = idx + thread_id;
                        if x_idx < n {
                            y_thread += x[(x_start + x_idx as i32 * stride_axis) as usize].cast::<A>();
                        }
                        idx += subgroup_threads;
                    }
                    unsafe {
                        y_thread = y_thread.subgroup_add();
                    };
                    let y_idx = (i0 as i32 * sy0 + i1 as i32 * sy1 as i32 + i2 as i32 * sy2 + offset_y as i32) as usize;
                    if thread_id == 0 {
                        if beta != A::default() {
                            unsafe {
                                y_thread += beta * y.unsafe_index(y_idx).cast::<A>();
                            }
                        }
                        unsafe {
                            *y.unsafe_index_mut(y_idx) = y_thread.cast::<T>();
                        }
                    }
                }
            }
        };
        ($($t:ty),*) => {
            $(
                impl_sum!($t => $t);
            )*
        };
        ($($t:ty),* => $a:ty) => {
            $(
                impl_sum!($t => $a);
            )*
        }
    }

    impl_sum!(bf16 => f32);
    impl_sum!(u32, i32, f32, u64, i64, f64);
}
