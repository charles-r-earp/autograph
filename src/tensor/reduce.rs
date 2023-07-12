use super::*;
#[cfg(feature = "device")]
use krnl::macros::module;

impl<T: Scalar, S: Data<Elem = T>, D: Dimension> TensorBase<S, D> {
    pub fn sum(&self) -> Result<T> {
        if let Some(input) = self.as_array() {
            Ok(input.sum())
        } else {
            let mut output = unsafe { Tensor::uninit(self.device(), ())? };
            self.sum_with(T::default(), &mut output)?;
            Ok(output.into_array()?.into_scalar())
        }
    }
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
            Ok(())
        } else {
            sum(
                self.view().into_dyn().into(),
                beta.into(),
                output.view_mut().into_dyn().into(),
            )
        }
    }
}

impl<T: Scalar, S: Data<Elem = T>, D: RemoveAxis> TensorBase<S, D> {
    pub fn sum_axis(&self, axis: Axis) -> Result<Tensor<T, D::Smaller>> {
        if let Some(input) = self.as_array() {
            Ok(input.sum_axis(axis).into())
        } else {
            let mut output =
                unsafe { Tensor::uninit(self.device(), self.raw_dim().remove_axis(axis))? };
            self.sum_axis_with(axis, T::default(), &mut output)?;
            Ok(output)
        }
    }
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
            Ok(())
        } else {
            sum_axis(
                self.view().into_dyn().into(),
                axis,
                beta.into(),
                output.view_mut().into_dyn().into(),
            )
        }
    }
}

fn sum(x: ScalarTensorViewD, beta: ScalarElem, mut y: ScalarTensorViewMutD) -> Result<()> {
    if x.scalar_type() != y.scalar_type() {
        todo!();
    }
    if x.device() != y.device() {
        todo!();
    }

    let groups: u32 = y.len() as u32;
    let threads = 64;

    let x = if x.is_contiguous() {
        x.into()
    } else {
        x.as_standard_layout()?
    };
    let x = x.as_scalar_slice().unwrap();
    let y = y.as_scalar_slice_mut().unwrap();

    macro_for!($T in [u32, i32, f32, u64, i64, f64] {
        if x.scalar_type() == $T::scalar_type() {
            let x = Slice::try_from(x).unwrap();
            let y = SliceMut::try_from(y).unwrap();
            let beta = beta.cast::<$T>();
            let kernel = paste! {
                kernels::[<sum_ $T>]::builder()?.specialize(
                    threads
                )?.build(y.device())?
            };
            kernel.with_groups(groups).dispatch(
                x,
                beta,
                y,
            )?;
            return Ok(());
        }
    });
    todo!()
}

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

    let groups: u32 = y.len() as u32;
    let threads = 64;

    let [rsx, csx]: [isize; 2] = x.strides().try_into().unwrap();
    let [bsx, sx] = if axis.0 == 0 {
        [csx as i32, rsx as i32]
    } else {
        [rsx as i32, csx as i32]
    };

    let (x, offset_x) = x.as_raw_scalar_slice_offset();
    let offset_x = offset_x as u32;
    let y = y.as_scalar_slice_mut().unwrap();

    macro_for!($T in [u32, i32, f32, u64, i64, f64] {
        if x.scalar_type() == $T::scalar_type() {
            let x = Slice::try_from(x).unwrap();
            let y = SliceMut::try_from(y).unwrap();
            let beta = beta.cast::<$T>();
            let kernel = paste! {
                kernels::[<sum_axis2_ $T>]::builder()?.specialize(
                    threads, bsx, sx,
                )?.build(y.device())?
            };
            kernel.with_groups(groups).dispatch(
                x,
                offset_x,
                beta,
                y,
            )?;
            return Ok(());
        }
    });
    todo!()
}

#[cfg(feature = "device")]
#[module]
mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;
    #[cfg(target_arch = "spirv")]
    use krnl_core::{buffer::UnsafeIndex, scalar::Scalar};
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
        ($t:ty => $a:ty) => {
            #[cfg(target_arch = "spirv")]
            impl SubgroupAdd for $t {
                unsafe fn subgroup_add(self) -> Self {
                    unsafe {
                        paste! {
                            [<subgroup_add_ $a>](self.cast::<$a>()).cast()
                        }
                    }
                }
            }
        };
        ($($t:ty),*) => {
            $(
                impl_subgroup_add!($t => $t);
            )*
        };
        ($($t:ty),* => $a:ty) => {
            $(
                impl_subgroup_add!($t => $a);
            )*
        }
    }

    impl_subgroup_add!(u32, i32, f32, u64, i64, f64);

    macro_rules! impl_sum {
        ($t:ty => $a:ty) => {
            paste! {
                #[kernel(threads(TS))]
                pub fn [<sum_ $t>]<const TS: u32>(
                    #[global] x: Slice<$t>,
                    beta: $t,
                    #[global] y: UnsafeSlice<$t>,
                ) {
                    type T = $t;
                    type A = $a;
                    let n = x.len() / y.len();
                    let thread_id = kernel.thread_id() as usize;
                    let subgroup_id = kernel.subgroup_id() as usize;
                    if subgroup_id > 0 {
                        return;
                    }
                    let subgroup_threads = (TS / kernel.subgroups()) as usize;
                    let mut y_thread = A::default();
                    let mut idx = 0;
                    while idx < n {
                        let x_idx = idx + thread_id;
                        if x_idx < n {
                            y_thread += x[x_idx].cast::<A>();
                        }
                        idx += subgroup_threads;
                    }
                    unsafe {
                        y_thread = y_thread.subgroup_add().cast::<T>();
                    };
                    if thread_id == 0 {
                        if beta == T::default() {
                            unsafe {
                                *y.unsafe_index_mut(0) = y_thread;
                            }
                        } else {
                            unsafe {
                                *y.unsafe_index_mut(0) = y_thread + beta * *y.unsafe_index(0);
                            }
                        }
                    }
                }

                #[kernel(threads(TS))]
                pub fn [<sum_axis2_ $t>]<const TS: u32, const BSX: i32, const SX: i32>(
                    #[global] x: Slice<$t>,
                    offset_x: u32,
                    beta: $t,
                    #[global] y: UnsafeSlice<$t>,
                ) {
                    type T = $t;
                    type A = $a;
                    let n = x.len() / y.len();
                    let group_id = kernel.group_id() as usize;
                    let thread_id = kernel.thread_id() as usize;
                    let subgroup_id = kernel.subgroup_id() as usize;
                    if subgroup_id > 0 {
                        return;
                    }
                    let mut x_start = group_id as i32 * BSX + offset_x as i32;
                    let subgroup_threads = (TS / kernel.subgroups()) as usize;
                    let mut y_thread = A::default();
                    let mut idx = 0;
                    while idx < n {
                        let x_idx = idx + thread_id;
                        if x_idx < n {
                            y_thread += x[(x_start + x_idx as i32 * SX) as usize].cast::<A>();
                        }
                        idx += subgroup_threads;
                    }
                    unsafe {
                        y_thread = y_thread.subgroup_add().cast::<T>();
                    };
                    if thread_id == 0 {
                        if beta == T::default() {
                            unsafe {
                                *y.unsafe_index_mut(group_id) = y_thread;
                            }
                        } else {
                            unsafe {
                                *y.unsafe_index_mut(group_id) = y_thread + beta * *y.unsafe_index(group_id);
                            }
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

    impl_sum!(u32, i32, f32, u64, i64, f64);
}
