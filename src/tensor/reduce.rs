use super::*;
#[cfg(feature = "device")]
use half::f16;
#[cfg(feature = "device")]
use krnl::macros::module;
use parallel::parallel_size;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::mem::size_of;

impl<T: Scalar, S: Data<Elem = T>, D: Dimension> TensorBase<S, D> {
    /// Sums the tensor.
    pub fn sum(&self) -> Result<T> {
        if let Some(input) = self.as_array() {
            if input.len() * size_of::<T>() > parallel_size() && rayon::current_num_threads() > 1 {
                Ok(input
                    .into_par_iter()
                    .copied()
                    .reduce(T::default, |a, b| a + b))
            } else {
                Ok(input.sum())
            }
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
        if let Some((_input, mut output)) = self.as_array().zip(output.as_array_mut()) {
            if beta == T::default() {
                output[()] = self.sum().unwrap();
            } else {
                output[()] = self.sum().unwrap() + beta * output[()];
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
        if beta == T::default() && self.device().is_host() {
            output.fill(T::default())?;
        }
        if let Some((x, mut y)) = self.as_array().zip(output.as_array_mut()) {
            // TODO: impl in parallel
            for (i, x) in x.axis_iter(axis).enumerate() {
                if i == 0 {
                    if beta != T::default() {
                        y.zip_mut_with(&x, |y, x| *y = *x + beta * *y);
                    } else {
                        y.zip_mut_with(&x, |y, x| *y = *x);
                    }
                } else {
                    y.zip_mut_with(&x, |y, x| *y += *x);
                }
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
    let device = y.device();
    let info = device.info().unwrap();
    let subgroup_threads = if info.min_subgroup_threads() == info.max_subgroup_threads() {
        info.max_subgroup_threads()
    } else {
        0
    };
    let threads = info.max_subgroup_threads();
    let groups = y.len().to_u32().unwrap();
    let x = if x.is_contiguous() {
        x.into()
    } else {
        x.as_standard_layout()?
    };
    let x = x.as_scalar_slice().unwrap();
    let y = y.as_scalar_slice_mut().unwrap();
    macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
        if x.scalar_type() == $T::SCALAR_TYPE {
            let x = Slice::try_from(x).unwrap();
            let y = SliceMut::try_from(y).unwrap();
            let kernel = paste! {
                kernels::[<sum_ $T>]::builder()?.with_threads(threads).specialize(subgroup_threads).build(device)?
            };
            kernel.with_groups(groups).dispatch(
                x,
                beta.cast(),
                y,
            )?;
            return Ok(());
        }
    });
    unreachable!()
}

#[cfg(feature = "device")]
fn sum_axis(
    x: ScalarTensorViewD,
    axis: Axis,
    beta: ScalarElem,
    mut y: ScalarTensorViewMutD,
) -> Result<()> {
    if x.device() != y.device() {
        bail!("Expected {:?}, found {:?}", x.device(), y.device());
    }
    let device = y.device();
    let info = device.info().unwrap();
    let subgroup_threads = if info.min_subgroup_threads() == info.max_subgroup_threads() {
        info.max_subgroup_threads()
    } else {
        0
    };
    let global_subgroups = y.len().to_u32().unwrap();
    let threads = info.default_threads().max(subgroup_threads);
    let subgroups = (threads / info.max_subgroup_threads()).min(global_subgroups);
    let threads = subgroups * info.max_subgroup_threads();
    let groups = global_subgroups / subgroups + (global_subgroups % subgroups != 0) as u32;
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
        macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            if x.scalar_type() == $T::SCALAR_TYPE {
                let x = Slice::try_from(x).unwrap();
                let y = SliceMut::try_from(y).unwrap();
                let kernel = paste! {
                    kernels::[<sum_axis2_ $T>]::builder()?
                        .with_threads(threads)
                        .specialize(subgroups, subgroup_threads, axis)
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
    if ndim == 3 || ndim == 4 {
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

        macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            if x.scalar_type() == $T::SCALAR_TYPE {
                let x = Slice::try_from(x).unwrap();
                let y = SliceMut::try_from(y).unwrap();
                let kernel = paste! {
                    kernels::[<sum_axis4_ $T>]::builder()?
                        .with_threads(threads)
                        .specialize(subgroups, subgroup_threads, axis)
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
    if ndim == 5 || ndim == 6 {
        let axis = if ndim == 5 { axis + 1 } else { axis };
        let [d0, d1, d2, d3, d4, d5] = match x.shape() {
            [d0, d1, d2, d3, d4, d5] => [
                d0.to_u32().unwrap(),
                d1.to_u32().unwrap(),
                d2.to_u32().unwrap(),
                d3.to_u32().unwrap(),
                d4.to_u32().unwrap(),
                d5.to_u32().unwrap(),
            ],
            [d1, d2, d3, d4, d5] => [
                1,
                d1.to_u32().unwrap(),
                d2.to_u32().unwrap(),
                d3.to_u32().unwrap(),
                d4.to_u32().unwrap(),
                d5.to_u32().unwrap(),
            ],
            _ => unreachable!(),
        };
        let [sx0, sx1, sx2, sx3, sx4, sx5] = match x.strides() {
            [sx0, sx1, sx2, sx3, sx4, sx5] => [
                sx0.to_i32().unwrap(),
                sx1.to_i32().unwrap(),
                sx2.to_i32().unwrap(),
                sx3.to_i32().unwrap(),
                sx4.to_i32().unwrap(),
                sx5.to_i32().unwrap(),
            ],
            [sx1, sx2, sx3, sx4, sx5] => [
                1,
                sx1.to_i32().unwrap(),
                sx2.to_i32().unwrap(),
                sx3.to_i32().unwrap(),
                sx4.to_i32().unwrap(),
                sx5.to_i32().unwrap(),
            ],
            _ => unreachable!(),
        };
        let [sy0, sy1, sy2, sy3, sy4] = match y.strides() {
            [sy0, sy1, sy2, sy3, sy4] => [
                sy0.to_i32().unwrap(),
                sy1.to_i32().unwrap(),
                sy2.to_i32().unwrap(),
                sy3.to_i32().unwrap(),
                sy4.to_i32().unwrap(),
            ],
            [sy1, sy2, sy3, sy4] => [
                1,
                sy1.to_i32().unwrap(),
                sy2.to_i32().unwrap(),
                sy3.to_i32().unwrap(),
                sy4.to_i32().unwrap(),
            ],
            _ => unreachable!(),
        };
        let (x, offset_x) = x.as_raw_scalar_slice_offset();
        let offset_x = offset_x.to_u32().unwrap();
        let (y, offset_y) = y.as_raw_scalar_slice_offset_mut();
        let offset_y = offset_y.to_u32().unwrap();

        macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            if x.scalar_type() == $T::SCALAR_TYPE {
                let x = Slice::try_from(x).unwrap();
                let y = SliceMut::try_from(y).unwrap();
                let kernel = paste! {
                    kernels::[<sum_axis6_ $T>]::builder()?
                        .with_threads(threads)
                        .specialize(subgroups, subgroup_threads, axis)
                        .build(device)?
                };
                kernel.with_groups(groups).dispatch(
                    d0,
                    d1,
                    d2,
                    d3,
                    d4,
                    d5,
                    x,
                    sx0,
                    sx1,
                    sx2,
                    sx3,
                    sx4,
                    sx5,
                    offset_x,
                    beta.cast(),
                    y,
                    sy0,
                    sy1,
                    sy2,
                    sy3,
                    sy4,
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
    #[cfg(target_arch = "spirv")]
    use crunchy::unroll;
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;
    #[cfg(target_arch = "spirv")]
    use krnl_core::{
        buffer::UnsafeIndex,
        half::{bf16, f16},
        scalar::Scalar,
    };
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

    #[cfg(target_arch = "spirv")]
    fn remove_from_array4<T: Copy + Default>(x: [T; 4], index: usize) -> [T; 3] {
        assert!(index < x.len());
        let mut y = <[T; 3]>::default();
        let mut u = 0;
        unroll! { for i in 0 .. 4 {
            if i != index {
                y[u] = x[i];
                #[allow(unused_assignments)] {
                    u += 1;
                }
            }
        }}
        y
    }

    #[cfg(target_arch = "spirv")]
    fn remove_from_array6<T: Copy + Default>(x: [T; 6], index: usize) -> [T; 5] {
        assert!(index < x.len());
        let mut y = <[T; 5]>::default();
        let mut u = 0;
        unroll! { for i in 0 .. 6 {
            if i != index {
                y[u] = x[i];
                #[allow(unused_assignments)] {
                    u += 1;
                }
            }
        }}
        y
    }

    macro_rules! impl_sum {
        ($t:ty => $a:ty) => {
            paste! {
                #[kernel]
                pub fn [<sum_ $t>]<const SUBGROUP_THREADS: u32>(
                    #[global] x: Slice<$t>,
                    beta: $a,
                    #[global] y: UnsafeSlice<$t>,
                ) {
                    type T = $t;
                    type A = $a;
                    if SUBGROUP_THREADS == 0 && kernel.subgroup_id() > 0 {
                        return;
                    }
                    let subgroup_threads = if SUBGROUP_THREADS > 0 {
                        SUBGROUP_THREADS as usize
                    } else {
                        kernel.threads() / kernel.subgroups()
                    };
                    let mut y_thread = A::default();
                    let mut idx = 0;
                    let n = x.len() / y.len();
                    while idx < n {
                        let x_idx = idx + kernel.thread_id();
                        if x_idx < n {
                            y_thread += x[x_idx].cast::<A>();
                        }
                        idx += subgroup_threads;
                    }
                    unsafe {
                        y_thread = y_thread.subgroup_add();
                    };
                    if kernel.thread_id() == 0 {
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

                #[allow(clippy::too_many_arguments)]
                #[kernel]
                pub fn [<sum_axis2_ $t>]<const SUBGROUPS: u32, const SUBGROUP_THREADS: u32, const AXIS: u32>(
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
                    let subgroups = SUBGROUPS as usize;
                    if SUBGROUP_THREADS == 0 && kernel.subgroup_id() >= subgroups {
                        return;
                    }
                    let subgroup_threads = if SUBGROUP_THREADS > 0 {
                        SUBGROUP_THREADS as usize
                    } else {
                        kernel.threads() / kernel.subgroups()
                    };
                    let axis = AXIS as usize;
                    let global_subgroup_id = kernel.group_id() * subgroups + kernel.subgroup_id();
                     if global_subgroup_id >= y.len() {
                        return;
                    }
                    let n = [d0, d1][axis] as usize;
                    let stride_group = [sx1, sx0][axis];
                    let stride_axis = [sx0, sx1][axis];
                    let mut x_start = global_subgroup_id as i32 * stride_group + offset_x as i32;
                    let mut y_thread = A::default();
                    let mut idx = 0;
                    while idx < n {
                        let x_idx = idx + kernel.subgroup_thread_id();
                        if x_idx < n {
                            y_thread += x[(x_start + x_idx as i32 * stride_axis) as usize].cast::<A>();
                        }
                        idx += subgroup_threads;
                    }
                    unsafe {
                        y_thread = y_thread.subgroup_add();
                    };
                    let y_idx = (global_subgroup_id as i32 * sy0 + offset_y as i32) as usize;
                    if kernel.subgroup_thread_id() == 0 {
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

                #[allow(clippy::too_many_arguments)]
                #[kernel]
                pub fn [<sum_axis4_ $t>]<const SUBGROUPS: u32, const SUBGROUP_THREADS: u32, const AXIS: u32>(
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
                    let subgroups = SUBGROUPS as usize;
                    if SUBGROUP_THREADS == 0 && kernel.subgroup_id() >= subgroups {
                        return;
                    }
                    let subgroup_threads = if SUBGROUP_THREADS > 0 {
                        SUBGROUP_THREADS as usize
                    } else {
                        kernel.threads() / kernel.subgroups()
                    };
                    let axis = AXIS as usize;
                    let global_subgroup_id = kernel.group_id() * subgroups + kernel.subgroup_id();
                     if global_subgroup_id >= y.len() {
                        return;
                    }
                    let n = [d0, d1, d2, d3][axis] as usize;
                    let stride_axis = [sx0, sx1, sx2, sx3][axis];
                    let [gd0, gd1, gd2] = remove_from_array4([d0, d1, d2, d3], axis);
                    let [sg0, sg1, sg2] = remove_from_array4([sx0, sx1, sx2, sx3], axis);
                    let [i0, i1, i2] = {
                        let global_subgroup_id = global_subgroup_id as u32;
                        let i0 = global_subgroup_id / (gd1 * gd2);
                        let r0 = global_subgroup_id % (gd1 * gd2);
                        let i1 = r0 / gd2;
                        let i2 = r0 % gd2;
                        [i0 as i32, i1 as i32, i2 as i32]
                    };
                    let mut x_start = i0 * sg0 + i1 * sg1 + i2 * sg2 + offset_x as i32;
                    let mut y_thread = A::default();
                    let mut idx = 0;
                    while idx < n {
                        let x_idx = idx + kernel.subgroup_thread_id();
                        if x_idx < n {
                            y_thread += x[(x_start + x_idx as i32 * stride_axis) as usize].cast::<A>();
                        }
                        idx += subgroup_threads;
                    }
                    unsafe {
                        y_thread = y_thread.subgroup_add();
                    };
                    let y_idx = (i0 * sy0 + i1 * sy1 + i2 * sy2 + offset_y as i32) as usize;
                    if kernel.subgroup_thread_id() == 0 {
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

                #[allow(clippy::too_many_arguments)]
                #[kernel]
                pub fn [<sum_axis6_ $t>]<const SUBGROUPS: u32, const SUBGROUP_THREADS: u32, const AXIS: u32>(
                    d0: u32,
                    d1: u32,
                    d2: u32,
                    d3: u32,
                    d4: u32,
                    d5: u32,
                    #[global] x: Slice<$t>,
                    sx0: i32,
                    sx1: i32,
                    sx2: i32,
                    sx3: i32,
                    sx4: i32,
                    sx5: i32,
                    offset_x: u32,
                    beta: $a,
                    #[global] y: UnsafeSlice<$t>,
                    sy0: i32,
                    sy1: i32,
                    sy2: i32,
                    sy3: i32,
                    sy4: i32,
                    offset_y: u32,
                ) {
                    type T = $t;
                    type A = $a;
                    let subgroups = SUBGROUPS as usize;
                    if SUBGROUP_THREADS == 0 && kernel.subgroup_id() >= subgroups {
                        return;
                    }
                    let subgroup_threads = if SUBGROUP_THREADS > 0 {
                        SUBGROUP_THREADS as usize
                    } else {
                        kernel.threads() / kernel.subgroups()
                    };
                    let axis = AXIS as usize;
                    let global_subgroup_id = kernel.group_id() * subgroups + kernel.subgroup_id();
                     if global_subgroup_id >= y.len() {
                        return;
                    }
                    let n = [d0, d1, d2, d3, d4, d5][axis] as usize;
                    let stride_axis = [sx0, sx1, sx2, sx3, sx4, sx5][axis];
                    let [gd0, gd1, gd2, gd3, gd4] = remove_from_array6([d0, d1, d2, d3, d4, d5], axis);
                    let [sg0, sg1, sg2, sg3, sg4] = remove_from_array6([sx0, sx1, sx2, sx3, sx4, sx5], axis);
                    let [i0, i1, i2, i3, i4] = {
                        let global_subgroup_id = global_subgroup_id as u32;
                        let i0 = global_subgroup_id / (gd1 * gd2 * gd3 * gd4);
                        let r0 = global_subgroup_id % (gd1 * gd2 * gd3 * gd4);
                        let i1 = r0 / (gd2 * gd3 * gd4);
                        let r1 = r0 % (gd2 * gd3 * gd4);
                        let i2 = r1 / (gd3 * gd4);
                        let r2 = r1 % (gd3 * gd4);
                        let i3 = r2 / gd4;
                        let i4 = r2 % gd4;
                        [i0 as i32, i1 as i32, i2 as i32, i3 as i32, i4 as i32]
                    };
                    let mut x_start = i0 * sg0 + i1 * sg1 + i2 * sg2 + i3 * sg3 + i4 * sg4 + offset_x as i32;
                    let mut y_thread = A::default();
                    let mut idx = 0;
                    while idx < n {
                        let x_idx = idx + kernel.subgroup_thread_id();
                        if x_idx < n {
                            y_thread += x[(x_start + x_idx as i32 * stride_axis) as usize].cast::<A>();
                        }
                        idx += subgroup_threads;
                    }
                    unsafe {
                        y_thread = y_thread.subgroup_add();
                    };
                    let y_idx = (i0 * sy0 + i1 * sy1 + i2 * sy2 + i3 * sy3 + i4 * sy4 + offset_y as i32) as usize;
                    if kernel.subgroup_thread_id() == 0 {
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

    impl_sum!(u8, u16 => u32);
    impl_sum!(i8, i16 => i32);
    impl_sum!(f16, bf16 => f32);
    impl_sum!(u32, i32, f32, u64, i64, f64);
}
