use super::*;
use crate::ops::AddAssign;
#[cfg(feature = "device")]
use anyhow::format_err;
use dry::macro_for;
use half::{bf16, f16};
use krnl::krnl_core::num_traits::ToPrimitive;
#[cfg(feature = "device")]
use krnl::macros::module;

impl<S: ScalarData, D: Dimension> ScalarTensorBase<S, D> {
    /*
    /// Converts to standard layout.
    ///
    /// If in standard layout, borrows the tensor. Otherwise, copies into a new standard layout tensor.
    ///
    /// **Errors**
    /// See [`.into_standard_layout()`](TensorBase::into_standard_layout()).
    pub fn as_standard_layout(&self) -> Result<ScalarCowTensor<D>> {
        if self.is_standard_layout() {
            Ok(self.view().into())
        } else {
            self.view().into_standard_layout().map(Into::into)
        }
    }
    /// Converts into standard layout.
    ///
    /// If in standard layout, converts into an owned [`Tensor`]. Otherwise, copies the data into a new standard layout tensor.
    ///
    /// **Errors**
    /// - Supports 2 dimensional inputs.
    /// - [`DeviceLost`]
    /// - The kernel could not be dispatched.
    /// - See [`.into_owned()`](TensorBase::into_owned()).
    ///
    /// **Panics**
    /// Only u32, i32, and f32 are currently implemented.
    pub fn into_standard_layout(self) -> Result<ScalarTensor<D>> {
        if self.is_standard_layout() {
            self.into_owned()
        } else {
            let mut output =
                unsafe { ScalarTensor::uninit(self.device(), self.raw_dim(), self.scalar_type())? };
            output.assign(&self)?;
            Ok(output)
        }
    }
    /// Converts to an [`ArcTensor`] in standard layout.
    ///
    /// If in standard layout, converts to an [`ArcTensor`] (or clones the [`ArcTensor`]), otherwise copies the data into a new [`ArcTensor`].
    pub fn to_standard_layout_shared(&self) -> Result<ScalarArcTensor<D>> {
        if self.is_standard_layout() {
            self.to_shared()
        } else {
            self.as_standard_layout()?.into_shared()
        }
    }*/
    pub fn scaled_add<S2, D2>(
        &mut self,
        alpha: ScalarElem,
        rhs: &ScalarTensorBase<S2, D2>,
    ) -> Result<()>
    where
        S: ScalarDataMut,
        S2: ScalarData,
        D2: Dimension,
    {
        scalar_assign(
            BinaryOp::Add,
            alpha,
            rhs.view().into_dyn(),
            self.view_mut().into_dyn(),
        )
    }
    pub fn scaled_cast(&self, alpha: ScalarElem) -> Result<ScalarTensor<D>> {
        let mut output =
            unsafe { ScalarTensor::uninit(self.device(), self.raw_dim(), alpha.scalar_type())? };
        scalar_assign(
            BinaryOp::Identity,
            alpha,
            self.view().into_dyn(),
            output.view_mut().into_dyn(),
        )?;
        Ok(output)
    }
    pub fn assign<S2, D2>(&mut self, rhs: &ScalarTensorBase<S2, D2>) -> Result<()>
    where
        S: ScalarDataMut,
        S2: ScalarData,
        D2: Dimension,
    {
        scalar_assign(
            BinaryOp::Identity,
            ScalarElem::one(self.scalar_type()),
            rhs.view().into_dyn(),
            self.view_mut().into_dyn(),
        )
    }
}

impl<S: ScalarDataMut, D: Dimension, S2: ScalarData, D2: Dimension>
    AddAssign<ScalarTensorBase<S2, D2>> for ScalarTensorBase<S, D>
{
    fn add_assign(&mut self, rhs: ScalarTensorBase<S2, D2>) -> Result<()> {
        self.scaled_add(ScalarElem::one(self.scalar_type()), &rhs)
    }
}

impl<S: ScalarDataMut, D: Dimension, S2: ScalarData, D2: Dimension>
    AddAssign<&ScalarTensorBase<S2, D2>> for ScalarTensorBase<S, D>
{
    fn add_assign(&mut self, rhs: &ScalarTensorBase<S2, D2>) -> Result<()> {
        self.scaled_add(ScalarElem::one(self.scalar_type()), rhs)
    }
}

impl<T: Scalar, S: Data<Elem = T>, D: Dimension> TensorBase<S, D> {
    /// Converts to standard layout.
    ///
    /// If in standard layout, borrows the tensor. Otherwise, copies into a new standard layout tensor.
    ///
    /// **Errors**
    /// See [`.into_standard_layout()`](TensorBase::into_standard_layout()).
    pub fn as_standard_layout(&self) -> Result<CowTensor<T, D>> {
        if self.is_standard_layout() {
            Ok(self.view().into())
        } else {
            self.view().into_standard_layout().map(Into::into)
        }
    }
    /// Converts into standard layout.
    ///
    /// If in standard layout, converts into an owned [`Tensor`]. Otherwise, copies the data into a new standard layout tensor.
    ///
    /// **Errors**
    /// - Supports 2 dimensional inputs.
    /// - [`DeviceLost`]
    /// - The kernel could not be dispatched.
    /// - See [`.into_owned()`](TensorBase::into_owned()).
    ///
    /// **Panics**
    /// Only u32, i32, and f32 are currently implemented.
    pub fn into_standard_layout(self) -> Result<Tensor<T, D>> {
        if self.is_standard_layout() {
            self.into_owned()
        } else {
            let mut output = unsafe { Tensor::uninit(self.device(), self.raw_dim())? };
            output.assign(&self)?;
            Ok(output)
        }
    }
    /// Converts to an [`ArcTensor`] in standard layout.
    ///
    /// If in standard layout, converts to an [`ArcTensor`] (or clones the [`ArcTensor`]), otherwise copies the data into a new [`ArcTensor`].
    pub fn to_standard_layout_shared(&self) -> Result<ArcTensor<T, D>> {
        if self.is_standard_layout() {
            self.to_shared()
        } else {
            self.as_standard_layout()?.into_shared()
        }
    }
    pub fn scaled_add<S2, D2>(&mut self, alpha: T, rhs: &TensorBase<S2, D2>) -> Result<()>
    where
        S: DataMut,
        S2: Data<Elem = T>,
        D2: Dimension,
    {
        assign(
            BinaryOp::Add,
            alpha,
            rhs.view().into_dyn(),
            self.view_mut().into_dyn(),
        )
    }
    pub fn assign<S2, D2>(&mut self, rhs: &TensorBase<S2, D2>) -> Result<()>
    where
        S: DataMut,
        S2: Data<Elem = T>,
        D2: Dimension,
    {
        assign(
            BinaryOp::Identity,
            T::one(),
            rhs.view().into_dyn(),
            self.view_mut().into_dyn(),
        )
    }
}

impl<T: Scalar, S: DataMut<Elem = T>, D: Dimension, S2: Data<Elem = T>, D2: Dimension>
    AddAssign<TensorBase<S2, D2>> for TensorBase<S, D>
{
    fn add_assign(&mut self, rhs: TensorBase<S2, D2>) -> Result<()> {
        self.scaled_add(T::one(), &rhs)
    }
}

impl<T: Scalar, S: DataMut<Elem = T>, D: Dimension, S2: Data<Elem = T>, D2: Dimension>
    AddAssign<&TensorBase<S2, D2>> for TensorBase<S, D>
{
    fn add_assign(&mut self, rhs: &TensorBase<S2, D2>) -> Result<()> {
        self.scaled_add(T::one(), rhs)
    }
}

fn assign<X: Scalar, Y: Scalar>(
    op: BinaryOp,
    alpha: Y,
    x: TensorViewD<X>,
    mut y: TensorViewMutD<Y>,
) -> Result<()> {
    let (x, mut y) = if let Some((x, y)) = x.as_array().zip(y.as_array_mut()) {
        (x, y)
    } else {
        return scalar_assign(op, alpha.into(), x.into(), y.into());
    };
    let x = if let Some(x) = x.broadcast(y.shape()) {
        x
    } else {
        bail!("Broadcast not possible! {x:?} -> {y:?}");
    };
    y.zip_mut_with(&x, |y, x| {
        *y = op.eval(alpha * x.cast(), y.cast()).cast();
    });
    Ok(())
}

fn scalar_assign(
    op: BinaryOp,
    alpha: ScalarElem,
    x: ScalarTensorViewD,
    mut y: ScalarTensorViewMutD,
) -> Result<()> {
    if let BinaryOp::Identity = op {
        if x.device() != y.device() {
            if let Some((x, mut y)) = x.as_scalar_slice().zip(y.as_scalar_slice_mut()) {
                return y.copy_from_scalar_slice(&x);
            } else {
                todo!()
            }
        }
    }
    if alpha.scalar_type() != y.scalar_type() {
        bail!(
            "alpha scalar_type {:?} != {:?}",
            alpha.scalar_type(),
            y.scalar_type()
        );
    }
    let x = if let Some(x) = x.broadcast(y.shape()) {
        x
    } else {
        bail!("Broadcast not possible! {x:?} -> {y:?}");
    };
    let device = y.device();
    if device.is_host() && x.device().is_host() {
        macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            if let Ok(x) = TensorViewD::<$X>::try_from(x.clone()) {
                macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                    if let Ok(y) = TensorViewMutD::<$Y>::try_from(y.view_mut()) {
                        let alpha = $Y::try_from(alpha).unwrap();
                        return assign(op, alpha, x, y);
                    }
                });
            }
        });
        bail!(
            "assign<{}, {}> not implemented!",
            x.scalar_type().name(),
            y.scalar_type().name(),
        );
    }
    #[cfg(not(feature = "device"))]
    {
        unreachable!()
    }
    #[cfg(feature = "device")]
    {
        let ndim = y.ndim();
        if ndim <= 2 {
            let (rows, cols) = match y.shape() {
                [rows, cols] => (rows.to_u32().unwrap(), cols.to_u32().unwrap()),
                [cols] => (1, cols.to_u32().unwrap()),
                [] => (1, 1),
                _ => unreachable!(),
            };
            let (rsx, csx) = match x.strides() {
                [rsx, csx] => (rsx.to_i32().unwrap(), csx.to_i32().unwrap()),
                [csx] => (1, csx.to_i32().unwrap()),
                [] => (1, 1),
                _ => unreachable!(),
            };
            let (rsy, csy) = match y.strides() {
                [rsy, csy] => (rsy.to_i32().unwrap(), csy.to_i32().unwrap()),
                [csy] => (1, csy.to_i32().unwrap()),
                [] => (1, 1),
                _ => unreachable!(),
            };
            let (x, offset_x) = x.as_raw_scalar_slice_offset();
            let offset_x = offset_x.to_u32().unwrap();
            let (mut y, offset_y) = y.as_raw_scalar_slice_offset_mut();
            let offset_y = offset_y.to_u32().unwrap();

            macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                if let Ok(x) = Slice::<$X>::try_from(x.clone()) {
                    macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                        if let Ok(y) = SliceMut::<$Y>::try_from(y.as_scalar_slice_mut()) {
                            let builder = paste! {
                                kernels::[<assign2_ $X _ $Y>]::builder()?
                            };
                            builder
                            .specialize(op.as_u32())?
                            .build(device)?
                            .with_global_threads([cols, rows].into())
                            .dispatch(
                                rows,
                                cols,
                                alpha.cast(),
                                x,
                                rsx,
                                csx,
                                offset_x,
                                y,
                                rsy,
                                csy,
                                offset_y,
                            )?;
                            return Ok(());
                        }
                    });
                }
            });
        }
        Err(format_err!(
            "assign<{}, {}> not implemented!",
            x.scalar_type().name(),
            y.scalar_type().name()
        ))
    }
}

#[allow(unused_imports)]
#[cfg_attr(feature = "device", module)]
mod kernels {
    use dry::macro_for;
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;
    #[cfg(target_arch = "spirv")]
    use krnl_core::{buffer::UnsafeIndex, glam::Vec2Swizzles};
    use krnl_core::{
        half::{bf16, f16},
        scalar::Scalar,
    };
    use paste::paste;

    #[repr(u32)]
    pub enum BinaryOp {
        Identity = 1,
        Add = 2,
        Sub = 3,
        Mul = 4,
        Div = 5,
    }

    #[cfg(not(target_arch = "spirv"))]
    impl BinaryOp {
        pub fn as_u32(self) -> u32 {
            self as u32
        }
    }

    impl TryFrom<u32> for BinaryOp {
        type Error = ();
        fn try_from(x: u32) -> Result<Self, ()> {
            Ok(match x {
                1 => Self::Identity,
                2 => Self::Add,
                3 => Self::Sub,
                4 => Self::Mul,
                5 => Self::Div,
                _ => {
                    return Err(());
                }
            })
        }
    }

    impl BinaryOp {
        pub fn eval<T: Scalar>(&self, a: T, b: T) -> T {
            match self {
                Self::Identity => a,
                Self::Add => a + b,
                Self::Sub => a - b,
                Self::Mul => a * b,
                Self::Div => a / b,
            }
        }
    }

    #[cfg(any(target_arch = "spirv", feature = "device"))]
    macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
        macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            paste! {
                #[kernel(threads(16, 16))]
                pub fn [<assign2_ $X _ $Y>]<const OP: u32>(
                    rows: u32,
                    cols: u32,
                    alpha: $Y,
                    #[global]
                    x: Slice<$X>,
                    rsx: i32,
                    csx: i32,
                    offset_x: u32,
                    #[global]
                    y: UnsafeSlice<$Y>,
                    rsy: i32,
                    csy: i32,
                    offset_y: u32,
                ) {
                    let op = BinaryOp::try_from(OP).ok().unwrap();
                    let [global_row, global_col] = kernel.global_id().yx().to_array();
                    if global_row < rows && global_col < cols {
                        let x_idx = (global_row as i32 * rsx + global_col as i32 * csx + offset_x as i32) as usize;
                        let y_idx = (global_row as i32 * rsy + global_col as i32 * csy + offset_y as i32) as usize;
                        unsafe {
                            *y.unsafe_index_mut(y_idx) = op.eval(alpha * x[x_idx].cast::<$Y>(), *y.unsafe_index(y_idx));
                        }
                    }
                }
            }
        });
    });
}
use kernels::BinaryOp;

#[cfg(all(test, feature = "device_tests"))]
mod tests {
    use super::*;

    async fn into_standard_layout<T: Scalar, E: IntoDimension>(shape: E, axes: E) -> Result<()> {
        let shape = shape.into_dimension();
        let x_vec = (0..shape.size())
            .into_iter()
            .map(|x| T::from_usize(x).unwrap())
            .collect();
        let x_array = Array::from_shape_vec(shape, x_vec)?;
        let axes = E::Dim::from_dimension(&axes.into_dimension()).unwrap();
        let y_array = x_array
            .view()
            .permuted_axes(axes.clone())
            .as_standard_layout()
            .to_owned();
        let device = Device::new()?;
        let x = Tensor::from(x_array).into_device(device).await?;
        let y = x.permuted_axes(axes).into_standard_layout()?.read().await?;
        assert_eq!(y_array.view(), y.as_array());
        Ok(())
    }

    #[tokio::test]
    async fn into_standard_layout_4d_u32() -> Result<()> {
        into_standard_layout::<u32, _>([2, 12, 12, 16], [0, 3, 1, 2]).await?;
        Ok(())
    }

    #[tokio::test]
    async fn into_standard_layout_4d_f32() -> Result<()> {
        into_standard_layout::<f32, _>([1000, 24, 24, 6], [0, 3, 1, 2]).await?;
        into_standard_layout::<f32, _>([1000, 8, 8, 16], [0, 3, 1, 2]).await?;
        Ok(())
    }

    /*#[tokio::test]
    async fn into_standard_layout_6d_u32() -> Result<()> {
        into_standard_layout::<u32, _>([2, 3, 4, 5, 6, 7]).await?;
        Ok(())
    }*/

    async fn reorder<T1, Sh1, T2, Sh2>(input_shape: Sh1, output_shape: Sh2) -> Result<()>
    where
        T1: Scalar,
        Sh1: ShapeBuilder,
        T2: Scalar,
        Sh2: ShapeBuilder<Dim = Sh1::Dim> + Clone,
    {
        let mut x_array = Array::<T1, _>::zeros(input_shape);
        for (x, i) in x_array.iter_mut().zip(1..) {
            *x = T1::from_usize(i).unwrap();
        }
        let mut y_array = Array::<T2, _>::zeros(output_shape.clone());
        for (indices, x) in x_array.indexed_iter() {
            if let Some(y) = y_array.get_mut(indices.into_dimension()) {
                *y = T2::from_usize(x.to_usize().unwrap()).unwrap();
            }
        }

        let device = Device::new()?;
        let x = CowTensor::from(x_array.view())
            .into_device(device.clone())
            .await?;
        let y = x
            .as_reordered::<T2, _>(output_shape)?
            .read()
            .await?
            .as_array()
            .to_owned();
        assert_eq!(y, y_array);
        Ok(())
    }

    #[tokio::test]
    async fn reorder_2d_f32_f32() -> Result<()> {
        reorder::<f32, _, f32, _>([3, 3], [4, 4].set_f(true)).await?;
        Ok(())
    }
}
