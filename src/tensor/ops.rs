use super::parallel::parallel_size;
use super::*;
use crate::ops::AddAssign;
#[cfg(feature = "neural-network")]
use crate::ops::{
    Col2ImConv2, Col2ImConv2Options, Im2ColConv2, Im2ColConv2Options, MaxPool2, MaxPool2Backward,
    MaxPool2Options,
};
#[cfg(feature = "device")]
use anyhow::format_err;
#[cfg(feature = "neural-network")]
use dry::{macro_for, macro_wrap};
#[allow(unused_imports)]
use half::{bf16, f16};
#[cfg(feature = "device")]
use krnl::macros::module;
#[cfg(feature = "neural-network")]
use ndarray::{Array2, Array4, Data as ArrayData, DataMut as ArrayDataMut};
use num_traits::Unsigned;
use std::mem::size_of;

impl<S: ScalarData, D: Dimension> ScalarTensorBase<S, D> {
    /// Converts to standard layout.
    ///
    /// If in standard layout, borrows the tensor. Otherwise, copies into a new standard layout tensor.
    ///
    /// # Errors
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
    /// # Errors
    /// See [`.into_standard_layout()`](TensorBase::into_standard_layout()).
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
    ///
    /// # Errors
    /// See [`.into_standard_layout()`](TensorBase::into_standard_layout()).
    pub fn to_standard_layout_shared(&self) -> Result<ScalarArcTensor<D>> {
        if self.is_standard_layout() {
            self.to_shared()
        } else {
            self.as_standard_layout()?.into_shared()
        }
    }
    /// Performs the operation `self += alpha * rhs`.
    ///
    /// Broadcasts `rhs` to the shape of `self`.
    ///
    /// # Errors
    /// - Broadcasting is not possible.
    /// - The operation could not be executed on the device.
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
    /// Performs the operation `self as _ * alpha`.
    ///
    /// # Errors
    /// The operation could not be executed on the device.
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
    /// Copies `rhs` to `self`.
    ///
    /// Broadcasts `rhs` to shape of `self`.
    ///
    /// # Errors
    /// - Broadcasting is not possible.
    /// - The operation could not be executed on the device.
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
    /// # Errors
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
    /// # Errors
    /// - On device, supports up to 6 dimensional inputs.
    /// - [`DeviceLost`]
    /// - The kernel could not be dispatched.
    /// See [`.into_owned()`](TensorBase::into_owned()).
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
    /// Performs the operation `self += alpha * rhs`.
    ///
    /// Broadcasts `rhs` to the shape of `self`.
    ///
    /// # Errors
    /// - Broadcasting is not possible.
    /// - The operation could not be executed on the device.
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
    /// Performs the operation `self as T2 * alpha`.
    ///
    /// # Errors
    /// - The operation could not be executed on the device.
    pub fn scaled_cast<T2: Scalar>(&self, alpha: T2) -> Result<Tensor<T2, D>> {
        let mut output = unsafe { Tensor::<T2, D>::uninit(self.device(), self.raw_dim())? };
        assign(
            BinaryOp::Identity,
            alpha,
            self.view().into_dyn(),
            output.view_mut().into_dyn(),
        )?;
        Ok(output)
    }
    /// Copies `rhs` to `self`.
    ///
    /// Broadcasts `rhs` to shape of `self`.
    ///
    /// # Errors
    /// - Broadcasting is not possible.
    /// - The operation could not be executed on the device.
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
    #[cfg(feature = "device")]
    {
        if matches!(op, BinaryOp::Identity)
            && x.device() != y.device()
            && alpha == Y::one()
            && x.strides() == y.strides()
            && X::SCALAR_TYPE == Y::SCALAR_TYPE
        {
            if let Some((x, mut y)) = x.as_slice_memory_order().zip(y.as_slice_memory_order_mut()) {
                return y.copy_from_slice(&x.as_scalar_slice().try_into().unwrap());
            }
        }
    }

    let (x, mut y) = if let Some((x, y)) = x.as_array().zip(y.as_array_mut()) {
        (x, y)
    } else {
        return scalar_assign(op, alpha.into(), x.into(), y.into());
    };
    let x = if let Some(x) = x.broadcast(y.raw_dim()) {
        x
    } else {
        bail!("Broadcast not possible! {x:?} -> {y:?}");
    };

    use ndarray::Zip;
    use rayon::iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        IntoParallelRefMutIterator, ParallelIterator,
    };

    let parallel = (x.len() * size_of::<X>() + y.len() * size_of::<Y>()) > parallel_size();
    let eval = |(x, y): (&X, &mut Y)| *y = op.eval(alpha * x.cast(), *y);
    if x.strides() == y.strides()
        && x.as_slice_memory_order().is_some()
        && y.as_slice_memory_order_mut().is_some()
    {
        {
            let x = x.as_slice_memory_order().unwrap();
            let y = y.as_slice_memory_order_mut().unwrap();
            if parallel {
                x.par_iter().zip(y.par_iter_mut()).for_each(eval);
            } else {
                x.iter().zip(y.iter_mut()).for_each(eval);
            }
        }
    } else {
        let zip = Zip::from(&x).and(&mut y);
        if parallel {
            zip.into_par_iter().for_each(eval);
        } else {
            zip.for_each(|a, b| eval((a, b)));
        }
    }
    Ok(())
}

fn scalar_assign(
    op: BinaryOp,
    alpha: ScalarElem,
    x: ScalarTensorViewD,
    mut y: ScalarTensorViewMutD,
) -> Result<()> {
    if alpha.scalar_type() != y.scalar_type() {
        bail!(
            "alpha scalar_type {:?} != {:?}",
            alpha.scalar_type(),
            y.scalar_type()
        );
    }
    if op.is_identity()
        && alpha == ScalarElem::zero(y.scalar_type())
        && x.scalar_type() == y.scalar_type()
        && x.strides() == y.strides()
    {
        if let Some((x, mut y)) = x
            .as_scalar_slice_memory_order()
            .zip(y.as_scalar_slice_memory_order_mut())
        {
            return y.copy_from_scalar_slice(&x);
        }
    }
    let device = y.device();
    if device.is_host() && x.device().is_host() {
        macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            if $X::SCALAR_TYPE == x.scalar_type() {
                macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                    if $Y::SCALAR_TYPE == y.scalar_type() {
                        let alpha = $Y::try_from(alpha).unwrap();
                        let x = x.try_into_tensor_view::<$X>().unwrap();
                        let y = y.try_into_tensor_view_mut::<$Y>().unwrap();
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
        let x = if let Some(x) = x.broadcast(y.raw_dim()) {
            x
        } else {
            bail!("Broadcast not possible! {x:?} -> {y:?}");
        };
        if x.strides() == y.strides() {
            if let Some((x, mut y)) = x
                .as_scalar_slice_memory_order()
                .zip(y.as_scalar_slice_memory_order_mut())
            {
                macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                    if let Ok(x) = Slice::<$X>::try_from(x.clone()) {
                        macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                            if let Ok(y) = SliceMut::<$Y>::try_from(y.as_scalar_slice_mut()) {
                                let builder = paste! {
                                    kernels::[<assign_ $X _ $Y>]::builder()?
                                };
                                builder
                                .specialize(op.as_u32())
                                .build(device)?
                                .dispatch(
                                    alpha.cast(),
                                    x,
                                    y,
                                )?;
                                return Ok(());
                            }
                        });
                    }
                });
            }
        }
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
                            .specialize(op.as_u32())
                            .build(device)?
                            .with_global_threads(rows * cols)
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
        if ndim == 3 || ndim == 4 {
            let [d0, d1, d2, d3] = match y.shape() {
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
                    (d1 * d2 * d3) as i32,
                    sx1.to_i32().unwrap(),
                    sx2.to_i32().unwrap(),
                    sx3.to_i32().unwrap(),
                ],
                _ => unreachable!(),
            };
            let [sy0, sy1, sy2, sy3] = match y.strides() {
                [sy0, sy1, sy2, sy3] => [
                    sy0.to_i32().unwrap(),
                    sy1.to_i32().unwrap(),
                    sy2.to_i32().unwrap(),
                    sy3.to_i32().unwrap(),
                ],
                [sy1, sy2, sy3] => [
                    (d1 * d2 * d3) as i32,
                    sy1.to_i32().unwrap(),
                    sy2.to_i32().unwrap(),
                    sy3.to_i32().unwrap(),
                ],
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
                                kernels::[<assign4_ $X _ $Y>]::builder()?
                            };
                            let kernel = builder
                                .specialize(op.as_u32())
                                .build(device)?
                                .with_global_threads(d0 * d1 * d2 * d3);
                            unsafe {
                                kernel.dispatch(
                                    d0,
                                    d1,
                                    d2,
                                    d3,
                                    alpha.cast(),
                                    x,
                                    sx0,
                                    sx1,
                                    sx2,
                                    sx3,
                                    offset_x,
                                    y,
                                    sy0,
                                    sy1,
                                    sy2,
                                    sy3,
                                    offset_y,
                                )?;
                            }
                            return Ok(());
                        }
                    });
                }
            });
        }
        if ndim == 5 || ndim == 6 {
            let [d0, d1, d2, d3, d4, d5] = match y.shape() {
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
                    (d1 * d2 * d3 * d4 * d5) as i32,
                    sx1.to_i32().unwrap(),
                    sx2.to_i32().unwrap(),
                    sx3.to_i32().unwrap(),
                    sx4.to_i32().unwrap(),
                    sx5.to_i32().unwrap(),
                ],
                _ => unreachable!(),
            };
            let [sy0, sy1, sy2, sy3, sy4, sy5] = match y.strides() {
                [sy0, sy1, sy2, sy3, sy4, sy5] => [
                    sy0.to_i32().unwrap(),
                    sy1.to_i32().unwrap(),
                    sy2.to_i32().unwrap(),
                    sy3.to_i32().unwrap(),
                    sy4.to_i32().unwrap(),
                    sy5.to_i32().unwrap(),
                ],
                [sy1, sy2, sy3, sy4, sy5] => [
                    (d1 * d2 * d3 * d4 * d5) as i32,
                    sy1.to_i32().unwrap(),
                    sy2.to_i32().unwrap(),
                    sy3.to_i32().unwrap(),
                    sy4.to_i32().unwrap(),
                    sy5.to_i32().unwrap(),
                ],
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
                                kernels::[<assign6_ $X _ $Y>]::builder()?
                            };
                            let kernel = builder
                                .specialize(op.as_u32())
                                .build(device)?
                                .with_global_threads(d0 * d1 * d2 * d3 * d4 * d5);
                            unsafe {
                                kernel.dispatch(
                                    d0,
                                    d1,
                                    d2,
                                    d3,
                                    d4,
                                    d5,
                                    alpha.cast(),
                                    x,
                                    sx0,
                                    sx1,
                                    sx2,
                                    sx3,
                                    sx4,
                                    sx5,
                                    offset_x,
                                    y,
                                    sy0,
                                    sy1,
                                    sy2,
                                    sy3,
                                    sy4,
                                    sy5,
                                    offset_y,
                                )?;
                            }
                            return Ok(());
                        }
                    });
                }
            });
        }
        Err(format_err!(
            "assign{}<{}, {}> not implemented!",
            ndim,
            x.scalar_type().name(),
            y.scalar_type().name()
        ))
    }
}

impl<S: ScalarData, D: Dimension> ScalarTensorBase<S, D> {
    /// A one hot vector given class labels.
    ///
    /// See [`TensorBase::to_one_hot`].
    pub fn to_one_hot(
        &self,
        classes: usize,
        scalar_type: ScalarType,
    ) -> Result<ScalarTensor<D::Larger>> {
        macro_for!($X in [u8, u16, u32, u64] {
            macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                if self.scalar_type() == $X::SCALAR_TYPE && scalar_type == $Y::SCALAR_TYPE {
                    let input = self.view().try_into_tensor_view::<$X>().unwrap();
                    let output = input.to_one_hot::<$Y>(classes)?;
                    return Ok(output.into());
                }
            });
        });
        bail!(
            "to_one_hot {:?} {:?} unimplemented!",
            self.scalar_type(),
            scalar_type
        );
    }
}

impl<T: Scalar + Unsigned, S: Data<Elem = T>, D: Dimension> TensorBase<S, D> {
    /// A one hot vector given class labels.
    ///
    /// Output shape = [input_shape.., `classes`].
    pub fn to_one_hot<T2: Scalar>(&self, classes: usize) -> Result<Tensor<T2, D::Larger>> {
        let mut dim = D::Larger::zeros(self.dim.ndim() + 1);
        for (x, y) in self
            .shape()
            .iter()
            .copied()
            .chain([classes])
            .zip(dim.slice_mut())
        {
            *y = x;
        }
        if let Some(input) = self.as_array() {
            let mut output = Array::zeros(dim);
            for (x, y) in input
                .iter()
                .zip(output.as_slice_mut().unwrap().chunks_mut(classes))
            {
                y[x.to_usize().unwrap()] = T2::one();
            }
            return Ok(Tensor::from(output));
        }
        #[cfg(feature = "device")]
        let input = self.as_standard_layout()?;
        #[cfg(feature = "device")]
        macro_for!($X in [u8, u16, u32, u64] {
            if T::SCALAR_TYPE == $X::SCALAR_TYPE {
                let input = ScalarTensorView::from(input.view()).try_into_tensor_view::<$X>().unwrap();
                macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                    if T2::SCALAR_TYPE == $Y::SCALAR_TYPE {
                        let mut output = unsafe {
                            Tensor::<$Y, _>::uninit(self.device(), dim)?
                        };
                        let kernel = paste! {
                            kernels::[<one_hot_ $X _ $Y>]::builder()?.build(input.device())?
                        };
                        kernel.dispatch(input.as_slice().unwrap(), output.as_slice_mut().unwrap())?;
                        return Ok(output.cast_into().unwrap());
                    }
                });
            }
        });
        bail!(
            "to_one_hot {:?} {:?} unimplemented!",
            self.scalar_type(),
            T2::SCALAR_TYPE
        );
    }
}

#[cfg(feature = "neural-network")]
impl<T: Scalar, S: ArrayData<Elem = T>> Im2ColConv2 for ArrayBase<S, Ix4> {
    type Output = Array2<T>;
    fn im2col_conv2(&self, options: &Im2ColConv2Options) -> Result<Self::Output> {
        let input = self.view();
        let (bs, c, ih, iw) = input.dim();
        let [oh, ow] = options.output_shape([ih, iw]);
        let Im2ColConv2Options {
            filter: [fh, fw],
            padding: [ph, pw],
            stride: [sh, sw],
            dilation: [dh, dw],
        } = options.clone();
        let is_default_padding_stride_dilation =
            options.padding == [0, 0] && options.stride == [1, 1] && options.dilation == [1, 1];
        let mut output = unsafe { Array::<T, _>::uninit([bs, oh, ow, c, fh, fw]).assume_init() };
        {
            use crate::tensor::parallel::array_par_outer_iter_mut_for_each;

            if is_default_padding_stride_dilation {
                array_par_outer_iter_mut_for_each(output.view_mut(), |bid, mut output| {
                    let input = input.index_axis(Axis(0), bid);
                    for hid in 0..oh {
                        for wid in 0..ow {
                            for cid in 0..c {
                                for fi in 0..fh {
                                    let hidx = fi + hid;
                                    for fj in 0..fw {
                                        let widx = fj + wid;
                                        unsafe {
                                            *output.uget_mut([hid, wid, cid, fi, fj]) =
                                                *input.uget([cid, hidx, widx]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
            } else {
                array_par_outer_iter_mut_for_each(output.view_mut(), |bid, mut output| {
                    let input = input.index_axis(Axis(0), bid);
                    for hid in 0..oh {
                        for wid in 0..ow {
                            for cid in 0..c {
                                for fi in 0..fh {
                                    for fj in 0..fw {
                                        let hidx = -(ph as isize) + (fi * dh + sh * hid) as isize;
                                        let widx = -(pw as isize) + (fj * dw + sw * wid) as isize;
                                        let x = if hidx >= 0
                                            && hidx < ih as isize
                                            && widx >= 0
                                            && widx < iw as isize
                                        {
                                            unsafe {
                                                *input.uget([cid, hidx as usize, widx as usize])
                                            }
                                        } else {
                                            T::default()
                                        };
                                        unsafe {
                                            *output.uget_mut([hid, wid, cid, fi, fj]) = x;
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
            }
        }
        Ok(output.into_shape([bs * oh * ow, c * fh * fw]).unwrap())
    }
}

#[cfg(feature = "neural-network")]
impl<T: Scalar, S: Data<Elem = T>> Im2ColConv2 for TensorBase<S, Ix4> {
    type Output = Tensor2<T>;
    fn im2col_conv2(&self, options: &Im2ColConv2Options) -> Result<Self::Output> {
        if let Some(input) = self.as_array() {
            input.im2col_conv2(options).map(Into::into)
        } else {
            Ok(ScalarTensorView::from(self.view())
                .im2col_conv2(options)?
                .try_into_tensor()
                .unwrap())
        }
    }
}

#[cfg(feature = "neural-network")]
impl<S: ScalarData> Im2ColConv2 for ScalarTensorBase<S, Ix4> {
    type Output = ScalarTensor2;
    fn im2col_conv2(&self, options: &Im2ColConv2Options) -> Result<Self::Output> {
        macro_wrap!(
            paste! { #[allow(clippy::single_match)] match self.scalar_type() {
                macro_for!($T in [bf16, f32] {
                   ScalarType::[<$T:upper>] => {
                        let input = self.view().try_into_tensor_view::<$T>().unwrap();
                        if let Some(input) = input.as_array() {
                            return Ok(Tensor::from(input.im2col_conv2(options)?).into());
                        }
                        #[cfg(feature = "device")] {
                            let input = input.as_standard_layout()?;
                            let (bs, c, ih, iw) = input.dim();
                            let [oh, ow] = options.output_shape([ih, iw]);
                            let Im2ColConv2Options {
                                filter: [fh, fw],
                                padding: [ph, pw],
                                stride: [sh, sw],
                                dilation: [dh, dw],
                            } = options.clone();
                            let mut output = unsafe {
                                Tensor::<$T, _>::uninit(input.device(), [bs * oh * ow, c * fh * fw])?
                            };
                            neural_network_kernels::[<im2col_conv2_ $T>]::builder()?
                                .with_threads(256)
                                .specialize(
                                    bs.to_u32().unwrap(),
                                    c.to_u32().unwrap(),
                                    ih.to_u32().unwrap(),
                                    iw.to_u32().unwrap(),
                                    oh.to_u32().unwrap(),
                                    ow.to_u32().unwrap(),
                                    fh.to_u32().unwrap(),
                                    fw.to_u32().unwrap(),
                                    ph.to_u32().unwrap(),
                                    pw.to_u32().unwrap(),
                                    sh.to_u32().unwrap(),
                                    sw.to_u32().unwrap(),
                                    dh.to_u32().unwrap(),
                                    dw.to_u32().unwrap(),
                                )
                                .build(output.device())?
                                .with_global_threads(output.len().to_u32().unwrap())
                                .dispatch(
                                    input.as_slice().unwrap(),
                                    output.as_slice_mut().unwrap(),
                                )?;
                            return Ok(output.into());
                        }
                   }
                })
                _ => (),
            }}
        );
        bail!("im2col_conv2 {:?} unimplemented!()", self.scalar_type())
    }
}

#[cfg(feature = "neural-network")]
impl<T: Scalar, S: ArrayData<Elem = T>> Col2ImConv2 for ArrayBase<S, Ix2> {
    type Output = Array4<T>;
    fn col2im_conv2(&self, options: &Col2ImConv2Options) -> Result<Self::Output> {
        use crate::tensor::parallel::array_par_outer_iter_mut_for_each;

        let input = self.view();
        let (rows, cols) = input.dim();
        let [oh, ow] = options.output_shape();
        let Col2ImConv2Options {
            shape: [ih, iw],
            filter: [fh, fw],
            padding: [ph, pw],
            stride: [sh, sw],
            dilation: [dh, dw],
        } = options.clone();
        let is_default_padding_stride_dilation =
            options.padding == [0, 0] && options.stride == [1, 1] && options.dilation == [1, 1];
        let bs = rows / (ih * iw);
        let c = cols / (fh * fw);
        let input = input.into_shape([bs, ih, iw, c, fh, fw]).unwrap();
        let mut output = Array::zeros([bs, c, oh, ow]);
        if is_default_padding_stride_dilation {
            array_par_outer_iter_mut_for_each(output.view_mut(), |bid, mut output| {
                let input = input.index_axis(Axis(0), bid);
                for cid in 0..c {
                    for hid in 0..ih {
                        for wid in 0..iw {
                            for fi in 0..fh {
                                for fj in 0..fw {
                                    let hidy = fi + hid;
                                    let widy = fj + wid;
                                    unsafe {
                                        *output.uget_mut([cid, hidy, widy]) +=
                                            *input.uget([hid, wid, cid, fi, fj]);
                                    }
                                }
                            }
                        }
                    }
                }
            });
        } else {
            array_par_outer_iter_mut_for_each(output.view_mut(), |bid, mut output| {
                let input = input.index_axis(Axis(0), bid);
                for cid in 0..c {
                    for hid in 0..ih {
                        for wid in 0..iw {
                            for fi in 0..fh {
                                for fj in 0..fw {
                                    let hidy = -(ph as isize) + (fi * dh + sh * hid) as isize;
                                    let widy = -(pw as isize) + (fj * dw + sw * wid) as isize;
                                    if hidy >= 0
                                        && hidy < oh as isize
                                        && widy >= 0
                                        && widy < ow as isize
                                    {
                                        unsafe {
                                            *output.uget_mut([
                                                cid,
                                                hidy as usize,
                                                widy as usize,
                                            ]) += *input.uget([hid, wid, cid, fi, fj]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }
        Ok(output)
    }
}

#[cfg(feature = "neural-network")]
impl<T: Scalar, S: Data<Elem = T>> Col2ImConv2 for TensorBase<S, Ix2> {
    type Output = Tensor4<T>;
    fn col2im_conv2(&self, options: &Col2ImConv2Options) -> Result<Self::Output> {
        if let Some(input) = self.as_array() {
            input.col2im_conv2(options).map(Into::into)
        } else {
            Ok(ScalarTensorView::from(self.view())
                .col2im_conv2(options)?
                .try_into_tensor()
                .unwrap())
        }
    }
}

#[cfg(feature = "neural-network")]
impl<S: ScalarData> Col2ImConv2 for ScalarTensorBase<S, Ix2> {
    type Output = ScalarTensor4;
    fn col2im_conv2(&self, options: &Col2ImConv2Options) -> Result<Self::Output> {
        macro_wrap!(paste! { match self.scalar_type() {
            macro_for!($T in [bf16, f32] {
               $T::SCALAR_TYPE => {
                    let input = self.view().try_into_tensor_view::<$T>().unwrap();
                    if let Some(input) = input.as_array() {
                        return Ok(Tensor::from(input.col2im_conv2(options)?).into());
                    }
                    #[cfg(feature = "device")] {
                        let input = input.as_standard_layout()?;
                        let (rows, cols) = input.dim();
                        let [oh, ow] = options.output_shape();
                        let Col2ImConv2Options {
                            shape: [ih, iw],
                            filter: [fh, fw],
                            padding: [ph, pw],
                            stride: [sh, sw],
                            dilation: [dh, dw],
                        } = options.clone();
                        let bs = rows / (ih * iw);
                        let c = cols / (fh * fw);
                        let mut output = unsafe {
                            Tensor::<$T, _>::uninit(input.device(), [bs, c, oh, ow])?
                        };
                        neural_network_kernels::[<col2im_conv2_ $T>]::builder()?
                            .specialize(
                                c.to_u32().unwrap(),
                                ih.to_u32().unwrap(),
                                iw.to_u32().unwrap(),
                                oh.to_u32().unwrap(),
                                ow.to_u32().unwrap(),
                                fh.to_u32().unwrap(),
                                fw.to_u32().unwrap(),
                                ph.to_u32().unwrap(),
                                pw.to_u32().unwrap(),
                                sh.to_u32().unwrap(),
                                sw.to_u32().unwrap(),
                                dh.to_u32().unwrap(),
                                dw.to_u32().unwrap(),
                            )
                            .build(output.device())?
                            .dispatch(
                                input.as_slice().unwrap(),
                                 output.as_slice_mut().unwrap(),
                            )?;
                        return Ok(output.into());
                    }
               }
            })
            _ => (),
        }});
        bail!("im2col_conv2 {:?} unimplemented!()", self.scalar_type())
    }
}

#[cfg(feature = "neural-network")]
impl<T: Scalar, S: ArrayData<Elem = T>> MaxPool2 for ArrayBase<S, Ix4> {
    type Output = Array4<T>;
    fn max_pool2(&self, options: MaxPool2Options) -> Result<Self::Output> {
        let (bs, c, ih, iw) = self.dim();
        let [oh, ow] = options.output_shape([ih, iw]);
        let MaxPool2Options {
            size: [h, w],
            strides: [sh, sw],
        } = options;
        let input = self.view();
        let mut output = unsafe { Array::uninit([bs, c, oh, ow]).assume_init() };

        use rayon::prelude::*;

        let threads = rayon::current_num_threads();
        let chunk_size = bs / threads + (bs % threads != 0) as usize;

        output
            .axis_chunks_iter_mut(ndarray::Axis(0), chunk_size)
            .into_par_iter()
            .enumerate()
            .for_each(|(chunk_id, mut output)| {
                for bid_slice in 0..output.dim().0 {
                    let bid = chunk_id * chunk_size + bid_slice;
                    for cid in 0..c {
                        for hid in 0..oh {
                            for wid in 0..ow {
                                let mut m = T::default();
                                for fi in 0..h {
                                    for fj in 0..w {
                                        let hidx = fi + sh * hid;
                                        let widx = fj + sw * wid;
                                        let x = unsafe { *input.uget([bid, cid, hidx, widx]) };
                                        if (fi == 0 && fj == 0) || x > m {
                                            m = x;
                                        }
                                    }
                                }
                                unsafe {
                                    *output.uget_mut([bid_slice, cid, hid, wid]) = m;
                                }
                            }
                        }
                    }
                }
            });
        Ok(output)
    }
}

#[cfg(feature = "neural-network")]
impl<T: Scalar, S: Data<Elem = T>> MaxPool2 for TensorBase<S, Ix4> {
    type Output = Tensor4<T>;
    fn max_pool2(&self, options: MaxPool2Options) -> Result<Self::Output> {
        if let Some(input) = self.as_array() {
            input.max_pool2(options).map(Into::into)
        } else {
            Ok(ScalarTensorView::from(self.view())
                .max_pool2(options)?
                .try_into_tensor()
                .unwrap())
        }
    }
}

#[cfg(feature = "neural-network")]
impl<S: ScalarData> MaxPool2 for ScalarTensorBase<S, Ix4> {
    type Output = ScalarTensor4;
    fn max_pool2(&self, options: MaxPool2Options) -> Result<Self::Output> {
        macro_wrap!(
            paste! { #[allow(clippy::single_match)] match self.scalar_type() {
                macro_for!($T in [bf16, f32] {
                   ScalarType::[<$T:upper>] => {
                        let input = self.view().try_into_tensor_view::<$T>().unwrap();
                        if let Some(input) = input.as_array() {
                            return Ok(Tensor::from(input.max_pool2(options)?).into());
                        }
                        #[cfg(feature = "device")] {
                            let input = input.as_standard_layout()?;
                            let (bs, c, ih, iw) = input.dim();
                            let [oh, ow] = options.output_shape([ih, iw]);
                            let MaxPool2Options {
                                size: [h, w],
                                strides: [sh, sw],
                            } = options;
                            let mut output = unsafe {
                                Tensor::<$T, _>::uninit(input.device(), [bs, c, oh, ow])?
                            };
                            neural_network_kernels::[<max_pool2_ $T>]::builder()?
                                .specialize(h.to_u32().unwrap(), w.to_u32().unwrap(), sh.to_u32().unwrap(), sw.to_u32().unwrap())
                                .build(input.device())?
                                .dispatch(input.as_slice().unwrap(), ih.to_u32().unwrap(), iw.to_u32().unwrap(), output.as_slice_mut().unwrap(), oh.to_u32().unwrap(), ow.to_u32().unwrap())?;
                            return Ok(output.into());
                        }
                   }
                })
                _ => (),
            }}
        );
        bail!("max_pool2 {:?} unimplemented!()", self.scalar_type())
    }
}

#[cfg(feature = "neural-network")]
impl<T: Scalar, S1: ArrayDataMut<Elem = T>, S2: ArrayData<Elem = T>>
    MaxPool2Backward<ArrayBase<S2, Ix4>> for ArrayBase<S1, Ix4>
{
    fn max_pool2_backward(
        &mut self,
        output_grad: ArrayBase<S2, Ix4>,
        options: MaxPool2Options,
    ) -> Result<()> {
        let MaxPool2Options {
            size: [h, w],
            strides: [sh, sw],
        } = options;
        let mut input_grad = self.view_mut();
        let output_grad = output_grad.view();
        let (bs, c, _ih, _iw) = input_grad.dim();
        let (_bs, _c, oh, ow) = output_grad.dim();
        debug_assert_eq!(bs, _bs);
        debug_assert_eq!(c, _c);

        use rayon::prelude::*;

        let threads = rayon::current_num_threads();
        let chunk_size = bs / threads + (bs % threads != 0) as usize;

        input_grad
            .axis_chunks_iter_mut(ndarray::Axis(0), chunk_size)
            .into_par_iter()
            .enumerate()
            .for_each(|(chunk_id, mut input_grad)| {
                for bid_slice in 0..input_grad.dim().0 {
                    let bid = chunk_id * chunk_size + bid_slice;
                    for cid in 0..c {
                        for hid in 0..oh {
                            for wid in 0..ow {
                                let mut m = T::default();
                                let mut hidx_max = 0;
                                let mut widx_max = 0;
                                for fi in 0..h {
                                    for fj in 0..w {
                                        let hidx = fi + sh * hid;
                                        let widx = fj + sw * wid;
                                        let dx = unsafe {
                                            input_grad.uget_mut([bid_slice, cid, hidx, widx])
                                        };
                                        if (fi == 0 && fj == 0) || *dx > m {
                                            m = *dx;
                                            hidx_max = hidx;
                                            widx_max = widx;
                                        }
                                        *dx = T::default();
                                    }
                                }
                                unsafe {
                                    *input_grad.uget_mut([bid_slice, cid, hidx_max, widx_max]) +=
                                        *output_grad.uget([bid, cid, hid, wid]);
                                }
                            }
                        }
                    }
                }
            });
        Ok(())
    }
}

#[cfg(feature = "neural-network")]
impl<T: Scalar, S1: DataMut<Elem = T>, S2: Data<Elem = T>> MaxPool2Backward<TensorBase<S2, Ix4>>
    for TensorBase<S1, Ix4>
{
    fn max_pool2_backward(
        &mut self,
        output_grad: TensorBase<S2, Ix4>,
        options: MaxPool2Options,
    ) -> Result<()> {
        if let Some((mut dx, dy)) = self.as_array_mut().zip(output_grad.as_array()) {
            dx.max_pool2_backward(dy, options)
        } else {
            ScalarTensorViewMut::from(self.view_mut())
                .max_pool2_backward(output_grad.view().into(), options)
        }
    }
}

#[cfg(feature = "neural-network")]
impl<S1: ScalarDataMut, S2: ScalarData> MaxPool2Backward<ScalarTensorBase<S2, Ix4>>
    for ScalarTensorBase<S1, Ix4>
{
    fn max_pool2_backward(
        &mut self,
        output_grad: ScalarTensorBase<S2, Ix4>,
        options: MaxPool2Options,
    ) -> Result<()> {
        if self.scalar_type() != output_grad.scalar_type() {
            bail!(
                "Expected {:?} found {:?}",
                self.scalar_type(),
                output_grad.scalar_type()
            );
        }
        macro_wrap!(
            paste! { #[allow(clippy::single_match)] match self.scalar_type() {
                macro_for!($T in [bf16, f32] {
                   ScalarType::[<$T:upper>] => {
                        let mut input_grad = self.view_mut().try_into_tensor_view_mut::<$T>().unwrap();
                        let output_grad = output_grad.view().try_into_tensor_view().unwrap();
                        if let Some((mut dx, dy)) = input_grad.as_array_mut().zip(output_grad.as_array()) {
                            return dx.max_pool2_backward(dy, options);
                        }
                        #[cfg(feature = "device")] {
                            let (_bs, _c, ih, iw) = input_grad.dim();
                            let [oh, ow] = options.output_shape([ih, iw]);
                            let MaxPool2Options {
                                size: [h, w],
                                strides: [sh, sw],
                            } = options;
                            neural_network_kernels::[<max_pool2_backward_ $T>]::builder()?
                                .specialize(h.to_u32().unwrap(), w.to_u32().unwrap(), sh.to_u32().unwrap(), sw.to_u32().unwrap())
                                .build(input_grad.device())?
                                .dispatch(input_grad.as_slice_mut().unwrap(), ih.to_u32().unwrap(), iw.to_u32().unwrap(), output_grad.as_slice().unwrap(), oh.to_u32().unwrap(), ow.to_u32().unwrap())?;
                            return Ok(());
                        }
                   }
                })
                _ => (),
            }}
        );
        bail!(
            "max_pool2_backward {:?} unimplemented!()",
            self.scalar_type()
        )
    }
}

#[cfg_attr(feature = "device", module)]
mod binary_op {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::scalar::Scalar;

    #[cfg_attr(not(target_arch = "spirv"), derive(Debug, derive_more::IsVariant))]
    #[repr(u32)]
    pub enum BinaryOp {
        Identity = 1,
        Add = 2,
        Sub = 3,
        Mul = 4,
        Div = 5,
    }

    #[cfg(feature = "device")]
    impl BinaryOp {
        #[inline]
        #[allow(clippy::wrong_self_convention)]
        pub fn as_u32(self) -> u32 {
            self as u32
        }
    }

    impl TryFrom<u32> for BinaryOp {
        type Error = ();
        #[inline]
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
        #[inline]
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
}
use binary_op::BinaryOp;

#[cfg(feature = "device")]
#[module]
mod kernels {
    #[cfg(target_arch = "spirv")]
    use crate::tensor::ops::binary_op::BinaryOp;
    use dry::macro_for;
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    #[cfg(target_arch = "spirv")]
    #[allow(unused_imports)]
    use krnl_core::buffer::UnsafeIndex;
    use krnl_core::macros::kernel;
    #[allow(unused_imports)]
    use krnl_core::{
        half::{bf16, f16},
        scalar::Scalar,
    };
    use paste::paste;

    macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
        macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            paste! {
                #[kernel]
                pub fn [<assign_ $X _ $Y>]<const OP: u32>(
                    alpha: $Y,
                    #[item]
                    x: $X,
                    #[item]
                    y: &mut $Y,
                ) {
                    let op = BinaryOp::try_from(OP).ok().unwrap();
                    *y = op.eval(alpha * x.cast::<$Y>(), *y);
                }

                #[allow(clippy::too_many_arguments)]
                #[kernel]
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
                    let global_id = kernel.global_id() as u32;
                    let global_row = global_id / cols;
                    let global_col = global_id % cols;
                    if global_row < rows && global_col < cols {
                        let x_idx = (global_row as i32 * rsx + global_col as i32 * csx + offset_x as i32) as usize;
                        let y_idx = (global_row as i32 * rsy + global_col as i32 * csy + offset_y as i32) as usize;
                        unsafe {
                            *y.unsafe_index_mut(y_idx) = op.eval(alpha * x[x_idx].cast::<$Y>(), *y.unsafe_index(y_idx));
                        }
                    }
                }

                #[allow(clippy::too_many_arguments)]
                #[kernel]
                pub unsafe fn [<assign4_ $X _ $Y>]<const OP: u32>(
                    d0: u32,
                    d1: u32,
                    d2: u32,
                    d3: u32,
                    alpha: $Y,
                    #[global]
                    x: Slice<$X>,
                    sx0: i32,
                    sx1: i32,
                    sx2: i32,
                    sx3: i32,
                    offset_x: u32,
                    #[global]
                    y: UnsafeSlice<$Y>,
                    sy0: i32,
                    sy1: i32,
                    sy2: i32,
                    sy3: i32,
                    offset_y: u32,
                ) {
                    let op = BinaryOp::try_from(OP).ok().unwrap();
                    let idx = kernel.global_id() as u32;
                    if idx >= d0 * d1 * d2 * d3 {
                        return;
                    }
                    let i0 = idx / (d1 * d2 * d3);
                    let r0 = idx % (d1 * d2 * d3);
                    let i1 = r0 / (d2 * d3);
                    let r1 = r0 % (d2 * d3);
                    let i2 = r1 / d3;
                    let i3 = r1 % d3;
                    let [i0, i1, i2, i3] = [i0 as i32, i1 as i32, i2 as i32, i3 as i32];
                    let x_idx = (sx0 * i0 + sx1 * i1 + sx2 * i2 + sx3 * i3 + offset_x as i32) as usize;
                    let y_idx = (sy0 * i0 + sy1 * i1 + sy2 * i2 + sy3 * i3 + offset_y as i32) as usize;
                    unsafe {
                        *y.unsafe_index_mut(y_idx) = op.eval(alpha * x[x_idx].cast::<$Y>(), *y.unsafe_index(y_idx));
                    }
                }

                #[allow(clippy::too_many_arguments)]
                #[kernel]
                pub unsafe fn [<assign6_ $X _ $Y>]<const OP: u32>(
                    d0: u32,
                    d1: u32,
                    d2: u32,
                    d3: u32,
                    d4: u32,
                    d5: u32,
                    alpha: $Y,
                    #[global]
                    x: Slice<$X>,
                    sx0: i32,
                    sx1: i32,
                    sx2: i32,
                    sx3: i32,
                    sx4: i32,
                    sx5: i32,
                    offset_x: u32,
                    #[global]
                    y: UnsafeSlice<$Y>,
                    sy0: i32,
                    sy1: i32,
                    sy2: i32,
                    sy3: i32,
                    sy4: i32,
                    sy5: i32,
                    offset_y: u32,
                ) {
                    let op = BinaryOp::try_from(OP).ok().unwrap();
                    let idx = kernel.global_id() as u32;
                    if idx >= d0 * d1 * d2 * d3 * d4 * d5 {
                        return;
                    }
                    let i0 = idx / (d1 * d2 * d3 * d4 * d5);
                    let r0 = idx % (d1 * d2 * d3 * d4 * d5);
                    let i1 = r0 / (d2 * d3 * d4 * d5);
                    let r1 = r0 % (d2 * d3 * d4 * d5);
                    let i2 = r1 / (d3 * d4 * d5);
                    let r2 = r1 % (d3 * d4 * d5);
                    let i3 = r2 / (d4 * d5);
                    let r3 = r2 % (d4 * d5);
                    let i4 = r3 / d5;
                    let i5 = r3 % d5;
                    let [i0, i1, i2, i3, i4, i5] = [i0 as i32, i1 as i32, i2 as i32, i3 as i32, i4 as i32, i5 as i32];
                    let x_idx = (sx0 * i0 + sx1 * i1 + sx2 * i2 + sx3 * i3 + sx4 * i4 + sx5 * i5 + offset_x as i32) as usize;
                    let y_idx = (sy0 * i0 + sy1 * i1 + sy2 * i2 + sy3 * i3 + sy4 * i4 + sy5 * i5 + offset_y as i32) as usize;
                    unsafe {
                        *y.unsafe_index_mut(y_idx) = op.eval(alpha * x[x_idx].cast::<$Y>(), *y.unsafe_index(y_idx));
                    }
                }
            }
        });
    });

    macro_for!($X in [u8, u16, u32, u64] {
        macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            paste! {
                #[kernel]
                pub fn [<one_hot_ $X _ $Y>](
                    #[global] x: Slice<$X>,
                    #[item] y: &mut $Y,
                ) {
                    type Y = $Y;
                    use krnl_core::num_traits::{Zero, One, ToPrimitive};

                    let idx = kernel.item_id();
                    let classes = kernel.items() / x.len();
                    let x_idx = idx / classes;
                    let y_class = idx % classes;
                    let class = x[x_idx].to_usize().unwrap();
                    *y = if y_class == class {
                        Y::one()
                    } else {
                        Y::zero()
                    };
                }
            }
        });
    });
}

#[cfg(feature = "device")]
#[module]
mod neural_network_kernels {
    use dry::macro_for;
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;
    #[cfg(target_arch = "spirv")]
    use krnl_core::{buffer::UnsafeIndex, half::bf16, num_traits::Zero, scalar::Scalar};
    use paste::paste;

    macro_for!($T in [bf16, f32] {
        paste! {
            #[kernel]
            pub fn [<im2col_conv2_ $T>]<
                const BS: u32,
                const C: u32,
                const IH: u32,
                const IW: u32,
                const OH: u32,
                const OW: u32,
                const FH: u32,
                const FW: u32,
                const PH: u32,
                const PW: u32,
                const SH: u32,
                const SW: u32,
                const DH: u32,
                const DW: u32,
            >(
                #[global] x: Slice<$T>,
                #[global] y: UnsafeSlice<$T>,
            ) {
                let bs = BS;
                let c = C;
                let ih = IH;
                let iw = IW;
                let oh = OH;
                let ow = OW;
                let fh = FH;
                let fw = FW;
                let ph = PH;
                let pw = PW;
                let sh = SH;
                let sw = SW;
                let dh = DH;
                let dw = DW;

                let idx = kernel.global_id() as u32;
                if idx >= bs * oh * ow * c * fh * fw {
                    return;
                }
                let bcid = idx / (oh * ow * fh * fw);
                let hwfid = idx % (oh * ow * fh * fw);
                let bid = bcid / c;
                let cid = bcid % c;
                let hwid = hwfid / (fh * fw);
                let fid = hwfid % (fh * fw);
                let hid = hwid / ow;
                let wid = hwid % ow;
                let fi = fid / fw;
                let fj = fid % fw;
                let hidx = -(ph as i32) + (fi * dh + sh * hid) as i32;
                let widx = -(pw as i32) + (fj * dw + sw * wid) as i32;
                let x = if hidx >= 0 && hidx < ih as i32 && widx >= 0 && widx < iw as i32 {
                    x[(bcid * ih * iw + hidx as u32 * iw + widx as u32) as usize]
                } else {
                    $T::zero()
                };
                let y_idx = bid * oh * ow * c * fh * fw
                    + hid * ow * c * fh * fw
                    + wid * c * fh * fw
                    + cid * fh * fw
                    + fid;
                unsafe {
                    *y.unsafe_index_mut(y_idx as usize) = x;
                }
            }

            #[kernel]
            pub fn [<col2im_conv2_ $T>]<
                const C: u32,
                const IH: u32,
                const IW: u32,
                const OH: u32,
                const OW: u32,
                const FH: u32,
                const FW: u32,
                const PH: u32,
                const PW: u32,
                const SH: u32,
                const SW: u32,
                const DH: u32,
                const DW: u32,
            >(
                #[global] x: Slice<$T>,
                #[item] y: &mut $T,
            ) {
                let c = C;
                let [ih, iw] = [IH, IW];
                let [oh, ow] = [OH, OW];
                let [fh, fw] = [FH, FW];
                let [ph, pw] = [PH, PW];
                let [sh, sw] = [SH, SW];
                let [dh, dw] = [DH, DW];

                let idx = kernel.item_id() as u32;
                let bcid = idx / (oh * ow);
                let hwid = idx % (oh * ow);
                let bid = bcid / c;
                let cid = bcid % c;
                let bidx = bid * ih * iw * c * fh * fw;
                let bidy = bid * c * oh * ow;
                let cidx = cid * fh * fw;
                let cidy = cid * oh * ow;
                let hidy = hwid / ow + ph;
                let widy = hwid % ow + pw;

                #[inline]
                fn div_exact(a: u32, b: u32) -> Option<u32> {
                    if a % b == 0 {
                        Some(a / b)
                    } else {
                        None
                    }
                }

                let fh_max = fh.min(hidy / dh + 1);
                let fw_max = fw.min(widy / dw + 1);

                let mut acc = 0f32;
                for fi in 0 .. fh_max {
                    if let Some(hidx) = div_exact(hidy - fi * dh, sh)
                    .filter(|hidx| *hidx < ih) {
                        for fj in 0 .. fw_max {
                            if let Some(widx) = div_exact(widy - fj * dw, sw)
                            .filter(|widx| *widx < iw) {
                                let fidx = fi * fw + fj;
                                let patch_idx = (hidx * iw + widx) * c * fh * fw;
                                acc += x[(bidx + patch_idx + cidx + fidx) as usize].cast::<f32>();
                            }
                        }
                    }
                }
                *y = acc.cast();
            }

            #[kernel]
            pub fn [<max_pool2_ $T>]<const H: u32, const W: u32, const SH: u32, const SW: u32>(
                #[global] x: Slice<$T>,
                ih: u32,
                iw: u32,
                #[item] y: &mut $T,
                oh: u32,
                ow: u32,
            ) {
                let idx = kernel.item_id() as u32;
                let bid = idx / (oh * ow);
                let hwid = idx % (oh * ow);
                let hid = hwid / ow;
                let wid = hwid % ow;

                let x_start = bid * ih * iw;
                let mut m = 0f32;

                let mut row = hid * SH;
                for i in 0..H {
                    let mut col = wid * SW;
                    for j in 0..W {
                        let x = x[(x_start + row * iw + col) as usize].cast::<f32>();
                        if i == 0 && j == 0 {
                            m = x;
                        } else {
                            m = m.max(x);
                        }
                        col += 1;
                    }
                    row += 1;
                }
                *y = m.cast();
            }

            #[kernel]
            pub fn [<max_pool2_backward_ $T>]<const H: u32, const W: u32, const SH: u32, const SW: u32>(
                #[global] dx: UnsafeSlice<$T>,
                ih: u32,
                iw: u32,
                #[item] dy: $T,
                oh: u32,
                ow: u32,
            ) {
                let idx = kernel.item_id() as u32;
                let bid = idx / (oh * ow);
                let hwid = idx % (oh * ow);
                let hid = hwid / ow;
                let wid = hwid % ow;
                let dx_start = bid * ih * iw;
                let mut m = 0f32;
                let mut mi = 0;
                let mut mj = 0;
                let mut row = hid * SH;
                for i in 0..H {
                    let mut col = wid * SW;
                    for j in 0..W {
                        let dx = unsafe { dx.unsafe_index_mut((dx_start + row * ih + col) as usize) };
                        let x = dx.cast::<f32>();
                        *dx = $T::zero();
                        if (i == 0 && j == 0) || x > m {
                            m = x;
                            mi = i;
                            mj = j;
                        }
                        col += 1;
                    }
                    row += 1;
                }

                let row = hid * SH + mi;
                let col = wid * SW + mj;
                unsafe {
                    *dx.unsafe_index_mut((dx_start + row * iw + col) as usize) = dy.cast();
                }
            }
        }
    });
}
