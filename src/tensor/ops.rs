use super::*;
use crate::ops::AddAssign;
#[cfg(feature = "neural-network")]
use crate::ops::{
    Col2ImConv2, Col2ImConv2Options, Im2ColConv2, Im2ColConv2Options, MaxPool2, MaxPool2Backward,
    MaxPool2Options,
};
#[cfg(feature = "device")]
use anyhow::format_err;
use dry::macro_for;
#[cfg(feature = "neural-network")]
use dry::macro_wrap;
use half::{bf16, f16};
use krnl::krnl_core::num_traits::ToPrimitive;
#[cfg(feature = "device")]
use krnl::macros::module;
#[cfg(feature = "neural-network")]
use ndarray::{Array2, Array4, Data as ArrayData, DataMut as ArrayDataMut};

impl<S: ScalarData, D: Dimension> ScalarTensorBase<S, D> {
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
    }
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
    if op.is_identity() && x.scalar_type() == y.scalar_type() {
        if let Some((x, mut y)) = x.as_scalar_slice().zip(y.as_scalar_slice_mut()) {
            return y.copy_from_scalar_slice(&x);
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
        if let Some((x, mut y)) = x.as_scalar_slice().zip(y.as_scalar_slice_mut()) {
            macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                if let Ok(x) = Slice::<$X>::try_from(x.clone()) {
                    macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                        if let Ok(y) = SliceMut::<$Y>::try_from(y.as_scalar_slice_mut()) {
                            let builder = paste! {
                                kernels::[<assign_ $X _ $Y>]::builder()?
                            };
                            builder
                            .specialize(op.as_u32())?
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

#[cfg(feature = "neural-network")]
impl<T: Scalar, S: ArrayData<Elem = T>> Im2ColConv2 for ArrayBase<S, Ix4> {
    type Output = Array2<T>;
    fn im2col_conv2(&self, options: Im2ColConv2Options) -> Result<Self::Output> {
        let input = self.as_standard_layout();
        let (bs, c, ih, iw) = input.dim();
        let [oh, ow] = options.output_shape([ih, iw]);
        let Im2ColConv2Options {
            filter: [fh, fw],
            padding: [ph, pw],
            stride: [sh, sw],
            dilation: [dh, dw],
        } = options;
        let mut output = Array::uninit([bs, oh, ow, c, fh * fw]);
        for (input, mut output) in input.outer_iter().zip(output.outer_iter_mut()) {
            for (input, mut output) in input.outer_iter().zip(output.axis_iter_mut(Axis(2))) {
                for (hid, mut output) in output.outer_iter_mut().enumerate() {
                    for (wid, mut output) in output.outer_iter_mut().enumerate() {
                        for fi in 0..fh {
                            for fj in 0..fw {
                                let hidx = -(ph as isize) + (fi * dh + sh * hid) as isize;
                                let widx = -(pw as isize) + (fj * dw + sw * wid) as isize;
                                let fidx = fi * fw + fj;
                                if hidx >= 0
                                    && hidx < ih as isize
                                    && widx >= 0
                                    && widx < iw as isize
                                {
                                    unsafe {
                                        output.uget_mut(fidx).write(*input.uget((hidx as usize, widx as usize)));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let output = unsafe { output.assume_init() };
        Ok(output.into_shape([bs * oh * ow, c * fh * fw]).unwrap())
    }
}

#[cfg(feature = "neural-network")]
impl<T: Scalar, S: Data<Elem = T>> Im2ColConv2 for TensorBase<S, Ix4> {
    type Output = Tensor2<T>;
    fn im2col_conv2(&self, options: Im2ColConv2Options) -> Result<Self::Output> {
        if let Some(input) = self.as_array() {
            return input.im2col_conv2(options).map(Into::into);
        }
        todo!()
    }
}

#[cfg(feature = "neural-network")]
impl<S: ScalarData> Im2ColConv2 for ScalarTensorBase<S, Ix4> {
    type Output = ScalarTensor2;
    fn im2col_conv2(&self, options: Im2ColConv2Options) -> Result<Self::Output> {
        macro_wrap!(paste! { match self.scalar_type() {
            macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
               ScalarType::[<$T:upper>] => {
                   return self.view().try_into_tensor_view::<$T>().unwrap().im2col_conv2(options).map(Into::into);
               }
            })
            _ => (),
        }});
        bail!("im2col_conv2 {:?} unimplemented!()", self.scalar_type())
    }
}

#[cfg(feature = "neural-network")]
impl<T: Scalar, S: ArrayData<Elem = T>> Col2ImConv2 for ArrayBase<S, Ix2> {
    type Output = Array4<T>;
    fn col2im_conv2(&self, options: Col2ImConv2Options) -> Result<Self::Output> {
        let input = self.as_standard_layout();
        let (rows, cols) = input.dim();
        let [oh, ow] = options.output_shape();
        let Col2ImConv2Options {
            shape: [ih, iw],
            filter: [fh, fw],
            padding: [ph, pw],
            stride: [sh, sw],
            dilation: [dh, dw],
        } = options;
        let bs = rows / (ih * iw);
        let c = cols / (fh * fw);
        let input = input.into_shape([bs, ih, iw, c, fh * fw]).unwrap();
        let mut output = Array::uninit([bs, c, oh, ow]);
        for (input, mut output) in input.outer_iter().zip(output.outer_iter_mut()) {
            for (input, mut output) in input.axis_iter(Axis(2)).zip(output.outer_iter_mut()) {
                for (hid, input) in input.outer_iter().enumerate() {
                    for (wid, input) in input.outer_iter().enumerate() {
                        for fi in 0..fh {
                            for fj in 0..fw {
                                let hidx = -(ph as isize) + (fi * dh + sh * hid) as isize;
                                let widx = -(pw as isize) + (fj * dw + sw * wid) as isize;
                                let fidx = fi * fw + fj;
                                if hidx >= 0
                                    && hidx < oh as isize
                                    && widx >= 0
                                    && widx < ow as isize
                                {
                                    unsafe {
                                        output.uget_mut((hidx as usize, widx as usize)).write(*input.uget(fidx));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let output = unsafe {
            output.assume_init()
        };
        Ok(output)
    }
}

#[cfg(feature = "neural-network")]
impl<T: Scalar, S: Data<Elem = T>> Col2ImConv2 for TensorBase<S, Ix2> {
    type Output = Tensor4<T>;
    fn col2im_conv2(&self, options: Col2ImConv2Options) -> Result<Self::Output> {
        if let Some(input) = self.as_array() {
            return input.col2im_conv2(options).map(Into::into);
        }
        todo!()
    }
}

#[cfg(feature = "neural-network")]
impl<S: ScalarData> Col2ImConv2 for ScalarTensorBase<S, Ix2> {
    type Output = ScalarTensor4;
    fn col2im_conv2(&self, options: Col2ImConv2Options) -> Result<Self::Output> {
        macro_wrap!(paste! { match self.scalar_type() {
            macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
               ScalarType::[<$T:upper>] => {
                   return self.view().try_into_tensor_view::<$T>().unwrap().col2im_conv2(options).map(Into::into);
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
        let mut output = Array::uninit([bs, c, oh, ow]);
        for (x, mut y) in self.outer_iter().zip(output.outer_iter_mut()) {
            for (x, mut y) in x.outer_iter().zip(y.outer_iter_mut()) {
                for ((row, col), y) in y.indexed_iter_mut() {
                    let mut m = T::default();
                    for i in 0..h {
                        for j in 0..w {
                            let x = x[(row * sh + i, col * sw + j)];
                            if (i == 0 && j == 0) || x > m {
                                m = x;
                            }
                        }
                    }
                    y.write(m);
                }
            }
        }
        let output = unsafe {
            output.assume_init()
        };
        Ok(output)
    }
}

#[cfg(feature = "neural-network")]
impl<T: Scalar, S: Data<Elem = T>> MaxPool2 for TensorBase<S, Ix4> {
    type Output = Tensor4<T>;
    fn max_pool2(&self, options: MaxPool2Options) -> Result<Self::Output> {
        if let Some(input) = self.as_array() {
            return input.max_pool2(options).map(Into::into);
        }
        todo!()
    }
}

#[cfg(feature = "neural-network")]
impl<S: ScalarData> MaxPool2 for ScalarTensorBase<S, Ix4> {
    type Output = ScalarTensor4;
    fn max_pool2(&self, options: MaxPool2Options) -> Result<Self::Output> {
        macro_wrap!(paste! { match self.scalar_type() {
            macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
               ScalarType::[<$T:upper>] => {
                   return self.view().try_into_tensor_view::<$T>().unwrap().max_pool2(options).map(Into::into);
               }
            })
            _ => (),
        }});
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
        for (mut dx, dy) in self.outer_iter_mut().zip(output_grad.outer_iter()) {
            for (mut dx, dy) in dx.outer_iter_mut().zip(dy.outer_iter()) {
                for ((row, col), dy) in dy.indexed_iter() {
                    let mut m = T::default();
                    let mut mi = 0;
                    let mut mj = 0;
                    for i in 0..h {
                        for j in 0..w {
                            let dx = unsafe { dx.uget_mut((row * sh + i, col * sw + j)) };
                            if (i == 0 && j == 0) || *dx > m {
                                m = *dx;
                                mi = i;
                                mj = j;
                            }
                            *dx = T::default();
                        }
                    }
                    unsafe {
                        *dx.uget_mut((mi, mj)) = *dy;
                    }
                }
            }
        }
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
            return dx.max_pool2_backward(dy, options);
        }
        todo!()
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
        macro_wrap!(paste! { match self.scalar_type() {
            macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
               ScalarType::[<$T:upper>] => {
                   return self.view_mut().try_into_tensor_view_mut::<$T>().unwrap().max_pool2_backward(output_grad.view().try_into_tensor_view().unwrap(), options).map(Into::into);
               }
            })
            _ => (),
        }});
        bail!(
            "max_pool2_backward {:?} unimplemented!()",
            self.scalar_type()
        )
    }
}

#[cfg_attr(feature = "device", module)]
mod kernels {
    use dry::macro_for;
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;
    #[cfg(target_arch = "spirv")]
    #[allow(unused_imports)]
    use krnl_core::{buffer::UnsafeIndex, glam::Vec2Swizzles};
    #[allow(unused_imports)]
    use krnl_core::{
        half::{bf16, f16},
        scalar::Scalar,
    };
    use paste::paste;

    #[cfg_attr(not(target_arch = "spirv"), derive(derive_more::IsVariant))]
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
                #[kernel(threads(256))]
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
            }
        });
    });

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
