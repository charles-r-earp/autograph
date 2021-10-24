use crate::{
    buffer::{
        float::{FloatArcBuffer, FloatBuffer, FloatCowBuffer, FloatSlice, FloatSliceMut},
        CowBuffer,
    },
    device::Device,
    linalg::{Dot, DotAcc, DotBias},
    ops::{AddAssign, Im2Col, KernelArgs, KernelKind, ScaledAdd},
    result::Result,
    scalar::{AsFloat, Float, FloatType, Scalar, Uint},
    tensor::{
        dim_strides_from_shape, into_dimensionality, into_shape, is_standard_layout, permuted_axes,
        ArcRepr, ArcTensor, CowRepr, CowTensor, Data, DataBase, OwnedRepr, Tensor, TensorBase,
        TensorView, TensorViewMut, ViewMutRepr, ViewRepr,
    },
};
use half::bf16;
use num_traits::NumCast;
use std::mem::transmute;

use ndarray::{
    Axis, Dimension, IntoDimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, RemoveAxis,
    ShapeBuilder,
};
use serde::{Deserialize, Serialize};

macro_rules! impl_data_base {
    ($($data:ident $(<$a:lifetime>)?),+) => {
        $(
            impl DataBase for $data $(<$a>)? {
                fn device(&self) -> Device {
                    self.0.device()
                }
                fn len(&self) -> usize {
                    self.0.len()
                }
                fn is_empty(&self) -> bool {
                    self.0.is_empty()
                }
            }
        )+
    };
}

impl_data_base! {FloatOwnedRepr, FloatArcRepr, FloatViewRepr<'_>, FloatViewMutRepr<'_>, FloatCowRepr<'_>}

/// Marker trait for FloatTensorBase representation.
///
/// Typically use [`FloatTensor'] / [`FloatArcTensor`] / [`FloatTensorView`] / [`FloatTensorViewMut`] / [`FloatCowTensor`] types directly.
pub trait FloatData: Sized + DataBase {
    #[doc(hidden)]
    fn try_into_buffer(self) -> Result<FloatBuffer, Self> {
        Err(self)
    }
    #[doc(hidden)]
    fn into_owned(self) -> Result<FloatOwnedRepr> {
        match self.try_into_buffer() {
            Ok(buffer) => Ok(FloatOwnedRepr(buffer)),
            Err(this) => Ok(FloatOwnedRepr(this.as_slice().to_owned()?)),
        }
    }
    #[doc(hidden)]
    fn try_into_arc_buffer(self) -> Result<FloatArcBuffer, Self> {
        self.try_into_buffer().map(Into::into)
    }
    #[doc(hidden)]
    fn into_shared(self) -> Result<FloatArcRepr> {
        match self.try_into_arc_buffer() {
            Ok(buffer) => Ok(FloatArcRepr(buffer)),
            Err(this) => Ok(FloatArcRepr(this.as_slice().to_owned()?.into())),
        }
    }
    #[doc(hidden)]
    fn to_shared(&self) -> Result<FloatArcRepr> {
        Ok(FloatArcRepr(self.as_slice().to_owned()?.into()))
    }
    #[doc(hidden)]
    fn as_slice(&self) -> FloatSlice;
    #[doc(hidden)]
    fn float_type(&self) -> FloatType {
        self.as_slice().float_type()
    }
}

/// Marker trait for owned float tensors [`FloatTensor`] / [`FloatArcTensor`] / [`FloatCowTensor`].
pub trait FloatDataOwned: FloatData {
    #[doc(hidden)]
    fn from_buffer(buffer: FloatBuffer) -> Self;
}

/// Marker trait for mutable float tensors [`FloatTensor`] / [`FloatTensorViewMut`]
pub trait FloatDataMut: FloatData {
    #[doc(hidden)]
    fn as_slice_mut(&mut self) -> FloatSliceMut;
}

#[doc(hidden)]
/// Marker trait for potentially mutable float tensors [`FloatArcTensor`] / ['FloatCowTensor`].
pub trait FloatDataTryMut: FloatData {
    #[doc(hidden)]
    fn get_slice_mut(&mut self) -> Option<FloatSliceMut>;
    #[doc(hidden)]
    fn make_slice_mut(&mut self) -> Result<FloatSliceMut>;
}

/// FloatTensor representation.
#[derive(Debug, Serialize, Deserialize)]
pub struct FloatOwnedRepr(FloatBuffer);

impl FloatData for FloatOwnedRepr {
    fn try_into_buffer(self) -> Result<FloatBuffer, Self> {
        Ok(self.0)
    }
    fn as_slice(&self) -> FloatSlice {
        self.0.as_slice()
    }
}

impl FloatDataOwned for FloatOwnedRepr {
    fn from_buffer(buffer: FloatBuffer) -> Self {
        Self(buffer)
    }
}

impl FloatDataMut for FloatOwnedRepr {
    fn as_slice_mut(&mut self) -> FloatSliceMut {
        self.0.as_slice_mut()
    }
}

/// FloatArcTensor representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloatArcRepr(FloatArcBuffer);

impl FloatData for FloatArcRepr {
    fn try_into_buffer(self) -> Result<FloatBuffer, Self> {
        self.0.try_unwrap().map_err(Self)
    }
    fn try_into_arc_buffer(self) -> Result<FloatArcBuffer, Self> {
        Ok(self.0)
    }
    fn to_shared(&self) -> Result<FloatArcRepr> {
        Ok(self.clone())
    }
    fn as_slice(&self) -> FloatSlice {
        self.0.as_slice()
    }
}

impl FloatDataOwned for FloatArcRepr {
    fn from_buffer(buffer: FloatBuffer) -> Self {
        Self(buffer.into())
    }
}

impl FloatDataTryMut for FloatArcRepr {
    fn get_slice_mut(&mut self) -> Option<FloatSliceMut> {
        self.0.get_mut()
    }
    fn make_slice_mut(&mut self) -> Result<FloatSliceMut> {
        self.0.make_mut()
    }
}

/// FloatTensorView representation.
#[derive(Debug, Clone)]
pub struct FloatViewRepr<'a>(FloatSlice<'a>);

impl FloatData for FloatViewRepr<'_> {
    fn as_slice(&self) -> FloatSlice {
        self.0.as_slice()
    }
}

/// FloatTensorViewMut representation.
#[derive(Debug)]
pub struct FloatViewMutRepr<'a>(FloatSliceMut<'a>);

impl FloatData for FloatViewMutRepr<'_> {
    fn as_slice(&self) -> FloatSlice {
        self.0.as_slice()
    }
}

impl FloatDataMut for FloatViewMutRepr<'_> {
    fn as_slice_mut(&mut self) -> FloatSliceMut {
        self.0.as_slice_mut()
    }
}

/// FloatCowTensor representation.
#[derive(Debug)]
pub struct FloatCowRepr<'a>(FloatCowBuffer<'a>);

impl FloatDataOwned for FloatCowRepr<'_> {
    fn from_buffer(buffer: FloatBuffer) -> Self {
        Self(buffer.into())
    }
}

impl FloatData for FloatCowRepr<'_> {
    fn try_into_buffer(self) -> Result<FloatBuffer, Self> {
        self.0.try_unwrap().map_err(Self)
    }
    fn as_slice(&self) -> FloatSlice {
        self.0.as_slice()
    }
}

/// Dynamically Float typed [`TensorBase`]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloatTensorBase<S: FloatData, D: Dimension> {
    dim: D,
    strides: D,
    data: S,
}

/// Owned FloatTensor
///
/// See [`FloatTensorBase`].
pub type FloatTensor<D> = FloatTensorBase<FloatOwnedRepr, D>;
/// FloatTensor with 1 element
pub type FloatTensor0 = FloatTensor<Ix0>;
/// FloatTensor with 1 dimension
pub type FloatTensor1 = FloatTensor<Ix1>;
/// FloatTensor with 2 dimensions
pub type FloatTensor2 = FloatTensor<Ix2>;
/// FloatTensor with 3 dimensions
pub type FloatTensor3 = FloatTensor<Ix3>;
/// FloatTensor with 4 dimensions
pub type FloatTensor4 = FloatTensor<Ix4>;
/// FloatTensor with 5 dimensions
pub type FloatTensor5 = FloatTensor<Ix5>;
/// FloatTensor with 6 dimensions
pub type FloatTensor6 = FloatTensor<Ix6>;
/// FloatTensor with dynamic dimensions
pub type FloatTensorD = FloatTensor<IxDyn>;

/// Shared FloatTensor
///
/// See [`FloatTensorBase`].
pub type FloatArcTensor<D> = FloatTensorBase<FloatArcRepr, D>;
/// FloatArcTensor with 1 element
pub type FloatArcTensor0 = FloatArcTensor<Ix0>;
/// FloatArcTensor with 1 dimension
pub type FloatArcTensor1 = FloatArcTensor<Ix1>;
/// FloatArcTensor with 2 dimensions
pub type FloatArcTensor2 = FloatArcTensor<Ix2>;
/// FloatArcTensor with 3 dimensions
pub type FloatArcTensor3 = FloatArcTensor<Ix3>;
/// FloatArcTensor with 4 dimensions
pub type FloatArcTensor4 = FloatArcTensor<Ix4>;
/// FloatArcTensor with 5 dimensions
pub type FloatArcTensor5 = FloatArcTensor<Ix5>;
/// FloatArcTensor with 6 dimensions
pub type FloatArcTensor6 = FloatArcTensor<Ix6>;
/// FloatArcTensor with dynamic dimensions
pub type FloatArcTensorD = FloatArcTensor<IxDyn>;

/// Borrowed FloatTensor
///
/// See [`FloatTensorBase`].
pub type FloatTensorView<'a, D> = FloatTensorBase<FloatViewRepr<'a>, D>;
/// FloatTensorView with 1 element
pub type FloatTensorView0<'a> = FloatTensorView<'a, Ix0>;
/// FloatTensorView with 1 dimension
pub type FloatTensorView1<'a> = FloatTensorView<'a, Ix1>;
/// FloatTensorView with 2 dimensions
pub type FloatTensorView2<'a> = FloatTensorView<'a, Ix2>;
/// FloatTensorView with 3 dimensions
pub type FloatTensorView3<'a> = FloatTensorView<'a, Ix3>;
/// FloatTensorView with 4 dimensions
pub type FloatTensorView4<'a> = FloatTensorView<'a, Ix4>;
/// FloatTensorView with 5 dimensions
pub type FloatTensorView5<'a> = FloatTensorView<'a, Ix5>;
/// FloatTensorView with 6 dimensions
pub type FloatTensorView6<'a> = FloatTensorView<'a, Ix6>;
/// FloatTensorView with dynamic dimensions
pub type FloatTensorViewD<'a> = FloatTensorView<'a, IxDyn>;

/// Mutably borrowed FloatTensor
///
/// See [`FloatTensorBase`].
pub type FloatTensorViewMut<'a, D> = FloatTensorBase<FloatViewMutRepr<'a>, D>;
/// FloatTensorViewMut with 1 element
pub type FloatTensorViewMut0<'a> = FloatTensorViewMut<'a, Ix0>;
/// FloatTensorViewMut with 1 dimension
pub type FloatTensorViewMut1<'a> = FloatTensorViewMut<'a, Ix1>;
/// FloatTensorViewMut with 2 dimensions
pub type FloatTensorViewMut2<'a> = FloatTensorViewMut<'a, Ix2>;
/// FloatTensorViewMut with 3 dimensions
pub type FloatTensorViewMut3<'a> = FloatTensorViewMut<'a, Ix3>;
/// FloatTensorViewMut with 4 dimensions
pub type FloatTensorViewMut4<'a> = FloatTensorViewMut<'a, Ix4>;
/// FloatTensorViewMut with 5 dimensions
pub type FloatTensorViewMut5<'a> = FloatTensorViewMut<'a, Ix5>;
/// FloatTensorViewMut with 6 dimensions
pub type FloatTensorViewMut6<'a> = FloatTensorViewMut<'a, Ix6>;
/// FloatTensorViewMut with dynamic dimensions
pub type FloatTensorViewMutD<'a> = FloatTensorViewMut<'a, IxDyn>;

/// FloatTensor that is either borrowed or owned.
///
/// See [`FloatTensorBase`].
pub type FloatCowTensor<'a, D> = FloatTensorBase<FloatCowRepr<'a>, D>;
/// FloatCowTensor with 1 element
pub type FloatCowTensor0<'a> = FloatCowTensor<'a, Ix0>;
/// FloatCowTensor with 1 dimension
pub type FloatCowTensor1<'a> = FloatCowTensor<'a, Ix1>;
/// FloatCowTensor with 2 dimensions
pub type FloatCowTensor2<'a> = FloatCowTensor<'a, Ix2>;
/// FloatCowTensor with 3 dimensions
pub type FloatCowTensor3<'a> = FloatCowTensor<'a, Ix3>;
/// FloatCowTensor with 4 dimensions
pub type FloatCowTensor4<'a> = FloatCowTensor<'a, Ix4>;
/// FloatCowTensor with 5 dimensions
pub type FloatCowTensor5<'a> = FloatCowTensor<'a, Ix5>;
/// FloatCowTensor with 6 dimensions
pub type FloatCowTensor6<'a> = FloatCowTensor<'a, Ix6>;
/// FloatCowTensor with dynamic dimensions
pub type FloatCowTensorD<'a> = FloatCowTensor<'a, IxDyn>;

impl<S: FloatData, D: Dimension> FloatTensorBase<S, D> {
    /// Allocates a float tensor of type `float_type` on `device` with `shape`.
    ///
    /// # Safety
    ///
    /// The tensor is not initialized.
    ///
    /// **Errors**
    /// See [`Buffer::alloc()`](crate::device::buffer::BufferBase::alloc()).
    pub unsafe fn alloc<Sh>(float_type: FloatType, device: Device, shape: Sh) -> Result<Self>
    where
        S: FloatDataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        let (dim, strides) = dim_strides_from_shape(shape.into_shape());
        let data = S::from_buffer(FloatBuffer::alloc(float_type, device, dim.size())?);
        Ok(Self { dim, strides, data })
    }
    /// Creates a float tensor on `device` with `shape` filled with `elem`.
    ///
    /// **Errors**
    /// See [`Buffer::alloc()`](crate::device::buffer::BufferBase::alloc()).
    pub fn from_elem<T, Sh>(device: Device, shape: Sh, elem: T) -> Result<Self>
    where
        T: Float,
        S: FloatDataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        let (dim, strides) = dim_strides_from_shape(shape.into_shape());
        let data = S::from_buffer(FloatBuffer::from_elem(device, dim.size(), elem)?);
        Ok(Self { dim, strides, data })
    }
    /// Creates a float tensor of type `float_type` on `device` with `shape` filled with 0's.
    ///
    /// **Errors**
    ///
    /// See [`Buffer::alloc()`](crate::device::buffer::BufferBase::alloc()).
    pub fn zeros<Sh>(float_type: FloatType, device: Device, shape: Sh) -> Result<Self>
    where
        S: FloatDataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        let (dim, strides) = dim_strides_from_shape(shape.into_shape());
        let data = S::from_buffer(FloatBuffer::zeros(float_type, device, dim.size())?);
        Ok(Self { dim, strides, data })
    }
    /// Creates a float tensor of type `float_type` on `device` with `shape` filled with 0's.
    ///
    /// **Errors**
    ///
    /// See [`Buffer::alloc()`](crate::device::buffer::BufferBase::alloc()).
    pub fn ones<Sh>(float_type: FloatType, device: Device, shape: Sh) -> Result<Self>
    where
        S: FloatDataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        let (dim, strides) = dim_strides_from_shape(shape.into_shape());
        let data = S::from_buffer(FloatBuffer::ones(float_type, device, dim.size())?);
        Ok(Self { dim, strides, data })
    }
    /// The device of the tensor.
    pub fn device(&self) -> Device {
        self.data.device()
    }
    /// The dimensions of the tensor in pattern form.
    pub fn dim(&self) -> D::Pattern {
        self.dim.clone().into_pattern()
    }
    /// The dimensions of the tensor.
    pub fn raw_dim(&self) -> D {
        self.dim.clone()
    }
    #[allow(dead_code)]
    pub(crate) unsafe fn with_raw_dim(self, dim: D) -> Self {
        Self { dim, ..self }
    }
    /// The dimensions of the tensor as a slice.
    pub fn shape(&self) -> &[usize] {
        self.dim.slice()
    }
    /// The strides of the tensor as a slice.
    pub fn strides(&self) -> &[isize] {
        bytemuck::cast_slice(self.strides.slice())
    }
    #[allow(dead_code)]
    pub(crate) fn raw_strides(&self) -> D {
        self.strides.clone()
    }
    #[allow(dead_code)]
    pub(crate) unsafe fn with_raw_strides(self, strides: D) -> Self {
        Self { strides, ..self }
    }
    /// The length of the tensor.
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.data.len(), self.dim.size());
        self.data.len()
    }
    /// Whether the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    /// The dimensionality of the tensor.
    pub fn ndim(&self) -> usize {
        self.dim.ndim()
    }
    /// The [`FloatType`] of the tensor.
    pub fn float_type(&self) -> FloatType {
        self.data.float_type()
    }
    /// Converts the tensor into dimension `D2`.
    ///
    /// Typically this is used to downcast from [`IxDyn`](type@ndarray::IxDyn) to a static dimensionality. For conversions to [`IxDyn`](type@ndarray::IxDyn), use [`.into_dyn()`](TensorBase::into_dyn()).
    ///
    /// **Errors**
    ///
    /// The number of axes of `D2` must be the same as `D`.
    pub fn into_dimensionality<D2>(self) -> Result<FloatTensorBase<S, D2>>
    where
        D2: Dimension,
    {
        let (dim, strides) = into_dimensionality(&self.dim, &self.strides)?;
        Ok(FloatTensorBase {
            dim,
            strides,
            data: self.data,
        })
    }
    /// Returns the tensor with dim `shape`.
    ///
    /// **Errors**
    ///
    /// The tensor must be contiguous, with default strides.
    pub fn into_shape<E>(self, shape: E) -> Result<FloatTensorBase<S, E::Dim>>
    where
        E: IntoDimension,
    {
        let (dim, strides) = into_shape(&self.dim, &self.strides, shape)?;
        Ok(FloatTensorBase {
            dim,
            strides,
            data: self.data,
        })
    }
    #[allow(unused)]
    pub(crate) fn flatten(self) -> Result<FloatTensorBase<S, Ix2>> {
        let batch_size = self.shape()[0];
        let n = self.shape()[1..].iter().product();
        self.into_shape([batch_size, n])
    }
    /// Converts the dimensionality of the tensor to [`IxDyn`](type@ndarray::IxDyn).
    pub fn into_dyn(self) -> FloatTensorBase<S, IxDyn> {
        FloatTensorBase {
            dim: self.dim.into_dyn(),
            strides: self.strides.into_dyn(),
            data: self.data,
        }
    }
    /// Borrows the tensor as a [`FloatTensorView`].
    pub fn view(&self) -> FloatTensorView<D> {
        FloatTensorView {
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            data: FloatViewRepr(self.data.as_slice()),
        }
    }
    /// Borrows the tensor as a [`FloatTensorViewMut`].
    pub fn view_mut(&mut self) -> FloatTensorViewMut<D>
    where
        S: FloatDataMut,
    {
        FloatTensorViewMut {
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            data: FloatViewMutRepr(self.data.as_slice_mut()),
        }
    }
    /// Permute the axes of the tensor.
    ///
    /// See [`TensorBase::permuted_axes`].
    pub fn permuted_axes<A>(self, axes: A) -> Self
    where
        A: IntoDimension<Dim = D>,
    {
        let (dim, strides) = permuted_axes(self.dim, self.strides, axes.into_dimension());
        Self {
            dim,
            strides,
            data: self.data,
        }
    }
    /// Reverses (transposes) the axes of the tensor.
    pub fn reversed_axes(mut self) -> Self {
        self.dim.slice_mut().reverse();
        self.strides.slice_mut().reverse();
        self
    }
    /// Retunrs a view with reversed (transposed) axes.
    pub fn t(&self) -> FloatTensorView<D> {
        self.view().reversed_axes()
    }
    /// Whether the tensor is standard layout.
    ///
    /// See [`TensorBase::is_standard_layout()`].
    pub fn is_standard_layout(&self) -> bool {
        is_standard_layout(&self.dim, &self.strides)
    }
    /// Returns a [`FloatCowBuffer`] in standard layout.
    ///
    /// If the data is default strided, ie standard layout (C or RowMajor), borrows the data as a slice. Otherwise, clones the data.
    ///
    /// See also [`as_raw_slice()`](FloatTensorBase::as_raw_slice()).
    ///
    /// **Errors**
    /// See [`.to_owned()`](FloatTensorBase::to_owned()).
    pub fn to_slice(&self) -> Result<FloatCowBuffer> {
        if self.strides == self.dim.default_strides() {
            Ok(self.data.as_slice().into())
        } else {
            Ok(self.data.as_slice().to_owned()?.into())
        }
    }
    /// Borrows the tensor as a [`FloatSlice`].
    ///
    /// # Note
    /// If the tensor is not standard layout (C or RowMajor), this may not be what you want. See [`to_slice()`](TensorBase::to_slice()).
    pub fn as_raw_slice(&self) -> FloatSlice {
        self.data.as_slice()
    }
    /// Mutably borrows the tensor as a [`FloatSliceMut`].
    ///
    /// # Note
    /// If the tensor is not standard layout (C or RowMajor), this may not be what you want.
    pub fn as_raw_slice_mut(&mut self) -> FloatSliceMut
    where
        S: FloatDataMut,
    {
        self.data.as_slice_mut()
    }
    #[allow(unused)]
    pub(crate) fn get_mut(&mut self) -> Option<FloatTensorViewMut<D>>
    where
        S: FloatDataTryMut,
    {
        Some(FloatTensorViewMut {
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            data: FloatViewMutRepr(self.data.get_slice_mut()?),
        })
    }
    #[allow(unused)]
    pub(crate) fn make_mut(&mut self) -> Result<FloatTensorViewMut<D>>
    where
        S: FloatDataTryMut,
    {
        Ok(FloatTensorViewMut {
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            data: FloatViewMutRepr(self.data.make_slice_mut()?),
        })
    }
    /// Transfers the tensor into the `device`.
    ///
    /// **Errors**
    ///
    /// See [`Buffer::into_device()`](crate::device::buffer::BufferBase::into_device()).
    pub async fn into_device(self, device: Device) -> Result<FloatTensor<D>> {
        if device == self.device() {
            self.into_owned()
        } else {
            let buffer = self.data.as_slice().into_device(device).await?;
            Ok(FloatTensor {
                dim: self.dim,
                strides: self.strides,
                data: FloatOwnedRepr(buffer),
            })
        }
    }
    /// Transfers the tensor into the `device`.
    ///
    /// **Errors**
    ///
    /// See [`Buffer::into_device()`](crate::device::buffer::BufferBase::into_device()).
    pub async fn into_device_shared(self, device: Device) -> Result<FloatArcTensor<D>> {
        if device == self.device() {
            self.into_shared()
        } else {
            match self.data.try_into_arc_buffer() {
                Ok(buffer) => {
                    let buffer = buffer.into_device_shared(device).await?;
                    Ok(FloatArcTensor {
                        dim: self.dim,
                        strides: self.strides,
                        data: FloatArcRepr(buffer),
                    })
                }
                Err(data) => Self {
                    dim: self.dim,
                    strides: self.strides,
                    data,
                }
                .into_device(device)
                .await?
                .into_shared(),
            }
        }
    }
    #[allow(dead_code)]
    pub(crate) fn into_raw_buffer(self) -> Result<FloatBuffer> {
        Ok(self.data.into_owned()?.0)
    }
    /// Converts into a [`FloatTensor`].
    pub fn into_owned(self) -> Result<FloatTensor<D>> {
        Ok(FloatTensor {
            dim: self.dim,
            strides: self.strides,
            data: self.data.into_owned()?,
        })
    }
    /// Converts to a [`FloatTensor`].
    pub fn to_owned(self) -> Result<FloatTensor<D>> {
        self.view().into_owned()
    }
    /// Converts into an [`FloatArcTensor`].
    pub fn into_shared(self) -> Result<FloatArcTensor<D>> {
        Ok(FloatArcTensor {
            dim: self.dim,
            strides: self.strides,
            data: self.data.into_shared()?,
        })
    }
    /// Converts to an [`FloatArcTensor`].
    pub fn to_shared(&self) -> Result<FloatArcTensor<D>> {
        Ok(FloatArcTensor {
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            data: self.data.to_shared()?,
        })
    }
}

impl<S: FloatDataOwned> From<FloatBuffer> for FloatTensorBase<S, Ix1> {
    fn from(buffer: FloatBuffer) -> Self {
        let dim = buffer.len().into_dimension();
        let strides = dim.default_strides();
        let data = S::from_buffer(buffer);
        Self { dim, strides, data }
    }
}

impl<'a> From<FloatSlice<'a>> for FloatTensorView<'a, Ix1> {
    fn from(slice: FloatSlice<'a>) -> Self {
        let dim = slice.len().into_dimension();
        let strides = dim.default_strides();
        let data = FloatViewRepr(slice);
        Self { dim, strides, data }
    }
}

impl<'a> From<FloatSliceMut<'a>> for FloatTensorViewMut<'a, Ix1> {
    fn from(slice: FloatSliceMut<'a>) -> Self {
        let dim = slice.len().into_dimension();
        let strides = dim.default_strides();
        let data = FloatViewMutRepr(slice);
        Self { dim, strides, data }
    }
}

impl<T: Float, D: Dimension> From<Tensor<T, D>> for FloatTensor<D> {
    fn from(tensor: Tensor<T, D>) -> Self {
        let data = FloatOwnedRepr(tensor.data.0.into());
        Self {
            dim: tensor.dim,
            strides: tensor.strides,
            data,
        }
    }
}

impl<T: Float, D: Dimension> From<ArcTensor<T, D>> for FloatArcTensor<D> {
    fn from(tensor: ArcTensor<T, D>) -> Self {
        let data = FloatArcRepr(tensor.data.0.into());
        Self {
            dim: tensor.dim,
            strides: tensor.strides,
            data,
        }
    }
}

impl<D: Dimension> From<FloatTensor<D>> for FloatArcTensor<D> {
    fn from(tensor: FloatTensor<D>) -> Self {
        let data = FloatArcRepr(tensor.data.0.into());
        Self {
            dim: tensor.dim,
            strides: tensor.strides,
            data,
        }
    }
}

impl<'a, T: Float, D: Dimension> From<TensorView<'a, T, D>> for FloatTensorView<'a, D> {
    fn from(tensor: TensorView<'a, T, D>) -> Self {
        let data = FloatViewRepr(tensor.data.0.into());
        Self {
            dim: tensor.dim,
            strides: tensor.strides,
            data,
        }
    }
}

impl<'a, T: Float, D: Dimension> From<TensorViewMut<'a, T, D>> for FloatTensorViewMut<'a, D> {
    fn from(tensor: TensorViewMut<'a, T, D>) -> Self {
        let data = FloatViewMutRepr(tensor.data.0.into());
        Self {
            dim: tensor.dim,
            strides: tensor.strides,
            data,
        }
    }
}

impl<'a, T: Float, D: Dimension> From<CowTensor<'a, T, D>> for FloatCowTensor<'a, D> {
    fn from(tensor: CowTensor<'a, T, D>) -> Self {
        let data = FloatCowRepr(tensor.data.0.into());
        Self {
            dim: tensor.dim,
            strides: tensor.strides,
            data,
        }
    }
}

impl<D: Dimension> From<FloatTensor<D>> for FloatCowTensor<'_, D> {
    fn from(tensor: FloatTensor<D>) -> Self {
        let data = FloatCowRepr(tensor.data.0.into());
        Self {
            dim: tensor.dim,
            strides: tensor.strides,
            data,
        }
    }
}

impl<'a, D: Dimension> From<FloatTensorView<'a, D>> for FloatCowTensor<'a, D> {
    fn from(tensor: FloatTensorView<'a, D>) -> Self {
        let data = FloatCowRepr(tensor.data.0.into());
        Self {
            dim: tensor.dim,
            strides: tensor.strides,
            data,
        }
    }
}

macro_rules! impl_downcast {
    ($($float_tensor:ident | $float_data:ident $(<$a:lifetime>)? => $tensor:ident | $data:ident,)+) => {
        $(
            impl<$($a,)? D: Dimension> $float_tensor<$($a,)? D> {
                #[allow(unused)]
                pub(crate) fn downcast<T: Float>(self) -> Result<$tensor<$($a,)? T, D>, Self> {
                    match self.data.0.downcast() {
                        Ok(buffer) => Ok(TensorBase {
                            dim: self.dim,
                            strides: self.strides,
                            data: $data(buffer)
                        }),
                        Err(buffer) => Err(FloatTensorBase {
                            dim: self.dim,
                            strides: self.strides,
                            data: $float_data(buffer)
                        })
                    }
                }
            }
        )+
    };
}

impl_downcast! {
    FloatTensor | FloatOwnedRepr => Tensor | OwnedRepr,
    FloatArcTensor | FloatArcRepr => ArcTensor | ArcRepr,
    FloatTensorView | FloatViewRepr<'a> => TensorView | ViewRepr,
    FloatTensorViewMut | FloatViewMutRepr<'a> => TensorViewMut | ViewMutRepr,
    FloatCowTensor | FloatCowRepr<'a> => CowTensor | CowRepr,
}

/// Casts
#[allow(unused)]
impl<S: FloatData, D: Dimension> FloatTensorBase<S, D> {
    /// Casts the tensor into a new tensor.
    ///
    /// See [`TensorBase::cast_into()`].
    pub fn cast_into<T2: Scalar>(self) -> Result<Tensor<T2, D>> {
        let buffer = match self.data.try_into_buffer() {
            Ok(buffer) => buffer.cast_into()?,
            Err(data) => data.as_slice().cast_into()?,
        };
        Ok(TensorBase {
            dim: self.dim,
            strides: self.strides,
            data: OwnedRepr(buffer),
        })
    }
    /// Casts the tensor into a new tensor.
    ///
    /// See [`TensorBase::cast_to()`].
    pub fn cast_to<T2: Scalar>(&self) -> Result<CowTensor<T2, D>> {
        let slice = self.data.as_slice();
        let buffer: CowBuffer<T2> = slice.cast_to::<T2>()?;
        Ok(TensorBase {
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            data: CowRepr(unsafe { transmute(buffer) }),
        })
    }
    /// Scales the tensor into a new tensor.
    ///
    /// See [`TensorBase::scale_into()`].
    pub fn scale_into<T2: Scalar>(self, alpha: T2) -> Result<Tensor<T2, D>> {
        let buffer = match self.data.try_into_buffer() {
            Ok(buffer) => buffer.scale_into(alpha)?,
            Err(data) => data.as_slice().scale_into(alpha)?,
        };
        Ok(TensorBase {
            dim: self.dim,
            strides: self.strides,
            data: OwnedRepr(buffer),
        })
    }
    pub(crate) fn downcast_ref<T: Float>(&self) -> Option<TensorView<T, D>> {
        self.view().downcast().ok()
    }
    pub(crate) fn downcast_mut<T: Float>(&mut self) -> Option<TensorViewMut<T, D>>
    where
        S: FloatDataMut,
    {
        self.view_mut().downcast().ok()
    }
}

#[allow(dead_code)]
impl<T: Uint, S: Data<Elem = T>> TensorBase<S, Ix1> {
    pub(crate) fn to_one_hot_float(
        &self,
        float_type: FloatType,
        nclasses: usize,
    ) -> Result<FloatTensor2> {
        match float_type {
            FloatType::BF16 => Ok(self.to_one_hot::<bf16>(nclasses)?.into()),
            FloatType::F32 => Ok(self.to_one_hot::<f32>(nclasses)?.into()),
        }
    }
}

macro_rules! map_float_tensor {
    (ref $tensor:ident, $x:ident => $e:expr) => {
        map_float_tensor! {Downcast $tensor, $x => downcast_ref $e}
    };
    (mut $tensor:ident, $x:ident => $e:expr) => {
        map_float_tensor! {Downcast $tensor, $x => downcast_mut { let mut $x = $x; $e }}
    };
    (Downcast $tensor:ident, $x:ident => $downcast:ident $e:expr) => {{
        use FloatType::*;
        match $tensor.float_type() {
            BF16 => {
                let $x = $tensor.$downcast::<bf16>().unwrap();
                $e
            }
            F32 => {
                let $x = $tensor.$downcast::<f32>().unwrap();
                $e
            }
        }
    }};
    ($tensor:ident, $x:ident => $e:expr) => {{
        use FloatType::*;
        match $tensor.float_type() {
            BF16 => {
                let $x = $tensor.cast_into::<bf16>().unwrap();
                $e
            }
            F32 => {
                let $x = $tensor.cast_into::<f32>().unwrap();
                $e
            }
        }
    }};
}

/// Reorder
impl<S: FloatData, D: Dimension> FloatTensorBase<S, D> {
    /// Converts to standard layout.
    ///
    /// See [`TensorBase::as_standard_layout()`].
    pub fn as_standard_layout(&self) -> Result<FloatCowTensor<D>> {
        if self.is_standard_layout() {
            Ok(self.view().into())
        } else {
            self.view().into_standard_layout().map(Into::into)
        }
    }
    /// Converts into standard layout.
    ///
    /// See [`TensorBase::into_standard_layout()`].
    pub fn into_standard_layout(self) -> Result<FloatTensor<D>> {
        map_float_tensor!(self, x => x.into_standard_layout().map(Into::into))
    }
    /// Converts to a [`FloatArcTensor`] in standard layout.
    ///
    /// See [`TensorBase::to_standard_layout_shared()`].
    pub fn to_standard_layout_shared(&self) -> Result<FloatArcTensor<D>> {
        map_float_tensor!(ref self, x => x.to_standard_layout_shared().map(Into::into))
    }
}

/// Reductions
#[allow(unused)]
impl<S: FloatData, D: Dimension> FloatTensorBase<S, D> {
    pub(crate) fn sum(&self) -> Result<FloatTensor0> {
        map_float_tensor!(ref self, tensor => Ok(tensor.sum()?.into()))
    }
    /// Computes the sum along the given axis
    pub(crate) fn sum_axis(&self, axis: Axis) -> Result<FloatTensor<D::Smaller>>
    where
        D: RemoveAxis,
    {
        map_float_tensor!(ref self, tensor => Ok(tensor.sum_axis(axis)?.into()))
    }
    pub(crate) fn sum_with(&self, output: &mut FloatTensorViewMut0) -> Result<()> {
        map_float_tensor!(mut output, output => self.cast_to()?.sum_with(&mut output))
    }
    pub(crate) fn sum_axis_with(
        &self,
        axis: Axis,
        output: &mut FloatTensorViewMut<D::Smaller>,
    ) -> Result<()>
    where
        D: RemoveAxis,
    {
        map_float_tensor!(mut output, output => self.cast_to()?.sum_axis_with(axis, &mut output))
    }
    /// Computes the min value along the given axis
    pub(crate) fn min_axis(&self, axis: Axis) -> Result<FloatTensor<D::Smaller>>
    where
        D: RemoveAxis,
    {
        map_float_tensor!(ref self, tensor => Ok(tensor.min_axis(axis)?.into()))
    }
    /// Computes the index of the min value along the given axis.
    ///
    /// For multiple min values, the first will be selected. NaN values are ignored, returns 0 if all values are NaN.
    pub(crate) fn argmin_axis<U: Uint>(&self, axis: Axis) -> Result<Tensor<U, D::Smaller>>
    where
        D: RemoveAxis,
    {
        map_float_tensor!(ref self, tensor => tensor.argmin_axis(axis))
    }
    /// Computes the index of the max value along the given axis.
    ///
    /// For multiple max values, the first will be selected. NaN values are ignored, returns 0 if all values are NaN.
    pub(crate) fn argmax_axis<U: Uint>(&self, axis: Axis) -> Result<Tensor<U, D::Smaller>>
    where
        D: RemoveAxis,
    {
        map_float_tensor!(ref self, tensor => tensor.argmax_axis(axis))
    }
    /// Indexes an `axis` with `indices`.
    ///
    /// **Errors**
    /// - The `axis` is out of range.
    /// - The operation could not be performed.
    pub(crate) fn index_select(
        &self,
        axis: Axis,
        indices: &TensorView<u32, D::Smaller>,
    ) -> Result<FloatTensor<D::Smaller>>
    where
        D: RemoveAxis,
    {
        map_float_tensor!(ref self, tensor => tensor.index_select(axis, indices).map(Into::into))
    }
}

// ops

impl<S1: FloatDataMut, S2: FloatData, D: Dimension> AddAssign<FloatTensorBase<S2, D>>
    for FloatTensorBase<S1, D>
{
    fn add_assign(&mut self, rhs: &FloatTensorBase<S2, D>) -> Result<()> {
        self.scaled_add(1f32, rhs)
    }
}

impl<T: Float, S1: FloatDataMut, S2: FloatData, D: Dimension> ScaledAdd<T, FloatTensorBase<S2, D>>
    for FloatTensorBase<S1, D>
{
    fn scaled_add(&mut self, alpha: T, rhs: &FloatTensorBase<S2, D>) -> Result<()> {
        map_float_tensor!(mut self, tensor => tensor.scaled_add(NumCast::from(alpha).unwrap(), &rhs.cast_to()?))
    }
}

// linalg

impl<S1: FloatData, S2: FloatData> Dot<FloatTensorBase<S2, Ix2>> for FloatTensorBase<S1, Ix2> {
    type Output = FloatTensor2;
    fn dot(&self, rhs: &FloatTensorBase<S2, Ix2>) -> Result<Self::Output> {
        map_float_tensor!(ref self, lhs => Ok(lhs.dot(&rhs.cast_to()?)?.into()))
    }
}

impl<S1: FloatData, S2: FloatData, S3: FloatData>
    DotBias<FloatTensorBase<S2, Ix2>, FloatTensorBase<S3, Ix1>> for FloatTensorBase<S1, Ix2>
{
    fn dot_bias(
        &self,
        rhs: &FloatTensorBase<S2, Ix2>,
        bias: Option<&FloatTensorBase<S3, Ix1>>,
    ) -> Result<Self::Output> {
        map_float_tensor!(ref self, lhs => {
            let bias = if let Some(bias) = bias {
                Some(bias.cast_to()?)
            } else {
                None
            };
            Ok(lhs.dot_bias(&rhs.cast_to()?, bias.as_ref())?.into())
        })
    }
}

impl<T: Float, S1: FloatData, S2: FloatData, S3: FloatDataMut>
    DotAcc<T, FloatTensorBase<S2, Ix2>, FloatTensorBase<S3, Ix2>> for FloatTensorBase<S1, Ix2>
{
    fn dot_acc(
        &self,
        alpha: T,
        rhs: &FloatTensorBase<S2, Ix2>,
        output: &mut FloatTensorBase<S3, Ix2>,
    ) -> Result<()> {
        map_float_tensor!(mut output, output => self.cast_to()?.dot_acc(alpha.to_f32().unwrap().as_(), &rhs.cast_to()?, &mut output))
    }
}

/// Conv
impl<S: FloatData> Im2Col<Ix2> for FloatTensorBase<S, Ix4> {
    type Output = FloatTensor2;
    fn im2col(
        &self,
        kernel: &Ix2,
        kind: KernelKind,
        args: &KernelArgs<Ix2>,
    ) -> Result<Self::Output> {
        map_float_tensor!(ref self, x => x.im2col(kernel, kind, args).map(Into::into))
    }
}
