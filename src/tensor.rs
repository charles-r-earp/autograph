#[cfg(doc)]
use crate::device::error::DeviceLost;
use crate::{
    buffer::{
        ArcBuffer, ArcBufferRepr, Buffer, BufferBase, BufferRepr, CowBuffer, CowBufferRepr, Data,
        DataMut, DataOwned, Slice, SliceMut, SliceMutRepr, SliceRepr,
    },
    device::Device,
    scalar::{Scalar, ScalarType},
    util::{elem_type_name, size_eq},
};
use anyhow::{anyhow, bail, Result};
use bytemuck::Pod;
use ndarray::{
    Array, ArrayBase, ArrayView, ArrayViewMut, Dimension, IntoDimension, Ix0, Ix1, Ix2, Ix3, Ix4,
    Ix5, Ix6, IxDyn, RawArrayView, ShapeBuilder, StrideShape,
};
use serde::{Deserialize, Serialize};
use std::{
    convert::{TryFrom, TryInto},
    fmt::{self, Debug},
    mem::{size_of, transmute},
    num::NonZeroUsize,
};

//mod accuracy;
mod linalg;
//mod ops;
//mod reduce;
mod reorder;

fn strides_from_array<S, D>(array: &ArrayBase<S, D>) -> D
where
    S: ndarray::RawData,
    D: Dimension,
{
    let strides_slice: &[usize] = bytemuck::cast_slice(array.strides());
    let mut strides = D::zeros(strides_slice.len());
    for (i, s) in strides_slice.iter().copied().enumerate() {
        strides[i] = s;
    }
    strides
}

fn dim_strides_from_shape<D: Dimension>(shape: impl Into<StrideShape<D>>) -> (D, D) {
    let array = unsafe { RawArrayView::from_shape_ptr(shape, &()) };
    let dim = array.raw_dim();
    let strides = strides_from_array(&array);
    (dim, strides)
}

fn into_dimensionality<D1, D2>(dim: &D1, strides: &D1) -> Result<(D2, D2)>
where
    D1: Dimension,
    D2: Dimension,
{
    D2::from_dimension(dim)
        .and_then(|dim| D2::from_dimension(strides).map(|strides| (dim, strides)))
        .ok_or_else(|| {
            let strides = bytemuck::cast_slice::<_, isize>(strides.slice());
            anyhow!(
                "Incompatible Shapes! {:?} {:?} => {:?}",
                dim.slice(),
                strides,
                D2::NDIM
            )
        })
}

fn into_shape<D1, E>(dim: &D1, strides: &D1, shape: E) -> Result<(E::Dim, E::Dim)>
where
    D1: Dimension,
    E: IntoDimension,
{
    let shape = shape.into_dimension();
    let zero_strides = strides.slice().iter().any(|s| *s == 0);
    if shape.size() == dim.size() && (zero_strides || strides == &dim.default_strides()) {
        let strides = shape.default_strides();
        Ok((shape, strides))
    } else if dim.ndim() > 1 && (zero_strides || strides == &dim.fortran_strides()) {
        let strides = shape.fortran_strides();
        Ok((shape, strides))
    } else {
        Err(anyhow!(
            "Incompatible Shapes! {:?} {:?} => {:?}",
            dim.slice(),
            strides.slice(),
            shape.slice()
        ))
    }
}

fn is_contiguous<D: Dimension>(dim: &D, strides: &D, offset: usize) -> bool {
    if offset > 0 {
        return false;
    }
    let zero_strides = strides.slice().iter().any(|s| *s == 0);
    zero_strides || strides == &dim.default_strides() || strides == &dim.fortran_strides()
}

fn is_standard_layout<D: Dimension>(dim: &D, strides: &D, offset: usize) -> bool {
    if offset > 0 {
        return false;
    }
    let zero_strides = strides.slice().iter().any(|s| *s == 0);
    (zero_strides || strides == &dim.default_strides())
}

// adapted from https://docs.rs/ndarray/0.15.3/ndarray/struct.ArrayBase.html#method.permuted_axes
fn permuted_axes<D: Dimension>(dim: D, strides: D, axes: D) -> (D, D) {
    // Ensure that each axis is used exactly once.
    let mut usage_counts = D::zeros(dim.ndim());
    for axis in axes.slice() {
        usage_counts[*axis] += 1;
    }
    for count in usage_counts.slice() {
        assert_eq!(*count, 1, "each axis must be listed exactly once");
    }
    // Determine the new shape and strides.
    let mut new_dim = usage_counts; // reuse to avoid an allocation
    let mut new_strides = D::zeros(dim.ndim());
    {
        let dim = dim.slice();
        let strides = strides.slice();
        for (new_axis, &axis) in axes.slice().iter().enumerate() {
            new_dim[new_axis] = dim[axis];
            new_strides[new_axis] = strides[axis];
        }
    }
    (new_dim, new_strides)
}

/// Multi-dimensional matrix.
#[derive(Clone)]
pub struct TensorBase<S: Data, D: Dimension> {
    dim: D,
    strides: D,
    buffer: BufferBase<S>,
    offset: usize,
}

/// Owned Tensor
///
/// See [`TensorBase`].
pub type Tensor<T, D> = TensorBase<BufferRepr<T>, D>;
/// Tensor with 1 element
pub type Tensor0<T> = Tensor<T, Ix0>;
/// Tensor with 1 dimension
pub type Tensor1<T> = Tensor<T, Ix1>;
/// Tensor with 2 dimensions
pub type Tensor2<T> = Tensor<T, Ix2>;
/// Tensor with 3 dimensions
pub type Tensor3<T> = Tensor<T, Ix3>;
/// Tensor with 4 dimensions
pub type Tensor4<T> = Tensor<T, Ix4>;
/// Tensor with 5 dimensions
pub type Tensor5<T> = Tensor<T, Ix5>;
/// Tensor with 6 dimensions
pub type Tensor6<T> = Tensor<T, Ix6>;
/// Tensor with dynamic dimensions
pub type TensorD<T> = Tensor<T, IxDyn>;

/// Shared Tensor
///
/// See [`TensorBase`].
pub type ArcTensor<T, D> = TensorBase<ArcBufferRepr<T>, D>;
/// ArcTensor with 1 element
pub type ArcTensor0<T> = ArcTensor<T, Ix0>;
/// ArcTensor with 1 dimension
pub type ArcTensor1<T> = ArcTensor<T, Ix1>;
/// ArcTensor with 2 dimensions
pub type ArcTensor2<T> = ArcTensor<T, Ix2>;
/// ArcTensor with 3 dimensions
pub type ArcTensor3<T> = ArcTensor<T, Ix3>;
/// ArcTensor with 4 dimensions
pub type ArcTensor4<T> = ArcTensor<T, Ix4>;
/// ArcTensor with 5 dimensions
pub type ArcTensor5<T> = ArcTensor<T, Ix5>;
/// ArcTensor with 6 dimensions
pub type ArcTensor6<T> = ArcTensor<T, Ix6>;
/// ArcTensor with dynamic dimensions
pub type ArcTensorD<T> = ArcTensor<T, IxDyn>;

/// Borrowed Tensor
///
/// See [`TensorBase`].
pub type TensorView<'a, T, D> = TensorBase<SliceRepr<'a, T>, D>;
/// TensorView with 1 element
pub type TensorView0<'a, T> = TensorView<'a, T, Ix0>;
/// TensorView with 1 dimension
pub type TensorView1<'a, T> = TensorView<'a, T, Ix1>;
/// TensorView with 2 dimensions
pub type TensorView2<'a, T> = TensorView<'a, T, Ix2>;
/// TensorView with 3 dimensions
pub type TensorView3<'a, T> = TensorView<'a, T, Ix3>;
/// TensorView with 4 dimensions
pub type TensorView4<'a, T> = TensorView<'a, T, Ix4>;
/// TensorView with 5 dimensions
pub type TensorView5<'a, T> = TensorView<'a, T, Ix5>;
/// TensorView with 6 dimensions
pub type TensorView6<'a, T> = TensorView<'a, T, Ix6>;
/// TensorView with dynamic dimensions
pub type TensorViewD<'a, T> = TensorView<'a, T, IxDyn>;

/// Mutably borrowed Tensor
///
/// See [`TensorBase`].
pub type TensorViewMut<'a, T, D> = TensorBase<SliceMutRepr<'a, T>, D>;
/// TensorViewMut with 1 element
pub type TensorViewMut0<'a, T> = TensorViewMut<'a, T, Ix0>;
/// TensorViewMut with 1 dimension
pub type TensorViewMut1<'a, T> = TensorViewMut<'a, T, Ix1>;
/// TensorViewMut with 2 dimensions
pub type TensorViewMut2<'a, T> = TensorViewMut<'a, T, Ix2>;
/// TensorViewMut with 3 dimensions
pub type TensorViewMut3<'a, T> = TensorViewMut<'a, T, Ix3>;
/// TensorViewMut with 4 dimensions
pub type TensorViewMut4<'a, T> = TensorViewMut<'a, T, Ix4>;
/// TensorViewMut with 5 dimensions
pub type TensorViewMut5<'a, T> = TensorViewMut<'a, T, Ix5>;
/// TensorViewMut with 6 dimensions
pub type TensorViewMut6<'a, T> = TensorViewMut<'a, T, Ix6>;
/// TensorViewMut with dynamic dimensions
pub type TensorViewMutD<'a, T> = TensorViewMut<'a, T, IxDyn>;

/// Tensor that is either borrowed or owned.
///
/// See [`TensorBase`].
pub type CowTensor<'a, T, D> = TensorBase<CowBufferRepr<'a, T>, D>;
/// CowTensor with 1 element
pub type CowTensor0<'a, T> = CowTensor<'a, T, Ix0>;
/// CowTensor with 1 dimension
pub type CowTensor1<'a, T> = CowTensor<'a, T, Ix1>;
/// CowTensor with 2 dimensions
pub type CowTensor2<'a, T> = CowTensor<'a, T, Ix2>;
/// CowTensor with 3 dimensions
pub type CowTensor3<'a, T> = CowTensor<'a, T, Ix3>;
/// CowTensor with 4 dimensions
pub type CowTensor4<'a, T> = CowTensor<'a, T, Ix4>;
/// CowTensor with 5 dimensions
pub type CowTensor5<'a, T> = CowTensor<'a, T, Ix5>;
/// CowTensor with 6 dimensions
pub type CowTensor6<'a, T> = CowTensor<'a, T, Ix6>;
/// CowTensor with dynamic dimensions
pub type CowTensorD<'a, T> = CowTensor<'a, T, IxDyn>;

impl<T: Scalar, S: DataOwned<Elem = T>, D: Dimension> TensorBase<S, D> {
    /// Allocates a tensor on `device` with `shape`.
    ///
    /// # Safety
    ///
    /// The tensor is not initialized.
    ///
    /// **Errors**
    /// See [`Buffer::uninit()`].
    pub unsafe fn uninit<Sh>(device: Device, shape: Sh) -> Result<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        let (dim, strides) = dim_strides_from_shape(shape.into_shape());
        let buffer = unsafe { BufferBase::uninit(device, dim.size())? };
        Ok(Self {
            dim,
            strides,
            buffer,
            offset: 0,
        })
    }
    /// Creates a tensor on `device` with `shape` filled with `elem`.
    ///
    /// **Errors**
    /// See [`Buffer::from_elem()`].
    pub fn from_elem<Sh>(device: Device, shape: Sh, elem: T) -> Result<Self>
    where
        T: Scalar,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        let (dim, strides) = dim_strides_from_shape(shape.into_shape());
        let buffer = BufferBase::from_elem(device, dim.size(), elem)?;
        Ok(Self {
            dim,
            strides,
            buffer,
            offset: 0,
        })
    }
    /// Creates a tensor on `device` with `shape` filled with 0's.
    ///
    /// **Errors**
    /// See [`Buffer::zeros()`].
    pub fn zeros<Sh>(device: Device, shape: Sh) -> Result<Self>
    where
        T: Scalar,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_elem(device, shape, T::default())
    }
    /// Creates a tensor on `device` with `shape` filled with 1's.
    ///
    /// **Errors**
    /// See [`Buffer::ones()`].
    pub fn ones<Sh>(device: Device, shape: Sh) -> Result<Self>
    where
        T: Scalar,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_elem(device, shape, T::one())
    }
}

impl<T: Scalar, S: Data<Elem = T>, D: Dimension> TensorBase<S, D> {
    /// The device of the tensor.
    pub fn device(&self) -> Device {
        self.buffer.device()
    }
    /// The dimensions of the tensor in pattern form.
    pub fn dim(&self) -> D::Pattern {
        self.dim.clone().into_pattern()
    }
    /// The dimensions of the tensor.
    pub fn raw_dim(&self) -> D {
        self.dim.clone()
    }
    /// The dimensions of the tensor as a slice.
    pub fn shape(&self) -> &[usize] {
        self.dim.slice()
    }
    /// The strides of the tensor as a slice.
    pub fn strides(&self) -> &[isize] {
        bytemuck::cast_slice(self.strides.slice())
    }
    /// The length of the tensor.
    pub fn len(&self) -> usize {
        self.dim.size()
    }
    /// Whether the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.shape().iter().any(|x| *x == 0)
    }
    /// The dimensionality of the tensor.
    pub fn ndim(&self) -> usize {
        self.dim.ndim()
    }
    /// Converts the tensor into dimension `D2`.
    ///
    /// Typically this is used to downcast from [`IxDyn`](type@ndarray::IxDyn) to a static dimensionality. For conversions to [`IxDyn`](type@ndarray::IxDyn), use [`.into_dyn()`](TensorBase::into_dyn()).
    ///
    /// **Errors**
    /// The number of axes of `D2` must be the same as `D`.
    pub fn into_dimensionality<D2>(self) -> Result<TensorBase<S, D2>>
    where
        D2: Dimension,
    {
        let (dim, strides) = into_dimensionality(&self.dim, &self.strides)?;
        Ok(TensorBase {
            dim,
            strides,
            buffer: self.buffer,
            offset: self.offset,
        })
        /*
        if let Some(dim) = D2::from_dimension(&self.dim) {
            if let Some(strides) = D2::from_dimension(&self.strides) {
                return Ok(TensorBase {
                    dim,
                    strides,
                    data: self.data,
                });
            }
        }
        let strides = bytemuck::cast_slice::<_, isize>(self.strides());
        Err(anyhow!(
            "Incompatible Shapes! {:?} {:?} => {:?}",
            self.shape(),
            strides,
            D2::NDIM
        ))*/
    }
    /// Returns the tensor with dim `shape`.
    ///
    /// **Errors**
    /// The tensor must be contiguous, with default strides.
    pub fn into_shape<E>(self, shape: E) -> Result<TensorBase<S, E::Dim>>
    where
        E: IntoDimension,
    {
        let (dim, strides) = into_shape(&self.dim, &self.strides, shape)?;
        assert_eq!(self.offset, 0);
        Ok(TensorBase {
            dim,
            strides,
            buffer: self.buffer,
            offset: self.offset,
        })
        /*
        let dim = shape.into_dimension();
        // TODO potentially handle Fotran layout
        if self.dim.size() == dim.size() && self.strides == self.dim.default_strides() {
            let strides = dim.default_strides();
            return Ok(TensorBase {
                dim,
                strides,
                data: self.data,
            });
        }
        Err(anyhow!("Incompatible Shapes!"))
        */
    }
    /// Converts the dimensionality of the tensor to [`IxDyn`](type@ndarray::IxDyn).
    pub fn into_dyn(self) -> TensorBase<S, IxDyn> {
        TensorBase {
            dim: self.dim.into_dyn(),
            strides: self.strides.into_dyn(),
            buffer: self.buffer,
            offset: self.offset,
        }
    }
    /// Borrows the tensor as a [`TensorView`].
    pub fn view(&self) -> TensorView<T, D> {
        TensorView {
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            buffer: self.buffer.as_slice(),
            offset: self.offset,
        }
    }
    /// Borrows the tensor as a [`TensorViewMut`].
    pub fn view_mut(&mut self) -> TensorViewMut<T, D>
    where
        S: DataMut,
    {
        TensorViewMut {
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            buffer: self.buffer.as_slice_mut(),
            offset: self.offset,
        }
    }
    /// Whether the tensor is contiguous.
    ///
    /// Contiguous is either C (Standard) or Fortran layout.
    pub fn is_contiguous(&self) -> bool {
        is_contiguous(&self.dim, &self.strides, self.offset)
    }
    /// Whether the tensor is standard layout.
    ///
    /// In standard layout, the strides increase from right to left by the product of each dimension.
    pub fn is_standard_layout(&self) -> bool {
        is_standard_layout(&self.dim, &self.strides, self.offset)
    }
    /// Permute the axes of the tensor.
    ///
    /// Reorders the dimensions of the tensor, where for each a in `axes`, a is the index of that axis in the new tensor.
    ///
    /// # Note
    /// This operation merely reorders the dimensions / strides and does not copy the data. Combine with [`.into_standard_layout()`](TensorBase::into_standard_layout()) to execute the operation, returning a tensor in standard layout.
    ///
    /// **Errors**
    ///
    /// Each axis 0 .. ndim must be used exactly once.
    pub fn permuted_axes<A>(self, axes: A) -> Self
    where
        A: IntoDimension<Dim = D>,
    {
        let (dim, strides) = permuted_axes(self.dim, self.strides, axes.into_dimension());
        Self {
            dim,
            strides,
            ..self
        }
    }
    /// Reverses (transposes) the axes of the tensor.
    pub fn reversed_axes(mut self) -> Self {
        self.dim.slice_mut().reverse();
        self.strides.slice_mut().reverse();
        self
    }
    /// Retunrs a view with reversed (transposed) axes.
    pub fn t(&self) -> TensorView<T, D> {
        self.view().reversed_axes()
    }
    /// Borrows the tensor as a [`Slice`] if standard layout.
    pub fn as_slice(&self) -> Option<Slice<T>> {
        if self.is_standard_layout() {
            Some(self.buffer.as_slice())
        } else {
            None
        }
    }
    /// Borrows the tensor as a [`Slice`] if contiguous.
    pub fn as_slice_memory_order(&self) -> Option<Slice<T>> {
        if self.is_contiguous() {
            Some(self.buffer.as_slice())
        } else {
            None
        }
    }
    /// Mutably borrows the tensor as a [`SliceMut`] if standard layout.
    pub fn as_slice_mut(&mut self) -> Option<SliceMut<T>>
    where
        S: DataMut,
    {
        if self.is_standard_layout() {
            Some(self.buffer.as_slice_mut())
        } else {
            None
        }
    }
    /// Mutably borrows the tensor as a [`SliceMut`] if contiguous.
    pub fn as_slice_memory_order_mut(&mut self) -> Option<SliceMut<T>>
    where
        S: DataMut,
    {
        if self.is_contiguous() {
            Some(self.buffer.as_slice_mut())
        } else {
            None
        }
    }
    /// Transfers the tensor into the `device`.
    ///
    /// See [`Buffer::into_device()`](crate::device::buffer::BufferBase::into_device()).
    ///
    /// **Errors**
    /// See [`BufferBase::into_device()`].
    pub fn into_device(self, device: Device) -> Result<Tensor<T, D>>
    where
        T: Pod,
    {
        if device == self.device() {
            self.into_owned()
        } else if !self.is_contiguous() {
            todo!()
        } else {
            let buffer = self.buffer.to_device(device)?;
            Ok(Tensor {
                dim: self.dim,
                strides: self.strides,
                buffer,
                offset: 0,
            })
        }
    }
    /// Converts into a [`Tensor`].
    pub fn into_owned(self) -> Result<Tensor<T, D>>
    where
        T: Copy,
    {
        if !self.is_contiguous() {
            todo!();
        }
        Ok(TensorBase {
            dim: self.dim,
            strides: self.strides,
            buffer: self.buffer.into_owned()?,
            offset: 0,
        })
    }
    /// Converts to a [`Tensor`].
    pub fn to_owned(&self) -> Result<Tensor<T, D>>
    where
        T: Copy,
    {
        self.view().into_owned()
    }
    /// Converts into an [`ArcTensor`].
    pub fn into_shared(self) -> Result<ArcTensor<T, D>>
    where
        T: Copy,
    {
        if !self.is_contiguous() {
            todo!()
        }
        Ok(TensorBase {
            dim: self.dim,
            strides: self.strides,
            buffer: self.buffer.into_shared()?,
            offset: 0,
        })
    }
    /// Converts to an [`ArcTensor`].
    ///
    /// For [`ArcTensor`] clones the [`Arc`](std::sync::Arc), otherwise copies the data into a new [`ArcTensor`].
    pub fn to_shared(&self) -> Result<ArcTensor<T, D>>
    where
        T: Copy,
    {
        if !self.is_contiguous() {
            todo!()
        }
        Ok(TensorBase {
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            buffer: self.buffer.to_shared()?,
            offset: 0,
        })
    }
    /// Fills the tensor with `elem`.
    ///
    /// **Errors**
    ///
    /// See [`BufferBase::fill()`](crate::buffer::BufferBase::fill()).
    pub fn fill(&mut self, elem: T) -> Result<()>
    where
        T: Scalar,
        S: DataMut,
    {
        self.buffer.as_slice_mut().fill(elem)
    }
    /// Moves the tensor into an [`Array`].
    ///
    /// **errors**
    /// See [`Buffer::into_vec()`].
    pub fn into_array(self) -> Result<Array<T, D>> {
        if !self.is_contiguous() {
            todo!()
        }
        let vec = self.buffer.into_vec()?;
        Ok(Array::from_shape_vec(self.dim.clone().strides(self.strides.clone()), vec).unwrap())
    }
    /// Returns the tensor as an array if on the host and standard layout
    // TODO: impl if contiguous
    pub fn as_array(&self) -> Option<ArrayView<T, D>> {
        if self.is_contiguous() {
            if let Some(host_slice) = self.buffer.as_host_slice() {
                return Some(
                    ArrayView::from_shape(
                        self.dim.clone().strides(self.strides.clone()),
                        host_slice,
                    )
                    .unwrap(),
                );
            }
        }
        None
    }
    pub fn as_array_mut(&mut self) -> Option<ArrayViewMut<T, D>>
    where
        S: DataMut,
    {
        if self.is_contiguous() {
            if let Some(host_slice) = self.buffer.as_host_slice_mut() {
                return Some(
                    ArrayViewMut::from_shape(
                        self.dim.clone().strides(self.strides.clone()),
                        host_slice,
                    )
                    .unwrap(),
                );
            }
        }
        None
    }
}

impl<T: Scalar, S: DataOwned<Elem = T>> From<Buffer<T>> for TensorBase<S, Ix1> {
    fn from(buffer: Buffer<T>) -> Self {
        let dim = buffer.len().into_dimension();
        let strides = dim.default_strides();
        let buffer = BufferBase::from_buffer(buffer);
        Self {
            dim,
            strides,
            buffer,
            offset: 0,
        }
    }
}

impl<'a, T: Scalar> From<Slice<'a, T>> for TensorView<'a, T, Ix1> {
    fn from(slice: Slice<'a, T>) -> Self {
        let dim = slice.len().into_dimension();
        let strides = dim.default_strides();
        Self {
            dim,
            strides,
            buffer: slice,
            offset: 0,
        }
    }
}

impl<T: Scalar, S: DataOwned<Elem = T>, D: Dimension> From<Array<T, D>> for TensorBase<S, D> {
    fn from(array: Array<T, D>) -> Self {
        let dim = array.raw_dim();
        let strides = strides_from_array(&array);
        let buffer = BufferBase::from_vec(array.into_raw_vec());
        Self {
            dim,
            strides,
            buffer,
            offset: 0,
        }
    }
}

impl<'a, T: Scalar, D: Dimension> From<ArrayView<'a, T, D>> for CowTensor<'a, T, D> {
    fn from(array: ArrayView<'a, T, D>) -> Self {
        if let Some(slice) = array.to_slice_memory_order() {
            let dim = array.raw_dim();
            let strides = strides_from_array(&array);
            let buffer = Slice::from(slice).into();
            Self {
                dim,
                strides,
                buffer,
                offset: 0,
            }
        } else {
            Self::from(array.to_owned())
        }
    }
}

impl<'a, T: Scalar, D: Dimension> TryFrom<ArrayView<'a, T, D>> for TensorView<'a, T, D> {
    type Error = anyhow::Error;
    fn try_from(array: ArrayView<'a, T, D>) -> Result<Self> {
        let slice = array
            .as_slice_memory_order()
            .ok_or_else(|| anyhow!("Shape not contiguous!"))?;
        // We want to return 'a, not a new borrow.
        let slice = unsafe { std::slice::from_raw_parts(slice.as_ptr(), slice.len()) };
        let dim = array.raw_dim();
        let strides = strides_from_array(&array);
        Ok(Self {
            dim,
            strides,
            buffer: slice.into(),
            offset: 0,
        })
    }
}

impl<'a, T: Scalar, D: Dimension> From<TensorView<'a, T, D>> for CowTensor<'a, T, D> {
    fn from(view: TensorView<'a, T, D>) -> Self {
        Self {
            dim: view.dim,
            strides: view.strides,
            buffer: view.buffer.into(),
            offset: view.offset,
        }
    }
}

impl<T: Scalar, D: Dimension> From<Tensor<T, D>> for CowTensor<'_, T, D> {
    fn from(tensor: Tensor<T, D>) -> Self {
        Self {
            dim: tensor.dim,
            strides: tensor.strides,
            buffer: tensor.buffer.into(),
            offset: tensor.offset,
        }
    }
}

impl<T: Scalar, D: Dimension> From<Tensor<T, D>> for ArcTensor<T, D> {
    fn from(tensor: Tensor<T, D>) -> Self {
        Self {
            dim: tensor.dim,
            strides: tensor.strides,
            buffer: tensor.buffer.into(),
            offset: tensor.offset,
        }
    }
}

/*
impl<S: Data + Clone, D: Dimension> Clone for TensorBase<S, D> {
    fn clone(&self) -> Self {
        Self {
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            buffer: self.buffer.clone(),
            offset: self.offset.clone(),
        }
    }
}*/

/// Casts
#[allow(unused)]
impl<T: Scalar, S: Data<Elem = T>, D: Dimension> TensorBase<S, D> {
    /// Casts the tensor into a new tensor.
    ///
    /// See [`BufferBase::cast_into()`](crate::buffer::BufferBase::cast_into()).
    pub fn cast_into<Y: Scalar>(self) -> Result<Tensor<Y, D>> {
        if !self.is_contiguous() {
            todo!()
        }
        Ok(TensorBase {
            dim: self.dim,
            strides: self.strides,
            buffer: self.buffer.cast_into()?,
            offset: 0,
        })
    }
    /// Casts the tensor to a new tensor.
    ///
    /// See [`BufferBase::cast()`](crate::buffer::BufferBase::cast()).
    pub fn cast<Y: Scalar>(&self) -> Result<Tensor<Y, D>> {
        if !self.is_contiguous() {
            todo!();
        }
        Ok(TensorBase {
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            buffer: self.buffer.cast()?,
            offset: 0,
        })
    }
    /// Scales the tensor into a new tensor.
    ///
    /// See [`BufferBase::scale_into()`](crate::buffer::BufferBase::scale_into()).
    pub fn scale_into<T2: Scalar>(self, alpha: T2) -> Result<Tensor<T2, D>> {
        todo!()
        /*
        let buffer = match self.data.try_into_buffer() {
            Ok(buffer) => buffer.scale_into(alpha)?,
            Err(data) => data.as_slice().scale_into(alpha)?,
        };
        Ok(TensorBase {
            dim: self.dim,
            strides: self.strides,
            data: OwnedRepr(buffer),
        })*/
    }
}

/*
#[allow(dead_code)]
impl<T: Uint, S: Data<Elem = T>> TensorBase<S, Ix1> {
    pub(crate) fn to_one_hot<T2: Scalar>(&self, nclasses: usize) -> Result<Tensor2<T2>> {
        let n = self.dim();
        let mut output = unsafe { Tensor::alloc(self.device(), [n, nclasses])? };
        if size_of::<T2>() < 4 {
            output.fill(T2::zero())?;
        }
        todo!()
        /*
        let builder = glsl_shaders::module(&format!(
            "one_hot_{}_{}",
            T::scalar_name(),
            T2::scalar_name()
        ))?
        .compute_pass("main")?
        .slice(self.as_raw_slice())?
        .slice_mut(output.as_raw_slice_mut())?
        .push([n as u32, nclasses as u32])?;
        unsafe {
            builder.submit([n as u32, 1, 1])?;
        }
        Ok(output)*/
    }
}*/

/*
#[cfg(test)]
mod tests {
    #[allow(unused)]
    use super::*;
    #[cfg(feature = "device_tests")]
    use half::bf16;
    #[cfg(feature = "device_tests")]
    use ndarray::{Array1, Array2};

    async fn tensor_from_array<D: Dimension>(x: Array<u32, D>) -> Result<()> {
        let y = TensorView::try_from(x.view())?.read().await?;
        assert_eq!(x.view(), y.as_array());
        let y_t = TensorView::try_from(x.t())?.read().await?;
        assert_eq!(x.t(), y_t.as_array());
        Ok(())
    }

    #[tokio::test]
    async fn tensor_from_array0() -> Result<()> {
        tensor_from_array(Array::from_elem((), 1)).await
    }

    #[tokio::test]
    async fn tensor_from_array1() -> Result<()> {
        tensor_from_array(Array::from_shape_vec(3, (1..=3).into_iter().collect())?).await
    }

    #[tokio::test]
    async fn tensor_from_array2() -> Result<()> {
        tensor_from_array(Array::from_shape_vec(
            [2, 3],
            (1..=6).into_iter().collect(),
        )?)
        .await
    }

    #[tokio::test]
    async fn tensor_from_array3() -> Result<()> {
        tensor_from_array(Array::from_shape_vec(
            [2, 3, 4],
            (1..=24).into_iter().collect(),
        )?)
        .await
    }

    #[tokio::test]
    async fn tensor_from_array4() -> Result<()> {
        tensor_from_array(Array::from_shape_vec(
            [2, 3, 4, 5],
            (1..=120).into_iter().collect(),
        )?)
        .await
    }

    #[tokio::test]
    async fn test_from_array5() -> Result<()> {
        tensor_from_array(Array::from_shape_vec(
            [2, 3, 4, 5, 6],
            (1..=120 * 6).into_iter().collect(),
        )?)
        .await
    }

    #[tokio::test]
    async fn tensor_from_array6() -> Result<()> {
        tensor_from_array(Array::from_shape_vec(
            [2, 3, 4, 5, 6, 7],
            (1..=120 * 6 * 7).into_iter().collect(),
        )?)
        .await
    }

    #[allow(non_snake_case)]
    #[tokio::test]
    async fn tensor_from_arrayD() -> Result<()> {
        tensor_from_array(Array::from_shape_vec(
            [2, 3, 4, 5, 6, 7, 8].as_ref(),
            (1..=120 * 6 * 7 * 8).into_iter().collect(),
        )?)
        .await
    }

    async fn tensor_serde(device: Device) -> Result<()> {
        let x = (0..4 * 5 * 6 * 7).into_iter().collect::<Vec<u32>>();
        let array = Array::from(x).into_shape([4, 5, 6, 7])?;
        let tensor = TensorView::try_from(array.view())?
            .into_device(device)
            .await?;
        let tensor: Tensor4<u32> = bincode::deserialize(&bincode::serialize(&tensor)?)?;
        assert_eq!(array.view(), tensor.read().await?.as_array());
        Ok(())
    }

    #[tokio::test]
    async fn tensor_serde_host() -> Result<()> {
        tensor_serde(Device::host()).await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn tensor_serde_device() -> Result<()> {
        let device = Device::new()?;
        tensor_serde(device).await
    }

    /*
    use autograph::{
        backend::Device,
        tensor::{Num, Scalar, Tensor, Unsigned},
        Result,
    };
    use half::bf16;
    use ndarray::{Array, Array1, Array2};
    use num_traits::{FromPrimitive, ToPrimitive};
    */

    #[cfg(feature = "device_tests")]
    fn array_scaled_cast<T1: Scalar, T2: Scalar>(x: &Array1<T1>, alpha: f64) -> Array1<T2> {
        x.iter()
            .map(|x| T2::from_f64(x.to_f64().unwrap() * alpha).unwrap())
            .collect()
    }

    #[cfg(feature = "device_tests")]
    async fn scaled_cast<T1: Scalar + From<u8>, T2: Scalar + From<u8>>() -> Result<()> {
        let n = 100;
        let alpha = 2;
        let data: Vec<T1> = (0..n as u8).into_iter().map(Into::into).collect();
        let x_array = Array::from(data);
        let y_true = array_scaled_cast(&x_array, alpha.into());
        let device = Device::new()?;
        let x = CowTensor::from(x_array.view()).into_device(device).await?;
        let y = x.scale_into::<T2>((alpha as u8).into())?;
        let y_array = y.read().await?;
        assert_eq!(y_array.as_array(), y_true.view());
        Ok(())
    }

    #[cfg_attr(windows, ignore)]
    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scaled_cast_u8_bf16() -> Result<()> {
        scaled_cast::<u8, bf16>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scaled_cast_u8_u32() -> Result<()> {
        scaled_cast::<u8, u32>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scaled_cast_u8_i32() -> Result<()> {
        scaled_cast::<u8, i32>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scaled_cast_u8_f32() -> Result<()> {
        scaled_cast::<u8, f32>().await
    }

    #[cfg_attr(windows, ignore)]
    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scaled_cast_u16_bf16() -> Result<()> {
        scaled_cast::<u16, bf16>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scaled_cast_u16_u32() -> Result<()> {
        scaled_cast::<u16, u32>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scaled_cast_u16_i32() -> Result<()> {
        scaled_cast::<u16, i32>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scaled_cast_u16_f32() -> Result<()> {
        scaled_cast::<u16, f32>().await
    }

    #[cfg_attr(windows, ignore)]
    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scaled_cast_bf16_bf16() -> Result<()> {
        scaled_cast::<bf16, bf16>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scaled_cast_bf16_u32() -> Result<()> {
        scaled_cast::<bf16, u32>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scaled_cast_bf16_i32() -> Result<()> {
        scaled_cast::<bf16, i32>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scaled_cast_bf16_f32() -> Result<()> {
        scaled_cast::<bf16, f32>().await
    }

    #[cfg_attr(windows, ignore)]
    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scaled_cast_u32_bf16() -> Result<()> {
        scaled_cast::<u32, bf16>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scaled_cast_u32_u32() -> Result<()> {
        scaled_cast::<u32, u32>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scaled_cast_u32_i32() -> Result<()> {
        scaled_cast::<u32, i32>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scaled_cast_u32_f32() -> Result<()> {
        scaled_cast::<u32, f32>().await
    }

    #[cfg_attr(windows, ignore)]
    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scaled_cast_i32_bf16() -> Result<()> {
        scaled_cast::<i32, bf16>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scaled_cast_i32_u32() -> Result<()> {
        scaled_cast::<i32, u32>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scaled_cast_i32_i32() -> Result<()> {
        scaled_cast::<i32, i32>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scaled_cast_i32_f32() -> Result<()> {
        scaled_cast::<i32, f32>().await
    }

    #[cfg(feature = "device_tests")]
    fn to_one_hot<U: Uint, T: Scalar>(x: &Array1<U>, nclasses: usize) -> Array2<T> {
        let mut y = Array::from_elem([x.len(), nclasses], T::zero());
        for (mut y, x) in y.outer_iter_mut().zip(x.iter().copied()) {
            y[x.to_usize().unwrap()] = T::one();
        }
        y
    }

    #[cfg(feature = "device_tests")]
    async fn one_hot<U: Uint + Into<u64> + From<u8>, T: Scalar>() -> Result<()> {
        let batch_size = 100;
        let nclasses = 10;
        let data: Vec<U> = (0..nclasses as u8)
            .into_iter()
            .cycle()
            .take(batch_size)
            .map(Into::into)
            .collect();
        let x_array = Array::from(data.clone());
        let y_true = to_one_hot(&x_array, nclasses);
        let device = Device::new()?;
        let x = CowTensor::from(x_array.view()).into_device(device).await?;
        let y = x.to_one_hot::<T>(nclasses)?;
        let y_array = y.read().await?;
        assert_eq!(y_array.as_array(), y_true.view());
        Ok(())
    }

    #[cfg_attr(windows, ignore)]
    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn one_hot_u8_bf16() -> Result<()> {
        one_hot::<u8, bf16>().await
    }

    #[cfg_attr(windows, ignore)]
    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn one_hot_u16_bf16() -> Result<()> {
        one_hot::<u16, bf16>().await
    }

    #[cfg_attr(windows, ignore)]
    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn one_hot_u32_bf16() -> Result<()> {
        one_hot::<u32, bf16>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn one_hot_u8_u32() -> Result<()> {
        one_hot::<u8, u32>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn one_hot_u16_u32() -> Result<()> {
        one_hot::<u16, u32>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn one_hot_u32_u32() -> Result<()> {
        one_hot::<u32, u32>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn one_hot_u8_i32() -> Result<()> {
        one_hot::<u8, i32>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn one_hot_u16_i32() -> Result<()> {
        one_hot::<u16, i32>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn one_hot_u32_i32() -> Result<()> {
        one_hot::<u32, i32>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn one_hot_u8_f32() -> Result<()> {
        one_hot::<u8, f32>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn one_hot_u16_f32() -> Result<()> {
        one_hot::<u16, f32>().await
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn one_hot_u32_f32() -> Result<()> {
        one_hot::<u32, f32>().await
    }
}
*/
