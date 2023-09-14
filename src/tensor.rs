/*!
```
# use krnl::device::Device;
use autograph::tensor::{Tensor, TensorView};
use anyhow::Result;
use ndarray::{Array, arr2, linalg::Dot};

# fn main() -> Result<()> {
# let device = Device::host();
// Create a tensor from an array
let a = Tensor::from(arr2(&[
    [1f32, 2.],
    [3., 4.],
]))
    // Moves to device, no copy if device is Device::host().
    .into_device(device.clone())?;
// Alternatively, create a tensor from an array view.
let a = arr2(&[
    [1f32, 2.],
    [3., 4.],
]);
// This will fail if the view has negative strides.
let a = TensorView::try_from(a.view()).unwrap()
    .to_device(device.clone())?;
// Create a tensor from a vec, same as above.
let b = Tensor::from(vec![5f32, 6., 7., 8.]).into_shape([2, 2]).unwrap()
    .into_device(device.clone())?;
// Compute a dot product (matrix multiplication) and move back to host.
let c = a.dot(&b)?.into_device(Device::host())?;
// Borrow as an array view.
let c_view = c.as_array().unwrap();
// Move into an array
let c = c.into_array()?;

# Ok(())
# }
```
*/

#[cfg(doc)]
use crate::device::error::DeviceLost;
use crate::{
    buffer::{
        ArcBufferRepr, Buffer, BufferBase, BufferRepr, CowBuffer, CowBufferRepr, Data, DataMut,
        DataOwned, ScalarArcBufferRepr, ScalarBuffer, ScalarBufferBase, ScalarBufferRepr,
        ScalarCowBuffer, ScalarCowBufferRepr, ScalarData, ScalarDataMut, ScalarDataOwned,
        ScalarSlice, ScalarSliceMut, ScalarSliceMutRepr, ScalarSliceRepr, Slice, SliceMut,
        SliceMutRepr, SliceRepr,
    },
    device::Device,
    scalar::{Scalar, ScalarElem, ScalarType},
};
use anyhow::{anyhow, bail, Result};
use dry::macro_for;
use krnl::krnl_core::half::bf16;
use ndarray::{
    Array, ArrayBase, ArrayView, ArrayViewMut, Axis, Dimension, IntoDimension, Ix0, Ix1, Ix2, Ix3,
    Ix4, Ix5, Ix6, IxDyn, RawArrayView, RemoveAxis, ShapeBuilder, ShapeError, StrideShape,
};
use num_traits::ToPrimitive;
use paste::paste;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    convert::{TryFrom, TryInto},
    fmt::{self, Debug},
};

mod linalg;
mod ops;
mod reduce;

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

fn into_dimensionality<D1, D2>(dim: &D1, strides: &D1) -> Result<(D2, D2), ShapeError>
where
    D1: Dimension,
    D2: Dimension,
{
    D2::from_dimension(dim)
        .and_then(|dim| D2::from_dimension(strides).map(|strides| (dim, strides)))
        .ok_or(ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape))
}

fn into_shape<D1, E>(dim: &D1, strides: &D1, shape: E) -> Result<(E::Dim, E::Dim), ShapeError>
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
        Err(ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape))
    }
}

pub(crate) fn flatten(shape: &[usize]) -> [usize; 2] {
    let mut iter = shape.iter().copied();
    let rows = iter.next().unwrap_or(1);
    let cols = iter.product();
    [rows, cols]
}

fn is_contiguous<D: Dimension>(dim: &D, strides: &D) -> bool {
    let zero_strides = strides.slice().iter().any(|s| *s == 0);
    zero_strides || strides == &dim.default_strides() || strides == &dim.fortran_strides()
}

fn is_standard_layout<D: Dimension>(dim: &D, strides: &D) -> bool {
    let zero_strides = strides.slice().iter().any(|s| *s == 0);
    zero_strides || strides == &dim.default_strides()
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

// adapted from https://docs.rs/crate/ndarray/0.15.6/source/src/dimension/mod.rs
/// Returns the `size` of the `dim`, checking that the product of non-zero axis
/// lengths does not exceed `isize::MAX`.
///
/// If `size_of_checked_shape(dim)` returns `Ok(size)`, the data buffer is a
/// slice or `Vec` of length `size`, and `strides` are created with
/// `self.default_strides()` or `self.fortran_strides()`, then the invariants
/// are met to construct an array from the data buffer, `dim`, and `strides`.
/// (The data buffer being a slice or `Vec` guarantees that it contains no more
/// than `isize::MAX` bytes.)
fn size_of_shape_checked<D: Dimension>(dim: &D) -> Result<usize, ShapeError> {
    use ndarray::ErrorKind;
    let size_nonzero = dim
        .slice()
        .iter()
        .filter(|&&d| d != 0)
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| ShapeError::from_kind(ErrorKind::Overflow))?;
    if size_nonzero > ::std::isize::MAX as usize {
        Err(ShapeError::from_kind(ErrorKind::Overflow))
    } else {
        Ok(dim.size())
    }
}

// adapted from https://docs.rs/ndarray/0.15.6/ndarray/struct.ArrayBase.html#method.broadcast
fn broadcast<D: Dimension, E: IntoDimension>(
    from: &D,
    strides: &D,
    dim: E,
) -> Option<(E::Dim, E::Dim)> {
    /// Return new stride when trying to grow `from` into shape `to`
    ///
    /// Broadcasting works by returning a "fake stride" where elements
    /// to repeat are in axes with 0 stride, so that several indexes point
    /// to the same element.
    ///
    /// **Note:** Cannot be used for mutable iterators, since repeating
    /// elements would create aliasing pointers.
    fn upcast<D: Dimension, E: Dimension>(to: &D, from: &E, stride: &E) -> Option<D> {
        // Make sure the product of non-zero axis lengths does not exceed
        // `isize::MAX`. This is the only safety check we need to perform
        // because all the other constraints of `ArrayBase` are guaranteed
        // to be met since we're starting from a valid `ArrayBase`.
        let _ = size_of_shape_checked(to).ok()?;

        let mut new_stride = to.clone();
        // begin at the back (the least significant dimension)
        // size of the axis has to either agree or `from` has to be 1
        if to.ndim() < from.ndim() {
            return None;
        }

        {
            let mut new_stride_iter = new_stride.slice_mut().iter_mut().rev();
            for ((er, es), dr) in from
                .slice()
                .iter()
                .rev()
                .zip(stride.slice().iter().rev())
                .zip(new_stride_iter.by_ref())
            {
                /* update strides */
                if *dr == *er {
                    /* keep stride */
                    *dr = *es;
                } else if *er == 1 {
                    /* dead dimension, zero stride */
                    *dr = 0
                } else {
                    return None;
                }
            }

            /* set remaining strides to zero */
            for dr in new_stride_iter {
                *dr = 0;
            }
        }
        Some(new_stride)
    }
    let dim = dim.into_dimension();

    // Note: zero strides are safe precisely because we return an read-only view
    let broadcast_strides = match upcast(&dim, from, strides) {
        Some(st) => st,
        None => return None,
    };
    Some((dim, broadcast_strides))
}

fn collapse_axis<D: Dimension>(dims: &mut D, strides: &D, Axis(axis): Axis, index: usize) -> isize {
    let dim = dims[axis];
    assert!(index < dim);
    dims[0] = 1;
    index as isize * strides[axis] as isize
}

/// Dynamically typed multi-dimensional matrix.
///
/// Use [`TryInto`] to convert into a [`TensorBase`].
/// Use [`From`] to convert from a ['TensorBase'].
#[derive(Clone)]
pub struct ScalarTensorBase<S: ScalarData, D: Dimension> {
    dim: D,
    strides: D,
    buffer: ScalarBufferBase<S>,
    offset: usize,
}

/// Owned Scalar Tensor
///
/// See [`ScalarTensorBase`].
pub type ScalarTensor<D> = ScalarTensorBase<ScalarBufferRepr, D>;
/// ScalarTensor with 1 element
pub type ScalarTensor0 = ScalarTensor<Ix0>;
/// ScalarTensor with 1 dimension
pub type ScalarTensor1 = ScalarTensor<Ix1>;
/// ScalarTensor with 2 dimensions
pub type ScalarTensor2 = ScalarTensor<Ix2>;
/// ScalarTensor with 3 dimensions
pub type ScalarTensor3 = ScalarTensor<Ix3>;
/// ScalarTensor with 4 dimensions
pub type ScalarTensor4 = ScalarTensor<Ix4>;
/// ScalarTensor with 5 dimensions
pub type ScalarTensor5 = ScalarTensor<Ix5>;
/// ScalarTensor with 6 dimensions
pub type ScalarTensor6 = ScalarTensor<Ix6>;
/// ScalarTensor with dynamic dimensions
pub type ScalarTensorD = ScalarTensor<IxDyn>;

/// Shared Scalar Tensor
///
/// See [`ScalarTensorBase`].
pub type ScalarArcTensor<D> = ScalarTensorBase<ScalarArcBufferRepr, D>;
/// ScalarArcTensor with 1 element
pub type ScalarArcTensor0 = ScalarArcTensor<Ix0>;
/// ScalarArcTensor with 1 dimension
pub type ScalarArcTensor1 = ScalarArcTensor<Ix1>;
/// ScalarArcTensor with 2 dimensions
pub type ScalarArcTensor2 = ScalarArcTensor<Ix2>;
/// ScalarArcTensor with 3 dimensions
pub type ScalarArcTensor3 = ScalarArcTensor<Ix3>;
/// ScalarArcTensor with 4 dimensions
pub type ScalarArcTensor4 = ScalarArcTensor<Ix4>;
/// ScalarArcTensor with 5 dimensions
pub type ScalarArcTensor5 = ScalarArcTensor<Ix5>;
/// ScalarArcTensor with 6 dimensions
pub type ScalarArcTensor6 = ScalarArcTensor<Ix6>;
/// ScalarArcTensor with dynamic dimensions
pub type ScalarArcTensorD = ScalarArcTensor<IxDyn>;

/// Borrowed Scalar Tensor
///
/// See [`ScalarTensorBase`].
pub type ScalarTensorView<'a, D> = ScalarTensorBase<ScalarSliceRepr<'a>, D>;
/// ScalarTensorView with 1 element
pub type ScalarTensorView0<'a> = ScalarTensorView<'a, Ix0>;
/// ScalarTensorView with 1 dimension
pub type ScalarTensorView1<'a> = ScalarTensorView<'a, Ix1>;
/// ScalarTensorView with 2 dimensions
pub type ScalarTensorView2<'a> = ScalarTensorView<'a, Ix2>;
/// ScalarTensorView with 3 dimensions
pub type ScalarTensorView3<'a> = ScalarTensorView<'a, Ix3>;
/// ScalarTensorView with 4 dimensions
pub type ScalarTensorView4<'a> = ScalarTensorView<'a, Ix4>;
/// ScalarTensorView with 5 dimensions
pub type ScalarTensorView5<'a> = ScalarTensorView<'a, Ix5>;
/// ScalarTensorView with 6 dimensions
pub type ScalarTensorView6<'a> = ScalarTensorView<'a, Ix6>;
/// ScalarTensorView with dynamic dimensions
pub type ScalarTensorViewD<'a> = ScalarTensorView<'a, IxDyn>;

/// Mutably borrowed Scalar Tensor
///
/// See [`ScalarTensorBase`].
pub type ScalarTensorViewMut<'a, D> = ScalarTensorBase<ScalarSliceMutRepr<'a>, D>;
/// ScalarTensorViewMut with 1 element
pub type ScalarTensorViewMut0<'a> = ScalarTensorViewMut<'a, Ix0>;
/// ScalarTensorViewMut with 1 dimension
pub type ScalarTensorViewMut1<'a> = ScalarTensorViewMut<'a, Ix1>;
/// ScalarTensorViewMut with 2 dimensions
pub type ScalarTensorViewMut2<'a> = ScalarTensorViewMut<'a, Ix2>;
/// ScalarTensorViewMut with 3 dimensions
pub type ScalarTensorViewMut3<'a> = ScalarTensorViewMut<'a, Ix3>;
/// ScalarTensorViewMut with 4 dimensions
pub type ScalarTensorViewMut4<'a> = ScalarTensorViewMut<'a, Ix4>;
/// ScalarTensorViewMut with 5 dimensions
pub type ScalarTensorViewMut5<'a> = ScalarTensorViewMut<'a, Ix5>;
/// ScalarTensorViewMut with 6 dimensions
pub type ScalarTensorViewMut6<'a> = ScalarTensorViewMut<'a, Ix6>;
/// ScalarTensorViewMut with dynamic dimensions
pub type ScalarTensorViewMutD<'a> = ScalarTensorViewMut<'a, IxDyn>;

/// Scalar Tensor that is either borrowed or owned.
///
/// See [`ScalarTensorBase`].
pub type ScalarCowTensor<'a, D> = ScalarTensorBase<ScalarCowBufferRepr<'a>, D>;
/// ScalarCowTensor with 1 element
pub type ScalarCowTensor0<'a> = ScalarCowTensor<'a, Ix0>;
/// ScalarCowTensor with 1 dimension
pub type ScalarCowTensor1<'a> = ScalarCowTensor<'a, Ix1>;
/// ScalarCowTensor with 2 dimensions
pub type ScalarCowTensor2<'a> = ScalarCowTensor<'a, Ix2>;
/// ScalarCowTensor with 3 dimensions
pub type ScalarCowTensor3<'a> = ScalarCowTensor<'a, Ix3>;
/// ScalarCowTensor with 4 dimensions
pub type ScalarCowTensor4<'a> = ScalarCowTensor<'a, Ix4>;
/// ScalarCowTensor with 5 dimensions
pub type ScalarCowTensor5<'a> = ScalarCowTensor<'a, Ix5>;
/// ScalarCowTensor with 6 dimensions
pub type ScalarCowTensor6<'a> = ScalarCowTensor<'a, Ix6>;
/// ScalarCowTensor with dynamic dimensions
pub type ScalarCowTensorD<'a> = ScalarCowTensor<'a, IxDyn>;

impl<S: ScalarDataOwned, D: Dimension> ScalarTensorBase<S, D> {
    /// Allocates a scalar tensor on `device` with `shape`.
    ///
    /// # Safety
    ///
    /// The tensor is not initialized.
    ///
    /// **Errors**
    /// See [`ScalarBuffer::uninit()`].
    pub unsafe fn uninit<Sh>(device: Device, shape: Sh, scalar_type: ScalarType) -> Result<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        let (dim, strides) = dim_strides_from_shape(shape.into_shape());
        let buffer = unsafe { ScalarBufferBase::uninit(device, dim.size(), scalar_type)? };
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
    /// See [`ScalarBuffer::from_elem()`].
    pub fn from_elem<Sh>(device: Device, shape: Sh, elem: ScalarElem) -> Result<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        let (dim, strides) = dim_strides_from_shape(shape.into_shape());
        let buffer = ScalarBufferBase::from_elem(device, dim.size(), elem)?;
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
    /// See [`ScalarBuffer::zeros()`].
    pub fn zeros<Sh>(device: Device, shape: Sh, scalar_type: ScalarType) -> Result<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_elem(device, shape, ScalarElem::zero(scalar_type))
    }
    /// Creates a tensor on `device` with `shape` filled with 1's.
    ///
    /// **Errors**
    /// See [`ScalarBuffer::ones()`].
    pub fn ones<Sh>(device: Device, shape: Sh, scalar_type: ScalarType) -> Result<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_elem(device, shape, ScalarElem::one(scalar_type))
    }
}

impl<S: ScalarData, D: Dimension> ScalarTensorBase<S, D> {
    /// The device of the tensor.
    pub fn device(&self) -> Device {
        self.buffer.device()
    }
    /// The scalar type of the tensor.
    pub fn scalar_type(&self) -> ScalarType {
        self.buffer.scalar_type()
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
    pub fn into_dimensionality<D2>(self) -> Result<ScalarTensorBase<S, D2>, ShapeError>
    where
        D2: Dimension,
    {
        let (dim, strides) = into_dimensionality(&self.dim, &self.strides)?;
        Ok(ScalarTensorBase {
            dim,
            strides,
            buffer: self.buffer,
            offset: self.offset,
        })
    }
    /// Converts the dimensionality of the tensor to [`IxDyn`](type@ndarray::IxDyn).
    pub fn into_dyn(self) -> ScalarTensorBase<S, IxDyn> {
        ScalarTensorBase {
            dim: self.dim.into_dyn(),
            strides: self.strides.into_dyn(),
            buffer: self.buffer,
            offset: self.offset,
        }
    }
    /// Returns the tensor with dim `shape`.
    ///
    /// **Errors**
    /// The tensor must be contiguous, with default strides.
    pub fn into_shape<E>(self, shape: E) -> Result<ScalarTensorBase<S, E::Dim>, ShapeError>
    where
        E: IntoDimension,
    {
        let (dim, strides) = into_shape(&self.dim, &self.strides, shape)?;
        assert_eq!(self.offset, 0);
        Ok(ScalarTensorBase {
            dim,
            strides,
            buffer: self.buffer,
            offset: self.offset,
        })
    }
    /// Act like a larger size and/or shape array by *broadcasting* into a larger shape, if possible.
    ///
    /// See [`TensorBase::broadcast`].
    pub fn broadcast<E>(&self, dim: E) -> Option<ScalarTensorView<E::Dim>>
    where
        E: IntoDimension,
    {
        let (dim, strides) = broadcast(&self.dim, &self.strides, dim)?;
        Some(ScalarTensorView {
            dim,
            strides,
            buffer: self.buffer.as_scalar_slice(),
            offset: self.offset,
        })
    }
    /// Borrows the tensor as a [`ScalarTensorView`].
    pub fn view(&self) -> ScalarTensorView<D> {
        ScalarTensorView {
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            buffer: self.buffer.as_scalar_slice(),
            offset: self.offset,
        }
    }
    /// Borrows the tensor as a [`ScalarTensorViewMut`].
    pub fn view_mut(&mut self) -> ScalarTensorViewMut<D>
    where
        S: ScalarDataMut,
    {
        ScalarTensorViewMut {
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            buffer: self.buffer.as_scalar_slice_mut(),
            offset: self.offset,
        }
    }
    /// Mutably borrows the tensor as a mutable view if possible.
    pub fn get_view_mut(&mut self) -> Option<ScalarTensorViewMut<D>> {
        if self.offset == 0 && self.is_contiguous() {
            let buffer = self.buffer.get_scalar_slice_mut()?;
            Some(ScalarTensorViewMut {
                dim: self.dim.clone(),
                strides: self.strides.clone(),
                buffer,
                offset: 0,
            })
        } else {
            None
        }
    }
    /// Mutably borrows the tensor as a mutable view.
    ///
    /// See [`TensorBase::make_view_mut`].
    pub fn make_view_mut(&mut self) -> Result<ScalarTensorViewMut<D>>
    where
        S: ScalarDataOwned,
    {
        if self.offset == 0 && self.is_contiguous() {
            Ok(ScalarTensorViewMut {
                dim: self.dim.clone(),
                strides: self.strides.clone(),
                buffer: self.buffer.make_scalar_slice_mut()?,
                offset: 0,
            })
        } else {
            let tensor = self.to_owned()?;
            *self = Self {
                dim: tensor.dim,
                strides: tensor.strides,
                buffer: ScalarBufferBase::from_scalar_buffer(tensor.buffer),
                offset: 0,
            };
            Ok(ScalarTensorViewMut {
                dim: self.dim.clone(),
                strides: self.strides.clone(),
                buffer: self.buffer.get_scalar_slice_mut().unwrap(),
                offset: 0,
            })
        }
    }
    /// Whether the tensor is contiguous.
    ///
    /// Contiguous is either C (Standard) or Fortran layout.
    pub fn is_contiguous(&self) -> bool {
        is_contiguous(&self.dim, &self.strides)
    }
    /// Whether the tensor is standard layout.
    ///
    /// In standard layout, the strides increase from right to left by the product of each dimension.
    pub fn is_standard_layout(&self) -> bool {
        is_standard_layout(&self.dim, &self.strides)
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
    pub fn t(&self) -> ScalarTensorView<D> {
        self.view().reversed_axes()
    }
    /// Returns a view restricted to `index` along the `axis`, with the axis removed.
    ///
    /// See [`TensorBase::index_axis`].
    pub fn index_axis(&self, axis: Axis, index: usize) -> ScalarTensorView<D::Smaller>
    where
        D: RemoveAxis,
    {
        self.view().index_axis_into(axis, index)
    }
    /// Returns a mutable view restricted to index along the `axis`, with the `axis` removed.
    ///
    /// See [`TensorBase::index_axis_mut`].
    pub fn index_axis_mut(&mut self, axis: Axis, index: usize) -> ScalarTensorViewMut<D::Smaller>
    where
        S: ScalarDataMut,
        D: RemoveAxis,
    {
        self.view_mut().index_axis_into(axis, index)
    }
    /// Returns a mutable view restricted to index along the `axis`, with the `axis` removed.
    ///
    /// See [`TensorBase::index_axis_into`].
    pub fn index_axis_into(mut self, axis: Axis, index: usize) -> ScalarTensorBase<S, D::Smaller>
    where
        D: RemoveAxis,
    {
        self.collapse_axis(axis, index);
        let dim = self.dim.remove_axis(axis);
        let strides = self.strides.remove_axis(axis);
        ScalarTensorBase {
            dim,
            strides,
            buffer: self.buffer,
            offset: self.offset,
        }
    }
    /// Selects `index` along the `axis`, collapsing the axis into length one.
    ///
    /// **Panics**
    /// - `axis` or `index` is out of bounds.
    ///
    /// See [`TensorBase::collapse_axis`].
    pub fn collapse_axis(&mut self, axis: Axis, index: usize) {
        let offset =
            collapse_axis(&mut self.dim, &self.strides, axis, index) + self.offset as isize;
        debug_assert!(offset >= 0);
        self.offset = offset as usize;
        debug_assert!(self.offset < self.buffer.len());
    }
    /// Borrows the tensor as a [`ScalarSlice`] if standard layout.
    pub fn as_scalar_slice(&self) -> Option<ScalarSlice> {
        if self.is_standard_layout() {
            Some(self.buffer.as_scalar_slice())
        } else {
            None
        }
    }
    /// Borrows the tensor as a [`ScalarSlice`] if contiguous.
    pub fn as_scalar_slice_memory_order(&self) -> Option<ScalarSlice> {
        if self.is_contiguous() {
            if self.offset > 0 {
                Some(
                    self.buffer
                        .slice(self.offset..self.offset + self.len())
                        .unwrap(),
                )
            } else {
                Some(self.buffer.as_scalar_slice())
            }
        } else {
            None
        }
    }
    /// Mutably borrows the tensor as a [`ScalarSliceMut`] if standard layout.
    pub fn as_scalar_slice_mut(&mut self) -> Option<ScalarSliceMut>
    where
        S: ScalarDataMut,
    {
        if self.is_contiguous() {
            if self.offset > 0 {
                Some(
                    self.buffer
                        .slice_mut(self.offset..self.offset + self.len())
                        .unwrap(),
                )
            } else {
                Some(self.buffer.as_scalar_slice_mut())
            }
        } else {
            None
        }
    }
    /// Mutably borrows the tensor as a [`ScalarSliceMut`] if contiguous.
    pub fn as_scalar_slice_memory_order_mut(&mut self) -> Option<ScalarSliceMut>
    where
        S: ScalarDataMut,
    {
        if self.is_contiguous() {
            Some(self.buffer.as_scalar_slice_mut())
        } else {
            None
        }
    }
    /// Borrows the tensor as a slice and offset.
    pub fn as_raw_scalar_slice_offset(&self) -> (ScalarSlice, usize) {
        (self.buffer.as_scalar_slice(), self.offset)
    }
    /// Mutably borrows the tensor as a mutable slice and offset.
    pub fn as_raw_scalar_slice_offset_mut(&mut self) -> (ScalarSliceMut, usize)
    where
        S: ScalarDataMut,
    {
        (self.buffer.as_scalar_slice_mut(), self.offset)
    }
    /// Transfers the tensor into the `device`.
    ///
    /// See [`TensorBase::into_device`].
    pub fn into_device(self, device: Device) -> Result<ScalarTensor<D>> {
        if self.device() == device {
            self.into_owned()
        } else if let Some(slice) = self.as_scalar_slice_memory_order() {
            let buffer = slice.to_device(device)?;
            Ok(ScalarTensor {
                dim: self.dim,
                strides: self.strides,
                buffer,
                offset: 0,
            })
        } else {
            self.into_owned()?.into_device(device)
        }
    }
    /// Transfers the tensor to the `device`.
    ///
    /// See [`Tensor::to_device`].
    pub fn to_device(&self, device: Device) -> Result<ScalarTensor<D>> {
        if self.device() == device {
            self.to_owned()
        } else {
            self.view().into_device(device)
        }
    }
    /// Transfers the tensor into the `device` in place.
    ///
    /// See [`Tensor::to_device_mut`].
    pub fn to_device_mut(&mut self, device: Device) -> Result<()>
    where
        S: ScalarDataOwned,
    {
        if self.device() == device {
            Ok(())
        } else {
            let ScalarTensor {
                dim,
                strides,
                buffer,
                offset,
            } = self.to_device(device)?;
            *self = Self {
                dim,
                strides,
                buffer: ScalarBufferBase::from_scalar_buffer(buffer),
                offset,
            };
            Ok(())
        }
    }
    /// Transfers the tensor into the `device` as a scalar arc tensor.
    ///
    /// See [`TensorBase::into_device_shared`].
    pub fn into_device_shared(self, device: Device) -> Result<ScalarArcTensor<D>> {
        if self.device() == device {
            self.into_shared()
        } else {
            self.to_device(device).map(Into::into)
        }
    }
    /// Transfers the tensor to the `device` as a scalar arc tensor.
    ///
    /// See [`TensorBase::to_device_shared`].
    pub fn to_device_shared(&self, device: Device) -> Result<ScalarArcTensor<D>> {
        if device == self.device() {
            self.to_shared()
        } else {
            self.to_device(device).map(Into::into)
        }
    }
    /// Converts into a [`ScalarTensor`].
    pub fn into_owned(self) -> Result<ScalarTensor<D>> {
        if self.offset == 0 && self.is_contiguous() {
            return Ok(ScalarTensorBase {
                dim: self.dim,
                strides: self.strides,
                buffer: self.buffer.into_owned()?,
                offset: 0,
            });
        }
        if let Some(slice) = self.as_scalar_slice_memory_order() {
            let buffer = slice.to_owned()?;
            return Ok(ScalarTensorBase {
                dim: self.dim,
                strides: self.strides,
                buffer,
                offset: 0,
            });
        }
        let mut output =
            unsafe { ScalarTensor::uninit(self.device(), self.raw_dim(), self.scalar_type())? };
        output.assign(&self)?;
        Ok(output)
    }
    /// Converts to a [`ScalarTensor`].
    pub fn to_owned(&self) -> Result<ScalarTensor<D>> {
        self.view().into_owned()
    }
    /// Converts into an [`ScalarArcTensor`].
    pub fn into_shared(self) -> Result<ScalarArcTensor<D>> {
        if !self.is_contiguous() {
            todo!()
        }
        Ok(ScalarTensorBase {
            dim: self.dim,
            strides: self.strides,
            buffer: self.buffer.into_shared()?,
            offset: 0,
        })
    }
    /// Converts to an [`ScalarArcTensor`].
    pub fn to_shared(&self) -> Result<ScalarArcTensor<D>> {
        if !self.is_contiguous() {
            todo!()
        }
        Ok(ScalarTensorBase {
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            buffer: self.buffer.to_shared()?,
            offset: 0,
        })
    }
}

impl<D: Dimension> ScalarTensor<D> {
    /// Attempt to convert to a tensor.
    pub fn try_into_tensor<T: Scalar>(self) -> Result<Tensor<T, D>, Self> {
        self.try_into()
    }
}

impl<D: Dimension> ScalarArcTensor<D> {
    /// Attempt to convert to an arc tensor.
    pub fn try_into_arc_tensor<T: Scalar>(self) -> Result<ArcTensor<T, D>, Self> {
        self.try_into()
    }
}

impl<'a, D: Dimension> ScalarTensorView<'a, D> {
    /// Attempt to convert to a tensor view.
    pub fn try_into_tensor_view<T: Scalar>(self) -> Result<TensorView<'a, T, D>, Self> {
        self.try_into()
    }
}

impl<'a, D: Dimension> ScalarTensorViewMut<'a, D> {
    /// Attempt to convert to a mutable tensor view.
    pub fn try_into_tensor_view_mut<T: Scalar>(self) -> Result<TensorViewMut<'a, T, D>, Self> {
        self.try_into()
    }
}

impl<D: Dimension> ScalarArcTensor<D> {
    /// Act like a larger size and/or shape array by *broadcasting* into a larger shape, if possible.
    ///
    /// See [`ArcTensor::broadcast_shared`].
    pub fn broadcast_shared<E>(&self, dim: E) -> Option<ScalarArcTensor<E::Dim>>
    where
        E: IntoDimension,
    {
        let (dim, strides) = broadcast(&self.dim, &self.strides, dim)?;
        Some(ScalarArcTensor {
            dim,
            strides,
            buffer: self.buffer.clone(),
            offset: self.offset,
        })
    }
}

impl<S: ScalarDataOwned> From<ScalarBuffer> for ScalarTensorBase<S, Ix1> {
    fn from(buffer: ScalarBuffer) -> Self {
        let dim = buffer.len().into_dimension();
        let strides = dim.default_strides();
        let buffer = ScalarBufferBase::from_scalar_buffer(buffer);
        Self {
            dim,
            strides,
            buffer,
            offset: 0,
        }
    }
}

impl<S: ScalarDataOwned, T: Scalar, D: Dimension> From<Tensor<T, D>> for ScalarTensorBase<S, D> {
    fn from(tensor: Tensor<T, D>) -> Self {
        Self {
            dim: tensor.dim,
            strides: tensor.strides,
            buffer: tensor.buffer.into(),
            offset: tensor.offset,
        }
    }
}

impl<D: Dimension> From<ScalarTensor<D>> for ScalarArcTensor<D> {
    fn from(tensor: ScalarTensor<D>) -> Self {
        Self {
            dim: tensor.dim,
            strides: tensor.strides,
            buffer: tensor.buffer.into(),
            offset: tensor.offset,
        }
    }
}

impl<T: Scalar, D: Dimension> From<ArcTensor<T, D>> for ScalarArcTensor<D> {
    fn from(tensor: ArcTensor<T, D>) -> Self {
        Self {
            dim: tensor.dim,
            strides: tensor.strides,
            buffer: tensor.buffer.into(),
            offset: tensor.offset,
        }
    }
}

impl<D: Dimension> From<ScalarTensor<D>> for ScalarCowTensor<'_, D> {
    fn from(tensor: ScalarTensor<D>) -> Self {
        Self {
            dim: tensor.dim,
            strides: tensor.strides,
            buffer: tensor.buffer.into(),
            offset: tensor.offset,
        }
    }
}

impl<'a, D: Dimension> From<ScalarTensorView<'a, D>> for ScalarCowTensor<'a, D> {
    fn from(tensor: ScalarTensorView<'a, D>) -> Self {
        Self {
            dim: tensor.dim,
            strides: tensor.strides,
            buffer: tensor.buffer.into(),
            offset: tensor.offset,
        }
    }
}

macro_for!($Tensor in [Tensor, ArcTensor] {
    paste! {
        impl<T: Scalar, D: Dimension> TryFrom<[<Scalar $Tensor>]<D>> for $Tensor<T, D> {
            type Error = [<Scalar $Tensor>]<D>;
            fn try_from(tensor: [<Scalar $Tensor>]<D>) -> Result<Self, Self::Error> {
                match tensor.buffer.try_into() {
                    Ok(buffer) => Ok(Self {
                        dim: tensor.dim,
                        strides: tensor.strides,
                        buffer,
                        offset: tensor.offset,
                    }),
                    Err(buffer) => Err(Self::Error {
                        dim: tensor.dim,
                        strides: tensor.strides,
                        buffer,
                        offset: tensor.offset,
                    })
                }
            }
        }
    }
});

macro_for!($Tensor in [TensorView, TensorViewMut, CowTensor] {
    paste! {
        impl<'a, T: Scalar, D: Dimension> From<$Tensor<'a, T, D>> for [<Scalar $Tensor>]<'a, D> {
            fn from(tensor: $Tensor<'a, T, D>) -> Self {
                Self {
                    dim: tensor.dim,
                    strides: tensor.strides,
                    buffer: tensor.buffer.into(),
                    offset: tensor.offset,
                }
            }
        }
        impl<'a, T: Scalar, D: Dimension> TryFrom<[<Scalar $Tensor>]<'a, D>> for $Tensor<'a, T, D> {
            type Error = [<Scalar $Tensor>]<'a, D>;
            fn try_from(tensor: [<Scalar $Tensor>]<'a, D>) -> Result<Self, Self::Error> {
                match tensor.buffer.try_into() {
                    Ok(buffer) => Ok(Self {
                        dim: tensor.dim,
                        strides: tensor.strides,
                        buffer,
                        offset: tensor.offset,
                    }),
                    Err(buffer) => Err(Self::Error {
                        dim: tensor.dim,
                        strides: tensor.strides,
                        buffer,
                        offset: tensor.offset,
                    })
                }
            }
        }
    }
});

impl<S: ScalarData, D: Dimension> Debug for ScalarTensorBase<S, D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut builder = f.debug_struct("TensorBase");
        builder
            .field("device", &self.device())
            .field("scalar_type", &self.scalar_type())
            .field("shape", &self.shape());
        if self.strides != self.dim.default_strides() {
            builder.field("strides", &self.strides());
        }
        if self.offset > 0 {
            builder.field("offset", &self.offset);
        }
        builder.finish()
    }
}

/// Casts
#[allow(unused)]
impl<S: ScalarData, D: Dimension> ScalarTensorBase<S, D> {
    /// Casts the tensor into a new tensor.
    ///
    /// See [`BufferBase::cast_into()`](crate::buffer::BufferBase::cast_into()).
    pub fn cast_into(self, scalar_type: ScalarType) -> Result<ScalarTensor<D>> {
        if !self.is_contiguous() {
            todo!()
        }
        Ok(ScalarTensorBase {
            dim: self.dim,
            strides: self.strides,
            buffer: self.buffer.cast_into(scalar_type)?,
            offset: 0,
        })
    }
    /// Casts the tensor to a new tensor.
    ///
    /// See [`BufferBase::cast()`](crate::buffer::BufferBase::cast()).
    pub fn cast(&self, scalar_type: ScalarType) -> Result<ScalarTensor<D>> {
        if !self.is_contiguous() {
            todo!();
        }
        Ok(ScalarTensorBase {
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            buffer: self.buffer.cast(scalar_type)?,
            offset: 0,
        })
    }
    /// Casts the tensor into a new tensor.
    ///
    /// See [`BufferBase::cast()`](crate::buffer::BufferBase::cast()).
    pub fn cast_into_tensor<T: Scalar>(self) -> Result<Tensor<T, D>> {
        Ok(self.cast_into(T::scalar_type())?.try_into().unwrap())
    }
}

/*
// Logits
impl<S: ScalarData, D: Dimension> ScalarTensorBase<S, D> {
    /// A one hot vector given class labels.
    ///
    /// Output shape = [input_shape.., `classes`].
    pub fn to_one_hot(
        self,
        classes: usize,
        scalar_type: ScalarType,
    ) -> Result<ScalarTensor<D::Larger>> {
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
        macro_for!($X in [u8, u16, u32, u64] {
            if let Ok(input) = TensorView::<$X, D>::try_from(self.view()) {
                let input = input.as_array().unwrap();

                macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                    if scalar_type == $Y::scalar_type() {
                        let mut output = Array::zeros(dim);
                        for (x, y) in input
                            .iter()
                            .zip(output.as_slice_mut().unwrap().chunks_mut(classes))
                        {
                            y[x.to_usize().unwrap()] = $Y::one();
                        }
                        return Ok(Tensor::from(output).into());
                    }
                });
            }
        });
        bail!(
            "to_one_hot {:?} {:?} unimplemented!",
            self.scalar_type(),
            scalar_type
        );
    }
}*/

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "S: ScalarData, D: Dimension + Serialize",
    deserialize = "S: ScalarDataOwned, D: Dimension + Deserialize<'de>"
))]
#[serde(rename = "Tensor")]
struct ScalarTensorSerde<S: ScalarData, D: Dimension> {
    dim: D,
    buffer: ScalarBufferBase<S>,
}

impl<S1: ScalarData, D: Dimension + Serialize> Serialize for ScalarTensorBase<S1, D> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::Error;
        let buffer = if let Some(slice) = self.as_scalar_slice() {
            ScalarCowBuffer::from(slice)
        } else {
            self.to_device(Device::host())
                .map_err(S::Error::custom)?
                .buffer
                .into()
        };
        ScalarTensorSerde {
            dim: self.dim.clone(),
            buffer,
        }
        .serialize(serializer)
    }
}

impl<'de, S: ScalarDataOwned, D1: Dimension + Deserialize<'de>> Deserialize<'de>
    for ScalarTensorBase<S, D1>
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;
        let ScalarTensorSerde { dim, buffer } =
            ScalarTensorSerde::<ScalarBufferRepr, D1>::deserialize(deserializer)?;
        ScalarTensorBase::from(buffer)
            .into_shape(dim)
            .map_err(D::Error::custom)
    }
}

/// Multi-dimensional matrix.
///
/// Use [`Into`] to convert to a [`ScalarTensorBase`].
/// Use [`TryFrom`] to convert from a [`ScalarTensorBase`].
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
        /*let buffer = if device.is_host() || dim.ndim() == 1 {
            unsafe { BufferBase::uninit(device, dim.size())? }
        } else {
            unsafe {
                BufferBase::zeros(device, dim.size())?
            }
        };*/
        let buffer = unsafe { BufferBase::uninit(device, dim.size())? };
        //buffer.device().wait()?;
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
    /// The scalar type of the tensor.
    pub fn scalar_type(&self) -> ScalarType {
        T::scalar_type()
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
    pub fn into_dimensionality<D2>(self) -> Result<TensorBase<S, D2>, ShapeError>
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
    /// Returns the tensor with dim `shape`.
    ///
    /// **Errors**
    /// The tensor must be contiguous, with default strides.
    pub fn into_shape<E>(self, shape: E) -> Result<TensorBase<S, E::Dim>, ShapeError>
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
    }
    /// Flattens the trailing dimensions into a 2 dimensional tensor.
    ///
    /// The output has shape [d0, d1 * d2 .. * dn].
    ///
    /// **Errors**
    /// See [`TensorBase::into_shape`].
    pub fn flatten(self) -> Result<TensorBase<S, Ix2>, ShapeError> {
        let dim = flatten(self.shape());
        self.into_shape(dim)
    }
    /// Act like a larger size and/or shape array by *broadcasting* into a larger shape, if possible.
    ///
    /// See <https://docs.rs/ndarray/0.15.6/ndarray/struct.ArrayBase.html#method.broadcast>
    pub fn broadcast<E>(&self, dim: E) -> Option<TensorView<T, E::Dim>>
    where
        E: IntoDimension,
    {
        let (dim, strides) = broadcast(&self.dim, &self.strides, dim)?;
        Some(TensorView {
            dim,
            strides,
            buffer: self.buffer.as_slice(),
            offset: self.offset,
        })
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
    /// Mutably borrows the tensor as a mutable view if possible.
    pub fn get_view_mut(&mut self) -> Option<TensorViewMut<T, D>> {
        if self.offset == 0 && self.is_contiguous() {
            let buffer = self.buffer.get_slice_mut()?;
            Some(TensorViewMut {
                dim: self.dim.clone(),
                strides: self.strides.clone(),
                buffer,
                offset: 0,
            })
        } else {
            None
        }
    }
    /// Mutably borrows the tensor as a mutable view.
    ///
    /// Copies the data into a new tensor if necessary.
    ///
    /// See [`TensorBase::to_owned`].
    pub fn make_view_mut(&mut self) -> Result<TensorViewMut<T, D>>
    where
        S: DataOwned,
    {
        if self.offset == 0 && self.is_contiguous() {
            Ok(TensorViewMut {
                dim: self.dim.clone(),
                strides: self.strides.clone(),
                buffer: self.buffer.make_slice_mut()?,
                offset: 0,
            })
        } else {
            let tensor = self.to_owned()?;
            *self = Self {
                dim: tensor.dim,
                strides: tensor.strides,
                buffer: BufferBase::from_buffer(tensor.buffer),
                offset: 0,
            };
            Ok(TensorViewMut {
                dim: self.dim.clone(),
                strides: self.strides.clone(),
                buffer: self.buffer.get_slice_mut().unwrap(),
                offset: 0,
            })
        }
    }
    /// Whether the tensor is contiguous.
    ///
    /// Contiguous is either C (Standard) or Fortran layout.
    pub fn is_contiguous(&self) -> bool {
        is_contiguous(&self.dim, &self.strides)
    }
    /// Whether the tensor is standard layout.
    ///
    /// In standard layout, the strides increase from right to left by the product of each dimension.
    pub fn is_standard_layout(&self) -> bool {
        is_standard_layout(&self.dim, &self.strides)
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
    /// Returns a view restricted to index along the `axis`, with the `axis` removed.
    ///
    /// **Panics**
    /// - `axis` or `index` is out of bounds.
    /// See <https://docs.rs/ndarray/0.15.6/ndarray/struct.ArrayBase.html#method.index_axis>
    pub fn index_axis(&self, axis: Axis, index: usize) -> TensorView<T, D::Smaller>
    where
        D: RemoveAxis,
    {
        self.view().index_axis_into(axis, index)
    }
    /// Returns a mutable view restricted to index along the `axis`, with the `axis` removed.
    ///
    /// **Panics**
    /// - `axis` or `index` is out of bounds.
    /// See <https://docs.rs/ndarray/0.15.6/ndarray/struct.ArrayBase.html#method.index_axis_mut>
    pub fn index_axis_mut(&mut self, axis: Axis, index: usize) -> TensorViewMut<T, D::Smaller>
    where
        S: DataMut,
        D: RemoveAxis,
    {
        self.view_mut().index_axis_into(axis, index)
    }
    /// Returns a tensor restricted to index along the `axis`, with the `axis` removed.
    ///
    /// **Panics**
    /// - `axis` or `index` is out of bounds.
    pub fn index_axis_into(mut self, axis: Axis, index: usize) -> TensorBase<S, D::Smaller>
    where
        D: RemoveAxis,
    {
        self.collapse_axis(axis, index);
        let dim = self.dim.remove_axis(axis);
        let strides = self.strides.remove_axis(axis);
        TensorBase {
            dim,
            strides,
            buffer: self.buffer,
            offset: self.offset,
        }
    }
    /// Selects `index` along the `axis`, collapsing the axis into length one.
    ///
    /// **Panics**
    /// - `axis` or `index` is out of bounds.
    ///
    /// See <https://docs.rs/ndarray/0.15.6/ndarray/struct.ArrayBase.html#method.collapse_axis>
    pub fn collapse_axis(&mut self, axis: Axis, index: usize) {
        let offset =
            collapse_axis(&mut self.dim, &self.strides, axis, index) + self.offset as isize;
        debug_assert!(offset >= 0);
        self.offset = offset as usize;
        debug_assert!(self.offset < self.buffer.len());
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
            if self.offset > 0 {
                Some(
                    self.buffer
                        .slice(self.offset..self.offset + self.len())
                        .unwrap(),
                )
            } else {
                Some(self.buffer.as_slice())
            }
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
            if self.offset > 0 {
                Some(
                    self.buffer
                        .slice_mut(self.offset..self.offset + self.len())
                        .unwrap(),
                )
            } else {
                Some(self.buffer.as_slice_mut())
            }
        } else {
            None
        }
    }
    /// Borrows the tensor as a slice and offset.
    pub fn as_raw_slice_offset(&self) -> (Slice<T>, usize) {
        (self.buffer.as_slice(), self.offset)
    }
    /// Mutably borrows the tensor as a mutable slice and offset.
    pub fn as_raw_slice_offset_mut(&mut self) -> (SliceMut<T>, usize)
    where
        S: DataMut,
    {
        (self.buffer.as_slice_mut(), self.offset)
    }
    /// Transfers the tensor to the `device`.
    ///
    /// See [`Tensor::into_device`].
    pub fn to_device(&self, device: Device) -> Result<Tensor<T, D>> {
        if self.device() == device {
            self.to_owned()
        } else {
            self.view().into_device(device)
        }
    }
    /// Transfers the tensor to the `device`.
    ///
    /// See [`Tensor::to_device`].
    pub fn to_device_shared(&self, device: Device) -> Result<ArcTensor<T, D>> {
        if self.device() == device {
            self.to_shared()
        } else {
            self.to_device(device).map(Into::into)
        }
    }
    /// Transfers the tensor to the `device` in place.
    ///
    /// See [`crate::buffer::BufferBase::to_device_mut`].
    pub fn to_device_mut(&mut self, device: Device) -> Result<()>
    where
        S: DataOwned,
    {
        if self.device() == device {
            Ok(())
        } else {
            let Tensor {
                dim,
                strides,
                buffer,
                offset,
            } = self.to_device(device)?;
            *self = Self {
                dim,
                strides,
                buffer: BufferBase::from_buffer(buffer),
                offset,
            };
            Ok(())
        }
    }
    /// Transfers the tensor into the `device`.
    ///
    /// See [`Buffer::into_device()`](crate::buffer::BufferBase::into_device()).
    pub fn into_device(self, device: Device) -> Result<Tensor<T, D>> {
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
    /// Transfers the tensor into the `device`.
    ///
    /// See [`ArcBuffer::into_device_shared()`](crate::buffer::BufferBase::into_device_shared()).
    pub fn into_device_shared(self, device: Device) -> Result<ArcTensor<T, D>> {
        if device == self.device() {
            self.into_shared()
        } else if !self.is_contiguous() {
            self.view()
                .into_standard_layout()?
                .into_device_shared(device)
        } else {
            let buffer = self.buffer.to_device_shared(device)?;
            Ok(ArcTensor {
                dim: self.dim,
                strides: self.strides,
                buffer,
                offset: 0,
            })
        }
    }
    /// Converts into a [`Tensor`].
    pub fn into_owned(self) -> Result<Tensor<T, D>> {
        if !self.is_contiguous() {
            return self.into_standard_layout();
        }
        Ok(TensorBase {
            dim: self.dim,
            strides: self.strides,
            buffer: self.buffer.into_owned()?,
            offset: 0,
        })
    }
    /// Converts to a [`Tensor`].
    pub fn to_owned(&self) -> Result<Tensor<T, D>> {
        self.view().into_owned()
    }
    /// Converts into an [`ArcTensor`].
    pub fn into_shared(self) -> Result<ArcTensor<T, D>> {
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
    pub fn to_shared(&self) -> Result<ArcTensor<T, D>> {
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
        S: DataMut,
    {
        if self.is_contiguous() {
            self.buffer.as_slice_mut().fill(elem)
        } else {
            todo!()
        }
    }
    /// Moves the tensor into an [`Array`].
    ///
    /// **Errors**
    /// See [`Buffer::into_vec()`].
    pub fn into_array(self) -> Result<Array<T, D>> {
        if !self.is_contiguous() {
            todo!()
        }
        let vec = self.buffer.into_vec()?;
        Ok(Array::from_shape_vec(self.dim.strides(self.strides), vec).unwrap())
    }
    /// Borrows the tensor as an array view if on the host.
    pub fn as_array(&self) -> Option<ArrayView<T, D>> {
        self.buffer.as_host_slice().map(|host_slice| unsafe {
            ArrayView::from_shape_ptr(
                self.dim.clone().strides(self.strides.clone()),
                &host_slice[self.offset] as *const T,
            )
        })
    }
    /// Mutably borrows the tensor as an a mutable array view if on the host.
    pub fn as_array_mut(&mut self) -> Option<ArrayViewMut<T, D>>
    where
        S: DataMut,
    {
        if let Some(host_slice) = self.buffer.as_host_slice_mut() {
            let host_slice = unsafe {
                std::slice::from_raw_parts_mut(host_slice.as_mut_ptr(), host_slice.len())
            };
            Some(unsafe {
                ArrayViewMut::from_shape_ptr(
                    self.dim.clone().strides(self.strides.clone()),
                    &mut host_slice[self.offset] as *mut T,
                )
            })
        } else {
            None
        }
    }
}

impl<T: Scalar, D: Dimension> Tensor<T, D> {
    /// Converts to a scalar tensor.
    pub fn into_scalar_tensor(self) -> ScalarTensor<D> {
        self.into()
    }
}

impl<'a, T: Scalar, D: Dimension> CowTensor<'a, T, D> {
    /// Converts to a scalar cow tensor.
    pub fn into_scalar_cow_tensor(self) -> ScalarCowTensor<'a, D> {
        self.into()
    }
}

impl<T: Scalar, D: Dimension> ArcTensor<T, D> {
    /// Act like a larger size and/or shape array by *broadcasting* into a larger shape, if possible.
    ///
    /// See [`TensorBase::broadcast`].
    pub fn broadcast_shared<E>(&self, dim: E) -> Option<ArcTensor<T, E::Dim>>
    where
        E: IntoDimension,
    {
        let (dim, strides) = broadcast(&self.dim, &self.strides, dim)?;
        Some(ArcTensor {
            dim,
            strides,
            buffer: self.buffer.clone(),
            offset: self.offset,
        })
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

impl<T: Scalar, S: DataOwned<Elem = T>> From<Vec<T>> for TensorBase<S, Ix1> {
    fn from(vec: Vec<T>) -> Self {
        let dim = vec.len().into_dimension();
        let strides = dim.default_strides();
        let buffer = BufferBase::from_buffer(Buffer::from(vec));
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

impl<'a, T: Scalar> From<SliceMut<'a, T>> for TensorViewMut<'a, T, Ix1> {
    fn from(slice: SliceMut<'a, T>) -> Self {
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
    /// **Errors**
    /// - The `array` is not contiguous.
    fn try_from(array: ArrayView<'a, T, D>) -> Result<Self> {
        let slice = array
            .as_slice_memory_order()
            .ok_or_else(|| anyhow!("Not contiguous!"))?;
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

impl<S: Data, D: Dimension> Debug for TensorBase<S, D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        ScalarTensorView::from(self.view()).fmt(f)
    }
}

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
    /*
    /// Scales the tensor into a new tensor.
    pub fn scale_into<Y: Scalar>(self, alpha: Y) -> Result<Tensor<Y, D>> {
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
    }*/
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "S: Data, D: Dimension + Serialize",
    deserialize = "S: DataOwned, D: Dimension + Deserialize<'de>"
))]
#[serde(rename = "Tensor")]
struct TensorSerde<S: Data, D: Dimension> {
    dim: D,
    buffer: BufferBase<S>,
}

impl<S1: Data, D: Dimension + Serialize> Serialize for TensorBase<S1, D> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::Error;
        let buffer = if let Some(slice) = self.as_slice() {
            CowBuffer::from(slice)
        } else {
            self.to_device(Device::host())
                .map_err(S::Error::custom)?
                .buffer
                .into()
        };
        TensorSerde {
            dim: self.dim.clone(),
            buffer,
        }
        .serialize(serializer)
    }
}

impl<'de, S: DataOwned, D1: Dimension + Deserialize<'de>> Deserialize<'de> for TensorBase<S, D1> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;
        let TensorSerde { dim, buffer } =
            TensorSerde::<BufferRepr<S::Elem>, D1>::deserialize(deserializer)?;
        TensorBase::from(buffer)
            .into_shape(dim)
            .map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_test::{assert_tokens, Token};

    #[test]
    fn tensor_serde() {
        let data = vec![1u32, 2, 3, 4];
        let items: Vec<u64> = bytemuck::cast_slice(data.as_slice()).to_vec();
        let tensor = Tensor::from(Buffer::from(data));
        let tokens = [
            Token::Struct {
                name: "Tensor",
                len: 2,
            },
            Token::Str("dim"),
            Token::Tuple { len: 1 },
            Token::U64(4),
            Token::TupleEnd,
            Token::Str("buffer"),
            Token::TupleStruct {
                name: "Buffer",
                len: 3,
            },
            Token::Str("U32"),
            Token::U64(4),
            Token::Seq { len: Some(2) },
            Token::U64(items[0].to_be()),
            Token::U64(items[1].to_be()),
            Token::SeqEnd,
            Token::TupleStructEnd,
            Token::StructEnd,
        ];

        #[derive(Debug, Serialize, Deserialize)]
        #[serde(transparent)]
        struct TensorWrap(Tensor1<u32>);

        impl PartialEq for TensorWrap {
            fn eq(&self, other: &Self) -> bool {
                self.0.as_array().unwrap() == other.0.as_array().unwrap()
            }
        }

        impl Eq for TensorWrap {}

        assert_tokens(&TensorWrap(tensor), &tokens);
    }
}
