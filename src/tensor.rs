use crate::backend::{Buffer, BufferSlice, BufferSliceMut, Device};
pub use crate::backend::{Float, Num, Scalar};
use crate::Result;
use anyhow::{anyhow, ensure};
use ndarray::{Array, ArrayBase, CowArray, RawArrayView};
pub use ndarray::{
    Axis, Dimension, IntoDimension, Ix, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, RemoveAxis,
    ShapeBuilder, StrideShape,
};
use smol::future::Future;
use std::borrow::Cow;
use std::fmt::{self, Debug};
use std::sync::Arc;

mod binary;
mod index_select;
pub mod linalg;
mod reduce;

mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

pub trait Data: Sealed + Sized {
    type Elem: Scalar;
    #[doc(hidden)]
    fn into_buffer(self) -> Result<Buffer<Self::Elem>>;
    #[doc(hidden)]
    fn into_arc_buffer(self) -> Result<Arc<Buffer<Self::Elem>>> {
        Ok(Arc::new(self.into_buffer()?))
    }
    #[doc(hidden)]
    fn as_buffer_slice(&self) -> BufferSlice<Self::Elem>;
}

pub trait DataOwned: Data {
    #[doc(hidden)]
    fn from_buffer(buffer: Buffer<Self::Elem>) -> Self;
}

pub trait DataMut: Data {
    #[doc(hidden)]
    fn as_buffer_slice_mut(&mut self) -> BufferSliceMut<Self::Elem>;
}

pub struct OwnedRepr<T>(Buffer<T>);

impl<T> Sealed for OwnedRepr<T> {}

impl<T: Scalar> Data for OwnedRepr<T> {
    type Elem = T;
    fn into_buffer(self) -> Result<Buffer<T>> {
        Ok(self.0)
    }
    fn as_buffer_slice(&self) -> BufferSlice<T> {
        self.0.as_buffer_slice()
    }
}

impl<T: Scalar> DataMut for OwnedRepr<T> {
    fn as_buffer_slice_mut(&mut self) -> BufferSliceMut<T> {
        self.0.as_buffer_slice_mut()
    }
}

impl<T: Scalar> DataOwned for OwnedRepr<T> {
    fn from_buffer(buffer: Buffer<T>) -> Self {
        Self(buffer)
    }
}

pub struct ArcRepr<T>(Arc<Buffer<T>>);

impl<T> Sealed for ArcRepr<T> {}

impl<T: Scalar> Data for ArcRepr<T> {
    type Elem = T;
    fn into_buffer(self) -> Result<Buffer<T>> {
        match Arc::try_unwrap(self.0) {
            Ok(buffer) => Ok(buffer),
            Err(arc_buffer) => Ok(arc_buffer.to_buffer()?),
        }
    }
    fn into_arc_buffer(self) -> Result<Arc<Buffer<T>>> {
        Ok(self.0)
    }
    fn as_buffer_slice(&self) -> BufferSlice<T> {
        self.0.as_buffer_slice()
    }
}

impl<T: Scalar> DataOwned for ArcRepr<T> {
    fn from_buffer(buffer: Buffer<T>) -> Self {
        Self(Arc::new(buffer))
    }
}

pub struct ViewRepr<'a, T>(BufferSlice<'a, T>);

impl<T> Sealed for ViewRepr<'_, T> {}

impl<T: Scalar> Data for ViewRepr<'_, T> {
    type Elem = T;
    fn into_buffer(self) -> Result<Buffer<T>> {
        self.0.to_buffer()
    }
    fn as_buffer_slice(&self) -> BufferSlice<T> {
        self.0.as_buffer_slice()
    }
}

pub struct ViewMutRepr<'a, T>(BufferSliceMut<'a, T>);

impl<T> Sealed for ViewMutRepr<'_, T> {}

impl<T: Scalar> Data for ViewMutRepr<'_, T> {
    type Elem = T;
    fn into_buffer(self) -> Result<Buffer<T>> {
        self.0.to_buffer()
    }
    fn as_buffer_slice(&self) -> BufferSlice<T> {
        self.0.as_buffer_slice()
    }
}

impl<T: Scalar> DataMut for ViewMutRepr<'_, T> {
    fn as_buffer_slice_mut(&mut self) -> BufferSliceMut<T> {
        self.0.as_buffer_slice_mut()
    }
}

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

pub struct TensorBase<S: Data, D: Dimension> {
    device: Device,
    dim: D,
    strides: D,
    data: S,
}

pub type Tensor<T, D> = TensorBase<OwnedRepr<T>, D>;
pub type Tensor0<T> = Tensor<T, Ix0>;
pub type Tensor1<T> = Tensor<T, Ix1>;
pub type Tensor2<T> = Tensor<T, Ix2>;
pub type Tensor3<T> = Tensor<T, Ix3>;
pub type Tensor4<T> = Tensor<T, Ix4>;
pub type Tensor5<T> = Tensor<T, Ix5>;
pub type Tensor6<T> = Tensor<T, Ix6>;
pub type TensorD<T> = Tensor<T, IxDyn>;

pub type ArcTensor<T, D> = TensorBase<ArcRepr<T>, D>;
pub type ArcTensor0<T> = ArcTensor<T, Ix0>;
pub type ArcTensor1<T> = ArcTensor<T, Ix1>;
pub type ArcTensor2<T> = ArcTensor<T, Ix2>;
pub type ArcTensor3<T> = ArcTensor<T, Ix3>;
pub type ArcTensor4<T> = ArcTensor<T, Ix4>;
pub type ArcTensor5<T> = ArcTensor<T, Ix5>;
pub type ArcTensor6<T> = ArcTensor<T, Ix6>;
pub type ArcTensorD<T> = ArcTensor<T, IxDyn>;

pub type TensorView<'a, T, D> = TensorBase<ViewRepr<'a, T>, D>;
pub type TensorView0<'a, T> = TensorView<'a, T, Ix0>;
pub type TensorView1<'a, T> = TensorView<'a, T, Ix1>;
pub type TensorView2<'a, T> = TensorView<'a, T, Ix2>;
pub type TensorView3<'a, T> = TensorView<'a, T, Ix3>;
pub type TensorView4<'a, T> = TensorView<'a, T, Ix4>;
pub type TensorView5<'a, T> = TensorView<'a, T, Ix5>;
pub type TensorView6<'a, T> = TensorView<'a, T, Ix6>;
pub type TensorViewD<'a, T> = TensorView<'a, T, IxDyn>;

pub type TensorViewMut<'a, T, D> = TensorBase<ViewMutRepr<'a, T>, D>;
pub type TensorViewMut0<'a, T> = TensorViewMut<'a, T, Ix0>;
pub type TensorViewMut1<'a, T> = TensorViewMut<'a, T, Ix1>;
pub type TensorViewMut2<'a, T> = TensorViewMut<'a, T, Ix2>;
pub type TensorViewMut3<'a, T> = TensorViewMut<'a, T, Ix3>;
pub type TensorViewMut4<'a, T> = TensorViewMut<'a, T, Ix4>;
pub type TensorViewMut5<'a, T> = TensorViewMut<'a, T, Ix5>;
pub type TensorViewMut6<'a, T> = TensorViewMut<'a, T, Ix6>;
pub type TensorViewMutD<'a, T> = TensorViewMut<'a, T, IxDyn>;

impl<T: Scalar, S: Data<Elem = T>, D: Dimension> TensorBase<S, D> {
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn dim(&self) -> D::Pattern {
        self.dim.clone().into_pattern()
    }
    pub fn raw_dim(&self) -> D {
        self.dim.clone()
    }
    pub fn shape(&self) -> &[usize] {
        self.dim.slice()
    }
    pub fn strides(&self) -> &[isize] {
        bytemuck::cast_slice(self.strides.slice())
    }
    pub fn len(&self) -> usize {
        self.dim.size()
    }
    pub fn is_empty(&self) -> bool {
        self.shape().iter().any(|x| *x == 0)
    }
    pub fn ndim(&self) -> usize {
        self.dim.ndim()
    }
    pub fn into_tensor(self) -> Result<Tensor<T, D>> {
        Ok(TensorBase {
            device: self.device,
            dim: self.dim,
            strides: self.strides,
            data: OwnedRepr(self.data.into_buffer()?),
        })
    }
    pub fn into_arc_tensor(self) -> Result<ArcTensor<T, D>> {
        Ok(TensorBase {
            device: self.device,
            dim: self.dim,
            strides: self.strides,
            data: ArcRepr(self.data.into_arc_buffer()?),
        })
    }
    pub fn into_dimensionality<D2>(self) -> Result<TensorBase<S, D2>>
    where
        D2: Dimension,
    {
        if let Some(dim) = D2::from_dimension(&self.dim) {
            if let Some(strides) = D2::from_dimension(&self.strides) {
                return Ok(TensorBase {
                    device: self.device,
                    dim,
                    strides,
                    data: self.data,
                });
            }
        }
        Err(anyhow!("Incompatible Shapes!"))
    }
}

impl<T: Scalar, S: DataOwned<Elem = T>, D: Dimension> TensorBase<S, D> {
    pub fn from_shape_cow<'a, Sh>(
        device: &Device,
        shape: Sh,
        cow: impl Into<Cow<'a, [T]>>,
    ) -> Result<Self>
    where
        Sh: Into<StrideShape<D>>,
    {
        let (dim, strides) = dim_strides_from_shape(shape);
        let cow = cow.into();
        let len = cow.len();
        ensure!(dim.size() == len);
        let data = S::from_buffer(Buffer::from_cow(device, cow)?);
        Ok(Self {
            device: device.clone(),
            dim,
            strides,
            data,
        })
    }
    /// Creates a new Tensor with the given shape\
    ///
    /// # Safety
    ///
    /// The Tensor is uninitialized.
    pub unsafe fn uninitialized<Sh>(device: &Device, shape: Sh) -> Result<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        let (dim, strides) = dim_strides_from_shape(shape.into_shape());
        let data = S::from_buffer(Buffer::uninitialized(device, dim.size())?);
        Ok(Self {
            device: device.clone(),
            dim,
            strides,
            data,
        })
    }
    pub fn from_elem<Sh>(device: &Device, shape: Sh, elem: T) -> Result<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        let (dim, strides) = dim_strides_from_shape(shape.into_shape());
        let data = S::from_buffer(Buffer::from_elem(device, elem, dim.size())?);
        Ok(Self {
            device: device.clone(),
            dim,
            strides,
            data,
        })
    }
    pub fn zeros<Sh>(device: &Device, shape: Sh) -> Result<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_elem(device, shape, T::zero())
    }
    pub fn ones<Sh>(device: &Device, shape: Sh) -> Result<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_elem(device, shape, T::one())
    }
    pub fn from_array<'a>(device: &Device, array: impl Into<CowArray<'a, T, D>>) -> Result<Self> {
        let array = array.into();
        let dim = array.raw_dim();
        let strides = strides_from_array(&array);
        let buffer = if let Some(slice) = array.as_slice_memory_order() {
            Buffer::from_cow(device, slice.into())?
        } else {
            Buffer::from_cow(
                device,
                array.as_standard_layout().as_slice().unwrap().into(),
            )?
        };
        let data = S::from_buffer(buffer);
        Ok(Self {
            device: device.clone(),
            dim,
            strides,
            data,
        })
    }
}

impl<T: Scalar, S: Data<Elem = T>, D: Dimension> TensorBase<S, D> {
    pub fn view(&self) -> TensorView<T, D> {
        TensorBase {
            device: self.device.clone(),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            data: ViewRepr(self.data.as_buffer_slice()),
        }
    }
    pub fn reversed_axes(mut self) -> Self {
        self.dim.slice_mut().reverse();
        self.strides.slice_mut().reverse();
        self
    }
    pub fn t(&self) -> TensorView<T, D> {
        self.view().reversed_axes()
    }
    /// Borrows the tensor as a BufferSlice if standard layout\
    ///
    /// Some: If the data is default strided, ie standard layout (C or RowMajor)
    pub fn as_buffer_slice(&self) -> Option<BufferSlice<T>> {
        if self.strides == self.dim.default_strides() {
            Some(self.data.as_buffer_slice())
        } else {
            None
        }
    }
    /// Borrows the tensor as a BufferSlice
    pub fn as_unordered_buffer_slice(&self) -> BufferSlice<T> {
        self.data.as_buffer_slice()
    }
    /// Copies self into a new Tensor
    pub fn to_tensor(&self) -> Result<Tensor<T, D>> {
        Ok(TensorBase {
            device: self.device.clone(),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            data: OwnedRepr(self.data.as_buffer_slice().to_buffer()?),
        })
    }
    /// Copies self into a new Tensor
    pub fn to_arc_tensor(&self) -> Result<ArcTensor<T, D>> {
        Ok(TensorBase {
            device: self.device.clone(),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            data: ArcRepr::from_buffer(self.data.as_buffer_slice().to_buffer()?),
        })
    }
    pub fn to_vec(&self) -> Result<impl Future<Output = Result<Vec<T>>> + '_> {
        // TODO: Convert to contiguous layout here instead of erroring
        ensure!(self.strides == self.dim.default_strides());
        self.data.as_buffer_slice().to_vec()
    }
    pub fn to_array(&self) -> Result<impl Future<Output = Result<Array<T, D>>> + '_> {
        // TODO: Convert to contiguous layout here instead of erroring
        ensure!(self.strides().iter().all(|s| *s > 0));
        let vec_future = self.data.as_buffer_slice().to_vec()?;
        let dim = self.dim.clone();
        let strides = self.strides.clone();
        Ok(async move {
            let vec = vec_future.await?;
            Ok(unsafe { Array::from_shape_vec_unchecked(dim.strides(strides), vec) })
        })
    }
}

impl<T: Scalar, S: DataMut<Elem = T>, D: Dimension> TensorBase<S, D> {
    pub fn view_mut(&mut self) -> TensorViewMut<T, D> {
        TensorBase {
            device: self.device.clone(),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            data: ViewMutRepr(self.data.as_buffer_slice_mut()),
        }
    }
    pub fn as_buffer_slice_mut(&mut self) -> Option<BufferSliceMut<T>> {
        if self.strides == self.dim.default_strides() {
            Some(self.data.as_buffer_slice_mut())
        } else {
            None
        }
    }
    /// Borrows the tensor mutably as a BufferSliceMut
    pub fn as_unordered_buffer_slice_mut(&mut self) -> BufferSliceMut<T> {
        self.data.as_buffer_slice_mut()
    }
    pub fn fill(&mut self, x: T) -> Result<()>
    where
        T: Scalar,
    {
        self.data.as_buffer_slice_mut().fill(x)
    }
}

impl<T: Scalar, D: Dimension> From<Tensor<T, D>> for ArcTensor<T, D> {
    fn from(tensor: Tensor<T, D>) -> ArcTensor<T, D> {
        TensorBase {
            device: tensor.device,
            dim: tensor.dim,
            strides: tensor.strides,
            data: ArcRepr::from_buffer(tensor.data.0),
        }
    }
}

impl<S: Data, D: Dimension> Debug for TensorBase<S, D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut builder = f.debug_struct("TensorBase");
        builder
            .field("device", &self.device)
            .field("dim", &self.dim);
        if self.strides != self.dim.default_strides() {
            builder.field("strides", &self.strides);
        }
        builder.finish()
    }
}

pub trait Dot<R> {
    type Output;
    fn dot(&self, rhs: &R) -> Result<Self::Output>;
}

impl<T: Num, S1: Data<Elem = T>, S2: Data<Elem = T>> Dot<TensorBase<S2, Ix2>>
    for TensorBase<S1, Ix2>
{
    type Output = Tensor2<T>;
    fn dot(&self, rhs: &TensorBase<S2, Ix2>) -> Result<Tensor2<T>> {
        let (m, k) = self.dim();
        let (k2, n) = rhs.dim();
        ensure!(k == k2);
        let mut output = unsafe { Tensor::uninitialized(self.device(), [m, n])? };
        linalg::gemm(
            T::one(),
            &self.view(),
            &rhs.view(),
            T::zero(),
            &mut output.view_mut(),
        )?;
        Ok(output)
    }
}

impl<S: Data, D: Dimension> TensorBase<S, D> {
    pub fn dot<R>(&self, rhs: &R) -> Result<<Self as Dot<R>>::Output>
    where
        Self: Dot<R>,
    {
        Dot::dot(self, rhs)
    }
}
