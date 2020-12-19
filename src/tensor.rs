use crate::backend::{Buffer, BufferSlice, BufferSliceMut, Device};
use crate::error::ShapeError;
use crate::Result;
use bytemuck::Pod;
use ndarray::{Array, ArrayBase, CowArray, RawArrayView};
pub use ndarray::{
    Dimension, IntoDimension, Ix, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, ShapeBuilder, StrideShape,
};
use num_traits::One;
use smol::future::Future;
use std::borrow::Cow;
use std::fmt::{self, Debug};
use std::sync::Arc;

pub mod linalg;

mod sealed {
    pub trait Sealed {}
}

pub trait Data: sealed::Sealed + Sized {
    type Elem;
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

impl<T> sealed::Sealed for OwnedRepr<T> {}

impl<T> Data for OwnedRepr<T> {
    type Elem = T;
    fn into_buffer(self) -> Result<Buffer<T>> {
        Ok(self.0)
    }
    fn as_buffer_slice(&self) -> BufferSlice<T> {
        self.0.as_buffer_slice()
    }
}

impl<T> DataMut for OwnedRepr<T> {
    fn as_buffer_slice_mut(&mut self) -> BufferSliceMut<T> {
        self.0.as_buffer_slice_mut()
    }
}

impl<T> DataOwned for OwnedRepr<T> {
    fn from_buffer(buffer: Buffer<T>) -> Self {
        Self(buffer)
    }
}

pub struct ArcRepr<T>(Arc<Buffer<T>>);

impl<T> sealed::Sealed for ArcRepr<T> {}

impl<T> Data for ArcRepr<T> {
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

pub struct ViewRepr<'a, T>(BufferSlice<'a, T>);

impl<T> sealed::Sealed for ViewRepr<'_, T> {}

impl<T> Data for ViewRepr<'_, T> {
    type Elem = T;
    fn into_buffer(self) -> Result<Buffer<T>> {
        self.0.to_buffer()
    }
    fn as_buffer_slice(&self) -> BufferSlice<T> {
        self.0.as_buffer_slice()
    }
}

pub struct ViewMutRepr<'a, T>(BufferSliceMut<'a, T>);

impl<T> sealed::Sealed for ViewMutRepr<'_, T> {}

impl<T> Data for ViewMutRepr<'_, T> {
    type Elem = T;
    fn into_buffer(self) -> Result<Buffer<T>> {
        self.0.to_buffer()
    }
    fn as_buffer_slice(&self) -> BufferSlice<T> {
        self.0.as_buffer_slice()
    }
}

impl<T> DataMut for ViewMutRepr<'_, T> {
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
pub type Tensor1<T> = Tensor<T, Ix1>;
pub type Tensor2<T> = Tensor<T, Ix2>;

pub type ArcTensor<T, D> = TensorBase<ArcRepr<T>, D>;

pub type TensorView<'a, T, D> = TensorBase<ViewRepr<'a, T>, D>;
pub type TensorView1<'a, T> = TensorView<'a, T, Ix1>;
pub type TensorView2<'a, T> = TensorView<'a, T, Ix2>;

pub type TensorViewMut<'a, T, D> = TensorBase<ViewMutRepr<'a, T>, D>;
pub type TensorViewMut2<'a, T> = TensorViewMut<'a, T, Ix2>;

impl<S: Data, D: Dimension> TensorBase<S, D> {
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn dim(&self) -> D::Pattern {
        self.dim.clone().into_pattern()
    }
    pub fn raw_dim(&self) -> D {
        self.dim.clone()
    }
    pub fn strides(&self) -> &[isize] {
        bytemuck::cast_slice(self.strides.slice())
    }
}

impl<T, S: DataOwned<Elem = T>, D: Dimension> TensorBase<S, D> {
    pub fn from_shape_cow<'a, Sh>(
        device: &Device,
        shape: Sh,
        cow: impl Into<Cow<'a, [T]>>,
    ) -> Result<Self>
    where
        T: Pod,
        Sh: Into<StrideShape<D>>,
    {
        let (dim, strides) = dim_strides_from_shape(shape);
        let cow = cow.into();
        let len = cow.len();
        if dim.size() == len {
            let data = S::from_buffer(Buffer::from_cow(device, cow)?);
            Ok(Self {
                device: device.clone(),
                dim,
                strides,
                data,
            })
        } else {
            Err(ShapeError::IncompatibleShape.into())
        }
    }
    /// Fills the Tensor with elem on creation\
    ///
    /// T must be 32 or 64 bits
    pub fn from_elem<Sh>(device: &Device, shape: Sh, elem: T) -> Result<Self>
    where
        T: Pod,
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
        let (dim, strides) = dim_strides_from_shape(shape.into_shape());
        let data = S::from_buffer(Buffer::zeros(device, dim.size())?);
        Ok(Self {
            device: device.clone(),
            dim,
            strides,
            data,
        })
    }
    pub fn ones<Sh>(device: &Device, shape: Sh) -> Result<Self>
    where
        T: Pod + One,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_elem(device, shape, T::one())
    }
    pub fn from_array<'a>(device: &Device, array: impl Into<CowArray<'a, T, D>>) -> Result<Self>
    where
        T: Pod,
    {
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

impl<T, S: Data<Elem = T>, D: Dimension> TensorBase<S, D> {
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
    pub fn as_buffer_slice(&self) -> BufferSlice<T> {
        self.data.as_buffer_slice()
    }
    pub fn to_vec(&self) -> Result<impl Future<Output = Result<Vec<T>>> + '_>
    where
        T: Pod,
    {
        // TODO: Convert to contiguous layout here instead of erroring
        if self.strides != self.dim.default_strides() {
            return Err(ShapeError::IncompatibleLayout.into());
        }
        self.data.as_buffer_slice().to_vec()
    }
    pub fn to_array(&self) -> Result<impl Future<Output = Result<Array<T, D>>> + '_>
    where
        T: Pod,
    {
        // TODO: Convert to contiguous layout here instead of erroring
        if self.strides().iter().any(|s| *s <= 0) {
            return Err(ShapeError::IncompatibleLayout.into());
        }
        let vec_future = self.data.as_buffer_slice().to_vec()?;
        let dim = self.dim.clone();
        let strides = self.strides.clone();
        Ok(async move {
            let vec = vec_future.await?;
            Ok(unsafe { Array::from_shape_vec_unchecked(dim.strides(strides), vec) })
        })
    }
}

impl<T, S: DataMut<Elem = T>, D: Dimension> TensorBase<S, D> {
    pub fn view_mut(&mut self) -> TensorViewMut<T, D> {
        TensorBase {
            device: self.device.clone(),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            data: ViewMutRepr(self.data.as_buffer_slice_mut()),
        }
    }
    pub fn as_buffer_slice_mut(&mut self) -> BufferSliceMut<T> {
        self.data.as_buffer_slice_mut()
    }
    pub fn fill(&mut self, x: T) -> Result<()>
    where
        T: Pod,
    {
        self.data.as_buffer_slice_mut().fill(x)
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

impl<T: linalg::Scalar, S1: Data<Elem = T>, S2: Data<Elem = T>> Dot<TensorBase<S2, Ix2>>
    for TensorBase<S1, Ix2>
{
    type Output = Tensor2<T>;
    fn dot(&self, rhs: &TensorBase<S2, Ix2>) -> Result<Tensor2<T>> {
        let (m, k) = self.dim();
        let (k2, n) = rhs.dim();
        if k != k2 {
            return Err(ShapeError::IncompatibleShape.into());
        }
        let mut output = Tensor::zeros(self.device(), [m, n])?;
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
