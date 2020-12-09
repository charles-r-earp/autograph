use crate::backend::{Buffer, BufferSlice, BufferSliceMut, Device};
use crate::error::ShapeError;
use crate::Result;
use bytemuck::Pod;
use ndarray::{Array, ArrayBase, CowArray, RawArrayView};
pub use ndarray::{
    Dimension, IntoDimension, Ix, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, ShapeBuilder, StrideShape,
};
use smol::future::Future;
use std::borrow::Cow;
use std::fmt::{self, Debug};

pub trait Data {
    type Elem;
    /*#[doc(hidden)]
    fn into_buffer(self) -> Result<Buffer<Self::Elem>>;
    #[doc(hidden)]
    fn into_arc_buffer(self) -> Result<Arc<Buffer<Self::Elem>>>;*/
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

impl<T> Data for OwnedRepr<T> {
    type Elem = T;
    fn as_buffer_slice(&self) -> BufferSlice<T> {
        self.0.as_buffer_slice()
    }
}

impl<T> DataOwned for OwnedRepr<T> {
    fn from_buffer(buffer: Buffer<T>) -> Self {
        Self(buffer)
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

impl<S: Data, D: Dimension> TensorBase<S, D> {
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
