use crate::backend::{AsSlice, AsSliceMut, Buffer, Device, Slice, SliceMut};
use crate::Result;
use async_std::future::Future;
use bytemuck::Pod;
use ndarray::{Array, CowArray, ShapeBuilder};
#[allow(unused)]
pub use ndarray::{Dimension, IntoDimension, Ix, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
use std::sync::Arc;

mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

pub trait Data {
    type Elem;
}

pub trait DataOwned: Data + Sized {
    #[doc(hidden)]
    fn from_buffer(buffer: Buffer<Self::Elem>) -> Self;
}

pub trait DataRef: Data {
    #[doc(hidden)]
    fn as_slice(&self) -> Slice<Self::Elem>;
}

pub trait DataMut: DataRef {
    #[doc(hidden)]
    fn as_slice_mut(&mut self) -> SliceMut<Self::Elem>;
}

pub struct OwnedRepr<T>(Buffer<T>);

impl<T> Sealed for OwnedRepr<T> {}

impl<T> Data for OwnedRepr<T> {
    type Elem = T;
}

impl<T> DataOwned for OwnedRepr<T> {
    fn from_buffer(buffer: Buffer<T>) -> Self {
        Self(buffer)
    }
}

impl<T> DataRef for OwnedRepr<T> {
    fn as_slice(&self) -> Slice<T> {
        self.0.as_slice()
    }
}

impl<T> DataMut for OwnedRepr<T> {
    fn as_slice_mut(&mut self) -> SliceMut<T> {
        self.0.as_slice_mut()
    }
}

#[derive(Clone)]
pub struct ArcRepr<T>(Arc<Buffer<T>>);

impl<T> Sealed for ArcRepr<T> {}

impl<T> Data for ArcRepr<T> {
    type Elem = T;
}

impl<T> DataOwned for ArcRepr<T> {
    fn from_buffer(buffer: Buffer<T>) -> Self {
        Self(Arc::new(buffer))
    }
}

impl<T> DataRef for ArcRepr<T> {
    fn as_slice(&self) -> Slice<T> {
        self.0.as_slice()
    }
}

pub struct ViewRepr<S>(S);

impl<S: AsSlice> Sealed for ViewRepr<S> {}

impl<T, S: AsSlice<Elem = T>> Data for ViewRepr<S> {
    type Elem = T;
}

impl<T, S: AsSlice<Elem = T>> DataRef for ViewRepr<S> {
    fn as_slice(&self) -> Slice<T> {
        self.0.as_slice()
    }
}

impl<T, S: AsSliceMut<Elem = T>> DataMut for ViewRepr<S> {
    fn as_slice_mut(&mut self) -> SliceMut<T> {
        self.0.as_slice_mut()
    }
}

pub struct TensorBase<S: Data, D: Dimension> {
    #[allow(unused)]
    device: Device,
    dim: D,
    strides: D,
    data: S,
}

pub type Tensor<T, D> = TensorBase<OwnedRepr<T>, D>;
pub type ArcTensor<T, D> = TensorBase<ArcRepr<T>, D>;
pub type TensorView<'a, T, D> = TensorBase<ViewRepr<Slice<'a, T>>, D>;
pub type TensorViewMut<'a, T, D> = TensorBase<ViewRepr<SliceMut<'a, T>>, D>;

impl<T, S: Data<Elem = T>, D: Dimension> TensorBase<S, D> {
    pub fn strides(&self) -> &[isize] {
        bytemuck::cast_slice(self.strides.slice())
    }
}

impl<T, S: DataOwned<Elem = T>, D: Dimension> TensorBase<S, D> {
    pub fn from_array<'a>(device: &Device, array: impl Into<CowArray<'a, T, D>>) -> Result<Self>
    where
        T: Pod + 'a,
        D: 'a,
    {
        let device = device.clone();
        let array = array.into();
        let dim = array.raw_dim();
        let (strides, buffer) = if let Some(slice) = array.as_slice_memory_order() {
            let mut strides = D::zeros(dim.ndim());
            for (i, s) in bytemuck::cast_slice(array.strides())
                .iter()
                .copied()
                .enumerate()
            {
                strides[i] = s;
            }
            let buffer = Buffer::from_vec(&device, slice)?;
            (strides, buffer)
        } else {
            let array = array.as_standard_layout();
            let strides = dim.default_strides();
            let buffer = Buffer::from_vec(&device, array.as_slice().unwrap())?;
            (strides, buffer)
        };

        let data = S::from_buffer(buffer);
        Ok(Self {
            device,
            dim,
            strides,
            data,
        })
    }
}

impl<T, S: DataRef<Elem = T>, D: Dimension> TensorBase<S, D> {
    pub fn as_slice(&self) -> Option<Slice<T>> {
        if self.strides == self.dim.default_strides() {
            Some(self.data.as_slice())
        } else {
            None
        }
    }
    pub fn as_slice_memory_order(&self) -> Option<Slice<T>> {
        if D::is_contiguous(&self.dim, &self.strides) {
            Some(self.data.as_slice())
        } else {
            None
        }
    }
    pub fn to_array(&self) -> Result<impl Future<Output = Result<Array<T, D>>>>
    where
        T: Pod + Sync,
    {
        let vec_future = self.data.as_slice().to_vec()?;
        let dim = self.dim.clone();
        let mut strides = D::zeros(dim.ndim());
        for (i, s) in bytemuck::cast_slice(self.strides())
            .iter()
            .copied()
            .enumerate()
        {
            strides[i] = s;
        }
        Ok(async move {
            let vec = vec_future.await?;
            let array = Array::from_shape_vec(dim.strides(strides), vec)?;
            Ok(array)
        })
    }
}

impl<T, S: DataMut<Elem = T>, D: Dimension> TensorBase<S, D> {
    pub fn as_slice_mut(&mut self) -> Option<SliceMut<T>> {
        if self.strides == self.dim.default_strides() {
            Some(self.data.as_slice_mut())
        } else {
            None
        }
    }
    pub fn as_slice_memory_order_mut(&mut self) -> Option<SliceMut<T>> {
        if D::is_contiguous(&self.dim, &self.strides) {
            Some(self.data.as_slice_mut())
        } else {
            None
        }
    }
}
