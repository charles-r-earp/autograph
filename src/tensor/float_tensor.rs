use super::{DataBase, Dimension, IxDyn, Sealed, TensorBase};
use crate::{
    backend::{Buffer, BufferSlice, BufferSliceMut},
    Result,
};
use half::bf16;
use std::sync::{Arc, Weak};

pub enum FloatBuffer {
    BF16(Buffer<bf16>),
    F32(Buffer<f32>),
}

impl FloatBuffer {
    fn zeros_like(buffer: &FloatBufferSlice) -> Result<Self> {
        todo!()
    }
    fn ones_like(buffer: &FloatBufferSlice) -> Result<Self> {
        todo!()
    }
    fn zeros_like_weak(buffer: &FloatWeakBuffer) -> Result<Self> {
        todo!()
    }
    fn ones_like_weak(buffer: &FloatWeakBuffer) -> Result<Self> {
        todo!()
    }
    fn to_buffer(&self) -> Result<FloatBuffer> {
        match self {
            Self::BF16(x) => Ok(FloatBuffer::BF16(x.to_buffer()?)),
            Self::F32(x) => Ok(FloatBuffer::F32(x.to_buffer()?)),
        }
    }
    fn as_float_buffer_slice(&self) -> FloatBufferSlice {
        match self {
            Self::BF16(x) => FloatBufferSlice::BF16(x.as_buffer_slice()),
            Self::F32(x) => FloatBufferSlice::F32(x.as_buffer_slice()),
        }
    }
    fn as_float_buffer_slice_mut(&mut self) -> FloatBufferSliceMut {
        match self {
            Self::BF16(x) => FloatBufferSliceMut::BF16(x.as_buffer_slice_mut()),
            Self::F32(x) => FloatBufferSliceMut::F32(x.as_buffer_slice_mut()),
        }
    }
}

#[derive(Clone)]
pub enum FloatArcBuffer {
    BF16(Arc<Buffer<bf16>>),
    F32(Arc<Buffer<f32>>),
}

impl FloatArcBuffer {
    fn to_buffer(&self) -> Result<FloatBuffer> {
        match self {
            Self::BF16(x) => Ok(FloatBuffer::BF16(x.to_buffer()?)),
            Self::F32(x) => Ok(FloatBuffer::F32(x.to_buffer()?)),
        }
    }
    fn as_float_buffer_slice(&self) -> FloatBufferSlice {
        match self {
            Self::BF16(x) => FloatBufferSlice::BF16(x.as_buffer_slice()),
            Self::F32(x) => FloatBufferSlice::F32(x.as_buffer_slice()),
        }
    }
}

impl From<FloatBuffer> for FloatArcBuffer {
    fn from(buffer: FloatBuffer) -> Self {
        match buffer {
            FloatBuffer::BF16(x) => Self::BF16(Arc::new(x)),
            FloatBuffer::F32(x) => Self::F32(Arc::new(x)),
        }
    }
}

#[derive(Clone)]
pub enum FloatWeakBuffer {
    BF16(Weak<Buffer<bf16>>),
    F32(Weak<Buffer<f32>>),
}

impl From<&FloatArcBuffer> for FloatWeakBuffer {
    fn from(buffer: &FloatArcBuffer) -> Self {
        match buffer {
            FloatArcBuffer::BF16(x) => Self::BF16(Arc::downgrade(x)),
            FloatArcBuffer::F32(x) => Self::F32(Arc::downgrade(x)),
        }
    }
}

pub enum FloatBufferSlice<'a> {
    BF16(BufferSlice<'a, bf16>),
    F32(BufferSlice<'a, f32>),
}

impl FloatBufferSlice<'_> {
    fn to_buffer(&self) -> Result<FloatBuffer> {
        match self {
            Self::BF16(x) => Ok(FloatBuffer::BF16(x.to_buffer()?)),
            Self::F32(x) => Ok(FloatBuffer::F32(x.to_buffer()?)),
        }
    }
    fn as_float_buffer_slice(&self) -> FloatBufferSlice {
        match self {
            Self::BF16(x) => FloatBufferSlice::BF16(x.as_buffer_slice()),
            Self::F32(x) => FloatBufferSlice::F32(x.as_buffer_slice()),
        }
    }
}

pub enum FloatBufferSliceMut<'a> {
    BF16(BufferSliceMut<'a, bf16>),
    F32(BufferSliceMut<'a, f32>),
}

impl FloatBufferSliceMut<'_> {
    fn to_buffer(&self) -> Result<FloatBuffer> {
        match self {
            Self::BF16(x) => Ok(FloatBuffer::BF16(x.to_buffer()?)),
            Self::F32(x) => Ok(FloatBuffer::F32(x.to_buffer()?)),
        }
    }
    fn as_float_buffer_slice(&self) -> FloatBufferSlice {
        match self {
            Self::BF16(x) => FloatBufferSlice::BF16(x.as_buffer_slice()),
            Self::F32(x) => FloatBufferSlice::F32(x.as_buffer_slice()),
        }
    }
    fn as_float_buffer_slice_mut(&mut self) -> FloatBufferSliceMut {
        match self {
            Self::BF16(x) => FloatBufferSliceMut::BF16(x.as_buffer_slice_mut()),
            Self::F32(x) => FloatBufferSliceMut::F32(x.as_buffer_slice_mut()),
        }
    }
}

pub trait FloatData: DataBase + Sized {
    #[doc(hidden)]
    fn into_float_buffer(self) -> Result<FloatBuffer>;
    #[doc(hidden)]
    fn into_float_arc_buffer(self) -> Result<FloatArcBuffer> {
        Ok(self.into_float_buffer()?.into())
    }
    #[doc(hidden)]
    fn as_float_buffer_slice(&self) -> FloatBufferSlice;
}

pub trait FloatDataOwned: FloatData {
    #[doc(hidden)]
    fn from_float_buffer(buffer: FloatBuffer) -> Self;
}

pub trait FloatDataMut: FloatData {
    #[doc(hidden)]
    fn as_float_buffer_slice_mut(&mut self) -> FloatBufferSliceMut;
}

pub struct FloatOwnedRepr(FloatBuffer);

pub type FloatTensor<D> = TensorBase<FloatOwnedRepr, D>;
pub type FloatTensorD = FloatTensor<IxDyn>;

impl Sealed for FloatOwnedRepr {}

impl DataBase for FloatOwnedRepr {}

impl FloatData for FloatOwnedRepr {
    fn into_float_buffer(self) -> Result<FloatBuffer> {
        Ok(self.0)
    }
    fn into_float_arc_buffer(self) -> Result<FloatArcBuffer> {
        Ok(self.0.into())
    }
    fn as_float_buffer_slice(&self) -> FloatBufferSlice {
        self.0.as_float_buffer_slice()
    }
}

impl FloatDataOwned for FloatOwnedRepr {
    fn from_float_buffer(buffer: FloatBuffer) -> Self {
        Self(buffer)
    }
}

impl FloatDataMut for FloatOwnedRepr {
    fn as_float_buffer_slice_mut(&mut self) -> FloatBufferSliceMut {
        self.0.as_float_buffer_slice_mut()
    }
}

#[derive(Clone)]
pub struct FloatArcRepr(FloatArcBuffer);

pub type FloatArcTensor<D> = TensorBase<FloatArcRepr, D>;
pub type FloatArcTensorD = FloatArcTensor<IxDyn>;

impl Sealed for FloatArcRepr {}

impl DataBase for FloatArcRepr {}

impl FloatData for FloatArcRepr {
    fn into_float_buffer(self) -> Result<FloatBuffer> {
        Ok(self.0.to_buffer()?)
    }
    fn into_float_arc_buffer(self) -> Result<FloatArcBuffer> {
        Ok(self.0)
    }
    fn as_float_buffer_slice(&self) -> FloatBufferSlice {
        self.0.as_float_buffer_slice()
    }
}

#[derive(Clone)]
pub struct FloatWeakRepr(FloatWeakBuffer);

impl Sealed for FloatWeakRepr {}

impl DataBase for FloatWeakRepr {}

pub type FloatWeakTensor<D> = TensorBase<FloatWeakRepr, D>;
pub type FloatWeakTensorD = FloatWeakTensor<IxDyn>;

pub struct FloatViewRepr<'a>(FloatBufferSlice<'a>);

pub type FloatTensorView<'a, D> = TensorBase<FloatViewRepr<'a>, D>;
pub type FloatTensorViewD<'a> = FloatTensorView<'a, IxDyn>;

impl Sealed for FloatViewRepr<'_> {}

impl DataBase for FloatViewRepr<'_> {}

impl FloatData for FloatViewRepr<'_> {
    fn into_float_buffer(self) -> Result<FloatBuffer> {
        Ok(self.0.to_buffer()?)
    }
    fn as_float_buffer_slice(&self) -> FloatBufferSlice {
        self.0.as_float_buffer_slice()
    }
}

pub struct FloatViewMutRepr<'a>(FloatBufferSliceMut<'a>);

pub type FloatTensorViewMut<'a, D> = TensorBase<FloatViewMutRepr<'a>, D>;
pub type FloatTensorViewMutD<'a> = FloatTensorViewMut<'a, IxDyn>;

impl Sealed for FloatViewMutRepr<'_> {}

impl DataBase for FloatViewMutRepr<'_> {}

impl FloatData for FloatViewMutRepr<'_> {
    fn into_float_buffer(self) -> Result<FloatBuffer> {
        Ok(self.0.to_buffer()?)
    }
    fn as_float_buffer_slice(&self) -> FloatBufferSlice {
        self.0.as_float_buffer_slice()
    }
}

impl FloatDataMut for FloatViewMutRepr<'_> {
    fn as_float_buffer_slice_mut(&mut self) -> FloatBufferSliceMut {
        self.0.as_float_buffer_slice_mut()
    }
}

impl<S: FloatData, D: Dimension> TensorBase<S, D> {
    pub fn into_float_tensor(self) -> Result<FloatTensor<D>> {
        Ok(TensorBase {
            device: self.device.clone(),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            data: FloatOwnedRepr(self.data.into_float_buffer()?),
        })
    }
    pub fn into_float_arc_tensor(self) -> Result<FloatArcTensor<D>> {
        Ok(TensorBase {
            device: self.device.clone(),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            data: FloatArcRepr(self.data.into_float_arc_buffer()?),
        })
    }
    pub fn float_view(&self) -> FloatTensorView<D> {
        TensorBase {
            device: self.device.clone(),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            data: FloatViewRepr(self.data.as_float_buffer_slice()),
        }
    }
}

impl<S: FloatDataOwned, D: Dimension> TensorBase<S, D> {
    pub(crate) fn float_zeros_like<S2: FloatData>(tensor: &TensorBase<S, D>) -> Result<Self> {
        let view = tensor.float_view();
        Ok(Self {
            device: view.device,
            dim: view.dim,
            strides: view.strides,
            data: S::from_float_buffer(FloatBuffer::zeros_like(&view.data.0)?),
        })
    }
    pub(crate) fn float_ones_like<S2: FloatData>(tensor: &TensorBase<S, D>) -> Result<Self> {
        let view = tensor.float_view();
        Ok(Self {
            device: view.device,
            dim: view.dim,
            strides: view.strides,
            data: S::from_float_buffer(FloatBuffer::ones_like(&view.data.0)?),
        })
    }
    pub(crate) fn float_zeros_like_weak(tensor: &FloatWeakTensor<D>) -> Result<Self> {
        Ok(Self {
            device: tensor.device.clone(),
            dim: tensor.dim.clone(),
            strides: tensor.strides.clone(),
            data: S::from_float_buffer(FloatBuffer::zeros_like_weak(&tensor.data.0)?),
        })
    }
    pub(crate) fn float_ones_like_weak(tensor: &FloatWeakTensor<D>) -> Result<Self> {
        Ok(Self {
            device: tensor.device.clone(),
            dim: tensor.dim.clone(),
            strides: tensor.strides.clone(),
            data: S::from_float_buffer(FloatBuffer::ones_like_weak(&tensor.data.0)?),
        })
    }
}

impl<S: FloatDataMut, D: Dimension> TensorBase<S, D> {
    pub fn float_view_mut(&mut self) -> FloatTensorViewMut<D> {
        TensorBase {
            device: self.device.clone(),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            data: FloatViewMutRepr(self.data.as_float_buffer_slice_mut()),
        }
    }
}

impl<D: Dimension> FloatWeakTensor<D> {
    pub(crate) fn vertex_key(&self) -> usize {
        match &self.data.0 {
            FloatWeakBuffer::BF16(x) => Weak::as_ptr(x) as usize,
            FloatWeakBuffer::F32(x) => Weak::as_ptr(x) as usize,
        }
    }
}

impl<D: Dimension> From<FloatTensor<D>> for FloatArcTensor<D> {
    fn from(tensor: FloatTensor<D>) -> Self {
        Self {
            device: tensor.device,
            dim: tensor.dim,
            strides: tensor.strides,
            data: FloatArcRepr(tensor.data.0.into()),
        }
    }
}

impl<D: Dimension> From<&FloatArcTensor<D>> for FloatWeakTensor<D> {
    fn from(tensor: &FloatArcTensor<D>) -> Self {
        Self {
            device: tensor.device.clone(),
            dim: tensor.dim.clone(),
            strides: tensor.strides.clone(),
            data: FloatWeakRepr((&tensor.data.0).into()),
        }
    }
}
