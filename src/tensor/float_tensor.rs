use super::{
    ArcRepr, ArcTensor, Axis, CowRepr, CowTensor, DataBase, Dimension, Ix2, IxDyn, OwnedRepr,
    RemoveAxis, Scalar, Sealed, ShapeBuilder, Tensor, TensorBase, TensorView, TensorViewMut,
    TryIntoData, ViewMutRepr, ViewRepr,
};
pub use crate::backend::FloatType;
use crate::{
    backend::{Buffer, BufferSlice, BufferSliceMut, CowBuffer, Device, Float, Num},
    util::type_eq,
    Result,
};
use anyhow::anyhow;
use half::bf16;
use serde::{Deserialize, Serialize};
use std::{
    any::type_name,
    convert::{TryFrom, TryInto},
    future::Future,
    mem::transmute,
    sync::Arc,
};

#[allow(clippy::upper_case_acronyms)]
#[derive(Serialize, Deserialize)]
pub enum FloatBuffer {
    BF16(Buffer<bf16>),
    F32(Buffer<f32>),
}

impl FloatBuffer {
    fn zeros(device: &Device, float_type: FloatType, len: usize) -> Result<Self> {
        match float_type {
            FloatType::BF16 => Ok(Self::BF16(Buffer::zeros(device, len)?)),
            FloatType::F32 => Ok(Self::F32(Buffer::zeros(device, len)?)),
        }
    }
    fn ones(device: &Device, float_type: FloatType, len: usize) -> Result<Self> {
        match float_type {
            FloatType::BF16 => Ok(Self::BF16(Buffer::ones(device, len)?)),
            FloatType::F32 => Ok(Self::F32(Buffer::ones(device, len)?)),
        }
    }
    /*fn to_buffer(&self) -> Result<FloatBuffer> {
        match self {
            Self::BF16(x) => Ok(FloatBuffer::BF16(x.to_buffer()?)),
            Self::F32(x) => Ok(FloatBuffer::F32(x.to_buffer()?)),
        }
    }*/
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
    fn float_type(&self) -> FloatType {
        match self {
            Self::BF16(_) => FloatType::BF16,
            Self::F32(_) => FloatType::F32,
        }
    }
}

impl<T: Float> From<Buffer<T>> for FloatBuffer {
    fn from(buffer: Buffer<T>) -> Self {
        if type_eq::<T, bf16>() {
            Self::BF16(unsafe { transmute(buffer) })
        } else if type_eq::<T, f32>() {
            Self::F32(unsafe { transmute(buffer) })
        } else {
            unreachable!()
        }
    }
}

impl<T: Scalar> TryFrom<FloatBuffer> for Buffer<T> {
    type Error = anyhow::Error;
    fn try_from(from: FloatBuffer) -> Result<Self> {
        let float_type = from.float_type();
        match from {
            FloatBuffer::BF16(buffer) => {
                if type_eq::<T, bf16>() {
                    return Ok(unsafe { transmute(buffer) });
                }
            }
            FloatBuffer::F32(buffer) => {
                if type_eq::<T, f32>() {
                    return Ok(unsafe { transmute(buffer) });
                }
            }
        }
        Err(anyhow!(
            "Expected {} found {:?}!",
            type_name::<T>(),
            float_type
        ))
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Serialize, Deserialize)]
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
    fn float_type(&self) -> FloatType {
        match self {
            Self::BF16(_) => FloatType::BF16,
            Self::F32(_) => FloatType::F32,
        }
    }
    fn make_mut(&mut self) -> Result<FloatBufferSliceMut> {
        match self {
            Self::BF16(buffer) => Ok(Buffer::make_mut(buffer)?.into()),
            Self::F32(buffer) => Ok(Buffer::make_mut(buffer)?.into()),
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

impl<T: Float> From<Arc<Buffer<T>>> for FloatArcBuffer {
    fn from(buffer: Arc<Buffer<T>>) -> Self {
        if type_eq::<T, bf16>() {
            Self::BF16(unsafe { transmute(buffer) })
        } else if type_eq::<T, f32>() {
            Self::F32(unsafe { transmute(buffer) })
        } else {
            unreachable!()
        }
    }
}

impl<T: Scalar> TryFrom<FloatArcBuffer> for Arc<Buffer<T>> {
    type Error = anyhow::Error;
    fn try_from(from: FloatArcBuffer) -> Result<Self> {
        let float_type = from.float_type();
        match from {
            FloatArcBuffer::BF16(buffer) => {
                if type_eq::<T, bf16>() {
                    return Ok(unsafe { transmute(buffer) });
                }
            }
            FloatArcBuffer::F32(buffer) => {
                if type_eq::<T, f32>() {
                    return Ok(unsafe { transmute(buffer) });
                }
            }
        }
        Err(anyhow!(
            "Expected {} found {:?}!",
            type_name::<T>(),
            float_type
        ))
    }
}

#[allow(clippy::upper_case_acronyms)]
pub enum FloatBufferSlice<'a> {
    BF16(BufferSlice<'a, bf16>),
    F32(BufferSlice<'a, f32>),
}

impl FloatBufferSlice<'_> {
    fn float_type(&self) -> FloatType {
        match self {
            Self::BF16(_) => FloatType::BF16,
            Self::F32(_) => FloatType::F32,
        }
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
}

impl<'a, T: Float> From<BufferSlice<'a, T>> for FloatBufferSlice<'a> {
    fn from(buffer: BufferSlice<'a, T>) -> Self {
        if type_eq::<T, bf16>() {
            Self::BF16(unsafe { transmute(buffer) })
        } else if type_eq::<T, f32>() {
            Self::F32(unsafe { transmute(buffer) })
        } else {
            unreachable!()
        }
    }
}

impl<'a, T: Scalar> TryFrom<FloatBufferSlice<'a>> for BufferSlice<'a, T> {
    type Error = anyhow::Error;
    fn try_from(from: FloatBufferSlice<'a>) -> Result<Self> {
        let float_type = from.float_type();
        match from {
            FloatBufferSlice::BF16(buffer) => {
                if type_eq::<T, bf16>() {
                    return Ok(unsafe { transmute(buffer) });
                }
            }
            FloatBufferSlice::F32(buffer) => {
                if type_eq::<T, f32>() {
                    return Ok(unsafe { transmute(buffer) });
                }
            }
        }
        Err(anyhow!(
            "Expected {} found {:?}!",
            type_name::<T>(),
            float_type
        ))
    }
}

#[allow(clippy::upper_case_acronyms)]
pub enum FloatBufferSliceMut<'a> {
    BF16(BufferSliceMut<'a, bf16>),
    F32(BufferSliceMut<'a, f32>),
}

impl FloatBufferSliceMut<'_> {
    fn float_type(&self) -> FloatType {
        match self {
            Self::BF16(_) => FloatType::BF16,
            Self::F32(_) => FloatType::F32,
        }
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

impl<'a, T: Float> From<BufferSliceMut<'a, T>> for FloatBufferSliceMut<'a> {
    fn from(buffer: BufferSliceMut<'a, T>) -> Self {
        if type_eq::<T, bf16>() {
            Self::BF16(unsafe { transmute(buffer) })
        } else if type_eq::<T, f32>() {
            Self::F32(unsafe { transmute(buffer) })
        } else {
            unreachable!()
        }
    }
}

impl<'a, T: Scalar> TryFrom<FloatBufferSliceMut<'a>> for BufferSliceMut<'a, T> {
    type Error = anyhow::Error;
    fn try_from(from: FloatBufferSliceMut<'a>) -> Result<Self> {
        let float_type = from.float_type();
        match from {
            FloatBufferSliceMut::BF16(buffer) => {
                if type_eq::<T, bf16>() {
                    return Ok(unsafe { transmute(buffer) });
                }
            }
            FloatBufferSliceMut::F32(buffer) => {
                if type_eq::<T, f32>() {
                    return Ok(unsafe { transmute(buffer) });
                }
            }
        }
        Err(anyhow!(
            "Expected {} found {:?}!",
            type_name::<T>(),
            float_type
        ))
    }
}

pub enum FloatCowBuffer<'a> {
    BF16(CowBuffer<'a, bf16>),
    F32(CowBuffer<'a, f32>),
}

impl FloatCowBuffer<'_> {
    fn float_type(&self) -> FloatType {
        match self {
            Self::BF16(_) => FloatType::BF16,
            Self::F32(_) => FloatType::F32,
        }
    }
    fn into_float_buffer(self) -> Result<FloatBuffer> {
        match self {
            Self::BF16(buffer) => Ok(buffer.into_buffer()?.into()),
            Self::F32(buffer) => Ok(buffer.into_buffer()?.into()),
        }
    }
    fn as_float_buffer_slice(&self) -> FloatBufferSlice {
        match self {
            Self::BF16(buffer) => buffer.as_buffer_slice().into(),
            Self::F32(buffer) => buffer.as_buffer_slice().into(),
        }
    }
}

impl From<FloatBuffer> for FloatCowBuffer<'_> {
    fn from(from: FloatBuffer) -> Self {
        match from {
            FloatBuffer::BF16(buffer) => Self::BF16(buffer.into()),
            FloatBuffer::F32(buffer) => Self::F32(buffer.into()),
        }
    }
}

impl<'a> From<FloatBufferSlice<'a>> for FloatCowBuffer<'a> {
    fn from(from: FloatBufferSlice<'a>) -> Self {
        match from {
            FloatBufferSlice::BF16(buffer) => Self::BF16(buffer.into()),
            FloatBufferSlice::F32(buffer) => Self::F32(buffer.into()),
        }
    }
}

impl<'a, T: Float> From<CowBuffer<'a, T>> for FloatCowBuffer<'a> {
    fn from(from: CowBuffer<'a, T>) -> Self {
        if type_eq::<T, bf16>() {
            Self::BF16(unsafe { transmute(from) })
        } else if type_eq::<T, f32>() {
            Self::F32(unsafe { transmute(from) })
        } else {
            unreachable!()
        }
    }
}

impl<'a, T: Scalar> TryFrom<FloatCowBuffer<'a>> for CowBuffer<'a, T> {
    type Error = anyhow::Error;
    fn try_from(from: FloatCowBuffer<'a>) -> Result<Self> {
        let float_type = from.float_type();
        match from {
            FloatCowBuffer::BF16(buffer) => {
                if type_eq::<T, bf16>() {
                    return Ok(unsafe { transmute(buffer) });
                }
            }
            FloatCowBuffer::F32(buffer) => {
                if type_eq::<T, f32>() {
                    return Ok(unsafe { transmute(buffer) });
                }
            }
        }
        Err(anyhow!(
            "Expected {} found {:?}!",
            type_name::<T>(),
            float_type
        ))
    }
}

pub trait FloatData: DataBase + Sized + TryIntoData<bf16> + TryIntoData<f32> {
    #[doc(hidden)]
    fn float_type(&self) -> FloatType;
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

#[derive(Serialize, Deserialize)]
pub struct FloatOwnedRepr(FloatBuffer);

pub type FloatTensor<D> = TensorBase<FloatOwnedRepr, D>;
pub type FloatTensor2 = TensorBase<FloatOwnedRepr, Ix2>;
pub type FloatTensorD = FloatTensor<IxDyn>;

impl Sealed for FloatOwnedRepr {}

impl DataBase for FloatOwnedRepr {}

impl FloatData for FloatOwnedRepr {
    fn float_type(&self) -> FloatType {
        self.0.float_type()
    }
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

impl<T: Scalar> TryIntoData<T> for FloatOwnedRepr {
    type Data = OwnedRepr<T>;
    fn try_into_data(self) -> Result<Self::Data> {
        Ok(OwnedRepr(self.0.try_into()?))
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct FloatArcRepr(FloatArcBuffer);

pub type FloatArcTensor<D> = TensorBase<FloatArcRepr, D>;
pub type FloatArcTensorD = FloatArcTensor<IxDyn>;

impl Sealed for FloatArcRepr {}

impl DataBase for FloatArcRepr {}

impl FloatData for FloatArcRepr {
    fn float_type(&self) -> FloatType {
        self.0.float_type()
    }
    fn into_float_buffer(self) -> Result<FloatBuffer> {
        self.0.to_buffer()
    }
    fn into_float_arc_buffer(self) -> Result<FloatArcBuffer> {
        Ok(self.0)
    }
    fn as_float_buffer_slice(&self) -> FloatBufferSlice {
        self.0.as_float_buffer_slice()
    }
}

impl<T: Scalar> TryIntoData<T> for FloatArcRepr {
    type Data = ArcRepr<T>;
    fn try_into_data(self) -> Result<Self::Data> {
        Ok(ArcRepr(self.0.try_into()?))
    }
}

impl FloatDataOwned for FloatArcRepr {
    fn from_float_buffer(buffer: FloatBuffer) -> Self {
        Self(buffer.into())
    }
}

pub struct FloatViewRepr<'a>(FloatBufferSlice<'a>);

pub type FloatTensorView<'a, D> = TensorBase<FloatViewRepr<'a>, D>;
pub type FloatTensorViewD<'a> = FloatTensorView<'a, IxDyn>;

impl Sealed for FloatViewRepr<'_> {}

impl DataBase for FloatViewRepr<'_> {}

impl FloatData for FloatViewRepr<'_> {
    fn float_type(&self) -> FloatType {
        self.0.float_type()
    }
    fn into_float_buffer(self) -> Result<FloatBuffer> {
        self.0.to_buffer()
    }
    fn as_float_buffer_slice(&self) -> FloatBufferSlice {
        self.0.as_float_buffer_slice()
    }
}

impl<'a, T: Scalar> TryIntoData<T> for FloatViewRepr<'a> {
    type Data = ViewRepr<'a, T>;
    fn try_into_data(self) -> Result<Self::Data> {
        Ok(ViewRepr(self.0.try_into()?))
    }
}

pub struct FloatViewMutRepr<'a>(FloatBufferSliceMut<'a>);

pub type FloatTensorViewMut<'a, D> = TensorBase<FloatViewMutRepr<'a>, D>;
pub type FloatTensorViewMutD<'a> = FloatTensorViewMut<'a, IxDyn>;

impl Sealed for FloatViewMutRepr<'_> {}

impl DataBase for FloatViewMutRepr<'_> {}

impl FloatData for FloatViewMutRepr<'_> {
    fn float_type(&self) -> FloatType {
        self.0.float_type()
    }
    fn into_float_buffer(self) -> Result<FloatBuffer> {
        self.0.to_buffer()
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

impl<'a, T: Scalar> TryIntoData<T> for FloatViewMutRepr<'a> {
    type Data = ViewMutRepr<'a, T>;
    fn try_into_data(self) -> Result<Self::Data> {
        Ok(ViewMutRepr(self.0.try_into()?))
    }
}

pub struct FloatCowRepr<'a>(FloatCowBuffer<'a>);

impl Sealed for FloatCowRepr<'_> {}

impl DataBase for FloatCowRepr<'_> {}

impl FloatData for FloatCowRepr<'_> {
    fn float_type(&self) -> FloatType {
        self.0.float_type()
    }
    fn into_float_buffer(self) -> Result<FloatBuffer> {
        self.0.into_float_buffer()
    }
    fn as_float_buffer_slice(&self) -> FloatBufferSlice {
        self.0.as_float_buffer_slice()
    }
}

impl<'a, T: Scalar> TryIntoData<T> for FloatCowRepr<'a> {
    type Data = CowRepr<'a, T>;
    fn try_into_data(self) -> Result<Self::Data> {
        Ok(CowRepr(self.0.try_into()?))
    }
}

pub type FloatCowTensor<'a, D> = TensorBase<FloatCowRepr<'a>, D>;

impl<S: FloatData, D: Dimension> TensorBase<S, D> {
    pub fn float_type(&self) -> FloatType {
        self.data.float_type()
    }
    pub fn into_float_tensor(self) -> Result<FloatTensor<D>> {
        Ok(TensorBase {
            device: self.device,
            dim: self.dim,
            strides: self.strides,
            data: FloatOwnedRepr(self.data.into_float_buffer()?),
        })
    }
    /*fn into_float_cow_tensor(self) -> Result<FloatCowTensor<'static, D>> {
        Ok(TensorBase {
            device: self.device,
            dim: self.dim,
            strides: self.strides,
            data: FloatCowRepr(self.data.into_float_buffer()?.into()),
        })
    }*/
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
    pub fn float_cast_into<T: Num>(self) -> Result<Tensor<T, D>>
    where
        S: TryIntoData<bf16> + TryIntoData<f32>,
    {
        match self.float_type() {
            FloatType::BF16 => self.try_into_::<bf16>()?.cast_into(),
            FloatType::F32 => self.try_into_::<f32>()?.cast_into(),
        }
    }
    pub fn float_cast_to<T: Num>(&self) -> Result<CowTensor<T, D>> {
        match self.float_type() {
            FloatType::BF16 => {
                if type_eq::<T, bf16>() {
                    Ok(self.float_view().try_into_::<T>()?.into())
                } else {
                    Ok(self.float_view().try_into_::<bf16>()?.cast_into()?.into())
                }
            }
            FloatType::F32 => {
                if type_eq::<T, f32>() {
                    Ok(self.float_view().try_into_::<T>()?.into())
                } else {
                    Ok(self.float_view().try_into_::<f32>()?.cast_into()?.into())
                }
            }
        }
    }
    pub fn to_float_cow_tensor(&self) -> Result<FloatCowTensor<D>> {
        Ok(TensorBase {
            device: self.device.clone(),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            data: FloatCowRepr(self.data.as_float_buffer_slice().into()),
        })
    }
    pub fn float_into_device(
        self,
        device: &Device,
    ) -> Result<impl Future<Output = Result<FloatTensor<D>>>> {
        let device = device.clone();
        Ok(async move {
            if self.device == device {
                self.into_float_tensor()
            } else {
                self.float_to_device(&device)?.await?.into_float_tensor()
            }
        })
    }
    pub fn float_into_device_arc(
        self,
        device: &Device,
    ) -> Result<impl Future<Output = Result<FloatArcTensor<D>>>> {
        let device = device.clone();
        Ok(async move {
            if self.device == device {
                self.into_float_arc_tensor()
            } else {
                self.float_to_device(&device)?
                    .await?
                    .into_float_arc_tensor()
            }
        })
    }
    pub fn float_to_device<'a>(
        &'a self,
        device: &Device,
    ) -> Result<impl Future<Output = Result<FloatCowTensor<'a, D>>> + 'a> {
        let device = device.clone();
        Ok(async move {
            if self.device == device {
                self.to_float_cow_tensor()
            } else {
                match self.data.as_float_buffer_slice() {
                    FloatBufferSlice::BF16(slice) => {
                        let buffer: FloatBuffer = slice.into_device(&device)?.await?.into();
                        Ok(FloatCowTensor {
                            device,
                            dim: self.dim.clone(),
                            strides: self.strides.clone(),
                            data: FloatCowRepr(buffer.into()),
                        })
                    }
                    FloatBufferSlice::F32(slice) => {
                        let buffer: FloatBuffer = slice.into_device(&device)?.await?.into();
                        Ok(FloatCowTensor {
                            device,
                            dim: self.dim.clone(),
                            strides: self.strides.clone(),
                            data: FloatCowRepr(buffer.into()),
                        })
                    }
                }
            }
        })
    }
    pub fn float_argmax(&self, axis: Axis) -> Result<Tensor<u32, D::Smaller>>
    where
        D: RemoveAxis,
    {
        match self.float_type() {
            FloatType::BF16 => self.float_view().try_into_::<bf16>()?.argmax(axis),
            FloatType::F32 => self.float_view().try_into_::<f32>()?.argmax(axis),
        }
    }
}

impl<D: Dimension> FloatArcTensor<D> {
    pub fn float_make_mut(&mut self) -> Result<FloatTensorViewMut<D>> {
        Ok(TensorBase {
            device: self.device.clone(),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            data: FloatViewMutRepr(self.data.0.make_mut()?),
        })
    }
}

impl<S: FloatDataOwned, D: Dimension> TensorBase<S, D> {
    pub fn float_zeros<Sh>(device: &Device, float_type: FloatType, shape: Sh) -> Result<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        let (dim, strides) = super::dim_strides_from_shape(shape.into_shape());
        let data = S::from_float_buffer(FloatBuffer::zeros(device, float_type, dim.size())?);
        Ok(Self {
            device: device.clone(),
            dim,
            strides,
            data,
        })
    }
    pub fn float_ones<Sh>(device: &Device, float_type: FloatType, shape: Sh) -> Result<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        let (dim, strides) = super::dim_strides_from_shape(shape.into_shape());
        let data = S::from_float_buffer(FloatBuffer::ones(device, float_type, dim.size())?);
        Ok(Self {
            device: device.clone(),
            dim,
            strides,
            data,
        })
    }
    pub fn float_to_device_mut<'a>(
        &'a mut self,
        device: &Device,
    ) -> Result<impl Future<Output = Result<()>> + 'a> {
        let device = device.clone();
        Ok(async move {
            if self.device != device {
                let buffer = self
                    .float_view()
                    .float_into_device(&device)?
                    .await?
                    .data
                    .into_float_buffer()?;
                self.device = device;
                self.data = S::from_float_buffer(buffer);
            }
            Ok(())
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

impl<T: Float, D: Dimension> From<Tensor<T, D>> for FloatTensor<D> {
    fn from(tensor: Tensor<T, D>) -> Self {
        Self {
            device: tensor.device,
            dim: tensor.dim,
            strides: tensor.strides,
            data: FloatOwnedRepr(tensor.data.0.into()),
        }
    }
}

impl<T: Float, D: Dimension> From<ArcTensor<T, D>> for FloatArcTensor<D> {
    fn from(tensor: ArcTensor<T, D>) -> Self {
        Self {
            device: tensor.device,
            dim: tensor.dim,
            strides: tensor.strides,
            data: FloatArcRepr(tensor.data.0.into()),
        }
    }
}

impl<T: Float, D: Dimension> TryFrom<FloatTensor<D>> for Tensor<T, D> {
    type Error = anyhow::Error;
    fn try_from(tensor: FloatTensor<D>) -> Result<Self> {
        tensor.try_into_()
    }
}

impl<'a, T: Float, D: Dimension> TryFrom<FloatTensorView<'a, D>> for TensorView<'a, T, D> {
    type Error = anyhow::Error;
    fn try_from(tensor: FloatTensorView<'a, D>) -> Result<Self> {
        tensor.try_into_()
    }
}

impl<'a, T: Float, D: Dimension> TryFrom<FloatTensorViewMut<'a, D>> for TensorViewMut<'a, T, D> {
    type Error = anyhow::Error;
    fn try_from(tensor: FloatTensorViewMut<'a, D>) -> Result<Self> {
        tensor.try_into_()
    }
}

impl<T: Float, D: Dimension> TryFrom<FloatArcTensor<D>> for ArcTensor<T, D> {
    type Error = anyhow::Error;
    fn try_from(tensor: FloatArcTensor<D>) -> Result<Self> {
        tensor.try_into_()
    }
}
