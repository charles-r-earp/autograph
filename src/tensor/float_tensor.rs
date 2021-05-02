use super::{
    ArcTensor, Axis, DataBase, Dimension, Ix2, IxDyn, RemoveAxis, Sealed, ShapeBuilder, Tensor,
    TensorBase, TensorView, TensorViewMut,
};
pub use crate::backend::FloatType;
use crate::{
    backend::{Buffer, BufferSlice, BufferSliceMut, Device, Float, Num},
    util::type_eq,
    Result,
};
use anyhow::bail;
use half::bf16;
use std::{
    convert::{TryFrom, TryInto},
    mem::transmute,
    sync::{Arc, Weak},
};

#[allow(clippy::upper_case_acronyms)]
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

#[allow(clippy::upper_case_acronyms)]
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

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone)]
pub enum FloatWeakBuffer {
    BF16(Weak<Buffer<bf16>>),
    F32(Weak<Buffer<f32>>),
}

impl FloatWeakBuffer {
    fn float_type(&self) -> FloatType {
        match self {
            Self::BF16(_) => FloatType::BF16,
            Self::F32(_) => FloatType::F32,
        }
    }
}

impl From<&FloatArcBuffer> for FloatWeakBuffer {
    fn from(buffer: &FloatArcBuffer) -> Self {
        match buffer {
            FloatArcBuffer::BF16(x) => Self::BF16(Arc::downgrade(x)),
            FloatArcBuffer::F32(x) => Self::F32(Arc::downgrade(x)),
        }
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

pub trait FloatDataBase: DataBase {
    #[doc(hidden)]
    fn float_type(&self) -> FloatType;
}

pub trait FloatData: FloatDataBase + Sized {
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
pub type FloatTensor2 = TensorBase<FloatOwnedRepr, Ix2>;
pub type FloatTensorD = FloatTensor<IxDyn>;

impl Sealed for FloatOwnedRepr {}

impl DataBase for FloatOwnedRepr {}

impl FloatDataBase for FloatOwnedRepr {
    fn float_type(&self) -> FloatType {
        self.0.float_type()
    }
}

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

impl FloatDataBase for FloatArcRepr {
    fn float_type(&self) -> FloatType {
        self.0.float_type()
    }
}

impl FloatData for FloatArcRepr {
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

#[derive(Clone)]
pub struct FloatWeakRepr(FloatWeakBuffer);

impl Sealed for FloatWeakRepr {}

impl DataBase for FloatWeakRepr {}

impl FloatDataBase for FloatWeakRepr {
    fn float_type(&self) -> FloatType {
        self.0.float_type()
    }
}

pub type FloatWeakTensor<D> = TensorBase<FloatWeakRepr, D>;
pub type FloatWeakTensorD = FloatWeakTensor<IxDyn>;

pub struct FloatViewRepr<'a>(FloatBufferSlice<'a>);

pub type FloatTensorView<'a, D> = TensorBase<FloatViewRepr<'a>, D>;
pub type FloatTensorViewD<'a> = FloatTensorView<'a, IxDyn>;

impl Sealed for FloatViewRepr<'_> {}

impl DataBase for FloatViewRepr<'_> {}

impl FloatDataBase for FloatViewRepr<'_> {
    fn float_type(&self) -> FloatType {
        self.0.float_type()
    }
}

impl FloatData for FloatViewRepr<'_> {
    fn into_float_buffer(self) -> Result<FloatBuffer> {
        self.0.to_buffer()
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

impl FloatDataBase for FloatViewMutRepr<'_> {
    fn float_type(&self) -> FloatType {
        self.0.float_type()
    }
}

impl FloatData for FloatViewMutRepr<'_> {
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

impl<S: FloatDataBase, D: Dimension> TensorBase<S, D> {
    pub(crate) fn float_type(&self) -> FloatType {
        self.data.float_type()
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
    pub(crate) fn float_zeros<Sh>(device: &Device, float_type: FloatType, shape: Sh) -> Result<Self>
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
    pub(crate) fn float_ones<Sh>(device: &Device, float_type: FloatType, shape: Sh) -> Result<Self>
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
        // The ugly transmute is required in order to have a bound on T: Float
        let data = if type_eq::<T, bf16>() {
            match tensor.data.0 {
                FloatBuffer::BF16(x) => unsafe { transmute(super::OwnedRepr(x)) },
                FloatBuffer::F32(_) => {
                    bail!("Expected bf16 found f32!");
                }
            }
        } else if type_eq::<T, f32>() {
            match tensor.data.0 {
                FloatBuffer::BF16(_) => {
                    bail!("Expected f32 found bf16!");
                }
                FloatBuffer::F32(x) => unsafe { transmute(super::OwnedRepr(x)) },
            }
        } else {
            unreachable!()
        };
        Ok(Self {
            device: tensor.device,
            dim: tensor.dim,
            strides: tensor.strides,
            data,
        })
    }
}

impl<'a, T: Float, D: Dimension> TryFrom<FloatTensorView<'a, D>> for TensorView<'a, T, D> {
    type Error = anyhow::Error;
    fn try_from(tensor: FloatTensorView<'a, D>) -> Result<Self> {
        // The ugly transmute is required in order to have a bound on T: Float
        let data = if type_eq::<T, bf16>() {
            match tensor.data.0 {
                FloatBufferSlice::BF16(x) => unsafe { transmute(super::ViewRepr(x)) },
                FloatBufferSlice::F32(_) => {
                    bail!("Expected bf16 found f32!");
                }
            }
        } else if type_eq::<T, f32>() {
            match tensor.data.0 {
                FloatBufferSlice::BF16(_) => {
                    bail!("Expected f32 found bf16!");
                }
                FloatBufferSlice::F32(x) => unsafe { transmute(super::ViewRepr(x)) },
            }
        } else {
            unreachable!()
        };
        Ok(Self {
            device: tensor.device,
            dim: tensor.dim,
            strides: tensor.strides,
            data,
        })
    }
}

impl<'a, T: Float, D: Dimension> TryFrom<FloatTensorViewMut<'a, D>> for TensorViewMut<'a, T, D> {
    type Error = anyhow::Error;
    fn try_from(tensor: FloatTensorViewMut<'a, D>) -> Result<Self> {
        // The ugly transmute is required in order to have a bound on T: Float
        let data = if type_eq::<T, bf16>() {
            match tensor.data.0 {
                FloatBufferSliceMut::BF16(x) => unsafe { transmute(super::ViewMutRepr(x)) },
                FloatBufferSliceMut::F32(_) => {
                    bail!("Expected bf16 found f32!");
                }
            }
        } else if type_eq::<T, f32>() {
            match tensor.data.0 {
                FloatBufferSliceMut::BF16(_) => {
                    bail!("Expected f32 found bf16!");
                }
                FloatBufferSliceMut::F32(x) => unsafe { transmute(super::ViewMutRepr(x)) },
            }
        } else {
            unreachable!()
        };
        Ok(Self {
            device: tensor.device,
            dim: tensor.dim,
            strides: tensor.strides,
            data,
        })
    }
}

impl<T: Float, D: Dimension> TryFrom<FloatArcTensor<D>> for ArcTensor<T, D> {
    type Error = anyhow::Error;
    fn try_from(tensor: FloatArcTensor<D>) -> Result<Self> {
        // The ugly transmute is required in order to have a bound on T: Float
        let data = if type_eq::<T, bf16>() {
            match tensor.data.0 {
                FloatArcBuffer::BF16(x) => unsafe { transmute(super::ArcRepr(x)) },
                FloatArcBuffer::F32(_) => {
                    bail!("Expected bf16 found f32!");
                }
            }
        } else if type_eq::<T, f32>() {
            match tensor.data.0 {
                FloatArcBuffer::BF16(_) => {
                    bail!("Expected f32 found bf16!");
                }
                FloatArcBuffer::F32(x) => unsafe { transmute(super::ArcRepr(x)) },
            }
        } else {
            unreachable!()
        };
        Ok(Self {
            device: tensor.device,
            dim: tensor.dim,
            strides: tensor.strides,
            data,
        })
    }
}

pub trait FloatTensorExt: Sized {
    type Dim: Dimension;
    fn scale_into<T2: Num>(self, alpha: T2) -> Result<Tensor<T2, Self::Dim>>;
    fn cast_into<T2: Num>(self) -> Result<Tensor<T2, Self::Dim>> {
        self.scale_into(T2::one())
    }
    fn argmax(&self, axis: Axis) -> Result<Tensor<u32, <Self::Dim as Dimension>::Smaller>>
    where
        Self::Dim: RemoveAxis;
}

impl<S: FloatData, D: Dimension> FloatTensorExt for TensorBase<S, D> {
    type Dim = D;
    fn scale_into<T2: Num>(self, alpha: T2) -> Result<Tensor<T2, D>> {
        match self.data.into_float_buffer()? {
            FloatBuffer::BF16(buffer) => TensorBase {
                device: self.device,
                dim: self.dim,
                strides: self.strides,
                data: super::OwnedRepr(buffer),
            }
            .scale_into(alpha),
            FloatBuffer::F32(buffer) => TensorBase {
                device: self.device,
                dim: self.dim,
                strides: self.strides,
                data: super::OwnedRepr(buffer),
            }
            .scale_into(alpha),
        }
    }
    fn argmax(&self, axis: Axis) -> Result<Tensor<u32, <Self::Dim as Dimension>::Smaller>>
    where
        Self::Dim: RemoveAxis,
    {
        match self.float_type() {
            FloatType::BF16 => {
                let input: TensorView<bf16, Self::Dim> = self.float_view().try_into()?;
                input.argmax(axis)
            }
            FloatType::F32 => {
                let input: TensorView<f32, Self::Dim> = self.float_view().try_into()?;
                input.argmax(axis)
            }
        }
    }
}
