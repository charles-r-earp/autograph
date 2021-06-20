use super::{
    ArcTensor, Axis, CowTensor, Data, DataMut, DataOwned, Device, Dimension, Dot, Float,
    IntoDimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, Num, RemoveAxis, Result, ShapeBuilder,
    Tensor, TensorBase, TensorView, TensorViewMut,
};
pub use crate::backend::FloatType;
use crate::util::type_eq;
use anyhow::ensure;
use half::bf16;
use num_traits::FromPrimitive;
use serde::{Deserialize, Serialize};
use std::{future::Future, marker::PhantomData, mem::transmute};

mod sealed {
    use super::{Data, DataMut, DataOwned, Float};

    pub trait FloatDataBase<T: Float> {
        type Data: Data<Elem = T>;
    }

    pub trait FloatDataBaseOwned<T: Float> {
        type Data: DataOwned<Elem = T>;
    }

    pub trait FloatDataBaseMut<T: Float>: FloatDataBase<T> {
        type Data: DataMut<Elem = T>;
    }
}
use sealed::{FloatDataBase, FloatDataBaseMut, FloatDataBaseOwned};

pub trait FloatData: FloatDataBase<bf16> + FloatDataBase<f32> {}

pub trait FloatDataOwned:
    FloatData
    + FloatDataBase<bf16, Data = <Self as FloatDataBaseOwned<bf16>>::Data>
    + FloatDataBase<f32, Data = <Self as FloatDataBaseOwned<f32>>::Data>
    + FloatDataBaseOwned<bf16>
    + FloatDataBaseOwned<f32>
{
}

pub trait FloatDataMut:
    FloatData
    + FloatDataBase<bf16, Data = <Self as FloatDataBaseMut<bf16>>::Data>
    + FloatDataBase<f32, Data = <Self as FloatDataBaseMut<f32>>::Data>
    + FloatDataBaseMut<bf16>
    + FloatDataBaseMut<f32>
{
}

pub struct FloatOwnedRepr {}

impl<T: Float> FloatDataBase<T> for FloatOwnedRepr {
    type Data = super::OwnedRepr<T>;
}

impl FloatData for FloatOwnedRepr {}

impl<T: Float> FloatDataBaseOwned<T> for FloatOwnedRepr {
    type Data = super::OwnedRepr<T>;
}

impl FloatDataOwned for FloatOwnedRepr {}

impl<T: Float> FloatDataBaseMut<T> for FloatOwnedRepr {
    type Data = super::OwnedRepr<T>;
}

impl FloatDataMut for FloatOwnedRepr {}

pub struct FloatArcRepr {}

impl<T: Float> FloatDataBase<T> for FloatArcRepr {
    type Data = super::ArcRepr<T>;
}

impl FloatData for FloatArcRepr {}

impl<T: Float> FloatDataBaseOwned<T> for FloatArcRepr {
    type Data = super::ArcRepr<T>;
}

impl FloatDataOwned for FloatArcRepr {}

pub struct FloatViewRepr<'a>(PhantomData<&'a ()>);

impl<'a, T: Float> FloatDataBase<T> for FloatViewRepr<'a> {
    type Data = super::ViewRepr<'a, T>;
}

impl FloatData for FloatViewRepr<'_> {}

pub struct FloatViewMutRepr<'a>(PhantomData<&'a mut ()>);

impl<'a, T: Float> FloatDataBase<T> for FloatViewMutRepr<'a> {
    type Data = super::ViewMutRepr<'a, T>;
}

impl FloatData for FloatViewMutRepr<'_> {}

impl<'a, T: Float> FloatDataBaseMut<T> for FloatViewMutRepr<'a> {
    type Data = super::ViewMutRepr<'a, T>;
}

impl FloatDataMut for FloatViewMutRepr<'_> {}

pub struct FloatCowRepr<'a>(PhantomData<&'a ()>);

impl<'a, T: Float> FloatDataBase<T> for FloatCowRepr<'a> {
    type Data = super::CowRepr<'a, T>;
}

impl FloatData for FloatCowRepr<'_> {}

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "
        <S as FloatDataBase<bf16>>::Data: Serialize,
        <S as FloatDataBase<f32>>::Data: Serialize,
        D: Dimension + Serialize
    ",
    deserialize = "
        <S as FloatDataBase<bf16>>::Data: Deserialize<'de>,
        <S as FloatDataBase<f32>>::Data: Deserialize<'de>,
        D: Dimension + Deserialize<'de>
    "
))]
#[non_exhaustive]
pub enum FloatTensorBase<S: FloatData, D: Dimension> {
    BF16(TensorBase<<S as FloatDataBase<bf16>>::Data, D>),
    F32(TensorBase<<S as FloatDataBase<f32>>::Data, D>),
}

pub type FloatTensor<D> = FloatTensorBase<FloatOwnedRepr, D>;
pub type FloatTensor0 = FloatTensor<Ix0>;
pub type FloatTensor1 = FloatTensor<Ix1>;
pub type FloatTensor2 = FloatTensor<Ix2>;
pub type FloatTensor3 = FloatTensor<Ix3>;
pub type FloatTensor4 = FloatTensor<Ix4>;
pub type FloatTensor5 = FloatTensor<Ix5>;
pub type FloatTensor6 = FloatTensor<Ix6>;
pub type FloatTensorD = FloatTensor<IxDyn>;

pub type FloatArcTensor<D> = FloatTensorBase<FloatArcRepr, D>;
pub type FloatArcTensor0 = FloatArcTensor<Ix0>;
pub type FloatArcTensor1 = FloatArcTensor<Ix1>;
pub type FloatArcTensor2 = FloatArcTensor<Ix2>;
pub type FloatArcTensor3 = FloatArcTensor<Ix3>;
pub type FloatArcTensor4 = FloatArcTensor<Ix4>;
pub type FloatArcTensor5 = FloatArcTensor<Ix5>;
pub type FloatArcTensor6 = FloatArcTensor<Ix6>;
pub type FloatArcTensorD = FloatArcTensor<IxDyn>;

pub type FloatTensorView<'a, D> = FloatTensorBase<FloatViewRepr<'a>, D>;
pub type FloatTensorView0<'a> = FloatTensorView<'a, Ix0>;
pub type FloatTensorView1<'a> = FloatTensorView<'a, Ix1>;
pub type FloatTensorView2<'a> = FloatTensorView<'a, Ix2>;
pub type FloatTensorView3<'a> = FloatTensorView<'a, Ix3>;
pub type FloatTensorView4<'a> = FloatTensorView<'a, Ix4>;
pub type FloatTensorView5<'a> = FloatTensorView<'a, Ix5>;
pub type FloatTensorView6<'a> = FloatTensorView<'a, Ix6>;
pub type FloatTensorViewD<'a> = FloatTensorView<'a, IxDyn>;

pub type FloatTensorViewMut<'a, D> = FloatTensorBase<FloatViewMutRepr<'a>, D>;
pub type FloatTensorViewMut0<'a> = FloatTensorView<'a, Ix0>;
pub type FloatTensorViewMut1<'a> = FloatTensorViewMut<'a, Ix1>;
pub type FloatTensorViewMut2<'a> = FloatTensorViewMut<'a, Ix2>;
pub type FloatTensorViewMut3<'a> = FloatTensorViewMut<'a, Ix3>;
pub type FloatTensorViewMut4<'a> = FloatTensorViewMut<'a, Ix4>;
pub type FloatTensorViewMut5<'a> = FloatTensorViewMut<'a, Ix5>;
pub type FloatTensorViewMut6<'a> = FloatTensorViewMut<'a, Ix6>;
pub type FloatTensorViewMutD<'a> = FloatTensorViewMut<'a, IxDyn>;

pub type FloatCowTensor<'a, D> = FloatTensorBase<FloatCowRepr<'a>, D>;
pub type FloatCowTensor0<'a> = FloatCowTensor<'a, Ix0>;
pub type FloatCowTensor1<'a> = FloatCowTensor<'a, Ix1>;
pub type FloatCowTensor2<'a> = FloatCowTensor<'a, Ix2>;
pub type FloatCowTensor3<'a> = FloatCowTensor<'a, Ix3>;
pub type FloatCowTensor4<'a> = FloatCowTensor<'a, Ix4>;
pub type FloatCowTensor5<'a> = FloatCowTensor<'a, Ix5>;
pub type FloatCowTensor6<'a> = FloatCowTensor<'a, Ix6>;
pub type FloatCowTensorD<'a> = FloatCowTensor<'a, IxDyn>;

macro_rules! float_tensor_impl {
    ($this:ident, $i:ident => $e:expr) => {{
        match $this {
            FloatTensorBase::BF16($i) => $e,
            FloatTensorBase::F32($i) => $e,
        }
    }};
    ($this:ident, map $i:ident => $e:expr) => {{
        match $this {
            FloatTensorBase::BF16($i) => FloatTensorBase::BF16($e),
            FloatTensorBase::F32($i) => FloatTensorBase::F32($e),
        }
    }};
    ($this:ident, map_ok $i:ident => $e:expr) => {{
        match $this {
            FloatTensorBase::BF16($i) => Ok(FloatTensorBase::BF16($e?)),
            FloatTensorBase::F32($i) => Ok(FloatTensorBase::F32($e?)),
        }
    }};
}

impl<S: FloatDataOwned, D: Dimension> FloatTensorBase<S, D> {
    /// Creates a new Tensor with the given shape\
    ///
    /// # Safety
    ///
    /// The Tensor is uninitialized.
    pub unsafe fn uninitialized<Sh>(
        device: &Device,
        float_type: FloatType,
        shape: Sh,
    ) -> Result<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        match float_type {
            FloatType::BF16 => Ok(Self::BF16(TensorBase::uninitialized(device, shape)?)),
            FloatType::F32 => Ok(Self::F32(TensorBase::uninitialized(device, shape)?)),
        }
    }
    pub fn from_elem<Sh>(
        device: &Device,
        float_type: FloatType,
        shape: Sh,
        elem: f32,
    ) -> Result<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        match float_type {
            FloatType::BF16 => Ok(Self::BF16(TensorBase::from_elem(
                device,
                shape,
                bf16::from_f32(elem),
            )?)),
            FloatType::F32 => Ok(Self::F32(TensorBase::from_elem(device, shape, elem)?)),
        }
    }
    pub fn zeros<Sh>(device: &Device, float_type: FloatType, shape: Sh) -> Result<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_elem(device, float_type, shape, 0f32)
    }
    pub fn ones<Sh>(device: &Device, float_type: FloatType, shape: Sh) -> Result<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_elem(device, float_type, shape, 1f32)
    }
    #[allow(clippy::wrong_self_convention)]
    pub fn to_device_mut<'a>(
        &'a mut self,
        device: &Device,
    ) -> Result<impl Future<Output = Result<()>> + 'a> {
        let device = device.clone();
        Ok(async move {
            if self.device() != &device {
                float_tensor_impl!(self, x => x.to_device_mut(&device)?.await?);
            }
            Ok(())
        })
    }
}

impl<S: FloatData, D: Dimension> FloatTensorBase<S, D> {
    pub fn device(&self) -> &Device {
        float_tensor_impl!(self, x => x.device())
    }
    pub fn float_type(&self) -> FloatType {
        match self {
            Self::BF16(_) => FloatType::BF16,
            Self::F32(_) => FloatType::F32,
        }
    }
    pub fn dim(&self) -> D::Pattern {
        float_tensor_impl!(self, x => x.dim())
    }
    pub fn raw_dim(&self) -> D {
        float_tensor_impl!(self, x => x.raw_dim())
    }
    pub fn shape(&self) -> &[usize] {
        float_tensor_impl!(self, x => x.shape())
    }
    pub fn strides(&self) -> &[isize] {
        float_tensor_impl!(self, x => x.strides())
    }
    pub fn len(&self) -> usize {
        float_tensor_impl!(self, x => x.len())
    }
    pub fn is_empty(&self) -> bool {
        float_tensor_impl!(self, x => x.is_empty())
    }
    pub fn ndim(&self) -> usize {
        float_tensor_impl!(self, x => x.ndim())
    }
    pub fn into_dimensionality<D2>(self) -> Result<FloatTensorBase<S, D2>>
    where
        D2: Dimension,
    {
        float_tensor_impl!(self, map_ok x => x.into_dimensionality::<D2>())
    }
    pub fn into_shape<E>(self, shape: E) -> Result<FloatTensorBase<S, E::Dim>>
    where
        E: IntoDimension,
    {
        float_tensor_impl!(self, map_ok x => x.into_shape(shape))
    }
    pub fn into_dyn(self) -> FloatTensorBase<S, IxDyn> {
        float_tensor_impl!(self, map x => x.into_dyn())
    }
    pub fn view(&self) -> FloatTensorView<D> {
        float_tensor_impl!(self, map x => x.view())
    }
    pub fn reversed_axes(self) -> Self {
        float_tensor_impl!(self, map x => x.reversed_axes())
    }
    pub fn t(&self) -> FloatTensorView<D> {
        self.view().reversed_axes()
    }
    pub fn into_device(
        self,
        device: &Device,
    ) -> Result<impl Future<Output = Result<FloatTensor<D>>>> {
        let device = device.clone();
        Ok(async move { float_tensor_impl!(self, map_ok x => x.into_device(&device)?.await) })
    }
    pub fn into_device_arc(
        self,
        device: &Device,
    ) -> Result<impl Future<Output = Result<FloatArcTensor<D>>>> {
        let device = device.clone();
        Ok(
            async move { float_tensor_impl!(self, map_ok x => x.into_device_shared(&device)?.await) },
        )
    }
    pub fn to_device<'a>(
        &'a self,
        device: &Device,
    ) -> Result<impl Future<Output = Result<FloatCowTensor<'a, D>>> + 'a> {
        let device = device.clone();
        Ok(async move { float_tensor_impl!(self, map_ok x => x.to_device(&device)?.await) })
    }
}

impl<S: FloatData, D: Dimension> FloatTensorBase<S, D> {
    pub fn into_owned(self) -> Result<FloatTensor<D>> {
        float_tensor_impl!(self, map_ok x => x.into_owned())
    }
    pub fn into_shared(self) -> Result<FloatArcTensor<D>> {
        float_tensor_impl!(self, map_ok x => x.into_shared())
    }
}

impl<T: Float, S: Data<Elem = T>, D: Dimension> TensorBase<S, D> {
    pub fn into_float_owned(self) -> Result<FloatTensor<D>> {
        Ok(self.into_owned()?.into())
    }
    pub fn into_float_shared(self) -> Result<FloatArcTensor<D>> {
        Ok(self.into_shared()?.into())
    }
}

impl<S: FloatDataMut, D: Dimension> FloatTensorBase<S, D> {
    pub fn view_mut(&mut self) -> FloatTensorViewMut<D> {
        float_tensor_impl!(self, x => x.view_mut().into())
    }
}

impl<D: Dimension> FloatArcTensor<D> {
    pub fn make_shared_mut(&mut self) -> Result<FloatTensorViewMut<D>> {
        float_tensor_impl!(self, map_ok x => x.make_shared_mut())
    }
}

impl<D: Dimension> Clone for FloatArcTensor<D> {
    fn clone(&self) -> Self {
        float_tensor_impl!(self, map x => x.clone())
    }
}

impl<T: Float, D: Dimension> From<Tensor<T, D>> for FloatTensor<D> {
    fn from(tensor: Tensor<T, D>) -> Self {
        let Tensor {
            device,
            dim,
            strides,
            data,
        } = tensor;
        if type_eq::<T, bf16>() {
            Self::BF16(Tensor {
                device,
                dim,
                strides,
                data: unsafe { transmute(data) },
            })
        } else if type_eq::<T, f32>() {
            Self::F32(Tensor {
                device,
                dim,
                strides,
                data: unsafe { transmute(data) },
            })
        } else {
            unreachable!()
        }
    }
}

impl<T: Float, D: Dimension> From<ArcTensor<T, D>> for FloatArcTensor<D> {
    fn from(tensor: ArcTensor<T, D>) -> Self {
        let ArcTensor {
            device,
            dim,
            strides,
            data,
        } = tensor;
        if type_eq::<T, bf16>() {
            Self::BF16(ArcTensor {
                device,
                dim,
                strides,
                data: unsafe { transmute(data) },
            })
        } else if type_eq::<T, f32>() {
            Self::F32(ArcTensor {
                device,
                dim,
                strides,
                data: unsafe { transmute(data) },
            })
        } else {
            unreachable!()
        }
    }
}

impl<D: Dimension> From<FloatTensor<D>> for FloatArcTensor<D> {
    fn from(tensor: FloatTensor<D>) -> Self {
        float_tensor_impl!(tensor, map x => x.into())
    }
}

impl<'a, T: Float, D: Dimension> From<TensorView<'a, T, D>> for FloatTensorView<'a, D> {
    fn from(tensor: TensorView<'a, T, D>) -> Self {
        let TensorView {
            device,
            dim,
            strides,
            data,
        } = tensor;
        if type_eq::<T, bf16>() {
            Self::BF16(TensorView {
                device,
                dim,
                strides,
                data: unsafe { transmute(data) },
            })
        } else if type_eq::<T, f32>() {
            Self::F32(TensorView {
                device,
                dim,
                strides,
                data: unsafe { transmute(data) },
            })
        } else {
            unreachable!()
        }
    }
}

impl<'a, T: Float, D: Dimension> From<TensorViewMut<'a, T, D>> for FloatTensorViewMut<'a, D> {
    fn from(tensor: TensorViewMut<'a, T, D>) -> Self {
        let TensorViewMut {
            device,
            dim,
            strides,
            data,
        } = tensor;
        if type_eq::<T, bf16>() {
            Self::BF16(TensorViewMut {
                device,
                dim,
                strides,
                data: unsafe { transmute(data) },
            })
        } else if type_eq::<T, f32>() {
            Self::F32(TensorViewMut {
                device,
                dim,
                strides,
                data: unsafe { transmute(data) },
            })
        } else {
            unreachable!()
        }
    }
}

impl<S: FloatData, D: Dimension> FloatTensorBase<S, D> {
    pub fn cast_into<T: Num>(self) -> Result<Tensor<T, D>> {
        float_tensor_impl!(self, x => x.cast_into())
    }
    pub fn cast_into_float(self, float_type: FloatType) -> Result<FloatTensor<D>> {
        match float_type {
            FloatType::BF16 => Ok(FloatTensor::BF16(self.cast_into()?)),
            FloatType::F32 => Ok(FloatTensor::F32(self.cast_into()?)),
        }
    }
    pub fn cast_to<T: Num>(&self) -> Result<CowTensor<T, D>> {
        float_tensor_impl!(self, x => x.cast_to())
    }
    pub fn cast_to_float(&self, float_type: FloatType) -> Result<FloatCowTensor<D>> {
        match float_type {
            FloatType::BF16 => Ok(FloatCowTensor::BF16(self.cast_to()?)),
            FloatType::F32 => Ok(FloatCowTensor::F32(self.cast_to()?)),
        }
    }
}

impl<S: FloatData, D: RemoveAxis> FloatTensorBase<S, D> {
    pub fn argmin(&self, axis: Axis) -> Result<Tensor<u32, D::Smaller>> {
        float_tensor_impl!(self, x => x.argmin(axis))
    }
    pub fn argmax(&self, axis: Axis) -> Result<Tensor<u32, D::Smaller>> {
        float_tensor_impl!(self, x => x.argmax(axis))
    }
}

// TODO: docs
pub fn float_gemm(
    alpha: f32,
    a: &FloatTensorView2,
    b: &FloatTensorView2,
    beta: f32,
    c: &mut FloatTensorViewMut2,
) -> Result<()> {
    float_tensor_impl!(c, c => Ok(super::linalg::gemm(
        FromPrimitive::from_f32(alpha).unwrap(),
        &a.cast_to()?.view(),
        &b.cast_to()?.view(),
        FromPrimitive::from_f32(beta).unwrap(),
        c
    )?))
}

pub fn float_gemm_bias(
    alpha: f32,
    a: &FloatTensorView2,
    b: &FloatTensorView2,
    bias: Option<&FloatTensorView1>,
    beta: f32,
    c: &mut FloatTensorViewMut2,
) -> Result<()> {
    float_tensor_impl!(c, c => {
        let bias = if let Some(bias) = bias {
            Some(bias.cast_to()?)
        } else {
            None
        };
        let bias = bias.as_ref().map(|bias| bias.view());
        Ok(
            super::linalg::gemm_bias(
                FromPrimitive::from_f32(alpha).unwrap(),
                &a.cast_to()?.view(),
                &b.cast_to()?.view(),
                FromPrimitive::from_f32(beta).unwrap(),
                bias.as_ref(),
                c
            )?
        )
    })
}

impl<S1: FloatData, S2: FloatData> Dot<FloatTensorBase<S2, Ix2>> for FloatTensorBase<S1, Ix2> {
    type Output = FloatTensor2;
    fn dot(&self, rhs: &FloatTensorBase<S2, Ix2>) -> Result<FloatTensor2> {
        let (m, k) = self.dim();
        let (k2, n) = rhs.dim();
        ensure!(k == k2, "{:?} x {:?}", self.shape(), rhs.shape());
        let mut output =
            unsafe { FloatTensor::uninitialized(self.device(), self.float_type(), [m, n])? };
        float_gemm(1., &self.view(), &rhs.view(), 0., &mut output.view_mut())?;
        Ok(output)
    }
}

impl<S: FloatData, D: Dimension> FloatTensorBase<S, D> {
    pub fn dot<R>(&self, rhs: &R) -> Result<<Self as Dot<R>>::Output>
    where
        Self: Dot<R>,
    {
        Dot::dot(self, rhs)
    }
}

impl<S: FloatData> FloatTensorBase<S, Ix2> {
    pub(crate) fn mm_bias(
        &self,
        rhs: &FloatTensorView2,
        bias: Option<&FloatTensorView1>,
    ) -> Result<FloatTensor2> {
        let (m, k) = self.dim();
        let (k2, n) = rhs.dim();
        ensure!(k == k2, "{:?} x {:?}", self.shape(), rhs.shape());
        let mut output =
            unsafe { FloatTensor::uninitialized(self.device(), self.float_type(), [m, n])? };
        float_gemm_bias(1., &self.view(), rhs, bias, 0., &mut output.view_mut())?;
        Ok(output)
    }
}
