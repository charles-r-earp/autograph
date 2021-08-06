use crate::{
    device::{
        buffer::{ArcBuffer, Buffer, CowBuffer, Slice, SliceMut},
        builders::ComputePassBuilder,
        Device,
    },
    result::Result,
    scalar::Scalar,
};
use half::bf16;
use num_traits::FromPrimitive;
use serde::{Deserialize, Serialize};
use std::mem::transmute;

/// Float types.
#[allow(missing_docs)]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FloatType {
    BF16,
    F32,
}

/// Base trait for float scalar types.
pub trait Float: Scalar {
    /// The float type.
    fn float_type() -> FloatType;
}

impl Float for bf16 {
    fn float_type() -> FloatType {
        FloatType::BF16
    }
}

impl Float for f32 {
    fn float_type() -> FloatType {
        FloatType::BF16
    }
}

/*
macro_rules! float_buffer_methods {
    ($($vis:vis fn $fn:ident $(<$($gen:tt),+>)? (& $($a:lifetime)? $self:ident $(, $arg:ident : $arg_ty:ty)*) $(-> $ret:ty)? $(where $($gen2:ident : $bound:tt),+)? { $this:ident => $body:expr })*) => (
        $(
            fn $fn $(<$($gen),+>)? (& $($a)? $self $(, $arg : $arg_ty)*) $(-> $ret)? $(where $($gen2 : $bound),+)? {
                match $self {
                    Self::BF16($this) => $body,
                    Self::F32($this) => $body,
                }
            }
        )*
    )
}*/

macro_rules! map_float_buffer {
    ($buffer:ident: $t:ident, $x:ident => $e:expr) => (
        match $buffer {
            $t::BF16($x) => $e,
            $t::F32($x) => $e,
        }
    );
    ($self:ident, $x:ident => $e:expr) => (
        map_float_buffer!($self: Self, $x => $e)
    );
}

macro_rules! impl_float_buffer {
    ($(($doc:literal, $float_buffer:ident $(<$a:lifetime>)?, $buffer:ident $(<$b:lifetime>)?, derive($($derive:ident),*) ),)+) => (
        $(
            #[doc = $doc]
            #[allow(missing_docs)]
            #[derive($($derive,)*)]
            pub enum $float_buffer $(<$a>)? {
                BF16($buffer <$($b,)? bf16>),
                F32($buffer <$($b,)? f32>),
            }

            impl<$($a,)? T: Float> From<$buffer <$($b,)? T>> for $float_buffer $(<$a>)? {
                fn from(buffer: $buffer <$($b,)? T>) -> Self {
                    match T::float_type() {
                        FloatType::BF16 => Self::BF16(unsafe { transmute(buffer) }),
                        FloatType::F32 => Self::F32(unsafe { transmute(buffer) }),
                    }
                }
            }

            impl<$($a,)?> $float_buffer $(<$a>)? {
                /// Returns the device.
                pub fn device(&self) -> Device {
                    map_float_buffer!(self, buffer => buffer.device())
                }
                /// The length of the buffer.
                pub fn len(&self) -> usize {
                    map_float_buffer!(self, buffer => buffer.len())
                }
                /// Whether the buffer is empty.
                pub fn is_empty(&self) -> bool {
                    map_float_buffer!(self, buffer => buffer.is_empty())
                }
                /// The [`FloatType`] of the buffer.
                pub fn float_type(&self) -> FloatType {
                    match &self {
                        Self::BF16(_) => FloatType::BF16,
                        Self::F32(_) => FloatType::F32,
                    }
                }
                /// Borrows the buffer as a slice.
                pub fn as_slice(&self) -> FloatSlice {
                    map_float_buffer!(self, buffer => buffer.as_slice().into())
                }
                /// Converts into an owned buffer.
                ///
                /// **Errors**
                /// - Potentially allocates the buffer [`Buffer::alloc`](crate::device::buffer::BufferBase::alloc).
                pub fn into_owned(self) -> Result<FloatBuffer> {
                    map_float_buffer!(self, buffer => Ok(buffer.into_owned()?.into()))
                }
                /// Copies into a new buffer.
                ///
                /// **Errors**
                /// - Allocates the buffer [`Buffer::alloc`](crate::device::buffer::BufferBase::alloc).
                pub fn to_owned(&self) -> Result<FloatBuffer> {
                    map_float_buffer!(self, buffer => Ok(buffer.to_owned()?.into()))
                }
                /// Transfers the buffer into the `device`.
                ///
                /// See [`Buffer::into_device()`](crate::device::buffer::BufferBase::into_device()).
                pub async fn into_device(self, device: Device) -> Result<FloatBuffer>
                {
                    map_float_buffer!(self, buffer => Ok(buffer.into_device(device).await?.into()))
                }
            }


            /// Casts
            impl<$($a,)?> $float_buffer $(<$a>)? {
                /// Casts the buffer into a new buffer.
                ///
                /// See [`BufferBase::cast_into()`].
                #[allow(unused)]
                pub(crate) fn cast_into<T2: Scalar>(self) -> Result<Buffer<T2>> {
                    map_float_buffer!(self, buffer => buffer.cast_into())
                }
                /// Casts the buffer into a new buffer.
                ///
                /// Returns a CowBuffer, where for T -> T, borrows the buffer.
                ///
                /// See [`BufferBase::cast_into()`].
                #[allow(unused)]
                pub(crate) fn cast_to<T2: Scalar>(&self) -> Result<CowBuffer<T2>> {
                    map_float_buffer!(self, buffer => buffer.cast_to())
                }
                /// Scales the buffer into a new buffer.
                ///
                /// See [`BufferBase::scale_into()`].
                #[allow(unused)]
                pub(crate) fn scale_into<T2: Scalar>(self, alpha: T2) -> Result<Buffer<T2>> {
                    map_float_buffer!(self, buffer => buffer.scale_into(alpha))
                }

                #[allow(unused)]
                pub(crate) fn downcast<T: Float>(self) -> Result<$buffer <$($b,)? T>, Self>  {
                    use FloatType::*;
                    match (self, T::float_type()) {
                        (Self::BF16(buffer), BF16)  => Ok(unsafe { transmute(buffer) }),
                        _ => todo!()
                    }
                }
            }
        )+
    );
}

impl_float_buffer! {
    ("A Float Buffer.", FloatBuffer, Buffer, derive(Debug, Serialize, Deserialize)),
    ("A Float Slice.", FloatSlice<'a>, Slice<'a>, derive(Debug, Clone)),
    ("A Float SliceMut.", FloatSliceMut<'a>, SliceMut<'a>, derive(Debug)),
    ("A Float ArcBuffer.", FloatArcBuffer, ArcBuffer, derive(Debug, Clone, Serialize, Deserialize)),
    ("A Float CowBuffer.", FloatCowBuffer<'a>, CowBuffer<'a>, derive(Debug)),
}

macro_rules! impl_float_buffer_owned {
    ($(($float_buffer:ident $(<$a:lifetime>)?, $buffer:ident $(<$b:lifetime>)?),)+) => {
        $(
            impl<$($a,)?> $float_buffer $(<$a>)? {
                /// Allocate a buffer with type `float_type` and length `len`.
                ///
                /// # Safety
                /// The buffer will not be initialized.
                ///
                /// **Errors**
                /// - AllocationTooLarge: Device allocations are limited to 256 MB per Buffer.
                /// - OutOfDeviceMemory: Device memory is exhausted.
                /// - DeviceLost: The device panicked or disconnected.
                ///
                /// # Note
                /// - For constructing a buffer on the host, prefer [`FloatBuffer::from`].
                /// - See [`.zeros()`](FloatBuffer::zeros) for a safe alternative.
                pub unsafe fn alloc(float_type: FloatType, device: Device, len: usize) -> Result<Self> {
                    match float_type {
                        FloatType::BF16 => Ok(Self::BF16($buffer::alloc(device, len)?)),
                        FloatType::F32 => Ok(Self::F32($buffer::alloc(device, len)?)),
                    }
                }
                /// Creates a buffer with type `float_type` and length `len` filled with `elem`.
                ///
                /// **Errors**
                /// - AllocationTooLarge: Device allocations are limited to 256 MB per Buffer.
                /// - OutOfDeviceMemory: Device memory is exhausted.
                /// - DeviceLost: The device panicked or disconnected.
                pub fn from_elem<T>(device: Device, len: usize, elem: T) -> Result<Self>
                    where T: Float {
                    Ok($buffer::from_elem(device, len, elem)?.into())
                }
                /// Creates a buffer with length `len` filled with 0's.
                ///
                /// **Errors**
                /// - AllocationTooLarge: Device allocations are limited to 256 MB per Buffer.
                /// - OutOfDeviceMemory: Device memory is exhausted.
                /// - DeviceLost: The device panicked or disconnected.
                pub fn zeros(float_type: FloatType, device: Device, len: usize) -> Result<Self> {
                    match float_type {
                        FloatType::BF16 => Ok(Self::BF16($buffer::zeros(device, len)?)),
                        FloatType::F32 => Ok(Self::F32($buffer::zeros(device, len)?)),
                    }
                }
            }
        )+
    };
}

impl_float_buffer_owned! {
    (FloatBuffer, Buffer),
    (FloatArcBuffer, ArcBuffer),
    (FloatCowBuffer<'a>, CowBuffer<'a>),
}

impl From<FloatBuffer> for FloatArcBuffer {
    fn from(buffer: FloatBuffer) -> Self {
        map_float_buffer!(buffer: FloatBuffer, buffer => ArcBuffer::from(buffer).into())
    }
}

impl From<FloatBuffer> for FloatCowBuffer<'_> {
    fn from(buffer: FloatBuffer) -> Self {
        map_float_buffer!(buffer: FloatBuffer, buffer => CowBuffer::from(buffer).into())
    }
}

impl<'a> From<FloatSlice<'a>> for FloatCowBuffer<'a> {
    fn from(slice: FloatSlice<'a>) -> Self {
        map_float_buffer!(slice: FloatSlice, slice => CowBuffer::from(slice).into())
    }
}

macro_rules! impl_float_buffer_mut {
    ($(($float_buffer:ident $(<$a:lifetime>)?, $buffer:ident $(<$b:lifetime>)?),)+) => {
        $(
            impl<$($a,)?> $float_buffer $(<$a>)? {
                /// Borrows the buffer as a slice.
                pub fn as_slice_mut(&mut self) -> FloatSliceMut {
                    map_float_buffer!(self, buffer => buffer.as_slice_mut().into())
                }
                /// Fills the buffer with `elem`.
                ///
                /// **Errors**
                /// - Not supported on the host.
                /// - The device panicked or disconnected.
                /// - The operation could not be performed.
                pub fn fill(&mut self, elem: f32) -> Result<()> {
                    map_float_buffer!(self, buffer => buffer.fill(FromPrimitive::from_f32(elem).unwrap()))
                }
            }
        )+
    };
}

impl_float_buffer_mut! {
    (FloatBuffer, Buffer),
    (FloatSliceMut<'a>, SliceMut<'a>),
}

macro_rules! impl_float_buffer_try_mut {
    ($(($float_buffer:ident $(<$a:lifetime>)?, $buffer:ident $(<$b:lifetime>)?),)+) => {
        $(
            impl<$($a,)?> $float_buffer $(<$a>)? {
                #[cfg(feature = "tensor")]
                pub(crate) fn try_unwrap(self) -> Result<FloatBuffer, Self> {
                    map_float_buffer!(self, buffer => buffer.try_unwrap().map(Into::into).map_err(Into::into))
                }
                #[allow(unused)]
                #[cfg(feature = "tensor")]
                pub(crate) fn get_mut(&mut self) -> Option<FloatSliceMut> {
                    map_float_buffer!(self, buffer => buffer.get_mut().map(Into::into))
                }
                #[allow(unused)]
                #[cfg(feature = "tensor")]
                pub(crate) fn make_mut(&mut self) -> Result<FloatSliceMut> {
                    map_float_buffer!(self, buffer => buffer.make_mut().map(Into::into))
                }
            }
        )+
    }
}

impl_float_buffer_try_mut! {
    (FloatArcBuffer, ArcBuffer),
    (FloatCowBuffer<'a>, CowBuffer<'a>),
}

/// Float methods.
///
/// See [`float`](crate::float).
impl<'m, 'b> ComputePassBuilder<'m, 'b> {
    /// Adds a float slice as an argument to the shader at the next binding.
    ///
    /// See [`.slice()`](ComputePassBuilder::slice())
    pub fn float_slice<'b2>(self, slice: FloatSlice<'b2>) -> Result<ComputePassBuilder<'m, 'b2>> {
        map_float_buffer!(slice: FloatSlice, slice => self.slice(slice))
    }
    /// Adds a mutable float slice as an argument to the shader at the next binding.
    ///
    /// See [`.slice_mut()`](ComputePassBuilder::slice_mut())
    pub fn float_slice_mut<'b2>(
        self,
        slice: FloatSliceMut<'b2>,
    ) -> Result<ComputePassBuilder<'m, 'b2>> {
        map_float_buffer!(slice: FloatSliceMut, slice => self.slice_mut(slice))
    }
}
