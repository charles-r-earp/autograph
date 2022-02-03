//!
//! # Compute Example
//! This example shows the basics of creating buffers, executing compute, and reading back the results.
/*!```no_run
use autograph::{
    result::Result, device::Device, shader::Module, buffer::{Buffer, Slice},
 };
 #[tokio::main]
 async fn main() -> Result<()> {
     // The spirv source can be created at runtime and imported via include_bytes! or compiled
     // at runtime (JIT).
     let spirv: Vec<u8> = todo!();
     // The module stores the spirv and does reflection on it to extract all of the entry
     // functions and their arguments. Module can be serialized and deserialized with serde so
     // it can be created at compile time and loaded at runtime as well.
     let module = Module::from_spirv(spirv)?;
     // Create a device.
     let device = Device::new()?;
     // Construct a Buffer from a vec and transfer it to the device.
     // Note that this actually copies into a "staging buffer", Host -> Device transfers do not
     // block. Instead, the device will execute the copy from the staging buffer to device
     // memory lazily, in a batch of operations, when it is ready.
     let a = Buffer::from(vec![1, 2, 3, 4]).into_device(device.clone()).await?;
     // Slice can be created from a &[T] and transfered into a device buffer.
     let b = Slice::from([1, 2, 3, 4].as_ref()).into_device(device).await?;
     // Allocate the result on the device. This is unsafe because it is not initialized.
     // Safe alternative: Buffer::zeros().
     let mut y = unsafe { Buffer::<u32>::alloc(device, a.len())? };
     let n = y.len() as u32;
     // Enqueue the compute pass
     let builder = module
        // entry "add"
        .compute_pass("add")?
        // buffer at binding = 0
        .slice(a.as_slice())?
        // buffer at binding = 1
        .slice(b.as_slice())?
        // buffer at binding = 2
        .slice_mut(y.as_slice_mut())?
        // push constant for the work size.
        // Can be chained or passed as a struct.
        .push(n)?;
     // Executing compute shaders is unsafe, it's like a foreign function call.
     unsafe { builder.submit([n, 1, 1])?; }
     // Read the data back. This will wait for all previous operations to finish.
     let output = y.read().await?;
     println!("{:?}", output.as_slice());
     Ok(())
}
```*/
use crate::{device::Device, result::Result, scalar::Scalar, util::elem_type_name};
use bytemuck::Pod;
use serde::{de::Deserializer, ser::Serializer, Deserialize, Serialize};
use std::{
    fmt::{self, Debug},
    mem::{size_of, transmute},
    sync::Arc,
};

#[doc(inline)]
pub use crate::device::buffer::{
    Buffer, BufferBase, Data, DataMut, OwnedRepr, ReadGuard, Slice, SliceMut, SliceMutRepr,
    SliceRepr,
};

/// Float buffers.
pub mod float;

impl<T, S: Data<Elem = T>> BufferBase<S> {
    /// Fills the buffer with `elem`.
    ///
    /// **Errors**
    /// - Not supported on the host.
    /// - The device panicked or disconnected.
    /// - The operation could not be performed.
    pub fn fill(&mut self, elem: T) -> Result<()>
    where
        T: Scalar,
        S: DataMut,
    {
        let n = self.len() as u32;
        let size = size_of::<T>();
        let core = crate::rust_shaders::core()?;
        let (builder, gs) = match size {
            1 => {
                let builder = core.compute_pass("fill::fill_u32")?;
                let n = if n % 4 == 0 { n / 4 } else { n / 4 + 1 };
                let builder = builder.push(n)?.push([elem; 4])?;
                (builder, n)
            }
            2 => {
                let builder = core.compute_pass("fill::fill_u32")?;
                let n = if n % 2 == 0 { n / 2 } else { n / 2 + 1 };
                let builder = builder.push(n)?.push([elem; 2])?;
                (builder, n)
            }
            4 if n % 2 == 0 => {
                let builder = core.compute_pass("fill::fill_u32x2")?;
                let n = n / 2;
                let builder = builder.push(n)?.push([elem; 2])?;
                (builder, n)
            }
            4 => {
                let builder = core.compute_pass("fill::fill_u32")?;
                let builder = builder.push(n)?.push(elem)?;
                (builder, n)
            }
            8 => {
                let builder = core.compute_pass("fill::fill_u32x2")?;
                let builder = builder.push(n)?.push(elem)?;
                (builder, n)
            }
            _ => unreachable!(),
        };
        let builder = builder.slice_mut(self.as_slice_mut())?;
        unsafe { builder.submit([gs, 1, 1]) }
    }
}

/// Shared Buffer
#[derive(Clone)]
pub struct ArcBuffer<T> {
    buffer: Arc<Buffer<T>>,
    offset: usize,
    len: usize,
}

impl<T> ArcBuffer<T> {
    unsafe fn from_raw_parts(buffer: Arc<Buffer<T>>, offset: usize, len: usize) -> Self {
        debug_assert!(
            offset + len <= buffer.len(),
            "{} > {}",
            offset + len,
            buffer.len()
        );
        Self {
            buffer,
            offset,
            len,
        }
    }
    /// Allocate a buffer with length `len`.
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
    /// - For constructing a buffer on the host, prefer [`Buffer::from`].
    /// - See [`.zeros()`](BufferBase::zeros) for a safe alternative.
    pub unsafe fn alloc(device: Device, len: usize) -> Result<Self>
    where
        T: Default + Copy,
    {
        Ok(unsafe { Buffer::alloc(device, len)?.into() })
    }
    /// Creates a buffer with length `len` filled with `elem`.
    ///
    /// **Errors**
    /// - AllocationTooLarge: Device allocations are limited to 256 MB per Buffer.
    /// - OutOfDeviceMemory: Device memory is exhausted.
    /// - DeviceLost: The device panicked or disconnected.
    pub fn from_elem(device: Device, len: usize, elem: T) -> Result<Self>
    where
        T: Scalar,
    {
        Ok(Buffer::from_elem(device, len, elem)?.into())
    }
    /// Creates a buffer with length `len` filled with 0's.
    ///
    /// **Errors**
    /// - AllocationTooLarge: Device allocations are limited to 256 MB per Buffer.
    /// - OutOfDeviceMemory: Device memory is exhausted.
    /// - DeviceLost: The device panicked or disconnected.
    pub fn zeros(device: Device, len: usize) -> Result<Self>
    where
        T: Scalar,
    {
        Ok(Buffer::zeros(device, len)?.into())
    }
    /// Borrows the buffer as a Slice.
    pub fn as_slice(&self) -> Slice<T> {
        self.buffer.slice_impl(self.offset, self.len)
    }
    pub(crate) fn try_unwrap(self) -> Result<Buffer<T>, Self> {
        if self.offset == 0 && self.len == self.buffer.len() {
            match Arc::try_unwrap(self.buffer) {
                Ok(buffer) => Ok(buffer),
                Err(buffer) => Err(Self {
                    buffer,
                    offset: self.offset,
                    len: self.len,
                }),
            }
        } else {
            Err(self)
        }
    }
    #[allow(unused)]
    pub(crate) fn get_mut(&mut self) -> Option<SliceMut<T>> {
        if self.offset == 0 && self.len == self.buffer.len() {
            Arc::get_mut(&mut self.buffer).map(Buffer::as_slice_mut)
        } else {
            None
        }
    }
    #[allow(unused)]
    pub(crate) fn make_mut(&mut self) -> Result<SliceMut<T>>
    where
        T: Copy,
    {
        if let Some(slice) = self.get_mut() {
            return Ok(unsafe { transmute(slice) });
        }
        let buffer = self.to_owned()?;
        self.buffer = Arc::new(buffer);
        let buffer = unsafe { &mut *(Arc::as_ptr(&self.buffer) as *mut Buffer<T>) };
        Ok(unsafe { transmute(buffer.as_slice_mut()) })
    }
    /// Converts into a [`Buffer`].
    ///
    /// **Errors**
    /// - Potentially allocates the buffer [`Buffer::alloc`](BufferBase::alloc).
    pub fn into_owned(self) -> Result<Buffer<T>>
    where
        T: Copy,
    {
        match self.try_unwrap() {
            Ok(buffer) => Ok(buffer),
            Err(buffer) => buffer.to_owned(),
        }
    }
    /// Copies into a new [`Buffer`].
    ///
    /// **Errors**
    /// - Allocates the buffer [`Buffer::alloc`](BufferBase::alloc).
    pub fn to_owned(&self) -> Result<Buffer<T>>
    where
        T: Copy,
    {
        self.as_slice().to_owned()
    }
    /// Transfers the buffer into the `device`.
    ///
    /// **Errors**
    ///
    /// See [`Buffer::into_device()`](BufferBase::into_device()).
    pub async fn into_device(self, device: Device) -> Result<Buffer<T>>
    where
        T: Pod,
    {
        if self.device() == device {
            self.into_owned()
        } else {
            self.as_slice().into_device(device).await
        }
    }
    /// Transfers the buffer into the `device` as an [`ArcBuffer`].
    ///
    /// NOOP if the buffer is on `device`.
    ///
    /// **Errors**
    ///
    /// See [`Buffer::into_device()`](BufferBase::into_device()).
    pub async fn into_device_shared(self, device: Device) -> Result<Self>
    where
        T: Pod,
    {
        if self.device() == device {
            Ok(self)
        } else {
            Ok(self.into_device(device).await?.into())
        }
    }
    /// The device of the buffer.
    pub fn device(&self) -> Device {
        self.buffer.device()
    }
    /// The length of the buffer.
    pub fn len(&self) -> usize {
        self.len
    }
    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> From<Buffer<T>> for ArcBuffer<T> {
    fn from(buffer: Buffer<T>) -> Self {
        let len = buffer.len();
        unsafe { Self::from_raw_parts(buffer.into(), 0, len) }
    }
}

impl<T: Pod + Serialize> Serialize for ArcBuffer<T> {
    fn serialize<Se>(&self, serializer: Se) -> Result<Se::Ok, Se::Error>
    where
        Se: Serializer,
    {
        self.as_slice().serialize(serializer)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for ArcBuffer<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Buffer::deserialize(deserializer)?.into())
    }
}

impl<T> Debug for ArcBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ArcBuffer")
            .field("device", &self.buffer.device())
            .field("len", &self.len)
            .field("elem", &elem_type_name::<T>())
            .finish()
    }
}

/// Casts
impl<T: Scalar> ArcBuffer<T> {
    /// Casts the buffer into a new buffer.
    ///
    /// # Note
    /// NOOP for Buffer<T> -> Buffer<T>.
    ///
    /// See [`BufferBase::cast_into()`].
    pub fn cast_into<T2: Scalar>(self) -> Result<Buffer<T2>> {
        match self.try_unwrap() {
            Ok(buffer) => buffer.cast_into(),
            Err(this) => this.as_slice().cast_into(),
        }
    }
    /// Casts the buffer into a new buffer.
    ///
    /// Returns a CowBuffer, where for T -> T, borrows the buffer.
    ///
    /// See [`BufferBase::cast_into()`].
    pub fn cast_to<T2: Scalar>(&self) -> Result<CowBuffer<T2>> {
        Ok(self.as_slice().cast_into()?.into())
    }
    /// Scales the buffer into a new buffer.
    ///
    /// # Note
    /// NOOP for Buffer<T> -> Buffer<T> where alpha == 1.
    ///
    /// See [`BufferBase::scale_into()`].
    pub fn scale_into<T2: Scalar>(self, alpha: T2) -> Result<Buffer<T2>> {
        self.as_slice().scale_into(alpha)
    }
}

/// A Borrowed or Owned Buffer
#[derive(Debug)]
pub enum CowBuffer<'a, T> {
    /// Borrowed
    Slice(Slice<'a, T>),
    /// Owned
    Buffer(Buffer<T>),
}

impl<T> CowBuffer<'_, T> {
    /// Allocate a buffer with length `len`.
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
    /// - For constructing a buffer on the host, prefer [`Buffer::from`].
    /// - See [`.zeros()`](BufferBase::zeros) for a safe alternative.
    pub unsafe fn alloc(device: Device, len: usize) -> Result<Self>
    where
        T: Default + Copy,
    {
        Ok(unsafe { Buffer::alloc(device, len)?.into() })
    }
    /// Creates a buffer with length `len` filled with `elem`.
    ///
    /// **Errors**
    /// - AllocationTooLarge: Device allocations are limited to 256 MB per Buffer.
    /// - OutOfDeviceMemory: Device memory is exhausted.
    /// - DeviceLost: The device panicked or disconnected.
    pub fn from_elem(device: Device, len: usize, elem: T) -> Result<Self>
    where
        T: Scalar,
    {
        Ok(Buffer::from_elem(device, len, elem)?.into())
    }
    /// Creates a buffer with length `len` filled with 0's.
    ///
    /// **Errors**
    /// - AllocationTooLarge: Device allocations are limited to 256 MB per Buffer.
    /// - OutOfDeviceMemory: Device memory is exhausted.
    /// - DeviceLost: The device panicked or disconnected.
    pub fn zeros(device: Device, len: usize) -> Result<Self>
    where
        T: Scalar,
    {
        Ok(Buffer::zeros(device, len)?.into())
    }
    /// Borrows the buffer as a Slice.
    pub fn as_slice(&self) -> Slice<T> {
        match self {
            Self::Slice(slice) => slice.as_slice(),
            Self::Buffer(buffer) => buffer.as_slice(),
        }
    }
    pub(crate) fn try_unwrap(self) -> Result<Buffer<T>, Self> {
        match self {
            Self::Buffer(buffer) => Ok(buffer),
            slice => Err(slice),
        }
    }
    #[allow(unused)]
    pub(crate) fn get_mut(&mut self) -> Option<SliceMut<T>> {
        match self {
            Self::Slice(_) => None,
            Self::Buffer(buffer) => Some(buffer.as_slice_mut()),
        }
    }
    #[allow(unused)]
    pub(crate) fn make_mut(&mut self) -> Result<SliceMut<T>>
    where
        T: Copy,
    {
        match self {
            Self::Slice(slice) => {
                let buffer = slice.to_owned()?;
                *self = Self::Buffer(buffer);
                match self {
                    Self::Slice(_) => unreachable!(),
                    Self::Buffer(buffer) => Ok(buffer.as_slice_mut()),
                }
            }
            Self::Buffer(buffer) => Ok(buffer.as_slice_mut()),
        }
    }
    /// Transfers the buffer into the `device`.
    ///
    /// **Errors**
    /// See [`Buffer::into_device()`](BufferBase::into_device()).
    pub async fn into_device(self, device: Device) -> Result<Buffer<T>>
    where
        T: Pod,
    {
        if self.device() == device {
            self.into_owned()
        } else {
            self.as_slice().into_device(device).await
        }
    }
    /// Transfers the buffer into the `device` as an [`ArcBuffer`].
    ///
    /// **Errors**
    ///
    /// See [`Buffer::into_device()`](BufferBase::into_device()).
    pub async fn into_device_shared(self, device: Device) -> Result<ArcBuffer<T>>
    where
        T: Pod,
    {
        if self.device() == device {
            self.into_shared()
        } else {
            Ok(self.into_device(device).await?.into())
        }
    }
    /// Converts into a [`Buffer`].
    ///
    /// **Errors**
    /// - Potentially allocates the buffer [`Buffer::alloc`](BufferBase::alloc).
    pub fn into_owned(self) -> Result<Buffer<T>>
    where
        T: Copy,
    {
        match self {
            Self::Slice(slice) => slice.to_owned(),
            Self::Buffer(buffer) => Ok(buffer),
        }
    }
    /// Copies into a new [`Buffer`].
    ///
    /// **Errors**
    /// - Allocates the buffer [`Buffer::alloc`](BufferBase::alloc).
    pub fn to_owned(&self) -> Result<Buffer<T>>
    where
        T: Copy,
    {
        match self {
            Self::Slice(slice) => slice.to_owned(),
            Self::Buffer(buffer) => buffer.to_owned(),
        }
    }
    /// Converts into a [`ArcBuffer`].
    ///
    /// **Errors**
    /// - Potentially allocates the buffer [`Buffer::alloc`](BufferBase::alloc).
    pub fn into_shared(self) -> Result<ArcBuffer<T>>
    where
        T: Copy,
    {
        Ok(self.into_owned()?.into())
    }
    /// The device of the buffer.
    pub fn device(&self) -> Device {
        match self {
            Self::Slice(slice) => slice.device(),
            Self::Buffer(buffer) => buffer.device(),
        }
    }
    /// The length of the buffer.
    pub fn len(&self) -> usize {
        match self {
            Self::Slice(slice) => slice.len(),
            Self::Buffer(buffer) => buffer.len(),
        }
    }
    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a, T> From<Slice<'a, T>> for CowBuffer<'a, T> {
    fn from(slice: Slice<'a, T>) -> Self {
        Self::Slice(slice)
    }
}

impl<'a, T> From<&'a [T]> for CowBuffer<'a, T> {
    fn from(slice: &'a [T]) -> Self {
        Self::Slice(slice.into())
    }
}

impl<T> From<Buffer<T>> for CowBuffer<'_, T> {
    fn from(buffer: Buffer<T>) -> Self {
        Self::Buffer(buffer)
    }
}

impl<T> From<Vec<T>> for CowBuffer<'_, T> {
    fn from(vec: Vec<T>) -> Self {
        Self::Buffer(vec.into())
    }
}

/// Casts
impl<T: Scalar> CowBuffer<'_, T> {
    /// Casts the buffer into a new buffer.
    ///
    /// See [`BufferBase::cast_into()`].
    pub fn cast_into<T2: Scalar>(self) -> Result<Buffer<T2>> {
        match self.try_unwrap() {
            Ok(buffer) => buffer.cast_into(),
            Err(this) => this.as_slice().cast_into(),
        }
    }
    /// Casts the buffer into a new buffer.
    ///
    /// Returns a CowBuffer, where for T -> T, borrows the buffer.
    ///
    /// See [`BufferBase::cast_into()`].
    pub fn cast_to<T2: Scalar>(&self) -> Result<CowBuffer<T2>> {
        Ok(self.as_slice().cast_into()?.into())
    }
    /// Scales the buffer into a new buffer.
    ///
    /// See [`BufferBase::scale_into()`].
    pub fn scale_into<T2: Scalar>(self, alpha: T2) -> Result<Buffer<T2>> {
        self.as_slice().scale_into(alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::Result;
    #[cfg(feature = "device_tests")]
    use crate::{device::Device, scalar::Scalar};
    #[cfg(feature = "device_tests")]
    use half::{bf16, f16};

    #[tokio::test]
    async fn host_buffer_copy_from_slice() -> Result<()> {
        let mut a = Buffer::from(vec![0; 4]);
        let b_vec = vec![1, 2, 3, 4];
        a.copy_from_slice(b_vec.as_slice().into())?;
        let a_guard = a.read().await?;
        assert_eq!(a_guard.as_slice(), b_vec.as_slice());
        Ok(())
    }

    #[tokio::test]
    async fn host_buffer_serde() -> Result<()> {
        let x = [1u32, 2, 3, 4];
        let slice = Slice::from(x.as_ref());
        let buffer: Buffer<u32> = bincode::deserialize(&bincode::serialize(&slice)?)?;
        assert_eq!(x.as_ref(), buffer.read().await?.as_slice());
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn device_buffer_from_vec() -> Result<()> {
        let device = Device::new()?;
        Buffer::<u32>::from(vec![1, 2, 3, 4])
            .into_device(device)
            .await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn device_buffer_copy_from_slice() -> Result<()> {
        let device = Device::new()?;
        let _s = device.acquire().await;
        let mut a = Slice::from([0; 4].as_ref())
            .into_device(device.clone())
            .await?;
        let b_vec = vec![1, 2, 3, 4];
        let b = Slice::from(b_vec.as_slice())
            .into_device(device.clone())
            .await?;
        a.copy_from_slice(b.as_slice())?;
        let a_guard = a.read().await?;
        assert_eq!(a_guard.as_slice(), b_vec.as_slice());
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn device_buffer_serde() -> Result<()> {
        let device = Device::new()?;
        let _s = device.acquire().await;
        let x = [1u32, 2, 3, 4];
        let buffer = Slice::from(x.as_ref()).into_device(device.clone()).await?;
        let buffer: Buffer<u32> = bincode::deserialize(&bincode::serialize(&buffer)?)?;
        assert_eq!(x.as_ref(), buffer.read().await?.as_slice());
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    async fn fill<T: Scalar>(n: usize, elem: T) -> Result<()> {
        let vec: Vec<T> = (0..n)
            .into_iter()
            .map(|n| T::from_usize(n).unwrap())
            .collect();
        let device = Device::new()?;
        let _s = device.acquire().await;
        let mut y = Slice::from(vec.as_slice())
            .into_device(device.clone())
            .await?;
        y.fill(elem)?;
        let y_guard = y.read().await?;
        assert!(
            y_guard.iter().copied().all(|y| y == elem),
            "{:?} != {:?}",
            y_guard.as_slice(),
            elem
        );
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_u8() -> Result<()> {
        fill(15, 11u8).await?;
        fill(100, 251).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_i8() -> Result<()> {
        fill(10, 11u8).await?;
        fill(100, -111i8).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_u16() -> Result<()> {
        fill(10, 11u16).await?;
        fill(1000, 211u16).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_i16() -> Result<()> {
        fill(10, 11i16).await?;
        fill(1000, -211i16).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_f16() -> Result<()> {
        fill(10, f16::from_f32(11.11)).await?;
        fill(1000, f16::from_f32(-211.11)).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_bf16() -> Result<()> {
        fill(10, bf16::from_f32(11.11)).await?;
        fill(1000, bf16::from_f32(-211.11)).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_u32() -> Result<()> {
        fill(10, 11u32).await?;
        fill(1000, 211u32).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_i32() -> Result<()> {
        fill(10, 11i32).await?;
        fill(1000, -211i32).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_f32() -> Result<()> {
        fill(10, 11.11f32).await?;
        fill(1000, -211.11f32).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_u64() -> Result<()> {
        fill(10, 11u64).await?;
        fill(1000, 211u64).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_i64() -> Result<()> {
        fill(10, 11i64).await?;
        fill(1000, -211i64).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn fill_f64() -> Result<()> {
        fill(10, 11.11f64).await?;
        fill(1000, -211.11f64).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    mod scale {
        use super::{Result, Scalar};

        async fn scale<T1: Scalar, T2: Scalar>(n: usize, alpha: T2) -> Result<()> {
            use super::*;

            let device = Device::new()?;
            let _s = device.acquire().await;
            let x_vec = (0..n)
                .into_iter()
                .map(|x| T1::from(x).unwrap())
                .collect::<Vec<_>>();
            let y_vec = x_vec
                .iter()
                .copied()
                .map(|x| T2::from_f32(x.to_f32().unwrap() * alpha.to_f32().unwrap()).unwrap())
                .collect::<Vec<_>>();
            let x_buffer = Slice::from(x_vec.as_slice())
                .into_device(device.clone())
                .await?;
            let y_buffer = x_buffer.scale_into(alpha)?;
            let y_guard = y_buffer.read().await?;
            assert_eq!(y_guard.as_slice(), y_vec.as_slice());
            Ok(())
        }

        macro_rules! impl_scale_tests {
            (($($t1:ident),+) => $t2s:tt) => {
                $(
                    mod $t1 {
                        use super::scale;
                        use crate::result::Result;

                        impl_scale_tests!{@Inner $t1 => $t2s}
                    }
                )+
            };
            (@Inner $t1:ident => ($($t2:ident),+)) => {
                $(
                    #[tokio::test]
                    async fn $t2() -> Result<()> {
                        #[allow(unused_imports)]
                        use half::{f16, bf16};
                        use num_traits::FromPrimitive;

                        for n in [10, 67] {
                            for alpha in [2i8, -2i8, 3i8] {
                                if let Some(alpha) = $t2::from_i8(alpha) {
                                    scale::<$t1, _>(n, alpha).await?;
                                }
                            }
                        }
                        Ok(())
                    }
                )+
            };
        }

        #[cfg(not(any(target_os = "ios", target_os = "macos")))]
        mod not_mac {
            use super::*;

            impl_scale_tests! { (u8, u16, bf16, u32, i32, f32) => (bf16) }
        }

        impl_scale_tests! { (u8, u16, bf16, u32, i32, f32) => (u32, i32, f32) }
    }
    /*
    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u8_bf16() -> Result<()> {
        scale::<u8, bf16>(10, 3u8.into()).await?;
        scale::<u8, bf16>(100, bf16::from_f32(-2f32)).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u8_u32() -> Result<()> {
        scale::<u8, u32>(10, 3).await?;
        scale::<u8, u32>(100, 17).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u8_i32() -> Result<()> {
        scale::<u8, i32>(10, -3).await?;
        scale::<u8, i32>(100, 17).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u8_f32() -> Result<()> {
        scale::<u8, f32>(10, -3.).await?;
        scale::<u8, f32>(100, 17.).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u16_bf16() -> Result<()> {
        scale::<u16, bf16>(10, 3u8.into()).await?;
        scale::<u16, bf16>(100, bf16::from_f32(-2f32)).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u16_u32() -> Result<()> {
        scale::<u16, u32>(10, 3).await?;
        scale::<u16, u32>(100, 17).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u16_i32() -> Result<()> {
        scale::<u16, i32>(10, -3).await?;
        scale::<u16, i32>(100, 17).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u16_f32() -> Result<()> {
        scale::<u16, f32>(10, -3.).await?;
        scale::<u16, f32>(100, 17.).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u16_bf16() -> Result<()> {
        scale::<u16, bf16>(10, 3u8.into()).await?;
        scale::<u16, bf16>(100, bf16::from_f32(-2f32)).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u16_u32() -> Result<()> {
        scale::<u16, u32>(10, 3).await?;
        scale::<u16, u32>(100, 17).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u16_i32() -> Result<()> {
        scale::<u16, i32>(10, -3).await?;
        scale::<u16, i32>(100, 17).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_u16_f32() -> Result<()> {
        scale::<u16, f32>(10, -3.).await?;
        scale::<u16, f32>(100, 17.).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_bf16_bf16() -> Result<()> {
        scale::<bf16, _>(10, bf16::from_f32(-3.)).await?;
        scale::<bf16, _>(100, bf16::from_f32(17)).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_bf16_u32() -> Result<()> {
        scale::<bf16, u32>(10, 3).await?;
        scale::<bf16, u32>(100, 17).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_bf16_i32() -> Result<()> {
        scale::<bf16, i32>(10, -3).await?;
        scale::<bf16, i32>(100, 17).await?;
        Ok(())
    }

    #[cfg(feature = "device_tests")]
    #[tokio::test]
    async fn scale_bf16_f32() -> Result<()> {
        scale::<u16, f32>(10, -3.).await?;
        scale::<u16, f32>(100, 17.).await?;
        Ok(())
    }*/
}
