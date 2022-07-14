use crate::{result::Result, shader::Module};
use std::{
    fmt::{self, Debug},
    future::Future,
    sync::Arc,
};

#[cfg(feature = "profile")]
mod profiler;

mod engine;
use engine::{Engine, StorageBuffer, StorageBufferReadGuard};

#[doc(hidden)]
pub mod buffer;

#[doc(hidden)]
pub mod shader;
use shader::EntryId;

/// Device builders.
pub mod builders {
    use super::*;
    use crate::{
        buffer::{Slice, SliceMut},
        device::shader::EntryDescriptor,
        shader::Module,
    };
    use anyhow::{anyhow, bail};
    use bytemuck::Pod;
    use std::marker::PhantomData;

    /// A builder for executing compute shaders.
    ///
    /// See [`Module::compute_pass()`].
    pub struct ComputePassBuilder<'m, 'b> {
        device: Option<DeviceBase>,
        module: &'m Module,
        entry: String,
        descriptor: &'m EntryDescriptor,
        buffers: Vec<BufferBinding>,
        push_constants: Vec<u8>,
        _m: PhantomData<&'b ()>,
    }

    impl<'m, 'b> ComputePassBuilder<'m, 'b> {
        fn new(module: &'m Module, entry: String) -> Result<Self> {
            let descriptor = module
                .descriptor
                .entries
                .get(&entry)
                .ok_or_else(|| anyhow!("Entry not found {:?}!", entry))?;
            Ok(Self {
                device: None,
                module,
                entry,
                descriptor,
                buffers: Vec::with_capacity(descriptor.buffers.len()),
                push_constants: Vec::with_capacity(descriptor.push_constant_size()),
                _m: PhantomData::default(),
            })
        }
        /// Adds a slice as an argument to the shader at the next binding.
        ///
        /// **Errors**
        /// - The slice is on the host.
        /// - The slice is not on the same device as previous arguments.
        /// - The slice is modified in the shader.
        /// - There are no more arguments to be bound.
        ///
        /// # Note
        /// - Does not check the numerical type of the buffer.
        pub fn slice<'b2, T>(self, slice: Slice<'b2, T>) -> Result<ComputePassBuilder<'m, 'b2>>
        where
            'b2: 'b,
            T: Pod,
        {
            self.slice_impl(
                slice
                    .into_device_buffer_base()
                    .map(|buffer| (buffer.binding(), buffer.device())),
            )
        }
        /// Adds a mutable slice as an argument to the shader at the next binding.
        ///
        /// **Errors**
        /// - The slice is on the host.
        /// - The slice is not on the same device as previous arguments.
        /// - There are no more arguments to be bound.
        ///
        /// # Note
        /// - Does not check the numerical type of the buffer.
        pub fn slice_mut<'b2, T>(
            self,
            slice: SliceMut<'b2, T>,
        ) -> Result<ComputePassBuilder<'m, 'b2>>
        where
            'b2: 'b,
            T: Pod,
        {
            self.slice_impl(
                slice
                    .into_device_buffer_base()
                    .map(|buffer| (buffer.binding_mut(), buffer.device())),
            )
        }
        fn slice_impl<'b2>(
            mut self,
            buffer_device: Option<(BufferBinding, DeviceBase)>,
        ) -> Result<ComputePassBuilder<'m, 'b2>> {
            let (buffer, device) = buffer_device.ok_or_else(|| {
                anyhow!("Compute passes can only be executed on a device, not the host!")
            })?;
            if let Some(current_device) = self.device.as_ref() {
                if current_device != &device {
                    bail!(
                        "Provided slice at binding {} is on {:?} but previous slices are on {:?}!",
                        self.buffers.len(),
                        Device::from(device),
                        Device::from(current_device.clone()),
                    );
                }
            } else {
                self.device.replace(device);
            }
            self.check_num_buffers(self.buffers.len() + 1, false)?;
            let declared_mutable = *self.descriptor.buffers.get(self.buffers.len()).unwrap();
            if !buffer.mutable && declared_mutable {
                bail!(
                    "Provided slice at binding {}, but it is modified in {:?} entry {:?}!",
                    self.buffers.len(),
                    self.module,
                    &self.entry,
                )
            }
            self.buffers.push(buffer);
            Ok(ComputePassBuilder {
                device: self.device,
                module: self.module,
                entry: self.entry,
                descriptor: self.descriptor,
                buffers: self.buffers,
                push_constants: self.push_constants,
                _m: PhantomData::default(),
            })
        }
        fn check_num_buffers(&mut self, size: usize, finished: bool) -> Result<()> {
            let buffers_size = self.descriptor.buffers.len();
            if size > buffers_size || (finished && size != buffers_size) {
                Err(anyhow!(
                    "Provided number of slices {} does not match the number of bindings {} declared in {:?} entry {:?}!",
                    size,
                    buffers_size,
                    self.module,
                    &self.entry,
                ))
            } else {
                Ok(())
            }
        }
        /// Adds one or more push constants.
        ///
        /// `push_constant` can be a numerical type, an array, or a struct implementing [`Pod`](bytemuck::Pod).
        ///
        /// **Errors**
        /// - The push constants exceed the size declared in the module.
        pub fn push<T: Pod>(self, push_constant: T) -> Result<Self> {
            self.push_bytes(bytemuck::cast_slice(&[push_constant]))
        }
        /// Adds push constants as bytes.
        ///
        /// Like [`.push()`](ComputePassBuilder::push), but accepts bytes directly.
        ///
        /// **Errors**
        /// - The push constants exceed the size declared in the module.
        pub fn push_bytes(mut self, bytes: &[u8]) -> Result<Self> {
            let push_constant_size = self.push_constants.len() + bytes.len();
            self.check_push_constant_size(push_constant_size, false)?;
            self.push_constants.extend_from_slice(bytes);
            Ok(self)
        }
        fn check_push_constant_size(&self, size: usize, finished: bool) -> Result<()> {
            let declared_size = self.descriptor.push_constant_size();
            if size > declared_size || finished && size != declared_size {
                Err(anyhow!(
                    "Provided push constant size {} does not match declared size {} in {:?} entry {:?}!",
                    size,
                    declared_size,
                    self.module,
                    self.entry,
                ))
            } else {
                Ok(())
            }
        }
        /// Submits the compute pass to the device with the given `global_size`.
        ///
        /// The compute pass will be executed with enough "work groups" such that for each dimension GLOBAL_SIZE >= WORK_GROUPS * LOCAL_SIZE.
        ///
        /// # Non-Blocking
        /// Does not block, the implementation submits work to the driver asynchronously.
        ///
        /// # Safety
        /// The caller must ensure:
        /// 1. The type `T` of each slice argument is valid. For example `Slice<u8>` can be bound to shader buffer that is a uint or u32, or any arbitrary type.
        /// 2. The type(s) of the push constants are valid. Only the size (in bytes) is checked, the shader can interpret the bytes arbitrarily.
        /// 3. The `global_size` is valid. If too large, the shader may perform out of bounds reads or writes, which is undefined behavior. If too small, the shader may not fully compute the output(s). Note that the shader will be executed with potentially more invocations than `global_size`, in blocks of the `local_size` declared in the shader. Typically the actual work size (ie the length of the buffer(s)) is passed as a push constant.
        /// 4. The shader is valid. Executing a compute shader is essentially a foreign function call, and is inherently unsafe.
        ///
        /// **Errors**
        /// - The number of arguments does not match the number declared in the module.
        /// - The total size in bytes of all push constants does not match that declared in the module.
        /// - The total size in bytes of all specialization constants does not match that declared in the module.
        /// - The device panicked or disconnected.
        /// - The device could not compile the module (compiles all entries on first use).
        /// - The device could not compile the specialization (compiles each entry / specialization on first use).
        pub unsafe fn submit(mut self, global_size: [u32; 3]) -> Result<()> {
            fn work_groups(global_size: [u32; 3], local_size: [u32; 3]) -> [u32; 3] {
                let mut work_groups = [0; 3];
                for (wg, (gs, ls)) in work_groups
                    .iter_mut()
                    .zip(global_size.iter().copied().zip(local_size.iter().copied()))
                {
                    *wg = if gs % ls == 0 { gs / ls } else { gs / ls + 1 };
                }
                work_groups
            }
            if self.buffers.is_empty() {
                bail!("No slices passed to compute pass builder!");
            }
            self.check_num_buffers(self.buffers.len(), true)?;
            self.check_push_constant_size(self.push_constants.len(), true)?;
            let local_size = self.descriptor.local_size;
            let work_groups = work_groups(global_size, local_size);
            let compute_pass = ComputePass {
                module: self.module,
                entry: self.descriptor.id,
                entry_name: self.entry,
                work_groups,
                #[cfg(feature = "profile")]
                local_size,
                buffers: self.buffers,
                push_constants: self.push_constants,
            };
            self.device
                .expect("No device!")
                .compute_pass(compute_pass)?;
            Ok(())
        }
    }

    impl Module {
        /// Returns a [`ComputePassBuilder`] used to setup and submit a compute shader.
        ///
        /// **Errors**
        /// - The `entry` was not found in the module.
        pub fn compute_pass(&self, entry: impl Into<String>) -> Result<ComputePassBuilder> {
            ComputePassBuilder::new(self, entry.into())
        }
    }
}

struct BufferBinding {
    storage: Arc<StorageBuffer>,
    offset: usize,
    len: usize,
    mutable: bool,
}

struct ComputePass<'a> {
    module: &'a Module,
    entry: EntryId,
    entry_name: String,
    work_groups: [u32; 3],
    #[cfg(feature = "profile")]
    local_size: [u32; 3],
    buffers: Vec<BufferBinding>,
    push_constants: Vec<u8>,
}

pub(crate) struct DeviceBase {
    engine: Arc<Engine>,
}

impl DeviceBase {
    fn new() -> Result<Self> {
        Ok(Self {
            engine: Engine::new()?,
        })
    }
    fn index(&self) -> usize {
        self.engine.index()
    }
    unsafe fn alloc(&self, len: usize) -> Result<Arc<StorageBuffer>> {
        unsafe { self.engine.alloc(len) }
    }
    fn upload(&self, data: &[u8]) -> Result<Arc<StorageBuffer>> {
        self.engine.upload(data)
    }
    fn download(
        &self,
        storage: Arc<StorageBuffer>,
        offset: usize,
        len: usize,
    ) -> impl Future<Output = Result<StorageBufferReadGuard>> {
        let result = self.engine.download(storage, offset, len);
        async move { result?.await }
    }
    fn compute_pass(&self, compute_pass: ComputePass<'_>) -> Result<()> {
        self.engine.compute_pass(compute_pass)
    }
    fn sync(&self) -> Result<impl Future<Output = Result<()>>> {
        self.engine.sync()
    }
}

impl Clone for DeviceBase {
    fn clone(&self) -> Self {
        Self {
            engine: self.engine.clone(),
        }
    }
}

impl PartialEq for DeviceBase {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.engine, &other.engine)
    }
}

impl Eq for DeviceBase {}

/// Device.
#[derive(Clone, PartialEq, Eq)]
pub struct Device {
    base: Option<DeviceBase>,
}

impl Device {
    /// Creates a device.
    pub fn new() -> Result<Self> {
        Ok(Self::from(DeviceBase::new()?))
    }
    /// Returns a host device.
    ///
    /// The host is stateless and trivially constructable. That is, all [`host()`](Device::host())'s are equivalent. The host supports basic operations like initialization and copying. It does not support shader execution, so most computations will fail.
    pub fn host() -> Self {
        Self { base: None }
    }
    pub(crate) fn into_base(self) -> Option<DeviceBase> {
        self.base
    }
    /// Synchronizes pending operations.
    ///
    /// The returned future will resolve when all previous transfer and compute operations are complete.
    ///
    /// NOOP on the host.
    ///
    /// **Errors**
    /// - The device panicked or disconnected.
    pub fn sync(&self) -> impl Future<Output = Result<()>> {
        let fut = self.base.as_ref().map(DeviceBase::sync);
        async move {
            if let Some(fut) = fut {
                fut?.await?;
            }
            Ok(())
        }
    }
}

impl Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(base) = self.base.as_ref() {
            f.write_str(&format!("Device({})", base.index()))
        } else {
            f.write_str("Host")
        }
    }
}

impl From<DeviceBase> for Device {
    fn from(base: DeviceBase) -> Self {
        Self { base: Some(base) }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "device_tests")]
    use super::*;

    #[cfg(feature = "device_tests")]
    #[test]
    fn device_new() -> Result<()> {
        Device::new()?;
        Ok(())
    }
}
