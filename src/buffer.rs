//!
//! # Compute Example
//! This example shows the basics of creating buffers, executing compute, and reading back the results.
/*!```no_run
use autograph::{
    result::Result, device::{Device, Module}, buffer::{Buffer, Slice},
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

#[doc(inline)]
pub use crate::device::buffer::*;

/// Float buffers.
pub mod float;
