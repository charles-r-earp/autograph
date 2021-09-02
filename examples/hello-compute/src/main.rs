use autograph::{
    result::Result,
    device::{Device, shader::Module},
    buffer::{Buffer, Slice},
};
use once_cell::sync::OnceCell;
use anyhow::anyhow;

static MODULE: OnceCell<Module> = OnceCell::new();

/// Loads the module.
///
/// The module is deserialized from the bytes injected into the binary.
fn module() -> Result<&'static Module> {
    Ok(MODULE.get_or_try_init(|| {
        bincode::deserialize(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/shaders/shader.bincode"
        )))
    })?)
}

/// Adds `a` to `b`.
fn add(a: Slice<u32>, b: Slice<u32>) -> Result<Buffer<u32>> {
    if a.len() != b.len() {
        return Err(anyhow!("{} != {}", a.len(), b.len()));
    }
    // Typically use `Buffer::alloc` here but it's unsafe.
    // `zeros()` runs a shader to fill the buffer, so it's unnecessary if it will be overwritten.
    let mut y = Buffer::zeros(a.device(), a.len())?;
    // The shader executes in "WorkGroups", which in the shader is defined to be [64, 1, 1]. This
    // means that even though we have just 1 item to process, it will actually run 64 invocations
    // aka threads. We have to pass `n` to prevent the extra invocations from writing outside of
    // the buffer.
    let n = y.len() as u32;
    let builder = module()?
        .compute_pass("add")?
        // `storage_buffer` at binding 0, must not be modified in the shader.
        .slice(a)?
        // `storage_buffer` at binding 1, must not be modified in the shader.
        .slice(b)?
        // `storage_buffer` at binding 2
        .slice_mut(y.as_slice_mut())?
        .push(n)?;
    unsafe {
        // Enqueues the shader with global size [n, 1, 1].
        // This method validates the arguments, and compiles the module for the device on first
        // use. Otherwise, this doesn't block, the internal device thread will submit work to the
        // device driver when it is ready.
        builder.submit([n, 1, 1])?;
    }
    Ok(y)
}

#[tokio::main]
async fn main() -> Result<()> {
    let device = Device::new()?;
    let x_in = [2];
    // Here we create a Slice<u32> from a &[u32].
    // We could also create a Buffer from a Vec, without copying.
    let x = Slice::from(x_in.as_ref())
        // Note that Host -> Device transfers are non-blocking, not async.
        .into_device(device.clone())
        .await?;
    /// Get the result of the addition.
    let y = add(x.as_slice(), x.as_slice())?;
    // Print out the result!
    println!("{:?} + {:?} = {:?}", x_in, x_in, y.read().await?.as_slice());
    Ok(())
}
