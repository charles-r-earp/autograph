#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr),
    register_attr(spirv)
)]
#![deny(warnings)]

use spirv_std::glam::UVec3;

// Declare the push constants. These can potentially be shared with the runtime crate, but that
// may require a separate `shared` crate. There is a limit in autograph of 64 B of push constants,
// and the size must be a multiple of 4 bytes (ie a u32). Use `#[repr(C)]` to ensure that fields
// are not reordered.
#[repr(C)]
pub struct PushConsts {
    n: u32,
}

/// Computes `y' = `a` + `b`
///
/// `threads` can be up to 3 dimensions (x, y, z). This is the size of the `WorkGroup`. Generally
/// this should be a multiple of the hardware specific size, NVidia refers to this as the
/// `warp size`, which for NVidia is often 32 but sometimes 64. For AMD this is generally 64. 64
/// is a good default. Note that autograph will automatically choose the number of work groups to
/// execute given the global size, so it is not necessary for the function submitting the shader
/// to know the work group size.
///
/// # Note
/// autograph does check the size of the push constants, as well as the mutability of buffers. It
/// DOES NOT check their types. For example, a buffer can be declared like `&[u32]` but bound to a
/// `Slice<u8>`.
#[allow(unused)]
#[spirv(compute(threads(64)))]
pub fn add(
    // This is the unique id of the invocation, and is 3D (x, y, z) even though we are just using x.
    // This tells the invocation what index to compute.
    #[spirv(global_invocation_id)] global_id: UVec3,
    // Buffer `a`. As of now, `storage_buffer, descriptor_set, binding, non_writable` must all be
    // specified. `non_writable` corresponds to immutable ie `Slice`.
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0, non_writable)] a: &[u32],
    // Buffer `b`.
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1, non_writable)] b: &[u32],
    // Buffer `y`, the output. Here `non_readable` is added for clarity, but likely has no affect.
    // Because this is not `non_writable`, it can be bound to a `SliceMut`.
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2, non_readable)] y: &mut [u32],
    // Push constants, ie additional arguments passed at runtime.
    #[spirv(push_constant)] push_consts: &PushConsts,
) {
    let gid = global_id.x as usize;
    // Only process up to n, which is the length of the buffers.
    if global_id.x < push_consts.n {
        // The indexing operation is implemented by rust-gpu, and is the only way to access
        // the data, ie using `[T]::get()` and dereferencing &T or *const T will fail to compile.
        y[gid] = a[gid] + b[gid];
    }
}
