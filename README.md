[![LicenseBadge]][License]
[![DocsBadge]][Docs]
[![Build Status](https://github.com/charles-r-earp/autograph/workflows/Continuous%20Integration/badge.svg?branch=main)](https://github.com/charles-r-earp/autograph/actions)

[License]: https://github.com/charles-r-earp/autograph/blob/main/LICENSE-APACHE
[LicenseBadge]: https://img.shields.io/badge/license-MIT/Apache_2.0-blue.svg

[Docs]: https://docs.rs/autograph
[DocsBadge]: https://docs.rs/autograph/badge.svg


# **autograph**
A machine learning library for Rust.

To use **autograph** in your crate, add it as a dependency in Cargo.toml:
```
[dependencies]
autograph = { git = https://github.com/charles-r-earp/autograph }
```

# Requirements
- Rust <https://www.rust-lang.org/>
- A device (typically a gpu) with drivers for a supported API:
    - Vulkan (All platforms) <https://www.vulkan.org/>
    - Metal (MacOS / iOS) <https://developer.apple.com/metal/>
    - DX12 (Windows) <https://docs.microsoft.com/windows/win32/directx>

# Tests
To check that you have a valid device, run `cargo test device_new`.

Run the tests with `cargo test`. Some drivers have limits on the number of logical devices that can be created. Run the tests with `cargo test -- --test-threads 1` or a small value to fix.

# Custom Shader Code
You can write your own shaders and execute them with **autograph**. See the [Hello Compute](examples/hello-compute/README.md) example.
