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
- To check that you have a valid device, run `cargo test device_new --features device_tests`.
- Run all the tests with `cargo test --features full`.

# Custom Shader Code
You can write your own shaders and execute them with **autograph**. See the [Hello Compute](examples/hello-compute/README.md) example.

# Help Wanted!
Lots of things are under construction. Ideas for contributions:
- Easy
    - Add a dataset. See the iris dataset for an example. Larger datasets like MNIST should load from a directory, the user can download the files themselves.
- Medium
    - Add a new method or algorithm. Neural Networks are in maintenance. May start with KMeans, just needs to be brought up-to-date.
        - Need full implementations for casts, elementwise ops, etc.
        - Softmax
    - Improve the shader parser, better error messages? Need to implement specialization constants, they are implemented in the engine but not in `Module`.
- Hard
    - Improve the performance of the GEMM / Dot implementation. Ideally implement it in Rust. This will likely require specialization constants, and those aren't available in rust-gpu yet.
    - Add larger datasets, that will be loaded dynamically, potentially using async io. Also need pre-processing ops like crops / normalization.

# Developement Platforms
1. Ubuntu 18.04 | (Vulkan) NVidia GeForce GTX 1060 with Max-Q Design
2. Wondows 10 Home | (Vulkan + DX12) AMD RX 580 / (DX12) Microsoft Basic Render Driver.

Shaders are tested on Github Actions:
- Windows Server 2019 | (DX12) Microsoft Basic Render Driver.

## Metal
Shaders are untested on Metal / Apple platforms. If you have problems, please create an issue!

# License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.

# Contribution
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
