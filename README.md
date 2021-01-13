[![LicenseBadge]][License]
[![DocsBadge]][Docs]
[![Build Status](https://github.com/charles-r-earp/autograph/workflows/Continuous%20Integration/badge.svg?branch=main)](https://github.com/charles-r-earp/autograph/actions)

[License]: https://github.com/charles-r-earp/autograph/blob/main/LICENSE-APACHE
[LicenseBadge]: https://img.shields.io/badge/license-MIT/Apache_2.0-blue.svg

[Docs]: https://docs.rs/autograph
[DocsBadge]: https://docs.rs/autograph/badge.svg


# autograph
Machine Learning Library for Rust

***undergoing maintenance***

# Features
  - Portable accelerated compute
  - Run SPIR-V shaders on GPU's that support Vulkan / Metal / DX12
  - Interop with ndarray, Tensor emulates Array
  - Lightweight Async / Non Blocking API 

Currently using GLSL as a shader language. When rust-gpu gains enough compute shader support, it will be possible to write portable GPU code in Rust! 

# Platforms

## Linux / Unix 
Supports GPU's with Vulkan. Tested on Ubuntu 18.04 AMD RX 580 / NV GTX 1060

## MacOs / iOS
Supports GPU's with Metal. Planned support for Vulkan. GPU execution untested. 

## Windows 
Supports GPU's with DX12. Planned support for Vulkan. Tested on Windows 10, AMD RX 580. 

Note: Run the windows tests with `cargo test -- --num-threads 1` to avoid creating too many instances of the gpu on too many threads. Shared access across threads is safe, but creating a Device for each of several processes may fail. 

# KMeans
Coming soon!

# Neural Networks
Coming soon!

# Datasets 
Coming sonn!

  
