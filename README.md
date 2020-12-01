[![LicenseBadge]][License]
[![DocsBadge]][Docs]
![Rust](https://github.com/charles-r-earp/autograph/workflows/Rust/badge.svg?branch=main)

[License]: https://github.com/charles-r-earp/autograph/blob/main/LICENSE-APACHE
[LicenseBadge]: https://img.shields.io/badge/license-MIT/Apache_2.0-blue.svg

[Docs]: https://docs.rs/autograph
[DocsBadge]: https://docs.rs/autograph/badge.svg


# autograph
Deep Neural Network Library for Rust

***undergoing maintenance***

# Future changes from v0.0.3
  - Pure Rust
  - Build on wgpu, consuming spirv as gpu source
  - Async / Non-blocking API
  - Local Node of 1 or more gpus
  - Parallel execution and training 
  - Run custom ops (ie shaders), probably limited to precompiled sources
  - Target all backends supported by wgpu, including mobile and macos 
  
The emergence of both wgpu and rust-gpu, async, as well as my experience building backends for cpu, CUDA, ROCm, OpenCL, and wgpu, motivate moving to a single backend. That is, consume SPIR-V and target gpus with wgpu. Potentially there will be some way to run shaders on cpu as well. This means that the backend can be very lean and well optimized, and development will be be much faster and better optimized. 
  
The general approach to layers and autograd will remain the same.

# Long term goal
  - Remote Node of 1 or more gpus
  - Parallel execution and training of multiple nodes 
  - Execution on a cpu node / device
  - autograph may add additional functionality for applications other than neural networks, but this may be left to contribution  
    * (the backend will not know anything about autograd so it will support arbitrary computation)
  
