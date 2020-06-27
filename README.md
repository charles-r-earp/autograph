[![License-MIT/Apache-2.0](https://img.shields.io/badge/license-MIT/Apache_2.0-blue.svg)](https://github.com/charles-r-earp/autograph/LICENSE-MIT)
![Rust](https://github.com/charles-r-earp/autograph/workflows/Rust/badge.svg?branch=master)
# autograph
Machine Learning Library for Rust

# Features
  - Safe API
  - Thread Safe
  - CPU and CUDA are fully supported
  - Flexible (Dynamic Backward Graph)

## Layers
  - Dense
  - Conv2d
  - MaxPool2d
  - Relu
 
## Loss Functions
  - CrossEntropyLoss
 
## Datasets
  - MNIST

# Graphs
During training, first a forward pass is run to determine the outputs of a model and compute the loss. Then a backward pass is run to compute the gradients which are used to update the model parameters. Autograph constructs a graph of operations in the forward pass that is then used for the backward pass. This allows for intermediate values and gradients to be lazily allocated and deallocated using RAII in order to minimize memory usage. Native control flow like loops, if statements etc. can be used to define a forward pass, which does not need to be the same each time. This allows for novel deep learning structures like RNN's and GAN's to be constructed, without special hardcoding. 
  
# Extension
See branch extend_api. This branch adds feature xapi which enables additional methods needed to add ops to autograph. This is experimental and feedback welcome. Currently only supports cpu operations. Autograph is highly modularized, cpu operations can be implemented using pure Rust, or via c/c++ to oneDNN (see https://oneapi-src.github.io/oneDNN/), or some other means. The xapi feature provides access to the dnnl::engine and dnnl::stream needed to perform operations with dnnl (now oneDNN). 

# Supported Platforms
Tested on Ubuntu-18.04, Windows Server 2019, and macOS Catalina 10.15. Generally you will want to make sure that OpenMP is installed. Currently cmake / oneDNN has trouble finding OpenMP on mac and builds in sequential mode. This greatly degrades performance (approx 10x slower) but it will otherwise run without issues. If you have trouble building autograph, please create an issue. 

# CUDA
Cuda can be enabled by passing the feature "cuda" to cargo. CUDA https://developer.nvidia.com/cuda-downloads and cuDNN https://developer.nvidia.com/cudnn must be installed. See https://github.com/bheisler/RustaCUDA for additional help. CUDA is only tested during developement and is not tested via Actions, so it may not work on different platforms and CUDA versions. Currently it will only work on linux.
```
cargo run --features cuda
```

# Datasets
Autograph includes a datasets module enabled with the features datasets. This currently has the MNIST dataset, which is downloaded and saved automatically. The implementation of this is old and outdated (it uses reqwest among others which now uses async), and compiles slowly. Potentially overkill for such a small dataset, but for adding new datasets (like ImageNet), we will need an updated, fast implementation. 

# Getting Started
If you are new to Rust, you can get it and find documentation here: https://www.rust-lang.org/
View the documentation by:
```
cargo doc --open [--features "[datasets] [cuda]"]
```
To add autograph to your project add it as a dependency in your cargo.toml (features are optional):
```
[dependencies]
autograph = { git = "https://github.com/charles-r-earp/autograph", features = ["datasets", "cuda"] }
```
Autograph is also available on crates.io, but it is not up-to-date. With the next release, you can use the following instead:
```
autograph = { version = 0.0.2, features = ["datasets", "cuda"] }
```
# Tests
Run the unit-tests with (passing the feature cuda additionally runs cuda tests):
```
cargo test --lib [--features cuda]
```

# Examples
Run the examples with:
```
cargo run --example [example] --features "datasets [cuda]" --release
```
See the examples directory for the examples.

# Benchmarks
Run the benchmarks with:
```
cargo bench [--features cuda]
```

# Roadmap 
  - Optimizers (SGD, Adam)
  - Saving and loading of models / Serde
  - Data transfers between devices (local model parallel)
  - Data Parallel (multi-gpu)
  - Remote Distributed Parallel (ie training on multiple machines)
