[![License-MIT/Apache-2.0](https://img.shields.io/badge/license-MIT/Apache_2.0-blue.svg)](https://github.com/charles-r-earp/autograph/blob/master/LICENSE-APACHE)
![](https://docs.rs/autograph/badge.svg?)
![Rust](https://github.com/charles-r-earp/autograph/workflows/Rust/badge.svg?branch=master)

# autograph
Machine Learning Library for Rust

[User Guide](https://charles-r-earp.github.io/autograph/?) ***Under construction***

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
  - Sequential

## Optimizers
  - SGD
 
## Loss Functions
  - CrossEntropyLoss
 
## Datasets
  - MNIST
  
## Saving 
Model parameters can be saved to a file or serialized using serde (see https://github.com/serde-rs/serde). Training progress can also be saved via checkpoints. Data saved this way is portable between cpu and gpu. 

# Graphs
During training, first a forward pass is run to determine the outputs of a model and compute the loss. Then a backward pass is run to compute the gradients which are used to update the model parameters. Autograph constructs a graph of operations in the forward pass that is then used for the backward pass. This allows for intermediate values and gradients to be lazily allocated and deallocated using RAII in order to minimize memory usage. Native control flow like loops, if statements etc. can be used to define a forward pass, which does not need to be the same each time. This allows for novel deep learning structures like RNN's and GAN's to be constructed, without special hardcoding. 

# Supported Platforms
Tested on Ubuntu-18.04, Windows Server 2019, and macOS Catalina 10.15. Generally you will want to make sure that OpenMP is installed. Currently cmake / oneDNN has trouble finding OpenMP on mac and builds in sequential mode. This greatly degrades performance (approx 10x slower) but it will otherwise run without issues. If you have trouble building autograph, please create an issue. 

# CUDA
Cuda can be enabled by passing the feature "cuda" to cargo. CUDA https://developer.nvidia.com/cuda-downloads and cuDNN https://developer.nvidia.com/cudnn must be installed. See https://github.com/bheisler/RustaCUDA and https://github.com/charles-r-earp/cuda-cudnn-sys for additional information. 

# Datasets
Autograph includes a datasets module enabled with the features datasets. This currently has the MNIST dataset, which is downloaded and saved automatically. The implementation of this is old and outdated (it uses reqwest among others which now uses async), and compiles slowly. Potentially overkill for such a small dataset, but for adding new datasets (like ImageNet), we will need an updated, fast implementation. The datasets module may be moved to its own crate in the future. 

# Getting Started
If you are new to Rust, you can get it and find documentation here: https://www.rust-lang.org/

Read the [user guide](https://charles-r-earp.github.io/autograph/?) or the [docs](https://docs.rs/autograph/0.0.3/autograph/)

If you have git installed (see https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) you can clone the repo by:
```
git clone https://github.com/charles-r-earp/autograph
```

OpenMP is used when available in oneDNN. Without it, execution will be very slow. This requires a C++ compiler that supports OpenMP 2.0 or later:
  - gcc 4.2.0 or later `gcc --version`
  - clang/llvm 
    - Linux with Clang: may need to install `libomp-dev` for Debian/Ubuntu, `libomp-devel` for Void

View the documentation by:
```
cargo doc --open [--features cuda]
```
To add autograph to your project add it as a dependency in your cargo.toml (features are optional):
```
[dependencies]
autograph = { version = 0.0.3, features = ["cuda"] }
// or from github
autograph = { git = https://github.com/charles-r-earp/autograph, features = ["cuda"] }
```

# Tests
Run the unit-tests with (passing the feature cuda additionally runs cuda tests):
```
cargo test --lib [--features cuda]
```

# Examples
Run the examples with:
```
cargo run --example [example] [--features cuda] --release
```
See the examples directory for the examples.

# Benchmarks
Run the benchmarks with:
```
cargo bench [--features cuda]
```

# Roadmap 
  - Residual Layers
  - AMD / Rocm support
  - Datasets FashionMNIST, ImageNet, etc (potentially with a Dataset trait to enable shuffling, cropping, etc) 
  - Optimizers Adam (others?)
  - Loss functions Mean Squared Error / Binary Cross Entropy / L1 
  - Activations Leaky Relu / Tanh / Sigmoid
  - Data transfers between devices (local model parallel)
  - Data Parallel (multi-gpu)
  - Remote Distributed Parallel (ie training on multiple machines)
