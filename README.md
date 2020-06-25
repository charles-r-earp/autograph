[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/charles-r-earp/autograph/LICENSE)
![Rust](https://github.com/charles-r-earp/autograph/workflows/Rust/badge.svg?branch=master)
# autograph
Machine Learning Library for Rust

# Features
  - Safe API
  - Thread Safe
  - CPU and CUDA are fully supported
  - Flexible (Dynamic Backward Graph)

# Graphs
During training, first a forward pass is run to determine the outputs of a model and compute the loss. Then a backward pass is run to compute the gradients which are used to update the model parameters. Autograph constructs a graph of operations in the forward pass that is then used for the backward pass. This allows for intermediate values and gradients to be lazily allocated and deallocated using RAII in order to minimize memory usage. Native control flow like loops, if statements etc. can be used to define a forward pass, which does not need to be the same each time. This allows for novel deep learning structures like RNN's and GAN's to be constructed, without special hardcoding. 

# Layers
  - Dense
  - Conv2d
  - MaxPool2d
  - Relu
  
# Extension
Currently autograph hides it's implementation, and new ops cannot be added externally. Backward ops are stored in an enum rather than a Box<dyn _>. Further experimentation is needed to determine the cost of using dynamic dispatch (ie a vtable). Ops should also be implemented for both CPU and CUDA and potentially additional devices. Exposing an interface to rapid developement is worthwhile but isn't trivial. 

# Supported Platforms
Autograph currently is only tested and will only work on Linux, due to linking with OpenMP. Support for Windows and Mac is coming soon. 

# CUDA
Cuda can be enabled by passing the feature "cuda" to cargo. CUDA https://developer.nvidia.com/cuda-downloads and cuDNN https://developer.nvidia.com/cudnn must be installed.
```
cargo run --features cuda
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
  - Configure Github Actions for testing on multiple platforms
  - Add support for Windows and Mac
  - Documentation
  - Experiment with public methods for extending autograph
  - Optimizers (SGD, Adam)
  - Saving and loading of models / Serde
  - Data transfers between devices (local model parallel)
  - Data Parallel (multi-gpu)
  - Remote Distributed Parallel (ie training on multiple machines)
