[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/charles-r-earp/autograph/LICENSE)
[![Build Status](https://api.travis-ci.com/charles-r-earp/autograph.svg?branch=master)](https://travis-ci.com/charles-r-earp/autograph)

# autograph
Machine Learning Library for Rust

# Concept
High performance Machine Learning Library inspired by Pytorch, built on top of ndarray. Flexible design allows for parallel and incremental execution, enabling dynamic composition of layers and functions, instead of purely statically defined models. 

# Note: No GPU Support
No means of supporting GPU acceleration, unless or until ndarray is extended as such. Longterm we may see Rust to cuda / SPIR-V, but until then this crate makes no abstractions to allow for gpu acceleration, due to simplicity and to minimize interior mutability and laziness, and because such abstractions are not zero cost.  

# Features
- Layers
  - Dense
- Functions
  - Dense
  - Softmax
  - CrossEntropyLoss
  - ClassificationMatches
- Initializers
  - Zeros
  - Random(rand::Distribution)
  - HeNormal
- Optimizers
  - () ie Learning Rate only

# Example
See /examples/mnist_dense.rs

# Benchmarks
Compared to similar fully connected layer in Pytorch, the above example achieves equalivent or better accuracy and trains + tests 30x faster, even against a moderate GPU. For larger networks, GPU acceleration should get an edge. Will benchmark well known models when the crate is more functional.

# Next Steps
Pull Requests welcome!
- Stochastic Gradient Descent with Momentum
- Conv Functions / Layers 
- Seq (Static list) Layer
- DynLayer
- Sequential Layer (Like a vec of DynLayers ie Box<Layer>)
- Saving / Loading of Model Structure and Parameters
  



