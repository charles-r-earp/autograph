[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/charles-r-earp/autograph/LICENSE)
[![Build Status](https://api.travis-ci.com/charles-r-earp/autograph.svg?branch=master)](https://travis-ci.com/charles-r-earp/autograph)

# autograph
Machine Learning Library for Rust

# Status
Still experimenting. Currently using ndarray for tensor math, found that it has very good performance, gpu acceleration would be nice but presents additional overhead, and is only beneficial with very large tensors / graphs, and that extra overhead complicates native implementations. 

# Concept
Layer is the core trait in autograph, which describes either a Dense / Conv layer, an activation function / normalization / dropout, or a composition such as Sequential. A layer takes an input and returns an output, and then accepts the gradient and computes the input gradients (and gradients of the parameters). It also has a params_mut() method, which allows for these parameters to be stepped with an Optimizer. 

# Example
See /examples/mnist_dense.rs

