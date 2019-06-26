# autograph
OpenCL accelerated Tensor computation library for Rust

## Goals

  - Initialization Ops (Zeros, Ones, He, Xavier)
  - Elementwise (Add, Sub, Mul, Div)
  - Activations (Relu, Sigmoid, Softmax)
  - Matrix Multiplication
  - Explicit Reshaping and Broadcasting
  - Support for Backward / Gradient Ops
  
## Concept
Tensors are created with a shape and some data, and then copied into a buffer which is stored in a Graph. Operations can be applied to tensors, adding more buffers and ops to the graph. The graph is then executed. Then tensors can read the data, copying it from the buffers on the device. Read / Write ops are readily executed. Graph operations, such as a forward inference or a full forward / backward / weight update are executed as a group, no external copying of data to and from the device. The OpenCL source can be compiled once at startup, which will greatly reduce overhead. 

Tensors and Graphs are thread safe, by explicitly passing references at each op call. There is no interior mutability, either by RefCell or Arc, for maximum safety and performance, at the cost of more verbose code, particularly for mathmatical expressions. For machine learning applications, a more functional approach is acceptable and thus this is a small price to pay. 

## Status 
Work in progress. Very experimental, api is not stable. 

