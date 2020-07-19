use autograph::{Device, Cpu};
use autograph::autograd::{Variable, Variable2, ParameterD};
use autograph::layer::{
    Layer, 
    Forward,
    Dense,
    Conv2d,
    MaxPool2d,
    Relu,
    Flatten
}; 
use ndarray::{Ix2, Ix4, RemoveAxis};

#[macro_use]
extern crate autograph_derive;


#[impl_forward(Ix2, Ix2)]
#[derive(Layer)]
struct DenseNet1 (
    Dense 
);

#[impl_forward(Ix4, Ix2)]
#[derive(Layer)]
struct Lenet5 (
    Conv2d,
    Relu,
    MaxPool2d,
    Conv2d,
    Relu,
    MaxPool2d,
    Flatten,
    Dense,
    Relu,
    Dense,
    Relu,
    Dense
);

#[impl_forward(Ix4, Ix2)]
#[derive(Layer)]
struct TestSkip {
    flatten1: Flatten,
    #[autograph(skip_layer)]
    meta: String,
    dense1: Dense
}
