# Forward

```
/// Trait for forward pass, implemented by layers\
/// Typically this will call a method or custom Trait method on Variable\
/// A layer like Conv2d will implement Forward, and a model composed of layers will also implement forward.
pub trait Forward<D: Dimension> {
    type OutputDim: Dimension;
    fn forward(&self, input: &Variable<D>) -> Variable<Self::OutputDim>;
}
```

The Forward trait is used to compute the output given the input. When training, backward ops are enqueued into the graph attached to the input. The Layer and Forward traits combine to form the common interface of all layers / models. 

# Implementing Forward

Implementing forward for a struct is straightforward. For an image classifier, the input will be a Variable4 (NCHW), with D = Ix4, and the output will be Variable2 (NC). Use Variable's forward() method to sequence several methods, like this:

```
struct ConvNet {
    conv: Conv2d,
    dense: Dense
}

impl Forward<Ix4> for ConvNet { // 
    type OutputDim = Ix2;
    fn forward(&self, input: &Variable4) -> Variable2 {
        input
            .forward(&self.conv)
            .relu()
            .flatten()
            .forward(&self.dense)
    }
}
```

Similar to how Layer can be derived, Forward can be implemented with the impl_forward macro. We have to add the Relu and Flatten layers to our struct:

```
#[impl_forward(Ix4, Ix2)]
#[derive(Layer)]
struct ConvNet (
    Conv2d,
    Relu,
    Dense,
    Flatten
);
```

Note that Rust has zero sized types. Relu and Flatten have no cost, they will only affect the derived implementations, not the size of the struct. Now you only have to write a constructor for the model, potentially something like this:

```
impl ConvNet {
    pub fn new(device: &Device) -> Self {
        let conv = Conv2d::builder()
            .device(device)
            .inputs(1)
            .outputs(8)
            .args(
                Conv2dArgs::default()
                    .kernel(7)
            )
            .build();
        let relu = Relu::default();
        let flatten = Flatten::default();
        let Dense = Dense::builder()
            .inputs(8*22*22)
            .outputs(10)
            .bias()
            .build();
        ConvNet(
            conv,
            relu,
            flatten,
            dense
        ) 
    }
}
```

When developing it may be useful to experiment with several models. You can hide the type of the model in a few ways:

```
// some models, with a new method like above
struct Model1 { .. }
struct Model2 { .. }

// define a type alias for use in training code
type MyModel = ConvNet; // ConvNet2

fn train() {
    let device = Device::default();
    let model = MyModel::new(&device);
}
```
```
// use a function that returns impl Trait

fn create_model(device: &Device) -> impl Layer + Forward<Ix4, Output=Ix2> {
    Model1::new(device) // Model2::new(device)
}

fn train() {
    let device = Device::default();
    let model = create_model();
}
```

Use a boxed dyn Trait:

```
#[impl_forward(Ix4, Ix2)]
#[derive(Layer)]
struct DynModel(Box<dyn (Layer + Forward<Ix4, OutputDim=Ix2>)>);

impl DynModel {
    fn new(device: &Device) -> Self {
        let model = Model1::new(device) // Model2::new(device)
        DynModel(Box::new(model))
    }
}
```




