# Layers

It is common to compose neural networks with layers. Layers represent functions, but may also store both parameters like weights, and hyper parameters like the padding of a convolution. Layers make it easy to define a model, as they translate hyper parameters into the shape of the parameters. For example, you could create a simple Dense layer with this:

```
let w = Parameter::new(
    ArcTensor::zeros(&device, `[outputs, inputs]`)
);
let b = Parameter::new(
    ArcTensor::zeros(&device, outputs)
);

// forward 
let y = x.dense(&w, Some(&b));
```

Or you could use a Dense layer:

```
let dense = Dense::builder()
    .device(&device)
    .inputs(inputs)
    .outputs(outputs)
    .build();
    
/// forward
let y = x.forward(&dense);
```

## Layer Trait 

```
/// Trait for Layers\
/// Custom Models should impl Layer
pub trait Layer {
    /// Returns a Vec of all the parameters in the Layer (including its children). Parameter acts like an Arc so it can be cloned to copy references. Layers that do not have parameters (like Activations) do not have to implement this method.
    fn parameters(&self) -> Vec<ParameterD> {
        Vec::new()
    }
    /// Prepares the layer for training if training is true, else prepares for evaluation / inference. This method should be called prior to a forward step ie:
    ///```
    /// for data in training_set {
    ///   let graph = Graph::new();
    ///   let (x, t) = // data
    ///   model.set_training(true);
    ///   let y = model.forward(&x);
    ///   let loss = // loss function
    ///   loss.backward(graph);
    ///   // update model
    /// }
    /// for data in evaluation_set {
    ///   let (x, t) = // data
    ///   model.set_training(false);
    ///   let y = model.forward(&x);
    ///   let loss = // loss function
    /// }
    ///```
    /// The implementation should recursively call set_training on all of its child layers, and or all of its parameters.
    fn set_training(&mut self, training: bool) {}
}
```

The Layer trait provides a minimal interface for accessing parameters and setting the training mode. It is similar to PyTorch's Module. Layers can be composed in a struct:

```
struct MyModel {
    layer1: Layer1,
    layer2: Layer2,
    // etc... 
}

// Note: you can also use a tuple struct, with unnamed fields
struct MyModel (
    Layer1, // my_model.0
    Layer2, // my_model.1
    // etc... 
);
```

There is a derive macro that will generate a Layer implementation for a struct composed of other layers:
```
#[derive(Layer)]
struct MyModel { /*fields*/ }
```
