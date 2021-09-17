use autograph::{
    learn::neural_network::{
        autograd::{
            Autograd, Gradient2, Parameter1, Parameter2, ParameterD, Variable2, VariableD,
            VariableGradientD,
        },
        layer::{Forward, Layer},
    },
    result::Result,
    tensor::float::{FloatArcTensor2, FloatArcTensorD},
};

#[derive(Layer)]
struct DenseLayer {
    #[autograph(parameter)]
    weight: ParameterD,
    #[autograph(optional_parameter)]
    bias: Option<ParameterD>,
}

impl Forward for DenseLayer {
    fn forward(&self, _input: VariableD) -> Result<VariableD> {
        unimplemented!()
    }
}

#[derive(Layer, Forward)]
struct SeqLayer {
    #[autograph(layer)]
    dense1: DenseLayer,
    #[autograph(optional_layer)]
    dense2: Option<DenseLayer>,
}

#[derive(Autograd)]
struct DenseBackward {
    #[autograph(vertex)]
    input: Variable2,
    #[autograph(vertex)]
    weight: Parameter2,
    #[autograph(optional_vertex)]
    bias: Option<Parameter1>,
}

#[derive(Autograd)]
struct DotBackward {
    input: FloatArcTensor2,
    #[autograph(gradient)]
    input_grad: Gradient2,
    weight: FloatArcTensor2,
    #[autograph(optional_gradient)]
    weight_grad: Option<Gradient2>,
}

#[derive(Autograd)]
struct ReluBackward {
    #[autograph(gradient)]
    input_grad: VariableGradientD,
    output: FloatArcTensorD,
}
