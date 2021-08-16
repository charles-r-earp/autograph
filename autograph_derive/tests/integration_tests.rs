use autograph::{
    learn::neural_network::{
        autograd::{ParameterD, VariableD},
        layer::{Forward, Layer},
    },
    result::Result,
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
