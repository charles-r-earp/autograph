use autograph::{
    result::Result,
    learn::neural_network::{
        layer::{Layer, Forward},
        autograd::{VariableD, ParameterD},
    },
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
    dense2: Option<DenseLayer>
}
