use autograph::{
    Result,
    neural_network::{
        Dense, Forward, Identity, Network,
        autograd::{VariableD, Parameter2, Parameter1}
    },
};

#[derive(Network)]
struct Net1 {
    #[autograph(parameter)]
    weight: Parameter2,
    #[autograph(optional_parameter)]
    bias: Option<Parameter1>,
    #[autograph(layer)]
    dense1: Dense,
    #[autograph(optional_layer)]
    dense2: Option<Dense<Identity>>,
}

impl Forward for Net1 {
    fn forward(&self, _input: VariableD) -> Result<VariableD> {
        unimplemented!()
    }
}

#[derive(Network, Forward)]
struct Net2 {
    #[autograph(layer)]
    dense1: Dense,
    #[autograph(optional_layer)]
    dense2: Option<Dense<Identity>>,
    _name: String,
}
