use autograph::{
    neural_network::{
        autograd::{Parameter1, Parameter2, VariableD},
        Dense, Forward, Identity, Network,
    },
    Result,
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
