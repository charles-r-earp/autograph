use autograph::{
    neural_network::{Network, Forward, Dense, Identity},
};
#[allow(unused_imports)]
#[macro_use]
extern crate autograph_derive;
use autograph_derive::*;

#[derive(Network, Forward)]
struct Net {
    dense1: Dense,
    dense2: Dense<Identity>
}
/*
#[derive(Network, Forward)]
struct Net2 (
    Dense,
    Dense,
    Dense,
);

#[derive(Network, Forward)]
struct Net2 (
    Dense,
    Dense,
    Dense,
);

struct Thing {}

#[derive(Network, Forward)]
struct Net3 (
    #[autograph(ignore)]
    Tning,
    Dense,
    Dense,
    Dense,
);

#[derive(Network, Forward)]
struct Net3 (
    // ignore the field
    #[autograph(ignore)]
    String,
    Dense,
    Dense,
);
*/
