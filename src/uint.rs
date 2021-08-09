use crate::scalar::Scalar;

/// Marker trait for Uint types
pub trait Uint: Scalar + Into<u32> {}

impl Uint for u32 {}
/*
impl Uint for u16 {}

impl Uint for u8 {}
*/
