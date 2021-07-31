use super::*;
use crate::{
    result::Result,
    linalg::Dot,
    tensor::{Tensor, Tensor1, Tensor2, Ix1, Ix2},
};
use anyhow::anyhow;


impl Dot<FloatTensor<Ix2>> for FloatTensor<Ix2> {
    type Output = Self;
    type Bias = FloatTensor<Ix1>;
    fn dot_bias(self, rhs: FloatTensor<Ix2>, bias: Option<Self::Bias>) -> Result<Self::Output> {
        match self.data.float_type() {
            FloatType::BF16 => {
                let lhs = Tensor2::<bf16>::try_from(self).ok().unwrap();
                let rhs = Tensor2::<bf16>::try_from(rhs).map_err(|_| anyhow!("Unimplemented!"))?;
                let bias = if let Some(bias) = bias {
                    Some(Tensor1::<bf16>::try_from(bias).map_err(|_| anyhow!("Unimplemented!"))?)
                } else {
                    None
                };
                Ok(lhs.dot_bias(rhs.view(), bias.as_ref().map(Tensor::view))?.into())
            }
            _ => todo!()
        }
    }
}
