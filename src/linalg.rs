use crate::result::Result;

/// Dot (matrix) product.
pub(crate) trait Dot<R> {
    /// Type of the output.
    type Output;
    /// Computes the dot product `self` * `rhs`.
    fn dot(&self, rhs: &R) -> Result<Self::Output>;
}

/*
pub(crate) trait DotBias<R, B>: Dot<R> {
    /// Computes the dot product `self` * `rhs` + `bias`.
    fn dot_bias(&self, rhs: &R, bias: Option<&B>) -> Result<Self::Output>;
}

pub(crate) trait DotAcc<R>: Dot<R> {
    fn dot_acc(&self, rhs: &R, output: &mut Self::Output) -> Result<()>;
}

pub(crate) trait DotBiasAcc<R, B>: DotBias<R, B> {
    fn dot_bias_acc(
        &self,
        rhs: &R,
        bias: Option<&B>,
        output: &mut Self::Output,
    ) -> Result<()>;
}
*/
