use crate::result::Result;

/// Dot (matrix) product.
pub trait Dot<R>: Sized {
    /// Type of the output.
    type Output;
    /// Type of the bias.
    type Bias;
    /// Computes the dot product `self` * `rhs`.
    fn dot(self, rhs: R) -> Result<Self::Output> {
        self.dot_bias(rhs, None)
    }
    /// Computes the dot product `self` * `rhs` + `bias`.
    fn dot_bias(self, rhs: R, bias: Option<Self::Bias>) -> Result<Self::Output>;
}

pub(crate) trait DotAccumulate<R>: Dot<R> {
    fn dot_acc(self, rhs: R, output: &mut Self::Output) -> Result<()> {
        self.dot_bias_acc(rhs, None, output)
    }
    fn dot_bias_acc(
        self,
        rhs: R,
        bias: Option<Self::Bias>,
        output: &mut Self::Output,
    ) -> Result<()>;
}

pub(crate) trait Add<R>: Sized {
    type Output;
    fn add(self, rhs: R) -> Result<Self::Output>;
}

pub(crate) trait Assign<R> {
    fn assign(&mut self, rhs: R) -> Result<()>;
}

pub(crate) trait AddAssign<R>: Add<R> + Assign<R> {
    fn add_assign(&mut self, rhs: R) -> Result<()>;
}

pub(crate) trait ScaledAdd<T, R>: AddAssign<R> {
    fn scaled_add(&mut self, alpha: T, rhs: R) -> Result<()>;
}
