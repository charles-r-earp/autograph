use super::*;
use crate::ops::{AddAssign, ScaledAdd};
use anyhow::ensure;

impl<T: Scalar, S1: DataMut<Elem = T>, S2: Data<Elem = T>, D: Dimension>
    AddAssign<TensorBase<S2, D>> for TensorBase<S1, D>
{
    fn add_assign(&mut self, rhs: &TensorBase<S2, D>) -> Result<()> {
        self.scaled_add(T::one(), rhs)
    }
}

impl<T: Scalar, S1: DataMut<Elem = T>, S2: Data<Elem = T>, D: Dimension>
    ScaledAdd<T, TensorBase<S2, D>> for TensorBase<S1, D>
{
    fn scaled_add(&mut self, alpha: T, rhs: &TensorBase<S2, D>) -> Result<()> {
        ensure!(self.dim == rhs.dim);
        ensure!(self.strides == rhs.strides);
        let n = self.len() as u32;
        let builder = glsl_shaders::module(&format!("scaled_add_{}", T::scalar_name()))?
            .compute_pass("main")?
            .slice_mut(self.as_raw_slice_mut())?
            .slice(rhs.as_raw_slice())?
            .push(n)?
            .push(alpha)?;
        unsafe { builder.submit([n, 1, 1]) }
    }
}
