use super::*;
use crate::{
    ops::{AddAssign, Im2Col, KernelArgs, KernelKind, ScaledAdd},
    rust_shaders,
    scalar::Float,
};
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

impl<T: Float, S: Data<Elem = T>> Im2Col<Ix2> for TensorBase<S, Ix4> {
    type Output = Tensor4<T>;
    fn im2col(
        &self,
        kernel: &Ix2,
        kind: KernelKind,
        args: &KernelArgs<Ix2>,
    ) -> Result<Self::Output> {
        let (kh, kw) = kernel.into_pattern();
        let (bs, ic, ih, iw) = self.dim();
        let strides = &args.strides;
        let padding = &args.padding;
        let dilation = &args.dilation;
        let oh = (ih + 2 * padding[0] - dilation[0] * (kh - 1) - 1) / strides[0] + 1;
        let ow = (iw + 2 * padding[1] - dilation[1] * (kw - 1) - 1) / strides[1] + 1;
        let mut output = Tensor::zeros(self.device(), [bs, ic * kh * kw, oh, ow])?;
        let input_slice = self.to_slice()?;
        let builder = rust_shaders::core()?
            .compute_pass(&format!(
                "kernel::im2col_2d_{}_{}",
                kind.as_str(),
                T::scalar_name()
            ))?
            .slice(input_slice.as_slice())?
            .slice_mut(output.as_raw_slice_mut())?
            .push(ic as u32)?
            .push([ih as u32, iw as u32])?
            .push([oh as u32, ow as u32])?
            .push([kh as u32, kw as u32])?
            .push([strides[0] as u32, strides[1] as u32])?
            .push([padding[0] as u32, padding[1] as u32])?
            .push([dilation[0] as u32, dilation[1] as u32])?;
        unsafe {
            builder.submit([iw as u32, (ic * ic) as u32, 1])?;
        }
        Ok(output)
    }
}

/*
impl<T: Float, S: Data<Elem=T>, D: Dimension> Col2Im<D> for TensorBase<S, Ix2> {
    type Output = Tensor<T, <<D as ndarray::Dimension>::Larger as Dimension>::Larger>;
    fn col2im(&self, args: &ConvArgs<D>) -> Result<Self::Output> {
        todo!()
    }
}
*/
