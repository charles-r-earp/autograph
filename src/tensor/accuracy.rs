use super::*;

impl<S: Data<Elem = u32>, D: Dimension> TensorBase<S, D> {
    /*
    pub(crate) fn accuracy<U: Uint>(&self, target: &TensorView<U, D>) -> Result<Tensor0<u32>> {
        let device = self.device();
        let mut output = Tensor::zeros(device, ())?;
        self.accuracy_with(target, &mut output.view_mut())?;
        Ok(output)
    }*/
    pub(crate) fn accuracy_with<U2: Uint>(
        &self,
        target: &TensorView<U2, D>,
        output: &mut TensorViewMut0<u32>,
    ) -> Result<()> {
        let n = self.len() as u32;
        let builder = glsl_shaders::module(&format!("accuracy_{}", U2::scalar_name()))?
            .compute_pass("main")?
            .slice(self.to_slice()?.as_slice())?
            .slice(target.to_slice()?.as_slice())?
            .slice_mut(output.as_raw_slice_mut())
            .unwrap()
            .push(n)?;
        unsafe { builder.submit([n, 1, 1]) }
    }
}
