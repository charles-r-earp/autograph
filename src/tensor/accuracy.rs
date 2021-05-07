use super::{Result, Data, Unsigned, TensorBase, TensorView1, Tensor, Tensor0, TensorViewMut0, Ix1};
use crate::util::type_eq;

impl<S: Data<Elem=u32>> TensorBase<S, Ix1> {
    pub fn accuracy<U2: Unsigned>(&self, target: &TensorView1<U2>) -> Result<Tensor0<u32>> {
        let device = self.device();
        let mut output = Tensor::zeros(device, ())?;
        self.accuracy_mut(target, &mut output.view_mut())?;
        Ok(output)
    }
    pub fn accuracy_mut<U2: Unsigned>(&self, target: &TensorView1<U2>, output: &mut TensorViewMut0<u32>) -> Result<()> {
        let device = self.device();
        let src = if type_eq::<U2, u8>() {
            include_shader!("glsl/accuracy_u8.spv")
        } else if type_eq::<U2, u32>() {
            include_shader!("glsl/accuracy_u32.spv")
        } else {
            unreachable!()
        };

        let n = self.dim() as u32;
        device
            .compute_pass(&src, "main")?
            .buffer_slice(self.as_buffer_slice().unwrap())?
            .buffer_slice(target.as_buffer_slice().unwrap())?
            .buffer_slice_mut(output.as_unordered_buffer_slice_mut())?
            .push_constants(bytemuck::cast_slice(&[n]))?
            .global_size([n, 1, 1])
            .enqueue()
    }
}
