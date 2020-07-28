use super::{Conv2dArgs, Pool2dArgs, Conv2d, Dense, MaxPool2d};
use crate::{Device, Into2d};

#[derive(Default, Clone)]
pub struct DenseBuilder {
    pub(super) device: Option<Device>,
    pub(super) inputs: Option<usize>,
    pub(super) outputs: Option<usize>,
    pub(super) use_bias: bool,
}

impl DenseBuilder {
    pub fn device(mut self, device: &Device) -> Self {
        self.device.replace(device.clone());
        self
    }
    pub fn inputs(mut self, inputs: usize) -> Self {
        debug_assert_ne!(inputs, 0);
        self.inputs.replace(inputs);
        self
    }
    pub fn outputs(mut self, outputs: usize) -> Self {
        debug_assert_ne!(outputs, 0);
        self.outputs.replace(outputs);
        self
    }
    pub fn bias(mut self) -> Self {
        self.use_bias = true;
        self
    }
    pub fn build(self) -> Dense {
        self.into()
    }
}

#[derive(Default, Clone)]
pub struct Conv2dBuilder {
    pub(super) device: Option<Device>,
    pub(super) inputs: Option<usize>,
    pub(super) outputs: Option<usize>,
    pub(super) kernel: Option<[usize; 2]>,
    pub(super) use_bias: bool,
    pub(super) args: Conv2dArgs,
}

impl Conv2dBuilder {
    pub fn device(mut self, device: &Device) -> Self {
        self.device.replace(device.clone());
        self
    }
    pub fn inputs(mut self, inputs: usize) -> Self {
        debug_assert_ne!(inputs, 0);
        self.inputs.replace(inputs);
        self
    }
    pub fn outputs(mut self, outputs: usize) -> Self {
        debug_assert_ne!(outputs, 0);
        self.outputs.replace(outputs);
        self
    }
    pub fn kernel(mut self, kernel: impl Into2d) -> Self {
        let kernel = kernel.into_2d();
        debug_assert_ne!(kernel[0] * kernel[1], 0);
        self.kernel.replace(kernel);
        self
    }
    pub fn bias(mut self) -> Self {
        self.use_bias = true;
        self
    }
    pub fn args(mut self, args: Conv2dArgs) -> Self {
        self.args = args;
        self
    }
    pub fn build(self) -> Conv2d {
        self.into()
    }
}

#[derive(Default, Clone)]
pub struct MaxPool2dBuilder {
    pub(super) args: Pool2dArgs,
}

impl MaxPool2dBuilder {
    pub fn args(mut self, args: Pool2dArgs) -> Self {
        self.args = args;
        self
    }
    pub fn build(self) -> MaxPool2d {
        self.into()
    }
}
