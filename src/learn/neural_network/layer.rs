use super::autograd::{
    Parameter, Parameter1, Parameter2, Parameter4, ParameterViewMut1, ParameterViewMut2,
    ParameterViewMut4, ParameterViewMutD, Variable, Variable2, Variable4,
};
use crate::{
    buffer::{Buffer, ScalarBuffer, ScalarData, ScalarSliceMut},
    device::Device,
    krnl::krnl_core::half::bf16,
    ops::{AddAssign, Col2ImConv2, Col2ImConv2Options, Im2ColConv2, Im2ColConv2Options, MaxPool2 as _, MaxPool2Options, MaxPool2Backward as _},
    scalar::{Scalar, ScalarType},
    tensor::{ScalarArcTensor, ScalarTensor, ScalarTensorBase, Tensor, TensorView, TensorViewMut},
};
use anyhow::{bail, Result};
#[cfg(feature = "device")]
use krnl::macros::module;
use ndarray::{linalg::Dot, Array, Dimension};
use rand::{
    distributions::{Distribution, Uniform},
    thread_rng,
};
use serde::{Deserialize, Serialize};
use std::any::Any;

pub mod builder {
    use super::*;

    pub struct Conv2Builder<A = Identity> {
        inputs: usize,
        outputs: usize,
        filter: [usize; 2],
        bias: bool,
        scalar_type: ScalarType,
        device: Device,
        activation: A,
    }

    impl Conv2Builder {
        pub(super) fn new() -> Self {
            Self {
                inputs: 0,
                outputs: 0,
                filter: [0, 0],
                bias: false,
                scalar_type: ScalarType::F32,
                device: Device::host(),
                activation: Identity,
            }
        }
    }

    impl<A> Conv2Builder<A> {
        pub fn inputs(self, inputs: usize) -> Self {
            Self { inputs, ..self }
        }
        pub fn outputs(self, outputs: usize) -> Self {
            Self { outputs, ..self }
        }
        pub fn filter(self, filter: [usize; 2]) -> Self {
            Self { filter, ..self }
        }
        pub fn bias(self, bias: bool) -> Self {
            Self { bias, ..self }
        }
        pub fn activation<A2>(self, activation: A2) -> Conv2Builder<A2> {
            let Self {
                inputs,
                outputs,
                filter,
                bias,
                activation: _,
                scalar_type,
                device,
            } = self;
            Conv2Builder {
                inputs,
                outputs,
                filter,
                bias,
                activation,
                scalar_type,
                device,
            }
        }
        pub fn scalar_type(self, scalar_type: ScalarType) -> Self {
            Self {
                scalar_type,
                ..self
            }
        }
        pub fn device(self, device: Device) -> Self {
            Self { device, ..self }
        }
        pub fn build(self) -> Result<Conv2<A>> {
            let Self {
                inputs,
                outputs,
                filter,
                bias,
                activation,
                scalar_type,
                device,
            } = self;
            if !matches!(scalar_type, ScalarType::BF16 | ScalarType::F32) {
                bail!("Dense {scalar_type:?} not supported!");
            }
            let a = if inputs > 0 {
                f32::sqrt(2. / inputs as f32)
            } else {
                0.
            };
            let mut rng = thread_rng();
            let weight_iter = Uniform::new(-a, a)
                .sample_iter(&mut rng)
                .take(outputs * inputs * filter[0] * filter[1]);
            let weight = match scalar_type {
                ScalarType::BF16 => ScalarBuffer::from(Buffer::from(
                    weight_iter.map(bf16::from_f32).collect::<Vec<_>>(),
                )),
                ScalarType::F32 => {
                    ScalarBuffer::from(Buffer::from(weight_iter.collect::<Vec<_>>()))
                }
                _ => unreachable!(),
            };
            let weight = weight.into_device(device.clone())?;
            let weight = Parameter::from(
                ScalarTensor::from(weight)
                    .into_shape([outputs, inputs, filter[0], filter[1]])
                    .unwrap(),
            );
            let bias = if bias {
                let bias_iter = Uniform::new(-a, a).sample_iter(rng).take(outputs);
                let bias = match scalar_type {
                    ScalarType::BF16 => ScalarBuffer::from(Buffer::from(
                        bias_iter.map(bf16::from_f32).collect::<Vec<_>>(),
                    )),
                    ScalarType::F32 => {
                        ScalarBuffer::from(Buffer::from(bias_iter.collect::<Vec<_>>()))
                    }
                    _ => unreachable!(),
                };
                let bias = bias.into_device(device)?;
                Some(Parameter::from(ScalarTensor::from(bias)))
            } else {
                None
            };
            Ok(Conv2 {
                weight,
                bias,
                activation,
            })
        }
    }

    pub struct DenseBuilder<A = Identity> {
        inputs: usize,
        outputs: usize,
        bias: bool,
        scalar_type: ScalarType,
        device: Device,
        activation: A,
    }

    impl DenseBuilder {
        pub(super) fn new() -> Self {
            Self {
                inputs: 0,
                outputs: 0,
                bias: false,
                scalar_type: ScalarType::F32,
                device: Device::host(),
                activation: Identity,
            }
        }
    }

    impl<A> DenseBuilder<A> {
        pub fn inputs(self, inputs: usize) -> Self {
            Self { inputs, ..self }
        }
        pub fn outputs(self, outputs: usize) -> Self {
            Self { outputs, ..self }
        }
        pub fn bias(self, bias: bool) -> Self {
            Self { bias, ..self }
        }
        pub fn activation<A2>(self, activation: A2) -> DenseBuilder<A2> {
            let Self {
                inputs,
                outputs,
                bias,
                activation: _,
                scalar_type,
                device,
            } = self;
            DenseBuilder {
                inputs,
                outputs,
                bias,
                activation,
                scalar_type,
                device,
            }
        }
        pub fn scalar_type(self, scalar_type: ScalarType) -> Self {
            Self {
                scalar_type,
                ..self
            }
        }
        pub fn device(self, device: Device) -> Self {
            Self { device, ..self }
        }
        pub fn build(self) -> Result<Dense<A>> {
            let Self {
                inputs,
                outputs,
                bias,
                activation,
                scalar_type,
                device,
            } = self;
            if !matches!(scalar_type, ScalarType::BF16 | ScalarType::F32) {
                bail!("Dense {scalar_type:?} not supported!");
            }
            let a = if inputs > 0 {
                f32::sqrt(2. / inputs as f32)
            } else {
                0.
            };
            let mut rng = thread_rng();
            let weight_iter = Uniform::new(-a, a)
                .sample_iter(&mut rng)
                .take(inputs * outputs);
            let weight = match scalar_type {
                ScalarType::BF16 => ScalarBuffer::from(Buffer::from(
                    weight_iter.map(bf16::from_f32).collect::<Vec<_>>(),
                )),
                ScalarType::F32 => {
                    ScalarBuffer::from(Buffer::from(weight_iter.collect::<Vec<_>>()))
                }
                _ => unreachable!(),
            };
            let weight = weight.into_device(device.clone())?;
            let weight = Parameter::from(
                ScalarTensor::from(weight)
                    .into_shape([inputs, outputs])
                    .unwrap(),
            );
            let bias = if bias {
                let bias_iter = Uniform::new(-a, a).sample_iter(rng).take(outputs);
                let bias = match scalar_type {
                    ScalarType::BF16 => ScalarBuffer::from(Buffer::from(
                        bias_iter.map(bf16::from_f32).collect::<Vec<_>>(),
                    )),
                    ScalarType::F32 => {
                        ScalarBuffer::from(Buffer::from(bias_iter.collect::<Vec<_>>()))
                    }
                    _ => unreachable!(),
                };
                let bias = bias.into_device(device)?;
                Some(Parameter::from(ScalarTensor::from(bias)))
            } else {
                None
            };
            Ok(Dense {
                weight,
                bias,
                activation,
            })
        }
    }

    pub struct MaxPool2Builder {
        size: [usize; 2],
        strides: [usize; 2],
    }

    impl MaxPool2Builder {
        pub(super) fn new() -> Self {
            Self {
                size: [0, 0],
                strides: [1, 1],
            }
        }
        pub fn size(self, size: [usize; 2]) -> Self {
            Self { size, ..self }
        }
        pub fn strides(self, strides: [usize; 2]) -> Self {
            Self { strides, ..self }
        }
        pub fn build(self) -> MaxPool2 {
            let Self { size, strides } = self;
            MaxPool2 { size, strides }
        }
    }
}
use builder::*;

pub struct LayerMut<'a> {
    inner: &'a mut dyn Layer,
}

impl<'a> LayerMut<'a> {
    pub fn new(layer: &'a mut dyn Layer) -> Self {
        Self { inner: layer }
    }
}

impl Layer for LayerMut<'_> {
    fn set_training(&mut self, training: bool) -> Result<()> {
        self.inner.set_training(training)
    }
    fn parameters_mut(&mut self) -> Result<Vec<ParameterViewMutD>> {
        self.inner.parameters_mut()
    }
    fn layers_mut(&mut self) -> Result<Vec<LayerMut>> {
        self.inner.layers_mut()
    }
    fn to_device_mut(&mut self, device: Device) -> Result<()> {
        self.inner.to_device_mut(device)
    }
}

pub trait Layer {
    fn set_training(&mut self, training: bool) -> Result<()> {
        for mut layer in self.layers_mut()? {
            layer.set_training(training)?;
        }
        Ok(())
    }
    fn parameters_mut(&mut self) -> Result<Vec<ParameterViewMutD>> {
        todo!()
    }
    fn layers_mut(&mut self) -> Result<Vec<LayerMut>> {
        Ok(Vec::new())
    }
    fn to_device_mut(&mut self, device: Device) -> Result<()> {
        for layer in self.layers_mut()?.iter_mut() {
            layer.to_device_mut(device.clone())?;
        }
        Ok(())
    }
}

pub trait Forward<X> {
    type Output;
    fn forward(&self, input: X) -> Result<Self::Output>;
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Conv2<A = Identity> {
    weight: Parameter4,
    bias: Option<Parameter1>,
    activation: A,
}

impl Conv2 {
    pub fn builder() -> Conv2Builder {
        Conv2Builder::new()
    }
    pub fn weight_view_mut(&mut self) -> ParameterViewMut4 {
        todo!()
    }
    pub fn bias_view_mut(&mut self) -> Result<Option<ParameterViewMut1>> {
        todo!()
    }
    pub fn into_device(self, _device: Device) -> Result<Self> {
        todo!()
    }
    pub fn to_device_mut(&mut self, _device: Device) -> Result<()> {
        todo!()
    }
}

impl<A> Layer for Conv2<A> {
    fn set_training(&mut self, training: bool) -> Result<()> {
        self.weight.set_training(training);
        if let Some(bias) = self.bias.as_mut() {
            bias.set_training(training);
        }
        Ok(())
    }
    fn parameters_mut(&mut self) -> Result<Vec<ParameterViewMutD>> {
        let mut parameters = Vec::new();
        parameters.push(self.weight.make_view_mut()?.into_dyn());
        if let Some(bias) = self.bias.as_mut() {
            parameters.push(bias.make_view_mut()?.into_dyn());
        }
        Ok(parameters)
    }
}

impl<A: Forward<Variable4, Output = Variable4>> Forward<Variable4> for Conv2<A> {
    type Output = Variable4;
    fn forward(&self, input: Variable4) -> Result<Variable4> {
        let (batch_size, inputs, ih, iw) = input.dim();
        let (outputs, inputs2, fh, fw) = self.weight.dim();
        assert_eq!(inputs, inputs2);
        let options = Im2ColConv2Options {
            filter: [fh, fw],
            ..Im2ColConv2Options::default()
        };
        let [oh, ow] = options.output_shape([ih, iw]);
        let im2col_matrix = input.value().im2col_conv2(options)?;
        let weight_matrix = self
            .weight
            .value()
            .clone()
            .into_shape([outputs, inputs * fh * fw])
            .unwrap();
        let output_matrix = im2col_matrix.dot(&weight_matrix.t())?;
        let mut builder = Variable::builder();
        if let Some(node) = input.node() {
            let weight_matrix = weight_matrix.clone();
            builder.edge(node, move |output_grad| {
                let options = Col2ImConv2Options {
                    shape: [oh, ow],
                    filter: [fh, fw],
                    ..Col2ImConv2Options::default()
                };
                output_grad
                    .dot(&weight_matrix)?
                    .col2im_conv2(options)
                    .map(Into::into)
            });
        }
        let weight = self.weight.to_variable();
        if let Some(node) = weight.node() {
            builder.edge(node, move |output_grad| {
                let weight_grad = output_grad
                    .t()
                    .dot(&im2col_matrix)?
                    .into_shape([outputs, inputs, fh, fw])
                    .unwrap();
                Ok(weight_grad.into())
            });
        }
        let output_matrix = builder.build(output_matrix.into());
        let mut builder = Variable::builder();
        if let Some(node) = output_matrix.node() {
            builder.edge(node, move |output_grad| {
                Ok(output_grad
                    .permuted_axes([0, 2, 3, 1])
                    .into_owned()?
                    .into_shape([batch_size * oh * ow, outputs])
                    .unwrap()
                    .into())
            });
        }
        let output = output_matrix
            .value()
            .view()
            .into_shape([batch_size, oh, ow, outputs])
            .unwrap()
            .permuted_axes([0, 3, 1, 2])
            .to_owned()?;
        let output = builder.build(output.into());
        self.activation.forward(output)
    }
}

///```no_run
/// # fn main() -> anyhow::Result<()> {
/// # let device = Device::host();
/// let dense = Dense::builder()
///    .inputs(1)
///    .outputs(1)
///    .bias(true))
///    .activation(Relu)
///    .scalar_type(ScalarType::BF16)
///    .device(device.clone())
///    .build()?;
/// # }
///```
#[derive(Debug, Serialize, Deserialize)]
pub struct Dense<A = Identity> {
    weight: Parameter2,
    bias: Option<Parameter1>,
    activation: A,
}

impl Dense {
    pub fn builder() -> DenseBuilder {
        DenseBuilder::new()
    }
}

impl<A> Dense<A> {
    pub fn weight_view_mut(&mut self) -> ParameterViewMut2 {
        todo!()
    }
    pub fn bias_view_mut(&mut self) -> Result<Option<ParameterViewMut1>> {
        todo!()
    }
    pub fn into_device(self, _device: Device) -> Result<Self> {
        todo!()
    }
    pub fn to_device_mut(&mut self, _device: Device) -> Result<()> {
        todo!()
    }
}

impl<A> Layer for Dense<A> {
    fn set_training(&mut self, training: bool) -> Result<()> {
        self.weight.set_training(training);
        if let Some(bias) = self.bias.as_mut() {
            bias.set_training(training);
        }
        Ok(())
    }
    fn parameters_mut(&mut self) -> Result<Vec<ParameterViewMutD>> {
        let mut parameters = Vec::new();
        parameters.push(self.weight.make_view_mut()?.into_dyn());
        if let Some(bias) = self.bias.as_mut() {
            parameters.push(bias.make_view_mut()?.into_dyn());
        }
        Ok(parameters)
    }
}

impl<A: Forward<Variable2, Output = Variable2> + Any> Forward<Variable2> for Dense<A> {
    type Output = Variable2;
    fn forward(&self, input: Variable2) -> Result<Self::Output> {
        let mut output = input.dot(&self.weight.to_variable())?;
        if let Some(bias) = self.bias.as_ref() {
            output.add_assign(&bias.to_variable())?;
        }
        self.activation.forward(output)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MaxPool2 {
    size: [usize; 2],
    strides: [usize; 2],
}

impl MaxPool2 {
    pub fn builder() -> MaxPool2Builder {
        MaxPool2Builder::new()
    }
}

impl Forward<Variable4> for MaxPool2 {
    type Output = Variable4;
    fn forward(&self, input: Variable4) -> Result<Self::Output> {
        let options = MaxPool2Options {
            size: self.size,
            strides: self.strides,
        };
        let mut builder = Variable::builder();
        if let Some(node) = input.node() {
            let mut input = input.value().clone();
            let options = MaxPool2Options {
                size: self.size,
                strides: self.strides,
            };
            builder.edge(node, |output_grad| {
                input.make_view_mut()?.max_pool2_backward(output_grad, options)?;
                Ok(input)
            });
        }
        let output = input.value().max_pool2(options)?;
        Ok(builder.build(output.into()))
    }
}

#[derive(Default, Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Identity;

impl<X> Forward<X> for Identity {
    type Output = X;
    fn forward(&self, input: X) -> Result<Self::Output> {
        Ok(input)
    }
}

#[derive(Default, Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Relu;

impl<D: Dimension + 'static> Forward<Variable<D>> for Relu {
    type Output = Variable<D>;
    fn forward(&self, input: Variable<D>) -> Result<Self::Output> {
        let mut builder = Variable::builder();
        if let Some(node) = input.node() {
            let input = input.value().clone();
            builder.edge(&node, move |output_grad| {
                scalar_relu_backward(input, output_grad)
            });
        }
        Ok(builder.build(scalar_relu(input.into_value())?.into()))
    }
}

fn scalar_relu<S: ScalarData, D: Dimension>(
    mut input: ScalarTensorBase<S, D>,
) -> Result<ScalarArcTensor<D>> {
    let scalar_type = input.scalar_type();
    if input.is_standard_layout() {
        if let Some(input_mut) = input.get_view_mut() {
            match scalar_type {
                ScalarType::F32 => {
                    relu_mut::<f32, D>(input_mut.try_into().unwrap())?;
                }
                _ => todo!(),
            }
            return input.into_shared();
        }
    }
    match scalar_type {
        ScalarType::F32 => Ok(relu::<f32, D>(input.view().try_into().unwrap())?
            .into_shared()?
            .into()),
        _ => todo!(),
    }
}

fn relu_mut<T: Scalar, D: Dimension>(mut input: TensorViewMut<T, D>) -> Result<()> {
    if let Some(mut x) = input.as_array_mut() {
        for x in x.iter_mut() {
            *x = relu_impl(*x);
        }
        return Ok(());
    }
    #[cfg(not(feature = "device"))]
    {
        unreachable!()
    }
    #[cfg(feature = "device")]
    {
        let device = input.device();
        let mut x = input.as_slice_mut().unwrap();
        if let Ok(x) = x.as_scalar_slice_mut().try_into() {
            kernels::relu_mut_f32::builder()?
                .build(device)?
                .dispatch(x)?;
            return Ok(());
        }
        todo!()
    }
}

fn relu<T: Scalar, D: Dimension>(input: TensorView<T, D>) -> Result<Tensor<T, D>> {
    if let Some(x) = input.as_array() {
        let y = x.map(|x| relu_impl(*x));
        return Ok(y.into());
    }
    #[cfg(not(feature = "device"))]
    {
        unreachable!()
    }
    #[cfg(feature = "device")]
    {
        let mut output = unsafe { Tensor::<T, D>::uninit(input.device(), input.raw_dim())? };
        let x = input.as_slice().unwrap();
        let mut y = output.as_slice_mut().unwrap();
        if let Ok(x) = x.as_scalar_slice().try_into() {
            let y = y.as_scalar_slice_mut().try_into().unwrap();
            kernels::relu_f32::builder()?
                .build(input.device())?
                .dispatch(x, y)?;
            return Ok(output);
        }
        todo!()
    }
}

fn scalar_relu_backward<D: Dimension>(
    output: ScalarArcTensor<D>,
    mut output_grad: ScalarArcTensor<D>,
) -> Result<ScalarArcTensor<D>> {
    let scalar_type = output.scalar_type();
    if let Some(output_grad_mut) = output_grad.get_view_mut() {
        match scalar_type {
            ScalarType::F32 => {
                relu_backward_mut::<f32, D>(
                    output.view().try_into().unwrap(),
                    output_grad_mut.try_into().unwrap(),
                )?;
            }
            _ => todo!(),
        }
        return Ok(output_grad);
    }
    todo!()
}

fn relu_backward_mut<T: Scalar, D: Dimension>(
    input: TensorView<T, D>,
    mut output_grad: TensorViewMut<T, D>,
) -> Result<()> {
    if let Some((x, mut dy)) = input.as_array().zip(output_grad.as_array_mut()) {
        dy.zip_mut_with(&x, |dy, x| {
            *dy = relu_backward_impl(*x, *dy);
        });
        return Ok(());
    }
    #[cfg(not(feature = "device"))]
    {
        unreachable!()
    }
    #[cfg(feature = "device")]
    {
        let x = input.as_slice().unwrap();
        let mut dy = output_grad.as_slice_mut().unwrap();
        if let Some((x, dy)) = x
            .as_scalar_slice()
            .try_into()
            .ok()
            .zip(dy.as_scalar_slice_mut().try_into().ok())
        {
            kernels::relu_backward_mut_f32::builder()?
                .build(input.device())?
                .dispatch(x, dy)?;
            return Ok(());
        }
        todo!()
    }
}

fn relu_backward<T: Scalar, D: Dimension>(
    input: TensorView<T, D>,
    output_grad: TensorView<T, D>,
) -> Result<Tensor<T, D>> {
    if let Some((x, dy)) = input.as_array().zip(output_grad.as_array()) {
        let dx: Vec<T> = x
            .iter()
            .copied()
            .zip(dy.iter().copied())
            .map(|(x, dy)| relu_backward_impl(x, dy))
            .collect();
        return Ok(Array::from(dx).into_shape(input.raw_dim()).unwrap().into());
    }
    #[cfg(not(feature = "device"))]
    {
        unreachable!()
    }
    #[cfg(feature = "device")]
    {
        let x = input.as_slice().unwrap();
        let dy = output_grad.as_slice().unwrap();
        let mut input_grad = unsafe { Tensor::uninit(input.device(), input.raw_dim())? };
        if let Some((x, dy)) = x
            .as_scalar_slice()
            .try_into()
            .ok()
            .zip(dy.as_scalar_slice().try_into().ok())
        {
            let dx = ScalarSliceMut::from(input_grad.as_slice_mut().unwrap())
                .try_into()
                .unwrap();
            kernels::relu_backward_f32::builder()?
                .build(input.device())?
                .dispatch(x, dy, dx)?;
            return Ok(input_grad);
        }
        todo!()
    }
}

#[cfg_attr(feature = "device", module)]
mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::{macros::kernel, scalar::Scalar};

    pub fn relu_impl<T: Scalar>(x: T) -> T {
        if x >= T::zero() {
            x
        } else {
            T::zero()
        }
    }

    pub fn relu_backward_impl<T: Scalar>(x: T, dy: T) -> T {
        if x >= T::zero() {
            dy
        } else {
            T::zero()
        }
    }

    #[cfg(any(feature = "device", target_arch = "spirv"))]
    pub mod device {
        use super::*;

        #[kernel(threads(256))]
        pub fn relu_mut_f32(#[item] x: &mut f32) {
            *x = relu_impl(*x);
        }

        #[kernel(threads(256))]
        pub fn relu_f32(#[item] x: f32, #[item] y: &mut f32) {
            *y = relu_impl(x);
        }

        #[kernel(threads(256))]
        pub fn relu_backward_mut_f32(#[item] x: f32, #[item] dy: &mut f32) {
            *dy = relu_backward_impl(x, *dy);
        }

        #[kernel(threads(256))]
        pub fn relu_backward_f32(#[item] x: f32, #[item] dy: f32, #[item] dx: &mut f32) {
            *dx = relu_backward_impl(x, dy);
        }
    }
    #[cfg(feature = "device")]
    pub use device::*;
}
use kernels::{relu_backward_impl, relu_impl};
