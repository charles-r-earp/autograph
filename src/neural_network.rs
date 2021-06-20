use crate::{
    backend::Device,
    learn::{Fit, Predict},
    tensor::{
        float::{
            float_gemm, FloatArcTensor, FloatArcTensor2, FloatTensor, FloatTensor2,
            FloatTensorViewMut,
        },
        Axis, Data, Dimension, Float, Ix1, Ix2, Tensor, Tensor0, Tensor1, Tensor2, TensorBase,
        TensorView0, TensorView2, TensorViewD, TensorViewMut1, TensorViewMut2, TensorViewMutD,
        Unsigned,
    },
    util::type_eq,
    Result,
};
use anyhow::{bail, ensure};
use half::bf16;
use serde::{Deserialize, Serialize};
use std::{
    fmt::{self, Debug},
    future::Future,
    pin::Pin,
};

pub use autograph_derive::{Forward, Network};

pub mod autograd;
use autograd::{
    Backward, GradientD, GradientEntry, GradientVec, Parameter1, Parameter2, ParameterMutD,
    Variable, Variable0, Variable2, VariableD,
};

pub mod builders;
use builders::{DenseBuilder, SgdBuilder};

#[cfg(test)]
mod tests;

// for data parallel, a special DataParallel layer that adds ops to compute average gradients
// for distr data parallel, a special DistrDataParallel layer

pub trait Optimizer {
    fn update<'a>(
        &mut self,
        parameters: impl ExactSizeIterator<Item = ParameterMutD<'a>>,
    ) -> Result<()>;
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Sgd {
    learning_rate: f32,
    momentum: f32,
    // velocities
}

impl Default for Sgd {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Sgd {
    pub fn builder() -> SgdBuilder {
        SgdBuilder::default()
    }
}

fn sgd<T: Float>(value: &mut TensorViewMutD<T>, grad: &TensorViewD<T>, alpha: T) -> Result<()> {
    value.scaled_add(alpha, grad)
}

impl Optimizer for Sgd {
    fn update<'a>(
        &mut self,
        parameters: impl ExactSizeIterator<Item = ParameterMutD<'a>>,
    ) -> Result<()> {
        if self.momentum > 0. {
            todo!()
        } else {
            for mut parameter in parameters {
                if let Some(parameter_grad) = parameter.take_grad() {
                    let parameter_grad = parameter_grad.into_dense()?;
                    let mut parameter = parameter.make_mut()?;
                    let parameter = parameter.value_view_mut();
                    match parameter {
                        FloatTensorViewMut::BF16(mut parameter) => sgd(
                            &mut parameter,
                            &parameter_grad.cast_to()?.view(),
                            bf16::from_f32(-self.learning_rate),
                        )?,
                        FloatTensorViewMut::F32(mut parameter) => sgd(
                            &mut parameter,
                            &parameter_grad.cast_to()?.view(),
                            -self.learning_rate,
                        )?,
                    }
                }
            }
            Ok(())
        }
    }
}

/// Forward is a trait for Neural Networks and layers.\
///
/// Can be [derived](`autograph_derive`).
pub trait Forward {
    /// Forward pass\
    ///
    /// If [input.training()](`Variable::training`), the implementation should apply a [`Backward`] with\
    /// [`Variable::builder`].
    fn forward(&self, input: VariableD) -> Result<VariableD>;
    /// Mutable forward pass\
    ///
    /// Like [forward](`Forward::forward`), but can initialize or update parameters (like for normalization).
    fn forward_mut(&mut self, input: VariableD) -> Result<VariableD> {
        self.forward(input)
    }
}

/// Network is a trait for Neural Networks and layers.\
///
/// Can be [derived](`autograph_derive`).
pub trait Network: Forward {
    /// Returns the total number of parameters.
    fn parameters_count(&self) -> usize {
        0
    }
    /// Collects all of the parameters as ParameterMutD's.
    #[allow(unused_variables)]
    fn collect_parameters_mut<'a>(&'a mut self, parameters: &mut Vec<ParameterMutD<'a>>) {}
    /// Returns the parameters of the network as [`ParameterMutD`].
    fn parameters_mut(&mut self) -> Vec<ParameterMutD> {
        let mut parameters = Vec::with_capacity(self.parameters_count());
        self.collect_parameters_mut(&mut parameters);
        parameters
    }
    /// Returns mutable references to the layers of the network\
    ///
    /// Composite layers should return their immediate children.
    fn layers_mut(&mut self) -> Vec<&mut dyn Network> {
        Vec::new()
    }
    /// Moves the network's data to the device in place.\
    ///
    /// This method does not need to be implemented. Combine with layers_mut to transfer layers to\
    /// different devices.
    #[allow(clippy::wrong_self_convention)]
    fn to_device_mut(&mut self, device: &Device) -> Result<()> {
        smol::block_on(async {
            for mut parameter in self.parameters_mut() {
                parameter.to_device_mut(device)?.await?;
            }
            Ok(())
        })
    }
    /// Moves the network's data to the device.\
    ///
    /// Generally this method should not be implemented, the implementation calls to_device_mut.
    fn into_device(mut self, device: &Device) -> Result<Self>
    where
        Self: Sized,
    {
        self.to_device_mut(device)?;
        Ok(self)
    }
}

#[derive(Default, Debug, Clone, Copy, Network, Forward, Serialize, Deserialize)]
// This replaces ::autograph with crate
#[autograph(crate)]
pub struct Identity;

#[derive(Network, Clone, Serialize, Deserialize)]
#[autograph(crate)]
#[serde(bound(serialize = "A: Serialize", deserialize = "A: Deserialize<'de>"))]
pub struct Dense<A = Identity> {
    #[autograph(parameter)]
    weight: Parameter2,
    #[autograph(optional_parameter)]
    bias: Option<Parameter1>,
    #[autograph(layer)]
    activation: A,
}

impl Dense<Identity> {
    pub fn builder() -> DenseBuilder<Identity> {
        DenseBuilder::default()
    }
}

impl<A: Debug> Debug for Dense<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Dense")
            .field("weight", &self.weight)
            .field("bias", &self.bias)
            .field("activation", &self.activation)
            .finish()
    }
}

// Note: 'static needed for specialization via type_eq
impl<A: Forward + 'static> Forward for Dense<A> {
    fn forward(&self, input: VariableD) -> Result<VariableD> {
        // maybe auto flatten?
        let input = input.into_dimensionality()?;
        let output = {
            /* specialization of fuzed ops
            if type_eq::<A, Relu>() {
                todo!()
            }*/
            input.dense(&self.weight, self.bias.as_ref())?
        };
        self.activation.forward(output.into_dyn())
    }
    fn forward_mut(&mut self, input: VariableD) -> Result<VariableD> {
        let weight_value = self.weight.value();
        let (outputs, inputs) = weight_value.dim();
        if inputs == 0 {
            let inputs = input.value().shape()[1..].iter().product();
            self.weight = builders::dense_weight(
                weight_value.device(),
                weight_value.float_type(),
                (outputs, inputs),
            )?;
        }
        // TODO: avoid dup of forward impl here, note that we need to call forward_mut on\
        // activation because it may be a normalization layer
        let input = input.into_dimensionality()?;
        let output = {
            /* specialization of fuzed ops
            if type_eq::<A, Relu>() {
                todo!()
            }*/
            input.dense(&self.weight, self.bias.as_ref())?
        };
        self.activation.forward_mut(output.into_dyn())
    }
}

fn bias_backward<T: Float>(
    bias_grad: &mut TensorViewMut1<T>,
    output_grad: &TensorView2<T>,
) -> Result<()> {
    let device = output_grad.device();
    let (n, c) = output_grad.dim();
    ensure!(bias_grad.dim() == c);
    let src = if type_eq::<T, bf16>() {
        include_shader!("glsl/bias_backward_bf16.spv")
    } else if type_eq::<T, f32>() {
        include_shader!("glsl/bias_backward_f32.spv")
    } else {
        unreachable!()
    };
    device
        .compute_pass(src, "main")?
        // TODO: covert unwrap panics into errors
        .buffer_slice_mut(bias_grad.as_unordered_buffer_slice_mut())?
        .buffer_slice(output_grad.as_buffer_slice().unwrap())?
        .push_constants(bytemuck::cast_slice(&[n as u32, c as u32]))?
        .global_size([c as u32, 1, 1])
        .enqueue()
}

struct DenseBackward {
    input: FloatArcTensor2,
    weight: Parameter2,
    bias: Option<Parameter1>,
    output_grad: Option<FloatTensor2>,
}

impl Backward for DenseBackward {
    fn backward(&mut self, input_grads: GradientVec, output_grad: GradientD) -> Result<()> {
        let [input_grad] = input_grads.try_into_array()?;
        let output_grad = output_grad.into_dimensionality()?.into_dense()?;
        if let Some(input_grad) = input_grad {
            let input_grad = input_grad.into_dimensionality()?;
            let weight = self.weight.value_view();
            let (beta, mut input_grad) = match input_grad {
                GradientEntry::Occupied(input_grad) => (1., input_grad.into_dense_view_mut()?),
                GradientEntry::Vacant(input_grad) => {
                    (0., unsafe { input_grad.into_dense_uninitialized()? })
                }
            };
            float_gemm(1., &output_grad.view(), &weight, beta, &mut input_grad)?;
        }
        self.output_grad.replace(output_grad);
        Ok(())
    }
    fn backward_parameters(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + Sync + 'static>> {
        let input = self.input.clone();
        let weight = self.weight.clone();
        let bias = self.bias.clone();
        let output_grad = self.output_grad.take().unwrap();
        Box::pin(async move {
            let mut weight_grad = weight.grad_lock().await;
            let (beta, mut weight_grad) = match weight_grad.entry() {
                GradientEntry::Occupied(weight_grad) => (1., weight_grad.into_dense_view_mut()?),
                GradientEntry::Vacant(weight_grad) => {
                    (0., unsafe { weight_grad.into_dense_uninitialized()? })
                }
            };
            float_gemm(1., &output_grad.t(), &input.view(), beta, &mut weight_grad)?;
            if let Some(bias) = bias {
                let mut bias_grad = bias.grad_lock().await;
                // TODO: impl with beta
                let (_beta, bias_grad) = match bias_grad.entry() {
                    GradientEntry::Occupied(bias_grad) => (1., bias_grad.into_dense_view_mut()?),
                    GradientEntry::Vacant(bias_grad) => {
                        (0., unsafe { bias_grad.into_dense_uninitialized()? })
                    }
                };
                match bias_grad {
                    FloatTensorViewMut::BF16(mut bias_grad) => {
                        bias_backward(&mut bias_grad, &output_grad.cast_to()?.view())?;
                    }
                    FloatTensorViewMut::F32(mut bias_grad) => {
                        bias_backward(&mut bias_grad, &output_grad.cast_to()?.view())?;
                    }
                }
            }
            Ok(())
        })
    }
}

impl Variable2 {
    pub fn dense(self, weight: &Parameter2, bias: Option<&Parameter1>) -> Result<Self> {
        Ok(Variable2::builder([self.graph()])
            .parameterized()
            .with_backward(|| {
                Box::new(DenseBackward {
                    input: self.value().clone(),
                    weight: weight.clone(),
                    bias: bias.map(Clone::clone),
                    output_grad: None,
                })
            })
            .build(
                self.into_value()
                    .mm_bias(
                        &weight.value_view().t(),
                        bias.map(Parameter1::value_view).as_ref(),
                    )?
                    .into(),
            ))
    }
}

#[allow(unused_variables)]
fn cross_entropy_loss<T: Float>(
    input: &TensorView2<T>,
    target: &TensorView2<T>,
) -> Result<Tensor2<T>> {
    let device = input.device();
    ensure!(
        input.dim() == target.dim(),
        "{:?} != {:?}",
        input.dim(),
        target.dim()
    );
    let (n, nclasses) = input.dim();
    let mut output = unsafe { Tensor::uninitialized(device, [n, nclasses])? };
    let src = if type_eq::<T, bf16>() {
        if nclasses <= 64 {
            include_shader!("glsl/cross_entropy_loss_bf16_64.spv")
        } else if nclasses <= 256 {
            include_shader!("glsl/cross_entropy_loss_bf16_256.spv")
        } else if nclasses <= 1024 {
            include_shader!("glsl/cross_entropy_loss_bf16_1024.spv")
        } else {
            bail!("nclasses > 1024 unimplemented!")
        }
    } else if type_eq::<T, f32>() {
        if nclasses <= 64 {
            include_shader!("glsl/cross_entropy_loss_f32_64.spv")
        } else if nclasses <= 256 {
            include_shader!("glsl/cross_entropy_loss_f32_256.spv")
        } else if nclasses <= 1024 {
            include_shader!("glsl/cross_entropy_loss_f32_1024.spv")
        } else {
            bail!("nclasses > 1024 unimplemented!")
        }
    } else {
        unreachable!()
    };
    device
        .compute_pass(src, "main")?
        // TODO: covert unwrap panics into errors
        .buffer_slice(input.as_buffer_slice().unwrap())?
        .buffer_slice(target.as_buffer_slice().unwrap())?
        .buffer_slice_mut(output.as_unordered_buffer_slice_mut())?
        .push_constants(bytemuck::cast_slice(&[n as u32, nclasses as u32]))?
        .global_size([n as u32, 1, 1])
        .enqueue()?;
    Ok(output)
}

fn cross_entropy_loss_backward<T: Float>(
    input: &TensorView2<T>,
    input_grad: &mut TensorViewMut2<T>,
    target: &TensorView2<T>,
    output_grad: &TensorView0<T>,
) -> Result<()> {
    let device = input.device();
    let n = input.dim().0 as u32;
    let src = if type_eq::<T, bf16>() {
        include_shader!("glsl/cross_entropy_loss_backward_bf16.spv")
    } else if type_eq::<T, f32>() {
        include_shader!("glsl/cross_entropy_loss_backward_f32.spv")
    } else {
        unreachable!()
    };
    device
        .compute_pass(src, "main")?
        // TODO: covert unwrap panics into errors
        .buffer_slice(input.as_buffer_slice().unwrap())?
        .buffer_slice_mut(input_grad.as_buffer_slice_mut().unwrap())?
        .buffer_slice(target.as_buffer_slice().unwrap())?
        .buffer_slice(output_grad.as_buffer_slice().unwrap())?
        .push_constants(bytemuck::cast_slice(&[n]))?
        .global_size([n, 1, 1])
        .enqueue()
}

struct CrossEntropyLossBackward {
    input: FloatArcTensor2,
    target: FloatArcTensor2,
}

impl Backward for CrossEntropyLossBackward {
    fn backward(&mut self, input_grads: GradientVec, output_grad: GradientD) -> Result<()> {
        let [input_grad] = input_grads.try_into_array()?;
        if let Some(input_grad) = input_grad {
            let output_grad = output_grad.into_dense()?.into_dimensionality()?;
            match input_grad.or_dense_zeroed()? {
                FloatTensorViewMut::BF16(input_grad) => cross_entropy_loss_backward(
                    &self.input.cast_to()?.view(),
                    &mut input_grad.into_dimensionality()?,
                    &self.target.cast_to()?.view(),
                    &output_grad.cast_to()?.view(),
                )?,
                FloatTensorViewMut::F32(input_grad) => cross_entropy_loss_backward(
                    &self.input.cast_to()?.view(),
                    &mut input_grad.into_dimensionality()?,
                    &self.target.cast_to()?.view(),
                    &output_grad.cast_to()?.view(),
                )?,
            }
        }
        Ok(())
    }
}

impl Variable2 {
    pub fn cross_entropy_loss(self, target: FloatArcTensor2) -> Result<Variable0> {
        Ok(Variable0::builder([self.graph()])
            .with_backward(|| {
                Box::new(CrossEntropyLossBackward {
                    input: self.value().clone(),
                    target: target.clone(),
                })
            })
            .build(match self.into_value() {
                FloatArcTensor::BF16(input) => {
                    cross_entropy_loss(&input.view(), &target.cast_to()?.view())?
                        .into_shape(input.len())?
                        .sum(Axis(0))?
                        .into_shared()?
                        .into()
                }
                FloatArcTensor::F32(input) => {
                    cross_entropy_loss(&input.view(), &target.cast_to()?.view())?
                        .into_shape(input.len())?
                        .sum(Axis(0))?
                        .into_shared()?
                        .into()
                }
            }))
        /*
        let mut output = self.forward_op(|input| {
            cross_entropy_loss::<T>(&input, &target.view())?
                .into_shape(self.value().len())?
                .sum(Axis(0))
        })?;
        // TODO: move out of variable here
        let input_value: ArcTensor2<T> = self.value().clone().try_into()?;
        output.backward_op(move |mut input_grad, output_grad| {
            cross_entropy_loss_backward(
                &input_value.view(),
                &mut input_grad,
                &target.view(),
                &output_grad,
            )
        });
        Ok(output.build())
        */
    }
}

// TODO: likely need to add fit stats here to save.
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "N: Network + Serialize, O: Optimizer + Serialize",
    deserialize = "N: Network + Deserialize<'de>, O: Optimizer + Deserialize<'de>"
))]
pub struct ClassificationTrainer<N: Network, O: Optimizer> {
    network: N,
    optimizer: O,
}

impl<N: Network, O: Optimizer> ClassificationTrainer<N, O> {
    pub fn from_network_optimizer(network: N, optimizer: O) -> Self {
        Self { network, optimizer }
    }
}

impl<
        T: Float,
        S1: Data<Elem = T>,
        U: Unsigned,
        S2: Data<Elem = U>,
        D: Dimension,
        N: Network,
        O: Optimizer,
    > Fit<(TensorBase<S1, D>, TensorBase<S2, Ix1>)> for ClassificationTrainer<N, O>
{
    fn train_epoch<I>(
        &mut self,
        device: &Device,
        train_iter: I,
    ) -> Result<(Tensor0<f32>, Option<Tensor0<u32>>)>
    where
        I: Iterator<Item = Result<(TensorBase<S1, D>, TensorBase<S2, Ix1>)>>,
    {
        let mut total_loss = Tensor0::<f32>::zeros(device, ())?;
        let mut correct = Tensor::zeros(device, ())?;
        let mut num_samples = 0;
        for xt in train_iter {
            let (x, t) = xt?;
            num_samples += x.shape()[0];
            let x =
                Variable::from(FloatTensor::from(x.into_owned()?).into_dyn()).with_training(true);
            let y = self.network.forward(x)?.into_dimensionality::<Ix2>()?;
            let nclasses = y.value().dim().1;
            let t = t.into_owned()?;
            let pred = y
                .value()
                .view()
                .into_dimensionality::<Ix2>()?
                .argmax(Axis(1))?;
            pred.accuracy_mut(&t.view(), &mut correct.view_mut())?;
            let t_hot = t.to_one_hot::<T>(nclasses)?;
            let loss = y.cross_entropy_loss(t_hot.into_shared()?.into())?;
            total_loss.add_assign(&loss.value().cast_to()?.view())?;
            smol::block_on(loss.backward())?;
            self.optimizer
                .update(self.network.parameters_mut().into_iter())?;
        }
        let alpha = if num_samples > 0 {
            1. / num_samples as f32
        } else {
            0.
        };
        let loss = total_loss.scale_into(alpha)?;
        Ok((loss, Some(correct)))
    }
    fn test_epoch<I>(
        &self,
        device: &Device,
        test_iter: I,
    ) -> Result<(Tensor0<f32>, Option<Tensor0<u32>>)>
    where
        I: Iterator<Item = Result<(TensorBase<S1, D>, TensorBase<S2, Ix1>)>>,
    {
        let mut total_loss = Tensor0::<f32>::zeros(device, ())?;
        let mut correct = Tensor::zeros(device, ())?;
        let mut num_samples = 0;
        for xt in test_iter {
            let (x, t) = xt?;
            num_samples += x.shape()[0];
            let x =
                Variable::from(FloatTensor::from(x.into_owned()?).into_dyn()).with_training(false);
            let y = self.network.forward(x)?.into_dimensionality::<Ix2>()?;
            let nclasses = y.value().dim().1;
            let t = t.into_owned()?;
            let pred = y
                .value()
                .view()
                .into_dimensionality::<Ix2>()?
                .argmax(Axis(1))?;
            pred.accuracy_mut(&t.view(), &mut correct.view_mut())?;
            let t_hot = t.to_one_hot::<T>(nclasses)?;
            let loss = y.cross_entropy_loss(t_hot.into_shared()?.into())?;
            total_loss.add_assign(&loss.value().cast_to()?.view())?;
        }
        let alpha = if num_samples > 0 {
            1. / num_samples as f32
        } else {
            0.
        };
        let loss = total_loss.scale_into(alpha)?;
        Ok((loss, Some(correct)))
    }
}

impl<T: Float, S: Data<Elem = T>, N: Network, O: Optimizer> Predict<TensorBase<S, Ix2>>
    for ClassificationTrainer<N, O>
{
    fn predict(&self, input: TensorBase<S, Ix2>) -> Result<Tensor1<u32>> {
        let input = Variable::from(FloatTensor::from(input.into_owned()?).into_dyn());
        self.network
            .forward(input)?
            .value()
            .view()
            .into_dimensionality::<Ix2>()?
            .argmax(Axis(1))
    }
}
