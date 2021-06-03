use crate::{
    backend::Device,
    learn::{Fit, Predict},
    tensor::{
        float_tensor::{FloatTensor, FloatTensorD, FloatType},
        linalg::{gemm, gemm_bias},
        ArcTensor2, Axis, Data, Dimension, Float, Ix1, Ix2, Tensor, Tensor0, Tensor1, Tensor2,
        TensorBase, TensorView0, TensorView2, TensorViewD, TensorViewMut1, TensorViewMut2,
        TensorViewMutD, Unsigned,
    },
    util::type_eq,
    Result,
};
use anyhow::{bail, ensure};
use half::bf16;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    convert::TryInto,
    fmt::{self, Debug},
};

pub use autograph_derive::{Forward, Network};

pub mod autograd;
use autograd::{
    Graph, Parameter1, Parameter2, ParameterViewMutD, Variable, Variable0, Variable2, VariableD,
    Vertex,
};

pub mod builders;
use builders::{DenseBuilder, SgdBuilder};

#[cfg(test)]
mod tests;

pub trait Optimizer {
    fn update(
        &mut self,
        parameters: &mut [ParameterViewMutD],
        gradients: &HashMap<Vertex, FloatTensorD>,
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
    fn update(
        &mut self,
        parameters: &mut [ParameterViewMutD],
        gradients: &HashMap<Vertex, FloatTensorD>,
    ) -> Result<()> {
        if self.momentum > 0. {
            todo!()
        } else {
            for parameter in parameters {
                if let Some(parameter_grad) = gradients.get(parameter.vertex()) {
                    let parameter_value = parameter.value_view_mut();
                    match parameter_value.float_type() {
                        FloatType::BF16 => sgd::<bf16>(
                            &mut parameter_value.try_into()?,
                            &parameter_grad.float_view().try_into()?,
                            bf16::from_f32(-self.learning_rate),
                        )?,
                        FloatType::F32 => sgd::<f32>(
                            &mut parameter_value.try_into()?,
                            &parameter_grad.float_view().try_into()?,
                            -self.learning_rate,
                        )?,
                    }
                }
            }
            Ok(())
        }
    }
}

/// Forward is a trait for Neural Networks and layers, that represent a Variable function.\
///
/// /// # Derive
/// Forward can be derived for a composite struct of layers:\
///```
/// use autograph::neural_network::{Forward, Dense};
///
/// #[derive(Forward)]
/// struct Net { // tuple structs ie Net(Dense, Dense) also supported
///     dense1: Dense,
///     dense2: Dense
/// }
///```
pub trait Forward {
    fn forward(&self, input: VariableD) -> Result<VariableD>;
    fn forward_mut(&mut self, input: VariableD) -> Result<VariableD> {
        self.forward(input)
    }
}

/// Network is a trait for Neural Networks and layers\
///
/// Implementation layers should implement collect_paramters_mut and to_device_mut if they have\
/// parameters, and layers_mut if they have parameters.\
/// # Derive
/// Network (and Forward) can be derived for a composite struct of layers:\
///```
/// use autograph::neural_network::{Network, Forward, Dense};
///
/// #[derive(Network, Forward)]
/// struct Net { // tuple structs ie Net(Dense, Dense) also supported
///     dense1: Dense,
///     dense2: Dense
/// }
///```
pub trait Network: Forward {
    /// Implementation method for parameters_mut\
    ///
    /// Mutable references to the parameters of the network (or layer) should be pushed into the\
    /// provided vec. Note that this includes all parameters (including those of child layers).
    #[allow(unused_variables)]
    fn collect_paramters_mut<'a>(
        &'a mut self,
        parameters: &mut Vec<ParameterViewMutD<'a>>,
    ) -> Result<()> {
        Ok(())
    }
    /// Returns a Vec containing mutable references to all the parameters in the network.
    ///
    /// Generally this does should not be implemented, as the default implementation calls
    /// collect_paramters_mut.
    fn parameters_mut(&mut self) -> Result<Vec<ParameterViewMutD>> {
        let mut parameters = Vec::new();
        self.collect_paramters_mut(&mut parameters)?;
        Ok(parameters)
    }
    /// Returns mutable references to the layers of the network\
    fn layers_mut(&mut self) -> Vec<&mut dyn Network> {
        Vec::new()
    }
    /// Moves the network's data to the device in place.\
    ///
    /// This method needs to be implemented when the layer has Parameters or other data that needs\
    /// to be transfered to the new device. However, composite layers that implement layers_mut do\
    /// not need to implement this method.
    #[allow(clippy::wrong_self_convention)]
    fn to_device_mut(&mut self, device: &Device) -> Result<()> {
        // issue is that if any transfers fail some data will be on the previous devices
        for layer in self.layers_mut() {
            layer.to_device_mut(device)?;
        }
        Ok(())
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

#[derive(Default, Debug, Clone, Copy, Serialize, Deserialize, Network, Forward)]
// This replaces ::autograph with crate
#[autograph(crate)]
pub struct Identity;

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "A: Serialize", deserialize = "A: Deserialize<'de>"))]
pub struct Dense<A = Identity> {
    weight: Parameter2,
    bias: Option<Parameter1>,
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

impl<A: Network + 'static> Network for Dense<A> {
    fn collect_paramters_mut<'a>(
        &'a mut self,
        parameters: &mut Vec<ParameterViewMutD<'a>>,
    ) -> Result<()> {
        parameters.push(self.weight.make_mut()?.into_dyn());
        if let Some(bias) = self.bias.as_mut() {
            parameters.push(bias.make_mut()?.into_dyn());
        }
        // For normalization layer?
        self.activation.collect_paramters_mut(parameters)?;
        Ok(())
    }
    fn layers_mut(&mut self) -> Vec<&mut dyn Network> {
        vec![&mut self.activation]
    }
    fn to_device_mut(&mut self, device: &Device) -> Result<()> {
        smol::block_on(async {
            self.weight.to_device_mut(device)?.await?;
            if let Some(bias) = self.bias.as_mut() {
                bias.to_device_mut(device)?.await?;
            }
            self.activation.to_device_mut(device)
        })
    }
}

impl<A: Network + Clone> Clone for Dense<A> {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            activation: self.activation.clone(),
        }
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

impl Variable2 {
    pub fn dense(self, weight: &Parameter2, bias: Option<&Parameter1>) -> Result<Self> {
        fn dense_impl<T: Float>(
            input: Variable2,
            weight: &Parameter2,
            bias: Option<&Parameter1>,
        ) -> Result<Variable2> {
            let device = weight.value().device();
            // This will panic if input isn't T. Can use cast_into to convert but this may be unnecessary
            let input_value: ArcTensor2<T> = input.value().clone().try_into()?;
            let weight_value: ArcTensor2<T> = weight.value().clone().try_into()?;
            let bias_value = if let Some(bias) = bias.as_ref() {
                Some(bias.value().clone().float_cast_into::<T>()?)
            } else {
                None
            };
            let bias_value_view = bias_value.as_ref().map(|bias| bias.view());
            let batch_size = input_value.dim().0;
            let outputs = weight_value.dim().0;
            let mut output = input.forward_op(|input_value: TensorView2<T>| {
                let input_value = input_value.cast_into::<T>()?;
                let mut output_value =
                    unsafe { Tensor::uninitialized(device, [batch_size, outputs])? };
                gemm_bias(
                    T::one(),
                    &input_value.view(),
                    &weight_value.t(),
                    T::zero(),
                    bias_value_view.as_ref(),
                    &mut output_value.view_mut(),
                )?;
                Ok(output_value)
            })?;
            output.backward_op(move |mut input_grad, output_grad| {
                gemm(
                    T::one(),
                    &output_grad,
                    &weight_value.view(),
                    T::one(),
                    &mut input_grad,
                )
            });
            output.backward_parameter_op(&weight.view(), move |mut weight_grad, output_grad| {
                gemm(
                    T::one(),
                    &output_grad.t(),
                    &input_value.view(),
                    T::one(),
                    &mut weight_grad,
                )
            });
            if let Some(bias) = bias {
                output.backward_parameter_op(&bias.view(), |mut bias_grad, output_grad| {
                    bias_backward(&mut bias_grad, &output_grad)
                });
            }
            Ok(output.build())
        }
        // may want to automatically convert input to weight type
        // this operation isn't implemented yet (ie with gradient ops)
        match weight.value().float_type() {
            FloatType::BF16 => dense_impl::<bf16>(self, weight, bias),
            FloatType::F32 => dense_impl::<f32>(self, weight, bias),
        }
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

impl Variable2 {
    pub fn cross_entropy_loss<T: Float>(self, target: ArcTensor2<T>) -> Result<Variable0> {
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
        let mut total_loss = Tensor::zeros(device, ())?;
        let mut correct = Tensor::zeros(device, ())?;
        let mut num_samples = 0;
        for xt in train_iter {
            let (x, t) = xt?;
            num_samples += x.shape()[0];
            let graph = Graph::default();
            let x = Variable::from(FloatTensor::from(x.into_tensor()?).into_dyn())
                .with_graph(&graph)
                .with_training(true);
            let y = self.network.forward(x)?.into_dimensionality::<Ix2>()?;
            let nclasses = y.value().dim().1;
            let t = t.into_tensor()?;
            let pred = y
                .value()
                .float_view()
                .into_dimensionality::<Ix2>()?
                .float_argmax(Axis(1))?;
            pred.accuracy_mut(&t.view(), &mut correct.view_mut())?;
            let t_hot = t.to_one_hot::<T>(nclasses)?;
            let loss = y.cross_entropy_loss(t_hot.into())?;
            let loss_value: Tensor0<T> = loss.into_value().into_float_tensor()?.try_into()?;
            total_loss.add_assign(&loss_value.view())?;
            let parameter_grads = graph.backward()?;
            self.optimizer
                .update(&mut self.network.parameters_mut()?, &parameter_grads)?;
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
        let mut total_loss = Tensor::zeros(device, ())?;
        let mut correct = Tensor::zeros(device, ())?;
        let mut num_samples = 0;
        for xt in test_iter {
            let (x, t) = xt?;
            num_samples += x.shape()[0];
            let graph = Graph::default();
            let x = Variable::from(FloatTensor::from(x.into_tensor()?).into_dyn())
                .with_graph(&graph)
                .with_training(false);
            let y = self.network.forward(x)?.into_dimensionality::<Ix2>()?;
            let nclasses = y.value().dim().1;
            let t = t.into_tensor()?;
            let pred = y
                .value()
                .float_view()
                .into_dimensionality::<Ix2>()?
                .float_argmax(Axis(1))?;
            pred.accuracy_mut(&t.view(), &mut correct.view_mut())?;
            let t_hot = t.to_one_hot::<T>(nclasses)?;
            let loss = y.cross_entropy_loss(t_hot.into())?;
            let loss_value: Tensor0<T> = loss.into_value().into_float_tensor()?.try_into()?;
            total_loss.add_assign(&loss_value.view())?;
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
        let input = Variable::from(FloatTensor::from(input.into_tensor()?).into_dyn());
        self.network
            .forward(input)?
            .value()
            .float_view()
            .into_dimensionality::<Ix2>()?
            .float_argmax(Axis(1))
    }
}
