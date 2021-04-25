use crate::{
    learn::{Fit, Predict},
    tensor::{
        float_tensor::{FloatTensor, FloatTensorD, FloatTensorExt, FloatTensorViewD, FloatType},
        linalg::{gemm, gemm_bias},
        ArcTensor2, Axis, Data, Dimension, Float, Ix1, Ix2, Tensor, Tensor0, Tensor1, TensorBase,
        TensorView0, TensorView2, TensorViewD, TensorViewMut1, TensorViewMut2, TensorViewMutD,
        Unsigned,
    },
    Result,
};
use half::bf16;
use std::{collections::HashMap, convert::TryInto};

pub mod autograd;
use autograd::{
    Graph, Parameter1, Parameter2, ParameterD, Variable, Variable0, Variable2, VariableD, Vertex,
};

pub mod builders;
use builders::{DenseBuilder, SgdBuilder};

pub trait Optimizer {
    fn step(&mut self, parameter: &mut ParameterD, gradient: FloatTensorViewD) -> Result<()>;
}

pub struct Sgd {
    learning_rate: f32,
    momentum: f32,
    #[allow(unused)]
    velocities: HashMap<Vertex, FloatTensorD>,
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

impl Optimizer for Sgd {
    fn step(&mut self, parameter: &mut ParameterD, gradient: FloatTensorViewD) -> Result<()> {
        fn step_impl<T: Float>(
            sgd: &mut Sgd,
            parameter: &mut ParameterD,
            gradient: &TensorViewD<T>,
            alpha: T,
            _momentum: T,
        ) -> Result<()> {
            //use std::collections::hash_map::Entry;
            let mut parameter_value: TensorViewMutD<T> =
                parameter.value_mut().float_make_mut()?.try_into()?;
            if sgd.momentum > 0. {
                /*let mut velocity = match sgd.velocities.entry(parameter.vertex()) {
                    Entry::Occupied(occupied) => occupied.into_mut(),
                    Entry::Vacant(vacant) => vacant.insert(
                        Tensor::zeros(parameter_value.device(), parameter_value.raw_dim())?.into()
                    ),
                };*/
                todo!()
            } else {
                parameter_value.scaled_add(alpha, gradient)?;
            }
            Ok(())
        }
        match parameter.value().float_type() {
            FloatType::BF16 => step_impl::<bf16>(
                self,
                parameter,
                &gradient.try_into()?,
                bf16::from_f32(-self.learning_rate),
                bf16::from_f32(self.momentum),
            ),
            FloatType::F32 => step_impl::<f32>(
                self,
                parameter,
                &gradient.try_into()?,
                -self.learning_rate,
                self.momentum,
            ),
        }
    }
}

pub trait Forward {
    fn forward(&self, input: VariableD) -> Result<VariableD>;
}

pub trait Network: Forward {
    /// Implementation method for parameters_mut\
    ///
    /// Mutable references to the parameters of the network (or layer) should be pushed into the\
    /// provided vec.
    #[allow(unused_variables)]
    fn collect_paramters_mut<'a>(&'a mut self, parameters: &mut Vec<&'a mut ParameterD>) {}
    /// Returns a Vec containing mutable references to all the parameters in the network.
    ///
    /// Generally this does should not be implemented, as the default implementation calls
    /// collect_paramters_mut.
    fn parameters_mut(&mut self) -> Vec<&mut ParameterD> {
        let mut parameters = Vec::new();
        self.collect_paramters_mut(&mut parameters);
        parameters
    }
    /// Returns mutable references to the layers of the network\
    fn layers_mut(&mut self) -> Vec<&mut dyn Network> {
        Vec::new()
    }
    /*
    fn to_device_mut(&mut self, device: &Device) -> Result<()> {
        for parameter in self.parameters_mut() {
            todo!() // parameter.to_device_mut(device)?;
        }
    }
    */
}

#[derive(Default)]
pub struct Identity;

impl Forward for Identity {
    fn forward(&self, input: VariableD) -> Result<VariableD> {
        Ok(input)
    }
}

impl Network for Identity {}

pub struct Dense<A> {
    weight: ParameterD,
    bias: Option<ParameterD>,
    activation: A,
}

impl Dense<Identity> {
    pub fn builder() -> DenseBuilder<Identity> {
        DenseBuilder::default()
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
            let weight = self.weight.clone().into_dimensionality()?;
            let bias = if let Some(bias) = self.bias.as_ref() {
                Some(bias.clone().into_dimensionality()?)
            } else {
                None
            };
            input.dense(&weight, bias.as_ref())?
        };
        self.activation.forward(output.into_dyn())
    }
}

impl<A: Forward + 'static> Network for Dense<A> {
    fn collect_paramters_mut<'a>(&'a mut self, parameters: &mut Vec<&'a mut ParameterD>) {
        parameters.push(&mut self.weight);
        if let Some(bias) = self.bias.as_mut() {
            parameters.push(bias);
        }
    }
}

#[allow(unused_variables)]
fn bias_backward<T: Float>(
    bias_grad: &mut TensorViewMut1<T>,
    output_grad: &TensorView2<T>,
) -> Result<()> {
    todo!()
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
                Some(bias.value().clone().cast_into::<T>()?)
            } else {
                None
            };
            let bias_value_view = if let Some(bias_value) = bias_value.as_ref() {
                Some(bias_value.view())
            } else {
                None
            };
            let batch_size = input_value.dim().0;
            let outputs = weight_value.dim().0;
            let mut output = input.forward_op(|input_value: TensorView2<T>| {
                let input_value = input_value.cast_into::<T>()?;
                let mut output_value =
                    unsafe { Tensor::uninitialized(device, [batch_size, outputs])? };
                gemm_bias(
                    T::one(),
                    &input_value.view(),
                    &weight_value.view(),
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
            output.backward_op(move |mut weight_grad, output_grad| {
                gemm(
                    T::one(),
                    &output_grad.t(),
                    &input_value.view(),
                    T::one(),
                    &mut weight_grad,
                )
            });
            if let Some(bias) = bias {
                output.backward_parameter_op(bias, |mut bias_grad, output_grad| {
                    bias_backward(&mut bias_grad, &output_grad)
                });
            }
            Ok(output.into())
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
) -> Result<Tensor0<T>> {
    todo!()
}

#[allow(unused_variables)]
fn cross_entropy_loss_backward<T: Float>(
    input: &TensorView2<T>,
    input_grad: &mut TensorViewMut2<T>,
    target: &TensorView2<T>,
    output_grad: &TensorView0<T>,
) -> Result<()> {
    todo!()
}

impl Variable2 {
    pub fn cross_entropy_loss<T: Float>(self, target: ArcTensor2<T>) -> Result<Variable0> {
        let mut output =
            self.forward_op(|input| cross_entropy_loss::<T>(&input, &target.view()))?;
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
        Ok(output.into())
    }
}

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
    fn train_epoch<I>(&mut self, train_iter: I) -> Result<(Tensor0<f32>, Option<Tensor0<u32>>)>
    where
        I: Iterator<Item = Result<(TensorBase<S1, D>, TensorBase<S2, Ix1>)>>,
    {
        // TODO: accuracy
        let mut total_loss: Option<Tensor0<T>> = None;
        let mut graph = Graph::default();
        let mut num_samples = 0;
        for xt in train_iter {
            let (x, t) = xt?;
            num_samples += x.shape()[0];
            let x = Variable::from(FloatTensor::from(x.into_tensor()?).into_dyn())
                .with_graph(&graph)
                .with_training(true);
            let y = self.network.forward(x)?.into_dimensionality::<Ix2>()?;
            let nclasses = y.value().dim().1;
            let t = t.into_tensor()?.to_one_hot::<T>(nclasses)?;
            let loss = y.cross_entropy_loss(t.into())?;
            let loss_value: Tensor0<T> = loss.into_value().into_float_tensor()?.try_into()?;
            if let Some(total_loss) = total_loss.as_mut() {
                total_loss.add_assign(&loss_value.view())?;
            } else {
                total_loss.replace(loss_value);
            }
            graph.backward()?;
            graph.update(self.network.parameters_mut(), &mut self.optimizer)?;
        }
        let alpha = if num_samples > 0 {
            1. / num_samples as f32
        } else {
            0.
        };
        // TODO: unwrap will panic if train_iter is empty!
        let loss = total_loss.unwrap().scale_into(alpha)?;
        Ok((loss, None))
    }
    #[allow(unused_variables)]
    fn test_epoch<I>(&self, test_iter: I) -> Result<(Tensor0<f32>, Option<Tensor0<u32>>)>
    where
        I: Iterator<Item = Result<(TensorBase<S1, D>, TensorBase<S2, Ix1>)>>,
    {
        todo!()
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
            .argmax(Axis(1))
    }
}
