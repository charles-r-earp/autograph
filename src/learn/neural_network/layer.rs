use super::autograd::{
    Parameter, Parameter1, Parameter2, ParameterD, ParameterViewMut, ParameterViewMut1,
    ParameterViewMut2, ParameterViewMutD, Variable, Variable1, Variable2, Variable3, Variable4,
};
//#[cfg(doc)]
//use super::optimizer::Optimizer;
use crate::{
    ops::{
        AddAssign, Col2ImConv2, Col2ImConv2Options, Im2ColConv2, Im2ColConv2Options, MaxPool2 as _,
        MaxPool2Backward as _, MaxPool2Options,
    },
    tensor::{
        ScalarArcTensor, ScalarArcTensor4, ScalarTensor, ScalarTensor4, ScalarTensorBase,
        ScalarTensorView4, Tensor, TensorView, TensorViewMut,
    },
};
use anyhow::{bail, Error, Result};
pub use autograph_derive::*;
#[cfg(feature = "device")]
use dry::macro_for;
use half::bf16;
#[cfg(feature = "device")]
use krnl::buffer::ScalarSliceMut;
use krnl::{
    buffer::{Buffer, ScalarBuffer, ScalarData},
    device::Device,
    scalar::{Scalar, ScalarType},
};
#[cfg(feature = "device")]
use paste::paste;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[cfg(feature = "device")]
use krnl::macros::module;
use ndarray::{linalg::Dot, Dimension, IntoDimension, Ix1, Ix2};

use rand::{
    distributions::{Distribution, Uniform},
    thread_rng,
};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::any::Any;

mod conv_direct;

/// Layer builders.
pub mod builder {
    use super::*;

    fn dim_ones<D: Dimension>() -> D {
        let mut dim = D::default();
        dim.slice_mut().iter_mut().for_each(|x| *x = 1);
        dim
    }

    /// Builder for creating a [`Conv`].
    pub struct ConvBuilder<D: Dimension, A = Identity> {
        inputs: usize,
        outputs: usize,
        filter: D,
        padding: D,
        stride: D,
        dilation: D,
        bias: bool,
        scalar_type: ScalarType,
        device: Device,
        activation: A,
    }

    impl<D: Dimension> ConvBuilder<D> {
        pub(super) fn new() -> Self {
            Self {
                inputs: 0,
                outputs: 0,
                filter: D::default(),
                padding: D::default(),
                stride: dim_ones(),
                dilation: dim_ones(),
                bias: false,
                scalar_type: ScalarType::F32,
                device: Device::host(),
                activation: Identity,
            }
        }
    }

    impl<D: Dimension, A> ConvBuilder<D, A> {
        /// Sets the number of input channels.
        pub fn inputs(self, inputs: usize) -> Self {
            Self { inputs, ..self }
        }
        /// Sets the number of output channels.
        pub fn outputs(self, outputs: usize) -> Self {
            Self { outputs, ..self }
        }
        /// Sets size of the filter.
        pub fn filter(self, filter: impl IntoDimension<Dim = D>) -> Self {
            Self {
                filter: filter.into_dimension(),
                ..self
            }
        }
        /// Adds padding.
        pub fn padding(self, padding: impl IntoDimension<Dim = D>) -> Self {
            Self {
                padding: padding.into_dimension(),
                ..self
            }
        }
        /// Sets the stride. Defaults to 1.
        pub fn stride(self, stride: impl IntoDimension<Dim = D>) -> Self {
            Self {
                stride: stride.into_dimension(),
                ..self
            }
        }
        /// Sets the dilation. Defaults to 1.
        pub fn dilation(self, dilation: impl IntoDimension<Dim = D>) -> Self {
            Self {
                dilation: dilation.into_dimension(),
                ..self
            }
        }
        /// Add a bias. Defaults to false.
        pub fn bias(self, bias: bool) -> Self {
            Self { bias, ..self }
        }
        /// Add an activation layer.
        pub fn activation<A2>(self, activation: A2) -> ConvBuilder<D, A2> {
            let Self {
                inputs,
                outputs,
                filter,
                padding,
                stride,
                dilation,
                bias,
                activation: _,
                scalar_type,
                device,
            } = self;
            ConvBuilder {
                inputs,
                outputs,
                filter,
                padding,
                stride,
                dilation,
                bias,
                activation,
                scalar_type,
                device,
            }
        }
        /// Sets the scalar type. Defaults to F32.
        ///
        /// BF16 and F32 are implemented.
        pub fn scalar_type(self, scalar_type: ScalarType) -> Self {
            Self {
                scalar_type,
                ..self
            }
        }
        /// Sets the device. Defaults to the host.
        pub fn device(self, device: Device) -> Self {
            Self { device, ..self }
        }
        /// Builds the layer.
        ///
        /// **Errors**
        /// - The `scalar_type` is not BF16 or F32.
        /// - Initializing parameters on the `device` failed.
        pub fn build(self) -> Result<Conv<D, A>> {
            let Self {
                inputs,
                outputs,
                filter,
                padding,
                stride,
                dilation,
                bias,
                activation,
                scalar_type,
                device,
            } = self;
            if !matches!(scalar_type, ScalarType::BF16 | ScalarType::F32) {
                bail!("Conv {scalar_type:?} not implemented!");
            }
            let a = if inputs > 0 {
                f32::sqrt(2. / (inputs * filter.size()) as f32)
            } else {
                0.
            };
            let mut rng = thread_rng();
            let mut weight_dim = <D::Larger as Dimension>::Larger::zeros(2 + filter.ndim());
            weight_dim[0] = outputs;
            weight_dim[1] = inputs;
            weight_dim.slice_mut()[2..].copy_from_slice(filter.slice());
            let weight_iter = Uniform::new(-a, a)
                .sample_iter(&mut rng)
                .take(weight_dim.size());
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
            let weight =
                Parameter::from(ScalarTensor::from(weight).into_shape(weight_dim).unwrap());
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
            Ok(Conv {
                weight,
                padding,
                stride,
                dilation,
                bias,
                activation,
            })
        }
    }

    /// Builder for creating a [`Dense`].
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
        /// Sets the number of input channels.
        pub fn inputs(self, inputs: usize) -> Self {
            Self { inputs, ..self }
        }
        /// Sets the number of output channels.
        pub fn outputs(self, outputs: usize) -> Self {
            Self { outputs, ..self }
        }
        /// Adds a bias. Defaults to false.
        pub fn bias(self, bias: bool) -> Self {
            Self { bias, ..self }
        }
        /// Adds and activation layer.
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
        /// Sets the scalar type. Defaults to F32.
        ///
        /// BF16 and F32 are implemented.
        pub fn scalar_type(self, scalar_type: ScalarType) -> Self {
            Self {
                scalar_type,
                ..self
            }
        }
        /// Sets the device. Defaults to the host.
        pub fn device(self, device: Device) -> Self {
            Self { device, ..self }
        }
        /// Builds the layer.
        ///
        /// **Errors**
        /// - The `scalar_type` is not BF16 or F32.
        /// - Initializing parameters on the `device` failed.
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
                bail!("Dense {scalar_type:?} not implemented!");
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

    /// Builder for creating a [`MaxPool`].
    pub struct MaxPoolBuilder<D: Dimension> {
        filter: D,
        stride: Option<D>,
    }

    impl<D: Dimension> MaxPoolBuilder<D> {
        pub(super) fn new() -> Self {
            Self {
                filter: D::default(),
                stride: None,
            }
        }
        /// Sets the size of the pool filter.
        pub fn filter(self, filter: impl IntoDimension<Dim = D>) -> Self {
            Self {
                filter: filter.into_dimension(),
                ..self
            }
        }
        /// Sets the stride. Defaults to filter.
        pub fn stride(self, stride: impl IntoDimension<Dim = D>) -> Self {
            Self {
                stride: Some(stride.into_dimension()),
                ..self
            }
        }
        /// Builds the layer.
        pub fn build(self) -> MaxPool<D> {
            let Self { filter, stride } = self;
            let stride = stride.unwrap_or(filter.clone());
            MaxPool { filter, stride }
        }
    }
}
use builder::*;

/// ParameterVec
///
/// See [`Layer::parameters()`](Layer::parameters).
pub type ParameterVec = SmallVec<[ParameterD; 2]>;
/// ParameterMutVec
///
/// See [`Layer::parameters_mut()`](Layer::parameters_mut).
pub type ParameterMutVec<'a> = SmallVec<[ParameterViewMutD<'a>; 2]>;

/// Layer.
///
/// Typically Layers implement [`Forward<Variable<D>>`](Forward) for the appropriate
/// dimension `D`.
///
/// Layers with parameters or those that store the `device` or `scalar_type` should implement the
/// relevant methods. Functional layers and activations may only need the default implementation.
///
/// Layer can be [derived](autograph_derive) for structs and enums where each field or variant
/// is a layer.
pub trait Layer {
    /// Prepares for training or inference.
    ///
    /// Calls [`.set_training(training)`](Parameter::set_training) on each parameter and
    /// [`.set_training(training)`][Layer::set_training] on each child layer as appropriate.
    fn set_training(&mut self, #[allow(unused_variables)] training: bool) -> Result<()> {
        Ok(())
    }
    /// Parameters of the layer.
    fn parameters(&self) -> ParameterVec {
        ParameterVec::new()
    }
    /// Mutable parameter views of the parameters of the layer.
    ///
    /// The mutable parameter views can be provided to [`Optimizer::update()`](Optimizer::update).
    ///
    /// See [`Parameter::make_view_mut()`](Parameter::make_view_mut).
    fn parameters_mut(&mut self) -> Result<ParameterMutVec> {
        Ok(ParameterMutVec::new())
    }
    /// Casts the layer to `scalar_type` in place.
    fn cast_mut(&mut self, #[allow(unused_variables)] scalar_type: ScalarType) -> Result<()> {
        Ok(())
    }
    /// Transfers the layer to `device` in place.
    fn to_device_mut(&mut self, #[allow(unused_variables)] device: Device) -> Result<()> {
        Ok(())
    }
    /// Moves the layer into `device`.
    fn into_device(self, #[allow(unused_variables)] device: Device) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(self)
    }
}

/// Forward.
///
/// Forward can be [derived](autograph_derive).
pub trait Forward<X> {
    /// The type of the Output.
    type Output;
    /// Executes the forward pass given `input`.
    fn forward(&self, input: X) -> Result<Self::Output>;
}

impl<T: Layer> Layer for Option<T> {
    fn set_training(&mut self, training: bool) -> Result<()> {
        if let Some(layer) = self.as_mut() {
            layer.set_training(training)
        } else {
            Ok(())
        }
    }
    fn parameters(&self) -> ParameterVec {
        self.as_ref()
            .map(|layer| layer.parameters())
            .unwrap_or_default()
    }
    fn parameters_mut(&mut self) -> Result<ParameterMutVec> {
        self.as_mut()
            .map(|layer| layer.parameters_mut())
            .unwrap_or(Ok(ParameterMutVec::new()))
    }
    fn cast_mut(&mut self, scalar_type: ScalarType) -> Result<()> {
        if let Some(layer) = self.as_mut() {
            layer.cast_mut(scalar_type)?;
        }
        Ok(())
    }
    fn to_device_mut(&mut self, #[allow(unused_variables)] device: Device) -> Result<()> {
        if let Some(layer) = self.as_mut() {
            layer.to_device_mut(device)?;
        }
        Ok(())
    }
    fn into_device(self, device: Device) -> Result<Self>
    where
        Self: Sized,
    {
        self.map(|layer| layer.into_device(device)).transpose()
    }
}

impl<X, T: Forward<X, Output = X>> Forward<X> for Option<T> {
    type Output = X;
    fn forward(&self, input: X) -> Result<Self::Output> {
        if let Some(layer) = self.as_ref() {
            layer.forward(input)
        } else {
            Ok(input)
        }
    }
}

impl<T: Layer> Layer for Vec<T> {
    fn set_training(&mut self, training: bool) -> Result<()> {
        self.iter_mut()
            .try_for_each(|layer| layer.set_training(training))
    }
    fn parameters(&self) -> ParameterVec {
        self.iter().flat_map(Layer::parameters).collect()
    }
    fn parameters_mut(&mut self) -> Result<ParameterMutVec> {
        if self.is_empty() {
            Ok(ParameterMutVec::new())
        } else if self.len() == 1 {
            self.first_mut().unwrap().parameters_mut()
        } else {
            let mut parameter_vecs = SmallVec::<[ParameterMutVec; 8]>::with_capacity(self.len());
            for layer in self.iter_mut() {
                parameter_vecs.push(layer.parameters_mut()?);
            }
            Ok(parameter_vecs.into_iter().flatten().collect())
        }
    }
    fn cast_mut(&mut self, scalar_type: ScalarType) -> Result<()> {
        self.iter_mut()
            .try_for_each(|layer| layer.cast_mut(scalar_type))
    }
    fn to_device_mut(&mut self, device: Device) -> Result<()> {
        self.iter_mut()
            .try_for_each(|layer| layer.to_device_mut(device.clone()))
    }
    fn into_device(mut self, device: Device) -> Result<Self>
    where
        Self: Sized,
    {
        self.to_device_mut(device)?;
        Ok(self)
    }
}

impl<X, T: Forward<X, Output = X>> Forward<X> for Vec<T> {
    type Output = X;
    fn forward(&self, mut input: X) -> Result<Self::Output> {
        for layer in self.iter() {
            input = layer.forward(input)?;
        }
        Ok(input)
    }
}

/// Convolutional layer.
///
/// See [`Conv1`] and [`Conv2`].
///
/// Implemented for bf16 and f32.
///
/// # Example
///```no_run
/// # use autograph::{krnl::{scalar::ScalarType, device::Device}, learn::neural_network::layer::{Conv2, Relu}};
/// # fn main() -> anyhow::Result<()> {
/// # let device = Device::host();
/// let conv = Conv2::builder()
///    .inputs(1)
///    .outputs(1)
///    .filter([5, 5])
///    .bias(true)
///    .activation(Relu)
///    .scalar_type(ScalarType::BF16)
///    .device(device.clone())
///    .build()?;
/// # Ok(())
/// # }
///```
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "D: Serialize, <D::Larger as Dimension>::Larger: Serialize, A: Serialize",
    deserialize = "D: Deserialize<'de>, <D::Larger as Dimension>::Larger: Deserialize<'de>, A: Deserialize<'de>",
))]
pub struct Conv<D: Dimension, A = Identity> {
    weight: Parameter<<D::Larger as Dimension>::Larger>,
    padding: D,
    stride: D,
    dilation: D,
    bias: Option<Parameter1>,
    activation: A,
}

/// Convolutional layer with 1 dimension.
///
/// See [`Conv`].
pub type Conv1<A = Identity> = Conv<Ix1, A>;
/// Convolutional layer with 2 dimensions.
///
/// See [`Conv`].
pub type Conv2<A = Identity> = Conv<Ix2, A>;

impl<D: Dimension> Conv<D> {
    /// Returns a builder for creating a [`Conv`].
    pub fn builder() -> ConvBuilder<D> {
        ConvBuilder::new()
    }
}

impl<D: Dimension, A> Conv<D, A> {
    /// The weight as a mutable parameter view.
    pub fn weight_view_mut(
        &mut self,
    ) -> Result<ParameterViewMut<<D::Larger as Dimension>::Larger>> {
        self.weight.make_view_mut()
    }
    /// The bias as a mutable parameter_view.
    pub fn bias_view_mut(&mut self) -> Result<Option<ParameterViewMut1>> {
        self.bias.as_mut().map(Parameter::make_view_mut).transpose()
    }
}

impl<D: Dimension, A> Layer for Conv<D, A> {
    fn set_training(&mut self, training: bool) -> Result<()> {
        self.weight.set_training(training);
        if let Some(bias) = self.bias.as_mut() {
            bias.set_training(training);
        }
        Ok(())
    }
    fn parameters(&self) -> ParameterVec {
        let mut parameters = ParameterVec::new();
        parameters.push(self.weight.clone().into_dyn());
        if let Some(bias) = self.bias.as_ref() {
            parameters.push(bias.clone().into_dyn());
        }
        parameters
    }
    fn parameters_mut(&mut self) -> Result<ParameterMutVec> {
        let mut parameters = ParameterMutVec::new();
        parameters.push(self.weight.make_view_mut()?.into_dyn());
        if let Some(bias) = self.bias.as_mut() {
            parameters.push(bias.make_view_mut()?.into_dyn());
        }
        Ok(parameters)
    }
    fn to_device_mut(&mut self, device: Device) -> Result<()> {
        self.weight.to_device_mut(device.clone())?;
        if let Some(bias) = self.bias.as_mut() {
            bias.to_device_mut(device)?;
        }
        Ok(())
    }
    fn into_device(self, device: Device) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            weight: self.weight.into_device(device.clone())?,
            bias: self.bias.map(|b| b.into_device(device)).transpose()?,
            ..self
        })
    }
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct ConvOptions<D: Dimension> {
    pub padding: D,
    pub stride: D,
    pub dilation: D,
}

impl<D: Dimension> ConvOptions<D> {
    #[inline]
    pub fn is_default(&self) -> bool {
        self.padding.slice().iter().all(|x| *x == 0)
            && self.stride.slice().iter().all(|x| *x == 1)
            && self.dilation.slice().iter().all(|x| *x == 1)
    }
    #[inline]
    pub fn input_shape(&self, output_shape: D, filter: &D) -> Option<D> {
        let Self {
            padding,
            stride,
            dilation,
        } = self;
        let mut shape = output_shape;
        for ((a, f), (p, (s, d))) in shape
            .slice_mut()
            .iter_mut()
            .zip(filter.slice().iter().copied())
            .zip(
                padding.slice().iter().copied().zip(
                    stride
                        .slice()
                        .iter()
                        .copied()
                        .zip(dilation.slice().iter().copied()),
                ),
            )
        {
            *a = (d as isize * (f as isize - 1) + 1 + s as isize * (*a as isize - 1)
                - 2 * p as isize)
                .try_into()
                .ok()?;
            if *a == 0 {
                return None;
            }
        }
        Some(shape)
    }
    #[inline]
    pub fn output_shape(&self, input_shape: D, filter: &D) -> Option<D> {
        let Self {
            padding,
            stride,
            dilation,
        } = self;
        let mut shape = input_shape;
        for ((a, f), (p, (s, d))) in shape
            .slice_mut()
            .iter_mut()
            .zip(filter.slice().iter().copied())
            .zip(
                padding.slice().iter().copied().zip(
                    stride
                        .slice()
                        .iter()
                        .copied()
                        .zip(dilation.slice().iter().copied()),
                ),
            )
        {
            *a = ((*a as isize + 2 * p as isize - d as isize * (f as isize - 1) - 1) / s as isize
                + 1)
            .try_into()
            .ok()?;
            if *a == 0 {
                return None;
            }
        }
        Some(shape)
    }
}

impl<D: Dimension> Default for ConvOptions<D> {
    fn default() -> Self {
        let zeros = D::zeros(D::NDIM.unwrap_or_default());
        let mut ones = zeros.clone();
        ones.slice_mut().iter_mut().for_each(|x| *x = 1);
        Self {
            padding: zeros,
            stride: ones.clone(),
            dilation: ones,
        }
    }
}

#[doc(hidden)]
pub type Conv2Options = ConvOptions<Ix2>;

fn conv2<A>(
    input: Variable4,
    weight: Variable4,
    options: &ConvOptions<Ix2>,
    bias: Option<Variable1>,
    activation: &A,
) -> Result<Variable4>
where
    A: Forward<Variable4, Output = Variable4>,
{
    let mut output = if input.device().is_host()
        && options.is_default()
        && input.scalar_type() == ScalarType::F32
    {
        conv_direct::conv2_direct(input, weight, options)?
    } else {
        conv2_im2col(input, weight, options)?
    };
    if let Some(bias) = bias.as_ref() {
        output.add_assign(bias)?;
    }
    output.forward(activation)
}

fn conv2_im2col(
    input: Variable4,
    weight: Variable4,
    options: &ConvOptions<Ix2>,
) -> Result<Variable4> {
    let (batch_size, inputs, ih, iw) = input.dim();
    let (outputs, inputs2, fh, fw) = weight.dim();
    debug_assert_eq!(inputs, inputs2);
    let (ph, pw) = options.padding.into_pattern();
    let (sh, sw) = options.stride.into_pattern();
    let (dh, dw) = options.dilation.into_pattern();
    let options = Im2ColConv2Options {
        filter: [fh, fw],
        padding: [ph, pw],
        stride: [sh, sw],
        dilation: [dh, dw],
    };
    let [oh, ow] = options.output_shape([ih, iw]);
    let im2col_matrix = input.value().im2col_conv2(&options)?;
    let weight_matrix = weight
        .value()
        .clone()
        .into_shape([outputs, inputs * fh * fw])
        .unwrap();
    let output_matrix = im2col_matrix.dot(&weight_matrix.t())?;
    let mut builder = Variable::builder();
    if let Some(node) = input.node() {
        builder.edge(node, move |output_grad| {
            let options = Col2ImConv2Options {
                shape: [oh, ow],
                filter: [fh, fw],
                ..Col2ImConv2Options::default()
            };
            output_grad
                .dot(&weight_matrix)?
                .col2im_conv2(&options)
                .map(Into::into)
        });
    }
    if let Some(node) = weight.node() {
        builder.edge(node, move |output_grad| {
            Ok(output_grad
                .t()
                .dot(&im2col_matrix)?
                .into_shape([outputs, inputs, fh, fw])
                .unwrap()
                .into())
        });
    }
    let output_matrix = builder.build(output_matrix.into());
    output_matrix
        .into_shape([batch_size, oh, ow, outputs])
        .unwrap()
        .permuted_axes([0, 3, 1, 2])
        .to_standard_layout()
}

#[doc(hidden)]
pub fn conv2_im2col_forward(
    input: ScalarTensorView4,
    weight: ScalarTensorView4,
    options: &ConvOptions<Ix2>,
) -> Result<ScalarTensor4> {
    let (batch_size, inputs, ih, iw) = input.dim();
    let (outputs, inputs2, fh, fw) = weight.dim();
    debug_assert_eq!(inputs, inputs2);
    let (ph, pw) = options.padding.into_pattern();
    let (sh, sw) = options.stride.into_pattern();
    let (dh, dw) = options.dilation.into_pattern();
    let options = Im2ColConv2Options {
        filter: [fh, fw],
        padding: [ph, pw],
        stride: [sh, sw],
        dilation: [dh, dw],
    };
    let [oh, ow] = options.output_shape([ih, iw]);
    let im2col_matrix = input.im2col_conv2(&options)?;
    let weight_matrix = weight.into_shape([outputs, inputs * fh * fw]).unwrap();
    let output_matrix = im2col_matrix.dot(&weight_matrix.t())?;
    let y = output_matrix
        .into_shape([batch_size, oh, ow, outputs])
        .unwrap();
    y.permuted_axes([0, 3, 1, 2]).into_standard_layout()
}

/*
fn conv2_direct(input: Variable4, weight: Variable4, options: &Conv2Options) -> Result<Variable4>,
{
    //use once_cell::sync::OnceCell;

    let (batch_size, inputs, ih, iw) = input.dim();
    let (outputs, inputs, fh, fw) = weight.dim();

    let bias_value = bias.as_ref().map(|bias| bias.value().view());
    if bias_value.is_some() {
        todo!();
    }

    let mut builder = Variable::builder();

    /*if input.node().is_some() && std::env::var("BACKWARD_INPUT").is_err() {
        use std::sync::Arc;
        let output_grad_cell = Arc::new(OnceCell::new());
        if let Some(node) = input.node() {
            let output_grad_cell = output_grad_cell.clone();
            let weight_matrix = weight
                .value()
                .clone()
                .into_shape([outputs, inputs * fh * fw])
                .unwrap();
            builder.edge(node, move |output_grad| {
                let (batch_size, outputs, oh, ow) = output_grad.dim();
                let options = Col2ImConv2Options {
                    shape: [oh, ow],
                    filter: [fh, fw],
                    ..Col2ImConv2Options::default()
                };
                let output_grad = output_grad_cell.get_or_try_init(|| -> Result<_> {
                    Ok(output_grad
                        .permuted_axes([0, 2, 3, 1])
                        .to_owned()?
                        .into_shape([batch_size * oh * ow, outputs])
                        .unwrap())
                })?;
                output_grad
                    .dot(&weight_matrix)?
                    .col2im_conv2(&options)
                    .map(Into::into)
            });
        }
        if let Some(node) = weight.node() {
            let input = input.value().clone();
            let (ph, pw) = options.padding.into_pattern();
            let (sh, sw) = options.stride.into_pattern();
            let (dh, dw) = options.dilation.into_pattern();
            let options = Im2ColConv2Options {
                filter: [fh, fw],
                padding: [ph, pw],
                stride: [sh, sw],
                dilation: [dh, dw],
            };
            let output_grad_cell = output_grad_cell.clone();
            builder.edge(node, move |output_grad| {
                let (batch_size, outputs, oh, ow) = output_grad.dim();
                let im2col_matrix = input.im2col_conv2(&options)?;
                let output_grad = output_grad_cell.get_or_try_init(|| -> Result<_> {
                    Ok(output_grad
                        .permuted_axes([0, 2, 3, 1])
                        .to_owned()?
                        .into_shape([batch_size * oh * ow, outputs])
                        .unwrap())
                })?;
                Ok(output_grad
                    .t()
                    .dot(&im2col_matrix)?
                    .into_shape([outputs, inputs, fh, fw])
                    .unwrap()
                    .into())
            });
        }
    } else {*/
    if let Some(node) = input.node() {
        let weight = weight.value().clone();
        let options = options.clone();
        builder.edge(node, move |output_grad| {
            conv2_direct_backward_input(weight.view(), output_grad.view(), &options).map(Into::into)
        });
    }
    if let Some(bias) = bias.as_ref() {
        if let Some(node) = bias.node() {
            todo!();
        }
    }
    if let Some(node) = weight.node() {
        let input = input.value().clone();
        let options = options.clone();
        builder.edge(node, move |output_grad| {
            conv2_direct_backward_weight(input.view(), output_grad.view(), [fh, fw], &options)
                .map(Into::into)
        });
    }
    let output = conv2_direct_forward(
        input.value().view(),
        weight.value().view(),
        bias_value,
        options,
    )?;
    builder.build(output.into()).forward(activation)
}*/

#[doc(hidden)]
pub fn conv2_direct_forward(
    input: ScalarTensorView4,
    weight: ScalarTensorView4,
    options: &Conv2Options,
) -> Result<ScalarTensor4> {
    conv_direct::conv2_direct_forward(input, weight, options)
    /*macro_for!($T in [bf16, f32] {
        if $T::scalar_type() == input.scalar_type() {
            let input = input.try_into_tensor_view::<$T>().unwrap();
            let weight = weight.try_into_tensor_view::<$T>().unwrap();
            let bias = if let Some(bias) = bias {
                Some(bias.try_into_tensor_view::<$T>().unwrap())
            } else {
                None
            };
            if let Some((input, weight)) = input.as_array().zip(weight.as_array()) {
                let bias = if let Some(bias) = bias.as_ref() {
                    Some(bias.as_array().unwrap())
                } else {
                    None
                };
                return Ok(Tensor::from(conv2_direct_host(input, weight, bias, options)).into());
            }
            todo!();
        }
    });
    todo!();*/
}

/*
fn conv2_direct_host<T: Scalar>(
    x: ArrayView4<T>,
    w: ArrayView4<T>,
    b: Option<ArrayView1<T>>,
    options: &ConvOptions<Ix2>,
) -> Array4<T> {
    let start = Instant::now();
    let (bs, ic, ih, iw) = x.dim();
    let (oc, _ic, fh, fw) = w.dim();
    debug_assert_eq!(ic, _ic);
    let Conv2Options {
        padding,
        stride,
        dilation,
    } = options;
    let (ph, pw) = padding.into_pattern();
    let (sh, sw) = stride.into_pattern();
    let (dh, dw) = dilation.into_pattern();
    let (oh, ow) = options
        .output_shape([ih, iw].into_dimension(), &[fh, fw].into_dimension())
        .unwrap()
        .into_pattern();
    //let default_padding = [ph, pw] == [0, 0];
    //let default_stride_dilation = [sh, sw] == [1, 1] && [dh, dw] == [1, 1];
    let mut y = Array4::<T>::zeros([bs, oc, oh, ow]);
    let sync_y = SyncRawArrayViewMut::try_from(y.view_mut()).unwrap();
    let threads = rayon::current_num_threads();
    let threads_oc = oc.min(threads);
    let h_blocks = (0..oh).step_by(8).len();
    let w_blocks = (0..ow).step_by(8).len();
    let threads_bhw = (threads / threads_oc).min(bs * h_blocks * w_blocks);
    /*dbg!(
        [bs, ic, oh, ow],
        h_blocks,
        w_blocks,
        oc,
        threads_oc,
        threads_bhw
    );*/
    let [thy, twy] = [8, 8];
    let [thx, twx] = [
        dh * (fh - 1) + (thy - 1) * sh + 1,
        dw * (fw - 1) + (twy - 1) * sw + 1,
    ];
    crate::tensor::parallel::broadcast(Some(threads_bhw * threads_oc), |thread_id, threads| {
        let thread_bhwid = thread_id / threads_oc;
        let thread_cidy = thread_id % threads_oc;
        let mut y = sync_y.clone();
        let mut x_tile = Array::<f32, _>::zeros([thx, twx]);
        let mut w_tile = Array::<f32, _>::zeros([fh, fw]);
        for bhwid in (thread_bhwid..bs * h_blocks * w_blocks).step_by(threads_bhw) {
            let bid = bhwid / (h_blocks * w_blocks);
            let hwid = bhwid % (h_blocks * w_blocks);
            let h_block_id = hwid / w_blocks;
            let w_block_id = hwid % w_blocks;
            let hidy = h_block_id * thy;
            let widy = w_block_id * twy;
            for cidy in (thread_cidy..oc).step_by(threads_oc) {
                let mut y_tile = [f32x8::default(); 8];
                for cidx in 0..ic {
                    for tix in 0..thx {
                        let hidx = hidy + tix;
                        for tjx in 0..twx {
                            let widx = widy + tjx;
                            let x = if hidx < ih && widx < iw {
                                unsafe { x.uget([bid, cidx, hidx, widx]).cast() }
                            } else {
                                0f32
                            };
                            unsafe {
                                *x_tile.uget_mut([tix, tjx]) = x;
                            }
                        }
                    }
                    for fi in 0..fh {
                        for fj in 0..fw {
                            unsafe {
                                *w_tile.uget_mut([fi, fj]) = w.uget([cidy, cidx, fi, fj]).cast();
                            }
                        }
                    }
                    for fi in 0..fh {
                        for fj in 0..fw {
                            let w = unsafe { *w_tile.uget([fi, fj]) };
                            let w_vec = f32x8::splat(w);
                            unroll! { for tiy in 0 .. 8 {
                                let tix = tiy + fi;
                                let mut x_vec = f32x8::default();
                                unroll! { for tjy in 0 .. 8 {
                                    let tjx = tjy + fj;
                                    unsafe {
                                        x_vec.as_array_mut()[tjy] = *x_tile.uget([tix, tjx]);
                                    }
                                }}
                                y_tile[tiy] = x_vec.mul_add(w_vec, y_tile[tiy]);
                            }}
                        }
                    }
                }
                unroll! { for tiy in 0 .. 8 {
                    let hidy = hidy + tiy;
                    unroll! { for tjy in 0 .. 8 {
                        let widy = widy + tjy;
                        if hidy < oh && widy < ow {
                            unsafe {
                                *y.uget_mut([bid, cidy, hidy, widy]) = y_tile[tiy].as_array_mut()[tjy].cast();
                            }
                        }
                    }}
                }}
            }
        }
    });
    println!("conv2_direct {:?}: {:?}", x.shape(), start.elapsed());
    y
}*/

#[doc(hidden)]
pub fn conv2_direct_backward_input(
    weight: ScalarTensorView4,
    output_grad: ScalarTensorView4,
    options: &ConvOptions<Ix2>,
) -> Result<ScalarTensor4> {
    conv_direct::conv2_direct_backward_input(weight, output_grad, options)
}

/*
fn conv2_direct_backward_input_host<T: Scalar>(
    w: ArrayView4<T>,
    dy: ArrayView4<T>,
    options: &ConvOptions<Ix2>,
) -> Array4<T> {
    let start = Instant::now();
    let (oc, ic, fh, fw) = w.dim();
    let (bs, oc, oh, ow) = dy.dim();
    let (ih, iw) = options
        .input_shape([oh, ow].into_dimension(), &[fh, fw].into_dimension())
        .unwrap()
        .into_pattern();
    let Conv2Options {
        padding,
        dilation,
        stride,
    } = options;
    let (ph, pw) = padding.into_pattern();
    let (sh, sw) = stride.into_pattern();
    let (dh, dw) = dilation.into_pattern();
    let default_padding = [ph, pw] == [0, 0];
    let mut dx = Array::zeros([bs, ic, ih, iw]);
    let sync_dx = SyncRawArrayViewMut::try_from(dx.view_mut()).unwrap();
    crate::tensor::parallel::broadcast(Some(bs), |thread_id, threads| {
        let mut dx = sync_dx.clone();
        let [thy, twy] = [8, 8];
        let [thx, twx] = [
            dh * (fh - 1) + (thy - 1) * sh + 1,
            dw * (fw - 1) + (twy - 1) * sw + 1,
        ];
        let mut dx_tile = Array::<f32, _>::zeros([thx, twx]);
        let mut w_tile = Array::<f32, _>::zeros([fh, fw]);
        //let mut dx_vec = [f32x8::default(); 8];
        let mut dy_tile = [f32x8::default(); 8];
        for bid in (thread_id..bs).step_by(threads) {
            for cidx in 0..ic {
                for hidy in (0..oh).step_by(thy) {
                    for widy in (0..ow).step_by(twy) {
                        //dx_tile.iter_mut().for_each(|dx| *dx = 0f32);
                        for cidy in 0..oc {
                            for i in 0..fh {
                                for j in 0..fw {
                                    unsafe {
                                        *w_tile.uget_mut([i, j]) =
                                            w.uget([cidy, cidx, i, j]).cast();
                                    }
                                }
                            }
                            for (tiy, hidy) in (hidy..).take(thy).enumerate() {
                                for (tjy, widy) in (widy..).take(twy).enumerate() {
                                    let dy = if hidy < oh && widy < ow {
                                        unsafe { dy.uget([bid, cidy, hidy, widy]).cast() }
                                    } else {
                                        0f32
                                    };
                                    unsafe {
                                        dy_tile[tiy].as_array_mut()[tjy] = dy;
                                    }
                                }
                            }
                            for fi in 0..fh {
                                for fj in 0..fw {
                                    let w = unsafe { *w_tile.uget([fi, fj]) };
                                    //let w = f32x8::splat(w);
                                    /*unroll! { for tiy in 0..8 {
                                        let tix = tiy * sh + fi * dh;
                                        unroll! { for tjy in 0..8 {
                                            let tjx = tjy * sw + fj * dw;
                                            unsafe {
                                                dx_vec[tiy].as_array_mut()[tjy] = *dx_tile.uget([tix, tjx]);
                                            }
                                        }}
                                    }}*/
                                    /*unroll! { for tiy in 0 .. 8 {
                                        dx_vec[tiy] = w.mul_add(dy_tile[tiy], dx_vec[tiy]);
                                        //dx_vec[tiy] += w * dy_tile[tiy];
                                    }}*/
                                    /*unroll! { for tiy in 0 .. 8 {
                                        let tix = tiy * sh + fi * dh;
                                        unroll! { for tjy in 0..8 {
                                            let tjx = tjy * sw + fj * dw;
                                            unsafe {
                                                // *dx_tile.uget_mut([tix, tjx]) = dx_vec[tiy].as_array_ref()[tjy];
                                                // *dx_tile.as_slice_mut().unwrap_unchecked().get_unchecked_mut(tix * twx + tjx) = w;
                                            }
                                        }}
                                    }}*/
                                    unroll! { for tiy in 0 .. 8 {
                                        let tix = tiy + fi;
                                        /*unroll! { for tjy in 0..8 {
                                            let tjx = tjy + fj;
                                            unsafe {
                                                // *dx_tile.uget_mut([tix, tjx]) = dx_vec[tiy].as_array_ref()[tjy];
                                                *dx_tile.as_slice_mut().unwrap_unchecked().get_unchecked_mut(tix * twx + tjx) = w;
                                            }
                                        }}*/
                                        unsafe {
                                            dx_tile.as_slice_mut().unwrap_unchecked().get_unchecked_mut(tix * twx..tix * twx + 8).copy_from_slice(&[w; 8]);
                                        }
                                    }}
                                }
                            }
                        }
                        /* if default_padding */
                        {
                            for (tix, hidx) in (hidy..ih).take(thx).enumerate() {
                                for (tjx, widx) in (widy..iw).take(twx).enumerate() {
                                    unsafe {
                                        *dx.uget_mut([bid, cidx, hidx, widx]) +=
                                            dx_tile.uget([tix, tjx]).cast();
                                    }
                                }
                            }
                        } /* else
                          {
                              for (tix, hidx) in (hidy as isize - ph as isize..ih as isize)
                                  .take(thx)
                                  .enumerate()
                              {
                                  if let Ok(hidx) = usize::try_from(hidx) {
                                      for (tjx, widx) in (widy as isize - pw as isize..iw as isize)
                                          .take(twx)
                                          .enumerate()
                                      {
                                          if let Ok(widx) = usize::try_from(widx) {
                                              unsafe {
                                                  *dx.uget_mut([bid, cidx, hidx, widx]) +=
                                                      dx_tile.uget([tix, tjx]).cast();
                                              }
                                          }
                                      }
                                  }
                              }
                          } */
                    }
                }
            }
        }
    });
    /*println!(
        "conv_backward_input {:?}: {:?}",
        dx.shape(),
        start.elapsed()
    );*/
    dx
}*/

#[doc(hidden)]
pub fn conv2_direct_backward_weight(
    input: ScalarTensorView4,
    output_grad: ScalarTensorView4,
    filter: [usize; 2],
    options: &ConvOptions<Ix2>,
) -> Result<ScalarTensor4> {
    conv_direct::conv2_direct_backward_weight(input, output_grad, filter, options)
}

/*
fn conv2_direct_backward_weight_host<T: Scalar>(
    x: ArrayView4<T>,
    dy: ArrayView4<T>,
    filter: [usize; 2],
    options: &ConvOptions<Ix2>,
) -> Array4<T> {
    let start = Instant::now();
    let (bs, ic, ih, iw) = x.dim();
    let (_bs, oc, oh, ow) = dy.dim();
    let [fh, fw] = filter;
    let Conv2Options {
        padding,
        dilation,
        stride,
    } = options;
    let (ph, pw) = padding.into_pattern();
    let (sh, sw) = stride.into_pattern();
    let (dh, dw) = dilation.into_pattern();
    let default_padding = [ph, pw] == [0, 0];
    let threads = rayon::current_num_threads();
    //let threads_oc = threads.min(oc);
    //let threads_oc = (threads / threads_bs).min(oc);
    return Array::zeros([oc, ic, fh, fw]);
    /*
    let mut w_grad = unsafe { Array::uninitialized([threads_bs, oc, ic, fh, fw]) };
    let sync_w_grad = SyncRawArrayViewMut::try_from(w_grad.view_mut()).unwrap();
    crate::tensor::parallel::broadcast(Some(threads_bs * threads_oc), |thread_id| {
        let thread_bid = thread_id / threads_oc;
        let thread_cidy = thread_id % threads_oc;
        let mut w_grad = sync_w_grad.clone();
        for bid in (thread_bid..bs).step_by(threads_bs) {
            for cidx in (0..ic).step_by(8) {
                for hidy in 0..oh {
                    for widy in (0..ow).step_by(8) {
                        for fi in 0 .. fh {
                            let hidx = -(ph as isize) + (sh * hidy) as isize + (fi * dh) as isize;
                            if (0..ih as isize).contains(&hidx) {
                                let hidx = hidx as usize;
                                for fj in 0 .. fw {
                                    for cidx in (cidx..ic).take(8) {
                                        for widy in (widy..ow).take(8) {
                                            let widx = -(pw as isize)
                                                + (sw * widy) as isize
                                                + (fj * dw) as isize;
                                            if (0..iw as isize).contains(&widx) {
                                                let widx = widx as usize;
                                                for cidy in (thread_cidy..oc).step_by(threads_oc) {
                                                    unsafe {
                                                        *y.uget_mut([bid, cidy, hidy, widy]) += *x.uget([bid, cidx, widx, hidx])
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if default_padding {
                                for (tix, hidx) in (hidy..ih).take(thx).enumerate() {
                                    for (tjx, widx) in (widy..iw).take(twx).enumerate() {
                                        unsafe {
                                            *x_tile.uget_mut([tix, tjx]) =
                                                x.uget([bid, cidx, hidx, widx]).cast();
                                        }
                                    }
                                }
                            } else {
                                x_tile.iter_mut().for_each(|x| *x = 0f32);
                                for (tix, hidx) in (hidy as isize - ph as isize..ih as isize)
                                    .take(thx)
                                    .enumerate()
                                {
                                    if let Ok(hidx) = usize::try_from(hidx) {
                                        for (tjx, widx) in (widy as isize - pw as isize..iw as isize)
                                            .take(twx)
                                            .enumerate()
                                        {
                                            if let Ok(widx) = usize::try_from(widx) {
                                                unsafe {
                                                    *x_tile.uget_mut([tix, tjx]) =
                                                        x.uget([bid, cidx, hidx, widx]).cast();
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            for cidy in (thread_cidy..oc).step_by(threads_oc) {
                                dw_tile.iter_mut().for_each(|dw| *dw = 0f32);
                                for (tiy, hidy) in (hidy..).take(thy).enumerate() {
                                    for (tjy, widy) in (widy..).take(twy).enumerate() {
                                        let dy = if hidy < oh && widy < ow {
                                            unsafe { dy.uget([bid, cidy, hidy, widy]).cast() }
                                        } else {
                                            0f32
                                        };
                                        dy_tile[tiy].as_array_mut()[tjy] = dy;
                                    }
                                }
                                for fi in 0..fh {
                                    for fj in 0..fw {
                                        let mut dw_vec = f32x8::default();
                                        unroll! { for tiy in 0..8 {
                                            let tix = tiy * sh + fi * dh;
                                            let mut x_vec = f32x8::default();
                                            unroll! { for tjy in 0..8 {
                                                let tjx = tjy * sw + fj * dw;
                                                x_vec.as_array_mut()[tjy] = unsafe {
                                                    *x_tile
                                                        .uget([tix, tjx])
                                                };
                                            }}
                                            dw_vec = x_vec.mul_add(dy_tile[tiy], dw_vec);
                                        }}
                                        unsafe {
                                            *dw_tile.uget_mut([fi, fj]) += dw_vec.reduce_add();
                                        }
                                    }
                                }
                                for fi in 0..fh {
                                    for fj in 0..fw {
                                        if [bid, hidy, widy] == [thread_bid, 0, 0] {
                                            unsafe {
                                                *w_grad.uget_mut([thread_bid, cidy, cidx, fi, fj]) =
                                                    dw_tile.uget([fi, fj]).cast();
                                            }
                                        } else {
                                            unsafe {
                                                *w_grad.uget_mut([thread_bid, cidy, cidx, fi, fj]) +=
                                                    dw_tile.uget([fi, fj]).cast();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    });
    let w_grad = if threads_bs == 1 {
        w_grad.into_shape([oc, ic, fh, fw]).unwrap()
    } else {
        let w_grad = w_grad
            .into_shape([threads_bs, oc * ic * fh * fw])
            .unwrap()
            .axis_iter(Axis(1))
            .into_par_iter()
            .map(|x| x.sum())
            .collect();
        Array::from_shape_vec([oc, ic, fh, fw], w_grad).unwrap()
    };
    /*println!(
        "conv_backward_weight {:?}: {:?}",
        x.shape(),
        start.elapsed()
    );*/
    w_grad*/
}*/

/*
impl<A: Forward<Variable3, Output = Variable3>> Forward<Variable3> for Conv1<A> {
    type Output = Variable3;
    fn forward(&self, input: Variable3) -> Result<Variable3> {
        let (n, ic, ih) = input.dim();
        let input = input.into_shape([n, ic, ih, 1]).map_err(Error::msg)?;
        let (outputs, inputs, fh) = self.weight.dim();
        let weight = self
            .weight
            .to_variable()
            .into_shape([outputs, inputs, fh, 1])
            .map_err(Error::msg)?;
        let ph = self.padding.into_pattern();
        let sh = self.stride.into_pattern();
        let dh = self.dilation.into_pattern();
        let options = ConvOptions {
            padding: [ph, 1].into_dimension(),
            stride: [sh, 1].into_dimension(),
            dilation: [dh, 1].into_dimension(),
        };
        let bias = self.bias.as_ref().map(Parameter::to_variable);
        let relu = if std::any::TypeId::of::<A>() == std::any::TypeId::of::<Relu>() {
            Some(Relu)
        } else {
            None
        };
        let output = conv2(input, weight, options, bias, &relu)?;
        let (n2, oc, oh, ow) = output.dim();
        debug_assert_eq!(n, n2);
        debug_assert_eq!(ow, 1);
        let output = output.into_shape([n, oc, oh]).map_err(Error::msg)?;
        self.activation.forward(output)
    }
}*/

impl<A: Forward<Variable4, Output = Variable4>> Forward<Variable4> for Conv2<A> {
    type Output = Variable4;
    fn forward(&self, input: Variable4) -> Result<Variable4> {
        let weight = self.weight.to_variable();
        let options = ConvOptions {
            padding: self.padding,
            stride: self.stride,
            dilation: self.dilation,
        };
        let bias = self.bias.as_ref().map(Parameter::to_variable);
        conv2(input, weight, &options, bias, &self.activation)
    }
}

/// A fully connected linear layer.
///
/// Implemented for bf16 and f32.
///
/// # Example
///```no_run
/// # use autograph::{krnl::{scalar::ScalarType, device::Device}, learn::neural_network::layer::{Dense, Relu}};
/// # fn main() -> anyhow::Result<()> {
/// # let device = Device::host();
/// let dense = Dense::builder()
///    .inputs(1)
///    .outputs(1)
///    .bias(true)
///    .activation(Relu)
///    .scalar_type(ScalarType::BF16)
///    .device(device.clone())
///    .build()?;
/// # Ok(())
/// # }
///```
#[derive(Debug, Serialize, Deserialize)]
pub struct Dense<A = Identity> {
    weight: Parameter2,
    bias: Option<Parameter1>,
    activation: A,
}

impl Dense {
    /// Returns a builder for creating a [`Dense`].
    pub fn builder() -> DenseBuilder {
        DenseBuilder::new()
    }
}

impl<A> Dense<A> {
    /// The weight as a mutable parameter view.
    pub fn weight_view_mut(&mut self) -> Result<ParameterViewMut2> {
        self.weight.make_view_mut()
    }
    /// The bias as a mutable parameter view.
    pub fn bias_view_mut(&mut self) -> Result<Option<ParameterViewMut1>> {
        self.bias.as_mut().map(Parameter::make_view_mut).transpose()
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
    fn parameters(&self) -> ParameterVec {
        let mut parameters = ParameterVec::new();
        parameters.push(self.weight.clone().into_dyn());
        if let Some(bias) = self.bias.as_ref() {
            parameters.push(bias.clone().into_dyn());
        }
        parameters
    }
    fn parameters_mut(&mut self) -> Result<ParameterMutVec> {
        let mut parameters = ParameterMutVec::new();
        parameters.push(self.weight.make_view_mut()?.into_dyn());
        if let Some(bias) = self.bias.as_mut() {
            parameters.push(bias.make_view_mut()?.into_dyn());
        }
        Ok(parameters)
    }
    fn to_device_mut(&mut self, device: Device) -> Result<()> {
        self.weight.to_device_mut(device.clone())?;
        if let Some(bias) = self.bias.as_mut() {
            bias.to_device_mut(device)?;
        }
        Ok(())
    }
    fn into_device(self, device: Device) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            weight: self.weight.into_device(device.clone())?,
            bias: self.bias.map(|b| b.into_device(device)).transpose()?,
            ..self
        })
    }
}

impl<A: Forward<Variable2, Output = Variable2> + Any> Forward<Variable2> for Dense<A> {
    type Output = Variable2;
    fn forward(&self, input: Variable2) -> Result<Self::Output> {
        let mut output = input.dot(&self.weight.to_variable())?;
        if let Some(bias) = self.bias.as_ref() {
            output.add_assign(&bias.to_variable())?;
        }
        let output = self.activation.forward(output);
        output
    }
}

/// MaxPool.
///
/// See [`MaxPool1`] and [`MaxPool2`].
/// Implemented for bf16 and f32.
#[derive(Debug, Serialize, Deserialize)]
pub struct MaxPool<D: Dimension> {
    filter: D,
    stride: D,
}

/// MaxPool with 1 dimension.
///
/// See [`MaxPool`].
pub type MaxPool1 = MaxPool<Ix1>;
/// MaxPool with 2 dimensions.
///
/// See [`MaxPool`].
pub type MaxPool2 = MaxPool<Ix2>;

impl<D: Dimension> MaxPool<D> {
    /// Returns a builder for creating a [`MaxPool`].
    pub fn builder() -> MaxPoolBuilder<D> {
        MaxPoolBuilder::new()
    }
}

impl<D: Dimension> Layer for MaxPool<D> {}

impl Forward<Variable3> for MaxPool1 {
    type Output = Variable3;
    fn forward(&self, input: Variable3) -> Result<Self::Output> {
        let (n, c, ih) = input.dim();
        let input = input.into_shape([n, c, ih, 1]).map_err(Error::msg)?;
        let fh = self.filter.into_pattern();
        let sh = self.stride.into_pattern();
        let output = MaxPool2 {
            filter: [fh, 1].into_dimension(),
            stride: [sh, 1].into_dimension(),
        }
        .forward(input)?;
        let (n2, c2, oh, ow) = output.dim();
        debug_assert_eq!(n, n2);
        debug_assert_eq!(c, c2);
        debug_assert_eq!(ow, 1);
        output.into_shape([n, c, oh]).map_err(Error::msg)
    }
}

impl Forward<Variable4> for MaxPool2 {
    type Output = Variable4;
    fn forward(&self, input: Variable4) -> Result<Self::Output> {
        let (fh, fw) = self.filter.into_pattern();
        let (sh, sw) = self.stride.into_pattern();
        let options = MaxPool2Options {
            size: [fh, fw],
            strides: [sh, sw],
        };
        let mut builder = Variable::builder();
        if let Some(node) = input.node() {
            let mut input = input.value().clone();
            let options = options.clone();
            builder.edge(node, move |output_grad| {
                input
                    .make_view_mut()?
                    .max_pool2_backward(output_grad, options)?;
                Ok(input)
            });
        }
        let output = input.value().max_pool2(options)?;
        Ok(builder.build(output.into()))
    }
}

// for testing
#[doc(hidden)]
impl MaxPool2 {
    pub fn backward(
        &self,
        mut input: ScalarArcTensor4,
        output_grad: ScalarArcTensor4,
    ) -> Result<ScalarArcTensor4> {
        let (fh, fw) = self.filter.into_pattern();
        let (sh, sw) = self.stride.into_pattern();
        let options = MaxPool2Options {
            size: [fh, fw],
            strides: [sh, sw],
        };
        input
            .make_view_mut()?
            .max_pool2_backward(output_grad, options)?;
        Ok(input)
    }
}

/// Flatten.
///
/// See [`Variable::flatten()`](Variable::flatten).
#[derive(Default, Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Flatten;

impl Layer for Flatten {}

impl<D: Dimension + 'static> Forward<Variable<D>> for Flatten {
    type Output = Variable2;
    fn forward(&self, input: Variable<D>) -> Result<Variable2> {
        input.flatten().map_err(Error::msg)
    }
}

/// Identity.
#[derive(Default, Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Identity;

impl Layer for Identity {}

impl<X> Forward<X> for Identity {
    type Output = X;
    fn forward(&self, input: X) -> Result<Self::Output> {
        Ok(input)
    }
}

/// ReLU.
///
/// Implemented for bf16 and f32.
#[derive(Default, Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Relu;

impl Layer for Relu {}

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
        let output = scalar_relu(input.into_value())?;
        Ok(builder.build(output))
    }
}

// for testing
#[doc(hidden)]
impl Relu {
    pub fn backward<D: Dimension>(
        &self,
        output: ScalarArcTensor<D>,
        output_grad: ScalarArcTensor<D>,
    ) -> Result<ScalarArcTensor<D>> {
        scalar_relu_backward(output, output_grad)
    }
}

fn scalar_relu<S: ScalarData, D: Dimension>(
    mut input: ScalarTensorBase<S, D>,
) -> Result<ScalarArcTensor<D>> {
    let scalar_type = input.scalar_type();
    if input.is_standard_layout() {
        if let Some(input_mut) = input.get_view_mut() {
            match scalar_type {
                ScalarType::BF16 => {
                    relu_mut::<bf16, D>(input_mut.try_into().unwrap())?;
                }
                ScalarType::F32 => {
                    relu_mut::<f32, D>(input_mut.try_into().unwrap())?;
                }
                _ => bail!("relu {scalar_type:?} unimplemented!"),
            }
            return input.into_shared();
        }
    } else if input.device().is_device() {
        return scalar_relu(input.to_standard_layout_shared()?);
    }
    match scalar_type {
        ScalarType::BF16 => Ok(relu::<bf16, D>(input.view().try_into().unwrap())?
            .into_shared()?
            .into()),
        ScalarType::F32 => Ok(relu::<f32, D>(input.view().try_into().unwrap())?
            .into_shared()?
            .into()),
        _ => bail!("relu {scalar_type:?} unimplemented!"),
    }
}

const RELU_PAR_LEN: usize = 40_000;

fn relu_mut<T: Scalar + num_traits::Float, D: Dimension>(
    mut input: TensorViewMut<T, D>,
) -> Result<()> {
    if let Some(mut x) = input.as_array_mut() {
        if x.len() >= RELU_PAR_LEN {
            x.into_par_iter().for_each(|x| *x = relu_impl(*x));
        } else {
            x.iter_mut().for_each(|x| *x = relu_impl(*x));
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
        macro_for!($T in [bf16, f32] {
            if let Ok(x) = x.as_scalar_slice_mut().try_into() {
                let kernel = paste! {
                    kernels::[<relu_mut_ $T>]::builder()?
                    .build(device)?
                };
                kernel
                    .dispatch(x)?;
                return Ok(());
            }
        });
        bail!("relu_mut {:?} unimplemented!", T::scalar_type())
    }
}

fn relu<T: Scalar, D: Dimension>(input: TensorView<T, D>) -> Result<Tensor<T, D>> {
    let scalar_type = T::scalar_type();
    if !matches!(scalar_type, ScalarType::BF16 | ScalarType::F32) {
        bail!("Relu {scalar_type:?} unimplemented!");
    }
    if let Some(x) = input.as_array() {
        let y = if x.len() > RELU_PAR_LEN {
            ndarray::Zip::from(&x).par_map_collect(|x| relu_impl(*x))
        } else {
            x.map(|x| relu_impl(*x))
        };
        return Ok(y.into());
    }
    #[cfg(not(feature = "device"))]
    {
        unreachable!()
    }
    #[cfg(feature = "device")]
    {
        macro_for!($T in [bf16, f32] {
            if scalar_type == $T::scalar_type() {
                let mut output = unsafe { Tensor::<$T, D>::uninit(input.device(), input.raw_dim())? };
                let x = input.as_slice().unwrap();
                let mut y = output.as_slice_mut().unwrap();
                let kernel = paste!{ kernels::[<relu_ $T>]::builder()?.build(input.device())? };
                kernel.dispatch(
                    x.as_scalar_slice().try_into().unwrap(),
                    y.as_scalar_slice_mut().try_into().unwrap(),
                )?;
                return Ok(output.cast_into().unwrap());
            }
        });
        unreachable!()
    }
}

fn scalar_relu_backward<D: Dimension>(
    output: ScalarArcTensor<D>,
    mut output_grad: ScalarArcTensor<D>,
) -> Result<ScalarArcTensor<D>> {
    let scalar_type = output.scalar_type();
    if output.device().is_host() || output.is_standard_layout() {
        if let Some(output_grad_mut) = output_grad.get_view_mut() {
            match scalar_type {
                ScalarType::BF16 => {
                    relu_backward_mut::<bf16, D>(
                        output.view().try_into().unwrap(),
                        output_grad_mut.try_into().unwrap(),
                    )?;
                }
                ScalarType::F32 => {
                    relu_backward_mut::<f32, D>(
                        output.view().try_into().unwrap(),
                        output_grad_mut.try_into().unwrap(),
                    )?;
                }
                _ => unreachable!(),
            }
            return Ok(output_grad);
        }
    }
    match scalar_type {
        ScalarType::BF16 => Ok(relu_backward::<bf16, D>(
            output.view().try_into().unwrap(),
            output_grad.view().try_into().unwrap(),
        )?
        .into_shared()?
        .into()),
        ScalarType::F32 => Ok(relu_backward::<f32, D>(
            output.view().try_into().unwrap(),
            output_grad.view().try_into().unwrap(),
        )?
        .into_shared()?
        .into()),
        _ => unreachable!(),
    }
}

fn relu_backward_mut<T: Scalar, D: Dimension>(
    input: TensorView<T, D>,
    mut output_grad: TensorViewMut<T, D>,
) -> Result<()> {
    if let Some((x, mut dy)) = input.as_array().zip(output_grad.as_array_mut()) {
        if x.len() >= RELU_PAR_LEN {
            ndarray::Zip::from(&x)
                .and(&mut dy)
                .par_for_each(|x, dy| *dy = relu_backward_impl(*x, *dy));
        } else {
            dy.zip_mut_with(&x, |dy, x| {
                *dy = relu_backward_impl(*x, *dy);
            });
        }
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
        macro_for!($T in [bf16, f32] {
            if let Some((x, dy)) = x
                .as_scalar_slice()
                .try_into()
                .ok()
                .zip(dy.as_scalar_slice_mut().try_into().ok())
            {
                let kernel = paste! {
                    kernels::[<relu_backward_mut_ $T>]::builder()?
                    .build(input.device())?
                };
                kernel.dispatch(x, dy)?;
                return Ok(());
            }
        });
        bail!(
            "relu_backward_mut {:?} unimplemented!()",
            input.scalar_type()
        );
    }
}

fn relu_backward<T: Scalar, D: Dimension>(
    input: TensorView<T, D>,
    output_grad: TensorView<T, D>,
) -> Result<Tensor<T, D>> {
    if let Some((x, dy)) = input.as_array().zip(output_grad.as_array()) {
        let y = if x.len() >= RELU_PAR_LEN {
            ndarray::Zip::from(&x)
                .and(&dy)
                .par_map_collect(|x, dy| relu_backward_impl(*x, *dy))
        } else {
            ndarray::Zip::from(&x)
                .and(&dy)
                .map_collect(|x, dy| relu_backward_impl(*x, *dy))
        };
        return Ok(y.into());
    }
    #[cfg(not(feature = "device"))]
    {
        unreachable!()
    }
    #[cfg(feature = "device")]
    {
        let input = input.as_standard_layout()?;
        let x = input.as_slice().unwrap();
        let output_grad = output_grad.as_standard_layout()?;
        let dy = output_grad.as_slice().unwrap();
        macro_for!($T in [bf16, f32] {
            if let Some((x, dy)) = x
                .as_scalar_slice()
                .try_into()
                .ok()
                .zip(dy.as_scalar_slice().try_into().ok())
            {
                let mut input_grad = unsafe { Tensor::uninit(input.device(), input.raw_dim())? };
                let dx = ScalarSliceMut::from(input_grad.as_slice_mut().unwrap())
                    .try_into()
                    .unwrap();
                let kernel = paste! {
                    kernels::[<relu_backward_ $T>]::builder()?
                        .build(input.device())?
                };
                kernel.dispatch(x, dy, dx)?;
                return Ok(input_grad);
            }
        });
        bail!("relu_backward {:?} unimplemented!()", input.scalar_type());
    }
}

#[cfg_attr(feature = "device", module)]
mod kernels {
    #[cfg(any(feature = "device", target_arch = "spirv"))]
    use dry::macro_for;
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    #[cfg(target_arch = "spirv")]
    use krnl_core::half::bf16;
    #[cfg(any(feature = "device", target_arch = "spirv"))]
    use krnl_core::macros::kernel;
    use krnl_core::scalar::Scalar;
    #[cfg(any(feature = "device", target_arch = "spirv"))]
    use paste::paste;

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
    macro_for!($T in [bf16, f32] {
        paste! {
            #[kernel]
            pub fn [<relu_mut_ $T>](#[item] x: &mut $T) {
                *x = relu_impl(*x);
            }

            #[kernel]
            pub fn [<relu_ $T>](#[item] x: $T, #[item] y: &mut $T) {
                *y = relu_impl(x);
            }

            #[kernel]
            pub fn [<relu_backward_mut_ $T>](#[item] x: $T, #[item] dy: &mut $T) {
                *dy = relu_backward_impl(x, *dy);
            }

            #[kernel]
            pub fn [<relu_backward_ $T>](#[item] x: $T, #[item] dy: $T, #[item] dx: &mut $T) {
                *dx = relu_backward_impl(x, dy);
            }
        }
    });
}
use kernels::{relu_backward_impl, relu_impl};
