use super::{
    autograd::{Parameter, ParameterD, VariableD},
    optimizer::Optimizer,
};
use crate::{
    buffer::{float::FloatBuffer, Buffer},
    device::Device,
    linalg::DotBias,
    result::Result,
    tensor::float::FloatArcTensor,
    util::type_eq,
};
use anyhow::bail;
#[doc(hidden)]
pub use async_trait::async_trait;
#[doc(hidden)]
pub use autograph_derive::*;
use ndarray::{Dimension, IntoDimension, IxDyn};
use rand::distributions::{Distribution, Standard, Uniform};
use serde::{Deserialize, Serialize};
use std::fmt::{self, Debug};

mod sealed {
    pub trait PoolKindBase {}
}
use sealed::PoolKindBase;

/// A trait for networks and layers.
///
/// [`Layer`] provides reflection and utility methods.
///
/// # Derive
/// [`Layer`] should be [derived](autograph_derive).
///
/// # Clone
/// Implement [`Clone`] (typically this can be derived) to make it easier to share the layer (potentially between threads).
///
/// # serde
/// Implement [`Serialize`](serde::Serialize) and [`Deserialize`](serde::Deserialize) for saving and loading the layer. This can generally be [derived](<https://serde.rs/derive.html>).
#[async_trait]
pub trait Layer: Forward + Send + Sync + 'static {
    /// The number of parameters.
    ///
    /// This is the length of [`.parameters()`](Self::parameters()).
    fn parameters_len(&self) -> usize {
        0
    }
    #[doc(hidden)]
    #[allow(unused)]
    fn collect_parameters(&self, parameters: &mut Vec<ParameterD>) {}
    /// Enumerates the parameters of the layer, including child layers.
    fn parameters(&self) -> Vec<ParameterD> {
        let mut parameters = Vec::with_capacity(self.parameters_len());
        self.collect_parameters(&mut parameters);
        parameters
    }
    #[doc(hidden)]
    #[allow(unused)]
    fn collect_parameters_mut<'a>(&'a mut self, parameters: &mut Vec<&'a mut ParameterD>) {}
    /// Enumerates mutable references to the parameters of the layer, including child layers.
    fn parameters_mut(&mut self) -> Vec<&mut ParameterD> {
        let mut parameters = Vec::with_capacity(self.parameters_len());
        self.collect_parameters_mut(&mut parameters);
        parameters
    }
    /// Enumerates the immediate child layers of the layer.
    fn layers(&self) -> Vec<&dyn Layer> {
        Vec::new()
    }
    /// Enumerates mutable references to the immediate child layers of the layer.
    fn layers_mut(&mut self) -> Vec<&mut dyn Layer> {
        Vec::new()
    }
    /// Transfers parameters to `device` inplace.
    ///
    /// See [`Parameter::into_device()`](super::autograd::VertexBase::into_device()).
    ///
    /// # Note
    /// If an error occurs, some parameters may not have been transfered.
    async fn to_device_mut(&mut self, device: Device) -> Result<()> {
        for parameter in self.parameters_mut() {
            if parameter.device() != device {
                let new = parameter.clone().into_device(device.clone()).await?;
                *parameter = new;
            }
        }
        Ok(())
    }
    /// Transfers parameters into `device`.
    ///
    /// See [`.to_device_mut()`](Self::to_device_mut()).
    async fn into_device(mut self, device: Device) -> Result<Self>
    where
        Self: Sized,
    {
        self.to_device_mut(device).await?;
        Ok(self)
    }
    /// Updates the layer with the optimizer.
    ///
    /// Call this method on the network after one or more backward passes.
    fn update<O: Optimizer>(&mut self, optimizer: &mut O) -> Result<()>
    where
        Self: Sized,
    {
        optimizer.update(&mut self.parameters_mut())
    }
}

/// A trait for the forward pass.
///
/// [`Layer`]'s implement [`Forward`], which computes the output as a function of the input.
///
/// # Derive
/// [`Forward`] can be [derived](autograph_derive) for sequential layers (ie typical feed-foward networks).
pub trait Forward {
    /// Computes the forward pass.
    ///
    /// # Autograd
    /// Operations on [`Variable`](super::autograd::Variable) are expected to apply backward ops via [`VariableBuilder`](super::autograd::VariableBuilder).
    ///
    /// **Errors**
    ///
    /// Returns an error if the operation could not be performed. Generally the implemenation should return an error instead of panicking.
    fn forward(&self, input: VariableD) -> Result<VariableD>;
}

fn xavier(inputs: usize, outputs: usize) -> Uniform<f32> {
    let a = (6. / (inputs as f32 * outputs as f32)).sqrt();
    Uniform::new(-a, a)
}

fn he_normal(mut inputs: usize) -> impl Distribution<f32> {
    if inputs == 0 {
        inputs = 1;
    }
    let a = (2. / inputs as f32).sqrt();
    Standard.map(move |x: f32| x * a)
}

/// Convolutional layer.
#[derive(Layer, Clone, Serialize, Deserialize)]
#[autograph(crate)]
pub struct Conv {
    #[autograph(parameter)]
    kernel: ParameterD,
    #[autograph(optional_parameter)]
    bias: Option<ParameterD>,
    strides: IxDyn,
    padding: IxDyn,
}

impl Conv {
    /// Creates a new [`Conv`] for 'inputs`, `outputs`, and `kernel`.
    ///
    /// Defaults:
    /// - strides: 1
    /// - padding: 0
    /// - bias: None
    ///
    /// The kernel is initialized with a uniform distribution of (-a, a) where a = sqrt(6 / (inputs * outputs)).
    ///
    /// # Example
    /*
    ```
    # use autograph::{result::Result, learn::neural_network::layer::Conv};
    # fn main() -> Result() {
    let conv = Conv::from_inputs_outputs_kernel(1, 64, [3, 3])
        .with_padding(1)?
        .with_bias(true)?;
    # Ok(())
    # }
    ```
    */
    pub fn from_inputs_outputs_kernel<E>(inputs: usize, outputs: usize, kernel: E) -> Self
    where
        E: IntoDimension,
    {
        let kernel_size = kernel.into_dimension();
        let mut kernel_dim = IxDyn::zeros(kernel_size.ndim() + 2);
        let mut kernel_dim_array = kernel_dim.as_array_view_mut();
        let kernel_dim_slice = kernel_dim_array.as_slice_mut().unwrap();
        kernel_dim_slice[..2].copy_from_slice([inputs, outputs].as_ref());
        kernel_dim_slice[2..].copy_from_slice(kernel_size.slice());
        let data = xavier(inputs, outputs)
            .sample_iter(&mut rand::thread_rng())
            .take(kernel_dim.size())
            .collect::<Vec<_>>();
        let buffer = FloatBuffer::from(Buffer::from(data));
        let kernel = Parameter::from(FloatArcTensor::from(buffer).into_shape(kernel_dim).unwrap());
        let strides = vec![1; kernel.ndim()].into_dimension();
        let padding = IxDyn::zeros(kernel.ndim());
        Self {
            kernel,
            bias: None,
            strides,
            padding,
        }
    }
    /// Adds `strides`.
    ///
    /// If strides are 1 dimensional, they may be broadcasted to the dimensionality of the kernel.
    ///
    /// **Errors**
    ///
    /// If the strides are not 1 dimensional and a different dimensionality than the kernel.
    pub fn with_strides<E>(mut self, strides: E) -> Result<Self>
    where
        E: IntoDimension,
    {
        let mut strides = strides.into_dimension().into_dyn();
        let kernel = &self.kernel.shape()[2..];
        if strides.ndim() != kernel.len() {
            if strides.ndim() == 1 {
                strides = vec![1; kernel.len()].into_dimension();
            } else {
                bail!(
                    "Strides {:?} do not match kernel {:?}!",
                    strides.slice(),
                    kernel
                );
            }
        }
        self.strides = strides;
        Ok(self)
    }
    /// Adds `padding`.
    ///
    /// If padding is 1 dimensional, they may be broadcasted to the dimensionality of the kernel.
    ///
    /// **Errors**
    ///
    /// If padding is not 1 dimensional and a different dimensionality than the kernel.
    pub fn with_padding<E>(mut self, padding: E) -> Result<Self>
    where
        E: IntoDimension,
    {
        let mut padding = padding.into_dimension().into_dyn();
        let kernel = &self.kernel.shape()[2..];
        if padding.ndim() != kernel.len() {
            if padding.ndim() == 1 {
                padding = vec![1; kernel.len()].into_dimension();
            } else {
                bail!(
                    "Padding {:?} do not match kernel {:?}!",
                    padding.slice(),
                    kernel
                );
            }
        }
        self.padding = padding;
        Ok(self)
    }
    /// Adds a bias to the layer.
    ///
    /// The bias is initialized with 0`s.
    ///
    /// **Errors**
    ///
    /// Allocates the bias on the device of the kernel (cannot fail on host). See [`Parameter::zeros()`](super::autograd::VertexBase::zeros()).
    pub fn with_bias(mut self, bias: bool) -> Result<Self> {
        if bias {
            let float_type = self.kernel.float_type();
            let device = self.kernel.device();
            let outputs = self.kernel.shape()[0];
            self.bias
                .replace(Parameter::zeros(float_type, device, [outputs].as_ref())?);
        } else {
            self.bias = None;
        }
        Ok(self)
    }
}

impl Debug for Conv {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut builder = f.debug_struct("Conv");
        builder.field("kernel", &self.kernel);
        if let Some(bias) = self.bias.as_ref() {
            builder.field("bias", bias);
        }
        if self.strides.slice().iter().any(|x| *x != 1) {
            builder.field("strides", &self.strides.slice());
        }
        if self.padding.slice().iter().any(|x| *x != 0) {
            builder.field("padding", &self.padding.slice());
        }
        builder.finish()
    }
}

impl Forward for Conv {
    #[allow(unused)]
    fn forward(&self, input: VariableD) -> Result<VariableD> {
        todo!()
    }
}

/// Dense / fully connected layer.
#[derive(Layer, Clone, Debug, Serialize, Deserialize)]
#[autograph(crate)]
pub struct Dense {
    #[autograph(parameter)]
    weight: ParameterD,
    #[autograph(optional_parameter)]
    bias: Option<ParameterD>,
}

impl Dense {
    /// Creates a new [`Dense`] for `inputs` and `outputs`.
    ///
    /// The weight is initialized with a normal distribution with std_dev = sqrt(2 / inputs).
    pub fn from_inputs_outputs(inputs: usize, outputs: usize) -> Self {
        let data = he_normal(inputs)
            .sample_iter(&mut rand::thread_rng())
            .take(inputs * outputs)
            .collect::<Vec<_>>();
        let buffer = FloatBuffer::from(Buffer::from(data));
        let weight = Parameter::from(
            FloatArcTensor::from(buffer)
                .into_shape([outputs, inputs].as_ref())
                .unwrap(),
        );
        Self { weight, bias: None }
    }
    /// Adds a bias to the layer.
    ///
    /// The bias is initialized with 0`s.
    ///
    /// **Errors**
    ///
    /// Allocates the bias on the device of the weight (cannot fail on host). See [`Parameter::zeros()`](super::autograd::VertexBase::zeros()).
    pub fn with_bias(mut self, bias: bool) -> Result<Self> {
        if bias {
            let float_type = self.weight.float_type();
            let device = self.weight.device();
            let outputs = self.weight.shape()[0];
            self.bias
                .replace(Parameter::zeros(float_type, device, [outputs].as_ref())?);
        } else {
            self.bias = None;
        }
        Ok(self)
    }
}

impl Forward for Dense {
    fn forward(&self, input: VariableD) -> Result<VariableD> {
        let input = smol::block_on(input.into_device(self.weight.device()))?.flatten()?;
        // TODO: convert to float type of weight
        let weight = self.weight.clone().into_dimensionality()?;
        let bias = if let Some(bias) = self.bias.as_ref().map(Parameter::clone) {
            Some(bias.into_dimensionality()?)
        } else {
            None
        };
        Ok(input.dot_bias(&weight.t(), bias.as_ref())?.into_dyn())
        //Ok(input.dense(&weight, bias.as_ref())?.into_dyn())
    }
}

/// ReLU activation.
#[derive(Default, Layer, Clone, Debug, Serialize, Deserialize)]
#[autograph(crate)]
pub struct Relu {}

impl Forward for Relu {
    #[allow(unused)]
    fn forward(&self, input: VariableD) -> Result<VariableD> {
        todo!()
    }
}

/// Marker trait for [`PoolBase`].
pub trait PoolKind: Default + Send + Sync + 'static + PoolKindBase {}

/// Marker for [`MaxPool`].
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct PoolMax {}

impl PoolKindBase for PoolMax {}

impl PoolKind for PoolMax {}

/// Marker for [`MeanPool`].
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct PoolMean {}

impl PoolKindBase for PoolMean {}

impl PoolKind for PoolMean {}

/// Pooling layer.
#[derive(Clone, Serialize, Deserialize)]
pub struct PoolBase<K: PoolKind> {
    kernel: IxDyn,
    strides: IxDyn,
    padding: IxDyn,
    kind: K,
}

/// MaxPool
///
/// See [`PoolBase`].
pub type MaxPool = PoolBase<PoolMax>;

/// MeanPool
///
/// See [`PoolBase`].
pub type MeanPool = PoolBase<PoolMean>;

impl<K: PoolKind> PoolBase<K> {
    /// Creates a new pool with `kernel`.
    ///
    /// Defaults:
    /// - strides: 1
    /// - padding: 0
    pub fn from_kernel<E>(kernel: E) -> Self
    where
        E: IntoDimension,
    {
        let kernel = kernel.into_dimension().into_dyn();
        let strides = vec![1; kernel.ndim()].into_dimension();
        let padding = IxDyn::zeros(kernel.ndim());
        Self {
            kernel,
            strides,
            padding,
            kind: K::default(),
        }
    }
    /// Adds `strides`.
    ///
    /// If strides are 1 dimensional, they may be broadcasted to the dimensionality of the kernel.
    ///
    /// **Errors**
    ///
    /// If the strides are not 1 dimensional and a different dimensionality than the kernel.
    pub fn with_strides<E>(mut self, strides: E) -> Result<Self>
    where
        E: IntoDimension,
    {
        let mut strides = strides.into_dimension().into_dyn();
        if strides.ndim() != self.kernel.ndim() {
            if strides.ndim() == 1 {
                strides = vec![1; self.kernel.ndim()].into_dimension();
            } else {
                bail!(
                    "Strides {:?} do not match kernel {:?}!",
                    strides.slice(),
                    self.kernel.slice()
                );
            }
        }
        self.strides = strides;
        Ok(self)
    }
    /// Adds `padding`.
    ///
    /// If padding is 1 dimensional, they may be broadcasted to the dimensionality of the kernel.
    ///
    /// **Errors**
    ///
    /// If padding is not 1 dimensional and a different dimensionality than the kernel.
    pub fn with_padding<E>(mut self, padding: E) -> Result<Self>
    where
        E: IntoDimension,
    {
        let mut padding = padding.into_dimension().into_dyn();
        if padding.ndim() != self.kernel.ndim() {
            if padding.ndim() == 1 {
                padding = vec![1; self.kernel.ndim()].into_dimension();
            } else {
                bail!(
                    "Padding {:?} do not match kernel {:?}!",
                    padding.slice(),
                    self.kernel.slice()
                );
            }
        }
        self.padding = padding;
        Ok(self)
    }
}

impl<K: PoolKind> Debug for PoolBase<K> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ty = if type_eq::<K, PoolMax>() {
            "MaxPool"
        } else if type_eq::<K, PoolMean>() {
            "MeanPool"
        } else {
            unreachable!()
        };
        let mut builder = f.debug_struct(ty);
        builder.field("kernel", &self.kernel.slice());
        if self.strides.slice().iter().any(|x| *x != 1) {
            builder.field("strides", &self.strides.slice());
        }
        if self.padding.slice().iter().any(|x| *x != 0) {
            builder.field("padding", &self.padding.slice());
        }
        builder.finish()
    }
}

// derive doesn't handle generics yet.
impl<K: PoolKind> Layer for PoolBase<K> {}

impl<K: PoolKind> Forward for PoolBase<K> {
    #[allow(unused)]
    fn forward(&self, input: VariableD) -> Result<VariableD> {
        todo!()
    }
}
