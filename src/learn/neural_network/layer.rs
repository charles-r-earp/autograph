use super::{
    autograd::{Autograd, Backward, Parameter, ParameterD, Variable, VariableD, VariableGradientD},
    optimizer::Optimizer,
};
use crate::{
    buffer::{float::FloatBuffer, Buffer},
    device::Device,
    linalg::DotBias,
    ops::{Im2Col, KernelArgs, KernelKind},
    result::Result,
    rust_shaders,
    scalar::FloatType,
    tensor::{
        float::{
            FloatArcTensor, FloatArcTensorD, FloatTensor, FloatTensorD, FloatTensorViewD,
            FloatTensorViewMutD,
        },
        Tensor, TensorD,
    },
    util::{eclid_gcd, type_eq},
};
use anyhow::bail;
#[doc(hidden)]
pub use async_trait::async_trait;
#[doc(hidden)]
pub use autograph_derive::*;
use ndarray::{Dim, Dimension, IntoDimension, Ix1, Ix4, IxDyn};
use rand::distributions::{Distribution, Uniform};
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

/// Convolutional layer.
#[derive(Layer, Clone, Serialize, Deserialize)]
#[autograph(crate)]
pub struct Conv {
    #[autograph(parameter)]
    weight: ParameterD,
    #[autograph(optional_parameter)]
    bias: Option<ParameterD>,
    strides: IxDyn,
    padding: IxDyn,
    dilation: IxDyn,
}

impl Conv {
    /// Creates a new [`Conv`] for 'inputs`, `outputs`, and `kernel`.
    ///
    /// Defaults:
    /// - strides: 1
    /// - padding: 0
    /// - dilation: 1,
    /// - bias: None
    ///
    /// The weight is initialized with a uniform distribution of (-a, a) where a = sqrt(1 / (inputs * kernel.size())).
    ///
    /// # Note
    /// Only 2D convolutions of 4D imputs are currently implemented.
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
        let kernel = kernel.into_dimension();
        let mut kernel_dim = IxDyn::zeros(kernel.ndim() + 2);
        let mut kernel_dim_array = kernel_dim.as_array_view_mut();
        let kernel_dim_slice = kernel_dim_array.as_slice_mut().unwrap();
        kernel_dim_slice[..2].copy_from_slice([outputs, inputs].as_ref());
        kernel_dim_slice[2..].copy_from_slice(kernel.slice());
        let a = f32::sqrt(1. / (inputs * kernel.size()) as f32);
        let data = Uniform::new(-a, a)
            .sample_iter(&mut rand::thread_rng())
            .take(kernel_dim.size())
            .collect::<Vec<_>>();
        let buffer = FloatBuffer::from(Buffer::from(data));
        let weight = Parameter::from(FloatArcTensor::from(buffer).into_shape(kernel_dim).unwrap());
        let strides = vec![1; kernel.ndim()].into_dimension();
        let padding = IxDyn::zeros(kernel.ndim());
        let dilation = vec![1; kernel.ndim()].into_dimension();
        Self {
            weight,
            bias: None,
            strides,
            padding,
            dilation,
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
        let kernel = &self.weight.shape()[2..];
        if strides.ndim() != kernel.len() {
            if let Some(s) = Ix1::from_dimension(&strides).map(Dimension::into_pattern) {
                strides = vec![s; kernel.len()].into_dimension();
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
        let kernel = &self.weight.shape()[2..];
        if padding.ndim() != kernel.len() {
            if let Some(p) = Ix1::from_dimension(&padding).map(Dimension::into_pattern) {
                padding = vec![p; kernel.len()].into_dimension();
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
    /// The bias is initialized with a uniform distribution of (-a, a) where a = sqrt(1 / inputs).
    pub fn with_bias(mut self, bias: bool) -> Result<Self> {
        if bias {
            let inputs = self.weight.shape()[1];
            let outputs = self.weight.shape()[0];
            let a = f32::sqrt(1. / inputs as f32);
            let data = Uniform::new(-a, a)
                .sample_iter(&mut rand::thread_rng())
                .take(outputs)
                .collect::<Vec<_>>();
            let device = self.weight.device();
            let buffer = FloatBuffer::from(smol::block_on(Buffer::from(data).into_device(device))?);
            self.bias.replace(Parameter::from(
                FloatArcTensor::from(buffer)
                    .into_shape([outputs].as_ref())
                    .unwrap(),
            ));
        } else {
            self.bias = None;
        }
        Ok(self)
    }
}

impl Debug for Conv {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut builder = f.debug_struct("Conv");
        builder.field("weight", &self.weight);
        if let Some(bias) = self.bias.as_ref() {
            builder.field("bias", bias);
        }
        if self.strides.slice().iter().any(|x| *x != 1) {
            builder.field("strides", &self.strides.slice());
        }
        if self.padding.slice().iter().any(|x| *x != 0) {
            builder.field("padding", &self.padding.slice());
        }
        if self.dilation.slice().iter().any(|x| *x != 1) {
            builder.field("dilation", &self.dilation.slice());
        }
        builder.finish()
    }
}

impl Forward for Conv {
    fn forward(&self, input: VariableD) -> Result<VariableD> {
        smol::block_on(async {
            // TODO: convert input to float type of weight
            let input = input
                .into_device(self.weight.device())
                .await?
                .into_dimensionality::<Ix4>()?;
            let (bs, ic, ih, iw) = input.dim();
            let weight = self.weight.clone().into_dimensionality::<Ix4>()?;
            let (oc, _ic, kh, kw) = weight.dim();
            debug_assert_eq!(_ic, ic);
            let weight = weight.into_shape([oc, ic * kh * kw])?;
            let bias = self
                .bias
                .clone()
                .map(Parameter::into_dimensionality)
                .transpose()?;
            let kernel = [kh, kw].into_dimension();
            let args = KernelArgs {
                strides: Dim::from_dimension(&self.strides).unwrap(),
                padding: Dim::from_dimension(&self.padding).unwrap(),
                dilation: Dim::from_dimension(&self.dilation).unwrap(),
            };
            let (oh, ow) = args.im2col_shape([ih, iw], &kernel).into_pattern();
            let output = input.im2col(&kernel, KernelKind::Convolution, &args)?;
            let output = output.dot_bias(&weight.t(), bias.as_ref())?;
            let output = output
                .into_shape([bs, oh, ow, oc])?
                .nhwc_into_nchw()?
                .into_dyn();
            Ok(output)
        })
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
    /// The weight is initialized with a uniform distribution of (-a, a) where a = sqrt(1 / inputs).
    pub fn from_inputs_outputs(inputs: usize, outputs: usize) -> Self {
        let a = f32::sqrt(1. / inputs as f32);
        let data = Uniform::new(-a, a)
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
    /// The bias is initialized with a uniform distribution of (-a, a) where a = sqrt(1 / inputs).
    pub fn with_bias(mut self, bias: bool) -> Result<Self> {
        if bias {
            let inputs = self.weight.shape()[1];
            let outputs = self.weight.shape()[0];
            let a = f32::sqrt(1. / inputs as f32);
            let data = Uniform::new(-a, a)
                .sample_iter(&mut rand::thread_rng())
                .take(outputs)
                .collect::<Vec<_>>();
            let device = self.weight.device();
            let buffer = FloatBuffer::from(smol::block_on(Buffer::from(data).into_device(device))?);
            self.bias.replace(Parameter::from(
                FloatArcTensor::from(buffer)
                    .into_shape([outputs].as_ref())
                    .unwrap(),
            ));
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
    }
}

/// ReLU activation.
#[derive(Default, Layer, Clone, Debug, Serialize, Deserialize)]
#[autograph(crate)]
pub struct Relu {}

fn relu(input: &FloatTensorViewD) -> Result<FloatTensorD> {
    let float_type = input.float_type();
    let device = input.device();
    let dim = input.raw_dim();
    let mut output = match float_type {
        FloatType::BF16 => FloatTensor::zeros(float_type, device, dim)?,
        FloatType::F32 => unsafe { FloatTensor::alloc(float_type, device, dim)? },
    };
    unsafe {
        output = output.with_raw_strides(input.raw_strides());
    }
    let n = input.len() as u32;
    let ws = match float_type {
        FloatType::BF16 => {
            if n % 2 == 0 {
                n / 2
            } else {
                n / 2 + 1
            }
        }
        FloatType::F32 => n,
    };
    let builder = rust_shaders::core()?
        .compute_pass(&format!("activation::relu_{}", float_type.as_str()))?
        .float_slice(input.as_raw_slice())?
        .float_slice_mut(output.as_raw_slice_mut())?
        .push(n)?;
    unsafe {
        builder.submit([ws, 1, 1])?;
    }
    Ok(output)
}

fn relu_backward(
    input: &FloatTensorViewD,
    input_grad: &mut FloatTensorViewMutD,
    output_grad: &FloatTensorViewD,
) -> Result<()> {
    debug_assert_eq!(input.shape(), input_grad.shape());
    debug_assert_eq!(input.shape(), output_grad.shape());
    debug_assert_eq!(input.raw_strides(), input_grad.raw_strides());
    debug_assert_eq!(input_grad.raw_strides(), output_grad.raw_strides());
    let float_type = input.float_type();
    let n = input.len() as u32;
    let ws = match float_type {
        FloatType::BF16 => {
            if n % 2 == 0 {
                n / 2
            } else {
                n / 2 + 1
            }
        }
        FloatType::F32 => n,
    };
    let output_grad_slice = output_grad.to_slice()?;
    let builder = rust_shaders::core()?
        .compute_pass(&format!(
            "activation::relu_backward_{}",
            float_type.as_str()
        ))?
        .float_slice(input.to_slice()?.as_slice())?
        .float_slice_mut(input_grad.as_raw_slice_mut())?
        .float_slice(output_grad_slice.as_slice())?
        .push(n)?;
    unsafe { builder.submit([ws, 1, 1]) }
}

#[derive(Autograd)]
#[autograph(crate)]
struct ReluBackward {
    #[autograph(gradient)]
    input_grad: VariableGradientD,
    output: FloatArcTensorD,
}

impl Backward for ReluBackward {
    fn backward(&self, output_grad: FloatTensorD) -> Result<()> {
        // TODO: implace impl
        let mut dx = self.input_grad.lock();
        relu_backward(
            &self.output.view(),
            &mut dx.zeroed_mut()?,
            &output_grad.view(),
        )?;
        Ok(())
    }
}

impl Forward for Relu {
    fn forward(&self, input: VariableD) -> Result<VariableD> {
        // TODO: inplace impl
        let mut output =
            Variable::from(relu(&input.value().view())?).with_training(input.training());
        if let Some(input_grad) = input.grad() {
            let output_value = output.value().clone();
            output = output.with_backward(ReluBackward {
                input_grad,
                output: output_value,
            });
        }
        Ok(output)
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum PoolingKind {
    Max,
    Mean,
}

impl PoolingKind {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Max => "max",
            Self::Mean => "mean",
        }
    }
}

/// Marker trait for [`PoolBase`].
pub trait PoolKind: Default + Send + Sync + 'static + PoolKindBase {
    #[doc(hidden)]
    fn kind() -> PoolingKind;
}

/// Marker for [`MaxPool`].
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct PoolMax {}

impl PoolKindBase for PoolMax {}

impl PoolKind for PoolMax {
    fn kind() -> PoolingKind {
        PoolingKind::Max
    }
}

/// Marker for [`MeanPool`].
#[derive(Default, Clone, Serialize, Deserialize)]
/* backward not working yet */
struct PoolMean {}

impl PoolKindBase for PoolMean {}

impl PoolKind for PoolMean {
    fn kind() -> PoolingKind {
        PoolingKind::Mean
    }
}

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

/*
/// MeanPool
///
/// See [`PoolBase`]. */
#[allow(unused)]
type MeanPool = PoolBase<PoolMean>;

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
            if let Some(s) = Ix1::from_dimension(&strides).map(Dimension::into_pattern) {
                strides = vec![s; self.kernel.ndim()].into_dimension();
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
            if let Some(p) = Ix1::from_dimension(&padding).map(Dimension::into_pattern) {
                padding = vec![p; self.kernel.ndim()].into_dimension();
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

fn pool_forward(
    kind: PoolingKind,
    indices: bool,
    input: &FloatArcTensorD,
    kernel: &IxDyn,
    strides: &IxDyn,
    padding: &IxDyn,
) -> Result<(Option<TensorD<u32>>, FloatTensorD)> {
    debug_assert!(input.is_standard_layout());
    debug_assert_eq!(kernel.ndim() + 2, input.ndim());

    let float_type = input.float_type();
    match kernel.ndim() {
        2 => {
            let bs = input.shape()[0];
            let ic = input.shape()[1];
            let ih = input.shape()[2];
            let iw = input.shape()[3];
            let oh = (ih + 2 * padding[0] - kernel[0]) / strides[0] + 1;
            let ow = (iw + 2 * padding[1] - kernel[1]) / strides[1] + 1;
            let mut output =
                unsafe { FloatTensor::alloc(float_type, input.device(), [bs, ic, oh, ow])? };
            let mut indices = if indices {
                Some(unsafe { Tensor::alloc(input.device(), output.raw_dim())? })
            } else {
                None
            };
            let bs = bs * ic;
            let builder = rust_shaders::core()?
                .compute_pass(&format!(
                    "pool::{}_pool{}_2d_{}",
                    kind.as_str(),
                    indices.as_ref().map_or("", |_| "_indices"),
                    float_type.as_str()
                ))?
                .float_slice(input.as_raw_slice())?;
            let builder = if let Some(indices) = indices.as_mut() {
                builder.slice_mut(indices.as_raw_slice_mut())?
            } else {
                builder
            };
            let builder = builder
                .float_slice_mut(output.as_raw_slice_mut())?
                .push(bs as u32)?
                .push([ih as u32, iw as u32])?
                .push([oh as u32, ow as u32])?
                .push([kernel[0] as u32, kernel[1] as u32])?
                .push([strides[0] as u32, strides[1] as u32])?
                .push([padding[0] as u32, padding[1] as u32])?;
            unsafe {
                builder.submit([(bs * oh) as u32, ow as u32, 1])?;
            }
            Ok((indices.map(Tensor::into_dyn), output.into_dyn()))
        }
        _ => bail!("Unimplemented!"),
    }
}

fn pool_backward(
    kind: PoolingKind,
    indices: Option<&TensorD<u32>>,
    input_shape: &[usize],
    output_grad: FloatTensorD,
    kernel: &IxDyn,
    strides: &IxDyn,
    padding: &IxDyn,
) -> Result<FloatTensorD> {
    debug_assert_eq!(kernel.ndim() + 2, input_shape.len());
    let output_grad = output_grad.into_standard_layout()?;
    let mut input_grad =
        unsafe { FloatTensor::alloc(output_grad.float_type(), output_grad.device(), input_shape)? };
    match kernel.ndim() {
        2 => {
            let (bs, ic, ih, iw) = input_grad.view().into_dimensionality::<Ix4>()?.dim();
            let dilation = [1, 1].into_dimension();
            let oh = output_grad.shape()[2];
            let ow = output_grad.shape()[3];
            let (s_bez_h, d_bez_h, gcd_h) = eclid_gcd(strides[0], dilation[0]);
            let (s_bez_w, d_bez_w, gcd_w) = eclid_gcd(strides[1], dilation[1]);
            let builder = rust_shaders::core()?.compute_pass(&format!(
                "pool::{}_pool_2d_backward_{}",
                kind.as_str(),
                input_grad.float_type().as_str()
            ))?;
            let builder = if let Some(indices) = indices {
                builder.slice(indices.as_raw_slice())?
            } else {
                builder
            };
            let builder = builder
                .float_slice_mut(input_grad.as_raw_slice_mut())?
                .float_slice(output_grad.as_raw_slice())?
                .push(bs as u32)?
                .push(ic as u32)?
                .push([ih as u32, iw as u32])?
                .push([oh as u32, ow as u32])?
                .push([kernel[0] as u32, kernel[1] as u32])?
                .push([strides[0] as u32, strides[1] as u32])?
                .push([padding[0] as u32, padding[1] as u32])?
                .push([dilation[0] as u32, dilation[1] as u32])?
                .push([s_bez_h as u32, s_bez_w as u32])?
                .push([d_bez_h as u32, d_bez_w as u32])?
                .push([gcd_h as u32, gcd_w as u32])?;
            let h = (ih - 1) / gcd_h + 1;
            let w = (iw - 1) / gcd_w + 1;
            unsafe {
                builder.submit([(bs * ic) as u32, (h * w) as u32, 1])?;
            }
        }
        _ => bail!("Unimplemented!"),
    }
    Ok(input_grad)
}

#[derive(Autograd)]
#[autograph(crate)]
struct PoolBackward {
    #[autograph(gradient)]
    input_grad: VariableGradientD,
    indices: Option<TensorD<u32>>,
    kind: PoolingKind,
    kernel: IxDyn,
    strides: IxDyn,
    padding: IxDyn,
}

impl Backward for PoolBackward {
    fn backward(&self, output_grad: FloatTensorD) -> Result<()> {
        self.input_grad.lock().add_assign(pool_backward(
            self.kind,
            self.indices.as_ref(),
            self.input_grad.shape(),
            output_grad,
            &self.kernel,
            &self.strides,
            &self.padding,
        )?)?;
        Ok(())
    }
}

impl<K: PoolKind> Forward for PoolBase<K> {
    #[allow(unused)]
    fn forward(&self, input: VariableD) -> Result<VariableD> {
        let (indices, output) = pool_forward(
            K::kind(),
            K::kind() == PoolingKind::Max && input.requires_grad(),
            &input.value().to_standard_layout_shared()?,
            &self.kernel,
            &self.strides,
            &self.padding,
        )?;
        let mut output = Variable::from(output).with_training(input.training());
        if let Some(input_grad) = input.grad() {
            if !input.is_standard_layout() {
                bail!("Must be standard layout!");
            }
            output = output.with_backward(PoolBackward {
                input_grad,
                indices,
                kind: K::kind(),
                kernel: self.kernel.clone(),
                strides: self.strides.clone(),
                padding: self.padding.clone(),
            });
        }
        Ok(output)
    }
}

#[cfg(all(test, feature = "device_tests"))]
mod tests {
    use super::*;
    use crate::{
        scalar::{Float, Scalar},
        tensor::{Tensor, TensorView},
    };
    use half::bf16;
    use ndarray::{Array, ArrayD, ArrayViewD};
    use std::convert::TryFrom;

    use ndarray::{azip, Array1, ArrayView1, ArrayViewMut1};

    fn array_relu<T: Scalar>(input: &ArrayView1<T>) -> Array1<T> {
        input.map(|x| {
            if x.to_f32().unwrap() > 0. {
                *x
            } else {
                T::zero()
            }
        })
    }

    async fn relu<T: Float>() -> Result<()> {
        let x_array = (-5..5)
            .into_iter()
            .map(|x| T::from_f32(x as f32).unwrap())
            .collect::<Array1<_>>();
        let y_array = array_relu(&x_array.view());
        let device = Device::new()?;
        let _s = device.acquire().await;
        let x = TensorView::try_from(x_array.view())?
            .into_device(device)
            .await?;
        let y_guard = super::relu(&x.into_float().view().into_dyn())?
            .into_dimensionality()?
            .cast_into::<T>()?
            .read()
            .await?;
        assert_eq!(y_guard.as_array(), y_array.view());
        Ok(())
    }

    #[tokio::test]
    async fn relu_bf16() -> Result<()> {
        relu::<bf16>().await
    }

    #[tokio::test]
    async fn relu_f32() -> Result<()> {
        relu::<f32>().await
    }

    fn array_relu_backward<T: Scalar>(
        input: &ArrayView1<T>,
        input_grad: &mut ArrayViewMut1<T>,
        output_grad: &ArrayView1<T>,
    ) {
        azip!((x in input, dx in input_grad, dy in output_grad) {
            if x.to_f32().unwrap() > 0. {
                *dx = T::from_f32(dx.to_f32().unwrap() + dy.to_f32().unwrap()).unwrap();
            }
        });
    }

    async fn relu_backward<T: Float>() -> Result<()> {
        let x_array = (-5..5)
            .into_iter()
            .map(|x| T::from_f32(x as f32).unwrap())
            .collect::<Array1<_>>();
        let dx_array_in = (-2..8)
            .into_iter()
            .map(|x| T::from_f32(x as f32).unwrap())
            .collect::<Array1<_>>();
        let dy_array = (-3..7)
            .into_iter()
            .map(|x| T::from_f32(x as f32).unwrap())
            .collect::<Array1<_>>();
        let mut dx_array_out = dx_array_in.clone();
        array_relu_backward(
            &x_array.view(),
            &mut dx_array_out.view_mut(),
            &dy_array.view(),
        );
        let device = Device::new()?;
        let _s = device.acquire().await;
        let x = TensorView::try_from(x_array.view())?
            .into_device(device.clone())
            .await?;
        let mut dx = TensorView::try_from(dx_array_in.view())?
            .into_device(device.clone())
            .await?;
        let dy = TensorView::try_from(dy_array.view())?
            .into_device(device)
            .await?;
        super::relu_backward(
            &x.view().into_dyn().into_float(),
            &mut dx.view_mut().into_dyn().into_float(),
            &dy.view().into_dyn().into_float(),
        )?;
        let dx_guard = dx.read().await?;
        assert_eq!(dx_guard.as_array(), dx_array_out.view());
        Ok(())
    }

    #[tokio::test]
    async fn relu_backward_bf16() -> Result<()> {
        relu_backward::<bf16>().await
    }

    #[tokio::test]
    async fn relu_backward_f32() -> Result<()> {
        relu_backward::<f32>().await
    }

    fn array_pool(
        kind: PoolingKind,
        x: ArrayViewD<f32>,
        kernel: &IxDyn,
        strides: &IxDyn,
        padding: &IxDyn,
    ) -> ArrayD<f32> {
        let x = x.into_dimensionality::<Ix4>().unwrap();
        let (bs, ic, ih, iw) = x.dim();
        let oh = (ih + 2 * padding[0] - kernel[0]) / strides[0] + 1;
        let ow = (iw + 2 * padding[1] - kernel[1]) / strides[1] + 1;
        let mut y = Array::<f32, _>::zeros([bs, ic, oh, ow]);
        for (x, mut y) in x.outer_iter().zip(y.outer_iter_mut()) {
            for (x, mut y) in x.outer_iter().zip(y.outer_iter_mut()) {
                for ((yi, yj), y) in y.indexed_iter_mut() {
                    let mut acc = match kind {
                        PoolingKind::Max => f32::NEG_INFINITY,
                        PoolingKind::Mean => 0.,
                    };
                    for ki in 0..kernel[0] {
                        if let Some(xi) = (yi * strides[0] + ki).checked_sub(padding[0]) {
                            for kj in 0..kernel[1] {
                                if let Some(xj) = (yj * strides[1] + kj).checked_sub(padding[1]) {
                                    if let Some(&val) = x.get((xi, xj)) {
                                        match kind {
                                            PoolingKind::Max => {
                                                acc = f32::max(val, acc);
                                            }
                                            PoolingKind::Mean => {
                                                acc += val / kernel.size() as f32;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    *y = acc;
                }
            }
        }
        y.into_dyn()
    }

    async fn pool<T: Float, K: PoolKind>(shape: &[usize], pool: &PoolBase<K>) -> Result<()> {
        let dim = shape.into_dimension();
        let x_vec = (1..=dim.size())
            .into_iter()
            .map(|x| x as f32)
            .collect::<Vec<_>>();
        let x_array = Array::from(x_vec).into_shape(shape)?;
        let y_array = array_pool(
            K::kind(),
            x_array.view(),
            &pool.kernel,
            &pool.strides,
            &pool.padding,
        )
        .map(|y| T::from_f32(*y).unwrap());
        let device = Device::new()?;
        let _s = device.acquire().await;
        let x = Tensor::from(x_array.map(|x| T::from_f32(*x).unwrap()))
            .into_device(device)
            .await?
            .into_float()
            .into_shared()?;
        let y = pool_forward(
            K::kind(),
            false,
            &x,
            &pool.kernel,
            &pool.strides,
            &pool.padding,
        )?
        .1
        .cast_into::<T>()?
        .read()
        .await?;
        assert_eq!(y.as_array(), y_array.view());
        Ok(())
    }

    #[tokio::test]
    async fn mean_pool_2d_f32() -> Result<()> {
        pool::<f32, _>(
            &[1, 1, 4, 4],
            &MeanPool::from_kernel([2, 2]).with_strides(2)?,
        )
        .await?;
        Ok(())
    }

    #[tokio::test]
    async fn max_pool_2d_f32() -> Result<()> {
        pool::<f32, _>(
            &[1, 1, 4, 4],
            &MaxPool::from_kernel([2, 2]).with_strides(2)?,
        )
        .await?;
        Ok(())
    }

    fn array_pool_backward(
        kind: PoolingKind,
        x: ArrayViewD<f32>,
        dy: ArrayViewD<f32>,
        kernel: &IxDyn,
        strides: &IxDyn,
        padding: &IxDyn,
    ) -> ArrayD<f32> {
        let x = x.view().into_dimensionality::<Ix4>().unwrap();
        let dy = dy.into_dimensionality::<Ix4>().unwrap();
        let mut dx = Array::<f32, _>::zeros(x.raw_dim());
        for ((x, mut dx), dy) in x.outer_iter().zip(dx.outer_iter_mut()).zip(dy.outer_iter()) {
            for ((x, mut dx), dy) in x.outer_iter().zip(dx.outer_iter_mut()).zip(dy.outer_iter()) {
                for ((yi, yj), dy) in dy.indexed_iter() {
                    let mut idx = (0, 0);
                    let mut acc = f32::NEG_INFINITY;
                    for ki in 0..kernel[0] {
                        if let Some(xi) = (yi * strides[0] + ki).checked_sub(padding[0]) {
                            for kj in 0..kernel[1] {
                                if let Some(xj) = (yj * strides[1] + kj).checked_sub(padding[1]) {
                                    if let Some(&val) = x.get((xi, xj)) {
                                        match kind {
                                            PoolingKind::Max => {
                                                if val > acc {
                                                    idx = (xi, xj);
                                                    acc = val;
                                                }
                                            }
                                            PoolingKind::Mean => {
                                                dx[(xi, xj)] += *dy;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if kind == PoolingKind::Max {
                        dx[idx] = *dy;
                    }
                }
            }
        }
        dx.into_dyn()
    }

    async fn pool_backward<T: Float, K: PoolKind>(
        shape: &[usize],
        pool: &PoolBase<K>,
    ) -> Result<()> {
        let dim = shape.into_dimension();
        let x_vec = (1..=dim.size())
            .into_iter()
            .map(|x| x as f32)
            .collect::<Vec<_>>();
        let x_array = Array::from(x_vec).into_shape(shape)?;
        let y_array = array_pool(
            K::kind(),
            x_array.view(),
            &pool.kernel,
            &pool.strides,
            &pool.padding,
        );
        let dy_vec = (1..=y_array.len())
            .into_iter()
            .map(|x| x as f32)
            .collect::<Vec<_>>();
        let dy_array = Array::from(dy_vec).into_shape(y_array.shape())?;
        let dx_array = array_pool_backward(
            K::kind(),
            x_array.view(),
            dy_array.view(),
            &pool.kernel,
            &pool.strides,
            &pool.padding,
        )
        .map(|y| T::from_f32(*y).unwrap());
        let device = Device::new()?;
        let _s = device.acquire().await;
        let x = Tensor::from(x_array.map(|x| T::from_f32(*x).unwrap()))
            .into_device(device.clone())
            .await?
            .into_float()
            .into_shared()?;
        let ix = if type_eq::<K, PoolMax>() {
            pool_forward(
                K::kind(),
                true,
                &x,
                &pool.kernel,
                &pool.strides,
                &pool.padding,
            )?
            .0
        } else {
            None
        };
        let dy = Tensor::from(dy_array.map(|y| T::from_f32(*y).unwrap()))
            .into_device(device)
            .await?
            .into_float();
        let dx = super::pool_backward(
            K::kind(),
            ix.as_ref(),
            x.shape(),
            dy,
            &pool.kernel,
            &pool.strides,
            &pool.padding,
        )?
        .cast_into::<T>()?
        .read()
        .await?;
        assert_eq!(dx.as_array(), dx_array.view());
        Ok(())
    }

    #[tokio::test]
    async fn max_pool_2d_backward_f32() -> Result<()> {
        pool_backward::<f32, _>(
            &[1, 1, 4, 4],
            &MaxPool::from_kernel([2, 2]).with_strides(2)?,
        )
        .await?;
        pool_backward::<f32, _>(
            &[2, 4, 9, 9],
            &MaxPool::from_kernel([3, 3]).with_strides(3)?,
        )
        .await?;
        Ok(())
    }

    #[ignore]
    #[tokio::test]
    async fn mean_pool_2d_backward_f32() -> Result<()> {
        pool_backward::<f32, _>(
            &[1, 1, 4, 4],
            &MeanPool::from_kernel([2, 2]).with_strides(2)?,
        )
        .await?;
        Ok(())
    }
}
