use super::{
    autograd::{Autograd, Backward, Parameter, ParameterD, Variable, VariableD, VariableGradientD},
    optimizer::Optimizer,
};
use crate::{
    buffer::{float::FloatBuffer, Buffer},
    device::Device,
    linalg::DotBias,
    result::Result,
    rust_shaders,
    scalar::FloatType,
    tensor::float::{
        FloatArcTensor, FloatArcTensorD, FloatTensor, FloatTensorD, FloatTensorViewD,
        FloatTensorViewMutD,
    },
    //ops::{KernelKind, KernelArgs, Im2Col},
    util::type_eq,
};
use anyhow::bail;
#[doc(hidden)]
pub use async_trait::async_trait;
#[doc(hidden)]
pub use autograph_derive::*;
use ndarray::{Dimension, IntoDimension, IxDyn /*Dim*/};
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
    /// The weight is initialized with a uniform distribution of (-a, a) where a = sqrt(6 / (inputs * outputs)).
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
        kernel_dim_slice[..2].copy_from_slice([inputs, outputs].as_ref());
        kernel_dim_slice[2..].copy_from_slice(kernel.slice());
        let data = xavier(inputs, outputs)
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
        let kernel = &self.weight.shape()[2..];
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
    #[allow(unused)]
    fn forward(&self, input: VariableD) -> Result<VariableD> {
        todo!()
        /*
        let input = input.into_dimensionality()?;
        let (bs, ic, ih, iw) = input.dim();
        let weight = self.weight.clone().into_dimensionality()?;
        let (_ic, oc, kh, kw) = weight.dim();
        debug_assert_eq!(_ic, ic);
        let bias = self.bias.clone().map(Parameter::into_dimensionality).transpose()?;
        let kernel = [kh, kw].into_dimension();
        let args = KernelArgs {
            strides: Dim::from_dimension(&self.strides).unwrap(),
            padding: Dim::from_dimension(&self.padding).unwrap(),
            dilation: Dim::from_dimension(&self.dilation).unwrap(),
        };
        let input = smol::block_on(input.into_device(weight.device()))?;
        let input = input.im2col(&kernel, KernelKind::Convolution, &args)?;
        let (bs, _, oh, ow) = input.dim();
        let input = input.into_shape([bs, ic * kh * kw, oh * ow])?;
        input.dot_bias(&weight, bias.as_ref())?
            .into_shape([bs, oc, oh, ow])
        // [bs, ic * kh *kw, oh * ow]
        // [ic, oc, kh, kw]
        // [bs, oc, oh, ow]
        //
        // [bs * ow * ow, ic * kh * kw] [ic * kh * kw, oc] [bs * oh * ow, oc]
        */
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

fn relu(input: &FloatTensorViewD) -> Result<FloatTensorD> {
    let float_type = input.float_type();
    let device = input.device();
    let dim = input.raw_dim();
    let mut output = match float_type {
        FloatType::BF16 => FloatTensor::zeros(float_type, device, dim)?,
        FloatType::F32 => unsafe { FloatTensor::alloc(float_type, device, dim)? },
    };
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
        .float_slice(input.to_slice()?.as_slice())?
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

#[cfg(all(test, feature = "device_tests"))]
mod tests {
    use super::*;
    use crate::{
        scalar::{Float, Scalar},
        tensor::TensorView,
    };
    use half::bf16;
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
}
