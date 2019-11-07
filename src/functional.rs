use super::{autograd::{Var, Param}, iter_ext::ArgMaxExt};
use std::{rc::Rc, sync::Arc, iter};
use ndarray as nd;

pub trait Dense<T> {
  type Output;
  fn dense(&self, kernel: &Param<T>, bias: Option<&Param<T>>) -> Self::Output;
}

impl<T: nd::LinalgScalar, S: nd::Data<Elem=T>, D: nd::Dimension> Dense<T> for nd::ArrayBase<S, D> {
  type Output = nd::Array<T, D>;
  fn dense(&self, kernel: &Param<T>, bias: Option<&Param<T>>) -> Self::Output {
    let batch_size = self.shape()[0];
    let in_channels = self.shape()[1..].iter()
      .product();
    let input = self.view()
      .into_shape([batch_size, in_channels])
      .unwrap();
    let kernel = kernel.view()
      .into_dimensionality()
      .unwrap();
    let out_channels = kernel.shape()[0];
    let out_dim = [batch_size, out_channels];
    let out = if let Some(ref bias) = bias {
      let mut out = bias.value()
        .broadcast(out_dim)
        .unwrap()
        .to_owned();
      nd::linalg::general_mat_mul(T::one(), &input, &kernel.t(), T::one(), &mut out);
      out
    }
    else {
      let mut out = unsafe { nd::Array2::uninitialized(out_dim) };
      nd::linalg::general_mat_mul(T::one(), &input, &kernel.t(), T::zero(), &mut out);
      out
    };
    out.into_dimensionality()
      .unwrap()
  }
}

impl<T: nd::LinalgScalar + num_traits::Float> Dense<T> for Var<T> {
  type Output = Self;
  fn dense(&self, kernel: &Param<T>, bias: Option<&Param<T>>) -> Self {
    let req_grad = self.req_grad() || kernel.req_grad() || bias.map_or(false, |b| b.req_grad()); 
    let out = Self::new(self.tape(), self.value().dense(kernel, bias),  req_grad);
    if let Some(ref out_grad) = out.grad() {
      let out_grad = Rc::clone(out_grad);
      let input_value = Rc::clone(self.value());
      let input_grad = self.grad().map(|ref grad| Rc::clone(grad));
      let kernel_value = nd::ArcArray::clone(kernel.value());
      let kernel_grad = kernel.grad().map(|ref grad| Arc::downgrade(grad));
      let bias_grad = if let Some(ref bias) = bias {
        if let Some(ref bias_grad) = bias.grad() {
          Some(Arc::downgrade(bias_grad))
        }
        else { None }
      }
      else { None };
      out.tape()
        .backward_op(move || {
        let out_grad = out_grad.borrow();
        let out_grad = out_grad.view()
          .into_dimensionality()
          .unwrap();
        let batch_size = input_value.shape()[0];
        let in_channels = input_value.shape()[1..].iter()
          .product();
        let input_value = input_value.view()
          .into_shape([batch_size, in_channels])
          .unwrap();
        if let Some(ref kernel_grad) = kernel_grad {
          let kernel_grad_arc = kernel_grad.upgrade()
            .unwrap();
          let mut kernel_grad_lock = kernel_grad_arc.lock()
            .unwrap();
          let kernel_grad = kernel_grad_lock.view_mut()
            .into_dimensionality()
            .unwrap();
          nd::linalg::general_mat_mul(T::one(), &input_value.t(), &out_grad, T::one(), &mut kernel_grad.reversed_axes());
        }
        if let Some(ref bias_grad) = bias_grad {
          let bias_grad_arc = bias_grad.upgrade()
            .unwrap();
          let mut bias_grad_lock = bias_grad_arc.lock()
            .unwrap();
          let units = bias_grad_lock.shape()[0];
          let mut bias_grad = bias_grad_lock.view_mut()
            .into_shape([1, units])
            .unwrap();
          let ones = nd::Array::ones([1, batch_size]);
          nd::linalg::general_mat_mul(T::one(), &ones, &out_grad, T::one(), &mut bias_grad);
        }
        if let Some(ref input_grad) = input_grad {
          let mut input_grad = input_grad.borrow_mut();
          let mut input_grad = input_grad.view_mut()
            .into_shape([batch_size, in_channels])
            .unwrap();
          let kernel_value = kernel_value.view()
            .into_dimensionality()
            .unwrap();
          nd::linalg::general_mat_mul(T::one(), &out_grad, &kernel_value, T::one(), &mut input_grad);
        }
      });
    }
    out
  }
}

#[allow(unused)]
pub struct ConvArgs<T, D: nd::Dimension> {
  stride: D,
  padding: D,
  pad_elem: T,
  dilation: D,
  groups: usize
}

impl<T: Default + num_traits::Zero, D: nd::Dimension> Default for ConvArgs<T, D> {
  fn default() -> Self {
    let mut stride = D::default();
    let mut dilation = D::default();
    for u in 0 .. stride.ndim() {
      stride[u] = 1;
      dilation[u] = 1;
    }
    Self {
      stride,
      padding: D::default(),
      pad_elem: T::zero(),
      dilation,
      groups: 1
    }
  }
}
  
pub trait Conv<T, D: nd::Dimension> {
  type Output;
  fn conv(&self, kernel: &Param<T>, bias: Option<&Param<T>>, args: &ConvArgs<T, D>) -> Self::Output;
} 

impl<T: nd::LinalgScalar + std::fmt::Debug, S: nd::Data<Elem=T>, D: nd::Dimension> Conv<T, nd::Ix2> for nd::ArrayBase<S, D> {
  type Output = nd::Array<T, D>;
  
  /// - input: [n, c_in, h_in, w_in]
  /// - kernel: [n, c_out, kernel_size[0], kernel_size[1]]
  /// - output: [n, c_out, h_out, w_out]
  /// - h_out = (h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
  /// - w_out = (w_in + 2 * padding[1] - dilation[0] * (kernel_size[1] - 1) - 1) / stride[1] + 1
  
  fn conv(&self, kernel: &Param<T>, bias: Option<&Param<T>>, args: &ConvArgs<T, nd::Ix2>) -> Self::Output {
    self.as_standard_layout();
    use nd::Dimension;
    let input = self.view()
      .into_dimensionality::<nd::Ix4>()
      .unwrap();
    let (n, c_in, h_in, w_in) = input.dim();
    let kernel = kernel.value()
      .view()
      .into_dimensionality::<nd::Ix4>()
      .unwrap();
    let (c_out, _c_in, kh, kw) = kernel.dim();
    debug_assert_eq!(c_in, _c_in);
    let h_out = (h_in + 2 * args.padding[0] - args.dilation[0] * (kh - 1) - 1) / args.stride[0] + 1;
    let w_out = (w_in + 2 * args.padding[1] - args.dilation[1] * (kw - 1) - 1) / args.stride[1] + 1;
    let input = if args.padding.size() > 0 { 
      unimplemented!();
      //let mut _input = unsafe { nd::Array::uninitialized(input.raw_dim()) };
    }
    else { 
      input
    };
    let mut out = nd::Array::zeros([n, c_out, h_out, w_out]);
    input.windows([n, c_in, kh, kw])
      .into_iter()
      .zip(out.exact_chunks_mut([n, c_out, 1, 1])
        .into_iter())
      .for_each(|(input, mut out)| {
        let mut out = out.index_axis_mut(nd::Axis(3), 0);
        let mut out = out.index_axis_mut(nd::Axis(2), 0);
        input.axis_iter(nd::Axis(2))
          .zip(kernel.axis_iter(nd::Axis(2)))
          .for_each(|(input, kernel)| {
            input.axis_iter(nd::Axis(2))
              .zip(kernel.axis_iter(nd::Axis(2)))
              .for_each(|(input, kernel)| {
              nd::linalg::general_mat_mul(T::one(), &input, &kernel.t(), T::one(), &mut out);
          });
      });
    });
    out.into_dimensionality()
      .unwrap()
  }
}

pub trait Relu {
  type Output;
  fn relu(&self) -> Self::Output;
}

impl<T: num_traits::Float, S: nd::Data<Elem=T>, D: nd::Dimension> Relu for nd::ArrayBase<S, D> {
  type Output = nd::Array<T, D>;
  fn relu(&self) -> Self::Output {
    self.map(|&x| if x > T::zero() { x } else { T::zero() })
  }
}

impl<T: 'static + num_traits::Float + num_traits::NumAssign> Relu for Var<T> {
  type Output = Self;
  fn relu(&self) -> Self {
    let out = Self::new(self.tape(), self.value().relu(), self.req_grad());
    if let Some(ref out_grad) = out.grad() {
      let out_grad = Rc::clone(out_grad);
      let input = Rc::clone(self.value());
      let input_grad = Rc::clone(self.grad().as_ref().unwrap());
      let dim = input.len();
      out.tape().backward_op(move || {
        let out_grad = out_grad
          .borrow();
        let out_grad = out_grad.view()
          .into_shape(dim)
          .unwrap();
        let input = input.view()
          .into_shape(dim)
          .unwrap();
        let mut input_grad = input_grad.borrow_mut();
        let mut input_grad = input_grad.view_mut()
          .into_shape(dim)
          .unwrap();
        nd::Zip::from(&input).and(&mut input_grad).and(&out_grad)
          .apply(|&x, dx, &dy| if x > T::zero() { *dx += dy }); 
      });
    }
    out
  }
}

pub trait Softmax {
  type Output;
  fn softmax(&self, axis: nd::Axis) -> Self::Output;
}

impl<T: num_traits::Float + num_traits::NumAssign, S: nd::Data<Elem=T>, D: nd::RemoveAxis> Softmax for nd::ArrayBase<S, D> {
  type Output = nd::Array<T, D>; 
  fn softmax(&self, axis: nd::Axis) -> Self::Output {
    let max = self.fold_axis(axis, T::infinity(), |&m, &x| {
      if m.is_infinite() { x }
      else if m >= x { m }
      else { x }
    });
    let mut out = self.to_owned();
    out.axis_iter_mut(axis)
      .for_each(|mut a| a.zip_mut_with(&max, |x, &m| *x = (*x - m).exp()));
    let sum = out.sum_axis(axis);
    out.axis_iter_mut(axis)
      .for_each(|mut a| a.zip_mut_with(&sum, |x, &s| *x /= s));
    out
  }
}

pub trait CrossEntropyLoss<L> {
  type Output;
  fn cross_entropy_loss(&self, labels: &L) -> Self::Output;
}

impl<T: num_traits::Float + num_traits::NumAssign + iter::Sum, U: num_traits::AsPrimitive<usize>, S1: nd::Data<Elem=T>, D1: nd::RemoveAxis, S2: nd::Data<Elem=U>> 
  CrossEntropyLoss<nd::ArrayBase<S2, nd::Ix1>> for nd::ArrayBase<S1, D1>
  where Self: Softmax {
  type Output = nd::Array0<T>;
  fn cross_entropy_loss(&self, labels: &nd::ArrayBase<S2, nd::Ix1>) -> Self::Output {
    let batch_size = self.shape()[0];
    let nclasses = self.shape()[1..].iter()
      .product();
    let input = self.view()
      .into_shape([batch_size, nclasses])
      .unwrap();
    let pred = input.softmax(nd::Axis(1));
    let loss = pred.axis_iter(nd::Axis(0))
      .zip(labels.iter().copied().map(|u| u.as_()))
      .map(|(p, u)| -p[u].ln())
      .sum();
    nd::arr0(loss)
  }
}

impl<T: 'static + num_traits::Float + num_traits::NumAssign + iter::Sum, U: num_traits::AsPrimitive<usize>, S2: nd::Data<Elem=U>> 
  CrossEntropyLoss<nd::ArrayBase<S2, nd::Ix1>> for Var<T>
  where nd::ArrayD<T>: CrossEntropyLoss<nd::ArrayBase<S2, nd::Ix1>, Output=nd::Array0<T>> {
  type Output = Self;
  fn cross_entropy_loss(&self, labels: &nd::ArrayBase<S2, nd::Ix1>) -> Self {
    let out = Self::new(self.tape(), self.value().cross_entropy_loss(labels), self.req_grad());
    if let Some(ref out_grad) = out.grad() {
      let out_grad = Rc::clone(out_grad);
      let input = Rc::clone(self.value());
      let input_grad = Rc::clone(self.grad().as_ref().unwrap());
      let labels = labels.to_owned();
      out.tape()
        .backward_op(move || {
        let dy = *out_grad.borrow()
          .first()
          .unwrap();
        let mut input_grad = input_grad.borrow_mut();
        input.softmax(nd::Axis(1))
          .axis_iter(nd::Axis(0))
          .zip(input_grad.axis_iter_mut(nd::Axis(0)))
          .zip(labels.iter().copied().map(|u| u.as_()))
          .for_each(|((x, mut dx), u)| {
          x.iter()
            .copied()
            .enumerate()
            .map(|(i, x)| if i == u { x - T::one() } else { x })
            .zip(dx.iter_mut())
            .for_each(|(px, dx)| *dx = px * dy);   
        });
      });
    }
    out
  }
}

pub trait ClassificationMatches<L> {
  fn classification_matches(&self, labels: &L) -> usize;
}

impl<T: num_traits::Float, U: num_traits::AsPrimitive<usize>, S1: nd::Data<Elem=T>, D1: nd::Dimension, S2: nd::Data<Elem=U>> 
  ClassificationMatches<nd::ArrayBase<S2, nd::Ix1>> for nd::ArrayBase<S1, D1> {
  fn classification_matches(&self, labels: &nd::ArrayBase<S2, nd::Ix1>) -> usize {
    let batch_size = self.shape()[0];
    let nclasses = self.shape()[1..].iter()
      .product();
    let input = self.view()
      .into_shape([batch_size, nclasses])
      .unwrap();
    let correct = input.axis_iter(nd::Axis(0))
      .zip(labels.iter().copied().map(|u| u.as_()))
      .filter(|(p, u)| p.iter().arg_max().unwrap() == *u)
      .count();
    correct
  }
}
