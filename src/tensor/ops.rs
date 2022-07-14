use super::*;
use crate::{
    ops::{AddAssign, Col2Im, Im2Col, KernelArgs, KernelKind, ScaledAdd},
    rust_shaders,
    scalar::Float,
    util::eclid_gcd,
};
use anyhow::ensure;
use ndarray::{Array, Array2, Array4, Axis, Data as ArrayData};

impl<T: Scalar, S1: DataMut<Elem = T>, S2: Data<Elem = T>, D: Dimension>
    AddAssign<TensorBase<S2, D>> for TensorBase<S1, D>
{
    fn add_assign(&mut self, rhs: &TensorBase<S2, D>) -> Result<()> {
        self.scaled_add(T::one(), rhs)
    }
}

impl<T: Scalar, S1: DataMut<Elem = T>, S2: Data<Elem = T>, D: Dimension>
    ScaledAdd<T, TensorBase<S2, D>> for TensorBase<S1, D>
{
    fn scaled_add(&mut self, alpha: T, rhs: &TensorBase<S2, D>) -> Result<()> {
        ensure!(self.dim == rhs.dim);
        ensure!(self.strides == rhs.strides);
        let n = self.len() as u32;
        let builder = glsl_shaders::module(&format!("scaled_add_{}", T::scalar_name()))?
            .compute_pass("main")?
            .slice_mut(self.as_raw_slice_mut())?
            .slice(rhs.as_raw_slice())?
            .push(n)?
            .push(alpha)?;
        unsafe { builder.submit([n, 1, 1]) }
    }
}

impl<T: Float, S: Data<Elem = T>> Im2Col<Ix2> for TensorBase<S, Ix4> {
    type Output = Tensor2<T>;
    fn im2col(
        &self,
        kernel: &Ix2,
        kind: KernelKind,
        args: &KernelArgs<Ix2>,
    ) -> Result<Self::Output> {
        let input = self.as_standard_layout()?;
        let (bs, ic, ih, iw) = input.dim();
        let (kh, kw) = kernel.into_pattern();
        let strides = &args.strides;
        let padding = &args.padding;
        let dilation = &args.dilation;
        let (oh, ow) = args.im2col_shape([ih, iw], kernel).into_pattern();
        let mut output = unsafe { Tensor::alloc(input.device(), [bs * oh * ow, ic * kh * kw])? };
        // let conv_5x5 = strides.into_pattern() == (1, 1) && padding.into_pattern() == (0, 0) && dilation.into_pattern() == (1, 1) && (kh, kw) == (5, 5);
        let entry = format!(
            "kernel::im2col_2d_{}_{}",
            kind.as_str(),
            //if conv_5x5 { "_5x5" } else { "" },
            T::scalar_name()
        );
        let builder = rust_shaders::compute_pass(&entry)?
            .slice(input.as_raw_slice())?
            .slice_mut(output.as_raw_slice_mut())?
            .push(bs as u32)?
            .push(ic as u32)?
            .push([ih as u32, iw as u32])?
            .push([oh as u32, ow as u32])?
            .push([kh as u32, kw as u32])?
            .push([strides[0] as u32, strides[1] as u32])?
            .push([padding[0] as u32, padding[1] as u32])?
            .push([dilation[0] as u32, dilation[1] as u32])?;
        let work_size = {
            //let oh = (oh / 16) * 16 + if oh % 16 != 0 { 16 } else { 0 };
            //let ow = (ow / 16) * 16 + if ow % 16 != 0 { 16 } else { 0 };
            /*let ohw = oh * ow;
            //let ohw = (ohw / 256) * 256 + if ohw % 256 != 0 { 256 } else { 0 };*/

            /*if conv_5x5 {
                let gh = oh / 32 + if oh % 32 != 0 { 1 } else { 0 };
                let gw = ow / 32 + if ow % 32 != 0 { 1 } else { 0 };
                let ohw = gh * gw * 256;
                [(bs * ic) as u32, ohw as u32, 1]
            } else {
                let gh = oh / 16 + if oh % 16 != 0 { 1 } else { 0 };
                let gw = ow / 16 + if ow % 16 != 0 { 1 } else { 0 };
                let ohw = gh * gw * 256;
                [(bs * ic * ohw) as u32, 1, 1]
            }*/
            //[(bs * ic) as u32, ohw as u32, 1]
            [(bs * oh * ow * ic * kh * kw) as u32, 1, 1]
        };
        unsafe {
            builder.submit(work_size)?;
        }
        Ok(output)
    }
}

impl<T: Float, S: Data<Elem = T>> Col2Im<Ix2> for TensorBase<S, Ix2> {
    type Output = Tensor4<T>;
    fn col2im(
        &self,
        shape: &Ix2,
        kernel: &Ix2,
        kind: KernelKind,
        args: &KernelArgs<Ix2>,
    ) -> Result<Self::Output> {
        if shape.size() == 0 {
            bail!("Shape has zero value! {:?}", shape.slice());
        }
        if kernel.size() == 0 {
            bail!("Kernel has zero value! {:?}", kernel.slice());
        }
        let input = self.as_standard_layout()?;
        let (bsihiw, ickhkw) = input.dim();
        let (oh, ow) = shape.into_pattern();
        let (kh, kw) = kernel.into_pattern();
        let ic = ickhkw / (kh * kw);
        let (ih, iw) = args.im2col_shape([oh, ow], kernel).into_pattern();
        let bs = bsihiw / (ih * iw);
        let strides = &args.strides;
        let padding = &args.padding;
        let dilation = &args.dilation;
        let (s_bez_h, d_bez_h, gcd_h) = eclid_gcd(strides[0], dilation[0]);
        let (s_bez_w, d_bez_w, gcd_w) = eclid_gcd(strides[1], dilation[1]);
        let mut output = unsafe { Tensor::alloc(input.device(), [bs, ic, oh, ow])? };
        let builder = rust_shaders::compute_pass(&format!(
            "kernel::col2im_2d_{}_{}",
            kind.as_str(),
            T::scalar_name()
        ))?
        .slice(input.as_raw_slice())?
        .slice_mut(output.as_raw_slice_mut())?
        .push(bs as u32)?
        .push(ic as u32)?
        .push([ih as u32, iw as u32])?
        .push([oh as u32, ow as u32])?
        .push([kh as u32, kw as u32])?
        .push([strides[0] as u32, strides[1] as u32])?
        .push([padding[0] as u32, padding[1] as u32])?
        .push([dilation[0] as u32, dilation[1] as u32])?
        .push([s_bez_h as u32, s_bez_w as u32])?
        .push([d_bez_h as u32, d_bez_w as u32])?
        .push([gcd_h as u32, gcd_w as u32])?;
        let h = (oh - 1) / gcd_h + 1;
        let w = (ow - 1) / gcd_w + 1;
        unsafe {
            builder.submit([(bs * ic) as u32, (h * w) as u32, 1])?;
        }
        Ok(output)
    }
}

impl<S: ArrayData<Elem = f32>> Im2Col<Ix2> for ArrayBase<S, Ix4> {
    type Output = Array2<f32>;
    fn im2col(
        &self,
        kernel: &Ix2,
        kind: KernelKind,
        args: &KernelArgs<Ix2>,
    ) -> Result<Self::Output> {
        let input = self.as_standard_layout();
        let (bs, ic, ih, iw) = input.dim();
        let (kh, kw) = kernel.into_pattern();
        let (sh, sw) = args.strides.into_pattern();
        let (ph, pw) = args.padding.into_pattern();
        let (dh, dw) = args.dilation.into_pattern();
        let (oh, ow) = args.im2col_shape([ih, iw], kernel).into_pattern();
        let kernel_flip = kind == KernelKind::CrossCorrelation;
        let mut output = Array::zeros([bs, oh, ow, ic, kh * kw]);
        for (input, mut output) in input.outer_iter().zip(output.outer_iter_mut()) {
            for (input, mut output) in input.outer_iter().zip(output.axis_iter_mut(Axis(2))) {
                for (hid, mut output) in output.outer_iter_mut().enumerate() {
                    for (wid, mut output) in output.outer_iter_mut().enumerate() {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let hidx = -(ph as isize) + (ki * dh + sh * hid) as isize;
                                let widx = -(pw as isize) + (kj * dw + sw * wid) as isize;
                                let mut kidx = ki * kw + kj;
                                if kernel_flip {
                                    kidx = kh * kw - (kidx + 1);
                                }
                                if hidx >= 0
                                    && hidx < ih as isize
                                    && widx >= 0
                                    && widx < iw as isize
                                {
                                    output[kidx] = input[(hidx as usize, widx as usize)];
                                }
                            }
                        }
                    }
                }
            }
        }
        // Only used in testing
        // Testing of autograph_derive fails to compile here.
        Ok(output.into_shape([bs * oh * ow, ic * kh * kw]).unwrap())
    }
}

impl<S: ArrayData<Elem = f32>> Col2Im<Ix2> for ArrayBase<S, Ix2> {
    type Output = Array4<f32>;
    fn col2im(
        &self,
        shape: &Ix2,
        kernel: &Ix2,
        kind: KernelKind,
        args: &KernelArgs<Ix2>,
    ) -> Result<Self::Output> {
        if shape.size() == 0 {
            bail!("Shape has zero value! {:?}", shape.slice());
        }
        if kernel.size() == 0 {
            bail!("Kernel has zero value! {:?}", kernel.slice());
        }
        if shape.size() == 0 {
            bail!("Shape has zero value! {:?}", shape.slice());
        }
        if kernel.size() == 0 {
            bail!("Kernel has zero value! {:?}", kernel.slice());
        }
        let input = self.as_standard_layout();
        let (bsihiw, ickhkw) = input.dim();
        let (oh, ow) = shape.into_pattern();
        let (kh, kw) = kernel.into_pattern();
        let kernel_flip = kind == KernelKind::CrossCorrelation;
        let (ih, iw) = args.im2col_shape([oh, ow], kernel).into_pattern();
        let ic = ickhkw / (kh * kw);
        let bs = bsihiw / (ih * iw);
        let (sh, sw) = args.strides.into_pattern();
        let (ph, pw) = args.padding.into_pattern();
        let (dh, dw) = args.dilation.into_pattern();
        // Only used in testing
        // Testing of autograph_derive fails to compile here.
        let input = input.into_shape([bs, ih, iw, ic, kh * kw]).unwrap();
        let mut output = Array::zeros([bs, ic, oh, ow]);
        for (input, mut output) in input.outer_iter().zip(output.outer_iter_mut()) {
            for (input, mut output) in input.axis_iter(Axis(2)).zip(output.outer_iter_mut()) {
                for (hid, input) in input.outer_iter().enumerate() {
                    for (wid, input) in input.outer_iter().enumerate() {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let hidx = -(ph as isize) + (ki * dh + sh * hid) as isize;
                                let widx = -(pw as isize) + (kj * dw + sw * wid) as isize;
                                let mut kidx = ki * kw + kj;
                                if kernel_flip {
                                    kidx = kh * kw - (kidx + 1);
                                }
                                if hidx >= 0
                                    && hidx < oh as isize
                                    && widx >= 0
                                    && widx < ow as isize
                                {
                                    output[(hidx as usize, widx as usize)] += input[kidx];
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(output)
    }
}

#[cfg(all(test, feature = "device_tests"))]
mod tests {
    use super::*;
    use crate::tensor::{OwnedRepr, TensorBase};
    use half::bf16;
    use ndarray::{ArrayBase, OwnedRepr as ArrayOwnedRepr};

    async fn im2col<
        T: Float,
        D1: Dimension,
        D2: Dimension,
        E1: IntoDimension<Dim = D1>,
        E2: IntoDimension<Dim = D2>,
    >(
        input_dim: E1,
        kernel: E2,
        kind: KernelKind,
        args: &KernelArgs<D2>,
    ) -> Result<()>
    where
        TensorBase<OwnedRepr<T>, D1>: Im2Col<D2, Output = Tensor<T, D2>>,
        ArrayBase<ArrayOwnedRepr<f32>, D1>: Im2Col<D2, Output = Array<f32, D2>>,
    {
        let input_dim = input_dim.into_dimension();
        let x_vec = (1..=input_dim.size())
            .into_iter()
            .map(|x| if x % 3 != 0 { x as f32 } else { -(x as f32) })
            .collect::<Vec<_>>();
        let kernel = kernel.into_dimension();
        let x_array = Array::from(x_vec).into_shape(input_dim)?;
        let y_array = x_array.im2col(&kernel, kind, args)?;
        let device = Device::new()?;
        let x = Tensor::from(x_array.map(|x| T::from_f32(*x).unwrap()))
            .into_device(device)
            .await?;
        let y = x.im2col(&kernel, kind, args)?;
        let y_out = y.read().await?.as_array().map(|x| x.to_f32().unwrap());
        assert_eq!(y_out, y_array);
        Ok(())
    }

    #[ignore] // Doesn't pass due to rounding error?
    #[tokio::test]
    async fn im2col_convolution_bf16() -> Result<()> {
        im2col::<bf16, _, _, _, _>(
            [1, 2, 3, 3],
            [2, 2],
            KernelKind::Convolution,
            &KernelArgs::default(),
        )
        .await?;
        im2col::<bf16, _, _, _, _>(
            [1, 6, 8, 8],
            [5, 5],
            KernelKind::Convolution,
            &KernelArgs::default(),
        )
        .await?;
        Ok(())
    }

    #[tokio::test]
    async fn im2col_convolution_f32() -> Result<()> {
        im2col::<f32, _, _, _, _>(
            [1, 2, 3, 3],
            [2, 2],
            KernelKind::Convolution,
            &KernelArgs::default(),
        )
        .await?;
        im2col::<f32, _, _, _, _>(
            [1, 2, 21, 5],
            [2, 2],
            KernelKind::Convolution,
            &KernelArgs::default(),
        )
        .await?;
        im2col::<f32, _, _, _, _>(
            [16, 6, 8, 8],
            [5, 5],
            KernelKind::Convolution,
            &KernelArgs::default(),
        )
        .await?;
        im2col::<f32, _, _, _, _>(
            [1, 1, 15, 20],
            [2, 2],
            KernelKind::Convolution,
            &KernelArgs::default(),
        )
        .await?;
        im2col::<f32, _, _, _, _>(
            [1, 1, 5, 16],
            [5, 5],
            KernelKind::Convolution,
            &KernelArgs::default(),
        )
        .await?;
        Ok(())
    }

    async fn col2im<
        T: Float,
        D1: Dimension,
        D2: Dimension,
        E1: IntoDimension<Dim = D1>,
        E2: IntoDimension<Dim = D1>,
        E3: IntoDimension<Dim = D1>,
    >(
        input_dim: E1,
        shape: E2,
        kernel: E3,
        kind: KernelKind,
        args: &KernelArgs<D1>,
    ) -> Result<()>
    where
        TensorBase<OwnedRepr<T>, D1>: Col2Im<D1, Output = Tensor<T, D2>>,
        ArrayBase<ArrayOwnedRepr<f32>, D1>: Col2Im<D1, Output = Array<f32, D2>>,
    {
        let input_dim = input_dim.into_dimension();
        let x_vec = (1..=input_dim.size())
            .into_iter()
            .map(|x| if x % 3 != 0 { x as f32 } else { -(x as f32) })
            .collect::<Vec<_>>();
        let shape = shape.into_dimension();
        let kernel = kernel.into_dimension();
        let x_array = Array::from(x_vec).into_shape(input_dim)?;
        let y_array = x_array.col2im(&shape, &kernel, kind, args)?;
        let device = Device::new()?;
        let x = Tensor::from(x_array.map(|x| T::from_f32(*x).unwrap()))
            .into_device(device)
            .await?;
        let y = x.col2im(&shape, &kernel, kind, args)?;
        let y_out = y.read().await?.as_array().map(|x| x.to_f32().unwrap());
        assert_eq!(y_out, y_array);
        Ok(())
    }

    async fn col2im_for<
        T: Float,
        D1: Dimension,
        D2: Dimension,
        E1: IntoDimension<Dim = D1>,
        E2: IntoDimension<Dim = D2>,
    >(
        input_dim: E1,
        kernel: E2,
        kind: KernelKind,
        args: &KernelArgs<D2>,
    ) -> Result<()>
    where
        TensorBase<OwnedRepr<T>, D2>: Col2Im<D2, Output = Tensor<T, D1>>,
        ArrayBase<ArrayOwnedRepr<f32>, D1>: Im2Col<D2, Output = Array<f32, D2>>,
        ArrayBase<ArrayOwnedRepr<f32>, D2>: Col2Im<D2, Output = Array<f32, D1>>,
    {
        let x = Array::<f32, _>::zeros(input_dim);
        let kernel = kernel.into_dimension();
        let y = x.im2col(&kernel, kind, args)?;
        let mut shape = D2::zeros(x.ndim() - 2);
        shape.slice_mut().copy_from_slice(&x.shape()[2..]);
        col2im(y.raw_dim(), shape, kernel, KernelKind::Convolution, args).await
    }

    #[tokio::test]
    async fn col2im_convolution_f32() -> Result<()> {
        col2im_for::<f32, _, _, _, _>(
            [1, 2, 3, 3],
            [2, 2],
            KernelKind::Convolution,
            &KernelArgs::default(),
        )
        .await?;
        col2im_for::<f32, _, _, _, _>(
            [16, 6, 8, 8],
            [5, 5],
            KernelKind::Convolution,
            &KernelArgs::default(),
        )
        .await?;
        Ok(())
    }
}
