use super::*;
use ndarray::{ArrayView2, ArrayViewMut2};

fn test_tensor_from_vec(device: impl Into<Device>) {
    let device = device.into();
    let vec = vec![1., 2., 3., 4.];
    let x = Tensor::from_shape_vec(&device, vec.len(), &vec);
    let vec_out = x.as_slice().into_owned();
    assert_eq!(vec, vec_out);
}
#[test]
fn test_tensor_from_vec_cpu() {
    test_tensor_from_vec(Cpu::new());
}
#[cfg(feature = "cuda")]
#[test]
fn test_tensor_from_vec_cuda() {
    test_tensor_from_vec(CudaGpu::new(0));
}
fn test_u8_to_f32(device: impl Into<Device>) {
    let device = device.into();
    let vec: Vec<u8> = vec![1, 2, 3, 4];
    let x = Tensor::from_shape_vec(&device, vec.len(), &vec);
    let y = x.to_f32();
    let vec_out = y.as_slice().into_owned();
    let scale = 255f32.recip();
    let vec_true: Vec<f32> = vec.iter().map(|x| scale * x.to_f32().unwrap()).collect();
    assert_eq!(vec_out, vec_true);
}
#[test]
fn test_u8_to_f32_cpu() {
    test_u8_to_f32(Cpu::new());
}
#[cfg(feature = "cuda")]
fn test_u8_to_f32_cuda() {
    test_u8_to_f32(CudaGpu::new(0));
}
fn test_u8_to_one_hot_f32(device: impl Into<Device>) {
    let device = device.into();
    let vec: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
    let x = Tensor::from_shape_vec(&device, vec.len(), &vec);
    let y = x.to_one_hot_f32(8);
    let mut y_true = Array::zeros([6, 8]);
    y_true
        .outer_iter_mut()
        .into_iter()
        .zip(vec.iter())
        .for_each(|(mut y, &x)| {
            y[x as usize] = 1.;
        });
    let y_out = y.as_array().into_owned();
    assert_eq!(y_out, y_true);
}
#[test]
fn test_u8_to_one_hot_f32_cpu() {
    test_u8_to_one_hot_f32(Cpu::new());
}
#[cfg(feature = "cuda")]
#[test]
fn test_u8_to_one_hot_f32_cuda() {
    test_u8_to_one_hot_f32(CudaGpu::new(0));
}
fn test_fill_u8(device: impl Into<Device>) {
    let device = device.into();
    let n = 10;
    let mut x = Tensor::zeros(&device, n);
    assert_eq!(x.as_slice(), vec![0u8; n].as_slice());
    x.fill(1u8);
    assert_eq!(x.as_slice(), vec![1u8; n].as_slice());
}
#[test]
fn test_fill_u8_cpu() {
    test_fill_u8(Cpu::new());
}
#[cfg(feature = "cuda")]
#[test]
fn test_fill_u8_cuda() {
    test_fill_u8(CudaGpu::new(0));
}
fn test_fill_f32(device: impl Into<Device>) {
    let device = device.into();
    let n = 10;
    let mut x = Tensor::zeros(&device, n);
    assert_eq!(x.as_slice(), vec![0f32; n].as_slice());
    x.fill(1f32);
    assert_eq!(x.as_slice(), vec![1f32; n].as_slice());
}
#[test]
fn test_fill_f32_cpu() {
    test_fill_f32(Cpu::new());
}
#[cfg(feature = "cuda")]
#[test]
fn test_fill_f32_cuda() {
    test_fill_f32(CudaGpu::new(0));
}
fn test_broadcast(device: impl Into<Device>) {
    let device = device.into();
    let x = Tensor::from_shape_vec(&device, 4, vec![1., 2., 3., 4.]);
    let mut y = Tensor::zeros(&device, [2, 4]);
    broadcast(&x, &mut y);
    let y_out = y.as_slice().into_owned();
    let y_true = vec![1., 2., 3., 4., 1., 2., 3., 4.];
    assert_eq!(y_out, y_true);
}
#[test]
fn test_broadcast_cpu() {
    test_broadcast(Cpu::new());
}
#[cfg(feature = "cuda")]
#[test]
fn test_broadcast_cuda() {
    test_broadcast(CudaGpu::new(0));
}
fn test_broadcast_backward(device: impl Into<Device>) {
    let device = device.into();
    let mut dx = Tensor::zeros(&device, 4);
    let dy = Tensor::from_shape_vec(&device, [2, 4], vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    broadcast_backward(&mut dx, &dy);
    let dx_out = dx.as_slice().into_owned();
    let dx_true = vec![6., 8., 10., 12.];
    assert_eq!(dx_out, dx_true);
}
#[test]
fn test_broadcast_backward_cpu() {
    test_broadcast_backward(Cpu::new());
}
#[cfg(feature = "cuda")]
#[test]
fn test_broadcast_backward_cuda() {
    test_broadcast_backward(CudaGpu::new(0));
}
fn compare_vectors(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    // dnnl and cuda have fast mul / approx ops
    // compared to ndarray / matrixmultiply which performs exact ops
    // assert_eq fails for gemm with large matrices
    // https://oneapi-src.github.io/oneDNN/cpu_sgemm_and_matmul_8cpp-example.html
    let mut v1_l2 = 0f64;
    let mut diff_l2 = 0f64;
    a.iter().zip(b.iter()).for_each(|(&a, &b)| {
        v1_l2 += (a * b) as f64;
        diff_l2 += (a - b).powi(2) as f64;
    });
    let threshold = (f32::EPSILON as f64) * f64::ln(f64::max(2., k.to_f64().unwrap()));
    assert!(
        diff_l2.sqrt() <= threshold * v1_l2.sqrt(),
        "m: {} k: {} n: {} ({} !<= {})",
        m,
        k,
        n,
        diff_l2.sqrt(),
        threshold * v1_l2.sqrt()
    );
}
fn test_gemm_mkn(m: usize, k: usize, n: usize, device: impl Into<Device>) {
    let device = device.into();

    let vec1: Vec<f32> = (1..=m * k)
        .into_iter()
        .map(|x| x.to_f32().unwrap())
        .collect();
    let vec2: Vec<f32> = (1..=k * n)
        .into_iter()
        .map(|x| x.to_f32().unwrap())
        .collect();

    {
        // MxK * KxN
        let x1 = Tensor::from_shape_vec(&device, [m, k], &vec1);
        let x2 = Tensor::from_shape_vec(&device, [k, n], &vec2);
        let mut y = Tensor::zeros(&device, [m, n]);
        gemm(1., &x1, Transpose::No, &x2, Transpose::No, 0., &mut y);
        let y_true = x1.as_array().dot(&x2.as_array());
        compare_vectors(&y.as_slice(), y_true.as_slice().unwrap(), m, k, n);
    }
    {
        // KxM^T * KxN
        let x1 = Tensor::from_shape_vec(&device, [k, m], &vec1);
        let x2 = Tensor::from_shape_vec(&device, [k, n], &vec2);
        let mut y = Tensor::zeros(&device, [m, n]);
        gemm(1., &x1, Transpose::Yes, &x2, Transpose::No, 0., &mut y);
        let y_true = x1.as_array().t().dot(&x2.as_array());
        compare_vectors(&y.as_slice(), y_true.as_slice().unwrap(), m, k, n);
    }
    {
        // MxK * NxK^T
        let x1 = Tensor::from_shape_vec(&device, [m, k], &vec1);
        let x2 = Tensor::from_shape_vec(&device, [n, k], &vec2);
        let mut y = Tensor::zeros(&device, [m, n]);
        gemm(1., &x1, Transpose::No, &x2, Transpose::Yes, 0., &mut y);
        let y_true = x1.as_array().dot(&x2.as_array().t());
        compare_vectors(&y.as_slice(), y_true.as_slice().unwrap(), m, k, n);
    }
    {
        // KxM^T * NxK^T
        let x1 = Tensor::from_shape_vec(&device, [k, m], &vec1);
        let x2 = Tensor::from_shape_vec(&device, [n, k], &vec2);
        let mut y = Tensor::zeros(&device, [m, n]);
        gemm(1., &x1, Transpose::Yes, &x2, Transpose::Yes, 0., &mut y);
        let y_true: Vec<f32> = x1
            .as_array()
            .t()
            .dot(&x2.as_array().t())
            .iter()
            .copied()
            .collect();
        compare_vectors(&y.as_slice(), &y_true, m, k, n);
    }
}
fn test_gemm(device: impl Into<Device>) {
    test_gemm_mkn(33, 43, 53, device);
}
#[test]
fn test_gemm_cpu() {
    test_gemm(Cpu::new());
}
#[cfg(feature = "cuda")]
#[test]
fn test_gemm_cuda() {
    test_gemm(CudaGpu::new(0));
}
fn test_sum(device: impl Into<Device>) {
    let device = device.into();

    let vec: Vec<f32> = (1..=100).into_iter().map(|x| x.to_f32().unwrap()).collect();

    let x = Tensor::from_shape_vec(&device, vec.len(), &vec);
    let y = x.sum().as_slice()[0];
    assert_eq!(y, vec.iter().sum::<f32>());
}
#[test]
fn test_sum_cpu() {
    test_sum(Cpu::new());
}
#[cfg(feature = "cuda")]
#[test]
fn test_sum_cuda() {
    test_sum(CudaGpu::new(0));
}
fn test_relu(device: impl Into<Device>) {
    let device = device.into();
    let x = Tensor::from_shape_vec(&device, 6, vec![-0.1, -100., 0.0, 0.1, 1., 100.]);
    let y = x.relu();
    let y_vec = y.as_slice().into_owned();
    debug_assert_eq!(y_vec, vec![0., 0., 0., 0.1, 1., 100.]);
}
#[test]
fn test_relu_cpu() {
    test_relu(Cpu::new());
}
#[cfg(feature = "cuda")]
fn test_relu_cuda() {
    test_relu(CudaGpu::new(0));
}
fn test_relu_backward(device: impl Into<Device>) {
    let device = device.into();
    let x = Tensor::from_shape_vec(&device, 6, vec![-0.1, -100., 0.0, 0.1, 1., 100.]);
    let mut dx = Tensor1::<f32>::ones(&device, 6);
    let dy = Tensor::from_shape_vec(&device, 6, vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6]);
    let dx_vec = dx.as_slice().into_owned();
    let mut dx_vec_true = vec![0.; dx_vec.len()];
    x.as_slice()
        .iter()
        .zip(dx_vec_true.iter_mut())
        .zip(dy.as_slice().iter())
        .for_each(|((&x, dx), &dy)| {
            if x >= 0. {
                *dx += dy;
            }
        });
    debug_assert_eq!(dx_vec, vec![0., 0., 0., 0.1, 1., 100.]);
}
#[test]
fn test_relu_backard_cpu() {
    test_relu(Cpu::new());
}
#[cfg(feature = "cuda")]
fn test_relu_backward_cuda() {
    test_relu(CudaGpu::new(0));
}
fn test_add(device: impl Into<Device>) {
    let device = device.into();
    let x1_vec = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
    let x2_vec = vec![11., 12., 13., 14., 15., 16., 17., 18., 19., 20.];
    let x1 = Tensor::from_shape_vec(&device, 10, x1_vec.as_slice());
    let x2 = Tensor::from_shape_vec(&device, 10, x2_vec.as_slice());
    let y = x1.add(&x2);
    let y_true: Vec<f32> = x1_vec.iter()
        .zip(x2_vec.iter())
        .map(|(&x1, &x2)| x1 + x2)
        .collect(); 
    assert_eq!(&*y.as_slice(), &*y_true);
}
#[test]
fn test_add_cpu() {
    test_add(Cpu::new());
}
#[cfg(feature="cuda")]
#[test]
fn test_add_cuda() {
    test_add(CudaGpu::new(0));
}
fn test_scaled_add(device: impl Into<Device>) {
    let device = device.into();

    let mut lhs = Tensor::zeros(&device, 10);
    let rhs = Tensor::from_shape_vec(&device, 10, vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);

    let alpha = 2.;

    lhs.scaled_add(alpha, &rhs);

    let mut lhs_true = Array::zeros(lhs.raw_dim());
    lhs_true.scaled_add(alpha, &rhs.as_array());

    let success = lhs
        .as_slice()
        .iter()
        .zip(lhs_true.as_slice().unwrap())
        .all(|(a, b)| approx::relative_eq!(a, b, max_relative = 0.00001));
    assert!(
        success,
        "{:?} {:?}",
        lhs.as_slice(),
        lhs_true.as_slice().unwrap()
    );
}
fn test_cross_entropy(device: impl Into<Device>) {
    let device = device.into();

    let batch_size = 3;
    let nclasses = 4;

    let input = Tensor::from_shape_vec(
        &device,
        [batch_size, nclasses],
        vec![1., 2., -3., -4., 5., 6., 7., 8., -9., -10., 11., 12.],
    );

    let target = Tensor::from_shape_vec(
        &device,
        [batch_size, nclasses],
        vec![1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
    );

    let mut output = Tensor::zeros(&device, [batch_size, nclasses]);

    match &device {
        Device::Cpu(_) => {
            cpu::cross_entropy(&input, &target, &mut output);
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            cuda::cross_entropy(&input, &target, &mut output);
        }
    }

    let mut output_true = vec![0.; batch_size * nclasses];
    input
        .as_slice()
        .chunks_exact(nclasses)
        .zip(target.as_slice().chunks_exact(nclasses))
        .zip(output_true.chunks_exact_mut(nclasses))
        .for_each(|((input, target), mut output)| {
            let mut m = input[0];
            input.iter().for_each(|&x| m = f32::max(x, m));
            output
                .iter_mut()
                .zip(input.iter())
                .for_each(|(y, &x)| *y = x - m);
            let s: f32 = output.iter().map(|&y| y.exp()).sum();
            let ln_s = s.ln();
            output
                .iter_mut()
                .zip(target.iter())
                .for_each(|(y, t)| *y = (ln_s - *y) * t);
        });
    let output = output.as_slice();
    let success = output
        .iter()
        .zip(output_true.as_slice())
        .all(|(a, b)| approx::relative_eq!(a, b, max_relative = 0.00001));
    assert!(success, "{:?} {:?}", output, output_true);
}
#[test]
fn test_cross_entropy_cpu() {
    test_cross_entropy(Cpu::new());
}
#[cfg(feature = "cuda")]
#[test]
fn test_cross_entropy_cuda() {
    test_cross_entropy(CudaGpu::new(0));
}
fn test_cross_entropy_backward(device: impl Into<Device>) {
    let device = device.into();

    let batch_size = 3;
    let nclasses = 4;

    let input = Tensor::from_shape_vec(
        &device,
        [batch_size, nclasses],
        vec![1., 2., -3., -4., 5., 6., 7., 8., -9., -10., 11., 12.],
    );

    let mut input_grad = Tensor::zeros(&device, [batch_size, nclasses]);

    let target = Tensor::from_shape_vec(
        &device,
        [batch_size, nclasses],
        vec![1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
    );

    let output_grad = Tensor::from_shape_vec(&device, (), vec![1.]);

    cross_entropy_backward(&input, &mut input_grad, &target, &output_grad);

    let mut input_grad_true = vec![0.; batch_size * nclasses];
    input
        .as_slice()
        .iter()
        .zip(input_grad_true.iter_mut())
        .zip(target.as_slice().iter())
        .for_each(|((x, mut dx), t)| {
            *dx = x - t;
        });
    let input_grad = input_grad.as_slice();
    let success = input_grad
        .iter()
        .zip(input_grad_true.as_slice())
        .all(|(a, b)| approx::relative_eq!(a, b, max_relative = 0.00001));
    assert!(success, "{:?} {:?}", input_grad, input_grad_true);
}
#[test]
fn test_cross_entropy_backward_cpu() {
    test_cross_entropy_backward(Cpu::new());
}
#[cfg(feature = "cuda")]
#[test]
fn test_cross_entropy_backward_cuda() {
    test_cross_entropy_backward(CudaGpu::new(0));
}
fn test_conv2d_with_args(
    input_dim: impl IntoDimension<Dim = Ix4>,
    outputs: usize,
    kernel: impl Into2d,
    use_bias: bool,
    args: &Conv2dArgs,
    device: impl Into<Device>,
) {
    let kernel = kernel.into_2d();
    let device = device.into();
    let input_dim = input_dim.into_dimension();
    let (batch_size, inputs, ih, iw) = input_dim.into_pattern();
    let [kh, kw] = kernel;
    let input_vec: Vec<f32> = (1..=input_dim.size())
        .into_iter()
        .map(|x| x.to_f32().unwrap())
        .collect();
    let weight_dim = [outputs, inputs, kh, kw].into_dimension();
    let weight_vec: Vec<f32> = (1..=weight_dim.size())
        .into_iter()
        .map(|x| x.to_f32().unwrap())
        .collect();
    let bias_vec: Vec<f32> = (1..=outputs)
        .into_iter()
        .map(|x| x.to_f32().unwrap())
        .collect();
    let output = {
        let input = Tensor::from_shape_vec(&device, input_dim, input_vec.as_slice());
        let weight = Tensor::from_shape_vec(&device, weight_dim, weight_vec.as_slice());
        let bias = if use_bias {
            Some(Tensor::from_shape_vec(
                &device,
                outputs,
                bias_vec.as_slice(),
            ))
        } else {
            None
        };
        input.conv2d(
            &weight.view(),
            bias.as_ref().map(|b| b.view()).as_ref(),
            args,
        )
    };
    let output_vec = output.as_slice().into_owned();
    let output_true = {
        let input = tch::Tensor::of_slice(&input_vec).reshape(&[
            batch_size as i64,
            inputs as i64,
            ih as i64,
            iw as i64,
        ]);
        let weight = tch::Tensor::of_slice(&weight_vec).reshape(&[
            outputs as i64,
            inputs as i64,
            kh as i64,
            kw as i64,
        ]);
        let bias = if use_bias {
            Some(tch::Tensor::of_slice(&bias_vec).reshape(&[outputs as i64]))
        } else {
            None
        };
        input.conv2d(
            &weight,
            bias.as_ref(),
            &[args.strides[0] as i64, args.strides[1] as i64],
            &[args.padding[0] as i64, args.padding[1] as i64],
            &[1, 1],
            1,
        )
    };
    let mut output_true_vec = vec![0f32; output_vec.len()];
    output_true.copy_data(&mut output_true_vec, output_vec.len());
    compare_vectors(
        &output_vec,
        &output_true_vec,
        batch_size,
        inputs * kh * kw,
        outputs,
    );
}
fn test_conv2d(device: impl Into<Device>) {
    let device = device.into();
    test_conv2d_with_args(
        [8, 16, 20, 20],
        12,
        [3, 3],
        false,
        &Conv2dArgs::default(),
        device.clone(),
    );
    test_conv2d_with_args(
        [8, 16, 20, 20],
        12,
        [3, 3],
        true,
        &Conv2dArgs::default().strides(2).padding(1),
        device.clone(),
    );
}
#[test]
fn test_conv2d_cpu() {
    test_conv2d(Cpu::new());
}
#[cfg(feature = "cuda")]
#[test]
fn test_conv2d_cuda() {
    test_conv2d(CudaGpu::new(0));
}
fn test_conv2d_backward_input_with_args(
    input_dim: impl IntoDimension<Dim = Ix4>,
    outputs: usize,
    kernel: impl Into2d,
    args: &Conv2dArgs,
    device: impl Into<Device>,
) {
    let kernel = kernel.into_2d();
    let device = device.into();
    let input_dim = input_dim.into_dimension();
    let (batch_size, inputs, ih, iw) = input_dim.into_pattern();
    let [kh, kw] = kernel;
    let [sh, sw] = args.strides;
    let [ph, pw] = args.padding;
    let oh = (ih - kh + 2 * ph) / sh + 1;
    let ow = (iw - kw + 2 * pw) / sw + 1;
    let weight_dim = [outputs, inputs, kh, kw].into_dimension();
    let weight_vec: Vec<f32> = (1..=weight_dim.size())
        .into_iter()
        .map(|x| x.to_f32().unwrap())
        .collect();
    let output_dim = [batch_size, outputs, oh, ow].into_dimension();
    let output_grad_vec: Vec<f32> = (1..=output_dim.size())
        .into_iter()
        .map(|x| x.to_f32().unwrap())
        .collect();
    let input_grad_vec = {
        let mut input_grad = Tensor::zeros(&device, input_dim);
        let weight = Tensor::from_shape_vec(&device, weight_dim, weight_vec.as_slice());
        let output_grad = Tensor::from_shape_vec(&device, output_dim, output_grad_vec.as_slice());
        conv2d_backward_input(&mut input_grad, &weight.view(), args, &output_grad.view());
        let input_grad_vec = input_grad.as_slice().into_owned();
        input_grad_vec
    };
    let input_grad_true_vec = {
        // testing cuda against cpu
        let device = Device::from(Cpu::new());
        let mut input_grad = Tensor::zeros(&device, input_dim);
        let weight = Tensor::from_shape_vec(&device, weight_dim, weight_vec.as_slice());
        let output_grad = Tensor::from_shape_vec(&device, output_dim, output_grad_vec.as_slice());
        conv2d_backward_input(&mut input_grad, &weight.view(), args, &output_grad.view());
        let input_grad_vec = input_grad.as_slice().into_owned();
        input_grad_vec
    };
    compare_vectors(
        &input_grad_vec,
        &input_grad_true_vec,
        batch_size,
        outputs,
        inputs,
    );
    assert!(
        approx::relative_eq!(
            &*input_grad_vec,
            &*input_grad_true_vec,
            max_relative = 0.001
        ),
        "\n{:?}\n !=\n{:?}",
        &input_grad_vec,
        &input_grad_true_vec
    );
}
fn test_conv2d_backward_input(device: impl Into<Device>) {
    let device = device.into();
    test_conv2d_backward_input_with_args(
        [1, 1, 3, 3],
        1,
        [2, 2],
        &Conv2dArgs::default(),
        device.clone(),
    );
    test_conv2d_backward_input_with_args(
        [40, 1, 28, 28],
        6,
        [5, 5],
        &Conv2dArgs::default(),
        device.clone(),
    );
    test_conv2d_backward_input_with_args(
        [40, 6, 24, 24],
        16,
        [5, 5],
        &Conv2dArgs::default(),
        device.clone(),
    );
}
/*#[test]
fn test_conv2d_backward_input_cpu() {
  test_conv2d_backward_input(Cpu::new());
}*/
#[cfg(feature = "cuda")]
#[test]
fn test_conv2d_backward_input_cuda() {
    test_conv2d_backward_input(CudaGpu::new(0));
}
fn test_conv2d_backward_weight_bias_with_args(
    input_dim: impl IntoDimension<Dim = Ix4>,
    outputs: usize,
    use_bias: bool,
    kernel: impl Into2d,
    args: &Conv2dArgs,
    device: impl Into<Device>,
) {
    let kernel = kernel.into_2d();
    let device = device.into();
    let input_dim = input_dim.into_dimension();
    let (batch_size, inputs, ih, iw) = input_dim.into_pattern();
    let [kh, kw] = kernel;
    let [sh, sw] = args.strides;
    let [ph, pw] = args.padding;
    let oh = (ih - kh + 2 * ph) / sh + 1;
    let ow = (iw - kw + 2 * pw) / sw + 1;
    let weight_dim = [outputs, inputs, kh, kw].into_dimension();
    let input_vec: Vec<f32> = (1..=input_dim.size())
        .into_iter()
        .map(|x| x.to_f32().unwrap())
        .collect();
    let output_dim = [batch_size, outputs, oh, ow].into_dimension();
    let output_grad_vec: Vec<f32> = (1..=output_dim.size())
        .into_iter()
        .map(|x| x.to_f32().unwrap())
        .collect();
    let (weight_grad_vec, bias_grad_vec) = {
        let input = Tensor::from_shape_vec(&device, input_dim, input_vec.as_slice());
        let mut weight_grad = Tensor::zeros(&device, weight_dim);
        let mut bias_grad = Tensor::zeros(&device, outputs);
        let output_grad = Tensor::from_shape_vec(&device, output_dim, output_grad_vec.as_slice());
        if use_bias {
            conv2d_backward_weight_bias(
                &input,
                &mut weight_grad.view_mut(),
                Some(&mut bias_grad.view_mut()),
                args,
                &output_grad.view(),
            );
        } else {
            conv2d_backward_weight_bias(
                &input,
                &mut weight_grad.view_mut(),
                None,
                args,
                &output_grad.view(),
            );
        }
        let weight_grad_vec = weight_grad.as_slice().into_owned();
        let bias_grad_vec = if use_bias {
            Some(bias_grad.as_slice().into_owned())
        } else {
            None
        };
        (weight_grad_vec, bias_grad_vec)
    };
    let (weight_grad_true_vec, bias_grad_true_vec) = {
        // testing cuda against cpu
        let device = Device::from(Cpu::new());
        let input = Tensor::from_shape_vec(&device, input_dim, input_vec.as_slice());
        let mut weight_grad = Tensor::zeros(&device, weight_dim);
        let mut bias_grad = Tensor::zeros(&device, outputs);
        let output_grad = Tensor::from_shape_vec(&device, output_dim, output_grad_vec.as_slice());
        if use_bias {
            conv2d_backward_weight_bias(
                &input,
                &mut weight_grad.view_mut(),
                Some(&mut bias_grad.view_mut()),
                args,
                &output_grad.view(),
            );
        } else {
            conv2d_backward_weight_bias(
                &input,
                &mut weight_grad.view_mut(),
                None,
                args,
                &output_grad.view(),
            );
        }
        let weight_grad_vec = weight_grad.as_slice().into_owned();
        let bias_grad_vec = if use_bias {
            Some(bias_grad.as_slice().into_owned())
        } else {
            None
        };
        (weight_grad_vec, bias_grad_vec)
    };
    assert!(
        approx::relative_eq!(
            &*weight_grad_vec,
            &*weight_grad_true_vec,
            max_relative = 0.001
        ),
        "\n{:?}\n !=\n{:?}",
        &weight_grad_vec,
        &weight_grad_true_vec
    );
    if use_bias {
        let bias_grad_vec = bias_grad_vec.unwrap();
        let bias_grad_true_vec = bias_grad_true_vec.unwrap();
        assert!(
            approx::relative_eq!(&*bias_grad_vec, &*bias_grad_true_vec, max_relative = 0.001),
            "\n{:?}\n !=\n{:?}",
            &bias_grad_vec,
            &bias_grad_true_vec
        );
    }
}
fn test_conv2d_backward_weight_bias(device: impl Into<Device>) {
    let device = device.into();
    test_conv2d_backward_weight_bias_with_args(
        [1, 1, 20, 20],
        6,
        false,
        [5, 5],
        &Conv2dArgs::default(),
        device.clone(),
    );
    test_conv2d_backward_weight_bias_with_args(
        [1, 6, 20, 20],
        16,
        true,
        [5, 5],
        &Conv2dArgs::default(),
        device.clone(),
    );
}
#[cfg(feature = "cuda")]
#[test]
fn test_conv2d_backward_weight_bias_cuda() {
    test_conv2d_backward_weight_bias(CudaGpu::new(0));
}
fn test_max_pool2d_with_args(
    input_dim: impl IntoDimension<Dim = Ix4>,
    args: &Pool2dArgs,
    device: impl Into<Device>,
) {
    let device = device.into();
    let input_dim = input_dim.into_dimension();
    let (batch_size, inputs, ih, iw) = input_dim.into_pattern();
    let input_vec: Vec<f32> = (1..=input_dim.size())
        .into_iter()
        .map(|x| x.to_f32().unwrap())
        .collect();
    let output = {
        let input = Tensor::from_shape_vec(&device, input_dim, input_vec.as_slice());
        input.max_pool2d(&args)
    };
    let output_vec = output.as_slice().into_owned();
    let output_true = {
        let input = tch::Tensor::of_slice(&input_vec).reshape(&[
            batch_size as i64,
            inputs as i64,
            ih as i64,
            iw as i64,
        ]);
        input.max_pool2d(
            &[args.kernel[0] as i64, args.kernel[1] as i64],
            &[args.strides[0] as i64, args.strides[1] as i64],
            &[args.padding[0] as i64, args.padding[1] as i64],
            &[1, 1],
            false,
        )
    };
    let (bs, o, oh, ow) = output_true.size4().unwrap();
    let output_dim_true = [bs as usize, o as usize, oh as usize, ow as usize].into_dimension();
    assert_eq!(output.raw_dim(), output_dim_true);
    let mut output_true_vec = vec![0f32; output_vec.len()];
    output_true.copy_data(&mut output_true_vec, output_vec.len());
    assert_eq!(output_vec, output_true_vec);
}
fn test_max_pool2d(device: impl Into<Device>) {
    let device = device.into();
    test_max_pool2d_with_args([8, 16, 20, 20], &Pool2dArgs::default(), device.clone());
}
#[test]
fn test_max_pool2d_cpu() {
    test_max_pool2d(Cpu::new());
}
#[cfg(feature = "cuda")]
#[test]
fn test_max_pool2d_cuda() {
    test_max_pool2d(CudaGpu::new(0));
}
fn test_max_pool2d_backward_with_args(
    input_dim: impl IntoDimension<Dim = Ix4>,
    args: &Pool2dArgs,
    device: impl Into<Device>,
) {
    let device = device.into();
    let input_dim = input_dim.into_dimension();
    let (batch_size, inputs, ih, iw) = input_dim.into_pattern();
    let [kh, kw] = args.kernel;
    let [sh, sw] = args.strides;
    let [ph, pw] = args.padding;
    let oh = (ih - (kh - 1) + 2 * ph - 1) / sh + 1;
    let ow = (iw - (kw - 1) + 2 * pw - 1) / sw + 1;
    let input_vec: Vec<f32> = (1..=input_dim.size())
        .into_iter()
        .map(|x| x.to_f32().unwrap())
        .collect();
    let output_dim = [batch_size, inputs, oh, ow].into_dimension();
    let output_grad_vec: Vec<f32> = (1..=output_dim.size())
        .into_iter()
        .map(|x| x.to_f32().unwrap())
        .collect();
    let input_grad_vec = {
        let input = Tensor::from_shape_vec(&device, input_dim, input_vec.as_slice());
        let (output, workspace) = max_pool2d_forward(&input, args, true);
        let mut input_grad = Tensor::zeros(&device, input_dim);
        let output_grad = Tensor::from_shape_vec(&device, output_dim, output_grad_vec.as_slice());
        max_pool2d_backward(
            &input,
            &mut input_grad,
            args,
            workspace.as_ref(),
            &output_grad,
        );
        input_grad.as_slice().into_owned()
    };
    let input_grad_true_vec = {
        // Currently testing cuda against cpu
        let device = Device::from(Cpu::new());
        let input = Tensor::from_shape_vec(&device, input_dim, input_vec.as_slice());
        let (output, workspace) = max_pool2d_forward(&input, args, true);
        let mut input_grad = Tensor::zeros(&device, input_dim);
        let output_grad = Tensor::from_shape_vec(&device, output_dim, output_grad_vec.as_slice());
        max_pool2d_backward(
            &input,
            &mut input_grad,
            args,
            workspace.as_ref(),
            &output_grad,
        );
        input_grad.as_slice().into_owned()
    };
    assert_eq!(input_grad_vec, input_grad_true_vec);
}
fn test_max_pool2d_backward(device: impl Into<Device>) {
    let device = device.into();
    test_max_pool2d_with_args([8, 16, 20, 20], &Pool2dArgs::default(), device.clone());
}
/*#[test]
fn test_max_pool2d_backward_cpu() {
  test_max_pool2d(Cpu::new());
}*/
#[cfg(feature = "cuda")]
fn test_max_pool_backward_cuda() {
    test_max_pool2d_backward(CudaGpu::new(0));
}
