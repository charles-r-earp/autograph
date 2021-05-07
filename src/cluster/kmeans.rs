#![allow(unused)]
use crate::{
    backend::{Device, Float},
    dataset::{train_test_split, Dataset},
    learn::{Fit, FitOptions, FitStats, Predict},
    tensor::{
        Axis, Data, Ix2, Tensor, Tensor0, Tensor1, Tensor2, TensorBase, TensorView1, TensorView2,
        TensorViewMut1, TensorViewMut2,
    },
    util::type_eq,
    Result,
};
use anyhow::{anyhow, ensure};
use futures_util::try_join;
use half::bf16;
use ndarray::{Array, ArrayBase, Dimension, Ix1};
use rand::distributions::{Distribution, Uniform};
use std::future::Future;

pub struct KMeans<T: Float> {
    centroids: Tensor2<T>,
}

impl<T: Float> KMeans<T> {
    pub fn from_centroids(centroids: Tensor2<T>) -> Self {
        Self { centroids }
    }
    pub fn new(device: &Device, k: usize) -> Result<Self> {
        Ok(Self::from_centroids(Tensor2::zeros(device, [k, 0])?))
    }
    pub async fn init_random<X, S, A>(&mut self, dataset: &A) -> Result<()>
    where
        X: Float,
        S: Data<Elem = X>,
        A: Dataset<Item = TensorBase<S, Ix2>>,
    {
        /// TODO: implement without copy back to host
        let uniform = Uniform::new(0, dataset.sample_count());
        let k = self.centroids.dim().0;
        let device = self.centroids.device();
        let n = dataset.sample(device, 0, 1).unwrap().await?.dim().1;
        let mut centroids = Array::from_elem([k, n], T::zero());
        for (i, c) in uniform
            .sample_iter(&mut rand::thread_rng())
            .zip(centroids.outer_iter_mut())
        {
            let x = dataset.sample(device, i, 1).unwrap().await?;
            let x = x.cast_into()?;
            c.into_shape([1, n]).unwrap().assign(&x.to_array()?.await?);
        }
        self.centroids = Tensor::from_array(device, centroids)?;
        Ok(())
    }
    /*pub async fn init_plus_plus<X, S, A>(&mut self, dataset: &A) -> Result<()>
        where X: Float, S: Data<Elem=X>, A: Dataset<Item=TensorBase<S, Ix2>> {
        todo!()
    }*/
    /// Computes the distance between the input batch and each centroid
    fn compute_distances(&self, input: &TensorView2<T>) -> Result<Tensor2<T>> {
        if let Some((x_slice, c_slice)) = input
            .as_buffer_slice()
            .zip(self.centroids.as_buffer_slice())
        {
            let device = self.centroids.device();
            let src = if type_eq::<T, bf16>() {
                include_shader!("glsl/kmeans_distance_bf16.spv")
            } else if type_eq::<T, f32>() {
                include_shader!("glsl/kmeans_distance_f32.spv")
            } else {
                unreachable!()
            };
            let (batch_size, ndims) = input.dim();
            let (nclasses, _ndims) = self.centroids.dim();
            ensure!(ndims == _ndims);
            let mut output = unsafe { Tensor::uninitialized(&device, (batch_size, nclasses))? };
            let batch_size = batch_size as u32;
            let nclasses = nclasses as u32;
            let ndims = ndims as u32;
            device
                .compute_pass(src, "main")?
                .buffer_slice(x_slice)?
                .buffer_slice(c_slice)?
                .buffer_slice_mut(output.as_unordered_buffer_slice_mut())?
                .push_constants(bytemuck::cast_slice(&[batch_size, nclasses, ndims]))?
                .global_size([batch_size, nclasses, 1])
                .enqueue()?;
            Ok(output)
        } else {
            Err(anyhow!("Tensors must be standard layout!"))
        }
    }
    fn update_centroids(
        &mut self,
        next_centroids: &TensorView2<T>,
        counts: &TensorView1<u32>,
    ) -> Result<()> {
        let device = next_centroids.device();
        let src = if type_eq::<T, bf16>() {
            include_shader!("glsl/kmeans_update_centroids_bf16.spv")
        } else if type_eq::<T, f32>() {
            include_shader!("glsl/kmeans_update_centroids_f32.spv")
        } else {
            unreachable!()
        };
        let (nclasses, ndim) = self.centroids.dim();
        let nclasses = nclasses as u32;
        let ndim = ndim as u32;
        device
            .compute_pass(src, "main")?
            .buffer_slice(next_centroids.as_buffer_slice().unwrap())?
            .buffer_slice(counts.as_buffer_slice().unwrap())?
            .buffer_slice_mut(self.centroids.as_buffer_slice_mut().unwrap())?
            .push_constants(bytemuck::cast_slice(&[nclasses, ndim]))?
            .global_size([nclasses, ndim, 1])
            .enqueue()
    }
    fn accumulate_next_centroids(
        next_centroids: &mut TensorViewMut2<T>,
        counts: &mut TensorViewMut1<u32>,
        input: &TensorView2<T>,
        classes: &TensorView1<u32>,
    ) -> Result<()> {
        let device = input.device();
        let src = if type_eq::<T, bf16>() {
            include_shader!("glsl/kmeans_accumulate_next_centroids_bf16.spv")
        } else if type_eq::<T, f32>() {
            include_shader!("glsl/kmeans_accumulate_next_centroids_f32.spv")
        } else {
            unreachable!()
        };
        let (nclasses, ndim) = next_centroids.dim();
        let nclasses = nclasses as u32;
        let ndim = ndim as u32;
        let batch_size = classes.dim() as u32;
        device
            .compute_pass(src, "main")?
            .buffer_slice_mut(next_centroids.as_buffer_slice_mut().unwrap())?
            .buffer_slice_mut(counts.as_buffer_slice_mut().unwrap())?
            .buffer_slice(input.as_buffer_slice().unwrap())?
            .buffer_slice(classes.as_buffer_slice().unwrap())?
            .push_constants(bytemuck::cast_slice(&[batch_size, nclasses, ndim]))?
            .global_size([nclasses, ndim, 1])
            .enqueue()
    }
    /// Trains the model with the train data, returns the average loss\
    ///
    /// Loss is the sum of the squared distances between the input and the closest centroid
    pub fn centroids(&self) -> &Tensor2<T> {
        &self.centroids
    }
}

impl<T: Float, X: Float, S: Data<Elem = X>> Fit<TensorBase<S, Ix2>> for KMeans<T> {
    fn initialize_from_dataset<A>(&mut self, dataset: &A, options: &FitOptions) -> Result<FitStats>
    where
        A: Dataset<Item = TensorBase<S, Ix2>>,
    {
        // TODO: Replace with init_plus_plus
        smol::block_on(self.init_random(dataset))?;
        Ok(FitStats::default())
    }
    fn train_epoch<I>(&mut self, device: &Device, train_iter: I) -> Result<(Tensor0<f32>, Option<Tensor0<u32>>)>
    where
        I: Iterator<Item = Result<TensorBase<S, Ix2>>>,
    {
        let device = self.centroids.device().clone();
        let mut loss = Tensor::zeros(&device, ())?;
        let mut next_centroids = Tensor::zeros(&device, self.centroids.raw_dim())?;
        let mut counts = Tensor::zeros(&device, self.centroids.dim().0)?;
        let mut num_samples = 0;
        for x in train_iter {
            let x = x?.cast_into()?;
            let distances = self.compute_distances(&x.view())?;
            let classes = distances.argmin(Axis(1))?;
            Self::accumulate_next_centroids(
                &mut next_centroids.view_mut(),
                &mut counts.view_mut(),
                &x.view(),
                &classes.view(),
            )?;
            distances
                .index_select(Axis(1), &classes.view())?
                .sum_with(Axis(0), &mut loss.view_mut())?;
            num_samples += x.dim().0;
        }
        self.update_centroids(&next_centroids.view(), &counts.view())?;
        let alpha = if num_samples > 0 {
            1. / num_samples as f32
        } else {
            0.
        };
        let loss = loss.scale_into(alpha)?;
        Ok((loss, None))
    }
    fn test_epoch<I>(&self, device: &Device, test_iter: I) -> Result<(Tensor0<f32>, Option<Tensor0<u32>>)>
    where
        I: Iterator<Item = Result<TensorBase<S, Ix2>>>,
    {
        let device = self.centroids.device().clone();
        let mut loss = Tensor::zeros(&device, ())?;
        let mut num_samples = 0;
        for x in test_iter {
            let x = x?.cast_into()?;
            let distances = self.compute_distances(&x.view())?;
            let classes = distances.argmin(Axis(1))?;
            distances
                .index_select(Axis(1), &classes.view())?
                .sum_with(Axis(0), &mut loss.view_mut())?;
            num_samples += x.dim().0;
        }
        let alpha = if num_samples > 0 {
            1. / num_samples as f32
        } else {
            0.
        };
        let loss = loss.scale_into(alpha)?;
        Ok((loss, None))
    }
}

impl<T: Float, S: Data<Elem = T>> Predict<TensorBase<S, Ix2>> for KMeans<T> {
    fn predict(&self, input: TensorBase<S, Ix2>) -> Result<Tensor1<u32>> {
        self.compute_distances(&input.view())?.argmin(Axis(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{Array1, Array2, ArrayView2};

    fn array_compute_distances(x: &ArrayView2<f32>, c: &ArrayView2<f32>) -> Array2<f32> {
        let mut y = Array::zeros([x.dim().0, c.dim().0]);
        for (x, mut y) in x.outer_iter().zip(y.outer_iter_mut()) {
            for (c, y) in c.outer_iter().zip(y.iter_mut()) {
                *y = x.iter().zip(c.iter()).map(|(x, c)| (x - c) * (x - c)).sum();
            }
        }
        y
    }

    fn compute_distances<T: Float + From<u8>>(m: usize, k: usize, n: usize) -> Result<()> {
        use crate::tensor::Scalar;
        for device in Device::list() {
            let v1: Vec<T> = (1..=100)
                .into_iter()
                .cycle()
                .step_by(7)
                .map(|x| T::from(x as u8))
                .take(m * k)
                .collect();
            let v2: Vec<T> = (1..=100)
                .into_iter()
                .cycle()
                .skip(13)
                .step_by(3)
                .map(|x| T::from(x as u8))
                .take(n * k)
                .collect();
            let a1 = Array::from_shape_vec([m, k], v1)?;
            let a2 = Array::from_shape_vec([n, k], v2)?;
            let y_true = {
                let a1 = a1.map(|x| x.to_f32().unwrap());
                let a2 = a2.map(|x| x.to_f32().unwrap());
                array_compute_distances(&a1.view(), &a2.view())
            };
            let t1 = Tensor::from_array(&device, a1.view())?;
            let t2 = Tensor::from_array(&device, a2.view())?;
            let kmeans = KMeans::from_centroids(t2);
            let y = smol::block_on(kmeans.compute_distances(&t1.view())?.to_array()?)?;
            let y = y.map(|&x| x.to_f32().unwrap());
            if type_eq::<T, f32>() {
                assert_relative_eq!(y, y_true, max_relative = 0.000_1);
            } else if type_eq::<T, bf16>() {
                assert_relative_eq!(y, y_true, epsilon = 0.01, max_relative = 0.01);
            }
        }
        Ok(())
    }

    #[test]
    fn compute_distances_bf16_m11_k5_n13() -> Result<()> {
        compute_distances::<bf16>(11, 5, 13)
    }

    #[test]
    fn compute_distances_f32_m11_k5_n13() -> Result<()> {
        compute_distances::<f32>(11, 5, 13)
    }

    #[test]
    fn train_epoch_f32() -> Result<()> {
        let k: usize = 3;
        let n: usize = 4;
        let num_samples: usize = 21;
        let data: Vec<f32> = (1..=100)
            .into_iter()
            .map(|x| x as f32)
            .cycle()
            .take(num_samples * n)
            .collect();
        let data = Array::from_shape_vec([num_samples, n], data)?;
        for device in Device::list() {
            let mut kmeans = KMeans::<f32>::from_centroids(Tensor::ones(&device, [k, n])?);
            let data_iter = data
                .axis_chunks_iter(Axis(0), 13)
                .into_iter()
                .map(|x| Tensor::from_array(&device, x));
            let (loss, _) = kmeans.train_epoch(&device, data_iter)?;
            let loss = smol::block_on(loss.to_array()?)?;
        }
        Ok(())
    }

    #[test]
    fn test_epoch_f32() -> Result<()> {
        let k: usize = 3;
        let n: usize = 4;
        let num_samples: usize = 21;
        let data: Vec<f32> = (1..=100)
            .into_iter()
            .map(|x| x as f32)
            .cycle()
            .take(num_samples * n)
            .collect();
        let data = Array::from_shape_vec([num_samples, n], data)?;
        for device in Device::list() {
            let kmeans = KMeans::<f32>::from_centroids(Tensor::ones(&device, [k, n])?);
            let data_iter = data
                .axis_chunks_iter(Axis(0), 13)
                .into_iter()
                .map(|x| Tensor::from_array(&device, x));
            let (loss, _) = kmeans.test_epoch(&device, data_iter)?;
            let loss = smol::block_on(loss.to_array()?)?;
        }
        Ok(())
    }
}
