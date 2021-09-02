use super::{Infer, Stats, Summarize, Summary, Test, Train};
use crate::{
    buffer::float::FloatSlice,
    device::Device,
    glsl_shaders,
    result::Result,
    scalar::{FloatType, Uint},
    tensor::{
        float::{
            FloatArcTensor2, FloatData, FloatTensor, FloatTensor2, FloatTensorBase, FloatTensorD,
            FloatTensorView2, FloatTensorViewMut2,
        },
        CowTensor, Tensor, Tensor2, TensorD, TensorView1, TensorViewMut1,
    },
};
use anyhow::bail;
use ndarray::{s, Array, Array2, ArrayView2, Axis, Dimension};
use rand::distributions::{Distribution, Uniform};
use std::future::Future;

async fn read_loss(count: usize, loss: FloatSlice<'_>) -> Result<f32> {
    if count > 0 {
        let loss = match loss {
            FloatSlice::BF16(loss) => loss.read().await?[0].to_f32(),
            FloatSlice::F32(loss) => loss.read().await?[0],
        };
        Ok(loss / count as f32)
    } else {
        Ok(0.)
    }
}

fn compute_distances(
    centroids: &FloatTensorView2,
    input: &FloatTensorView2,
) -> Result<FloatTensor2> {
    let device = centroids.device();
    let float_type = centroids.float_type();
    let name = format!("kmeans_distance_{}", float_type.as_str());
    let module = glsl_shaders::module(name)?;
    let (batch_size, ndims) = input.dim();
    let (nclasses, _ndims) = centroids.dim();
    if ndims != _ndims {
        bail!("Input ndim {:?} != Centroid ndim {:?}!", ndims, _ndims);
    }
    let mut output = match float_type {
        FloatType::BF16 => FloatTensor::zeros(float_type, device, (batch_size, nclasses))?,
        FloatType::F32 => unsafe {
            FloatTensor::alloc(float_type, device, (batch_size, nclasses))?
        },
    };
    let batch_size = batch_size as u32;
    let nclasses = nclasses as u32;
    let ndims = ndims as u32;
    let builder = module
        .compute_pass("main")?
        .float_slice(input.to_slice()?.as_slice())?
        .float_slice(centroids.to_slice()?.as_slice())?
        .float_slice_mut(output.as_raw_slice_mut())?
        .push([batch_size, nclasses, ndims])?;
    unsafe {
        builder.submit([batch_size, nclasses, 1])?;
    }
    Ok(output)
}

fn array_compute_distances(x: &ArrayView2<f32>, c: &ArrayView2<f32>) -> Array2<f32> {
    let mut y = Array::zeros([x.dim().0, c.dim().0]);
    for (x, mut y) in x.outer_iter().zip(y.outer_iter_mut()) {
        for (c, y) in c.outer_iter().zip(y.iter_mut()) {
            *y = x.iter().zip(c.iter()).map(|(x, c)| (x - c) * (x - c)).sum();
        }
    }
    y
}

// TODO: impl tests
#[allow(unused)]
fn array_init_kplus_plus(k: usize, input: &ArrayView2<f32>) -> Array2<f32> {
    let mut centroids = Array2::zeros([k, input.dim().1]);
    let index = Uniform::new(0, input.dim().0).sample(&mut rand::thread_rng());
    centroids
        .index_axis_mut(Axis(0), 0)
        .assign(&input.index_axis(Axis(0), index));
    for i in 1..centroids.dim().0 {
        let centroids_slice = centroids.slice(s![..i, ..]);
        let distances = array_compute_distances(input, &centroids_slice).map_axis(Axis(1), |x| {
            x.iter().fold(*x.first().unwrap(), |acc, x| acc.min(*x))
        });
        let distance_value = Uniform::new(0f32, distances.sum()).sample(&mut rand::thread_rng());
        let mut distance_counter = 0f32;
        let index = distances
            .iter()
            .enumerate()
            .find_map(|(u, d)| {
                if *d > 0f32 && distance_counter + *d >= distance_value {
                    Some(u)
                } else {
                    distance_counter += *d;
                    None
                }
            })
            .unwrap_or_else(|| distances.len());
        centroids
            .index_axis_mut(Axis(0), i)
            .assign(&input.index_axis(Axis(0), index));
    }
    centroids
}

/// KMeans Classifier
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KMeans {
    centroids: FloatArcTensor2,
}

impl KMeans {
    /// Creates a new KMeans with `k` means.
    ///
    /// # Note
    /// The model is unitialized. Use [`KMeansTrainer`] with [`Train`] to initialize and train the classifier.
    pub fn new(k: usize) -> Self {
        let centroids = FloatTensor::zeros(FloatType::F32, Device::host(), [k, 0]).unwrap();
        Self::from_centroids(centroids.into())
    }
    /// Transfers the model to `device`.
    ///
    /// See [`Tensor::into_device()`](crate::tensor::TensorBase::into_device()).
    pub async fn into_device(self, device: Device) -> Result<Self> {
        // TODO: impl into_device_shared
        Ok(Self::from_centroids(
            self.centroids.into_device(device).await?.into_shared()?,
        ))
    }
    /// Constructs a new [`KMeans`] from `centroids`.
    ///
    /// # Centroids
    /// - Should be standard layout.
    /// - Shape = [k, c], with k means of c dimensional points.
    pub fn from_centroids(centroids: FloatArcTensor2) -> Self {
        Self { centroids }
    }
    /// The centroids of the model.
    pub fn centroids(&self) -> &FloatArcTensor2 {
        &self.centroids
    }
    #[allow(clippy::many_single_char_names)]
    async fn init_kplus_plus<S, D, F, I>(&mut self, f: F) -> Result<()>
    where
        S: FloatData,
        D: Dimension,
        F: FnOnce(usize) -> I,
        I: IntoIterator,
        <I as IntoIterator>::Item: IntoIterator<Item = Result<FloatTensorBase<S, D>>>,
    {
        // TODO: impl fully on device.
        use half::bf16;
        let k = self.centroids.dim().0;
        let device = self.centroids.device();
        let float_type = self.centroids.float_type();
        let mut c = 0;
        let mut centroids = Tensor2::<f32>::zeros(device.clone(), [k, c])?;
        let mut iters = f(2 * k).into_iter();
        let mut n_samples = 0;
        for x in iters.next().unwrap() {
            let x = x?.flatten()?;
            if c == 0 {
                c = x.dim().1;
            }
            n_samples += x.dim().0;
        }
        let index = Uniform::new(0, n_samples).sample(&mut rand::thread_rng());
        let mut index_counter = 0;
        for x in iters.next().unwrap() {
            let x = x?.flatten()?;
            let batch_size = x.dim().0;
            if index_counter + batch_size >= index {
                let centroid = x.cast_to::<f32>()?;
                let centroid = centroid.read().await?;
                let centroid = centroid.as_array();
                let centroid = centroid
                    .index_axis(Axis(0), index - index_counter)
                    .into_shape([1, c])?;
                centroids = CowTensor::from(centroid)
                    .into_device(device.clone())
                    .await?;
                break;
            }
            index_counter += batch_size;
        }
        for _ in 1..k {
            let mut total_distance = FloatTensor::zeros(float_type, device.clone(), ())?;
            for x in iters.next().unwrap() {
                let x = x?.flatten()?;
                let distances =
                    compute_distances(&centroids.view().into(), &x.view())?.min_axis(Axis(1))?;
                let n = distances.len();
                distances
                    .into_shape(n)?
                    .sum_axis_with(Axis(0), &mut total_distance.view_mut())?;
            }
            let total_distance = total_distance.cast_to::<f32>()?;
            let total_distance = total_distance.as_raw_slice().read().await?[0];
            let distance_value = Uniform::new(0f32, total_distance).sample(&mut rand::thread_rng());
            let mut distance_counter = 0f32;
            for x in iters.next().unwrap() {
                let x = x?.flatten()?;
                let distances =
                    compute_distances(&centroids.view().into(), &x.view())?.min_axis(Axis(1))?;
                let n = distances.len();
                let distance_fut = distances
                    .view()
                    .into_shape(n)?
                    .sum_axis(Axis(0))?
                    .cast_into::<f32>()?
                    .read();
                let distances = distances.cast_into::<f32>()?.read().await?;
                let distance = distance_fut.await?.as_array().as_slice().unwrap()[0];
                let distances = distances.as_array();
                if distance_counter + distance > distance_value {
                    let mut index = 0;
                    let mut distance_counter = distance_counter;
                    for (u, d) in distances.iter().enumerate() {
                        if *d > 0f32 && distance_counter + *d > distance_value {
                            index = u;
                            break;
                        }
                        distance_counter += d;
                    }
                    let x = x.cast_to::<f32>()?;
                    let x_array_fut = x.read();
                    let array_centroids = centroids.read().await?;
                    let x_array = x_array_fut.await?;
                    let x_array = x_array.as_array();
                    let centroid = x_array.index_axis(Axis(0), index).into_shape([1, c])?;
                    let array_centroids =
                        ndarray::concatenate(Axis(0), &[array_centroids.as_array(), centroid])?;
                    centroids = CowTensor::from(array_centroids)
                        .into_device(device.clone())
                        .await?;
                    break;
                } else {
                    distance_counter += distance;
                }
            }
        }
        match self.centroids.float_type() {
            FloatType::BF16 => {
                self.centroids = centroids.cast_into::<bf16>()?.into_shared()?.into();
            }
            FloatType::F32 => {
                self.centroids = centroids.into_shared()?.into();
            }
        }
        Ok(())
    }

    fn compute_distances(&self, input: &FloatTensorView2) -> Result<FloatTensor2> {
        compute_distances(&self.centroids.view(), input)
    }
    fn update_centroids(
        &mut self,
        next_centroids: &FloatTensorView2,
        counts: &TensorView1<u32>,
    ) -> Result<()> {
        let float_type = self.centroids.float_type();
        let name = format!("kmeans_update_centroids_{}", float_type.as_str());
        let module = glsl_shaders::module(name)?;
        let (nclasses, ndim) = self.centroids.dim();
        let nclasses = nclasses as u32;
        let ndim = ndim as u32;
        let mut centroids = self.centroids.make_mut()?;
        let builder = module
            .compute_pass("main")?
            .float_slice(next_centroids.as_raw_slice())?
            .slice(counts.as_raw_slice())?
            .float_slice_mut(centroids.as_raw_slice_mut())?
            .push([nclasses, ndim])?;
        unsafe {
            builder.submit([nclasses, ndim, 1])?;
        }
        Ok(())
    }
    fn accumulate_next_centroids(
        next_centroids: &mut FloatTensorViewMut2,
        counts: &mut TensorViewMut1<u32>,
        input: &FloatTensorView2,
        classes: &TensorView1<u32>,
    ) -> Result<()> {
        let float_type = next_centroids.float_type();
        let name = format!("kmeans_accumulate_next_centroids_{}", float_type.as_str());
        let module = glsl_shaders::module(name)?;
        let (nclasses, ndim) = next_centroids.dim();
        let nclasses = nclasses as u32;
        let ndim = ndim as u32;
        let batch_size = classes.dim() as u32;
        let builder = module
            .compute_pass("main")?
            .float_slice_mut(next_centroids.as_raw_slice_mut())?
            .slice_mut(counts.as_raw_slice_mut())?
            .float_slice(input.to_slice()?.as_slice())?
            .slice(classes.as_raw_slice())?
            .push([batch_size, nclasses, ndim])?;
        unsafe { builder.submit([nclasses, ndim, 1]) }
    }
    fn train_impl<S, D, I>(
        &mut self,
        train_iter: I,
    ) -> Result<(usize, impl Future<Output = Result<f32>>)>
    where
        S: FloatData,
        D: Dimension,
        I: IntoIterator<Item = Result<FloatTensorBase<S, D>>>,
    {
        let device = self.centroids.device();
        let float_type = self.centroids.float_type();
        let mut loss = FloatTensor::zeros(float_type, device.clone(), ())?;
        let mut next_centroids =
            FloatTensor::zeros(float_type, device.clone(), self.centroids.raw_dim())?;
        let mut counts = Tensor::zeros(device, self.centroids.dim().0)?;
        let mut num_samples = 0;
        for x in train_iter {
            let x = x?.flatten()?;
            let distances = self.compute_distances(&x.view())?;
            let classes = distances.argmin_axis(Axis(1))?;
            Self::accumulate_next_centroids(
                &mut next_centroids.view_mut(),
                &mut counts.view_mut(),
                &x.view(),
                &classes.view(),
            )?;
            distances
                .index_select(Axis(1), &classes.view())?
                .sum_axis_with(Axis(0), &mut loss.view_mut())?;
            num_samples += x.dim().0;
        }
        self.update_centroids(&next_centroids.view(), &counts.view())?;
        let loss_fut = async move { read_loss(num_samples, loss.as_raw_slice()).await };
        Ok((num_samples, loss_fut))
    }
    fn test_impl<S, D, I>(&self, test_iter: I) -> Result<(usize, impl Future<Output = Result<f32>>)>
    where
        S: FloatData,
        D: Dimension,
        I: IntoIterator<Item = Result<FloatTensorBase<S, D>>>,
    {
        let device = self.centroids.device();
        let float_type = self.centroids.float_type();
        let mut loss = FloatTensor::zeros(float_type, device, ())?;
        let mut num_samples = 0;
        for x in test_iter {
            let x = x?.flatten()?;
            let distances = self.compute_distances(&x.view())?;
            let classes = distances.argmin_axis(Axis(1))?;
            distances
                .index_select(Axis(1), &classes.view())?
                .sum_axis_with(Axis(0), &mut loss.view_mut())?;
            num_samples += x.dim().0;
        }
        let loss_fut = async move { read_loss(num_samples, loss.as_raw_slice()).await };
        Ok((num_samples, loss_fut))
    }
}

impl<S: FloatData, D: Dimension> Test<FloatTensorBase<S, D>> for KMeans {
    fn test<I>(&self, test_iter: I) -> Result<Stats>
    where
        I: IntoIterator<Item = Result<FloatTensorBase<S, D>>>,
    {
        let (test_count, test_loss) = self.test_impl(test_iter)?;
        let test_loss = smol::block_on(test_loss)?;
        Ok(Stats {
            count: test_count,
            loss: Some(test_loss),
            ..Default::default()
        })
    }
}

impl<S: FloatData, D: Dimension> Infer<FloatTensorBase<S, D>> for KMeans {
    fn infer(&self, input: &FloatTensorBase<S, D>) -> Result<FloatTensorD> {
        let dim = input.raw_dim();
        self.compute_distances(&input.view().flatten()?)?
            .into_shape(dim.into_dyn())
    }
    fn predict<U: Uint>(&self, input: &FloatTensorBase<S, D>) -> Result<TensorD<U>> {
        let mut shape = input.shape();
        if shape.len() > 1 {
            shape = &shape[..shape.len() - 1];
        }
        self.compute_distances(&input.view().flatten()?)?
            .argmin_axis(Axis(1))?
            .into_shape(shape)
    }
}

impl From<KMeansTrainer> for KMeans {
    fn from(trainer: KMeansTrainer) -> Self {
        trainer.kmeans
    }
}

#[doc(hidden)]
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KMeansTrainer {
    kmeans: KMeans,
    summary: Summary,
}

impl From<KMeans> for KMeansTrainer {
    fn from(kmeans: KMeans) -> Self {
        Self {
            kmeans,
            summary: Summary::default(),
        }
    }
}

impl<S: FloatData, D: Dimension> Test<FloatTensorBase<S, D>> for KMeansTrainer {
    fn test<I>(&self, test_iter: I) -> Result<Stats>
    where
        I: IntoIterator<Item = Result<FloatTensorBase<S, D>>>,
    {
        self.kmeans.test(test_iter)
    }
}

impl<S: FloatData, D: Dimension> Infer<FloatTensorBase<S, D>> for KMeansTrainer {
    fn infer(&self, input: &FloatTensorBase<S, D>) -> Result<FloatTensorD> {
        self.kmeans.infer(input)
    }
}

impl Summarize for KMeansTrainer {
    fn summarize(&self) -> Summary {
        self.summary.clone()
    }
}

impl<S: FloatData, D: Dimension> Train<FloatTensorBase<S, D>> for KMeansTrainer {
    fn init<F, I>(&mut self, f: F) -> Result<()>
    where
        F: FnOnce(usize) -> I,
        I: IntoIterator,
        <I as IntoIterator>::Item: IntoIterator<Item = Result<FloatTensorBase<S, D>>>,
    {
        let kmeans = &mut self.kmeans;
        self.summary
            .run_init(|_| smol::block_on(kmeans.init_kplus_plus(f)))
    }
    fn train_test<I1, I2>(&mut self, train_iter: I1, test_iter: I2) -> Result<(Stats, Stats)>
    where
        I1: IntoIterator<Item = Result<FloatTensorBase<S, D>>>,
        I2: IntoIterator<Item = Result<FloatTensorBase<S, D>>>,
    {
        let kmeans = &mut self.kmeans;
        self.summary.run_epoch(move |_| {
            let (train_count, train_loss) = kmeans.train_impl(train_iter)?;
            let (test_count, test_loss) = kmeans.test_impl(test_iter)?;
            let (train_loss, test_loss) = smol::block_on(async move {
                Result::<_, anyhow::Error>::Ok((train_loss.await?, test_loss.await?))
            })?;
            let train_stats = Stats {
                count: train_count,
                loss: Some(train_loss),
                ..Default::default()
            };
            let test_stats = Stats {
                count: test_count,
                loss: Some(test_loss),
                ..Default::default()
            };
            Ok((train_stats, test_stats))
        })
    }
}

#[cfg(all(test, feature = "device_tests"))]
mod tests {
    use super::*;
    use crate::{device::Device, scalar::Float, tensor::TensorView, util::type_eq};
    use approx::assert_relative_eq;
    use half::bf16;
    use ndarray::{Array, Array2, ArrayView2};
    use std::convert::TryFrom;

    fn array_compute_distances(x: &ArrayView2<f32>, c: &ArrayView2<f32>) -> Array2<f32> {
        let mut y = Array::zeros([x.dim().0, c.dim().0]);
        for (x, mut y) in x.outer_iter().zip(y.outer_iter_mut()) {
            for (c, y) in c.outer_iter().zip(y.iter_mut()) {
                *y = x.iter().zip(c.iter()).map(|(x, c)| (x - c) * (x - c)).sum();
            }
        }
        y
    }

    async fn compute_distances<T: Float + From<u8>>(m: usize, k: usize, n: usize) -> Result<()> {
        use num_traits::ToPrimitive;
        let device = Device::new()?;
        let _s = device.acquire().await;
        let v1: Vec<T> = (1..=100)
            .into_iter()
            .cycle()
            .step_by(7)
            .map(|x| From::from(x as u8))
            .take(m * k)
            .collect();
        let v2: Vec<T> = (1..=100)
            .into_iter()
            .cycle()
            .skip(13)
            .step_by(3)
            .map(|x| From::from(x as u8))
            .take(n * k)
            .collect();
        let a1 = Array::from_shape_vec([m, k], v1)?;
        let a2 = Array::from_shape_vec([n, k], v2)?;
        let y_true = {
            let a1 = a1.map(|x| x.to_f32().unwrap());
            let a2 = a2.map(|x| x.to_f32().unwrap());
            array_compute_distances(&a1.view(), &a2.view())
        };
        let t1 = TensorView::try_from(a1.view())?
            .into_device(device.clone())
            .await?;
        let t1 = FloatTensor::from(t1);
        let t2 = TensorView::try_from(a2.view())?
            .into_device(device.clone())
            .await?
            .into_shared()?;
        let kmeans = KMeans::from_centroids(t2.into());
        let y = kmeans
            .compute_distances(&t1.view())?
            .cast_into::<f32>()?
            .read()
            .await?
            .into_array();
        let y = y.map(|&x| x.to_f32().unwrap());
        if type_eq::<T, f32>() {
            assert_relative_eq!(y, y_true, max_relative = 0.000_1);
        } else if type_eq::<T, bf16>() {
            assert_relative_eq!(y, y_true, epsilon = 0.01, max_relative = 0.01);
        }
        Ok(())
    }

    #[tokio::test]
    async fn compute_distances_bf16_m11_k5_n13() -> Result<()> {
        compute_distances::<bf16>(11, 5, 13).await
    }

    #[tokio::test]
    async fn compute_distances_f32_m11_k5_n13() -> Result<()> {
        compute_distances::<f32>(11, 5, 13).await
    }
}
