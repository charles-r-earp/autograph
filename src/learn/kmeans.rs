use super::{Stats, Summary, Train};
use crate::{
    float_tensor::{
        FloatArcTensor2, FloatData, FloatTensor, FloatTensor2, FloatTensorBase, FloatTensorView2,
        FloatTensorViewMut2,
    },
    glsl_shaders,
    result::Result,
    tensor::{Tensor, Tensor0, TensorView1, TensorViewMut1},
};
use anyhow::bail;
use ndarray::{Axis, Ix2};

#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct KMeans {
    centroids: FloatArcTensor2,
}

impl KMeans {
    /// Constructs a new [`KMeans`] from `centroids`.
    ///
    /// # Centroids
    /// - Should be standard layout.
    /// - Shape = [k, c], with k means of c dimensional points.
    pub fn from_centroids(centroids: FloatArcTensor2) -> Self {
        Self { centroids }
    }
    fn compute_distances(&self, input: &FloatTensorView2) -> Result<FloatTensor2> {
        let device = self.centroids.device();
        let float_type = self.centroids.float_type();
        let name = format!("kmean_dstance_{:?}", float_type);
        let module = glsl_shaders::module(name)?;
        let (batch_size, ndims) = input.dim();
        let (nclasses, _ndims) = self.centroids.dim();
        if ndims != _ndims {
            bail!("Input ndim {:?} != Centroid ndim {:?}!", ndims, _ndims);
        }
        let mut output = unsafe { FloatTensor::alloc(float_type, device, (batch_size, nclasses))? };
        let batch_size = batch_size as u32;
        let nclasses = nclasses as u32;
        let ndims = ndims as u32;
        let builder = module
            .compute_pass("main")?
            .float_slice(input.to_slice()?.as_slice())?
            .float_slice(input.to_slice()?.as_slice())?
            .float_slice_mut(output.as_raw_slice_mut())?
            .push([batch_size, nclasses, ndims])?;
        unsafe {
            builder.submit([batch_size, nclasses, 1])?;
        }
        Ok(output)
    }
    fn update_centroids(
        &mut self,
        next_centroids: &FloatTensorView2,
        counts: &TensorView1<u32>,
    ) -> Result<()> {
        let float_type = self.centroids.float_type();
        let name = format!("kmeans_update_centroids_{:?}", float_type);
        let module = glsl_shaders::module(name)?;
        let (nclasses, ndim) = self.centroids.dim();
        let nclasses = nclasses as u32;
        let ndim = ndim as u32;
        let builder = module
            .compute_pass("main")?
            .float_slice(next_centroids.as_raw_slice())?
            .slice(counts.as_raw_slice())?
            .float_slice_mut(self.centroids.make_slice_mut()?)?
            .push([nclasses, ndim])?;
        unsafe { builder.submit([nclasses, ndim, 1]) }
    }
    fn accumulate_next_centroids(
        next_centroids: &mut FloatTensorViewMut2,
        counts: &mut TensorViewMut1<u32>,
        input: &FloatTensorView2,
        classes: &TensorView1<u32>,
    ) -> Result<()> {
        let float_type = next_centroids.float_type();
        let name = format!("kmeans_accumulate_next_centroids_{:?}", float_type);
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
    fn train<S, I>(&mut self, train_iter: I) -> Result<Tensor0<f32>>
    where
        S: FloatData,
        I: IntoIterator<Item = Result<FloatTensorBase<S, Ix2>>>,
    {
        let device = self.centroids.device();
        let float_type = self.centroids.float_type();
        #[allow(unused_mut)]
        let mut loss = FloatTensor::zeros(float_type, device.clone(), ())?;
        let mut next_centroids =
            FloatTensor::zeros(float_type, device.clone(), self.centroids.raw_dim())?;
        let mut counts = Tensor::zeros(device, self.centroids.dim().0)?;
        let mut num_samples = 0;
        for x in train_iter {
            let x = x?;
            let distances = self.compute_distances(&x.view())?;
            let classes = distances.argmin(Axis(1))?;
            Self::accumulate_next_centroids(
                &mut next_centroids.view_mut(),
                &mut counts.view_mut(),
                &x.view(),
                &classes.view(),
            )?;
            /*distances
            .index_select(Axis(1), &classes.view())?
            .sum_with(Axis(0), &mut loss.view_mut())?;*/
            num_samples += x.dim().0;
        }
        self.update_centroids(&next_centroids.view(), &counts.view())?;
        let alpha = if num_samples > 0 {
            1. / num_samples as f32
        } else {
            0.
        };
        loss.scale_into(alpha)
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct KMeansTrainer {
    kmeans: KMeans,
    summary: Summary,
}

#[doc(hidden)]
impl<S: FloatData> Train<FloatTensorBase<S, Ix2>> for KMeansTrainer {
    #[allow(unused)]
    fn train_test<I1, I2>(&mut self, train_iter: I1, test_iter: I2) -> Result<(Stats, Stats)>
    where
        I1: IntoIterator<Item = Result<FloatTensorBase<S, Ix2>>>,
        I2: IntoIterator<Item = Result<FloatTensorBase<S, Ix2>>>,
    {
        self.kmeans.train(train_iter)?;
        todo!()
    }
}
