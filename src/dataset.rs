use crate::{
    Result,
    backend::Device,
    tensor::{Tensor, Dimension, Scalar, Axis},
    ndarray::{Array, Array1, Array2, Data as ArrayData, ArrayBase, RawArrayView},
};
use std::{
    future::Future,
    ops::{Range, RangeBounds, Bound},
    str::FromStr,
    vec::IntoIter as VecIntoIter,
};
use http::StatusCode;
use smol::future::{ready, Ready};
use rand::prelude::SliceRandom;
use anyhow::{anyhow, bail, ensure};
use downloader::{Download, Downloader};
use futures_util::future::{TryJoin, try_join};

pub trait Dataset {
    type Item;
    type Future: Future<Output=Result<Self::Item>>;
    fn sample_count(&self) -> usize;
    fn sample(&self, device: &Device, index: usize, batch_size: usize) -> Option<Self::Future>;
    fn slice(&self, bounds: impl RangeBounds<usize>) -> Slice<'_, Self> {
        let start = match bounds.start_bound() {
            Bound::Included(start) => *start,
            Bound::Excluded(start) => *start + 1,
            Bound::Unbounded => 0,
        };
        let end = match bounds.end_bound() {
            Bound::Included(end) => *end + 1,
            Bound::Excluded(end) => *end,
            Bound::Unbounded => self.sample_count(),
        };
        Slice {
            dataset: self,
            range: start..end
        }
    }
    fn batches(&self, device: &Device, batch_size: usize, shuffle: bool) -> Batches<'_, Self> {
        let sample_count = self.sample_count();
        let mut indices: Vec<usize> = (0..sample_count)
            .into_iter()
            .step_by(batch_size)
            .collect();
        if shuffle {
            indices.shuffle(&mut rand::thread_rng());
        }
        Batches {
            dataset: self,
            device: device.clone(),
            sample_count,
            indices: indices.into_iter(),
            batch_size,
        }
    }
}

pub fn train_test_split<A>(dataset: &A, test_ratio: f32) -> (Slice<'_, A>, Slice<'_, A>)
    where A: Dataset {
    let sample_count = dataset.sample_count();
    let test_offset = sample_count - ((test_ratio * sample_count as f32).round() as usize).min(sample_count);
    let train = dataset.slice(..test_offset);
    let test = dataset.slice(test_offset..);
    (train, test)
}

pub struct Slice<'a, A: ?Sized> {
    dataset: &'a A,
    range: Range<usize>,
}

impl<A: Dataset> Dataset for Slice<'_, A> {
    type Item = A::Item;
    type Future = A::Future;
    fn sample_count(&self) -> usize {
        self.range.len()
    }
    fn sample(&self, device: &Device, index: usize, batch_size: usize) -> Option<Self::Future> {
        if index < self.range.len() && index + batch_size <= self.range.len() {
            self.dataset.sample(device, self.range.start + index, batch_size)
        } else {
            None
        }
    }
}

pub struct Batches<'a, A: ?Sized> {
    dataset: &'a A,
    device: Device,
    sample_count: usize,
    indices: VecIntoIter<usize>,
    batch_size: usize,
}

impl<A: Dataset> Iterator for Batches<'_, A> {
    type Item = A::Future;
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.indices.next()?;
        let batch_size = (self.sample_count - index).min(self.batch_size);
        self.dataset.sample(&self.device, index, batch_size)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let sample_count = self.sample_count;
        (sample_count, Some(sample_count))
    }
}

impl<A: Dataset> ExactSizeIterator for Batches<'_, A> {}


impl<T: Scalar, S: ArrayData<Elem=T>, D: Dimension> Dataset for ArrayBase<S, D> {
    type Item = Tensor<T, D>;
    type Future = Ready<Result<Self::Item>>;
    fn sample_count(&self) -> usize {
        self.shape().first().copied().unwrap_or(0)
    }
    fn sample(&self, device: &Device, index: usize, batch_size: usize) -> Option<Self::Future> {
        let sample_count = self.sample_count();
        if index < sample_count && index + batch_size <= sample_count {
            if let Some(slice) = self.as_slice() {
                let mut array = unsafe {
                    RawArrayView::from_shape_ptr(self.raw_dim(), slice.as_ptr())
                        .deref_into_view()
                };
                array.slice_axis_inplace(Axis(0), (index..index + batch_size).into());
                Some(ready(Tensor::from_array(device, array)))
            } else {
                Some(ready(Err(anyhow!("Array must be standard layout!"))))
            }
        } else {
            None
        }
    }
}

impl<A: Dataset, B: Dataset> Dataset for (A, B) {
    type Item = (A::Item, B::Item);
    type Future = TryJoin<A::Future, B::Future>;
    fn sample_count(&self) -> usize {
        self.0.sample_count()
            .min(self.1.sample_count())
    }
    fn sample(&self, device: &Device, index: usize, batch_size: usize) -> Option<Self::Future> {
        self.0.sample(device, index, batch_size)
            .zip(self.1.sample(device, index, batch_size))
            .map(|(a, b)| try_join(a, b))
    }
}

pub fn iris() -> Result<(Array2<f32>, Array1<u32>)> {
    let dir_path = dirs::download_dir().unwrap_or_else(std::env::temp_dir);
    let data_path = dir_path.join("iris").with_extension("data");
    if !data_path.exists() {
        let url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data";
        let download = Download::new(url).file_name(&data_path);
        let mut downloader = Downloader::builder().download_folder(&dir_path).build()?;
        let summaries = downloader.download(&[download])?;
        let summary = summaries
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("Download of Iris dataset failed!"))??;
        let (_, status) = summary
            .status
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("Download of Iris dataset failed!"))?;
        let status = StatusCode::from_u16(status)?;
        ensure!(
            status.is_success(),
            "Iris download failed with status code: {}!",
            status
        );
    }
    let csv = smol::block_on(smol::fs::read_to_string(data_path))?;
    let mut data = vec![-1.; 150 * 4];
    let mut labels = vec![150; 150];
    for (line, (x, t)) in csv
        .lines()
        .zip(data.chunks_exact_mut(4).zip(labels.iter_mut()))
    {
        let mut items = line.split(',');
        for (x, s) in x.iter_mut().zip(&mut items) {
            *x = f32::from_str(s)?;
        }
        *t = match items.next() {
            Some("Iris-setosa") => 0,
            Some("Iris-versicolor") => 1,
            Some("Iris-virginica") => 2,
            Some(s) => bail!("Unable to parse iris class {:?}", s),
            None => bail!("Unable to parse iris.data: expected class, found None!"),
        };
    }
    ensure!(data.iter().all(|x| *x > 0.));
    ensure!(labels.iter().all(|x| *x <= 2));
    let data = Array::from_shape_vec([150, 4], data)?;
    let labels = Array::from_shape_vec(150, labels)?;
    Ok((data, labels))
}


#[cfg(test)]
mod tests {
    use super::*;

    #[ignore]
    #[test]
    fn iris() -> Result<()> {
        super::iris()?;
        Ok(())
    }

    #[ignore]
    #[test]
    fn iris_train_test() -> Result<()> {
        smol::block_on(async {
            let xy_dataset = super::iris()?;
            let (train_set, test_set) = train_test_split(&xy_dataset, 0.2);
            if let Some(device) = Device::new_gpu(0) {
                let device = device?;
                for xy_future in train_set.batches(&device, 11, true) {
                    let (x, y) = xy_future.await?;
                    assert_eq!(x.dim().0, y.dim());
                }
                for xy_future in test_set.batches(&device, 33, false) {
                    let (x, y) = xy_future.await?;
                    assert_eq!(x.dim().0, y.dim());
                }
            }
            Ok(())
        })
    }

    #[ignore]
    #[test]
    fn iris_unsupervised() -> Result<()> {
        smol::block_on(async {
            let (train_set, _) = super::iris()?;
            if let Some(device) = Device::new_gpu(0) {
                let device = device?;
                for x_future in train_set.batches(&device, 43, false) {
                    let x = x_future.await?;
                    assert!(x.dim().0 <= 43);
                }
            }
            Ok(())
        })
    }
}
