use crate::{
    backend::Device,
    ndarray::{Array, Array1, Array2, Array4, ArrayBase, Data as ArrayData, RawArrayView},
    tensor::{Axis, Dimension, Scalar, Tensor},
    Result,
};
use anyhow::{anyhow, bail, ensure};
use byteorder::{BigEndian, ReadBytesExt};
use downloader::{progress::Reporter, Download, Downloader};
use flate2::read::GzDecoder;
use futures_util::future::{try_join, TryJoin};
use http::StatusCode;
use indicatif::{MultiProgress, ProgressBar};
use rand::prelude::SliceRandom;
use smol::future::{ready, Ready};
use std::{
    fs::{self, File},
    future::Future,
    ops::{Bound, Range, RangeBounds},
    str::FromStr,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    vec::IntoIter as VecIntoIter,
};

pub trait Dataset {
    type Item;
    type Future: Future<Output = Result<Self::Item>>;
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
            range: start..end,
        }
    }
    fn batches(&self, device: &Device, batch_size: usize, shuffle: bool) -> Batches<'_, Self> {
        let sample_count = self.sample_count();
        let mut indices: Vec<usize> = (0..sample_count).into_iter().step_by(batch_size).collect();
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
where
    A: Dataset,
{
    let sample_count = dataset.sample_count();
    let test_offset =
        sample_count - ((test_ratio * sample_count as f32).round() as usize).min(sample_count);
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
            self.dataset
                .sample(device, self.range.start + index, batch_size)
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

impl<T: Scalar, S: ArrayData<Elem = T>, D: Dimension> Dataset for ArrayBase<S, D> {
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
                    RawArrayView::from_shape_ptr(self.raw_dim(), slice.as_ptr()).deref_into_view()
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
        self.0.sample_count().min(self.1.sample_count())
    }
    fn sample(&self, device: &Device, index: usize, batch_size: usize) -> Option<Self::Future> {
        self.0
            .sample(device, index, batch_size)
            .zip(self.1.sample(device, index, batch_size))
            .map(|(a, b)| try_join(a, b))
    }
}

struct ProgressBarWrapper {
    bar: ProgressBar,
    max_progress: AtomicUsize,
}

impl Reporter for ProgressBarWrapper {
    fn setup(&self, max_progess: Option<u64>, message: &str) {
        if let Some(max_progess) = max_progess {
            self.max_progress
                .store(max_progess as usize, Ordering::SeqCst);
        }
    }

    fn progress(&self, current: u64) {
        let max_progess = self.max_progress.load(Ordering::SeqCst) as u64;
        if max_progess > 0 {
            let pos = current * self.bar.length() / max_progess;
            self.bar.set_position(pos);
        }
    }

    fn set_message(&self, _: &str) {}

    fn done(&self) {
        self.bar.finish()
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

pub fn mnist() -> Result<(Array4<u8>, Array1<u8>)> {
    // TODO: Need to show progress to user because downloading will take awhile
    // though it does put it in downloads so you can just look at that, or download manually.
    use std::io::Read;
    let dir_path = dirs::download_dir().unwrap_or_else(std::env::temp_dir);
    let mnist_path = dir_path.join("mnist");
    if !mnist_path.exists() {
        std::fs::create_dir(&mnist_path)?;
    }
    let names = &[
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ];
    {
        // download
        let names: Vec<_> = names
            .iter()
            .filter(|name| !mnist_path.join(name).with_extension("gz").exists())
            .collect();
        if !names.is_empty() {
            let downloads: Vec<_> = names
                .iter()
                .map(|name| {
                    let path = mnist_path.join(name).with_extension("gz");
                    let url = format!("http://yann.lecun.com/exdb/mnist/{}.gz", name);
                    Download::new(&url).file_name(&path)
                })
                .collect();
            let mut downloader = Downloader::builder()
                .download_folder(&mnist_path)
                .retries(10)
                .build()?;
            let summaries = downloader.download(&downloads)?;
            for summary in summaries {
                match summary {
                    Ok(_) => (),
                    Err(downloader::Error::Download(summary)) => {
                        if let Some((_, status)) = summary.status.last() {
                            StatusCode::from_u16(*status)?;
                        }
                    }
                    _ => {
                        summary?;
                    }
                }
            }
        }
    }
    let mut images = Vec::new();
    let mut labels = Vec::new();
    {
        // unzip
        for &name in names.iter() {
            let (train, image) = match name {
                "train-images-idx3-ubyte" => (true, true),
                "train-labels-idx1-ubyte" => (true, false),
                "t10k-images-idx3-ubyte" => (false, true),
                "t10k-labels-idx1-ubyte" => (false, false),
                _ => unreachable!(),
            };
            let magic = if image { 2_051 } else { 2_049 };
            let n = if train { 60_000 } else { 10_000 };
            let data_path = mnist_path.join(name).with_extension("data");
            if let Some(data) = std::fs::read(&data_path).ok() {
                if image {
                    ensure!(data.len() == n * 28 * 28);
                    images.extend(data);
                } else {
                    ensure!(data.len() == n);
                    labels.extend(data);
                }
            } else {
                let gz_path = mnist_path.join(name).with_extension("gz");
                let mut data = Vec::new();
                let mut decoder = GzDecoder::new(File::open(&gz_path)?);
                ensure!(decoder.read_i32::<BigEndian>().unwrap() == magic);
                ensure!(decoder.read_i32::<BigEndian>().unwrap() == n as i32);
                if image {
                    ensure!(decoder.read_i32::<BigEndian>().unwrap() == 28);
                    ensure!(decoder.read_i32::<BigEndian>().unwrap() == 28);
                }
                decoder.read_to_end(&mut data)?;
                if image {
                    ensure!(data.len() == n * 28 * 28);
                    images.extend(&data);
                } else {
                    ensure!(data.len() == n);
                    labels.extend(&data);
                }
                std::fs::write(data_path, data)?;
            }
        }
    }
    let images = Array::from_shape_vec([70_000, 1, 28, 28], images)?;
    let labels = Array::from_shape_vec([70_000], labels)?;
    Ok((images, labels))
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

    #[ignore]
    #[test]
    fn mnist() -> Result<()> {
        super::mnist()?;
        Ok(())
    }
}
