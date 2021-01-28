use crate::Result;
use anyhow::{anyhow, bail, ensure};
use downloader::{Download, Downloader};
use http::StatusCode;
use ndarray::{
    Array, Array1, Array2, ArrayBase, ArrayView, ArrayView1, ArrayView2, Axis, Dimension,
    RawArrayView,
};
use rand::seq::SliceRandom;
use smol::future::{ready, Ready};
use std::{
    future::Future,
    ops::{Bound, Range, RangeBounds},
    str::FromStr,
};

pub trait Dataset {
    type Item;
    type Future: Future<Output = Result<Self::Item>>;
    fn sample_count(&self) -> usize;
    fn sample(&self, range: Range<usize>) -> Option<Self::Future>;
    fn slice(self, bounds: impl RangeBounds<usize>) -> Option<Slice<Self>>
    where
        Self: Sized,
    {
        let len = self.sample_count();
        let start = match bounds.start_bound() {
            Bound::Included(start) => *start,
            Bound::Excluded(start) => *start + 1,
            Bound::Unbounded => 0,
        };
        let end = match bounds.end_bound() {
            Bound::Included(end) => *end + 1,
            Bound::Excluded(end) => *end,
            Bound::Unbounded => len,
        };
        if start < end && end <= len {
            Some(Slice {
                dataset: self,
                range: start..end,
            })
        } else {
            None
        }
    }
    fn batches(&self, batch_size: usize, shuffle: bool) -> Batches<'_, Self> {
        let mut indices: Vec<usize> = (0..self.sample_count()).step_by(batch_size).collect();
        if shuffle {
            indices.shuffle(&mut rand::thread_rng());
        }
        Batches {
            dataset: self,
            indices: indices.into_iter(),
            batch_size,
        }
    }
}

pub struct Slice<T> {
    dataset: T,
    range: Range<usize>,
}

impl<T: Dataset> Dataset for Slice<T> {
    type Item = T::Item;
    type Future = T::Future;
    fn sample_count(&self) -> usize {
        self.range.len()
    }
    fn sample(&self, range: Range<usize>) -> Option<Self::Future> {
        let start = self.range.start + range.start;
        let end = (self.range.start + range.end).min(self.range.end);
        self.dataset.sample(start..end)
    }
}

pub fn train_test_split<'a, T>(dataset: &'a T, test_ratio: f32) -> (Slice<&'a T>, Slice<&'a T>)
where
    &'a T: Dataset,
{
    let len = dataset.sample_count();
    let test_offset = len - ((test_ratio * len as f32).round() as usize).min(len);
    let train = dataset.slice(..test_offset).unwrap();
    let test = dataset.slice(test_offset..).unwrap();
    (train, test)
}

pub struct Batches<'a, T: Dataset + ?Sized> {
    dataset: &'a T,
    indices: std::vec::IntoIter<usize>,
    batch_size: usize,
}

impl<T: Dataset> Iterator for Batches<'_, T> {
    type Item = T::Future;
    fn next(&mut self) -> Option<Self::Item> {
        let start = self.indices.next()?;
        let end = (start + self.batch_size).min(self.dataset.sample_count());
        Some(self.dataset.sample(start..end).unwrap())
    }
}

#[derive(Clone)]
pub struct SimpleDataset<X, Y> {
    input: X,
    output: Y,
}

impl<X> SimpleDataset<X, ()> {
    pub fn from_input(input: X) -> Self {
        Self { input, output: () }
    }
}

impl<X, Y> SimpleDataset<X, Y> {
    pub fn from_input_output(input: X, output: Y) -> Self {
        Self { input, output }
    }
    pub fn unsupervised(self) -> SimpleDataset<X, ()> {
        SimpleDataset::from_input(self.input)
    }
}

impl<'a, T: Copy + 'a, S: ndarray::Data<Elem = T> + 'a, D: Dimension> Dataset
    for &'a SimpleDataset<ArrayBase<S, D>, ()>
{
    type Item = ArrayView<'a, T, D>;
    type Future = Ready<Result<Self::Item>>;
    fn sample_count(&self) -> usize {
        self.input.shape().first().map_or(1, |x| *x)
    }
    fn sample(&self, range: Range<usize>) -> Option<Self::Future> {
        let len = self.sample_count();
        if range.start < range.end && range.end <= len {
            if let Some(slice) = self.input.as_slice() {
                let mut input = unsafe {
                    RawArrayView::from_shape_ptr(self.input.raw_dim(), slice.as_ptr())
                        .deref_into_view()
                };
                input.slice_axis_inplace(Axis(0), range.into());
                Some(ready(Ok(input)))
            } else {
                Some(ready(Err(anyhow!(
                    "SimpleDataset: Array must be standard layout!"
                ))))
            }
        } else {
            None
        }
    }
}

impl<
        'a,
        T1: 'a,
        S1: ndarray::Data<Elem = T1> + 'a,
        D1: Dimension,
        T2: 'a,
        S2: ndarray::Data<Elem = T2> + 'a,
        D2: Dimension,
    > Dataset for &'a SimpleDataset<ArrayBase<S1, D1>, ArrayBase<S2, D2>>
{
    type Item = (ArrayView<'a, T1, D1>, ArrayView<'a, T2, D2>);
    type Future = Ready<Result<Self::Item>>;
    fn sample_count(&self) -> usize {
        debug_assert_eq!(
            self.input.shape().first().map_or(1, |x| *x),
            self.output.shape().first().map_or(1, |x| *x)
        );
        self.input.shape().first().map_or(1, |x| *x)
    }
    fn sample(&self, range: Range<usize>) -> Option<Self::Future> {
        let len = self.sample_count();
        if range.start < range.end && range.end <= len {
            if let Some((input_slice, output_slice)) =
                self.input.as_slice().zip(self.output.as_slice())
            {
                let mut input = unsafe {
                    RawArrayView::from_shape_ptr(self.input.raw_dim(), input_slice.as_ptr())
                        .deref_into_view()
                };
                input.slice_axis_inplace(Axis(0), range.clone().into());
                let mut output = unsafe {
                    RawArrayView::from_shape_ptr(self.output.raw_dim(), output_slice.as_ptr())
                        .deref_into_view()
                };
                output.slice_axis_inplace(Axis(0), range.into());
                Some(ready(Ok((input, output))))
            } else {
                Some(ready(Err(anyhow!(
                    "SimpleDataset: Array must be standard layout!"
                ))))
            }
        } else {
            None
        }
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct Iris(SimpleDataset<Array2<f32>, Array1<u8>>);

impl Iris {
    pub fn new() -> Result<Self> {
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
        Ok(Self(SimpleDataset::from_input_output(data, labels)))
    }
    pub fn unsupervised(self) -> IrisUnsupervised {
        IrisUnsupervised(self.0.unsupervised())
    }
}

impl<'a> Dataset for &'a Iris {
    type Item = (ArrayView2<'a, f32>, ArrayView1<'a, u8>);
    type Future = Ready<Result<Self::Item>>;
    fn sample_count(&self) -> usize {
        150
    }
    fn sample(&self, range: Range<usize>) -> Option<Self::Future> {
        (&self.0).sample(range)
    }
}

pub struct IrisUnsupervised(SimpleDataset<Array2<f32>, ()>);

impl<'a> Dataset for &'a IrisUnsupervised {
    type Item = ArrayView2<'a, f32>;
    type Future = Ready<Result<Self::Item>>;
    fn sample_count(&self) -> usize {
        150
    }
    fn sample(&self, range: Range<usize>) -> Option<Self::Future> {
        (&self.0).sample(range)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ignore]
    #[test]
    fn iris_new() -> Result<()> {
        Iris::new()?;
        Ok(())
    }

    #[ignore]
    #[test]
    fn iris_train_test() -> Result<()> {
        let iris = Iris::new()?;
        let (train_data, test_data) = train_test_split(&iris, 0.2);
        smol::block_on(async {
            for xy_future in train_data.batches(11, true) {
                let (x, y) = xy_future.await?;
                assert_eq!(x.dim().0, y.dim());
            }
            for xy_future in test_data.batches(33, false) {
                let (x, y) = xy_future.await?;
                assert_eq!(x.dim().0, y.dim());
            }
            Ok(())
        })
    }

    #[ignore]
    #[test]
    fn iris_unsupervised() -> Result<()> {
        let iris = Iris::new()?.unsupervised();
        smol::block_on(async {
            for x_future in (&iris).batches(43, false) {
                let x = x_future.await?;
                assert!(x.dim().0 <= 43);
            }
            Ok(())
        })
    }
}
