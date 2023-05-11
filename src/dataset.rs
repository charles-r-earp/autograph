use crate::{
    buffer::Data,
    device::Device,
    scalar::Scalar,
    tensor::{CowTensor, Tensor, TensorBase, TensorView},
};
use anyhow::Result;
use ndarray::{Array, ArrayBase, ArrayView, Axis, Data as ArrayData, Dimension, IxDyn, RemoveAxis};
use rand::{seq::index::IndexVec, thread_rng};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        IntoParallelRefMutIterator, ParallelIterator,
    },
    prelude::ParallelSliceMut,
};
use std::{
    borrow::Cow,
    iter::Peekable,
    marker::PhantomData,
    ops::{Range, RangeBounds},
};

/// The Iris dataset.
#[cfg(feature = "iris")]
pub mod iris;

/// The MNIST dataset.
#[cfg(feature = "mnist")]
pub mod mnist;

pub trait Dataset {
    type Item;
    fn sample_count(&self) -> usize;
    fn sample(&self, index: usize) -> Option<Self::Item>;
    fn zip<B>(self, b: B) -> Zip<(Self, B)>
    where
        Self: Sized,
        B: Dataset,
    {
        Zip::from(self).and(b)
    }
    fn split_at(self, index: usize) -> (Slice<Self>, Slice<Self>)
    where
        Self: Clone,
    {
        Slice::from_split_at(self, index)
    }
    fn map<B, F>(self, f: F) -> Map<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Item) -> B,
    {
        Map { dataset: self, f }
    }
    fn tensor_batches<T>(self, size: usize) -> TensorBatches<Self, T>
    where
        Self: Sized,
    {
        TensorBatches::new(self, size)
    }
}

impl<'a, T, D: RemoveAxis> Dataset for ArrayView<'a, T, D> {
    type Item = ArrayView<'a, T, D::Smaller>;
    fn sample_count(&self) -> usize {
        self.shape().first().copied().unwrap_or_default()
    }
    fn sample(&self, index: usize) -> Option<Self::Item> {
        if index > self.sample_count() {
            return None;
        }
        Some(self.clone().index_axis_move(Axis(0), index))
    }
}

pub struct Zip<A> {
    datasets: A,
}

impl<A: Dataset> Zip<(A,)> {
    pub fn from(a: A) -> Self {
        Self { datasets: (a,) }
    }
    pub fn and<B: Dataset>(self, b: B) -> Zip<(A, B)> {
        let (a,) = self.datasets;
        Zip { datasets: (a, b) }
    }
}

impl<A: Dataset, B: Dataset> Dataset for Zip<(A, B)> {
    type Item = (A::Item, B::Item);
    fn sample_count(&self) -> usize {
        let (a, b) = &self.datasets;
        a.sample_count().min(b.sample_count())
    }
    fn sample(&self, index: usize) -> Option<Self::Item> {
        let (a, b) = &self.datasets;
        a.sample(index).zip(b.sample(index))
    }
}

#[derive(Clone)]
pub struct Slice<A> {
    dataset: A,
    offset: usize,
    len: usize,
}

impl<A: Dataset + Clone> Slice<A> {
    fn from_split_at(dataset: A, index: usize) -> (Self, Self) {
        todo!()
    }
}

impl<A: Dataset> Dataset for Slice<A> {
    type Item = A::Item;
    fn sample_count(&self) -> usize {
        self.len
    }
    fn sample(&self, index: usize) -> Option<Self::Item> {
        if index >= self.len {
            return None;
        }
        self.dataset.sample(self.offset + index)
    }
}

pub struct Map<A, F> {
    dataset: A,
    f: F,
}

struct IndexIter {
    range: Range<usize>,
    index_vec: Option<IndexVec>,
}

impl IndexIter {
    fn new(len: usize, shuffle: bool) -> Self {
        let range = 0..len;
        let index_vec = if len > 0 && shuffle {
            Some(rand::seq::index::sample(&mut thread_rng(), len, len))
        } else {
            None
        };
        Self { range, index_vec }
    }
}

impl Iterator for IndexIter {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.range.next()?;
        if let Some(index_vec) = self.index_vec.as_ref() {
            Some(index_vec.index(index))
        } else {
            Some(index)
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl ExactSizeIterator for IndexIter {
    fn len(&self) -> usize {
        self.range.len()
    }
}

#[derive(Clone)]
pub struct TensorBatches<A, T> {
    dataset: A,
    size: usize,
    device: Device,
    shuffle: bool,
    _m: PhantomData<T>,
}

impl<A, T> TensorBatches<A, T> {
    fn new(dataset: A, size: usize) -> Self {
        Self {
            dataset,
            size,
            device: Device::host(),
            shuffle: false,
            _m: PhantomData::default(),
        }
    }
    pub fn device(self, device: Device) -> Self {
        Self { device, ..self }
    }
    pub fn shuffle(self, shuffle: bool) -> Self {
        Self { shuffle, ..self }
    }
}

impl<T: Scalar, S: ArrayData<Elem = T>, D: Dimension, A: Dataset<Item = ArrayBase<S, D>>>
    IntoIterator for TensorBatches<A, (T,)>
{
    type Item = Result<Tensor<T, D::Larger>>;
    type IntoIter = TensorBatchIter<A, (T,)>;
    fn into_iter(self) -> Self::IntoIter {
        let TensorBatches {
            dataset,
            size,
            device,
            shuffle,
            _m,
        } = self;
        let len = if size > 0 { dataset.sample_count() } else { 0 };
        let index_iter = IndexIter::new(len, self.shuffle);
        TensorBatchIter {
            dataset,
            size,
            device,
            index_iter,
            indices: Vec::new(),
            dims: Vec::new(),
            _m,
        }
    }
}

impl<
        T1: Scalar,
        S1: ArrayData<Elem = T1>,
        D1: Dimension,
        T2: Scalar,
        S2: ArrayData<Elem = T2>,
        D2: Dimension,
        A: Dataset<Item = (ArrayBase<S1, D1>, ArrayBase<S2, D2>)> + Send + Sync,
    > IntoIterator for TensorBatches<A, (T1, T2)>
{
    type Item = Result<(Tensor<T1, D1::Larger>, Tensor<T2, D2::Larger>)>;
    type IntoIter = TensorBatchIter<A, (T1, T2)>;
    fn into_iter(self) -> Self::IntoIter {
        let TensorBatches {
            dataset,
            size,
            device,
            shuffle,
            _m,
        } = self;
        let len = if size > 0 { dataset.sample_count() } else { 0 };
        let index_iter = IndexIter::new(len, self.shuffle);
        TensorBatchIter {
            dataset,
            size,
            device,
            index_iter,
            indices: Vec::new(),
            dims: Vec::new(),
            _m,
        }
    }
}

pub struct TensorBatchIter<A, T> {
    dataset: A,
    size: usize,
    device: Device,
    index_iter: IndexIter,
    indices: Vec<usize>,
    dims: Vec<IxDyn>,
    _m: PhantomData<T>,
}

impl<A, T> ExactSizeIterator for TensorBatchIter<A, T>
where
    Self: Iterator,
{
    fn len(&self) -> usize {
        if self.size == 0 {
            return 0;
        }
        let len = self.index_iter.len();
        len / self.size + (len % self.size != 0) as usize
    }
}

impl<T: Scalar, S: ndarray::Data<Elem = T>, D: Dimension, A: Dataset<Item = ArrayBase<S, D>>>
    Iterator for TensorBatchIter<A, (T,)>
where
    S::Elem: Scalar,
{
    type Item = Result<Tensor<T, D::Larger>>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index_iter.len() == 0 {
            return None;
        }
        let batch_size = self.index_iter.len().min(self.size);
        let mut first = None;
        if self.dims.is_empty() {
            let index = self.index_iter.next().unwrap();
            let sample = self.dataset.sample(index).unwrap();
            let mut dim = IxDyn::zeros(sample.ndim() + 1);
            dim.slice_mut()[1..].copy_from_slice(sample.shape());
            self.dims.push(dim);
            first.replace(sample);
        }
        let mut dim = D::Larger::from_dimension(&self.dims[0]).unwrap();
        dim[0] = batch_size;
        let mut output = Array::zeros(dim);
        first
            .into_iter()
            .chain(
                self.index_iter
                    .by_ref()
                    .map(|index| self.dataset.sample(index).unwrap()),
            )
            .take(self.size)
            .zip(output.outer_iter_mut())
            .for_each(|(sample, mut output)| output.assign(&sample));
        Some(Ok(output.into()))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<
        T1: Scalar,
        S1: ArrayData<Elem = T1>,
        D1: Dimension,
        T2: Scalar,
        S2: ArrayData<Elem = T2>,
        D2: Dimension,
        A: Dataset<Item = (ArrayBase<S1, D1>, ArrayBase<S2, D2>)> + Send + Sync,
    > Iterator for TensorBatchIter<A, (T1, T2)>
{
    type Item = Result<(Tensor<S1::Elem, D1::Larger>, Tensor<S2::Elem, D2::Larger>)>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index_iter.len() == 0 {
            return None;
        }
        let batch_size = self.index_iter.len().min(self.size);
        let mut first = None;
        if self.dims.is_empty() {
            let index = self.index_iter.next().unwrap();
            let (sample1, sample2) = self.dataset.sample(index).unwrap();
            for shape in [sample1.shape(), sample2.shape()] {
                let mut dim = IxDyn::zeros(shape.len() + 1);
                dim.slice_mut()[1..].copy_from_slice(shape);
                self.dims.push(dim);
            }
            first.replace((sample1, sample2));
        }
        let mut dim1 = D1::Larger::from_dimension(&self.dims[0]).unwrap();
        dim1[0] = batch_size;
        let mut dim2 = D2::Larger::from_dimension(&self.dims[1]).unwrap();
        dim2[0] = batch_size;
        let mut output1 = vec![T1::default(); dim1.size()];
        let mut output2 = vec![T2::default(); dim2.size()];
        let chunk_size1 = dim1.slice()[1..].iter().product();
        let chunk_isze2 = dim2.slice()[1..].iter().product();
        if let Some((sample1, sample2)) = first.as_ref() {
            output1[..chunk_size1].copy_from_slice(sample1.as_slice().unwrap());
            output2[..chunk_isze2].copy_from_slice(sample2.as_slice().unwrap());
        }
        self.indices.clear();
        self.indices.extend(
            self.index_iter
                .by_ref()
                .take(batch_size - first.is_some() as usize),
        );
        output1
            .par_chunks_exact_mut(chunk_size1)
            .zip(output2.par_chunks_exact_mut(chunk_isze2))
            .zip(self.indices.par_iter())
            .for_each(|((mut output1, mut output2), index)| {
                let (sample1, sample2) = self.dataset.sample(*index).unwrap();
                output1.copy_from_slice(sample1.as_slice().unwrap());
                output2.copy_from_slice(sample2.as_slice().unwrap());
            });
        let output1 = Array::from_shape_vec(dim1, output1).unwrap();
        let output2 = Array::from_shape_vec(dim2, output2).unwrap();
        Some(Ok((output1.into(), output2.into())))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

#[test]
fn batches1() {
    use ndarray::Array;
    let x = Array::from_shape_vec([10, 4], (1..).take(40).collect()).unwrap();
    for batch in x.view().tensor_batches(2) {
        let x = batch.unwrap();
        let x = x.as_array().unwrap();
        dbg!(x);
    }
    todo!()
}

#[test]
fn batches2() {
    use ndarray::Array;
    let x = Array::from_shape_vec([10, 4], (1..).map(|x| x as f32).take(40).collect()).unwrap();
    let y = Array::from_shape_vec([10, 4], (1..).take(40).collect()).unwrap();
    for batch in x.view().zip(y.view()).tensor_batches(2) {
        let (x, y) = batch.unwrap();
        let x = x.as_array().unwrap();
        let y = y.as_array().unwrap();
        dbg!(x, y);
    }
    todo!()
}
