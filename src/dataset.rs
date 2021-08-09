/*use crate::result::Result;
#[cfg(feature = "tensor")]
use crate::tensor::ArcTensor;
#[cfg(feature = "tensor")]
use anyhow::bail;
#[cfg(feature = "tensor")]
use ndarray::{Array, ArrayBase, Axis, Data as ArrayData, RemoveAxis};
#[cfg(feature = "rand")]
use rand::seq::SliceRandom as _;
use std::ops::Range;
#[cfg(feature = "rand")]
use std::vec::IntoIter as VecIntoIter;
*/
/// The Iris dataset.
#[cfg(feature = "iris")]
pub mod iris;
/*
/// A dataset that can be sampled.
pub trait Dataset {
    /// The type of each sample / batch.
    type Item;
    /// Samples the dataset with `indices`.
    ///
    /// **Errors**
    /// - An index is out of range (ie greater than [`size()`](Dataset::size())).
    /// - The operation could not be performed (ie unable to read from a file).
    fn sample<I>(&self, indices: I) -> Result<Self::Item>
    where
        I: IntoIterator<Item = usize>;
    /// Returns an iterator over batches with size `batch_size`.
    ///
    /// See [`Batches`].
    fn batches(self, batch_size: usize) -> Batches<Self>
    where
        Self: Sized,
    {
        Batches::new(self, batch_size)
    }
    /// Borrows a dataset.
    ///
    /// Typically [`Dataset`] adapters consume `self`. Use this to apply adapters to &Self instead.
    ///
    ///```
    /// # #[cfg(feature = "ndarray")]
    /// # {
    /// use autograph::dataset::Dataset;
    /// use ndarray::Array;
    /// let array = Array::from(vec![1, 2, 3, 4]);
    /// for x in array.by_ref().batches(2) {
    /// // process x
    /// }
    /// # }
    ///```
    fn by_ref(&self) -> &Self {
        self
    }
    /// Zips with another dataset (like [`Iterator::zip()`]).
    ///
    /// - [`.sample()`](Dataset::sample()) will sample each dataset with `indices`, returning an error if either returns and error.
    /// - [`.size()`](Dataset::size()) will be the min of [`Self::size()`] and [`B::size()`](Dataset::size).
    fn zip<B>(self, other: B) -> Zip<Self, B>
    where
        Self: Sized,
    {
        Zip::new(self, other)
    }
    /// The number of samples in the dataset.
    fn size(&self) -> usize;
    /// Returns whether the dataset is empty.
    fn is_empty(&self) -> bool {
        self.size() == 0
    }
}

impl<A: Dataset> Dataset for &A {
    type Item = A::Item;
    fn sample<I>(&self, indices: I) -> Result<Self::Item>
    where
        I: IntoIterator<Item = usize>,
    {
        A::sample(self, indices)
    }
    fn size(&self) -> usize {
        A::size(self)
    }
}

#[derive(Debug, Clone)]
enum Indices {
    Range(Range<usize>),
    #[cfg(feature = "rand")]
    Vec(VecIntoIter<usize>),
}

impl Iterator for Indices {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        match self {
            Self::Range(range) => range.next(),
            #[cfg(feature = "rand")]
            Self::Vec(iter) => iter.next(),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Range(range) => range.size_hint(),
            #[cfg(feature = "rand")]
            Self::Vec(iter) => iter.size_hint(),
        }
    }
}

impl ExactSizeIterator for Indices {}

/// [`Iterator`] over batches. See [`Dataset::batches()`].
#[derive(Debug, Clone)]
pub struct Batches<A> {
    dataset: A,
    indices: Indices,
    batch_size: usize,
}

impl<A: Dataset> Batches<A> {
    fn new(dataset: A, batch_size: usize) -> Self {
        let indices = Indices::Range(0..dataset.size());
        Self {
            dataset,
            indices,
            batch_size,
        }
    }
    #[cfg(feature = "rand")]
    /// Potentially randomly shuffles the batches using [`rand::seq::SliceRandom::shuffle()`].
    ///
    /// Shuffling generally greatly improves training speed.
    pub fn shuffled(mut self, shuffled: bool) -> Self {
        let mut indices = self.indices.collect::<Vec<_>>();
        indices.shuffle(&mut rand::thread_rng());
        Self {
            dataset: self.dataset,
            indices: Indices::Vec(indices.into_iter()),
            batch_size: self.batch_size,
        }
    }
}

impl<A: Dataset> Iterator for Batches<A> {
    type Item = Result<A::Item>;
    fn next(&mut self) -> Option<Result<A::Item>> {
        let indices = self.indices.by_ref().take(self.batch_size);
        if indices.len() == 0 {
            None
        } else {
            Some(self.dataset.sample(indices))
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let mut len = self.indices.len() / self.batch_size;
        if self.indices.len() % self.batch_size != 0 {
            len += 1;
        }
        (len, Some(len))
    }
}

impl<A: Dataset> ExactSizeIterator for Batches<A> {}

/// A pair of datasets "zipped" together. See [`Dataset::zip()`].
#[derive(Debug, Clone)]
pub struct Zip<A, B> {
    a: A,
    b: B,
}

impl<A, B> Zip<A, B> {
    fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A: Dataset, B: Dataset> Dataset for Zip<A, B> {
    type Item = (A::Item, B::Item);
    fn sample<I>(&self, indices: I) -> Result<Self::Item>
    where
        I: IntoIterator<Item = usize>,
    {
        let indices = indices.into_iter().collect::<Vec<_>>();
        let a = self.a.sample(indices.iter().copied())?;
        let b = self.b.sample(indices.iter().copied())?;
        Ok((a, b))
    }
    fn size(&self) -> usize {
        self.a.size().min(self.b.size())
    }
}

#[cfg(feature = "tensor")]
impl<T: Default + Copy, S: ArrayData<Elem = T>, D: RemoveAxis> Dataset for ArrayBase<S, D> {
    type Item = ArcTensor<T, D>;
    fn sample<I>(&self, indices: I) -> Result<Self::Item>
    where
        I: IntoIterator<Item = usize>,
    {
        let indices = indices.into_iter().collect::<Vec<_>>();
        let size = self.size();
        if let Some(index) = indices.iter().find(|i| *i > &size) {
            bail!("Index {} out of range {}!", index, size);
        }
        let mut dim = self.raw_dim();
        dim[0] = indices.len();
        let mut output = Array::from_elem(dim, T::default());
        for (mut y, i) in output.outer_iter_mut().zip(indices) {
            y.assign(&self.index_axis(Axis(0), i));
        }
        Ok(ArcTensor::from(output))
    }
    fn size(&self) -> usize {
        self.shape().first().map_or(0, |x| *x)
    }
}
*/
