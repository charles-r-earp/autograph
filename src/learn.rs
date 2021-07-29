use crate::{
    result::Result,
};
use rand::{thread_rng, seq::SliceRandom};
use std::{
    ops::Range,
};

#[derive(Default, Clone, Copy, Debug)]
struct Stats {
    mean_loss: Option<f32>,
    correct: Option<usize>,
    len: usize,
}

trait Infer<X, Y> {
    fn infer(&self, input: X) -> Result<Y>;
}

trait Test<B> {
    fn test<I>(&self, input_iter: I) -> Result<Stats>
        where I: IntoIterator<Item=Result<B>>;
}

trait Train<B>: Test<B> {
    fn train<I>(&mut self, input_iter: I) -> Result<Stats>
        where I: IntoIterator<Item=Result<B>>;
}

trait Fit<B>: Train<B> {
    #[allow(unused)]
    fn init<S: Dataset<Item=B>>(&mut self, dataset: &S) -> Result<()> {
        Ok(())
    }
    fn fit<S>(&mut self, dataset: &S, batch_size: usize, test_ratio: f32) -> Result<(Stats, Stats)>
        where S: Dataset<Item=B> {
        let test_len = (dataset.len() as f32 * test_ratio).round() as usize;
        let train_len = dataset.len() - test_len;
        self.init(dataset)?; // <-- need to limit length, ie slice the dataset
        let mut train_indices = (0 .. train_len).into_iter().collect::<Vec<_>>();
        train_indices.shuffle(&mut thread_rng());
        let train_stats = self.train(dataset.batches(Indices::Vec(train_indices), batch_size)?)?;
        let test_stats = if test_len > 0 {
            self.test(dataset.batches(Indices::Range(train_len .. dataset.len()), batch_size)?)?
        } else {
            Stats::default()
        };
        Ok((train_stats, test_stats))
    }
}

enum Indices {
    Range(Range<usize>),
    Vec(Vec<usize>),
}

trait Dataset {
    type Item;
    type Iter: Iterator<Item=Result<Self::Item>>;
    fn sample(&self, indices: Indices) -> Result<Self::Item>;
    fn batches(&self, indices: Indices, batch_size: usize) -> Result<Self::Iter>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

trait Batches<I, B> {
    type Iter: Iterator<Item=Result<B>>;
    fn batches(&self, indices: I, batch_size: usize) -> Result<Self::Iter>;
}
