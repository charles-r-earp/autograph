use crate::{float_tensor::FloatTensorD, result::Result, tensor::TensorD, uint::Uint};
use ndarray::Axis;
#[cfg(feature = "Serde")]
use serde::{Deserialize, Serialize};

use std::{
    iter::empty,
    time::{Duration, Instant},
};

/// KMeans classifier.
pub mod kmeans;

/// Testing / Evaluation.
///
/// [`Test`] is a general purpose trait for testing / evaluating a trainer / model.
pub trait Test<X> {
    /// Tests the model with the test data.
    ///
    /// Unlike [`Train::train_test()`], this method does not require mutable (exclusive) access. This may be useful for evaluating the model on a large dataset, since it can be run in parallel (ie with [rayon](https://docs.rs/rayon/rayon/)).
    ///
    /// Returns the testing stats.
    ///
    /// **Errors**
    /// Returns an error if testing could not be performed.
    fn test<I>(&self, test_iter: I) -> Result<Stats>
    where
        I: IntoIterator<Item = Result<X>>;
}

/// Training / Testing statistics.
#[derive(Default, Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Stats {
    count: usize,
    loss: Option<f32>,
    correct: Option<usize>,
}

/// Summary of training.
#[allow(missing_docs)]
#[derive(Default, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Summary {
    pub epoch: usize,
    pub epoch_time: Duration,
    pub total_time: Duration,
    pub train_stats: Stats,
    pub test_stats: Stats,
}

impl Summary {
    /// Runs an epoch with `f`.
    ///
    /// Times `f`. If `f` returns `Ok((train_stats, test_stats))`, updates the stats and epoch time, and accumulates the total time and the epoch. Otherwise returns the error.
    pub fn run_epoch<F>(&mut self, f: F) -> Result<(Stats, Stats)>
    where
        F: FnOnce(&Self) -> Result<(Stats, Stats)>,
    {
        let start = Instant::now();
        let (train_stats, test_stats) = f(self)?;
        self.epoch_time = start.elapsed();
        self.total_time += self.epoch_time;
        self.epoch += 1;
        self.train_stats = train_stats;
        self.test_stats = test_stats;
        Ok((train_stats, test_stats))
    }
}

/// Summarizes the trainer.
///
/// The trainer is expected to compute a summary on each call to [`.train()`](Train::train()). Use [`Summary::run_epoch()`] to compute the next summary.
pub trait Summarize {
    /// Returns a summary.
    fn summarize(&self) -> Summary;
}

/// Training.
///
/// [`Train`] is a general purpose trait for machine learning "trainers" that train a model, potentially iteratively with several "epochs". [`.train()`](Train::train()) trains the model.
///
/// Typically a trainer will include a "model" that implements [`Infer`], and a [`Summary`] that stores training statistics, as well as any additional training state.
///
/// Trainers should implement [`Infer`] where appropriate, by deferring to the model implementation.
///
/// # serde
/// Implement [`Serialize`](serde::Serialize) and [`Deserialize`](serde::Deserialize) for saving and loading checkpoints.
pub trait Train<X>: Test<X> + Summarize {
    /// Initializes the model / trainer.
    ///
    /// The closure `f` takes the number of iterations and returns an iterator of training set iterators, where each training set iterator iterates over the data in the same order.
    ///
    /// The implementation should reset training state, and initialize the model if it is not initialized.
    ///
    /// **Errors**
    ///
    /// Returns an error if initialization could not be performed. The trainer may be modified even when returning an error.
    fn init<F, I>(&mut self, f: F) -> Result<()>
    where
        F: FnOnce(usize) -> I,
        I: IntoIterator,
        <I as IntoIterator>::Item: IntoIterator<Item = Result<X>>;
    /// Trains the model with the training and testing sets.
    ///
    /// Returns (`train_stats`, `test_stats`).
    ///
    /// **Errors**
    ///
    /// Returns an error if training / testing could not be performed. The trainer may be modified even when returning an error.
    fn train_test<I1, I2>(&mut self, train_iter: I1, test_iter: I2) -> Result<(Stats, Stats)>
    where
        I1: IntoIterator<Item = Result<X>>,
        I2: IntoIterator<Item = Result<X>>;
    /// Trains the model with the training set.
    ///
    /// Returns the training stats.
    ///
    /// **Errors**
    ///
    /// Returns an error if training could not be performed. The trainer may be modified even when returning an error.
    fn train<I>(&mut self, train_iter: I) -> Result<Stats>
    where
        I: IntoIterator<Item = Result<X>>,
    {
        Ok(self.train_test(train_iter, empty())?.0)
    }
}

/// Inference
///
/// [`Infer`] is a trait for models (and trainers).
///
/// # serde
/// Implement [`Serialize`](serde::Serialize) and [`Deserialize`](serde::Deserialize) for saving and loading the model.
#[allow(missing_docs)]
pub trait Infer<X>: Test<X> {
    fn infer(&self, input: &X) -> Result<FloatTensorD>;
    /*fn classify<F: Float>(&self, input: &X) -> Result<TensorD<F>> {
        todo!() //self.infer(input)?.softmax_axis(Axis(1))
    }*/
    fn predict<U: Uint>(&self, input: &X) -> Result<TensorD<U>> {
        self.infer(input)?.argmax_axis(Axis(1))
    }
}
