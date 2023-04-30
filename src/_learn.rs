use crate::{
    result::Result,
    scalar::Uint,
    tensor::{float::FloatTensorD, TensorD},
};
use ndarray::Axis;
use serde::{Deserialize, Serialize};

use std::{
    fmt::{self, Debug},
    iter::empty,
    time::{Duration, Instant},
};

/// Criterions
pub mod criterion;

/// KMeans classifier.
#[cfg(feature = "kmeans")]
pub mod kmeans;

/// Neural Networks
#[cfg(feature = "neural_network")]
pub mod neural_network;

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
#[non_exhaustive]
#[derive(Default, Clone, Copy, Serialize, Deserialize)]
pub struct Stats {
    /// The number of samples.
    pub count: usize,
    /// The mean loss.
    pub loss: Option<f32>,
    /// The number of correct predictions.
    pub correct: Option<usize>,
}

impl Stats {
    /// The accuracy as a ratio between 0. and 1.
    ///
    /// If correct is Some, correct / count, else None.
    pub fn accuracy(&self) -> Option<f32> {
        self.correct
            .map(|correct| correct as f32 / self.count as f32)
    }
}

impl Debug for Stats {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut builder = f.debug_struct("Stats");
        builder.field("count", &self.count);
        if let Some(loss) = self.loss.as_ref() {
            builder.field("loss", loss);
        }
        if let Some(correct) = self.correct.as_ref() {
            builder.field("correct", correct);
            builder.field("accuracy", &(*correct as f32 / self.count as f32));
        }
        builder.finish()
    }
}

/// Summary of training.
#[non_exhaustive]
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Summary {
    /// The current epoch (starting at 0).
    pub epoch: usize,
    /// The run time for initialization.
    pub init_time: Duration,
    /// The run time for the current epoch.
    pub epoch_time: Duration,
    /// The total run time.
    pub total_time: Duration,
    /// The training stats.
    pub train_stats: Stats,
    /// The testing stats.
    pub test_stats: Stats,
}

impl Summary {
    /// Runs an epoch with `f`.
    ///
    /// - Initializes (resets to Self::default()).
    /// - Times `f`.
    /// - If `f` returns `Ok`, updates the epoch time, and accumulates the total time. Otherwise returns the error.
    pub fn run_init<F>(&mut self, f: F) -> Result<()>
    where
        F: FnOnce(&Self) -> Result<()>,
    {
        *self = Self::default();
        let start = Instant::now();
        f(self)?;
        self.init_time = start.elapsed();
        self.total_time += self.epoch_time;
        Ok(())
    }
    /// Runs an epoch with `f`.
    ///
    /// - Times `f`.
    /// - If `f` returns `Ok((train_stats, test_stats))`, updates the stats and epoch time, and accumulates the total time and the epoch. Otherwise returns the error.
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
/// # [`serde`]
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
/// [`Infer`] is a trait for models that take one or more inputs and produce an output.
///
/// # serde
/// Implement [`Serialize`](serde::Serialize) and [`Deserialize`](serde::Deserialize) for saving and loading the model.
pub trait Infer<X> {
    /// Performs inference.
    ///
    /// **Errors**
    ///
    /// Returns an error if the operation cannot be performed.
    fn infer(&self, input: &X) -> Result<FloatTensorD>;
    /*fn classify<F: Float>(&self, input: &X) -> Result<TensorD<F>> {
        todo!() //self.infer(input)?.softmax_axis(Axis(1))
    }*/
    /// Predicts the class.
    ///
    /// **Errors**
    ///
    /// Returns an error if the operation cannot be performed.
    fn predict<U: Uint>(&self, input: &X) -> Result<TensorD<U>> {
        self.infer(input)?.argmax_axis(Axis(1))
    }
}
