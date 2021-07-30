use crate::result::Result;
#[cfg(feature = "tensor")]
use crate::tensor::{ArcTensor1, ArcTensor2};
#[cfg(feature = "Serde")]
use serde::{Deserialize, Serialize};

use std::{
    iter::empty,
    time::{Duration, Instant},
};

trait Infer<X> {
    type Output;
    fn infer(&self, input: X) -> Result<Self::Output>;
}

#[cfg(feature = "tensor")]
trait Classify<X> {
    // May change to dynamic types id FloatArcTensor2.
    fn classify<F>(input: X) -> Result<ArcTensor2<F>>;
    #[allow(unused)]
    fn predict<U>(input: X) -> Result<ArcTensor1<U>> {
        todo!()
    }
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
#[derive(Default, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Summary {
    epoch: usize,
    epoch_time: Duration,
    total_time: Duration,
    train: Stats,
    test: Stats,
}

impl Summary {
    /// Runs an epoch with `f`.
    ///
    /// Times `f`. If `f` returns `Ok`, updates the epoch time and accumulates the total time and the epoch. Otherwise returns the error.
    pub fn run_epoch<F>(&mut self, mut f: F) -> Result<(Stats, Stats)>
    where
        F: FnMut(&Self) -> Result<(Stats, Stats)>,
    {
        let start = Instant::now();
        let (train, test) = f(self)?;
        self.epoch_time = start.elapsed();
        self.total_time += self.epoch_time;
        self.epoch += 1;
        self.train = train;
        self.test = test;
        Ok((train, test))
    }
}

/// Summerizes the trainer.
pub trait Summarize {
    /// Returns a summary.
    fn summarize(&self) -> Summary;
}

/// Training.
///
/// [`Train`] is a general purpose trait for machine learning "trainers" that train a model, potentially iteratively with several "epochs". [`.train()`](Train::train()) trains the model
///
/// # Summary
/// Implement [`Summarize`], as the trainer is expected to compute a summary on each call to [`.train()`](Train::train()). Use [`Summary::run_epoch()`] to compute the next summary.
///
/// # Test
/// Implement [`Test`] so that the model can be tested without exclusive access.
///
/// # serde
/// Implement [`Serialize`](serde::Serialize) and [`Deserialize`](serde::Deserialize) for saving and loading checkpoints.
pub trait Train<X> {
    /// Trains the model with the training and testing sets.
    ///
    /// Returns (`train_stats`, `test_stats`).
    ///
    /// **Errors**
    /// Returns an error if training / testing could not be performed. The trainer may be modified even when returning an error.
    fn train_test<I1, I2>(&mut self, train_iter: I1, test_iter: I2) -> Result<(Stats, Stats)>
    where
        I1: IntoIterator<Item = X>,
        I2: IntoIterator<Item = X>;
    /// Trains the model with the training set.
    ///
    /// Returns the training stats.
    ///
    /// **Errors**
    /// Returns an error if training could not be performed. The trainer may be modified even when returning an error.
    fn train<I>(&mut self, train_iter: I) -> Result<Stats>
    where
        I: IntoIterator<Item = X>,
    {
        Ok(self.train_test(train_iter, empty())?.0)
    }
}

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
        I: IntoIterator<Item = X>;
}
