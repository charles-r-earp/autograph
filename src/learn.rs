use crate::{
    backend::Device,
    dataset::{train_test_split, Dataset},
    tensor::{Tensor0, Tensor1, Tensor2},
    Result,
};
use std::time::Duration;

pub struct FitOptions {
    test_ratio: f32,
    train_batch_size: usize,
    test_batch_size: usize,
    shuffle: bool,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            test_ratio: 0.,
            train_batch_size: 64,
            test_batch_size: 1024,
            shuffle: true,
        }
    }
}

impl FitOptions {
    pub fn test_ratio(mut self, test_ratio: f32) -> Self {
        self.test_ratio = test_ratio;
        self
    }
    pub fn get_test_ratio(&self) -> f32 {
        self.test_ratio
    }
    pub fn train_batch_size(mut self, train_batch_size: usize) -> Self {
        self.train_batch_size = train_batch_size;
        self
    }
    pub fn get_train_batch_size(&self) -> usize {
        self.train_batch_size
    }
    pub fn test_batch_size(mut self, test_batch_size: usize) -> Self {
        self.test_batch_size = test_batch_size;
        self
    }
    pub fn get_test_batch_size(&self) -> usize {
        self.test_batch_size
    }
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }
    pub fn get_shuffle(&self) -> bool {
        self.shuffle
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FitStats {
    epoch: usize, // epoch starts at 1 not 0
    duration: Duration,
    train_loss: f32,
    train_count: usize,
    train_correct: Option<usize>,
    test_loss: f32,
    test_count: usize,
    test_correct: Option<usize>,
    best_epoch: Option<usize>,
}

impl Default for FitStats {
    fn default() -> Self {
        Self {
            epoch: 1,
            duration: Duration::default(),
            train_loss: 0.,
            train_count: 0,
            train_correct: None,
            test_loss: 0.,
            test_count: 0,
            test_correct: None,
            best_epoch: None,
        }
    }
}

impl FitStats {
    pub(crate) fn next_epoch(&mut self) {
        self.epoch += 1;
        self.train_loss = 0.;
        self.train_count = 0;
        self.train_correct = None;
        self.test_loss = 0.;
        self.test_count = 0;
        self.test_correct = None;
    }
}

pub trait Fit<X> {
    #[allow(unused_variables)]
    fn initialize_from_dataset<A>(&mut self, dataset: &A, options: &FitOptions) -> Result<FitStats>
    where
        A: Dataset<Item = X>,
    {
        Ok(FitStats::default())
    }
    fn train_epoch<I>(&mut self, train_iter: I) -> Result<(Tensor0<f32>, Option<Tensor0<u32>>)>
    where
        I: Iterator<Item = Result<X>>;
    fn test_epoch<I>(&self, test_iter: I) -> Result<(Tensor0<f32>, Option<Tensor0<u32>>)>
    where
        I: Iterator<Item = Result<X>>;
    fn fit<A, F>(
        &mut self,
        device: &Device,
        dataset: &A,
        options: FitOptions,
        mut callback: F,
    ) -> Result<FitStats>
    where
        A: Dataset<Item = X>,
        F: FnMut(&mut Self, &FitStats) -> Result<bool>,
    {
        smol::block_on(async {
            let (train_set, test_set) = train_test_split(dataset, options.test_ratio);
            let mut stats = self.initialize_from_dataset(&train_set, &options)?;
            loop {
                let mut train_iter = train_set
                    .batches(device, options.train_batch_size, options.shuffle)
                    .peekable();
                let train_iter = std::iter::from_fn(move || {
                    train_iter.peek();
                    Some(smol::block_on(train_iter.next()?))
                });
                let (train_loss, train_correct) = self.train_epoch(train_iter)?;
                let train_loss = train_loss.to_vec()?;
                let train_correct = train_correct.as_ref();
                let train_correct = if let Some(train_correct) = train_correct {
                    Some(train_correct.to_vec()?)
                } else {
                    None
                };
                if test_set.sample_count() > 0 {
                    let mut test_iter = test_set
                        .batches(device, options.test_batch_size, false)
                        .peekable();
                    let test_iter = std::iter::from_fn(move || {
                        test_iter.peek();
                        Some(smol::block_on(test_iter.next()?))
                    });
                    let (test_loss, test_correct) = self.test_epoch(test_iter)?;
                    let test_loss = test_loss.to_vec()?;
                    let test_correct = test_correct.as_ref();
                    let test_correct = if let Some(test_correct) = test_correct {
                        Some(test_correct.to_vec()?)
                    } else {
                        None
                    };
                    stats.test_count = test_set.sample_count();
                    stats.test_loss = *test_loss.await?.first().unwrap();
                    stats.test_correct = if let Some(test_correct) = test_correct {
                        Some(*test_correct.await?.first().unwrap() as usize)
                    } else {
                        None
                    };
                }
                stats.train_count = train_set.sample_count();
                stats.train_loss = *train_loss.await?.first().unwrap();
                stats.train_correct = if let Some(train_correct) = train_correct {
                    Some(*train_correct.await?.first().unwrap() as usize)
                } else {
                    None
                };
                if !callback(self, &stats)? {
                    return Ok(stats);
                }
                stats.next_epoch();
            }
        })
    }
}

pub trait Infer<X> {
    type Output;
    fn infer(&self, input: X) -> Result<Self::Output>;
}

pub trait Classify<X> {
    fn classify(&self, input: X) -> Result<Tensor2<f32>>;
}

pub trait Predict<X> {
    fn predict(&self, input: X) -> Result<Tensor1<u32>>;
}

/*
pub struct FitStats {
    train_loss: Option<f32>,
    train_correct: Option<usize>,
    test_loss: Option<f32>,
    test_correct: Option<usize>,
    /// Used for signaling training has reached termination
    completed: bool,
}



pub trait Fit<X> {
    fn fit<F1, F2>(&mut self, train_iter: impl Iterator<Item=F1>, test_iter: impl Iterator<Item=F2>) -> Result<FitStats>
        where F1: Future<Output=Result<X>>, F2: Future<Output=Result<X>>;
}
*/
/*
#[derive(Default, Clone, Copy, Debug)]
pub struct FitStats {
    pub epochs: usize,
    pub duration: Duration,
    pub train_loss: Option<f32>,
    pub train_correct: Option<usize>,
    pub train_len: usize,
    pub test_loss: Option<f32>,
    pub test_correct: Option<usize>,
    pub test_len: usize,
}

impl Display for FitStats {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = format!("epochs: {:>5} elapsed: {:7.2?}", self.epochs, self.duration);
        if let Some(train_loss) = self.train_loss {
            s.push_str(&format!(" train_loss: {:7.5}", train_loss));
        }
        if let Some(train_correct) = self.train_correct {
            let train_accuracy = train_correct as f32 / self.train_len as f32;
            s.push_str(&format!(" train_accuracy: {:>7}/{:<7} {:4.2}%", train_correct, self.train_len, train_accuracy * 100.));
        }
        if let Some(test_loss) = self.test_loss {
            s.push_str(&format!(" test_loss: {:7.5}", test_loss));
        }
        if let Some(test_correct) = self.test_correct {
            let test_accuracy = test_correct as f32 / self.test_len as f32;
            s.push_str(&format!(" test_accuracy: {:>7}/{:<7} {:4.2}%", test_correct, self.test_len, test_accuracy * 100.));
        }
        write!(f, "{}", s)
    }
}

#[derive(Default, Clone, Copy)]
pub struct EpochStats {
    pub train_loss: Option<f32>,
    pub train_correct: Option<usize>,
    pub test_loss: Option<f32>,
    pub test_correct: Option<usize>,
}

pub trait FitEpoch<X> {
    fn fit_epoch<I1, F1, I2, F2>(&mut self, train_iter: I1, test_iter: Option<I2>) -> Result<EpochStats>
        where I1: Iterator<Item=F1>, F1: Future<Output=Result<X>>, I2: Iterator<Item=F2>, F2: Future<Output=Result<X>>;
    fn fit()
}

#[derive(Clone, Copy, Debug)]
pub struct FitOptions {
    start_epoch: usize,
    start_duration: Duration,
    train_batch_size: usize,
    test_batch_size: usize,
    shuffle: bool,
    max_epochs: Option<usize>,
    max_duration: Option<Duration>,
    patience: Option<usize>,
}

impl FitOptions {
    pub fn batch_size(self, batch_size: usize) -> Self {
        self.train_batch_size(batch_size)
            .test_batch_size(batch_size)
    }
    pub fn train_batch_size(mut self, train_batch_size: usize) -> Self {
        self.train_batch_size = train_batch_size;
        self
    }
    pub fn test_batch_size(mut self, test_batch_size: usize) -> Self {
        self.test_batch_size = test_batch_size;
        self
    }
    pub fn max_epochs(mut self, max_epochs: usize) -> Self {
        self.max_epochs.replace(max_epochs);
        self
    }
    pub fn max_duration(mut self, max_duration: Duration) -> Self {
        self.max_duration.replace(max_duration);
        self
    }
    pub fn patience(mut self, patience: Option<usize>) -> Self {
        self.patience = patience;
        self
    }
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            start_epoch: 1,
            start_duration: Duration::default(),
            train_batch_size: 100,
            test_batch_size: 1000,
            shuffle: true,
            max_epochs: None,
            max_duration: None,
            patience: Some(10),
        }
    }
}

pub fn fit_x<'a, T, X>(trainer: &mut T, train_loader: &DataLoader<'a, X>, test_loader: Option<&DataLoader<'a, X>>, options: FitOptions) -> Result<FitStats>
    where &'a X: InputData<'a>, T: FitX<<&'a X as InputData<'a>>::Input> {
    fn log_epoch(progress: &ProgressBar, epoch: usize, duration: Duration, stats: &FitStats) {
        let mut log = format!("epoch: {:>5} elapsed: {:7.2?}", epoch, duration);
        if let Some(train_loss) = stats.train_loss {
            log.push_str(&format!(" train_loss: {:7.5}", train_loss));
        }
        if let Some(train_correct) = stats.train_correct {
            let train_accuracy = train_correct as f32 / stats.train_len as f32;
            log.push_str(&format!(" train_accuracy: {:>7}/{:<7} {:4.2}%", train_correct, stats.train_len, train_accuracy * 100.));
        }
        if let Some(test_loss) = stats.test_loss {
            log.push_str(&format!(" test_loss: {:7.5}", test_loss));
        }
        if let Some(test_correct) = stats.test_correct {
            let test_accuracy = test_correct as f32 / stats.test_len as f32;
            log.push_str(&format!(" test_accuracy: {:>7}/{:<7} {:4.2}%", test_correct, stats.test_len, test_accuracy * 100.));
        }
        progress.println(log);
    }
    let start = Instant::now();
    let style = ProgressStyle::default_bar()
        .template("{prefix} [{wide_bar}] {msg} {pos:>7}/{len:<7} {percent}% elapsed: {elapsed:5} eta: {eta:5}")
        .progress_chars("=> ");
    let ref progress = ProgressBar::new(1000)
        .with_style(style);

    let train_len = train_loader.len();
    let test_len = test_loader.as_ref().map_or(0, |x| x.len());
    let mut stats = FitStats {
        train_len,
        test_len,
        ..FitStats::default()
    };
    let mut best_stats = stats.clone();
    let mut best_loss = f32::max;
    let mut best_accuracy = 0.;
    let mut patience_counter = 0;
    for epoch in options.start_epoch.. {
        stats.epochs = epoch - 1;
        stats.duration = start.elapsed() + options.start_duration;
        // TODO: Update best stats
        if let Some(max_epochs) = options.max_epochs {
            if stats.epochs >= max_epochs {
                return Ok(best_stats);
            }
        }
        if let Some(max_duration) = options.max_duration {
            if stats.duration >= max_duration {
                return Ok(best_stats);
            }
        }
        if let Some(patience) = options.patience {
            // TODO
        }
        progress.set_prefix(&format!("epoch: {}", epoch));
        let mut train_iter = train_loader.input_iter(options.train_batch_size, options.shuffle)
            .enumerate()
            .peekable();
        let train_iter = std::iter::from_fn(move || {
            train_iter.peek(); // prefetch next future
            train_iter.next()
                .map(|(i, x)| {
                    if i == 0 {
                        progress.set_length(train_len as u64);
                        progress.set_message("Train");
                    }
                    let pos = (i*options.train_batch_size).min(train_len);
                    progress.set_position(pos as u64);
                    smol::block_on(x)
                })
        });
        let test_iter = test_loader.as_ref()
            .map(|test_loader| {
                let mut test_iter = test_loader.input_iter(options.train_batch_size, options.shuffle)
                    .enumerate()
                    .peekable();
                std::iter::from_fn(move || {
                    test_iter.peek(); // prefetch next future
                    test_iter.next()
                        .map(|(i, x)| {
                            if i == 0 {
                                progress.set_length(test_len as u64);
                                progress.set_message("Test");
                            }
                            let pos = (i*options.test_batch_size).min(test_len);
                            progress.set_position(pos as u64);
                            smol::block_on(x)
                        })
                })
            });
        let epoch_start = Instant::now();
        let epoch_stats = if let Some(test_iter) = test_iter {
            trainer.fit_x_epoch(train_iter, test_iter)?
        } else {
            trainer.fit_x_epoch(train_iter, std::iter::empty())?
        };
        let epoch_elapsed = epoch_start.elapsed();
        stats.train_loss = epoch_stats.train_loss;
        stats.train_correct = epoch_stats.train_correct;
        stats.test_loss = epoch_stats.test_loss;
        stats.test_correct = epoch_stats.test_correct;
        log_epoch(&progress, epoch, epoch_elapsed, &stats);
    }
    unreachable!()
}
*/
