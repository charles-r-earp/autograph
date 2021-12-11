use anyhow::Result;

pub mod autograph_trainer;
#[cfg(feature = "tch")]
pub mod tch_trainer;

#[derive(Clone, Copy, Debug)]
pub enum DatasetKind {
    Mnist,
}

#[derive(Debug)]
pub struct DatasetDescriptor {
    pub kind: DatasetKind,
    pub train_batch_size: usize,
    pub test_batch_size: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum NetworkKind {
    Lenet5,
}

#[derive(Debug)]
pub struct NetworkDescriptor {
    pub kind: NetworkKind,
}

#[derive(Debug)]
pub struct TrainerDescriptor {
    pub dataset: DatasetDescriptor,
    pub network: NetworkDescriptor,
    pub epochs: usize,
}

#[derive(Debug)]
pub struct TrainerStats {
    pub total_time: Vec<f32>,
    pub test_accuracy: Vec<f32>,
}

pub trait Library {
    fn benchmark(
        trainer_descriptor: &TrainerDescriptor,
        epoch_cb: impl FnMut(usize),
    ) -> Result<TrainerStats>;
}
