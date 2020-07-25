use crate::{Device, Tensor};
use super::autograd::{Parameter, ParameterD, ParameterMeta, OptimizerDataEntry};
use ndarray::{Dimension, IntoDimension, IxDyn};
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use std::{io, error::Error, fs::{self, File}, path::{Path, PathBuf}, str::FromStr};

/// Saved model parameters\
///
/// Typically the "model" implements Layer, which has a parameters() method. Only the parameter data is stored, not dimensions or model hyperparameters. A different model representation can be used to load the parameters, as long as the parameters are in the same order and the same shape. Models trained on a gpu can be saved and loaded into a model on the cpu and vice versa.  
/// Saving:\
///```
/// SavedModel::new(model.parameters())
///     .save("mymodel")
///     .expect("Unable to save model!");
///```
/// Loading:\
///```
/// SavedModel::load("mymodel")
///     .expect("Unable to load model!")
///     .load_parameters(model.parameters());
///```
#[derive(Serialize, Deserialize)]
pub struct SavedModel {
    parameters: Vec<SavedParameter>
}

impl SavedModel {
    /// Prepare a collection of parameters for serialization\
    pub fn new(parameters: impl IntoIterator<Item=ParameterD>) -> Self {
        let parameters = parameters.into_iter()
            .map(|parameter| parameter.to_saved(false))
            .collect();
        Self { parameters }
    }
    /// Move the data in self to the collection of parameters 
    pub fn load_parameters(self, parameters: impl IntoIterator<Item=ParameterD>) {
        self.parameters.into_iter()
            .zip(parameters)
            .for_each(|(saved, parameter)| parameter.load(saved, false));
    } 
    /// Save the parameters to a file with extension ".model"
    pub fn save(&self, name: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let name = name.as_ref()
            .with_extension("model");
        let mut file = File::create(name)?;
        bincode::serialize_into(&mut file, self)?;
        Ok(())
    }
    /// Load the parameters from a file with extension ".model"
    pub fn load(name: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        let name = name.as_ref()
            .with_extension("model");
        let file = File::open(name)?;
        let model = bincode::deserialize_from(&file)?;
        Ok(model)  
    }
}

/// Saved checkpoint\
///
/// A checkpoint saves training progress, allowing training to be resumed if interupted. As with SavedModel, training saved from one device can be resumed from a different device.\
/// Saving:\
///```
/// SavedCheckpoint::new(epoch, model.parameters(), &optim)
///     .save("mymodel")
///     .expect("Unable to save checkpoint!");
///```
/// Loading:\
///```
/// let (epoch, optim) = SavedCheckpoint::load("mymodel")
///     .expect("Unable to load checkpoint!")
///     .load_parameters(model.parameters());
///```
#[derive(Serialize, Deserialize)]
pub struct SavedCheckpoint<O> {
    epoch: usize,
    parameters: Vec<SavedParameter>,
    optimizer: O
}

impl<O> SavedCheckpoint<O> {
    /// Prepare a collection of parameters and an optimizer for serialization
    pub fn new(epoch: usize, parameters: impl IntoIterator<Item=ParameterD>, optimizer: O) -> Self {
        let parameters = parameters.into_iter()
            .map(|parameter| parameter.to_saved(true))
            .collect();
        Self { epoch, parameters, optimizer }
    }
    /// Move the data in self to the parameters, and return the epoch and optimizer
    pub fn load_parameters(self, parameters: impl IntoIterator<Item=ParameterD>) -> (usize, O) {
        self.parameters.into_iter()
            .zip(parameters)
            .for_each(|(saved, parameter)| parameter.load(saved, true));
        (self.epoch, self.optimizer)
    }
    /// Save the checkpoint to a file: name + "_epoch" + epoch, with extension ".checkpoint"\
    /// ie "mymodel_epoch10.checkpoint"\
    /// Err: Returns an error if the file cannot be created or serialiation fails
    pub fn save(&self, name: impl AsRef<Path>) -> Result<(), Box<dyn Error>>
        where O: Serialize {
        let name = name.as_ref();
        let stem = name.file_stem()
            .unwrap()
            .to_str()
            .unwrap();
        let name = name
            .with_file_name(&format!("{}_epoch{}", stem, self.epoch))
            .with_extension("checkpoint");
        let mut file = File::create(name)?;
        bincode::serialize_into(&mut file, self)?;
        Ok(())
    }
    /// Load a checkpoint created by save(). The most recent checkpoint, ie the one with the largest epoch, will be loaded.\
    /// Err: Returns an error if the file cannot be opened or deserialization fails
    pub fn load(name: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> 
        where O: DeserializeOwned {
        let name = name.as_ref();
        let stem = name.file_stem()
            .unwrap()
            .to_str()
            .unwrap();
        let dir = {
            let dir = name.parent().unwrap();
            if !dir.to_str().unwrap().is_empty() {
                dir.to_owned()
            }
            else {
                PathBuf::from(".")
            }
        };
        let epoch = fs::read_dir(dir)?
            .filter_map(|entry| {
                if let Ok(entry) = entry {
                    if let Ok(fname) = entry.file_name().into_string() {
                        if let Some(fname) = fname.strip_suffix(".checkpoint") {
                            if let Some(fname) = fname.strip_prefix(&format!("{}_epoch", stem)) {
                                usize::from_str(fname).ok()
                            }
                            else {
                                None
                            }
                        }
                        else {
                            None
                        }
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                }
           })
           .max()
           .unwrap_or(0);
        let name = name
            .with_file_name(&format!("{}_epoch{}", stem, epoch))
            .with_extension("checkpoint");
        let file = File::open(name)?;
        let checkpoint = bincode::deserialize_from(&file)?;
        Ok(checkpoint)
    }
}

#[derive(Serialize, Deserialize)]
enum SavedOptimizerDataEntry {
    VelocityTensor(Vec<f32>)    
}

impl OptimizerDataEntry {
    fn to_saved(&self) -> SavedOptimizerDataEntry {
        match self {
            OptimizerDataEntry::VelocityTensor(tensor) => SavedOptimizerDataEntry::VelocityTensor(tensor.as_slice().into_owned()),
        }
    }
    fn load_from(device: &Device, dim: &IxDyn, saved: SavedOptimizerDataEntry) -> Self {
        match saved {
            SavedOptimizerDataEntry::VelocityTensor(vec) => {
                OptimizerDataEntry::VelocityTensor(
                    Tensor::from_shape_vec(device, dim.clone(), vec)
                )
            }
        }
    }  
    fn load(&mut self, saved: SavedOptimizerDataEntry) {
        match self {
            OptimizerDataEntry::VelocityTensor(tensor) => {
                match saved {
                    SavedOptimizerDataEntry::VelocityTensor(vec) => {
                        tensor.copy_from_slice(vec)
                    }
                }
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
struct SavedParameterMeta {
    optimizer_data: Vec<SavedOptimizerDataEntry>
}

impl ParameterMeta {
    fn to_saved(&self, with_optimizer_data: bool) -> SavedParameterMeta {
        let optimizer_data = if with_optimizer_data {
            self.optimizer_data()
                .read()
                .unwrap()
                .iter()
                .map(|entry| entry.to_saved())
                .collect()
        } 
        else {
            Vec::new()
        };
        SavedParameterMeta {
            optimizer_data
        }
    }
    fn load(&self, device: &Device, shape: impl IntoDimension<Dim=IxDyn>, saved: SavedParameterMeta, with_optimizer_data: bool) {
        if with_optimizer_data { 
            let mut optimizer_data = self.optimizer_data()
                .write()
                .unwrap();
            if optimizer_data.is_empty() {
                let dim = shape.into_dimension();
                optimizer_data.extend(
                    saved.optimizer_data.into_iter()
                        .map(|saved| OptimizerDataEntry::load_from(device, &dim, saved))
                );
            }
            else {
                assert_eq!(optimizer_data.len(), saved.optimizer_data.len());
                optimizer_data.iter_mut()
                    .zip(saved.optimizer_data.into_iter())
                    .for_each(|(data, saved)| data.load(saved));
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
struct SavedParameter {
    value: Vec<f32>,
    meta: SavedParameterMeta
}

impl<D: Dimension> Parameter<D> {
    fn to_saved(&self, with_optimizer_data: bool) -> SavedParameter {
        let value = self.value()
            .read()
            .unwrap()
            .as_slice()
            .into_owned();
        let meta = self.meta()
            .to_saved(with_optimizer_data);
        SavedParameter {
            value,
            meta
        }
    }
    fn load(&self, saved: SavedParameter, with_optimizer_data: bool) {
        let mut value = self.value()
            .write()
            .unwrap();
        value.copy_from_slice(saved.value);
        self.meta()
            .load(value.device(), value.raw_dim().into_dyn(), saved.meta, with_optimizer_data);
    }
}
