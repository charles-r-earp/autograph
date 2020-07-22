use crate::{Device, Tensor};
use super::{Parameter, ParameterD, ParameterMeta, OptimizerDataEntry};
use ndarray::{Dimension, IntoDimension, IxDyn};
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use std::{io, error::Error};

pub struct CheckpointWriter<'o, 'm, O, M> {
    parameters: Vec<ParameterD>,
    optimizer: Option<&'o O>,
    meta: Option<&'m M>
}

impl CheckpointWriter<'static, 'static, (), ()> {
    pub fn new(parameters: Vec<ParameterD>) -> Self {
        Self {
            parameters,
            optimizer: None,
            meta: None
        }
    }
}

impl<'o, 'm, O, M> CheckpointWriter<'o, 'm, O, M> {
    pub fn optimizer<'o2, O2>(self, optimizer: &'o2 O2) -> CheckpointWriter<'o2, 'm, O2, M> {
        CheckpointWriter {
            parameters: self.parameters,
            optimizer: Some(optimizer),
            meta: self.meta
        }
    }
    pub fn meta<'m2, M2>(self, meta: &'m2 M2) -> CheckpointWriter<'o, 'm2, O, M2> {
        CheckpointWriter {
            parameters: self.parameters,
            optimizer: self.optimizer,
            meta: Some(meta)
        }
    }
}

impl<'o, 'm, O: Serialize, M: Serialize> CheckpointWriter<'o, 'm, O, M> {
    pub fn write_into(&self, mut writer: impl io::Write) -> Result<(), Box<dyn Error>> {
        if let Some(meta) = self.meta {
            bincode::serialize_into(&mut writer, meta)?;
        }
        if let Some(optimizer) = self.optimizer {
            bincode::serialize_into(&mut writer, optimizer)?;
        }
        let with_optimizer_data = self.optimizer.is_some();
        for parameter in &self.parameters {
            bincode::serialize_into(
                &mut writer,
                &parameter.to_saved(with_optimizer_data)
            )?;
        }
        Ok(())
    }
}

pub struct CheckpointReader<'o, 'm, O, M> {
    parameters: Vec<ParameterD>,
    optimizer: Option<&'o mut O>,
    meta: Option<&'m mut M>
}

impl CheckpointReader<'static, 'static, (), ()> {
    pub fn new(parameters: Vec<ParameterD>) -> Self {
        Self {
            parameters,
            optimizer: None,
            meta: None
        }
    }
}

impl<'o, 'm, O, M> CheckpointReader<'o, 'm, O, M> {
    pub fn optimizer<'o2, O2>(self, optimizer: &'o2 mut O2) -> CheckpointReader<'o2, 'm, O2, M> {
        CheckpointReader {
            parameters: self.parameters,
            optimizer: Some(optimizer),
            meta: self.meta
        }
    }
    pub fn meta<'m2, M2>(self, meta: &'m2 mut M2) -> CheckpointReader<'o, 'm2, O, M2> {
        CheckpointReader {
            parameters: self.parameters,
            optimizer: self.optimizer,
            meta: Some(meta)
        }
    }
}

impl<'o, 'm, O: DeserializeOwned, M: DeserializeOwned> CheckpointReader<'o, 'm, O, M> {
    pub fn read_from(&mut self, mut reader: impl io::Read) -> Result<(), Box<dyn Error>> {
        if let Some(meta) = &mut self.meta {
            **meta = bincode::deserialize_from(&mut reader)?;
        }
        if let Some(optimizer) = &mut self.optimizer {
            **optimizer = bincode::deserialize_from(&mut reader)?;
        }
        let with_optimizer_data = self.optimizer.is_some();
        for parameter in &self.parameters {
            let saved = bincode::deserialize_from(&mut reader)?;
            parameter.load(saved, with_optimizer_data);
        }
        Ok(())
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
            self.optimizer_data.read()
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
            let mut optimizer_data = self.optimizer_data.write()
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
        let value = self.value.read()
            .unwrap()
            .as_slice()
            .into_owned();
        let meta = self.meta.to_saved(with_optimizer_data);
        SavedParameter {
            value,
            meta
        }
    }
    fn load(&self, saved: SavedParameter, with_optimizer_data: bool) {
        let mut value = self.value.write()
            .unwrap();
        value.copy_from_slice(saved.value);
        self.meta.load(value.device(), value.raw_dim().into_dyn(), saved.meta, with_optimizer_data);
    }
}
