use crate::Unsigned;
use argmm::ArgMinMax;
use ndarray::{ArrayView1, ArrayView2};

pub fn classification_accuracy<U: Unsigned>(
    pred: &ArrayView2<f32>,
    labels: &ArrayView1<U>,
) -> usize {
    let (batch_size, nclasses) = pred.dim();
    debug_assert_eq!(labels.dim(), batch_size);
    let mut correct = 0;
    pred.outer_iter().zip(labels.iter()).for_each(|(y, &c)| {
        if y.as_slice().unwrap().argmax() == Some(c.to_usize().unwrap()) {
            correct += 1;
        }
    });
    correct
}
