# ArcTensor

```
pub type ArcTensor<T, D> = TensorBase<ArcRepr<T>, D>;
```
An ArcTensor is like Arc<Vec<T>>. Cloning an ArcTensor copies the pointer, sharing the data. The data is dropped when the last ArcTensor pointing to it is dropped.
