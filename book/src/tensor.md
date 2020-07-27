# Tensor

```
pub type Tensor<T, D> = TensorBase<OwnedRepr<T>, D>;
```
A Tensor owns its data exclusively. Most tensor operations that return an output will return a Tensor. When a tensor is cloned, its data will be copied into a new tensor. 
