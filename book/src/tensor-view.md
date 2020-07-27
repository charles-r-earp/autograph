# TensorView

```
pub type TensorView<'a, T, D> = TensorBase<ViewRepr<&'a Buffer<T>>, D>;
```
A TensorView is like a &`[T]`, it represents a borrowed tensor. Tensors with S: DataRef can be borrowed as views.  
