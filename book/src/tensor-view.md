# TensorView

```
pub type TensorView<'a, T, D> = TensorBase<ViewRepr<&'a Buffer<T>>, D>;
```
A TensorView is like a &`[T]`, it represents a borrowed tensor. Tensors with S: DataRef can be borrowed as views with the view() method. 

# TensorViewMut

```
pub type TensorViewMut<'a, T, D> = TensorBase<ViewRepr<&'a mut Buffer<T>>, D>;
```
A TensorViewMut is lke a &`[T]`, it represents a mutably borrowed tensor. Tensors with S: DataMut can be borrowed as mutable views with the view_mut() method.
```
