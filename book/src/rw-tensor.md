# RwTensor

```
pub type RwTensor<T, D> = TensorBase<RwRepr<T>, D>;
```
A RwTensor is like an Arc<RwLock<Vec<T>>>. Cloning the RwTensor copies the pointer, like ArcTensor. Unlike ArcTensor, the data is wrapped in a RwLock. This allows either shared immutable access with the read() method (returning a RwReadTensor), or exclusive access with the write() method (returning a RwWriteTensor). 

## RwReadTensor

```
pub type RwReadTensor<'a, T, D> = TensorBase<RwReadRepr<'a, T>, D>;
```
A RwReadTensor is like a RwLockReadGuard<Vec<T>>. On creation, the RwLock is locked in read mode, which blocks until a writer has released the lock, and then blocks any writers from obtaining the lock. A RwReadTensor can be borrowed as a TensorView with the view() method, and many methods are implemented generically over both. This is safe because the view will borrow the RwReadTensor, which holds the lock.

## RwWWriteTensor

```
pub type RwWriteTensor<'a, T, D> = TensorBase<RwWriteRepr<'a, T>, D>;
```
A RwReadTensor is like a RwLockWriteGuard<Vec<T>>. On creation, the RwLock is locked in write mode, which blocks until all readers and writers have released the lock, and then blocks any readers or writers from obtaining the lock. A RwWriteTensor can be borrowed as a TensorViewMut with the view_mut() method, and many methods are implemented generically over both. This is safe because the view will borrow the RwWriteTensor, which holds the lock.
