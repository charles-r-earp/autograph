# TensorBase

A tensor is a mathmatically generalization of a vector in n dimensions. Typically this means storing a vector of data and a vector of dimensions. 

```
#[derive(Clone)]
pub struct TensorBase<S: Data, D: Dimension> { ... }
```
Similar to ndarray's ArrayBase, TensorBase<S, D> is generic over the storage S and dimension D. S is required to implement the Data trait, which is sealed (not able to be implemented outside the crate):
```
/// Main trait for Tensor S generic parameter, similar to ndarray::Data. Elem indicates that the Tensor stores that datatype.
pub trait Data: PrivateData {
    type Elem: Num;
}
```
The [Dimension](https://docs.rs/ndarray/0.13.1/ndarray/trait.Dimension.html) trait looks like this:
```
pub trait Dimension: Clone + Eq + Debug + Send + Sync + Default + IndexMut<usize, Output = usize> + Add<Self, Output = Self> + AddAssign + for<'x> AddAssign<&'x Self> + Sub<Self, Output = Self> + SubAssign + for<'x> SubAssign<&'x Self> + Mul<usize, Output = Self> + Mul<Self, Output = Self> + MulAssign + for<'x> MulAssign<&'x Self> + MulAssign<usize> {
    type SliceArg: ?Sized + AsRef<[SliceOrIndex]>;
    type Pattern: IntoDimension<Dim = Self> + Clone + Debug + PartialEq + Eq + Default;
    type Smaller: Dimension;
    type Larger: Dimension + RemoveAxis;

    const NDIM: Option<usize>;

    fn ndim(&self) -> usize;
    fn into_pattern(self) -> Self::Pattern;
    fn zeros(ndim: usize) -> Self;
    fn __private__(&self) -> PrivateMarker;

    fn size(&self) -> usize { ... }
    fn size_checked(&self) -> Option<usize> { ... }
    fn as_array_view(&self) -> ArrayView1<Ix> { ... }
    fn as_array_view_mut(&mut self) -> ArrayViewMut1<Ix> { ... }
    fn into_dyn(self) -> IxDyn { ... }
}
```
Dimensions:
    - Ix0 (scalar)
    - Ix1 (vector)
    - Ix2 (matrix)
    - Ix3 (3d tensor)
    - Ix4 (4d tensor ie 2d image)
    - Ix5 (5d tensor ie 3d image)
    - Ix6 (6d tensor)
    - IxDyn (Nd tensor)
```
Rather than constructing a Dimension directly, most functions in autograph use the [IntoDimension](https://docs.rs/ndarray/0.13.1/ndarray/trait.IntoDimension.html) trait:
```
pub trait IntoDimension {
    type Dim: Dimension;
    fn into_dimension(self) -> Self::Dim;
}
```
This is implemented in ndarray for several types:
    - usize -> Ix0
    - (usize, usize) | `[usize; 2]` -> Ix2
    - (usize, usize, usize) | `[usize; 3]` -> Ix3
    - (usize, usize, usize, usize) | `[usize; 4]` -> Ix4
    - (usize, usize, usize, usize, usize) | `[usize; 5]` -> Ix5
    - (usize, usize, usize, usize, usize, usize) | `[usize; 6]` -> Ix6
    - &`[usize]` -> IxDyn
```

Similar to Array1, Array2, ArrayD, there are type aliases for Tensors as well, like Tensor1, Tensor2, TensorD, or TensorView1, TensorView2, TensorViewD etc.

## DataOwned

The DataOwned trait is a subtrait of Data for tensors that own their data (ie without a borrow / lifetime). Implemented by:
    - Tensor
    - ArcTensor
    - RwTensor

This is used for all constructors, like:
```
let x1 = Tensor::zeros(&device, 1);
let x2 = ArcTensor::zeros(&device, 1);
```

### Cows 

[Cow](https://doc.rust-lang.org/nightly/alloc/borrow/enum.Cow.html) (Copy On Write) is a "Smart Pointer" that is an enum, representing either Owned or Borrowed data. If the data is owned, it can be moved into the new container (ie a cpu Tensor). For gpu tensors, the data must be copied to gpu memory, so there is no advantage to passing an owned Vec over a slice. Often, data is not owned but borrowed. Instead of forcing the user to copy the data before calling the constructor, the user can provide the slice instead, allowing the implementation to choose when to copy the data, avoiding a second copy from the Vec to gpu memory. 

The primary way to contruct a Tensor is with `TensorBase::from_shape_vec`:
```
/// Constructs a Tensor on the device with the given shape. If a Vec is provided, will move the data (ie no copy) if the device is a cpu. Can also provide a slice, which allows for the data to only be copied once (rather than twice in the case of copying to the gpu).
    /// Panics: Asserts that the provided shape matches the length of the provided vec or slice.
    ///
pub fn from_shape_vec<'a>(
    device: &Device,
    shape: impl IntoDimension<Dim = D>,
    vec: impl Into<Cow<'a, [T]>>,
) -> Self;
```
Both `Vec<T>` and `&[T]` implement `Into<Cow<[T]>`, so this method can be called with a slice as well. The same is true for `TensorBase::from_array`:
```
/// Similar to from_shape_vec, can accept either an Array or an ArrayView
    /// Copies that data to standard (packed) layout if necessary
pub fn from_array<'a>(device: &Device, array: impl Into<CowArray<'a, T, D>>) -> Self;
```

## DataRef

The DataRef trait is a subtrait of Data for tensors that can borrow their data, ie they can return a TensorView. Implemented by:
    - Tensor
    - ArcTensor
    - TensorView
    - TensorViewMut
    - RwReadTensor
    - RwWriteTensor

Generally tensor operations will require their inputs to be DataRef. This ensures that the data is not modified while executing. 

## DataMut 

Like DataRef, but allows mutable borrows. Implemented by:
    - Tensor
    - TensorViewMut
    - RwWriteTensor

