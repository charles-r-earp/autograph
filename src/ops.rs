use crate::result::Result;
#[cfg(feature = "tensor")]
use ndarray::Dimension;

/*
pub(crate) trait Add<R>: Sized {
    type Output;
    fn add(self, rhs: R) -> Result<Self::Output>;
}

pub(crate) trait Assign<R> {
    fn assign(&mut self, rhs: R) -> Result<()>;
}*/

pub(crate) trait AddAssign<R> /*: Add<R> + Assign<R>*/ {
    fn add_assign(&mut self, rhs: &R) -> Result<()>;
}

pub(crate) trait ScaledAdd<T, R>: AddAssign<R> {
    fn scaled_add(&mut self, alpha: T, rhs: &R) -> Result<()>;
}

#[cfg(feature = "tensor")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(unused)]
pub(crate) enum KernelKind {
    Convolution,
    CrossCorrelation,
}

#[cfg(feature = "tensor")]
impl KernelKind {
    #[allow(unused)]
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Convolution => "convolution",
            Self::CrossCorrelation => "cross_correlation",
        }
    }
}

#[cfg(feature = "tensor")]
#[derive(Debug)]
pub(crate) struct KernelArgs<D: Dimension> {
    pub(crate) strides: D,
    pub(crate) padding: D,
    pub(crate) dilation: D,
}

#[cfg(feature = "ndarray")]
pub(crate) trait Im2Col<D: Dimension> {
    type Output;
    fn im2col(&self, kernel: &D, kind: KernelKind, args: &KernelArgs<D>) -> Result<Self::Output>;
}
