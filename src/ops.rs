use crate::result::Result;

pub(crate) trait Add<R>: Sized {
    type Output;
    fn add(self, rhs: R) -> Result<Self::Output>;
}

pub(crate) trait Assign<R> {
    fn assign(&mut self, rhs: R) -> Result<()>;
}

pub(crate) trait AddAssign<R>: Add<R> + Assign<R> {
    fn add_assign(&mut self, rhs: R) -> Result<()>;
}

pub(crate) trait ScaledAdd<T, R>: AddAssign<R> {
    fn scaled_add(&mut self, alpha: T, rhs: R) -> Result<()>;
}
