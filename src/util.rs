use std::{any::TypeId, mem::size_of};

pub(crate) fn type_eq<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

pub(crate) fn size_eq<A, B>() -> bool {
    size_of::<A>() == size_of::<B>()
}
