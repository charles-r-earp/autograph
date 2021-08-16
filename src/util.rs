use std::{
    any::TypeId,
    hash::{BuildHasher, Hasher},
    hint::unreachable_unchecked,
    mem::size_of,
};

pub(crate) fn elem_type_name<T>() -> &'static str {
    let name = std::any::type_name::<T>();
    name.strip_prefix("half::binary16::")
        .or_else(|| name.strip_prefix("half::bfloat::"))
        .unwrap_or(name)
}

pub(crate) fn type_eq<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

pub(crate) fn size_eq<A, B>() -> bool {
    size_of::<A>() == size_of::<B>()
}

pub(crate) trait UnwrapUnchecked {
    type Output;
    unsafe fn _unwrap_unchecked(self) -> Self::Output;
}

impl<T> UnwrapUnchecked for Option<T> {
    type Output = T;
    unsafe fn _unwrap_unchecked(self) -> T {
        debug_assert!(self.is_some());
        match self {
            Some(x) => x,
            None => unreachable_unchecked(),
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub(crate) struct U32Hasher(u32);

impl Hasher for U32Hasher {
    fn finish(&self) -> u64 {
        self.0 as u64
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!()
    }
    fn write_u32(&mut self, i: u32) {
        self.0 = i;
    }
}

impl BuildHasher for U32Hasher {
    type Hasher = Self;
    fn build_hasher(&self) -> Self {
        *self
    }
}

#[cfg(test)]
mod tests {
    use half::f16;

    #[test]
    fn type_eq() {
        assert!(super::type_eq::<f32, f32>());
        assert!(!super::type_eq::<f32, u32>());
    }

    #[test]
    fn size_eq() {
        assert!(super::size_eq::<f16, u16>());
        assert!(!super::size_eq::<f64, u32>());
    }
}
