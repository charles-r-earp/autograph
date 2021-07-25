use std::{any::TypeId, mem::size_of};

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
