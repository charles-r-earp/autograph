use bytemuck::Pod;
use half::{bf16, f16};
use num_traits::{FromPrimitive, ToPrimitive};
use std::fmt::Debug;

mod sealed {
    use half::{bf16, f16};

    #[doc(hidden)]
    pub trait Sealed {}

    macro_rules! impl_sealed {
        ($($t:ty),+) => {
            $(
                impl Sealed for $t {}
            )+
        };
    }

    impl_sealed! {u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64}
}
use sealed::Sealed;

/// Numerical types supported in autograph.
#[allow(missing_docs)]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarType {
    U8,
    I8,
    U16,
    I16,
    F16,
    BF16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
}

/// Extends ToPrimitive to include half types.
pub trait ToPrimitiveExt: ToPrimitive {
    /// Converts to bf16.
    fn to_f16(&self) -> Option<f16> {
        self.to_f32().map(f16::from_f32)
    }
    /// Converts to bf16.
    fn to_bf16(&self) -> Option<bf16> {
        self.to_f32().map(bf16::from_f32)
    }
}

/// Extends FromPrimitive to include half types.
pub trait FromPrimitiveExt: FromPrimitive {
    /// Converts from f16.
    fn from_f16(n: f16) -> Option<Self> {
        Self::from_f32(n.into())
    }
    /// Converts from bf16.
    fn from_bf16(n: bf16) -> Option<Self> {
        Self::from_f32(n.into())
    }
}

macro_rules! impl_primitive_ext {
    ($($t:ty),+) => {
        $(
            impl ToPrimitiveExt for $t {}
            impl FromPrimitiveExt for $t {}
        )+
    };
}

impl_primitive_ext! {u8, i8, u16, i16, u32, i32, f32, u64, i64, f64}

impl ToPrimitiveExt for f16 {
    fn to_f16(&self) -> Option<Self> {
        Some(*self)
    }
}

impl ToPrimitiveExt for bf16 {
    fn to_bf16(&self) -> Option<Self> {
        Some(*self)
    }
}

impl FromPrimitiveExt for f16 {
    fn from_f16(n: f16) -> Option<Self> {
        Some(n)
    }
}

impl FromPrimitiveExt for bf16 {
    fn from_bf16(n: bf16) -> Option<Self> {
        Some(n)
    }
}

/// Types with a `0` value.
pub trait Zero: Default + Sealed {
    /// Returns 0.
    fn zero() -> Self {
        Self::default()
    }
}

macro_rules! impl_zero {
    ($($t:ty),+) => {
        $(
            impl Zero for $t {}
        )+
    }
}

impl_zero! {u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64}

/// Types with a `1` value.
pub trait One: Sealed {
    /// Returns `1`.
    fn one() -> Self;
}

macro_rules! impl_one {
    (@Int $($t:ty),+) => {
        $(
            impl One for $t {
                fn one() -> Self {
                    1
                }
            }
        )+
    };
    (@Half $($t:ty),+) => {
        $(
            impl One for $t {
                fn one() -> Self {
                    <$t>::ONE
                }
            }
        )+
    };
    (@Float $($t:ty),+) => {
        $(
            impl One for $t {
                fn one() -> Self {
                    1.
                }
            }
        )+
    }
}

impl_one! {@Int u8, i8, u16, i16, u32, i32, u64, i64}
impl_one! {@Half f16, bf16}
impl_one! {@Float f32, f64}

/// Named scalar.
///
/// For example, u32 is "u32", bf16 is "bf16", etc.
pub trait ScalarName: Sealed {
    /// Returns the type name.
    fn scalar_name() -> &'static str;
}

macro_rules! impl_scalar_name {
    ($($t:ty),+) => {
        $(
            impl ScalarName for $t {
                fn scalar_name() -> &'static str {
                    stringify!($t)
                }
            }
        )+
    };
}

impl_scalar_name! {u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64}

/// Base trait for numerical types supported in autograph.
pub trait Scalar:
    Zero + One + ScalarName + Pod + ToPrimitiveExt + FromPrimitiveExt + Debug + PartialEq + Sealed
{
    /// The [`ScalarType`] of the scalar.
    fn scalar_type() -> ScalarType;
}

impl Scalar for u8 {
    fn scalar_type() -> ScalarType {
        ScalarType::U8
    }
}

impl Scalar for i8 {
    fn scalar_type() -> ScalarType {
        ScalarType::I8
    }
}

impl Scalar for u16 {
    fn scalar_type() -> ScalarType {
        ScalarType::U16
    }
}

impl Scalar for i16 {
    fn scalar_type() -> ScalarType {
        ScalarType::I16
    }
}

impl Scalar for f16 {
    fn scalar_type() -> ScalarType {
        ScalarType::F16
    }
}

impl Scalar for bf16 {
    fn scalar_type() -> ScalarType {
        ScalarType::BF16
    }
}

impl Scalar for u32 {
    fn scalar_type() -> ScalarType {
        ScalarType::U32
    }
}

impl Scalar for i32 {
    fn scalar_type() -> ScalarType {
        ScalarType::I32
    }
}

impl Scalar for f32 {
    fn scalar_type() -> ScalarType {
        ScalarType::F32
    }
}

impl Scalar for u64 {
    fn scalar_type() -> ScalarType {
        ScalarType::U64
    }
}

impl Scalar for i64 {
    fn scalar_type() -> ScalarType {
        ScalarType::I64
    }
}

impl Scalar for f64 {
    fn scalar_type() -> ScalarType {
        ScalarType::F64
    }
}
