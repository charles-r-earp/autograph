use bytemuck::Pod;
use half::{bf16, f16};
use num_traits::{FromPrimitive, Num, NumCast, ToPrimitive};
use std::fmt::{Debug, Display};

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
    Default
    + Num
    + NumCast
    + FromPrimitiveExt
    + ToPrimitiveExt
    + ScalarName
    + Pod
    + Debug
    + Display
    + PartialEq
    + Sealed
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

/// Float types.
#[allow(missing_docs)]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FloatType {
    BF16,
    F32,
}

impl FloatType {
    /// Returns the type name (ie "f32").
    pub fn as_str(&self) -> &'static str {
        use FloatType::*;
        match self {
            BF16 => "bf16",
            F32 => "f32",
        }
    }
}

pub(crate) trait AsFloat<T: Float> {
    fn as_(self) -> T;
}

impl AsFloat<bf16> for bf16 {
    fn as_(self) -> Self {
        self
    }
}

impl AsFloat<f32> for bf16 {
    fn as_(self) -> f32 {
        self.to_f32()
    }
}

impl AsFloat<bf16> for f32 {
    fn as_(self) -> bf16 {
        bf16::from_f32(self)
    }
}

impl AsFloat<f32> for f32 {
    fn as_(self) -> Self {
        self
    }
}

/// Base trait for float scalar types.
pub trait Float: Scalar {
    /// The float type.
    fn float_type() -> FloatType;
}

impl Float for bf16 {
    fn float_type() -> FloatType {
        FloatType::BF16
    }
}

impl Float for f32 {
    fn float_type() -> FloatType {
        FloatType::F32
    }
}

/// Marker trait for Uint types
pub trait Uint: Scalar + Into<u32> {}

impl Uint for u8 {}

impl Uint for u16 {}

impl Uint for u32 {}
