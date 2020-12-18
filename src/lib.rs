pub mod backend;
pub mod error;
pub mod tensor;

pub use ndarray;

pub type Result<T, E = error::Error> = std::result::Result<T, E>;

#[macro_export]
macro_rules! include_spirv {
    ($file:expr) => {{
        #[repr(C)]
        pub struct AlignedAs<Align, Bytes: ?Sized> {
            pub _align: [Align; 0],
            pub bytes: Bytes,
        }

        static ALIGNED: &AlignedAs<u32, [u8]> = &AlignedAs {
            _align: [],
            bytes: *include_bytes!($file),
        };

        &ALIGNED.bytes
    }};
}
