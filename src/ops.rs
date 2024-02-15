use anyhow::Result;

#[cfg(feature = "neural-network")]
#[doc(hidden)]
pub mod __private {
    use anyhow::Result;

    #[derive(Clone)]
    pub struct Im2ColConv2Options {
        pub filter: [usize; 2],
        pub padding: [usize; 2],
        pub stride: [usize; 2],
        pub dilation: [usize; 2],
    }

    impl Default for Im2ColConv2Options {
        fn default() -> Self {
            Self {
                filter: [0, 0],
                padding: [0, 0],
                stride: [1, 1],
                dilation: [1, 1],
            }
        }
    }

    impl Im2ColConv2Options {
        pub fn output_shape(&self, input_shape: [usize; 2]) -> [usize; 2] {
            let mut shape = input_shape;
            for ((a, f), (s, (p, d))) in shape.iter_mut().zip(self.filter).zip(
                self.stride
                    .into_iter()
                    .zip(self.padding.into_iter().zip(self.dilation)),
            ) {
                *a = (*a + 2 * p - d * (f - 1) - 1) / s + 1;
            }
            shape
        }
    }

    pub trait Im2ColConv2 {
        type Output;
        fn im2col_conv2(&self, options: &Im2ColConv2Options) -> Result<Self::Output>;
    }

    #[derive(Clone)]
    pub struct Col2ImConv2Options {
        pub shape: [usize; 2],
        pub filter: [usize; 2],
        pub padding: [usize; 2],
        pub stride: [usize; 2],
        pub dilation: [usize; 2],
    }

    impl Default for Col2ImConv2Options {
        fn default() -> Self {
            Self {
                shape: [0, 0],
                filter: [0, 0],
                padding: [0, 0],
                stride: [1, 1],
                dilation: [1, 1],
            }
        }
    }

    impl Col2ImConv2Options {
        pub fn output_shape(&self) -> [usize; 2] {
            let mut shape = self.shape;
            for ((a, f), (s, (p, d))) in shape.iter_mut().zip(self.filter).zip(
                self.stride
                    .into_iter()
                    .zip(self.padding.into_iter().zip(self.dilation)),
            ) {
                *a = (*a - 1) * s + d * (f - 1) + 1 - (2 * p);
            }
            shape
        }
    }

    pub trait Col2ImConv2 {
        type Output;
        fn col2im_conv2(&self, options: &Col2ImConv2Options) -> Result<Self::Output>;
    }

    #[derive(Clone)]
    pub(crate) struct MaxPool2Options {
        pub(crate) size: [usize; 2],
        pub(crate) strides: [usize; 2],
    }

    impl MaxPool2Options {
        pub(crate) fn output_shape(&self, input_shape: [usize; 2]) -> [usize; 2] {
            let mut shape = input_shape;
            for (a, (x, s)) in shape
                .iter_mut()
                .zip(self.size.into_iter().zip(self.strides))
            {
                *a = (*a - x) / s + 1;
            }
            shape
        }
    }

    pub(crate) trait MaxPool2 {
        type Output;
        fn max_pool2(&self, options: MaxPool2Options) -> Result<Self::Output>;
    }

    pub(crate) trait MaxPool2Backward<DY> {
        fn max_pool2_backward(&mut self, output_grad: DY, options: MaxPool2Options) -> Result<()>;
    }
}
#[cfg(feature = "neural-network")]
pub(crate) use __private::*;

/// AddAssign that returns a [`Result`].
pub trait AddAssign<R> {
    /// Performs the operation `self += rhs`.
    fn add_assign(&mut self, rhs: R) -> Result<()>;
}
