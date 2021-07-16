use std::future::Future;

/// Auto trait that converts async code to blocking code
pub trait BlockingFuture: Future {
    /// Blocks on the future
    fn block(self) -> Self::Output;
}

impl<F: Future> BlockingFuture for F {
    fn block(self) -> Self::Output {
        smol::block_on(self)
    }
}
