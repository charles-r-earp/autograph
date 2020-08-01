use std::{fmt::{self, Debug, Display}, error::Error};
use super::{cudaError_t, cublasStatus_t, cudnnStatus_t};

pub(super) enum CudaError {
    CudaError(cudaError_t),
    CublasStatus(cublasStatus_t),
    CudnnStatus(cudnnStatus_t)
}

impl From<cudaError_t> for CudaError {
    fn from(e: cudaError_t) -> Self {
        CudaError::CudaError(e)
    }
}

impl From<cublasStatus_t> for CudaError {
    fn from(s: cublasStatus_t) -> Self {
        CudaError::CublasStatus(s)
    }
}

impl From<cudnnStatus_t> for CudaError {
    fn from(s: cudnnStatus_t) -> Self {
        CudaError::CudnnStatus(s)
    }
}

impl Debug for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CudaError::CudaError(e) => e.fmt(f),
            CudaError::CublasStatus(s) => s.fmt(f),
            CudaError::CudnnStatus(s) => s.fmt(f)
        }
    }
}

impl Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self, f)
    }
}

impl Error for CudaError {
    fn description(&self) -> &'static str {
        "CudaError"
    }
}

pub(super) type CudaResult<T> = Result<T, CudaError>;

pub(super) trait IntoResult {
    type Error: Error;
    fn into_result(self) -> Result<(), Self::Error>;
}

impl IntoResult for cudaError_t {
    type Error = CudaError;
    fn into_result(self) -> Result<(), Self::Error> {
        match self {
            cudaError_t::CUDA_SUCCESS => Ok(()),
            _ => Err(self.into())
        }
    }
}

impl IntoResult for cublasStatus_t {
    type Error = CudaError;
    fn into_result(self) -> Result<(), Self::Error> {
        match self {
            cublasStatus_t::SUCCESS => Ok(()),
            _ => Err(self.into())
        }
    }
}

impl IntoResult for cudnnStatus_t {
    type Error = CudaError;
    fn into_result(self) -> Result<(), Self::Error> {
        match self {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            _ => Err(self.into())
        }
    }
}
