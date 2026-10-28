//! Library entry points that cudakit checks before cudarc loads a library
//!
//! Each list covers the functions reached by cudakit's own wrappers and the cudarc
//! code they call, including handle and descriptor destructors and version queries.
//! A consumer that calls more functions directly declares them in
//! [`ExtraSymbols`](crate::ExtraSymbols)

use std::ffi::CStr;

/// cuBLAS symbols reached by [`Sgemm`](crate::Sgemm), [`SgemmBatched`](crate::SgemmBatched)
/// and library version queries
pub const CUBLAS: &[&CStr] = &[
    c"cublasCreate_v2",
    c"cublasDestroy_v2",
    c"cublasSetStream_v2",
    c"cublasSetMathMode",
    c"cublasSgemm_v2",
    c"cublasSgemmStridedBatched",
    c"cublasGetVersion_v2",
    c"cublasGetCudartVersion",
];

/// cuDNN symbols reached by the handle, the convolution planner and version queries
pub const CUDNN: &[&CStr] = &[
    c"cudnnCreate",
    c"cudnnDestroy",
    c"cudnnSetStream",
    c"cudnnGetVersion",
    c"cudnnCreateTensorDescriptor",
    c"cudnnDestroyTensorDescriptor",
    c"cudnnSetTensor4dDescriptor",
    c"cudnnCreateConvolutionDescriptor",
    c"cudnnDestroyConvolutionDescriptor",
    c"cudnnSetConvolution2dDescriptor",
    c"cudnnSetConvolutionMathType",
    c"cudnnCreateFilterDescriptor",
    c"cudnnDestroyFilterDescriptor",
    c"cudnnSetFilter4dDescriptor",
    c"cudnnGetConvolutionForwardAlgorithm_v7",
    c"cudnnGetConvolutionForwardWorkspaceSize",
    c"cudnnConvolutionForward",
    c"cudnnConvolutionBiasActivationForward",
    c"cudnnCreateActivationDescriptor",
    c"cudnnDestroyActivationDescriptor",
    c"cudnnSetActivationDescriptor",
];

/// NVRTC symbols that cuDNN's runtime-compiled kernels use; cuDNN compiles device
/// code itself, not through a Rust wrapper, so its compilation, diagnostic and
/// artifact APIs are checked before cuDNN builds anything
pub const NVRTC: &[&CStr] = &[
    c"nvrtcVersion",
    c"nvrtcGetErrorString",
    c"nvrtcGetNumSupportedArchs",
    c"nvrtcGetSupportedArchs",
    c"nvrtcCreateProgram",
    c"nvrtcDestroyProgram",
    c"nvrtcCompileProgram",
    c"nvrtcGetProgramLogSize",
    c"nvrtcGetProgramLog",
    c"nvrtcGetPTXSize",
    c"nvrtcGetPTX",
    c"nvrtcGetCUBINSize",
    c"nvrtcGetCUBIN",
    c"nvrtcAddNameExpression",
    c"nvrtcGetLoweredName",
    c"nvrtcGetLTOIRSize",
    c"nvrtcGetLTOIR",
];
