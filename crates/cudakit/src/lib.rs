//! Driver-only CUDA host stack shared by speakrs and scriptrs
//!
//! cudakit loads the NVIDIA driver (`libcuda`) at run time through cudarc and never
//! loads the CUDA Runtime (`libcudart`), so a build needs no CUDA toolkit and a
//! process needs only the driver. With the `libraries` feature, and only when the
//! consumer's [`Profile`] allows it, cuBLAS, cuBLASLt, cuDNN and NVRTC are loaded
//! lazily, and used only after a symbol preflight of the loaded library.
//!
//! It runs on Linux and Windows x86_64. On Windows the driver is `nvcuda.dll` from
//! System32, and the libraries are CUDA 12's or 13's `cublas64_*.dll`,
//! `cublasLt64_*.dll` and `nvrtc64_*0_0.dll` and cuDNN 9's `cudnn64_9.dll`, found in an absolute `PATH` entry
//! or `%CUDA_PATH%\bin\x64` and `%CUDA_PATH%\bin`, never in the current directory.
//! CI builds and tests the Windows build, but it hasn't run on a GPU yet.
//!
//! - [`Gpu`] owns one context and one stream, the device attributes and the optional
//!   library handles
//! - [`Runtime`] adds a module cache for a consumer's [`KernelArea`]s: embedded PTX and
//!   exact-architecture cubins, loaded by exact identity ([`ModuleRequest`])
//! - [`DeviceTensor`] is a shaped device buffer
//! - [`Sgemm`] and [`SgemmBatched`] describe row-major cuBLAS products
//! - `MatmulPlan` runs a row-major cuBLASLt product with a fixed, persistable
//!   algorithm, FP32, FP16 or BF16 operands and a fused bias, ReLU or GELU epilogue;
//!   `Gpu::tune_matmul` times the heuristic's top candidates and `MatmulCache` keeps
//!   one plan per shape (feature `libraries`)
//! - [`Gpu::capture`] records stream work as a replayable [`Graph`]
//! - `dnn` plans cuDNN convolutions (feature `libraries`)
//! - [`identity`] reads the driver release for persisted tuning identities
//!
//! Kernels are written with cuda-oxide in each consumer's own kernels crate and built
//! offline by `cudakit-build`; see the [guide](https://github.com/avencera/cudakit/blob/master/docs/guide.md)

#![cfg_attr(docsrs, feature(doc_cfg))]

mod artifact;
mod conv;
mod device;
#[cfg(feature = "libraries")]
pub mod dnn;
mod error;
mod gemm;
mod gpu;
mod graph;
pub mod identity;
#[cfg(feature = "libraries")]
mod libraries;
#[cfg(feature = "libraries")]
mod matmul;
mod profile;
mod runtime;
pub mod symbols;
mod tensor;
mod tier;

pub use artifact::{
    AreaPtx, ArtifactHash, ArtifactLoadError, EmbeddedCubin, EmbeddedPtx, KernelArea,
    LoadedArtifact, ModuleRequest, load_artifact,
};
pub use conv::Conv2d;
pub use device::DeviceAttributes;
#[cfg(feature = "test-support")]
pub use device::DeviceAttributesBuilder;
pub use error::{Error, Library, check_len, element_count, to_c_int};
pub use gemm::{Math, Sgemm, SgemmBatched};
pub use gpu::Gpu;
pub use graph::Graph;
#[cfg(feature = "libraries")]
pub use libraries::{BlasGuard, LibraryVersions, MATMUL_WORKSPACE_BYTES};
#[cfg(feature = "libraries")]
pub use matmul::{
    AlgoConfig, CandidateTiming, Epilogue, MATMUL_ALIGNMENT, Matmul, MatmulAlgo, MatmulCache,
    MatmulCandidate, MatmulElement, MatmulPlan, Operands, Output, Scale, TuneReport, Tuning,
};
pub use profile::{ExtraSymbols, LibraryPolicy, Profile};
pub use runtime::{LoadedKernels, Options, Runtime};
pub use tensor::{DeviceTensor, unfold_windows};
pub use tier::{ComputeCapability, PtxTier, TierSet};

/// cudarc, re-exported so consumers use the exact version cudakit was built against
pub use cudarc;
