//! Driver-only CUDA host stack shared by speakrs and scriptrs
//!
//! cudakit loads the NVIDIA driver (`libcuda`) at run time through cudarc and never
//! loads the CUDA Runtime (`libcudart`), so a build needs no CUDA toolkit and a
//! process needs only the driver. With the `libraries` feature, and only when the
//! consumer's [`Profile`] allows it, cuBLAS, cuDNN and NVRTC are loaded lazily after a
//! symbol preflight.
//!
//! - [`Gpu`] owns one context and one stream, the device attributes and the optional
//!   library handles
//! - [`Runtime`] adds a module cache for a consumer's [`KernelArea`]s: embedded PTX and
//!   exact-architecture cubins, loaded by exact identity ([`ModuleRequest`])
//! - [`DeviceTensor`] is a shaped device buffer
//! - [`Sgemm`] and [`SgemmBatched`] describe row-major cuBLAS products
//! - [`Gpu::capture`] records stream work as a replayable [`Graph`]
//! - [`dnn`] plans cuDNN convolutions (feature `libraries`)
//! - [`identity`] reads the driver release for persisted tuning identities
//!
//! Kernels are written with cuda-oxide in each consumer's own kernels crate and built
//! offline by `cudakit-build`; see the repository's `HANDOFF.md`

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
pub use graph::{EventTrackingPause, Graph};
#[cfg(feature = "libraries")]
pub use libraries::LibraryVersions;
pub use profile::{ExtraSymbols, LibraryPolicy, Profile};
pub use runtime::{LoadedKernels, Options, Runtime};
pub use tensor::{DeviceTensor, unfold_windows};
pub use tier::{ComputeCapability, PtxTier, TierSet};

/// cudarc, re-exported so consumers use the exact version cudakit was built against
pub use cudarc;
