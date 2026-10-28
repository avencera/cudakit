use std::fmt;

#[cfg(feature = "libraries")]
use cudarc::cublas::result::CublasError;
#[cfg(feature = "libraries")]
use cudarc::cudnn::CudnnError;
use cudarc::driver::DriverError;

use crate::artifact::LoadedArtifact;
use crate::tier::{ComputeCapability, PtxTier};

/// A CUDA shared library that cudakit loads at run time
///
/// cudakit never loads the CUDA Runtime (`libcudart`); the driver is the only library
/// every process needs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Library {
    /// The driver API, `libcuda`, installed with the NVIDIA driver
    Driver,
    /// cuBLAS
    Cublas,
    /// cuDNN 9
    Cudnn,
    /// Runtime compilation, which cuDNN uses for some dynamic kernels
    Nvrtc,
}

impl fmt::Display for Library {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Driver => "CUDA driver (libcuda)",
            Self::Cublas => "cuBLAS",
            Self::Cudnn => "cuDNN",
            Self::Nvrtc => "NVRTC",
        })
    }
}

/// Errors from cudakit
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A CUDA shared library is absent or lacks a required symbol
    #[error("could not load the {library} shared library")]
    LibraryUnavailable {
        /// The library that failed to load
        library: Library,
    },
    /// The consumer's driver-only policy forbids an optional library
    #[error("driver-only CUDA prohibits a direct request for {library}")]
    LibraryForbidden {
        /// The requested library
        library: Library,
    },
    /// A library was used before it was prepared
    #[error("{library} was not prepared before use")]
    LibraryUnprepared {
        /// The library that was not prepared
        library: Library,
    },
    /// A thread-affine library handle was requested on a thread other than the one
    /// that prepared it
    #[error("{library} was prepared on another thread and may be used only there")]
    LibraryOnAnotherThread {
        /// The library whose handle is bound to another thread
        library: Library,
    },
    /// A required PTX tier is absent from this build
    #[error("PTX tier {tier} for device {device} is not compiled in; enable `{feature}`")]
    TierNotCompiledIn {
        /// Required PTX tier
        tier: PtxTier,
        /// Exact device capability
        device: ComputeCapability,
        /// Consumer Cargo feature that embeds the tier
        feature: &'static str,
    },
    /// An area has no embedded variant that the device can run within the tier limit
    #[error(
        "CUDA area {area} has no runnable PTX at tier {tier} for device {device}; enable `{feature}`"
    )]
    AreaTierNotCompiledIn {
        /// Kernel area without a runnable variant
        area: &'static str,
        /// Runtime's PTX tier limit
        tier: PtxTier,
        /// Exact device capability
        device: ComputeCapability,
        /// Consumer Cargo feature that embeds a runnable variant
        feature: &'static str,
    },
    /// The requested device does not exist
    #[error("CUDA device {ordinal} is not available; {count} device(s) found")]
    NoDevice {
        /// The requested device ordinal
        ordinal: usize,
        /// Devices the driver reports
        count: usize,
    },
    /// The device is older than the `sm_75` baseline
    #[error(
        "GPU compute capability {capability} is below the native CUDA backend's {baseline} baseline (sm_75, Turing)"
    )]
    UnsupportedDevice {
        /// The device's compute capability
        capability: ComputeCapability,
        /// The lowest PTX tier
        baseline: PtxTier,
    },
    /// A PTX tier name is not one of the known tiers
    #[error("unknown PTX tier `{value}`; expected one of sm75, sm80, sm90, sm120")]
    InvalidPtxTier {
        /// The rejected value
        value: String,
    },
    /// The requested PTX tier needs a newer GPU than the device
    #[error("PTX tier {tier} needs compute capability {}, but the GPU has {capability}", tier.min_capability())]
    PtxTierAboveDevice {
        /// The requested tier
        tier: PtxTier,
        /// The device's compute capability
        capability: ComputeCapability,
    },
    /// The CUDA driver API returned an error
    #[error("CUDA driver: {0}")]
    Driver(#[from] DriverError),
    /// cuBLAS returned an error
    #[cfg(feature = "libraries")]
    #[error("cuBLAS: {0}")]
    Cublas(#[from] CublasError),
    /// cuDNN returned an error
    #[cfg(feature = "libraries")]
    #[error("cuDNN: {0}")]
    Cudnn(#[from] CudnnError),
    /// The requested artifact is absent or conflicts with a cached load
    #[error("CUDA module `{module}` cannot supply requested artifact {artifact:?}")]
    ArtifactUnavailable {
        /// The kernel area
        module: &'static str,
        /// The requested identity
        artifact: LoadedArtifact,
    },
    /// The driver rejected the requested artifact
    #[error("CUDA module `{module}` rejected requested artifact {artifact:?}: {source}")]
    ArtifactLoad {
        /// The kernel area
        module: &'static str,
        /// The requested identity
        artifact: LoadedArtifact,
        /// The driver's refusal
        #[source]
        source: DriverError,
    },
    /// A loaded module has no kernel with the requested name
    #[error("PTX module `{module}` has no kernel `{kernel}`: {source}")]
    KernelMissing {
        /// The kernel area
        module: &'static str,
        /// The requested kernel entry name
        kernel: String,
        /// The driver error
        #[source]
        source: DriverError,
    },
    /// A buffer length does not match the shape or operation it is used for
    #[error("{context}: buffer holds {actual} elements, needs {expected}")]
    BufferLength {
        /// Which operation checked the buffer
        context: &'static str,
        /// Elements the operation needs
        expected: usize,
        /// Elements the buffer holds
        actual: usize,
    },
    /// A dimension does not fit the 32-bit integers that cuBLAS and cuDNN take
    #[error("{context}: dimension {value} does not fit in a 32-bit CUDA library argument")]
    DimensionOverflow {
        /// Which operation checked the dimension
        context: &'static str,
        /// The dimension that overflowed
        value: usize,
    },
    /// A stream capture recorded no work
    #[error("CUDA graph capture recorded no work")]
    EmptyCapture,
    /// An operation was asked for a shape or option it does not support
    #[error("{context}: {reason}")]
    Unsupported {
        /// Which operation refused
        context: &'static str,
        /// What it does not support
        reason: String,
    },
}

impl Error {
    /// Returns true when this machine has no usable NVIDIA GPU: the driver library is
    /// missing, the driver reports no device, or the GPU is older than the `sm_75`
    /// baseline
    ///
    /// A missing cuBLAS or cuDNN on a machine with a GPU is a broken install, not a
    /// missing GPU, so it does not count
    pub fn is_device_unavailable(&self) -> bool {
        matches!(
            self,
            Self::LibraryUnavailable {
                library: Library::Driver
            } | Self::NoDevice { .. }
                | Self::UnsupportedDevice { .. }
        )
    }
}

/// Converts a dimension for a cuBLAS or cuDNN call
pub fn to_c_int(context: &'static str, value: usize) -> Result<i32, Error> {
    i32::try_from(value).map_err(|_| Error::DimensionOverflow { context, value })
}

/// Checks that a buffer holds exactly the elements an operation reads or writes
pub fn check_len(context: &'static str, expected: usize, actual: usize) -> Result<(), Error> {
    if expected != actual {
        return Err(Error::BufferLength {
            context,
            expected,
            actual,
        });
    }

    Ok(())
}

/// Number of elements in a shape, failing on overflow
pub fn element_count(context: &'static str, shape: &[usize]) -> Result<usize, Error> {
    shape.iter().try_fold(1_usize, |count, &dim| {
        count.checked_mul(dim).ok_or(Error::DimensionOverflow {
            context,
            value: dim,
        })
    })
}
