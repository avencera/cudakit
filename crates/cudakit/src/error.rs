use std::fmt;
use std::path::PathBuf;

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
    /// The driver API, `libcuda` (`nvcuda.dll` on Windows), installed with the NVIDIA
    /// driver
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
            Self::Driver if cfg!(windows) => "CUDA driver (nvcuda.dll)",
            Self::Driver => "CUDA driver (libcuda)",
            Self::Cublas => "cuBLAS",
            Self::Cudnn => "cuDNN",
            Self::Nvrtc => "NVRTC",
        })
    }
}

impl Library {
    /// The files cudarc opens for this library on Windows, and where they are found
    ///
    /// cudarc's candidates include CUDA 10, 11 and 12 names but no CUDA 13 one, so the
    /// hints name the CUDA 12 files cudakit's symbols come from, and CUDA 13's as
    /// unsupported rather than left for the user to discover
    #[cfg(windows)]
    fn search_hint(self) -> &'static str {
        match self {
            Self::Driver => {
                "nvcuda.dll comes with the NVIDIA driver; install one for CUDA 12.4 or newer"
            }
            Self::Cublas => {
                "cudakit needs CUDA 12's cublas64_12.dll, which loads cublasLt64_12.dll beside \
                 it; put the directory holding them on PATH as an absolute path (a CUDA 12 \
                 toolkit keeps them in %CUDA_PATH_V12_x%\\bin). CUDA 13's cublas64_13.dll, in \
                 %CUDA_PATH%\\bin\\x64, is not supported yet"
            }
            Self::Cudnn => {
                "cudakit needs cuDNN 9's cudnn64_9.dll; put the directory holding it on PATH as an \
                 absolute path"
            }
            Self::Nvrtc => {
                "cudakit needs CUDA 12's nvrtc64_120_0.dll; put the directory holding it on PATH \
                 as an absolute path (a CUDA 12 toolkit keeps it in %CUDA_PATH_V12_x%\\bin). \
                 CUDA 13's nvrtc64_130_0.dll, in %CUDA_PATH%\\bin\\x64, is not supported yet"
            }
        }
    }

    /// The files cudarc opens for this library on Linux, and where they are found
    #[cfg(not(windows))]
    fn search_hint(self) -> &'static str {
        match self {
            Self::Driver => {
                "libcuda.so.1 comes with the NVIDIA driver; install one for CUDA 12.4 or newer"
            }
            Self::Cublas => {
                "cudakit opens a toolkit's libcublas.so, or else libcublas.so.12, through \
                 LD_LIBRARY_PATH or the ldconfig cache"
            }
            Self::Cudnn => {
                "cudakit opens libcudnn.so, or else libcudnn.so.9, through LD_LIBRARY_PATH or \
                 the ldconfig cache"
            }
            Self::Nvrtc => {
                "cudakit opens a toolkit's libnvrtc.so, or else libnvrtc.so.12, through \
                 LD_LIBRARY_PATH or the ldconfig cache"
            }
        }
    }
}

/// Errors from cudakit
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A CUDA shared library is absent or lacks a required symbol
    ///
    /// The message names the files cudakit looks for and where it looks on this OS
    #[error(
        "could not load the {library} shared library, or it lacks a symbol cudakit needs; {}",
        library.search_hint()
    )]
    LibraryUnavailable {
        /// The library that failed to load
        library: Library,
    },
    /// Windows would load a CUDA library from the current directory or a relative `PATH`
    /// entry, where a planted DLL would run its code (Windows only)
    #[error(
        "refusing to load the {library} shared library from {}, which Windows reaches through \
         the current directory or a relative PATH entry; put it in the executable's directory \
         or an absolute PATH entry, or run from another directory",
        path.display()
    )]
    UntrustedLibrary {
        /// The library that was refused
        library: Library,
        /// The file Windows would have loaded
        path: PathBuf,
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
    #[error("PTX tier {tier} for device {device} is not compiled in; {}", enable(*.feature, *.tier))]
    TierNotCompiledIn {
        /// Required PTX tier
        tier: PtxTier,
        /// Exact device capability
        device: ComputeCapability,
        /// Consumer Cargo feature that embeds the tier, when the profile names one
        feature: Option<&'static str>,
    },
    /// An area has no embedded variant that the device can run within the tier limit
    #[error(
        "CUDA area {area} has no runnable PTX at tier {tier} for device {device}; {}",
        enable(*.feature, *.tier)
    )]
    AreaTierNotCompiledIn {
        /// Kernel area without a runnable variant
        area: &'static str,
        /// Runtime's PTX tier limit
        tier: PtxTier,
        /// Exact device capability
        device: ComputeCapability,
        /// Consumer Cargo feature that embeds a runnable variant, when the profile names
        /// one
        feature: Option<&'static str>,
    },
    /// A module request names a tier above the runtime's PTX tier limit
    #[error("CUDA area {area} requested tier {tier}, above this runtime's {limit} limit")]
    TierAboveLimit {
        /// Kernel area of the request
        area: &'static str,
        /// Tier the request names
        tier: PtxTier,
        /// Runtime's PTX tier limit
        limit: PtxTier,
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

/// The hint that ends a missing-tier message
fn enable(feature: Option<&'static str>, tier: PtxTier) -> String {
    match feature {
        Some(feature) => format!("enable `{feature}`"),
        None => format!("enable a feature that embeds {tier}"),
    }
}
