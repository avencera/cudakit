//! Per-GPU fallible library cells, requested only while a consumer builds a plan

use std::ffi::CStr;
use std::ops::Deref;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::{self, ThreadId};

use libloading::Library as SharedObject;

use cudarc::cublas::{CudaBlas, result::CublasError};
use cudarc::cudnn::{Cudnn, CudnnError};
use cudarc::driver::CudaStream;

use crate::profile::Profile;
use crate::symbols::{CUBLAS, CUDNN, NVRTC};
use crate::{Error, Library, Math};

#[derive(Debug, Clone)]
enum Failure {
    Missing(Library),
    Blas(CublasError),
    Dnn(CudnnError),
}

impl From<Failure> for Error {
    fn from(failure: Failure) -> Self {
        match failure {
            Failure::Missing(library) => Self::LibraryUnavailable { library },
            Failure::Blas(error) => Self::Cublas(error),
            Failure::Dnn(error) => Self::Cudnn(error),
        }
    }
}

#[derive(Debug)]
struct Blas {
    handle: CudaBlas,
    // keep mode selection and enqueue in one critical section
    math: Mutex<Math>,
}

/// The shared cuDNN handle, bound to the thread that created it
///
/// cuDNN allows a handle on any thread but never on two at once, and cudarc's
/// descriptors and plans each keep a clone of it. Handing it out only on `owner`
/// keeps every use on one thread, since the clones are `!Send` and stay there too
#[derive(Debug)]
struct Dnn {
    owner: ThreadId,
    handle: Arc<Cudnn>,
}

// SAFETY: other threads never reach `handle` except to drop it: `Libraries::dnn`
// returns it only on `owner`. Dropping on another thread only decrements the atomic
// count, and a final drop there destroys the handle when no clone, and so no use on
// `owner`, remains; cuDNN allows that from any thread, and `Gpu`'s drop binds the
// context first
unsafe impl Send for Dnn {}
// SAFETY: see `Send`; a shared reference gives other threads nothing but `owner`
unsafe impl Sync for Dnn {}

/// The shared cuBLAS handle in one math mode, from [`Gpu::blas`](crate::Gpu::blas)
///
/// Holds the handle's mode lock until dropped, so keep it only around the calls that
/// need it
#[derive(Debug)]
pub struct BlasGuard<'a> {
    handle: &'a CudaBlas,
    // private, so callers can read the mode only through the handle
    _math: MutexGuard<'a, Math>,
}

impl Deref for BlasGuard<'_> {
    type Target = CudaBlas;

    fn deref(&self) -> &CudaBlas {
        self.handle
    }
}

/// Versions of the loaded numerical libraries, for tune-file identities
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LibraryVersions {
    /// `cudnnGetVersion`
    pub cudnn: usize,
    /// `cublasGetVersion`
    pub cublas: i32,
}

/// Optional libraries drop before the GPU's stream and context
#[derive(Debug, Default)]
pub(crate) struct Libraries {
    blas: OnceLock<Result<Blas, Failure>>,
    dnn: OnceLock<Result<Dnn, Failure>>,
    nvrtc: OnceLock<Result<(), Failure>>,
}

impl Libraries {
    /// The policy check is deliberately before the cell lookup and every probe
    pub(crate) fn prepare(
        &self,
        profile: &Profile,
        library: Library,
        stream: &Arc<CudaStream>,
    ) -> Result<(), Error> {
        profile.allow(library)?;
        match library {
            Library::Cublas => cached(self.blas.get_or_init(|| {
                present(profile, library)?;
                let handle = CudaBlas::new(stream.clone()).map_err(Failure::Blas)?;
                set_math(&handle, Math::Fp32).map_err(Failure::Blas)?;
                Ok(Blas {
                    handle,
                    math: Mutex::new(Math::Fp32),
                })
            }))
            .map(|_| ()),
            Library::Cudnn => cached(self.dnn.get_or_init(|| {
                present(profile, library)?;
                let handle = Cudnn::new(stream.clone()).map_err(Failure::Dnn)?;
                Ok(Dnn {
                    owner: thread::current().id(),
                    handle,
                })
            }))
            .map(|_| ()),
            Library::Nvrtc => {
                cached(self.nvrtc.get_or_init(|| present(profile, library))).map(|_| ())
            }
            Library::Driver => Ok(()),
        }
    }

    /// Query the same loaded libraries and handle that execute library plans
    pub(crate) fn versions(
        &self,
        profile: &Profile,
        stream: &Arc<CudaStream>,
    ) -> Result<LibraryVersions, Error> {
        self.prepare(profile, Library::Cublas, stream)?;
        self.prepare(profile, Library::Cudnn, stream)?;
        let mut cublas = 0;
        // SAFETY: the prepared handle is live and the version output is writable
        unsafe { cudarc::cublas::sys::cublasGetVersion_v2(*self.blas()?.handle(), &mut cublas) }
            .result()?;
        let cudnn = cudarc::cudnn::result::get_version();
        if cublas <= 0 || cudnn == 0 {
            return Err(Error::Unsupported {
                context: "library versions",
                reason: "numerical-library versions must be positive".to_owned(),
            });
        }
        Ok(LibraryVersions { cudnn, cublas })
    }

    /// Access an already prepared handle without loading during forward or capture
    pub(crate) fn blas(&self) -> Result<&CudaBlas, Error> {
        let value = self.blas.get().ok_or(unprepared(Library::Cublas))?;
        cached(value).map(|blas| &blas.handle)
    }

    /// The shared cuDNN handle, only on the thread that prepared it
    pub(crate) fn dnn(&self) -> Result<&Arc<Cudnn>, Error> {
        let dnn = cached(self.dnn.get().ok_or(unprepared(Library::Cudnn))?)?;
        if dnn.owner != thread::current().id() {
            return Err(Error::LibraryOnAnotherThread {
                library: Library::Cudnn,
            });
        }

        Ok(&dnn.handle)
    }

    pub(crate) fn lock_blas(&self, math: Math) -> Result<BlasGuard<'_>, Error> {
        let blas = cached(self.blas.get().ok_or(unprepared(Library::Cublas))?)?;
        let mut current = blas.math.lock().unwrap_or_else(PoisonError::into_inner);
        if *current != math {
            set_math(&blas.handle, math)?;
            *current = math;
        }
        Ok(BlasGuard {
            handle: &blas.handle,
            _math: current,
        })
    }
}

fn cached<T>(value: &Result<T, Failure>) -> Result<&T, Error> {
    value.as_ref().map_err(|error| error.clone().into())
}

fn unprepared(library: Library) -> Error {
    Error::LibraryUnprepared { library }
}

fn set_math(blas: &CudaBlas, math: Math) -> Result<(), CublasError> {
    // SAFETY: the handle is live and the caller holds its mode lock
    unsafe { cudarc::cublas::sys::cublasSetMathMode(*blas.handle(), math.cublas()) }.result()
}

fn present(profile: &Profile, library: Library) -> Result<(), Failure> {
    let extra = &profile.extra_symbols;
    let (base, extra) = match library {
        Library::Cublas => (CUBLAS, extra.cublas),
        Library::Cudnn => (CUDNN, extra.cudnn),
        Library::Nvrtc => (NVRTC, extra.nvrtc),
        Library::Driver => (&[][..], &[][..]),
    };
    preflight(library, opened(library), base.iter().chain(extra)).map(|_| ())
}

/// The file cudarc opened for `library`, which it keeps for the process and every later
/// call uses, so the preflight checks exactly what cudarc runs
fn opened(library: Library) -> Option<&'static SharedObject> {
    use cudarc::{cublas, cudnn, driver, nvrtc};

    // SAFETY: CUDA library initializers have no caller preconditions, and `culib` only
    // returns the handle `is_culib_present` just opened
    unsafe {
        match library {
            Library::Cublas => cublas::sys::is_culib_present().then(|| cublas::sys::culib()),
            Library::Cudnn => cudnn::sys::is_culib_present().then(|| cudnn::sys::culib()),
            Library::Nvrtc => nvrtc::sys::is_culib_present().then(|| nvrtc::sys::culib()),
            Library::Driver => driver::sys::is_culib_present().then(|| driver::sys::culib()),
        }
    }
}

/// Fails unless `opened` has every symbol
///
/// cudarc keeps the first file that opens, so a missing symbol in that file must fail
/// here rather than be found later as a panic in a cudarc call
fn preflight<'l, 'a>(
    library: Library,
    opened: Option<&'l SharedObject>,
    symbols: impl IntoIterator<Item = &'a &'a CStr>,
) -> Result<&'l SharedObject, Failure> {
    let checked = opened.ok_or(Failure::Missing(library))?;
    for symbol in symbols {
        // SAFETY: lookup only; the untyped address is never dereferenced or called
        unsafe { checked.get::<*const ()>(symbol.to_bytes_with_nul()) }
            .map_err(|_| Failure::Missing(library))?;
    }

    Ok(checked)
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use std::path::PathBuf;
    #[cfg(unix)]
    use std::sync::atomic::{AtomicU64, Ordering};

    use libloading::Library as SharedObject;

    use super::preflight;
    use crate::{Error, Library};

    /// Stub libraries compiled with `cc`, which Unix runners have and Windows ones lack;
    /// Windows tests use system DLLs instead
    #[cfg(unix)]
    struct StubDirectory(PathBuf);

    #[cfg(unix)]
    impl StubDirectory {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let sequence = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "cudakit-library-preflight-{}-{sequence}",
                std::process::id()
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        /// Compiles and opens a stub library
        fn build(
            &self,
            index: usize,
            symbols: &[&std::ffi::CStr],
            omitted: Option<&std::ffi::CStr>,
        ) -> SharedObject {
            use std::process::Command;

            let source = self.0.join(format!("stub-{index}.c"));
            let output = self
                .0
                .join(format!("stub-{index}.{}", std::env::consts::DLL_EXTENSION));
            let exports = symbols
                .iter()
                .filter(|symbol| Some(**symbol) != omitted)
                .map(|symbol| format!("void {}(void) {{}}\n", symbol.to_str().unwrap()))
                .collect::<String>();
            std::fs::write(&source, exports).unwrap();
            let result = Command::new("cc")
                .args(["-shared", "-fPIC"])
                .arg(&source)
                .arg("-o")
                .arg(&output)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );

            // SAFETY: the stub has no initializers
            unsafe { SharedObject::new(&output) }.unwrap()
        }
    }

    #[cfg(unix)]
    impl Drop for StubDirectory {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    fn unavailable(error: Error, expected: Library) {
        assert!(matches!(
            error,
            Error::LibraryUnavailable { library } if library == expected
        ));
    }

    #[cfg(unix)]
    #[test]
    fn every_required_symbol_is_checked_before_cudarc_can_run() {
        use crate::symbols::{CUBLAS, CUDNN, NVRTC};

        for (library, symbols) in [
            (Library::Cublas, CUBLAS),
            (Library::Cudnn, CUDNN),
            (Library::Nvrtc, NVRTC),
        ] {
            let directory = StubDirectory::new();
            let complete = directory.build(0, symbols, None);
            assert!(preflight(library, Some(&complete), symbols).is_ok());
            for (index, omitted) in symbols.iter().enumerate() {
                let incomplete = directory.build(index + 1, symbols, Some(omitted));
                let error = preflight(library, Some(&incomplete), symbols).unwrap_err();
                unavailable(error.into(), library);
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn an_incomplete_library_fails_once_and_the_failure_is_cached() {
        use std::sync::OnceLock;

        use super::cached;
        use crate::symbols::CUBLAS;

        let directory = StubDirectory::new();
        let incomplete = directory.build(0, CUBLAS, Some(CUBLAS[0]));
        let cell = OnceLock::new();
        let mut initializations = 0;
        for _ in 0..2 {
            let result = cell.get_or_init(|| {
                initializations += 1;
                preflight(Library::Cublas, Some(&incomplete), CUBLAS).map(|_| ())
            });
            unavailable(cached(result).unwrap_err(), Library::Cublas);
        }

        assert_eq!(initializations, 1);
    }

    /// System DLLs stand in for the stubs on Windows: kernel32 lacks advapi32's
    /// `CryptAcquireContextW`
    #[cfg(windows)]
    #[test]
    fn windows_preflight_checks_the_opened_dll() {
        // SAFETY: system DLLs with no caller preconditions
        let open = |name: &str| unsafe { SharedObject::new(name) }.unwrap();
        let (kernel32, advapi32) = (open("kernel32.dll"), open("advapi32.dll"));
        assert!(preflight(Library::Cublas, Some(&kernel32), &[c"GetProcAddress"]).is_ok());

        let error =
            preflight(Library::Cudnn, Some(&kernel32), &[c"CryptAcquireContextW"]).unwrap_err();
        unavailable(error.into(), Library::Cudnn);
        assert!(preflight(Library::Cudnn, Some(&advapi32), &[c"CryptAcquireContextW"]).is_ok());
    }

    #[test]
    fn a_library_cudarc_could_not_open_is_a_typed_error() {
        for library in [Library::Cublas, Library::Cudnn, Library::Nvrtc] {
            unavailable(preflight(library, None, &[]).unwrap_err().into(), library);
        }
    }
}
