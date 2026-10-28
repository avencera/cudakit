//! Per-GPU fallible library cells, requested only while a consumer builds a plan

use std::ffi::{CStr, OsStr};
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

#[derive(Debug, Clone, Copy)]
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
    // drop the handle before releasing the checked shared object
    _library: SharedObject,
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
    _library: SharedObject,
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
    nvrtc: OnceLock<Result<SharedObject, Failure>>,
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
                let checked = present(profile, library)?;
                let handle = CudaBlas::new(stream.clone()).map_err(Failure::Blas)?;
                set_math(&handle, Math::Fp32).map_err(Failure::Blas)?;
                Ok(Blas {
                    handle,
                    math: Mutex::new(Math::Fp32),
                    _library: checked,
                })
            }))
            .map(|_| ()),
            Library::Cudnn => cached(self.dnn.get_or_init(|| {
                let checked = present(profile, library)?;
                let handle = Cudnn::new(stream.clone()).map_err(Failure::Dnn)?;
                Ok(Dnn {
                    owner: thread::current().id(),
                    handle,
                    _library: checked,
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
    value.as_ref().map_err(|error| (*error).into())
}

fn unprepared(library: Library) -> Error {
    Error::LibraryUnprepared { library }
}

fn set_math(blas: &CudaBlas, math: Math) -> Result<(), CublasError> {
    // SAFETY: the handle is live and the caller holds its mode lock
    unsafe { cudarc::cublas::sys::cublasSetMathMode(*blas.handle(), math.cublas()) }.result()
}

fn present(profile: &Profile, library: Library) -> Result<SharedObject, Failure> {
    let extra = &profile.extra_symbols;
    let (name, base, extra) = match library {
        Library::Cublas => ("cublas", CUBLAS, extra.cublas),
        Library::Cudnn => ("cudnn", CUDNN, extra.cudnn),
        Library::Nvrtc => ("nvrtc", NVRTC, extra.nvrtc),
        Library::Driver => ("cuda", &[][..], &[][..]),
    };
    preflight(
        library,
        cudarc::get_lib_name_candidates(name),
        base.iter().chain(extra),
    )
}

fn preflight<'a>(
    library: Library,
    choices: impl IntoIterator<Item = impl AsRef<OsStr>>,
    symbols: impl IntoIterator<Item = &'a &'a CStr> + Clone,
) -> Result<SharedObject, Failure> {
    for choice in choices {
        // SAFETY: CUDA library initializers have no caller preconditions
        let Ok(checked) = (unsafe { SharedObject::new(choice.as_ref()) }) else {
            continue;
        };
        // cudarc selects the first file that opens in this exact order; a missing
        // symbol in that file must fail, not fall through to a different version
        for symbol in symbols.clone() {
            // SAFETY: lookup only; the untyped address is never dereferenced or called
            unsafe { checked.get::<*const ()>(symbol.to_bytes_with_nul()) }
                .map_err(|_| Failure::Missing(library))?;
        }

        return Ok(checked);
    }

    Err(Failure::Missing(library))
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::{cached, preflight};
    use crate::symbols::{CUBLAS, CUDNN, NVRTC};
    use crate::{Error, Library};

    struct StubDirectory(PathBuf);

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

        fn build(&self, index: usize, symbols: &[&CStr], omitted: Option<&CStr>) -> PathBuf {
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
            output
        }
    }

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

    #[test]
    fn every_required_symbol_is_checked_before_cudarc_can_run() {
        for (library, symbols) in [
            (Library::Cublas, CUBLAS),
            (Library::Cudnn, CUDNN),
            (Library::Nvrtc, NVRTC),
        ] {
            let directory = StubDirectory::new();
            let complete = directory.build(0, symbols, None);
            let checked = preflight(library, [&complete], symbols).unwrap();
            // keep the checked object open after its name disappears
            std::fs::remove_file(&complete).unwrap();
            // SAFETY: the symbol is looked up, never called as a stub CUDA API
            assert!(unsafe { checked.get::<*const ()>(symbols[0].to_bytes_with_nul()) }.is_ok());
            for (index, omitted) in symbols.iter().enumerate() {
                let incomplete = directory.build(index + 1, symbols, Some(omitted));
                let error = preflight(library, [&incomplete], symbols).unwrap_err();
                unavailable(error.into(), library);
            }
        }
    }

    #[test]
    fn incomplete_first_library_does_not_fall_through_and_failure_is_cached() {
        let directory = StubDirectory::new();
        let incomplete = directory.build(0, CUBLAS, Some(CUBLAS[0]));
        let complete = directory.build(1, CUBLAS, None);
        let cell = OnceLock::new();
        let mut initializations = 0;
        for _ in 0..2 {
            let result = cell.get_or_init(|| {
                initializations += 1;
                preflight(Library::Cublas, [&incomplete, &complete], CUBLAS)
            });
            unavailable(cached(result).unwrap_err(), Library::Cublas);
        }

        assert_eq!(initializations, 1);
        assert!(preflight(Library::Cublas, [&complete], CUBLAS).is_ok());
        let missing = directory.0.join("not-installed");
        unavailable(
            preflight(Library::Nvrtc, [&missing], NVRTC)
                .unwrap_err()
                .into(),
            Library::Nvrtc,
        );
    }

    #[test]
    fn nonexistent_file_is_a_typed_error() {
        let directory = StubDirectory::new();
        let missing: &Path = &directory.0.join("missing-library");
        for library in [Library::Cublas, Library::Cudnn, Library::Nvrtc] {
            unavailable(
                preflight(library, [missing], &[]).unwrap_err().into(),
                library,
            );
        }
    }
}
