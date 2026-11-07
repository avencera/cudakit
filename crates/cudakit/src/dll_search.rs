//! Windows DLL search for the libraries cudarc opens by name
//!
//! cudarc opens the driver, cuBLAS, cuDNN and NVRTC by bare name with Windows' default
//! search, which looks in the current directory before `PATH`. A DLL planted in the
//! folder a program runs from would load and run its code. cudakit can't change cudarc's
//! names or flags, so before cudarc loads a library it finds the file that search would
//! pick for each of cudarc's names in order. It refuses a file that comes from the
//! current directory or a relative `PATH` entry, and otherwise loads it by full path.
//! Loading a bare name reuses a module already loaded under that name, so cudarc's own
//! loads then get that same file, and the preflight still checks exactly what cudarc
//! runs
//!
//! The full-path load uses `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR`, so a DLL's own imports
//! come from its directory: `cublas64_12.dll` binds the `cublasLt64_12.dll` beside it,
//! not another install's copy found earlier on `PATH`, and never one in the current
//! directory

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// A file that Windows' default search would load from the current directory or a
/// relative `PATH` entry
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Untrusted(pub(crate) PathBuf);

/// The directories of Windows' default DLL search, in order, each marked trusted or not
///
/// With SafeDllSearchMode, which is on by default, a bare name is looked for in the
/// executable's directory, System32, the 16-bit System directory, the Windows directory,
/// the current directory and then each `PATH` entry. The current directory and relative
/// or empty `PATH` entries are untrusted unless they are also one of the others
pub(crate) struct SearchOrder(Vec<(PathBuf, bool)>);

impl SearchOrder {
    /// This process's search order
    #[cfg(windows)]
    pub(crate) fn current() -> Self {
        use std::env;

        let exe_dir = env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(Path::to_path_buf));
        Self::new(
            exe_dir,
            env::var_os("SystemRoot").map(PathBuf::from),
            env::current_dir().ok(),
            &env::var_os("PATH").unwrap_or_default(),
        )
    }

    pub(crate) fn new(
        exe_dir: Option<PathBuf>,
        system_root: Option<PathBuf>,
        cwd: Option<PathBuf>,
        path: &OsStr,
    ) -> Self {
        let mut fixed: Vec<PathBuf> = exe_dir.into_iter().collect();
        if let Some(root) = system_root {
            fixed.extend([root.join("System32"), root.join("System"), root]);
        }

        let entries: Vec<PathBuf> = std::env::split_paths(path).collect();
        let trusted: Vec<&Path> = fixed
            .iter()
            .chain(entries.iter().filter(|entry| entry.is_absolute()))
            .map(PathBuf::as_path)
            .collect();
        let is_trusted = |dir: &Path| trusted.iter().any(|other| same_dir(other, dir));

        let mut order: Vec<(PathBuf, bool)> = fixed.iter().map(|dir| (dir.clone(), true)).collect();
        // relative entries resolve against the current directory, so without one they
        // name nothing
        if let Some(cwd) = &cwd {
            order.push((cwd.clone(), is_trusted(cwd)));
        }

        for entry in &entries {
            if entry.is_absolute() {
                order.push((entry.clone(), true));
            } else if let Some(cwd) = &cwd {
                let dir = cwd.join(entry);
                let trusted = is_trusted(&dir);
                order.push((dir, trusted));
            }
        }

        Self(order)
    }

    /// The file the default search finds for `name`: the first directory that has it
    pub(crate) fn locate(&self, name: &OsStr) -> Option<Result<PathBuf, Untrusted>> {
        self.0
            .iter()
            .map(|(dir, trusted)| (dir.join(name), *trusted))
            .find(|(file, _)| file.is_file())
            .map(|(file, trusted)| {
                if trusted {
                    Ok(file)
                } else {
                    Err(Untrusted(file))
                }
            })
    }
}

/// Paths that name the same directory, compared after resolving case and links
fn same_dir(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

/// Loads the first of cudarc's names that Windows' default search would load, as cudarc
/// would, but never from an untrusted directory
///
/// Returns `Ok(None)` when no name is found, like cudarc's own search
#[cfg(windows)]
pub(crate) fn open_first(
    names: impl IntoIterator<Item = impl AsRef<OsStr>>,
) -> Result<Option<libloading::Library>, Untrusted> {
    use libloading::os::windows::{
        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, Library,
    };

    let order = SearchOrder::current();
    for name in names {
        let name = name.as_ref();
        // a bare-name load returns a module already loaded under that name, wherever it
        // came from, so cudarc gets that one
        if let Ok(loaded) = Library::open_already_loaded(name) {
            return Ok(Some(loaded.into()));
        }

        let Some(found) = order.locate(name) else {
            continue;
        };
        let file = found?;
        let flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
        // SAFETY: CUDA library initializers have no caller preconditions
        match unsafe { Library::load_with_flags(&file, flags) } {
            Ok(library) => return Ok(Some(library.into())),
            // cudarc moves on to its next name when a file is found but fails to load
            Err(_) => continue,
        }
    }

    Ok(None)
}

/// Loads the driver as [`open_first`] does, before cudarc's first driver call, and keeps
/// it loaded for the process as cudarc does
#[cfg(windows)]
pub(crate) fn pin_driver() -> Result<(), crate::Error> {
    use std::sync::OnceLock;

    static DRIVER: OnceLock<Result<Option<libloading::Library>, Untrusted>> = OnceLock::new();
    // cudarc's driver names, in its order
    let names = ["cuda", "nvcuda"]
        .into_iter()
        .flat_map(cudarc::get_lib_name_candidates);
    match DRIVER.get_or_init(|| open_first(names)) {
        Err(Untrusted(path)) => Err(crate::Error::UntrustedLibrary {
            library: crate::Library::Driver,
            path: path.clone(),
        }),
        // cudarc's own probe reports a missing driver next
        Ok(_) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::fs;
    use std::path::PathBuf;

    use super::{SearchOrder, Untrusted};

    /// The search finds the first directory holding a name, and refuses the current
    /// directory and relative `PATH` entries unless they are also a trusted directory
    #[test]
    fn untrusted_directories_are_refused_where_the_default_search_reaches_them() {
        let root = tempfile::tempdir().unwrap();
        let dir = |name: &str| {
            let dir = root.path().join(name);
            fs::create_dir_all(&dir).unwrap();
            dir
        };
        let (exe, cwd, absolute) = (dir("exe"), dir("cwd"), dir("absolute"));
        let relative = dir("cwd/relative");
        let place = |dir: &PathBuf, name: &str| fs::write(dir.join(name), b"").unwrap();
        place(&cwd, "planted.dll");
        place(&relative, "relative.dll");
        place(&absolute, "absolute.dll");
        place(&absolute, "shadowed.dll");
        place(&cwd, "shadowed.dll");
        place(&exe, "beside-exe.dll");
        place(&cwd, "beside-exe.dll");

        let path = env::join_paths([PathBuf::from("relative"), absolute.clone()]).unwrap();
        let order = SearchOrder::new(Some(exe.clone()), None, Some(cwd.clone()), &path);
        let locate = |name: &str| order.locate(name.as_ref());
        assert_eq!(
            locate("planted.dll"),
            Some(Err(Untrusted(cwd.join("planted.dll"))))
        );
        assert_eq!(
            locate("relative.dll"),
            Some(Err(Untrusted(cwd.join("relative").join("relative.dll"))))
        );
        assert_eq!(
            locate("absolute.dll"),
            Some(Ok(absolute.join("absolute.dll")))
        );
        // the current directory comes before PATH, so its copy is the one Windows loads
        assert_eq!(
            locate("shadowed.dll"),
            Some(Err(Untrusted(cwd.join("shadowed.dll"))))
        );
        assert_eq!(
            locate("beside-exe.dll"),
            Some(Ok(exe.join("beside-exe.dll")))
        );
        assert_eq!(locate("missing.dll"), None);

        // a current directory that is also an absolute PATH entry is searched as one
        let path = env::join_paths([cwd.clone()]).unwrap();
        let order = SearchOrder::new(Some(exe), None, Some(cwd.clone()), &path);
        assert_eq!(
            order.locate("planted.dll".as_ref()),
            Some(Ok(cwd.join("planted.dll")))
        );
    }

    /// A real DLL in the current directory is refused, not loaded
    #[cfg(windows)]
    #[test]
    fn a_dll_in_the_current_directory_is_refused() {
        /// Puts the working directory back even when an assertion fails
        struct Restore(PathBuf);

        impl Drop for Restore {
            fn drop(&mut self) {
                env::set_current_dir(&self.0).unwrap();
            }
        }

        // a copy of a system DLL under a name no other directory has; other tests in
        // this binary use absolute paths or system DLL names, so the directory change
        // doesn't reach them
        let name = format!("cudakit_search_probe_{}.dll", std::process::id());
        let dir = tempfile::tempdir().unwrap();
        let system = PathBuf::from(env::var_os("SystemRoot").unwrap()).join("System32");
        fs::copy(system.join("version.dll"), dir.path().join(&name)).unwrap();
        let _restore = Restore(env::current_dir().unwrap());
        env::set_current_dir(dir.path()).unwrap();

        let result = super::open_first([&name]);
        assert!(
            matches!(&result, Err(Untrusted(path)) if path.file_name() == Some(name.as_ref())),
            "{result:?}"
        );
    }
}
