//! Driver releases for persisted identities, such as tune files, with API-only
//! identities kept apart so they are never saved as a release

use std::ffi::{CStr, c_char, c_uint};

use libloading::Library;

use crate::Error;

/// Only a full driver release can identify saved measurements
///
/// The serialized form is `{"source": "nvml" | "sysfs" | "procfs", "version": "..."}`;
/// the same text from different sources never compares equal
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(
        tag = "source",
        content = "version",
        rename_all = "snake_case",
        deny_unknown_fields
    )
)]
pub enum DriverRelease {
    /// `nvmlSystemGetDriverVersion`
    Nvml(String),
    /// First line of `/sys/module/nvidia/version`
    Sysfs(String),
    /// First line of `/proc/driver/nvidia/version`
    Procfs(String),
}

/// A working CUDA API does not imply that the driver release is readable
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriverIdentity {
    /// A full driver release
    Release(DriverRelease),
    /// Only the CUDA API level from `cuDriverGetVersion`, such as 13000
    ApiOnly(i32),
}

impl DriverIdentity {
    /// Reads NVML, then sysfs, then procfs, then the CUDA API level
    ///
    /// Call it with a live CUDA context; NVML is loaded and released within the call
    pub fn read() -> Result<Self, Error> {
        Self::select(
            nvml_version,
            |path| std::fs::read_to_string(path),
            cuda_version,
        )
    }

    fn select(
        nvml: impl FnOnce() -> Option<String>,
        mut file: impl FnMut(&str) -> std::io::Result<String>,
        cuda: impl FnOnce() -> Result<i32, Error>,
    ) -> Result<Self, Error> {
        if let Some(version) = nvml().and_then(|text| first_line(&text)) {
            return Ok(Self::Release(DriverRelease::Nvml(version)));
        }
        if let Some(version) = file("/sys/module/nvidia/version")
            .ok()
            .and_then(|text| first_line(&text))
        {
            return Ok(Self::Release(DriverRelease::Sysfs(version)));
        }
        if let Some(version) = file("/proc/driver/nvidia/version")
            .ok()
            .and_then(|text| first_line(&text))
        {
            return Ok(Self::Release(DriverRelease::Procfs(version)));
        }

        let version = cuda()?;
        if version <= 0 {
            return Err(Error::Unsupported {
                context: "driver identity",
                reason: "CUDA driver API version is not positive".into(),
            });
        }

        Ok(Self::ApiOnly(version))
    }
}

fn first_line(text: &str) -> Option<String> {
    text.lines()
        .next()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
}

fn nvml_version() -> Option<String> {
    // safety: NVML library initializers have no caller preconditions
    let library = unsafe { Library::new("libnvidia-ml.so.1") }.ok()?;
    nvml_version_from_library(&library)
}

fn nvml_version_from_library(library: &Library) -> Option<String> {
    type Init = unsafe extern "C" fn() -> c_uint;
    type GetVersion = unsafe extern "C" fn(*mut c_char, c_uint) -> c_uint;
    type Shutdown = unsafe extern "C" fn() -> c_uint;

    // safety: signatures match NVML's C API; symbols cannot outlive the library
    let (init, get_version, shutdown) = unsafe {
        (
            library.get::<Init>(b"nvmlInit_v2\0").ok()?,
            library
                .get::<GetVersion>(b"nvmlSystemGetDriverVersion\0")
                .ok()?,
            library.get::<Shutdown>(b"nvmlShutdown\0").ok()?,
        )
    };
    // safety: all symbols were checked before initialization, and init takes no arguments
    if unsafe { init() } != 0 {
        return None;
    }

    // nvml specifies 80 bytes including the terminator; decode only inside this buffer
    let mut bytes = [0u8; 80];
    // safety: NVML is initialized and the writable buffer has the given length
    let status = unsafe { get_version(bytes.as_mut_ptr().cast(), bytes.len() as c_uint) };
    // safety: balance exactly this initialization, including a failed version query
    let _ = unsafe { shutdown() };
    if status != 0 {
        return None;
    }

    CStr::from_bytes_until_nul(&bytes)
        .ok()?
        .to_str()
        .ok()
        .map(str::to_owned)
}

fn cuda_version() -> Result<i32, Error> {
    let mut version = 0;
    // safety: callers already have a live CUDA context, and the output is valid
    unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut version) }.result()?;
    Ok(version)
}

#[cfg(test)]
mod tests {

    //! Host-only checks of driver-source selection and identity

    use super::{DriverIdentity, DriverRelease};
    use crate::Error;

    fn missing() -> std::io::Result<String> {
        Err(std::io::ErrorKind::NotFound.into())
    }

    #[test]
    fn nvml_release_wins_without_reading_files_or_cuda() {
        let version = DriverIdentity::select(
            || Some("580.95.05".into()),
            |_| panic!("NVML must avoid file reads"),
            || panic!("NVML must avoid the CUDA fallback"),
        )
        .unwrap();
        assert_eq!(
            version,
            DriverIdentity::Release(DriverRelease::Nvml("580.95.05".into()))
        );
    }

    #[test]
    fn missing_nvml_uses_sysfs_before_procfs() {
        let mut paths = Vec::new();
        let version = DriverIdentity::select(
            || None,
            |path| {
                paths.push(path.to_owned());
                Ok(" 580.95.05 \n".into())
            },
            || panic!("a module release must avoid the CUDA fallback"),
        )
        .unwrap();
        assert_eq!(
            version,
            DriverIdentity::Release(DriverRelease::Sysfs("580.95.05".into()))
        );
        assert_eq!(paths, ["/sys/module/nvidia/version"]);
    }

    #[test]
    fn unavailable_sysfs_uses_the_procfs_first_line() {
        let first_line = "NVRM version: NVIDIA UNIX x86_64 Kernel Module 580.95.05";
        for sysfs in [None, Some(" \n")] {
            let mut paths = Vec::new();
            let version = DriverIdentity::select(
                || None,
                |path| {
                    paths.push(path.to_owned());
                    match path {
                        "/sys/module/nvidia/version" => {
                            sysfs.map(str::to_owned).map_or_else(missing, Ok)
                        }
                        "/proc/driver/nvidia/version" => {
                            Ok(format!("{first_line}\nGCC version: ignored\n"))
                        }
                        _ => panic!("unexpected driver path {path}"),
                    }
                },
                || panic!("procfs must avoid the CUDA fallback"),
            )
            .unwrap();
            assert_eq!(
                version,
                DriverIdentity::Release(DriverRelease::Procfs(first_line.into()))
            );
            assert_eq!(
                paths,
                ["/sys/module/nvidia/version", "/proc/driver/nvidia/version"]
            );
        }
    }

    #[test]
    fn missing_or_empty_sources_use_the_cuda_api_level() {
        for nvml in [None, Some(" \n")] {
            for file in [None, Some("\n")] {
                let version = DriverIdentity::select(
                    || nvml.map(str::to_owned),
                    |_| file.map(str::to_owned).map_or_else(missing, Ok),
                    || Ok(13000),
                )
                .unwrap();
                assert_eq!(version, DriverIdentity::ApiOnly(13000));
            }
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialized_release_names_its_source() {
        let release = DriverRelease::Nvml("580.95.05".into());
        assert_eq!(
            serde_json::to_value(&release).unwrap(),
            serde_json::json!({"source": "nvml", "version": "580.95.05"})
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn equal_text_from_different_sources_never_matches() {
        let versions = [
            DriverRelease::Nvml("13000".into()),
            DriverRelease::Sysfs("13000".into()),
            DriverRelease::Procfs("13000".into()),
        ];
        for (index, version) in versions.iter().enumerate() {
            let json = serde_json::to_vec(version).unwrap();
            assert_eq!(
                serde_json::from_slice::<DriverRelease>(&json).unwrap(),
                *version
            );
            for other in &versions[index + 1..] {
                assert_ne!(version, other);
            }
        }
    }

    #[test]
    fn an_unavailable_or_invalid_cuda_api_version_returns_the_error() {
        let result = DriverIdentity::select(
            || None,
            |_| missing(),
            || {
                Err(Error::Unsupported {
                    context: "test",
                    reason: "driver query failed".into(),
                })
            },
        );
        assert!(
            matches!(result, Err(Error::Unsupported { reason, .. }) if reason == "driver query failed")
        );
        for version in [0, -1] {
            let result = DriverIdentity::select(|| None, |_| missing(), || Ok(version));
            assert!(
                matches!(result, Err(Error::Unsupported { reason, .. }) if reason == "CUDA driver API version is not positive")
            );
        }
    }

    #[test]
    fn nvml_symbols_return_the_release_and_balance_initialization() {
        use std::process::Command;

        use libloading::Library;

        let directory =
            std::env::temp_dir().join(format!("cudakit-nvml-stub-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        for (index, init_status, query_status, terminate, expected) in [
            (0, 0, 0, true, Some("580.95.05")),
            (1, 1, 0, true, None),
            (2, 0, 1, true, None),
            (3, 0, 0, false, None),
        ] {
            let source = directory.join(format!("stub-{index}.c"));
            let path = directory.join(format!("stub-{index}.{}", std::env::consts::DLL_EXTENSION));
            let copy = if terminate {
                "memcpy(version, \"580.95.05\", 10);"
            } else {
                "memset(version, 'x', length);"
            };
            std::fs::write(
                &source,
                format!(
                    r#"
    #include <string.h>
    static unsigned int shutdown_count;
    unsigned int nvmlInit_v2(void) {{ return {init_status}; }}
    unsigned int nvmlSystemGetDriverVersion(char *version, unsigned int length) {{
        if (length != 80) return 2;
        {copy}
        return {query_status};
    }}
    unsigned int nvmlShutdown(void) {{ ++shutdown_count; return 0; }}
    unsigned int stubShutdownCount(void) {{ return shutdown_count; }}
    "#
                ),
            )
            .unwrap();
            let output = Command::new("cc")
                .args(["-shared", "-fPIC"])
                .arg(&source)
                .arg("-o")
                .arg(&path)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            // safety: the host stub has no library initializer or caller preconditions
            let library = unsafe { Library::new(&path) }.unwrap();
            assert_eq!(
                super::nvml_version_from_library(&library).as_deref(),
                expected
            );
            // safety: the test stub exports this exact signature, and the library is live
            let shutdown_count = unsafe {
                library
                    .get::<unsafe extern "C" fn() -> u32>(b"stubShutdownCount\0")
                    .unwrap()()
            };
            assert_eq!(shutdown_count, u32::from(init_status == 0));
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
}
