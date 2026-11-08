//! Proves that cuBLAS and cuBLASLt come from one install: cudarc (cudakit-cudarc)
//! opens both by versioned soname, so libcublas's own import of libcublasLt and
//! cudakit's cuBLASLt resolve to the same file
//!
//! This is its own test binary so no other test can have loaded a library first. Run
//! it once as is and once with a second install (a CUDA wheel's `nvidia/cu13/lib`) on
//! `LD_LIBRARY_PATH`; set `CUDAKIT_EXPECT_CUBLAS_DIR` to the directory both must come
//! from. Like `tests/gpu.rs`, it touches the device only with `CUDAKIT_REQUIRE_GPU=1`

#![cfg(target_os = "linux")]

use std::collections::BTreeSet;
use std::path::Path;

use cudakit::{Error, ExtraSymbols, Gpu, Library, LibraryPolicy, Profile, PtxTier, TierSet};

static PROFILE: Profile = Profile {
    consumer: "cudakit-soname-tests",
    compiled: TierSet::of(&[PtxTier::Sm75]),
    shipped: TierSet::of(&[PtxTier::Sm75]),
    tier_features: &[(PtxTier::Sm75, "test-sm75")],
    libraries: LibraryPolicy::Allowed,
    extra_symbols: ExtraSymbols::NONE,
};

#[test]
fn cublas_and_cublaslt_map_once_from_one_directory() -> Result<(), Error> {
    if std::env::var_os("CUDAKIT_REQUIRE_GPU").is_none_or(|value| value != "1") {
        eprintln!("skipping: set CUDAKIT_REQUIRE_GPU=1 to run GPU tests");
        return Ok(());
    }

    let gpu = Gpu::open(0, &PROFILE, None)?;
    gpu.prepare(Library::Cublas)?;
    gpu.prepare(Library::CublasLt)?;
    let versions = gpu.library_versions()?;
    eprintln!("cuBLAS {versions:?}, cuBLASLt {}", gpu.cublaslt_version()?);

    let maps = std::fs::read_to_string("/proc/self/maps").expect("readable /proc/self/maps");
    let mapped = |prefix: &str| {
        maps.lines()
            .filter_map(|line| line.split_whitespace().nth(5))
            .filter(|path| {
                Path::new(path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(prefix))
            })
            .map(|path| std::fs::canonicalize(path).unwrap_or_else(|_| path.into()))
            .collect::<BTreeSet<_>>()
    };
    let blas = mapped("libcublas.so");
    let lt = mapped("libcublasLt.so");
    eprintln!("mapped {blas:?} and {lt:?}");
    assert_eq!(blas.len(), 1, "one libcublas, not {blas:?}");
    assert_eq!(lt.len(), 1, "one libcublasLt, not {lt:?}");
    let directory = |set: &BTreeSet<std::path::PathBuf>| {
        set.first()
            .and_then(|path| path.parent())
            .map(Path::to_path_buf)
    };
    assert_eq!(directory(&blas), directory(&lt), "one install");
    if let Some(expected) = std::env::var_os("CUDAKIT_EXPECT_CUBLAS_DIR") {
        let expected = std::fs::canonicalize(expected).expect("an existing directory");
        assert_eq!(directory(&lt), Some(expected));
    }
    Ok(())
}
