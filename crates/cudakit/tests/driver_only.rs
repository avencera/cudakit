//! Proves that a driver-only profile keeps every CUDA library except the driver out of
//! the process, even when the `libraries` feature is compiled in
//!
//! This is its own test binary so no other test can have loaded a library first. Like
//! `tests/gpu.rs`, it touches the device only with `CUDAKIT_REQUIRE_GPU=1`

#![cfg(target_os = "linux")]

use cudakit::cudarc::driver::{LaunchConfig, PushKernelArg};
use cudakit::{
    AreaPtx, Error, ExtraSymbols, KernelArea, Library, LibraryPolicy, Options, Profile, PtxTier,
    Runtime, TierSet,
};

static DRIVER_ONLY: Profile = Profile {
    consumer: "cudakit-tests",
    compiled: TierSet::of(&[PtxTier::Sm75]),
    shipped: TierSet::of(&[PtxTier::Sm75]),
    tier_features: &[(PtxTier::Sm75, "test-sm75")],
    libraries: LibraryPolicy::DriverOnly,
    extra_symbols: ExtraSymbols::NONE,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct AddOne;

impl KernelArea for AddOne {
    const PROFILE: &'static Profile = &DRIVER_ONLY;

    fn name(self) -> &'static str {
        "add-one"
    }

    fn variants(self) -> AreaPtx {
        AreaPtx::fixture(&[(PtxTier::Sm75, include_str!("fixtures/add_one.ptx"))])
    }
}

/// Shared objects that a driver-only process must never map. `libcublas` also matches
/// `libcublasLt`, and `libcudnn` its sub-libraries
const FORBIDDEN: [&str; 5] = [
    "libcudart",
    "libcublas",
    "libcudnn",
    "libnvrtc",
    "libnvJitLink",
];

#[test]
fn a_driver_only_runtime_maps_no_cuda_library_but_the_driver() -> Result<(), Error> {
    if std::env::var_os("CUDAKIT_REQUIRE_GPU").is_none_or(|value| value != "1") {
        eprintln!("skipping: set CUDAKIT_REQUIRE_GPU=1 to run GPU tests");
        return Ok(());
    }

    let runtime = Runtime::<AddOne>::open(0, Options::default())?;
    for library in [Library::Cublas, Library::Cudnn, Library::Nvrtc] {
        assert!(runtime.prepare(library).is_err());
    }

    // the whole driver path: module load, launch, capture, replay and copies
    let function = runtime.load_embedded(AddOne)?.function("add_one")?;
    let stream = runtime.stream();
    let len = 256_u32;
    let mut data = stream.clone_htod(&vec![0.0_f32; len as usize])?;
    let mut enqueue = || {
        let mut launch = stream.launch_builder(&function);
        launch.arg(&mut data).arg(&len);
        // SAFETY: the kernel reads and writes `len` floats of `data`, which holds that many
        unsafe { launch.launch(LaunchConfig::for_num_elems(len)) }?;
        Ok::<(), Error>(())
    };
    enqueue()?;
    // SAFETY: `data` outlives the graph, which is launched only here
    let graph = unsafe { runtime.capture_nonempty(enqueue) }?;
    graph.launch()?;
    assert!(stream.clone_dtoh(&data)?.iter().all(|value| *value == 2.0));

    let maps = std::fs::read_to_string("/proc/self/maps").expect("readable /proc/self/maps");
    let mapped = |name: &str| {
        maps.lines()
            .filter_map(|line| line.rsplit('/').next())
            .any(|file| file.starts_with(name))
    };
    assert!(mapped("libcuda.so"), "the driver should be mapped");
    let leaked: Vec<_> = FORBIDDEN.into_iter().filter(|name| mapped(name)).collect();
    assert!(leaked.is_empty(), "driver-only process mapped {leaked:?}");
    Ok(())
}
