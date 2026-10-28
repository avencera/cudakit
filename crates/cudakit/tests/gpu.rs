//! GPU checks of the shared host stack: module loading, graph capture and the cuBLAS
//! products
//!
//! Each test skips when no usable device is present, unless `CUDAKIT_REQUIRE_GPU=1`
//! makes that a failure, so a GPU runner cannot silently pass

use cudakit::cudarc::driver::{LaunchConfig, PushKernelArg};
use cudakit::{
    AreaPtx, Error, ExtraSymbols, Gpu, KernelArea, Library, LibraryPolicy, LoadedArtifact, Options,
    Profile, PtxTier, Runtime, Sgemm, SgemmBatched, TierSet,
};

const BASE: Profile = Profile {
    consumer: "cudakit-tests",
    compiled: TierSet::of(&[PtxTier::Sm75]),
    shipped: TierSet::of(&[PtxTier::Sm75]),
    tier_features: ["test-sm75", "test-sm80", "test-sm90", "test-sm120"],
    libraries: LibraryPolicy::Allowed,
    extra_symbols: ExtraSymbols::NONE,
};

static PROFILE: Profile = BASE;

static DRIVER_ONLY: Profile = Profile {
    libraries: LibraryPolicy::DriverOnly,
    ..BASE
};

// hand-written so the test needs no kernel toolchain; `data[i] += 1` for `i < len`
const ADD_ONE_PTX: &str = r#"
.version 7.0
.target sm_75
.address_size 64

.visible .entry add_one(
    .param .u64 data,
    .param .u32 len
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .f32 %f<3>;
    .reg .b64 %rd<5>;

    ld.param.u64 %rd1, [data];
    ld.param.u32 %r1, [len];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;
    cvta.to.global.u64 %rd2, %rd1;
    mul.wide.u32 %rd3, %r5, 4;
    add.s64 %rd4, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4];
    add.f32 %f2, %f1, 0f3F800000;
    st.global.f32 [%rd4], %f2;
DONE:
    ret;
}
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Area {
    AddOne,
}

impl KernelArea for Area {
    const PROFILE: &'static Profile = &PROFILE;

    fn name(self) -> &'static str {
        "add-one"
    }

    fn variants(self) -> AreaPtx {
        AreaPtx::fixture(&[(PtxTier::Sm75, ADD_ONE_PTX)])
    }
}

/// `None` when the machine has no usable GPU and the run does not require one
fn opened<T>(result: Result<T, Error>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(error)
            if error.is_device_unavailable()
                && std::env::var_os("CUDAKIT_REQUIRE_GPU").is_none_or(|value| value != "1") =>
        {
            eprintln!("skipping: {error}");
            None
        }
        Err(error) => panic!("opening the GPU failed: {error}"),
    }
}

fn runtime() -> Option<Runtime<Area>> {
    opened(Runtime::open(0, Options::default()))
}

/// Row-major `op(a) · op(b)` on the host
fn reference(spec: Sgemm, a: &[f32], b: &[f32]) -> Vec<f32> {
    let (m, n, k) = (spec.m, spec.n, spec.k);
    let at = |row: usize, col: usize| {
        if spec.a_transposed {
            a[col * m + row]
        } else {
            a[row * k + col]
        }
    };
    let bt = |row: usize, col: usize| {
        if spec.b_transposed {
            b[col * k + row]
        } else {
            b[row * n + col]
        }
    };

    (0..m * n)
        .map(|index| {
            let (row, col) = (index / n, index % n);
            (0..k).map(|inner| at(row, inner) * bt(inner, col)).sum()
        })
        .collect()
}

fn values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|index| ((index * 7 + seed * 13) % 17) as f32 / 8.0 - 1.0)
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-4 * expected.abs().max(1.0),
            "element {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn embedded_ptx_loads_once_and_replays_in_a_graph() -> Result<(), Error> {
    let Some(runtime) = runtime() else {
        return Ok(());
    };
    let kernels = runtime.load_embedded(Area::AddOne)?;
    assert_eq!(kernels.tier(), PtxTier::Sm75);
    assert!(matches!(kernels.artifact(), LoadedArtifact::PtxJit { .. }));
    // a second load returns the cached module
    assert_eq!(
        runtime.load_embedded(Area::AddOne)?.request(),
        kernels.request()
    );

    let function = kernels.function("add_one")?;
    assert!(matches!(
        kernels.function("missing"),
        Err(Error::KernelMissing { .. })
    ));

    let len = 1000_u32;
    let stream = runtime.stream();
    // allocated and written before the capture, so a tracked buffer would wait on a
    // pre-capture event inside it
    let mut data = stream.clone_htod(&vec![0.0_f32; len as usize])?;
    let graph = runtime.capture_nonempty(|| {
        let mut launch = stream.launch_builder(&function);
        launch.arg(&mut data).arg(&len);
        // SAFETY: the kernel reads and writes `len` floats of `data`, which holds that many
        unsafe { launch.launch(LaunchConfig::for_num_elems(len)) }?;
        Ok::<(), Error>(())
    })?;
    assert!(
        stream.clone_dtoh(&data)?.iter().all(|value| *value == 0.0),
        "capture must record, not run"
    );

    graph.launch()?;
    graph.launch()?;
    assert!(stream.clone_dtoh(&data)?.iter().all(|value| *value == 2.0));
    Ok(())
}

#[test]
fn capture_reports_empty_work_and_survives_a_failed_enqueue() -> Result<(), Error> {
    let Some(runtime) = runtime() else {
        return Ok(());
    };
    assert!(runtime.capture(|| Ok::<(), Error>(()))?.is_none());
    assert!(matches!(
        runtime.capture_nonempty(|| Ok::<(), Error>(())),
        Err(Error::EmptyCapture)
    ));

    let failed = runtime.capture(|| {
        Err::<(), Error>(Error::Unsupported {
            context: "test",
            reason: "enqueue failed".into(),
        })
    });
    assert!(matches!(failed, Err(Error::Unsupported { .. })));

    // the stream left capture mode, so ordinary work still runs
    let stream = runtime.stream();
    let data = stream.clone_htod(&[1.0_f32, 2.0])?;
    assert_eq!(stream.clone_dtoh(&data)?, [1.0, 2.0]);
    runtime.synchronize()
}

#[test]
fn sgemm_matches_the_row_major_reference() -> Result<(), Error> {
    let Some(runtime) = runtime() else {
        return Ok(());
    };
    runtime.prepare(Library::Cublas)?;
    let stream = runtime.stream();

    for (a_transposed, b_transposed) in [(false, false), (true, false), (false, true), (true, true)]
    {
        let spec = Sgemm {
            a_transposed,
            b_transposed,
            ..Sgemm::new(5, 7, 3)
        };
        let a = values(spec.m * spec.k, 1);
        let b = values(spec.k * spec.n, 2);
        let (a_device, b_device) = (stream.clone_htod(&a)?, stream.clone_htod(&b)?);
        let mut c = stream.alloc_zeros::<f32>(spec.m * spec.n)?;
        runtime.sgemm(spec, &a_device, &b_device, &mut c)?;
        assert_close(&stream.clone_dtoh(&c)?, &reference(spec, &a, &b));
    }

    // beta keeps the existing output
    let spec = Sgemm {
        beta: 1.0,
        ..Sgemm::new(4, 4, 4)
    };
    let (a, b) = (values(16, 3), values(16, 4));
    let mut c = stream.clone_htod(&[1.0_f32; 16])?;
    runtime.sgemm(
        spec,
        &stream.clone_htod(&a)?,
        &stream.clone_htod(&b)?,
        &mut c,
    )?;
    let expected: Vec<f32> = reference(spec, &a, &b)
        .iter()
        .map(|value| value + 1.0)
        .collect();
    assert_close(&stream.clone_dtoh(&c)?, &expected);

    let short = stream.alloc_zeros::<f32>(3)?;
    assert!(matches!(
        runtime.sgemm(spec, &short, &short, &mut c),
        Err(Error::BufferLength { .. })
    ));
    Ok(())
}

#[test]
fn batched_sgemm_matches_each_product() -> Result<(), Error> {
    let Some(runtime) = runtime() else {
        return Ok(());
    };
    runtime.prepare(Library::Cublas)?;
    let stream = runtime.stream();

    let gemm = Sgemm {
        b_transposed: true,
        ..Sgemm::new(6, 4, 5)
    };
    let batch = 3;
    let spec = SgemmBatched::packed(gemm, batch);
    let a = values(spec.stride_a * batch, 5);
    let b = values(spec.stride_b * batch, 6);
    let mut c = stream.alloc_zeros::<f32>(spec.stride_c * batch)?;
    runtime.sgemm_batched(
        spec,
        &stream.clone_htod(&a)?,
        &stream.clone_htod(&b)?,
        &mut c,
    )?;

    let expected: Vec<f32> = (0..batch)
        .flat_map(|item| {
            reference(
                gemm,
                &a[item * spec.stride_a..][..spec.stride_a],
                &b[item * spec.stride_b..][..spec.stride_b],
            )
        })
        .collect();
    assert_close(&stream.clone_dtoh(&c)?, &expected);

    // a shared weight: stride 0 reuses one `b` for every item
    let shared = SgemmBatched {
        stride_b: 0,
        ..spec
    };
    let weight = &b[..spec.stride_b];
    runtime.sgemm_batched(
        shared,
        &stream.clone_htod(&a)?,
        &stream.clone_htod(weight)?,
        &mut c,
    )?;
    let expected: Vec<f32> = (0..batch)
        .flat_map(|item| reference(gemm, &a[item * spec.stride_a..][..spec.stride_a], weight))
        .collect();
    assert_close(&stream.clone_dtoh(&c)?, &expected);

    let overlapping = SgemmBatched {
        stride_c: 1,
        ..spec
    };
    let mut small = stream.alloc_zeros::<f32>(spec.stride_c + batch - 1)?;
    assert!(matches!(
        runtime.sgemm_batched(
            overlapping,
            &stream.clone_htod(&a)?,
            &stream.clone_htod(&b)?,
            &mut small
        ),
        Err(Error::Unsupported { .. })
    ));
    Ok(())
}

#[test]
fn driver_only_profile_refuses_libraries_before_loading_them() -> Result<(), Error> {
    let Some(gpu) = opened(Gpu::open(0, &DRIVER_ONLY, None)) else {
        return Ok(());
    };
    for library in [Library::Cublas, Library::Cudnn, Library::Nvrtc] {
        assert!(matches!(
            gpu.prepare(library),
            Err(Error::LibraryForbidden { library: refused }) if refused == library
        ));
    }

    let stream = gpu.stream();
    let a = stream.alloc_zeros::<f32>(4)?;
    let mut c = stream.alloc_zeros::<f32>(4)?;
    assert!(matches!(
        gpu.sgemm(Sgemm::new(2, 2, 2), &a, &a, &mut c),
        Err(Error::LibraryForbidden {
            library: Library::Cublas
        })
    ));
    gpu.prepare(Library::Driver)
}
