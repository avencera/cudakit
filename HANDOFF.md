# cudakit handoff

For consumers adopting cudakit; scriptrs is the first new one. See `docs/design.md` for
why things are shaped this way.

## Stability

| Part | Status |
| --- | --- |
| `Gpu`, `Runtime<K>`, `Options`, `Profile`, `LibraryPolicy`, `ExtraSymbols` | Stable for consumers. speakrs runs on it |
| `KernelArea`, `AreaPtx`, `EmbeddedPtx`, `EmbeddedCubin`, `embed_tier!`, `ModuleRequest`, `LoadedKernels` | Stable. speakrs's module registry is built on it |
| `DeviceTensor`, `unfold_windows` | Stable |
| `Sgemm`, `Gpu::sgemm` | Stable. Moved unchanged from speakrs |
| `SgemmBatched`, `Gpu::sgemm_batched` | New. Covered by GPU tests in `crates/cudakit/tests/gpu.rs`, but no production consumer yet |
| `Gpu::capture`, `capture_nonempty`, `Graph` | Stable. All three speakrs capture sites use them. Capture became `unsafe` after `0a36e85` |
| `dnn::{ConvPlanner, ConvPlan, Residual}`, `Conv2d` | Stable. speakrs's convolution fallback |
| `identity::{DriverIdentity, DriverRelease}` | Stable |
| `Error` | `#[non_exhaustive]`. New variants may appear |
| `cudakit-build` (`Project`, `Area`, `Tier`, `Pins`, `TierCrate`, `CubinArch`) | Usable. speakrs's `cargo xtask cuda-kernels` runs on it and its committed artifacts check unchanged. Not published |

`Math` is exhaustive on purpose. Consumers persist it, so adding a variant is a
breaking change.

## Consumer recipe

### 1. Dependency and features

```toml
[features]
cuda = ["dep:cudakit", "cudakit/libraries"]   # library-first build
cuda-sm75 = ["dep:cudakit", "dep:myapp-ptx-sm75"]   # one per tier you ship

[dependencies]
cudakit = { path = "../cudakit/crates/cudakit", optional = true }
```

Enable cudakit only from CUDA features, so CPU and CoreML builds never compile cudarc.

### 2. Profile

```rust
use cudakit::{ExtraSymbols, LibraryPolicy, Profile, PtxTier, TierSet};

pub static PROFILE: Profile = Profile {
    consumer: "scriptrs",
    compiled: TierSet::EMPTY
        .with_if(PtxTier::Sm75, cfg!(feature = "cuda-sm75"))
        .with_if(PtxTier::Sm80, cfg!(feature = "cuda-sm80"))
        .with_if(PtxTier::Sm90, cfg!(feature = "cuda-sm90"))
        .with_if(PtxTier::Sm120, cfg!(feature = "cuda-sm120")),
    shipped: TierSet::of(&[PtxTier::Sm75]),
    tier_features: ["cuda-sm75", "cuda-sm80", "cuda-sm90", "cuda-sm120"],
    libraries: LibraryPolicy::Allowed,
    extra_symbols: ExtraSymbols::NONE,
};
```

If you call cuBLAS or cuDNN entry points directly, beyond cudakit's wrappers, list them
in `extra_symbols`. The preflight then rejects a library that lacks them before cudarc
loads it. speakrs, for example, lists the cuDNN RNN API.

### 3. Kernel areas

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelModule { Encoder, Decode }

impl cudakit::KernelArea for KernelModule {
    const PROFILE: &'static cudakit::Profile = &PROFILE;

    fn name(self) -> &'static str {
        match self { Self::Encoder => "encoder", Self::Decode => "decode" }
    }

    #[inline(never)]
    fn variants(self) -> cudakit::AreaPtx {
        match self {
            Self::Encoder => cudakit::AreaPtx::baseline(cudakit::embed_tier!(
                shipped_ptx, shipped_cubin;
                ["cuda-sm75", "cuda-sm80", "cuda-sm90", "cuda-sm120"],
                "ptx/encoder.sm75",
                [75, 80, 86, 89, 90, 120]
            )),
            // ...
        }
    }
}
```

`shipped_ptx` and `shipped_cubin` are your own `const fn(&str)` lookups over your
`<app>-ptx-<tier>` crates. speakrs's `src/inference/cuda/kernels.rs` is the model to
copy. Keep the lookups in const context: a run-time call links every shipped file into
the binary.

### 4. Open, prepare, load

```rust
let runtime = cudakit::Runtime::<KernelModule>::open(0, cudakit::Options::default())?;
runtime.prepare(cudakit::Library::Cublas)?;          // at model load, never in capture
let kernels = runtime.load_embedded(KernelModule::Encoder)?;
let softmax = kernels.function("softmax_rows")?;      // PTX entry name = Rust fn name
```

`Runtime` dereferences to `Gpu`, so `runtime.stream()`, `runtime.sgemm(..)` and
`runtime.capture(..)` all work directly on it. Use one `Runtime` per worker thread.
Everything it issues runs in order on its one stream.

### 5. GEMMs

```rust
use cudakit::{Math, Sgemm, SgemmBatched};

// y[m×n] = x[m×k] · w[n×k]ᵀ
let linear = Sgemm { b_transposed: true, math: Math::Tf32, ..Sgemm::new(m, n, k) };
runtime.sgemm(linear, &x, &w, &mut y)?;

// per-head attention scores: q[h][t×d] · k[h][t×d]ᵀ
let scores = SgemmBatched::packed(Sgemm { b_transposed: true, ..Sgemm::new(t, t, d) }, heads);
runtime.sgemm_batched(scores, &q, &k, &mut s)?;
```

A stride of 0 on `a` or `b` reuses one operand for every item. Output strides must not
overlap.

### 6. Graphs

```rust
// buffers allocated once; run eagerly first so cuBLAS finishes lazy setup
encoder.enqueue(&runtime)?;
// SAFETY: the encoder owns every captured buffer and never reallocates them, and the
// graph is launched only while the encoder is alive
let graph = unsafe { runtime.capture_nonempty(|| encoder.enqueue(&runtime)) }?;
for chunk in batches {
    input.copy_from_host(runtime.stream(), chunk)?;   // refill, never reallocate
    graph.launch()?;
}
```

The graph holds raw pointers to every captured buffer, which is why `capture` and
`capture_nonempty` are `unsafe`: every launch must find each buffer allocated at its
captured address, and the buffers stay on the GPU's stream. The simplest way to keep
that is to store the graph beside the buffers' owner and launch it only through that
owner. Dropping a graph only destroys driver handles, so a graph that can no longer be
launched may outlive its buffers. If you reallocate buffers, drop or recapture every
graph that recorded them before its next launch. `capture` returns `None` when the
closure queued nothing, and a panic inside the closure leaves the stream capturing.

## What scriptrs should call (Phase 3)

- Model load: `Runtime::open`, `prepare(Library::Cublas)`, `DeviceTensor::upload` for
  the weights, `load_embedded` for each kernel area
- Encoder: `Sgemm` for the projections and feed-forward layers, `SgemmBatched` for the
  attention products, your own kernels for softmax, norms and activations, then one
  `capture_nonempty` per fixed chunk shape
- TDT decode: capture one decode step (predictor, joint, argmax, duration advance)
  that reads and writes its control state in device buffers, and replay it with
  `Graph::launch` without host syncs. `Graph::as_cuda` exposes the cudarc graph if you
  later need conditional nodes
- Errors: `impl From<cudakit::Error> for YourError`. `Error::is_device_unavailable`
  separates a missing GPU or driver from a broken install

## Kernel toolchain

`cudakit-build` is the speakrs toolchain as a library. Your xtask declares one const
`Project` and forwards its subcommands:

```rust
use cudakit_build::{Area, CubinArch, Pins, Project, Tier, TierCrate};

const PROJECT: Project = Project {
    name: "scriptrs",
    command: "cargo xtask cuda-kernels",
    kernel_crate: "crates/scriptrs-cuda-kernels",
    kernel_package: "scriptrs-cuda-kernels",
    manifest_dir: "src/cuda/ptx",
    host_kernels: "src/cuda/kernels.rs",
    host_feature: "cuda-{tier}",
    tier_crate: TierCrate {
        path: "crates/scriptrs-ptx-{tier}",
        description: "Prebuilt {tier} tier CUDA kernels for scriptrs",
        example_area: "encoder",
    },
    areas: &[Area::new("encoder", &[Tier::Sm75]), Area::new("decode", &[Tier::Sm75])],
    // write the pins out in full, not `Pins::CURRENT`: a cudakit-build default bump would
    // otherwise make every committed artifact stale
    pins: Pins {
        cuda_oxide_rev: "918bbde123671153b29a89542f1781f1c7494c14",
        cuda_oxide_nightly: "nightly-2026-08-28",
        max_ptx_isa: (9, 0),
        ptxas: "Cuda compilation tools, release 13.0, V13.0.88",
    },
    cubin_arches: &[
        CubinArch(75), CubinArch(80), CubinArch(86), CubinArch(89), CubinArch(90), CubinArch(120),
    ],
};

// `build` (PTX + cubins) and `build-cubins` need the GPU box; `check` runs anywhere
PROJECT.check(&workspace_root, false)?;
```

- `build` regenerates PTX with the pinned cuda-oxide rev and nightly, then cubins with
  the pinned ptxas, and rewrites the tier crates' `lib.rs`
- `build_cubins` rebuilds cubins from the committed PTX bytes only
- `check` verifies source and PTX hashes, PTX headers and `MAX_PTX_ISA`, entry points
  across an area's variants, the cubin matrix and hashes, the generated tier crates, and
  the feature and architecture masks of every `tier_ptx!` call in `host_kernels`.
  `check(root, true)` also rebuilds cubins and requires identical bytes
- `cubin_arches` must be strictly ascending; every entry point rejects any other order
- the host check scans every `tier_ptx!(` call outside `//` comments, macro bodies
  included, so write your wrapper as a pass-through
  (`($($variant:tt)*) => { cudakit::embed_tier!(..; $($variant)*) }`)
- `ptx_lint` is public for consumers that lint PTX outside `check`

`scripts/setup-gpu-box.sh --probe <kernel-crate>` prepares the box with the same pins.
