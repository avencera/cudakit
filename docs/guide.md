# Using cudakit

This guide walks through adopting cudakit in an application that ships its own CUDA
kernels. [speakrs](https://github.com/avencera/speakrs) is a complete example: see its
`src/inference/cuda` and `xtask`. `docs/design.md` explains why things are shaped this
way.

## 1. Dependency and features

```toml
[features]
cuda = ["dep:cudakit", "cudakit/libraries"]         # library-first build
cuda-sm75 = ["dep:cudakit", "dep:myapp-ptx-sm75"]   # one per tier you ship

[dependencies]
cudakit = { version = "0.1", optional = true }
```

Enable cudakit only from your CUDA features, so CPU builds never compile cudarc.

## 2. Profile

A `Profile` is your build policy: the tiers this build compiled in, the tiers you ship
production kernels for, the Cargo feature behind each tier (for error messages), and
whether cuBLAS and cuDNN may load.

```rust
use cudakit::{ExtraSymbols, LibraryPolicy, Profile, PtxTier, TierSet};

pub static PROFILE: Profile = Profile {
    consumer: "myapp",
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

`LibraryPolicy::DriverOnly` refuses every cuBLAS, cuDNN and NVRTC request before
anything is looked up, even when another crate in the binary enabled the `libraries`
feature.

If you call cuBLAS or cuDNN entry points directly, beyond cudakit's wrappers, list them
in `extra_symbols`. The preflight then rejects a library that lacks them before cudarc
loads it.

## 3. Kernel areas

A kernel area is one PTX module. Your area enum says which variants this build embeds:

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
`<app>-ptx-<tier>` crates, which `cudakit-build` generates. Keep the lookups in const
context: a run-time call links every shipped file into the binary. `#[inline(never)]`
stops each codegen unit from copying the embedded bytes.

## 4. Open, prepare, load

```rust
let runtime = cudakit::Runtime::<KernelModule>::open(0, cudakit::Options::default())?;
runtime.prepare(cudakit::Library::Cublas)?;          // at model load, never in capture
let kernels = runtime.load_embedded(KernelModule::Encoder)?;
let softmax = kernels.function("softmax_rows")?;      // PTX entry name = Rust fn name
```

`Runtime` dereferences to `Gpu`, so `runtime.stream()`, `runtime.sgemm(..)` and
`runtime.capture(..)` all work directly on it. Everything it issues runs in order on
its one stream. `load_embedded` picks the exact cubin for the device when you ship one,
otherwise the PTX for the driver to JIT, and never silently substitutes another
artifact.

### Threads

`Runtime` and `Gpu` are `Send + Sync`, so a loaded model can move to a worker thread or
sit behind an `Arc<Mutex<..>>`. `Graph` is `Send` but not `Sync`. cudakit makes the
GPU's context current on whichever thread calls it, so a fresh worker needs no setup.
Two limits:

- Capture needs the GPU to itself: the stream records whatever any thread queues
  during the capture, so serialize capture against other work on the same `Runtime`
- The cuDNN handle is bound to the thread that prepared cuDNN. `Gpu::dnn` returns
  `Error::LibraryOnAnotherThread` elsewhere, and `dnn::ConvPlan`s stay on that thread
  (they are `!Send`). cuBLAS has no such limit

## 5. GEMMs

```rust
use cudakit::{Math, Sgemm, SgemmBatched};

// y[m×n] = x[m×k] · w[n×k]ᵀ
let linear = Sgemm { b_transposed: true, math: Math::Tf32, ..Sgemm::new(m, n, k) };
runtime.sgemm(linear, &x, &w, &mut y)?;

// per-head attention scores: q[h][t×d] · k[h][t×d]ᵀ
let scores = SgemmBatched::packed(Sgemm { b_transposed: true, ..Sgemm::new(t, t, d) }, heads);
runtime.sgemm_batched(scores, &q, &k, &mut s)?;
```

Products are row-major. Each buffer must hold exactly its matrix, and cudakit checks the
lengths. In a batch, a stride of 0 on `a` or `b` reuses one operand for every item, and
output strides must not overlap. `Math` is exhaustive, so you can persist it in tuning
keys and match it.

## 6. Graphs

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

`Graph::as_cuda` exposes the cudarc graph if you need conditional nodes.

## 7. Convolutions

With the `libraries` feature, `dnn::ConvPlanner` picks a cuDNN forward algorithm from
cuDNN's own ranking, without timing runs, so the same GPU, cuDNN version and shape
always get the same algorithm. `ConvPlan::forward_bias_relu` fuses bias, an optional
residual and ReLU into the convolution.

## 8. Errors

`impl From<cudakit::Error> for YourError`. `Error::is_device_unavailable` separates a
missing GPU or driver from a broken install. `Error` is `#[non_exhaustive]`.

## Kernel toolchain

`cudakit-build` builds your cuda-oxide kernels crate to PTX, pins cubins with ptxas,
generates your tier crates and checks the committed artifacts. Your xtask declares one
const `Project` and forwards its subcommands:

```rust
use cudakit_build::{Area, CubinArch, Pins, Project, Tier, TierCrate};

const PROJECT: Project = Project {
    name: "myapp",
    command: "cargo xtask cuda-kernels",
    kernel_crate: "crates/myapp-cuda-kernels",
    kernel_package: "myapp-cuda-kernels",
    manifest_dir: "src/cuda/ptx",
    host_kernels: "src/cuda/kernels.rs",
    host_feature: "cuda-{tier}",
    tier_crate: TierCrate {
        path: "crates/myapp-ptx-{tier}",
        description: "Prebuilt {tier} tier CUDA kernels for myapp",
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

// `build` (PTX + cubins) and `build_cubins` need a GPU box; `check` runs anywhere
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

`scripts/setup-gpu-box.sh --probe <kernel-crate>` prepares an Ubuntu 24.04 NVIDIA box
with the same pins.
