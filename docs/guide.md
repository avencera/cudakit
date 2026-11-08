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
cudakit = { version = "0.3", optional = true }
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
    tier_features: &[
        (PtxTier::Sm75, "cuda-sm75"),
        (PtxTier::Sm80, "cuda-sm80"),
        (PtxTier::Sm90, "cuda-sm90"),
        (PtxTier::Sm120, "cuda-sm120"),
    ],
    libraries: LibraryPolicy::Allowed,
    extra_symbols: ExtraSymbols::NONE,
};
```

`LibraryPolicy::DriverOnly` refuses every cuBLAS, cuBLASLt, cuDNN and NVRTC request before
anything is looked up, even when another crate in the binary enabled the `libraries`
feature.

If you call cuBLAS or cuDNN entry points directly, beyond cudakit's wrappers, list them
in `extra_symbols`. Once cudarc opens a library, the preflight checks its handle and
rejects a library that lacks them before any call into it. `unsafe fn Gpu::blas(math)` hands out the shared cuBLAS handle under its mode
lock; leave its pointer mode, math mode, stream and workspace as they are.

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
            // an sm_80 variant serves Ampere and newer; the baseline keeps Turing
            Self::Decode => cudakit::AreaPtx::baseline(cudakit::embed_tier!(
                shipped_ptx, shipped_cubin;
                ["cuda-sm75"],
                "ptx/decode.sm75",
                [75, 80, 86, 89, 90, 120]
            ))
            .with(cudakit::PtxTier::Sm80, cudakit::embed_tier!(
                shipped_ptx, shipped_cubin;
                ["cuda-sm80", "cuda-sm90", "cuda-sm120"],
                "ptx/decode.sm80",
                [80, 86, 89, 90, 120]
            )),
        }
    }
}
```

Each `embed_tier!` call embeds one variant under exactly the features for which it is
the best one you ship, and is `None` in builds without them. `shipped_ptx` and
`shipped_cubin` are your own `const fn(&str)` lookups over your
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

`Sgemm` runs whatever algorithm cuBLAS picks, which comes from cuBLASLt's heuristic. On
small or odd shapes that choice can leave much of a large GPU idle: on an RTX 5090,
other candidates ran 32-row TF32 products 1.49x and 201-row ones 1.25x faster. A
`MatmulPlan` fixes the algorithm instead, and runs bias, ReLU or GELU in the product's
epilogue:

```rust
use cudakit::{Epilogue, Library, Matmul, MatmulCache, Operands, Output, Scale, Tuning};

runtime.prepare(Library::CublasLt)?;            // at model load, never in capture

// a BF16 `Linear` with a fused bias and GELU, written as BF16
let spec = Matmul::linear(rows, outputs, inputs, Operands::Bf16)
    .with_output(Output::Bf16)
    .with_epilogue(Epilogue::GeluBias);

// one plan per shape; with a tuning, a new shape times the heuristic's top 8 and
// keeps its first unless another is over 5% faster
let plans = MatmulCache::new(Some(Tuning::default()));
let plan = plans.plan(&runtime, spec)?;
runtime.matmul_bias(&plan, Scale::OVERWRITE, &x, &w, &bias, &mut y)?;
```

`a` and `b` hold the `Operands` type: `f32` for `F32(Math)` and `F32Fast16Bf`, `u16`
bit patterns for `F16` and `Bf16`. `c` and the bias hold the `Output` type, FP32 by
default; a 16-bit output needs operands of the same type, and cuBLASLt takes the bias in
the output's type, so a BF16 product needs a BF16 bias. GELU is the tanh approximation.
With TF32 operands on an RTX 5090, cuBLASLt ran GELU as the GEMM plus a separate
epilogue kernel, while bias alone fused; check nsys before counting on GELU fusing.

`with_strides` describes views with wider rows, such as one head's columns of a packed
QKV buffer. A stride below the row length would make rows overlap, which cuBLASLt
rejects, so cudakit refuses it with `Error::Unsupported`: express a sliding-window
product (a 3×3 convolution as 3 matmuls over windows that slide by one row) as one
product per tap instead. Every buffer must hold exactly its stored matrix,
`(rows - 1) * stride + columns` elements, and start on a `MATMUL_ALIGNMENT` (16-byte)
boundary, which whole allocations do; cudakit asks the heuristic only for algorithms
that work at that alignment, so an offset view fails with `Error::Unsupported` where
`Sgemm` would run. Buffers can be any cudarc `DevicePtr`, so memory from your own pool
works through a small wrapper type.

A fixed algorithm also fixes the summation order, and so the output bits. Tuning by
timing picks per process, and the 5% margin keeps near-ties from flipping on noise,
but a candidate near the margin can still win in one process and lose in the next.
Where outputs must match across runs, persist the chosen algorithm, keyed on the device
and `runtime.cublaslt_version()?`, and build from it:

```rust
let (plan, report) = runtime.tune_matmul(spec, &Tuning::default())?;
store(spec, plan.algo());                       // `MatmulAlgo` is serde-able
// at load: the stored algorithm, or the heuristic's first when it is rejected
let plan = runtime.matmul_plan_with(spec, stored).or_else(|_| runtime.matmul_plan(spec))?;
plans.insert(&runtime, plan)?;
```

Tuning runs every candidate on scratch buffers of constant values and synchronizes, so
time it on an idle GPU, and never during a capture: `tune_matmul`, and a tuned
`MatmulCache` meeting a new shape, fail with `Error::Capturing` there. Run the model
once eagerly before capturing so every shape is cached. Plans hold host descriptors
only, are `Send + Sync`, and may be recorded in a graph: every plan shares one
workspace of `MATMUL_WORKSPACE_BYTES` (32 MiB) that lives as long as the `Gpu`. Matmul
calls and `Graph::launch` queue under one lock, so threads sharing a `Gpu` can run
products and replay graphs at once; launching a graph through `Graph::as_cuda` skips
that lock.

## 6. Graphs

```rust
// buffers allocated once; run eagerly first so cuBLAS finishes lazy setup and every
// matmul shape is tuned
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
graph that recorded them before its next launch.

Replays record no cudarc events, so a buffer a replay touches stays allocated and on
the GPU's stream until that replay finishes, and dropping the graph does not wait.
cudarc frees a buffer in order on the stream it was allocated on, so buffers allocated
on the GPU's stream may simply be dropped; synchronize the stream before touching one
from another stream or freeing one allocated elsewhere. Buffers allocated inside the
closure carry no events for their whole life.
`capture` returns `None` when the closure queued nothing, and a panic inside the
closure leaves the stream capturing.

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
    // the tiers you have features for; a later cudakit-build tier is ignored until listed
    targets: &[Tier::Sm75, Tier::Sm80, Tier::Sm90, Tier::Sm120],
    tier_crate: TierCrate {
        path: "crates/myapp-ptx-{tier}",
        description: "Prebuilt {tier} tier CUDA kernels for myapp",
        example_area: "encoder",
    },
    areas: &[
        Area::new("encoder", &[Tier::Sm75]),
        Area::new("decode", &[Tier::Sm75, Tier::Sm80]),
    ],
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
- `build` and `build_cubins` take ptxas from `$CUDA13_HOME/bin` or
  `$CUDA_TOOLKIT_PATH/bin`, never from `PATH`. On Windows that is `ptxas.exe`; set
  `CUDA13_HOME` to the toolkit, such as `%CUDA_PATH_V13_0%`
- `check` runs on Windows too, given LF line endings: manifests hash the committed
  bytes, so put `* text=auto eol=lf` in your `.gitattributes`. When a hash fails and a
  file it covers has CRLF endings, `check` names the file. `build` has only run on Linux

`scripts/setup-gpu-box.sh --probe <kernel-crate>` prepares an Ubuntu 24.04 NVIDIA box
with the same pins.
