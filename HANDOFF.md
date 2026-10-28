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
| `cudakit-build` (kernel toolchain library) | **Not ready yet.** Being ported from speakrs's xtask |

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
the binary. Until `cudakit-build` lands, `embed_tier!` is usable with hand-built PTX.

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
// SAFETY: the encoder owns every captured buffer and drops this graph before them
let graph = unsafe { runtime.capture_nonempty(|| encoder.enqueue(&runtime)) }?;
for chunk in batches {
    input.copy_from_host(runtime.stream(), chunk)?;   // refill, never reallocate
    graph.launch()?;
}
```

The graph holds raw pointers to every captured buffer, which is why `capture` and
`capture_nonempty` are `unsafe`: every buffer must outlive the graph at the same address
and stay on the GPU's stream. Store the graph beside the buffers' owner, declared before
the buffers (or dropped explicitly first). `capture` returns `None` when the closure
queued nothing.

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

## Kernel toolchain (pending)

`cudakit-build` will give your xtask the speakrs toolchain as a library:

- variant builds with the pinned cuda-oxide rev and nightly
- ptxas pinning and cubin manifests
- entry-point and header checks, and PTX lint
- tier-crate rendering and the host embed-mask check

You configure it with a `Project` that names your kernels crate, areas and pins. This
section will be filled in when it lands.
