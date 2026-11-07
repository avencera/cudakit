# cudakit

[![crates.io](https://img.shields.io/crates/v/cudakit.svg)](https://crates.io/crates/cudakit)
[![docs.rs](https://docs.rs/cudakit/badge.svg)](https://docs.rs/cudakit)
[![CI](https://github.com/avencera/cudakit/actions/workflows/ci.yml/badge.svg)](https://github.com/avencera/cudakit/actions/workflows/ci.yml)

**A CUDA host stack that needs only the NVIDIA driver.** cudakit loads `libcuda`
(`nvcuda.dll` on Windows) at run time and never links or loads the CUDA Runtime
(`libcudart`). Building needs no CUDA
toolkit, and running needs only the driver. cuBLAS and cuDNN are optional and also
loaded at run time. Each one is loaded only when the application's policy allows it,
and only after a symbol check passes.

cudakit is the shared GPU layer of [speakrs](https://github.com/avencera/speakrs)
(speaker diarization) and [scriptrs](https://github.com/avencera/scriptrs) (speech
recognition). Each application ships its own kernels, written with
[cuda-oxide](https://github.com/NVlabs/cuda-oxide) and embedded as PTX plus
exact-architecture cubins.

## What it provides

- `Gpu`: one context and one stream, device attributes, cooperative-launch capacity, and
  lazily prepared cuBLAS and cuDNN handles
- `Runtime<K>`: a `Gpu` plus a module cache for your `KernelArea` enum. It loads
  embedded cubins or PTX by exact identity and never falls back silently
- `Profile`: your build policy (compiled tiers, shipped tiers, feature names, and
  driver-only or library use), checked when the GPU opens
- `DeviceTensor`: a shaped device buffer
- `Sgemm` and `SgemmBatched`: row-major cuBLAS products with length checks
- `Gpu::capture`: CUDA graph capture that pauses cudarc's event tracking and ends the
  capture even when the enqueue fails
- `dnn`: a deterministic cuDNN convolution planner with fused bias, residual and ReLU
- `identity`: the driver release, for persisted tuning keys

`Gpu` and `Runtime` are `Send + Sync`, and cudakit makes the context current on
whichever thread calls it.

## Usage

```toml
[dependencies]
cudakit = { version = "0.1", features = ["libraries"] }
```

```rust
let runtime = cudakit::Runtime::<MyKernels>::open(0, cudakit::Options::default())?;
runtime.prepare(cudakit::Library::Cublas)?;
let kernels = runtime.load_embedded(MyKernels::Encoder)?;
let softmax = kernels.function("softmax_rows")?;

// y[m×n] = x[m×k] · w[n×k]ᵀ
let linear = cudakit::Sgemm { b_transposed: true, ..cudakit::Sgemm::new(m, n, k) };
runtime.sgemm(linear, &x, &w, &mut y)?;
```

The [guide](https://github.com/avencera/cudakit/blob/master/docs/guide.md) covers the
whole setup: the profile, kernel areas, threads, graphs and the kernel toolchain.

## Features

| Feature | Adds |
| --- | --- |
| (none) | Driver API only |
| `libraries` | cuBLAS and cuDNN through cudarc, the symbol preflight, the GEMMs and `dnn` |
| `serde` | serde for `identity::DriverRelease` |
| `test-support` | Fixtures for host-only tests |

Even with `libraries` enabled, a driver-only `Profile` refuses every library before
anything is looked up. Policy is a run-time value rather than a Cargo feature, so two
crates with different policies can share one binary.

Make cudakit an optional dependency that only your CUDA features enable. That keeps CPU
builds free of cudarc.

## Requirements

- Linux, or Windows x86_64 (built and tested in CI, not yet run on a GPU), with an
  NVIDIA driver for CUDA 12.4 or newer
- A GPU of compute capability 7.5 (Turing) or newer
- For the `libraries` feature only: cuBLAS and cuDNN 9 (tested on Linux with CUDA 13
  and cuDNN 9.27)

### Windows

Windows x86_64 builds with the MSVC toolchain, and CI runs clippy and the unit tests
there, but nothing has run on a Windows GPU yet.

The driver is `nvcuda.dll`, which the NVIDIA driver installs. With the `libraries`
feature, cudakit loads the names cudarc looks for, which on Windows are CUDA 10 to 12
names: it needs CUDA 12's `cublas64_12.dll` (with `cublasLt64_12.dll` beside it) and
`nvrtc64_120_0.dll`, and cuDNN 9's `cudnn64_9.dll`. Put the directory holding them on
`PATH` as an absolute path; a CUDA 12 toolkit keeps its DLLs in
`%CUDA_PATH_V12_x%\bin`. cudarc doesn't look for CUDA 13's `cublas64_13.dll` or `nvrtc64_130_0.dll` (in
`%CUDA_PATH%\bin\x64`), so a machine with only CUDA 13's libraries can run driver-only
work but not cuBLAS or NVRTC. When a library is missing, the error names the files and
`PATH`.

Windows' default DLL search looks in the current directory before `PATH`, so a DLL
planted in the folder a program runs from would load. Before cudarc opens the driver,
cuBLAS, cuDNN or NVRTC, cudakit finds the file that search would pick and loads it by
full path. It fails with `Error::UntrustedLibrary` instead when a copy of a name it
tries is anywhere in the current directory or a relative `PATH` entry, unless that
directory is also the executable's directory, a Windows directory or an absolute `PATH`
entry. cuBLAS's own import of cuBLASLt then comes from its directory, unless a cuBLASLt
of that name is already loaded. cuDNN loads its sub-libraries itself, which cudakit
doesn't control, and `cudarc::cublaslt` calls a consumer makes directly load cuBLASLt
through cudarc's search, so still run programs from directories you trust.

`identity::DriverIdentity::read` opens `nvml.dll` from System32 only; Windows has no
sysfs or procfs fallback.

## Kernel toolchain

[`cudakit-build`](https://crates.io/crates/cudakit-build) is the offline toolchain that
applications' xtasks call. It builds cuda-oxide kernels to PTX, pins cubins with ptxas,
generates the PTX tier crates and checks the committed artifacts.
`scripts/setup-gpu-box.sh` prepares an Ubuntu 24.04 NVIDIA box for those builds.
Repositories that check artifacts on Windows need `* text=auto eol=lf` in
`.gitattributes`, since the manifests hash the committed bytes.

## Development

```sh
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
# GPU tests run only with CUDAKIT_REQUIRE_GPU=1 (and then fail without a GPU), so the
# workspace tests above never take a shared GPU outside its scheduler
CUDAKIT_REQUIRE_GPU=1 cargo test -p cudakit --features "libraries test-support" --tests
```

See [`docs/design.md`](https://github.com/avencera/cudakit/blob/master/docs/design.md)
for the design.

## License

Apache-2.0
