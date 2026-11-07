# Changelog

## 0.2.0

Windows x86_64 support. CI builds and tests it, but nothing has run on a Windows GPU
yet.

### cudakit

- cudakit now depends on [`cudakit-cudarc`](https://crates.io/crates/cudakit-cudarc)
  0.19.10, a cudarc fork with library-loading fixes, in place of `cudarc`. Its library
  is still `cudarc`, re-exported as `cudakit::cudarc`. Consumers that also depend on
  upstream `cudarc` directly should use `cudakit::cudarc` instead
- On Linux, versioned sonames such as `libcublas.so.13` and `libcublas.so.12` open
  before a toolkit's unversioned `libcublas.so`, so a process no longer loads two
  copies of cuBLAS or pairs a cuBLAS with another install's cuBLASLt
- Windows x86_64 with the MSVC toolchain. The driver is `nvcuda.dll` from System32.
  The `libraries` feature opens CUDA 12's or CUDA 13's cuBLAS and NVRTC
  (`cublas64_12.dll`, `cublas64_13.dll`, `nvrtc64_120_0.dll`, `nvrtc64_130_0.dll`) and
  cuDNN 9's `cudnn64_9.dll` from the executable's directory, System32, absolute `PATH`
  entries, `%CUDA_PATH%\bin\x64` and `%CUDA_PATH%\bin`, never from the current
  directory or a relative `PATH` entry
- The symbol preflight checks the library handle cudarc itself opened, instead of
  repeating cudarc's search
- `Error::LibraryUnavailable` names the files cudakit looks for and where: `PATH` and
  `%CUDA_PATH%` on Windows, `LD_LIBRARY_PATH` and the ldconfig cache on Linux. On
  Windows the driver displays as `CUDA driver (nvcuda.dll)`
- `identity::DriverIdentity::read` opens `nvml.dll` from System32 on Windows. It used
  to look for NVML only under its Linux name and then read `\sys\...` and `\proc\...`
  on the current drive. It now reads sysfs and procfs only on Linux

### cudakit-build

- `build` and `build_cubins` find `ptxas.exe` on Windows
- When a sources, PTX or tier crate hash fails, `check` names any file it covers that
  has CRLF line endings, as a Windows checkout without `eol=lf` makes. Repositories
  checked out on Windows need `* text=auto eol=lf` in `.gitattributes`
- Generated-file paths in messages use `/` on every OS

## 0.1.0

First release.

### cudakit

- `Gpu` and `Runtime<K>`: one context and stream per device, with a module cache for
  an application's kernel areas that loads exact cubins or PTX by identity and never
  substitutes another artifact. Both are `Send + Sync`, and cudakit makes the context
  current on whichever thread calls it
- `Profile` and `LibraryPolicy`: build policy checked at open, including driver-only
  operation that refuses every library before anything is looked up
- `PtxTier`, `TierSet` and `AreaPtx` leave room for new tiers: `PtxTier::ALL` is a
  slice, `AreaPtx` is built with `EMPTY`, `baseline` and `with`, and a profile names
  features only for the tiers it ships
- cuBLAS and cuDNN loaded at run time after a symbol preflight (feature `libraries`)
- `Sgemm` and `SgemmBatched`: row-major products with length checks, and
  `unsafe fn Gpu::blas` for other cuBLAS calls on the shared handle
- `Gpu::capture` and `Graph`: graph capture that pauses cudarc's event tracking and
  detects empty captures
- `dnn`: a deterministic cuDNN convolution planner with fused bias, residual and ReLU,
  and `Conv2d::check` for convolution geometry
- `DeviceTensor`, `unfold_windows` and `identity::DriverRelease`

### cudakit-build

- `Project` with `build`, `build_cubins` and `check` for cuda-oxide kernel crates,
  pinned cubins, per-area manifests and generated PTX tier crates
- `Project::targets`: the tiers the consumer has features for, so a tier added to
  cudakit-build later is not required of existing consumers
- `ptx_lint`, which rejects generic shared-memory addresses narrowed without a
  conversion to the shared space
