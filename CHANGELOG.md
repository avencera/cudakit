# Changelog

## 0.1.1

Windows x86_64 support. CI builds and tests it, but nothing has run on a Windows GPU
yet.

### cudakit

- Windows x86_64 with the MSVC toolchain. The driver is `nvcuda.dll`, and the
  `libraries` feature opens the names cudarc looks for: CUDA 12's `cublas64_12.dll`
  and `nvrtc64_120_0.dll`, and cuDNN 9's `cudnn64_9.dll`. CUDA 13's `cublas64_13.dll`
  and `nvrtc64_130_0.dll` are not among them, so cuBLAS and NVRTC need CUDA 12's
  libraries on Windows
- On Windows, the driver, cuBLAS, cuDNN and NVRTC are never loaded from the current
  directory or a relative `PATH` entry. cudakit loads the file Windows' default search
  would pick by full path before cudarc opens it, and returns the new
  `Error::UntrustedLibrary` when a copy of a name it tries is in one of those
  directories. cuBLAS then
  binds the cuBLASLt beside it, unless one of that name is already loaded
- `Error::LibraryUnavailable` names the files cudakit looks for and where: `PATH` and
  `%CUDA_PATH_V12_x%\bin` on Windows, `LD_LIBRARY_PATH` and the ldconfig cache on
  Linux. On Windows the driver displays as `CUDA driver (nvcuda.dll)`
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
