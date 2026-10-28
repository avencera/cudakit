# Changelog

## 0.1.0

First release.

### cudakit

- `Gpu` and `Runtime<K>`: one context and stream per device, with a module cache for
  an application's kernel areas that loads exact cubins or PTX by identity and never
  substitutes another artifact. Both are `Send + Sync`, and cudakit makes the context
  current on whichever thread calls it
- `Profile` and `LibraryPolicy`: build policy checked at open, including driver-only
  operation that refuses every library before anything is looked up
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
- `ptx_lint`, which rejects generic shared-memory addresses narrowed without a
  conversion to the shared space
