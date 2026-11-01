# Changelog

## 0.1.1

### cudakit

- `MatmulPlan`: row-major cuBLASLt products with FP32 (`Math::Fp32` or `Math::Tf32`)
  or FP16 operands and one fixed algorithm. `Gpu::matmul_candidates` lists the
  heuristic's algorithms best first, `Gpu::matmul_plan_with` checks a stored
  `MatmulAlgo` before use, and `Gpu::cublaslt_version` keys tune files. Plans share
  one workspace, `MATMUL_WORKSPACE_BYTES`, and can be recorded in graphs. Buffers must
  start on `MATMUL_ALIGNMENT` (16-byte) boundaries; `Gpu::matmul_algo_config` reads an
  algorithm's tile, stages and split-K back for logs
- `Library::CublasLt` and `Error::CublasLt`, with the cuBLASLt symbol preflight
- serde for `Math`, `Operands`, `Matmul` and `MatmulAlgo` under the `serde` feature

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
