# cudakit

**A CUDA host stack that needs only the NVIDIA driver.** cudakit loads `libcuda` at run
time and never links or loads the CUDA Runtime (`libcudart`). Building needs no CUDA
toolkit, and running needs only the driver. cuBLAS and cuDNN are optional and also
loaded at run time. Each one is loaded only when the consumer's policy allows it, and
only after a symbol check passes.

The name is meant to avoid "cuda-runtime", which would suggest libcudart.

cudakit is the shared GPU layer of [speakrs](https://github.com/avencera/speakrs)
(diarization) and scriptrs (Parakeet ASR). Each consumer ships its own kernels, written
with cuda-oxide and embedded as PTX plus exact-architecture cubins.

## What it provides

- `Gpu`: one context and one stream, device attributes, cooperative-launch capacity, and
  lazily prepared cuBLAS/cuDNN handles
- `Runtime<K>`: a `Gpu` plus a module cache for the consumer's `KernelArea` enum. It
  loads embedded cubins or PTX by exact identity and never falls back silently
- `Profile`: the consumer's build policy (compiled tiers, shipped tiers, feature names,
  and driver-only or library use), checked when the GPU opens
- `DeviceTensor`: a shaped device buffer
- `Sgemm` and `SgemmBatched`: row-major cuBLAS products with length checks
- `Gpu::capture`: CUDA graph capture that pauses cudarc event tracking and always ends
  the capture
- `dnn`: a cuDNN convolution planner
- `identity`: the driver release, for persisted tuning keys
- `cudakit-build`: the offline kernel toolchain library that consumers' xtasks call (in progress)

## Features

| Feature | Adds |
| --- | --- |
| (none) | Driver API only |
| `libraries` | cuBLAS and cuDNN through cudarc, the symbol preflight, the GEMMs and `dnn` |
| `serde` | serde for `identity::DriverRelease` |
| `test-support` | Fixtures for consumers' host-only tests |

Even with `libraries` enabled, a driver-only `Profile` refuses every library before
anything is looked up. Policy is a run-time value rather than a Cargo feature, so two
consumers with different policies can share one binary.

Make cudakit an optional dependency that only your CUDA features enable. That keeps
CPU and CoreML builds free of cudarc.

## Development

```sh
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
# GPU checks fail instead of skipping when CUDAKIT_REQUIRE_GPU=1
CUDAKIT_REQUIRE_GPU=1 cargo test -p cudakit --features "libraries test-support" --test gpu
```

See `docs/design.md` for the design and `HANDOFF.md` for the consumer recipe.

## License

Apache-2.0
