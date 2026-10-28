# cudakit-build

[![crates.io](https://img.shields.io/crates/v/cudakit-build.svg)](https://crates.io/crates/cudakit-build)
[![docs.rs](https://docs.rs/cudakit-build/badge.svg)](https://docs.rs/cudakit-build)

The offline kernel toolchain for [cudakit](https://crates.io/crates/cudakit)
applications, as a library that your xtask calls.

An application writes its kernels with [cuda-oxide](https://github.com/NVlabs/cuda-oxide)
in its own kernels crate. That crate needs a pinned nightly and CUDA 13, so its PTX is
built on a GPU box and committed, one generated crate per PTX tier. `cudakit-build`:

- builds each kernel area to PTX with the pinned cuda-oxide rev and nightly
- compiles exact-architecture cubins from the committed PTX bytes with the pinned ptxas
- writes a manifest per area that binds sources, PTX and cubins by hash
- generates the `<app>-ptx-<tier>` crates, which are `no_std` and have no dependencies
- checks everything anywhere, without a GPU: hashes, PTX headers and ISA limit, entry
  points across variants, the cubin matrix, the generated crates, and the feature and
  architecture masks of every embed in your host code

## Usage

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
    areas: &[Area::new("encoder", &[Tier::Sm75])],
    pins: Pins {
        cuda_oxide_rev: "918bbde123671153b29a89542f1781f1c7494c14",
        cuda_oxide_nightly: "nightly-2026-08-28",
        max_ptx_isa: (9, 0),
        ptxas: "Cuda compilation tools, release 13.0, V13.0.88",
    },
    cubin_arches: &[CubinArch(75), CubinArch(80), CubinArch(86), CubinArch(89), CubinArch(90), CubinArch(120)],
};

// in CI: no GPU or CUDA toolkit needed
PROJECT.check(&workspace_root, false)?;
// on a GPU box with the pinned toolchain
PROJECT.build(&workspace_root, &[])?;
```

Write the pins out in full rather than using `Pins::CURRENT`, so a toolchain default
bump never makes your committed artifacts stale. The
[guide](https://github.com/avencera/cudakit/blob/main/docs/guide.md#kernel-toolchain)
covers the details, and the repository's `scripts/setup-gpu-box.sh` prepares an Ubuntu
24.04 NVIDIA box with the same pins.

## License

Apache-2.0
