//! Kernel toolchain for cudakit consumers: builds cuda-oxide kernels to PTX, pins
//! cubins and checks the committed artifacts
//!
//! A consumer's kernel crate needs a pinned nightly and CUDA 13, so its PTX is generated
//! on a GPU box and committed, one published crate per tier. Every area ships an `sm75`
//! baseline variant and may add higher tiers, written as `<area>.<tier>.ptx`, plus one
//! `<area>.manifest` in the consumer's manifest directory that records, per variant, the
//! target, the hash of the sources that produced it and the hash of the PTX itself.
//! Cubins use the exact PTX bytes and the pinned ptxas, with one file per compatible
//! exact GPU capability. Manifests bind each cubin to its source PTX
//!
//! Parallel work on different areas never edits the same generated file. Plain
//! [`Project::check`] only hashes and parses files, so it runs anywhere
//!
//! The consumer's xtask declares one const [`Project`] and forwards its subcommands to
//! [`Project::build`], [`Project::build_cubins`] and [`Project::check`]

use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use eyre::{Context, Result, bail, eyre};
use sha2::{Digest, Sha256};

use crate::entry_points::{entry_point_differences, entry_points};
use crate::manifest::{Manifest, ManifestCubin, ManifestVariant};

mod entry_points;
mod host_embeds;
mod manifest;
pub mod ptx_lint;

/// A PTX target an area can ship
///
/// Only plain `sm_XY` targets: the driver JIT-compiles their PTX for every newer GPU,
/// while `a`-suffixed targets run on one exact architecture only
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    /// Turing and newer; the baseline every area ships
    Sm75,
    /// Ampere and newer
    Sm80,
    /// Hopper and newer
    Sm90,
    /// Blackwell consumer GPUs and newer
    Sm120,
}

impl Tier {
    /// Every area ships this variant; it covers Turing and everything newer
    pub const BASELINE: Self = Self::Sm75;

    /// Every tier, oldest first
    pub const ALL: [Self; 4] = [Self::Sm75, Self::Sm80, Self::Sm90, Self::Sm120];

    /// The name used in file names, manifests, crate names and host features
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sm75 => "sm75",
            Self::Sm80 => "sm80",
            Self::Sm90 => "sm90",
            Self::Sm120 => "sm120",
        }
    }

    /// The cuda-oxide `--arch` and PTX `.target`
    pub const fn arch(self) -> &'static str {
        match self {
            Self::Sm75 => "sm_75",
            Self::Sm80 => "sm_80",
            Self::Sm90 => "sm_90",
            Self::Sm120 => "sm_120",
        }
    }

    /// The oldest compute capability this tier runs on, as `major * 10 + minor`
    pub const fn capability(self) -> u16 {
        match self {
            Self::Sm75 => 75,
            Self::Sm80 => 80,
            Self::Sm90 => 90,
            Self::Sm120 => 120,
        }
    }

    /// The kernel-crate feature that turns on tier-specific code
    const fn feature(self) -> Option<&'static str> {
        match self {
            Self::Sm75 => None,
            Self::Sm80 => Some("tier-sm80"),
            Self::Sm90 => Some("tier-sm90"),
            Self::Sm120 => Some("tier-sm120"),
        }
    }

    /// Oldest PTX ISA that can target this architecture, used for stub modules
    const fn min_ptx_isa(self) -> (u32, u32) {
        match self {
            Self::Sm75 => (6, 3),
            Self::Sm80 => (7, 0),
            Self::Sm90 => (7, 8),
            Self::Sm120 => (8, 7),
        }
    }

    fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|tier| tier.name() == name)
    }
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A kernel area: one module and feature in the kernel crate, one PTX file per tier
#[derive(Debug, Clone, Copy)]
pub struct Area {
    name: &'static str,
    tiers: &'static [Tier],
    /// A production binding pins the baseline's exact bytes, so every target embeds it
    /// and higher tiers may export entries the baseline lacks instead of stubs
    pinned_baseline: bool,
    /// A production binding selects the top tier on one measured device only, so that
    /// tier's targets also embed the tier below it for every other device
    bound_top: bool,
}

impl Area {
    /// An area shipping `tiers`, each selected by every target at or above it until a
    /// newer shipped tier takes over
    ///
    /// Rejects, at compile time in a const, an area without the baseline or with
    /// unordered tiers
    pub const fn new(name: &'static str, tiers: &'static [Tier]) -> Self {
        assert!(
            !tiers.is_empty() && tiers[0] as u8 == Tier::BASELINE as u8,
            "every kernel area must ship the sm75 baseline as its first tier"
        );
        let mut index = 1;
        while index < tiers.len() {
            assert!(
                (tiers[index - 1] as u8) < tiers[index] as u8,
                "kernel area tiers must be strictly ascending"
            );
            index += 1;
        }

        Self {
            name,
            tiers,
            pinned_baseline: false,
            bound_top: false,
        }
    }

    /// As [`Area::new`], for an area whose baseline a production binding pins
    pub const fn pinned_baseline(name: &'static str, tiers: &'static [Tier]) -> Self {
        Self {
            pinned_baseline: true,
            ..Self::new(name, tiers)
        }
    }

    /// As [`Area::new`], for an area whose top tier only a device binding selects
    pub const fn bound_top(name: &'static str, tiers: &'static [Tier]) -> Self {
        assert!(tiers.len() > 1, "a bound top tier needs a tier below it");
        Self {
            bound_top: true,
            ..Self::new(name, tiers)
        }
    }

    /// The area's module and feature name in the kernel crate
    pub const fn name(self) -> &'static str {
        self.name
    }

    /// The tiers this area ships, baseline first
    pub const fn tiers(self) -> &'static [Tier] {
        self.tiers
    }

    /// Whether `variant` is embedded for the targets of `tier`
    fn embeds(self, variant: Tier, tier: Tier) -> bool {
        let newest = self.tiers.iter().rev().find(|shipped| **shipped <= tier);
        let below_top = self.bound_top.then(|| self.tiers[self.tiers.len() - 2]);
        (self.pinned_baseline && variant == Tier::BASELINE)
            || newest == Some(&variant)
            || (below_top == Some(variant) && tier >= variant)
    }

    /// Whether `tier` may export entries the baseline lacks: every higher tier of a
    /// pinned baseline, and a bound top tier, whose binding's host plans the added
    /// entries only where that tier is loaded
    fn may_add(self, tier: Tier) -> bool {
        self.pinned_baseline || (self.bound_top && self.tiers.last() == Some(&tier))
    }

    fn variants(self) -> impl Iterator<Item = Variant> {
        self.tiers
            .iter()
            .map(move |&tier| Variant { area: self, tier })
    }
}

/// An exact GPU capability for which ready SASS is shipped, as `major * 10 + minor`
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CubinArch(pub u16);

impl CubinArch {
    /// Turing through Blackwell consumer GPUs, every capability the current pins ship
    /// cubins for
    pub const DEFAULT: &'static [Self] =
        &[Self(75), Self(80), Self(86), Self(89), Self(90), Self(120)];

    /// The ptxas `-arch` and cubin file-name form, such as `sm_86`
    pub fn name(self) -> String {
        format!("sm_{}", self.0)
    }

    fn parse(name: &str, arches: &[Self]) -> Option<Self> {
        arches.iter().copied().find(|arch| arch.name() == name)
    }
}

/// Toolchain pins that every source hash and manifest records
///
/// Any change makes every committed variant stale, so each consumer keeps its own pins
/// and moves them only together with a rebuild on the GPU box
#[derive(Debug, Clone, Copy)]
pub struct Pins {
    /// cuda-oxide commit; keep in sync with the kernel crate's Cargo.toml and the GPU
    /// box setup script
    pub cuda_oxide_rev: &'static str,
    /// Toolchain pinned by that cuda-oxide commit
    pub cuda_oxide_nightly: &'static str,
    /// Newest PTX ISA the oldest supported driver can JIT; a newer `.version` fails to
    /// load there
    pub max_ptx_isa: (u32, u32),
    /// The exact `ptxas --version` line, patch release included for byte identity
    pub ptxas: &'static str,
}

impl Pins {
    /// The current pins: cuda-oxide `918bbde`, its nightly, PTX ISA 9.0 so CUDA 13.0
    /// drivers (580.x) can JIT, and CUDA 13.0.88 ptxas
    pub const CURRENT: Self = Self {
        cuda_oxide_rev: "918bbde123671153b29a89542f1781f1c7494c14",
        cuda_oxide_nightly: "nightly-2026-08-28",
        max_ptx_isa: (9, 0),
        ptxas: "Cuda compilation tools, release 13.0, V13.0.88",
    };
}

/// The per-tier artifact crates that carry PTX and cubins
#[derive(Debug, Clone, Copy)]
pub struct TierCrate {
    /// Crate directory relative to the repository root, with `{tier}` standing for the
    /// tier name, such as `crates/speakrs-ptx-{tier}`
    pub path: &'static str,
    /// The generated `src/lib.rs` crate docs, without `//!` markers and with `{tier}`
    /// standing for the tier name; a generated-file note is appended
    pub description: &'static str,
    /// The area named in the generated lookup functions' doc examples
    pub example_area: &'static str,
}

/// A consumer's kernel artifacts: where they live, which areas and tiers they ship and
/// which toolchain pins produce them
///
/// Paths are relative to the repository root passed to each command
#[derive(Debug, Clone, Copy)]
pub struct Project {
    /// The consumer's name, written into the stub PTX of areas without kernels
    pub name: &'static str,
    /// The command that runs these subcommands, such as `cargo xtask cuda-kernels`,
    /// named in generated files and in hints
    pub command: &'static str,
    /// The cuda-oxide kernel crate directory
    pub kernel_crate: &'static str,
    /// The kernel crate's package name; cuda-oxide names its PTX after it
    pub kernel_package: &'static str,
    /// Directory of the per-area manifests, which stay in the consumer because they
    /// cover every tier
    pub manifest_dir: &'static str,
    /// The host source whose `tier_ptx!([features], "ptx/<area>.<tier>", [arches])`
    /// calls embed each variant for its selected GPU tiers
    pub host_kernels: &'static str,
    /// The consumer feature that selects a GPU tier, with `{tier}` standing for the tier
    /// name, such as `cuda-{tier}`
    pub host_feature: &'static str,
    /// The per-tier artifact crates
    pub tier_crate: TierCrate,
    /// Kernel areas and the tiers each one ships
    pub areas: &'static [Area],
    /// Toolchain pins
    pub pins: Pins,
    /// Exact GPU capabilities for which ready SASS is shipped, strictly ascending
    ///
    /// Manifests record them sorted, so any other order would fail the next `check`;
    /// every entry point rejects it up front
    pub cubin_arches: &'static [CubinArch],
}

/// Exact argv template: default optimization, no debug or line information
const PTXAS_FLAGS: &str = "-arch={arch} {input} -o {output}";

impl Project {
    /// Reject declarations that would produce artifacts the next `check` refuses
    fn validate(&self) -> Result<()> {
        if !self.cubin_arches.is_sorted_by(|a, b| a < b) {
            bail!(
                "cubin_arches must be strictly ascending, got {:?}",
                self.cubin_arches
            );
        }

        Ok(())
    }

    /// Regenerate PTX for every variant of the given areas, or of every area when none
    /// are given, then build their cubins with the pinned ptxas
    ///
    /// Needs cargo-oxide, the pinned nightly and CUDA 13, so it runs on the GPU box
    pub fn build(&self, root: &Path, areas: &[String]) -> Result<()> {
        self.validate()?;
        let areas = self.selected_areas(areas)?;
        let ptxas = Ptxas::pinned(self.pins.ptxas)?;
        let dirs = ArtifactDirs::new(self, root);
        fs::create_dir_all(dirs.manifests())?;
        for tier in self.shipped_tiers() {
            fs::create_dir_all(dirs.tier(tier))?;
        }

        for area in areas {
            dirs.build_area(area)?;
            dirs.build_area_cubins(area, &ptxas)?;
        }

        dirs.write_tier_libs()
    }

    /// Build only cubins from the exact PTX bytes already in the committed directory
    ///
    /// Requires the pinned ptxas, but never invokes cuda-oxide
    pub fn build_cubins(&self, root: &Path, areas: &[String]) -> Result<()> {
        self.validate()?;
        let areas = self.selected_areas(areas)?;
        let ptxas = Ptxas::pinned(self.pins.ptxas)?;
        let dirs = ArtifactDirs::new(self, root);
        for area in areas {
            dirs.build_area_cubins(area, &ptxas)?;
        }

        dirs.write_tier_libs()
    }

    /// Fail when any committed PTX is stale, edited by hand, missing, not embedded by
    /// the host, or exports different kernels than the other variants of its area
    ///
    /// Also checks the complete cubin matrix, hashes and build pins without a toolkit
    /// With `rebuild`, requires pinned ptxas and byte-identical rebuilt cubins
    pub fn check(&self, root: &Path, rebuild: bool) -> Result<()> {
        self.validate()?;
        let dirs = ArtifactDirs::new(self, root);
        let mut problems = Vec::new();

        for area in self.areas {
            if let Err(error) = dirs.check_area(*area) {
                problems.push(format!("{}: {error:#}", area.name));
            }
        }

        problems.extend(dirs.unexpected_ptx_files()?);
        problems.extend(dirs.tier_lib_problems());
        problems.extend(self.host_embed_problems(&root.join(self.host_kernels))?);
        if !problems.is_empty() {
            bail!(
                "committed CUDA PTX or cubins failed checks; use `{} build` for PTX or `build-cubins` for cubins on the GPU box\n  {}",
                self.command,
                problems.join("\n  ")
            );
        }

        let variants: Vec<_> = self
            .areas
            .iter()
            .flat_map(|area| area.variants())
            .map(|variant| format!("{}.{}", variant.area.name, variant.tier.name()))
            .collect();
        if rebuild {
            dirs.rebuild_cubins()?;
        }

        println!(
            "CUDA PTX and cubins are up to date for {}",
            variants.join(", ")
        );
        Ok(())
    }

    /// Tiers that ship at least one variant, and so have an artifact crate
    fn shipped_tiers(&self) -> Vec<Tier> {
        let mut tiers: Vec<_> = self
            .areas
            .iter()
            .flat_map(|area| area.tiers.iter().copied())
            .collect();
        tiers.sort();
        tiers.dedup();
        tiers
    }

    /// The consumer feature that selects `tier`
    fn host_feature(&self, tier: Tier) -> String {
        self.host_feature.replace("{tier}", tier.name())
    }

    /// Name cuda-oxide gives the PTX of the kernel crate
    fn oxide_ptx_name(&self) -> String {
        format!("{}.ptx", self.kernel_package.replace('-', "_"))
    }

    fn variant_cubins(&self, variant: Variant) -> impl Iterator<Item = Cubin> + use<> {
        let arches = self.cubin_arches;
        arches
            .iter()
            .copied()
            .filter(move |arch| arch.0 >= variant.tier.capability())
            .map(move |arch| Cubin { variant, arch })
    }

    /// The generated source of a tier crate, which embeds every file the tier ships
    fn render_tier_lib(&self, tier: Tier) -> String {
        let variants: Vec<_> = self
            .areas
            .iter()
            .flat_map(|area| area.variants())
            .filter(|variant| variant.tier == tier)
            .collect();
        let list = |names: Vec<String>| -> String {
            names
                .iter()
                .map(|name| format!("    \"{name}\",\n"))
                .collect()
        };
        let ptx = list(variants.iter().map(|variant| variant.file_name()).collect());
        let cubins = list(
            variants
                .iter()
                .flat_map(|variant| self.variant_cubins(*variant))
                .map(Cubin::file_name)
                .collect(),
        );
        let docs: String = self
            .tier_crate
            .description
            .replace("{tier}", tier.name())
            .lines()
            .map(|line| match line {
                "" => "//!\n".to_string(),
                line => format!("//! {line}\n"),
            })
            .collect();
        let command = self.command;
        let example = self.tier_crate.example_area;
        format!(
            r#"{docs}//!
//! Generated by `{command} build`; do not edit
#![no_std]

/// Embeds files from `ptx/`, keyed by their path relative to the crate root
macro_rules! files {{
    ($include:ident; $($name:literal,)*) => {{
        &[$((concat!("ptx/", $name), $include!(concat!("../ptx/", $name)))),*]
    }};
}}

const PTX: &[(&str, &str)] = files! {{ include_str;
{ptx}}};

const CUBINS: &[(&str, &[u8])] = files! {{ include_bytes;
{cubins}}};

/// The PTX at `path`, such as `ptx/{example}.sm75.ptx`, if this tier ships it
pub const fn ptx(path: &str) -> Option<&'static str> {{
    let mut index = 0;
    while index < PTX.len() {{
        if same(PTX[index].0, path) {{
            return Some(PTX[index].1);
        }}
        index += 1;
    }}
    None
}}

/// The cubin at `path`, such as `ptx/{example}.sm75.sm_75.cubin`, if this tier ships it
pub const fn cubin(path: &str) -> Option<&'static [u8]> {{
    let mut index = 0;
    while index < CUBINS.len() {{
        if same(CUBINS[index].0, path) {{
            return Some(CUBINS[index].1);
        }}
        index += 1;
    }}
    None
}}

// str equality is not const
const fn same(left: &str, right: &str) -> bool {{
    let (left, right) = (left.as_bytes(), right.as_bytes());
    if left.len() != right.len() {{
        return false;
    }}
    let mut index = 0;
    while index < left.len() {{
        if left[index] != right[index] {{
            return false;
        }}
        index += 1;
    }}
    true
}}
"#
        )
    }

    fn selected_areas(&self, areas: &[String]) -> Result<Vec<Area>> {
        if areas.is_empty() {
            return Ok(self.areas.to_vec());
        }

        areas
            .iter()
            .map(|name| {
                self.areas
                    .iter()
                    .copied()
                    .find(|area| area.name == name)
                    .ok_or_else(|| {
                        eyre!("unknown kernel area `{name}`; known: {}", self.area_names())
                    })
            })
            .collect()
    }

    fn area_names(&self) -> String {
        self.areas
            .iter()
            .map(|area| area.name)
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// A loadable module with no entries, so the host can embed and load every area
    /// before the area has kernels
    fn empty_module_ptx(&self, variant: Variant) -> String {
        let (major, minor) = variant.tier.min_ptx_isa();
        format!(
            "//\n// {}: the `{}` area has no kernels yet\n//\n\n.version {major}.{minor}\n.target {}\n.address_size 64\n",
            self.name,
            variant.area.name,
            variant.tier.arch()
        )
    }

    /// The area a PTX directory file belongs to, from its first dot-separated part
    fn file_area(&self, name: &str) -> Option<&'static str> {
        let stem = name.split('.').next()?;
        self.areas
            .iter()
            .map(|area| area.name)
            .find(|area| *area == stem)
    }

    /// Whether a crate file feeds the given area's PTX
    fn belongs_to(&self, relative: &str, area: &str) -> bool {
        let Some(module) = self.area_of(relative) else {
            return true;
        };

        module == area
    }

    /// The area that owns `src/<area>.rs` or `src/<area>/...`, if any
    fn area_of(&self, relative: &str) -> Option<&'static str> {
        let rest = relative.strip_prefix("src/")?;
        self.areas.iter().map(|area| area.name).find(|area| {
            rest.strip_prefix(area)
                .is_some_and(|tail| tail == ".rs" || tail.starts_with('/'))
        })
    }

    /// Checks `.version` and `.target` so a toolchain bump cannot silently produce PTX
    /// that older drivers or GPUs reject
    fn check_ptx_header(&self, variant: Variant, ptx: &str) -> Result<(u32, u32)> {
        let file = variant.file_name();
        let directive = |name: &str| {
            ptx.lines()
                .map(str::trim)
                .find_map(|line| line.strip_prefix(name))
                .map(str::trim)
        };

        let version = directive(".version")
            .and_then(|version| version.split_once('.'))
            .and_then(|(major, minor)| Some((major.parse().ok()?, minor.parse().ok()?)))
            .ok_or_else(|| eyre!("{file} has no readable `.version` directive"))?;
        let max = self.pins.max_ptx_isa;
        if version > max {
            bail!(
                "{file} uses PTX ISA {}.{}, newer than {}.{} that the oldest supported driver accepts",
                version.0,
                version.1,
                max.0,
                max.1
            );
        }

        let target = directive(".target").unwrap_or_default();
        let arch = variant.tier.arch();
        if target.split(',').next().map(str::trim) != Some(arch) {
            bail!("{file} targets `{target}`, expected `{arch}`");
        }

        Ok(version)
    }

    /// Hash of everything that determines one variant's PTX: the build pins, the target
    /// and features, the crate's shared files and this area's own module. Other areas'
    /// modules are left out, so a change in one area never marks another area stale
    fn sources_hash(&self, crate_dir: &Path, variant: Variant) -> Result<String> {
        let mut files = Vec::new();
        collect_files(crate_dir, crate_dir, &mut files)?;
        files.retain(|relative| self.belongs_to(relative, variant.area.name));
        files.sort();

        let mut digest = Sha256::new();
        let pins = format!(
            "cuda-oxide={} toolchain={} arch={} features={}",
            self.pins.cuda_oxide_rev,
            self.pins.cuda_oxide_nightly,
            variant.tier.arch(),
            variant.features()
        );
        hash_entry(&mut digest, "pins", pins.as_bytes());
        for relative in files {
            let contents = fs::read(crate_dir.join(&relative))?;
            hash_entry(&mut digest, &relative, &contents);
        }

        Ok(format!("{:x}", digest.finalize()))
    }
}

/// One PTX file: an area built for one tier
#[derive(Debug, Clone, Copy)]
struct Variant {
    area: Area,
    tier: Tier,
}

impl Variant {
    fn file_name(self) -> String {
        format!("{}.{}.ptx", self.area.name, self.tier.name())
    }

    fn features(self) -> String {
        match self.tier.feature() {
            Some(tier) => format!("{},{tier}", self.area.name),
            None => self.area.name.to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Cubin {
    variant: Variant,
    arch: CubinArch,
}

impl Cubin {
    fn file_name(self) -> String {
        format!(
            "{}.{}.{}.cubin",
            self.variant.area.name,
            self.variant.tier.name(),
            self.arch.name()
        )
    }
}

struct Ptxas(PathBuf);

impl Ptxas {
    fn pinned(version: &str) -> Result<Self> {
        let toolkit = std::env::var_os("CUDA13_HOME")
            .or_else(|| std::env::var_os("CUDA_TOOLKIT_PATH"))
            .ok_or_else(|| eyre!("set CUDA13_HOME or CUDA_TOOLKIT_PATH to the pinned CUDA toolkit; ptxas is never taken from PATH"))?;
        let path = PathBuf::from(toolkit).join("bin/ptxas");
        let output = Command::new(&path)
            .arg("--version")
            .output()
            .wrap_err_with(|| format!("running {} --version", path.display()))?;
        let actual = String::from_utf8(output.stdout)?;
        if !output.status.success() || !actual.lines().any(|line| line == version) {
            bail!(
                "wrong ptxas version from {}: expected `{version}`, got `{}`",
                path.display(),
                actual.trim()
            );
        }

        Ok(Self(path))
    }

    fn compile(&self, dirs: &ArtifactDirs, out_dir: &Path, cubin: Cubin) -> Result<()> {
        run_cmd(
            Command::new(&self.0)
                .arg(format!("-arch={}", cubin.arch.name()))
                .arg(dirs.ptx(cubin.variant))
                .arg("-o")
                .arg(out_dir.join(cubin.file_name())),
        )
        .wrap_err_with(|| format!("building {} from committed PTX", cubin.file_name()))
    }
}

/// Where generated kernel files live, under one repository root
///
/// Manifests stay in the consumer. Each tier's PTX and cubins ship in a per-tier crate,
/// because together they exceed the crates.io upload limit
struct ArtifactDirs<'a> {
    project: &'a Project,
    root: PathBuf,
}

impl<'a> ArtifactDirs<'a> {
    fn new(project: &'a Project, root: &Path) -> Self {
        Self {
            project,
            root: root.to_path_buf(),
        }
    }

    fn kernel_crate(&self) -> PathBuf {
        self.root.join(self.project.kernel_crate)
    }

    fn manifests(&self) -> PathBuf {
        self.root.join(self.project.manifest_dir)
    }

    fn manifest(&self, area: Area) -> PathBuf {
        self.manifests().join(format!("{}.manifest", area.name))
    }

    fn tier_crate(&self, tier: Tier) -> PathBuf {
        self.root
            .join(self.project.tier_crate.path.replace("{tier}", tier.name()))
    }

    fn tier(&self, tier: Tier) -> PathBuf {
        self.tier_crate(tier).join("ptx")
    }

    fn ptx(&self, variant: Variant) -> PathBuf {
        self.tier(variant.tier).join(variant.file_name())
    }

    fn cubin(&self, cubin: Cubin) -> PathBuf {
        self.tier(cubin.variant.tier).join(cubin.file_name())
    }

    /// The `src/lib.rs` that embeds a tier crate's files
    fn tier_lib(&self, tier: Tier) -> PathBuf {
        self.tier_crate(tier).join("src/lib.rs")
    }

    /// The display form of a generated file, relative to the project root
    fn relative(&self, path: &Path) -> String {
        path.strip_prefix(&self.root)
            .unwrap_or(path)
            .display()
            .to_string()
    }

    /// Write each tier crate's source so it embeds exactly the declared files
    fn write_tier_libs(&self) -> Result<()> {
        for tier in self.project.shipped_tiers() {
            let path = self.tier_lib(tier);
            fs::create_dir_all(path.parent().expect("lib.rs has a parent"))?;
            fs::write(&path, self.project.render_tier_lib(tier))?;
        }

        Ok(())
    }

    fn build_area(&self, area: Area) -> Result<()> {
        let project = self.project;
        let crate_dir = self.kernel_crate();
        let mut built = Vec::new();
        for variant in area.variants() {
            let ptx = self.compile_variant(variant)?;
            let version = project.check_ptx_header(variant, &ptx)?;
            ptx_lint::check_shared_truncation(&ptx)?;
            built.push((variant, ptx, version));
        }

        // compare before writing anything, so a mismatch leaves the committed files alone
        let modules: Vec<_> = built
            .iter()
            .map(|(variant, ptx, _)| (*variant, ptx.as_str()))
            .collect();
        check_entry_points(area, &modules)?;

        self.remove_undeclared_variants(area)?;
        let mut manifest = Manifest::default();
        for (variant, ptx, version) in built {
            fs::write(self.ptx(variant), &ptx)?;
            manifest.variants.push(ManifestVariant {
                tier: variant.tier,
                sources: project.sources_hash(&crate_dir, variant)?,
                ptx: sha256_hex(ptx.as_bytes()),
                cubins: Vec::new(),
            });
            println!(
                "{}: wrote {} (PTX ISA {}.{}, {})",
                area.name,
                self.relative(&self.ptx(variant)),
                version.0,
                version.1,
                variant.tier.arch()
            );
        }

        fs::write(self.manifest(area), manifest.render(project))?;
        Ok(())
    }

    fn build_area_cubins(&self, area: Area, ptxas: &Ptxas) -> Result<()> {
        let project = self.project;
        let mut manifest = self.read_manifest(area)?;
        self.check_area_ptx(area, &manifest)?;
        // keep temporary writes inside the checkout, including on the rented box
        let out = tempfile::tempdir_in(self.manifests())?;
        for (variant, entry) in area.variants().zip(&mut manifest.variants) {
            entry.cubins.clear();
            for cubin in project.variant_cubins(variant) {
                ptxas.compile(self, out.path(), cubin)?;
                entry.cubins.push(ManifestCubin {
                    arch: cubin.arch,
                    sha256: sha256_hex(&fs::read(out.path().join(cubin.file_name()))?),
                    ptx: entry.ptx.clone(),
                });
            }
        }

        // finish the whole area before replacing artifacts or their manifest
        for cubin in area
            .variants()
            .flat_map(|variant| project.variant_cubins(variant))
        {
            fs::copy(out.path().join(cubin.file_name()), self.cubin(cubin))?;
        }

        self.remove_undeclared_variants(area)?;
        manifest.ptxas = Some(project.pins.ptxas.into());
        manifest.ptxas_flags = Some(PTXAS_FLAGS.into());
        fs::write(self.manifest(area), manifest.render(project))?;
        println!("{}: built cubins from committed PTX", area.name);
        Ok(())
    }

    fn rebuild_cubins(&self) -> Result<()> {
        let project = self.project;
        let ptxas = Ptxas::pinned(project.pins.ptxas)?;
        let out = tempfile::tempdir_in(self.manifests())?;
        for area in project.areas {
            for cubin in area
                .variants()
                .flat_map(|variant| project.variant_cubins(variant))
            {
                ptxas.compile(self, out.path(), cubin)?;
                if fs::read(out.path().join(cubin.file_name()))? != fs::read(self.cubin(cubin))? {
                    bail!(
                        "{} rebuild is not byte-identical to the committed cubin",
                        cubin.file_name()
                    );
                }
            }

            println!("{}: rebuilt cubins are byte-identical", area.name);
        }

        Ok(())
    }

    fn compile_variant(&self, variant: Variant) -> Result<String> {
        let project = self.project;
        let crate_dir = self.kernel_crate();
        let out_dir = tempfile::tempdir()?;
        // a shared CARGO_TARGET_DIR is built by the stable toolchain; keep the nightly
        // kernel build in its own subdirectory so the two never invalidate each other
        let target_dir = std::env::var_os("CARGO_TARGET_DIR")
            .map(|dir| PathBuf::from(dir).join("cuda-kernels"))
            .unwrap_or_else(|| crate_dir.join("target"));

        // cuda-oxide writes PTX only while rustc compiles the crate, and Cargo skips a
        // crate it considers fresh, so clean it first to always get a new PTX file
        run_cmd(self.nightly_cargo(&crate_dir, &target_dir).args([
            "clean",
            "-p",
            project.kernel_package,
        ]))?;
        let mut cmd = self.nightly_cargo(&crate_dir, &target_dir);
        cmd.env("CUDA_OXIDE_PTX_DIR", out_dir.path())
            .args([
                "oxide",
                "build",
                "--arch",
                variant.tier.arch(),
                "--features",
            ])
            .arg(variant.features());
        run_cmd(&mut cmd).wrap_err_with(|| {
            format!(
                "building the `{}` kernels for {}",
                variant.area.name,
                variant.tier.arch()
            )
        })?;

        let ptx_path = out_dir.path().join(project.oxide_ptx_name());
        if ptx_path.exists() {
            return Ok(fs::read_to_string(&ptx_path)?);
        }

        // a crate without device code produces no PTX at all
        if self.area_has_kernels(&crate_dir, variant.area.name)? {
            bail!(
                "cuda-oxide did not write {} for `{}`",
                ptx_path.display(),
                variant.file_name()
            );
        }

        Ok(project.empty_module_ptx(variant))
    }

    /// Cargo pinned to the cuda-oxide nightly, independent of the toolchain running the
    /// consumer's xtask
    fn nightly_cargo(&self, crate_dir: &Path, target_dir: &Path) -> Command {
        let mut cmd = Command::new("cargo");
        cmd.current_dir(crate_dir)
            // `cargo xtask` runs under the stable toolchain, and rustup would pass that
            // choice down through these variables instead of the pinned nightly
            .env_remove("RUSTUP_TOOLCHAIN")
            .env_remove("CARGO")
            .env_remove("RUSTC")
            .env("CARGO_TARGET_DIR", target_dir)
            .arg(format!("+{}", self.project.pins.cuda_oxide_nightly));
        cmd
    }

    fn area_has_kernels(&self, crate_dir: &Path, area: &str) -> Result<bool> {
        let mut files = Vec::new();
        collect_files(crate_dir, &crate_dir.join("src"), &mut files)?;
        for relative in files
            .iter()
            .filter(|relative| self.project.area_of(relative) == Some(area))
        {
            if fs::read_to_string(crate_dir.join(relative))?.contains("#[kernel]") {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Deletes variant files of `area` that it no longer declares, including the
    /// untiered `<area>.ptx` from before tiers existed
    fn remove_undeclared_variants(&self, area: Area) -> Result<()> {
        let project = self.project;
        let declared: BTreeSet<_> = area
            .variants()
            .flat_map(|variant| {
                std::iter::once(variant.file_name())
                    .chain(project.variant_cubins(variant).map(Cubin::file_name))
            })
            .collect();
        // a variant in another tier's crate is in the wrong place, so it is removed too
        let in_place = |tier: Tier, name: &str| {
            declared.contains(name) && name.split('.').nth(1) == Some(tier.name())
        };
        for tier in project.shipped_tiers() {
            let tier_dir = self.tier(tier);
            if !tier_dir.is_dir() {
                continue;
            }

            for entry in fs::read_dir(&tier_dir)? {
                let path = entry?.path();
                let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };

                let ours = (name.ends_with(".ptx") || name.ends_with(".cubin"))
                    && project.file_area(name) == Some(area.name);
                if ours && !in_place(tier, name) {
                    fs::remove_file(&path)?;
                    println!("{}: removed undeclared {}", area.name, self.relative(&path));
                }
            }
        }

        Ok(())
    }

    fn read_manifest(&self, area: Area) -> Result<Manifest> {
        let manifest_path = self.manifest(area);
        fs::read_to_string(&manifest_path)
            .map_err(|_| eyre!("missing {}", manifest_path.display()))
            .and_then(|text| Manifest::parse(&text, self.project.cubin_arches))
    }

    fn check_area(&self, area: Area) -> Result<()> {
        let manifest = self.read_manifest(area)?;
        self.check_area_ptx(area, &manifest)?;
        self.check_area_cubins(area, &manifest)
    }

    fn check_area_cubins(&self, area: Area, manifest: &Manifest) -> Result<()> {
        let project = self.project;
        let ptxas = project.pins.ptxas;
        if manifest.ptxas.as_deref() != Some(ptxas) {
            bail!(
                "wrong ptxas version in manifest: expected `{ptxas}`, got {:?}",
                manifest.ptxas
            );
        }

        if manifest.ptxas_flags.as_deref() != Some(PTXAS_FLAGS) {
            bail!("wrong ptxas flags in manifest: expected `{PTXAS_FLAGS}`");
        }

        for (variant, entry) in area.variants().zip(&manifest.variants) {
            let expected: Vec<_> = project
                .variant_cubins(variant)
                .map(|cubin| cubin.arch)
                .collect();
            let actual: Vec<_> = entry.cubins.iter().map(|cubin| cubin.arch).collect();
            if expected != actual {
                bail!(
                    "{} manifest cubin arches {:?}, expected {:?}",
                    variant.file_name(),
                    actual,
                    expected
                );
            }

            for record in &entry.cubins {
                let cubin = Cubin {
                    variant,
                    arch: record.arch,
                };
                let name = cubin.file_name();
                if record.ptx != entry.ptx {
                    bail!("{name} PTX hash does not match its section's ptx hash");
                }

                let bytes = fs::read(self.cubin(cubin))
                    .wrap_err_with(|| format!("missing cubin {name}"))?;
                if sha256_hex(&bytes) != record.sha256 {
                    bail!("{name} cubin sha256 does not match its manifest");
                }
            }
        }

        Ok(())
    }

    fn check_area_ptx(&self, area: Area, manifest: &Manifest) -> Result<()> {
        let project = self.project;
        let crate_dir = self.kernel_crate();
        let listed: Vec<_> = manifest.variants.iter().map(|entry| entry.tier).collect();
        if listed != area.tiers {
            bail!(
                "manifest lists tiers {}, but the area declares {}",
                tier_names(&listed),
                tier_names(area.tiers)
            );
        }

        let mut modules = Vec::new();
        for (variant, entry) in area.variants().zip(&manifest.variants) {
            let ptx_path = self.ptx(variant);
            let ptx = fs::read_to_string(&ptx_path)
                .map_err(|_| eyre!("missing {}", ptx_path.display()))?;

            if project.sources_hash(&crate_dir, variant)? != entry.sources {
                bail!(
                    "kernel sources or build pins changed since {} was generated",
                    variant.file_name()
                );
            }

            if sha256_hex(ptx.as_bytes()) != entry.ptx {
                bail!(
                    "{} does not match its manifest; regenerate it instead of editing it",
                    variant.file_name()
                );
            }

            project.check_ptx_header(variant, &ptx)?;
            ptx_lint::check_shared_truncation(&ptx)
                .wrap_err_with(|| format!("linting {}", variant.file_name()))?;
            modules.push((variant, ptx));
        }

        let modules: Vec<_> = modules
            .iter()
            .map(|(variant, ptx)| (*variant, ptx.as_str()))
            .collect();
        check_entry_points(area, &modules)
    }

    /// Files in the manifest or tier directories that are not declared there
    ///
    /// A missing directory holds no stray files; the per-area checks report what it
    /// lacks
    fn unexpected_ptx_files(&self) -> Result<Vec<String>> {
        let project = self.project;
        let manifests: BTreeSet<String> = project
            .areas
            .iter()
            .map(|area| format!("{}.manifest", area.name))
            .collect();
        let mut expected = vec![(self.manifests(), manifests)];
        for tier in project.shipped_tiers() {
            let files = project
                .areas
                .iter()
                .flat_map(|area| area.variants())
                .filter(|variant| variant.tier == tier)
                .flat_map(|variant| {
                    std::iter::once(variant.file_name())
                        .chain(project.variant_cubins(variant).map(Cubin::file_name))
                })
                .collect();
            expected.push((self.tier(tier), files));
        }

        let mut problems = Vec::new();
        for (dir, files) in expected {
            if !dir.is_dir() {
                continue;
            }

            for entry in fs::read_dir(&dir)? {
                let path = entry?.path();
                let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };

                if !files.contains(name) {
                    problems.push(format!(
                        "{} is not a declared PTX, cubin or manifest for this directory; declare its tier in the project's areas or delete the file",
                        path.display()
                    ));
                }
            }
        }

        Ok(problems)
    }

    /// Each tier crate must embed exactly the files its tier declares
    fn tier_lib_problems(&self) -> Vec<String> {
        let project = self.project;
        project
            .shipped_tiers()
            .into_iter()
            .filter(|tier| {
                fs::read_to_string(self.tier_lib(*tier)).ok()
                    != Some(project.render_tier_lib(*tier))
            })
            .map(|tier| {
                format!(
                    "{} is stale; run `{} build-cubins` to regenerate it",
                    self.relative(&self.tier_lib(tier)),
                    project.command
                )
            })
            .collect()
    }
}

fn tier_names(tiers: &[Tier]) -> String {
    let names: Vec<_> = tiers.iter().map(|tier| tier.name()).collect();
    format!("[{}]", names.join(", "))
}

/// The host picks one variant per area at run time and looks kernels up by name, so
/// every variant must export the baseline's kernels with the same parameters
fn check_entry_points(area: Area, modules: &[(Variant, &str)]) -> Result<()> {
    let Some(((baseline, baseline_ptx), higher)) = modules.split_first() else {
        return Ok(());
    };

    let expected =
        entry_points(baseline_ptx).wrap_err_with(|| format!("parsing {}", baseline.file_name()))?;
    let mut problems = Vec::new();
    for (variant, ptx) in higher {
        let actual =
            entry_points(ptx).wrap_err_with(|| format!("parsing {}", variant.file_name()))?;
        problems.extend(entry_point_differences(
            &baseline.file_name(),
            &expected,
            &variant.file_name(),
            &actual,
            area.may_add(variant.tier),
        ));
    }

    if !problems.is_empty() {
        bail!(
            "variants export different kernels:\n    {}",
            problems.join("\n    ")
        );
    }

    Ok(())
}

fn collect_files(root: &Path, dir: &Path, files: &mut Vec<String>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        // build output and editor or OS litter never feed the PTX
        if name == "target" || name.to_string_lossy().starts_with('.') {
            continue;
        }

        if entry.file_type()?.is_dir() {
            collect_files(root, &path, files)?;
            continue;
        }

        let relative = path
            .strip_prefix(root)?
            .to_str()
            .ok_or_else(|| eyre!("non UTF-8 path {}", path.display()))?
            .replace('\\', "/");
        files.push(relative);
    }

    Ok(())
}

/// Length-prefixed so that moving bytes between a name and its contents changes the hash
fn hash_entry(digest: &mut Sha256, name: &str, contents: &[u8]) {
    digest.update((name.len() as u64).to_le_bytes());
    digest.update(name.as_bytes());
    digest.update((contents.len() as u64).to_le_bytes());
    digest.update(contents);
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Run a command, inheriting stdio, and bail on non-zero exit
fn run_cmd(cmd: &mut Command) -> Result<()> {
    let status = cmd.status()?;
    if !status.success() {
        bail!(
            "{} failed with {}",
            cmd.get_program().to_string_lossy(),
            status
        );
    }

    Ok(())
}

#[cfg(test)]
mod test_support {
    use super::{Area, CubinArch, Pins, Project, Tier, TierCrate, Variant};

    /// A speakrs-shaped project: probe first with two tiers, plus each binding kind
    pub(crate) const PROJECT: Project = Project {
        name: "speakrs",
        command: "cargo xtask cuda-kernels",
        kernel_crate: "crates/speakrs-cuda-kernels",
        kernel_package: "speakrs-cuda-kernels",
        manifest_dir: "src/inference/cuda/ptx",
        host_kernels: "src/inference/cuda/kernels.rs",
        host_feature: "cuda-{tier}",
        tier_crate: TierCrate {
            path: "crates/speakrs-ptx-{tier}",
            description: "Prebuilt {tier} tier CUDA kernels for speakrs",
            example_area: "fbank",
        },
        areas: &[
            Area::new("probe", &[Tier::Sm75, Tier::Sm80]),
            Area::new("fbank", &[Tier::Sm75]),
            Area::new("fbankdft", &[Tier::Sm75]),
            Area::new("embedding", &[Tier::Sm75]),
            Area::pinned_baseline("resnet", &[Tier::Sm75, Tier::Sm80]),
            Area::bound_top("wideconv", &[Tier::Sm75, Tier::Sm80, Tier::Sm120]),
        ],
        pins: Pins::CURRENT,
        cubin_arches: CubinArch::DEFAULT,
    };

    pub(crate) fn probe(tier: Tier) -> Variant {
        Variant {
            area: PROJECT.areas[0],
            tier,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{PROJECT, probe};
    use super::{
        ArtifactDirs, CubinArch, Manifest, ManifestCubin, ManifestVariant, PTXAS_FLAGS, Project,
        Tier, Variant, check_entry_points, sha256_hex,
    };
    use crate::Area;

    const PROBE_PTX: &str = "//\n// Generated by LLVM\n//\n.version 6.3\n.target sm_75\n.address_size 64\n\n\t// .globl\tprobe_scale_add // .entry fake(\n.visible .entry probe_scale_add(\n\t.param .f32 probe_scale_add_param_0,\n\t.param .u64 .ptr .align 4 probe_scale_add_param_1,\n\t.param .u64 probe_scale_add_param_2\n)\n{\n\tret;\n}\n";

    #[test]
    fn unsorted_cubin_arches_fail_before_any_artifact_is_touched() {
        let project = Project {
            cubin_arches: &[CubinArch(86), CubinArch(80)],
            ..PROJECT
        };
        // the root does not exist, so only validation can produce these errors
        let root = std::path::Path::new("/nonexistent/cudakit-build");
        for error in [
            project.check(root, false).unwrap_err(),
            project.build(root, &[]).unwrap_err(),
            project.build_cubins(root, &[]).unwrap_err(),
        ] {
            assert!(
                error.to_string().contains("strictly ascending"),
                "{error:#}"
            );
        }
    }

    #[test]
    fn area_files_only_feed_their_own_area() {
        assert_eq!(PROJECT.area_of("src/fbank.rs"), Some("fbank"));
        assert_eq!(PROJECT.area_of("src/embedding/pool.rs"), Some("embedding"));
        assert_eq!(PROJECT.area_of("src/fbank_common.rs"), None);
        assert!(PROJECT.belongs_to("src/lib.rs", "fbank"));
        assert!(PROJECT.belongs_to("Cargo.lock", "probe"));
        assert!(PROJECT.belongs_to("src/fbank.rs", "fbank"));
        assert!(!PROJECT.belongs_to("src/fbank.rs", "probe"));
        // the record-owned area shares a name prefix with the always-on area, so an
        // fbankdft edit must never mark the pinned fbank PTX stale
        assert_eq!(PROJECT.area_of("src/fbankdft.rs"), Some("fbankdft"));
        assert!(!PROJECT.belongs_to("src/fbankdft.rs", "fbank"));
        assert!(!PROJECT.belongs_to("src/fbank.rs", "fbankdft"));
    }

    #[test]
    fn ptx_header_rejects_isa_newer_than_cuda_13_0_and_wrong_targets() {
        let ok = ".version 6.3\n.target sm_75\n.address_size 64\n";
        assert_eq!(
            PROJECT.check_ptx_header(probe(Tier::Sm75), ok).ok(),
            Some((6, 3))
        );

        let too_new = ".version 9.2\n.target sm_75\n";
        assert!(
            PROJECT
                .check_ptx_header(probe(Tier::Sm75), too_new)
                .is_err()
        );

        let wrong_target = ".version 8.0\n.target sm_80\n";
        assert!(
            PROJECT
                .check_ptx_header(probe(Tier::Sm75), wrong_target)
                .is_err()
        );
        assert!(
            PROJECT
                .check_ptx_header(probe(Tier::Sm80), wrong_target)
                .is_ok()
        );
    }

    #[test]
    fn oxide_ptx_name_follows_the_kernel_package() {
        assert_eq!(PROJECT.oxide_ptx_name(), "speakrs_cuda_kernels.ptx");
    }

    #[test]
    fn entry_points_must_match_across_variants() {
        let probe_area = PROJECT.areas[0];
        let same = PROBE_PTX.replace("sm_75", "sm_80");
        assert!(
            check_entry_points(
                probe_area,
                &[(probe(Tier::Sm75), PROBE_PTX), (probe(Tier::Sm80), &same)]
            )
            .is_ok()
        );

        let renamed = same.replace("probe_scale_add", "probe_scale_add_v2");
        assert!(
            check_entry_points(
                probe_area,
                &[
                    (probe(Tier::Sm75), PROBE_PTX),
                    (probe(Tier::Sm80), &renamed)
                ]
            )
            .is_err()
        );

        let retyped = same.replace(".param .f32", ".param .f64");
        assert!(
            check_entry_points(
                probe_area,
                &[
                    (probe(Tier::Sm75), PROBE_PTX),
                    (probe(Tier::Sm80), &retyped)
                ]
            )
            .is_err()
        );

        // a pinned baseline lets a higher tier add entries, never rename or retype them
        let pinned = Area::pinned_baseline("probe", &[Tier::Sm75, Tier::Sm80]);
        let entry = same[same.find(".visible .entry").expect("entry")..]
            .replace("probe_scale_add", "probe_scale_add_tc");
        let added = format!("{same}\n{entry}");
        assert!(
            check_entry_points(
                pinned,
                &[(probe(Tier::Sm75), PROBE_PTX), (probe(Tier::Sm80), &added)]
            )
            .is_ok()
        );
        assert!(
            check_entry_points(
                probe_area,
                &[(probe(Tier::Sm75), PROBE_PTX), (probe(Tier::Sm80), &added)]
            )
            .is_err()
        );
        assert!(
            check_entry_points(
                pinned,
                &[
                    (probe(Tier::Sm75), PROBE_PTX),
                    (probe(Tier::Sm80), &renamed)
                ]
            )
            .is_err()
        );
    }

    #[test]
    fn tier_lib_renders_the_description_and_examples() {
        let lib = PROJECT.render_tier_lib(Tier::Sm80);
        assert!(lib.starts_with(
            "//! Prebuilt sm80 tier CUDA kernels for speakrs\n//!\n//! Generated by `cargo xtask cuda-kernels build`; do not edit\n#![no_std]\n"
        ));
        assert!(lib.contains("such as `ptx/fbank.sm75.ptx`"));
        assert!(lib.contains("    \"probe.sm80.ptx\",\n    \"resnet.sm80.ptx\",\n"));
        assert!(!lib.contains("probe.sm75"));
    }

    fn fixture_variant(variant: Variant) -> ManifestVariant {
        ManifestVariant {
            tier: variant.tier,
            sources: "sources".into(),
            ptx: "ptx-hash".into(),
            cubins: PROJECT
                .variant_cubins(variant)
                .map(|cubin| ManifestCubin {
                    arch: cubin.arch,
                    sha256: sha256_hex(b"cubin"),
                    ptx: "ptx-hash".into(),
                })
                .collect(),
        }
    }

    #[test]
    fn manifest_round_trips() {
        let manifest = Manifest {
            ptxas: Some(PROJECT.pins.ptxas.into()),
            ptxas_flags: Some(PTXAS_FLAGS.into()),
            variants: vec![
                ManifestVariant {
                    tier: Tier::Sm75,
                    sources: "abc".into(),
                    ptx: "def".into(),
                    cubins: fixture_variant(probe(Tier::Sm75)).cubins,
                },
                ManifestVariant {
                    tier: Tier::Sm80,
                    sources: "ghi".into(),
                    ptx: "jkl".into(),
                    cubins: fixture_variant(probe(Tier::Sm80)).cubins,
                },
            ],
        };
        assert_eq!(
            Manifest::parse(&manifest.render(&PROJECT), PROJECT.cubin_arches).ok(),
            Some(manifest)
        );
    }

    fn cubin_fixture() -> (tempfile::TempDir, ArtifactDirs<'static>, Manifest) {
        let dir = tempfile::tempdir().expect("temporary directory");
        let dirs = ArtifactDirs::new(&PROJECT, dir.path());
        let manifest = Manifest {
            ptxas: Some(PROJECT.pins.ptxas.into()),
            ptxas_flags: Some(PTXAS_FLAGS.into()),
            variants: PROJECT.areas[0].variants().map(fixture_variant).collect(),
        };
        for cubin in PROJECT.areas[0]
            .variants()
            .flat_map(|variant| PROJECT.variant_cubins(variant))
        {
            std::fs::create_dir_all(dirs.tier(cubin.variant.tier)).expect("tier directory");
            std::fs::write(dirs.cubin(cubin), b"cubin").expect("write cubin");
        }

        dirs.check_area_cubins(PROJECT.areas[0], &manifest)
            .expect("valid cubins");
        (dir, dirs, manifest)
    }

    fn first_cubin(variant: Variant) -> super::Cubin {
        PROJECT.variant_cubins(variant).next().expect("cubin")
    }

    #[test]
    fn check_rejects_tampered_cubin() {
        let (_dir, dirs, manifest) = cubin_fixture();
        std::fs::write(dirs.cubin(first_cubin(probe(Tier::Sm75))), b"tampered").expect("tamper");
        let error = dirs
            .check_area_cubins(PROJECT.areas[0], &manifest)
            .expect_err("tampered hash");
        assert!(error.to_string().contains("cubin sha256 does not match"));
    }

    #[test]
    fn check_rejects_cubin_ptx_mismatch() {
        let (_dir, dirs, mut manifest) = cubin_fixture();
        manifest.variants[0].cubins[0].ptx = "different-ptx".into();
        let error = dirs
            .check_area_cubins(PROJECT.areas[0], &manifest)
            .expect_err("PTX mismatch");
        assert!(error.to_string().contains("PTX hash does not match"));
    }

    #[test]
    fn check_rejects_wrong_ptxas_version_and_flags() {
        let (_dir, dirs, mut manifest) = cubin_fixture();
        manifest.ptxas = Some("Cuda compilation tools, release 12.8, V12.8.93".into());
        let error = dirs
            .check_area_cubins(PROJECT.areas[0], &manifest)
            .expect_err("wrong ptxas");
        assert!(error.to_string().contains("wrong ptxas version"));
        manifest.ptxas = Some(PROJECT.pins.ptxas.into());
        manifest.ptxas_flags = Some("-lineinfo".into());
        let error = dirs
            .check_area_cubins(PROJECT.areas[0], &manifest)
            .expect_err("wrong flags");
        assert!(error.to_string().contains("wrong ptxas flags"));
    }

    #[test]
    fn check_rejects_missing_cubin() {
        let (_dir, dirs, manifest) = cubin_fixture();
        std::fs::remove_file(dirs.cubin(first_cubin(probe(Tier::Sm75)))).expect("remove cubin");
        let error = dirs
            .check_area_cubins(PROJECT.areas[0], &manifest)
            .expect_err("missing cubin");
        assert!(
            error
                .to_string()
                .contains("missing cubin probe.sm75.sm_75.cubin")
        );
    }

    #[test]
    fn check_rejects_stray_cubin() {
        let (_dir, dirs, _) = cubin_fixture();
        std::fs::write(
            dirs.tier(Tier::Sm75).join("probe.sm75.sm_100.cubin"),
            b"stray",
        )
        .expect("stray cubin");
        // a declared file in another tier's crate is stray too
        let misplaced = first_cubin(probe(Tier::Sm80)).file_name();
        std::fs::write(dirs.tier(Tier::Sm75).join(&misplaced), b"cubin").expect("misplaced");
        let errors = dirs.unexpected_ptx_files().expect("scan directories");
        assert_eq!(errors.len(), 2);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("probe.sm75.sm_100.cubin is not a declared"))
        );
        assert!(
            errors
                .iter()
                .any(|error| error.contains(&format!("{misplaced} is not a declared")))
        );
    }

    #[test]
    fn check_rejects_missing_or_incompatible_manifest_arches() {
        let (_dir, dirs, mut manifest) = cubin_fixture();
        manifest.variants[0].cubins.pop();
        assert!(dirs.check_area_cubins(PROJECT.areas[0], &manifest).is_err());
        manifest.variants[1].cubins = fixture_variant(probe(Tier::Sm75)).cubins;
        manifest.variants[0] = fixture_variant(probe(Tier::Sm75));
        assert!(dirs.check_area_cubins(PROJECT.areas[0], &manifest).is_err());
    }
}
