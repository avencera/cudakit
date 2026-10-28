//! Embedded PTX and cubin artifacts and the exact identity of a loaded module

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{Mutex, OnceLock};

use sha2::{Digest, Sha256};

use crate::Error;
use crate::profile::Profile;
use crate::tier::{ComputeCapability, PtxTier};

/// A consumer's kernel areas: one cuda-oxide module each, embedded per PTX tier
///
/// Implement it for a fieldless enum. [`Self::variants`] should be built from
/// [`embed_tier!`](crate::embed_tier) calls and marked `#[inline(never)]`: without
/// that rustc may give every calling codegen unit a local copy, each with its own
/// copy of every embedded PTX and cubin
pub trait KernelArea: Copy + Eq + Hash + Debug + Send + Sync + 'static {
    /// The consumer's build-wide policy
    const PROFILE: &'static Profile;

    /// The area name, which is also the PTX file stem
    fn name(self) -> &'static str;

    /// The PTX variants embedded in this build
    fn variants(self) -> AreaPtx;
}

/// Embeds one area variant under exactly the consumer features for which it is the
/// best shipped PTX
///
/// `$ptx` and `$cubin` are the consumer's `const fn` lookups over its tier crates. They
/// are called in const context only, so a binary contains just the files it embeds; a
/// run-time call would link every file of every enabled tier crate. The `cfg` checks
/// expand in the consumer, so `$feature` names the consumer's Cargo features.
/// Consumers usually wrap this in a local `tier_ptx!` macro, whose call syntax
/// `cudakit-build` checks against the manifests
#[macro_export]
macro_rules! embed_tier {
    ($ptx:path, $cubin:path; [$($feature:literal),+], $stem:literal, [$($arch:literal),+]) => {{
        #[cfg(any($(feature = $feature),+))]
        const CUBINS: &[$crate::EmbeddedCubin] = &[$($crate::EmbeddedCubin {
            arch: $crate::ComputeCapability::new($arch / 10, $arch % 10),
            bytes: const { $cubin(concat!($stem, ".sm_", stringify!($arch), ".cubin")) },
        }),+];
        #[cfg(any($(feature = $feature),+))]
        let ptx = Some($crate::EmbeddedPtx {
            text: const { $ptx(concat!($stem, ".ptx")) },
            cubins: CUBINS,
        });
        #[cfg(not(any($(feature = $feature),+)))]
        let ptx = None;
        ptx
    }};
}

/// Content identity of bytes embedded in the running binary
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArtifactHash([u8; 32]);

impl ArtifactHash {
    /// Hash the actual artifact bytes, not a source-tree file or manifest claim
    pub fn of(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    /// Read a canonical pinned hash; invalid pins fail at compile time in constants
    pub const fn from_hex(text: &str) -> Self {
        const fn digit(byte: u8) -> u8 {
            match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                _ => panic!("hash must use lowercase hexadecimal"),
            }
        }
        assert!(
            text.len() == 64,
            "hash must contain 64 hexadecimal characters"
        );
        let mut bytes = [0; 32];
        let mut index = 0;
        while index < bytes.len() {
            bytes[index] =
                digit(text.as_bytes()[index * 2]) * 16 + digit(text.as_bytes()[index * 2 + 1]);
            index += 1;
        }
        Self(bytes)
    }

    /// Byte equality usable in compile-time table validation
    pub const fn const_eq(self, other: Self) -> bool {
        let mut index = 0;
        while index < self.0.len() {
            if self.0[index] != other.0[index] {
                return false;
            }
            index += 1;
        }
        true
    }
}

impl std::fmt::Display for ArtifactHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Immutable bytes whose content identity can be reused across plans and runtimes
struct EmbeddedArtifact(&'static [u8]);

impl EmbeddedArtifact {
    fn sha256(self) -> ArtifactHash {
        static HASHES: OnceLock<Mutex<HashMap<(usize, usize), ArtifactHash>>> = OnceLock::new();
        let mut hashes = HASHES
            .get_or_init(Mutex::default)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // static slices cannot change or reuse their address; length distinguishes subviews
        *hashes
            .entry((self.0.as_ptr() as usize, self.0.len()))
            .or_insert_with(|| ArtifactHash::of(self.0))
    }
}

/// The exact artifact accepted by the driver for a kernel area
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoadedArtifact {
    /// Ready SASS compiled for this exact device capability
    Cubin {
        /// Exact cubin architecture, never an older compatible architecture
        arch: ComputeCapability,
        /// Hash of the binary bytes handed to the driver
        sha256: ArtifactHash,
    },
    /// Embedded PTX compiled by the device driver
    PtxJit {
        /// Hash of the PTX text handed to the driver
        sha256: ArtifactHash,
    },
}

/// One embedded binary, available only alongside its source PTX variant
#[derive(Debug, Clone, Copy)]
pub struct EmbeddedCubin {
    /// The exact device capability the binary was assembled for
    pub arch: ComputeCapability,
    /// The cubin bytes
    pub bytes: &'static [u8],
}

impl EmbeddedCubin {
    /// Content hash, cached per embedded slice
    pub fn sha256(self) -> ArtifactHash {
        EmbeddedArtifact(self.bytes).sha256()
    }
}

/// The PTX fallback and exact-architecture binaries for one tier of one area
#[derive(Debug, Clone, Copy)]
pub struct EmbeddedPtx {
    /// The PTX text
    pub text: &'static str,
    /// Cubins assembled from exactly this PTX
    pub cubins: &'static [EmbeddedCubin],
}

impl EmbeddedPtx {
    /// Content hash of the PTX text, cached per embedded slice
    pub fn sha256(self) -> ArtifactHash {
        EmbeddedArtifact(self.text.as_bytes()).sha256()
    }

    /// The cubin for exactly `device`; an older compatible architecture never matches
    pub fn cubin(self, device: ComputeCapability) -> Option<EmbeddedCubin> {
        self.cubins
            .iter()
            .copied()
            .find(|cubin| cubin.arch == device)
    }
}

/// The PTX variants of one area embedded by this build
///
/// A tier feature may embed an older variant when that is the best the area ships.
/// Every variant exports the same kernels with the same parameters, which
/// `cudakit-build` checks, so the host can load any of them
#[derive(Debug, Clone, Copy, Default)]
pub struct AreaPtx {
    // indexed by tier; private so a new tier is not a breaking change
    variants: [Option<EmbeddedPtx>; PtxTier::COUNT],
}

impl AreaPtx {
    /// An area with no embedded variant yet; add them with [`Self::with`]
    pub const EMPTY: Self = Self {
        variants: [None; PtxTier::COUNT],
    };

    /// An area that ships only the baseline variant
    pub const fn baseline(sm75: Option<EmbeddedPtx>) -> Self {
        Self::EMPTY.with(PtxTier::BASELINE, sm75)
    }

    /// This area with `ptx` as its `tier` variant, usually from [`embed_tier!`], which
    /// is `None` when the build does not embed it
    pub const fn with(mut self, tier: PtxTier, ptx: Option<EmbeddedPtx>) -> Self {
        self.variants[tier.index()] = ptx;
        self
    }

    /// Embedded variants without cubins, for host-only tests such as a fake higher tier
    #[cfg(feature = "test-support")]
    pub fn fixture(variants: &[(PtxTier, &'static str)]) -> Self {
        variants.iter().fold(Self::EMPTY, |area, (tier, text)| {
            area.with(*tier, Some(EmbeddedPtx { text, cubins: &[] }))
        })
    }

    /// The embedded variants, lowest tier first
    pub fn iter(&self) -> impl Iterator<Item = (PtxTier, &'static str)> {
        PtxTier::ALL
            .iter()
            .filter_map(|tier| Some((*tier, self.embedded(*tier)?.text)))
    }

    /// The embedded bytes for one specific tier
    pub fn embedded(&self, tier: PtxTier) -> Option<EmbeddedPtx> {
        self.variants[tier.index()]
    }

    /// The highest runnable embedded tier, without hashing its artifact
    pub fn runnable_tier(self, limit: PtxTier, device: ComputeCapability) -> Option<PtxTier> {
        PtxTier::ALL.iter().copied().rev().find(|tier| {
            *tier <= limit && tier.min_capability() <= device && self.embedded(*tier).is_some()
        })
    }

    /// The best runnable tier and this device's exact cubin, or its PTX when the area
    /// embeds no cubin for the device
    pub fn driver_request<K: KernelArea>(
        self,
        area: K,
        limit: PtxTier,
        device: ComputeCapability,
    ) -> Option<ModuleRequest<K>> {
        let tier = self.runnable_tier(limit, device)?;
        let ptx = self.embedded(tier)?;
        let artifact = ptx.cubin(device).map_or_else(
            || LoadedArtifact::PtxJit {
                sha256: ptx.sha256(),
            },
            |cubin| LoadedArtifact::Cubin {
                arch: device,
                sha256: cubin.sha256(),
            },
        );
        Some(ModuleRequest::new(area, tier, artifact))
    }

    /// The highest embedded variant at or below `limit`, regardless of device
    pub fn select(&self, limit: PtxTier) -> Option<(PtxTier, &'static str)> {
        self.iter().filter(|(tier, _)| *tier <= limit).last()
    }
}

/// A complete module identity, resolved before the driver sees any bytes
///
/// Loading, the runtime's module cache and consumers' selection tokens compare whole
/// requests: the area, the PTX tier whose bytes are loaded, and the exact artifact
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleRequest<K> {
    area: K,
    tier: PtxTier,
    artifact: LoadedArtifact,
}

impl<K: KernelArea> ModuleRequest<K> {
    /// A request whose cubin, if any, can run the tier; invalid constants fail to compile
    pub const fn new(area: K, tier: PtxTier, artifact: LoadedArtifact) -> Self {
        if let LoadedArtifact::Cubin { arch, .. } = artifact {
            assert!(
                arch.at_least(tier.min_capability()),
                "a cubin cannot be older than its PTX tier"
            );
        }
        Self {
            area,
            tier,
            artifact,
        }
    }

    /// The kernel area
    pub const fn area(self) -> K {
        self.area
    }

    /// The PTX tier whose bytes are loaded
    pub const fn tier(self) -> PtxTier {
        self.tier
    }

    /// The exact artifact
    pub const fn artifact(self) -> LoadedArtifact {
        self.artifact
    }

    /// Reject a different cached execution identity without replacing its module
    pub fn check_cached(self, cached: Self) -> Result<(), Error> {
        if self != cached {
            return Err(Error::ArtifactUnavailable {
                module: self.area.name(),
                artifact: self.artifact,
            });
        }
        Ok(())
    }

    /// Resolve a forced JIT diagnostic without changing the selected tier
    pub fn diagnostic_request(self, force_jit: bool) -> Result<Self, Error> {
        if !force_jit {
            return Ok(self);
        }
        let embedded =
            self.area
                .variants()
                .embedded(self.tier)
                .ok_or(Error::ArtifactUnavailable {
                    module: self.area.name(),
                    artifact: self.artifact,
                })?;
        Ok(self.ptx_jit(ArtifactHash::of(embedded.text.as_bytes())))
    }

    /// The same area and tier, loaded through driver JIT of the tier's embedded PTX
    pub const fn ptx_jit(self, sha256: ArtifactHash) -> Self {
        Self::new(self.area, self.tier, LoadedArtifact::PtxJit { sha256 })
    }
}

/// Failure to load the requested artifact, without substituting another artifact
#[derive(Debug, PartialEq, Eq)]
pub enum ArtifactLoadError<E> {
    /// The requested bytes are not embedded
    Unavailable,
    /// The driver refused the requested bytes
    Driver(E),
}

/// Load exactly the requested bytes; driver rejection never tries the other format
pub fn load_artifact<T, E>(
    requested: LoadedArtifact,
    cubin: Option<EmbeddedCubin>,
    ptx_sha256: ArtifactHash,
    load_cubin: impl FnOnce(&[u8]) -> Result<T, E>,
    load_ptx: impl FnOnce() -> Result<T, E>,
) -> Result<(T, LoadedArtifact), ArtifactLoadError<E>> {
    let loaded = match requested {
        LoadedArtifact::PtxJit { sha256 } if sha256 == ptx_sha256 => load_ptx(),
        LoadedArtifact::Cubin { arch, sha256 } => {
            let Some(cubin) = cubin.filter(|cubin| cubin.arch == arch && cubin.sha256() == sha256)
            else {
                return Err(ArtifactLoadError::Unavailable);
            };
            load_cubin(cubin.bytes)
        }
        _ => return Err(ArtifactLoadError::Unavailable),
    };
    loaded
        .map(|loaded| (loaded, requested))
        .map_err(ArtifactLoadError::Driver)
}

#[cfg(test)]
mod tests {
    use super::{
        ArtifactHash, ArtifactLoadError, EmbeddedArtifact, EmbeddedCubin, LoadedArtifact,
        load_artifact,
    };
    use crate::ComputeCapability;

    #[test]
    fn embedded_hash_cache_preserves_content_and_subview_identity() {
        static FIRST: &[u8] = b"first artifact";
        static SECOND: &[u8] = b"other artifact";
        for bytes in [FIRST, SECOND, &FIRST[..5], FIRST, SECOND] {
            assert_eq!(EmbeddedArtifact(bytes).sha256(), ArtifactHash::of(bytes));
        }
        assert_ne!(
            EmbeddedArtifact(FIRST).sha256(),
            EmbeddedArtifact(&FIRST[..5]).sha256()
        );
    }

    #[test]
    fn hex_pins_round_trip_and_compare_in_const_context() {
        const PIN: ArtifactHash = ArtifactHash::from_hex(
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
        );
        assert_eq!(PIN, ArtifactHash::of(b"hello"));
        assert_eq!(
            PIN.to_string(),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
        assert!(PIN.const_eq(ArtifactHash::of(b"hello")));
        assert!(!PIN.const_eq(ArtifactHash::of(b"other")));
    }

    #[test]
    fn pinned_loading_never_substitutes_artifacts() {
        let hash = ArtifactHash::of(b"ptx");
        let jit = LoadedArtifact::PtxJit { sha256: hash };
        for device in [ComputeCapability::new(8, 9), ComputeCapability::new(12, 0)] {
            let cubin = EmbeddedCubin {
                arch: device,
                bytes: b"cubin",
            };
            let binary = LoadedArtifact::Cubin {
                arch: device,
                sha256: ArtifactHash::of(cubin.bytes),
            };
            let result = load_artifact(
                binary,
                Some(cubin),
                hash,
                |bytes| Ok::<_, &str>(bytes.to_vec()),
                || panic!("cubin request cannot JIT"),
            )
            .unwrap();
            assert_eq!(result, (cubin.bytes.to_vec(), binary));
            assert_eq!(
                load_artifact::<(), _>(
                    binary,
                    Some(cubin),
                    hash,
                    |_| Err("driver refusal"),
                    || panic!("refused cubin must not JIT")
                ),
                Err(ArtifactLoadError::Driver("driver refusal"))
            );
            assert_eq!(
                load_artifact::<(), &str>(
                    binary,
                    None,
                    hash,
                    |_| panic!("missing cubin"),
                    || panic!("missing cubin must not JIT")
                ),
                Err(ArtifactLoadError::Unavailable)
            );
            assert_eq!(
                load_artifact(
                    jit,
                    Some(cubin),
                    hash,
                    |_| panic!("PTX request cannot load cubin"),
                    || Ok::<_, &str>("jit")
                ),
                Ok(("jit", jit))
            );
            assert_eq!(
                load_artifact::<(), _>(
                    jit,
                    Some(cubin),
                    hash,
                    |_| panic!("refused PTX cannot load cubin"),
                    || Err("PTX rejected")
                ),
                Err(ArtifactLoadError::Driver("PTX rejected"))
            );
        }
        let wrong = LoadedArtifact::PtxJit {
            sha256: ArtifactHash::of(b"stale ptx"),
        };
        assert_eq!(
            load_artifact::<(), &str>(
                wrong,
                None,
                hash,
                |_| panic!("wrong hash"),
                || panic!("wrong hash")
            ),
            Err(ArtifactLoadError::Unavailable)
        );
    }
}
