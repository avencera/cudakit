use std::ffi::CStr;

use crate::tier::{ComputeCapability, PtxTier, TierSet};
use crate::{Error, Library};

/// Whether a consumer may load optional numerical libraries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LibraryPolicy {
    /// Only the driver; every cuBLAS, cuDNN or NVRTC request fails with
    /// [`Error::LibraryForbidden`] before any lookup
    DriverOnly,
    /// cuBLAS, cuDNN and NVRTC may be prepared when the `libraries` feature is on
    Allowed,
}

/// Library entry points a consumer calls directly through cudarc, beyond the ones
/// cudakit's own wrappers reach
///
/// The preflight checks these before cudarc touches the library, so a missing symbol
/// is a typed [`Error::LibraryUnavailable`] instead of a panic inside cudarc
#[derive(Debug, Clone, Copy, Default)]
pub struct ExtraSymbols {
    /// Extra cuBLAS symbols
    pub cublas: &'static [&'static CStr],
    /// Extra cuDNN symbols
    pub cudnn: &'static [&'static CStr],
    /// Extra NVRTC symbols
    pub nvrtc: &'static [&'static CStr],
}

impl ExtraSymbols {
    /// No extra symbols
    pub const NONE: Self = Self {
        cublas: &[],
        cudnn: &[],
        nvrtc: &[],
    };
}

/// A consumer's build-wide CUDA policy, declared once as a constant
///
/// cudakit reads no environment variables and no consumer Cargo features; everything
/// that depends on them comes through this value
#[derive(Debug, Clone, Copy)]
pub struct Profile {
    /// The consumer crate, for log context
    pub consumer: &'static str,
    /// GPU target tiers whose PTX this build embeds, usually from `cfg!(feature = ...)`
    pub compiled: TierSet,
    /// Tiers with shipped production kernels, independent of Cargo features
    pub shipped: TierSet,
    /// The consumer's Cargo feature for each tier in [`PtxTier::ALL`] order, named in
    /// errors that ask the user to enable a tier
    pub tier_features: [&'static str; 4],
    /// Whether optional libraries may be loaded
    pub libraries: LibraryPolicy,
    /// Library symbols the consumer calls directly
    pub extra_symbols: ExtraSymbols,
}

impl Profile {
    /// Whether the policy forbids optional libraries
    pub const fn driver_only(&self) -> bool {
        matches!(self.libraries, LibraryPolicy::DriverOnly)
    }

    /// The consumer's Cargo feature for `tier`
    pub const fn feature(&self, tier: PtxTier) -> &'static str {
        self.tier_features[tier.index()]
    }

    /// Whether this build embeds `tier`
    pub const fn is_compiled_in(&self, tier: PtxTier) -> bool {
        self.compiled.contains(tier)
    }

    /// Highest shipped tier the device can execute
    pub fn native(&self, device: ComputeCapability) -> PtxTier {
        self.shipped
            .iter()
            .filter(|tier| tier.min_capability() <= device)
            .max()
            .unwrap_or(PtxTier::BASELINE)
    }

    /// The highest tier a device may load: the highest compiled-in tier it supports,
    /// or `requested` when given
    ///
    /// Fails when the device is below the baseline, or when `requested` is not
    /// compiled in or is above what the device supports
    pub fn select(
        &self,
        capability: ComputeCapability,
        requested: Option<PtxTier>,
    ) -> Result<PtxTier, Error> {
        if capability < PtxTier::BASELINE.min_capability() {
            return Err(Error::UnsupportedDevice {
                capability,
                baseline: PtxTier::BASELINE,
            });
        }

        let Some(tier) = requested else {
            let supported = self
                .compiled
                .iter()
                .filter(|tier| tier.min_capability() <= capability);
            return supported.max().ok_or_else(|| {
                let tier = self.native(capability);
                Error::TierNotCompiledIn {
                    tier,
                    device: capability,
                    feature: self.feature(tier),
                }
            });
        };

        if !self.is_compiled_in(tier) {
            return Err(Error::TierNotCompiledIn {
                tier,
                device: capability,
                feature: self.feature(tier),
            });
        }

        if tier.min_capability() > capability {
            return Err(Error::PtxTierAboveDevice { tier, capability });
        }

        Ok(tier)
    }

    /// Fail before any library lookup when the policy forbids `library`
    pub(crate) fn allow(&self, library: Library) -> Result<(), Error> {
        if self.driver_only() && library != Library::Driver {
            return Err(Error::LibraryForbidden { library });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{ExtraSymbols, LibraryPolicy, Profile};
    use crate::tier::{ComputeCapability, PtxTier, TierSet};
    use crate::{Error, Library};

    fn profile(compiled: &[PtxTier]) -> Profile {
        Profile {
            consumer: "test",
            compiled: TierSet::of(compiled),
            shipped: TierSet::of(&[PtxTier::Sm75]),
            tier_features: ["t-sm75", "t-sm80", "t-sm90", "t-sm120"],
            libraries: LibraryPolicy::DriverOnly,
            extra_symbols: ExtraSymbols::NONE,
        }
    }

    #[test]
    fn automatic_target_requires_an_enabled_tier_the_device_supports() {
        for compiled in [&[][..], &[PtxTier::Sm75], &[PtxTier::Sm75, PtxTier::Sm80]] {
            let profile = profile(compiled);
            for device in [
                ComputeCapability::new(7, 5),
                ComputeCapability::new(8, 6),
                ComputeCapability::new(12, 0),
            ] {
                let expected = compiled
                    .iter()
                    .copied()
                    .filter(|tier| tier.min_capability() <= device)
                    .max();
                match (expected, profile.select(device, None)) {
                    (Some(tier), Ok(selected)) => assert_eq!(selected, tier),
                    (
                        None,
                        Err(Error::TierNotCompiledIn {
                            feature, device: d, ..
                        }),
                    ) => {
                        assert_eq!(feature, "t-sm75");
                        assert_eq!(d, device);
                    }
                    (expected, actual) => panic!("{expected:?} vs {actual:?}"),
                }
            }
            assert!(matches!(
                profile.select(ComputeCapability::new(7, 0), None),
                Err(Error::UnsupportedDevice { .. })
            ));
        }
    }

    #[test]
    fn overrides_require_the_target_feature_and_device_capability() {
        let profile = profile(&[PtxTier::Sm75, PtxTier::Sm80]);
        let device = ComputeCapability::new(12, 0);
        assert_eq!(
            profile.select(device, Some(PtxTier::Sm80)).unwrap(),
            PtxTier::Sm80
        );
        assert!(matches!(
            profile.select(ComputeCapability::new(7, 5), Some(PtxTier::Sm80)),
            Err(Error::PtxTierAboveDevice { .. })
        ));
        assert!(matches!(
            profile.select(device, Some(PtxTier::Sm120)),
            Err(Error::TierNotCompiledIn {
                tier: PtxTier::Sm120,
                feature: "t-sm120",
                ..
            })
        ));
    }

    #[test]
    fn driver_only_forbids_every_optional_library_and_names_it() {
        let profile = profile(&[]);
        for library in [Library::Cublas, Library::Cudnn, Library::Nvrtc] {
            let error = profile.allow(library).unwrap_err();
            assert!(matches!(error, Error::LibraryForbidden { library: l } if l == library));
            assert!(error.to_string().contains(&library.to_string()));
        }
        profile.allow(Library::Driver).unwrap();
    }
}
