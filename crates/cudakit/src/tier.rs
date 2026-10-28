use std::fmt;
use std::str::FromStr;

use crate::Error;

/// A device compute capability, ordered by major then minor version
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeCapability {
    /// Major version, 7 for Turing and Volta
    pub major: u32,
    /// Minor version, 5 for Turing
    pub minor: u32,
}

impl ComputeCapability {
    /// A compute capability from its major and minor version
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// Ordering usable in const context
    pub const fn at_least(self, other: Self) -> bool {
        self.major > other.major || self.major == other.major && self.minor >= other.minor
    }
}

impl fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// A GPU generation that committed PTX can target
///
/// The driver JIT-compiles PTX for its target and every newer GPU, so a tier runs on
/// any device at or above [`Self::min_capability`]. Only plain `sm_XY` targets exist:
/// `a`-suffixed targets run on one exact architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum PtxTier {
    /// `sm_75`: Turing (T4, RTX 20xx) and newer
    Sm75,
    /// `sm_80`: Ampere (A100, RTX 30xx) and newer
    Sm80,
    /// `sm_90`: Hopper and newer
    Sm90,
    /// `sm_120`: consumer Blackwell (RTX 50xx) and newer
    Sm120,
}

impl PtxTier {
    /// The tier every area ships and every supported GPU runs
    pub const BASELINE: Self = Self::Sm75;

    /// Every tier, lowest first
    pub const ALL: [Self; 4] = [Self::Sm75, Self::Sm80, Self::Sm90, Self::Sm120];

    /// The name used in PTX file names, such as `sm75`
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sm75 => "sm75",
            Self::Sm80 => "sm80",
            Self::Sm90 => "sm90",
            Self::Sm120 => "sm120",
        }
    }

    /// The oldest compute capability that can run this tier's PTX
    pub const fn min_capability(self) -> ComputeCapability {
        match self {
            Self::Sm75 => ComputeCapability::new(7, 5),
            Self::Sm80 => ComputeCapability::new(8, 0),
            Self::Sm90 => ComputeCapability::new(9, 0),
            Self::Sm120 => ComputeCapability::new(12, 0),
        }
    }

    /// Position in [`Self::ALL`]
    pub const fn index(self) -> usize {
        match self {
            Self::Sm75 => 0,
            Self::Sm80 => 1,
            Self::Sm90 => 2,
            Self::Sm120 => 3,
        }
    }
}

impl fmt::Display for PtxTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for PtxTier {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim();
        Self::ALL
            .into_iter()
            .find(|tier| tier.name() == value)
            .ok_or_else(|| Error::InvalidPtxTier {
                value: value.to_string(),
            })
    }
}

/// A set of PTX tiers, usable in const context
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TierSet(u8);

impl TierSet {
    /// No tiers
    pub const EMPTY: Self = Self(0);

    /// The set of exactly these tiers
    pub const fn of(tiers: &[PtxTier]) -> Self {
        let mut set = Self::EMPTY;
        let mut index = 0;
        while index < tiers.len() {
            set = set.with(tiers[index]);
            index += 1;
        }
        set
    }

    /// This set plus `tier`
    pub const fn with(self, tier: PtxTier) -> Self {
        Self(self.0 | 1 << tier.index())
    }

    /// Adds `tier` when `enabled`, for building a set from `cfg!` checks
    pub const fn with_if(self, tier: PtxTier, enabled: bool) -> Self {
        if enabled { self.with(tier) } else { self }
    }

    /// Whether `tier` is in the set
    pub const fn contains(self, tier: PtxTier) -> bool {
        self.0 & 1 << tier.index() != 0
    }

    /// The tiers in the set, lowest first
    pub fn iter(self) -> impl Iterator<Item = PtxTier> {
        PtxTier::ALL
            .into_iter()
            .filter(move |tier| self.contains(*tier))
    }
}
