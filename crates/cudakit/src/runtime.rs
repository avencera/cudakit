use std::collections::HashMap;
use std::ops::Deref;
use std::sync::{Arc, Mutex, PoisonError};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::Ptx;
use tracing::debug;

use crate::artifact::{
    ArtifactHash, ArtifactLoadError, KernelArea, LoadedArtifact, ModuleRequest, load_artifact,
};
use crate::{Error, Gpu, PtxTier};

/// How to open a [`Runtime`]; consumers fill it from their own environment variables
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Options {
    /// Lower the PTX tier limit below the highest compiled-in tier the device supports
    pub tier: Option<PtxTier>,
    /// Load embedded PTX through driver JIT instead of cubins, for diagnosis
    pub force_ptx_jit: bool,
}

/// A [`Gpu`] plus the kernel modules of one consumer's areas
///
/// Each area loads at most one module per runtime. A request with a different identity
/// than the cached one is an error, never a replacement, so a development load cannot
/// change what a later production request runs. Dereferences to [`Gpu`]
#[derive(Debug)]
pub struct Runtime<K: KernelArea> {
    modules: Mutex<HashMap<K, LoadedKernels<K>>>,
    force_ptx_jit: bool,
    // declared last so loaded modules drop before the context
    gpu: Gpu,
}

impl<K: KernelArea> Runtime<K> {
    /// Opens device `ordinal` with `K`'s profile; see [`Gpu::open`]
    pub fn open(ordinal: usize, options: Options) -> Result<Self, Error> {
        Ok(Self {
            modules: Mutex::new(HashMap::new()),
            force_ptx_jit: options.force_ptx_jit,
            gpu: Gpu::open(ordinal, K::PROFILE, options.tier)?,
        })
    }

    /// The underlying GPU
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// Whether this runtime forces PTX JIT
    pub fn force_ptx_jit(&self) -> bool {
        self.force_ptx_jit
    }

    /// The best runnable embedded artifact of `area` on this device, resolved without
    /// loading: the exact cubin when the area embeds one for this device, else its PTX
    ///
    /// Consumers with their own binding policy build a [`ModuleRequest`] themselves and
    /// pass it through [`Self::effective_request`]
    pub fn embedded_request(&self, area: K) -> Result<Option<ModuleRequest<K>>, Error> {
        area.variants()
            .driver_request(area, self.ptx_tier(), self.compute_capability())
            .map(|request| self.effective_request(request))
            .transpose()
    }

    /// Applies the forced-JIT diagnostic before a selection fixes a token or cache
    /// identity
    pub fn effective_request(&self, request: ModuleRequest<K>) -> Result<ModuleRequest<K>, Error> {
        request.diagnostic_request(self.force_ptx_jit)
    }

    /// Loads the best runnable embedded artifact of `area`; see [`Self::embedded_request`]
    pub fn load_embedded(&self, area: K) -> Result<LoadedKernels<K>, Error> {
        let request = self
            .embedded_request(area)?
            .ok_or(Error::AreaTierNotCompiledIn {
                area: area.name(),
                tier: self.ptx_tier(),
                device: self.compute_capability(),
                feature: K::PROFILE.feature(self.ptx_tier()),
            })?;
        self.load(request)
    }

    /// Loads exactly `request`; a cached module with another identity is an error, never
    /// a replacement, and a driver refusal never tries the other artifact format
    pub fn load(&self, request: ModuleRequest<K>) -> Result<LoadedKernels<K>, Error> {
        let area = request.area();
        let tier = request.tier();
        let requested = request.artifact();
        let unavailable = |artifact| Error::ArtifactUnavailable {
            module: area.name(),
            artifact,
        };
        let embedded = area
            .variants()
            .embedded(tier)
            .ok_or_else(|| unavailable(requested))?;
        let ptx = embedded.text;
        let ptx_sha256 = embedded.sha256();
        let cubin = embedded.cubin(self.compute_capability());
        let mut modules = self.modules.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(loaded) = modules.get(&area) {
            request.check_cached(loaded.request())?;
            return Ok(loaded.clone());
        }

        let context = self.context();
        let (inner, artifact) = load_artifact(
            requested,
            cubin,
            ptx_sha256,
            |bytes| context.load_module(Ptx::from_binary(bytes.to_vec())),
            || context.load_module(Ptx::from_src(ptx)),
        )
        .map_err(|error| match error {
            ArtifactLoadError::Unavailable => unavailable(requested),
            ArtifactLoadError::Driver(source) => Error::ArtifactLoad {
                module: area.name(),
                artifact: requested,
                source,
            },
        })?;
        debug!(area = area.name(), %tier, capability = %self.compute_capability(), ?artifact, "Loaded CUDA artifact");
        let loaded = LoadedKernels {
            request: ModuleRequest::new(area, tier, artifact),
            context: context.clone(),
            inner,
            ptx_sha256,
        };
        debug!(embedded_ptx_sha256 = %loaded.ptx_sha256(), "CUDA embedded PTX identity");

        modules.insert(area, loaded.clone());
        Ok(loaded)
    }
}

impl<K: KernelArea> Deref for Runtime<K> {
    type Target = Gpu;

    fn deref(&self) -> &Gpu {
        &self.gpu
    }
}

/// A kernel module loaded into a CUDA context
#[derive(Debug, Clone)]
pub struct LoadedKernels<K> {
    request: ModuleRequest<K>,
    // cudarc's function lookup does not bind the module's context, and a cached or
    // moved module may be used first on a thread where it is not current
    context: Arc<CudaContext>,
    inner: Arc<CudaModule>,
    ptx_sha256: ArtifactHash,
}

impl<K: KernelArea> LoadedKernels<K> {
    /// Wraps a module that the caller loaded itself, such as an artifact proof that
    /// loads every embedded variant. `context` must be the one `inner` was loaded into
    pub fn from_module(
        context: Arc<CudaContext>,
        request: ModuleRequest<K>,
        inner: Arc<CudaModule>,
        ptx_sha256: ArtifactHash,
    ) -> Self {
        Self {
            request,
            context,
            inner,
            ptx_sha256,
        }
    }

    /// The complete identity the driver accepted
    pub fn request(&self) -> ModuleRequest<K> {
        self.request
    }

    /// The PTX tier of the variant that was loaded
    pub fn tier(&self) -> PtxTier {
        self.request.tier()
    }

    /// Identity of the artifact that the driver loaded
    pub fn artifact(&self) -> LoadedArtifact {
        self.request.artifact()
    }

    /// Identity of the fallback PTX text embedded alongside the loaded artifact
    pub fn ptx_sha256(&self) -> ArtifactHash {
        self.ptx_sha256
    }

    /// Looks up a kernel by its PTX entry name, which is the Rust function name
    pub fn function(&self, kernel: &str) -> Result<CudaFunction, Error> {
        self.context.bind_to_thread()?;
        self.inner
            .load_function(kernel)
            .map_err(|source| Error::KernelMissing {
                module: self.request.area().name(),
                kernel: kernel.to_string(),
                source,
            })
    }
}
