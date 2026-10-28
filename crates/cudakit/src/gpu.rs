use std::sync::Arc;
#[cfg(feature = "libraries")]
use std::sync::MutexGuard;

#[cfg(feature = "libraries")]
use cudarc::cublas::CudaBlas;
#[cfg(feature = "libraries")]
use cudarc::cudnn::Cudnn;
use cudarc::driver::sys::CUresult;
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, DriverError};
use tracing::debug;

#[cfg(feature = "libraries")]
use crate::Math;
use crate::device::DeviceAttributes;
#[cfg(feature = "libraries")]
use crate::libraries::{Libraries, LibraryVersions};
use crate::profile::Profile;
use crate::{ComputeCapability, Error, Library, PtxTier};

/// One CUDA device context and stream, with optional libraries prepared on request
///
/// Everything issued through one `Gpu` runs in order on its stream. Use one per worker
/// thread; device buffers from one `Gpu` must not be used on another's stream without
/// explicit synchronization. Device attributes are queried once when it opens
///
/// Most consumers hold a [`Runtime`](crate::Runtime), which adds the kernel module
/// cache and dereferences to this type
#[derive(Debug)]
pub struct Gpu {
    // library handles must drop before driver state
    #[cfg(feature = "libraries")]
    libraries: Libraries,
    stream: Arc<CudaStream>,
    context: Arc<CudaContext>,
    device: DeviceAttributes,
    ptx_tier: PtxTier,
    profile: &'static Profile,
}

impl Gpu {
    /// Opens device `ordinal` and creates a stream without loading optional libraries
    ///
    /// `requested` lowers the PTX tier limit; `None` picks the highest compiled-in tier
    /// the device supports. Fails with [`Error::LibraryUnavailable`] instead of
    /// panicking when the driver library is missing, with [`Error::UnsupportedDevice`]
    /// below the `sm_75` baseline, and when `requested` is not compiled in or is above
    /// what the device supports. A driver-only profile also fails when the build does
    /// not embed the device's native shipped tier, since no library can stand in
    pub fn open(
        ordinal: usize,
        profile: &'static Profile,
        requested: Option<PtxTier>,
    ) -> Result<Self, Error> {
        ensure_driver()?;

        let count = device_count()?;
        if ordinal >= count {
            return Err(Error::NoDevice { ordinal, count });
        }

        let context = CudaContext::new(ordinal)?;
        let device = DeviceAttributes::query(&context)?;
        let capability = device.capability();
        // target features must support the device even when an override lowers the limit
        if profile.driver_only() {
            let compiled = profile.select(capability, None)?;
            let native = profile.native(capability);
            if compiled < native {
                return Err(Error::TierNotCompiledIn {
                    tier: native,
                    device: capability,
                    feature: profile.feature(native),
                });
            }
        }
        let ptx_tier = profile.select(capability, requested)?;
        debug!(
            consumer = profile.consumer,
            ordinal,
            %capability,
            %ptx_tier,
            ?requested,
            "CUDA device opened"
        );

        // a dedicated stream rather than the legacy default stream, so workers on
        // separate runtimes do not serialize against each other
        let stream = context.new_stream()?;
        debug!(
            device_name = device.name(),
            sm_count = device.multiprocessors(),
            l2_bytes = device.l2_bytes(),
            shared_optin_bytes = device.shared_optin_bytes(),
            "CUDA device properties"
        );

        Ok(Self {
            #[cfg(feature = "libraries")]
            libraries: Libraries::default(),
            stream,
            context,
            device,
            ptx_tier,
            profile,
        })
    }

    /// The consumer policy this GPU was opened with
    pub fn profile(&self) -> &'static Profile {
        self.profile
    }

    /// The device's compute capability
    pub fn compute_capability(&self) -> ComputeCapability {
        self.device.capability()
    }

    /// The device attributes queried when the GPU opened
    pub fn device(&self) -> &DeviceAttributes {
        &self.device
    }

    /// The PTX tier limit: a module whose tier is above it is never loaded
    pub fn ptx_tier(&self) -> PtxTier {
        self.ptx_tier
    }

    /// The device context
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// The stream that every operation of this GPU runs on
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Loads and checks `library` and creates its handle on this stream
    ///
    /// Call it while building a plan, never during forward or graph capture. Fails with
    /// [`Error::LibraryForbidden`] under a driver-only profile before any lookup
    pub fn prepare(&self, library: Library) -> Result<(), Error> {
        self.profile.allow(library)?;
        #[cfg(feature = "libraries")]
        {
            self.libraries.prepare(self.profile, library, &self.stream)
        }
        #[cfg(not(feature = "libraries"))]
        match library {
            Library::Driver => Ok(()),
            library => Err(Error::LibraryForbidden { library }),
        }
    }

    /// The already prepared cuBLAS handle
    #[cfg(feature = "libraries")]
    pub fn blas(&self) -> Result<&CudaBlas, Error> {
        self.profile.allow(Library::Cublas)?;
        self.libraries.blas()
    }

    /// Selects `math` on the prepared cuBLAS handle and holds the mode lock, so mode
    /// selection and the enqueue that follows are atomic
    #[cfg(feature = "libraries")]
    pub fn lock_blas(&self, math: Math) -> Result<MutexGuard<'_, Math>, Error> {
        self.profile.allow(Library::Cublas)?;
        self.libraries.lock_blas(math)
    }

    /// The already prepared cuDNN handle
    ///
    /// The handle is bound to the thread that prepared cuDNN, and any other thread gets
    /// [`Error::LibraryOnAnotherThread`]: cuDNN forbids using one handle on two threads
    /// at once, and the descriptors and plans made from it keep `!Send` clones, so they
    /// stay on that thread as well
    #[cfg(feature = "libraries")]
    pub fn dnn(&self) -> Result<&Arc<Cudnn>, Error> {
        self.profile.allow(Library::Cudnn)?;
        self.libraries.dnn()
    }

    /// Prepares cuBLAS and cuDNN and reports their versions
    #[cfg(feature = "libraries")]
    pub fn library_versions(&self) -> Result<LibraryVersions, Error> {
        self.context.bind_to_thread()?;
        self.libraries.versions(self.profile, &self.stream)
    }

    /// Streaming multiprocessors this context may use: the device's, or the client's
    /// share under MPS active-thread limits, as queried when the GPU opened
    pub fn multiprocessor_count(&self) -> usize {
        self.device.multiprocessors().get() as usize
    }

    /// Whether the device supports cooperative kernel launches
    pub fn supports_cooperative_launch(&self) -> Result<bool, Error> {
        let supported = self.context.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
        )?;
        Ok(supported != 0)
    }

    /// Blocks of `function` that one cooperative launch keeps resident at once: the
    /// occupancy API's active blocks per SM for `block_threads` threads and
    /// `dynamic_smem` bytes of dynamic shared memory, times the SM count, or 0 when the
    /// device has no cooperative launch
    ///
    /// Grids that run concurrently, for example on a side stream, share this capacity,
    /// so their sum must fit; size them with [`Self::concurrent_cooperative_capacity`]
    pub fn cooperative_capacity(
        &self,
        function: &CudaFunction,
        block_threads: u32,
        dynamic_smem: usize,
    ) -> Result<usize, Error> {
        let per_sm = self.cooperative_blocks_per_sm(function, block_threads, dynamic_smem)?;
        Ok(per_sm * self.multiprocessor_count())
    }

    /// Blocks of `function` that cooperative grids running at the same time can keep
    /// resident together in this context, or `None` when the context's SM limit is
    /// unknown, in which case grids must not run concurrently
    ///
    /// The driver checks co-residency per cooperative launch, not for grids launched on
    /// different streams, and Ampere and Ada start a cooperative grid before all its
    /// blocks fit. MPS active-thread limits already shrink the SM count attribute, but a
    /// context created with an SM-count execution affinity is limited further, so the
    /// joint budget also honors that limit. Green contexts, static MPS SM partitions and
    /// other spinning work on the device are not covered
    pub fn concurrent_cooperative_capacity(
        &self,
        function: &CudaFunction,
        block_threads: u32,
        dynamic_smem: usize,
    ) -> Result<Option<usize>, Error> {
        let per_sm = self.cooperative_blocks_per_sm(function, block_threads, dynamic_smem)?;
        let device_sms = self.multiprocessor_count();
        let budget_sms = match self.affinity_sm_limit()? {
            SmLimit::Unlimited => device_sms,
            SmLimit::Limited(sms) => sms.min(device_sms),
            SmLimit::Unknown => return Ok(None),
        };

        Ok(Some(per_sm * budget_sms))
    }

    /// Active blocks per SM for a cooperative launch of `function`, or 0 when the device
    /// has no cooperative launch
    fn cooperative_blocks_per_sm(
        &self,
        function: &CudaFunction,
        block_threads: u32,
        dynamic_smem: usize,
    ) -> Result<usize, Error> {
        if !self.supports_cooperative_launch()? {
            return Ok(0);
        }

        let per_sm = function.occupancy_max_active_blocks_per_multiprocessor(
            block_threads,
            dynamic_smem,
            None,
        )?;
        Ok(usize::try_from(per_sm).unwrap_or(0))
    }

    /// The SM-count execution affinity of this context
    fn affinity_sm_limit(&self) -> Result<SmLimit, Error> {
        use cudarc::driver::sys::{
            CUexecAffinityParam, CUexecAffinityParam_st__bindgen_ty_1, CUexecAffinitySmCount,
            CUexecAffinityType, cuCtxGetExecAffinity,
        };

        self.context.bind_to_thread()?;
        let mut param = CUexecAffinityParam {
            type_: CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT,
            param: CUexecAffinityParam_st__bindgen_ty_1 {
                smCount: CUexecAffinitySmCount { val: 0 },
            },
        };

        // SAFETY: `param` is a valid out-pointer for the call, and the context is bound
        // to this thread above; the driver writes only `param`
        let status = unsafe {
            cuCtxGetExecAffinity(
                &mut param,
                CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT,
            )
        };

        match status {
            CUresult::CUDA_SUCCESS => {
                // SAFETY: the SM-count query fills the `smCount` member
                let count = unsafe { param.param.smCount.val };
                Ok(usize::try_from(count)
                    .ok()
                    .filter(|count| *count > 0)
                    .map_or(SmLimit::Unknown, SmLimit::Limited))
            }
            // affinity is an MPS feature, so a context without it has no extra limit
            CUresult::CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY => Ok(SmLimit::Unlimited),
            status => {
                debug!(?status, "Context SM limit unavailable");
                Ok(SmLimit::Unknown)
            }
        }
    }

    /// Blocks until all work queued on [`Self::stream`] has finished
    pub fn synchronize(&self) -> Result<(), Error> {
        self.stream.synchronize()?;
        Ok(())
    }
}

/// An SM-count execution-affinity limit on a context
enum SmLimit {
    /// The context has no SM-count affinity
    Unlimited,
    /// The context may use this many SMs
    Limited(usize),
    /// The driver didn't report the limit
    Unknown,
}

/// Only the driver is needed before a consumer selects plans
fn ensure_driver() -> Result<(), Error> {
    // SAFETY: driver library initializers have no caller preconditions
    if !unsafe { cudarc::driver::sys::is_culib_present() } {
        return Err(Error::LibraryUnavailable {
            library: Library::Driver,
        });
    }
    Ok(())
}

fn device_count() -> Result<usize, Error> {
    match CudaContext::device_count() {
        Ok(count) => Ok(usize::try_from(count).unwrap_or(0)),
        // a driver with no visible GPU fails initialization instead of reporting zero
        Err(DriverError(CUresult::CUDA_ERROR_NO_DEVICE)) => Ok(0),
        Err(error) => Err(error.into()),
    }
}
