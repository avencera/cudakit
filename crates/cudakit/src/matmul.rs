//! Row-major products on cuBLASLt with an explicit, persistable algorithm
//!
//! [`Sgemm`](crate::Sgemm) leaves the algorithm to cuBLAS, which runs cuBLASLt's first
//! heuristic. That choice is a model of the GPU, not a measurement, and on small
//! products it often leaves most SMs idle. A [`MatmulPlan`] instead fixes one
//! algorithm: the heuristic's first, or any [`MatmulAlgo`] the consumer picked, for
//! example from a tune file it measured once. A fixed algorithm also fixes the
//! numerics, since split-K and tile choices change the summation order

use std::fmt;
use std::mem::MaybeUninit;

use cudarc::cublaslt::result as lt;
use cudarc::cublaslt::sys::{
    self, cublasComputeType_t, cublasLtMatmulAlgoConfigAttributes_t as Attribute,
    cublasLtMatmulDescAttributes_t, cublasLtMatmulPreferenceAttributes_t, cudaDataType,
};
// cuBLASLt takes the operation as an attribute value, so cuBLAS's enum is the type
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::{DevicePtr, DevicePtrMut, DeviceRepr};

use crate::error::check_len;
use crate::libraries::MATMUL_WORKSPACE_BYTES;

/// Byte alignment every [`Gpu::matmul`] buffer must start on, the alignment
/// cuBLASLt's heuristic assumes when it offers algorithms
pub const MATMUL_ALIGNMENT: usize = 16;
use crate::{Error, Gpu, Library, Math};

/// Precision of a [`Matmul`]'s `a` and `b`; `c` and the accumulation stay FP32
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Operands {
    /// FP32 `a` and `b` (`f32` buffers), multiplied in [`Math`] precision
    F32(Math),
    /// FP16 `a` and `b` (`u16` buffers of FP16 bit patterns) on tensor cores
    F16,
}

impl Operands {
    fn data_type(self) -> cudaDataType {
        match self {
            Self::F32(_) => cudaDataType::CUDA_R_32F,
            Self::F16 => cudaDataType::CUDA_R_16F,
        }
    }

    fn compute_type(self) -> cublasComputeType_t {
        match self {
            // FP32 compute never down-converts FP32 operands, as `Math::Fp32` promises
            Self::F32(Math::Fp32) | Self::F16 => cublasComputeType_t::CUBLAS_COMPUTE_32F,
            Self::F32(Math::Tf32) => cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
        }
    }

    fn element_bytes(self) -> usize {
        match self {
            Self::F32(_) => size_of::<f32>(),
            Self::F16 => size_of::<u16>(),
        }
    }
}

/// Element type of a [`Matmul`] operand buffer: `f32`, or `u16` holding FP16 bits
pub trait MatmulElement: DeviceRepr + sealed::Sealed {}

impl MatmulElement for f32 {}
impl MatmulElement for u16 {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u16 {}
}

/// A row-major product `c = alpha * op(a) · op(b) + beta * c` for a [`MatmulPlan`]
///
/// The layout conventions are [`Sgemm`](crate::Sgemm)'s: `op(a)` is `m × k`, `op(b)`
/// is `k × n` and `c` is `m × n`, all dense and row-major, and a transposed operand
/// is stored as its transpose. `alpha` and `beta` are given per call, so one plan
/// serves a product with and without accumulation into `c`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Matmul {
    /// Rows of `op(a)` and `c`
    pub m: usize,
    /// Columns of `op(b)` and `c`
    pub n: usize,
    /// Columns of `op(a)` and rows of `op(b)`
    pub k: usize,
    /// `a` is stored as `k × m`
    pub a_transposed: bool,
    /// `b` is stored as `n × k`
    pub b_transposed: bool,
    /// Precision of `a` and `b`
    pub operands: Operands,
}

impl Matmul {
    /// `c = a · b` with untransposed operands
    pub fn new(m: usize, n: usize, k: usize, operands: Operands) -> Self {
        Self {
            m,
            n,
            k,
            a_transposed: false,
            b_transposed: false,
            operands,
        }
    }

    /// A PyTorch `Linear` without bias: `c [rows, outputs] = a [rows, inputs] · wᵀ`,
    /// with `w` the `[outputs, inputs]` weight as `b`
    pub fn linear(rows: usize, outputs: usize, inputs: usize, operands: Operands) -> Self {
        Self::new(rows, outputs, inputs, operands).transposed(false, true)
    }

    /// The same product with `a` and `b` stored transposed or not
    pub fn transposed(self, a: bool, b: bool) -> Self {
        Self {
            a_transposed: a,
            b_transposed: b,
            ..self
        }
    }
}

/// A cuBLASLt algorithm and its configuration, as `cublasLtMatmulAlgo_t` stores them
///
/// cuBLASLt documents the structure as plain data that may be stored and reused with
/// the same library version, so a tune file keyed on [`Gpu::cublaslt_version`] and the
/// device can persist it. [`Gpu::matmul_plan_with`] checks it before use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MatmulAlgo(pub [u64; 8]);

impl MatmulAlgo {
    fn from_sys(algo: sys::cublasLtMatmulAlgo_t) -> Self {
        Self(algo.data)
    }

    fn to_sys(self) -> sys::cublasLtMatmulAlgo_t {
        sys::cublasLtMatmulAlgo_t { data: self.0 }
    }
}

/// The readable part of a [`MatmulAlgo`]'s configuration
///
/// Values are cuBLASLt's enum numbers (`cublasLtMatmulTile_t`, `cublasLtMatmulStages_t`,
/// `cublasLtReductionScheme_t`). It describes an algorithm; persist the
/// [`MatmulAlgo`] itself, which carries state this summary leaves out
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct AlgoConfig {
    /// Algorithm id
    pub id: i64,
    /// Tile shape id
    pub tile: i64,
    /// Pipeline stages id
    pub stages: i64,
    /// Split-K factor; 1 runs without splitting
    pub split_k: i64,
    /// How split-K partial sums are reduced
    pub reduction: i64,
    /// CTA swizzling
    pub swizzle: i64,
    /// Algorithm-specific option
    pub custom: i64,
}

impl fmt::Display for AlgoConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "algo {} tile {} stages {} split-k {} reduction {} swizzle {} custom {}",
            self.id,
            self.tile,
            self.stages,
            self.split_k,
            self.reduction,
            self.swizzle,
            self.custom
        )
    }
}

/// One algorithm cuBLASLt's heuristic offers for a [`Matmul`], in its ranking order
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct MatmulCandidate {
    /// The algorithm, to pass to [`Gpu::matmul_plan_with`]
    pub algo: MatmulAlgo,
    /// Workspace bytes it needs, at most [`MATMUL_WORKSPACE_BYTES`]
    pub workspace_bytes: usize,
    /// cuBLASLt's estimate of its waves over the GPU's SMs
    pub waves: f32,
}

struct Desc(sys::cublasLtMatmulDesc_t);

impl Drop for Desc {
    fn drop(&mut self) {
        // SAFETY: created by `cublasLtMatmulDescCreate` and destroyed once
        let _ = unsafe { lt::destroy_matmul_desc(self.0) };
    }
}

struct Layout(sys::cublasLtMatrixLayout_t);

impl Drop for Layout {
    fn drop(&mut self) {
        // SAFETY: created by `cublasLtMatrixLayoutCreate` and destroyed once
        let _ = unsafe { lt::destroy_matrix_layout(self.0) };
    }
}

struct Preference(sys::cublasLtMatmulPreference_t);

impl Drop for Preference {
    fn drop(&mut self) {
        // SAFETY: created by `cublasLtMatmulPreferenceCreate` and destroyed once
        let _ = unsafe { lt::destroy_matmul_pref(self.0) };
    }
}

/// The cuBLASLt descriptors of one [`Matmul`] in cuBLAS's column-major terms
struct Descriptors {
    desc: Desc,
    // `first` is the row-major `b`, `second` the row-major `a`; see `Descriptors::new`
    first: Layout,
    second: Layout,
    c: Layout,
}

impl Descriptors {
    /// Row-major `c` (`m × n`) is column-major `cᵀ` (`n × m`), so cuBLASLt computes
    /// `cᵀ = op(b)ᵀ · op(a)ᵀ` with `b` first, as [`Sgemm`](crate::Sgemm) does. A layout
    /// describes a matrix as stored: a row-major `r × c` buffer read column-major is
    /// `c × r` with leading dimension `c`
    fn new(spec: Matmul) -> Result<Self, Error> {
        let dim = |value: usize| -> Result<u64, Error> {
            u64::try_from(value).map_err(|_| Error::DimensionOverflow {
                context: "matmul",
                value,
            })
        };
        let lead = |value: usize| -> Result<i64, Error> {
            i64::try_from(value).map_err(|_| Error::DimensionOverflow {
                context: "matmul leading dimension",
                value,
            })
        };
        let op = |transposed: bool| {
            if transposed {
                cublasOperation_t::CUBLAS_OP_T
            } else {
                cublasOperation_t::CUBLAS_OP_N
            }
        };
        let (m, n, k) = (spec.m, spec.n, spec.k);
        let operand = spec.operands.data_type();

        let desc = Desc(lt::create_matmul_desc(
            spec.operands.compute_type(),
            cudaDataType::CUDA_R_32F,
        )?);
        for (attribute, transposed) in [
            (
                cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                spec.b_transposed,
            ),
            (
                cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                spec.a_transposed,
            ),
        ] {
            let value = op(transposed);
            // SAFETY: the descriptor is live and the attribute takes a `cublasOperation_t`
            unsafe {
                lt::set_matmul_desc_attribute(
                    desc.0,
                    attribute,
                    (&raw const value).cast(),
                    size_of::<cublasOperation_t>(),
                )
            }?;
        }

        // stored `b`: `n × k` when transposed, else `k × n`
        let (rows, cols) = if spec.b_transposed { (k, n) } else { (n, k) };
        let first = Layout(lt::create_matrix_layout(
            operand,
            dim(rows)?,
            dim(cols)?,
            lead(rows)?,
        )?);
        // stored `a`: `k × m` when transposed, else `m × k`
        let (rows, cols) = if spec.a_transposed { (m, k) } else { (k, m) };
        let second = Layout(lt::create_matrix_layout(
            operand,
            dim(rows)?,
            dim(cols)?,
            lead(rows)?,
        )?);
        let c = Layout(lt::create_matrix_layout(
            cudaDataType::CUDA_R_32F,
            dim(n)?,
            dim(m)?,
            lead(n)?,
        )?);
        Ok(Self {
            desc,
            first,
            second,
            c,
        })
    }
}

/// A [`Matmul`] bound to one cuBLASLt algorithm, run with [`Gpu::matmul`]
///
/// Build plans while setting up, not during graph capture: building one queries the
/// library. A plan holds only host descriptors, so it is cheap to keep one per shape
pub struct MatmulPlan {
    spec: Matmul,
    algo: MatmulAlgo,
    descriptors: Descriptors,
}

// SAFETY: the descriptors are host-side structures that cudakit never changes after
// building the plan; cuBLASLt only reads them, which is safe from several threads
unsafe impl Send for MatmulPlan {}
// SAFETY: see `Send`
unsafe impl Sync for MatmulPlan {}

impl fmt::Debug for MatmulPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatmulPlan")
            .field("spec", &self.spec)
            .field("algo", &self.algo)
            .finish_non_exhaustive()
    }
}

impl MatmulPlan {
    /// The product this plan runs
    pub fn spec(&self) -> Matmul {
        self.spec
    }

    /// The algorithm this plan runs
    pub fn algo(&self) -> MatmulAlgo {
        self.algo
    }
}

impl Gpu {
    /// The loaded cuBLASLt's version, `cublasLtGetVersion`, for tune-file identities
    ///
    /// A [`MatmulAlgo`] is only meaningful to the library version that produced it.
    /// cuBLASLt must be prepared with [`Gpu::prepare`] first
    pub fn cublaslt_version(&self) -> Result<usize, Error> {
        self.lt_handle()?;
        // SAFETY: the library is loaded and checked by the prepare above
        Ok(unsafe { sys::cublasLtGetVersion() })
    }

    /// Up to `limit` algorithms cuBLASLt's heuristic offers for `spec`, best first,
    /// each fitting the shared workspace
    ///
    /// The first is the one cuBLAS itself would run. Timing the rest is how a consumer
    /// builds a tune file; cudakit keeps no tuning of its own
    pub fn matmul_candidates(
        &self,
        spec: Matmul,
        limit: usize,
    ) -> Result<Vec<MatmulCandidate>, Error> {
        let handle = self.lt_handle()?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        let descriptors = Descriptors::new(spec)?;
        let preference = Preference(lt::create_matmul_pref()?);
        let workspace = MATMUL_WORKSPACE_BYTES;
        // SAFETY: the preference is live and the attribute takes a `usize`
        unsafe {
            lt::set_matmul_pref_attribute(
                preference.0,
                cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&raw const workspace).cast(),
                size_of::<usize>(),
            )
        }?;

        let requested = i32::try_from(limit).unwrap_or(i32::MAX);
        // zeroed rather than uninitialized: cuBLASLt leaves some reserved bytes unwritten
        let mut results =
            vec![MaybeUninit::<sys::cublasLtMatmulHeuristicResult_t>::zeroed(); limit];
        let mut found = 0;
        // SAFETY: every descriptor is live and `results` holds `requested` entries
        unsafe {
            sys::cublasLtMatmulAlgoGetHeuristic(
                handle,
                descriptors.desc.0,
                descriptors.first.0,
                descriptors.second.0,
                descriptors.c.0,
                descriptors.c.0,
                preference.0,
                requested,
                results.as_mut_ptr().cast(),
                &raw mut found,
            )
        }
        .result()?;

        let found = usize::try_from(found).unwrap_or(0).min(limit);
        Ok(results[..found]
            .iter()
            // SAFETY: all-zero bits are a valid result, and cuBLASLt filled the first
            // `found` entries' fields
            .map(|result| unsafe { result.assume_init() })
            .filter(|result| result.state == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS)
            .map(|result| MatmulCandidate {
                algo: MatmulAlgo::from_sys(result.algo),
                workspace_bytes: result.workspaceSize,
                waves: result.wavesCount,
            })
            .collect())
    }

    /// `algo`'s configuration, read back for logs and tune-file comments
    ///
    /// cuBLASLt must be prepared with [`Gpu::prepare`] first
    pub fn matmul_algo_config(&self, algo: MatmulAlgo) -> Result<AlgoConfig, Error> {
        self.lt_handle()?;
        let algo = algo.to_sys();
        let read = |attribute: Attribute| -> Result<i64, Error> {
            // the read must name the attribute's exact size, so ask for it first; every
            // config attribute is an integer of at most 8 bytes
            let mut size = 0usize;
            // SAFETY: `algo` is initialized; a zero-size read only reports the size
            unsafe {
                sys::cublasLtMatmulAlgoConfigGetAttribute(
                    &raw const algo,
                    attribute,
                    std::ptr::null_mut(),
                    0,
                    &raw mut size,
                )
            }
            .result()?;
            if size > size_of::<i64>() {
                return Err(Error::Unsupported {
                    context: "matmul algorithm config",
                    reason: format!("{attribute:?} is {size} bytes"),
                });
            }
            // little-endian into a zeroed `i64`, so a narrower unsigned value reads back
            // as itself
            let mut value = 0i64;
            let mut written = 0usize;
            // SAFETY: `value` holds at least `size` writable bytes
            unsafe {
                sys::cublasLtMatmulAlgoConfigGetAttribute(
                    &raw const algo,
                    attribute,
                    (&raw mut value).cast(),
                    size,
                    &raw mut written,
                )
            }
            .result()?;
            Ok(value)
        };
        Ok(AlgoConfig {
            id: read(Attribute::CUBLASLT_ALGO_CONFIG_ID)?,
            tile: read(Attribute::CUBLASLT_ALGO_CONFIG_TILE_ID)?,
            stages: read(Attribute::CUBLASLT_ALGO_CONFIG_STAGES_ID)?,
            split_k: read(Attribute::CUBLASLT_ALGO_CONFIG_SPLITK_NUM)?,
            reduction: read(Attribute::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)?,
            swizzle: read(Attribute::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING)?,
            custom: read(Attribute::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION)?,
        })
    }

    /// A plan for `spec` with the heuristic's first algorithm, the one cuBLAS runs
    pub fn matmul_plan(&self, spec: Matmul) -> Result<MatmulPlan, Error> {
        let first = self.matmul_candidates(spec, 1)?;
        let candidate = first.first().ok_or_else(|| Error::Unsupported {
            context: "matmul plan",
            reason: format!("cuBLASLt offers no algorithm for {spec:?}"),
        })?;
        self.matmul_plan_with(spec, candidate.algo)
    }

    /// A plan for `spec` with `algo`, typically read from a tune file
    ///
    /// Fails with [`Error::Unsupported`] when cuBLASLt rejects `algo` for this product
    /// or it needs more than the shared workspace, so a stale tune entry falls back
    /// instead of running
    pub fn matmul_plan_with(&self, spec: Matmul, algo: MatmulAlgo) -> Result<MatmulPlan, Error> {
        let handle = self.lt_handle()?;
        let descriptors = Descriptors::new(spec)?;
        let raw = algo.to_sys();
        // zeroed: the check leaves the reserved bytes unwritten
        let mut result = MaybeUninit::<sys::cublasLtMatmulHeuristicResult_t>::zeroed();
        // SAFETY: every descriptor is live, `raw` is initialized and `result` writable
        let status = unsafe {
            sys::cublasLtMatmulAlgoCheck(
                handle,
                descriptors.desc.0,
                descriptors.first.0,
                descriptors.second.0,
                descriptors.c.0,
                descriptors.c.0,
                &raw const raw,
                result.as_mut_ptr(),
            )
        }
        .result();
        let rejected = |reason: String| Error::Unsupported {
            context: "matmul plan",
            reason,
        };
        if let Err(error) = status {
            return Err(rejected(format!(
                "cuBLASLt rejects the algorithm for {spec:?}: {error}"
            )));
        }
        // SAFETY: all-zero bits are a valid result, and the check filled its fields
        let result = unsafe { result.assume_init() };
        if result.workspaceSize > MATMUL_WORKSPACE_BYTES {
            return Err(rejected(format!(
                "the algorithm needs {} workspace bytes, over the shared {MATMUL_WORKSPACE_BYTES}",
                result.workspaceSize
            )));
        }
        Ok(MatmulPlan {
            spec,
            algo,
            descriptors,
        })
    }

    /// Runs `plan`: `c = alpha * op(a) · op(b) + beta * c` on this GPU's stream
    ///
    /// `a` and `b` hold the plan's [`Operands`] element type and `c` is FP32. Each
    /// buffer must hold exactly its matrix and start on a [`MATMUL_ALIGNMENT`]-byte
    /// boundary, which a whole allocation always does: cuBLASLt's algorithms assume
    /// it, so a view of a larger buffer must start at a multiple of 16 bytes. Fails
    /// with [`Error::Unsupported`] otherwise. Safe to record in a graph capture: the workspace is one allocation
    /// that lives as long as this `Gpu`
    pub fn matmul<T, A, B, C>(
        &self,
        plan: &MatmulPlan,
        alpha: f32,
        beta: f32,
        a: &A,
        b: &B,
        c: &mut C,
    ) -> Result<(), Error>
    where
        T: MatmulElement,
        A: DevicePtr<T>,
        B: DevicePtr<T>,
        C: DevicePtrMut<f32>,
    {
        let spec = plan.spec;
        if size_of::<T>() != spec.operands.element_bytes() {
            return Err(Error::Unsupported {
                context: "matmul",
                reason: format!(
                    "{:?} operands are {} bytes, the buffers' elements {}",
                    spec.operands,
                    spec.operands.element_bytes(),
                    size_of::<T>()
                ),
            });
        }
        check_len("matmul a", spec.m.saturating_mul(spec.k), a.len())?;
        check_len("matmul b", spec.k.saturating_mul(spec.n), b.len())?;
        check_len("matmul c", spec.m.saturating_mul(spec.n), c.len())?;
        let handle = self.lt_handle()?;
        let misaligned = |name: &str, pointer: u64| {
            (!pointer.is_multiple_of(MATMUL_ALIGNMENT as u64)).then(|| Error::Unsupported {
                context: "matmul",
                reason: format!(
                    "{name} starts at {pointer:#x}, not on a {MATMUL_ALIGNMENT}-byte boundary"
                ),
            })
        };
        let lt = self.libraries().lt()?;

        let stream = self.stream();
        let (a_ptr, _a_sync) = a.device_ptr(stream);
        let (b_ptr, _b_sync) = b.device_ptr(stream);
        let (c_ptr, _c_sync) = c.device_ptr_mut(stream);
        let (workspace, _workspace_sync) = lt.workspace.device_ptr(stream);
        for (name, pointer) in [("a", a_ptr), ("b", b_ptr), ("c", c_ptr)] {
            if let Some(error) = misaligned(name, pointer) {
                return Err(error);
            }
        }
        let algo = plan.algo.to_sys();
        let descriptors = &plan.descriptors;
        // SAFETY: the lengths match the plan's layouts, checked above; every buffer
        // and the workspace are device allocations ordered on this stream, and
        // `alpha` and `beta` are host FP32 scalars as the FP32 scale type takes
        unsafe {
            lt::matmul(
                handle,
                descriptors.desc.0,
                (&raw const alpha).cast(),
                (&raw const beta).cast(),
                b_ptr as *const _,
                descriptors.first.0,
                a_ptr as *const _,
                descriptors.second.0,
                c_ptr as *const _,
                descriptors.c.0,
                c_ptr as *mut _,
                descriptors.c.0,
                &raw const algo,
                workspace as *mut _,
                MATMUL_WORKSPACE_BYTES,
                stream.cu_stream().cast(),
            )
        }?;
        Ok(())
    }

    /// The prepared cuBLASLt handle, with this GPU's context bound to the thread
    fn lt_handle(&self) -> Result<sys::cublasLtHandle_t, Error> {
        self.profile().allow(Library::CublasLt)?;
        let lt = self.libraries().lt()?;
        // cuBLASLt launches on the stream's context, which cudarc does not bind for it
        self.context().bind_to_thread()?;
        Ok(lt.handle.0)
    }
}
