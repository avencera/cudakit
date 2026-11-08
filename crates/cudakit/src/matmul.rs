//! Row-major products on cuBLASLt with an explicit algorithm and a fused epilogue
//!
//! [`Sgemm`](crate::Sgemm) leaves the algorithm to cuBLAS, which runs cuBLASLt's first
//! heuristic. That choice is a model of the GPU, not a measurement: on an RTX 5090 in
//! TF32, other candidates ran 32-row products 1.49x and 201-row products 1.25x faster
//! (sam3rs). A [`MatmulPlan`] instead fixes one algorithm: the heuristic's first, one
//! the consumer picked (for example from a tune file), or the fastest of the top few
//! by [`Gpu::tune_matmul`]. A fixed algorithm also fixes the numerics, since split-K
//! and tile choices change the summation order. Bias, ReLU and GELU run in the
//! product's epilogue instead of a separate pass over the output

use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt;
use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublaslt::sys as lt;
use cudarc::cublaslt::sys::{
    cublasComputeType_t, cublasLtEpilogue_t, cublasLtMatmulAlgo_t,
    cublasLtMatmulAlgoConfigAttributes_t as Attribute, cublasLtMatmulDesc_t,
    cublasLtMatmulDescAttributes_t as DescAttribute, cublasLtMatmulHeuristicResult_t,
    cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t as Preference,
    cublasLtMatrixLayout_t, cublasLtReductionScheme_t as Reduction, cublasStatus_t, cudaDataType,
};
use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{DevicePtr, DevicePtrMut, DeviceRepr};
use tracing::debug;

use crate::error::check_len;
use crate::libraries::MATMUL_WORKSPACE_BYTES;
use crate::{Error, Gpu, Library, Math};

/// Byte alignment every [`Gpu::matmul`] buffer, bias included, must start on
///
/// cudakit asks cuBLASLt's heuristic only for algorithms that work at this alignment,
/// instead of its default 256 bytes, so a view into a larger buffer works when its
/// start is a multiple of 16 bytes
pub const MATMUL_ALIGNMENT: usize = 16;

/// Element type and multiply precision of a [`Matmul`]'s `a` and `b`; accumulation is
/// always FP32
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Operands {
    /// FP32 `a` and `b` (`f32` buffers), multiplied in [`Math`] precision: plain FP32
    /// compute never rounds the inputs, TF32 runs them on tensor cores
    F32(Math),
    /// FP32 `a` and `b` on tensor cores that may round them to BF16
    /// (`CUBLAS_COMPUTE_32F_FAST_16BF`)
    ///
    /// On an RTX 5090 cuBLASLt still picked mostly TF32 kernels for this mode, but it
    /// changed accuracy (sam3rs parity), so compare it with [`Math::Tf32`] by results,
    /// not by name. BF16 tensor cores need BF16 storage, [`Operands::Bf16`]
    F32Fast16Bf,
    /// FP16 `a` and `b` (`u16` buffers of FP16 bit patterns) on tensor cores
    F16,
    /// BF16 `a` and `b` (`u16` buffers of BF16 bit patterns) on tensor cores
    Bf16,
}

impl Operands {
    fn data_type(self) -> cudaDataType {
        match self {
            Self::F32(_) | Self::F32Fast16Bf => cudaDataType::CUDA_R_32F,
            Self::F16 => cudaDataType::CUDA_R_16F,
            Self::Bf16 => cudaDataType::CUDA_R_16BF,
        }
    }

    fn compute_type(self) -> cublasComputeType_t {
        match self {
            // FP32 compute never down-converts FP32 operands, as `Math::Fp32` promises
            Self::F32(Math::Fp32) | Self::F16 | Self::Bf16 => {
                cublasComputeType_t::CUBLAS_COMPUTE_32F
            }
            Self::F32(Math::Tf32) => cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
            Self::F32Fast16Bf => cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF,
        }
    }

    fn element_bytes(self) -> usize {
        match self {
            Self::F32(_) | Self::F32Fast16Bf => size_of::<f32>(),
            Self::F16 | Self::Bf16 => size_of::<u16>(),
        }
    }

    /// A fill value about 1/3 in this type, for tuning operands
    fn third(self) -> Fill {
        match self {
            Self::F32(_) | Self::F32Fast16Bf => Output::F32.third(),
            Self::F16 => Output::F16.third(),
            Self::Bf16 => Output::Bf16.third(),
        }
    }
}

/// Element type of a [`Matmul`]'s output `c` and its bias
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Output {
    /// `f32` buffers
    #[default]
    F32,
    /// `u16` buffers of FP16 bits; needs [`Operands::F16`]
    F16,
    /// `u16` buffers of BF16 bits; needs [`Operands::Bf16`]
    Bf16,
}

impl Output {
    fn data_type(self) -> cudaDataType {
        match self {
            Self::F32 => cudaDataType::CUDA_R_32F,
            Self::F16 => cudaDataType::CUDA_R_16F,
            Self::Bf16 => cudaDataType::CUDA_R_16BF,
        }
    }

    fn element_bytes(self) -> usize {
        match self {
            Self::F32 => size_of::<f32>(),
            Self::F16 | Self::Bf16 => size_of::<u16>(),
        }
    }

    /// Split-K reductions allowed for this output: in-place and output-type schemes
    /// keep partial sums in the output type, which loses FP32 accumulation for a
    /// 16-bit output and can overflow FP16 where the final sum would not
    fn reductions(self) -> u32 {
        match self {
            Self::F32 => Reduction::CUBLASLT_REDUCTION_SCHEME_MASK as u32,
            Self::F16 | Self::Bf16 => Reduction::CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE as u32,
        }
    }

    fn third(self) -> Fill {
        match self {
            Self::F32 => Fill::Word(0x3eaa_aaab),
            Self::F16 => Fill::Half(0x3555),
            Self::Bf16 => Fill::Half(0x3eab),
        }
    }
}

/// What a [`Matmul`]'s epilogue applies to `alpha * op(a) · op(b) + beta * c`
///
/// The bias is a vector of `n` values, one per output column, in the
/// [`Output`] type: cuBLASLt takes the bias in the output's type, so a BF16-output
/// product needs a BF16 copy of an FP32 bias. GELU is the tanh approximation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Epilogue {
    /// The product alone
    #[default]
    None,
    /// Adds the bias to every row
    Bias,
    /// ReLU of the product
    Relu,
    /// ReLU of the product plus the bias
    ReluBias,
    /// GELU of the product
    ///
    /// With TF32 operands on an RTX 5090, cuBLASLt ran the GELU epilogues as a GEMM
    /// plus a separate epilogue kernel (86 µs per call in sam3rs) and none of the top 8
    /// candidates fused them, while bias-only epilogues fused. Check the kernels nsys
    /// shows before relying on the fusion for FP32 operands
    Gelu,
    /// GELU of the product plus the bias; see [`Epilogue::Gelu`]
    GeluBias,
}

impl Epilogue {
    fn raw(self) -> cublasLtEpilogue_t {
        match self {
            Self::None => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
            Self::Bias => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
            Self::Relu => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU,
            Self::ReluBias => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS,
            Self::Gelu => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU,
            Self::GeluBias => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS,
        }
    }

    /// Whether the epilogue reads a bias, which [`Gpu::matmul_bias`] supplies
    pub fn has_bias(self) -> bool {
        matches!(self, Self::Bias | Self::ReluBias | Self::GeluBias)
    }
}

/// Element type of a [`Matmul`] buffer: `f32`, or `u16` holding FP16 or BF16 bits
pub trait MatmulElement: DeviceRepr + sealed::Sealed {}

impl MatmulElement for f32 {}
impl MatmulElement for u16 {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u16 {}
}

/// A row-major product `c = epilogue(alpha * op(a) · op(b) + beta * c)` for a
/// [`MatmulPlan`]
///
/// The layout conventions are [`Sgemm`](crate::Sgemm)'s: `op(a)` is `m × k`, `op(b)`
/// is `k × n` and `c` is `m × n`, all row-major, and a transposed operand is stored as
/// its transpose. A stride is the number of elements from one stored row to the next;
/// `None` packs the rows densely. A stride below the stored row length would make rows
/// overlap, which cuBLASLt rejects, so cudakit refuses it up front: a 3x3 convolution
/// as 3 products over windows that slide by one row is such a case, and its 9-tap form
/// with one product per tap works
///
/// `alpha` and `beta` are given per call, so one plan serves a product with and
/// without accumulation into `c`
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
    /// Type and precision of `a` and `b`
    pub operands: Operands,
    /// Type of `c` and the bias
    pub output: Output,
    /// What runs after the product
    pub epilogue: Epilogue,
    /// Elements between stored rows of `a`, or `None` for dense rows
    pub a_stride: Option<usize>,
    /// Elements between stored rows of `b`, or `None` for dense rows
    pub b_stride: Option<usize>,
    /// Elements between rows of `c`, or `None` for dense rows
    pub c_stride: Option<usize>,
}

impl Matmul {
    /// `c = a · b` with untransposed, dense operands, FP32 output and no epilogue
    pub fn new(m: usize, n: usize, k: usize, operands: Operands) -> Self {
        Self {
            m,
            n,
            k,
            a_transposed: false,
            b_transposed: false,
            operands,
            output: Output::F32,
            epilogue: Epilogue::None,
            a_stride: None,
            b_stride: None,
            c_stride: None,
        }
    }

    /// A PyTorch `Linear`: `c [rows, outputs] = a [rows, inputs] · wᵀ`, with `w` the
    /// `[outputs, inputs]` weight as `b`; add the bias with [`Self::with_epilogue`]
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

    /// The same product writing `output`
    pub fn with_output(self, output: Output) -> Self {
        Self { output, ..self }
    }

    /// The same product with `epilogue`
    pub fn with_epilogue(self, epilogue: Epilogue) -> Self {
        Self { epilogue, ..self }
    }

    /// The same product with these row strides, `None` for dense rows
    pub fn with_strides(self, a: Option<usize>, b: Option<usize>, c: Option<usize>) -> Self {
        Self {
            a_stride: a,
            b_stride: b,
            c_stride: c,
            ..self
        }
    }

    /// The stored `a`, `b` and `c`, after checking the dimensions, strides and types
    fn geometry(&self) -> Result<[Stored; 3], Error> {
        let refuse = |reason: String| Error::Unsupported {
            context: "matmul",
            reason,
        };
        let (m, n, k) = (self.m, self.n, self.k);
        if m == 0 || n == 0 || k == 0 {
            return Err(refuse(format!(
                "m, n and k must be positive, not {m}, {n} and {k}"
            )));
        }
        let compatible = match self.output {
            Output::F32 => true,
            Output::F16 => self.operands == Operands::F16,
            Output::Bf16 => self.operands == Operands::Bf16,
        };
        if !compatible {
            return Err(refuse(format!(
                "{:?} output needs operands of the same type, not {:?}",
                self.output, self.operands
            )));
        }

        let stored = |name: &'static str, rows: usize, cols: usize, stride: Option<usize>| {
            let stride = stride.unwrap_or(cols);
            if stride < cols {
                return Err(refuse(format!(
                    "{name}'s row stride {stride} is below its {cols} columns, so its rows \
                     would overlap, which cuBLASLt rejects; use one product per row offset"
                )));
            }
            let span = (rows - 1)
                .checked_mul(stride)
                .and_then(|start| start.checked_add(cols))
                .ok_or(Error::DimensionOverflow {
                    context: "matmul",
                    value: stride,
                })?;
            Ok(Stored {
                rows,
                cols,
                stride,
                span,
            })
        };
        let a = if self.a_transposed { (k, m) } else { (m, k) };
        let b = if self.b_transposed { (n, k) } else { (k, n) };
        Ok([
            stored("a", a.0, a.1, self.a_stride)?,
            stored("b", b.0, b.1, self.b_stride)?,
            stored("c", m, n, self.c_stride)?,
        ])
    }
}

/// A row-major matrix as stored: `rows × cols` with `stride` elements between rows
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Stored {
    rows: usize,
    cols: usize,
    stride: usize,
    // elements from the first to one past the last
    span: usize,
}

/// `alpha` and `beta` of `c = alpha * op(a) · op(b) + beta * c`, host FP32 scalars
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scale {
    /// Scale for the product
    pub alpha: f32,
    /// Scale for the existing `c`; 0 overwrites it without reading it
    pub beta: f32,
}

impl Scale {
    /// `c = op(a) · op(b)`
    pub const OVERWRITE: Self = Self::new(1.0, 0.0);
    /// `c += op(a) · op(b)`, such as a residual connection
    pub const ACCUMULATE: Self = Self::new(1.0, 1.0);

    /// `alpha` and `beta`
    pub const fn new(alpha: f32, beta: f32) -> Self {
        Self { alpha, beta }
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
    fn from_sys(algo: cublasLtMatmulAlgo_t) -> Self {
        Self(algo.data)
    }

    fn to_sys(self) -> cublasLtMatmulAlgo_t {
        cublasLtMatmulAlgo_t { data: self.0 }
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

/// How [`Gpu::tune_matmul`] times the heuristic's candidates
///
/// The defaults are sam3rs's: 8 candidates, about 0.25 ms of launches per timing,
/// best of 2, and a candidate replaces the heuristic's first only when it is over 5%
/// faster. On an RTX 5090 that picked the same algorithms as 1 ms windows and best of
/// 3, for a third of the first-call cost (about 100 ms over a SAM 3 encode)
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Tuning {
    /// Heuristic candidates to time, best first
    pub candidates: usize,
    /// How much faster than the heuristic's first a candidate must be to replace it
    ///
    /// Without a margin near-ties flip on timing noise, so the chosen algorithm, and
    /// with it the summation order and the output bits, would change between
    /// processes. With 1.05, sam3rs's TF32 outputs hashed the same over three fresh
    /// processes, twice; a candidate near the margin can still win in one process and
    /// lose in the next, so persist the chosen [`MatmulAlgo`] where outputs must match
    /// across runs
    pub margin: f32,
    /// Milliseconds of back-to-back launches per timing
    pub window_ms: f32,
    /// Timings per candidate, of which the fastest counts
    pub rounds: usize,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            candidates: 8,
            margin: 1.05,
            window_ms: 0.25,
            rounds: 2,
        }
    }
}

impl Tuning {
    /// The same tuning with `candidates` candidates
    pub fn with_candidates(self, candidates: usize) -> Self {
        Self { candidates, ..self }
    }

    /// The same tuning with `margin`
    pub fn with_margin(self, margin: f32) -> Self {
        Self { margin, ..self }
    }

    /// The same tuning with `window_ms` milliseconds per timing and `rounds` timings
    pub fn with_timing(self, window_ms: f32, rounds: usize) -> Self {
        Self {
            window_ms,
            rounds,
            ..self
        }
    }

    fn check(&self) -> Result<(), Error> {
        let valid = self.candidates > 0
            && self.margin.is_finite()
            && self.margin >= 1.0
            && self.window_ms.is_finite()
            && self.window_ms > 0.0
            && self.rounds > 0;
        if valid {
            return Ok(());
        }

        Err(Error::Unsupported {
            context: "matmul tuning",
            reason: format!(
                "{self:?}: candidates and rounds must be positive, the margin at least 1 \
                 and the window positive"
            ),
        })
    }
}

/// One candidate's time in a [`TuneReport`]
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct CandidateTiming {
    /// The candidate
    pub candidate: MatmulCandidate,
    /// Best milliseconds per launch, or `None` when it failed to launch
    pub ms: Option<f32>,
}

/// What [`Gpu::tune_matmul`] measured
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct TuneReport {
    /// Every candidate in the heuristic's order; empty when it offered only one, which
    /// is then chosen without timing
    pub timings: Vec<CandidateTiming>,
    /// Index of the chosen candidate in the heuristic's order
    pub chosen: usize,
}

/// The candidate to keep: the heuristic's first unless another is faster by more than
/// `margin`, or the fastest when the first failed; `None` when none ran
fn choose(times: &[Option<f32>], margin: f32) -> Option<usize> {
    let (fastest, best) = times
        .iter()
        .enumerate()
        .filter_map(|(index, time)| time.map(|time| (index, time)))
        .min_by(|left, right| left.1.total_cmp(&right.1))?;
    match times.first().copied().flatten() {
        Some(first) if first <= best * margin => Some(0),
        _ => Some(fastest),
    }
}

/// A device fill pattern for tuning buffers
#[derive(Debug, Clone, Copy)]
enum Fill {
    Word(u32),
    Half(u16),
}

struct Desc(cublasLtMatmulDesc_t);

// SAFETY: a descriptor is a host structure that cuBLASLt reads during a call and that
// cudakit writes only behind the plan's lock, so it may move between threads
unsafe impl Send for Desc {}

impl Drop for Desc {
    fn drop(&mut self) {
        // SAFETY: created by this library's `cublasLtMatmulDescCreate`, destroyed once
        let _ = unsafe { lt::cublasLtMatmulDescDestroy(self.0) };
    }
}

impl Desc {
    /// Writes a descriptor attribute
    ///
    /// # Safety
    ///
    /// `T` must be the attribute's C type
    unsafe fn set<T>(&self, attribute: DescAttribute, value: &T) -> Result<(), Error> {
        // SAFETY: the descriptor is live and the caller names the attribute's type
        unsafe {
            lt::cublasLtMatmulDescSetAttribute(
                self.0,
                attribute,
                (value as *const T).cast(),
                size_of::<T>(),
            )
        }
        .result()?;
        Ok(())
    }
}

struct Layout(cublasLtMatrixLayout_t);

// SAFETY: a layout is a host structure that cudakit never changes after creating it;
// cuBLASLt only reads it, which is safe from several threads
unsafe impl Send for Layout {}
// SAFETY: see `Send`
unsafe impl Sync for Layout {}

impl Drop for Layout {
    fn drop(&mut self) {
        // SAFETY: created by this library's `cublasLtMatrixLayoutCreate`, destroyed once
        let _ = unsafe { lt::cublasLtMatrixLayoutDestroy(self.0) };
    }
}

struct PreferenceHandle(cublasLtMatmulPreference_t);

impl Drop for PreferenceHandle {
    fn drop(&mut self) {
        // SAFETY: created by this library's `cublasLtMatmulPreferenceCreate`, once
        let _ = unsafe { lt::cublasLtMatmulPreferenceDestroy(self.0) };
    }
}

impl PreferenceHandle {
    /// The heuristic's limits: the shared workspace, [`MATMUL_ALIGNMENT`] and the
    /// reductions `output` allows
    fn new(output: Output) -> Result<Self, Error> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: `raw` is a writable slot
        unsafe { lt::cublasLtMatmulPreferenceCreate(&mut raw) }.result()?;
        let preference = Self(raw);
        let set = |attribute: Preference, value: *const c_void, size: usize| {
            // SAFETY: the preference is live and each value has its attribute's type
            unsafe { lt::cublasLtMatmulPreferenceSetAttribute(raw, attribute, value, size) }
                .result()
        };
        let workspace = MATMUL_WORKSPACE_BYTES as u64;
        set(
            Preference::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            (&raw const workspace).cast(),
            size_of::<u64>(),
        )?;
        let alignment = MATMUL_ALIGNMENT as u32;
        for attribute in [
            Preference::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
            Preference::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
            Preference::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
            Preference::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
        ] {
            set(attribute, (&raw const alignment).cast(), size_of::<u32>())?;
        }

        let reductions = output.reductions();
        set(
            Preference::CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
            (&raw const reductions).cast(),
            size_of::<u32>(),
        )?;
        Ok(preference)
    }
}

/// The cuBLASLt descriptors of one [`Matmul`] in cuBLAS's column-major terms
struct Descriptors {
    // written per call with the bias pointer, so every call holds the lock until its
    // product is queued
    desc: Mutex<Desc>,
    // `first` is the row-major `b`, `second` the row-major `a`; see `Descriptors::new`
    first: Layout,
    second: Layout,
    c: Layout,
    geometry: [Stored; 3],
}

impl Descriptors {
    /// Row-major `c` (`m × n`) is column-major `cᵀ` (`n × m`), so cuBLASLt computes
    /// `cᵀ = op(b)ᵀ · op(a)ᵀ` with `b` first, as [`Sgemm`](crate::Sgemm) does. A layout
    /// describes a matrix as stored: a row-major `r × c` buffer with row stride `s`
    /// read column-major is `c × r` with leading dimension `s`
    fn new(spec: Matmul) -> Result<Self, Error> {
        let geometry = spec.geometry()?;
        let [a, b, c] = geometry;
        let mut raw = std::ptr::null_mut();
        // SAFETY: `raw` is a writable slot; the scale type is FP32 for every compute
        // type cudakit uses
        unsafe {
            lt::cublasLtMatmulDescCreate(
                &mut raw,
                spec.operands.compute_type(),
                cudaDataType::CUDA_R_32F,
            )
        }
        .result()?;
        let desc = Desc(raw);
        let op = |transposed: bool| {
            if transposed {
                cublasOperation_t::CUBLAS_OP_T
            } else {
                cublasOperation_t::CUBLAS_OP_N
            }
        };
        // SAFETY (each set): the attribute takes a `cublasOperation_t`, a
        // `cublasLtEpilogue_t` or a `cudaDataType`, all 32-bit enums
        unsafe {
            desc.set(
                DescAttribute::CUBLASLT_MATMUL_DESC_TRANSA,
                &op(spec.b_transposed),
            )?;
            desc.set(
                DescAttribute::CUBLASLT_MATMUL_DESC_TRANSB,
                &op(spec.a_transposed),
            )?;
            desc.set(
                DescAttribute::CUBLASLT_MATMUL_DESC_EPILOGUE,
                &spec.epilogue.raw(),
            )?;
            if spec.epilogue.has_bias() {
                // the bias always has the output's type, which cuBLASLt requires here
                desc.set(
                    DescAttribute::CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                    &spec.output.data_type(),
                )?;
            }
        }

        let layout = |data_type: cudaDataType, stored: Stored| -> Result<Layout, Error> {
            let dim = |value: usize| {
                u64::try_from(value).map_err(|_| Error::DimensionOverflow {
                    context: "matmul",
                    value,
                })
            };
            let lead = i64::try_from(stored.stride).map_err(|_| Error::DimensionOverflow {
                context: "matmul leading dimension",
                value: stored.stride,
            })?;
            let mut raw = std::ptr::null_mut();
            // SAFETY: `raw` is a writable slot
            unsafe {
                lt::cublasLtMatrixLayoutCreate(
                    &mut raw,
                    data_type,
                    dim(stored.cols)?,
                    dim(stored.rows)?,
                    lead,
                )
            }
            .result()?;
            Ok(Layout(raw))
        };
        let operand = spec.operands.data_type();
        Ok(Self {
            first: layout(operand, b)?,
            second: layout(operand, a)?,
            c: layout(spec.output.data_type(), c)?,
            desc: Mutex::new(desc),
            geometry,
        })
    }

    /// The descriptor, behind the per-call lock
    fn desc(&self) -> std::sync::MutexGuard<'_, Desc> {
        self.desc.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// A [`Matmul`] bound to one cuBLASLt algorithm, run with [`Gpu::matmul`] or
/// [`Gpu::matmul_bias`]
///
/// Building a plan only queries the library on the host, so it works during a graph
/// capture; [`Gpu::tune_matmul`] does not. A plan holds only host descriptors, so it is
/// cheap to keep one per shape, as [`MatmulCache`] does. It belongs to the [`Gpu`]
/// whose heuristic chose its algorithm, and another GPU's [`Gpu::matmul`] refuses it
pub struct MatmulPlan {
    owner: Owner,
    spec: Matmul,
    algo: MatmulAlgo,
    workspace_bytes: usize,
    descriptors: Descriptors,
}

impl fmt::Debug for MatmulPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatmulPlan")
            .field("spec", &self.spec)
            .field("algo", &self.algo)
            .field("workspace_bytes", &self.workspace_bytes)
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

    /// Workspace bytes the algorithm uses, at most [`MATMUL_WORKSPACE_BYTES`]
    pub fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }
}

/// The device and context a plan's algorithm was chosen for: an algorithm is only
/// valid on the device it came from
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Owner(usize, usize);

impl Owner {
    fn of(gpu: &Gpu) -> Self {
        let context = gpu.context();
        Self(context.ordinal(), context.cu_ctx() as usize)
    }

    fn check(self, gpu: &Gpu, context: &'static str) -> Result<(), Error> {
        if self == Self::of(gpu) {
            return Ok(());
        }

        Err(Error::Unsupported {
            context,
            reason: "the plan belongs to another GPU".to_owned(),
        })
    }
}

/// Plans for one [`Gpu`], built on first use of each [`Matmul`] and shared after that
///
/// With a [`Tuning`], a new shape is timed by [`Gpu::tune_matmul`], which synchronizes
/// the stream, so every shape must be seen once before a graph capture: a new shape
/// during a capture fails with [`Error::Capturing`] instead of syncing inside it, while
/// a cached one works. Without a tuning, a new shape takes the heuristic's first
/// algorithm, which needs no stream work and so works during a capture too
pub struct MatmulCache {
    tuning: Option<Tuning>,
    // the GPU the plans belong to, set on first use
    owner: OnceLock<Owner>,
    plans: Mutex<HashMap<Matmul, Arc<MatmulPlan>>>,
}

impl fmt::Debug for MatmulCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatmulCache")
            .field("tuning", &self.tuning)
            .field("plans", &self.len())
            .finish_non_exhaustive()
    }
}

impl MatmulCache {
    /// An empty cache that tunes new shapes with `tuning`, or takes the heuristic's
    /// first algorithm with `None`
    pub fn new(tuning: Option<Tuning>) -> Self {
        Self {
            tuning,
            owner: OnceLock::new(),
            plans: Mutex::new(HashMap::new()),
        }
    }

    /// The plan for `spec`, built on `gpu` when the cache has none
    ///
    /// Fails with [`Error::Unsupported`] when `gpu` is not the GPU the cache's plans
    /// were built on. The lock is held while a plan is built, so threads asking for a
    /// new shape at once tune it once
    pub fn plan(&self, gpu: &Gpu, spec: Matmul) -> Result<Arc<MatmulPlan>, Error> {
        self.check_gpu(gpu)?;
        let mut plans = self.plans.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(plan) = plans.get(&spec) {
            return Ok(plan.clone());
        }

        let plan = match self.tuning {
            Some(tuning) => gpu.tune_matmul(spec, &tuning)?.0,
            None => gpu.matmul_plan(spec)?,
        };
        let plan = Arc::new(plan);
        plans.insert(spec, plan.clone());
        Ok(plan)
    }

    /// Adds `plan`, for example one built from a tune file, replacing any plan for the
    /// same product
    ///
    /// Fails with [`Error::Unsupported`] when `gpu` or `plan` belongs to another GPU
    pub fn insert(&self, gpu: &Gpu, plan: MatmulPlan) -> Result<Arc<MatmulPlan>, Error> {
        self.check_gpu(gpu)?;
        plan.owner.check(gpu, "matmul cache")?;
        let plan = Arc::new(plan);
        self.plans
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(plan.spec, plan.clone());
        Ok(plan)
    }

    /// Number of cached plans
    pub fn len(&self) -> usize {
        self.plans
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Whether the cache holds no plan
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn check_gpu(&self, gpu: &Gpu) -> Result<(), Error> {
        self.owner
            .get_or_init(|| Owner::of(gpu))
            .check(gpu, "matmul cache")
    }
}

impl Gpu {
    /// The loaded cuBLASLt's version, `cublasLtGetVersion`, for tune-file identities
    ///
    /// A [`MatmulAlgo`] is only meaningful to the library version that produced it.
    /// cuBLASLt must be prepared with [`Gpu::prepare`] first
    pub fn cublaslt_version(&self) -> Result<usize, Error> {
        self.lt()?;
        // SAFETY: the library is loaded and the call takes no arguments
        Ok(unsafe { lt::cublasLtGetVersion() })
    }

    /// Up to `limit` algorithms cuBLASLt's heuristic offers for `spec`, best first,
    /// each fitting the shared workspace and [`MATMUL_ALIGNMENT`]
    ///
    /// The first is the heuristic's own pick under cudakit's alignment and workspace
    /// limits, which can differ from the algorithm [`Sgemm`](crate::Sgemm) runs. Timing
    /// the rest is what [`Gpu::tune_matmul`] does; a consumer can also time them itself
    /// for a tune file
    pub fn matmul_candidates(
        &self,
        spec: Matmul,
        limit: usize,
    ) -> Result<Vec<MatmulCandidate>, Error> {
        let handle = self.lt()?;
        let descriptors = Descriptors::new(spec)?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        let preference = PreferenceHandle::new(spec.output)?;
        let requested = i32::try_from(limit).unwrap_or(i32::MAX);
        let slots = usize::try_from(requested).unwrap_or(limit);
        // zeroed rather than uninitialized: cuBLASLt leaves some reserved bytes unwritten
        let mut results = vec![MaybeUninit::<cublasLtMatmulHeuristicResult_t>::zeroed(); slots];
        let mut found = 0;
        let desc = descriptors.desc();
        // SAFETY: every descriptor is live and `results` holds `requested` entries
        unsafe {
            lt::cublasLtMatmulAlgoGetHeuristic(
                handle,
                desc.0,
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

        let found = usize::try_from(found).unwrap_or(0).min(slots);
        Ok(results[..found]
            .iter()
            // SAFETY: all-zero bits are a valid result, and cuBLASLt filled the first
            // `found` entries' fields
            .map(|result| unsafe { result.assume_init() })
            // an entry's algorithm is only valid when its own state is success
            .filter(|result| result.state == cublasStatus_t::CUBLAS_STATUS_SUCCESS)
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
        self.lt()?;
        let algo = algo.to_sys();
        let read = |attribute: Attribute| -> Result<i64, Error> {
            // the read must name the attribute's exact size, so ask for it first; every
            // config attribute is an integer of at most 8 bytes
            let mut size = 0usize;
            // SAFETY: `algo` is initialized; a zero-size read only reports the size
            unsafe {
                lt::cublasLtMatmulAlgoConfigGetAttribute(
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
                lt::cublasLtMatmulAlgoConfigGetAttribute(
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
        let candidate = first.first().ok_or_else(|| no_algorithm(spec))?;
        self.matmul_plan_with(spec, candidate.algo)
    }

    /// A plan for `spec` with `algo`, typically read from a tune file
    ///
    /// Fails with [`Error::Unsupported`] when cuBLASLt rejects `algo` for this product,
    /// it needs more than the shared workspace, or it reduces split-K partial sums in a
    /// 16-bit output type, so a stale tune entry falls back instead of running. The check sees no pointers, so store only algorithms from
    /// [`Gpu::matmul_candidates`] or [`Gpu::tune_matmul`], which offer only ones that
    /// work at [`MATMUL_ALIGNMENT`]
    pub fn matmul_plan_with(&self, spec: Matmul, algo: MatmulAlgo) -> Result<MatmulPlan, Error> {
        let handle = self.lt()?;
        let descriptors = Descriptors::new(spec)?;
        let raw = algo.to_sys();
        // zeroed: the check leaves the reserved bytes unwritten
        let mut result = MaybeUninit::<cublasLtMatmulHeuristicResult_t>::zeroed();
        let status = {
            let desc = descriptors.desc();
            // SAFETY: every descriptor is live, `raw` is initialized and `result`
            // writable
            unsafe {
                lt::cublasLtMatmulAlgoCheck(
                    handle,
                    desc.0,
                    descriptors.first.0,
                    descriptors.second.0,
                    descriptors.c.0,
                    descriptors.c.0,
                    &raw const raw,
                    result.as_mut_ptr(),
                )
            }
            .result()
        };
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
        // the check accepts any reduction, so hold a restored algorithm to the rule
        // the heuristic follows for this output
        let reduction = self.matmul_algo_config(algo)?.reduction;
        if u32::try_from(reduction).map_or(true, |scheme| scheme & !spec.output.reductions() != 0) {
            return Err(rejected(format!(
                "the algorithm reduces split-K sums with scheme {reduction}, which {:?} output does not allow",
                spec.output
            )));
        }

        if result.workspaceSize > MATMUL_WORKSPACE_BYTES {
            return Err(rejected(format!(
                "the algorithm needs {} workspace bytes, over the shared {MATMUL_WORKSPACE_BYTES}",
                result.workspaceSize
            )));
        }
        Ok(MatmulPlan {
            owner: Owner::of(self),
            spec,
            algo,
            workspace_bytes: result.workspaceSize,
            descriptors,
        })
    }

    /// A plan for `spec` with the fastest of the heuristic's top candidates, as
    /// [`Tuning`] describes, and what was measured
    ///
    /// Each candidate runs on scratch buffers this call allocates and fills with a
    /// constant, with `beta = 0`, so the caller's data is untouched. It synchronizes the
    /// stream to read the timings, so it fails with [`Error::Capturing`] during a graph
    /// capture of this GPU's stream; tune every shape before capturing. Timings come
    /// from whatever else shares the GPU, so tune on an idle device
    ///
    /// Candidates are launched eagerly, so a product of a few microseconds is timed
    /// partly by how fast the host queues launches: on an RTX 5090, a 201×256×256
    /// TF32 product picked a candidate 1.2x faster by this timing that ran no faster
    /// in a paired comparison, while 32-row products kept their 1.45-1.58x wins.
    /// Such a pick costs no speed but changes the summation order
    pub fn tune_matmul(
        &self,
        spec: Matmul,
        tuning: &Tuning,
    ) -> Result<(MatmulPlan, TuneReport), Error> {
        tuning.check()?;
        self.refuse_while_capturing("matmul tuning")?;
        let candidates = self.matmul_candidates(spec, tuning.candidates)?;
        if candidates.len() <= 1 {
            let candidate = candidates.first().ok_or_else(|| no_algorithm(spec))?;
            let report = TuneReport {
                timings: Vec::new(),
                chosen: 0,
            };
            return Ok((self.matmul_plan_with(spec, candidate.algo)?, report));
        }

        let times = self.time_candidates(spec, tuning, &candidates)?;
        let chosen = choose(&times, tuning.margin).ok_or_else(|| Error::Unsupported {
            context: "matmul tuning",
            reason: format!("none of cuBLASLt's {} candidates ran {spec:?}", times.len()),
        })?;
        debug!(
            ?spec,
            chosen,
            first_ms = ?times[0],
            chosen_ms = ?times[chosen],
            "Tuned matmul"
        );
        let report = TuneReport {
            timings: candidates
                .iter()
                .zip(&times)
                .map(|(candidate, ms)| CandidateTiming {
                    candidate: *candidate,
                    ms: *ms,
                })
                .collect(),
            chosen,
        };
        Ok((
            self.matmul_plan_with(spec, candidates[chosen].algo)?,
            report,
        ))
    }

    /// Best milliseconds per launch of each candidate, `None` for one that fails
    fn time_candidates(
        &self,
        spec: Matmul,
        tuning: &Tuning,
        candidates: &[MatmulCandidate],
    ) -> Result<Vec<Option<f32>>, Error> {
        self.lt()?;
        let descriptors = Descriptors::new(spec)?;
        let [a, b, c] = descriptors.geometry;
        let stream = self.stream();
        let filled = |elements: usize, bytes: usize, fill: Fill| {
            let mut buffer = stream.alloc_zeros::<u8>(elements * bytes)?;
            let (pointer, sync) = buffer.device_ptr_mut(stream);
            // SAFETY: the buffer holds `elements` values of the fill's width, and the
            // memset is ordered on its stream
            unsafe {
                match fill {
                    Fill::Word(value) => cudarc::driver::sys::cuMemsetD32Async(
                        pointer,
                        value,
                        elements,
                        stream.cu_stream(),
                    ),
                    Fill::Half(value) => cudarc::driver::sys::cuMemsetD16Async(
                        pointer,
                        value,
                        elements,
                        stream.cu_stream(),
                    ),
                }
            }
            .result()?;
            drop(sync);
            Ok::<_, Error>(buffer)
        };
        let operand_bytes = spec.operands.element_bytes();
        let a_buffer = filled(a.span, operand_bytes, spec.operands.third())?;
        let b_buffer = filled(b.span, operand_bytes, spec.operands.third())?;
        let output_bytes = spec.output.element_bytes();
        let bias_buffer = filled(spec.n, output_bytes, spec.output.third())?;
        let mut c_buffer = stream.alloc_zeros::<u8>(c.span * output_bytes)?;

        let (a_ptr, _a_sync) = a_buffer.device_ptr(stream);
        let (b_ptr, _b_sync) = b_buffer.device_ptr(stream);
        let (bias_ptr, _bias_sync) = bias_buffer.device_ptr(stream);
        let (c_ptr, _c_sync) = c_buffer.device_ptr_mut(stream);
        let bias = spec.epilogue.has_bias().then_some(bias_ptr);
        let flags = Some(CUevent_flags::CU_EVENT_DEFAULT);
        let (start, end) = (
            self.context().new_event(flags)?,
            self.context().new_event(flags)?,
        );
        // milliseconds per launch over `launches` launches
        let time = |algo: &cublasLtMatmulAlgo_t, launches: usize| -> Result<f32, Error> {
            start.record(stream)?;
            for _ in 0..launches {
                let pointers = Pointers {
                    a: a_ptr,
                    b: b_ptr,
                    bias,
                    c: c_ptr,
                };
                self.launch(&descriptors, Scale::OVERWRITE, pointers, algo)?;
            }
            end.record(stream)?;
            end.synchronize()?;
            Ok(start.elapsed_ms(&end)? / launches as f32)
        };

        let mut times = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let algo = candidate.algo.to_sys();
            // the first launch warms up and sizes the window; one that fails never wins
            let Ok(first) = time(&algo, 1) else {
                times.push(None);
                continue;
            };
            let launches = ((tuning.window_ms / first.max(1e-3)) as usize).clamp(2, 50);
            let mut best = f32::INFINITY;
            for _ in 0..tuning.rounds {
                best = best.min(time(&algo, launches)?);
            }
            times.push(Some(best));
        }
        Ok(times)
    }

    /// Runs `plan`: `c = epilogue(alpha * op(a) · op(b) + beta * c)` on this GPU's
    /// stream, for a plan whose epilogue takes no bias
    ///
    /// `a` and `b` hold the plan's [`Operands`] element type and `c` its [`Output`]
    /// type. Each buffer must hold exactly its stored matrix, `(rows - 1) * stride +
    /// columns` elements, and start on a [`MATMUL_ALIGNMENT`]-byte boundary, which a
    /// whole allocation always does. Fails with [`Error::Unsupported`] otherwise.
    /// Safe to record in a graph capture: the workspace is one allocation that lives
    /// as long as this `Gpu`, so a graph holding the product must not be launched
    /// after the `Gpu` drops. Buffers may be any [`DevicePtr`], including a consumer's
    /// own type over memory from a private pool
    pub fn matmul<T, U, A, B, C>(
        &self,
        plan: &MatmulPlan,
        scale: Scale,
        a: &A,
        b: &B,
        c: &mut C,
    ) -> Result<(), Error>
    where
        T: MatmulElement,
        U: MatmulElement,
        A: DevicePtr<T>,
        B: DevicePtr<T>,
        C: DevicePtrMut<U>,
    {
        if plan.spec.epilogue.has_bias() {
            return Err(Error::Unsupported {
                context: "matmul",
                reason: format!(
                    "the {:?} epilogue reads a bias; run it with `matmul_bias`",
                    plan.spec.epilogue
                ),
            });
        }
        self.matmul_inner(plan, scale, a, b, None::<&cudarc::driver::CudaSlice<U>>, c)
    }

    /// [`Gpu::matmul`] for a plan whose epilogue adds a bias: `bias` holds `n` values
    /// of the plan's [`Output`] type, the same element type as `c`, and starts on a
    /// [`MATMUL_ALIGNMENT`]-byte boundary
    pub fn matmul_bias<T, U, A, B, V, C>(
        &self,
        plan: &MatmulPlan,
        scale: Scale,
        a: &A,
        b: &B,
        bias: &V,
        c: &mut C,
    ) -> Result<(), Error>
    where
        T: MatmulElement,
        U: MatmulElement,
        A: DevicePtr<T>,
        B: DevicePtr<T>,
        V: DevicePtr<U>,
        C: DevicePtrMut<U>,
    {
        if !plan.spec.epilogue.has_bias() {
            return Err(Error::Unsupported {
                context: "matmul",
                reason: format!(
                    "the {:?} epilogue reads no bias; run it with `matmul`",
                    plan.spec.epilogue
                ),
            });
        }
        self.matmul_inner(plan, scale, a, b, Some(bias), c)
    }

    fn matmul_inner<T, U, A, B, V, C>(
        &self,
        plan: &MatmulPlan,
        scale: Scale,
        a: &A,
        b: &B,
        bias: Option<&V>,
        c: &mut C,
    ) -> Result<(), Error>
    where
        T: MatmulElement,
        U: MatmulElement,
        A: DevicePtr<T>,
        B: DevicePtr<T>,
        V: DevicePtr<U>,
        C: DevicePtrMut<U>,
    {
        plan.owner.check(self, "matmul")?;
        let spec = plan.spec;
        let element = |name: &str, expected: usize, actual: usize| {
            if expected == actual {
                return Ok(());
            }
            Err(Error::Unsupported {
                context: "matmul",
                reason: format!(
                    "{name} elements must be {expected} bytes for {spec:?}, not {actual}"
                ),
            })
        };
        element("a and b", spec.operands.element_bytes(), size_of::<T>())?;
        element("c and bias", spec.output.element_bytes(), size_of::<U>())?;
        let [a_shape, b_shape, c_shape] = plan.descriptors.geometry;
        check_len("matmul a", a_shape.span, a.len())?;
        check_len("matmul b", b_shape.span, b.len())?;
        check_len("matmul c", c_shape.span, c.len())?;
        if let Some(bias) = bias {
            check_len("matmul bias", spec.n, bias.len())?;
        }

        let stream = self.stream();
        let (a_ptr, _a_sync) = a.device_ptr(stream);
        let (b_ptr, _b_sync) = b.device_ptr(stream);
        let bias = bias.map(|bias| bias.device_ptr(stream));
        let (c_ptr, _c_sync) = c.device_ptr_mut(stream);
        let pointers = Pointers {
            a: a_ptr,
            b: b_ptr,
            bias: bias.as_ref().map(|(pointer, _)| *pointer),
            c: c_ptr,
        };
        pointers.check_alignment()?;
        self.launch(&plan.descriptors, scale, pointers, &plan.algo.to_sys())
    }

    /// Queues one product on this GPU's stream
    fn launch(
        &self,
        descriptors: &Descriptors,
        scale: Scale,
        pointers: Pointers,
        algo: &cublasLtMatmulAlgo_t,
    ) -> Result<(), Error> {
        let handle = self.lt()?;
        let prepared = self.libraries().lt()?;
        // held until every kernel of this product is queued; see `Lt`
        let _queue = prepared
            .queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let stream = self.stream();
        let (workspace, _workspace_sync) = prepared.workspace.device_ptr(stream);
        let desc = descriptors.desc();
        if let Some(bias) = pointers.bias {
            // SAFETY: the bias pointer attribute takes a device pointer
            unsafe { desc.set(DescAttribute::CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias) }?;
        }

        let (alpha, beta) = (scale.alpha, scale.beta);
        // SAFETY: the buffers hold the plan's layouts, as the callers check; every
        // buffer and the workspace are device allocations ordered on this stream, and
        // `alpha` and `beta` are host FP32 scalars as the FP32 scale type takes. The
        // workspace and descriptor locks are held until the product is queued
        unsafe {
            lt::cublasLtMatmul(
                handle,
                desc.0,
                (&raw const alpha).cast(),
                pointers.b as *const c_void,
                descriptors.first.0,
                pointers.a as *const c_void,
                descriptors.second.0,
                (&raw const beta).cast(),
                pointers.c as *const c_void,
                descriptors.c.0,
                pointers.c as *mut c_void,
                descriptors.c.0,
                algo,
                workspace as *mut c_void,
                MATMUL_WORKSPACE_BYTES,
                stream.cu_stream().cast(),
            )
        }
        .result()?;
        Ok(())
    }

    /// The prepared cuBLASLt handle, with this GPU's context bound to the calling
    /// thread
    ///
    /// Every cuBLASLt call goes through here first: cudarc's entry points panic when the
    /// library is missing, and a prepared handle means the preflight found every symbol
    fn lt(&self) -> Result<lt::cublasLtHandle_t, Error> {
        self.profile().allow(Library::CublasLt)?;
        let prepared = self.libraries().lt()?;
        // cuBLASLt launches on the stream's context, which nothing binds for it
        self.context().bind_to_thread()?;
        Ok(prepared.handle.0)
    }
}

/// Device addresses of one product's buffers
#[derive(Debug, Clone, Copy)]
struct Pointers {
    a: u64,
    b: u64,
    bias: Option<u64>,
    c: u64,
}

impl Pointers {
    fn check_alignment(&self) -> Result<(), Error> {
        let named = [
            ("a", Some(self.a)),
            ("b", Some(self.b)),
            ("bias", self.bias),
            ("c", Some(self.c)),
        ];
        for (name, pointer) in named {
            let Some(pointer) = pointer else {
                continue;
            };
            if !pointer.is_multiple_of(MATMUL_ALIGNMENT as u64) {
                return Err(Error::Unsupported {
                    context: "matmul",
                    reason: format!(
                        "{name} starts at {pointer:#x}, not on a {MATMUL_ALIGNMENT}-byte boundary"
                    ),
                });
            }
        }
        Ok(())
    }
}

fn no_algorithm(spec: Matmul) -> Error {
    Error::Unsupported {
        context: "matmul plan",
        reason: format!("cuBLASLt offers no algorithm for {spec:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{Epilogue, Matmul, Operands, Output, Stored, Tuning, choose};
    use crate::{Error, Math};

    fn unsupported(result: Result<[Stored; 3], Error>, needle: &str) {
        match result {
            Err(Error::Unsupported { reason, .. }) => {
                assert!(reason.contains(needle), "{reason}");
            }
            other => panic!("expected a refusal mentioning {needle:?}, got {other:?}"),
        }
    }

    #[test]
    fn spec_checks_refuse_what_cublaslt_rejects_or_misreads() {
        let f32 = Operands::F32(Math::Tf32);
        // a sliding-window view: K = 3C rows that start C apart
        let windows = Matmul::new(64, 32, 3 * 16, f32).with_strides(Some(16), None, None);
        unsupported(windows.geometry(), "overlap");
        unsupported(Matmul::new(0, 4, 4, f32).geometry(), "positive");
        // the bias and output take the output's type, which must be the operands'
        unsupported(
            Matmul::new(4, 4, 4, f32)
                .with_output(Output::Bf16)
                .geometry(),
            "same type",
        );
        unsupported(
            Matmul::new(4, 4, 4, Operands::F16)
                .with_output(Output::Bf16)
                .geometry(),
            "same type",
        );

        // a strided `a` view of 3 rows, 5 apart, spans 2 * 5 + 4 elements
        let [a, b, c] = Matmul::new(3, 2, 4, Operands::Bf16)
            .with_output(Output::Bf16)
            .with_epilogue(Epilogue::GeluBias)
            .with_strides(Some(5), None, Some(7))
            .geometry()
            .unwrap();
        assert_eq!((a.rows, a.cols, a.stride, a.span), (3, 4, 5, 14));
        assert_eq!((b.rows, b.cols, b.stride, b.span), (4, 2, 2, 8));
        assert_eq!((c.rows, c.cols, c.stride, c.span), (3, 2, 7, 16));
        // transposed operands are stored as their transposes
        let [a, b, _] = Matmul::linear(3, 2, 4, f32)
            .transposed(true, true)
            .geometry()
            .unwrap();
        assert_eq!((a.rows, a.cols, b.rows, b.cols), (4, 3, 2, 4));
    }

    #[test]
    fn tuning_keeps_the_first_candidate_unless_another_clears_the_margin() {
        // within 5%: the heuristic's first stays, so near-ties do not flip on noise
        assert_eq!(choose(&[Some(1.0), Some(0.96)], 1.05), Some(0));
        assert_eq!(choose(&[Some(1.0), Some(0.9), Some(0.8)], 1.05), Some(2));
        // a first candidate that failed to launch never wins
        assert_eq!(choose(&[None, Some(2.0), Some(1.5)], 1.05), Some(2));
        assert_eq!(choose(&[None, None], 1.05), None);
        assert!(Tuning::default().with_margin(0.9).check().is_err());
        assert!(Tuning::default().with_candidates(0).check().is_err());
        assert!(Tuning::default().check().is_ok());
    }
}
