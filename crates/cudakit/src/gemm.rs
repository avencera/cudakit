//! Row-major FP32 matrix products over cuBLAS

#[cfg(feature = "libraries")]
use cudarc::cublas::sys::{cublasMath_t, cublasOperation_t};
#[cfg(feature = "libraries")]
use cudarc::cublas::{Gemm, GemmConfig, StridedBatchedConfig};
#[cfg(feature = "libraries")]
use cudarc::cudnn::sys::cudnnMathType_t;
use cudarc::driver::{DevicePtr, DevicePtrMut};

#[cfg(feature = "libraries")]
use crate::error::{check_len, to_c_int};
use crate::{Error, Gpu, Library};

/// Arithmetic precision of cuBLAS and cuDNN work on FP32 data
///
/// Exhaustive on purpose: consumers persist the mode in tuning identities and must map
/// every variant, so adding one is a breaking change
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Math {
    /// Full FP32 multiply and accumulate; tensor cores only where they keep FP32
    /// precision; never FP16 operands
    Fp32,
    /// FP32 storage and accumulation with TF32 (10-bit mantissa) tensor-core
    /// multiplies on Ampere and newer
    Tf32,
}

#[cfg(feature = "libraries")]
impl Math {
    /// The cuBLAS math mode
    pub fn cublas(self) -> cublasMath_t {
        match self {
            // the default math mode only uses tensor cores at full FP32 precision
            Self::Fp32 => cublasMath_t::CUBLAS_DEFAULT_MATH,
            Self::Tf32 => cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
        }
    }

    /// The cuDNN math type
    pub fn cudnn(self) -> cudnnMathType_t {
        match self {
            // cuDNN's default math type would allow TF32 for FP32 data on Ampere and newer
            Self::Fp32 => cudnnMathType_t::CUDNN_FMA_MATH,
            // tensor-op math runs FP32 data as TF32; `_ALLOW_CONVERSION` is avoided
            // because it may also down-convert FP32 tensors to FP16
            Self::Tf32 => cudnnMathType_t::CUDNN_TENSOR_OP_MATH,
        }
    }
}

/// A row-major GEMM on FP32 buffers: `c = alpha * op(a) · op(b) + beta * c`
///
/// `op(a)` is `m × k` and `op(b)` is `k × n`; `c` is `m × n`. A transposed operand
/// is stored as its transpose in row-major order, so a PyTorch `Linear` weight
/// (`out × in`) is `b` with `b_transposed = true`. cuBLAS is column-major, and
/// [`Gpu::sgemm`] does the operand swap
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sgemm {
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
    /// Scale for the product
    pub alpha: f32,
    /// Scale for the existing `c`; 0 overwrites it
    pub beta: f32,
    /// Precision of the multiply; FP32 unless the caller opts into TF32
    pub math: Math,
}

impl Sgemm {
    /// `c = a · b` with untransposed operands
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            a_transposed: false,
            b_transposed: false,
            alpha: 1.0,
            beta: 0.0,
            math: Math::Fp32,
        }
    }

    /// The column-major cuBLAS call for this row-major product
    ///
    /// Row-major `c` (`m × n`) is column-major `cᵀ` (`n × m`), so cuBLAS computes
    /// `cᵀ = op(b)ᵀ · op(a)ᵀ` with `b` as its first operand. A row-major matrix read as
    /// column-major is already transposed, so an untransposed row-major operand maps
    /// to `N` and a transposed one to `T`
    #[cfg(feature = "libraries")]
    fn column_major(&self) -> Result<GemmConfig<f32>, Error> {
        let context = "sgemm";
        let op = |transposed: bool| {
            if transposed {
                cublasOperation_t::CUBLAS_OP_T
            } else {
                cublasOperation_t::CUBLAS_OP_N
            }
        };

        Ok(GemmConfig {
            transa: op(self.b_transposed),
            transb: op(self.a_transposed),
            m: to_c_int(context, self.n)?,
            n: to_c_int(context, self.m)?,
            k: to_c_int(context, self.k)?,
            alpha: self.alpha,
            // leading dimension = row length of the stored row-major matrix
            lda: to_c_int(context, if self.b_transposed { self.k } else { self.n })?,
            ldb: to_c_int(context, if self.a_transposed { self.m } else { self.k })?,
            beta: self.beta,
            ldc: to_c_int(context, self.n)?,
        })
    }
}

/// `batch` independent [`Sgemm`] products whose operands sit at fixed element strides
/// in three buffers, such as one product per attention head
///
/// Product `i` reads `a[i * stride_a..]` and `b[i * stride_b..]` and writes
/// `c[i * stride_c..]`. A stride of 0 shares one operand across the batch. Strides
/// must not make two products write overlapping parts of `c`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SgemmBatched {
    /// The product each batch entry runs
    pub gemm: Sgemm,
    /// Number of products
    pub batch: usize,
    /// Elements between consecutive `a` operands
    pub stride_a: usize,
    /// Elements between consecutive `b` operands
    pub stride_b: usize,
    /// Elements between consecutive `c` outputs
    pub stride_c: usize,
}

impl SgemmBatched {
    /// Densely packed operands: each stride is one operand's element count
    pub fn packed(gemm: Sgemm, batch: usize) -> Self {
        Self {
            gemm,
            batch,
            stride_a: gemm.m * gemm.k,
            stride_b: gemm.k * gemm.n,
            stride_c: gemm.m * gemm.n,
        }
    }

    /// Elements a buffer must hold for `batch` operands of `size` at `stride`
    #[cfg(feature = "libraries")]
    fn span(&self, stride: usize, size: usize) -> usize {
        match self.batch {
            0 => 0,
            batch => stride.saturating_mul(batch - 1).saturating_add(size),
        }
    }
}

impl Gpu {
    /// Runs a row-major GEMM on this GPU's stream in `spec.math` precision
    ///
    /// Each buffer must hold exactly its matrix; take a view of a larger persistent
    /// buffer to multiply part of it. cuBLAS must be prepared with
    /// [`Gpu::prepare`] first
    pub fn sgemm<A, B, C>(&self, spec: Sgemm, a: &A, b: &B, c: &mut C) -> Result<(), Error>
    where
        A: DevicePtr<f32>,
        B: DevicePtr<f32>,
        C: DevicePtrMut<f32>,
    {
        self.profile().allow(Library::Cublas)?;
        #[cfg(feature = "libraries")]
        {
            check_len("sgemm a", spec.m.saturating_mul(spec.k), a.len())?;
            check_len("sgemm b", spec.k.saturating_mul(spec.n), b.len())?;
            check_len("sgemm c", spec.m.saturating_mul(spec.n), c.len())?;
            let config = spec.column_major()?;
            let _math = self.lock_blas(spec.math)?;

            // SAFETY: the lengths match the dimensions and leading dimensions checked
            // above, and all buffers are device allocations that cudarc orders on the
            // handle's stream
            unsafe { self.blas()?.gemm(config, b, a, c) }?;
            Ok(())
        }
        #[cfg(not(feature = "libraries"))]
        {
            let _ = (spec, a, b, c);
            Err(Error::LibraryForbidden {
                library: Library::Cublas,
            })
        }
    }

    /// Runs a strided batch of row-major GEMMs as one cuBLAS call
    ///
    /// Each buffer must hold exactly the span its strides cover: `stride * (batch - 1)`
    /// plus one operand
    pub fn sgemm_batched<A, B, C>(
        &self,
        spec: SgemmBatched,
        a: &A,
        b: &B,
        c: &mut C,
    ) -> Result<(), Error>
    where
        A: DevicePtr<f32>,
        B: DevicePtr<f32>,
        C: DevicePtrMut<f32>,
    {
        self.profile().allow(Library::Cublas)?;
        #[cfg(feature = "libraries")]
        {
            let gemm = spec.gemm;
            let (a_size, b_size, c_size) = (gemm.m * gemm.k, gemm.k * gemm.n, gemm.m * gemm.n);
            check_len("sgemm batched a", spec.span(spec.stride_a, a_size), a.len())?;
            check_len("sgemm batched b", spec.span(spec.stride_b, b_size), b.len())?;
            check_len("sgemm batched c", spec.span(spec.stride_c, c_size), c.len())?;
            if spec.batch > 1 && spec.stride_c < c_size {
                return Err(Error::Unsupported {
                    context: "sgemm batched",
                    reason: format!(
                        "output stride {} overlaps a {c_size}-element product",
                        spec.stride_c
                    ),
                });
            }
            let stride = |value: usize| {
                i64::try_from(value).map_err(|_| Error::DimensionOverflow {
                    context: "sgemm batched stride",
                    value,
                })
            };
            let config = StridedBatchedConfig {
                gemm: gemm.column_major()?,
                batch_size: to_c_int("sgemm batched", spec.batch)?,
                // operands swap for the column-major call, like the plain product
                stride_a: stride(spec.stride_b)?,
                stride_b: stride(spec.stride_a)?,
                stride_c: stride(spec.stride_c)?,
            };
            let _math = self.lock_blas(gemm.math)?;

            // SAFETY: each buffer holds the full strided span checked above, outputs do
            // not overlap, and cudarc orders the buffers on the handle's stream
            unsafe { self.blas()?.gemm_strided_batched(config, b, a, c) }?;
            Ok(())
        }
        #[cfg(not(feature = "libraries"))]
        {
            let _ = (spec, a, b, c);
            Err(Error::LibraryForbidden {
                library: Library::Cublas,
            })
        }
    }
}
