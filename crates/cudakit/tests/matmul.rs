//! GPU checks of cuBLASLt matmul plans against an f64 host reference: every operand and
//! output type, every epilogue, strided views, tuning, plan caching, graph capture and
//! buffers from a consumer's own memory pool
//!
//! The tests touch the device only with `CUDAKIT_REQUIRE_GPU=1`, which also makes a
//! missing GPU a failure so a GPU runner cannot silently pass. Outputs start as NaN and
//! every element is compared, so a product that skips elements fails

use std::marker::PhantomData;
use std::sync::Arc;

use cudakit::cudarc::driver::sys::{self as cu, CUdeviceptr, CUmemoryPool};
use cudakit::cudarc::driver::{
    CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceSlice, SyncOnDrop,
};
use cudakit::{
    Epilogue, Error, ExtraSymbols, Gpu, Library, LibraryPolicy, Math, Matmul, MatmulAlgo,
    MatmulCache, Operands, Output, Profile, PtxTier, Scale, TierSet, Tuning,
};

static PROFILE: Profile = Profile {
    consumer: "cudakit-matmul-tests",
    compiled: TierSet::of(&[PtxTier::Sm75]),
    shipped: TierSet::of(&[PtxTier::Sm75]),
    tier_features: &[(PtxTier::Sm75, "test-sm75")],
    libraries: LibraryPolicy::Allowed,
    extra_symbols: ExtraSymbols::NONE,
};

const EPILOGUES: [Epilogue; 6] = [
    Epilogue::None,
    Epilogue::Bias,
    Epilogue::Relu,
    Epilogue::ReluBias,
    Epilogue::Gelu,
    Epilogue::GeluBias,
];

/// The GPU with cuBLASLt prepared, or `None` without `CUDAKIT_REQUIRE_GPU=1`
fn gpu() -> Option<Gpu> {
    if std::env::var_os("CUDAKIT_REQUIRE_GPU").is_none_or(|value| value != "1") {
        eprintln!("skipping: set CUDAKIT_REQUIRE_GPU=1 to run GPU tests");
        return None;
    }

    let gpu = Gpu::open(0, &PROFILE, None).expect("opening the GPU failed");
    gpu.prepare(Library::CublasLt)
        .expect("preparing cuBLASLt failed");
    Some(gpu)
}

/// Multiples of 1/8 in [-1, 1], exact in every operand type, so products and sums are
/// exact in FP32 and only the output rounding and the epilogue differ from f64
fn values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|index| ((index * 7 + seed * 13) % 17) as f32 / 8.0 - 1.0)
        .collect()
}

/// A stored `rows × cols` matrix and its row stride
#[derive(Debug, Clone, Copy)]
struct Shape {
    rows: usize,
    cols: usize,
    stride: usize,
}

impl Shape {
    fn span(self) -> usize {
        (self.rows - 1) * self.stride + self.cols
    }

    /// `dense` laid out with this stride, the gaps filled with `gap`
    fn pack(self, dense: &[f32], gap: f32) -> Vec<f32> {
        assert_eq!(dense.len(), self.rows * self.cols);
        let mut packed = vec![gap; self.span()];
        for (row, values) in dense.chunks(self.cols).enumerate() {
            packed[row * self.stride..][..self.cols].copy_from_slice(values);
        }
        packed
    }

    fn is_gap(self, index: usize) -> bool {
        index % self.stride >= self.cols
    }
}

/// The stored shapes of `spec`'s `a`, `b` and `c`
fn shapes(spec: &Matmul) -> [Shape; 3] {
    let (m, n, k) = (spec.m, spec.n, spec.k);
    let shape = |(rows, cols): (usize, usize), stride: Option<usize>| Shape {
        rows,
        cols,
        stride: stride.unwrap_or(cols),
    };
    [
        shape(
            if spec.a_transposed { (k, m) } else { (m, k) },
            spec.a_stride,
        ),
        shape(
            if spec.b_transposed { (n, k) } else { (k, n) },
            spec.b_stride,
        ),
        shape((m, n), spec.c_stride),
    ]
}

/// `epilogue(op(a) · op(b))` in f64 for dense stored `a` and `b`
fn reference(spec: &Matmul, a: &[f32], b: &[f32], bias: &[f32]) -> Vec<f64> {
    let (m, n, k) = (spec.m, spec.n, spec.k);
    let at = |row: usize, col: usize| {
        f64::from(if spec.a_transposed {
            a[col * m + row]
        } else {
            a[row * k + col]
        })
    };
    let bt = |row: usize, col: usize| {
        f64::from(if spec.b_transposed {
            b[col * k + row]
        } else {
            b[row * n + col]
        })
    };
    let gelu = |x: f64| {
        let inner = (2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    };

    (0..m * n)
        .map(|index| {
            let (row, col) = (index / n, index % n);
            let mut value: f64 = (0..k).map(|inner| at(row, inner) * bt(inner, col)).sum();
            if spec.epilogue.has_bias() {
                value += f64::from(bias[col]);
            }
            match spec.epilogue {
                Epilogue::Relu | Epilogue::ReluBias => value.max(0.0),
                Epilogue::Gelu | Epilogue::GeluBias => gelu(value),
                _ => value,
            }
        })
        .collect()
}

/// FP16 bits of `value`, which must be zero or a normal FP16 with at most 10
/// mantissa bits, as the test values are
fn f16_bits(value: f32) -> u16 {
    if value == 0.0 {
        return 0;
    }
    if value.is_nan() {
        return 0x7e00;
    }
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    assert!(
        (1..31).contains(&exponent) && bits & 0x1fff == 0,
        "{value} is not an exact FP16"
    );
    sign | ((exponent as u16) << 10) | ((bits >> 13) & 0x3ff) as u16
}

fn f16_value(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        31 if mantissa == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

/// BF16 bits of `value`, which must be exact in BF16, as the test values are
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    assert!(
        value.is_nan() || bits & 0xffff == 0,
        "{value} is not an exact BF16"
    );
    (bits >> 16) as u16
}

fn bf16_value(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Bits of `value` in a 16-bit type, for FP16 or BF16 operands and outputs
fn half_bits(bf16: bool, values: &[f32]) -> Vec<u16> {
    let encode = if bf16 { bf16_bits } else { f16_bits };
    values.iter().copied().map(encode).collect()
}

/// A device buffer of `f32` or 16-bit values
enum Buffer {
    Word(CudaSlice<f32>),
    Half(CudaSlice<u16>),
}

impl Buffer {
    fn upload(stream: &Arc<CudaStream>, half: Option<bool>, values: &[f32]) -> Result<Self, Error> {
        Ok(match half {
            None => Self::Word(stream.clone_htod(values)?),
            Some(bf16) => Self::Half(stream.clone_htod(&half_bits(bf16, values))?),
        })
    }

    fn download(&self, stream: &Arc<CudaStream>, bf16: bool) -> Result<Vec<f32>, Error> {
        Ok(match self {
            Self::Word(buffer) => stream.clone_dtoh(buffer)?,
            Self::Half(buffer) => {
                let decode = if bf16 { bf16_value } else { f16_value };
                stream.clone_dtoh(buffer)?.into_iter().map(decode).collect()
            }
        })
    }
}

/// `Some(true)` for BF16, `Some(false)` for FP16, `None` for `f32` operand buffers
fn operand_half(operands: Operands) -> Option<bool> {
    match operands {
        Operands::F16 => Some(false),
        Operands::Bf16 => Some(true),
        _ => None,
    }
}

fn output_half(output: Output) -> Option<bool> {
    match output {
        Output::F16 => Some(false),
        Output::Bf16 => Some(true),
        _ => None,
    }
}

/// Runs `plan` on `a`, `b` and `bias`, given dense, and returns the whole stored `c`
/// after a NaN prefill; the gaps of strided `a` and `b` hold NaN too
fn run(
    gpu: &Gpu,
    plan: &cudakit::MatmulPlan,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>, Error> {
    let spec = plan.spec();
    let stream = gpu.stream();
    let [a_shape, b_shape, c_shape] = shapes(&spec);
    let operands = operand_half(spec.operands);
    let output = output_half(spec.output);
    let a = Buffer::upload(stream, operands, &a_shape.pack(a, f32::NAN))?;
    let b = Buffer::upload(stream, operands, &b_shape.pack(b, f32::NAN))?;
    let bias = Buffer::upload(stream, output, bias)?;
    let mut c = Buffer::upload(stream, output, &vec![f32::NAN; c_shape.span()])?;
    let scale = Scale::OVERWRITE;
    let with_bias = spec.epilogue.has_bias();
    match (&a, &b, &bias, &mut c) {
        (Buffer::Word(a), Buffer::Word(b), Buffer::Word(bias), Buffer::Word(c)) => {
            if with_bias {
                gpu.matmul_bias(plan, scale, a, b, bias, c)
            } else {
                gpu.matmul(plan, scale, a, b, c)
            }
        }
        (Buffer::Half(a), Buffer::Half(b), Buffer::Word(bias), Buffer::Word(c)) => {
            if with_bias {
                gpu.matmul_bias(plan, scale, a, b, bias, c)
            } else {
                gpu.matmul(plan, scale, a, b, c)
            }
        }
        (Buffer::Half(a), Buffer::Half(b), Buffer::Half(bias), Buffer::Half(c)) => {
            if with_bias {
                gpu.matmul_bias(plan, scale, a, b, bias, c)
            } else {
                gpu.matmul(plan, scale, a, b, c)
            }
        }
        _ => unreachable!("no product reads f32 operands into a 16-bit output"),
    }?;
    c.download(stream, output == Some(true))
}

/// Compares a whole stored `c` with the dense f64 reference: gaps stay NaN, every
/// element is finite and within one rounding of the output type
fn check(spec: &Matmul, context: &str, stored: &[f32], expected: &[f64]) {
    let c_shape = shapes(spec)[2];
    assert_eq!(stored.len(), c_shape.span(), "{context}: stored c length");
    let ulp = match spec.output {
        Output::F16 => 2f64.powi(-10),
        Output::Bf16 => 2f64.powi(-7),
        _ => 1e-5,
    };
    let mut dense = Vec::with_capacity(expected.len());
    for (index, value) in stored.iter().enumerate() {
        if c_shape.is_gap(index) {
            assert!(value.is_nan(), "{context}: gap element {index} was written");
        } else {
            dense.push(f64::from(*value));
        }
    }

    assert_eq!(dense.len(), expected.len(), "{context}: element count");
    for (index, (actual, expected)) in dense.iter().zip(expected).enumerate() {
        let tolerance = ulp * expected.abs() + 1e-4;
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance,
            "{context}, element {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn every_candidate_matches_the_reference_for_each_operand_type() -> Result<(), Error> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };
    assert!(gpu.cublaslt_version()? > 0);

    // large enough that the heuristic offers split-K and several tiles
    let (m, n, k) = (48, 96, 256);
    for (a_transposed, b_transposed) in [(false, false), (true, false), (false, true), (true, true)]
    {
        let (a, b) = (values(m * k, 5), values(k * n, 6));
        for operands in [
            Operands::F32(Math::Fp32),
            Operands::F32(Math::Tf32),
            Operands::F32Fast16Bf,
            Operands::F16,
            Operands::Bf16,
        ] {
            let spec = Matmul::new(m, n, k, operands).transposed(a_transposed, b_transposed);
            let expected = reference(&spec, &a, &b, &[]);
            let candidates = gpu.matmul_candidates(spec, 16)?;
            assert!(!candidates.is_empty(), "no algorithm for {spec:?}");
            assert_eq!(gpu.matmul_plan(spec)?.algo(), candidates[0].algo);
            for candidate in candidates {
                let config = gpu.matmul_algo_config(candidate.algo)?;
                let plan = gpu.matmul_plan_with(spec, candidate.algo)?;
                assert!(plan.workspace_bytes() <= cudakit::MATMUL_WORKSPACE_BYTES);
                let context = format!("{spec:?} with {config}");
                check(&spec, &context, &run(&gpu, &plan, &a, &b, &[])?, &expected);
            }
        }
    }
    Ok(())
}

#[test]
fn split_k_sums_stay_in_fp32_for_a_16_bit_output() -> Result<(), Error> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };

    // each half of the dot product sums to ±131072, past FP16's 65504, while the whole
    // sums to zero, so a split-K reduction in FP16 would store ±inf and then NaN
    let (m, n, k) = (16, 64, 8192);
    let a = vec![1.0_f32; m * k];
    let b: Vec<f32> = (0..k * n)
        .map(|index| if index / n < k / 2 { 32.0 } else { -32.0 })
        .collect();
    let spec = Matmul::new(m, n, k, Operands::F16).with_output(Output::F16);
    let expected = reference(&spec, &a, &b, &[]);
    assert!(expected.iter().all(|value| *value == 0.0));

    let candidates = gpu.matmul_candidates(spec, 32)?;
    assert!(!candidates.is_empty(), "no algorithm for {spec:?}");
    let mut split = 0;
    for candidate in candidates {
        let config = gpu.matmul_algo_config(candidate.algo)?;
        // none, or the FP32 compute type
        assert!(
            matches!(config.reduction, 0 | 2),
            "{spec:?} offered {config}"
        );
        if config.split_k > 1 {
            split += 1;
        }

        let plan = gpu.matmul_plan_with(spec, candidate.algo)?;
        let context = format!("{spec:?} with {config}");
        check(&spec, &context, &run(&gpu, &plan, &a, &b, &[])?, &expected);
    }
    // the heuristic offered 3 on an RTX 5090 with cuBLASLt 13.4; with none, the sums
    // above never run split
    assert!(split > 0, "no split-K candidate for {spec:?}");
    Ok(())
}

/// A split-K plan whose reduction reads the shared workspace, for `spec`
fn split_plan(gpu: &Gpu, spec: Matmul) -> Result<cudakit::MatmulPlan, Error> {
    for candidate in gpu.matmul_candidates(spec, 32)? {
        let config = gpu.matmul_algo_config(candidate.algo)?;
        if config.split_k > 1 && candidate.workspace_bytes > 0 {
            return gpu.matmul_plan_with(spec, candidate.algo);
        }
    }

    panic!("no split-K candidate that uses the workspace for {spec:?}");
}

#[test]
fn split_k_plans_on_two_threads_keep_their_own_partial_sums() -> Result<(), Error> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };

    // two products of different widths whose split-K partial sums both go through the
    // one shared workspace, queued from two threads at once many times over
    const ROUNDS: usize = 64;
    let specs = [(16, 64, 8192), (16, 32, 8192)]
        .map(|(m, n, k)| Matmul::new(m, n, k, Operands::F16).with_output(Output::F16));
    let stream = gpu.stream();
    let mut jobs = Vec::new();
    for (seed, spec) in specs.into_iter().enumerate() {
        let a = values(spec.m * spec.k, seed + 20);
        let b = values(spec.k * spec.n, seed + 30);
        let expected = reference(&spec, &a, &b, &[]);
        let plan = split_plan(&gpu, spec)?;
        let a = stream.clone_htod(&half_bits(false, &a))?;
        let b = stream.clone_htod(&half_bits(false, &b))?;
        let outputs = (0..ROUNDS)
            .map(|_| stream.alloc_zeros::<u16>(spec.m * spec.n))
            .collect::<Result<Vec<_>, _>>()?;
        jobs.push((spec, plan, a, b, outputs, expected));
    }

    let shared = &gpu;
    std::thread::scope(|scope| {
        let threads: Vec<_> = jobs
            .iter_mut()
            .map(|(_, plan, a, b, outputs, _)| {
                scope.spawn(move || {
                    for c in outputs.iter_mut() {
                        shared.matmul(plan, Scale::OVERWRITE, a, b, c)?;
                    }
                    Ok::<_, Error>(())
                })
            })
            .collect();
        threads
            .into_iter()
            .try_for_each(|thread| thread.join().expect("matmul thread panicked"))
    })?;

    for (spec, _, _, _, outputs, expected) in &jobs {
        for (round, c) in outputs.iter().enumerate() {
            let stored: Vec<f32> = stream.clone_dtoh(c)?.into_iter().map(f16_value).collect();
            check(spec, &format!("{spec:?}, round {round}"), &stored, expected);
        }
    }
    Ok(())
}

#[test]
fn a_graph_replay_on_another_thread_keeps_eager_partial_sums() -> Result<(), Error> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };

    // one thread queues split-K products eagerly while another replays a graph of a
    // different split-K product; both go through the one shared workspace
    const ROUNDS: usize = 64;
    let stream = gpu.stream();
    let [eager, replayed] = [(16, 64, 8192), (16, 32, 8192)]
        .map(|(m, n, k)| Matmul::new(m, n, k, Operands::F16).with_output(Output::F16));
    let upload = |spec: Matmul, seed: usize| -> Result<_, Error> {
        let a = values(spec.m * spec.k, seed);
        let b = values(spec.k * spec.n, seed + 1);
        let expected = reference(&spec, &a, &b, &[]);
        let a = stream.clone_htod(&half_bits(false, &a))?;
        let b = stream.clone_htod(&half_bits(false, &b))?;
        Ok((split_plan(&gpu, spec)?, a, b, expected))
    };
    let (eager_plan, eager_a, eager_b, eager_expected) = upload(eager, 40)?;
    let (graph_plan, graph_a, graph_b, graph_expected) = upload(replayed, 50)?;
    let mut eager_outputs = (0..ROUNDS)
        .map(|_| stream.alloc_zeros::<u16>(eager.m * eager.n))
        .collect::<Result<Vec<_>, _>>()?;
    let mut graph_c = stream.alloc_zeros::<u16>(replayed.m * replayed.n)?;

    // SAFETY: the inputs and `graph_c` stay allocated until after the last replay below
    // has been read back
    let graph = unsafe {
        gpu.capture_nonempty(|| {
            gpu.matmul(
                &graph_plan,
                Scale::OVERWRITE,
                &graph_a,
                &graph_b,
                &mut graph_c,
            )
        })
    }?;

    let shared = &gpu;
    let outputs = &mut eager_outputs;
    let (plan, a, b) = (&eager_plan, &eager_a, &eager_b);
    std::thread::scope(|scope| {
        let eager_thread = scope.spawn(move || {
            for c in outputs.iter_mut() {
                shared.matmul(plan, Scale::OVERWRITE, a, b, c)?;
            }
            Ok::<_, Error>(())
        });
        let replay_thread = scope.spawn(move || {
            for _ in 0..ROUNDS {
                graph.launch()?;
            }
            Ok::<_, Error>(())
        });
        eager_thread.join().expect("eager thread panicked")?;
        replay_thread.join().expect("replay thread panicked")
    })?;

    for (round, c) in eager_outputs.iter().enumerate() {
        let stored: Vec<f32> = stream.clone_dtoh(c)?.into_iter().map(f16_value).collect();
        check(
            &eager,
            &format!("eager round {round}"),
            &stored,
            &eager_expected,
        );
    }
    let stored: Vec<f32> = stream
        .clone_dtoh(&graph_c)?
        .into_iter()
        .map(f16_value)
        .collect();
    check(&replayed, "graph replay", &stored, &graph_expected);
    Ok(())
}

#[test]
fn epilogues_outputs_and_strided_views_match_the_reference() -> Result<(), Error> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };

    // an odd row count, like a 201-token sequence, through a `Linear`
    let (rows, outputs, inputs) = (33, 64, 96);
    let a = values(rows * inputs, 1);
    let w = values(outputs * inputs, 2);
    let bias = values(outputs, 3);
    for (operands, output) in [
        (Operands::F32(Math::Tf32), Output::F32),
        (Operands::F32(Math::Fp32), Output::F32),
        (Operands::Bf16, Output::F32),
        (Operands::Bf16, Output::Bf16),
        (Operands::F16, Output::F16),
    ] {
        // dense, then views into wider rows, each row starting 16-byte aligned, then a
        // transposed `a` (stored `inputs × rows`) as a view too
        for (a_transposed, strides) in [
            (false, (None, None, None)),
            (
                false,
                (Some(inputs + 8), Some(inputs + 16), Some(outputs + 8)),
            ),
            (true, (Some(rows + 7), Some(inputs + 16), Some(outputs + 8))),
        ] {
            for epilogue in EPILOGUES {
                let spec = Matmul::linear(rows, outputs, inputs, operands)
                    .transposed(a_transposed, true)
                    .with_output(output)
                    .with_epilogue(epilogue)
                    .with_strides(strides.0, strides.1, strides.2);
                let plan = gpu.matmul_plan(spec)?;
                let config = gpu.matmul_algo_config(plan.algo())?;
                let expected = reference(&spec, &a, &w, &bias);
                let context = format!("{spec:?} with {config}");
                check(
                    &spec,
                    &context,
                    &run(&gpu, &plan, &a, &w, &bias)?,
                    &expected,
                );
            }
        }
    }
    Ok(())
}

#[test]
fn a_plan_accumulates_replays_in_a_graph_and_checks_its_inputs() -> Result<(), Error> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };
    let stream = gpu.stream();
    let spec = Matmul::linear(16, 64, 32, Operands::F32(Math::Fp32));
    let (a, w) = (values(16 * 32, 7), values(64 * 32, 8));
    let product = reference(&spec, &a, &w, &[]);
    let (a_device, w_device) = (stream.clone_htod(&a)?, stream.clone_htod(&w)?);
    let plan = gpu.matmul_plan(spec)?;

    // `alpha = 0.5, beta = 1` adds half the product onto `c`, twice through a graph
    let mut c = stream.clone_htod(&vec![1.0_f32; 16 * 64])?;
    let half = Scale::new(0.5, 1.0);
    // SAFETY: the buffers outlive the graph's launches, which finish before the reads
    let graph =
        unsafe { gpu.capture_nonempty(|| gpu.matmul(&plan, half, &a_device, &w_device, &mut c)) }?;
    graph.launch()?;
    graph.launch()?;
    let expected: Vec<f64> = product.iter().map(|value| 1.0 + value).collect();
    check(&spec, "accumulated", &stream.clone_dtoh(&c)?, &expected);

    let overwrite = Scale::OVERWRITE;
    let short = stream.alloc_zeros::<f32>(3)?;
    assert!(matches!(
        gpu.matmul(&plan, overwrite, &short, &w_device, &mut c),
        Err(Error::BufferLength { .. })
    ));
    // FP16 buffers for an FP32 plan
    let halves = stream.alloc_zeros::<u16>(16 * 32)?;
    let weights = stream.alloc_zeros::<u16>(64 * 32)?;
    assert!(matches!(
        gpu.matmul(&plan, overwrite, &halves, &weights, &mut c),
        Err(Error::Unsupported { .. })
    ));
    // cuBLASLt was asked for 16-byte-aligned algorithms, so an offset view is refused
    let offset = stream.alloc_zeros::<f32>(16 * 32 + 1)?;
    assert!(matches!(
        gpu.matmul(&plan, overwrite, &offset.slice(1..), &w_device, &mut c),
        Err(Error::Unsupported { .. })
    ));
    // a bias needs a bias epilogue, and a bias epilogue needs `n` aligned values
    let bias = stream.alloc_zeros::<f32>(64)?;
    assert!(matches!(
        gpu.matmul_bias(&plan, overwrite, &a_device, &w_device, &bias, &mut c),
        Err(Error::Unsupported { .. })
    ));
    let biased = gpu.matmul_plan(spec.with_epilogue(Epilogue::Bias))?;
    assert!(matches!(
        gpu.matmul(&biased, overwrite, &a_device, &w_device, &mut c),
        Err(Error::Unsupported { .. })
    ));
    // the bias pointer is set per call, so one plan serves two biases
    for seed in [10, 11] {
        let bias = values(64, seed);
        let expected = reference(&biased.spec(), &a, &w, &bias);
        check(
            &biased.spec(),
            "bias per call",
            &run(&gpu, &biased, &a, &w, &bias)?,
            &expected,
        );
    }
    assert!(matches!(
        gpu.matmul_bias(&biased, overwrite, &a_device, &w_device, &short, &mut c),
        Err(Error::BufferLength { .. })
    ));
    let wide = stream.alloc_zeros::<f32>(65)?;
    assert!(matches!(
        gpu.matmul_bias(
            &biased,
            overwrite,
            &a_device,
            &w_device,
            &wide.slice(1..),
            &mut c
        ),
        Err(Error::Unsupported { .. })
    ));
    // a stale or corrupt tune entry is refused, not run
    assert!(matches!(
        gpu.matmul_plan_with(spec, MatmulAlgo([u64::MAX; 8])),
        Err(Error::Unsupported { .. })
    ));
    Ok(())
}

#[test]
fn tuning_runs_before_capture_and_cached_plans_run_inside_it() -> Result<(), Error> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };
    let stream = gpu.stream();

    // 201 rows in TF32 is a shape where the heuristic's first was 1.25x off on an RTX 5090
    let spec =
        Matmul::linear(201, 256, 256, Operands::F32(Math::Tf32)).with_epilogue(Epilogue::Bias);
    let tuning = Tuning::default();
    let (plan, report) = gpu.tune_matmul(spec, &tuning)?;
    let candidates = gpu.matmul_candidates(spec, tuning.candidates)?;
    if candidates.len() > 1 {
        assert_eq!(report.timings.len(), candidates.len());
        let chosen = report.timings[report.chosen];
        assert_eq!(chosen.candidate.algo, plan.algo());
        let first = report.timings[0].ms;
        let chosen_ms = chosen.ms.expect("the chosen candidate ran");
        assert!(chosen_ms > 0.0 && chosen_ms.is_finite());
        // the first stays unless another beats it by the margin
        if report.chosen != 0 {
            assert!(first.is_none_or(|first| first > chosen_ms * tuning.margin));
        }
        eprintln!(
            "tuned {spec:?}: chose {} of {}, {:?}",
            report.chosen,
            candidates.len(),
            report
                .timings
                .iter()
                .map(|timing| timing.ms)
                .collect::<Vec<_>>()
        );
    } else {
        assert!(report.timings.is_empty());
    }
    let (a, w, bias) = (values(201 * 256, 4), values(256 * 256, 5), values(256, 6));
    let expected = reference(&spec, &a, &w, &bias);
    check(&spec, "tuned", &run(&gpu, &plan, &a, &w, &bias)?, &expected);

    let tuned = MatmulCache::new(Some(tuning));
    let untuned = MatmulCache::new(None);
    tuned.plan(&gpu, spec)?;
    assert_eq!(tuned.len(), 1);
    let (a_device, w_device) = (stream.clone_htod(&a)?, stream.clone_htod(&w)?);
    let bias_device = stream.clone_htod(&bias)?;
    let mut c = stream.alloc_zeros::<f32>(201 * 256)?;
    let other = Matmul::linear(32, 256, 256, Operands::F32(Math::Tf32));
    let mut checked = false;
    // SAFETY: the buffers outlive the graph's launch, which finishes before the read
    let graph = unsafe {
        gpu.capture_nonempty(|| {
            // timing syncs the stream, which would break the capture, so it is refused
            assert!(matches!(
                gpu.tune_matmul(other, &tuning),
                Err(Error::Capturing { .. })
            ));
            assert!(matches!(
                tuned.plan(&gpu, other),
                Err(Error::Capturing { .. })
            ));
            // the first preparation allocates; cuBLAS is never prepared in this file
            assert!(matches!(
                gpu.prepare(Library::Cublas),
                Err(Error::Capturing { .. })
            ));
            gpu.prepare(Library::CublasLt)?;
            // a heuristic plan is host-only, and a cached tuned plan needs no timing
            untuned.plan(&gpu, other)?;
            let cached = tuned.plan(&gpu, spec)?;
            checked = true;
            gpu.matmul_bias(
                &cached,
                Scale::OVERWRITE,
                &a_device,
                &w_device,
                &bias_device,
                &mut c,
            )
        })
    }?;
    assert!(checked);
    graph.launch()?;
    check(
        &spec,
        "cached in a graph",
        &stream.clone_dtoh(&c)?,
        &expected,
    );
    assert_eq!((tuned.len(), untuned.len()), (1, 1));

    // the capture left the stream usable for tuning again
    gpu.tune_matmul(other, &tuning.with_candidates(2))?;
    Ok(())
}

/// A buffer from a private `cuMemPool`, standing in for a consumer's own allocator
struct PoolBuffer<T> {
    pointer: CUdeviceptr,
    len: usize,
    stream: Arc<CudaStream>,
    element: PhantomData<T>,
}

impl<T> PoolBuffer<T> {
    fn new(pool: CUmemoryPool, stream: &Arc<CudaStream>, len: usize) -> Result<Self, Error> {
        let mut pointer = 0;
        // SAFETY: `pool` is a live device pool and `pointer` a writable slot
        unsafe {
            cu::cuMemAllocFromPoolAsync(
                &mut pointer,
                len * size_of::<T>(),
                pool,
                stream.cu_stream(),
            )
        }
        .result()?;
        Ok(Self {
            pointer,
            len,
            stream: stream.clone(),
            element: PhantomData,
        })
    }
}

impl<T> Drop for PoolBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: allocated from the pool on this stream, freed once in stream order
        let _ = unsafe { cu::cuMemFreeAsync(self.pointer, self.stream.cu_stream()) };
    }
}

impl<T> DeviceSlice<T> for PoolBuffer<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

// one stream orders every use, so the buffers need no events
impl<T> DevicePtr<T> for PoolBuffer<T> {
    fn device_ptr<'a>(&'a self, _stream: &'a CudaStream) -> (CUdeviceptr, SyncOnDrop<'a>) {
        (self.pointer, SyncOnDrop::Sync(None))
    }
}

impl<T> DevicePtrMut<T> for PoolBuffer<T> {
    fn device_ptr_mut<'a>(&'a mut self, _stream: &'a CudaStream) -> (CUdeviceptr, SyncOnDrop<'a>) {
        (self.pointer, SyncOnDrop::Sync(None))
    }
}

#[test]
fn private_pool_buffers_run_in_a_graph_from_another_thread() -> Result<(), Error> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };
    let stream = gpu.stream().clone();
    gpu.context().bind_to_thread()?;
    let mut pool = std::ptr::null_mut();
    // SAFETY: zeroed props are valid; the pool is pinned device memory on this device
    unsafe {
        let mut props: cu::CUmemPoolProps = std::mem::zeroed();
        props.allocType = cu::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        props.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.id = gpu.context().cu_device();
        cu::cuMemPoolCreate(&mut pool, &props).result()?;
    }

    let (rows, outputs, inputs) = (64, 128, 96);
    let spec = Matmul::linear(rows, outputs, inputs, Operands::Bf16)
        .with_output(Output::Bf16)
        .with_epilogue(Epilogue::GeluBias);
    let (a, w, bias) = (
        values(rows * inputs, 7),
        values(outputs * inputs, 8),
        values(outputs, 9),
    );
    let expected = reference(&spec, &a, &w, &bias);
    let mut a_pool = PoolBuffer::<u16>::new(pool, &stream, a.len())?;
    let mut w_pool = PoolBuffer::<u16>::new(pool, &stream, w.len())?;
    let mut bias_pool = PoolBuffer::<u16>::new(pool, &stream, bias.len())?;
    let mut c_pool = PoolBuffer::<u16>::new(pool, &stream, rows * outputs)?;
    stream.memcpy_htod(&half_bits(true, &a), &mut a_pool)?;
    stream.memcpy_htod(&half_bits(true, &w), &mut w_pool)?;
    stream.memcpy_htod(&half_bits(true, &bias), &mut bias_pool)?;
    stream.memcpy_htod(&vec![0x7fc0_u16; rows * outputs], &mut c_pool)?;

    // the plan, the capture and the replay each run on a thread that never bound the
    // context, so every cuBLASLt call must bind it itself
    let graph = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let plan = gpu.matmul_plan(spec)?;
                // SAFETY: the pool buffers outlive the graph's launch, which finishes
                // before they are read and freed
                unsafe {
                    gpu.capture_nonempty(|| {
                        gpu.matmul_bias(
                            &plan,
                            Scale::OVERWRITE,
                            &a_pool,
                            &w_pool,
                            &bias_pool,
                            &mut c_pool,
                        )
                    })
                }
            })
            .join()
            .expect("capture thread panicked")
    })?;
    std::thread::spawn(move || graph.launch())
        .join()
        .expect("replay thread panicked")?;

    let mut c = vec![0_u16; rows * outputs];
    stream.memcpy_dtoh(&c_pool, &mut c)?;
    let c: Vec<f32> = c.into_iter().map(bf16_value).collect();
    check(&spec, "pool buffers", &c, &expected);

    drop((a_pool, w_pool, bias_pool, c_pool));
    gpu.synchronize()?;
    // SAFETY: every allocation from the pool has been freed and the stream is idle
    unsafe { cu::cuMemPoolDestroy(pool) }.result()?;
    Ok(())
}
