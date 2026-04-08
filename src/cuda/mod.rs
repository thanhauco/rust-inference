//! CUDA backend (cargo feature `cuda`).
//!
//! Built on cudarc's raw driver API with every library loaded at runtime
//! (`libcuda`, `libnvrtc`, `libcublas`), so the crate compiles on machines
//! without a CUDA toolkit. Kernels live in `kernels.cu` and are compiled
//! with NVRTC for the device's compute capability on first use.
//!
//! Like the C++ engine there is a single device and three streams: compute,
//! transfer (expert weight uploads) and a lower-priority secondary stream.

mod buffer;
pub mod moe;
pub mod nvml;
pub mod ops;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::cublas::{result as blas, sys as blas_sys};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

pub use buffer::{DeviceBuffer, PinnedBuffer};

use crate::dtype::DType;
use crate::error::{Error, Result};

pub(crate) const KERNEL_SRC: &str = include_str!("kernels.cu");

const KERNELS: &[&str] = &[
    "embedding",
    "rmsnorm",
    "add_rmsnorm",
    "rope",
    "swiglu",
    "add",
    "to_f32",
    "argmax",
    "topk_softmax",
    "moe_gather",
    "moe_scatter",
    "fp8_gemv",
    "fp8_gemv_swiglu",
    "kv_append",
    "paged_attention",
];

pub(crate) fn err(e: impl std::fmt::Debug) -> Error {
    Error::Cuda(format!("{e:?}"))
}

struct Blas(blas_sys::cublasHandle_t);
// SAFETY: the handle is only used behind the Mutex in `Gpu`.
unsafe impl Send for Blas {}

pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub compute: Arc<CudaStream>,
    pub transfer: Arc<CudaStream>,
    pub secondary: Arc<CudaStream>,
    blas: Mutex<Blas>,
    funcs: HashMap<String, CudaFunction>,
    pub compute_capability: (i32, i32),
}

static GPU: OnceLock<std::result::Result<Gpu, String>> = OnceLock::new();

thread_local! {
    static BOUND: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The process-wide GPU (device 0), initialised on first use. Also makes the
/// context current on the calling thread.
pub fn gpu() -> Result<&'static Gpu> {
    let g = GPU
        .get_or_init(|| {
            Gpu::init().map_err(|e| match e {
                Error::Cuda(m) => m,
                other => other.to_string(),
            })
        })
        .as_ref()
        .map_err(|e| Error::Cuda(e.clone()))?;
    if !BOUND.with(|b| b.get()) {
        g.ctx.bind_to_thread().map_err(err)?;
        BOUND.with(|b| b.set(true));
    }
    Ok(g)
}

/// Whether a usable CUDA device is present (for tests and the runner).
pub fn is_available() -> bool {
    gpu().is_ok()
}

/// Compile the kernel source to PTX for `arch` (e.g. `"compute_80"`).
/// Needs only NVRTC, not a GPU.
pub fn compile_kernels(arch: &'static str) -> Result<String> {
    // SAFETY: only probes whether the library can be loaded.
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return Err(Error::Cuda("libnvrtc not found (check LD_LIBRARY_PATH)".into()));
    }
    let opts = CompileOptions { arch: Some(arch), options: vec!["--std=c++17".into()], ..Default::default() };
    let ptx = compile_ptx_with_opts(KERNEL_SRC, opts).map_err(|e| Error::Cuda(format!("NVRTC: {e}")))?;
    Ok(ptx.to_src())
}

fn arch_name(major: i32, minor: i32) -> &'static str {
    // NVRTC wants a static string; cover the architectures CUDA 12 knows.
    match (major, minor) {
        (7, 0) => "compute_70",
        (7, 2) => "compute_72",
        (7, 5) => "compute_75",
        (8, 0) => "compute_80",
        (8, 6) => "compute_86",
        (8, 7) => "compute_87",
        (8, 9) => "compute_89",
        (9, 0) => "compute_90",
        (10, 0) => "compute_100",
        (12, 0) => "compute_120",
        (m, _) if m >= 12 => "compute_120",
        (m, _) if m >= 10 => "compute_100",
        (9, _) => "compute_90",
        (8, _) => "compute_80",
        _ => "compute_75",
    }
}

impl Gpu {
    fn init() -> Result<Self> {
        // cudarc panics when a library is missing; probe first so callers get
        // an error (and `is_available` a plain false) instead.
        // SAFETY: these only try to dlopen the libraries.
        let present = unsafe {
            [
                ("libcuda", cudarc::driver::sys::is_culib_present()),
                ("libnvrtc", cudarc::nvrtc::sys::is_culib_present()),
                ("libcublas", cudarc::cublas::sys::is_culib_present()),
            ]
        };
        if let Some((name, _)) = present.iter().find(|(_, ok)| !ok) {
            return Err(Error::Cuda(format!("{name} not found (check the driver / LD_LIBRARY_PATH)")));
        }
        let ctx = CudaContext::new(0).map_err(err)?;
        let compute_capability = ctx.compute_capability().map_err(err)?;
        // lower number = higher priority, mirroring the C++ stream priorities
        let compute = ctx.new_stream_with_priority(0).map_err(err)?;
        let transfer = ctx.new_stream_with_priority(-1).map_err(err)?;
        let secondary = ctx.new_stream_with_priority(0).map_err(err)?;

        let ptx = {
            let arch = arch_name(compute_capability.0, compute_capability.1);
            let opts = CompileOptions { arch: Some(arch), options: vec!["--std=c++17".into()], ..Default::default() };
            compile_ptx_with_opts(KERNEL_SRC, opts).map_err(|e| Error::Cuda(format!("NVRTC: {e}")))?
        };
        let module = ctx.load_module(ptx).map_err(err)?;
        let mut funcs = HashMap::new();
        for k in KERNELS {
            for t in ["bf16", "f16"] {
                let name = format!("{k}_{t}");
                funcs.insert(name.clone(), module.load_function(&name).map_err(err)?);
            }
        }

        let handle = blas::create_handle().map_err(err)?;
        Ok(Gpu { ctx, compute, transfer, secondary, blas: Mutex::new(Blas(handle)), funcs, compute_capability })
    }

    /// Kernel `name` instantiated for activation type `dtype`.
    pub(crate) fn func(&self, name: &str, dtype: DType) -> Result<&CudaFunction> {
        let suffix = match dtype {
            DType::BF16 => "bf16",
            DType::F16 => "f16",
            other => return Err(Error::Unsupported(format!("CUDA activations must be BF16/F16, got {other}"))),
        };
        self.funcs.get(&format!("{name}_{suffix}")).ok_or_else(|| Error::Cuda(format!("no kernel {name}_{suffix}")))
    }

    pub fn synchronize(&self) -> Result<()> {
        self.compute.synchronize().map_err(err)
    }

    /// `y[M, N] = x[M, K] W[N, K]^T` with cuBLAS (same column-major trick
    /// as the C++ `matmul_cublas`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemm(
        &self,
        y: u64,
        x: u64,
        w: u64,
        m: usize,
        n: usize,
        k: usize,
        dtype: DType,
        stream: &CudaStream,
    ) -> Result<()> {
        let ty = match dtype {
            DType::BF16 => blas_sys::cudaDataType_t::CUDA_R_16BF,
            DType::F16 => blas_sys::cudaDataType_t::CUDA_R_16F,
            DType::F32 => blas_sys::cudaDataType_t::CUDA_R_32F,
            other => return Err(Error::Unsupported(format!("cuBLAS gemm on {other}"))),
        };
        let (alpha, beta) = (1.0f32, 0.0f32);
        let h = self.blas.lock().unwrap();
        // SAFETY: pointers come from live device allocations of the right
        // sizes (checked by the callers); the handle is serialised by the lock.
        unsafe {
            blas::set_stream(h.0, stream.cu_stream() as _).map_err(err)?;
            blas::gemm_ex(
                h.0,
                blas_sys::cublasOperation_t::CUBLAS_OP_T,
                blas_sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const _,
                w as *const _,
                ty,
                k as i32,
                x as *const _,
                ty,
                k as i32,
                &beta as *const f32 as *const _,
                y as *mut _,
                ty,
                n as i32,
                blas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                blas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
            .map_err(err)
        }
    }
}

/// Launch a kernel: `launch!(stream, func, cfg, &arg0, &arg1, ...)`. The
/// argument list must match the kernel signature in kernels.cu; ops validate
/// buffer sizes before launching.
macro_rules! launch {
    ($stream:expr, $func:expr, $cfg:expr $(, $arg:expr)* $(,)?) => {{
        #[allow(unused_imports)]
        use cudarc::driver::PushKernelArg as _;
        let mut b = $stream.launch_builder($func);
        $( b.arg($arg); )*
        // SAFETY: see above.
        unsafe { b.launch($cfg) }.map(|_| ()).map_err($crate::cuda::err)
    }};
}
pub(crate) use launch;

pub(crate) fn grid1(n: usize, block: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n as u32).div_ceil(block).max(1), 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NVRTC-only: runs wherever libnvrtc can be loaded, GPU or not.
    #[test]
    #[ignore = "needs libnvrtc (set LD_LIBRARY_PATH); run with --ignored"]
    fn kernels_compile_for_all_archs() {
        for arch in ["compute_75", "compute_80", "compute_89", "compute_90"] {
            let ptx = compile_kernels(arch).unwrap();
            for k in KERNELS {
                for t in ["bf16", "f16"] {
                    assert!(ptx.contains(&format!(".entry {k}_{t}(")), "{k}_{t} missing for {arch}");
                }
            }
        }
    }
}
