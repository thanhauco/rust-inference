//! GPU energy readings through NVML, loaded at runtime (`libnvidia-ml`).

use std::ffi::c_void;

use crate::profiler::EnergyCounter;

type NvmlDevice = *mut c_void;
type InitFn = unsafe extern "C" fn() -> i32;
type HandleFn = unsafe extern "C" fn(u32, *mut NvmlDevice) -> i32;
type EnergyFn = unsafe extern "C" fn(NvmlDevice, *mut u64) -> i32;

/// Cumulative energy of GPU 0 since driver load, in joules.
pub struct NvmlEnergy {
    _lib: libloading::Library,
    device: NvmlDevice,
    energy: EnergyFn,
}

// SAFETY: NVML handles may be used from any thread.
unsafe impl Send for NvmlEnergy {}

impl NvmlEnergy {
    pub fn open() -> Option<Self> {
        // SAFETY: loading the NVML shared library and resolving documented
        // symbols with their documented signatures.
        unsafe {
            let lib = ["libnvidia-ml.so.1", "libnvidia-ml.so"].iter().find_map(|n| libloading::Library::new(n).ok())?;
            let init: InitFn = *lib.get::<InitFn>(b"nvmlInit_v2\0").ok()?;
            let handle: HandleFn = *lib.get::<HandleFn>(b"nvmlDeviceGetHandleByIndex_v2\0").ok()?;
            let energy: EnergyFn = *lib.get::<EnergyFn>(b"nvmlDeviceGetTotalEnergyConsumption\0").ok()?;
            if init() != 0 {
                return None;
            }
            let mut device: NvmlDevice = std::ptr::null_mut();
            if handle(0, &mut device) != 0 {
                return None;
            }
            // the fn pointers stay valid because the library is kept alive
            Some(NvmlEnergy { _lib: lib, device, energy })
        }
    }
}

impl EnergyCounter for NvmlEnergy {
    fn name(&self) -> &str {
        "gpu"
    }

    fn read(&mut self) -> Option<f64> {
        let mut mj = 0u64;
        // SAFETY: device handle from nvmlDeviceGetHandleByIndex.
        let rc = unsafe { (self.energy)(self.device, &mut mj) };
        (rc == 0).then(|| mj as f64 / 1000.0)
    }
}
