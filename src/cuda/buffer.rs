//! Raw device and pinned-host allocations.

use cudarc::driver::result;

use super::{err, gpu};
use crate::error::Result;

/// Stream-ordered device memory, freed on the compute stream when dropped.
pub struct DeviceBuffer {
    ptr: u64,
    len: usize,
}

impl DeviceBuffer {
    pub fn alloc(len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(DeviceBuffer { ptr: 0, len: 0 });
        }
        let g = gpu()?;
        // SAFETY: the context is current (gpu() binds it); the pool
        // allocation is ordered on the compute stream like every op.
        let ptr = unsafe { result::malloc_async(g.compute.cu_stream(), len) }.map_err(err)?;
        Ok(DeviceBuffer { ptr, len })
    }

    pub fn zeroed(len: usize) -> Result<Self> {
        let b = Self::alloc(len)?;
        if len > 0 {
            let g = gpu()?;
            // SAFETY: b owns `len` bytes at ptr.
            unsafe { result::memset_d8_async(b.ptr, 0, len, g.compute.cu_stream()) }.map_err(err)?;
        }
        Ok(b)
    }

    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copy from host memory at byte `offset` (ordered on the compute
    /// stream; pageable sources are staged before this returns).
    pub fn upload(&self, offset: usize, src: &[u8]) -> Result<()> {
        assert!(offset + src.len() <= self.len, "upload out of bounds");
        if src.is_empty() {
            return Ok(());
        }
        let g = gpu()?;
        // SAFETY: bounds checked above.
        unsafe { result::memcpy_htod_async(self.ptr + offset as u64, src, g.compute.cu_stream()) }.map_err(err)
    }

    /// Blocking copy to host.
    pub fn download(&self, offset: usize, dst: &mut [u8]) -> Result<()> {
        assert!(offset + dst.len() <= self.len, "download out of bounds");
        if dst.is_empty() {
            return Ok(());
        }
        let g = gpu()?;
        // SAFETY: bounds checked above; we synchronise before returning so
        // `dst` stays borrowed for the whole copy.
        unsafe { result::memcpy_dtoh_async(dst, self.ptr + offset as u64, g.compute.cu_stream()) }.map_err(err)?;
        g.compute.synchronize().map_err(err)
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != 0
            && let Ok(g) = gpu()
        {
            // SAFETY: we own the allocation; stream-ordered free.
            let _ = unsafe { result::free_async(self.ptr, g.compute.cu_stream()) };
        }
    }
}

/// Page-locked host memory (for asynchronous expert uploads and staging).
pub struct PinnedBuffer {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: plain host memory; access is synchronised by the owners.
unsafe impl Send for PinnedBuffer {}
unsafe impl Sync for PinnedBuffer {}

impl PinnedBuffer {
    pub fn alloc(len: usize) -> Result<Self> {
        gpu()?;
        // SAFETY: context current; flags 0 = portable default.
        let ptr = unsafe { result::malloc_host(len.max(1), 0) }.map_err(err)? as *mut u8;
        Ok(PinnedBuffer { ptr, len })
    }

    pub fn from_slice(src: &[u8]) -> Result<Self> {
        let mut b = Self::alloc(src.len())?;
        b.as_mut_slice().copy_from_slice(src);
        Ok(b)
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is a live allocation of len bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, with exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        // SAFETY: allocated by malloc_host.
        let _ = unsafe { result::free_host(self.ptr as *mut _) };
    }
}
