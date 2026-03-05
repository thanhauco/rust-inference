//! A small reference-counted tensor.
//!
//! A `Tensor` is a shape + dtype + byte offset into shared storage. Cloning is
//! shallow (like the C++ original); several tensors may view one buffer, which
//! is how a packed gate|up|down expert allocation is split into three weights.
//! FP8 weights optionally carry a 2-D block scale (`[ceil(N/128), ceil(K/128)]`
//! for a `[N, K]` matrix).

use std::alloc::{self, Layout};
use std::sync::Arc;

use memmap2::Mmap;

use crate::dtype::{self, DType};
use crate::error::{Error, Result};

/// Plain-old-data element types that can be viewed directly from tensor bytes.
///
/// # Safety
/// Implementors must be valid for every bit pattern and contain no padding.
pub unsafe trait Pod: Copy + Send + Sync + 'static {}

unsafe impl Pod for f32 {}
unsafe impl Pod for u32 {}
unsafe impl Pod for i32 {}
unsafe impl Pod for i64 {}
unsafe impl Pod for u16 {}
unsafe impl Pod for u8 {}

const CPU_ALIGN: usize = 64;

/// Heap bytes aligned for SIMD loads.
pub struct AlignedBytes {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for AlignedBytes {}
unsafe impl Sync for AlignedBytes {}

impl AlignedBytes {
    pub fn zeroed(len: usize) -> Self {
        if len == 0 {
            return AlignedBytes { ptr: std::ptr::null_mut(), len: 0 };
        }
        let layout = Layout::from_size_align(len, CPU_ALIGN).expect("tensor too large");
        // SAFETY: layout has non-zero size.
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        AlignedBytes { ptr, len }
    }

    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: ptr is a live allocation of len bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: ptr is a live allocation of len bytes and we hold &mut self.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        if self.len != 0 {
            let layout = Layout::from_size_align(self.len, CPU_ALIGN).unwrap();
            // SAFETY: allocated with the same layout in `zeroed`.
            unsafe { alloc::dealloc(self.ptr, layout) };
        }
    }
}

pub enum CpuBuffer {
    Owned(AlignedBytes),
    /// Zero-copy view into a memory-mapped safetensors shard.
    Mapped {
        map: Arc<Mmap>,
        start: usize,
        len: usize,
    },
}

impl CpuBuffer {
    pub fn bytes(&self) -> &[u8] {
        match self {
            CpuBuffer::Owned(b) => b.as_slice(),
            CpuBuffer::Mapped { map, start, len } => &map[*start..*start + *len],
        }
    }

    pub fn len(&self) -> usize {
        match self {
            CpuBuffer::Owned(b) => b.len,
            CpuBuffer::Mapped { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
}

#[derive(Clone)]
pub enum Storage {
    Cpu(Arc<CpuBuffer>),
    #[cfg(feature = "cuda")]
    Cuda(Arc<crate::cuda::DeviceBuffer>),
}

#[derive(Clone)]
pub struct Tensor {
    shape: Vec<usize>,
    dtype: DType,
    storage: Storage,
    /// Byte offset of element 0 inside `storage`.
    offset: usize,
    /// Block scale for FP8 weights, always F32 with 2 dims.
    scale: Option<Arc<Tensor>>,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("device", &self.device())
            .field("scale", &self.scale.as_ref().map(|s| s.shape.clone()))
            .finish()
    }
}

pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

impl Tensor {
    /// Zero-filled tensor. Panics if a device allocation fails (like `Vec`
    /// does on OOM); see `try_zeros`.
    pub fn zeros(shape: &[usize], dtype: DType, device: Device) -> Self {
        Self::try_zeros(shape, dtype, device).expect("device allocation failed")
    }

    pub fn try_zeros(shape: &[usize], dtype: DType, device: Device) -> Result<Self> {
        let len = numel(shape) * dtype.size();
        let storage = match device {
            Device::Cpu => Storage::Cpu(Arc::new(CpuBuffer::Owned(AlignedBytes::zeroed(len)))),
            #[cfg(feature = "cuda")]
            Device::Cuda => Storage::Cuda(Arc::new(crate::cuda::DeviceBuffer::zeroed(len)?)),
        };
        Ok(Tensor { shape: shape.to_vec(), dtype, storage, offset: 0, scale: None })
    }

    /// Uninitialised device tensor (CPU tensors are zeroed anyway).
    pub fn empty(shape: &[usize], dtype: DType, device: Device) -> Result<Self> {
        match device {
            Device::Cpu => Self::try_zeros(shape, dtype, device),
            #[cfg(feature = "cuda")]
            Device::Cuda => {
                let buf = crate::cuda::DeviceBuffer::alloc(numel(shape) * dtype.size())?;
                Ok(Tensor {
                    shape: shape.to_vec(),
                    dtype,
                    storage: Storage::Cuda(Arc::new(buf)),
                    offset: 0,
                    scale: None,
                })
            }
        }
    }

    /// Build a CPU tensor from raw little-endian bytes.
    pub fn from_bytes(bytes: &[u8], shape: &[usize], dtype: DType) -> Result<Self> {
        let expected = numel(shape) * dtype.size();
        if bytes.len() != expected {
            return Err(Error::Shape(format!(
                "{} bytes for shape {shape:?} {dtype}, expected {expected}",
                bytes.len()
            )));
        }
        let mut t = Tensor::zeros(shape, dtype, Device::Cpu);
        t.bytes_mut()?.copy_from_slice(bytes);
        Ok(t)
    }

    pub fn from_slice<T: Pod>(data: &[T], shape: &[usize], dtype: DType) -> Result<Self> {
        if std::mem::size_of::<T>() != dtype.size() {
            return Err(Error::Shape(format!("element size mismatch for {dtype}")));
        }
        // SAFETY: T is Pod, reading its bytes is always valid.
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) };
        Tensor::from_bytes(bytes, shape, dtype)
    }

    pub fn from_f32(data: &[f32], shape: &[usize]) -> Result<Self> {
        Tensor::from_slice(data, shape, DType::F32)
    }

    pub fn from_u32(data: &[u32], shape: &[usize]) -> Result<Self> {
        Tensor::from_slice(data, shape, DType::U32)
    }

    /// Zero-copy tensor over a mapped file region. Falls back to an aligned
    /// copy when the region is not aligned for the element type.
    pub fn from_mmap(map: Arc<Mmap>, start: usize, shape: &[usize], dtype: DType) -> Result<Self> {
        let len = numel(shape) * dtype.size();
        if start + len > map.len() {
            return Err(Error::Safetensors(format!(
                "tensor bytes {start}..{} past end of file ({})",
                start + len,
                map.len()
            )));
        }
        let aligned = (map.as_ptr() as usize + start).is_multiple_of(dtype.size());
        if !aligned {
            return Tensor::from_bytes(&map[start..start + len], shape, dtype);
        }
        Ok(Tensor {
            shape: shape.to_vec(),
            dtype,
            storage: Storage::Cpu(Arc::new(CpuBuffer::Mapped { map, start, len })),
            offset: 0,
            scale: None,
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dim(&self, i: usize) -> usize {
        self.shape[i]
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        numel(&self.shape)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> Device {
        match self.storage {
            Storage::Cpu(_) => Device::Cpu,
            #[cfg(feature = "cuda")]
            Storage::Cuda(_) => Device::Cuda,
        }
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn byte_offset(&self) -> usize {
        self.offset
    }

    pub fn size_in_bytes(&self) -> usize {
        self.numel() * self.dtype.size()
    }

    /// Change the logical shape; the element count must not change.
    pub fn reshape(&mut self, shape: &[usize]) -> Result<()> {
        if numel(shape) != self.numel() {
            return Err(Error::Shape(format!("cannot reshape {:?} to {shape:?}", self.shape)));
        }
        self.shape = shape.to_vec();
        Ok(())
    }

    pub fn reshaped(mut self, shape: &[usize]) -> Result<Self> {
        self.reshape(shape)?;
        Ok(self)
    }

    /// A tensor sharing this one's storage at `byte_offset` (relative to this
    /// tensor's own offset).
    pub fn view(&self, shape: &[usize], byte_offset: usize) -> Result<Self> {
        let avail = self.storage_len().saturating_sub(self.offset);
        if byte_offset + numel(shape) * self.dtype.size() > avail {
            return Err(Error::Shape(format!("view {shape:?} @ {byte_offset} out of bounds")));
        }
        Ok(Tensor {
            shape: shape.to_vec(),
            dtype: self.dtype,
            storage: self.storage.clone(),
            offset: self.offset + byte_offset,
            scale: None,
        })
    }

    /// Reinterpret the same bytes under a different dtype/shape. Used where a
    /// weight packs several logical tensors.
    pub fn view_as(&self, shape: &[usize], dtype: DType, byte_offset: usize) -> Result<Self> {
        let avail = self.storage_len().saturating_sub(self.offset);
        if byte_offset + numel(shape) * dtype.size() > avail {
            return Err(Error::Shape(format!("view {shape:?} {dtype} @ {byte_offset} out of bounds")));
        }
        Ok(Tensor {
            shape: shape.to_vec(),
            dtype,
            storage: self.storage.clone(),
            offset: self.offset + byte_offset,
            scale: None,
        })
    }

    fn storage_len(&self) -> usize {
        match &self.storage {
            Storage::Cpu(b) => b.len(),
            #[cfg(feature = "cuda")]
            Storage::Cuda(b) => b.len(),
        }
    }

    pub fn scale(&self) -> Option<&Tensor> {
        self.scale.as_deref()
    }

    pub fn set_scale(&mut self, scale: Tensor) -> Result<()> {
        if scale.ndim() != 2 || scale.dtype() != DType::F32 {
            return Err(Error::Shape(format!(
                "block scale must be 2-D F32, got {:?} {}",
                scale.shape(),
                scale.dtype()
            )));
        }
        self.scale = Some(Arc::new(scale));
        Ok(())
    }

    pub fn clear_scale(&mut self) {
        self.scale = None;
    }

    // -- CPU byte / element access -------------------------------------------

    /// Host bytes of a CPU tensor. Panics for device tensors (use `to_cpu`).
    pub fn bytes(&self) -> &[u8] {
        match &self.storage {
            Storage::Cpu(b) => &b.bytes()[self.offset..self.offset + self.size_in_bytes()],
            #[cfg(feature = "cuda")]
            Storage::Cuda(_) => panic!("bytes() on a CUDA tensor; call to_cpu() first"),
        }
    }

    /// Mutable bytes. Requires the storage to be uniquely owned heap memory;
    /// mapped weights are read-only.
    pub fn bytes_mut(&mut self) -> Result<&mut [u8]> {
        let range = self.offset..self.offset + self.size_in_bytes();
        match &mut self.storage {
            Storage::Cpu(arc) => match Arc::get_mut(arc) {
                Some(CpuBuffer::Owned(b)) => Ok(&mut b.as_mut_slice()[range]),
                Some(CpuBuffer::Mapped { .. }) => Err(Error::Unsupported("mutating a memory-mapped tensor".into())),
                None => Err(Error::Unsupported("mutating a shared tensor".into())),
            },
            #[cfg(feature = "cuda")]
            Storage::Cuda(_) => Err(Error::Unsupported("host access to a CUDA tensor".into())),
        }
    }

    fn check_elem<T: Pod>(&self) {
        assert_eq!(std::mem::size_of::<T>(), self.dtype.size(), "element size does not match dtype {}", self.dtype);
    }

    pub fn as_slice<T: Pod>(&self) -> &[T] {
        self.check_elem::<T>();
        let bytes = self.bytes();
        // SAFETY: T is Pod; alignment is checked by align_to below.
        let (pre, mid, post) = unsafe { bytes.align_to::<T>() };
        assert!(pre.is_empty() && post.is_empty(), "misaligned tensor data");
        mid
    }

    pub fn as_mut_slice<T: Pod>(&mut self) -> Result<&mut [T]> {
        self.check_elem::<T>();
        let bytes = self.bytes_mut()?;
        // SAFETY: as above.
        let (pre, mid, post) = unsafe { bytes.align_to_mut::<T>() };
        assert!(pre.is_empty() && post.is_empty(), "misaligned tensor data");
        Ok(mid)
    }

    /// Owned copy of the data, decoded to f32 (works for any float dtype; the
    /// FP8 block scale is *not* applied).
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        if self.device() != Device::Cpu {
            return self.to_cpu()?.to_f32_vec();
        }
        Ok(match self.dtype {
            DType::F32 => self.as_slice::<f32>().to_vec(),
            DType::BF16 => self.as_slice::<u16>().iter().map(|&b| dtype::bf16_to_f32(b)).collect(),
            DType::F16 => self.as_slice::<u16>().iter().map(|&h| dtype::f16_to_f32(h)).collect(),
            DType::F8E4M3 => self.as_slice::<u8>().iter().map(|&b| dtype::fp8_e4m3_to_f32(b)).collect(),
            other => return Err(Error::Unsupported(format!("to_f32_vec on {other}"))),
        })
    }

    /// Convert to an owned CPU tensor of `dtype` (float types only).
    pub fn to_dtype(&self, dtype: DType) -> Result<Tensor> {
        if dtype == self.dtype {
            return Ok(self.clone());
        }
        if self.device() != Device::Cpu {
            return Err(Error::Unsupported("to_dtype on a device tensor".into()));
        }
        let vals = self.to_f32_vec()?;
        let mut out = Tensor::zeros(&self.shape, dtype, Device::Cpu);
        match dtype {
            DType::F32 => out.as_mut_slice::<f32>()?.copy_from_slice(&vals),
            DType::BF16 => {
                for (o, v) in out.as_mut_slice::<u16>()?.iter_mut().zip(&vals) {
                    *o = dtype::f32_to_bf16(*v);
                }
            }
            DType::F16 => {
                for (o, v) in out.as_mut_slice::<u16>()?.iter_mut().zip(&vals) {
                    *o = dtype::f32_to_f16(*v);
                }
            }
            DType::F8E4M3 => {
                for (o, v) in out.as_mut_slice::<u8>()?.iter_mut().zip(&vals) {
                    *o = dtype::f32_to_fp8_e4m3(*v);
                }
            }
            other => return Err(Error::Unsupported(format!("to_dtype {other}"))),
        }
        Ok(out)
    }

    /// Deep copy into fresh owned storage (scale is shared).
    pub fn deep_clone(&self) -> Tensor {
        #[cfg(feature = "cuda")]
        if self.device() == Device::Cuda {
            let mut out = crate::cuda::ops::copy(self).expect("device copy failed");
            out.scale = self.scale.clone();
            return out;
        }
        let mut out = Tensor::zeros(&self.shape, self.dtype, self.device());
        out.bytes_mut().expect("fresh tensor").copy_from_slice(self.bytes());
        out.scale = self.scale.clone();
        out
    }

    /// Copy to `device` (the block scale moves along). Same-device is a
    /// shallow clone.
    pub fn to_device(&self, device: Device) -> Result<Tensor> {
        if device == self.device() {
            return Ok(self.clone());
        }
        #[cfg(not(feature = "cuda"))]
        unreachable!("only one device without the cuda feature");
        #[cfg(feature = "cuda")]
        {
            let scale = match &self.scale {
                Some(s) => Some(Arc::new(s.to_device(device)?)),
                None => None,
            };
            let mut out = match (&self.storage, device) {
                (Storage::Cpu(_), Device::Cuda) => {
                    let buf = crate::cuda::DeviceBuffer::alloc(self.size_in_bytes())?;
                    buf.upload(0, self.bytes())?;
                    Tensor {
                        shape: self.shape.clone(),
                        dtype: self.dtype,
                        storage: Storage::Cuda(Arc::new(buf)),
                        offset: 0,
                        scale: None,
                    }
                }
                (Storage::Cuda(buf), Device::Cpu) => {
                    let mut out = Tensor::zeros(&self.shape, self.dtype, Device::Cpu);
                    buf.download(self.offset, out.bytes_mut()?)?;
                    out
                }
                _ => unreachable!("same-device copies return early"),
            };
            out.scale = scale;
            Ok(out)
        }
    }

    pub fn to_cpu(&self) -> Result<Tensor> {
        self.to_device(Device::Cpu)
    }

    /// Device address of element 0. Panics for CPU tensors.
    #[cfg(feature = "cuda")]
    pub fn device_ptr(&self) -> u64 {
        match &self.storage {
            Storage::Cuda(b) => b.ptr() + self.offset as u64,
            Storage::Cpu(_) => panic!("device_ptr() on a CPU tensor"),
        }
    }

    /// Wrap a device buffer (used for expert slots and op outputs).
    #[cfg(feature = "cuda")]
    pub fn from_device_buffer(
        buf: Arc<crate::cuda::DeviceBuffer>,
        shape: &[usize],
        dtype: DType,
        offset: usize,
    ) -> Result<Self> {
        if offset + numel(shape) * dtype.size() > buf.len() {
            return Err(Error::Shape(format!("{shape:?} {dtype} @ {offset} exceeds device buffer of {}", buf.len())));
        }
        Ok(Tensor { shape: shape.to_vec(), dtype, storage: Storage::Cuda(buf), offset, scale: None })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_and_views() {
        let mut t = Tensor::zeros(&[2, 3], DType::F32, Device::Cpu);
        t.as_mut_slice::<f32>().unwrap().copy_from_slice(&[1., 2., 3., 4., 5., 6.]);
        let row1 = t.view(&[3], 3 * 4).unwrap();
        assert_eq!(row1.as_slice::<f32>(), &[4., 5., 6.]);
        assert!(t.view(&[4], 3 * 4).is_err());
        // the view shares storage so t is no longer uniquely owned
        assert!(t.bytes_mut().is_err());
        drop(row1);
        assert!(t.bytes_mut().is_ok());
    }

    #[test]
    fn reshape_checks_numel() {
        let mut t = Tensor::zeros(&[2, 3, 4], DType::BF16, Device::Cpu);
        t.reshape(&[6, 4]).unwrap();
        assert_eq!(t.shape(), &[6, 4]);
        assert!(t.reshape(&[5, 5]).is_err());
    }

    #[test]
    fn dtype_conversion() {
        let t = Tensor::from_f32(&[1.0, -2.5, 0.15625], &[3]).unwrap();
        let b = t.to_dtype(DType::BF16).unwrap();
        assert_eq!(b.to_f32_vec().unwrap(), vec![1.0, -2.5, 0.15625]);
        let f8 = t.to_dtype(DType::F8E4M3).unwrap();
        assert_eq!(f8.to_f32_vec().unwrap(), vec![1.0, -2.5, 0.15625]);
    }
}
