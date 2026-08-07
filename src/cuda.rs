//
// Copyright (c) 2025-2026 Łukasz Szpakowski
// Copyright (c) 2026 Mateusz Szpakowski
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
//! A module that contains a CUDA backend.
use std::default::Default;
use std::ffi::c_int;
use std::sync::Arc;
use std::sync::Mutex;
use crate::Backend;
use crate::BackendArray;
use crate::Error;
use crate::Result;
use crate::mutex_lock;

pub use cudarc::cublas::result::CublasError;
pub use cudarc::driver::DriverError;

use cudarc::cublas::result::sgemm;
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::CudaBlas;
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::CudaContext;
use cudarc::driver::CudaModule;
use cudarc::driver::CudaFunction;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::CompileError;
use cudarc::nvrtc::CompileOptions;
use cudarc::nvrtc::Ptx;
use cudarc::nvrtc::compile_ptx_with_opts;

const SOURCE: &'static str = include_str!("cuda.cu");

const PTX_SOURCE: &'static str = include_str!("ptx_mul.ptx");

/// A structure of CUDA backend array.
///
/// This structure contains the reference to the device memory.
#[derive(Debug)]
pub struct CudaBackendArray
{
    slice: Arc<Mutex<CudaSlice<f32>>>,
    len: usize,
}

struct CudaInnerBackend
{
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    ptx_module: Option<Arc<CudaModule>>,
    cublas: Option<CudaBlas>,
}

/// A structure of CUDA backend.
pub struct CudaBackend
{
    inner: Mutex<CudaInnerBackend>,
    has_cublas: bool,
    has_mma: bool,
    has_ptx: bool,
}

fn preferred_launch_config(n: usize, m: usize, item_row_count: usize, item_col_count: usize, are_swapped_dims: bool, is_mul: bool, is_mma: bool, is_ptx: bool) -> LaunchConfig
{
    if m <= item_col_count && !is_mul {
        let n2 = (((n + item_row_count - 1) / item_row_count + 1023) / 1024) as u32;
        if !are_swapped_dims {
            LaunchConfig {
                grid_dim: (n2, 1, 1),
                block_dim: (1024, 1, 1),
                shared_mem_bytes: 0,
            }
        } else {
            LaunchConfig {
                grid_dim: (1, n2, 1),
                block_dim: (1, 1024, 1),
                shared_mem_bytes: 0,
            }
        }
    } else if n <= item_row_count && !is_mul {
        let m2 = (((m + item_col_count - 1) / item_col_count + 1023) / 1024) as u32;
        if !are_swapped_dims {
            LaunchConfig {
                grid_dim: (1, m2, 1),
                block_dim: (1, 1024, 1),
                shared_mem_bytes: 0,
            }
        } else {
            LaunchConfig {
                grid_dim: (m2, 1, 1),
                block_dim: (1024, 1, 1),
                shared_mem_bytes: 0,
            }
        }
    } else if is_mul {
        if is_mma {
            let n2 = ((n + 63) / 64) as u32;
            let m2 = ((m + 63) / 64) as u32;
            if !are_swapped_dims {
                LaunchConfig {
                    grid_dim: (n2, m2, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 0,
                }
            } else {
                LaunchConfig {
                    grid_dim: (m2, n2, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 0,
                }
            }
        } else {
            if is_ptx {
                let n2 = (((n + 3) / 4 + 31) / 32) as u32;
                let m2 = (((m + 3) / 4 + 31) / 32) as u32;
                if !are_swapped_dims {
                    LaunchConfig {
                        grid_dim: (n2, m2, 1),
                        block_dim: (32, 32, 1),
                        shared_mem_bytes: 0,
                    }
                } else {
                    LaunchConfig {
                        grid_dim: (m2, n2, 1),
                        block_dim: (32, 32, 1),
                        shared_mem_bytes: 0,
                    }
                }
            } else {
                let n2 = (((n + 7) / 8 + 15) / 16) as u32;
                let m2 = (((m + 3) / 4 + 15) / 16) as u32;
                if !are_swapped_dims {
                    LaunchConfig {
                        grid_dim: (n2, m2, 1),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    }
                } else {
                    LaunchConfig {
                        grid_dim: (m2, n2, 1),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    }
                }
            }
        }
    } else {
        let n2 = (((n + item_row_count - 1) / item_row_count + 31) / 32) as u32;
        let m2 = (((m + item_col_count - 1) / item_col_count + 31) / 32) as u32;
        if !are_swapped_dims {
            LaunchConfig {
                grid_dim: (n2, m2, 1),
                block_dim: (32, 32, 1),
                shared_mem_bytes: 0,
            }
        } else {
            LaunchConfig {
                grid_dim: (m2, n2, 1),
                block_dim: (32, 32, 1),
                shared_mem_bytes: 0,
            }
        }
    }
}

impl CudaBackend
{
    /// Creates a CUDA backend for a first device.
    pub fn new() -> Result<CudaBackend>
    {
        if cfg!(feature = "default_cublas") {
            Self::new_with_ordinal_and_flags(0, true, false)
        } else if cfg!(feature = "default_mma") {
            Self::new_with_ordinal_and_flags(0, false, true)
        } else if cfg!(feature = "default_ptx") {
            Self::new_with_ordinal_and_flags_and_ptx_flag(0, false, false, true)
        } else {
            Self::new_with_ordinal_and_flags(0, false, false)
        }
    }
    
    /// Creates a CUDA backend with the ordinal number and the flags.
    ///
    /// This method takes the following flags:
    ///
    /// - `is_cublas` - use the cuBLAS library to multiplication of matrices
    /// - `is_mma` - use the mma instruction to multiplication of matrices
    pub fn new_with_ordinal_and_flags(ordinal: usize, is_cublas: bool, is_mma: bool) -> Result<CudaBackend>
    { Self::new_with_ordinal_and_flags_and_ptx_flag(ordinal, is_cublas, is_mma, false) }

    pub fn new_with_ordinal_and_flags_and_ptx_flag(ordinal: usize, is_cublas: bool, is_mma: bool, is_ptx: bool) -> Result<CudaBackend>
    {
        let context = match CudaContext::new(ordinal) {
            Ok(tmp_device) => tmp_device,
            Err(err) => return Err(Error::Cuda(err)),
        };
        let mut options: CompileOptions = Default::default();
        if is_mma {
            options.options = vec![String::from("-DUNMTX_GPU_MMA=1")];
            options.arch = Some("sm_80");
        }
        let ptx = match compile_ptx_with_opts(SOURCE, options) {
            Ok(tmp_ptx) => tmp_ptx,
            Err(CompileError::CompileError { log, .. }) => return Err(Error::Compilation(log.as_c_str().to_string_lossy().into_owned())),
            Err(err) => return Err(Error::Compilation(format!("{}", err))),
        };
        let module = match context.load_module(ptx) {
            Ok(module) => module,
            Err(err) => return Err(Error::Cuda(err)),
        };
        let tmp_is_ptx = if !is_cublas && !is_mma {
            is_ptx
        } else {
            false
        };
        let ptx_module = if tmp_is_ptx {
            match context.load_module(Ptx::from_src(PTX_SOURCE)) {
                Ok(ptx_module) => Some(ptx_module),
                Err(err) => return Err(Error::Cuda(err)),
            }
        } else {
            None
        };
        let stream = context.default_stream();
        let cublas = if is_cublas {
            match CudaBlas::new(stream.clone()) {
                Ok(tmp_cublas) => Some(tmp_cublas),
                Err(err) => return Err(Error::Cublas(err)),
            }
        } else {
            None
        };
        Ok(CudaBackend { inner: Mutex::new(CudaInnerBackend { context, stream, module, ptx_module, cublas, }), has_cublas: is_cublas, has_mma: is_mma, has_ptx: tmp_is_ptx, })
    }
    
    pub fn has_cublas(&self) -> bool
    { self.has_cublas }
    
    fn check_and_launch2<F, G>(&self, kernel_name: &str, a: &BackendArray, b: &BackendArray, f: F, g: G) -> Result<()>
        where F: FnOnce(&CudaBackendArray, &CudaBackendArray) -> Result<()>,
            G: FnOnce(&CudaInnerBackend, CudaFunction, CUdeviceptr, CUdeviceptr) -> Result<()>
    {
        #[allow(unreachable_patterns)]
        match (a, b) {
            (BackendArray::Cuda(a2), BackendArray::Cuda(b2)) => {
                f(a2, b2)?;
                let inner_g = mutex_lock(&self.inner)?;
                let kernel = match inner_g.module.load_function(kernel_name) {
                    Ok(tmp_kernel) => tmp_kernel,
                    Err(_) => return Err(Error::NoKernel(String::from(kernel_name))),
                };
                if !Arc::ptr_eq(&a2.slice, &b2.slice) {
                    let a_slice_g = mutex_lock(&a2.slice)?;
                    let mut b_slice_g = mutex_lock(&b2.slice)?;
                    let a_device_ptr = a_slice_g.device_ptr(&inner_g.stream).0;
                    let b_device_ptr = b_slice_g.device_ptr_mut(&inner_g.stream).0;
                    g(&*inner_g, kernel, a_device_ptr, b_device_ptr)?;
                } else {
                    let mut a_slice_g = mutex_lock(&a2.slice)?;
                    let a_device_ptr = a_slice_g.device_ptr_mut(&inner_g.stream).0;
                    g(&*inner_g, kernel, a_device_ptr, a_device_ptr)?;
                }
                match inner_g.context.synchronize() {
                    Ok(()) => (),
                    Err(err) => return Err(Error::Cuda(err)),
                }
                Ok(())
            },
            _ => Err(Error::InvalidBackendArray),
        }
    }

    fn check_and_launch3<F, G>(&self, kernel_name: &str, a: &BackendArray, b: &BackendArray, c: &BackendArray, f: F, g: G) -> Result<()>
        where F: FnOnce(&CudaBackendArray, &CudaBackendArray, &CudaBackendArray) -> Result<()>,
            G: FnOnce(&CudaInnerBackend, CudaFunction, CUdeviceptr, CUdeviceptr, CUdeviceptr) -> Result<()>
    {
        #[allow(unreachable_patterns)]
        match (a, b, c) {
            (BackendArray::Cuda(a2), BackendArray::Cuda(b2), BackendArray::Cuda(c2)) => {
                f(a2, b2, c2)?;
                let inner_g = mutex_lock(&self.inner)?;
                let kernel = match inner_g.module.load_function(kernel_name) {
                    Ok(tmp_kernel) => tmp_kernel,
                    Err(_) => return Err(Error::NoKernel(String::from(kernel_name))),
                };
                match (Arc::ptr_eq(&a2.slice, &b2.slice), Arc::ptr_eq(&a2.slice, &c2.slice), Arc::ptr_eq(&b2.slice, &c2.slice)) {
                    (false, false, false) => {
                        let a_slice_g = mutex_lock(&a2.slice)?;
                        let b_slice_g = mutex_lock(&b2.slice)?;
                        let mut c_slice_g = mutex_lock(&c2.slice)?;
                        let a_device_ptr = a_slice_g.device_ptr(&inner_g.stream).0;
                        let b_device_ptr = b_slice_g.device_ptr(&inner_g.stream).0;
                        let c_device_ptr = c_slice_g.device_ptr_mut(&inner_g.stream).0;
                        g(&*inner_g, kernel, a_device_ptr, b_device_ptr, c_device_ptr)?
                    },
                    (true, false, false) => {
                        let a_slice_g = mutex_lock(&a2.slice)?;
                        let mut c_slice_g = mutex_lock(&c2.slice)?;
                        let a_device_ptr = a_slice_g.device_ptr(&inner_g.stream).0;
                        let c_device_ptr = c_slice_g.device_ptr_mut(&inner_g.stream).0;
                        g(&*inner_g, kernel, a_device_ptr, a_device_ptr, c_device_ptr)?
                    },
                    (false, true, false) => {
                        let mut a_slice_g = mutex_lock(&a2.slice)?;
                        let b_slice_g = mutex_lock(&b2.slice)?;
                        let a_device_ptr = a_slice_g.device_ptr_mut(&inner_g.stream).0;
                        let b_device_ptr = b_slice_g.device_ptr(&inner_g.stream).0;
                        g(&*inner_g, kernel, a_device_ptr, b_device_ptr, a_device_ptr)?
                    },
                    (false, false, true) => {
                        let a_slice_g = mutex_lock(&a2.slice)?;
                        let mut b_slice_g = mutex_lock(&b2.slice)?;
                        let a_device_ptr = a_slice_g.device_ptr(&inner_g.stream).0;
                        let b_device_ptr = b_slice_g.device_ptr_mut(&inner_g.stream).0;
                        g(&*inner_g, kernel, a_device_ptr, b_device_ptr, b_device_ptr)?
                    },
                    _ => {
                        let mut a_slice_g = mutex_lock(&a2.slice)?;
                        let a_device_ptr = a_slice_g.device_ptr_mut(&inner_g.stream).0;
                        g(&*inner_g, kernel, a_device_ptr, a_device_ptr, a_device_ptr)?
                    },
                }
                match inner_g.context.synchronize() {
                    Ok(()) => (),
                    Err(err) => return Err(Error::Cuda(err)),
                }
                Ok(())
            },
            _ => Err(Error::InvalidBackendArray),
        }
    }    

    fn check_and_launch_ptx3<F, G>(&self, kernel_name: &str, a: &BackendArray, b: &BackendArray, c: &BackendArray, f: F, g: G) -> Result<()>
        where F: FnOnce(&CudaBackendArray, &CudaBackendArray, &CudaBackendArray) -> Result<()>,
            G: FnOnce(&CudaInnerBackend, CudaFunction, CUdeviceptr, CUdeviceptr, CUdeviceptr) -> Result<()>
    {
        #[allow(unreachable_patterns)]
        match (a, b, c) {
            (BackendArray::Cuda(a2), BackendArray::Cuda(b2), BackendArray::Cuda(c2)) => {
                f(a2, b2, c2)?;
                let inner_g = mutex_lock(&self.inner)?;
                let kernel = match &inner_g.ptx_module {
                    Some(ptx_module) => {
                        match ptx_module.load_function(kernel_name) {
                            Ok(tmp_kernel) => tmp_kernel,
                            Err(_) => return Err(Error::NoKernel(String::from(kernel_name))),
                        }
                    },
                    None => return Err(Error::NoPtxModule),
                };
                match (Arc::ptr_eq(&a2.slice, &b2.slice), Arc::ptr_eq(&a2.slice, &c2.slice), Arc::ptr_eq(&b2.slice, &c2.slice)) {
                    (false, false, false) => {
                        let a_slice_g = mutex_lock(&a2.slice)?;
                        let b_slice_g = mutex_lock(&b2.slice)?;
                        let mut c_slice_g = mutex_lock(&c2.slice)?;
                        let a_device_ptr = a_slice_g.device_ptr(&inner_g.stream).0;
                        let b_device_ptr = b_slice_g.device_ptr(&inner_g.stream).0;
                        let c_device_ptr = c_slice_g.device_ptr_mut(&inner_g.stream).0;
                        g(&*inner_g, kernel, a_device_ptr, b_device_ptr, c_device_ptr)?
                    },
                    (true, false, false) => {
                        let a_slice_g = mutex_lock(&a2.slice)?;
                        let mut c_slice_g = mutex_lock(&c2.slice)?;
                        let a_device_ptr = a_slice_g.device_ptr(&inner_g.stream).0;
                        let c_device_ptr = c_slice_g.device_ptr_mut(&inner_g.stream).0;
                        g(&*inner_g, kernel, a_device_ptr, a_device_ptr, c_device_ptr)?
                    },
                    (false, true, false) => {
                        let mut a_slice_g = mutex_lock(&a2.slice)?;
                        let b_slice_g = mutex_lock(&b2.slice)?;
                        let a_device_ptr = a_slice_g.device_ptr_mut(&inner_g.stream).0;
                        let b_device_ptr = b_slice_g.device_ptr(&inner_g.stream).0;
                        g(&*inner_g, kernel, a_device_ptr, b_device_ptr, a_device_ptr)?
                    },
                    (false, false, true) => {
                        let a_slice_g = mutex_lock(&a2.slice)?;
                        let mut b_slice_g = mutex_lock(&b2.slice)?;
                        let a_device_ptr = a_slice_g.device_ptr(&inner_g.stream).0;
                        let b_device_ptr = b_slice_g.device_ptr_mut(&inner_g.stream).0;
                        g(&*inner_g, kernel, a_device_ptr, b_device_ptr, b_device_ptr)?
                    },
                    _ => {
                        let mut a_slice_g = mutex_lock(&a2.slice)?;
                        let a_device_ptr = a_slice_g.device_ptr_mut(&inner_g.stream).0;
                        g(&*inner_g, kernel, a_device_ptr, a_device_ptr, a_device_ptr)?
                    },
                }
                match inner_g.context.synchronize() {
                    Ok(()) => (),
                    Err(err) => return Err(Error::Cuda(err)),
                }
                Ok(())
            },
            _ => Err(Error::InvalidBackendArray),
        }
    }
    
    fn check_and_launch_cublas3<F, G>(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, f: F, g: G) -> Result<()>
        where F: FnOnce(&CudaBackendArray, &CudaBackendArray, &CudaBackendArray) -> Result<()>,
            G: FnOnce(&CudaInnerBackend, CUdeviceptr, CUdeviceptr, CUdeviceptr) -> Result<()>
    {
        #[allow(unreachable_patterns)]
        match (a, b, c) {
            (BackendArray::Cuda(a2), BackendArray::Cuda(b2), BackendArray::Cuda(c2)) => {
                f(a2, b2, c2)?;
                let inner_g = mutex_lock(&self.inner)?;
                match (Arc::ptr_eq(&a2.slice, &b2.slice), Arc::ptr_eq(&a2.slice, &c2.slice), Arc::ptr_eq(&b2.slice, &c2.slice)) {
                    (false, false, false) => {
                        let a_slice_g = mutex_lock(&a2.slice)?;
                        let b_slice_g = mutex_lock(&b2.slice)?;
                        let mut c_slice_g = mutex_lock(&c2.slice)?;
                        let a_device_ptr = a_slice_g.device_ptr(&inner_g.stream).0;
                        let b_device_ptr = b_slice_g.device_ptr(&inner_g.stream).0;
                        let c_device_ptr = c_slice_g.device_ptr_mut(&inner_g.stream).0;
                        g(&*inner_g, a_device_ptr, b_device_ptr, c_device_ptr)?
                    },
                    (true, false, false) => {
                        let a_slice_g = mutex_lock(&a2.slice)?;
                        let mut c_slice_g = mutex_lock(&c2.slice)?;
                        let a_device_ptr = a_slice_g.device_ptr(&inner_g.stream).0;
                        let c_device_ptr = c_slice_g.device_ptr_mut(&inner_g.stream).0;
                        g(&*inner_g, a_device_ptr, a_device_ptr, c_device_ptr)?
                    },
                    (false, true, false) => {
                        let mut a_slice_g = mutex_lock(&a2.slice)?;
                        let b_slice_g = mutex_lock(&b2.slice)?;
                        let a_device_ptr = a_slice_g.device_ptr_mut(&inner_g.stream).0;
                        let b_device_ptr = b_slice_g.device_ptr(&inner_g.stream).0;
                        g(&*inner_g, a_device_ptr, b_device_ptr, a_device_ptr)?
                    },
                    (false, false, true) => {
                        let a_slice_g = mutex_lock(&a2.slice)?;
                        let mut b_slice_g = mutex_lock(&b2.slice)?;
                        let a_device_ptr = a_slice_g.device_ptr(&inner_g.stream).0;
                        let b_device_ptr = b_slice_g.device_ptr_mut(&inner_g.stream).0;
                        g(&*inner_g, a_device_ptr, b_device_ptr, b_device_ptr)?
                    },
                    _ => {
                        let mut a_slice_g = mutex_lock(&a2.slice)?;
                        let a_device_ptr = a_slice_g.device_ptr_mut(&inner_g.stream).0;
                        g(&*inner_g, a_device_ptr, a_device_ptr, a_device_ptr)?
                    },
                }
                match inner_g.context.synchronize() {
                    Ok(()) => (),
                    Err(err) => return Err(Error::Cuda(err)),
                }
                Ok(())
            },
            _ => Err(Error::InvalidBackendArray),
        }
    }
    
    fn check_and_launch_for_fun(&self, kernel_name: &str, a: &BackendArray, b: &BackendArray, n: usize, m: usize, item_row_count: usize, item_col_count: usize, are_swapped_dims: bool) -> Result<()>
    {
        let is_mma = self.has_mma;
        let is_ptx = self.has_ptx;
        self.check_and_launch2(kernel_name, a, b, |a2, b2| {
                if a2.len != n * m {
                    return Err(Error::BackendArrayElemCount(a2.len, n * m));
                }
                if b2.len != n * m {
                    return Err(Error::BackendArrayElemCount(b2.len, n * m));
                }
                Ok(())
        }, |inner_g, kernel, a_param, b_param| {
                let config = preferred_launch_config(n, m, item_row_count, item_col_count, are_swapped_dims, false, is_mma, is_ptx);
                let mut launch_args = inner_g.stream.launch_builder(&kernel);
                launch_args.arg(&a_param)
                    .arg(&b_param)
                    .arg(&n)
                    .arg(&m);
                unsafe {
                    match launch_args.launch(config) {
                        Ok(_) => Ok(()),
                        Err(err) => Err(Error::Cuda(err)),
                    }
                }
        })
    }

    fn check_and_launch_for_op(&self, kernel_name: &str, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, item_row_count: usize, item_col_count: usize, are_swapped_dims: bool) -> Result<()>
    {
        let is_mma = self.has_mma;
        let is_ptx = self.has_ptx;
        self.check_and_launch3(kernel_name, a, b, c, |a2, b2, c2| {
                if a2.len != n * m {
                    return Err(Error::BackendArrayElemCount(a2.len, n * m));
                }
                if b2.len != n * m {
                    return Err(Error::BackendArrayElemCount(b2.len, n * m));
                }
                if c2.len != n * m {
                    return Err(Error::BackendArrayElemCount(c2.len, n * m));
                }
                Ok(())
        }, |inner_g, kernel, a_param, b_param, c_param| {
                let config = preferred_launch_config(n, m, item_row_count, item_col_count, are_swapped_dims, false, is_mma, is_ptx);
                let mut launch_args = inner_g.stream.launch_builder(&kernel);
                launch_args.arg(&a_param)
                    .arg(&b_param)
                    .arg(&c_param)
                    .arg(&n)
                    .arg(&m);
                unsafe {
                    match launch_args.launch(config) {
                        Ok(_) => Ok(()),
                        Err(err) => Err(Error::Cuda(err)),
                    }
                }
        })
    }

    fn check_and_launch_for_mul(&self, kernel_name: &str, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize, item_row_count: usize, item_col_count: usize, are_swapped_dims: bool) -> Result<()>
    {
        let is_mma = self.has_mma;
        let is_ptx = self.has_ptx;
        self.check_and_launch3(kernel_name, a, b, c, |a2, b2, c2| {
                if a2.len != n * l {
                    return Err(Error::BackendArrayElemCount(a2.len, n * l));
                }
                if b2.len != l * m {
                    return Err(Error::BackendArrayElemCount(b2.len, l * m));
                }
                if c2.len != n * m {
                    return Err(Error::BackendArrayElemCount(c2.len, n * m));
                }
                Ok(())
        }, |inner_g, kernel, a_param, b_param, c_param| {
                let config = preferred_launch_config(n, m, item_row_count, item_col_count, are_swapped_dims, true, is_mma, is_ptx);
                let mut launch_args = inner_g.stream.launch_builder(&kernel);
                launch_args.arg(&a_param)
                    .arg(&b_param)
                    .arg(&c_param)
                    .arg(&n)
                    .arg(&m)
                    .arg(&l);
                unsafe {
                    match launch_args.launch(config) {
                        Ok(_) => Ok(()),
                        Err(err) => Err(Error::Cuda(err)),
                    }
                }
        })
    }

    fn check_and_launch_for_scalar(&self, kernel_name: &str, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize, item_row_count: usize, item_col_count: usize, are_swapped_dims: bool) -> Result<()>
    {
        let is_mma = self.has_mma;
        let is_ptx = self.has_ptx;
        self.check_and_launch2(kernel_name, a, c, |a2, c2| {
                if a2.len != n * m  {
                    return Err(Error::BackendArrayElemCount(a2.len, n * m));
                }
                if c2.len != n * m {
                    return Err(Error::BackendArrayElemCount(c2.len, n * m));
                }
                Ok(())
        }, |inner_g, kernel, a_param, c_param| {
                let config = preferred_launch_config(n, m, item_row_count, item_col_count, are_swapped_dims, false, is_mma, is_ptx);
                let mut launch_args = inner_g.stream.launch_builder(&kernel);
                launch_args.arg(&a_param)
                    .arg(&b)
                    .arg(&c_param)
                    .arg(&n)
                    .arg(&m);
                unsafe {
                    match launch_args.launch(config) {
                        Ok(_) => Ok(()),
                        Err(err) => Err(Error::Cuda(err)),
                    }
                }
        })
    }

    fn check_and_launch_for_fun_and_tiles(&self, kernel_name: &str, a: &BackendArray, b: &BackendArray, n: usize, m: usize, item_row_count: usize, item_col_count: usize, are_swapped_dims: bool) -> Result<()>
    {
        let is_mma = self.has_mma;
        let is_ptx = self.has_ptx;
        self.check_and_launch2(kernel_name, a, b, |a2, b2| {
                if a2.len != n * m {
                    return Err(Error::BackendArrayElemCount(a2.len, n * m));
                }
                if b2.len != n * m {
                    return Err(Error::BackendArrayElemCount(b2.len, n * m));
                }
                Ok(())
        }, |inner_g, kernel, a_param, b_param| {
                let config = preferred_launch_config(n, m, item_row_count, item_col_count, are_swapped_dims, false, is_mma, is_ptx);
                let mut launch_args = inner_g.stream.launch_builder(&kernel);
                launch_args.arg(&a_param)
                    .arg(&b_param)
                    .arg(&n)
                    .arg(&m);
                unsafe {
                    match launch_args.launch(config) {
                        Ok(_) => Ok(()),
                        Err(err) => Err(Error::Cuda(err)),
                    }
                }
        })
    }

    fn check_and_launch_for_repeat_col(&self, kernel_name: &str, a: &BackendArray, b: &BackendArray, n: usize, m: usize, item_row_count: usize, item_col_count: usize, are_swapped_dims: bool) -> Result<()>
    {
        let is_mma = self.has_mma;
        let is_ptx = self.has_ptx;
        self.check_and_launch2(kernel_name, a, b, |a2, b2| {
                if a2.len != n {
                    return Err(Error::BackendArrayElemCount(a2.len, n));
                }
                if b2.len != n * m {
                    return Err(Error::BackendArrayElemCount(b2.len, n * m));
                }
                Ok(())
        }, |inner_g, kernel, a_param, b_param| {
                let config = preferred_launch_config(n, m, item_row_count, item_col_count, are_swapped_dims, false, is_mma, is_ptx);
                let mut launch_args = inner_g.stream.launch_builder(&kernel);
                launch_args.arg(&a_param)
                    .arg(&b_param)
                    .arg(&n)
                    .arg(&m);
                unsafe {
                    match launch_args.launch(config) {
                        Ok(_) => Ok(()),
                        Err(err) => Err(Error::Cuda(err)),
                    }
                }
        })
    }

    fn check_and_launch_for_repeat_row(&self, kernel_name: &str, a: &BackendArray, b: &BackendArray, n: usize, m: usize, item_row_count: usize, item_col_count: usize, are_swapped_dims: bool) -> Result<()>
    {
        let is_mma = self.has_mma;
        let is_ptx = self.has_ptx;
        self.check_and_launch2(kernel_name, a, b, |a2, b2| {
                if a2.len != m {
                    return Err(Error::BackendArrayElemCount(a2.len, m));
                }
                if b2.len != n * m {
                    return Err(Error::BackendArrayElemCount(b2.len, n * m));
                }
                Ok(())
        }, |inner_g, kernel, a_param, b_param| {
                let config = preferred_launch_config(n, m, item_row_count, item_col_count, are_swapped_dims, false, is_mma, is_ptx);
                let mut launch_args = inner_g.stream.launch_builder(&kernel);
                launch_args.arg(&a_param)
                    .arg(&b_param)
                    .arg(&n)
                    .arg(&m);
                unsafe {
                    match launch_args.launch(config) {
                        Ok(_) => Ok(()),
                        Err(err) => Err(Error::Cuda(err)),
                    }
                }
        })
    }    
    
    fn check_and_launch_for_ptx_mul(&self, kernel_name: &str, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize, item_row_count: usize, item_col_count: usize, are_swapped_dims: bool) -> Result<()>
    {
        let is_mma = self.has_mma;
        let is_ptx = self.has_ptx;
        self.check_and_launch_ptx3(kernel_name, a, b, c, |a2, b2, c2| {
                if a2.len != n * l {
                    return Err(Error::BackendArrayElemCount(a2.len, n * l));
                }
                if b2.len != l * m {
                    return Err(Error::BackendArrayElemCount(b2.len, l * m));
                }
                if c2.len != n * m {
                    return Err(Error::BackendArrayElemCount(c2.len, n * m));
                }
                Ok(())
        }, |inner_g, kernel, a_param, b_param, c_param| {
                let config = preferred_launch_config(n, m, item_row_count, item_col_count, are_swapped_dims, true, is_mma, is_ptx);
                let mut launch_args = inner_g.stream.launch_builder(&kernel);
                launch_args.arg(&a_param)
                    .arg(&b_param)
                    .arg(&c_param)
                    .arg(&n)
                    .arg(&m)
                    .arg(&l);
                unsafe {
                    match launch_args.launch(config) {
                        Ok(_) => Ok(()),
                        Err(err) => Err(Error::Cuda(err)),
                    }
                }
        })
    }

    fn check_and_launch_cublas_for_mul(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize, is_trans_a: bool, is_trans_b: bool) -> Result<()>
    {
        self.check_and_launch_cublas3(a, b, c, |a2, b2, c2| {
                if a2.len != n * l {
                    return Err(Error::BackendArrayElemCount(a2.len, n * l));
                }
                if b2.len != l * m {
                    return Err(Error::BackendArrayElemCount(b2.len, l * m));
                }
                if c2.len != n * m {
                    return Err(Error::BackendArrayElemCount(c2.len, n * m));
                }
                Ok(())
        }, |inner, a_device_ptr, b_device_ptr, c_device_ptr| {
                unsafe {
                    match &inner.cublas {
                        Some(cublas) => {
                            let (transa, lda) = if is_trans_a {
                                (cublasOperation_t::CUBLAS_OP_T, n as c_int)
                            } else {
                                (cublasOperation_t::CUBLAS_OP_N, l as c_int)
                            };
                            let (transb, ldb) = if is_trans_b {
                                (cublasOperation_t::CUBLAS_OP_T, l as c_int)
                            } else {
                                (cublasOperation_t::CUBLAS_OP_N, m as c_int)
                            };
                            let alpha = 1.0f32;
                            let beta = 0.0f32;
                            let res = sgemm(*cublas.handle(),
                                transb, transa,
                                m as c_int, n as c_int, l as c_int,
                                (&alpha) as *const _,
                                b_device_ptr as *const _, ldb,
                                a_device_ptr as *const _, lda,
                                (&beta) as *const _,
                                c_device_ptr as *mut _, m as c_int);
                            match res {
                                Ok(()) => Ok(()),
                                Err(err) => Err(Error::Cublas(err)),
                            }
                        },
                        None => Err(Error::NoCublas),
                    }
                }
        })
    }
}

impl Backend for CudaBackend
{
    fn name(&self) -> &'static str
    {
        if self.has_cublas {
            "CUDA(cuBLAS)"
        } else if self.has_mma {
            "CUDA(mma)"
        } else if self.has_ptx {
            "CUDA(PTX)"
        } else {
            "CUDA"
        }
    }
    
    fn has_cublas(&self) -> bool
    { self.has_cublas }

    unsafe fn alloc(&self, n: usize) -> Result<BackendArray>
    {
        let inner_g = mutex_lock(&self.inner)?;
        let slice: CudaSlice<f32> = match inner_g.stream.alloc(n) {
            Ok(tmp_slice) => tmp_slice,
            Err(err) => return Err(Error::Cuda(err)),
        };
        let cuda_array = CudaBackendArray { slice: Arc::new(Mutex::new(slice)), len: n, };
        Ok(BackendArray::Cuda(cuda_array))
    }

    fn alloc_and_store_zeros(&self, n: usize) -> Result<BackendArray>
    {
        let inner_g = mutex_lock(&self.inner)?;
        let slice: CudaSlice<f32> = match inner_g.stream.alloc_zeros(n) {
            Ok(tmp_slice) => tmp_slice,
            Err(err) => return Err(Error::Cuda(err)),
        };
        let cuda_array = CudaBackendArray { slice: Arc::new(Mutex::new(slice)), len: n, };
        Ok(BackendArray::Cuda(cuda_array))
    }
    
    fn alloc_and_store(&self, elems: &[f32]) -> Result<BackendArray>
    {
        let inner_g = mutex_lock(&self.inner)?;
        let slice: CudaSlice<f32> = match inner_g.stream.clone_htod(elems) {
            Ok(tmp_slice) => tmp_slice,
            Err(err) => return Err(Error::Cuda(err)),
        };
        match inner_g.context.synchronize() {
            Ok(()) => (),
            Err(err) => return Err(Error::Cuda(err)),
        };
        let cuda_array = CudaBackendArray { slice: Arc::new(Mutex::new(slice)), len: elems.len(), };
        Ok(BackendArray::Cuda(cuda_array))
    }
    
    fn load(&self, a: &BackendArray, elems: &mut [f32]) -> Result<()>
    {
        #[allow(unreachable_patterns)]
        match a {
            BackendArray::Cuda(a2) => {
                if a2.len != elems.len() {
                    return Err(Error::BackendArrayElemCount(a2.len, elems.len()));
                }
                let inner_g = mutex_lock(&self.inner)?;
                let a_slice_g = mutex_lock(&a2.slice)?;
                match inner_g.stream.memcpy_dtoh(&(*a_slice_g), elems) {
                    Ok(()) => (),
                    Err(err) => return Err(Error::Cuda(err)),
                };
                match inner_g.context.synchronize() {
                    Ok(()) => (),
                    Err(err) => return Err(Error::Cuda(err)),
                }
            },
            _ => return Err(Error::InvalidBackendArray),
        }
        Ok(())
    }

    fn store(&self, a: &BackendArray, elems: &[f32]) -> Result<()>
    {
        #[allow(unreachable_patterns)]
        match a {
            BackendArray::Cuda(a2) => {
                if a2.len != elems.len() {
                    return Err(Error::BackendArrayElemCount(a2.len, elems.len()));
                }
                let inner_g = mutex_lock(&self.inner)?;
                let mut a_slice_g = mutex_lock(&a2.slice)?;
                match inner_g.stream.memcpy_htod(elems, &mut (*a_slice_g)) {
                    Ok(()) => (),
                    Err(err) => return Err(Error::Cuda(err)),
                };
                match inner_g.context.synchronize() {
                    Ok(()) => (),
                    Err(err) => return Err(Error::Cuda(err)),
                }
            },
            _ => return Err(Error::InvalidBackendArray),
        }
        Ok(())
    }
    
    fn copy(&self, a: &BackendArray, b: &BackendArray) -> Result<()>
    {
        #[allow(unreachable_patterns)]
        match (a, b) {
            (BackendArray::Cuda(a2), BackendArray::Cuda(b2)) => {
                if Arc::ptr_eq(&a2.slice, &b2.slice) {
                    return Ok(());
                }
                if a2.len != b2.len {
                    return Err(Error::TwoBackendArrayElemCounts(a2.len, b2.len));
                }
                let inner_g = mutex_lock(&self.inner)?;
                let a_slice_g = mutex_lock(&a2.slice)?;
                let mut b_slice_g = mutex_lock(&b2.slice)?;
                match inner_g.stream.memcpy_dtod(&(*a_slice_g), &mut (*b_slice_g)) {
                    Ok(()) => (),
                    Err(err) => return Err(Error::Cuda(err)),
                }
                match inner_g.context.synchronize() {
                    Ok(()) => (),
                    Err(err) => return Err(Error::Cuda(err)),
                }
            },
            _ => return Err(Error::InvalidBackendArray),
        }
        Ok(())
    }

    fn transpose_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("transpose_a", a, b, n, m, 2, 2, true) }

    fn add_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("add_a_b", a, b, c, n, m, 2, 2, true) }

    fn add_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("add_at_b", a, b, c, n, m, 2, 2, true) }
    
    fn add_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("add_a_bt", a, b, c, n, m, 2, 2, true) }

    fn add_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("add_at_bt", a, b, c, n, m, 2, 2, true) }

    fn sub_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("sub_a_b", a, b, c, n, m, 2, 2, true) }

    fn sub_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("sub_at_b", a, b, c, n, m, 2, 2, true) }
    
    fn sub_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("sub_a_bt", a, b, c, n, m, 2, 2, true) }

    fn sub_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>    
    { self.check_and_launch_for_op("sub_at_bt", a, b, c, n, m, 2, 2, true) }
    
    fn mul_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>
    {
        if self.has_cublas {
            self.check_and_launch_cublas_for_mul(a, b, c, n, m, l, false, false)
        } else {
            if self.has_ptx {
                self.check_and_launch_for_ptx_mul("ptx_mul_a_b", a, b, c, n, m, l, 4, 4, true)
            } else {
                self.check_and_launch_for_mul("mul_a_b", a, b, c, n, m, l, 8, 4, true)
            }
        }
    }

    fn mul_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>
    {
        if self.has_cublas {
            self.check_and_launch_cublas_for_mul(a, b, c, n, m, l, true, false)
        } else {
            if self.has_ptx {
                self.check_and_launch_for_ptx_mul("ptx_mul_at_b", a, b, c, n, m, l, 4, 4, false)
            } else {
                self.check_and_launch_for_mul("mul_at_b", a, b, c, n, m, l, 8, 4, false)
            }
        }
    }

    fn mul_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>
    {
        if self.has_cublas {
            self.check_and_launch_cublas_for_mul(a, b, c, n, m, l, false, true)
        } else {
            if self.has_ptx {
                self.check_and_launch_for_ptx_mul("ptx_mul_a_bt", a, b, c, n, m, l, 4, 4, true) 
            } else {
                self.check_and_launch_for_mul("mul_a_bt", a, b, c, n, m, l, 8, 4, true) 
            }
        }
    }

    fn mul_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>
    {
        if self.has_cublas {
            self.check_and_launch_cublas_for_mul(a, b, c, n, m, l, true, true)
        } else {
            if self.has_ptx {
                self.check_and_launch_for_ptx_mul("ptx_mul_at_bt", a, b, c, n, m, l, 4, 4, false)
            } else {
                self.check_and_launch_for_mul("mul_at_bt", a, b, c, n, m, l, 8, 4, false)
            }
        }
    }

    fn mul_a_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("mul_a_b_for_elems", a, b, c, n, m, 2, 2, true) }

    fn mul_at_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("mul_at_b_for_elems", a, b, c, n, m, 2, 2, true) }
    
    fn mul_a_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("mul_a_bt_for_elems", a, b, c, n, m, 2, 2, true) }
    
    fn mul_at_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("mul_at_bt_for_elems", a, b, c, n, m, 2, 2, true) }

    fn div_a_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("div_a_b_for_elems", a, b, c, n, m, 2, 2, true) }

    fn div_at_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("div_at_b_for_elems", a, b, c, n, m, 2, 2, true) }
    
    fn div_a_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("div_a_bt_for_elems", a, b, c, n, m, 2, 2, true) }
    
    fn div_at_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("div_at_bt_for_elems", a, b, c, n, m, 2, 2, true) }

    fn add_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("add_a_b_for_scalar", a, b, c, n, m, 2, 2, true) }

    fn add_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("add_at_b_for_scalar", a, b, c, n, m, 2, 2, true) }

    fn sub_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("sub_a_b_for_scalar", a, b, c, n, m, 2, 2, true) }

    fn sub_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("sub_at_b_for_scalar", a, b, c, n, m, 2, 2, true) }

    fn rsub_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("rsub_a_b_for_scalar", a, b, c, n, m, 2, 2, true) }

    fn rsub_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("rsub_at_b_for_scalar", a, b, c, n, m, 2, 2, true) }
    
    fn mul_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("mul_a_b_for_scalar", a, b, c, n, m, 2, 2, true) }

    fn mul_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("mul_at_b_for_scalar", a, b, c, n, m, 2, 2, true) }

    fn div_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("div_a_b_for_scalar", a, b, c, n, m, 2, 2, true) }

    fn div_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("div_at_b_for_scalar", a, b, c, n, m, 2, 2, true) }

    fn rdiv_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("rdiv_a_b_for_scalar", a, b, c, n, m, 2, 2, true) }

    fn rdiv_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("rdiv_at_b_for_scalar", a, b, c, n, m, 2, 2, true) }

    fn sigmoid_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("sigmoid_a", a, b, n, m, 2, 2, true) }

    fn sigmoid_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("sigmoid_at", a, b, n, m, 2, 2, true) }

    fn tanh_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("tanh_a", a, b, n, m, 2, 2, true) }

    fn tanh_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("tanh_at", a, b, n, m, 2, 2, true) }

    fn swish_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("swish_a", a, b, n, m, 2, 2, true) }

    fn swish_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("swish_at", a, b, n, m, 2, 2, true) }

    fn softmax_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun_and_tiles("softmax_a", a, b, n, m, 2, 2, true) }

    fn softmax_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun_and_tiles("softmax_at", a, b, n, m, 2, 2, false) }

    fn sqrt_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("sqrt_a", a, b, n, m, 2, 2, true) }

    fn sqrt_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("sqrt_at", a, b, n, m, 2, 2, true) }

    fn repeat_col_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_repeat_col("repeat_col_a", a, b, n, m, 2, 2, true) }

    fn repeat_row_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_repeat_row("repeat_row_a", a, b, n, m, 2, 2, true) }

    fn abs_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("abs_a", a, b, n, m, 2, 2, true) }

    fn abs_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("abs_at", a, b, n, m, 2, 2, true) }

    fn pow_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("pow_a_b", a, b, c, n, m, 2, 2, true) }

    fn pow_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("pow_at_b", a, b, c, n, m, 2, 2, true) }
    
    fn pow_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("pow_a_bt", a, b, c, n, m, 2, 2, true) }
    
    fn pow_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("pow_at_bt", a, b, c, n, m, 2, 2, true) }

    fn pow_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("pow_a_b_for_scalar", a, b, c, n, m, 2, 2, true) }

    fn pow_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("pow_at_b_for_scalar", a, b, c, n, m, 2, 2, true) }

    fn rpow_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("rpow_a_b_for_scalar", a, b, c, n, m, 2, 2, true) }

    fn rpow_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("rpow_at_b_for_scalar", a, b, c, n, m, 2, 2, true) }

    fn exp_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("exp_a", a, b, n, m, 2, 2, true) }

    fn exp_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("exp_at", a, b, n, m, 2, 2, true) }

    fn ln_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("ln_a", a, b, n, m, 2, 2, true) }

    fn ln_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("ln_at", a, b, n, m, 2, 2, true) }

    fn log2_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("log2_a", a, b, n, m, 2, 2, true) }

    fn log2_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("log2_at", a, b, n, m, 2, 2, true) }

    fn log10_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("log10_a", a, b, n, m, 2, 2, true) }

    fn log10_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("log10_at", a, b, n, m, 2, 2, true) }

    fn sin_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("sin_a", a, b, n, m, 2, 2, true) }

    fn sin_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("sin_at", a, b, n, m, 2, 2, true) }

    fn cos_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("cos_a", a, b, n, m, 2, 2, true) }

    fn cos_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("cos_at", a, b, n, m, 2, 2, true) }

    fn tan_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("tan_a", a, b, n, m, 2, 2, true) }

    fn tan_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("tan_at", a, b, n, m, 2, 2, true) }

    fn asin_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("asin_a", a, b, n, m, 2, 2, true) }

    fn asin_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("asin_at", a, b, n, m, 2, 2, true) }

    fn acos_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("acos_a", a, b, n, m, 2, 2, true) }

    fn acos_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("acos_at", a, b, n, m, 2, 2, true) }

    fn atan_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("atan_a", a, b, n, m, 2, 2, true) }

    fn atan_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("atan_at", a, b, n, m, 2, 2, true) }

    fn atan2_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("atan2_a_b", a, b, c, n, m, 2, 2, true) }

    fn atan2_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("atan2_at_b", a, b, c, n, m, 2, 2, true) }
    
    fn atan2_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("atan2_a_bt", a, b, c, n, m, 2, 2, true) }
    
    fn atan2_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("atan2_at_bt", a, b, c, n, m, 2, 2, true) }

    fn atan2_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("atan2_a_b_for_scalar", a, b, c, n, m, 2, 2, true) }

    fn atan2_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("atan2_at_b_for_scalar", a, b, c, n, m, 2, 2, true) }

    fn ratan2_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("ratan2_a_b_for_scalar", a, b, c, n, m, 2, 2, true) }

    fn ratan2_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("ratan2_at_b_for_scalar", a, b, c, n, m, 2, 2, true) }

    fn sinh_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("sinh_a", a, b, n, m, 2, 2, true) }

    fn sinh_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("sinh_at", a, b, n, m, 2, 2, true) }

    fn cosh_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("cosh_a", a, b, n, m, 2, 2, true) }

    fn cosh_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("cosh_at", a, b, n, m, 2, 2, true) }

    fn asinh_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("asinh_a", a, b, n, m, 2, 2, true) }

    fn asinh_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("asinh_at", a, b, n, m, 2, 2, true) }

    fn acosh_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("acosh_a", a, b, n, m, 2, 2, true) }

    fn acosh_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("acosh_at", a, b, n, m, 2, 2, true) }

    fn atanh_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("atanh_a", a, b, n, m, 2, 2, true) }

    fn atanh_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("atanh_at", a, b, n, m, 2, 2, true) }

    fn signum_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("signum_a", a, b, n, m, 2, 2, true) }

    fn signum_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("signum_at", a, b, n, m, 2, 2, true) }

    fn ceil_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("ceil_a", a, b, n, m, 2, 2, true) }

    fn ceil_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("ceil_at", a, b, n, m, 2, 2, true) }

    fn floor_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("floor_a", a, b, n, m, 2, 2, true) }

    fn floor_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("floor_at", a, b, n, m, 2, 2, true) }

    fn round_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("round_a", a, b, n, m, 2, 2, true) }

    fn round_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("round_at", a, b, n, m, 2, 2, true) }

    fn trunc_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("trunc_a", a, b, n, m, 2, 2, true) }

    fn trunc_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("trunc_at", a, b, n, m, 2, 2, true) }

    fn max_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("max_a_b", a, b, c, n, m, 2, 2, true) }

    fn max_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("max_at_b", a, b, c, n, m, 2, 2, true) }
    
    fn max_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("max_a_bt", a, b, c, n, m, 2, 2, true) }
    
    fn max_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("max_at_bt", a, b, c, n, m, 2, 2, true) }

    fn max_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("max_a_b_for_scalar", a, b, c, n, m, 2, 2, true) }

    fn max_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("max_at_b_for_scalar", a, b, c, n, m, 2, 2, true) }

    fn min_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("min_a_b", a, b, c, n, m, 2, 2, true) }

    fn min_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("min_at_b", a, b, c, n, m, 2, 2, true) }
    
    fn min_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("min_a_bt", a, b, c, n, m, 2, 2, true) }
    
    fn min_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("min_at_bt", a, b, c, n, m, 2, 2, true) }

    fn min_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("min_a_b_for_scalar", a, b, c, n, m, 2, 2, true) }

    fn min_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("min_at_b_for_scalar", a, b, c, n, m, 2, 2, true) }
}

#[cfg(test)]
mod tests;
