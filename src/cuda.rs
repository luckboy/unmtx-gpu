//
// Copyright (c) 2025 Łukasz Szpakowski
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::Mutex;
use crate::Backend;
use crate::BackendArray;
use crate::Error;
use crate::Result;
use crate::mutex_lock;

pub use cudarc::driver::DriverError;

use cudarc::driver::CudaSlice;
use cudarc::driver::CudaDevice;
use cudarc::driver::CudaFunction;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchAsync;
use cudarc::driver::LaunchConfig;
use cudarc::nvrtc::CompileError;
use cudarc::nvrtc::compile_ptx;

const SOURCE: &'static str = include_str!("cuda.cu");

const KERNELS: &'static [&'static str] = &[
    "transpose_a",
    "add_a_b",
    "add_at_b",
    "add_a_bt",
    "add_at_bt",
    "sub_a_b",
    "sub_at_b",
    "sub_a_bt",
    "sub_at_bt",
    "mul_a_b",
    "mul_at_b",
    "mul_a_bt",
    "mul_at_bt",
    "mul_a_b_for_elems",
    "mul_at_b_for_elems",
    "mul_a_bt_for_elems",
    "mul_at_bt_for_elems",
    "div_a_b_for_elems",
    "div_at_b_for_elems",
    "div_a_bt_for_elems",
    "div_at_bt_for_elems",
    "add_a_b_for_scalar",
    "add_at_b_for_scalar",
    "sub_a_b_for_scalar",
    "sub_at_b_for_scalar",
    "rsub_a_b_for_scalar",
    "rsub_at_b_for_scalar",
    "mul_a_b_for_scalar",
    "mul_at_b_for_scalar",
    "div_a_b_for_scalar",
    "div_at_b_for_scalar",
    "rdiv_a_b_for_scalar",
    "rdiv_at_b_for_scalar",
    "sigmoid_a",
    "sigmoid_at",
    "tanh_a",
    "tanh_at",
    "softmax_a",
    "softmax_at"
];

#[derive(Debug)]
pub struct CudaBackendArray
{
    slice: Arc<Mutex<CudaSlice<f32>>>,
    len: usize,
}

struct CudaInnerBackend
{
    device: Arc<CudaDevice>,
}

pub struct CudaBackend
{
    inner: Mutex<CudaInnerBackend>,
}

fn preferred_launch_config(n: usize, m: usize) -> LaunchConfig
{
    if m == 1 {
        let n2 = ((n + 1023) / 1024) as u32;
        LaunchConfig {
            grid_dim: (n2, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        }
    } else if n == 1 {
        let m2 = ((m + 1023) / 1024) as u32;
        LaunchConfig {
            grid_dim: (1, m2, 1),
            block_dim: (1, 1024, 1),
            shared_mem_bytes: 0,
        }
    } else {
        let n2 = ((n + 31) / 32) as u32;
        let m2 = ((m + 31) / 32) as u32;
        LaunchConfig {
            grid_dim: (n2, m2, 1),
            block_dim: (32, 32, 1),
            shared_mem_bytes: 0,
        }
    }
}

impl CudaBackend
{
    pub fn new() -> Result<CudaBackend>
    { Self::new_with_ordinal(0) }

    pub fn new_with_ordinal(ordinal: usize) -> Result<CudaBackend>
    {
        let device = match CudaDevice::new(ordinal) {
            Ok(tmp_device) => tmp_device,
            Err(err) => return Err(Error::Cuda(err)),
        };
        let ptx = match compile_ptx(SOURCE) {
            Ok(tmp_ptx) => tmp_ptx,
            Err(CompileError::CompileError { log, .. }) => return Err(Error::Compilation(log.as_c_str().to_string_lossy().into_owned())),
            Err(err) => return Err(Error::Compilation(format!("{}", err))),
        };
        match device.load_ptx(ptx, "unmtx_gpu", KERNELS) {
            Ok(()) => (),
            Err(err) => return Err(Error::Cuda(err)),
        }
        Ok(CudaBackend { inner: Mutex::new(CudaInnerBackend { device, }), })
    }
    
    fn check_and_launch2<F, G>(&self, kernel_name: &str, a: &BackendArray, b: &BackendArray, f: F, g: G) -> Result<()>
        where F: FnOnce(&CudaBackendArray, &CudaBackendArray) -> Result<()>,
            G: FnOnce(&CudaInnerBackend, CudaFunction, *mut c_void, *mut c_void) -> Result<()>
    {
        #[allow(unreachable_patterns)]
        match (a, b) {
            (BackendArray::Cuda(a2), BackendArray::Cuda(b2)) => {
                f(a2, b2)?;
                let inner_g = mutex_lock(&self.inner)?;
                let kernel = match inner_g.device.get_func("unmtx_gpu", kernel_name) {
                    Some(tmp_kernel) => tmp_kernel,
                    None => return Err(Error::NoKernel(String::from(kernel_name))),
                };
                if !Arc::ptr_eq(&a2.slice, &b2.slice) {
                    let a_slice_g = mutex_lock(&a2.slice)?;
                    let mut b_slice_g = mutex_lock(&b2.slice)?;
                    g(&*inner_g, kernel, (&(*a_slice_g)).as_kernel_param(), (&mut (*b_slice_g)).as_kernel_param())?;
                } else {
                    let mut a_slice_g = mutex_lock(&a2.slice)?;
                    g(&*inner_g, kernel, (&mut (*a_slice_g)).as_kernel_param(), (&mut (*a_slice_g)).as_kernel_param())?;
                }
                match inner_g.device.synchronize() {
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
            G: FnOnce(&CudaInnerBackend, CudaFunction, *mut c_void, *mut c_void, *mut c_void) -> Result<()>
    {
        #[allow(unreachable_patterns)]
        match (a, b, c) {
            (BackendArray::Cuda(a2), BackendArray::Cuda(b2), BackendArray::Cuda(c2)) => {
                f(a2, b2, c2)?;
                let inner_g = mutex_lock(&self.inner)?;
                let kernel = match inner_g.device.get_func("unmtx_gpu", kernel_name) {
                    Some(tmp_kernel) => tmp_kernel,
                    None => return Err(Error::NoKernel(String::from(kernel_name))),
                };
                match (Arc::ptr_eq(&a2.slice, &b2.slice), Arc::ptr_eq(&a2.slice, &c2.slice), Arc::ptr_eq(&b2.slice, &c2.slice)) {
                    (false, false, false) => {
                        let a_slice_g = mutex_lock(&a2.slice)?;
                        let b_slice_g = mutex_lock(&b2.slice)?;
                        let mut c_slice_g = mutex_lock(&c2.slice)?;
                        g(&*inner_g, kernel, (&(*a_slice_g)).as_kernel_param(), (&(*b_slice_g)).as_kernel_param(), (&mut (*c_slice_g)).as_kernel_param())?
                    },
                    (true, false, false) => {
                        let a_slice_g = mutex_lock(&a2.slice)?;
                        let mut c_slice_g = mutex_lock(&c2.slice)?;
                        g(&*inner_g, kernel, (&(*a_slice_g)).as_kernel_param(), (&(*a_slice_g)).as_kernel_param(), (&mut (*c_slice_g)).as_kernel_param())?
                    },
                    (false, true, false) => {
                        let mut a_slice_g = mutex_lock(&a2.slice)?;
                        let b_slice_g = mutex_lock(&b2.slice)?;
                        g(&*inner_g, kernel, (&(*a_slice_g)).as_kernel_param(), (&(*b_slice_g)).as_kernel_param(), (&mut (*a_slice_g)).as_kernel_param())?
                    },
                    (false, false, true) => {
                        let a_slice_g = mutex_lock(&a2.slice)?;
                        let mut b_slice_g = mutex_lock(&b2.slice)?;
                        g(&*inner_g, kernel, (&(*a_slice_g)).as_kernel_param(), (&mut (*b_slice_g)).as_kernel_param(), (&mut (*b_slice_g)).as_kernel_param())?
                    },
                    _ => {
                        let mut a_slice_g = mutex_lock(&a2.slice)?;
                        g(&*inner_g, kernel, (&mut (*a_slice_g)).as_kernel_param(), (&mut (*a_slice_g)).as_kernel_param(), (&mut (*a_slice_g)).as_kernel_param())?
                    },
                }
                match inner_g.device.synchronize() {
                    Ok(()) => (),
                    Err(err) => return Err(Error::Cuda(err)),
                }
                Ok(())
            },
            _ => Err(Error::InvalidBackendArray),
        }
    }
    
    fn check_and_launch_for_fun(&self, kernel_name: &str, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    {
        self.check_and_launch2(kernel_name, a, b, |a2, b2| {
                if a2.len != n * m {
                    return Err(Error::BackendArrayElemCount(a2.len, n * m));
                }
                if b2.len != n * m {
                    return Err(Error::BackendArrayElemCount(b2.len, n * m));
                }
                Ok(())
        }, |_, kernel, a_param, b_param| {
                let config = preferred_launch_config(n, m);
                let mut params = vec![
                    a_param,
                    b_param,
                    n.as_kernel_param(),
                    m.as_kernel_param()
                ];
                unsafe {
                    match kernel.launch(config, &mut params) {
                        Ok(()) => Ok(()),
                        Err(err) => Err(Error::Cuda(err)),
                    }
                }
        })
    }

    fn check_and_launch_for_op(&self, kernel_name: &str, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    {
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
        }, |_, kernel, a_param, b_param, c_param| {
                let config = preferred_launch_config(n, m);
                let mut params = vec![
                    a_param,
                    b_param,
                    c_param,
                    n.as_kernel_param(),
                    m.as_kernel_param()
                ];
                unsafe {
                    match kernel.launch(config, &mut params) {
                        Ok(()) => Ok(()),
                        Err(err) => Err(Error::Cuda(err)),
                    }
                }
        })
    }

    fn check_and_launch_for_mul(&self, kernel_name: &str, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>
    {
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
        }, |_, kernel, a_param, b_param, c_param| {
                let config = preferred_launch_config(n, m);
                let mut params = vec![
                    a_param,
                    b_param,
                    c_param,
                    n.as_kernel_param(),
                    m.as_kernel_param(),
                    l.as_kernel_param()
                ];
                unsafe {
                    match kernel.launch(config, &mut params) {
                        Ok(()) => Ok(()),
                        Err(err) => Err(Error::Cuda(err)),
                    }
                }
        })
    }

    fn check_and_launch_for_scalar(&self, kernel_name: &str, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    {
        self.check_and_launch2(kernel_name, a, c, |a2, c2| {
                if a2.len != n * m  {
                    return Err(Error::BackendArrayElemCount(a2.len, n * m));
                }
                if c2.len != n * m {
                    return Err(Error::BackendArrayElemCount(c2.len, n * m));
                }
                Ok(())
        }, |_, kernel, a_param, c_param| {
                let config = preferred_launch_config(n, m);
                let mut params = vec![
                    a_param,
                    b.as_kernel_param(),
                    c_param,
                    n.as_kernel_param(),
                    m.as_kernel_param()
                ];
                unsafe {
                    match kernel.launch(config, &mut params) {
                        Ok(()) => Ok(()),
                        Err(err) => Err(Error::Cuda(err)),
                    }
                }
        })
    }
}

impl Backend for CudaBackend
{
    unsafe fn alloc(&self, n: usize) -> Result<BackendArray>
    {
        let inner_g = mutex_lock(&self.inner)?;
        let slice: CudaSlice<f32> = match inner_g.device.alloc(n) {
            Ok(tmp_slice) => tmp_slice,
            Err(err) => return Err(Error::Cuda(err)),
        };
        let cuda_array = CudaBackendArray { slice: Arc::new(Mutex::new(slice)), len: n, };
        Ok(BackendArray::Cuda(cuda_array))
    }

    fn alloc_and_store_zeros(&self, n: usize) -> Result<BackendArray>
    {
        let inner_g = mutex_lock(&self.inner)?;
        let slice: CudaSlice<f32> = match inner_g.device.alloc_zeros(n) {
            Ok(tmp_slice) => tmp_slice,
            Err(err) => return Err(Error::Cuda(err)),
        };
        let cuda_array = CudaBackendArray { slice: Arc::new(Mutex::new(slice)), len: n, };
        Ok(BackendArray::Cuda(cuda_array))
    }
    
    fn alloc_and_store(&self, elems: &[f32]) -> Result<BackendArray>
    {
        let inner_g = mutex_lock(&self.inner)?;
        let slice: CudaSlice<f32> = match inner_g.device.htod_sync_copy(elems) {
            Ok(tmp_slice) => tmp_slice,
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
                match inner_g.device.dtoh_sync_copy_into(&(*a_slice_g), elems) {
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
                match inner_g.device.htod_sync_copy_into(elems, &mut (*a_slice_g)) {
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
                match inner_g.device.dtod_copy(&(*a_slice_g), &mut (*b_slice_g)) {
                    Ok(()) => (),
                    Err(err) => return Err(Error::Cuda(err)),
                }
            },
            _ => return Err(Error::InvalidBackendArray),
        }
        Ok(())
    }

    fn transpose_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("transpose_a", a, b, n, m) }

    fn add_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("add_a_b", a, b, c, n, m) }

    fn add_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("add_at_b", a, b, c, n, m) }
    
    fn add_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("add_a_bt", a, b, c, n, m) }

    fn add_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("add_at_bt", a, b, c, n, m) }

    fn sub_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("sub_a_b", a, b, c, n, m) }

    fn sub_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("sub_at_b", a, b, c, n, m) }
    
    fn sub_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("sub_a_bt", a, b, c, n, m) }

    fn sub_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>    
    { self.check_and_launch_for_op("sub_at_bt", a, b, c, n, m) }
    
    fn mul_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>
    { self.check_and_launch_for_mul("mul_a_b", a, b, c, n, m, l) }

    fn mul_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>
    { self.check_and_launch_for_mul("mul_at_b", a, b, c, n, m, l) }

    fn mul_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>
    { self.check_and_launch_for_mul("mul_a_bt", a, b, c, n, m, l) }

    fn mul_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>
    { self.check_and_launch_for_mul("mul_at_bt", a, b, c, n, m, l) }

    fn mul_a_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("mul_a_b_for_elems", a, b, c, n, m) }

    fn mul_at_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("mul_at_b_for_elems", a, b, c, n, m) }
    
    fn mul_a_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("mul_a_bt_for_elems", a, b, c, n, m) }
    
    fn mul_at_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("mul_at_bt_for_elems", a, b, c, n, m) }

    fn div_a_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("div_a_b_for_elems", a, b, c, n, m) }

    fn div_at_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("div_at_b_for_elems", a, b, c, n, m) }
    
    fn div_a_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("div_a_bt_for_elems", a, b, c, n, m) }
    
    fn div_at_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_op("div_at_bt_for_elems", a, b, c, n, m) }

    fn add_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("add_a_b_for_scalar", a, b, c, n, m) }

    fn add_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("add_at_b_for_scalar", a, b, c, n, m) }

    fn sub_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("sub_a_b_for_scalar", a, b, c, n, m) }

    fn sub_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("sub_at_b_for_scalar", a, b, c, n, m) }

    fn rsub_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("rsub_a_b_for_scalar", a, b, c, n, m) }

    fn rsub_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("rsub_at_b_for_scalar", a, b, c, n, m) }
    
    fn mul_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("mul_a_b_for_scalar", a, b, c, n, m) }

    fn mul_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("mul_at_b_for_scalar", a, b, c, n, m) }

    fn div_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("div_a_b_for_scalar", a, b, c, n, m) }

    fn div_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("div_at_b_for_scalar", a, b, c, n, m) }

    fn rdiv_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("rdiv_a_b_for_scalar", a, b, c, n, m) }

    fn rdiv_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_scalar("rdiv_at_b_for_scalar", a, b, c, n, m) }

    fn sigmoid_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("sigmoid_a", a, b, n, m) }

    fn sigmoid_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("sigmoid_at", a, b, n, m) }

    fn tanh_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("tanh_a", a, b, n, m) }

    fn tanh_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("tanh_at", a, b, n, m) }

    fn softmax_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("softmax_a", a, b, n, m) }

    fn softmax_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_launch_for_fun("softmax_at", a, b, n, m) }
}

#[cfg(test)]
mod tests;
