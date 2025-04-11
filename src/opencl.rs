//
// Copyright (c) 2025 Łukasz Szpakowski
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
use std::mem::size_of;
use std::ptr::null_mut;
use std::sync::Arc;
use std::sync::Mutex;
use crate::Backend;
use crate::BackendArray;
use crate::Error;
use crate::Result;
use crate::mutex_lock;

pub use opencl3::context::Context;
pub use opencl3::device::Device;
pub use opencl3::device::CL_DEVICE_TYPE_ACCELERATOR;
pub use opencl3::device::CL_DEVICE_TYPE_ALL;
pub use opencl3::device::CL_DEVICE_TYPE_CPU;
pub use opencl3::device::CL_DEVICE_TYPE_CUSTOM;
pub use opencl3::device::CL_DEVICE_TYPE_DEFAULT;
pub use opencl3::device::CL_DEVICE_TYPE_GPU;
pub use opencl3::device::cl_device_id;
pub use opencl3::error_codes::ClError;
pub use opencl3::platform::Platform;
pub use opencl3::platform::get_platforms;

use cl3::info_type::InfoType;
use opencl3::command_queue::CommandQueue;
use opencl3::device::CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE;
use opencl3::device::get_device_info;
use opencl3::event::Event;
use opencl3::kernel::ExecuteKernel;
use opencl3::kernel::Kernel;
use opencl3::memory::Buffer;
use opencl3::memory::ClMem;
use opencl3::memory::cl_mem;
use opencl3::memory::CL_MEM_READ_WRITE;
use opencl3::program::Program;
use opencl3::types::CL_TRUE;

const SOURCE: &'static str = include_str!("opencl.cl");

#[derive(Debug)]
pub struct ClBackendArray
{
    buffer: Arc<Mutex<Buffer<f32>>>,
    len: usize,
}

struct ClInnerBackend
{
    context: Context,
    command_queue: CommandQueue,
    program: Program,
    group_size: usize,
}

pub struct ClBackend
{
    inner: Mutex<ClInnerBackend>,
}

impl ClBackend
{
    pub fn new() -> Result<ClBackend>
    {
        let platforms = match get_platforms() {
            Ok(tmp_platforms) => tmp_platforms,
            Err(err) => return Err(Error::OpenCl(err)),
        };
        if platforms.is_empty() {
            return Err(Error::NoPlatform);
        }
        let device_ids = match platforms[0].get_devices(CL_DEVICE_TYPE_DEFAULT) {
            Ok(tmp_device_ids) => tmp_device_ids,
            Err(err) => return Err(Error::OpenCl(err)),
        };
        if device_ids.is_empty() {
            return Err(Error::NoDevice);
        }
        let device = Device::new(device_ids[0]);
        let context = match Context::from_device(&device) {
            Ok(tmp_context) => tmp_context,
            Err(err) => return Err(Error::OpenCl(err)),
        };
        Self::new_with_context(context)
    }
    
    pub fn new_with_context(context: Context) -> Result<ClBackend>
    {
        let command_queue = match CommandQueue::create_default_with_properties(&context, 0, 0) {
            Ok(tmp_command_queue) => tmp_command_queue,
            Err(err) => return Err(Error::OpenCl(err)),
        };
        let program = match Program::create_and_build_from_source(&context, SOURCE, "") {
            Ok(tmp_program) => tmp_program,
            Err(msg) => return Err(Error::Compilation(msg)),
        };
        let group_size = match get_device_info(context.default_device(), CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE) {
            Ok(InfoType::Size(tmp_group_size)) => tmp_group_size,
            _ => return Err(Error::InvalidDeviceInfoType),
        };
        let inner = ClInnerBackend {
            context,
            command_queue,
            program,
            group_size,
        };
        Ok(ClBackend { inner: Mutex::new(inner), })
    }
    
    fn check_and_enqueue_nd_range2<F, G>(&self, kernel_name: &str, a: &BackendArray, b: &BackendArray, f: F, g: G) -> Result<()>
        where F: FnOnce(&ClBackendArray, &ClBackendArray) -> Result<()>,
            G: FnOnce(&ClInnerBackend, &Kernel, cl_mem, cl_mem) -> Result<Event>
    {
        #[allow(unreachable_patterns)]
        match (a, b) {
            (BackendArray::OpenCl(a2), BackendArray::OpenCl(b2)) => {
                f(a2, b2)?;
                let inner_g = mutex_lock(&self.inner)?;
                let kernel = match Kernel::create(&inner_g.program, kernel_name) {
                    Ok(tmp_kernel) => tmp_kernel,
                    Err(err) => return Err(Error::OpenCl(err)),
                };
                let event = if !Arc::ptr_eq(&a2.buffer, &b2.buffer) {
                    let a_buffer_g = mutex_lock(&a2.buffer)?;
                    let mut b_buffer_g = mutex_lock(&b2.buffer)?;
                    g(&*inner_g, &kernel, a_buffer_g.get(), b_buffer_g.get_mut())?
                } else {
                    let mut a_buffer_g = mutex_lock(&a2.buffer)?;
                    g(&*inner_g, &kernel, a_buffer_g.get(), a_buffer_g.get_mut())?
                };
                match event.wait() {
                    Ok(()) => (),
                    Err(err) => return Err(Error::OpenCl(err)),
                }
                Ok(())
            },
            _ => Err(Error::InvalidBackendArray),
        }
    }

    fn check_and_enqueue_nd_range3<F, G>(&self, kernel_name: &str, a: &BackendArray, b: &BackendArray, c: &BackendArray, f: F, g: G) -> Result<()>
        where F: FnOnce(&ClBackendArray, &ClBackendArray, &ClBackendArray) -> Result<()>,
            G: FnOnce(&ClInnerBackend, &Kernel, cl_mem, cl_mem, cl_mem) -> Result<Event>
    {
        #[allow(unreachable_patterns)]
        match (a, b, c) {
            (BackendArray::OpenCl(a2), BackendArray::OpenCl(b2), BackendArray::OpenCl(c2)) => {
                f(a2, b2, c2)?;
                let inner_g = mutex_lock(&self.inner)?;
                let kernel = match Kernel::create(&inner_g.program, kernel_name) {
                    Ok(tmp_kernel) => tmp_kernel,
                    Err(err) => return Err(Error::OpenCl(err)),
                };
                let event = match (Arc::ptr_eq(&a2.buffer, &b2.buffer), Arc::ptr_eq(&a2.buffer, &c2.buffer), Arc::ptr_eq(&b2.buffer, &c2.buffer)) {
                    (false, false, false) => {
                        let a_buffer_g = mutex_lock(&a2.buffer)?;
                        let b_buffer_g = mutex_lock(&b2.buffer)?;
                        let mut c_buffer_g = mutex_lock(&c2.buffer)?;
                        g(&*inner_g, &kernel, a_buffer_g.get(), b_buffer_g.get(), c_buffer_g.get_mut())?
                    },
                    (true, false, false) => {
                        let a_buffer_g = mutex_lock(&a2.buffer)?;
                        let mut c_buffer_g = mutex_lock(&c2.buffer)?;
                        g(&*inner_g, &kernel, a_buffer_g.get(), a_buffer_g.get(), c_buffer_g.get_mut())?
                    },
                    (false, true, false) => {
                        let mut a_buffer_g = mutex_lock(&a2.buffer)?;
                        let b_buffer_g = mutex_lock(&b2.buffer)?;
                        g(&*inner_g, &kernel, a_buffer_g.get(), b_buffer_g.get(), a_buffer_g.get_mut())?
                    },
                    (false, false, true) => {
                        let a_buffer_g = mutex_lock(&a2.buffer)?;
                        let mut b_buffer_g = mutex_lock(&b2.buffer)?;
                        g(&*inner_g, &kernel, a_buffer_g.get(), b_buffer_g.get(), b_buffer_g.get_mut())?
                    },
                    _ => {
                        let mut a_buffer_g = mutex_lock(&a2.buffer)?;
                        g(&*inner_g, &kernel, a_buffer_g.get(), a_buffer_g.get(), a_buffer_g.get_mut())?
                    },
                };
                match event.wait() {
                    Ok(()) => (),
                    Err(err) => return Err(Error::OpenCl(err)),
                }
                Ok(())
            },
            _ => Err(Error::InvalidBackendArray),
        }
    }
    
    fn check_and_enqueue_nd_range_for_fun(&self, kernel_name: &str, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    {
        self.check_and_enqueue_nd_range2(kernel_name, a, b, |a2, b2| {
                if a2.len != n * m {
                    return Err(Error::BackendArrayElemCount(a2.len, n * m));
                }
                if b2.len != n * m {
                    return Err(Error::BackendArrayElemCount(b2.len, n * m));
                }
                Ok(())
        }, |inner, kernel, a_mem, b_mem| {
                let n2 = n as u64;
                let m2 = m as u64;
                let n3 = (n + inner.group_size - 1) % inner.group_size;
                let m3 = (m + inner.group_size - 1) % inner.group_size;
                unsafe {
                    let res = ExecuteKernel::new(kernel)
                    .set_arg(&a_mem)
                    .set_arg(&b_mem)
                    .set_arg(&n2)
                    .set_arg(&m2)
                    .set_local_work_sizes(&[inner.group_size, inner.group_size])
                    .set_global_work_sizes(&[n3, m3])
                    .enqueue_nd_range(&inner.command_queue);
                    match res {
                        Ok(event) => Ok(event),
                        Err(err) => Err(Error::OpenCl(err)),
                    }
                }
        })
    }

    fn check_and_enqueue_nd_range_for_op(&self, kernel_name: &str, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    {
        self.check_and_enqueue_nd_range3(kernel_name, a, b, c, |a2, b2, c2| {
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
        }, |inner, kernel, a_mem, b_mem, c_mem| {
                let n2 = n as u64;
                let m2 = m as u64;
                let n3 = (n + inner.group_size - 1) % inner.group_size;
                let m3 = (m + inner.group_size - 1) % inner.group_size;
                unsafe {
                    let res = ExecuteKernel::new(kernel)
                    .set_arg(&a_mem)
                    .set_arg(&b_mem)
                    .set_arg(&c_mem)
                    .set_arg(&n2)
                    .set_arg(&m2)
                    .set_local_work_sizes(&[inner.group_size, inner.group_size])
                    .set_global_work_sizes(&[n3, m3])
                    .enqueue_nd_range(&inner.command_queue);
                    match res {
                        Ok(event) => Ok(event),
                        Err(err) => Err(Error::OpenCl(err)),
                    }
                }
        })
    }

    fn check_and_enqueue_nd_range_for_mul(&self, kernel_name: &str, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>
    {
        self.check_and_enqueue_nd_range3(kernel_name, a, b, c, |a2, b2, c2| {
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
        }, |inner, kernel, a_mem, b_mem, c_mem| {
                let n2 = n as u64;
                let m2 = m as u64;
                let l2 = l as u64;
                let n3 = (n + inner.group_size - 1) % inner.group_size;
                let m3 = (m + inner.group_size - 1) % inner.group_size;
                unsafe {
                    let res = ExecuteKernel::new(kernel)
                    .set_arg(&a_mem)
                    .set_arg(&b_mem)
                    .set_arg(&c_mem)
                    .set_arg(&n2)
                    .set_arg(&m2)
                    .set_arg(&l2)
                    .set_local_work_sizes(&[inner.group_size, inner.group_size])
                    .set_global_work_sizes(&[n3, m3])
                    .enqueue_nd_range(&inner.command_queue);
                    match res {
                        Ok(event) => Ok(event),
                        Err(err) => Err(Error::OpenCl(err)),
                    }
                }
        })
    }

    fn check_and_enqueue_nd_range_for_scalar(&self, kernel_name: &str, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    {
        self.check_and_enqueue_nd_range2(kernel_name, a, c, |a2, c2| {
                if a2.len != n * m  {
                    return Err(Error::BackendArrayElemCount(a2.len, n * m));
                }
                if c2.len != n * m {
                    return Err(Error::BackendArrayElemCount(c2.len, n * m));
                }
                Ok(())
        }, |inner, kernel, a_mem, c_mem| {
                let n2 = n as u64;
                let m2 = m as u64;
                let n3 = (n + inner.group_size - 1) % inner.group_size;
                let m3 = (m + inner.group_size - 1) % inner.group_size;
                unsafe {
                    let res = ExecuteKernel::new(kernel)
                    .set_arg(&a_mem)
                    .set_arg(&b)
                    .set_arg(&c_mem)
                    .set_arg(&n2)
                    .set_arg(&m2)
                    .set_local_work_sizes(&[inner.group_size, inner.group_size])
                    .set_global_work_sizes(&[n3, m3])
                    .enqueue_nd_range(&inner.command_queue);
                    match res {
                        Ok(event) => Ok(event),
                        Err(err) => Err(Error::OpenCl(err)),
                    }
                }
        })
    }
}

impl Backend for ClBackend
{
    unsafe fn alloc(&self, n: usize) -> Result<BackendArray>
    {
        let inner_g = mutex_lock(&self.inner)?;
        let buffer: Buffer<f32> = match Buffer::create(&inner_g.context, CL_MEM_READ_WRITE, n, null_mut()) {
            Ok(tmp_buffer) => tmp_buffer,
            Err(err) => return Err(Error::OpenCl(err)),
        };
        let cl_array = ClBackendArray { buffer: Arc::new(Mutex::new(buffer)), len: n, };
        Ok(BackendArray::OpenCl(cl_array))
    }

    fn alloc_and_store_zeros(&self, n: usize) -> Result<BackendArray>
    {
        let inner_g = mutex_lock(&self.inner)?;
        let mut buffer: Buffer<f32> = match unsafe { Buffer::create(&inner_g.context, CL_MEM_READ_WRITE, n, null_mut()) } {
            Ok(tmp_buffer) => tmp_buffer,
            Err(err) => return Err(Error::OpenCl(err)),
        };
        let event = match unsafe { inner_g.command_queue.enqueue_fill_buffer(&mut buffer, &[0.0f32], 0, n * size_of::<f32>(), &[]) } {
            Ok(tmp_event) => tmp_event,
            Err(err) => return Err(Error::OpenCl(err)),
        };
        match event.wait() {
            Ok(()) => (),
            Err(err) => return Err(Error::OpenCl(err)),
        }
        let cl_array = ClBackendArray { buffer: Arc::new(Mutex::new(buffer)), len: n, };
        Ok(BackendArray::OpenCl(cl_array))
    }
    
    fn alloc_and_store(&self, elems: &[f32]) -> Result<BackendArray>
    {
        let inner_g = mutex_lock(&self.inner)?;
        let mut buffer: Buffer<f32> = match unsafe { Buffer::create(&inner_g.context, CL_MEM_READ_WRITE, elems.len(), null_mut()) } {
            Ok(tmp_buffer) => tmp_buffer,
            Err(err) => return Err(Error::OpenCl(err)),
        };
        let event = match unsafe { inner_g.command_queue.enqueue_write_buffer(&mut buffer, CL_TRUE, 0, elems, &[]) } {
            Ok(tmp_event) => tmp_event,
            Err(err) => return Err(Error::OpenCl(err)),
        };
        match event.wait() {
            Ok(()) => (),
            Err(err) => return Err(Error::OpenCl(err)),
        }
        let cl_array = ClBackendArray { buffer: Arc::new(Mutex::new(buffer)), len: elems.len(), };
        Ok(BackendArray::OpenCl(cl_array))
    }
    
    fn load(&self, a: &BackendArray, elems: &mut [f32]) -> Result<()>
    {
        #[allow(unreachable_patterns)]
        match a {
            BackendArray::OpenCl(a2) => {
                if a2.len != elems.len() {
                    return Err(Error::BackendArrayElemCount(a2.len, elems.len()));
                }
                let inner_g = mutex_lock(&self.inner)?;
                let a_buffer_g = mutex_lock(&a2.buffer)?;
                let event = match unsafe { inner_g.command_queue.enqueue_read_buffer(&*a_buffer_g, CL_TRUE, 0, elems, &[]) } {
                    Ok(tmp_event) => tmp_event,
                    Err(err) => return Err(Error::OpenCl(err)),
                };
                match event.wait() {
                    Ok(()) => (),
                    Err(err) => return Err(Error::OpenCl(err)),
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
            BackendArray::OpenCl(a2) => {
                if a2.len != elems.len() {
                    return Err(Error::BackendArrayElemCount(a2.len, elems.len()));
                }
                let inner_g = mutex_lock(&self.inner)?;
                let mut a_buffer_g = mutex_lock(&a2.buffer)?;
                let event = match unsafe { inner_g.command_queue.enqueue_write_buffer(&mut *a_buffer_g, CL_TRUE, 0, elems, &[]) } {
                    Ok(tmp_event) => tmp_event,
                    Err(err) => return Err(Error::OpenCl(err)),
                };
                match event.wait() {
                    Ok(()) => (),
                    Err(err) => return Err(Error::OpenCl(err)),
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
            (BackendArray::OpenCl(a2), BackendArray::OpenCl(b2)) => {
                if Arc::ptr_eq(&a2.buffer, &b2.buffer) {
                    return Ok(());
                }
                if a2.len != b2.len {
                    return Err(Error::TwoBackendArrayElemCounts(a2.len, b2.len));
                }
                let inner_g = mutex_lock(&self.inner)?;
                let a_buffer_g = mutex_lock(&a2.buffer)?;
                let mut b_buffer_g = mutex_lock(&b2.buffer)?;
                let event = match unsafe { inner_g.command_queue.enqueue_copy_buffer(&*a_buffer_g, &mut *b_buffer_g, 0, 0, a2.len * size_of::<f32>(), &[]) } {
                    Ok(tmp_event) => tmp_event,
                    Err(err) => return Err(Error::OpenCl(err)),
                };
                match event.wait() {
                    Ok(()) => (),
                    Err(err) => return Err(Error::OpenCl(err)),
                }
            },
            _ => return Err(Error::InvalidBackendArray),
        }
        Ok(())
    }

    fn transpose_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_fun("transpose_a", a, b, n, m) }

    fn add_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_op("add_a_b", a, b, c, n, m) }

    fn add_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_op("add_at_b", a, b, c, n, m) }
    
    fn add_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_op("add_a_bt", a, b, c, n, m) }

    fn add_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_op("add_at_bt", a, b, c, n, m) }

    fn sub_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_op("sub_a_b", a, b, c, n, m) }

    fn sub_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_op("sub_at_b", a, b, c, n, m) }
    
    fn sub_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_op("sub_a_bt", a, b, c, n, m) }

    fn sub_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>    
    { self.check_and_enqueue_nd_range_for_op("sub_at_bt", a, b, c, n, m) }
    
    fn mul_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_mul("mul_a_b", a, b, c, n, m, l) }

    fn mul_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_mul("mul_at_b", a, b, c, n, m, l) }

    fn mul_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_mul("mul_a_bt", a, b, c, n, m, l) }

    fn mul_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_mul("mul_at_bt", a, b, c, n, m, l) }

    fn mul_a_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_op("mul_a_b_for_elems", a, b, c, n, m) }

    fn mul_at_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_op("mul_at_b_for_elems", a, b, c, n, m) }
    
    fn mul_a_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_op("mul_a_bt_for_elems", a, b, c, n, m) }
    
    fn mul_at_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_op("mul_at_bt_for_elems", a, b, c, n, m) }

    fn div_a_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_op("div_a_b_for_elems", a, b, c, n, m) }

    fn div_at_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_op("div_at_b_for_elems", a, b, c, n, m) }
    
    fn div_a_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_op("div_a_bt_for_elems", a, b, c, n, m) }
    
    fn div_at_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_op("mul_at_b_for_elems", a, b, c, n, m) }

    fn add_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_scalar("add_a_b_for_scalar", a, b, c, n, m) }

    fn add_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_scalar("add_at_b_for_scalar", a, b, c, n, m) }

    fn sub_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_scalar("sub_a_b_for_scalar", a, b, c, n, m) }

    fn sub_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_scalar("sub_at_b_for_scalar", a, b, c, n, m) }

    fn rsub_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_scalar("rsub_a_b_for_scalar", a, b, c, n, m) }

    fn rsub_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_scalar("rsub_at_b_for_scalar", a, b, c, n, m) }
    
    fn mul_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_scalar("mul_a_b_for_scalar", a, b, c, n, m) }

    fn mul_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_scalar("mul_a_b_for_scalar", a, b, c, n, m) }

    fn div_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_scalar("div_a_b_for_scalar", a, b, c, n, m) }

    fn div_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_scalar("div_at_b_for_scalar", a, b, c, n, m) }

    fn rdiv_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_scalar("rdiv_a_b_for_scalar", a, b, c, n, m) }

    fn rdiv_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_scalar("rdiv_at_b_for_scalar", a, b, c, n, m) }

    fn sigmoid_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_fun("sigmoid_a", a, b, n, m) }

    fn sigmoid_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_fun("sigmoid_at", a, b, n, m) }

    fn tanh_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_fun("tanh_a", a, b, n, m) }

    fn tanh_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_fun("tanh_at", a, b, n, m) }

    fn softmax_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_fun("softmax_a", a, b, n, m) }

    fn softmax_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>
    { self.check_and_enqueue_nd_range_for_fun("softmax_at", a, b, n, m) }
}

#[cfg(test)]
mod tests;
