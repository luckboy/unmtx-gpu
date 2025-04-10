//
// Copyright (c) 2025 Łukasz Szpakowski
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
use crate::Backend;
use crate::BackendArray;
use crate::Result;

pub use opencl3::context::Context;
pub use opencl3::device::Device;
pub use opencl3::error_codes::ClError;
pub use opencl3::platform::Platform;
pub use opencl3::platform::get_platforms;

pub struct ClBackendArray
{
}

pub struct ClBackend
{
}

impl ClBackend
{
    pub fn new() -> Result<ClBackend>
    { Ok(ClBackend {}) }

    
    pub fn new_with_context(context: Context) -> Result<ClBackend>
    { Ok(ClBackend {}) }
}

impl Backend for ClBackend
{
    fn alloc(&self, n: usize, m: usize) -> Result<BackendArray>
    { Ok(BackendArray::OpenCl(ClBackendArray {})) }

    fn alloc_with_elems(&self, elems: &[f32], n: usize, m: usize) -> Result<BackendArray>
    { Ok(BackendArray::OpenCl(ClBackendArray {})) }
    
    fn load(&self, a: &BackendArray, n: usize, m: usize) -> Result<Vec<f32>>
    { Ok(Vec::new()) }
    
    fn add_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { Ok(()) }

    fn add_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { Ok(()) }
    
    fn add_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { Ok(()) }

    fn add_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { Ok(()) }

    fn sub_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { Ok(()) }

    fn sub_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { Ok(()) }
    
    fn sub_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { Ok(()) }

    fn sub_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>    
    { Ok(()) }
    
    fn mul_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>
    { Ok(()) }

    fn mul_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>
    { Ok(()) }

    fn mul_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>
    { Ok(()) }

    fn mul_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>
    { Ok(()) }

    fn mul_a_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { Ok(()) }

    fn mul_at_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { Ok(()) }
    
    fn mul_a_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { Ok(()) }
    
    fn mul_at_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { Ok(()) }

    fn div_a_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { Ok(()) }

    fn div_at_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { Ok(()) }
    
    fn div_a_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { Ok(()) }
    
    fn div_at_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>
    { Ok(()) }
}
