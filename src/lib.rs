//
// Copyright (c) 2025 Łukasz Szpakowski
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
use std::sync::Arc;
use std::sync::Mutex;
use std::result;

#[cfg(feature = "opencl")]
pub mod opencl;

pub trait Backend
{
    fn alloc(&self, n: usize, m: usize) -> Result<BackendArray>;

    fn alloc_with_elems(&self, elems: &[f32], n: usize, m: usize) -> Result<BackendArray>;
    
    fn load(&self, a: &BackendArray, n: usize, m: usize) -> Result<Vec<f32>>;

    fn add_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    fn add_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;
    
    fn add_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    fn add_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    fn sub_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    fn sub_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;
    
    fn sub_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    fn sub_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;    
    
    fn mul_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>;

    fn mul_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>;

    fn mul_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>;

    fn mul_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>;

    fn mul_a_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    fn mul_at_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;
    
    fn mul_a_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;
    
    fn mul_at_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    fn div_a_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    fn div_at_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;
    
    fn div_a_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;
    
    fn div_at_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;
}

pub enum Error
{
    OpSize(usize, usize, usize, usize),
    MulSize(usize, usize, usize, usize, usize, usize),
    ResTransposition,
    Mutex,
    #[cfg(feature = "opencl")]
    OpenCl(opencl::ClError),
    Compilation(String),
}

pub type Result<T> = result::Result<T, Error>;

pub enum BackendArray
{
    #[cfg(feature = "opencl")]
    OpenCl(opencl::ClBackendArray),
}

static mut DEFAULT_BACKEND: Mutex<Option<Arc<dyn Backend>>> = Mutex::new(None);

#[derive(Clone)]
pub struct Matrix
{
    row_count: usize,
    col_count: usize,
    is_transposed: bool,
    array: Arc<BackendArray>,
}

impl Matrix
{
    pub fn transpose(&self) -> Matrix
    {
        Matrix {
            row_count: self.col_count,
            col_count: self.row_count,
            is_transposed: !self.is_transposed,
            array: self.array.clone(),
        }
    }
}

pub struct Frontend
{
    backend: Arc<dyn Backend>,
}

impl Frontend
{
    pub fn add(&self, a: &Matrix, b: &Matrix, c: &Matrix) -> Result<()>
    {
        if a.row_count == b.row_count && a.col_count == b.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, b.row_count, b.col_count)); 
        }
        if a.row_count == c.row_count && a.col_count == c.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, c.row_count, c.col_count)); 
        }
        if c.is_transposed {
            return Err(Error::ResTransposition);
        }
        match (a.is_transposed, a.is_transposed) {
            (false, false) => self.backend.add_a_b(&*a.array, &*b.array, &*c.array, a.row_count, a.col_count),
            (true, false) => self.backend.add_at_b(&*a.array, &*b.array, &*c.array, a.row_count, a.col_count),
            (false, true) => self.backend.add_a_bt(&*a.array, &*b.array, &*c.array, a.row_count, a.col_count),
            (true, true) => self.backend.add_at_bt(&*a.array, &*b.array, &*c.array, a.row_count, a.col_count),
        }
    }

    pub fn sub(&self, a: &Matrix, b: &Matrix, c: &Matrix) -> Result<()>
    {
        if a.row_count == b.row_count && a.col_count == b.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, b.row_count, b.col_count)); 
        }
        if a.row_count == c.row_count && a.col_count == c.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, c.row_count, c.col_count)); 
        }
        if c.is_transposed {
            return Err(Error::ResTransposition);
        }
        match (a.is_transposed, a.is_transposed) {
            (false, false) => self.backend.sub_a_b(&*a.array, &*b.array, &*c.array, a.row_count, a.col_count),
            (true, false) => self.backend.sub_at_b(&*a.array, &*b.array, &*c.array, a.row_count, a.col_count),
            (false, true) => self.backend.sub_a_bt(&*a.array, &*b.array, &*c.array, a.row_count, a.col_count),
            (true, true) => self.backend.sub_at_bt(&*a.array, &*b.array, &*c.array, a.row_count, a.col_count),
        }
    }

    pub fn mul(&self, a: &Matrix, b: &Matrix, c: &Matrix) -> Result<()>
    {
        if a.row_count == c.row_count {
            return Err(Error::MulSize(a.row_count, a.col_count, b.row_count, b.col_count, c.row_count, c.col_count)); 
        }
        if b.col_count == c.col_count {
            return Err(Error::MulSize(a.row_count, a.col_count, b.row_count, b.col_count, c.row_count, c.col_count)); 
        }
        if a.col_count == b.row_count {
            return Err(Error::MulSize(a.row_count, a.col_count, b.row_count, b.col_count, c.row_count, c.col_count)); 
        }
        if c.is_transposed {
            return Err(Error::ResTransposition);
        }
        match (a.is_transposed, a.is_transposed) {
            (false, false) => self.backend.mul_a_b(&*a.array, &*b.array, &*c.array, a.row_count, b.col_count, a.col_count),
            (true, false) => self.backend.mul_at_b(&*a.array, &*b.array, &*c.array, a.row_count, b.col_count, a.col_count),
            (false, true) => self.backend.mul_a_bt(&*a.array, &*b.array, &*c.array, a.row_count, b.col_count, a.col_count),
            (true, true) => self.backend.mul_at_bt(&*a.array, &*b.array, &*c.array, a.row_count, b.col_count, a.col_count),
        }
    }

    pub fn mul_elems(&self, a: &Matrix, b: &Matrix, c: &Matrix) -> Result<()>
    {
        if a.row_count == b.row_count && a.col_count == b.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, b.row_count, b.col_count)); 
        }
        if a.row_count == c.row_count && a.col_count == c.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, c.row_count, c.col_count)); 
        }
        if c.is_transposed {
            return Err(Error::ResTransposition);
        }
        match (a.is_transposed, a.is_transposed) {
            (false, false) => self.backend.mul_a_b_for_elems(&*a.array, &*b.array, &*c.array, a.row_count, a.col_count),
            (true, false) => self.backend.mul_at_b_for_elems(&*a.array, &*b.array, &*c.array, a.row_count, a.col_count),
            (false, true) => self.backend.mul_a_bt_for_elems(&*a.array, &*b.array, &*c.array, a.row_count, a.col_count),
            (true, true) => self.backend.mul_at_bt_for_elems(&*a.array, &*b.array, &*c.array, a.row_count, a.col_count),
        }
    }

    pub fn div_elems(&self, a: &Matrix, b: &Matrix, c: &Matrix) -> Result<()>
    {
        if a.row_count == b.row_count && a.col_count == b.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, b.row_count, b.col_count)); 
        }
        if a.row_count == c.row_count && a.col_count == c.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, c.row_count, c.col_count)); 
        }
        if c.is_transposed {
            return Err(Error::ResTransposition);
        }
        match (a.is_transposed, a.is_transposed) {
            (false, false) => self.backend.div_a_b_for_elems(&*a.array, &*b.array, &*c.array, a.row_count, a.col_count),
            (true, false) => self.backend.div_at_b_for_elems(&*a.array, &*b.array, &*c.array, a.row_count, a.col_count),
            (false, true) => self.backend.div_a_bt_for_elems(&*a.array, &*b.array, &*c.array, a.row_count, a.col_count),
            (true, true) => self.backend.div_at_bt_for_elems(&*a.array, &*b.array, &*c.array, a.row_count, a.col_count),
        }
    }
}
