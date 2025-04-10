//
// Copyright (c) 2025 Łukasz Szpakowski
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Sub;
use std::ops::SubAssign;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Div;
use std::ops::DivAssign;
use std::sync::Arc;
use std::sync::Mutex;
use std::result;

#[cfg(feature = "opencl")]
pub mod opencl;

pub trait Backend
{
    fn alloc(&self, n: usize, m: usize) -> Result<BackendArray>;

    fn alloc_and_store(&self, elems: &[f32], n: usize, m: usize) -> Result<BackendArray>;
    
    fn load(&self, a: &BackendArray, n: usize, m: usize) -> Result<Vec<f32>>;

    fn trans_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>;
    
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

    fn add_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    fn add_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    fn sub_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    fn sub_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    fn rsub_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    fn rsub_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;
    
    fn mul_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    fn mul_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    fn div_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    fn div_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    fn rdiv_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    fn rdiv_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;
}

#[derive(Debug)]
pub enum Error
{
    UninitializedDefaultBackend,
    OpSize(usize, usize, usize, usize),
    MulSize(usize, usize, usize, usize, usize, usize),
    TransSize(usize, usize, usize, usize),
    ArgTransposition,
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

pub fn get_default_backend() -> Result<Option<Arc<dyn Backend>>>
{
    unsafe {
        let default_backend_g = match DEFAULT_BACKEND.lock() {
            Ok(guard) => guard,
            Err(_) => return Err(Error::Mutex),
        };
        Ok(default_backend_g.clone())
    }
}

pub fn set_default_backend(backend: Arc<dyn Backend>) -> Result<()>
{
    unsafe {
        let mut default_backend_g = match DEFAULT_BACKEND.lock() {
            Ok(guard) => guard,
            Err(_) => return Err(Error::Mutex),
        };
        *default_backend_g = Some(backend);
    }
    Ok(())
}

pub fn set_default_backend_for_uninitialized(backend: Arc<dyn Backend>) -> Result<()>
{
    unsafe {
        let mut default_backend_g = match DEFAULT_BACKEND.lock() {
            Ok(guard) => guard,
            Err(_) => return Err(Error::Mutex),
        };
        match *default_backend_g {
            Some(_) => (),
            None => *default_backend_g = Some(backend),
        }
    }
    Ok(())
}

pub fn initialize_default_backend_for_uninitialized() -> Result<()>
{
    #[cfg(feature = "opencl")]
    set_default_backend_for_uninitialized(Arc::new(opencl::ClBackend::new()?))?;
    Ok(())
}

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

impl Add for Matrix
{
    type Output = Self;
    
    fn add(self, rhs: Self) -> Self::Output
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        let res = frontend.create_matrix(self.row_count, self.col_count).unwrap();
        frontend.add(&self, &rhs, &res).unwrap();
        res
    }
}

impl Add<&Matrix> for Matrix
{
    type Output = Self;
    
    fn add(self, rhs: &Matrix) -> Self::Output
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        let res = frontend.create_matrix(self.row_count, self.col_count).unwrap();
        frontend.add(&self, rhs, &res).unwrap();
        res
    }
}

impl Add<f32> for Matrix
{
    type Output = Self;
    
    fn add(self, rhs: f32) -> Self::Output
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        let res = frontend.create_matrix(self.row_count, self.col_count).unwrap();
        frontend.add_for_scalar(&self, rhs, &res).unwrap();
        res
    }
}

impl Add<&f32> for Matrix
{
    type Output = Self;
    
    fn add(self, rhs: &f32) -> Self::Output
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        let res = frontend.create_matrix(self.row_count, self.col_count).unwrap();
        frontend.add_for_scalar(&self, *rhs, &res).unwrap();
        res
    }
}

impl AddAssign for Matrix
{
    fn add_assign(&mut self, rhs: Self)
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        frontend.add(&self, &rhs, &self).unwrap();
    }
}

impl AddAssign<&Matrix> for Matrix
{
    fn add_assign(&mut self, rhs: &Self)
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        frontend.add(&self, rhs, &self).unwrap();
    }
}

impl AddAssign<f32> for Matrix
{
    fn add_assign(&mut self, rhs: f32)
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        frontend.add_for_scalar(&self, rhs, &self).unwrap();
    }
}

impl AddAssign<&f32> for Matrix
{
    fn add_assign(&mut self, rhs: &f32)
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        frontend.add_for_scalar(&self, *rhs, &self).unwrap();
    }
}

impl Sub for Matrix
{
    type Output = Self;
    
    fn sub(self, rhs: Self) -> Self::Output
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        let res = frontend.create_matrix(self.row_count, self.col_count).unwrap();
        frontend.sub(&self, &rhs, &res).unwrap();
        res
    }
}

impl Sub<&Matrix> for Matrix
{
    type Output = Self;
    
    fn sub(self, rhs: &Matrix) -> Self::Output
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        let res = frontend.create_matrix(self.row_count, self.col_count).unwrap();
        frontend.sub(&self, rhs, &res).unwrap();
        res
    }
}

impl Sub<f32> for Matrix
{
    type Output = Self;
    
    fn sub(self, rhs: f32) -> Self::Output
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        let res = frontend.create_matrix(self.row_count, self.col_count).unwrap();
        frontend.sub_for_scalar(&self, rhs, &res).unwrap();
        res
    }
}

impl Sub<&f32> for Matrix
{
    type Output = Self;
    
    fn sub(self, rhs: &f32) -> Self::Output
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        let res = frontend.create_matrix(self.row_count, self.col_count).unwrap();
        frontend.sub_for_scalar(&self, *rhs, &res).unwrap();
        res
    }
}

impl SubAssign for Matrix
{
    fn sub_assign(&mut self, rhs: Self)
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        frontend.sub(&self, &rhs, &self).unwrap();
    }
}

impl SubAssign<&Matrix> for Matrix
{
    fn sub_assign(&mut self, rhs: &Self)
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        frontend.sub(&self, rhs, &self).unwrap();
    }
}

impl SubAssign<f32> for Matrix
{
    fn sub_assign(&mut self, rhs: f32)
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        frontend.sub_for_scalar(&self, rhs, &self).unwrap();
    }
}

impl SubAssign<&f32> for Matrix
{
    fn sub_assign(&mut self, rhs: &f32)
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        frontend.sub_for_scalar(&self, *rhs, &self).unwrap();
    }
}

impl Mul for Matrix
{
    type Output = Self;
    
    fn mul(self, rhs: Self) -> Self::Output
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        let res = frontend.create_matrix(self.row_count, rhs.col_count).unwrap();
        frontend.mul(&self, &rhs, &res).unwrap();
        res
    }
}

impl Mul<&Matrix> for Matrix
{
    type Output = Self;
    
    fn mul(self, rhs: &Matrix) -> Self::Output
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        let res = frontend.create_matrix(self.row_count, rhs.col_count).unwrap();
        frontend.mul(&self, rhs, &res).unwrap();
        res
    }
}

impl Mul<f32> for Matrix
{
    type Output = Self;
    
    fn mul(self, rhs: f32) -> Self::Output
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        let res = frontend.create_matrix(self.row_count, self.col_count).unwrap();
        frontend.mul_for_scalar(&self, rhs, &res).unwrap();
        res
    }
}

impl Mul<&f32> for Matrix
{
    type Output = Self;
    
    fn mul(self, rhs: &f32) -> Self::Output
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        let res = frontend.create_matrix(self.row_count, self.col_count).unwrap();
        frontend.mul_for_scalar(&self, *rhs, &res).unwrap();
        res
    }
}

impl MulAssign for Matrix
{
    fn mul_assign(&mut self, rhs: Self)
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        let res = frontend.create_matrix(self.row_count, rhs.col_count).unwrap();
        frontend.mul(&self, &rhs, &res).unwrap();
        *self = res;
    }
}

impl MulAssign<&Matrix> for Matrix
{
    fn mul_assign(&mut self, rhs: &Self)
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        let res = frontend.create_matrix(self.row_count, rhs.col_count).unwrap();
        frontend.sub(&self, rhs, &res).unwrap();
        *self = res;
    }
}

impl MulAssign<f32> for Matrix
{
    fn mul_assign(&mut self, rhs: f32)
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        frontend.mul_for_scalar(&self, rhs, &self).unwrap();
    }
}

impl MulAssign<&f32> for Matrix
{
    fn mul_assign(&mut self, rhs: &f32)
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        frontend.mul_for_scalar(&self, *rhs, &self).unwrap();
    }
}

impl Div<f32> for Matrix
{
    type Output = Self;
    
    fn div(self, rhs: f32) -> Self::Output
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        let res = frontend.create_matrix(self.row_count, self.col_count).unwrap();
        frontend.div_for_scalar(&self, rhs, &res).unwrap();
        res
    }
}

impl Div<&f32> for Matrix
{
    type Output = Self;
    
    fn div(self, rhs: &f32) -> Self::Output
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        let res = frontend.create_matrix(self.row_count, self.col_count).unwrap();
        frontend.div_for_scalar(&self, *rhs, &res).unwrap();
        res
    }
}

impl DivAssign<f32> for Matrix
{
    fn div_assign(&mut self, rhs: f32)
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        frontend.div_for_scalar(&self, rhs, &self).unwrap();
    }
}

impl DivAssign<&f32> for Matrix
{
    fn div_assign(&mut self, rhs: &f32)
    {
        initialize_default_backend_for_uninitialized().unwrap();
        let frontend = Frontend::new().unwrap();
        frontend.div_for_scalar(&self, *rhs, &self).unwrap();
    }
}

pub struct Frontend
{
    backend: Arc<dyn Backend>,
}

impl Frontend
{
    pub fn new() -> Result<Frontend>
    {
        initialize_default_backend_for_uninitialized()?;
        match get_default_backend()? {
            Some(backend) => Ok(Frontend { backend, }),
            None => Err(Error::UninitializedDefaultBackend),
        }
    }

    pub fn new_with_backend(backend: Arc<dyn Backend>) -> Frontend
    { Frontend { backend, } }
    
    pub fn create_matrix(&self, row_count: usize, col_count: usize) -> Result<Matrix>
    {
        Ok(Matrix {
                row_count,
                col_count,
                is_transposed: false,
                array: Arc::new(self.backend.alloc(row_count, col_count)?),
        })
    }

    pub fn create_matrix_and_set_elems(&self, row_count: usize, col_count: usize, elems: &[f32]) -> Result<Matrix>
    {
        Ok(Matrix {
                row_count,
                col_count,
                is_transposed: false,
                array: Arc::new(self.backend.alloc_and_store(elems, row_count, col_count)?),
        })
    }
    
    pub fn elems_and_transpose_flag(&self, a: &Matrix) -> Result<(Vec<f32>, bool)>
    {
        if !a.is_transposed {
            Ok((self.backend.load(&*a.array, a.row_count, a.col_count)?, false))
        } else {
            Ok((self.backend.load(&*a.array, a.col_count, a.row_count)?, true))
        }
    }
    
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
        match (a.is_transposed, b.is_transposed) {
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
        match (a.is_transposed, b.is_transposed) {
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
        match (a.is_transposed, b.is_transposed) {
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
        match (a.is_transposed, b.is_transposed) {
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
        match (a.is_transposed, b.is_transposed) {
            (false, false) => self.backend.div_a_b_for_elems(&*a.array, &*b.array, &*c.array, a.row_count, a.col_count),
            (true, false) => self.backend.div_at_b_for_elems(&*a.array, &*b.array, &*c.array, a.row_count, a.col_count),
            (false, true) => self.backend.div_a_bt_for_elems(&*a.array, &*b.array, &*c.array, a.row_count, a.col_count),
            (true, true) => self.backend.div_at_bt_for_elems(&*a.array, &*b.array, &*c.array, a.row_count, a.col_count),
        }
    }

    pub fn add_for_scalar(&self, a: &Matrix, b: f32, c: &Matrix) -> Result<()>
    {
        if a.row_count == c.row_count && a.col_count == c.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, c.row_count, c.col_count)); 
        }
        if c.is_transposed {
            return Err(Error::ResTransposition);
        }
        if !a.is_transposed {
            self.backend.add_a_b_for_scalar(&*a.array, b, &*c.array, a.row_count, a.col_count)
        } else {
            self.backend.add_at_b_for_scalar(&*a.array, b, &*c.array, a.row_count, a.col_count)
        }
    }

    pub fn sub_for_scalar(&self, a: &Matrix, b: f32, c: &Matrix) -> Result<()>
    {
        if a.row_count == c.row_count && a.col_count == c.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, c.row_count, c.col_count)); 
        }
        if c.is_transposed {
            return Err(Error::ResTransposition);
        }
        if !a.is_transposed {
            self.backend.sub_a_b_for_scalar(&*a.array, b, &*c.array, a.row_count, a.col_count)
        } else {
            self.backend.sub_at_b_for_scalar(&*a.array, b, &*c.array, a.row_count, a.col_count)
        }
    }

    pub fn rsub_for_scalar(&self, a: &Matrix, b: f32, c: &Matrix) -> Result<()>
    {
        if a.row_count == c.row_count && a.col_count == c.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, c.row_count, c.col_count)); 
        }
        if c.is_transposed {
            return Err(Error::ResTransposition);
        }
        if !a.is_transposed {
            self.backend.rsub_a_b_for_scalar(&*a.array, b, &*c.array, a.row_count, a.col_count)
        } else {
            self.backend.rsub_at_b_for_scalar(&*a.array, b, &*c.array, a.row_count, a.col_count)
        }
    }
    
    pub fn mul_for_scalar(&self, a: &Matrix, b: f32, c: &Matrix) -> Result<()>
    {
        if a.row_count == c.row_count && a.col_count == c.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, c.row_count, c.col_count)); 
        }
        if c.is_transposed {
            return Err(Error::ResTransposition);
        }
        if !a.is_transposed {
            self.backend.mul_a_b_for_scalar(&*a.array, b, &*c.array, a.row_count, a.col_count)
        } else {
            self.backend.mul_at_b_for_scalar(&*a.array, b, &*c.array, a.row_count, a.col_count)
        }
    }
    
    pub fn div_for_scalar(&self, a: &Matrix, b: f32, c: &Matrix) -> Result<()>
    {
        if a.row_count == c.row_count && a.col_count == c.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, c.row_count, c.col_count)); 
        }
        if c.is_transposed {
            return Err(Error::ResTransposition);
        }
        if !a.is_transposed {
            self.backend.div_a_b_for_scalar(&*a.array, b, &*c.array, a.row_count, a.col_count)
        } else {
            self.backend.div_at_b_for_scalar(&*a.array, b, &*c.array, a.row_count, a.col_count)
        }
    }

    pub fn rdiv_for_scalar(&self, a: &Matrix, b: f32, c: &Matrix) -> Result<()>
    {
        if a.row_count == c.row_count && a.col_count == c.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, c.row_count, c.col_count)); 
        }
        if c.is_transposed {
            return Err(Error::ResTransposition);
        }
        if !a.is_transposed {
            self.backend.rdiv_a_b_for_scalar(&*a.array, b, &*c.array, a.row_count, a.col_count)
        } else {
            self.backend.rdiv_at_b_for_scalar(&*a.array, b, &*c.array, a.row_count, a.col_count)
        }
    }
    
    pub fn force_transpose(&self, a: &Matrix, b: &Matrix) -> Result<()>
    {
        if a.row_count == b.col_count && a.col_count == b.row_count {
            return Err(Error::TransSize(a.row_count, a.col_count, b.row_count, b.col_count)); 
        }
        if a.is_transposed {
            return Err(Error::ArgTransposition);
        }
        if b.is_transposed {
            return Err(Error::ResTransposition);
        }
        self.backend.trans_a(&*a.array, &*b.array, a.col_count, a.row_count)
    }
}
