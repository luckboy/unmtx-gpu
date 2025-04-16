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
use std::error;
use std::fmt;
use std::result;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;

#[cfg(feature = "opencl")]
pub mod opencl;
#[cfg(feature = "cuda")]
pub mod cuda;

/// A backend trait.
pub trait Backend
{
    /// Returns the backend name.
    fn name(&self) -> &'static str;
    
    /// Allocates a backend array.
    unsafe fn alloc(&self, n: usize) -> Result<BackendArray>;

    /// Allocates a backend array and stores zeros to the backend array.
    fn alloc_and_store_zeros(&self, n: usize) -> Result<BackendArray>;

    /// Allocates a backend array and stores elements to the backend array.
    fn alloc_and_store(&self, elems: &[f32]) -> Result<BackendArray>;
    
    /// Loads elements from the backenc array.
    fn load(&self, a: &BackendArray, elems: &mut [f32]) -> Result<()>;

    /// Stores elements in the backend array.
    fn store(&self, a: &BackendArray, elems: &[f32]) -> Result<()>;

    /// Copies the `a` backend array to the `b` backend array.
    fn copy(&self, a: &BackendArray, b: &BackendArray) -> Result<()>;

    /// Transposes the `a` matrix and then the result is in the `b` matrix (B = A^T).
    fn transpose_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Adds the `a` matrix onto the `b` matrix and then the result is in the `c` matrix
    /// (C = A + B).
    fn add_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Adds the transposed `a` matrix onto the `b` matrix and then the result is in the `c` matrix
    /// (C = A^T + B).
    fn add_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;
    
    /// Adds the `a` matrix onto the transposed `b` matrix and then the result is in the `c` matrix
    /// (C = A + B^T).
    fn add_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Adds the transposed `a` matrix onto the transposed `b` matrix and then the result is in the
    /// `c` matrix (C = A^T + B^T).
    fn add_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Subtracts the `b` matrix from the `a` matrix and then the result is in the `c` matrix
    /// (C = A - B).
    fn sub_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Subtracts the `b` matrix from the transposed `a` matrix and then the result is in the `c`
    /// matrix (C = A^T - B).
    fn sub_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;
    
    /// Subtracts the transposed `b` matrix from the `a` matrix and then the result is in the `c`
    /// matrix (C = A - B^T).
    fn sub_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Subtracts the transposed `b` matrix from the transposed `a` matrix and then the result is
    /// in the `c` matrix (C = A^T - B^T).
    fn sub_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;    
    
    /// Multiplies the `a` matrix by the `b` matrix and then the result is in the `c` matrix
    /// (C = A * B).
    fn mul_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>;

    /// Multiplies the transposed `a` matrix by the `b` matrix and then the result is in the `c`
    /// matrix (C = A^T * B).
    fn mul_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>;

    /// Multiplies the `a` matrix by the transposed `b` matrix and then the result is in the `c`
    /// matrix (C = A * B^T).
    fn mul_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>;

    /// Multiplies the transposed `a` matrix by the transposed `b` matrix and then the result is in
    /// the `c` matrix (C = A^T * B^T).
    fn mul_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>;

    /// Multiplies the `a` matrix elements by the `b` matrix elements and then the result is in the
    /// `c` matrix (Cij = Aij * Bij).
    fn mul_a_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Multiplies the transposed `a` matrix elements by the `b` matrix elements and saves the
    /// result to the `c` matrix (Cij = (A^T)ij * Bij).
    fn mul_at_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Multiplies the `a` matrix elements by the transposed `b` matrix elements and then the
    /// result is in the `c` matrix (Cij = Aij * (B^T)ij).
    fn mul_a_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;
    
    /// Multiplies the transposed `a` matrix elements by the transposed `b` matrix elements and
    /// then the result is in the `c` matrix (Cij = (A^T)ij * (B^T)ij).
    fn mul_at_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Divides the `a` matrix elements by the `b` matrix elements and then the result is in the
    /// `c` matrix (Cij = Aij / Bij).
    fn div_a_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Divides the transposed `a` matrix elements by the `b` matrix elements and then the result
    /// is in the `c` matrix (Cij = (A^T)ij / Bij).
    fn div_at_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;
    
    /// Divides the transposed `a` matrix elements by the `b` matrix elements and then the result
    /// is in the `c` matrix (Cij = Aij / (B^T)ij).
    fn div_a_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;
    
    /// Divides the transposed `a` matrix elements by the transposed `b` matrix elements and then
    /// the result is in the `c` matrix (Cij = (A^T)ij / (B^T)ij).
    fn div_at_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Adds the `a` matrix onto the `b` scalar and then the result is in the `c` matrix (C = A + b).
    fn add_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Adds the transposed `a` matrix onto the `b` scalar and then the result is in the `c` matrix
    /// (C = A^T + b).
    fn add_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Subtracts the `b` scalar from the `a` matrix and then the result is in the `c` matrix.
    /// (C = A - b).
    fn sub_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Subtracts the `b` scalar from the transposed `a` matrix and then the result is in the `c`
    /// matrix (C = A^T - b).
    fn sub_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Subtracts the `a` matrix from the `b` scalar  and then the result is in the `c` matrix
    /// (C = b - A).
    fn rsub_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Subtracts the transposed `a` matrix from the `b` scalar  and then the result is in the `c`
    /// matrix (C = b - A^T).
    fn rsub_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;
    
    /// Multiplies the `a` matrix by the `b` scalar and then the result is in the `c` matrix
    /// (C = A * b).
    fn mul_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Multiplies the transposed `a` matrix by the `b` scalar and then the result is in the `c`
    /// matrix (C = A^T * b).
    fn mul_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Divides the `a` matrix by the `b` scalar and then the result is in the `c` matrix
    /// (C = A / b).
    fn div_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Divides the transposed `a` matrix by the `b` scalar and then the result is in the `c`
    /// matrix (C = A^T / b).
    fn div_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Divides the `b` scalar by the `a` matrix and then the result is in the `c` matrix
    /// (C = b / A).
    fn rdiv_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Divides the `b` scalar by the transposed `a` matrix and then the result is in the `c`
    /// matrix (C = b / A^T).
    fn rdiv_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Calculates sigmoid function for the `a` matrix adn the result is the `b` matrix
    /// (B = sigmoid(A)).
    fn sigmoid_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Calculates sigmoid function for the transposed `a` matrix and then the result is in the `b`
    /// matrix (B = sigmoid(A^T)).
    fn sigmoid_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Calculates hyperbolic tangent function for the `a` matrix and then the result is in `b`
    /// matrix (B = tanh(A)).
    fn tanh_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Calculates hyperbolic tangent function for the transposed `a` matrix and then the result is
    /// in the `b` matrix (B = tanh(A^T)).
    fn tanh_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Calculates softmax function for the `a` matrix and then the result is in the `b` matrix
    /// (B = softmax(A)).
    fn softmax_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Calculates softmax function for the transposed `a` matrix and then the result is in the `b`
    /// matrix (B = softmax(A^T)).
    fn softmax_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>;
}

/// An error enumeration.
#[derive(Debug)]
pub enum Error
{
    UninitializedDefaultBackend,
    OpSize(usize, usize, usize, usize),
    MulSize(usize, usize, usize, usize, usize, usize),
    TransposeSize(usize, usize, usize, usize),
    ArgTransposition,
    ResTransposition,
    MatrixElemCount(usize, usize),
    Mutex,
    #[cfg(feature = "opencl")]
    OpenCl(opencl::ClError),
    #[cfg(feature = "cuda")]
    Cuda(cuda::DriverError),
    Compilation(String),
    NoPlatform,
    NoDevice,
    NoKernel(String),
    InvalidDeviceInfoType,
    BackendArrayElemCount(usize, usize),
    TwoBackendArrayElemCounts(usize, usize),
    InvalidBackendArray,
}

impl error::Error for Error
{}

impl fmt::Display for Error
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    {
        match self {
            Error::UninitializedDefaultBackend => write!(f, "uninitialized default backend"),
            Error::OpSize(n1, m1, n2, m2) => write!(f, "mismatched sizes of matrices ({}x{}, {}x{})", n1, m1, n2, m2),
            Error::MulSize(n1, m1, n2, m2, n3, m3) => write!(f, "mismatched sizes of matrices for multiplication ({}x{}, {}x{}, {}x{})", n1, m1, n2, m2, n3, m3),
            Error::TransposeSize(n1, m1, n2, m2) => write!(f, "mismatched sizes of matrices for transposition ({}x{}, {}x{})", n1, m1, n2, m2),
            Error::ArgTransposition => write!(f, "argument matrix is transposed"),
            Error::ResTransposition => write!(f, "result matrix is transposed"),
            Error::MatrixElemCount(n1, n2) => write!(f, "number of matrix elements isn't equal to number of elements ({}, {})", n1, n2),
            Error::Mutex => write!(f, "can't lock mutex"),
            #[cfg(feature = "opencl")]
            Error::OpenCl(err) => write!(f, "OpenCL error: {}", err),
            #[cfg(feature = "cuda")]
            Error::Cuda(err) => write!(f, "CUDA error: {}", err),
            Error::Compilation(msg) => write!(f, "{}", msg),
            Error::NoPlatform => write!(f, "no platform"),
            Error::NoDevice => write!(f, "no device"),
            Error::NoKernel(name) => write!(f, "no kernel {}", name),
            Error::InvalidDeviceInfoType => write!(f, "no device info type"),
            Error::BackendArrayElemCount(n1, n2) => write!(f, "number of backend array elements isn't equal to number of elements ({}, {})", n1, n2),
            Error::TwoBackendArrayElemCounts(n1, n2) => write!(f, "two numbers of elements of backend arrays aren't equal ({}, {})", n1, n2),
            Error::InvalidBackendArray => write!(f, "invalid backend array"),
        }
    }
}

pub type Result<T> = result::Result<T, Error>;

/// An enumeration of backend array.
///
/// This enumeration contains a reference to a matrix array on device.
#[derive(Debug)]
pub enum BackendArray
{
    /// A backend array for OpenCL.
    #[cfg(feature = "opencl")]
    OpenCl(opencl::ClBackendArray),
    /// A backend array for CUDA.
    #[cfg(feature = "cuda")]
    Cuda(cuda::CudaBackendArray),
}

static mut DEFAULT_BACKEND: Mutex<Option<Arc<dyn Backend>>> = Mutex::new(None);

fn mutex_lock<T>(mutex: &Mutex<T>) -> Result<MutexGuard<'_, T>>
{
    match mutex.lock() {
        Ok(guard) => Ok(guard),
        Err(_) => return Err(Error::Mutex),
    }
}

pub fn get_default_backend() -> Result<Option<Arc<dyn Backend>>>
{
    unsafe {
        let default_backend_g = mutex_lock(&DEFAULT_BACKEND)?;
        Ok(default_backend_g.clone())
    }
}

pub fn set_default_backend(backend: Arc<dyn Backend>) -> Result<()>
{
    unsafe {
        let mut default_backend_g = mutex_lock(&DEFAULT_BACKEND)?;
        *default_backend_g = Some(backend);
    }
    Ok(())
}

pub fn unset_default_backend() -> Result<()>
{
    unsafe {
        let mut default_backend_g = mutex_lock(&DEFAULT_BACKEND)?;
        *default_backend_g = None;
    }
    Ok(())
}

pub fn set_default_backend_for_uninitialized<F>(f: F) -> Result<()>
    where F: FnOnce() -> Result<Arc<dyn Backend>>
{
    unsafe {
        let mut default_backend_g = mutex_lock(&DEFAULT_BACKEND)?;
        match *default_backend_g {
            Some(_) => (),
            None => *default_backend_g = Some(f()?),
        }
    }
    Ok(())
}

pub fn initialize_default_backend_for_uninitialized() -> Result<()>
{
    #[cfg(feature = "opencl")]
    set_default_backend_for_uninitialized(|| Ok(Arc::new(opencl::ClBackend::new()?)))?;
    #[cfg(all(not(feature = "opencl"), feature = "cuda"))]
    set_default_backend_for_uninitialized(|| Ok(Arc::new(cuda::CudaBackend::new()?)))?;
    Ok(())
}

pub fn finalize_default_backend() -> Result<()>
{ unset_default_backend() }

/// Creates a matrix from the arguments.
///
/// # Examples
///
/// ```
/// # use unmtx_gpu::*;
/// let a = matrix![
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0]
/// ];
/// assert_eq!(2, a.row_count());
/// assert_eq!(3, a.col_count());
/// assert_eq!(false, a.is_transposed());
/// assert_eq!(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], a.elems());
/// ```
#[macro_export]
macro_rules! matrix {
    ($([$($elem:expr),* $(,)*]),* $(,)*) => {
        Matrix::new_with_elem_vecs(vec![$(vec![$($elem),*]),*].as_slice())
    };
}

/// A matrix structure.
#[derive(Clone, Debug)]
pub struct Matrix
{
    row_count: usize,
    col_count: usize,
    is_transposed: bool,
    array: Arc<BackendArray>,
}

impl Matrix
{
    /// Creates a matrix with the number of rows and the number of columns.
    pub fn new(row_count: usize, col_count: usize) -> Self
    {
        let frontend = Frontend::new().unwrap();
        frontend.create_matrix_and_set_zeros(row_count, col_count).unwrap()
    }

    /// Creates a matrix with the number of rows, the number of columns, and the elements.
    pub fn new_with_elems(row_count: usize, col_count: usize, elems: &[f32]) -> Self
    {
        let frontend = Frontend::new().unwrap();
        frontend.create_matrix_and_set_elems(row_count, col_count, elems).unwrap()
    }

    /// Creates a matrix with the vector of rows.
    pub fn new_with_elem_vecs(elem_vecs: &[Vec<f32>]) -> Self
    {
        let frontend = Frontend::new().unwrap();
        let col_count = match elem_vecs.first() {
            Some(elems) => elems.len(),
            None => 0,
        };
        for row in elem_vecs {
            assert_eq!(col_count, row.len());
        }
        let row_count = elem_vecs.len();
        let elems: Vec<f32> = elem_vecs.iter().flatten().map(|e| *e).collect();
        frontend.create_matrix_and_set_elems(row_count, col_count, elems.as_slice()).unwrap()
    }

    /// Returns the number of matrix rows.
    pub fn row_count(&self) -> usize
    { self.row_count }
    
    /// Returns the number of matrix columns.
    pub fn col_count(&self) -> usize
    { self.col_count }

    /// Returns `true` if the matrix is transposed, otherwise `false`.
    ///
    /// This method indeed returns the transpose flag of matrix that is changed by
    /// [`transpose`](Self::transpose).
    pub fn is_transposed(&self) -> bool
    { self.is_transposed }
    
    /// Returns the matrix elements.
    pub fn elems(&self) -> Vec<f32>
    {
        let frontend = Frontend::new().unwrap();
        frontend.elems_and_transpose_flag(self).unwrap().0
    }
    
    /// Creates a matrix copy. 
    ///
    /// This method indeed copies the matrix array to a new matrix array.
    pub fn copy(&self) -> Self
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.copy(self, &res).unwrap();
        res
    }
    
    /// Transposes the matrix (A^T).
    ///
    /// This method doesn't indeed transpose the matrix but changes the transpose flag and
    /// exchanges the number of matrix rows with the number of matrix columns.
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0]
    /// ];
    /// let b = a.transpose();
    /// assert_eq!(3, b.row_count());
    /// assert_eq!(2, b.col_count());
    /// assert_eq!(true, b.is_transposed());
    /// assert_eq!(a.elems(), b.elems());
    /// let c = b.transpose();
    /// assert_eq!(2, c.row_count());
    /// assert_eq!(3, c.col_count());
    /// assert_eq!(false, c.is_transposed());
    /// assert_eq!(a.elems(), c.elems());
    /// ```
    pub fn transpose(&self) -> Self
    {
        Matrix {
            row_count: self.col_count,
            col_count: self.row_count,
            is_transposed: !self.is_transposed,
            array: self.array.clone(),
        }
    }
    
    /// Indeed transposes the matrix (A^T).
    ///
    /// This method indeed transposes the matrix without changing the transpose flag.
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0]
    /// ];
    /// let b = a.really_transpose();
    /// assert_eq!(3, b.row_count());
    /// assert_eq!(2, b.col_count());
    /// assert_eq!(false, b.is_transposed());
    /// assert_eq!(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], b.elems());
    /// ```
    pub fn really_transpose(&self) -> Self
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.col_count, self.row_count) }.unwrap();
        frontend.really_transpose(self, &res).unwrap();
        res
    }
    
    /// Multiplies the matrix elements by the `b` matrix elements (Aij * Bij).
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ];
    /// let b = matrix![
    ///     [5.0, 6.0],
    ///     [7.0, 8.0]
    /// ];
    /// let c = a.mul_elems(&b);
    /// assert_eq!(vec![1.0 * 5.0, 2.0 * 6.0, 3.0 * 7.0, 4.0 * 8.0], c.elems());
    /// ```
    pub fn mul_elems(&self, b: &Self) -> Self
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.mul_elems(self, b, &res).unwrap();
        res
    }

    /// Divides the matrix elements by the `b` matrix elements (Aij / Bij).
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ];
    /// let b = matrix![
    ///     [5.0, 6.0],
    ///     [7.0, 8.0]
    /// ];
    /// let c = a.div_elems(&b);
    /// let elems = c.elems();
    /// assert!((1.0 / 5.0 - elems[0]).abs() < 0.001);
    /// assert!((2.0 / 6.0 - elems[1]).abs() < 0.001);
    /// assert!((3.0 / 7.0 - elems[2]).abs() < 0.001);
    /// assert!((4.0 / 8.0 - elems[3]).abs() < 0.001);
    /// ```
    pub fn div_elems(&self, b: &Self) -> Self
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.div_elems(self, b, &res).unwrap();
        res
    }

    /// Subtracts the matrix from the scalar (b - A).
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ];
    /// let b = a.rsub(10.5);
    /// assert_eq!(vec![10.5 - 1.0, 10.5 - 2.0, 10.5 - 3.0, 10.5 - 4.0], b.elems());
    /// ```
    pub fn rsub(&self, b: f32) -> Self
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.rsub_for_scalar(self, b, &res).unwrap();
        res
    }

    /// Divides the scalar by the matrix (b / A).
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ];
    /// let b = a.rdiv(10.5);
    /// let elems = b.elems();
    /// assert!((10.5 / 1.0 - elems[0]).abs() < 0.001);
    /// assert!((10.5 / 2.0 - elems[1]).abs() < 0.001);
    /// assert!((10.5 / 3.0 - elems[2]).abs() < 0.001);
    /// assert!((10.5 / 4.0 - elems[3]).abs() < 0.001);
    /// ```
    pub fn rdiv(&self, b: f32) -> Self
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.rdiv_for_scalar(self, b, &res).unwrap();
        res
    }

    /// Calculates sigmoid function for the matrix (sigmoid(A)).
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ];
    /// let b = a.sigmoid();
    /// let elems = b.elems();
    /// assert!((1.0 / (1.0 + (-1.0f32).exp()) - elems[0]).abs() < 0.001);
    /// assert!((1.0 / (1.0 + (-2.0f32).exp()) - elems[1]).abs() < 0.001);
    /// assert!((1.0 / (1.0 + (-3.0f32).exp()) - elems[2]).abs() < 0.001);
    /// assert!((1.0 / (1.0 + (-4.0f32).exp()) - elems[3]).abs() < 0.001);
    /// ```
    pub fn sigmoid(&self) -> Self
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.sigmoid(self, &res).unwrap();
        res
    }

    /// Calculates hiperbolic tangent function for the matrix (tanh(A)).
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ];
    /// let b = a.tanh();
    /// let elems = b.elems();
    /// assert!((1.0f32.tanh() - elems[0]).abs() < 0.001);
    /// assert!((2.0f32.tanh() - elems[1]).abs() < 0.001);
    /// assert!((3.0f32.tanh() - elems[2]).abs() < 0.001);
    /// assert!((4.0f32.tanh() - elems[3]).abs() < 0.001);
    /// ```
    pub fn tanh(&self) -> Self
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.tanh(self, &res).unwrap();
        res
    }

    /// Calculates softmax function for the matrix (softmax(A)).
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ];
    /// let b = a.softmax();
    /// let elems = b.elems();
    /// let sum1 = 1.0f32.exp() + 3.0f32.exp();
    /// let sum2 = 2.0f32.exp() + 4.0f32.exp();
    /// assert!((1.0f32.exp() / sum1 - elems[0]).abs() < 0.001);
    /// assert!((2.0f32.exp() / sum2 - elems[1]).abs() < 0.001);
    /// assert!((3.0f32.exp() / sum1 - elems[2]).abs() < 0.001);
    /// assert!((4.0f32.exp() / sum2 - elems[3]).abs() < 0.001);
    /// ```
    pub fn softmax(&self) -> Self
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.softmax(self, &res).unwrap();
        res
    }
}

impl Add for Matrix
{
    type Output = Self;
    
    fn add(self, rhs: Self) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.add(&self, &rhs, &res).unwrap();
        res
    }
}

impl Add<&Matrix> for Matrix
{
    type Output = Self;
    
    fn add(self, rhs: &Matrix) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.add(&self, rhs, &res).unwrap();
        res
    }
}

impl Add<f32> for Matrix
{
    type Output = Self;
    
    fn add(self, rhs: f32) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.add_for_scalar(&self, rhs, &res).unwrap();
        res
    }
}

impl Add<&f32> for Matrix
{
    type Output = Self;
    
    fn add(self, rhs: &f32) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.add_for_scalar(&self, *rhs, &res).unwrap();
        res
    }
}

impl Add<Matrix> for &Matrix
{
    type Output = Matrix;
    
    fn add(self, rhs: Matrix) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.add(self, &rhs, &res).unwrap();
        res
    }
}

impl Add<&Matrix> for &Matrix
{
    type Output = Matrix;
    
    fn add(self, rhs: &Matrix) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.add(self, rhs, &res).unwrap();
        res
    }
}

impl Add<f32> for &Matrix
{
    type Output = Matrix;
    
    fn add(self, rhs: f32) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.add_for_scalar(self, rhs, &res).unwrap();
        res
    }
}

impl Add<&f32> for &Matrix
{
    type Output = Matrix;
    
    fn add(self, rhs: &f32) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.add_for_scalar(self, *rhs, &res).unwrap();
        res
    }
}

impl AddAssign for Matrix
{
    fn add_assign(&mut self, rhs: Self)
    {
        let frontend = Frontend::new().unwrap();
        frontend.add(self, &rhs, &self).unwrap();
    }
}

impl AddAssign<&Matrix> for Matrix
{
    fn add_assign(&mut self, rhs: &Self)
    {
        let frontend = Frontend::new().unwrap();
        frontend.add(&self, rhs, &self).unwrap();
    }
}

impl AddAssign<f32> for Matrix
{
    fn add_assign(&mut self, rhs: f32)
    {
        let frontend = Frontend::new().unwrap();
        frontend.add_for_scalar(&self, rhs, &self).unwrap();
    }
}

impl AddAssign<&f32> for Matrix
{
    fn add_assign(&mut self, rhs: &f32)
    {
        let frontend = Frontend::new().unwrap();
        frontend.add_for_scalar(&self, *rhs, &self).unwrap();
    }
}

impl Sub for Matrix
{
    type Output = Self;
    
    fn sub(self, rhs: Self) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.sub(&self, &rhs, &res).unwrap();
        res
    }
}

impl Sub<&Matrix> for Matrix
{
    type Output = Self;
    
    fn sub(self, rhs: &Matrix) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.sub(&self, rhs, &res).unwrap();
        res
    }
}

impl Sub<f32> for Matrix
{
    type Output = Self;
    
    fn sub(self, rhs: f32) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.sub_for_scalar(&self, rhs, &res).unwrap();
        res
    }
}

impl Sub<&f32> for Matrix
{
    type Output = Self;
    
    fn sub(self, rhs: &f32) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.sub_for_scalar(&self, *rhs, &res).unwrap();
        res
    }
}

impl Sub<Matrix> for &Matrix
{
    type Output = Matrix;
    
    fn sub(self, rhs: Matrix) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.sub(self, &rhs, &res).unwrap();
        res
    }
}

impl Sub<&Matrix> for &Matrix
{
    type Output = Matrix;
    
    fn sub(self, rhs: &Matrix) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.sub(self, rhs, &res).unwrap();
        res
    }
}

impl Sub<f32> for &Matrix
{
    type Output = Matrix;
    
    fn sub(self, rhs: f32) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.sub_for_scalar(self, rhs, &res).unwrap();
        res
    }
}

impl Sub<&f32> for &Matrix
{
    type Output = Matrix;
    
    fn sub(self, rhs: &f32) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.sub_for_scalar(self, *rhs, &res).unwrap();
        res
    }
}

impl SubAssign for Matrix
{
    fn sub_assign(&mut self, rhs: Self)
    {
        let frontend = Frontend::new().unwrap();
        frontend.sub(&self, &rhs, &self).unwrap();
    }
}

impl SubAssign<&Matrix> for Matrix
{
    fn sub_assign(&mut self, rhs: &Self)
    {
        let frontend = Frontend::new().unwrap();
        frontend.sub(&self, rhs, &self).unwrap();
    }
}

impl SubAssign<f32> for Matrix
{
    fn sub_assign(&mut self, rhs: f32)
    {
        let frontend = Frontend::new().unwrap();
        frontend.sub_for_scalar(&self, rhs, &self).unwrap();
    }
}

impl SubAssign<&f32> for Matrix
{
    fn sub_assign(&mut self, rhs: &f32)
    {
        let frontend = Frontend::new().unwrap();
        frontend.sub_for_scalar(&self, *rhs, &self).unwrap();
    }
}

impl Mul for Matrix
{
    type Output = Self;
    
    fn mul(self, rhs: Self) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, rhs.col_count) }.unwrap();
        frontend.mul(&self, &rhs, &res).unwrap();
        res
    }
}

impl Mul<&Matrix> for Matrix
{
    type Output = Self;
    
    fn mul(self, rhs: &Matrix) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, rhs.col_count) }.unwrap();
        frontend.mul(&self, rhs, &res).unwrap();
        res
    }
}

impl Mul<f32> for Matrix
{
    type Output = Self;
    
    fn mul(self, rhs: f32) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.mul_for_scalar(&self, rhs, &res).unwrap();
        res
    }
}

impl Mul<&f32> for Matrix
{
    type Output = Self;
    
    fn mul(self, rhs: &f32) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.mul_for_scalar(&self, *rhs, &res).unwrap();
        res
    }
}

impl Mul<Matrix> for &Matrix
{
    type Output = Matrix;
    
    fn mul(self, rhs: Matrix) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, rhs.col_count) }.unwrap();
        frontend.mul(self, &rhs, &res).unwrap();
        res
    }
}

impl Mul<&Matrix> for &Matrix
{
    type Output = Matrix;
    
    fn mul(self, rhs: &Matrix) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, rhs.col_count) }.unwrap();
        frontend.mul(self, rhs, &res).unwrap();
        res
    }
}

impl Mul<f32> for &Matrix
{
    type Output = Matrix;
    
    fn mul(self, rhs: f32) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.mul_for_scalar(self, rhs, &res).unwrap();
        res
    }
}

impl Mul<&f32> for &Matrix
{
    type Output = Matrix;
    
    fn mul(self, rhs: &f32) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.mul_for_scalar(self, *rhs, &res).unwrap();
        res
    }
}

impl MulAssign for Matrix
{
    fn mul_assign(&mut self, rhs: Self)
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, rhs.col_count) }.unwrap();
        frontend.mul(&self, &rhs, &res).unwrap();
        *self = res;
    }
}

impl MulAssign<&Matrix> for Matrix
{
    fn mul_assign(&mut self, rhs: &Self)
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, rhs.col_count) }.unwrap();
        frontend.mul(&self, rhs, &res).unwrap();
        *self = res;
    }
}

impl MulAssign<f32> for Matrix
{
    fn mul_assign(&mut self, rhs: f32)
    {
        let frontend = Frontend::new().unwrap();
        frontend.mul_for_scalar(&self, rhs, &self).unwrap();
    }
}

impl MulAssign<&f32> for Matrix
{
    fn mul_assign(&mut self, rhs: &f32)
    {
        let frontend = Frontend::new().unwrap();
        frontend.mul_for_scalar(&self, *rhs, &self).unwrap();
    }
}

impl Div<f32> for Matrix
{
    type Output = Self;
    
    fn div(self, rhs: f32) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.div_for_scalar(&self, rhs, &res).unwrap();
        res
    }
}

impl Div<&f32> for Matrix
{
    type Output = Self;
    
    fn div(self, rhs: &f32) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.div_for_scalar(&self, *rhs, &res).unwrap();
        res
    }
}

impl Div<f32> for &Matrix
{
    type Output = Matrix;
    
    fn div(self, rhs: f32) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.div_for_scalar(self, rhs, &res).unwrap();
        res
    }
}

impl Div<&f32> for &Matrix
{
    type Output = Matrix;
    
    fn div(self, rhs: &f32) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.div_for_scalar(self, *rhs, &res).unwrap();
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

/// A frontend structure.
pub struct Frontend
{
    backend: Arc<dyn Backend>,
}

impl Frontend
{
    /// Creates a frontend with a default backend.
    ///
    /// This method also automatically initializes a default backend if the default backend is
    /// uninitialized.
    pub fn new() -> Result<Frontend>
    {
        initialize_default_backend_for_uninitialized()?;
        match get_default_backend()? {
            Some(backend) => Ok(Frontend { backend, }),
            None => Err(Error::UninitializedDefaultBackend),
        }
    }

    /// Creates a frotend with the backend.
    pub fn new_with_backend(backend: Arc<dyn Backend>) -> Frontend
    { Frontend { backend, } }
    
    /// Returns the backend.
    pub fn backend(&self) -> Arc<dyn Backend>
    { self.backend.clone() }
    
    /// Creates a matrix with unset elements.
    pub unsafe fn create_matrix(&self, row_count: usize, col_count: usize) -> Result<Matrix>
    {
        Ok(Matrix {
                row_count,
                col_count,
                is_transposed: false,
                array: Arc::new(self.backend.alloc(row_count * col_count)?),
        })
    }

    /// Creates a matrix and sets the matrix elements on zeros.
    pub fn create_matrix_and_set_zeros(&self, row_count: usize, col_count: usize) -> Result<Matrix>
    {
        Ok(Matrix {
                row_count,
                col_count,
                is_transposed: false,
                array: Arc::new(self.backend.alloc_and_store_zeros(row_count * col_count)?),
        })
    }

    /// Creates a matrix and sets the matrix elements.
    pub fn create_matrix_and_set_elems(&self, row_count: usize, col_count: usize, elems: &[f32]) -> Result<Matrix>
    {
        if row_count * col_count != elems.len() {
            return Err(Error::MatrixElemCount(row_count * col_count, elems.len())); 
        }
        Ok(Matrix {
                row_count,
                col_count,
                is_transposed: false,
                array: Arc::new(self.backend.alloc_and_store(elems)?),
        })
    }

    /// Sets the matrix elements.
    pub fn set_elems(&self, a: &Matrix, elems: &[f32]) -> Result<()>
    {
        if a.row_count() * a.col_count() != elems.len() {
            return Err(Error::MatrixElemCount(a.row_count() * a.col_count(), elems.len())); 
        }
        self.backend.store(&*a.array, elems)
    }    

    /// Copies the `a` matrix to the `b` matrix.
    ///
    /// This method indeed copies the `a` matrix array to the `b` matrix array.
    pub fn copy(&self, a: &Matrix, b: &Matrix) -> Result<()>
    {
        if a.row_count != b.row_count || a.col_count != b.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, b.row_count, b.col_count)); 
        }
        self.backend.copy(&*a.array, &*b.array)        
    }    
    
    /// Copies matrix elements to the mutable slice and the transpose flag to the object that
    /// is referred by the reference.
    pub fn get_elems_and_transpose_flag(&self, a: &Matrix, elems: &mut [f32], is_transposed: &mut bool) -> Result<()>
    {
        if a.row_count * a.col_count != elems.len() {
            return Err(Error::MatrixElemCount(a.row_count * a.col_count, elems.len())); 
        }
        if !a.is_transposed {
            self.backend.load(&*a.array, elems)?;
        } else {
            self.backend.load(&*a.array, elems)?;
        }
        *is_transposed = true;
        Ok(())
    }
    
    /// Returns the elements and the transpose flag of matrix.
    pub fn elems_and_transpose_flag(&self, a: &Matrix) -> Result<(Vec<f32>, bool)>
    {
        let mut elems: Vec<f32> = vec![0.0; a.row_count * a.col_count];
        let mut is_transposed = false;
        self.get_elems_and_transpose_flag(a, elems.as_mut_slice(), &mut is_transposed)?;
        Ok((elems, is_transposed))
    }
    
    /// Adds the `a` matrix onto the `b` matrix and then the result is in the `c` matrix (C = A + B).
    pub fn add(&self, a: &Matrix, b: &Matrix, c: &Matrix) -> Result<()>
    {
        if a.row_count != b.row_count || a.col_count != b.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, b.row_count, b.col_count)); 
        }
        if a.row_count != c.row_count || a.col_count != c.col_count {
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

    /// Subtracts the `b` matrix from the `a` matrix and then the result is in the `c` matrix
    /// (C = A - B).
    pub fn sub(&self, a: &Matrix, b: &Matrix, c: &Matrix) -> Result<()>
    {
        if a.row_count != b.row_count || a.col_count != b.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, b.row_count, b.col_count)); 
        }
        if a.row_count != c.row_count || a.col_count != c.col_count {
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

    /// Multiplies the `a` matrix by the `b` matrix and then the result is in the `c` matrix
    /// (C = A * B).
    pub fn mul(&self, a: &Matrix, b: &Matrix, c: &Matrix) -> Result<()>
    {
        if a.row_count != c.row_count {
            return Err(Error::MulSize(a.row_count, a.col_count, b.row_count, b.col_count, c.row_count, c.col_count)); 
        }
        if b.col_count != c.col_count {
            return Err(Error::MulSize(a.row_count, a.col_count, b.row_count, b.col_count, c.row_count, c.col_count)); 
        }
        if a.col_count != b.row_count {
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

    /// Multiplies the `a` matrix elements by the `b` matrix elements and then the result is in the
    /// `c` matrix (Cij = Aij * Bij). 
    pub fn mul_elems(&self, a: &Matrix, b: &Matrix, c: &Matrix) -> Result<()>
    {
        if a.row_count != b.row_count || a.col_count != b.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, b.row_count, b.col_count)); 
        }
        if a.row_count != c.row_count || a.col_count != c.col_count {
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

    /// Divides the `a` matrix elements by the `b` matrix elements and then the result is in the `c`
    /// matrix (Cij = Aij / Bij).
    pub fn div_elems(&self, a: &Matrix, b: &Matrix, c: &Matrix) -> Result<()>
    {
        if a.row_count != b.row_count || a.col_count != b.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, b.row_count, b.col_count)); 
        }
        if a.row_count != c.row_count || a.col_count != c.col_count {
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

    /// Adds the `a` matrix onto the `b` scalar and then the result is in the `c` matrix (C = A + b).
    pub fn add_for_scalar(&self, a: &Matrix, b: f32, c: &Matrix) -> Result<()>
    {
        if a.row_count != c.row_count || a.col_count != c.col_count {
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

    /// Subtracts the `b` scalar from the `a` matrix and then the result is in the `c` matrix
    /// (C = A - b).
    pub fn sub_for_scalar(&self, a: &Matrix, b: f32, c: &Matrix) -> Result<()>
    {
        if a.row_count != c.row_count || a.col_count != c.col_count {
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

    /// Subtracts the `a` matrix from the `b` scalar and then the result is in the `c` matrix
    /// (C = b - A).
    pub fn rsub_for_scalar(&self, a: &Matrix, b: f32, c: &Matrix) -> Result<()>
    {
        if a.row_count != c.row_count || a.col_count != c.col_count {
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
    
    /// Multiplies the `a` matrix by the `b` scalar and then the result is in the `c` matrix
    /// (C = A * b).
    pub fn mul_for_scalar(&self, a: &Matrix, b: f32, c: &Matrix) -> Result<()>
    {
        if a.row_count != c.row_count || a.col_count != c.col_count {
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
    
    /// Divides the `a` matrix by the `b` scalar and then the result is in the `c` matrix
    /// (C = A / b). 
    pub fn div_for_scalar(&self, a: &Matrix, b: f32, c: &Matrix) -> Result<()>
    {
        if a.row_count != c.row_count || a.col_count != c.col_count {
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

    /// Divides the `b` scalar by the `a` matrix and then the result is in the `c` matrix
    /// (C = b / A). 
    pub fn rdiv_for_scalar(&self, a: &Matrix, b: f32, c: &Matrix) -> Result<()>
    {
        if a.row_count != c.row_count || a.col_count != c.col_count {
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

    /// Calculates sigmoid function for the `a` matrix and then the result is in the `b` matrix
    /// (B = sigmoid(A)).
    pub fn sigmoid(&self, a: &Matrix, b: &Matrix) -> Result<()>
    {
        if a.row_count != b.row_count || a.col_count != b.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, b.row_count, b.col_count)); 
        }
        if b.is_transposed {
            return Err(Error::ResTransposition);
        }
        if !a.is_transposed {
            self.backend.sigmoid_a(&*a.array, &*b.array, a.row_count, a.col_count)
        } else {
            self.backend.sigmoid_at(&*a.array, &*b.array, a.row_count, a.col_count)
        }
    }

    /// Calculates hyperbolic tangent function for the `a` matrix and then the result is in the `b`
    /// matrix (B = tanh(A)).
    pub fn tanh(&self, a: &Matrix, b: &Matrix) -> Result<()>
    {
        if a.row_count != b.row_count || a.col_count != b.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, b.row_count, b.col_count)); 
        }
        if b.is_transposed {
            return Err(Error::ResTransposition);
        }
        if !a.is_transposed {
            self.backend.tanh_a(&*a.array, &*b.array, a.row_count, a.col_count)
        } else {
            self.backend.tanh_at(&*a.array, &*b.array, a.row_count, a.col_count)
        }
    }    

    /// Calculates softmax function for the `a` matrix and then the result is in the `b` matrix
    /// (B = softmax(A)).
    pub fn softmax(&self, a: &Matrix, b: &Matrix) -> Result<()>
    {
        if a.row_count != b.row_count || a.col_count != b.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, b.row_count, b.col_count)); 
        }
        if b.is_transposed {
            return Err(Error::ResTransposition);
        }
        if !a.is_transposed {
            self.backend.softmax_a(&*a.array, &*b.array, a.row_count, a.col_count)
        } else {
            self.backend.softmax_at(&*a.array, &*b.array, a.row_count, a.col_count)
        }
    }    
    
    /// Indeed transposes the `a` matrix and then the result is in the `b` matrix (B = A^T).
    ///
    /// This method indeed transposes the `a` matrix without changing the transpose flag.
    pub fn really_transpose(&self, a: &Matrix, b: &Matrix) -> Result<()>
    {
        if a.row_count != b.col_count || a.col_count != b.row_count {
            return Err(Error::TransposeSize(a.row_count, a.col_count, b.row_count, b.col_count)); 
        }
        if a.is_transposed {
            return Err(Error::ArgTransposition);
        }
        if b.is_transposed {
            return Err(Error::ResTransposition);
        }
        self.backend.transpose_a(&*a.array, &*b.array, a.col_count, a.row_count)
    }
}

#[cfg(test)]
mod test_helpers;
#[cfg(all(test, not(feature = "test_only_backend")))]
mod tests;
