//
// Copyright (c) 2025 Łukasz Szpakowski
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
//! Micro neural matrix library for GPU is small library that operates on matrices.
//!
//! This library uses GPU by the following computing platforms:
//!
//! - OpenCL
//! - CUDA
//!
//! If this library uses CUDA, this library can use the cuBLAS library to multiplication of
//! matrices.
//!
//! A frontend-backend architecture is used by this library. The frontend of this library can use
//! one of two backends (OpenCL or CUDA). These backends allow to use GPU by the computing
//! platforms. The frontend and the backend can have many instances. This library provides a
//! high-level interfece to operations of matrices by the frontend and methods of a [`Matrix`]
//! structure.
//!
//! # Examples
//!
//! ```
//! # use unmtx_gpu::*;
//! let a = matrix![
//!     [1.0, 2.0],
//!     [3.0, 4.0]
//! ];
//! let x = matrix![
//!     [5.0],
//!     [6.0]
//! ];
//! let b = matrix![
//!     [7.0],
//!     [8.0]
//! ];
//! let c = a * x + b;
//! assert_eq!(vec![1.0 * 5.0 + 2.0 * 6.0 + 7.0, 3.0 * 5.0 + 4.0 * 6.0 + 8.0], c.elems());
//! ```
use std::ops::Neg;
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
///
/// The backend provides a low-level interface to computing platform (OpenCL or CUDA) for basic
/// operations and functions on matrices. The backend methods operate on backend arrays which
/// refers to areas of the device memory. The backend is low-level layer between a frontend and
/// computing platform.
pub trait Backend
{
    /// Returns the backend name.
    fn name(&self) -> &'static str;
    
    /// Returns `true` if the backend uses cuBLAS, otherwise `false`.
    fn has_cublas(&self) -> bool;
    
    /// Allocates a backend array.
    unsafe fn alloc(&self, n: usize) -> Result<BackendArray>;

    /// Allocates a backend array and stores zeros in the backend array.
    fn alloc_and_store_zeros(&self, n: usize) -> Result<BackendArray>;

    /// Allocates a backend array and stores the elements in the backend array.
    fn alloc_and_store(&self, elems: &[f32]) -> Result<BackendArray>;
    
    /// Loads elements from the backenc array.
    fn load(&self, a: &BackendArray, elems: &mut [f32]) -> Result<()>;

    /// Stores elements in the backend array.
    fn store(&self, a: &BackendArray, elems: &[f32]) -> Result<()>;

    /// Copies the `a` backend array to the `b` backend array.
    fn copy(&self, a: &BackendArray, b: &BackendArray) -> Result<()>;

    /// Transposes the `a` matrix and then the result is in the `b` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><msup><mi mathvariant="bold">A</mi><mi mathvariant="normal">T</mi></msup></mrow></math>).
    fn transpose_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Adds the `b` matrix to the `a` matrix and then the result is in the `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><mi mathvariant="bold">A</mi><mo>+</mo><mi mathvariant="bold">B</mi></mrow></math>).
    fn add_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Adds the `b` matrix to the transposed `a` matrix and then the result is in the `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><msup><mi mathvariant="bold">A</mi><mi mathvariant="normal">T</mi></msup><mo>+</mo><mi mathvariant="bold">B</mi></mrow></math>).
    fn add_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;
    
    /// Adds the transposed `b` matrix to the `a` matrix and then the result is in the `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><mi mathvariant="bold">A</mi><mo>+</mo><msup><mi mathvariant="bold">B</mi><mi mathvariant="normal">T</mi></msup></mrow></math>).
    fn add_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Adds the transposed `b` matrix to the transposed `a` matrix and then the result is in the
    /// `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><msup><mi mathvariant="bold">A</mi><mi mathvariant="normal">T</mi></msup><mo>+</mo><msup><mi mathvariant="bold">B</mi><mi mathvariant="normal">T</mi></msup></mrow></math>).
    fn add_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Subtracts the `b` matrix from the `a` matrix and then the result is in the `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><mi mathvariant="bold">A</mi><mo>-</mo><mi mathvariant="bold">B</mi></mrow></math>).
    fn sub_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Subtracts the `b` matrix from the transposed `a` matrix and then the result is in the `c`
    /// matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><msup><mi mathvariant="bold">A</mi><mi mathvariant="normal">T</mi></msup><mo>-</mo><mi mathvariant="bold">B</mi></mrow></math>).
    fn sub_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;
    
    /// Subtracts the transposed `b` matrix from the `a` matrix and then the result is in the `c`
    /// matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><mi mathvariant="bold">A</mi><mo>-</mo><msup><mi mathvariant="bold">B</mi><mi mathvariant="normal">T</mi></msup></mrow></math>).
    fn sub_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Subtracts the transposed `b` matrix from the transposed `a` matrix and then the result is
    /// in the `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><msup><mi mathvariant="bold">A</mi><mi mathvariant="normal">T</mi></msup><mo>-</mo><msup><mi mathvariant="bold">B</mi><mi mathvariant="normal">T</mi></msup></mrow></math>).
    fn sub_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;    
    
    /// Multiplies the `a` matrix by the `b` matrix and then the result is in the `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><mi mathvariant="bold">A</mi><mo>·</mo><mi mathvariant="bold">B</mi></mrow></math>).
    fn mul_a_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>;

    /// Multiplies the transposed `a` matrix by the `b` matrix and then the result is in the `c`
    /// matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><msup><mi mathvariant="bold">A</mi><mi mathvariant="normal">T</mi></msup><mo>·</mo><mi mathvariant="bold">B</mi></mrow></math>).
    fn mul_at_b(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>;

    /// Multiplies the `a` matrix by the transposed `b` matrix and then the result is in the `c`
    /// matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><mi mathvariant="bold">A</mi><mo>·</mo><msup><mi mathvariant="bold">B</mi><mi mathvariant="normal">T</mi></msup></mrow></math>).
    fn mul_a_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>;

    /// Multiplies the transposed `a` matrix by the transposed `b` matrix and then the result is in
    /// the `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><msup><mi mathvariant="bold">A</mi><mi mathvariant="normal">T</mi></msup><mo>·</mo><msup><mi mathvariant="bold">B</mi><mi mathvariant="normal">T</mi></msup></mrow></math>).
    fn mul_at_bt(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize, l: usize) -> Result<()>;

    /// Multiplies the `a` matrix elements by the `b` matrix elements and then the result is in the
    /// `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><msub><mi>a</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>·</mo><msub><mi>b</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></mrow></math>).
    fn mul_a_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Multiplies the transposed `a` matrix elements by the `b` matrix elements and saves the
    /// result to the `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><msub><mi>a</mi><mi mathvariant="italic">ji</mi></msub><mo>·</mo><msub><mi>b</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></mrow></math>).
    fn mul_at_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Multiplies the `a` matrix elements by the transposed `b` matrix elements and then the
    /// result is in the `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><msub><mi>a</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>·</mo><msub><mi>b</mi><mi mathvariant="italic">ji</mi></msub></mrow></math>).
    fn mul_a_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;
    
    /// Multiplies the transposed `a` matrix elements by the transposed `b` matrix elements and
    /// then the result is in the `c` matrix.
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><msub><mi>a</mi><mi mathvariant="italic">ji</mi></msub><mo>·</mo><msub><mi>b</mi><mi mathvariant="italic">ji</mi></msub></mrow></math>).
    fn mul_at_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Divides the `a` matrix elements by the `b` matrix elements and then the result is in the
    /// `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><mfrac><msub><mi>a</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><msub><mi>b</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></mfrac></mrow></math>).
    fn div_a_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Divides the transposed `a` matrix elements by the `b` matrix elements and then the result
    /// is in the `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><mfrac><msub><mi>a</mi><mi mathvariant="italic">ji</mi></msub><msub><mi>b</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></mfrac></mrow></math>).
    fn div_at_b_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;
    
    /// Divides the transposed `a` matrix elements by the `b` matrix elements and then the result
    /// is in the `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><mfrac><msub><mi>a</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><msub><mi>b</mi><mi mathvariant="italic">ji</mi></msub></mfrac></mrow></math>).
    fn div_a_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;
    
    /// Divides the transposed `a` matrix elements by the transposed `b` matrix elements and then
    /// the result is in the `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><mfrac><msub><mi>a</mi><mi mathvariant="italic">ji</mi></msub><msub><mi>b</mi><mi mathvariant="italic">ji</mi></msub></mfrac></mrow></math>).
    fn div_at_bt_for_elems(&self, a: &BackendArray, b: &BackendArray, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Adds the `b` scalar to the `a` matrix and then the result is in the `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><mi mathvariant="bold">A</mi><mo>+</mo><mi>b</mi></mrow></math>).
    fn add_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Adds the `b` scalar to the transposed `a` matrix and then the result is in the `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><msup><mi mathvariant="bold">A</mi><mi mathvariant="normal">T</mi></msup><mo>+</mo><mi>b</mi></mrow></math>).
    fn add_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Subtracts the `b` scalar from the `a` matrix and then the result is in the `c` matrix.
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><mi mathvariant="bold">A</mi><mo>-</mo><mi>b</mi></mrow></math>).
    fn sub_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Subtracts the `b` scalar from the transposed `a` matrix and then the result is in the `c`
    /// matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><msup><mi mathvariant="bold">A</mi><mi mathvariant="normal">T</mi></msup><mo>-</mo><mi>b</mi></mrow></math>).
    fn sub_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Subtracts the `a` matrix from the `b` scalar  and then the result is in the `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><mi>b</mi><mo>-</mo><mi mathvariant="bold">A</mi></mrow></math>).
    fn rsub_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Subtracts the transposed `a` matrix from the `b` scalar  and then the result is in the `c`
    /// matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><mi>b</mi><mo>-</mo><msup><mi mathvariant="bold">A</mi><mi mathvariant="normal">T</mi></msup></mrow></math>).
    fn rsub_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;
    
    /// Multiplies the `a` matrix by the `b` scalar and then the result is in the `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><mi mathvariant="bold">A</mi><mo>·</mo><mi>b</mi></mrow></math>).
    fn mul_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Multiplies the transposed `a` matrix by the `b` scalar and then the result is in the `c`
    /// matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><msup><mi mathvariant="bold">A</mi><mi mathvariant="normal">T</mi></msup><mo>·</mo><mi>b</mi></mrow></math>).
    fn mul_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Divides the `a` matrix by the `b` scalar and then the result is in the `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><mfrac><mi mathvariant="bold">A</mi><mi>b</mi></mfrac></mrow></math>).
    fn div_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Divides the transposed `a` matrix by the `b` scalar and then the result is in the `c`
    /// matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><mfrac><msup><mi mathvariant="bold">A</mi><mi mathvariant="normal">T</mi></msup><mi>b</mi></mfrac></mrow></math>).
    fn div_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Divides the `b` scalar by the `a` matrix elements and then the result is in the `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><mfrac><mi>b</mi><msub><mi>a</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></mfrac></mrow></math>).
    fn rdiv_a_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Divides the `b` scalar by the transposed `a` matrix elements and then the result is in the
    /// `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><mfrac><mi>b</mi><msub><mi>a</mi><mi mathvariant="italic">ji</mi></msub></mfrac></mrow></math>).
    fn rdiv_at_b_for_scalar(&self, a: &BackendArray, b: f32, c: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Calculates sigmoid function for the `a` matrix adn the result is the `b` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">B</mi><mo>=</mo><mi>sigmoid</mi><mo fence="true">(</mo><mi mathvariant="bold">A</mi><mo fence="true">)</mo></mrow></math>).
    fn sigmoid_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Calculates sigmoid function for the transposed `a` matrix and then the result is in the `b`
    /// matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">B</mi><mo>=</mo><mi>sigmoid</mi><mo fence="true">(</mo><msup><mi mathvariant="bold">A</mi><mi mathvariant="normal">T</mi></msup><mo fence="true">)</mo></mrow></math>).
    fn sigmoid_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Calculates hyperbolic tangent function for the `a` matrix and then the result is in `b`
    /// matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">B</mi><mo>=</mo><mi>tanh</mi><mo fence="true">(</mo><mi mathvariant="bold">A</mi><mo fence="true">)</mo></mrow></math>).
    fn tanh_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Calculates hyperbolic tangent function for the transposed `a` matrix and then the result is
    /// in the `b` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">B</mi><mo>=</mo><mi>tanh</mi><mo fence="true">(</mo><msup><mi mathvariant="bold">A</mi><mi mathvariant="normal">T</mi></msup><mo fence="true">)</mo></mrow></math>).
    fn tanh_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Calculates swish function for the `a` matrix adn the result is the `b` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">B</mi><mo>=</mo><mi>swish</mi><mo fence="true">(</mo><mi mathvariant="bold">A</mi><mo fence="true">)</mo></mrow></math>).
    fn swish_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Calculates swish function for the transposed `a` matrix and then the result is in the `b`
    /// matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">B</mi><mo>=</mo><mi>swish</mi><mo fence="true">(</mo><msup><mi mathvariant="bold">A</mi><mi mathvariant="normal">T</mi></msup><mo fence="true">)</mo></mrow></math>).
    fn swish_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>;
    
    /// Calculates softmax function for the `a` matrix and then the result is in the `b` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">B</mi><mo>=</mo><mi>softmax</mi><mo fence="true">(</mo><mi mathvariant="bold">A</mi><mo fence="true">)</mo></mrow></math>).
    fn softmax_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Calculates softmax function for the transposed `a` matrix and then the result is in the `b`
    /// matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">B</mi><mo>=</mo><mi>softmax</mi><mo fence="true">(</mo><msup><mi mathvariant="bold">A</mi><mi mathvariant="normal">T</mi></msup><mo fence="true">)</mo></mrow></math>).
    fn softmax_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Calculates square root of the `a` matrix adn the result is the `b` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">B</mi><mo>=</mo><msqrt><mi mathvariant="bold">A</mi></msqrt></mrow></math>).
    fn sqrt_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Calculates square root of the transposed `a` matrix and then the result is in the `b`
    /// matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">B</mi><mo>=</mo><msqrt><msup><mi mathvariant="bold">A</mi><mi mathvariant="normal">T</mi></msup></msqrt></mrow></math>).
    fn sqrt_at(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>;    
    
    /// Repeats the `a` vector as column
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>b</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><msub><mi>a</mi><mi>i</mi></msub></mrow></math>).
    fn repeat_col_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>;

    /// Repeats the `a` vector as row
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>b</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><msub><mi>a</mi><mi>j</mi></msub></mrow></math>).
    fn repeat_row_a(&self, a: &BackendArray, b: &BackendArray, n: usize, m: usize) -> Result<()>;
}

/// An error enumeration.
#[derive(Debug)]
pub enum Error
{
    /// Can't initialize a default backend.
    DefaultBackendInitialization,
    /// Mismatched sizes of matrices for a matrix operation.
    OpSize(usize, usize, usize, usize),
    /// Mismatched sizes of matrices for a matrix multiplication.
    MulSize(usize, usize, usize, usize, usize, usize),
    /// Mismatched sizes of matrices for a matrix transposition.
    TransposeSize(usize, usize, usize, usize),
    /// An argument matrix is transposed.
    ArgTransposition,
    /// A result matrix is transposed.
    ResTransposition,
    /// A number of matrix elements isn't equal to a number of elements.
    MatrixElemCount(usize, usize),
    /// A matrix isn't a vector.
    IsNotVector,
    /// A mutex can't be locked.
    Mutex,
    /// An OpenCL error.
    #[cfg(feature = "opencl")]
    OpenCl(opencl::ClError),
    /// A CUDA error.
    #[cfg(feature = "cuda")]
    Cuda(cuda::DriverError),
    /// A cuBLAS error.
    #[cfg(feature = "cuda")]
    Cublas(cuda::CublasError),
    /// No a cuBLAS.
    #[cfg(feature = "cuda")]
    NoCublas,
    /// A compilation error.
    Compilation(String),
    /// No a platform.
    NoPlatform,
    /// No a device.
    NoDevice,
    /// No a kernel.
    NoKernel(String),
    /// A type of device information is invalid.
    InvalidDeviceInfoType,
    /// A number of backend array elements isn't equal to a number of elements.
    BackendArrayElemCount(usize, usize),
    /// Two numbers of elements of backend arrays aren't equal.
    TwoBackendArrayElemCounts(usize, usize),
    /// A backend array is invalid.
    InvalidBackendArray,
}

impl error::Error for Error
{}

impl fmt::Display for Error
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    {
        match self {
            Error::DefaultBackendInitialization => write!(f, "can't initialize default backend"),
            Error::OpSize(n1, m1, n2, m2) => write!(f, "mismatched sizes of matrices ({}x{}, {}x{})", n1, m1, n2, m2),
            Error::MulSize(n1, m1, n2, m2, n3, m3) => write!(f, "mismatched sizes of matrices for multiplication ({}x{}, {}x{}, {}x{})", n1, m1, n2, m2, n3, m3),
            Error::TransposeSize(n1, m1, n2, m2) => write!(f, "mismatched sizes of matrices for transposition ({}x{}, {}x{})", n1, m1, n2, m2),
            Error::ArgTransposition => write!(f, "argument matrix is transposed"),
            Error::ResTransposition => write!(f, "result matrix is transposed"),
            Error::MatrixElemCount(n1, n2) => write!(f, "number of matrix elements isn't equal to number of elements ({}, {})", n1, n2),
            Error::IsNotVector => write!(f, "matrix isn't vector"),
            Error::Mutex => write!(f, "can't lock mutex"),
            #[cfg(feature = "opencl")]
            Error::OpenCl(err) => write!(f, "OpenCL error: {}", err),
            #[cfg(feature = "cuda")]
            Error::Cuda(err) => write!(f, "CUDA error: {}", err),
            #[cfg(feature = "cuda")]
            Error::Cublas(err) => write!(f, "cuBLAS error: {}", err),
            #[cfg(feature = "cuda")]
            Error::NoCublas => write!(f, "no cuBLAS"),
            Error::Compilation(msg) => write!(f, "{}", msg),
            Error::NoPlatform => write!(f, "no platform"),
            Error::NoDevice => write!(f, "no device"),
            Error::NoKernel(name) => write!(f, "no kernel {}", name),
            Error::InvalidDeviceInfoType => write!(f, "invalid device info type"),
            Error::BackendArrayElemCount(n1, n2) => write!(f, "number of backend array elements isn't equal to number of elements ({}, {})", n1, n2),
            Error::TwoBackendArrayElemCounts(n1, n2) => write!(f, "two numbers of elements of backend arrays aren't equal ({}, {})", n1, n2),
            Error::InvalidBackendArray => write!(f, "invalid backend array"),
        }
    }
}

/// A result type.
pub type Result<T> = result::Result<T, Error>;

/// An enumeration of backend array.
///
/// This enumeration contains the reference to the area of the device memory for computing
/// platform (OpenCL or CUDA).
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

/// Returns a default backend.
pub fn get_default_backend() -> Result<Option<Arc<dyn Backend>>>
{
    unsafe {
        let default_backend_g = mutex_lock(&DEFAULT_BACKEND)?;
        Ok(default_backend_g.clone())
    }
}

/// Sets a default backend.
pub fn set_default_backend(backend: Arc<dyn Backend>) -> Result<()>
{
    unsafe {
        let mut default_backend_g = mutex_lock(&DEFAULT_BACKEND)?;
        *default_backend_g = Some(backend);
    }
    Ok(())
}

/// Unsets a default backend.
pub fn unset_default_backend() -> Result<()>
{
    unsafe {
        let mut default_backend_g = mutex_lock(&DEFAULT_BACKEND)?;
        *default_backend_g = None;
    }
    Ok(())
}

/// Sets a default backend if the default backend is uninitialized and returns the default backend.
///
/// This method takes a closure that returns the backend and then the backend is set as the default
/// backend if the default backend is uninitialized. The closure is only called if the backend is
/// to be set.
pub fn set_default_backend_for_uninitialized<F>(f: F) -> Result<Arc<dyn Backend>>
    where F: FnOnce() -> Result<Arc<dyn Backend>>
{
    unsafe {
        let mut default_backend_g = mutex_lock(&DEFAULT_BACKEND)?;
        match &*default_backend_g {
            Some(default_backend) => Ok(default_backend.clone()),
            None => {
                let backend = f()?;
                *default_backend_g = Some(backend.clone());
                Ok(backend)
            },
        }
    }
}

/// Initializes a default backend if the backend is uninitialized and returns the default backend.
pub fn initialize_default_backend_for_uninitialized() -> Result<Arc<dyn Backend>>
{
    #[cfg(feature = "opencl")]
    let res = set_default_backend_for_uninitialized(|| Ok(Arc::new(opencl::ClBackend::new()?)));
    #[cfg(all(not(feature = "opencl"), feature = "cuda"))]
    let res = set_default_backend_for_uninitialized(|| Ok(Arc::new(cuda::CudaBackend::new()?)));
    #[cfg(all(not(feature = "opencl"), not(feature = "cuda")))]
    let res: Result<Arc<dyn Backend>> = Err(Error::DefaultBackendInitialization);
    res
}

/// Finalizes a default backend.
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
        $crate::Matrix::new_with_elem_vecs(vec![$(vec![$($elem),*]),*].as_slice())
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
    
    /// Transposes the matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="bold">A</mi><mi mathvariant="normal">T</mi></msup></mrow></math>).
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
    
    /// See [`transpose`](Self::transpose).
    pub fn t(&self) -> Self
    { self.transpose() }
    
    /// Indeed transposes the matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="bold">A</mi><mi mathvariant="normal">T</mi></msup></mrow></math>).
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
    
    /// See [`really_transpose`](Self::really_transpose).
    pub fn rt(&self) -> Self
    { self.really_transpose() }
    
    /// Multiplies the matrix elements by the `b` matrix elements
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>a</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>·</mo><msub><mi>b</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></mrow></math>).    
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

    /// Divides the matrix elements by the `b` matrix elements
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><msub><mi>a</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><msub><mi>b</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></mfrac></mrow></math>).
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

    /// Subtracts the matrix from the scalar
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi><mo>-</mo><mi mathvariant="bold">A</mi></mrow></math>).
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

    /// Divides the scalar by the matrix elements
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mi>b</mi><msub><mi>a</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></mfrac></mrow></math>).
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

    /// Calculates sigmoid function for the matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>sigmoid</mi><mo fence="true">(</mo><mi mathvariant="bold">A</mi><mo fence="true">)</mo></mrow></math>).
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

    /// Calculates hiperbolic tangent function for the matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>tanh</mi><mo fence="true">(</mo><mi mathvariant="bold">A</mi><mo fence="true">)</mo></mrow></math>).
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

    /// Calculates swish function for the matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>swish</mi><mo fence="true">(</mo><mi mathvariant="bold">A</mi><mo fence="true">)</mo></mrow></math>).
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ];
    /// let b = a.swish();
    /// let elems = b.elems();
    /// assert!((1.0 / (1.0 + (-1.0f32).exp()) - elems[0]).abs() < 0.001);
    /// assert!((2.0 / (1.0 + (-2.0f32).exp()) - elems[1]).abs() < 0.001);
    /// assert!((3.0 / (1.0 + (-3.0f32).exp()) - elems[2]).abs() < 0.001);
    /// assert!((4.0 / (1.0 + (-4.0f32).exp()) - elems[3]).abs() < 0.001);
    /// ```
    pub fn swish(&self) -> Self
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.swish(self, &res).unwrap();
        res
    }

    /// Calculates softmax function for the matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>softmax</mi><mo fence="true">(</mo><mi mathvariant="bold">A</mi><mo fence="true">)</mo></mrow></math>).
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
    
    /// Calculates square root of the matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mi mathvariant="bold">A</mi></msqrt></mrow></math>).
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ];
    /// let b = a.sqrt();
    /// let elems = b.elems();
    /// assert!((1.0f32.sqrt() - elems[0]).abs() < 0.001);
    /// assert!((2.0f32.sqrt() - elems[1]).abs() < 0.001);
    /// assert!((3.0f32.sqrt() - elems[2]).abs() < 0.001);
    /// assert!((4.0f32.sqrt() - elems[3]).abs() < 0.001);
    /// ```
    pub fn sqrt(&self) -> Self
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.sqrt(self, &res).unwrap();
        res
    }
    
    /// Repeats the vector as column or a row.
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0],
    ///     [2.0]
    /// ];
    /// let b = a.repeat(3);
    /// assert_eq!(vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0], b.elems());
    /// let c = matrix![[1.0, 2.0, 3.0]];
    /// let d = c.repeat(2);
    /// assert_eq!(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], d.elems());
    /// ```
    pub fn repeat(&self, n: usize) -> Self
    {
        assert!(self.col_count == 1 || self.row_count == 1); 
        let frontend = Frontend::new().unwrap();
        let res = if self.col_count == 1 {
            unsafe { frontend.create_matrix(self.row_count, n) }.unwrap()
        } else {
            unsafe { frontend.create_matrix(n, self.col_count) }.unwrap()
        };
        frontend.repeat(self, &res).unwrap();
        res
    }
}

impl Neg for Matrix
{
    type Output = Self;

    fn neg(self) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.rsub_for_scalar(&self, 0.0, &res).unwrap();
        res
    }
}

impl Neg for &Matrix
{
    type Output = Matrix;

    fn neg(self) -> Self::Output
    {
        let frontend = Frontend::new().unwrap();
        let res = unsafe { frontend.create_matrix(self.row_count, self.col_count) }.unwrap();
        frontend.rsub_for_scalar(self, 0.0, &res).unwrap();
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
        let res = if frontend.backend().has_cublas() {
            frontend.create_matrix_and_set_zeros(self.row_count, rhs.col_count).unwrap()
        } else {
            unsafe { frontend.create_matrix(self.row_count, rhs.col_count) }.unwrap()
        };
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
        let res = if frontend.backend().has_cublas() {
            frontend.create_matrix_and_set_zeros(self.row_count, rhs.col_count).unwrap()
        } else {
            unsafe { frontend.create_matrix(self.row_count, rhs.col_count) }.unwrap()
        };
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
        let res = if frontend.backend().has_cublas() {
            frontend.create_matrix_and_set_zeros(self.row_count, rhs.col_count).unwrap()
        } else {
            unsafe { frontend.create_matrix(self.row_count, rhs.col_count) }.unwrap()
        };
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
        let res = if frontend.backend().has_cublas() {
            frontend.create_matrix_and_set_zeros(self.row_count, rhs.col_count).unwrap()
        } else {
            unsafe { frontend.create_matrix(self.row_count, rhs.col_count) }.unwrap()
        };
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
        let res = if frontend.backend().has_cublas() {
            frontend.create_matrix_and_set_zeros(self.row_count, rhs.col_count).unwrap()
        } else {
            unsafe { frontend.create_matrix(self.row_count, rhs.col_count) }.unwrap()
        };
        frontend.mul(&self, &rhs, &res).unwrap();
        *self = res;
    }
}

impl MulAssign<&Matrix> for Matrix
{
    fn mul_assign(&mut self, rhs: &Self)
    {
        let frontend = Frontend::new().unwrap();
        let res = if frontend.backend().has_cublas() {
            frontend.create_matrix_and_set_zeros(self.row_count, rhs.col_count).unwrap()
        } else {
            unsafe { frontend.create_matrix(self.row_count, rhs.col_count) }.unwrap()
        };
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
///
/// The frontend contains methods which operate on matrices or calculate functions for the
/// matrices. Backend methods are called by the frontend to operate the matrices. The frontend is
/// high-level layer that can be directly used by programmer or a [`Matrix`] structure.
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
    { Ok(Frontend { backend: initialize_default_backend_for_uninitialized()?, }) }

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
    
    /// Copies the matrix elements to the mutable slice and the transpose flag to the object that
    /// is referred by the reference.
    pub fn get_elems_and_transpose_flag(&self, a: &Matrix, elems: &mut [f32], is_transposed: &mut bool) -> Result<()>
    {
        if a.row_count * a.col_count != elems.len() {
            return Err(Error::MatrixElemCount(a.row_count * a.col_count, elems.len())); 
        }
        self.backend.load(&*a.array, elems)?;
        *is_transposed = a.is_transposed;
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
    
    /// Adds the `b` matrix to the `a` matrix and then the result is in the `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><mi mathvariant="bold">A</mi><mo>+</mo><mi mathvariant="bold">B</mi></mrow></math>).
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
    /// let c = Matrix::new(2, 2);
    /// let frontend = Frontend::new().unwrap();
    /// frontend.add(&a, &b, &c).unwrap();
    /// assert_eq!(vec![1.0 + 5.0, 2.0 + 6.0, 3.0 + 7.0, 4.0 + 8.0], c.elems());
    /// ```
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
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><mi mathvariant="bold">A</mi><mo>-</mo><mi mathvariant="bold">B</mi></mrow></math>).
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
    /// let c = Matrix::new(2, 2);
    /// let frontend = Frontend::new().unwrap();
    /// frontend.sub(&a, &b, &c).unwrap();
    /// assert_eq!(vec![1.0 - 5.0, 2.0 - 6.0, 3.0 - 7.0, 4.0 - 8.0], c.elems());
    /// ```
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
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><mi mathvariant="bold">A</mi><mo>·</mo><mi mathvariant="bold">B</mi></mrow></math>).
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0]
    /// ];
    /// let b = matrix![
    ///     [7.0,  8.0],
    ///     [9.0,  10.0],
    ///     [11.0, 12.0]
    /// ];
    /// let c = Matrix::new(2, 2);
    /// let frontend = Frontend::new().unwrap();
    /// frontend.mul(&a, &b, &c).unwrap();
    /// let c11: f32 = 1.0 * 7.0 + 2.0 * 9.0 + 3.0 * 11.0;
    /// let c12: f32 = 1.0 * 8.0 + 2.0 * 10.0 + 3.0 * 12.0;
    /// let c21: f32 = 4.0 * 7.0 + 5.0 * 9.0 + 6.0 * 11.0;
    /// let c22: f32 = 4.0 * 8.0 + 5.0 * 10.0 + 6.0 * 12.0;
    /// assert_eq!(vec![c11, c12, c21, c22], c.elems());
    /// ```
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
    /// `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><msub><mi>a</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>·</mo><msub><mi>b</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></mrow></math>).
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
    /// let c = Matrix::new(2, 2);
    /// let frontend = Frontend::new().unwrap();
    /// frontend.mul_elems(&a, &b, &c).unwrap();
    /// assert_eq!(vec![1.0 * 5.0, 2.0 * 6.0, 3.0 * 7.0, 4.0 * 8.0], c.elems());
    /// ```
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
    /// matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><mfrac><msub><mi>a</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><msub><mi>b</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></mfrac></mrow></math>).
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
    /// let c = Matrix::new(2, 2);
    /// let frontend = Frontend::new().unwrap();
    /// frontend.div_elems(&a, &b, &c).unwrap();
    /// let elems = c.elems();
    /// assert!((1.0 / 5.0 - elems[0]).abs() < 0.001);
    /// assert!((2.0 / 6.0 - elems[1]).abs() < 0.001);
    /// assert!((3.0 / 7.0 - elems[2]).abs() < 0.001);
    /// assert!((4.0 / 8.0 - elems[3]).abs() < 0.001);
    /// ```
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

    /// Adds the `b` scalar to the `a` matrix and then the result is in the `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><mi mathvariant="bold">A</mi><mo>+</mo><mi>b</mi></mrow></math>).
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ];
    /// let c = Matrix::new(2, 2);
    /// let frontend = Frontend::new().unwrap();
    /// frontend.add_for_scalar(&a, 10.5, &c).unwrap();
    /// assert_eq!(vec![1.0 + 10.5, 2.0 + 10.5, 3.0 + 10.5, 4.0 + 10.5], c.elems());
    /// ```
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
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><mi mathvariant="bold">A</mi><mo>-</mo><mi>b</mi></mrow></math>).
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ];
    /// let c = Matrix::new(2, 2);
    /// let frontend = Frontend::new().unwrap();
    /// frontend.sub_for_scalar(&a, 10.5, &c).unwrap();
    /// assert_eq!(vec![1.0 - 10.5, 2.0 - 10.5, 3.0 - 10.5, 4.0 - 10.5], c.elems());
    /// ```
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
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><mi>b</mi><mo>-</mo><mi mathvariant="bold">A</mi></mrow></math>).
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ];
    /// let c = Matrix::new(2, 2);
    /// let frontend = Frontend::new().unwrap();
    /// frontend.rsub_for_scalar(&a, 10.5, &c).unwrap();
    /// assert_eq!(vec![10.5 - 1.0, 10.5 - 2.0, 10.5 - 3.0, 10.5 - 4.0], c.elems());
    /// ```
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
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><mi mathvariant="bold">A</mi><mo>·</mo><mi>b</mi></mrow></math>).
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ];
    /// let c = Matrix::new(2, 2);
    /// let frontend = Frontend::new().unwrap();
    /// frontend.mul_for_scalar(&a, 10.5, &c).unwrap();
    /// assert_eq!(vec![1.0 * 10.5, 2.0 * 10.5, 3.0 * 10.5, 4.0 * 10.5], c.elems());
    /// ```
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
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">C</mi><mo>=</mo><mfrac><mi mathvariant="bold">A</mi><mi>b</mi></mfrac></mrow></math>).
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ];
    /// let c = Matrix::new(2, 2);
    /// let frontend = Frontend::new().unwrap();
    /// frontend.div_for_scalar(&a, 10.5, &c).unwrap();
    /// let elems = c.elems();
    /// assert!((1.0 / 10.5 - elems[0]).abs() < 0.001);
    /// assert!((2.0 / 10.5 - elems[1]).abs() < 0.001);
    /// assert!((3.0 / 10.5 - elems[2]).abs() < 0.001);
    /// assert!((4.0 / 10.5 - elems[3]).abs() < 0.001);
    /// ```
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

    /// Divides the `b` scalar by the `a` matrix elements and then the result is in the `c` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><mfrac><mi>b</mi><msub><mi>a</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></mfrac></mrow></math>).
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ];
    /// let c = Matrix::new(2, 2);
    /// let frontend = Frontend::new().unwrap();
    /// frontend.rdiv_for_scalar(&a, 10.5, &c).unwrap();
    /// let elems = c.elems();
    /// assert!((10.5 / 1.0- elems[0]).abs() < 0.001);
    /// assert!((10.5 / 2.0 - elems[1]).abs() < 0.001);
    /// assert!((10.5 / 3.0 - elems[2]).abs() < 0.001);
    /// assert!((10.5 / 4.0 - elems[3]).abs() < 0.001);
    /// ```
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
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">B</mi><mo>=</mo><mi>sigmoid</mi><mo fence="true">(</mo><mi mathvariant="bold">A</mi><mo fence="true">)</mo></mrow></math>).
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ];
    /// let b = Matrix::new(2, 2);
    /// let frontend = Frontend::new().unwrap();
    /// frontend.sigmoid(&a, &b).unwrap();
    /// let elems = b.elems();
    /// assert!((1.0 / (1.0 + (-1.0f32).exp()) - elems[0]).abs() < 0.001);
    /// assert!((1.0 / (1.0 + (-2.0f32).exp()) - elems[1]).abs() < 0.001);
    /// assert!((1.0 / (1.0 + (-3.0f32).exp()) - elems[2]).abs() < 0.001);
    /// assert!((1.0 / (1.0 + (-4.0f32).exp()) - elems[3]).abs() < 0.001);
    /// ```
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
    /// matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">B</mi><mo>=</mo><mi>tanh</mi><mo fence="true">(</mo><mi mathvariant="bold">A</mi><mo fence="true">)</mo></mrow></math>).
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ];
    /// let b = Matrix::new(2, 2);
    /// let frontend = Frontend::new().unwrap();
    /// frontend.tanh(&a, &b).unwrap();
    /// let elems = b.elems();
    /// assert!((1.0f32.tanh() - elems[0]).abs() < 0.001);
    /// assert!((2.0f32.tanh() - elems[1]).abs() < 0.001);
    /// assert!((3.0f32.tanh() - elems[2]).abs() < 0.001);
    /// assert!((4.0f32.tanh() - elems[3]).abs() < 0.001);
    /// ```
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

    /// Calculates swish function for the `a` matrix and then the result is in the `b` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">B</mi><mo>=</mo><mi>swish</mi><mo fence="true">(</mo><mi mathvariant="bold">A</mi><mo fence="true">)</mo></mrow></math>).
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ];
    /// let b = Matrix::new(2, 2);
    /// let frontend = Frontend::new().unwrap();
    /// frontend.swish(&a, &b).unwrap();
    /// let elems = b.elems();
    /// assert!((1.0 / (1.0 + (-1.0f32).exp()) - elems[0]).abs() < 0.001);
    /// assert!((2.0 / (1.0 + (-2.0f32).exp()) - elems[1]).abs() < 0.001);
    /// assert!((3.0 / (1.0 + (-3.0f32).exp()) - elems[2]).abs() < 0.001);
    /// assert!((4.0 / (1.0 + (-4.0f32).exp()) - elems[3]).abs() < 0.001);
    /// ```
    pub fn swish(&self, a: &Matrix, b: &Matrix) -> Result<()>
    {
        if a.row_count != b.row_count || a.col_count != b.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, b.row_count, b.col_count)); 
        }
        if b.is_transposed {
            return Err(Error::ResTransposition);
        }
        if !a.is_transposed {
            self.backend.swish_a(&*a.array, &*b.array, a.row_count, a.col_count)
        } else {
            self.backend.swish_at(&*a.array, &*b.array, a.row_count, a.col_count)
        }
    }
    
    /// Calculates softmax function for the `a` matrix and then the result is in the `b` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">B</mi><mo>=</mo><mi>softmax</mi><mo fence="true">(</mo><mi mathvariant="bold">A</mi><mo fence="true">)</mo></mrow></math>).
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ];
    /// let b = Matrix::new(2, 2);
    /// let frontend = Frontend::new().unwrap();
    /// frontend.softmax(&a, &b).unwrap();
    /// let elems = b.elems();
    /// let sum1 = 1.0f32.exp() + 3.0f32.exp();
    /// let sum2 = 2.0f32.exp() + 4.0f32.exp();
    /// assert!((1.0f32.exp() / sum1 - elems[0]).abs() < 0.001);
    /// assert!((2.0f32.exp() / sum2 - elems[1]).abs() < 0.001);
    /// assert!((3.0f32.exp() / sum1 - elems[2]).abs() < 0.001);
    /// assert!((4.0f32.exp() / sum2 - elems[3]).abs() < 0.001);
    /// ```
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

    /// Calculates square root of the `a` matrix and then the result is in the `b` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">B</mi><mo>=</mo><msqrt><mi mathvariant="bold">A</mi></msqrt></mrow></math>).
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ];
    /// let b = Matrix::new(2, 2);
    /// let frontend = Frontend::new().unwrap();
    /// frontend.sqrt(&a, &b).unwrap();
    /// let elems = b.elems();
    /// assert!((1.0f32.sqrt() - elems[0]).abs() < 0.001);
    /// assert!((2.0f32.sqrt() - elems[1]).abs() < 0.001);
    /// assert!((3.0f32.sqrt() - elems[2]).abs() < 0.001);
    /// assert!((4.0f32.sqrt() - elems[3]).abs() < 0.001);
    /// ```
    pub fn sqrt(&self, a: &Matrix, b: &Matrix) -> Result<()>
    {
        if a.row_count != b.row_count || a.col_count != b.col_count {
            return Err(Error::OpSize(a.row_count, a.col_count, b.row_count, b.col_count)); 
        }
        if b.is_transposed {
            return Err(Error::ResTransposition);
        }
        if !a.is_transposed {
            self.backend.sqrt_a(&*a.array, &*b.array, a.row_count, a.col_count)
        } else {
            self.backend.sqrt_at(&*a.array, &*b.array, a.row_count, a.col_count)
        }
    }
    
    /// Indeed transposes the `a` matrix and then the result is in the `b` matrix
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="bold">B</mi><mo>=</mo><msup><mi mathvariant="bold">A</mi><mi mathvariant="normal">T</mi></msup></mrow></math>).
    ///
    /// This method indeed transposes the `a` matrix without changing the transpose flag.
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0]
    /// ];
    /// let b = Matrix::new(3, 2);
    /// let frontend = Frontend::new().unwrap();
    /// frontend.really_transpose(&a, &b).unwrap();
    /// assert_eq!(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], b.elems());
    /// ```
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

    /// Repeats the `a` vector as column or a row
    /// (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>b</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><msub><mi>a</mi><mi>i</mi></msub></mrow></math> or 
    /// <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>b</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><msub><mi>a</mi><mi>j</mi></msub></mrow></math>).
    ///
    /// # Examples
    ///
    /// ```
    /// # use unmtx_gpu::*;
    /// let a = matrix![
    ///     [1.0],
    ///     [2.0]
    /// ];
    /// let b = Matrix::new(2, 3);
    /// let frontend = Frontend::new().unwrap();
    /// frontend.repeat(&a, &b).unwrap();
    /// assert_eq!(vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0], b.elems());
    /// let c = matrix![[1.0, 2.0, 3.0]];
    /// let d = Matrix::new(2, 3);
    /// let frontend = Frontend::new().unwrap();
    /// frontend.repeat(&c, &d).unwrap();
    /// assert_eq!(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], d.elems());
    /// ```
    pub fn repeat(&self, a: &Matrix, b: &Matrix) -> Result<()>
    {
        if b.is_transposed {
            return Err(Error::ResTransposition);
        }
        if a.col_count == 1 {
            if a.row_count != b.row_count {
                return Err(Error::OpSize(a.row_count, a.col_count, b.row_count, b.col_count));
            }
            self.backend.repeat_col_a(&*a.array, &*b.array, a.row_count, b.col_count)
        } else if a.row_count == 1 {
            if a.col_count != b.col_count {
                return Err(Error::OpSize(a.row_count, a.col_count, b.row_count, b.col_count));
            }
            self.backend.repeat_row_a(&*a.array, &*b.array, b.row_count, a.col_count)
        } else {
            Err(Error::IsNotVector)
        }
    }
}

#[cfg(test)]
mod test_helpers;
#[cfg(all(test, not(feature = "test_only_backend")))]
mod tests;
