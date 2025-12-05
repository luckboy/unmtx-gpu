//
// Copyright (c) 2025 Łukasz Szpakowski
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
use crate::*;

pub(crate) fn fixture_a(n: usize, m: usize) -> Vec<f32>
{
    let mut a = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            a[m * i + j] = (((i as f32) + 1.0) * 10.0 + ((j as f32) + 1.0)) / 2.0;
        }
    }
    a
}

pub(crate) fn fixture_a_b(n1: usize, m1: usize, n2: usize, m2: usize) -> (Vec<f32>, Vec<f32>)
{
    let mut a = vec![0.0f32; n1 * m1];
    for i in 0..n1 {
        for j in 0..m1 {
            a[m1 * i + j] = (((i as f32) + 1.0) * 10.0 + ((j as f32) + 1.0)) / 2.0;
        }
    }
    let mut b = vec![0.0f32; n2 * m2];
    for i in 0..n2 {
        for j in 0..m2 {
            b[m2 * i + j] = (((i as f32) + (n1 as f32) + 1.0) * 10.0 + ((j as f32) + (m1 as f32) + 1.0)) / 2.0;
        }
    }
    (a, b)
}

pub(crate) fn fixture_a_for_activation_fun(n: usize, m: usize) -> Vec<f32>
{
    let mut a = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            a[m * i + j] = ((i as f32) - ((n as f32) - 1.0) / 2.0) * 2.0 / ((n as f32) - 1.0);
            a[m * i + j] += ((j as f32) - ((m as f32) - 1.0) / 2.0) * 2.0 / ((m as f32) - 1.0);
        }
    }
    a
}

pub(crate) fn fixture_a_for_common_math_fun(n: usize, m: usize) -> Vec<f32>
{
    let mut a = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            a[m * i + j] = ((i as f32) - ((n as f32) / 2.0)) * 2.0 + (j as f32) - ((m as f32) / 2.0);
        }
    }
    a
}

pub(crate) fn fixture_a_b_for_common_math_fun(n1: usize, m1: usize, n2: usize, m2: usize) -> (Vec<f32>, Vec<f32>)
{
    let mut a = vec![0.0f32; n1 * m1];
    for i in 0..n1 {
        for j in 0..m1 {
            a[m1 * i + j] = ((i as f32) - ((n1 as f32) / 2.0)) * 2.0 + (j as f32) - ((m1 as f32) / 2.0);
        }
    }
    let mut b = vec![0.0f32; n2 * m2];
    for i in 0..n2 {
        for j in 0..m2 {
            b[m2 * i + j] = (i as f32) - ((n2 as f32) / 2.0) + ((j as f32) - ((m2 as f32) / 2.0)) * 2.0;
        }
    }
    (a, b)
}

pub(crate) fn fixture_a_b_for_pow(n1: usize, m1: usize, n2: usize, m2: usize) -> (Vec<f32>, Vec<f32>)
{
    let mut a = vec![0.0f32; n1 * m1];
    for i in 0..n1 {
        for j in 0..m1 {
            a[m1 * i + j] = ((i as f32) - ((n1 as f32) / 2.0)) * 2.0 + (j as f32) - ((m1 as f32) / 2.0);
        }
    }
    let mut b = vec![0.0f32; n2 * m2];
    for i in 0..n2 {
        for j in 0..m2 {
            b[m2 * i + j] = (i as f32) + (j as f32) * 2.0 + 1.0;
        }
    }
    (a, b)
}

pub(crate) fn fixture_a_for_log(n: usize, m: usize) -> Vec<f32>
{
    let mut a = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            a[m * i + j] = (i as f32) * 2.0 + (j as f32) + 1.0;
        }
    }
    a
}

pub(crate) fn fixture_a_for_asin_or_acos(n: usize, m: usize) -> Vec<f32>
{
    let mut a = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            a[m * i + j] = (((i as f32) - ((n as f32) / 2.0)) * 2.0 + (j as f32) - ((m as f32) / 2.0)) / ((n as f32) * 2.0 + (m as f32));
        }
    }
    a
}

pub(crate) fn fixture_a_for_rounding(n: usize, m: usize) -> Vec<f32>
{
    let mut a = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            a[m * i + j] = (((i as f32) - ((n as f32) / 2.0)) * 2.0 + (j as f32) - ((m as f32) / 2.0)) / 3.0;
        }
    }
    a
}

pub(crate) fn backend_alloc_and_store_zeros(backend: &dyn Backend, n: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store_zeros(n)?;
    let mut elems = vec![0.0f32; n];
    backend.load(&a, &mut elems)?;
    Ok(elems)
}

pub(crate) fn backend_alloc_and_store(backend: &dyn Backend, elems: &[f32]) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(elems)?;
    let mut elems2 = vec![0.0f32; elems.len()];
    backend.load(&a, &mut elems2)?;
    Ok(elems2)
}

pub(crate) fn backend_store(backend: &dyn Backend, elems: &[f32]) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store_zeros(elems.len())?;
    backend.store(&a, elems)?;
    let mut elems2 = vec![0.0f32; elems.len()];
    backend.load(&a, &mut elems2)?;
    Ok(elems2)
}

pub(crate) fn backend_copy(backend: &dyn Backend, elems: &[f32]) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(elems)?;
    let b = backend.alloc_and_store_zeros(elems.len())?;
    backend.copy(&a, &b)?;
    let mut elems2 = vec![0.0f32; elems.len()];
    backend.load(&b, &mut elems2)?;
    Ok(elems2)
}

pub(crate) fn backend_transpose_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.transpose_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_add_a_b(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.add_a_b(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_add_at_b(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.add_at_b(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_add_a_bt(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.add_a_bt(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_add_at_bt(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.add_at_bt(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_sub_a_b(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.sub_a_b(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_sub_at_b(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.sub_at_b(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_sub_a_bt(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.sub_a_bt(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_sub_at_bt(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.sub_at_bt(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_mul_a_b(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize, l: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.mul_a_b(&a, &b, &c, n, m, l)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_mul_at_b(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize, l: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.mul_at_b(&a, &b, &c, n, m, l)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_mul_a_bt(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize, l: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.mul_a_bt(&a, &b, &c, n, m, l)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_mul_at_bt(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize, l: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.mul_at_bt(&a, &b, &c, n, m, l)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_mul_a_b_for_elems(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.mul_a_b_for_elems(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_mul_at_b_for_elems(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.mul_at_b_for_elems(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_mul_a_bt_for_elems(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.mul_a_bt_for_elems(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_mul_at_bt_for_elems(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.mul_at_bt_for_elems(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_div_a_b_for_elems(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.div_a_b_for_elems(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_div_at_b_for_elems(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.div_at_b_for_elems(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_div_a_bt_for_elems(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.div_a_bt_for_elems(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_div_at_bt_for_elems(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.div_at_bt_for_elems(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_add_a_b_for_scalar(backend: &dyn Backend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.add_a_b_for_scalar(&a, b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_add_at_b_for_scalar(backend: &dyn Backend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.add_at_b_for_scalar(&a, b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_sub_a_b_for_scalar(backend: &dyn Backend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.sub_a_b_for_scalar(&a, b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_sub_at_b_for_scalar(backend: &dyn Backend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.sub_at_b_for_scalar(&a, b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_rsub_a_b_for_scalar(backend: &dyn Backend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.rsub_a_b_for_scalar(&a, b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_rsub_at_b_for_scalar(backend: &dyn Backend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.rsub_at_b_for_scalar(&a, b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_mul_a_b_for_scalar(backend: &dyn Backend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.mul_a_b_for_scalar(&a, b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_mul_at_b_for_scalar(backend: &dyn Backend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.mul_at_b_for_scalar(&a, b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_div_a_b_for_scalar(backend: &dyn Backend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.div_a_b_for_scalar(&a, b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_div_at_b_for_scalar(backend: &dyn Backend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.div_at_b_for_scalar(&a, b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_rdiv_a_b_for_scalar(backend: &dyn Backend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.rdiv_a_b_for_scalar(&a, b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_rdiv_at_b_for_scalar(backend: &dyn Backend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.rdiv_at_b_for_scalar(&a, b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_sigmoid_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.sigmoid_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_sigmoid_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.sigmoid_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_tanh_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.tanh_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_tanh_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.tanh_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_swish_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.swish_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_swish_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.swish_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_softmax_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.softmax_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_softmax_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.softmax_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_sqrt_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.sqrt_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_sqrt_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.sqrt_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_repeat_col_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.repeat_col_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_repeat_row_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.repeat_row_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_sigmoid_a_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    backend.sigmoid_a(&a, &a, n, m)?;
    let mut a_elems2 = vec![0.0f32; n * m];
    backend.load(&a, &mut a_elems2)?;
    Ok(a_elems2)
}

pub(crate) fn backend_add_a_a_c(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.add_a_b(&a, &a, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_add_a_b_a(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    backend.add_a_b(&a, &b, &a, n, m)?;
    let mut a_elems2 = vec![0.0f32; n * m];
    backend.load(&a, &mut a_elems2)?;
    Ok(a_elems2)
}

pub(crate) fn backend_add_a_b_b(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    backend.add_a_b(&a, &b, &b, n, m)?;
    let mut b_elems2 = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems2)?;
    Ok(b_elems2)
}

pub(crate) fn backend_add_a_a_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    backend.add_a_b(&a, &a, &a, n, m)?;
    let mut a_elems2 = vec![0.0f32; n * m];
    backend.load(&a, &mut a_elems2)?;
    Ok(a_elems2)
}

pub(crate) fn backend_mul_a_a_c(backend: &dyn Backend, a_elems: &[f32], n: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let c = backend.alloc_and_store_zeros(n * n)?;
    backend.mul_a_b(&a, &a, &c, n, n, n)?;
    let mut c_elems = vec![0.0f32; n * n];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_abs_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.abs_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_abs_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.abs_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_pow_a_b(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.pow_a_b(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_pow_at_b(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.pow_at_b(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_pow_a_bt(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.pow_a_bt(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_pow_at_bt(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.pow_at_bt(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_pow_a_b_for_scalar(backend: &dyn Backend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.pow_a_b_for_scalar(&a, b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_pow_at_b_for_scalar(backend: &dyn Backend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.pow_at_b_for_scalar(&a, b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_rpow_a_b_for_scalar(backend: &dyn Backend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.rpow_a_b_for_scalar(&a, b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_rpow_at_b_for_scalar(backend: &dyn Backend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.rpow_at_b_for_scalar(&a, b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_exp_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.exp_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_exp_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.exp_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_ln_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.ln_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_ln_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.ln_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_log2_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.log2_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_log2_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.log2_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_log10_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.log10_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_log10_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.log10_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_sin_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.sin_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_sin_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.sin_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_cos_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.cos_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_cos_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.cos_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_tan_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.tan_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_tan_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.tan_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_asin_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.asin_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_asin_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.asin_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_acos_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.acos_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_acos_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.acos_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_atan_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.atan_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_atan_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.atan_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_atan2_a_b(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.atan2_a_b(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_atan2_at_b(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.atan2_at_b(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_atan2_a_bt(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.atan2_a_bt(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_atan2_at_bt(backend: &dyn Backend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store(b_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.atan2_at_bt(&a, &b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_atan2_a_b_for_scalar(backend: &dyn Backend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.atan2_a_b_for_scalar(&a, b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_atan2_at_b_for_scalar(backend: &dyn Backend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.atan2_at_b_for_scalar(&a, b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_ratan2_a_b_for_scalar(backend: &dyn Backend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.ratan2_a_b_for_scalar(&a, b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_ratan2_at_b_for_scalar(backend: &dyn Backend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let c = backend.alloc_and_store_zeros(n * m)?;
    backend.ratan2_at_b_for_scalar(&a, b, &c, n, m)?;
    let mut c_elems = vec![0.0f32; n * m];
    backend.load(&c, &mut c_elems)?;
    Ok(c_elems)
}

pub(crate) fn backend_sinh_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.sinh_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_sinh_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.sinh_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_cosh_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.cosh_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_cosh_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.cosh_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}


pub(crate) fn backend_asinh_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.asinh_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_asinh_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.asinh_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_acosh_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.acosh_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_acosh_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.acosh_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_atanh_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.atanh_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_atanh_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.atanh_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_signum_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.signum_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_signum_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.signum_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_ceil_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.ceil_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_ceil_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.ceil_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_floor_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.floor_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_floor_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.floor_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_round_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.round_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_round_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.round_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_trunc_a(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.trunc_a(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

pub(crate) fn backend_trunc_at(backend: &dyn Backend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = backend.alloc_and_store(a_elems)?;
    let b = backend.alloc_and_store_zeros(n * m)?;
    backend.trunc_at(&a, &b, n, m)?;
    let mut b_elems = vec![0.0f32; n * m];
    backend.load(&b, &mut b_elems)?;
    Ok(b_elems)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_create_matrix_and_set_zeros(frontend: &Frontend, n: usize, m: usize) -> Result<(Matrix, Vec<f32>)>
{
    let a = frontend.create_matrix_and_set_zeros(n, m)?;
    let elems = frontend.elems_and_transpose_flag(&a)?.0;
    Ok((a, elems))
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_create_matrix_and_set_elems(frontend: &Frontend, n: usize, m: usize, elems: &[f32]) -> Result<(Matrix, Vec<f32>)>
{
    let a = frontend.create_matrix_and_set_elems(n, m, elems)?;
    let elems = frontend.elems_and_transpose_flag(&a)?.0;
    Ok((a, elems))
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_set_elems(frontend: &Frontend, elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.set_elems(&a, elems)?;
    Ok(frontend.elems_and_transpose_flag(&a)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_copy(frontend: &Frontend, elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.copy(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_really_transpose(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.really_transpose(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_add_for_a_b(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_elems(n, m, b_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.add(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_add_for_at_b(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_elems(n, m, b_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.add(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_add_for_a_bt(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_elems(m, n, b_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.add(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_add_for_at_bt(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_elems(m, n, b_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.add(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_sub_for_a_b(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_elems(n, m, b_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.sub(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_sub_for_at_b(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_elems(n, m, b_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.sub(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_sub_for_a_bt(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_elems(m, n, b_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.sub(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_sub_for_at_bt(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_elems(m, n, b_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.sub(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_mul_for_a_b(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize, l: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, l, a_elems)?;
    let b = frontend.create_matrix_and_set_elems(l, m, b_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.mul(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_mul_for_at_b(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize, l: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(l, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_elems(l, m, b_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.mul(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_mul_for_a_bt(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize, l: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, l, a_elems)?;
    let b = frontend.create_matrix_and_set_elems(m, l, b_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.mul(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_mul_for_at_bt(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize, l: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(l, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_elems(m, l, b_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.mul(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_mul_elems_for_a_b(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_elems(n, m, b_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.mul_elems(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_mul_elems_for_at_b(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_elems(n, m, b_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.mul_elems(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_mul_elems_for_a_bt(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_elems(m, n, b_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.mul_elems(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_mul_elems_for_at_bt(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_elems(m, n, b_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.mul_elems(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_div_elems_for_a_b(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_elems(n, m, b_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.div_elems(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_div_elems_for_at_b(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_elems(n, m, b_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.div_elems(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_div_elems_for_a_bt(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_elems(m, n, b_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.div_elems(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_div_elems_for_at_bt(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_elems(m, n, b_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.div_elems(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_add_for_scalar_and_a_b(frontend: &Frontend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.add_for_scalar(&a, b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_add_for_scalar_and_at_b(frontend: &Frontend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.add_for_scalar(&a, b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_sub_for_scalar_and_a_b(frontend: &Frontend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.sub_for_scalar(&a, b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_sub_for_scalar_and_at_b(frontend: &Frontend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.sub_for_scalar(&a, b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_rsub_for_scalar_and_a_b(frontend: &Frontend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.rsub_for_scalar(&a, b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_rsub_for_scalar_and_at_b(frontend: &Frontend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.rsub_for_scalar(&a, b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_mul_for_scalar_and_a_b(frontend: &Frontend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.mul_for_scalar(&a, b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_mul_for_scalar_and_at_b(frontend: &Frontend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.mul_for_scalar(&a, b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_div_for_scalar_and_a_b(frontend: &Frontend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.div_for_scalar(&a, b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_div_for_scalar_and_at_b(frontend: &Frontend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.div_for_scalar(&a, b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_rdiv_for_scalar_and_a_b(frontend: &Frontend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.rdiv_for_scalar(&a, b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_rdiv_for_scalar_and_at_b(frontend: &Frontend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.rdiv_for_scalar(&a, b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_sigmoid_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.sigmoid(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_sigmoid_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.sigmoid(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_tanh_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.tanh(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_tanh_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.tanh(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_swish_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.swish(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_swish_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.swish(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_softmax_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.softmax(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_softmax_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.softmax(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_sqrt_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.sqrt(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_sqrt_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.sqrt(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_repeat_for_col_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, 1, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.repeat(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_repeat_for_row_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(1, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.repeat(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_abs_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.abs(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_abs_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.abs(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_pow_for_a_b(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_elems(n, m, b_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.pow(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_pow_for_at_b(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_elems(n, m, b_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.pow(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_pow_for_a_bt(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_elems(m, n, b_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.pow(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_pow_for_at_bt(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_elems(m, n, b_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.pow(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_pow_for_scalar_and_a_b(frontend: &Frontend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.pow_for_scalar(&a, b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_pow_for_scalar_and_at_b(frontend: &Frontend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.pow_for_scalar(&a, b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_rpow_for_scalar_and_a_b(frontend: &Frontend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.rpow_for_scalar(&a, b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_rpow_for_scalar_and_at_b(frontend: &Frontend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.rpow_for_scalar(&a, b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_exp_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.exp(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_exp_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.exp(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_ln_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.ln(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_ln_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.ln(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_log2_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.log2(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_log2_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.log2(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_log10_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.log10(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_log10_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.log10(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_sin_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.sin(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_sin_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.sin(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_cos_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.cos(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_cos_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.cos(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_tan_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.tan(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_tan_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.tan(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_asin_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.asin(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_asin_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.asin(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_acos_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.acos(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_acos_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.acos(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_atan_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.atan(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_atan_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.atan(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_atan2_for_a_b(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_elems(n, m, b_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.atan2(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_atan2_for_at_b(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_elems(n, m, b_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.atan2(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_atan2_for_a_bt(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_elems(m, n, b_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.atan2(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_atan2_for_at_bt(frontend: &Frontend, a_elems: &[f32], b_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_elems(m, n, b_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.atan2(&a, &b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_atan2_for_scalar_and_a_b(frontend: &Frontend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.atan2_for_scalar(&a, b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_atan2_for_scalar_and_at_b(frontend: &Frontend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.atan2_for_scalar(&a, b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_ratan2_for_scalar_and_a_b(frontend: &Frontend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.ratan2_for_scalar(&a, b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_ratan2_for_scalar_and_at_b(frontend: &Frontend, a_elems: &[f32], b: f32, n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let c = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.ratan2_for_scalar(&a, b, &c)?;
    Ok(frontend.elems_and_transpose_flag(&c)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_sinh_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.sinh(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_sinh_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.sinh(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_cosh_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.cosh(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_cosh_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.cosh(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_asinh_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.asinh(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_asinh_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.asinh(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_acosh_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.acosh(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_acosh_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.acosh(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_atanh_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.atanh(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_atanh_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.atanh(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_signum_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.signum(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_signum_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.signum(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_ceil_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.ceil(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_ceil_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.ceil(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_floor_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.floor(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_floor_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.floor(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_round_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.round(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_round_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.round(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_trunc_for_a(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(n, m, a_elems)?;
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.trunc(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

#[cfg(not(feature = "test_only_backend"))]
pub(crate) fn frontend_trunc_for_at(frontend: &Frontend, a_elems: &[f32], n: usize, m: usize) -> Result<Vec<f32>>
{
    let a = frontend.create_matrix_and_set_elems(m, n, a_elems)?.transpose();
    let b = frontend.create_matrix_and_set_zeros(n, m)?;
    frontend.trunc(&a, &b)?;
    Ok(frontend.elems_and_transpose_flag(&b)?.0)
}

pub(crate) fn expected_transpose_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i];
        }
    }
    b
}

pub(crate) fn expected_add_a_b(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[m * i + j] + b[m * i + j];
        }
    }
    c
}

pub(crate) fn expected_add_at_b(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[n * j + i] + b[m * i + j];
        }
    }
    c
}

pub(crate) fn expected_add_a_bt(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[m * i + j] + b[n * j + i];
        }
    }
    c
}

pub(crate) fn expected_add_at_bt(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[n * j + i] + b[n * j + i];
        }
    }
    c
}

pub(crate) fn expected_sub_a_b(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[m * i + j] - b[m * i + j];
        }
    }
    c
}

pub(crate) fn expected_sub_at_b(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[n * j + i] - b[m * i + j];
        }
    }
    c
}

pub(crate) fn expected_sub_a_bt(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[m * i + j] - b[n * j + i];
        }
    }
    c
}

pub(crate) fn expected_sub_at_bt(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[n * j + i] - b[n * j + i];
        }
    }
    c
}

pub(crate) fn expected_mul_a_b(a: &[f32], b: &[f32], n: usize, m: usize, l: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = 0.0f32;
            for k in 0..l {
                c[m * i + j] += a[l * i + k] * b[m * k + j];
            }
        }
    }
    c
}

pub(crate) fn expected_mul_at_b(a: &[f32], b: &[f32], n: usize, m: usize, l: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = 0.0f32;
            for k in 0..l {
                c[m * i + j] += a[n * k + i] * b[m * k + j];
            }
        }
    }
    c
}

pub(crate) fn expected_mul_a_bt(a: &[f32], b: &[f32], n: usize, m: usize, l: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = 0.0f32;
            for k in 0..l {
                c[m * i + j] += a[l * i + k] * b[l * j + k];
            }
        }
    }
    c
}

pub(crate) fn expected_mul_at_bt(a: &[f32], b: &[f32], n: usize, m: usize, l: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = 0.0f32;
            for k in 0..l {
                c[m * i + j] += a[n * k + i] * b[l * j + k];
            }
        }
    }
    c
}

pub(crate) fn expected_mul_a_b_for_elems(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[m * i + j] * b[m * i + j];
        }
    }
    c
}

pub(crate) fn expected_mul_at_b_for_elems(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[n * j + i] * b[m * i + j];
        }
    }
    c
}

pub(crate) fn expected_mul_a_bt_for_elems(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[m * i + j] * b[n * j + i];
        }
    }
    c
}

pub(crate) fn expected_mul_at_bt_for_elems(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[n * j + i] * b[n * j + i];
        }
    }
    c
}

pub(crate) fn expected_div_a_b_for_elems(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[m * i + j] / b[m * i + j];
        }
    }
    c
}

pub(crate) fn expected_div_at_b_for_elems(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[n * j + i] / b[m * i + j];
        }
    }
    c
}

pub(crate) fn expected_div_a_bt_for_elems(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[m * i + j] / b[n * j + i];
        }
    }
    c
}

pub(crate) fn expected_div_at_bt_for_elems(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[n * j + i] / b[n * j + i];
        }
    }
    c
}

pub(crate) fn expected_add_a_b_for_scalar(a: &[f32], b: f32, n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[m * i + j] + b;
        }
    }
    c
}

pub(crate) fn expected_add_at_b_for_scalar(a: &[f32], b: f32, n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[n * j + i] + b;
        }
    }
    c
}

pub(crate) fn expected_sub_a_b_for_scalar(a: &[f32], b: f32, n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[m * i + j] - b;
        }
    }
    c
}

pub(crate) fn expected_sub_at_b_for_scalar(a: &[f32], b: f32, n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[n * j + i] - b;
        }
    }
    c
}

pub(crate) fn expected_rsub_a_b_for_scalar(a: &[f32], b: f32, n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = b - a[m * i + j];
        }
    }
    c
}

pub(crate) fn expected_rsub_at_b_for_scalar(a: &[f32], b: f32, n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = b - a[n * j + i];
        }
    }
    c
}

pub(crate) fn expected_mul_a_b_for_scalar(a: &[f32], b: f32, n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[m * i + j] * b;
        }
    }
    c
}

pub(crate) fn expected_mul_at_b_for_scalar(a: &[f32], b: f32, n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[n * j + i] * b;
        }
    }
    c
}

pub(crate) fn expected_div_a_b_for_scalar(a: &[f32], b: f32, n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[m * i + j] / b;
        }
    }
    c
}

pub(crate) fn expected_div_at_b_for_scalar(a: &[f32], b: f32, n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[n * j + i] / b;
        }
    }
    c
}

pub(crate) fn expected_rdiv_a_b_for_scalar(a: &[f32], b: f32, n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = b / a[m * i + j];
        }
    }
    c
}

pub(crate) fn expected_rdiv_at_b_for_scalar(a: &[f32], b: f32, n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = b / a[n * j + i];
        }
    }
    c
}

pub(crate) fn expected_sigmoid_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = 1.0f32 / (1.0f32 + (-a[m * i + j]).exp());
        }
    }
    b
}

pub(crate) fn expected_sigmoid_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = 1.0f32 / (1.0f32 + (-a[n * j + i]).exp());
        }
    }
    b
}

pub(crate) fn expected_tanh_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].tanh();
        }
    }
    b
}

pub(crate) fn expected_tanh_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].tanh();
        }
    }
    b
}

pub(crate) fn expected_swish_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j] / (1.0f32 + (-a[m * i + j]).exp());
        }
    }
    b
}

pub(crate) fn expected_swish_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i] / (1.0f32 + (-a[n * j + i]).exp());
        }
    }
    b
}

pub(crate) fn expected_softmax_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += a[m * k + j].exp();
            }
            b[m * i + j] = a[m * i + j].exp() / sum;
        }
    }
    b
}

pub(crate) fn expected_softmax_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += a[n * j + k].exp();
            }
            b[m * i + j] = a[n * j + i].exp() / sum;
        }
    }
    b
}

pub(crate) fn expected_sqrt_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].sqrt();
        }
    }
    b
}

pub(crate) fn expected_sqrt_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].sqrt();
        }
    }
    b
}

pub(crate) fn expected_repeat_col_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[i];
        }
    }
    b
}

pub(crate) fn expected_repeat_row_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[j];
        }
    }
    b
}

pub(crate) fn expected_abs_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].abs();
        }
    }
    b
}

pub(crate) fn expected_abs_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].abs();
        }
    }
    b
}

pub(crate) fn expected_pow_a_b(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[m * i + j].powf(b[m * i + j]);
        }
    }
    c
}

pub(crate) fn expected_pow_at_b(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[n * j + i].powf(b[m * i + j]);
        }
    }
    c
}

pub(crate) fn expected_pow_a_bt(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[m * i + j].powf(b[n * j + i]);
        }
    }
    c
}

pub(crate) fn expected_pow_at_bt(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[n * j + i].powf(b[n * j + i]);
        }
    }
    c
}

pub(crate) fn expected_pow_a_b_for_scalar(a: &[f32], b: f32, n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[m * i + j].powf(b);
        }
    }
    c
}

pub(crate) fn expected_pow_at_b_for_scalar(a: &[f32], b: f32, n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[n * j + i].powf(b);
        }
    }
    c
}

pub(crate) fn expected_rpow_a_b_for_scalar(a: &[f32], b: f32, n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = b.powf(a[m * i + j]);
        }
    }
    c
}

pub(crate) fn expected_rpow_at_b_for_scalar(a: &[f32], b: f32, n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = b.powf(a[n * j + i]);
        }
    }
    c
}

pub(crate) fn expected_exp_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].exp();
        }
    }
    b
}

pub(crate) fn expected_exp_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].exp();
        }
    }
    b
}

pub(crate) fn expected_ln_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].ln();
        }
    }
    b
}

pub(crate) fn expected_ln_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].ln();
        }
    }
    b
}

pub(crate) fn expected_log2_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].log2();
        }
    }
    b
}

pub(crate) fn expected_log2_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].log2();
        }
    }
    b
}

pub(crate) fn expected_log10_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].log10();
        }
    }
    b
}

pub(crate) fn expected_log10_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].log10();
        }
    }
    b
}

pub(crate) fn expected_sin_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].sin();
        }
    }
    b
}

pub(crate) fn expected_sin_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].sin();
        }
    }
    b
}

pub(crate) fn expected_cos_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].cos();
        }
    }
    b
}

pub(crate) fn expected_cos_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].cos();
        }
    }
    b
}

pub(crate) fn expected_tan_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].tan();
        }
    }
    b
}

pub(crate) fn expected_tan_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].tan();
        }
    }
    b
}

pub(crate) fn expected_asin_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].asin();
        }
    }
    b
}

pub(crate) fn expected_asin_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].asin();
        }
    }
    b
}

pub(crate) fn expected_acos_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].acos();
        }
    }
    b
}

pub(crate) fn expected_acos_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].acos();
        }
    }
    b
}

pub(crate) fn expected_atan_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].atan();
        }
    }
    b
}

pub(crate) fn expected_atan_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].atan();
        }
    }
    b
}

pub(crate) fn expected_atan2_a_b(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[m * i + j].atan2(b[m * i + j]);
        }
    }
    c
}

pub(crate) fn expected_atan2_at_b(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[n * j + i].atan2(b[m * i + j]);
        }
    }
    c
}

pub(crate) fn expected_atan2_a_bt(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[m * i + j].atan2(b[n * j + i]);
        }
    }
    c
}

pub(crate) fn expected_atan2_at_bt(a: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[n * j + i].atan2(b[n * j + i]);
        }
    }
    c
}

pub(crate) fn expected_atan2_a_b_for_scalar(a: &[f32], b: f32, n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[m * i + j].atan2(b);
        }
    }
    c
}

pub(crate) fn expected_atan2_at_b_for_scalar(a: &[f32], b: f32, n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = a[n * j + i].atan2(b);
        }
    }
    c
}

pub(crate) fn expected_ratan2_a_b_for_scalar(a: &[f32], b: f32, n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = b.atan2(a[m * i + j]);
        }
    }
    c
}

pub(crate) fn expected_ratan2_at_b_for_scalar(a: &[f32], b: f32, n: usize, m: usize) -> Vec<f32>
{
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            c[m * i + j] = b.atan2(a[n * j + i]);
        }
    }
    c
}

pub(crate) fn expected_sinh_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].sinh();
        }
    }
    b
}

pub(crate) fn expected_sinh_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].sinh();
        }
    }
    b
}

pub(crate) fn expected_cosh_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].cosh();
        }
    }
    b
}

pub(crate) fn expected_cosh_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].cosh();
        }
    }
    b
}

pub(crate) fn expected_asinh_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].asinh();
        }
    }
    b
}

pub(crate) fn expected_asinh_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].asinh();
        }
    }
    b
}

pub(crate) fn expected_acosh_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].acosh();
        }
    }
    b
}

pub(crate) fn expected_acosh_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].acosh();
        }
    }
    b
}

pub(crate) fn expected_atanh_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].atanh();
        }
    }
    b
}

pub(crate) fn expected_atanh_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].atanh();
        }
    }
    b
}

pub(crate) fn expected_signum_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].signum();
        }
    }
    b
}

pub(crate) fn expected_signum_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].signum();
        }
    }
    b
}

pub(crate) fn expected_ceil_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].ceil();
        }
    }
    b
}

pub(crate) fn expected_ceil_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].ceil();
        }
    }
    b
}

pub(crate) fn expected_floor_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].floor();
        }
    }
    b
}

pub(crate) fn expected_floor_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].floor();
        }
    }
    b
}

pub(crate) fn expected_round_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].round();
        }
    }
    b
}

pub(crate) fn expected_round_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].round();
        }
    }
    b
}

pub(crate) fn expected_trunc_a(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[m * i + j].trunc();
        }
    }
    b
}

pub(crate) fn expected_trunc_at(a: &[f32], n: usize, m: usize) -> Vec<f32>
{
    let mut b = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            b[m * i + j] = a[n * j + i].trunc();
        }
    }
    b
}
