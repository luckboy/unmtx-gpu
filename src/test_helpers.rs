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
