//
// Copyright (c) 2025 Łukasz Szpakowski
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
use crate::test_helpers::*;
use super::*;

#[test]
fn test_cuda_backend_new_creates_backend()
{
    match CudaBackend::new() {
        Ok(_) => assert!(true),
        Err(_) => assert!(false),
        //Err(err) => {
        //    println!("{}", err);
        //    assert!(false)
        //},
    }
}

#[test]
fn test_cuda_backend_alloc_allocates_backend_array()
{
    match CudaBackend::new() {
        Ok(backend) => {
            match unsafe { backend.alloc(2 * 3) } {
                Ok(_) => assert!(true),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_alloc_and_store_zeros_allocates_backend_array_with_zeros()
{
    match CudaBackend::new() {
        Ok(backend) => {
            match backend_alloc_and_store_zeros(&backend, 2 * 3) {
                Ok(elems) => assert_eq!(vec![0.0f32; 2 * 3], elems),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_alloc_and_store_allocates_backend_array_with_elements()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a(2, 3);
            match backend_alloc_and_store(&backend, a.as_slice()) {
                Ok(elems) => assert_eq!(a, elems),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_store_stores_to_backend_array()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a(2, 3);
            match backend_store(&backend, a.as_slice()) {
                Ok(elems) => assert_eq!(a, elems),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_copy_copies_from_backend_array_to_backend_array()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a(2, 3);
            match backend_copy(&backend, a.as_slice()) {
                Ok(elems) => assert_eq!(a, elems),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_transpose_a_transposes_backend_array()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a(3, 2);
            match backend_transpose_a(&backend, a.as_slice(), 2, 3) {
                Ok(c) => assert_eq!(expected_transpose_a(a.as_slice(), 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_add_a_b_adds_backend_arrays()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let (a, b) = fixture_a_b(2, 3, 2, 3);
            match backend_add_a_b(&backend, a.as_slice(), b.as_slice(), 2, 3) {
                Ok(c) => assert_eq!(expected_add_a_b(a.as_slice(), b.as_slice(), 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_add_at_b_adds_backend_arrays()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let (a, b) = fixture_a_b(3, 2, 2, 3);
            match backend_add_at_b(&backend, a.as_slice(), b.as_slice(), 2, 3) {
                Ok(c) => assert_eq!(expected_add_at_b(a.as_slice(), b.as_slice(), 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_add_a_bt_adds_backend_arrays()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let (a, b) = fixture_a_b(2, 3, 3, 2);
            match backend_add_a_bt(&backend, a.as_slice(), b.as_slice(), 2, 3) {
                Ok(c) => assert_eq!(expected_add_a_bt(a.as_slice(), b.as_slice(), 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_add_at_bt_adds_backend_arrays()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let (a, b) = fixture_a_b(3, 2, 3, 2);
            match backend_add_at_bt(&backend, a.as_slice(), b.as_slice(), 2, 3) {
                Ok(c) => assert_eq!(expected_add_at_bt(a.as_slice(), b.as_slice(), 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_sub_a_b_subtracts_backend_arrays()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let (a, b) = fixture_a_b(2, 3, 2, 3);
            match backend_sub_a_b(&backend, a.as_slice(), b.as_slice(), 2, 3) {
                Ok(c) => assert_eq!(expected_sub_a_b(a.as_slice(), b.as_slice(), 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_sub_at_b_subtracts_backend_arrays()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let (a, b) = fixture_a_b(3, 2, 2, 3);
            match backend_sub_at_b(&backend, a.as_slice(), b.as_slice(), 2, 3) {
                Ok(c) => assert_eq!(expected_sub_at_b(a.as_slice(), b.as_slice(), 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_sub_a_bt_subtracts_backend_arrays()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let (a, b) = fixture_a_b(2, 3, 3, 2);
            match backend_sub_a_bt(&backend, a.as_slice(), b.as_slice(), 2, 3) {
                Ok(c) => assert_eq!(expected_sub_a_bt(a.as_slice(), b.as_slice(), 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_sub_at_bt_subtracts_backend_arrays()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let (a, b) = fixture_a_b(3, 2, 3, 2);
            match backend_sub_at_bt(&backend, a.as_slice(), b.as_slice(), 2, 3) {
                Ok(c) => assert_eq!(expected_sub_at_bt(a.as_slice(), b.as_slice(), 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_mul_a_b_multiplies_backend_arrays()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let (a1, b1) = fixture_a_b(2, 4, 4, 3);
            match backend_mul_a_b(&backend, a1.as_slice(), b1.as_slice(), 2, 3, 4) {
                Ok(c1) => assert_eq!(expected_mul_a_b(a1.as_slice(), b1.as_slice(), 2, 3, 4), c1),
                Err(_) => assert!(false),
            }
            let (a2, b2) = fixture_a_b(2, 5, 5, 3);
            match backend_mul_a_b(&backend, a2.as_slice(), b2.as_slice(), 2, 3, 5) {
                Ok(c2) => assert_eq!(expected_mul_a_b(a2.as_slice(), b2.as_slice(), 2, 3, 5), c2),
                Err(_) => assert!(false),
            }
            let (a3, b3) = fixture_a_b(2, 6, 6, 3);
            match backend_mul_a_b(&backend, a3.as_slice(), b3.as_slice(), 2, 3, 6) {
                Ok(c3) => assert_eq!(expected_mul_a_b(a3.as_slice(), b3.as_slice(), 2, 3, 6), c3),
                Err(_) => assert!(false),
            }
            let (a4, b4) = fixture_a_b(2, 7, 7, 3);
            match backend_mul_a_b(&backend, a4.as_slice(), b4.as_slice(), 2, 3, 7) {
                Ok(c4) => assert_eq!(expected_mul_a_b(a4.as_slice(), b4.as_slice(), 2, 3, 7), c4),
                Err(_) => assert!(false),
            }
            let (a5, b5) = fixture_a_b(2, 8, 8, 3);
            match backend_mul_a_b(&backend, a5.as_slice(), b5.as_slice(), 2, 3, 8) {
                Ok(c5) => assert_eq!(expected_mul_a_b(a5.as_slice(), b5.as_slice(), 2, 3, 8), c5),
                Err(_) => assert!(false),
            }
             let (a6, b6) = fixture_a_b(8, 8, 8, 8);
            match backend_mul_a_b(&backend, a6.as_slice(), b6.as_slice(), 8, 8, 8) {
                Ok(c6) => assert_eq!(expected_mul_a_b(a6.as_slice(), b6.as_slice(), 8, 8, 8), c6),
                Err(_) => assert!(false),
            }
       },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_mul_at_b_multiplies_backend_arrays()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let (a1, b1) = fixture_a_b(4, 2, 4, 3);
            match backend_mul_at_b(&backend, a1.as_slice(), b1.as_slice(), 2, 3, 4) {
                Ok(c1) => assert_eq!(expected_mul_at_b(a1.as_slice(), b1.as_slice(), 2, 3, 4), c1),
                Err(_) => assert!(false),
            }
            let (a2, b2) = fixture_a_b(5, 2, 5, 3);
            match backend_mul_at_b(&backend, a2.as_slice(), b2.as_slice(), 2, 3, 5) {
                Ok(c2) => assert_eq!(expected_mul_at_b(a2.as_slice(), b2.as_slice(), 2, 3, 5), c2),
                Err(_) => assert!(false),
            }
            let (a3, b3) = fixture_a_b(6, 2, 6, 3);
            match backend_mul_at_b(&backend, a3.as_slice(), b3.as_slice(), 2, 3, 6) {
                Ok(c3) => assert_eq!(expected_mul_at_b(a3.as_slice(), b3.as_slice(), 2, 3, 6), c3),
                Err(_) => assert!(false),
            }
            let (a4, b4) = fixture_a_b(7, 2, 7, 3);
            match backend_mul_at_b(&backend, a4.as_slice(), b4.as_slice(), 2, 3, 7) {
                Ok(c4) => assert_eq!(expected_mul_at_b(a4.as_slice(), b4.as_slice(), 2, 3, 7), c4),
                Err(_) => assert!(false),
            }
            let (a5, b5) = fixture_a_b(8, 2, 8, 3);
            match backend_mul_at_b(&backend, a5.as_slice(), b5.as_slice(), 2, 3, 8) {
                Ok(c5) => assert_eq!(expected_mul_at_b(a5.as_slice(), b5.as_slice(), 2, 3, 8), c5),
                Err(_) => assert!(false),
            }
            let (a6, b6) = fixture_a_b(8, 8, 8, 8);
            match backend_mul_at_b(&backend, a6.as_slice(), b6.as_slice(), 8, 8, 8) {
                Ok(c6) => assert_eq!(expected_mul_at_b(a6.as_slice(), b6.as_slice(), 8, 8, 8), c6),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_mul_a_bt_multiplies_backend_arrays()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let (a1, b1) = fixture_a_b(2, 4, 3, 4);
            match backend_mul_a_bt(&backend, a1.as_slice(), b1.as_slice(), 2, 3, 4) {
                Ok(c1) => assert_eq!(expected_mul_a_bt(a1.as_slice(), b1.as_slice(), 2, 3, 4), c1),
                Err(_) => assert!(false),
            }
            let (a2, b2) = fixture_a_b(2, 5, 3, 5);
            match backend_mul_a_bt(&backend, a2.as_slice(), b2.as_slice(), 2, 3, 5) {
                Ok(c2) => assert_eq!(expected_mul_a_bt(a2.as_slice(), b2.as_slice(), 2, 3, 5), c2),
                Err(_) => assert!(false),
            }
            let (a3, b3) = fixture_a_b(2, 6, 3, 6);
            match backend_mul_a_bt(&backend, a3.as_slice(), b3.as_slice(), 2, 3, 6) {
                Ok(c3) => assert_eq!(expected_mul_a_bt(a3.as_slice(), b3.as_slice(), 2, 3, 6), c3),
                Err(_) => assert!(false),
            }
            let (a4, b4) = fixture_a_b(2, 7, 3, 7);
            match backend_mul_a_bt(&backend, a4.as_slice(), b4.as_slice(), 2, 3, 7) {
                Ok(c4) => assert_eq!(expected_mul_a_bt(a4.as_slice(), b4.as_slice(), 2, 3, 7), c4),
                Err(_) => assert!(false),
            }
            let (a5, b5) = fixture_a_b(2, 8, 3, 8);
            match backend_mul_a_bt(&backend, a5.as_slice(), b5.as_slice(), 2, 3, 8) {
                Ok(c5) => assert_eq!(expected_mul_a_bt(a5.as_slice(), b5.as_slice(), 2, 3, 8), c5),
                Err(_) => assert!(false),
            }
            let (a6, b6) = fixture_a_b(8, 8, 8, 8);
            match backend_mul_a_bt(&backend, a6.as_slice(), b6.as_slice(), 8, 8, 8) {
                Ok(c6) => assert_eq!(expected_mul_a_bt(a6.as_slice(), b6.as_slice(), 8, 8, 8), c6),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_mul_at_bt_multiplies_backend_arrays()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let (a1, b1) = fixture_a_b(4, 2, 3, 4);
            match backend_mul_at_bt(&backend, a1.as_slice(), b1.as_slice(), 2, 3, 4) {
                Ok(c1) => assert_eq!(expected_mul_at_bt(a1.as_slice(), b1.as_slice(), 2, 3, 4), c1),
                Err(_) => assert!(false),
            }
            let (a2, b2) = fixture_a_b(5, 2, 3, 5);
            match backend_mul_at_bt(&backend, a2.as_slice(), b2.as_slice(), 2, 3, 5) {
                Ok(c2) => assert_eq!(expected_mul_at_bt(a2.as_slice(), b2.as_slice(), 2, 3, 5), c2),
                Err(_) => assert!(false),
            }
            let (a3, b3) = fixture_a_b(6, 2, 3, 6);
            match backend_mul_at_bt(&backend, a3.as_slice(), b3.as_slice(), 2, 3, 6) {
                Ok(c3) => assert_eq!(expected_mul_at_bt(a3.as_slice(), b3.as_slice(), 2, 3, 6), c3),
                Err(_) => assert!(false),
            }
            let (a4, b4) = fixture_a_b(7, 2, 3, 7);
            match backend_mul_at_bt(&backend, a4.as_slice(), b4.as_slice(), 2, 3, 7) {
                Ok(c4) => assert_eq!(expected_mul_at_bt(a4.as_slice(), b4.as_slice(), 2, 3, 7), c4),
                Err(_) => assert!(false),
            }
            let (a5, b5) = fixture_a_b(8, 2, 3, 8);
            match backend_mul_at_bt(&backend, a5.as_slice(), b5.as_slice(), 2, 3, 8) {
                Ok(c5) => assert_eq!(expected_mul_at_bt(a5.as_slice(), b5.as_slice(), 2, 3, 8), c5),
                Err(_) => assert!(false),
            }
            let (a6, b6) = fixture_a_b(8, 8, 8, 8);
            match backend_mul_at_bt(&backend, a6.as_slice(), b6.as_slice(), 8, 8, 8) {
                Ok(c6) => assert_eq!(expected_mul_at_bt(a6.as_slice(), b6.as_slice(), 8, 8, 8), c6),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_mul_a_b_for_elems_multiplies_backend_arrays()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let (a, b) = fixture_a_b(2, 3, 2, 3);
            match backend_mul_a_b_for_elems(&backend, a.as_slice(), b.as_slice(), 2, 3) {
                Ok(c) => assert_eq!(expected_mul_a_b_for_elems(a.as_slice(), b.as_slice(), 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_mul_at_b_for_elems_multiplies_backend_arrays()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let (a, b) = fixture_a_b(3, 2, 2, 3);
            match backend_mul_at_b_for_elems(&backend, a.as_slice(), b.as_slice(), 2, 3) {
                Ok(c) => assert_eq!(expected_mul_at_b_for_elems(a.as_slice(), b.as_slice(), 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_mul_a_bt_for_elems_multiplies_backend_arrays()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let (a, b) = fixture_a_b(2, 3, 3, 2);
            match backend_mul_a_bt_for_elems(&backend, a.as_slice(), b.as_slice(), 2, 3) {
                Ok(c) => assert_eq!(expected_mul_a_bt_for_elems(a.as_slice(), b.as_slice(), 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_mul_at_bt_for_elems_multiplies_backend_arrays()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let (a, b) = fixture_a_b(2, 3, 3, 2);
            match backend_mul_at_bt_for_elems(&backend, a.as_slice(), b.as_slice(), 2, 3) {
                Ok(c) => assert_eq!(expected_mul_at_bt_for_elems(a.as_slice(), b.as_slice(), 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_div_a_b_for_elems_divides_backend_arrays()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let (a, b) = fixture_a_b(2, 3, 2, 3);
            match backend_div_a_b_for_elems(&backend, a.as_slice(), b.as_slice(), 2, 3) {
                Ok(c) => {
                    let expected_c = expected_div_a_b_for_elems(a.as_slice(), b.as_slice(), 2, 3);
                    assert_eq!(expected_c.len(), c.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c[i] - c[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_div_at_b_for_elems_divides_backend_arrays()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let (a, b) = fixture_a_b(3, 2, 2, 3);
            match backend_div_at_b_for_elems(&backend, a.as_slice(), b.as_slice(), 2, 3) {
                Ok(c) => {
                    let expected_c = expected_div_at_b_for_elems(a.as_slice(), b.as_slice(), 2, 3);
                    assert_eq!(expected_c.len(), c.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c[i] - c[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_div_a_bt_for_elems_divides_backend_arrays()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let (a, b) = fixture_a_b(2, 3, 3, 2);
            match backend_div_a_bt_for_elems(&backend, a.as_slice(), b.as_slice(), 2, 3) {
                Ok(c) => {
                    let expected_c = expected_div_a_bt_for_elems(a.as_slice(), b.as_slice(), 2, 3);
                    assert_eq!(expected_c.len(), c.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c[i] - c[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_div_at_bt_for_elems_divides_backend_arrays()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let (a, b) = fixture_a_b(3, 2, 2, 3);
            match backend_div_at_bt_for_elems(&backend, a.as_slice(), b.as_slice(), 2, 3) {
                Ok(c) => {
                    let expected_c = expected_div_at_bt_for_elems(a.as_slice(), b.as_slice(), 2, 3);
                    assert_eq!(expected_c.len(), c.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c[i] - c[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_add_a_b_for_scalar_adds_backend_array_and_scalar()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a(2, 3);
            match backend_add_a_b_for_scalar(&backend, a.as_slice(), 10.5, 2, 3) {
                Ok(c) => assert_eq!(expected_add_a_b_for_scalar(a.as_slice(), 10.5, 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_add_at_b_for_scalar_adds_backend_array_and_scalar()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a(3, 2);
            match backend_add_at_b_for_scalar(&backend, a.as_slice(), 10.5, 2, 3) {
                Ok(c) => assert_eq!(expected_add_at_b_for_scalar(a.as_slice(), 10.5, 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_sub_a_b_for_scalar_subtracts_backend_array_and_scalar()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a(2, 3);
            match backend_sub_a_b_for_scalar(&backend, a.as_slice(), 10.5, 2, 3) {
                Ok(c) => assert_eq!(expected_sub_a_b_for_scalar(a.as_slice(), 10.5, 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_sub_at_b_for_scalar_subtracts_backend_array_and_scalar()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a(3, 2);
            match backend_sub_at_b_for_scalar(&backend, a.as_slice(), 10.5, 2, 3) {
                Ok(c) => assert_eq!(expected_sub_at_b_for_scalar(a.as_slice(), 10.5, 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_rsub_a_b_for_scalar_subtracts_backend_array_and_scalar()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a(2, 3);
            match backend_rsub_a_b_for_scalar(&backend, a.as_slice(), 10.5, 2, 3) {
                Ok(c) => assert_eq!(expected_rsub_a_b_for_scalar(a.as_slice(), 10.5, 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_rsub_at_b_for_scalar_subtracts_backend_array_and_scalar()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a(3, 2);
            match backend_rsub_at_b_for_scalar(&backend, a.as_slice(), 10.5, 2, 3) {
                Ok(c) => assert_eq!(expected_rsub_at_b_for_scalar(a.as_slice(), 10.5, 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_mul_a_b_for_scalar_multiplies_backend_array_and_scalar()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a(2, 3);
            match backend_mul_a_b_for_scalar(&backend, a.as_slice(), 10.5, 2, 3) {
                Ok(c) => assert_eq!(expected_mul_a_b_for_scalar(a.as_slice(), 10.5, 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_mul_at_b_for_scalar_multiplies_backend_array_and_scalar()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a(3, 2);
            match backend_mul_at_b_for_scalar(&backend, a.as_slice(), 10.5, 2, 3) {
                Ok(c) => assert_eq!(expected_mul_at_b_for_scalar(a.as_slice(), 10.5, 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_div_a_b_for_scalar_divides_backend_array_and_scalar()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a(2, 3);
            match backend_div_a_b_for_scalar(&backend, a.as_slice(), 10.5, 2, 3) {
                Ok(c) => {
                    let expected_c = expected_div_a_b_for_scalar(a.as_slice(), 10.5, 2, 3);
                    assert_eq!(expected_c.len(), c.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c[i] - c[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_div_at_b_for_scalar_divides_backend_array_and_scalar()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a(3, 2);
            match backend_div_at_b_for_scalar(&backend, a.as_slice(), 10.5, 2, 3) {
                Ok(c) => {
                    let expected_c = expected_div_at_b_for_scalar(a.as_slice(), 10.5, 2, 3);
                    assert_eq!(expected_c.len(), c.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c[i] - c[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_rdiv_a_b_for_scalar_divides_backend_array_and_scalar()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a(2, 3);
            match backend_rdiv_a_b_for_scalar(&backend, a.as_slice(), 10.5, 2, 3) {
                Ok(c) => {
                    let expected_c = expected_rdiv_a_b_for_scalar(a.as_slice(), 10.5, 2, 3);
                    assert_eq!(expected_c.len(), c.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c[i] - c[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_rdiv_at_b_for_scalar_divides_backend_array_and_scalar()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a(3, 2);
            match backend_rdiv_at_b_for_scalar(&backend, a.as_slice(), 10.5, 2, 3) {
                Ok(c) => {
                    let expected_c = expected_rdiv_at_b_for_scalar(a.as_slice(), 10.5, 2, 3);
                    assert_eq!(expected_c.len(), c.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c[i] - c[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_sigmoid_a_calculates_sigmoid_for_backend_array()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a_for_activation_fun(2, 3);
            match backend_sigmoid_a(&backend, a.as_slice(), 2, 3) {
                Ok(b) => {
                    let expected_b = expected_sigmoid_a(a.as_slice(), 2, 3);
                    assert_eq!(expected_b.len(), b.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b[i] - b[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_sigmoid_at_calculates_sigmoid_for_backend_array()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a_for_activation_fun(3, 2);
            match backend_sigmoid_at(&backend, a.as_slice(), 2, 3) {
                Ok(b) => {
                    let expected_b = expected_sigmoid_at(a.as_slice(), 2, 3);
                    assert_eq!(expected_b.len(), b.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b[i] - b[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_tanh_a_calculates_tanh_for_backend_array()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a_for_activation_fun(2, 3);
            match backend_tanh_a(&backend, a.as_slice(), 2, 3) {
                Ok(b) => {
                    let expected_b = expected_tanh_a(a.as_slice(), 2, 3);
                    assert_eq!(expected_b.len(), b.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b[i] - b[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_tanh_at_calculates_tanh_for_backend_array()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a_for_activation_fun(3, 2);
            match backend_tanh_at(&backend, a.as_slice(), 2, 3) {
                Ok(b) => {
                    let expected_b = expected_tanh_at(a.as_slice(), 2, 3);
                    assert_eq!(expected_b.len(), b.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b[i] - b[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_softmax_a_calculates_softmax_for_backend_array()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a1 = fixture_a(4, 3);
            match backend_softmax_a(&backend, a1.as_slice(), 4, 3) {
                Ok(b1) => {
                    let expected_b1 = expected_softmax_a(a1.as_slice(), 4, 3);
                    assert_eq!(expected_b1.len(), b1.len());
                    for i in 0usize..(4usize * 3usize) {
                        assert!((expected_b1[i] - b1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a(5, 3);
            match backend_softmax_a(&backend, a2.as_slice(), 5, 3) {
                Ok(b2) => {
                    let expected_b2 = expected_softmax_a(a2.as_slice(), 5, 3);
                    assert_eq!(expected_b2.len(), b2.len());
                    for i in 0usize..(5usize * 3usize) {
                        assert!((expected_b2[i] - b2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a3 = fixture_a(6, 3);
            match backend_softmax_a(&backend, a3.as_slice(), 6, 3) {
                Ok(b3) => {
                    let expected_b3 = expected_softmax_a(a3.as_slice(), 6, 3);
                    assert_eq!(expected_b3.len(), b3.len());
                    for i in 0usize..(6usize * 3usize) {
                        assert!((expected_b3[i] - b3[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a4 = fixture_a(7, 3);
            match backend_softmax_a(&backend, a4.as_slice(), 7, 3) {
                Ok(b4) => {
                    let expected_b4 = expected_softmax_a(a4.as_slice(), 7, 3);
                    assert_eq!(expected_b4.len(), b4.len());
                    for i in 0usize..(7usize * 3usize) {
                        assert!((expected_b4[i] - b4[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a5 = fixture_a(8, 3);
            match backend_softmax_a(&backend, a5.as_slice(), 8, 3) {
                Ok(b5) => {
                    let expected_b5 = expected_softmax_a(a5.as_slice(), 8, 3);
                    assert_eq!(expected_b5.len(), b5.len());
                    for i in 0usize..(8usize * 3usize) {
                        assert!((expected_b5[i] - b5[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a6 = fixture_a(4, 1);
            match backend_softmax_a(&backend, a6.as_slice(), 4, 1) {
                Ok(b6) => {
                    let expected_b6 = expected_softmax_a(a6.as_slice(), 4, 1);
                    assert_eq!(expected_b6.len(), b6.len());
                    for i in 0usize..4usize {
                        assert!((expected_b6[i] - b6[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a7 = fixture_a(1, 4);
            match backend_softmax_a(&backend, a7.as_slice(), 1, 4) {
                Ok(b7) => {
                    let expected_b7 = expected_softmax_a(a6.as_slice(), 1, 4);
                    assert_eq!(expected_b7.len(), b7.len());
                    for i in 0usize..4usize {
                        assert!((expected_b7[i] - b7[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_softmax_at_calculates_softmax_for_backend_array()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a1 = fixture_a(3, 4);
            match backend_softmax_at(&backend, a1.as_slice(), 4, 3) {
                Ok(b1) => {
                    let expected_b1 = expected_softmax_at(a1.as_slice(), 4, 3);
                    assert_eq!(expected_b1.len(), b1.len());
                    for i in 0usize..(4usize * 3usize) {
                        assert!((expected_b1[i] - b1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a(3, 5);
            match backend_softmax_at(&backend, a2.as_slice(), 5, 3) {
                Ok(b2) => {
                    let expected_b2 = expected_softmax_at(a2.as_slice(), 5, 3);
                    assert_eq!(expected_b2.len(), b2.len());
                    for i in 0usize..(5usize * 3usize) {
                        assert!((expected_b2[i] - b2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a3 = fixture_a(3, 6);
            match backend_softmax_at(&backend, a3.as_slice(), 6, 3) {
                Ok(b3) => {
                    let expected_b3 = expected_softmax_at(a3.as_slice(), 6, 3);
                    assert_eq!(expected_b3.len(), b3.len());
                    for i in 0usize..(6usize * 3usize) {
                        assert!((expected_b3[i] - b3[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a4 = fixture_a(3, 7);
            match backend_softmax_at(&backend, a4.as_slice(), 7, 3) {
                Ok(b4) => {
                    let expected_b4 = expected_softmax_at(a4.as_slice(), 7, 3);
                    assert_eq!(expected_b4.len(), b4.len());
                    for i in 0usize..(7usize * 3usize) {
                        assert!((expected_b4[i] - b4[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a5 = fixture_a(3, 8);
            match backend_softmax_at(&backend, a5.as_slice(), 8, 3) {
                Ok(b5) => {
                    let expected_b5 = expected_softmax_at(a5.as_slice(), 8, 3);
                    assert_eq!(expected_b5.len(), b5.len());
                    for i in 0usize..(8usize * 3usize) {
                        assert!((expected_b5[i] - b5[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a6 = fixture_a(1, 4);
            match backend_softmax_at(&backend, a6.as_slice(), 4, 1) {
                Ok(b6) => {
                    let expected_b6 = expected_softmax_at(a6.as_slice(), 4, 1);
                    assert_eq!(expected_b6.len(), b6.len());
                    for i in 0usize..4usize {
                        assert!((expected_b6[i] - b6[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a7 = fixture_a(4, 1);
            match backend_softmax_at(&backend, a7.as_slice(), 1, 4) {
                Ok(b7) => {
                    let expected_b7 = expected_softmax_at(a6.as_slice(), 1, 4);
                    assert_eq!(expected_b7.len(), b7.len());
                    for i in 0usize..4usize {
                        assert!((expected_b7[i] - b7[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_repeat_col_a_repeats_column_for_backend_array()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a(2, 1);
            match backend_repeat_col_a(&backend, a.as_slice(), 2, 3) {
                Ok(b) => assert_eq!(expected_repeat_col_a(a.as_slice(), 2, 3), b),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_repeat_row_a_repeats_row_for_backend_array()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a(1, 3);
            match backend_repeat_row_a(&backend, a.as_slice(), 2, 3) {
                Ok(b) => assert_eq!(expected_repeat_row_a(a.as_slice(), 2, 3), b),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_sigmoid_a_uses_one_backend_array_for_a_a()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a(2, 3);
            match backend_sigmoid_a_a(&backend, a.as_slice(), 2, 3) {
                Ok(b) => {
                    let expected_b = expected_sigmoid_a(a.as_slice(), 2, 3);
                    assert_eq!(expected_b.len(), b.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b[i] - b[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_add_a_b_uses_two_backend_arrays_for_a_a_c()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a(2, 3);
            match backend_add_a_a_c(&backend, a.as_slice(), 2, 3) {
                Ok(c) => assert_eq!(expected_add_a_b(a.as_slice(), a.as_slice(), 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_add_a_b_uses_two_backend_arrays_for_a_b_a()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let (a, b) = fixture_a_b(2, 3, 2, 3);
            match backend_add_a_b_a(&backend, a.as_slice(), b.as_slice(), 2, 3) {
                Ok(c) => assert_eq!(expected_add_a_b(a.as_slice(), b.as_slice(), 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_add_a_b_uses_two_backend_arrays_for_a_b_b()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let (a, b) = fixture_a_b(2, 3, 2, 3);
            match backend_add_a_b_b(&backend, a.as_slice(), b.as_slice(), 2, 3) {
                Ok(c) => assert_eq!(expected_add_a_b(a.as_slice(), b.as_slice(), 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_add_a_b_uses_one_backend_array_for_a_a_a()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a(2, 3);
            match backend_add_a_a_a(&backend, a.as_slice(), 2, 3) {
                Ok(c) => assert_eq!(expected_add_a_b(a.as_slice(), a.as_slice(), 2, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_cuda_backend_mul_a_b_uses_two_backend_arrays_for_a_a_c()
{
    match CudaBackend::new() {
        Ok(backend) => {
            let a = fixture_a(3, 3);
            match backend_mul_a_a_c(&backend, a.as_slice(), 3) {
                Ok(c) => assert_eq!(expected_mul_a_b(a.as_slice(), a.as_slice(), 3, 3, 3), c),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}
