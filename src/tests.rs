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
fn test_frontend_new_creates_frontend()
{
    match Frontend::new() {
        Ok(_) => assert!(true),
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_create_matrix_creates_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            match unsafe { frontend.create_matrix(2, 3) } {
                Ok(a) => {
                    assert_eq!(2, a.row_count());
                    assert_eq!(3, a.col_count());
                    assert_eq!(false, a.is_transposed());
                }, 
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_create_matrix_and_set_zeros_creates_matrix_with_zeros()
{
    match Frontend::new() {
        Ok(frontend) => {
            match frontend_create_matrix_and_set_zeros(&frontend, 2, 3) {
                Ok((a, elems)) => {
                    assert_eq!(2, a.row_count());
                    assert_eq!(3, a.col_count());
                    assert_eq!(false, a.is_transposed());
                    assert_eq!(vec![0.0f32; 2 * 3], elems);
                }, 
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_create_matrix_and_set_zeros_creates_matrix_with_elements()
{
    match Frontend::new() {
        Ok(frontend) => {
            let expected_elems = fixture_a(2, 3);
            match frontend_create_matrix_and_set_elems(&frontend, 2, 3, expected_elems.as_slice()) {
                Ok((a, elems)) => {
                    assert_eq!(2, a.row_count());
                    assert_eq!(3, a.col_count());
                    assert_eq!(false, a.is_transposed());
                    assert_eq!(expected_elems, elems);
                }, 
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_set_elems_sets_elements()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a = fixture_a(2, 3);
            match frontend_set_elems(&frontend, a.as_slice(), 2, 3) {
                Ok(b) => assert_eq!(a, b),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_copy_copies_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a = fixture_a(2, 3);
            match frontend_copy(&frontend, a.as_slice(), 2, 3) {
                Ok(b) => assert_eq!(a, b),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_really_transpose_transposes_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a = fixture_a(2, 3);
            match frontend_really_transpose(&frontend, a.as_slice(), 2, 3) {
                Ok(b) => assert_eq!(expected_transpose_a(a.as_slice(), 2, 3), b),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_add_adds_matrices()
{
    match Frontend::new() {
        Ok(frontend) => {
            let (a1, b1) = fixture_a_b(2, 3, 2, 3);
            match frontend_add_for_a_b(&frontend, a1.as_slice(), b1.as_slice(), 2, 3) {
                Ok(c1) => assert_eq!(expected_add_a_b(a1.as_slice(), b1.as_slice(), 2, 3), c1),
                Err(_) => assert!(false),
            }
            let (a2, b2) = fixture_a_b(3, 2, 2, 3);
            match frontend_add_for_at_b(&frontend, a2.as_slice(), b2.as_slice(), 2, 3) {
                Ok(c2) => assert_eq!(expected_add_at_b(a2.as_slice(), b2.as_slice(), 2, 3), c2),
                Err(_) => assert!(false),
            }
            let (a3, b3) = fixture_a_b(2, 3, 3, 2);
            match frontend_add_for_a_bt(&frontend, a3.as_slice(), b3.as_slice(), 2, 3) {
                Ok(c3) => assert_eq!(expected_add_a_bt(a3.as_slice(), b3.as_slice(), 2, 3), c3),
                Err(_) => assert!(false),
            }
            let (a4, b4) = fixture_a_b(3, 2, 3, 2);
            match frontend_add_for_at_bt(&frontend, a4.as_slice(), b4.as_slice(), 2, 3) {
                Ok(c4) => assert_eq!(expected_add_at_bt(a4.as_slice(), b4.as_slice(), 2, 3), c4),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_sub_subtracts_matrices()
{
    match Frontend::new() {
        Ok(frontend) => {
            let (a1, b1) = fixture_a_b(2, 3, 2, 3);
            match frontend_sub_for_a_b(&frontend, a1.as_slice(), b1.as_slice(), 2, 3) {
                Ok(c1) => assert_eq!(expected_sub_a_b(a1.as_slice(), b1.as_slice(), 2, 3), c1),
                Err(_) => assert!(false),
            }
            let (a2, b2) = fixture_a_b(3, 2, 2, 3);
            match frontend_sub_for_at_b(&frontend, a2.as_slice(), b2.as_slice(), 2, 3) {
                Ok(c2) => assert_eq!(expected_sub_at_b(a2.as_slice(), b2.as_slice(), 2, 3), c2),
                Err(_) => assert!(false),
            }
            let (a3, b3) = fixture_a_b(2, 3, 3, 2);
            match frontend_sub_for_a_bt(&frontend, a3.as_slice(), b3.as_slice(), 2, 3) {
                Ok(c3) => assert_eq!(expected_sub_a_bt(a3.as_slice(), b3.as_slice(), 2, 3), c3),
                Err(_) => assert!(false),
            }
            let (a4, b4) = fixture_a_b(3, 2, 3, 2);
            match frontend_sub_for_at_bt(&frontend, a4.as_slice(), b4.as_slice(), 2, 3) {
                Ok(c4) => assert_eq!(expected_sub_at_bt(a4.as_slice(), b4.as_slice(), 2, 3), c4),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_mul_mutliplies_matrices()
{
    match Frontend::new() {
        Ok(frontend) => {
            let (a1, b1) = fixture_a_b(2, 4, 4, 3);
            match frontend_mul_for_a_b(&frontend, a1.as_slice(), b1.as_slice(), 2, 3, 4) {
                Ok(c1) => assert_eq!(expected_mul_a_b(a1.as_slice(), b1.as_slice(), 2, 3, 4), c1),
                Err(_) => assert!(false),
            }
            let (a2, b2) = fixture_a_b(4, 2, 4, 3);
            match frontend_mul_for_at_b(&frontend, a2.as_slice(), b2.as_slice(), 2, 3, 4) {
                Ok(c2) => assert_eq!(expected_mul_at_b(a2.as_slice(), b2.as_slice(), 2, 3, 4), c2),
                Err(_) => assert!(false),
            }
            let (a3, b3) = fixture_a_b(2, 4, 3, 4);
            match frontend_mul_for_a_bt(&frontend, a3.as_slice(), b3.as_slice(), 2, 3, 4) {
                Ok(c3) => assert_eq!(expected_mul_a_bt(a3.as_slice(), b3.as_slice(), 2, 3, 4), c3),
                Err(_) => assert!(false),
            }
            let (a4, b4) = fixture_a_b(4, 2, 3, 4);
            match frontend_mul_for_at_bt(&frontend, a4.as_slice(), b4.as_slice(), 2, 3, 4) {
                Ok(c4) => assert_eq!(expected_mul_at_bt(a4.as_slice(), b4.as_slice(), 2, 3, 4), c4),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_mul_elems_multiples_matrices()
{
    match Frontend::new() {
        Ok(frontend) => {
            let (a1, b1) = fixture_a_b(2, 3, 2, 3);
            match frontend_mul_elems_for_a_b(&frontend, a1.as_slice(), b1.as_slice(), 2, 3) {
                Ok(c1) => assert_eq!(expected_mul_a_b_for_elems(a1.as_slice(), b1.as_slice(), 2, 3), c1),
                Err(_) => assert!(false),
            }
            let (a2, b2) = fixture_a_b(3, 2, 2, 3);
            match frontend_mul_elems_for_at_b(&frontend, a2.as_slice(), b2.as_slice(), 2, 3) {
                Ok(c2) => assert_eq!(expected_mul_at_b_for_elems(a2.as_slice(), b2.as_slice(), 2, 3), c2),
                Err(_) => assert!(false),
            }
            let (a3, b3) = fixture_a_b(2, 3, 3, 2);
            match frontend_mul_elems_for_a_bt(&frontend, a3.as_slice(), b3.as_slice(), 2, 3) {
                Ok(c3) => assert_eq!(expected_mul_a_bt_for_elems(a3.as_slice(), b3.as_slice(), 2, 3), c3),
                Err(_) => assert!(false),
            }
            let (a4, b4) = fixture_a_b(3, 2, 3, 2);
            match frontend_mul_elems_for_at_bt(&frontend, a4.as_slice(), b4.as_slice(), 2, 3) {
                Ok(c4) => assert_eq!(expected_mul_at_bt_for_elems(a4.as_slice(), b4.as_slice(), 2, 3), c4),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_div_elems_divides_matrices()
{
    match Frontend::new() {
        Ok(frontend) => {
            let (a1, b1) = fixture_a_b(2, 3, 2, 3);
            match frontend_div_elems_for_a_b(&frontend, a1.as_slice(), b1.as_slice(), 2, 3) {
                Ok(c1) => {
                    let expected_c1 = expected_div_a_b_for_elems(a1.as_slice(), b1.as_slice(), 2, 3);
                    assert_eq!(expected_c1.len(), c1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c1[i] - c1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let (a2, b2) = fixture_a_b(3, 2, 2, 3);
            match frontend_div_elems_for_at_b(&frontend, a2.as_slice(), b2.as_slice(), 2, 3) {
                Ok(c2) => {
                    let expected_c2 = expected_div_at_b_for_elems(a2.as_slice(), b2.as_slice(), 2, 3);
                    assert_eq!(expected_c2.len(), c2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c2[i] - c2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let (a3, b3) = fixture_a_b(2, 3, 3, 2);
            match frontend_div_elems_for_a_bt(&frontend, a3.as_slice(), b3.as_slice(), 2, 3) {
                Ok(c3) => {
                    let expected_c3 = expected_div_a_bt_for_elems(a3.as_slice(), b3.as_slice(), 2, 3);
                    assert_eq!(expected_c3.len(), c3.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c3[i] - c3[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let (a4, b4) = fixture_a_b(3, 2, 3, 2);
            match frontend_div_elems_for_at_bt(&frontend, a4.as_slice(), b4.as_slice(), 2, 3) {
                Ok(c4) => {
                    let expected_c4 = expected_div_at_bt_for_elems(a4.as_slice(), b4.as_slice(), 2, 3);
                    assert_eq!(expected_c4.len(), c4.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c4[i] - c4[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_add_for_scalar_adds_matrix_and_scalar()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a(2, 3);
            match frontend_add_for_scalar_and_a_b(&frontend, a1.as_slice(), 10.5, 2, 3) {
                Ok(c1) => assert_eq!(expected_add_a_b_for_scalar(a1.as_slice(), 10.5, 2, 3), c1),
                Err(_) => assert!(false),
            }
            let a2 = fixture_a(3, 2);
            match frontend_add_for_scalar_and_at_b(&frontend, a2.as_slice(), 10.5, 2, 3) {
                Ok(c2) => assert_eq!(expected_add_at_b_for_scalar(a2.as_slice(), 10.5, 2, 3), c2),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_sub_for_scalar_subtracts_matrix_and_scalar()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a(2, 3);
            match frontend_sub_for_scalar_and_a_b(&frontend, a1.as_slice(), 10.5, 2, 3) {
                Ok(c1) => assert_eq!(expected_sub_a_b_for_scalar(a1.as_slice(), 10.5, 2, 3), c1),
                Err(_) => assert!(false),
            }
            let a2 = fixture_a(3, 2);
            match frontend_sub_for_scalar_and_at_b(&frontend, a2.as_slice(), 10.5, 2, 3) {
                Ok(c2) => assert_eq!(expected_sub_at_b_for_scalar(a2.as_slice(), 10.5, 2, 3), c2),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_rsub_for_scalar_subtracts_matrix_and_scalar()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a(2, 3);
            match frontend_rsub_for_scalar_and_a_b(&frontend, a1.as_slice(), 10.5, 2, 3) {
                Ok(c1) => assert_eq!(expected_rsub_a_b_for_scalar(a1.as_slice(), 10.5, 2, 3), c1),
                Err(_) => assert!(false),
            }
            let a2 = fixture_a(3, 2);
            match frontend_rsub_for_scalar_and_at_b(&frontend, a2.as_slice(), 10.5, 2, 3) {
                Ok(c2) => assert_eq!(expected_rsub_at_b_for_scalar(a2.as_slice(), 10.5, 2, 3), c2),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_mul_for_scalar_multiplies_matrix_and_scalar()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a(2, 3);
            match frontend_mul_for_scalar_and_a_b(&frontend, a1.as_slice(), 10.5, 2, 3) {
                Ok(c1) => assert_eq!(expected_mul_a_b_for_scalar(a1.as_slice(), 10.5, 2, 3), c1),
                Err(_) => assert!(false),
            }
            let a2 = fixture_a(3, 2);
            match frontend_mul_for_scalar_and_at_b(&frontend, a2.as_slice(), 10.5, 2, 3) {
                Ok(c2) => assert_eq!(expected_mul_at_b_for_scalar(a2.as_slice(), 10.5, 2, 3), c2),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_div_for_scalar_divides_matrix_and_scalar()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a(2, 3);
            match frontend_div_for_scalar_and_a_b(&frontend, a1.as_slice(), 10.5, 2, 3) {
                Ok(c1) => {
                    let expected_c1 = expected_div_a_b_for_scalar(a1.as_slice(), 10.5, 2, 3);
                    assert_eq!(expected_c1.len(), c1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c1[i] - c1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a(3, 2);
            match frontend_div_for_scalar_and_at_b(&frontend, a2.as_slice(), 10.5, 2, 3) {
                Ok(c2) => {
                    let expected_c2 = expected_div_at_b_for_scalar(a2.as_slice(), 10.5, 2, 3);
                    assert_eq!(expected_c2.len(), c2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c2[i] - c2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_rdiv_for_scalar_divides_matrix_and_scalar()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a(2, 3);
            match frontend_rdiv_for_scalar_and_a_b(&frontend, a1.as_slice(), 10.5, 2, 3) {
                Ok(c1) => {
                    let expected_c1 = expected_rdiv_a_b_for_scalar(a1.as_slice(), 10.5, 2, 3);
                    assert_eq!(expected_c1.len(), c1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c1[i] - c1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a(3, 2);
            match frontend_rdiv_for_scalar_and_at_b(&frontend, a2.as_slice(), 10.5, 2, 3) {
                Ok(c2) => {
                    let expected_c2 = expected_rdiv_at_b_for_scalar(a2.as_slice(), 10.5, 2, 3);
                    assert_eq!(expected_c2.len(), c2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c2[i] - c2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_sigmoid_calculates_sigmoid_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_activation_fun(2, 3);
            match frontend_sigmoid_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => {
                    let expected_b1 = expected_sigmoid_a(a1.as_slice(), 2, 3);
                    assert_eq!(expected_b1.len(), b1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b1[i] - b1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_activation_fun(3, 2);
            match frontend_sigmoid_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => {
                    let expected_b2 = expected_sigmoid_at(a2.as_slice(), 2, 3);
                    assert_eq!(expected_b2.len(), b2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b2[i] - b2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_tanh_calculates_tanh_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_activation_fun(2, 3);
            match frontend_tanh_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => {
                    let expected_b1 = expected_tanh_a(a1.as_slice(), 2, 3);
                    assert_eq!(expected_b1.len(), b1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b1[i] - b1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_activation_fun(3, 2);
            match frontend_tanh_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => {
                    let expected_b2 = expected_tanh_at(a2.as_slice(), 2, 3);
                    assert_eq!(expected_b2.len(), b2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b2[i] - b2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_swish_calculates_swish_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_activation_fun(2, 3);
            match frontend_swish_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => {
                    let expected_b1 = expected_swish_a(a1.as_slice(), 2, 3);
                    assert_eq!(expected_b1.len(), b1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b1[i] - b1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_activation_fun(3, 2);
            match frontend_swish_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => {
                    let expected_b2 = expected_swish_at(a2.as_slice(), 2, 3);
                    assert_eq!(expected_b2.len(), b2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b2[i] - b2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_softmax_calculates_softmax_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a(2, 3);
            match frontend_softmax_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => {
                    let expected_b1 = expected_softmax_a(a1.as_slice(), 2, 3);
                    assert_eq!(expected_b1.len(), b1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b1[i] - b1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a(3, 2);
            match frontend_softmax_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => {
                    let expected_b2 = expected_softmax_at(a2.as_slice(), 2, 3);
                    assert_eq!(expected_b2.len(), b2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b2[i] - b2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_sqrt_calculates_sqrt_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a(2, 3);
            match frontend_sqrt_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => {
                    let expected_b1 = expected_sqrt_a(a1.as_slice(), 2, 3);
                    assert_eq!(expected_b1.len(), b1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b1[i] - b1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a(3, 2);
            match frontend_sqrt_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => {
                    let expected_b2 = expected_sqrt_at(a2.as_slice(), 2, 3);
                    assert_eq!(expected_b2.len(), b2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b2[i] - b2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_repeat_repeats_column_or_row()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a(2, 1);
            match frontend_repeat_for_col_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => assert_eq!(expected_repeat_col_a(a1.as_slice(), 2, 3), b1),
                Err(_) => assert!(false),
            }
            let a2 = fixture_a(1, 3);
            match frontend_repeat_for_row_a(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => assert_eq!(expected_repeat_row_a(a2.as_slice(), 2, 3), b2),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_abs_calculates_abs_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_common_math_fun(2, 3);
            match frontend_abs_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => assert_eq!(expected_abs_a(a1.as_slice(), 2, 3), b1),
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_common_math_fun(3, 2);
            match frontend_abs_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => assert_eq!(expected_abs_at(a2.as_slice(), 2, 3), b2),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_pow_calculates_pow_for_matrices()
{
    match Frontend::new() {
        Ok(frontend) => {
            let (a1, b1) = fixture_a_b_for_pow(2, 3, 2, 3);
            match frontend_pow_for_a_b(&frontend, a1.as_slice(), b1.as_slice(), 2, 3) {
                Ok(c1) => {
                    let expected_c1 = expected_pow_a_b(a1.as_slice(), b1.as_slice(), 2, 3);
                    assert_eq!(expected_c1.len(), c1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c1[i] - c1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let (a2, b2) = fixture_a_b_for_pow(3, 2, 2, 3);
            match frontend_pow_for_at_b(&frontend, a2.as_slice(), b2.as_slice(), 2, 3) {
                Ok(c2) => {
                    let expected_c2 = expected_pow_at_b(a2.as_slice(), b2.as_slice(), 2, 3);
                    assert_eq!(expected_c2.len(), c2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c2[i] - c2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let (a3, b3) = fixture_a_b_for_pow(2, 3, 3, 2);
            match frontend_pow_for_a_bt(&frontend, a3.as_slice(), b3.as_slice(), 2, 3) {
                Ok(c3) => {
                    let expected_c3 = expected_pow_a_bt(a3.as_slice(), b3.as_slice(), 2, 3);
                    assert_eq!(expected_c3.len(), c3.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c3[i] - c3[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let (a4, b4) = fixture_a_b_for_pow(3, 2, 3, 2);
            match frontend_pow_for_at_bt(&frontend, a4.as_slice(), b4.as_slice(), 2, 3) {
                Ok(c4) => {
                    let expected_c4 = expected_pow_at_bt(a4.as_slice(), b4.as_slice(), 2, 3);
                    assert_eq!(expected_c4.len(), c4.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c4[i] - c4[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_pow_for_scalar_calculates_pow_for_matrix_and_scalar()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_log(2, 3);
            match frontend_pow_for_scalar_and_a_b(&frontend, a1.as_slice(), 2.5, 2, 3) {
                Ok(c1) => {
                    let expected_c1 = expected_pow_a_b_for_scalar(a1.as_slice(), 2.5, 2, 3);
                    assert_eq!(expected_c1.len(), c1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c1[i] - c1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_log(3, 2);
            match frontend_pow_for_scalar_and_at_b(&frontend, a2.as_slice(), 2.5, 2, 3) {
                Ok(c2) => {
                    let expected_c2 = expected_pow_at_b_for_scalar(a2.as_slice(), 2.5, 2, 3);
                    assert_eq!(expected_c2.len(), c2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c2[i] - c2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_rpow_for_scalar_calculates_pow_for_matrix_and_scalar()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_common_math_fun(2, 3);
            match frontend_rpow_for_scalar_and_a_b(&frontend, a1.as_slice(), 10.5, 2, 3) {
                Ok(c1) => {
                    let expected_c1 = expected_rpow_a_b_for_scalar(a1.as_slice(), 10.5, 2, 3);
                    assert_eq!(expected_c1.len(), c1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c1[i] - c1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_common_math_fun(3, 2);
            match frontend_rpow_for_scalar_and_at_b(&frontend, a2.as_slice(), 10.5, 2, 3) {
                Ok(c2) => {
                    let expected_c2 = expected_rpow_at_b_for_scalar(a2.as_slice(), 10.5, 2, 3);
                    assert_eq!(expected_c2.len(), c2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c2[i] - c2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_exp_calculates_exp_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_common_math_fun(2, 3);
            match frontend_exp_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => {
                    let expected_b1 = expected_exp_a(a1.as_slice(), 2, 3);
                    assert_eq!(expected_b1.len(), b1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b1[i] - b1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_common_math_fun(3, 2);
            match frontend_exp_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => {
                    let expected_b2 = expected_exp_at(a2.as_slice(), 2, 3);
                    assert_eq!(expected_b2.len(), b2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b2[i] - b2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_ln_calculates_ln_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_log(2, 3);
            match frontend_ln_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => {
                    let expected_b1 = expected_ln_a(a1.as_slice(), 2, 3);
                    assert_eq!(expected_b1.len(), b1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b1[i] - b1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_log(3, 2);
            match frontend_ln_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => {
                    let expected_b2 = expected_ln_at(a2.as_slice(), 2, 3);
                    assert_eq!(expected_b2.len(), b2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b2[i] - b2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_log2_calculates_log2_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_log(2, 3);
            match frontend_log2_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => {
                    let expected_b1 = expected_log2_a(a1.as_slice(), 2, 3);
                    assert_eq!(expected_b1.len(), b1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b1[i] - b1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_log(3, 2);
            match frontend_log2_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => {
                    let expected_b2 = expected_log2_at(a2.as_slice(), 2, 3);
                    assert_eq!(expected_b2.len(), b2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b2[i] - b2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_log10_calculates_log10_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_log(2, 3);
            match frontend_log10_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => {
                    let expected_b1 = expected_log10_a(a1.as_slice(), 2, 3);
                    assert_eq!(expected_b1.len(), b1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b1[i] - b1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_log(3, 2);
            match frontend_log10_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => {
                    let expected_b2 = expected_log10_at(a2.as_slice(), 2, 3);
                    assert_eq!(expected_b2.len(), b2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b2[i] - b2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_sin_calculates_sin_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_common_math_fun(2, 3);
            match frontend_sin_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => {
                    let expected_b1 = expected_sin_a(a1.as_slice(), 2, 3);
                    assert_eq!(expected_b1.len(), b1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b1[i] - b1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_common_math_fun(3, 2);
            match frontend_sin_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => {
                    let expected_b2 = expected_sin_at(a2.as_slice(), 2, 3);
                    assert_eq!(expected_b2.len(), b2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b2[i] - b2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_cos_calculates_cos_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_common_math_fun(2, 3);
            match frontend_cos_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => {
                    let expected_b1 = expected_cos_a(a1.as_slice(), 2, 3);
                    assert_eq!(expected_b1.len(), b1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b1[i] - b1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_common_math_fun(3, 2);
            match frontend_cos_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => {
                    let expected_b2 = expected_cos_at(a2.as_slice(), 2, 3);
                    assert_eq!(expected_b2.len(), b2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b2[i] - b2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_tan_calculates_tan_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_common_math_fun(2, 3);
            match frontend_tan_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => {
                    let expected_b1 = expected_tan_a(a1.as_slice(), 2, 3);
                    assert_eq!(expected_b1.len(), b1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b1[i] - b1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_common_math_fun(3, 2);
            match frontend_tan_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => {
                    let expected_b2 = expected_tan_at(a2.as_slice(), 2, 3);
                    assert_eq!(expected_b2.len(), b2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b2[i] - b2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_asin_calculates_asin_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_asin_or_acos(2, 3);
            match frontend_asin_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => {
                    let expected_b1 = expected_asin_a(a1.as_slice(), 2, 3);
                    assert_eq!(expected_b1.len(), b1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b1[i] - b1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_asin_or_acos(3, 2);
            match frontend_asin_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => {
                    let expected_b2 = expected_asin_at(a2.as_slice(), 2, 3);
                    assert_eq!(expected_b2.len(), b2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b2[i] - b2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_acos_calculates_acos_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_asin_or_acos(2, 3);
            match frontend_acos_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => {
                    let expected_b1 = expected_acos_a(a1.as_slice(), 2, 3);
                    assert_eq!(expected_b1.len(), b1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b1[i] - b1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_asin_or_acos(3, 2);
            match frontend_acos_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => {
                    let expected_b2 = expected_acos_at(a2.as_slice(), 2, 3);
                    assert_eq!(expected_b2.len(), b2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b2[i] - b2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_atan_calculates_atan_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_common_math_fun(2, 3);
            match frontend_atan_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => {
                    let expected_b1 = expected_atan_a(a1.as_slice(), 2, 3);
                    assert_eq!(expected_b1.len(), b1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b1[i] - b1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_common_math_fun(3, 2);
            match frontend_atan_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => {
                    let expected_b2 = expected_atan_at(a2.as_slice(), 2, 3);
                    assert_eq!(expected_b2.len(), b2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b2[i] - b2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_atan2_calculates_atan2_for_matrices()
{
    match Frontend::new() {
        Ok(frontend) => {
            let (a1, b1) = fixture_a_b_for_common_math_fun(2, 3, 2, 3);
            match frontend_atan2_for_a_b(&frontend, a1.as_slice(), b1.as_slice(), 2, 3) {
                Ok(c1) => {
                    let expected_c1 = expected_atan2_a_b(a1.as_slice(), b1.as_slice(), 2, 3);
                    assert_eq!(expected_c1.len(), c1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c1[i] - c1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let (a2, b2) = fixture_a_b_for_common_math_fun(3, 2, 2, 3);
            match frontend_atan2_for_at_b(&frontend, a2.as_slice(), b2.as_slice(), 2, 3) {
                Ok(c2) => {
                    let expected_c2 = expected_atan2_at_b(a2.as_slice(), b2.as_slice(), 2, 3);
                    assert_eq!(expected_c2.len(), c2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c2[i] - c2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let (a3, b3) = fixture_a_b_for_common_math_fun(2, 3, 3, 2);
            match frontend_atan2_for_a_bt(&frontend, a3.as_slice(), b3.as_slice(), 2, 3) {
                Ok(c3) => {
                    let expected_c3 = expected_atan2_a_bt(a3.as_slice(), b3.as_slice(), 2, 3);
                    assert_eq!(expected_c3.len(), c3.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c3[i] - c3[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let (a4, b4) = fixture_a_b_for_common_math_fun(3, 2, 3, 2);
            match frontend_atan2_for_at_bt(&frontend, a4.as_slice(), b4.as_slice(), 2, 3) {
                Ok(c4) => {
                    let expected_c4 = expected_atan2_at_bt(a4.as_slice(), b4.as_slice(), 2, 3);
                    assert_eq!(expected_c4.len(), c4.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c4[i] - c4[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_atan2_for_scalar_calculates_atan2_for_matrix_and_scalar()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_log(2, 3);
            match frontend_atan2_for_scalar_and_a_b(&frontend, a1.as_slice(), 10.5, 2, 3) {
                Ok(c1) => {
                    let expected_c1 = expected_atan2_a_b_for_scalar(a1.as_slice(), 10.5, 2, 3);
                    assert_eq!(expected_c1.len(), c1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c1[i] - c1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_log(3, 2);
            match frontend_atan2_for_scalar_and_at_b(&frontend, a2.as_slice(), 10.5, 2, 3) {
                Ok(c2) => {
                    let expected_c2 = expected_atan2_at_b_for_scalar(a2.as_slice(), 10.5, 2, 3);
                    assert_eq!(expected_c2.len(), c2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c2[i] - c2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_ratan2_for_scalar_calculates_atan2_for_matrix_and_scalar()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_common_math_fun(2, 3);
            match frontend_ratan2_for_scalar_and_a_b(&frontend, a1.as_slice(), 10.5, 2, 3) {
                Ok(c1) => {
                    let expected_c1 = expected_ratan2_a_b_for_scalar(a1.as_slice(), 10.5, 2, 3);
                    assert_eq!(expected_c1.len(), c1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c1[i] - c1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_common_math_fun(3, 2);
            match frontend_ratan2_for_scalar_and_at_b(&frontend, a2.as_slice(), 10.5, 2, 3) {
                Ok(c2) => {
                    let expected_c2 = expected_ratan2_at_b_for_scalar(a2.as_slice(), 10.5, 2, 3);
                    assert_eq!(expected_c2.len(), c2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_c2[i] - c2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_sinh_calculates_sinh_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_common_math_fun(2, 3);
            match frontend_sinh_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => {
                    let expected_b1 = expected_sinh_a(a1.as_slice(), 2, 3);
                    assert_eq!(expected_b1.len(), b1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b1[i] - b1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_common_math_fun(3, 2);
            match frontend_sinh_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => {
                    let expected_b2 = expected_sinh_at(a2.as_slice(), 2, 3);
                    assert_eq!(expected_b2.len(), b2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b2[i] - b2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_cosh_calculates_cosh_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_common_math_fun(2, 3);
            match frontend_cosh_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => {
                    let expected_b1 = expected_cosh_a(a1.as_slice(), 2, 3);
                    assert_eq!(expected_b1.len(), b1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b1[i] - b1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_common_math_fun(3, 2);
            match frontend_cosh_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => {
                    let expected_b2 = expected_cosh_at(a2.as_slice(), 2, 3);
                    assert_eq!(expected_b2.len(), b2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b2[i] - b2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_asinh_calculates_asinh_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_common_math_fun(2, 3);
            match frontend_asinh_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => {
                    let expected_b1 = expected_asinh_a(a1.as_slice(), 2, 3);
                    assert_eq!(expected_b1.len(), b1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b1[i] - b1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_common_math_fun(3, 2);
            match frontend_asinh_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => {
                    let expected_b2 = expected_asinh_at(a2.as_slice(), 2, 3);
                    assert_eq!(expected_b2.len(), b2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b2[i] - b2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_acosh_calculates_acosh_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_log(2, 3);
            match frontend_acosh_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => {
                    let expected_b1 = expected_acosh_a(a1.as_slice(), 2, 3);
                    assert_eq!(expected_b1.len(), b1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b1[i] - b1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_log(3, 2);
            match frontend_acosh_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => {
                    let expected_b2 = expected_acosh_at(a2.as_slice(), 2, 3);
                    assert_eq!(expected_b2.len(), b2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b2[i] - b2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_atanh_calculates_atanh_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_asin_or_acos(2, 3);
            match frontend_atanh_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => {
                    let expected_b1 = expected_atanh_a(a1.as_slice(), 2, 3);
                    assert_eq!(expected_b1.len(), b1.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b1[i] - b1[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_asin_or_acos(3, 2);
            match frontend_atanh_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => {
                    let expected_b2 = expected_atanh_at(a2.as_slice(), 2, 3);
                    assert_eq!(expected_b2.len(), b2.len());
                    for i in 0usize..(2usize * 3usize) {
                        assert!((expected_b2[i] - b2[i]).abs() < 0.001);
                    }
                },
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_signum_calculates_signum_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_common_math_fun(2, 3);
            match frontend_signum_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => assert_eq!(expected_signum_a(a1.as_slice(), 2, 3), b1),
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_common_math_fun(3, 2);
            match frontend_signum_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => assert_eq!(expected_signum_at(a2.as_slice(), 2, 3), b2),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_ceil_calculates_ceil_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_rounding(2, 3);
            match frontend_ceil_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => assert_eq!(expected_ceil_a(a1.as_slice(), 2, 3), b1),
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_rounding(3, 2);
            match frontend_ceil_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => assert_eq!(expected_ceil_at(a2.as_slice(), 2, 3), b2),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_floor_calculates_floor_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_rounding(2, 3);
            match frontend_floor_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => assert_eq!(expected_floor_a(a1.as_slice(), 2, 3), b1),
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_rounding(3, 2);
            match frontend_floor_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => assert_eq!(expected_floor_at(a2.as_slice(), 2, 3), b2),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_round_calculates_round_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_rounding(2, 3);
            match frontend_round_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => assert_eq!(expected_round_a(a1.as_slice(), 2, 3), b1),
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_rounding(3, 2);
            match frontend_round_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => assert_eq!(expected_round_at(a2.as_slice(), 2, 3), b2),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_frontend_trunc_calculates_trunc_for_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a1 = fixture_a_for_rounding(2, 3);
            match frontend_trunc_for_a(&frontend, a1.as_slice(), 2, 3) {
                Ok(b1) => assert_eq!(expected_trunc_a(a1.as_slice(), 2, 3), b1),
                Err(_) => assert!(false),
            }
            let a2 = fixture_a_for_rounding(3, 2);
            match frontend_trunc_for_at(&frontend, a2.as_slice(), 2, 3) {
                Ok(b2) => assert_eq!(expected_trunc_at(a2.as_slice(), 2, 3), b2),
                Err(_) => assert!(false),
            }
        },
        Err(_) => assert!(false),
    }
}

#[test]
fn test_matrix_creates_matrix()
{
    let a = matrix![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ];
    assert_eq!(2, a.row_count());
    assert_eq!(3, a.col_count());
    assert_eq!(false, a.is_transposed());
    assert_eq!(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], a.elems());
}

#[test]
fn test_matrix_new_creates_matrix()
{
    let a = Matrix::new(2, 3);
    assert_eq!(2, a.row_count());
    assert_eq!(3, a.col_count());
    assert_eq!(false, a.is_transposed());
    assert_eq!(vec![0.0; 2 * 3], a.elems());
}

#[test]
fn test_matrix_new_with_elems_creates_matrix_with_elements()
{
    let elems = fixture_a(2, 3);
    let a = Matrix::new_with_elems(2, 3, elems.as_slice());
    assert_eq!(2, a.row_count());
    assert_eq!(3, a.col_count());
    assert_eq!(false, a.is_transposed());
    assert_eq!(elems, a.elems());
}

#[test]
fn test_matrix_copy_copies_matrix()
{
    let elems = fixture_a(2, 3);
    let a = Matrix::new_with_elems(2, 3, elems.as_slice());
    let b = a.copy();
    assert_eq!(elems, b.elems());
}

#[test]
fn test_matrix_transpose_transposes_matrix()
{
    let elems = fixture_a(2, 3);
    let a = Matrix::new_with_elems(2, 3, elems.as_slice()).transpose();
    assert_eq!(3, a.row_count());
    assert_eq!(2, a.col_count());
    assert_eq!(true, a.is_transposed());
    assert_eq!(elems, a.elems());
    let b = a.transpose();
    assert_eq!(2, b.row_count());
    assert_eq!(3, b.col_count());
    assert_eq!(false, b.is_transposed());
    assert_eq!(elems, b.elems());
}

#[test]
fn test_matrix_really_transpose_transposes_matrix()
{
    let a_elems = fixture_a(3, 2);
    let a = Matrix::new_with_elems(3, 2, a_elems.as_slice());
    assert_eq!(expected_transpose_a(a_elems.as_slice(), 2, 3), a.really_transpose().elems());
}

#[test]
fn test_matrix_neg_negates_matrix()
{
    let a_elems = fixture_a(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_b_elems = expected_rsub_a_b_for_scalar(a_elems.as_slice(), 0.0, 2, 3);
    assert_eq!(expected_b_elems, (-(a.clone())).elems());
    assert_eq!(expected_b_elems, (-(&a)).elems());
}

#[test]
fn test_matrix_add_adds_matrices_or_matrix_and_scalar()
{
    // matrix
    let (a_elems1, b_elems1) = fixture_a_b(2, 3, 2, 3);
    let a1 = Matrix::new_with_elems(2, 3, a_elems1.as_slice());
    let b1 = Matrix::new_with_elems(2, 3, b_elems1.as_slice());
    let expected_c_elems1 = expected_add_a_b(a_elems1.as_slice(), b_elems1.as_slice(), 2, 3);
    assert_eq!(expected_c_elems1, (a1.clone() + b1.clone()).elems());
    assert_eq!(expected_c_elems1, (a1.clone() + &b1).elems());
    assert_eq!(expected_c_elems1, (&a1 + b1.clone()).elems());
    assert_eq!(expected_c_elems1, (&a1 + &b1).elems());
    // scalar
    let a_elems2 = fixture_a(2, 3);
    let a2 = Matrix::new_with_elems(2, 3, a_elems2.as_slice());
    let expected_c_elems2 = expected_add_a_b_for_scalar(a_elems2.as_slice(), 10.5, 2, 3);
    assert_eq!(expected_c_elems2, (a2.clone() + 10.5).elems());
    assert_eq!(expected_c_elems2, (a2.clone() + &10.5).elems());
    assert_eq!(expected_c_elems2, (&a2 + 10.5).elems());
    assert_eq!(expected_c_elems2, (&a2 + &10.5).elems());
}

#[test]
fn test_matrix_add_assign_adds_matrices_or_matrix_and_scalar()
{
    // matrix
    let (a_elems1, b_elems1) = fixture_a_b(2, 3, 2, 3);
    let b1 = Matrix::new_with_elems(2, 3, b_elems1.as_slice());
    let expected_c_elems1 = expected_add_a_b(a_elems1.as_slice(), b_elems1.as_slice(), 2, 3);
    let mut a11 = Matrix::new_with_elems(2, 3, a_elems1.as_slice());
    a11 += b1.clone();
    assert_eq!(expected_c_elems1, a11.elems());
    let mut a12 = Matrix::new_with_elems(2, 3, a_elems1.as_slice());
    a12 += &b1;
    assert_eq!(expected_c_elems1, a12.elems());
    // scalar
    let a_elems2 = fixture_a(2, 3);
    let expected_c_elems2 = expected_add_a_b_for_scalar(a_elems2.as_slice(), 10.5, 2, 3);
    let mut a21 = Matrix::new_with_elems(2, 3, a_elems2.as_slice());
    a21 += 10.5;
    assert_eq!(expected_c_elems2, a21.elems());
    let mut a22 = Matrix::new_with_elems(2, 3, a_elems2.as_slice());
    a22 += &10.5;
    assert_eq!(expected_c_elems2, a22.elems());
}

#[test]
fn test_matrix_sub_subtracts_matrices_or_matrix_and_scalar()
{
    // matrix
    let (a_elems1, b_elems1) = fixture_a_b(2, 3, 2, 3);
    let a1 = Matrix::new_with_elems(2, 3, a_elems1.as_slice());
    let b1 = Matrix::new_with_elems(2, 3, b_elems1.as_slice());
    let expected_c_elems1 = expected_sub_a_b(a_elems1.as_slice(), b_elems1.as_slice(), 2, 3);
    assert_eq!(expected_c_elems1, (a1.clone() - b1.clone()).elems());
    assert_eq!(expected_c_elems1, (a1.clone() - &b1).elems());
    assert_eq!(expected_c_elems1, (&a1 - b1.clone()).elems());
    assert_eq!(expected_c_elems1, (&a1 - &b1).elems());
    // scalar
    let a_elems2 = fixture_a(2, 3);
    let a2 = Matrix::new_with_elems(2, 3, a_elems2.as_slice());
    let expected_c_elems2 = expected_sub_a_b_for_scalar(a_elems2.as_slice(), 10.5, 2, 3);
    assert_eq!(expected_c_elems2, (a2.clone() - 10.5).elems());
    assert_eq!(expected_c_elems2, (a2.clone() - &10.5).elems());
    assert_eq!(expected_c_elems2, (&a2 - 10.5).elems());
    assert_eq!(expected_c_elems2, (&a2 - &10.5).elems());
}

#[test]
fn test_matrix_sub_assign_subtracts_matrices_or_matrix_and_scalar()
{
    // matrix
    let (a_elems1, b_elems1) = fixture_a_b(2, 3, 2, 3);
    let b1 = Matrix::new_with_elems(2, 3, b_elems1.as_slice());
    let expected_c_elems1 = expected_sub_a_b(a_elems1.as_slice(), b_elems1.as_slice(), 2, 3);
    let mut a11 = Matrix::new_with_elems(2, 3, a_elems1.as_slice());
    a11 -= b1.clone();
    assert_eq!(expected_c_elems1, a11.elems());
    let mut a12 = Matrix::new_with_elems(2, 3, a_elems1.as_slice());
    a12 -= &b1;
    assert_eq!(expected_c_elems1, a12.elems());
    // scalar
    let a_elems2 = fixture_a(2, 3);
    let expected_c_elems2 = expected_sub_a_b_for_scalar(a_elems2.as_slice(), 10.5, 2, 3);
    let mut a21 = Matrix::new_with_elems(2, 3, a_elems2.as_slice());
    a21 -= 10.5;
    assert_eq!(expected_c_elems2, a21.elems());
    let mut a22 = Matrix::new_with_elems(2, 3, a_elems2.as_slice());
    a22 -= &10.5;
    assert_eq!(expected_c_elems2, a22.elems());
}

#[test]
fn test_matrix_mul_multiplies_matrices_or_matrix_and_scalar()
{
    // matrix
    let (a_elems1, b_elems1) = fixture_a_b(2, 4, 4, 3);
    let a1 = Matrix::new_with_elems(2, 4, a_elems1.as_slice());
    let b1 = Matrix::new_with_elems(4, 3, b_elems1.as_slice());
    let expected_c_elems1 = expected_mul_a_b(a_elems1.as_slice(), b_elems1.as_slice(), 2, 3, 4);
    assert_eq!(expected_c_elems1, (a1.clone() * b1.clone()).elems());
    assert_eq!(expected_c_elems1, (a1.clone() * &b1).elems());
    assert_eq!(expected_c_elems1, (&a1 * b1.clone()).elems());
    assert_eq!(expected_c_elems1, (&a1 * &b1).elems());
    // scalar
    let a_elems2 = fixture_a(2, 3);
    let a2 = Matrix::new_with_elems(2, 3, a_elems2.as_slice());
    let expected_c_elems2 = expected_mul_a_b_for_scalar(a_elems2.as_slice(), 10.5, 2, 3);
    assert_eq!(expected_c_elems2, (a2.clone() * 10.5).elems());
    assert_eq!(expected_c_elems2, (a2.clone() * &10.5).elems());
    assert_eq!(expected_c_elems2, (&a2 * 10.5).elems());
    assert_eq!(expected_c_elems2, (&a2 * &10.5).elems());
}

#[test]
fn test_matrix_mul_assign_multiplies_matrices_or_matrix_and_scalar()
{
    // matrix
    let (a_elems1, b_elems1) = fixture_a_b(2, 4, 4, 3);
    let b1 = Matrix::new_with_elems(4, 3, b_elems1.as_slice());
    let expected_c_elems1 = expected_mul_a_b(a_elems1.as_slice(), b_elems1.as_slice(), 2, 3, 4);
    let mut a11 = Matrix::new_with_elems(2, 4, a_elems1.as_slice());
    a11 *= b1.clone();
    assert_eq!(expected_c_elems1, a11.elems());
    let mut a12 = Matrix::new_with_elems(2, 4, a_elems1.as_slice());
    a12 *= &b1;
    assert_eq!(expected_c_elems1, a12.elems());
    // scalar
    let a_elems2 = fixture_a(2, 3);
    let expected_c_elems2 = expected_mul_a_b_for_scalar(a_elems2.as_slice(), 10.5, 2, 3);
    let mut a21 = Matrix::new_with_elems(2, 3, a_elems2.as_slice());
    a21 *= 10.5;
    assert_eq!(expected_c_elems2, a21.elems());
    let mut a22 = Matrix::new_with_elems(2, 3, a_elems2.as_slice());
    a22 *= &10.5;
    assert_eq!(expected_c_elems2, a22.elems());
}

#[test]
fn test_matrix_div_divides_matrix_and_scalar()
{
    let a_elems = fixture_a(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_c_elems = expected_div_a_b_for_scalar(a_elems.as_slice(), 10.5, 2, 3);
    let c_elems1 = (a.clone() / 10.5).elems();
    assert_eq!(expected_c_elems.len(), c_elems1.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_c_elems[i] - c_elems1[i]).abs() < 0.001);
    }
    let c_elems2 = (a.clone() / &10.5).elems();
    assert_eq!(expected_c_elems.len(), c_elems2.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_c_elems[i] - c_elems2[i]).abs() < 0.001);
    }
    let c_elems3 = (&a / 10.5).elems();
    assert_eq!(expected_c_elems.len(), c_elems3.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_c_elems[i] - c_elems3[i]).abs() < 0.001);
    }
    let c_elems4 = (&a / &10.5).elems();
    assert_eq!(expected_c_elems.len(), c_elems4.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_c_elems[i] - c_elems4[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_div_assign_divides_matrix_and_scalar()
{
    let a_elems = fixture_a(2, 3);
    let expected_c_elems = expected_div_a_b_for_scalar(a_elems.as_slice(), 10.5, 2, 3);
    let mut a1 = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    a1 /= 10.5;
    let c_elems1 = a1.elems();
    assert_eq!(expected_c_elems.len(), c_elems1.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_c_elems[i] - c_elems1[i]).abs() < 0.001);
    }
    let mut a2 = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    a2 /= &10.5;
    let c_elems2 = a2.elems();
    assert_eq!(expected_c_elems.len(), c_elems2.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_c_elems[i] - c_elems2[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_mul_elems_multiplies_matrices()
{
    let (a_elems, b_elems) = fixture_a_b(2, 3, 2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let b = Matrix::new_with_elems(2, 3, b_elems.as_slice());
    let expected_c_elems = expected_mul_a_b_for_elems(a_elems.as_slice(), b_elems.as_slice(), 2, 3);
    assert_eq!(expected_c_elems, a.mul_elems(&b).elems());
}

#[test]
fn test_matrix_div_elems_divides_matrices()
{
    let (a_elems, b_elems) = fixture_a_b(2, 3, 2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let b = Matrix::new_with_elems(2, 3, b_elems.as_slice());
    let expected_c_elems = expected_div_a_b_for_elems(a_elems.as_slice(), b_elems.as_slice(), 2, 3);
    let c_elems = a.div_elems(&b).elems();
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_c_elems[i] - c_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_rsub_subtracts_matrix_and_scalar()
{
    let a_elems = fixture_a(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_c_elems = expected_rsub_a_b_for_scalar(a_elems.as_slice(), 10.5, 2, 3);
    assert_eq!(expected_c_elems, a.rsub(10.5).elems());
}

#[test]
fn test_matrix_rdiv_divides_matrix_and_scalar()
{
    let a_elems = fixture_a(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_c_elems = expected_rdiv_a_b_for_scalar(a_elems.as_slice(), 10.5, 2, 3);
    let c = a.rdiv(10.5);
    let c_elems = c.elems();
    assert_eq!(expected_c_elems.len(), c_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_c_elems[i] - c_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_sigmoid_calculates_sigmoid_for_matrix()
{
    let a_elems = fixture_a_for_activation_fun(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_b_elems = expected_sigmoid_a(a_elems.as_slice(), 2, 3);
    let b = a.sigmoid();
    let b_elems = b.elems();
    assert_eq!(expected_b_elems.len(), b_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_b_elems[i] - b_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_tanh_calculates_tanh_for_matrix()
{
    let a_elems = fixture_a_for_activation_fun(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_b_elems = expected_tanh_a(a_elems.as_slice(), 2, 3);
    let b = a.tanh();
    let b_elems = b.elems();
    assert_eq!(expected_b_elems.len(), b_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_b_elems[i] - b_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_swish_calculates_swish_for_matrix()
{
    let a_elems = fixture_a_for_activation_fun(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_b_elems = expected_swish_a(a_elems.as_slice(), 2, 3);
    let b = a.swish();
    let b_elems = b.elems();
    assert_eq!(expected_b_elems.len(), b_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_b_elems[i] - b_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_softmax_calculates_softmax_for_matrix()
{
    let a_elems = fixture_a(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_b_elems = expected_softmax_a(a_elems.as_slice(), 2, 3);
    let b = a.softmax();
    let b_elems = b.elems();
    assert_eq!(expected_b_elems.len(), b_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_b_elems[i] - b_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_sqrt_calculates_sqrt_for_matrix()
{
    let a_elems = fixture_a(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_b_elems = expected_sqrt_a(a_elems.as_slice(), 2, 3);
    let b = a.sqrt();
    let b_elems = b.elems();
    assert_eq!(expected_b_elems.len(), b_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_b_elems[i] - b_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_repeat_repeats_column_or_row()
{
    let a_elems1 = fixture_a(2, 1);
    let a1 = Matrix::new_with_elems(2, 1, a_elems1.as_slice());
    assert_eq!(expected_repeat_col_a(a_elems1.as_slice(), 2, 3), a1.repeat(3).elems());
    let a_elems2 = fixture_a(1, 3);
    let a2 = Matrix::new_with_elems(1, 3, a_elems2.as_slice());
    assert_eq!(expected_repeat_row_a(a_elems2.as_slice(), 2, 3), a2.repeat(2).elems());
}

#[test]
fn test_matrix_abs_calculates_abs_for_matrix()
{
    let a_elems = fixture_a_for_common_math_fun(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    assert_eq!(expected_abs_a(a_elems.as_slice(), 2, 3), a.abs().elems());
}

#[test]
fn test_matrix_powm_calculates_pow_for_matrices()
{
    let (a_elems, b_elems) = fixture_a_b_for_pow(2, 3, 2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let b = Matrix::new_with_elems(2, 3, b_elems.as_slice());
    let expected_c_elems = expected_pow_a_b(a_elems.as_slice(), b_elems.as_slice(), 2, 3);
    let c_elems = a.powm(&b).elems();
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_c_elems[i] - c_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_powf_calculates_pow_for_matrix_and_scalar()
{
    let a_elems = fixture_a_for_log(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_c_elems = expected_pow_a_b_for_scalar(a_elems.as_slice(), 2.5, 2, 3);
    let c = a.powf(2.5);
    let c_elems = c.elems();
    assert_eq!(expected_c_elems.len(), c_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_c_elems[i] - c_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_rpowf_calculates_pow_for_matrix_and_scalar()
{
    let a_elems = fixture_a_for_common_math_fun(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_c_elems = expected_rpow_a_b_for_scalar(a_elems.as_slice(), 10.5, 2, 3);
    let c = a.rpowf(10.5);
    let c_elems = c.elems();
    assert_eq!(expected_c_elems.len(), c_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_c_elems[i] - c_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_exp_calculates_exp_for_matrix()
{
    let a_elems = fixture_a_for_common_math_fun(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_b_elems = expected_exp_a(a_elems.as_slice(), 2, 3);
    let b = a.exp();
    let b_elems = b.elems();
    assert_eq!(expected_b_elems.len(), b_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_b_elems[i] - b_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_ln_calculates_ln_for_matrix()
{
    let a_elems = fixture_a_for_log(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_b_elems = expected_ln_a(a_elems.as_slice(), 2, 3);
    let b = a.ln();
    let b_elems = b.elems();
    assert_eq!(expected_b_elems.len(), b_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_b_elems[i] - b_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_log2_calculates_log2_for_matrix()
{
    let a_elems = fixture_a_for_log(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_b_elems = expected_log2_a(a_elems.as_slice(), 2, 3);
    let b = a.log2();
    let b_elems = b.elems();
    assert_eq!(expected_b_elems.len(), b_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_b_elems[i] - b_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_log10_calculates_log10_for_matrix()
{
    let a_elems = fixture_a_for_log(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_b_elems = expected_log10_a(a_elems.as_slice(), 2, 3);
    let b = a.log10();
    let b_elems = b.elems();
    assert_eq!(expected_b_elems.len(), b_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_b_elems[i] - b_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_sin_calculates_sin_for_matrix()
{
    let a_elems = fixture_a_for_common_math_fun(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_b_elems = expected_sin_a(a_elems.as_slice(), 2, 3);
    let b = a.sin();
    let b_elems = b.elems();
    assert_eq!(expected_b_elems.len(), b_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_b_elems[i] - b_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_cos_calculates_cos_for_matrix()
{
    let a_elems = fixture_a_for_common_math_fun(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_b_elems = expected_cos_a(a_elems.as_slice(), 2, 3);
    let b = a.cos();
    let b_elems = b.elems();
    assert_eq!(expected_b_elems.len(), b_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_b_elems[i] - b_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_tan_calculates_tan_for_matrix()
{
    let a_elems = fixture_a_for_common_math_fun(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_b_elems = expected_tan_a(a_elems.as_slice(), 2, 3);
    let b = a.tan();
    let b_elems = b.elems();
    assert_eq!(expected_b_elems.len(), b_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_b_elems[i] - b_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_asin_calculates_asin_for_matrix()
{
    let a_elems = fixture_a_for_asin_or_acos(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_b_elems = expected_asin_a(a_elems.as_slice(), 2, 3);
    let b = a.asin();
    let b_elems = b.elems();
    assert_eq!(expected_b_elems.len(), b_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_b_elems[i] - b_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_acos_calculates_acos_for_matrix()
{
    let a_elems = fixture_a_for_asin_or_acos(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_b_elems = expected_acos_a(a_elems.as_slice(), 2, 3);
    let b = a.acos();
    let b_elems = b.elems();
    assert_eq!(expected_b_elems.len(), b_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_b_elems[i] - b_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_atan_calculates_atan_for_matrix()
{
    let a_elems = fixture_a_for_common_math_fun(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_b_elems = expected_atan_a(a_elems.as_slice(), 2, 3);
    let b = a.atan();
    let b_elems = b.elems();
    assert_eq!(expected_b_elems.len(), b_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_b_elems[i] - b_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_atan2_calculates_atan2_for_matrices()
{
    let (a_elems, b_elems) = fixture_a_b_for_pow(2, 3, 2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let b = Matrix::new_with_elems(2, 3, b_elems.as_slice());
    let expected_c_elems = expected_atan2_a_b(a_elems.as_slice(), b_elems.as_slice(), 2, 3);
    let c_elems = a.atan2(&b).elems();
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_c_elems[i] - c_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_atan2f_calculates_atan2_for_matrix_and_scalar()
{
    let a_elems = fixture_a_for_log(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_c_elems = expected_atan2_a_b_for_scalar(a_elems.as_slice(), 2.5, 2, 3);
    let c = a.atan2f(2.5);
    let c_elems = c.elems();
    assert_eq!(expected_c_elems.len(), c_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_c_elems[i] - c_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_ratan2f_calculates_atan2_for_matrix_and_scalar()
{
    let a_elems = fixture_a_for_common_math_fun(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_c_elems = expected_ratan2_a_b_for_scalar(a_elems.as_slice(), 10.5, 2, 3);
    let c = a.ratan2f(10.5);
    let c_elems = c.elems();
    assert_eq!(expected_c_elems.len(), c_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_c_elems[i] - c_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_sinh_calculates_sinh_for_matrix()
{
    let a_elems = fixture_a_for_common_math_fun(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_b_elems = expected_sinh_a(a_elems.as_slice(), 2, 3);
    let b = a.sinh();
    let b_elems = b.elems();
    assert_eq!(expected_b_elems.len(), b_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_b_elems[i] - b_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_cosh_calculates_cosh_for_matrix()
{
    let a_elems = fixture_a_for_common_math_fun(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_b_elems = expected_cosh_a(a_elems.as_slice(), 2, 3);
    let b = a.cosh();
    let b_elems = b.elems();
    assert_eq!(expected_b_elems.len(), b_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_b_elems[i] - b_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_asinh_calculates_asinh_for_matrix()
{
    let a_elems = fixture_a_for_common_math_fun(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_b_elems = expected_asinh_a(a_elems.as_slice(), 2, 3);
    let b = a.asinh();
    let b_elems = b.elems();
    assert_eq!(expected_b_elems.len(), b_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_b_elems[i] - b_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_acosh_calculates_acosh_for_matrix()
{
    let a_elems = fixture_a_for_log(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_b_elems = expected_acosh_a(a_elems.as_slice(), 2, 3);
    let b = a.acosh();
    let b_elems = b.elems();
    assert_eq!(expected_b_elems.len(), b_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_b_elems[i] - b_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_atanh_calculates_atanh_for_matrix()
{
    let a_elems = fixture_a_for_asin_or_acos(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    let expected_b_elems = expected_atanh_a(a_elems.as_slice(), 2, 3);
    let b = a.atanh();
    let b_elems = b.elems();
    assert_eq!(expected_b_elems.len(), b_elems.len());
    for i in 0usize..(2usize * 3usize) {
        assert!((expected_b_elems[i] - b_elems[i]).abs() < 0.001);
    }
}

#[test]
fn test_matrix_signum_calculates_signum_for_matrix()
{
    let a_elems = fixture_a_for_common_math_fun(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    assert_eq!(expected_signum_a(a_elems.as_slice(), 2, 3), a.signum().elems());
}

#[test]
fn test_matrix_ceil_calculates_ceil_for_matrix()
{
    let a_elems = fixture_a_for_rounding(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    assert_eq!(expected_ceil_a(a_elems.as_slice(), 2, 3), a.ceil().elems());
}

#[test]
fn test_matrix_floor_calculates_floor_for_matrix()
{
    let a_elems = fixture_a_for_rounding(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    assert_eq!(expected_floor_a(a_elems.as_slice(), 2, 3), a.floor().elems());
}

#[test]
fn test_matrix_round_calculates_round_for_matrix()
{
    let a_elems = fixture_a_for_rounding(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    assert_eq!(expected_round_a(a_elems.as_slice(), 2, 3), a.round().elems());
}

#[test]
fn test_matrix_trunc_calculates_trunc_for_matrix()
{
    let a_elems = fixture_a_for_rounding(2, 3);
    let a = Matrix::new_with_elems(2, 3, a_elems.as_slice());
    assert_eq!(expected_trunc_a(a_elems.as_slice(), 2, 3), a.trunc().elems());
}
