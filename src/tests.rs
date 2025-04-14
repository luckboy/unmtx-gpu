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
fn test_frontend_force_transpose_transposes_matrix()
{
    match Frontend::new() {
        Ok(frontend) => {
            let a = fixture_a(2, 3);
            match frontend_force_transpose(&frontend, a.as_slice(), 2, 3) {
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
fn test_frontend_softmax_calculates_sigmoid_for_matrix()
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
