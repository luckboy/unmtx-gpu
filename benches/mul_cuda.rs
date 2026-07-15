//
// Copyright (c) 2026 Łukasz Szpakowski
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
use std::sync::Arc;
use std::sync::Mutex;
use unmtx_gpu::cuda::CudaBackend;
use unmtx_gpu::Frontend;
use unmtx_gpu::Matrix;
use rand::random;

const WIDTH: usize = 4 * 1024;
const SAMPLE_COUNT: u32 = 5;

static FRONTEND: Mutex<Option<Frontend>> = Mutex::new(None);
static A: Mutex<Option<Matrix>> = Mutex::new(None);
static B: Mutex<Option<Matrix>> = Mutex::new(None);

fn main()
{
    {
        let mut frontend_g = FRONTEND.lock().unwrap();
        let backend = CudaBackend::new().unwrap();
        *frontend_g = Some(Frontend::new_with_backend(Arc::new(backend)));
        println!("Backend: {}", frontend_g.as_ref().unwrap().backend().name());
        let mut a_g = A.lock().unwrap();
        let mut a_elems: Vec<f32> = vec![0.0f32; WIDTH * WIDTH];
        for a_elem in &mut a_elems {
            *a_elem = random();
        }
        *a_g = Some(frontend_g.as_ref().unwrap().create_matrix_and_set_elems(WIDTH, WIDTH, a_elems.as_slice()).unwrap());
        let mut b_g = B.lock().unwrap();
        let mut b_elems: Vec<f32> = vec![0.0f32; WIDTH * WIDTH];
        for b_elem in &mut b_elems {
            *b_elem = random();
        }
        *b_g = Some(frontend_g.as_ref().unwrap().create_matrix_and_set_elems(WIDTH, WIDTH, b_elems.as_slice()).unwrap());
    }
    divan::main();
    {
        let mut frontend_g = FRONTEND.lock().unwrap();
        let mut a_g = A.lock().unwrap();
        let mut b_g = B.lock().unwrap();
        *b_g = None;
        *a_g = None;
        *frontend_g = None;
    }
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_mul_a_b()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c = if frontend_g.as_ref().unwrap().backend().has_cublas() {
        frontend_g.as_ref().unwrap().create_matrix_and_set_zeros(WIDTH, WIDTH).unwrap()
    } else {
        unsafe { frontend_g.as_ref().unwrap().create_matrix(WIDTH, WIDTH) }.unwrap()
    };
    frontend_g.as_ref().unwrap().mul(a_g.as_ref().unwrap(), b_g.as_ref().unwrap(), &c).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_mul_at_b()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c = if frontend_g.as_ref().unwrap().backend().has_cublas() {
        frontend_g.as_ref().unwrap().create_matrix_and_set_zeros(WIDTH, WIDTH).unwrap()
    } else {
        unsafe { frontend_g.as_ref().unwrap().create_matrix(WIDTH, WIDTH) }.unwrap()
    };
    frontend_g.as_ref().unwrap().mul(&a_g.as_ref().unwrap().t(), b_g.as_ref().unwrap(), &c).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_mul_a_bt()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c = if frontend_g.as_ref().unwrap().backend().has_cublas() {
        frontend_g.as_ref().unwrap().create_matrix_and_set_zeros(WIDTH, WIDTH).unwrap()
    } else {
        unsafe { frontend_g.as_ref().unwrap().create_matrix(WIDTH, WIDTH) }.unwrap()
    };
    frontend_g.as_ref().unwrap().mul(a_g.as_ref().unwrap(), &b_g.as_ref().unwrap().t(), &c).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_mul_at_bt()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c = if frontend_g.as_ref().unwrap().backend().has_cublas() {
        frontend_g.as_ref().unwrap().create_matrix_and_set_zeros(WIDTH, WIDTH).unwrap()
    } else {
        unsafe { frontend_g.as_ref().unwrap().create_matrix(WIDTH, WIDTH) }.unwrap()
    };
    frontend_g.as_ref().unwrap().mul(&a_g.as_ref().unwrap().t(), &b_g.as_ref().unwrap().t(), &c).unwrap();
}

