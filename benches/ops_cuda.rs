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
const SAMPLE_COUNT: u32 = 100;

static FRONTEND: Mutex<Option<Frontend>> = Mutex::new(None);
static A: Mutex<Option<Matrix>> = Mutex::new(None);
static B: Mutex<Option<Matrix>> = Mutex::new(None);
static C: Mutex<Option<Matrix>> = Mutex::new(None);
static D: Mutex<Option<Matrix>> = Mutex::new(None);
static SCALAR: Mutex<Option<f32>> = Mutex::new(None);

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
        let mut c_g = C.lock().unwrap();
        *c_g = Some(unsafe { frontend_g.as_ref().unwrap().create_matrix(WIDTH, WIDTH).unwrap() });
        let mut d_g = D.lock().unwrap();
        let mut d_elems: Vec<f32> = vec![0.0f32; WIDTH];
        for d_elem in &mut d_elems {
            *d_elem = random();
        }
        *d_g = Some(frontend_g.as_ref().unwrap().create_matrix_and_set_elems(WIDTH, 1, d_elems.as_slice()).unwrap());
        let mut scalar_g = SCALAR.lock().unwrap();
        *scalar_g = Some(random());
    }
    divan::main();
    {
        let mut frontend_g = FRONTEND.lock().unwrap();
        let mut a_g = A.lock().unwrap();
        let mut b_g = B.lock().unwrap();
        let mut c_g = C.lock().unwrap();
        let mut d_g = D.lock().unwrap();
        let mut scalar_g = SCALAR.lock().unwrap();
        *scalar_g = None;
        *d_g = None;
        *c_g = None;
        *b_g = None;
        *a_g = None;
        *frontend_g = None;
    }
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_add_a_b()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().add(a_g.as_ref().unwrap(), b_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_add_at_b()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().add(&a_g.as_ref().unwrap().t(), b_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_add_a_bt()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().add(a_g.as_ref().unwrap(), &b_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_add_at_bt()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().add(&a_g.as_ref().unwrap().t(), &b_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_sub_a_b()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().sub(a_g.as_ref().unwrap(), b_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_sub_at_b()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().sub(&a_g.as_ref().unwrap().t(), b_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_sub_a_bt()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().sub(a_g.as_ref().unwrap(), &b_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_sub_at_bt()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().sub(&a_g.as_ref().unwrap().t(), &b_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_mul_elems_a_b()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().mul_elems(a_g.as_ref().unwrap(), b_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_mul_elems_at_b()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().mul_elems(&a_g.as_ref().unwrap().t(), b_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_mul_elems_a_bt()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().mul_elems(a_g.as_ref().unwrap(), &b_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_mul_elems_at_bt()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().mul_elems(&a_g.as_ref().unwrap().t(), &b_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_div_elems_a_b()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().div_elems(a_g.as_ref().unwrap(), b_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_div_elems_at_b()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().div_elems(&a_g.as_ref().unwrap().t(), b_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_div_elems_a_bt()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().div_elems(a_g.as_ref().unwrap(), &b_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_div_elems_at_bt()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().div_elems(&a_g.as_ref().unwrap().t(), &b_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_add_for_scalar_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().add_for_scalar(a_g.as_ref().unwrap(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_add_for_scalar_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().add_for_scalar(&a_g.as_ref().unwrap().t(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_sub_for_scalar_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().sub_for_scalar(a_g.as_ref().unwrap(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_sub_for_scalar_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().sub_for_scalar(&a_g.as_ref().unwrap().t(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_rsub_for_scalar_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().rsub_for_scalar(a_g.as_ref().unwrap(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_rsub_for_scalar_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().rsub_for_scalar(&a_g.as_ref().unwrap().t(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_mul_for_scalar_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().mul_for_scalar(a_g.as_ref().unwrap(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_mul_for_scalar_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().mul_for_scalar(&a_g.as_ref().unwrap().t(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_div_for_scalar_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().div_for_scalar(a_g.as_ref().unwrap(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_div_for_scalar_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().div_for_scalar(&a_g.as_ref().unwrap().t(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_rdiv_for_scalar_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().rdiv_for_scalar(a_g.as_ref().unwrap(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_rdiv_for_scalar_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().rdiv_for_scalar(&a_g.as_ref().unwrap().t(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_really_transpose_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().really_transpose(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_repeat_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let d_g = D.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().repeat(d_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_repeat_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let d_g = D.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().repeat(&d_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}
