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
const SOFTMAX_SAMPLE_COUNT: u32 = 5;


static FRONTEND: Mutex<Option<Frontend>> = Mutex::new(None);
static A: Mutex<Option<Matrix>> = Mutex::new(None);
static C: Mutex<Option<Matrix>> = Mutex::new(None);

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
        let mut c_g = C.lock().unwrap();
        *c_g = Some(unsafe { frontend_g.as_ref().unwrap().create_matrix(WIDTH, WIDTH).unwrap() });
    }
    divan::main();
    {
        let mut frontend_g = FRONTEND.lock().unwrap();
        let mut a_g = A.lock().unwrap();
        let mut c_g = C.lock().unwrap();
        *c_g = None;
        *a_g = None;
        *frontend_g = None;
    }
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_sigmoid_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().sigmoid(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_sigmoid_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().sigmoid(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_tanh_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().tanh(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_tanh_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().tanh(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_swish_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().swish(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_swish_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().swish(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SOFTMAX_SAMPLE_COUNT, sample_size = 1)]
fn bench_softmax_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().softmax(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SOFTMAX_SAMPLE_COUNT, sample_size = 1)]
fn bench_softmax_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().softmax(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_sqrt_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().sqrt(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_sqrt_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().sqrt(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}
