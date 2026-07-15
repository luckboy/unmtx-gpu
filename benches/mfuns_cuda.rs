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
static A_FOR_ABS: Mutex<Option<Matrix>> = Mutex::new(None);
static A_FOR_ACOSH: Mutex<Option<Matrix>> = Mutex::new(None);
static B: Mutex<Option<Matrix>> = Mutex::new(None);
static C: Mutex<Option<Matrix>> = Mutex::new(None);
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
        let mut a2_g = A_FOR_ABS.lock().unwrap();
        let mut a2_elems: Vec<f32> = vec![0.0f32; WIDTH * WIDTH];
        for a2_elem in &mut a2_elems {
            *a2_elem = random::<f32>() * 2.0 - 1.0;
        }
        *a2_g = Some(frontend_g.as_ref().unwrap().create_matrix_and_set_elems(WIDTH, WIDTH, a2_elems.as_slice()).unwrap());
        let mut a3_g = A_FOR_ACOSH.lock().unwrap();
        let mut a3_elems: Vec<f32> = vec![0.0f32; WIDTH * WIDTH];
        for a3_elem in &mut a3_elems {
            *a3_elem = random::<f32>() + 1.0;
        }
        *a3_g = Some(frontend_g.as_ref().unwrap().create_matrix_and_set_elems(WIDTH, WIDTH, a3_elems.as_slice()).unwrap());
        let mut b_g = B.lock().unwrap();
        let mut b_elems: Vec<f32> = vec![0.0f32; WIDTH * WIDTH];
        for b_elem in &mut b_elems {
            *b_elem = random();
        }
        *b_g = Some(frontend_g.as_ref().unwrap().create_matrix_and_set_elems(WIDTH, WIDTH, b_elems.as_slice()).unwrap());
        let mut c_g = C.lock().unwrap();
        *c_g = Some(unsafe { frontend_g.as_ref().unwrap().create_matrix(WIDTH, WIDTH).unwrap() });
        let mut scalar_g = SCALAR.lock().unwrap();
        *scalar_g = Some(random());
    }
    divan::main();
    {
        let mut frontend_g = FRONTEND.lock().unwrap();
        let mut a_g = A.lock().unwrap();
        let mut a2_g = A_FOR_ABS.lock().unwrap();
        let mut a3_g = A_FOR_ACOSH.lock().unwrap();
        let mut b_g = B.lock().unwrap();
        let mut c_g = C.lock().unwrap();
        let mut scalar_g = SCALAR.lock().unwrap();
        *scalar_g = None;
        *c_g = None;
        *b_g = None;
        *a3_g = None;
        *a2_g = None;
        *a_g = None;
        *frontend_g = None;
    }
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_abs_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A_FOR_ABS.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().abs(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_abs_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A_FOR_ABS.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().abs(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_pow_a_b()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().pow(a_g.as_ref().unwrap(), b_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_pow_at_b()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().pow(&a_g.as_ref().unwrap().t(), b_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_pow_a_bt()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().pow(a_g.as_ref().unwrap(), &b_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_pow_at_bt()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().pow(&a_g.as_ref().unwrap().t(), &b_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_pow_for_scalar_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().pow_for_scalar(a_g.as_ref().unwrap(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_pow_for_scalar_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().pow_for_scalar(&a_g.as_ref().unwrap().t(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_rpow_for_scalar_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().rpow_for_scalar(a_g.as_ref().unwrap(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_rpow_for_scalar_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().rpow_for_scalar(&a_g.as_ref().unwrap().t(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_exp_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().abs(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_exp_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().abs(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_ln_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().ln(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_ln_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().ln(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_log2_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().log2(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_log2_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().log2(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_log10_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().log10(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_log10_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().log10(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_sin_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().sin(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_sin_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().sin(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_cos_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().cos(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_cos_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().cos(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_tan_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().tan(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_tan_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().tan(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_asin_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().asin(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_asin_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().asin(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_acos_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().acos(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_acos_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().acos(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_atan_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().atan(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_atan_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().atan(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_atan2_a_b()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().atan2(a_g.as_ref().unwrap(), b_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_atan2_at_b()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().atan2(&a_g.as_ref().unwrap().t(), b_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_atan2_a_bt()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().atan2(a_g.as_ref().unwrap(), &b_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_atan2_at_bt()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().atan2(&a_g.as_ref().unwrap().t(), &b_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_atan2_for_scalar_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().atan2_for_scalar(a_g.as_ref().unwrap(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_atan2_for_scalar_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().atan2_for_scalar(&a_g.as_ref().unwrap().t(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_ratan2_for_scalar_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().ratan2_for_scalar(a_g.as_ref().unwrap(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_ratan2_for_scalar_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().ratan2_for_scalar(&a_g.as_ref().unwrap().t(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_sinh_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().sinh(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_sinh_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().sinh(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_cosh_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().cosh(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_cosh_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().cosh(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_asinh_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().asinh(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_asinh_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().asinh(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_acosh_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A_FOR_ACOSH.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().acosh(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_acosh_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A_FOR_ACOSH.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().acosh(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_atanh_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().atanh(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_atanh_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().atanh(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_signum_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A_FOR_ABS.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().signum(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_signum_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A_FOR_ABS.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().signum(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_ceil_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A_FOR_ABS.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().ceil(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_ceil_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A_FOR_ABS.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().ceil(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_floor_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A_FOR_ABS.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().floor(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_floor_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A_FOR_ABS.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().floor(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_round_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A_FOR_ABS.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().round(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_round_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A_FOR_ABS.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().round(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_trunc_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A_FOR_ABS.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().trunc(a_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_trunc_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A_FOR_ABS.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().trunc(&a_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_max_a_b()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().max(a_g.as_ref().unwrap(), b_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_max_at_b()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().max(&a_g.as_ref().unwrap().t(), b_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_max_a_bt()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().max(a_g.as_ref().unwrap(), &b_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_max_at_bt()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().max(&a_g.as_ref().unwrap().t(), &b_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_max_for_scalar_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().max_for_scalar(a_g.as_ref().unwrap(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_max_for_scalar_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().max_for_scalar(&a_g.as_ref().unwrap().t(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_min_a_b()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().min(a_g.as_ref().unwrap(), b_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_min_at_b()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().min(&a_g.as_ref().unwrap().t(), b_g.as_ref().unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_min_a_bt()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().min(a_g.as_ref().unwrap(), &b_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_min_at_bt()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let b_g = B.lock().unwrap();
    let c_g = C.lock().unwrap();
    frontend_g.as_ref().unwrap().min(&a_g.as_ref().unwrap().t(), &b_g.as_ref().unwrap().t(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_min_for_scalar_a()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().min_for_scalar(a_g.as_ref().unwrap(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}

#[divan::bench(sample_count = SAMPLE_COUNT, sample_size = 1)]
fn bench_min_for_scalar_at()
{
    let frontend_g = FRONTEND.lock().unwrap();
    let a_g = A.lock().unwrap();
    let c_g = C.lock().unwrap();
    let scalar_g = SCALAR.lock().unwrap();
    frontend_g.as_ref().unwrap().min_for_scalar(&a_g.as_ref().unwrap().t(), scalar_g.unwrap(), c_g.as_ref().unwrap()).unwrap();
}
