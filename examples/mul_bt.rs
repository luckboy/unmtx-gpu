//
// Copyright (c) 2025 Łukasz Szpakowski
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
use std::env;
use std::process::exit;
use std::time::Instant;
use unmtx_gpu::*;

fn create_matrix(n: usize, m: usize) -> Matrix
{
    let mut elems: Vec<f32> = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            elems[m * i + j] = (m * i + j) as f32;
        }
    }
    Matrix::new_with_elems(n, m, elems.as_slice())
}

fn main()
{
    let args: Vec<String> = env::args().collect();
    let n: usize = match args.get(1) {
        Some(s) => {
            match s.parse::<usize>() {
                Ok(tmp_n) => tmp_n,
                Err(err) => {
                    eprintln!("{}", err);
                    exit(1);
                },
            }
        },
        None => 100,
    };
    let m: usize = match args.get(2) {
        Some(s) => {
            match s.parse::<usize>() {
                Ok(tmp_m) => tmp_m,
                Err(err) => {
                    eprintln!("{}", err);
                    exit(1);
                },
            }
        },
        None => 100,
    };
    let l: usize = match args.get(3) {
        Some(s) => {
            match s.parse::<usize>() {
                Ok(tmp_l) => tmp_l,
                Err(err) => {
                    eprintln!("{}", err);
                    exit(1);
                },
            }
        },
        None => 100,
    };
    let frontend = match Frontend::new() {
        Ok(tmp_frontend) => tmp_frontend,
        Err(err) => {
            eprintln!("{}", err);
            exit(1);
        },
    };
    println!("backend: {}", frontend.backend().name());
    let a = create_matrix(n, l);
    let b = create_matrix(m, l);
    let now = Instant::now();
    let c = a * b.transpose();
    let duration = now.elapsed();
    let elems = c.elems();
    let sum = elems.iter().fold(0.0f32, |x, y| x + y);
    println!("sum: {}", sum);
    println!("time: {}.{:06}", duration.as_secs(), duration.as_micros() % 1000000);
    match finalize_default_backend() {
        Ok(()) => (),
        Err(err) => {
            eprintln!("{}", err);
            exit(1);
        },
    }
}
