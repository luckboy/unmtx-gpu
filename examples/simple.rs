//
// Copyright (c) 2025 Łukasz Szpakowski
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
use unmtx_gpu::matrix;

fn main()
{
    let a = matrix![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ];
    let b = matrix![
        [1.0, 4.0, 7.0],
        [2.0, 5.0, 8.0],
        [3.0, 6.0, 9.0]
    ];
    let c = a * b;
    let elems = c.elems();
    for i in 0..c.row_count() {
        for j in 0..c.col_count() {
            print!("\t{}", elems[i * c.col_count() + j]);
        }
        println!("");
    }
}
