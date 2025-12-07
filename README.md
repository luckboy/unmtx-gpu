# Unmtx-gpu

Micro neural matrix library for GPU is small library that operates on matrices. This library provides
an interfece to operations of matrices on GPU for neural networks.

## Computing platforms

This library uses GPU by the following computing platforms:

- OpenCL
- CUDA

If this library uses CUDA, this library can use the cuBLAS library to multiplication of matrices.

## Usage

You can use this library by add the following lines in the `Cargo.toml` file:

```toml
[dependencies]
unmtx-gpu = "0.1.5"
```

## Features

The following features of this library can be used by you:

- `opencl` - use OpenCL (default)
- `cuda` - use CUDA
- `cuda-*` - choose CUDA version (for example `cuda-11050`)
- `default_cublas` - use the cuBLAS library to multiplication of matrices as default for CUDA
- `default_mma` - use the mma instruction to multiplication of matrices as default for CUDA
- `test_only_backend` - test only backend

## Examples

The following example presents multiplication of matrices:

```rust
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
```

If you want to find more examples, you can find them in the `examples` directory.

## License

This library is licensed under the Mozilla Public License v2.0. See the LICENSE file for the full
licensing terms.
