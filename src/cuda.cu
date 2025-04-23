//
// Copyright (c) 2025 Łukasz Szpakowski
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
#define TILE_WIDTH      32

#define MTHREAD_COUNT   32
#define MTILE_WIDTH     (32 << 2)

extern "C" {
  __global__ void transpose_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      b[m * i + j] = a[n * j + i];
    }
  }
  
  __global__ void add_a_b(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[m * i + j] + b[m * i + j];
    }
  }

  __global__ void add_at_b(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[n * j + i] + b[m * i + j];
    }
  }

  __global__ void add_a_bt(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[m * i + j] + b[n * j + i];
    }
  }

  __global__ void add_at_bt(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[n * j + i] + b[n * j + i];
    }
  }

  __global__ void sub_a_b(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[m * i + j] - b[m * i + j];
    }
  }

  __global__ void sub_at_b(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[n * j + i] - b[m * i + j];
    }
  }

  __global__ void sub_a_bt(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[m * i + j] - b[n * j + i];
    }
  }

  __global__ void sub_at_bt(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[n * j + i] - b[n * j + i];
    }
  }

  __global__ void mul_a_b(const float *a, const float *b, float *c, size_t n, size_t m, size_t l)
  {
    __shared__ float as[MTILE_WIDTH][MTHREAD_COUNT];
    __shared__ float bs[MTHREAD_COUNT][MTILE_WIDTH];
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t k;
    size_t ti = threadIdx.x;
    size_t tj = threadIdx.y;
    size_t bi = ti << 1;
    size_t bj = tj << 1;
    float ar1;
    float ar2;
    float br1;
    float br2;
    float cr11 = 0.0f;
    float cr12 = 0.0f;
    float cr21 = 0.0f;
    float cr22 = 0.0f;
    for(k = 0; k < l; k += MTHREAD_COUNT) {
      size_t tk;
      as[bi + 0][tj] = 0.0f;
      if(i + 0 < n && k + tj < l) {
        as[bi + 0][tj] = a[l * (i + 0) + k + tj];
      }
      as[bi + 1][tj] = 0.0f;
      if(i + 1 < n && k + tj < l) {
        as[bi + 1][tj] = a[l * (i + 1) + k + tj];
      }
      bs[ti][bj + 0] = 0.0f;
      if(j + 0 < m && k + ti < l) {
        bs[ti][bj + 0] = b[m * (k + ti) + j + 0];
      }
      bs[ti][bj + 1] = 0.0f;
      if(j + 1 < m && k + ti < l) {
        bs[ti][bj + 1] = b[m * (k + ti) + j + 1];
      }
      __syncthreads();
      for(tk = 0; tk < MTHREAD_COUNT; tk++) {
        ar1 = as[bi + 0][tk];
        ar2 = as[bi + 1][tk];
        br1 = bs[tk][bj + 0];
        br2 = bs[tk][bj + 1];
        cr11 += ar1 * br1;
        cr12 += ar1 * br2;
        cr21 += ar2 * br1;
        cr22 += ar2 * br2;
      }
      __syncthreads();
    }
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = cr11;
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = cr12;
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = cr21;
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = cr22;
    }
  }

  __global__ void mul_at_b(const float *a, const float *b, float *c, size_t n, size_t m, size_t l)
  {
    __shared__ float as[MTILE_WIDTH][MTHREAD_COUNT];
    __shared__ float bs[MTHREAD_COUNT][MTILE_WIDTH];
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t k;
    size_t ti = threadIdx.x;
    size_t tj = threadIdx.y;
    size_t bi = ti << 1;
    size_t bj = tj << 1;
    float ar1;
    float ar2;
    float br1;
    float br2;
    float cr11 = 0.0f;
    float cr12 = 0.0f;
    float cr21 = 0.0f;
    float cr22 = 0.0f;
    for(k = 0; k < l; k += MTHREAD_COUNT) {
      size_t tk;
      as[bi + 0][tj] = 0.0f;
      if(i + 0 < n && k + tj < l) {
        as[bi + 0][tj] = a[n * (k + tj) + i + 0];
      }
      as[bi + 1][tj] = 0.0f;
      if(i + 1 < n && k + tj < l) {
        as[bi + 1][tj] = a[n * (k + tj) + i + 1];
      }
      bs[ti][bj + 0] = 0.0f;
      if(j + 0 < m && k + ti < l) {
        bs[ti][bj + 0] = b[m * (k + ti) + j + 0];
      }
      bs[ti][bj + 1] = 0.0f;
      if(j + 1 < m && k + ti < l) {
        bs[ti][bj + 1] = b[m * (k + ti) + j + 1];
      }
      __syncthreads();
      for(tk = 0; tk < MTHREAD_COUNT; tk++) {
        ar1 = as[bi + 0][tk];
        ar2 = as[bi + 1][tk];
        br1 = bs[tk][bj + 0];
        br2 = bs[tk][bj + 1];
        cr11 += ar1 * br1;
        cr12 += ar1 * br2;
        cr21 += ar2 * br1;
        cr22 += ar2 * br2;
      }
      __syncthreads();
    }
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = cr11;
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = cr12;
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = cr21;
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = cr22;
    }
  }

  __global__ void mul_a_bt(const float *a, const float *b, float *c, size_t n, size_t m, size_t l)
  {
    __shared__ float as[MTILE_WIDTH][MTHREAD_COUNT];
    __shared__ float bs[MTHREAD_COUNT][MTILE_WIDTH];
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t k;
    size_t ti = threadIdx.x;
    size_t tj = threadIdx.y;
    size_t bi = ti << 1;
    size_t bj = tj << 1;
    float ar1;
    float ar2;
    float br1;
    float br2;
    float cr11 = 0.0f;
    float cr12 = 0.0f;
    float cr21 = 0.0f;
    float cr22 = 0.0f;
    for(k = 0; k < l; k += MTHREAD_COUNT) {
      size_t tk;
      as[bi + 0][tj] = 0.0f;
      if(i + 0 < n && k + tj < l) {
        as[bi + 0][tj] = a[l * (i + 0) + k + tj];
      }
      as[bi + 1][tj] = 0.0f;
      if(i + 1 < n && k + tj < l) {
        as[bi + 1][tj] = a[l * (i + 1) + k + tj];
      }
      bs[ti][bj + 0] = 0.0f;
      if(j + 0 < m && k + ti < l) {
        bs[ti][bj + 0] = b[l * (j + 0) + k + ti];
      }
      bs[ti][bj + 1] = 0.0f;
      if(j + 1 < m && k + ti < l) {
        bs[ti][bj + 1] = b[l * (j + 1) + k + ti];
      }
      __syncthreads();
      for(tk = 0; tk < MTHREAD_COUNT; tk++) {
        ar1 = as[bi + 0][tk];
        ar2 = as[bi + 1][tk];
        br1 = bs[tk][bj + 0];
        br2 = bs[tk][bj + 1];
        cr11 += ar1 * br1;
        cr12 += ar1 * br2;
        cr21 += ar2 * br1;
        cr22 += ar2 * br2;
      }
      __syncthreads();
    }
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = cr11;
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = cr12;
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = cr21;
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = cr22;
    }
  }

  __global__ void mul_at_bt(const float *a, const float *b, float *c, size_t n, size_t m, size_t l)
  {
    __shared__ float as[MTILE_WIDTH][MTHREAD_COUNT];
    __shared__ float bs[MTHREAD_COUNT][MTILE_WIDTH];
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t k;
    size_t ti = threadIdx.x;
    size_t tj = threadIdx.y;
    size_t bi = ti << 1;
    size_t bj = tj << 1;
    float ar1;
    float ar2;
    float br1;
    float br2;
    float cr11 = 0.0f;
    float cr12 = 0.0f;
    float cr21 = 0.0f;
    float cr22 = 0.0f;
    for(k = 0; k < l; k += MTHREAD_COUNT) {
      size_t tk;
      as[bi + 0][tj] = 0.0f;
      if(i + 0 < n && k + tj < l) {
        as[bi + 0][tj] = a[n * (k + tj) + i + 0];
      }
      as[bi + 1][tj] = 0.0f;
      if(i + 1 < n && k + tj < l) {
        as[bi + 1][tj] = a[n * (k + tj) + i + 1];
      }
      bs[ti][bj + 0] = 0.0f;
      if(j + 0 < m && k + ti < l) {
        bs[ti][bj + 0] = b[l * (j + 0) + k + ti];
      }
      bs[ti][bj + 1] = 0.0f;
      if(j + 1 < m && k + ti < l) {
        bs[ti][bj + 1] = b[l * (j + 1) + k + ti];
      }
      __syncthreads();
      for(tk = 0; tk < MTHREAD_COUNT; tk++) {
        ar1 = as[bi + 0][tk];
        ar2 = as[bi + 1][tk];
        br1 = bs[tk][bj + 0];
        br2 = bs[tk][bj + 1];
        cr11 += ar1 * br1;
        cr12 += ar1 * br2;
        cr21 += ar2 * br1;
        cr22 += ar2 * br2;
      }
      __syncthreads();
    }
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = cr11;
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = cr12;
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = cr21;
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = cr22;
    }
  }

  __global__ void mul_a_b_for_elems(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[m * i + j] * b[m * i + j];
    }
  }

  __global__ void mul_at_b_for_elems(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[n * j + i] * b[m * i + j];
    }
  }

  __global__ void mul_a_bt_for_elems(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[m * i + j] * b[n * j + i];
    }
  }

  __global__ void mul_at_bt_for_elems(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[n * j + i] * b[n * j + i];
    }
  }

  __global__ void div_a_b_for_elems(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[m * i + j] / b[m * i + j];
    }
  }

  __global__ void div_at_b_for_elems(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[n * j + i] / b[m * i + j];
    }
  }

  __global__ void div_a_bt_for_elems(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[m * i + j] / b[n * j + i];
    }
  }

  __global__ void div_at_bt_for_elems(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[n * j + i] / b[n * j + i];
    }
  }

  __global__ void add_a_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[m * i + j] + b;
    }
  }

  __global__ void add_at_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[n * j + i] + b;
    }
  }

  __global__ void sub_a_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[m * i + j] - b;
    }
  }

  __global__ void sub_at_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[n * j + i] - b;
    }
  }

  __global__ void rsub_a_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = b - a[m * i + j];
    }
  }

  __global__ void rsub_at_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = b - a[n * j + i];
    }
  }

  __global__ void mul_a_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[m * i + j] * b;
    }
  }

  __global__ void mul_at_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[n * j + i] * b;
    }
  }

  __global__ void div_a_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[m * i + j] / b;
    }
  }

  __global__ void div_at_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[n * j + i] / b;
    }
  }

  __global__ void rdiv_a_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = b / a[m * i + j];
    }
  }

  __global__ void rdiv_at_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = b / a[n * j + i];
    }
  }

  __global__ void sigmoid_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      b[m * i + j] = 1.0f / (1.0f + expf(-a[m * i + j]));
    }
  }

  __global__ void sigmoid_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      b[m * i + j] = 1.0f / (1.0f + expf(-a[n * j + i]));
    }
  }

  __global__ void tanh_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      b[m * i + j] = tanhf(a[m * i + j]);
    }
  }

  __global__ void tanh_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      b[m * i + j] = tanhf(a[n * j + i]);
    }
  }

  __global__ void softmax_a(const float *a, float *b, size_t n, size_t m)
  {
    __shared__ float es[TILE_WIDTH][TILE_WIDTH];
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    size_t k;
    size_t ti = threadIdx.x;
    size_t tj = threadIdx.y;
    float sum = 0.0f;
    for(k = 0; k < n; k += TILE_WIDTH) {
      size_t k_ti = k + ti;
      size_t tk;
      es[ti][tj] = 0.0f;
      if(j < m && k_ti < n) {
        es[ti][tj] = exp(a[m * k_ti + j]);
      }
      __syncthreads();
      for(tk = 0; tk < TILE_WIDTH; tk += 4) {
        float4 ev;
        ev.x = es[tk + 0][tj];
        ev.y = es[tk + 1][tj];
        ev.z = es[tk + 2][tj];
        ev.w = es[tk + 3][tj];
        sum += ev.x + ev.y + ev.z + ev.w;
      }
      __syncthreads();
    }
    if(i < n && j < m) {
      b[m * i + j] = exp(a[m * i + j]) / sum;
    }
  }

  __global__ void softmax_at(const float *a, float *b, size_t n, size_t m)
  {
    __shared__ float es[TILE_WIDTH][TILE_WIDTH];
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    size_t k;
    size_t ti = threadIdx.x;
    size_t tj = threadIdx.y;
    size_t n_j = n * j;
    float sum = 0.0f;
    for(k = 0; k < n; k += TILE_WIDTH) {
      size_t k_ti = k + ti;
      size_t tk;
      es[ti][tj] = 0.0f;
      if(j < m && k_ti < n) {
        es[ti][tj] = exp(a[n_j + k_ti]);
      }
      __syncthreads();
      for(tk = 0; tk < TILE_WIDTH; tk += 4) {
        float4 ev;
        ev.x = es[tk + 0][tj];
        ev.y = es[tk + 1][tj];
        ev.z = es[tk + 2][tj];
        ev.w = es[tk + 3][tj];
        sum += ev.x + ev.y + ev.z + ev.w;
      }
      __syncthreads();
    }
    if(i < n && j < m) {
      b[m * i + j] = exp(a[n * j + i]) / sum;
    }
  }
}
