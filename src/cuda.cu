//
// Copyright (c) 2025 Łukasz Szpakowski
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
#define TILE_WIDTH      32

#define MTHREAD_COUNT   16
#define MTILE_WIDTH     (MTHREAD_COUNT << 2)

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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 2;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 2;
    size_t k;
    size_t ti = threadIdx.x;
    size_t tj = threadIdx.y;
    size_t bi = ti << 2;
    size_t bj = tj << 2;
    float ar1;
    float ar2;
    float ar3;
    float ar4;
    float br1;
    float br2;
    float br3;
    float br4;
    float cr11 = 0.0f;
    float cr12 = 0.0f;
    float cr13 = 0.0f;
    float cr14 = 0.0f;
    float cr21 = 0.0f;
    float cr22 = 0.0f;
    float cr23 = 0.0f;
    float cr24 = 0.0f;
    float cr31 = 0.0f;
    float cr32 = 0.0f;
    float cr33 = 0.0f;
    float cr34 = 0.0f;
    float cr41 = 0.0f;
    float cr42 = 0.0f;
    float cr43 = 0.0f;
    float cr44 = 0.0f;
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
      as[bi + 2][tj] = 0.0f;
      if(i + 2 < n && k + tj < l) {
        as[bi + 2][tj] = a[l * (i + 2) + k + tj];
      }
      as[bi + 3][tj] = 0.0f;
      if(i + 3 < n && k + tj < l) {
        as[bi + 3][tj] = a[l * (i + 3) + k + tj];
      }
      bs[ti][bj + 0] = 0.0f;
      if(j + 0 < m && k + ti < l) {
        bs[ti][bj + 0] = b[m * (k + ti) + j + 0];
      }
      bs[ti][bj + 1] = 0.0f;
      if(j + 1 < m && k + ti < l) {
        bs[ti][bj + 1] = b[m * (k + ti) + j + 1];
      }
      bs[ti][bj + 2] = 0.0f;
      if(j + 2 < m && k + ti < l) {
        bs[ti][bj + 2] = b[m * (k + ti) + j + 2];
      }
      bs[ti][bj + 3] = 0.0f;
      if(j + 3 < m && k + ti < l) {
        bs[ti][bj + 3] = b[m * (k + ti) + j + 3];
      }
      __syncthreads();
      for(tk = 0; tk < MTHREAD_COUNT; tk++) {
        ar1 = as[bi + 0][tk];
        ar2 = as[bi + 1][tk];
        ar3 = as[bi + 2][tk];
        ar4 = as[bi + 3][tk];
        br1 = bs[tk][bj + 0];
        br2 = bs[tk][bj + 1];
        br3 = bs[tk][bj + 2];
        br4 = bs[tk][bj + 3];
        cr11 += ar1 * br1;
        cr12 += ar1 * br2;
        cr13 += ar1 * br3;
        cr14 += ar1 * br4;
        cr21 += ar2 * br1;
        cr22 += ar2 * br2;
        cr23 += ar2 * br3;
        cr24 += ar2 * br4;
        cr31 += ar3 * br1;
        cr32 += ar3 * br2;
        cr33 += ar3 * br3;
        cr34 += ar3 * br4;
        cr41 += ar4 * br1;
        cr42 += ar4 * br2;
        cr43 += ar4 * br3;
        cr44 += ar4 * br4;
      }
      __syncthreads();
    }
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = cr11;
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = cr12;
    }
    if(i + 0 < n && j + 2 < m) {
      c[m * (i + 0) + j + 2] = cr13;
    }
    if(i + 0 < n && j + 3 < m) {
      c[m * (i + 0) + j + 3] = cr14;
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = cr21;
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = cr22;
    }
    if(i + 1 < n && j + 2 < m) {
      c[m * (i + 1) + j + 2] = cr23;
    }
    if(i + 1 < n && j + 3 < m) {
      c[m * (i + 1) + j + 3] = cr24;
    }
    if(i + 2 < n && j + 0 < m) {
      c[m * (i + 2) + j + 0] = cr31;
    }
    if(i + 2 < n && j + 1 < m) {
      c[m * (i + 2) + j + 1] = cr32;
    }
    if(i + 2 < n && j + 2 < m) {
      c[m * (i + 2) + j + 2] = cr33;
    }
    if(i + 2 < n && j + 3 < m) {
      c[m * (i + 2) + j + 3] = cr34;
    }
    if(i + 3 < n && j + 0 < m) {
      c[m * (i + 3) + j + 0] = cr41;
    }
    if(i + 3 < n && j + 1 < m) {
      c[m * (i + 3) + j + 1] = cr42;
    }
    if(i + 3 < n && j + 2 < m) {
      c[m * (i + 3) + j + 2] = cr43;
    }
    if(i + 3 < n && j + 3 < m) {
      c[m * (i + 3) + j + 3] = cr44;
    }
  }

  __global__ void mul_at_b(const float *a, const float *b, float *c, size_t n, size_t m, size_t l)
  {
    __shared__ float as[MTILE_WIDTH][MTHREAD_COUNT];
    __shared__ float bs[MTHREAD_COUNT][MTILE_WIDTH];
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 2;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 2;
    size_t k;
    size_t ti = threadIdx.x;
    size_t tj = threadIdx.y;
    size_t bi = ti << 2;
    size_t bj = tj << 2;
    float ar1;
    float ar2;
    float ar3;
    float ar4;
    float br1;
    float br2;
    float br3;
    float br4;
    float cr11 = 0.0f;
    float cr12 = 0.0f;
    float cr13 = 0.0f;
    float cr14 = 0.0f;
    float cr21 = 0.0f;
    float cr22 = 0.0f;
    float cr23 = 0.0f;
    float cr24 = 0.0f;
    float cr31 = 0.0f;
    float cr32 = 0.0f;
    float cr33 = 0.0f;
    float cr34 = 0.0f;
    float cr41 = 0.0f;
    float cr42 = 0.0f;
    float cr43 = 0.0f;
    float cr44 = 0.0f;
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
      as[bi + 2][tj] = 0.0f;
      if(i + 2 < n && k + tj < l) {
        as[bi + 2][tj] = a[n * (k + tj) + i + 2];
      }
      as[bi + 3][tj] = 0.0f;
      if(i + 3 < n && k + tj < l) {
        as[bi + 3][tj] = a[n * (k + tj) + i + 3];
      }
      bs[ti][bj + 0] = 0.0f;
      if(j + 0 < m && k + ti < l) {
        bs[ti][bj + 0] = b[m * (k + ti) + j + 0];
      }
      bs[ti][bj + 1] = 0.0f;
      if(j + 1 < m && k + ti < l) {
        bs[ti][bj + 1] = b[m * (k + ti) + j + 1];
      }
      bs[ti][bj + 2] = 0.0f;
      if(j + 2 < m && k + ti < l) {
        bs[ti][bj + 2] = b[m * (k + ti) + j + 2];
      }
      bs[ti][bj + 3] = 0.0f;
      if(j + 3 < m && k + ti < l) {
        bs[ti][bj + 3] = b[m * (k + ti) + j + 3];
      }
      __syncthreads();
      for(tk = 0; tk < MTHREAD_COUNT; tk++) {
        ar1 = as[bi + 0][tk];
        ar2 = as[bi + 1][tk];
        ar3 = as[bi + 2][tk];
        ar4 = as[bi + 3][tk];
        br1 = bs[tk][bj + 0];
        br2 = bs[tk][bj + 1];
        br3 = bs[tk][bj + 2];
        br4 = bs[tk][bj + 3];
        cr11 += ar1 * br1;
        cr12 += ar1 * br2;
        cr13 += ar1 * br3;
        cr14 += ar1 * br4;
        cr21 += ar2 * br1;
        cr22 += ar2 * br2;
        cr23 += ar2 * br3;
        cr24 += ar2 * br4;
        cr31 += ar3 * br1;
        cr32 += ar3 * br2;
        cr33 += ar3 * br3;
        cr34 += ar3 * br4;
        cr41 += ar4 * br1;
        cr42 += ar4 * br2;
        cr43 += ar4 * br3;
        cr44 += ar4 * br4;
      }
      __syncthreads();
    }
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = cr11;
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = cr12;
    }
    if(i + 0 < n && j + 2 < m) {
      c[m * (i + 0) + j + 2] = cr13;
    }
    if(i + 0 < n && j + 3 < m) {
      c[m * (i + 0) + j + 3] = cr14;
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = cr21;
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = cr22;
    }
    if(i + 1 < n && j + 2 < m) {
      c[m * (i + 1) + j + 2] = cr23;
    }
    if(i + 1 < n && j + 3 < m) {
      c[m * (i + 1) + j + 3] = cr24;
    }
    if(i + 2 < n && j + 0 < m) {
      c[m * (i + 2) + j + 0] = cr31;
    }
    if(i + 2 < n && j + 1 < m) {
      c[m * (i + 2) + j + 1] = cr32;
    }
    if(i + 2 < n && j + 2 < m) {
      c[m * (i + 2) + j + 2] = cr33;
    }
    if(i + 2 < n && j + 3 < m) {
      c[m * (i + 2) + j + 3] = cr34;
    }
    if(i + 3 < n && j + 0 < m) {
      c[m * (i + 3) + j + 0] = cr41;
    }
    if(i + 3 < n && j + 1 < m) {
      c[m * (i + 3) + j + 1] = cr42;
    }
    if(i + 3 < n && j + 2 < m) {
      c[m * (i + 3) + j + 2] = cr43;
    }
    if(i + 3 < n && j + 3 < m) {
      c[m * (i + 3) + j + 3] = cr44;
    }
  }

  __global__ void mul_a_bt(const float *a, const float *b, float *c, size_t n, size_t m, size_t l)
  {
    __shared__ float as[MTILE_WIDTH][MTHREAD_COUNT];
    __shared__ float bs[MTHREAD_COUNT][MTILE_WIDTH];
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 2;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 2;
    size_t k;
    size_t ti = threadIdx.x;
    size_t tj = threadIdx.y;
    size_t bi = ti << 2;
    size_t bj = tj << 2;
    float ar1;
    float ar2;
    float ar3;
    float ar4;
    float br1;
    float br2;
    float br3;
    float br4;
    float cr11 = 0.0f;
    float cr12 = 0.0f;
    float cr13 = 0.0f;
    float cr14 = 0.0f;
    float cr21 = 0.0f;
    float cr22 = 0.0f;
    float cr23 = 0.0f;
    float cr24 = 0.0f;
    float cr31 = 0.0f;
    float cr32 = 0.0f;
    float cr33 = 0.0f;
    float cr34 = 0.0f;
    float cr41 = 0.0f;
    float cr42 = 0.0f;
    float cr43 = 0.0f;
    float cr44 = 0.0f;
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
      as[bi + 2][tj] = 0.0f;
      if(i + 2 < n && k + tj < l) {
        as[bi + 2][tj] = a[l * (i + 2) + k + tj];
      }
      as[bi + 3][tj] = 0.0f;
      if(i + 3 < n && k + tj < l) {
        as[bi + 3][tj] = a[l * (i + 3) + k + tj];
      }
      bs[ti][bj + 0] = 0.0f;
      if(j + 0 < m && k + ti < l) {
        bs[ti][bj + 0] = b[l * (j + 0) + k + ti];
      }
      bs[ti][bj + 1] = 0.0f;
      if(j + 1 < m && k + ti < l) {
        bs[ti][bj + 1] = b[l * (j + 1) + k + ti];
      }
      bs[ti][bj + 2] = 0.0f;
      if(j + 2 < m && k + ti < l) {
        bs[ti][bj + 2] = b[l * (j + 2) + k + ti];
      }
      bs[ti][bj + 3] = 0.0f;
      if(j + 3 < m && k + ti < l) {
        bs[ti][bj + 3] = b[l * (j + 3) + k + ti];
      }
      __syncthreads();
      for(tk = 0; tk < MTHREAD_COUNT; tk++) {
        ar1 = as[bi + 0][tk];
        ar2 = as[bi + 1][tk];
        ar3 = as[bi + 2][tk];
        ar4 = as[bi + 3][tk];
        br1 = bs[tk][bj + 0];
        br2 = bs[tk][bj + 1];
        br3 = bs[tk][bj + 2];
        br4 = bs[tk][bj + 3];
        cr11 += ar1 * br1;
        cr12 += ar1 * br2;
        cr13 += ar1 * br3;
        cr14 += ar1 * br4;
        cr21 += ar2 * br1;
        cr22 += ar2 * br2;
        cr23 += ar2 * br3;
        cr24 += ar2 * br4;
        cr31 += ar3 * br1;
        cr32 += ar3 * br2;
        cr33 += ar3 * br3;
        cr34 += ar3 * br4;
        cr41 += ar4 * br1;
        cr42 += ar4 * br2;
        cr43 += ar4 * br3;
        cr44 += ar4 * br4;
      }
      __syncthreads();
    }
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = cr11;
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = cr12;
    }
    if(i + 0 < n && j + 2 < m) {
      c[m * (i + 0) + j + 2] = cr13;
    }
    if(i + 0 < n && j + 3 < m) {
      c[m * (i + 0) + j + 3] = cr14;
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = cr21;
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = cr22;
    }
    if(i + 1 < n && j + 2 < m) {
      c[m * (i + 1) + j + 2] = cr23;
    }
    if(i + 1 < n && j + 3 < m) {
      c[m * (i + 1) + j + 3] = cr24;
    }
    if(i + 2 < n && j + 0 < m) {
      c[m * (i + 2) + j + 0] = cr31;
    }
    if(i + 2 < n && j + 1 < m) {
      c[m * (i + 2) + j + 1] = cr32;
    }
    if(i + 2 < n && j + 2 < m) {
      c[m * (i + 2) + j + 2] = cr33;
    }
    if(i + 2 < n && j + 3 < m) {
      c[m * (i + 2) + j + 3] = cr34;
    }
    if(i + 3 < n && j + 0 < m) {
      c[m * (i + 3) + j + 0] = cr41;
    }
    if(i + 3 < n && j + 1 < m) {
      c[m * (i + 3) + j + 1] = cr42;
    }
    if(i + 3 < n && j + 2 < m) {
      c[m * (i + 3) + j + 2] = cr43;
    }
    if(i + 3 < n && j + 3 < m) {
      c[m * (i + 3) + j + 3] = cr44;
    }
  }

  __global__ void mul_at_bt(const float *a, const float *b, float *c, size_t n, size_t m, size_t l)
  {
    __shared__ float as[MTILE_WIDTH][MTHREAD_COUNT];
    __shared__ float bs[MTHREAD_COUNT][MTILE_WIDTH];
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 2;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 2;
    size_t k;
    size_t ti = threadIdx.x;
    size_t tj = threadIdx.y;
    size_t bi = ti << 2;
    size_t bj = tj << 2;
    float ar1;
    float ar2;
    float ar3;
    float ar4;
    float br1;
    float br2;
    float br3;
    float br4;
    float cr11 = 0.0f;
    float cr12 = 0.0f;
    float cr13 = 0.0f;
    float cr14 = 0.0f;
    float cr21 = 0.0f;
    float cr22 = 0.0f;
    float cr23 = 0.0f;
    float cr24 = 0.0f;
    float cr31 = 0.0f;
    float cr32 = 0.0f;
    float cr33 = 0.0f;
    float cr34 = 0.0f;
    float cr41 = 0.0f;
    float cr42 = 0.0f;
    float cr43 = 0.0f;
    float cr44 = 0.0f;
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
      as[bi + 2][tj] = 0.0f;
      if(i + 2 < n && k + tj < l) {
        as[bi + 2][tj] = a[n * (k + tj) + i + 2];
      }
      as[bi + 3][tj] = 0.0f;
      if(i + 3 < n && k + tj < l) {
        as[bi + 3][tj] = a[n * (k + tj) + i + 3];
      }
      bs[ti][bj + 0] = 0.0f;
      if(j + 0 < m && k + ti < l) {
        bs[ti][bj + 0] = b[l * (j + 0) + k + ti];
      }
      bs[ti][bj + 1] = 0.0f;
      if(j + 1 < m && k + ti < l) {
        bs[ti][bj + 1] = b[l * (j + 1) + k + ti];
      }
      bs[ti][bj + 2] = 0.0f;
      if(j + 2 < m && k + ti < l) {
        bs[ti][bj + 2] = b[l * (j + 2) + k + ti];
      }
      bs[ti][bj + 3] = 0.0f;
      if(j + 3 < m && k + ti < l) {
        bs[ti][bj + 3] = b[l * (j + 3) + k + ti];
      }
      __syncthreads();
      for(tk = 0; tk < MTHREAD_COUNT; tk++) {
        ar1 = as[bi + 0][tk];
        ar2 = as[bi + 1][tk];
        ar3 = as[bi + 2][tk];
        ar4 = as[bi + 3][tk];
        br1 = bs[tk][bj + 0];
        br2 = bs[tk][bj + 1];
        br3 = bs[tk][bj + 2];
        br4 = bs[tk][bj + 3];
        cr11 += ar1 * br1;
        cr12 += ar1 * br2;
        cr13 += ar1 * br3;
        cr14 += ar1 * br4;
        cr21 += ar2 * br1;
        cr22 += ar2 * br2;
        cr23 += ar2 * br3;
        cr24 += ar2 * br4;
        cr31 += ar3 * br1;
        cr32 += ar3 * br2;
        cr33 += ar3 * br3;
        cr34 += ar3 * br4;
        cr41 += ar4 * br1;
        cr42 += ar4 * br2;
        cr43 += ar4 * br3;
        cr44 += ar4 * br4;
      }
      __syncthreads();
    }
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = cr11;
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = cr12;
    }
    if(i + 0 < n && j + 2 < m) {
      c[m * (i + 0) + j + 2] = cr13;
    }
    if(i + 0 < n && j + 3 < m) {
      c[m * (i + 0) + j + 3] = cr14;
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = cr21;
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = cr22;
    }
    if(i + 1 < n && j + 2 < m) {
      c[m * (i + 1) + j + 2] = cr23;
    }
    if(i + 1 < n && j + 3 < m) {
      c[m * (i + 1) + j + 3] = cr24;
    }
    if(i + 2 < n && j + 0 < m) {
      c[m * (i + 2) + j + 0] = cr31;
    }
    if(i + 2 < n && j + 1 < m) {
      c[m * (i + 2) + j + 1] = cr32;
    }
    if(i + 2 < n && j + 2 < m) {
      c[m * (i + 2) + j + 2] = cr33;
    }
    if(i + 2 < n && j + 3 < m) {
      c[m * (i + 2) + j + 3] = cr34;
    }
    if(i + 3 < n && j + 0 < m) {
      c[m * (i + 3) + j + 0] = cr41;
    }
    if(i + 3 < n && j + 1 < m) {
      c[m * (i + 3) + j + 1] = cr42;
    }
    if(i + 3 < n && j + 2 < m) {
      c[m * (i + 3) + j + 2] = cr43;
    }
    if(i + 3 < n && j + 3 < m) {
      c[m * (i + 3) + j + 3] = cr44;
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
      size_t tk;
      es[ti][tj] = 0.0f;
      if(j < m && k + ti < n) {
        es[ti][tj] = exp(a[m * (k + ti) + j]);
      }
      __syncthreads();
      for(tk = 0; tk < TILE_WIDTH; tk++) {
        sum += es[tk][tj];
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
    float sum = 0.0f;
    for(k = 0; k < n; k += TILE_WIDTH) {
      size_t tk;
      es[ti][tj] = 0.0f;
      if(j < m && k + ti < n) {
        es[ti][tj] = exp(a[n * j + k + ti]);
      }
      __syncthreads();
      for(tk = 0; tk < TILE_WIDTH; tk++) {
        sum += es[tk][tj];
      }
      __syncthreads();
    }
    if(i < n && j < m) {
      b[m * i + j] = exp(a[n * j + i]) / sum;
    }
  }
}
