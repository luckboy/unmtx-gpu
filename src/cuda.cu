//
// Copyright (c) 2025 Łukasz Szpakowski
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
#define TILE_WIDTH      32

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
    __shared__ float as[TILE_WIDTH][TILE_WIDTH];
    __shared__ float bs[TILE_WIDTH][TILE_WIDTH];
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    size_t k;
    size_t ti = threadIdx.x;
    size_t tj = threadIdx.y;
    size_t l_i = l * i;
    float cij = 0.0f;
    for(k = 0; k < l; k += TILE_WIDTH) {
      size_t k_ti = k + ti;
      size_t k_tj = k + tj;
      size_t tk;
      as[ti][tj] = 0.0f;
      if(i < n && k_tj < l) {
        as[ti][tj] = a[l_i + k_tj];
      }
      bs[ti][tj] = 0.0f;
      if(j < m && k_ti < l) {
        bs[ti][tj] = b[m * k_ti + j];
      }
      __syncthreads();
      for(tk = 0; tk < TILE_WIDTH; tk += 4) {
        float4 av;
        float4 bv;
        av.x = as[ti][tk + 0];
        av.y = as[ti][tk + 1];
        av.z = as[ti][tk + 2];
        av.w = as[ti][tk + 3];
        bv.x = bs[tk + 0][tj];
        bv.y = bs[tk + 1][tj];
        bv.z = bs[tk + 2][tj];
        bv.w = bs[tk + 3][tj];
        cij += av.x * bv.x + av.y * bv.y + av.z * bv.z + av.w * bv.w;
      }
      __syncthreads();
    }
    if(i < n && j < m) {
      c[m * i + j] = cij;
    }
  }

  __global__ void mul_at_b(const float *a, const float *b, float *c, size_t n, size_t m, size_t l)
  {
    __shared__ float as[TILE_WIDTH][TILE_WIDTH];
    __shared__ float bs[TILE_WIDTH][TILE_WIDTH];
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    size_t k;
    size_t ti = threadIdx.x;
    size_t tj = threadIdx.y;
    float cij = 0.0f;
    for(k = 0; k < l; k += TILE_WIDTH) {
      size_t k_ti = k + ti;
      size_t k_tj = k + tj;
      size_t tk;
      as[ti][tj] = 0.0f;
      if(i < n && k_tj < l) {
        as[ti][tj] = a[n * k_tj + i];
      }
      bs[ti][tj] = 0.0f;
      if(j < m && k_ti < l) {
        bs[ti][tj] = b[m * k_ti + j];
      }
      __syncthreads();
      for(tk = 0; tk < TILE_WIDTH; tk += 4) {
        float4 av;
        float4 bv;
        av.x = as[ti][tk + 0];
        av.y = as[ti][tk + 1];
        av.z = as[ti][tk + 2];
        av.w = as[ti][tk + 3];
        bv.x = bs[tk + 0][tj];
        bv.y = bs[tk + 1][tj];
        bv.z = bs[tk + 2][tj];
        bv.w = bs[tk + 3][tj];
        cij += av.x * bv.x + av.y * bv.y + av.z * bv.z + av.w * bv.w;
      }
      __syncthreads();
    }
    if(i < n && j < m) {
      c[m * i + j] = cij;
    }
  }

  __global__ void mul_a_bt(const float *a, const float *b, float *c, size_t n, size_t m, size_t l)
  {
    __shared__ float as[TILE_WIDTH][TILE_WIDTH];
    __shared__ float bs[TILE_WIDTH][TILE_WIDTH];
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    size_t k;
    size_t ti = threadIdx.x;
    size_t tj = threadIdx.y;
    size_t l_i = l * i;
    size_t l_j = l * j;
    float cij = 0.0f;
    for(k = 0; k < l; k += TILE_WIDTH) {
      size_t k_ti = k + ti;
      size_t k_tj = k + tj;
      size_t tk;
      as[ti][tj] = 0.0f;
      if(i < n && k_tj < l) {
        as[ti][tj] = a[l_i + k_tj];
      }
      bs[ti][tj] = 0.0f;
      if(j < m && k_ti < l) {
        bs[ti][tj] = b[l_j + k_ti];
      }
      __syncthreads();
      for(tk = 0; tk < TILE_WIDTH; tk += 4) {
        float4 av;
        float4 bv;
        av.x = as[ti][tk + 0];
        av.y = as[ti][tk + 1];
        av.z = as[ti][tk + 2];
        av.w = as[ti][tk + 3];
        bv.x = bs[tk + 0][tj];
        bv.y = bs[tk + 1][tj];
        bv.z = bs[tk + 2][tj];
        bv.w = bs[tk + 3][tj];
        cij += av.x * bv.x + av.y * bv.y + av.z * bv.z + av.w * bv.w;
      }
      __syncthreads();
    }
    if(i < n && j < m) {
      c[m * i + j] = cij;
    }
  }

  __global__ void mul_at_bt(const float *a, const float *b, float *c, size_t n, size_t m, size_t l)
  {
    __shared__ float as[TILE_WIDTH][TILE_WIDTH];
    __shared__ float bs[TILE_WIDTH][TILE_WIDTH];
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    size_t k;
    size_t ti = threadIdx.x;
    size_t tj = threadIdx.y;
    size_t l_j = l * j;
    float cij = 0.0f;
    for(k = 0; k < l; k += TILE_WIDTH) {
      size_t k_ti = k + ti;
      size_t k_tj = k + tj;
      size_t tk;
      as[ti][tj] = 0.0f;
      if(i < n && k_tj < l) {
        as[ti][tj] = a[n * k_tj + i];
      }
      bs[ti][tj] = 0.0f;
      if(j < m && k_ti < l) {
        bs[ti][tj] = b[l_j + k_ti];
      }
      __syncthreads();
      for(tk = 0; tk < TILE_WIDTH; tk += 4) {
        float4 av;
        float4 bv;
        av.x = as[ti][tk + 0];
        av.y = as[ti][tk + 1];
        av.z = as[ti][tk + 2];
        av.w = as[ti][tk + 3];
        bv.x = bs[tk + 0][tj];
        bv.y = bs[tk + 1][tj];
        bv.z = bs[tk + 2][tj];
        bv.w = bs[tk + 3][tj];
        cij += av.x * bv.x + av.y * bv.y + av.z * bv.z + av.w * bv.w;
      }
      __syncthreads();
    }
    if(i < n && j < m) {
      c[m * i + j] = cij;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      size_t n_3 = n & ~((size_t) 3);
      size_t k;
      float sum = 0.0f;
      for(k = 0; k < n_3; k += 4) {
        float4 av;
        av.x = a[m * (k + 0) + j];
        av.y = a[m * (k + 1) + j];
        av.z = a[m * (k + 2) + j];
        av.w = a[m * (k + 3) + j];
        sum += expf(av.x) + expf(av.y) + expf(av.z) + expf(av.w);
      }
      for(; k < n; k++) {
        sum += expf(a[m * k + j]);
      }
      b[m * i + j] = exp(a[m * i + j]) / sum;
    }
  }

  __global__ void softmax_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      size_t n_j = n * j;
      size_t n_3 = n & ~((size_t) 3);
      size_t k;
      float sum = 0.0f;
      for(k = 0; k < n_3; k += 4) {
        float4 av;
        av.x = a[n_j + k + 0];
        av.y = a[n_j + k + 1];
        av.z = a[n_j + k + 2];
        av.w = a[n_j + k + 3];
        sum += expf(av.x) + expf(av.y) + expf(av.z) + expf(av.w);
      }
      for(; k < n; k++) {
        sum += expf(a[n_j + k]);
      }
      b[m * i + j] = exp(a[n * j + i]) / sum;
    }
  }
}
