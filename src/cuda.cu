//
// Copyright (c) 2025-2026 Łukasz Szpakowski
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
#define THREAD_SIZE     1024

#define MTHREAD_SIZE    16

#define MMA_TILE_WIDTH  64

extern "C" {
  __global__ void transpose_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = a[n * (j + 0) + i + 0];
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = a[n * (j + 1) + i + 0];
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = a[n * (j + 0) + i + 1];
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = a[n * (j + 1) + i + 1];
    }
  }
  
  __global__ void add_a_b(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[m * (i + 0) + j + 0] + b[m * (i + 0) + j + 0];
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[m * (i + 0) + j + 1] + b[m * (i + 0) + j + 1];
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[m * (i + 1) + j + 0] + b[m * (i + 1) + j + 0];
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[m * (i + 1) + j + 1] + b[m * (i + 1) + j + 1];
    }
  }

  __global__ void add_at_b(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[n * (j + 0) + i + 0] + b[m * (i + 0) + j + 0];
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[n * (j + 1) + i + 0] + b[m * (i + 0) + j + 1];
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[n * (j + 0) + i + 1] + b[m * (i + 1) + j + 0];
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[n * (j + 1) + i + 1] + b[m * (i + 1) + j + 1];
    }
  }

  __global__ void add_a_bt(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[m * (i + 0) + j + 0] + b[n * (j + 0) + i + 0];
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[m * (i + 0) + j + 1] + b[n * (j + 1) + i + 0];
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[m * (i + 1) + j + 0] + b[n * (j + 0) + i + 1];
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[m * (i + 1) + j + 1] + b[n * (j + 1) + i + 1];
    }
  }

  __global__ void add_at_bt(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[n * (j + 0) + i + 0] + b[n * (j + 0) + i + 0];
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[n * (j + 1) + i + 0] + b[n * (j + 1) + i + 0];
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[n * (j + 0) + i + 1] + b[n * (j + 0) + i + 1];
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[n * (j + 1) + i + 1] + b[n * (j + 1) + i + 1];
    }
  }

  __global__ void sub_a_b(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[m * (i + 0) + j + 0] - b[m * (i + 0) + j + 0];
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[m * (i + 0) + j + 1] - b[m * (i + 0) + j + 1];
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[m * (i + 1) + j + 0] - b[m * (i + 1) + j + 0];
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[m * (i + 1) + j + 1] - b[m * (i + 1) + j + 1];
    }
  }

  __global__ void sub_at_b(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[n * (j + 0) + i + 0] - b[m * (i + 0) + j + 0];
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[n * (j + 1) + i + 0] - b[m * (i + 0) + j + 1];
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[n * (j + 0) + i + 1] - b[m * (i + 1) + j + 0];
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[n * (j + 1) + i + 1] - b[m * (i + 1) + j + 1];
    }
  }

  __global__ void sub_a_bt(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j +  0 < m) {
      c[m * (i + 0) + j + 0] = a[m * (i + 0) + j + 0] - b[n * (j + 0) + i + 0];
    }
    if(i + 0 < n && j +  1 < m) {
      c[m * (i + 0) + j + 1] = a[m * (i + 0) + j + 1] - b[n * (j + 1) + i + 0];
    }
    if(i + 1 < n && j +  0 < m) {
      c[m * (i + 1) + j + 0] = a[m * (i + 1) + j + 0] - b[n * (j + 0) + i + 1];
    }
    if(i + 1 < n && j +  1 < m) {
      c[m * (i + 1) + j + 1] = a[m * (i + 1) + j + 1] - b[n * (j + 1) + i + 1];
    }
  }

  __global__ void sub_at_bt(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[n * (j + 0) + i + 0] - b[n * (j + 0) + i + 0];
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[n * (j + 1) + i + 0] - b[n * (j + 1) + i + 0];
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[n * (j + 0) + i + 1] - b[n * (j + 0) + i + 1];
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[n * (j + 1) + i + 1] - b[n * (j + 1) + i + 1];
    }
  }

#ifdef UNMTX_GPU_MMA

  static inline __device__ unsigned float_to_tf32(float x)
  {
    unsigned y;
    asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(y) : "f"(x));
    return y;
  }

  __global__ void mul_a_b(const float *a, const float *b, float *c, size_t n, size_t m, size_t l)
  {
    size_t i = ((size_t) blockIdx.y) * MMA_TILE_WIDTH;
    size_t j = ((size_t) blockIdx.x) * MMA_TILE_WIDTH;
    size_t k;
    size_t tx = threadIdx.x >> 5;
    size_t stx = threadIdx.x & 31;
    size_t bi = (tx >> 3) << 4;
    size_t bj = (tx & 7) << 3;
    size_t ari = stx >> 2;
    size_t ark = stx & 3;
    size_t brj = ari;
    size_t brk = ark;
    size_t cri = ari;
    size_t crj = ark;
    unsigned ar1;
    unsigned ar2;
    unsigned ar3;
    unsigned ar4;
    unsigned br1;
    unsigned br2;
    float cr1 = 0.0f;
    float cr2 = 0.0f;
    float cr3 = 0.0f;
    float cr4 = 0.0f;
    unsigned zero = float_to_tf32(0.0f);
    for(k = 0; k < l; k += 8) {
      ar1 = zero;
      if(i + bi + 0 + ari < n && k + 0 + ark < l) {
        ar1 = float_to_tf32(a[l * (i + bi + 0 + ari) + k + 0 + ark]);
      }
      ar2 = zero;
      if(i + bi + 8 + ari < n && k + 0 + ark < l) {
        ar2 = float_to_tf32(a[l * (i + bi + 8 + ari) + k + 0 + ark]);
      }
      ar3 = zero;
      if(i + bi + 0 + ari < n && k + 4 + ark < l) {
        ar3 = float_to_tf32(a[l * (i + bi + 0 + ari) + k + 4 + ark]);
      }
      ar4 = zero;
      if(i + bi + 8 + ari < n && k + 4 + ark < l) {
        ar4 = float_to_tf32(a[l * (i + bi + 8 + ari) + k + 4 + ark]);
      }
      br1 = zero;
      if(j + bj + brj < m && k + 0 + brk < l) {
        br1 = float_to_tf32(b[m * (k + 0 + brk) + j + bj + brj]);
      }
      br2 = zero;
      if(j + bj + brj < m && k + 4 + brk < l) {
        br2 = float_to_tf32(b[m * (k + 4 + brk) + j + bj + brj]);
      }
      asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32\n"
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(cr1), "+f"(cr2), "+f"(cr3), "+f"(cr4)
        : "r"(ar1), "r"(ar2), "r"(ar3), "r"(ar4),
        "r"(br1), "r"(br2));
    }
    if(i + bi + 0 + cri < n && j + bj + (crj << 1) + 0 < m) {
      c[m * (i + bi + 0 + cri) + j + bj + (crj << 1) + 0] = cr1;
    }
    if(i + bi + 0 + cri < n && j + bj + (crj << 1) + 1 < m) {
      c[m * (i + bi + 0 + cri) + j + bj + (crj << 1) + 1] = cr2;
    }
    if(i + bi + 8 + cri < n && j + bj + (crj << 1) + 0 < m) {
      c[m * (i + bi + 8 + cri) + j + bj + (crj << 1) + 0] = cr3;
    }
    if(i + bi + 8 + cri < n && j + bj + (crj << 1) + 1) {
      c[m * (i + bi + 8 + cri) + j + bj + (crj << 1) + 1] = cr4;
    }
  }

  __global__ void mul_at_b(const float *a, const float *b, float *c, size_t n, size_t m, size_t l)
  {
    size_t i = ((size_t) blockIdx.x) * MMA_TILE_WIDTH;
    size_t j = ((size_t) blockIdx.y) * MMA_TILE_WIDTH;
    size_t k;
    size_t tx = threadIdx.x >> 5;
    size_t stx = threadIdx.x & 31;
    size_t bi = (tx >> 3) << 4;
    size_t bj = (tx & 7) << 3;
    size_t ari = stx >> 2;
    size_t ark = stx & 3;
    size_t brj = ari;
    size_t brk = ark;
    size_t cri = ari;
    size_t crj = ark;
    unsigned ar1;
    unsigned ar2;
    unsigned ar3;
    unsigned ar4;
    unsigned br1;
    unsigned br2;
    float cr1 = 0.0f;
    float cr2 = 0.0f;
    float cr3 = 0.0f;
    float cr4 = 0.0f;
    unsigned zero = float_to_tf32(0.0f);
    for(k = 0; k < l; k += 8) {
      ar1 = zero;
      if(i + bi + 0 + ari < n && k + 0 + ark < l) {
        ar1 = float_to_tf32(a[n * (k + 0 + ark) + i + bi + 0 + ari]);
      }
      ar2 = zero;
      if(i + bi + 8 + ari < n && k + 0 + ark < l) {
        ar2 = float_to_tf32(a[n * (k + 0 + ark) + i + bi + 8 + ari]);
      }
      ar3 = zero;
      if(i + bi + 0 + ari < n && k + 4 + ark < l) {
        ar3 = float_to_tf32(a[n * (k + 4 + ark) + i + bi + 0 + ari]);
      }
      ar4 = zero;
      if(i + bi + 8 + ari < n && k + 4 + ark < l) {
        ar4 = float_to_tf32(a[n * (k + 4 + ark) + i + bi + 8 + ari]);
      }
      br1 = zero;
      if(j + bj + brj < m && k + 0 + brk < l) {
        br1 = float_to_tf32(b[m * (k + 0 + brk) + j + bj + brj]);
      }
      br2 = zero;
      if(j + bj + brj < m && k + 4 + brk < l) {
        br2 = float_to_tf32(b[m * (k + 4 + brk) + j + bj + brj]);
      }
      asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32\n"
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(cr1), "+f"(cr2), "+f"(cr3), "+f"(cr4)
        : "r"(ar1), "r"(ar2), "r"(ar3), "r"(ar4),
        "r"(br1), "r"(br2));
    }
    if(i + bi + 0 + cri < n && j + bj + (crj << 1) + 0 < m) {
      c[m * (i + bi + 0 + cri) + j + bj + (crj << 1) + 0] = cr1;
    }
    if(i + bi + 0 + cri < n && j + bj + (crj << 1) + 1 < m) {
      c[m * (i + bi + 0 + cri) + j + bj + (crj << 1) + 1] = cr2;
    }
    if(i + bi + 8 + cri < n && j + bj + (crj << 1) + 0 < m) {
      c[m * (i + bi + 8 + cri) + j + bj + (crj << 1) + 0] = cr3;
    }
    if(i + bi + 8 + cri < n && j + bj + (crj << 1) + 1) {
      c[m * (i + bi + 8 + cri) + j + bj + (crj << 1) + 1] = cr4;
    }
  }

  __global__ void mul_a_bt(const float *a, const float *b, float *c, size_t n, size_t m, size_t l)
  {
    size_t i = ((size_t) blockIdx.y) * MMA_TILE_WIDTH;
    size_t j = ((size_t) blockIdx.x) * MMA_TILE_WIDTH;
    size_t k;
    size_t tx = threadIdx.x >> 5;
    size_t stx = threadIdx.x & 31;
    size_t bi = (tx >> 3) << 4;
    size_t bj = (tx & 7) << 3;
    size_t ari = stx >> 2;
    size_t ark = stx & 3;
    size_t brj = ari;
    size_t brk = ark;
    size_t cri = ari;
    size_t crj = ark;
    unsigned ar1;
    unsigned ar2;
    unsigned ar3;
    unsigned ar4;
    unsigned br1;
    unsigned br2;
    float cr1 = 0.0f;
    float cr2 = 0.0f;
    float cr3 = 0.0f;
    float cr4 = 0.0f;
    unsigned zero = float_to_tf32(0.0f);
    for(k = 0; k < l; k += 8) {
      ar1 = zero;
      if(i + bi + 0 + ari < n && k + 0 + ark < l) {
        ar1 = float_to_tf32(a[l * (i + bi + 0 + ari) + k + 0 + ark]);
      }
      ar2 = zero;
      if(i + bi + 8 + ari < n && k + 0 + ark < l) {
        ar2 = float_to_tf32(a[l * (i + bi + 8 + ari) + k + 0 + ark]);
      }
      ar3 = zero;
      if(i + bi + 0 + ari < n && k + 4 + ark < l) {
        ar3 = float_to_tf32(a[l * (i + bi + 0 + ari) + k + 4 + ark]);
      }
      ar4 = zero;
      if(i + bi + 8 + ari < n && k + 4 + ark < l) {
        ar4 = float_to_tf32(a[l * (i + bi + 8 + ari) + k + 4 + ark]);
      }
      br1 = zero;
      if(j + bj + brj < m && k + 0 + brk < l) {
        br1 = float_to_tf32(b[l * (j + bj + brj) + k + 0 + brk]);
      }
      br2 = zero;
      if(j + bj + brj < m && k + 4 + brk < l) {
        br2 = float_to_tf32(b[l * (j + bj + brj) + k + 4 + brk]);
      }
      asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32\n"
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(cr1), "+f"(cr2), "+f"(cr3), "+f"(cr4)
        : "r"(ar1), "r"(ar2), "r"(ar3), "r"(ar4),
        "r"(br1), "r"(br2));
    }
    if(i + bi + 0 + cri < n && j + bj + (crj << 1) + 0 < m) {
      c[m * (i + bi + 0 + cri) + j + bj + (crj << 1) + 0] = cr1;
    }
    if(i + bi + 0 + cri < n && j + bj + (crj << 1) + 1 < m) {
      c[m * (i + bi + 0 + cri) + j + bj + (crj << 1) + 1] = cr2;
    }
    if(i + bi + 8 + cri < n && j + bj + (crj << 1) + 0 < m) {
      c[m * (i + bi + 8 + cri) + j + bj + (crj << 1) + 0] = cr3;
    }
    if(i + bi + 8 + cri < n && j + bj + (crj << 1) + 1) {
      c[m * (i + bi + 8 + cri) + j + bj + (crj << 1) + 1] = cr4;
    }
  }
  
  __global__ void mul_at_bt(const float *a, const float *b, float *c, size_t n, size_t m, size_t l)
  {
    size_t i = ((size_t) blockIdx.x) * MMA_TILE_WIDTH;
    size_t j = ((size_t) blockIdx.y) * MMA_TILE_WIDTH;
    size_t k;
    size_t tx = threadIdx.x >> 5;
    size_t stx = threadIdx.x & 31;
    size_t bi = (tx >> 3) << 4;
    size_t bj = (tx & 7) << 3;
    size_t ari = stx >> 2;
    size_t ark = stx & 3;
    size_t brj = ari;
    size_t brk = ark;
    size_t cri = ari;
    size_t crj = ark;
    unsigned ar1;
    unsigned ar2;
    unsigned ar3;
    unsigned ar4;
    unsigned br1;
    unsigned br2;
    float cr1 = 0.0f;
    float cr2 = 0.0f;
    float cr3 = 0.0f;
    float cr4 = 0.0f;
    unsigned zero = float_to_tf32(0.0f);
    for(k = 0; k < l; k += 8) {
      ar1 = zero;
      if(i + bi + 0 + ari < n && k + 0 + ark < l) {
        ar1 = float_to_tf32(a[n * (k + 0 + ark) + i + bi + 0 + ari]);
      }
      ar2 = zero;
      if(i + bi + 8 + ari < n && k + 0 + ark < l) {
        ar2 = float_to_tf32(a[n * (k + 0 + ark) + i + bi + 8 + ari]);
      }
      ar3 = zero;
      if(i + bi + 0 + ari < n && k + 4 + ark < l) {
        ar3 = float_to_tf32(a[n * (k + 4 + ark) + i + bi + 0 + ari]);
      }
      ar4 = zero;
      if(i + bi + 8 + ari < n && k + 4 + ark < l) {
        ar4 = float_to_tf32(a[n * (k + 4 + ark) + i + bi + 8 + ari]);
      }
      br1 = zero;
      if(j + bj + brj < m && k + 0 + brk < l) {
        br1 = float_to_tf32(b[l * (j + bj + brj) + k + 0 + brk]);
      }
      br2 = zero;
      if(j + bj + brj < m && k + 4 + brk < l) {
        br2 = float_to_tf32(b[l * (j + bj + brj) + k + 4 + brk]);
      }
      asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32\n"
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(cr1), "+f"(cr2), "+f"(cr3), "+f"(cr4)
        : "r"(ar1), "r"(ar2), "r"(ar3), "r"(ar4),
        "r"(br1), "r"(br2));
    }
    if(i + bi + 0 + cri < n && j + bj + (crj << 1) + 0 < m) {
      c[m * (i + bi + 0 + cri) + j + bj + (crj << 1) + 0] = cr1;
    }
    if(i + bi + 0 + cri < n && j + bj + (crj << 1) + 1 < m) {
      c[m * (i + bi + 0 + cri) + j + bj + (crj << 1) + 1] = cr2;
    }
    if(i + bi + 8 + cri < n && j + bj + (crj << 1) + 0 < m) {
      c[m * (i + bi + 8 + cri) + j + bj + (crj << 1) + 0] = cr3;
    }
    if(i + bi + 8 + cri < n && j + bj + (crj << 1) + 1) {
      c[m * (i + bi + 8 + cri) + j + bj + (crj << 1) + 1] = cr4;
    }
  }
  
#else

  __global__ void mul_a_b(const float *a, const float *b, float *c, size_t n, size_t m, size_t l)
  {
    __shared__ float4 as[MTHREAD_SIZE << 1][MTHREAD_SIZE];
    __shared__ float4 bs[MTHREAD_SIZE][MTHREAD_SIZE];
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 3;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 2;
    size_t k;
    size_t ti = threadIdx.y;
    size_t tj = threadIdx.x;
    size_t ik = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    size_t jk = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    float4 ar1;
    float4 ar2;
    float4 br;
    float4 cr1 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr2 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr3 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr4 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr5 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr6 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr7 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr8 = { 0.0f, 0.0f, 0.0f, 0.0f };
    for(k = 0; k < l; k += MTHREAD_SIZE) {
      size_t tk;
      size_t tjik = (tj + ik) % MTHREAD_SIZE;
      size_t tijk = (ti + jk) % MTHREAD_SIZE;
      as[ti][tjik].x = 0.0f;
      if(i + 0 < n && k + tj < l) {
        as[ti][tjik].x = a[l * (i + 0) + k + tj];
      }
      as[ti][tjik].y = 0.0f;
      if(i + 1 < n && k + tj < l) {
        as[ti][tjik].y = a[l * (i + 1) + k + tj];
      }
      as[ti][tjik].z = 0.0f;
      if(i + 2 < n && k + tj < l) {
        as[ti][tjik].z = a[l * (i + 2) + k + tj];
      }
      as[ti][tjik].w = 0.0f;
      if(i + 3 < n && k + tj < l) {
        as[ti][tjik].w = a[l * (i + 3) + k + tj];
      }
      as[ti + MTHREAD_SIZE][tjik].x = 0.0f;
      if(i + 4 < n && k + tj < l) {
        as[ti + MTHREAD_SIZE][tjik].x = a[l * (i + 4) + k + tj];
      }
      as[ti + MTHREAD_SIZE][tjik].y = 0.0f;
      if(i + 5 < n && k + tj < l) {
        as[ti + MTHREAD_SIZE][tjik].y = a[l * (i + 5) + k + tj];
      }
      as[ti + MTHREAD_SIZE][tjik].z = 0.0f;
      if(i + 6 < n && k + tj < l) {
        as[ti + MTHREAD_SIZE][tjik].z = a[l * (i + 6) + k + tj];
      }
      as[ti + MTHREAD_SIZE][tjik].w = 0.0f;
      if(i + 7 < n && k + tj < l) {
        as[ti + MTHREAD_SIZE][tjik].w = a[l * (i + 7) + k + tj];
      }
      bs[tj][tijk].x = 0.0f;
      if(j + 0 < m && k + ti < l) {
        bs[tj][tijk].x = b[m * (k + ti) + j + 0];
      }
      bs[tj][tijk].y = 0.0f;
      if(j + 1 < m && k + ti < l) {
        bs[tj][tijk].y = b[m * (k + ti) + j + 1];
      }
      bs[tj][tijk].z = 0.0f;
      if(j + 2 < m && k + ti < l) {
        bs[tj][tijk].z = b[m * (k + ti) + j + 2];
      }
      bs[tj][tijk].w = 0.0f;
      if(j + 3 < m && k + ti < l) {
        bs[tj][tijk].w = b[m * (k + ti) + j + 3];
      }
      __syncthreads();
#pragma unroll
      for(tk = 0; tk < MTHREAD_SIZE; tk++) {
        size_t tkik = (tk + ik) % MTHREAD_SIZE;
        size_t tkjk = (tk + jk) % MTHREAD_SIZE;
        ar1 = as[ti][tkik];
        ar2 = as[ti + MTHREAD_SIZE][tkik];
        br = bs[tj][tkjk];
        cr1.x += ar1.x * br.x;
        cr1.y += ar1.x * br.y;
        cr1.z += ar1.x * br.z;
        cr1.w += ar1.x * br.w;
        cr2.x += ar1.y * br.x;
        cr2.y += ar1.y * br.y;
        cr2.z += ar1.y * br.z;
        cr2.w += ar1.y * br.w;
        cr3.x += ar1.z * br.x;
        cr3.y += ar1.z * br.y;
        cr3.z += ar1.z * br.z;
        cr3.w += ar1.z * br.w;
        cr4.x += ar1.w * br.x;
        cr4.y += ar1.w * br.y;
        cr4.z += ar1.w * br.z;
        cr4.w += ar1.w * br.w;
        cr5.x += ar2.x * br.x;
        cr5.y += ar2.x * br.y;
        cr5.z += ar2.x * br.z;
        cr5.w += ar2.x * br.w;
        cr6.x += ar2.y * br.x;
        cr6.y += ar2.y * br.y;
        cr6.z += ar2.y * br.z;
        cr6.w += ar2.y * br.w;
        cr7.x += ar2.z * br.x;
        cr7.y += ar2.z * br.y;
        cr7.z += ar2.z * br.z;
        cr7.w += ar2.z * br.w;
        cr8.x += ar2.w * br.x;
        cr8.y += ar2.w * br.y;
        cr8.z += ar2.w * br.z;
        cr8.w += ar2.w * br.w;
      }
      __syncthreads();
    }
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = cr1.x;
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = cr1.y;
    }
    if(i + 0 < n && j + 2 < m) {
      c[m * (i + 0) + j + 2] = cr1.z;
    }
    if(i + 0 < n && j + 3 < m) {
      c[m * (i + 0) + j + 3] = cr1.w;
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = cr2.x;
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = cr2.y;
    }
    if(i + 1 < n && j + 2 < m) {
      c[m * (i + 1) + j + 2] = cr2.z;
    }
    if(i + 1 < n && j + 3 < m) {
      c[m * (i + 1) + j + 3] = cr2.w;
    }
    if(i + 2 < n && j + 0 < m) {
      c[m * (i + 2) + j + 0] = cr3.x;
    }
    if(i + 2 < n && j + 1 < m) {
      c[m * (i + 2) + j + 1] = cr3.y;
    }
    if(i + 2 < n && j + 2 < m) {
      c[m * (i + 2) + j + 2] = cr3.z;
    }
    if(i + 2 < n && j + 3 < m) {
      c[m * (i + 2) + j + 3] = cr3.w;
    }
    if(i + 3 < n && j + 0 < m) {
      c[m * (i + 3) + j + 0] = cr4.x;
    }
    if(i + 3 < n && j + 1 < m) {
      c[m * (i + 3) + j + 1] = cr4.y;
    }
    if(i + 3 < n && j + 2 < m) {
      c[m * (i + 3) + j + 2] = cr4.z;
    }
    if(i + 3 < n && j + 3 < m) {
      c[m * (i + 3) + j + 3] = cr4.w;
    }
    if(i + 4 < n && j + 0 < m) {
      c[m * (i + 4) + j + 0] = cr5.x;
    }
    if(i + 4 < n && j + 1 < m) {
      c[m * (i + 4) + j + 1] = cr5.y;
    }
    if(i + 4 < n && j + 2 < m) {
      c[m * (i + 4) + j + 2] = cr5.z;
    }
    if(i + 4 < n && j + 3 < m) {
      c[m * (i + 4) + j + 3] = cr5.w;
    }
    if(i + 5 < n && j + 0 < m) {
      c[m * (i + 5) + j + 0] = cr6.x;
    }
    if(i + 5 < n && j + 1 < m) {
      c[m * (i + 5) + j + 1] = cr6.y;
    }
    if(i + 5 < n && j + 2 < m) {
      c[m * (i + 5) + j + 2] = cr6.z;
    }
    if(i + 5 < n && j + 3 < m) {
      c[m * (i + 5) + j + 3] = cr6.w;
    }
    if(i + 6 < n && j + 0 < m) {
      c[m * (i + 6) + j + 0] = cr7.x;
    }
    if(i + 6 < n && j + 1 < m) {
      c[m * (i + 6) + j + 1] = cr7.y;
    }
    if(i + 6 < n && j + 2 < m) {
      c[m * (i + 6) + j + 2] = cr7.z;
    }
    if(i + 6 < n && j + 3 < m) {
      c[m * (i + 6) + j + 3] = cr7.w;
    }
    if(i + 7 < n && j + 0 < m) {
      c[m * (i + 7) + j + 0] = cr8.x;
    }
    if(i + 7 < n && j + 1 < m) {
      c[m * (i + 7) + j + 1] = cr8.y;
    }
    if(i + 7 < n && j + 2 < m) {
      c[m * (i + 7) + j + 2] = cr8.z;
    }
    if(i + 7 < n && j + 3 < m) {
      c[m * (i + 7) + j + 3] = cr8.w;
    }
  }
  
  __global__ void mul_at_b(const float *a, const float *b, float *c, size_t n, size_t m, size_t l)
  {
    __shared__ float4 as[MTHREAD_SIZE << 1][MTHREAD_SIZE];
    __shared__ float4 bs[MTHREAD_SIZE][MTHREAD_SIZE];
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 3;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 2;
    size_t k;
    size_t ti = threadIdx.x;
    size_t tj = threadIdx.y;
    size_t ik = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t jk = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    float4 ar1;
    float4 ar2;
    float4 br;
    float4 cr1 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr2 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr3 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr4 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr5 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr6 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr7 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr8 = { 0.0f, 0.0f, 0.0f, 0.0f };
    for(k = 0; k < l; k += MTHREAD_SIZE) {
      size_t tk;
      size_t tjik = (tj + ik) % MTHREAD_SIZE;
      size_t tijk = (ti + jk) % MTHREAD_SIZE;
      as[ti][tjik].x = 0.0f;
      if(i + 0 < n && k + tj < l) {
        as[ti][tjik].x = a[n * (k + tj) + i + 0];
      }
      as[ti][tjik].y = 0.0f;
      if(i + 1 < n && k + tj < l) {
        as[ti][tjik].y = a[n * (k + tj) + i + 1];
      }
      as[ti][tjik].z = 0.0f;
      if(i + 2 < n && k + tj < l) {
        as[ti][tjik].z = a[n * (k + tj) + i + 2];
      }
      as[ti][tjik].w = 0.0f;
      if(i + 3 < n && k + tj < l) {
        as[ti][tjik].w = a[n * (k + tj) + i + 3];
      }
      as[ti + MTHREAD_SIZE][tjik].x = 0.0f;
      if(i + 4 < n && k + tj < l) {
        as[ti + MTHREAD_SIZE][tjik].x = a[n * (k + tj) + i + 4];
      }
      as[ti + MTHREAD_SIZE][tjik].y = 0.0f;
      if(i + 5 < n && k + tj < l) {
        as[ti + MTHREAD_SIZE][tjik].y = a[n * (k + tj) + i + 5];
      }
      as[ti + MTHREAD_SIZE][tjik].z = 0.0f;
      if(i + 6 < n && k + tj < l) {
        as[ti + MTHREAD_SIZE][tjik].z = a[n * (k + tj) + i + 6];
      }
      as[ti + MTHREAD_SIZE][tjik].w = 0.0f;
      if(i + 7 < n && k + tj < l) {
        as[ti + MTHREAD_SIZE][tjik].w = a[n * (k + tj) + i + 7];
      }
      bs[tj][tijk].x = 0.0f;
      if(j + 0 < m && k + ti < l) {
        bs[tj][tijk].x = b[m * (k + ti) + j + 0];
      }
      bs[tj][tijk].y = 0.0f;
      if(j + 1 < m && k + ti < l) {
        bs[tj][tijk].y = b[m * (k + ti) + j + 1];
      }
      bs[tj][tijk].z = 0.0f;
      if(j + 2 < m && k + ti < l) {
        bs[tj][tijk].z = b[m * (k + ti) + j + 2];
      }
      bs[tj][tijk].w = 0.0f;
      if(j + 3 < m && k + ti < l) {
        bs[tj][tijk].w = b[m * (k + ti) + j + 3];
      }
      __syncthreads();
#pragma unroll
      for(tk = 0; tk < MTHREAD_SIZE; tk++) {
        size_t tkik = (tk + ik) % MTHREAD_SIZE;
        size_t tkjk = (tk + jk) % MTHREAD_SIZE;
        ar1 = as[ti][tkik];
        ar2 = as[ti + MTHREAD_SIZE][tkik];
        br = bs[tj][tkjk];
        cr1.x += ar1.x * br.x;
        cr1.y += ar1.x * br.y;
        cr1.z += ar1.x * br.z;
        cr1.w += ar1.x * br.w;
        cr2.x += ar1.y * br.x;
        cr2.y += ar1.y * br.y;
        cr2.z += ar1.y * br.z;
        cr2.w += ar1.y * br.w;
        cr3.x += ar1.z * br.x;
        cr3.y += ar1.z * br.y;
        cr3.z += ar1.z * br.z;
        cr3.w += ar1.z * br.w;
        cr4.x += ar1.w * br.x;
        cr4.y += ar1.w * br.y;
        cr4.z += ar1.w * br.z;
        cr4.w += ar1.w * br.w;
        cr5.x += ar2.x * br.x;
        cr5.y += ar2.x * br.y;
        cr5.z += ar2.x * br.z;
        cr5.w += ar2.x * br.w;
        cr6.x += ar2.y * br.x;
        cr6.y += ar2.y * br.y;
        cr6.z += ar2.y * br.z;
        cr6.w += ar2.y * br.w;
        cr7.x += ar2.z * br.x;
        cr7.y += ar2.z * br.y;
        cr7.z += ar2.z * br.z;
        cr7.w += ar2.z * br.w;
        cr8.x += ar2.w * br.x;
        cr8.y += ar2.w * br.y;
        cr8.z += ar2.w * br.z;
        cr8.w += ar2.w * br.w;
      }
      __syncthreads();
    }
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = cr1.x;
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = cr1.y;
    }
    if(i + 0 < n && j + 2 < m) {
      c[m * (i + 0) + j + 2] = cr1.z;
    }
    if(i + 0 < n && j + 3 < m) {
      c[m * (i + 0) + j + 3] = cr1.w;
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = cr2.x;
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = cr2.y;
    }
    if(i + 1 < n && j + 2 < m) {
      c[m * (i + 1) + j + 2] = cr2.z;
    }
    if(i + 1 < n && j + 3 < m) {
      c[m * (i + 1) + j + 3] = cr2.w;
    }
    if(i + 2 < n && j + 0 < m) {
      c[m * (i + 2) + j + 0] = cr3.x;
    }
    if(i + 2 < n && j + 1 < m) {
      c[m * (i + 2) + j + 1] = cr3.y;
    }
    if(i + 2 < n && j + 2 < m) {
      c[m * (i + 2) + j + 2] = cr3.z;
    }
    if(i + 2 < n && j + 3 < m) {
      c[m * (i + 2) + j + 3] = cr3.w;
    }
    if(i + 3 < n && j + 0 < m) {
      c[m * (i + 3) + j + 0] = cr4.x;
    }
    if(i + 3 < n && j + 1 < m) {
      c[m * (i + 3) + j + 1] = cr4.y;
    }
    if(i + 3 < n && j + 2 < m) {
      c[m * (i + 3) + j + 2] = cr4.z;
    }
    if(i + 3 < n && j + 3 < m) {
      c[m * (i + 3) + j + 3] = cr4.w;
    }
    if(i + 4 < n && j + 0 < m) {
      c[m * (i + 4) + j + 0] = cr5.x;
    }
    if(i + 4 < n && j + 1 < m) {
      c[m * (i + 4) + j + 1] = cr5.y;
    }
    if(i + 4 < n && j + 2 < m) {
      c[m * (i + 4) + j + 2] = cr5.z;
    }
    if(i + 4 < n && j + 3 < m) {
      c[m * (i + 4) + j + 3] = cr5.w;
    }
    if(i + 5 < n && j + 0 < m) {
      c[m * (i + 5) + j + 0] = cr6.x;
    }
    if(i + 5 < n && j + 1 < m) {
      c[m * (i + 5) + j + 1] = cr6.y;
    }
    if(i + 5 < n && j + 2 < m) {
      c[m * (i + 5) + j + 2] = cr6.z;
    }
    if(i + 5 < n && j + 3 < m) {
      c[m * (i + 5) + j + 3] = cr6.w;
    }
    if(i + 6 < n && j + 0 < m) {
      c[m * (i + 6) + j + 0] = cr7.x;
    }
    if(i + 6 < n && j + 1 < m) {
      c[m * (i + 6) + j + 1] = cr7.y;
    }
    if(i + 6 < n && j + 2 < m) {
      c[m * (i + 6) + j + 2] = cr7.z;
    }
    if(i + 6 < n && j + 3 < m) {
      c[m * (i + 6) + j + 3] = cr7.w;
    }
    if(i + 7 < n && j + 0 < m) {
      c[m * (i + 7) + j + 0] = cr8.x;
    }
    if(i + 7 < n && j + 1 < m) {
      c[m * (i + 7) + j + 1] = cr8.y;
    }
    if(i + 7 < n && j + 2 < m) {
      c[m * (i + 7) + j + 2] = cr8.z;
    }
    if(i + 7 < n && j + 3 < m) {
      c[m * (i + 7) + j + 3] = cr8.w;
    }
  }

  __global__ void mul_a_bt(const float *a, const float *b, float *c, size_t n, size_t m, size_t l)
  {
    __shared__ float4 as[MTHREAD_SIZE << 1][MTHREAD_SIZE];
    __shared__ float4 bs[MTHREAD_SIZE][MTHREAD_SIZE];
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 3;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 2;
    size_t k;
    size_t ti = threadIdx.y;
    size_t tj = threadIdx.x;
    size_t ik = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    size_t jk = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    float4 ar1;
    float4 ar2;
    float4 br;
    float4 cr1 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr2 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr3 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr4 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr5 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr6 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr7 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr8 = { 0.0f, 0.0f, 0.0f, 0.0f };
    for(k = 0; k < l; k += MTHREAD_SIZE) {
      size_t tk;
      size_t tjik = (tj + ik) % MTHREAD_SIZE;
      size_t tijk = (ti + jk) % MTHREAD_SIZE;
      as[ti][tjik].x = 0.0f;
      if(i + 0 < n && k + tj < l) {
        as[ti][tjik].x = a[l * (i + 0) + k + tj];
      }
      as[ti][tjik].y = 0.0f;
      if(i + 1 < n && k + tj < l) {
        as[ti][tjik].y = a[l * (i + 1) + k + tj];
      }
      as[ti][tjik].z = 0.0f;
      if(i + 2 < n && k + tj < l) {
        as[ti][tjik].z = a[l * (i + 2) + k + tj];
      }
      as[ti][tjik].w = 0.0f;
      if(i + 3 < n && k + tj < l) {
        as[ti][tjik].w = a[l * (i + 3) + k + tj];
      }
      as[ti + MTHREAD_SIZE][tjik].x = 0.0f;
      if(i + 4 < n && k + tj < l) {
        as[ti + MTHREAD_SIZE][tjik].x = a[l * (i + 4) + k + tj];
      }
      as[ti + MTHREAD_SIZE][tjik].y = 0.0f;
      if(i + 5 < n && k + tj < l) {
        as[ti + MTHREAD_SIZE][tjik].y = a[l * (i + 5) + k + tj];
      }
      as[ti + MTHREAD_SIZE][tjik].z = 0.0f;
      if(i + 6 < n && k + tj < l) {
        as[ti + MTHREAD_SIZE][tjik].z = a[l * (i + 6) + k + tj];
      }
      as[ti + MTHREAD_SIZE][tjik].w = 0.0f;
      if(i + 7 < n && k + tj < l) {
        as[ti + MTHREAD_SIZE][tjik].w = a[l * (i + 7) + k + tj];
      }
      bs[tj][tijk].x = 0.0f;
      if(j + 0 < m && k + ti < l) {
        bs[tj][tijk].x = b[l * (j + 0) + k + ti];
      }
      bs[tj][tijk].y = 0.0f;
      if(j + 1 < m && k + ti < l) {
        bs[tj][tijk].y = b[l * (j + 1) + k + ti];
      }
      bs[tj][tijk].z = 0.0f;
      if(j + 2 < m && k + ti < l) {
        bs[tj][tijk].z = b[l * (j + 2) + k + ti];
      }
      bs[tj][tijk].w = 0.0f;
      if(j + 3 < m && k + ti < l) {
        bs[tj][tijk].w = b[l * (j + 3) + k + ti];
      }
      __syncthreads();
#pragma unroll
      for(tk = 0; tk < MTHREAD_SIZE; tk++) {
        size_t tkik = (tk + ik) % MTHREAD_SIZE;
        size_t tkjk = (tk + jk) % MTHREAD_SIZE;
        ar1 = as[ti][tkik];
        ar2 = as[ti + MTHREAD_SIZE][tkik];
        br = bs[tj][tkjk];
        cr1.x += ar1.x * br.x;
        cr1.y += ar1.x * br.y;
        cr1.z += ar1.x * br.z;
        cr1.w += ar1.x * br.w;
        cr2.x += ar1.y * br.x;
        cr2.y += ar1.y * br.y;
        cr2.z += ar1.y * br.z;
        cr2.w += ar1.y * br.w;
        cr3.x += ar1.z * br.x;
        cr3.y += ar1.z * br.y;
        cr3.z += ar1.z * br.z;
        cr3.w += ar1.z * br.w;
        cr4.x += ar1.w * br.x;
        cr4.y += ar1.w * br.y;
        cr4.z += ar1.w * br.z;
        cr4.w += ar1.w * br.w;
        cr5.x += ar2.x * br.x;
        cr5.y += ar2.x * br.y;
        cr5.z += ar2.x * br.z;
        cr5.w += ar2.x * br.w;
        cr6.x += ar2.y * br.x;
        cr6.y += ar2.y * br.y;
        cr6.z += ar2.y * br.z;
        cr6.w += ar2.y * br.w;
        cr7.x += ar2.z * br.x;
        cr7.y += ar2.z * br.y;
        cr7.z += ar2.z * br.z;
        cr7.w += ar2.z * br.w;
        cr8.x += ar2.w * br.x;
        cr8.y += ar2.w * br.y;
        cr8.z += ar2.w * br.z;
        cr8.w += ar2.w * br.w;
      }
      __syncthreads();
    }
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = cr1.x;
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = cr1.y;
    }
    if(i + 0 < n && j + 2 < m) {
      c[m * (i + 0) + j + 2] = cr1.z;
    }
    if(i + 0 < n && j + 3 < m) {
      c[m * (i + 0) + j + 3] = cr1.w;
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = cr2.x;
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = cr2.y;
    }
    if(i + 1 < n && j + 2 < m) {
      c[m * (i + 1) + j + 2] = cr2.z;
    }
    if(i + 1 < n && j + 3 < m) {
      c[m * (i + 1) + j + 3] = cr2.w;
    }
    if(i + 2 < n && j + 0 < m) {
      c[m * (i + 2) + j + 0] = cr3.x;
    }
    if(i + 2 < n && j + 1 < m) {
      c[m * (i + 2) + j + 1] = cr3.y;
    }
    if(i + 2 < n && j + 2 < m) {
      c[m * (i + 2) + j + 2] = cr3.z;
    }
    if(i + 2 < n && j + 3 < m) {
      c[m * (i + 2) + j + 3] = cr3.w;
    }
    if(i + 3 < n && j + 0 < m) {
      c[m * (i + 3) + j + 0] = cr4.x;
    }
    if(i + 3 < n && j + 1 < m) {
      c[m * (i + 3) + j + 1] = cr4.y;
    }
    if(i + 3 < n && j + 2 < m) {
      c[m * (i + 3) + j + 2] = cr4.z;
    }
    if(i + 3 < n && j + 3 < m) {
      c[m * (i + 3) + j + 3] = cr4.w;
    }
    if(i + 4 < n && j + 0 < m) {
      c[m * (i + 4) + j + 0] = cr5.x;
    }
    if(i + 4 < n && j + 1 < m) {
      c[m * (i + 4) + j + 1] = cr5.y;
    }
    if(i + 4 < n && j + 2 < m) {
      c[m * (i + 4) + j + 2] = cr5.z;
    }
    if(i + 4 < n && j + 3 < m) {
      c[m * (i + 4) + j + 3] = cr5.w;
    }
    if(i + 5 < n && j + 0 < m) {
      c[m * (i + 5) + j + 0] = cr6.x;
    }
    if(i + 5 < n && j + 1 < m) {
      c[m * (i + 5) + j + 1] = cr6.y;
    }
    if(i + 5 < n && j + 2 < m) {
      c[m * (i + 5) + j + 2] = cr6.z;
    }
    if(i + 5 < n && j + 3 < m) {
      c[m * (i + 5) + j + 3] = cr6.w;
    }
    if(i + 6 < n && j + 0 < m) {
      c[m * (i + 6) + j + 0] = cr7.x;
    }
    if(i + 6 < n && j + 1 < m) {
      c[m * (i + 6) + j + 1] = cr7.y;
    }
    if(i + 6 < n && j + 2 < m) {
      c[m * (i + 6) + j + 2] = cr7.z;
    }
    if(i + 6 < n && j + 3 < m) {
      c[m * (i + 6) + j + 3] = cr7.w;
    }
    if(i + 7 < n && j + 0 < m) {
      c[m * (i + 7) + j + 0] = cr8.x;
    }
    if(i + 7 < n && j + 1 < m) {
      c[m * (i + 7) + j + 1] = cr8.y;
    }
    if(i + 7 < n && j + 2 < m) {
      c[m * (i + 7) + j + 2] = cr8.z;
    }
    if(i + 7 < n && j + 3 < m) {
      c[m * (i + 7) + j + 3] = cr8.w;
    }
  }

  __global__ void mul_at_bt(const float *a, const float *b, float *c, size_t n, size_t m, size_t l)
  {
    __shared__ float4 as[MTHREAD_SIZE << 1][MTHREAD_SIZE];
    __shared__ float4 bs[MTHREAD_SIZE][MTHREAD_SIZE];
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 3;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 2;
    size_t k;
    size_t ti = threadIdx.x;
    size_t tj = threadIdx.y;
    size_t ik = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t jk = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    float4 ar1;
    float4 ar2;
    float4 br;
    float4 cr1 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr2 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr3 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr4 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr5 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr6 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr7 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 cr8 = { 0.0f, 0.0f, 0.0f, 0.0f };
    for(k = 0; k < l; k += MTHREAD_SIZE) {
      size_t tk;
      size_t tjik = (tj + ik) % MTHREAD_SIZE;
      size_t tijk = (ti + jk) % MTHREAD_SIZE;
      as[ti][tjik].x = 0.0f;
      if(i + 0 < n && k + tj < l) {
        as[ti][tjik].x = a[n * (k + tj) + i + 0];
      }
      as[ti][tjik].y = 0.0f;
      if(i + 1 < n && k + tj < l) {
        as[ti][tjik].y = a[n * (k + tj) + i + 1];
      }
      as[ti][tjik].z = 0.0f;
      if(i + 2 < n && k + tj < l) {
        as[ti][tjik].z = a[n * (k + tj) + i + 2];
      }
      as[ti][tjik].w = 0.0f;
      if(i + 3 < n && k + tj < l) {
        as[ti][tjik].w = a[n * (k + tj) + i + 3];
      }
      as[ti + MTHREAD_SIZE][tjik].x = 0.0f;
      if(i + 4 < n && k + tj < l) {
        as[ti + MTHREAD_SIZE][tjik].x = a[n * (k + tj) + i + 4];
      }
      as[ti + MTHREAD_SIZE][tjik].y = 0.0f;
      if(i + 5 < n && k + tj < l) {
        as[ti + MTHREAD_SIZE][tjik].y = a[n * (k + tj) + i + 5];
      }
      as[ti + MTHREAD_SIZE][tjik].z = 0.0f;
      if(i + 6 < n && k + tj < l) {
        as[ti + MTHREAD_SIZE][tjik].z = a[n * (k + tj) + i + 6];
      }
      as[ti + MTHREAD_SIZE][tjik].w = 0.0f;
      if(i + 7 < n && k + tj < l) {
        as[ti + MTHREAD_SIZE][tjik].w = a[n * (k + tj) + i + 7];
      }
      bs[tj][tijk].x = 0.0f;
      if(j + 0 < m && k + ti < l) {
        bs[tj][tijk].x = b[l * (j + 0) + k + ti];
      }
      bs[tj][tijk].y = 0.0f;
      if(j + 1 < m && k + ti < l) {
        bs[tj][tijk].y = b[l * (j + 1) + k + ti];
      }
      bs[tj][tijk].z = 0.0f;
      if(j + 2 < m && k + ti < l) {
        bs[tj][tijk].z = b[l * (j + 2) + k + ti];
      }
      bs[tj][tijk].w = 0.0f;
      if(j + 3 < m && k + ti < l) {
        bs[tj][tijk].w = b[l * (j + 3) + k + ti];
      }
      __syncthreads();
#pragma unroll
      for(tk = 0; tk < MTHREAD_SIZE; tk++) {
        size_t tkik = (tk + ik) % MTHREAD_SIZE;
        size_t tkjk = (tk + jk) % MTHREAD_SIZE;
        ar1 = as[ti][tkik];
        ar2 = as[ti + MTHREAD_SIZE][tkik];
        br = bs[tj][tkjk];
        cr1.x += ar1.x * br.x;
        cr1.y += ar1.x * br.y;
        cr1.z += ar1.x * br.z;
        cr1.w += ar1.x * br.w;
        cr2.x += ar1.y * br.x;
        cr2.y += ar1.y * br.y;
        cr2.z += ar1.y * br.z;
        cr2.w += ar1.y * br.w;
        cr3.x += ar1.z * br.x;
        cr3.y += ar1.z * br.y;
        cr3.z += ar1.z * br.z;
        cr3.w += ar1.z * br.w;
        cr4.x += ar1.w * br.x;
        cr4.y += ar1.w * br.y;
        cr4.z += ar1.w * br.z;
        cr4.w += ar1.w * br.w;
        cr5.x += ar2.x * br.x;
        cr5.y += ar2.x * br.y;
        cr5.z += ar2.x * br.z;
        cr5.w += ar2.x * br.w;
        cr6.x += ar2.y * br.x;
        cr6.y += ar2.y * br.y;
        cr6.z += ar2.y * br.z;
        cr6.w += ar2.y * br.w;
        cr7.x += ar2.z * br.x;
        cr7.y += ar2.z * br.y;
        cr7.z += ar2.z * br.z;
        cr7.w += ar2.z * br.w;
        cr8.x += ar2.w * br.x;
        cr8.y += ar2.w * br.y;
        cr8.z += ar2.w * br.z;
        cr8.w += ar2.w * br.w;
      }
      __syncthreads();
    }
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = cr1.x;
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = cr1.y;
    }
    if(i + 0 < n && j + 2 < m) {
      c[m * (i + 0) + j + 2] = cr1.z;
    }
    if(i + 0 < n && j + 3 < m) {
      c[m * (i + 0) + j + 3] = cr1.w;
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = cr2.x;
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = cr2.y;
    }
    if(i + 1 < n && j + 2 < m) {
      c[m * (i + 1) + j + 2] = cr2.z;
    }
    if(i + 1 < n && j + 3 < m) {
      c[m * (i + 1) + j + 3] = cr2.w;
    }
    if(i + 2 < n && j + 0 < m) {
      c[m * (i + 2) + j + 0] = cr3.x;
    }
    if(i + 2 < n && j + 1 < m) {
      c[m * (i + 2) + j + 1] = cr3.y;
    }
    if(i + 2 < n && j + 2 < m) {
      c[m * (i + 2) + j + 2] = cr3.z;
    }
    if(i + 2 < n && j + 3 < m) {
      c[m * (i + 2) + j + 3] = cr3.w;
    }
    if(i + 3 < n && j + 0 < m) {
      c[m * (i + 3) + j + 0] = cr4.x;
    }
    if(i + 3 < n && j + 1 < m) {
      c[m * (i + 3) + j + 1] = cr4.y;
    }
    if(i + 3 < n && j + 2 < m) {
      c[m * (i + 3) + j + 2] = cr4.z;
    }
    if(i + 3 < n && j + 3 < m) {
      c[m * (i + 3) + j + 3] = cr4.w;
    }
    if(i + 4 < n && j + 0 < m) {
      c[m * (i + 4) + j + 0] = cr5.x;
    }
    if(i + 4 < n && j + 1 < m) {
      c[m * (i + 4) + j + 1] = cr5.y;
    }
    if(i + 4 < n && j + 2 < m) {
      c[m * (i + 4) + j + 2] = cr5.z;
    }
    if(i + 4 < n && j + 3 < m) {
      c[m * (i + 4) + j + 3] = cr5.w;
    }
    if(i + 5 < n && j + 0 < m) {
      c[m * (i + 5) + j + 0] = cr6.x;
    }
    if(i + 5 < n && j + 1 < m) {
      c[m * (i + 5) + j + 1] = cr6.y;
    }
    if(i + 5 < n && j + 2 < m) {
      c[m * (i + 5) + j + 2] = cr6.z;
    }
    if(i + 5 < n && j + 3 < m) {
      c[m * (i + 5) + j + 3] = cr6.w;
    }
    if(i + 6 < n && j + 0 < m) {
      c[m * (i + 6) + j + 0] = cr7.x;
    }
    if(i + 6 < n && j + 1 < m) {
      c[m * (i + 6) + j + 1] = cr7.y;
    }
    if(i + 6 < n && j + 2 < m) {
      c[m * (i + 6) + j + 2] = cr7.z;
    }
    if(i + 6 < n && j + 3 < m) {
      c[m * (i + 6) + j + 3] = cr7.w;
    }
    if(i + 7 < n && j + 0 < m) {
      c[m * (i + 7) + j + 0] = cr8.x;
    }
    if(i + 7 < n && j + 1 < m) {
      c[m * (i + 7) + j + 1] = cr8.y;
    }
    if(i + 7 < n && j + 2 < m) {
      c[m * (i + 7) + j + 2] = cr8.z;
    }
    if(i + 7 < n && j + 3 < m) {
      c[m * (i + 7) + j + 3] = cr8.w;
    }
  }

#endif

  __global__ void mul_a_b_for_elems(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[m * (i + 0) + j + 0] * b[m * (i + 0) + j + 0];
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[m * (i + 0) + j + 1] * b[m * (i + 0) + j + 1];
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[m * (i + 1) + j + 0] * b[m * (i + 1) + j + 0];
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[m * (i + 1) + j + 1] * b[m * (i + 1) + j + 1];
    }
  }

  __global__ void mul_at_b_for_elems(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[n * (j + 0) + i + 0] * b[m * (i + 0) + j + 0];
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[n * (j + 1) + i + 0] * b[m * (i + 0) + j + 1];
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[n * (j + 0) + i + 1] * b[m * (i + 1) + j + 0];
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[n * (j + 1) + i + 1] * b[m * (i + 1) + j + 1];
    }
  }

  __global__ void mul_a_bt_for_elems(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[m * (i + 0) + j + 0] * b[n * (j + 0) + i + 0];
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[m * (i + 0) + j + 1] * b[n * (j + 1) + i + 0];
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[m * (i + 1) + j + 0] * b[n * (j + 0) + i + 1];
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[m * (i + 1) + j + 1] * b[n * (j + 1) + i + 1];
    }
  }

  __global__ void mul_at_bt_for_elems(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[n * (j + 0) + i + 0] * b[n * (j + 0) + i + 0];
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[n * (j + 1) + i + 0] * b[n * (j + 1) + i + 0];
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[n * (j + 0) + i + 1] * b[n * (j + 0) + i + 1];
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[n * (j + 1) + i + 1] * b[n * (j + 1) + i + 1];
    }
  }

  __global__ void div_a_b_for_elems(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[m * (i + 0) + j + 0] / b[m * (i + 0) + j + 0];
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[m * (i + 0) + j + 1] / b[m * (i + 0) + j + 1];
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[m * (i + 1) + j + 0] / b[m * (i + 1) + j + 0];
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[m * (i + 1) + j + 1] / b[m * (i + 1) + j + 1];
    }
  }

  __global__ void div_at_b_for_elems(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[n * (j + 0) + i + 0] / b[m * (i + 0) + j + 0];
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[n * (j + 1) + i + 0] / b[m * (i + 0) + j + 1];
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[n * (j + 0) + i + 1] / b[m * (i + 1) + j + 0];
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[n * (j + 1) + i + 1] / b[m * (i + 1) + j + 1];
    }
  }

  __global__ void div_a_bt_for_elems(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[m * (i + 0) + j + 0] / b[n * (j + 0) + i + 0];
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[m * (i + 0) + j + 1] / b[n * (j + 1) + i + 0];
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[m * (i + 1) + j + 0] / b[n * (j + 0) + i + 1];
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[m * (i + 1) + j + 1] / b[n * (j + 1) + i + 1];
    }
  }

  __global__ void div_at_bt_for_elems(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[n * (j + 0) + i + 0] / b[n * (j + 0) + i + 0];
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[n * (j + 1) + i + 0] / b[n * (j + 1) + i + 0];
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[n * (j + 0) + i + 1] / b[n * (j + 0) + i + 1];
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[n * (j + 1) + i + 1] / b[n * (j + 1) + i + 1];
    }
  }

  __global__ void add_a_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[m * (i + 0) + j + 0] + b;
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[m * (i + 0) + j + 1] + b;
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[m * (i + 1) + j + 0] + b;
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[m * (i + 1) + j + 1] + b;
    }
  }

  __global__ void add_at_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[n * (j + 0) + i + 0] + b;
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[n * (j + 1) + i + 0] + b;
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[n * (j + 0) + i + 1] + b;
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[n * (j + 1) + i + 1] + b;
    }
  }

  __global__ void sub_a_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[m * (i + 0) + j + 0] - b;
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[m * (i + 0) + j + 1] - b;
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[m * (i + 1) + j + 0] - b;
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[m * (i + 1) + j + 1] - b;
    }
  }

  __global__ void sub_at_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[n * (j + 0) + i + 0] - b;
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[n * (j + 1) + i + 0] - b;
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[n * (j + 0) + i + 1] - b;
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[n * (j + 1) + i + 1] - b;
    }
  }

  __global__ void rsub_a_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = b - a[m * (i + 0) + j + 0];
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = b - a[m * (i + 0) + j + 1];
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = b - a[m * (i + 1) + j + 0];
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = b - a[m * (i + 1) + j + 1];
    }
  }

  __global__ void rsub_at_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = b - a[n * (j + 0) + i + 0];
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = b - a[n * (j + 1) + i + 0];
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = b - a[n * (j + 0) + i + 1];
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = b - a[n * (j + 1) + i + 1];
    }
  }

  __global__ void mul_a_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[m * (i + 0) + j + 0] * b;
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[m * (i + 0) + j + 1] * b;
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[m * (i + 1) + j + 0] * b;
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[m * (i + 1) + j + 1] * b;
    }
  }

  __global__ void mul_at_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[n * (j + 0) + i + 0] * b;
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[n * (j + 1) + i + 0] * b;
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[n * (j + 0) + i + 1] * b;
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[n * (j + 1) + i + 1] * b;
    }
  }

  __global__ void div_a_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[m * (i + 0) + j + 0] / b;
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[m * (i + 0) + j + 1] / b;
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[m * (i + 1) + j + 0] / b;
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[m * (i + 1) + j + 1] / b;
    }
  }

  __global__ void div_at_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = a[n * (j + 0) + i + 0] / b;
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = a[n * (j + 1) + i + 0] / b;
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = a[n * (j + 0) + i + 1] / b;
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = a[n * (j + 1) + i + 1] / b;
    }
  }

  __global__ void rdiv_a_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = b / a[m * (i + 0) + j + 0];
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = b / a[m * (i + 0) + j + 1];
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = b / a[m * (i + 1) + j + 0];
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = b / a[m * (i + 1) + j + 1];
    }
  }

  __global__ void rdiv_at_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = b / a[n * (j + 0) + i + 0];
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1]  = b / a[n * (j + 1) + i + 0];
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = b / a[n * (j + 0) + i + 1];
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = b / a[n * (j + 1) + i + 1];
    }
  }

  __global__ void sigmoid_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = 1.0f / (1.0f + expf(-a[m * (i + 0) + j + 0]));
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = 1.0f / (1.0f + expf(-a[m * (i + 0) + j + 1]));
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = 1.0f / (1.0f + expf(-a[m * (i + 1) + j + 0]));
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = 1.0f / (1.0f + expf(-a[m * (i + 1) + j + 1]));
    }
  }

  __global__ void sigmoid_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = 1.0f / (1.0f + expf(-a[n * (j + 0) + i + 0]));
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = 1.0f / (1.0f + expf(-a[n * (j + 1) + i + 0]));
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = 1.0f / (1.0f + expf(-a[n * (j + 0) + i + 1]));
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = 1.0f / (1.0f + expf(-a[n * (j + 1) + i + 1]));
    }
  }

  __global__ void tanh_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = tanhf(a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = tanhf(a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = tanhf(a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = tanhf(a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void tanh_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = tanhf(a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = tanhf(a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = tanhf(a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = tanhf(a[n * (j + 1) + i + 1]);
    }
  }

  __global__ void swish_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = a[m * (i + 0) + j + 0] / (1.0f + expf(-a[m * (i + 0) + j + 0]));
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = a[m * (i + 0) + j + 1] / (1.0f + expf(-a[m * (i + 0) + j + 1]));
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = a[m * (i + 1) + j + 0] / (1.0f + expf(-a[m * (i + 1) + j + 0]));
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = a[m * (i + 1) + j + 1] / (1.0f + expf(-a[m * (i + 1) + j + 1]));
    }
  }

  __global__ void swish_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = a[n * (j + 0) + i + 0] / (1.0f + expf(-a[n * (j + 0) + i + 0]));
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = a[n * (j + 1) + i + 0] / (1.0f + expf(-a[n * (j + 1) + i + 0]));
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = a[n * (j + 0) + i + 1] / (1.0f + expf(-a[n * (j + 0) + i + 1]));
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = a[n * (j + 1) + i + 1] / (1.0f + expf(-a[n * (j + 1) + i + 1]));
    }
  }

  __global__ void softmax_a(const float *a, float *b, size_t n, size_t m)
  {
    __shared__ float4 es[THREAD_SIZE];
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t k;
    size_t thread_width = blockDim.x;
    size_t thread_height = blockDim.y;
    size_t tile_height = thread_height << 1;
    size_t ti = threadIdx.y;
    size_t tj = threadIdx.x;
    size_t bi = ti << 1;
    float2 sum = { 0.0f, 0.0f };
    for(k = 0; k < n; k += tile_height) {
      size_t tk;
      es[thread_width * ti + tj].x = 0.0f;
      if(j + 0 < m && k + bi + 0 < n) {
        es[thread_width * ti + tj].x = expf(a[m * (k + bi + 0) + j + 0]);
      }
      es[thread_width * ti + tj].y = 0.0f;
      if(j + 1 < m && k + bi + 0 < n) {
        es[thread_width * ti + tj].y = expf(a[m * (k + bi + 0) + j + 1]);
      }
      es[thread_width * ti + tj + 0].z = 0.0f;
      if(j + 0 < m && k + bi + 1 < n) {
        es[thread_width * ti + tj].z = expf(a[m * (k + bi + 1) + j + 0]);
      }
      es[thread_width * ti + tj].w = 0.0f;
      if(j + 1 < m && k + bi + 1 < n) {
        es[thread_width * ti + tj].w = expf(a[m * (k + bi + 1) + j + 1]);
      }
      __syncthreads();
#pragma unroll 32
      for(tk = 0; tk < thread_height; tk++) {
        float4 e = es[thread_width * tk + tj];
        sum.x += e.x;
        sum.y += e.y;
        sum.x += e.z;
        sum.y += e.w;
      }
      __syncthreads();
    }
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = expf(a[m * (i + 0) + j + 0]) / sum.x;
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = expf(a[m * (i + 0) + j + 1]) / sum.y;
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = expf(a[m * (i + 1) + j + 0]) / sum.x;
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = expf(a[m * (i + 1) + j + 1]) / sum.y;
    }
  }

  __global__ void softmax_at(const float *a, float *b, size_t n, size_t m)
  {
    __shared__ float4 es[THREAD_SIZE];
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t k;
    size_t thread_width = blockDim.y;
    size_t thread_height = blockDim.x;
    size_t tile_height = thread_height << 1;
    size_t ti = threadIdx.x;
    size_t tj = threadIdx.y;
    size_t bi = ti << 1;
    float2 sum = { 0.0f, 0.0f };
    for(k = 0; k < n; k += tile_height) {
      size_t tk;
      es[thread_width * ti + tj].x = 0.0f;
      if(j + 0 < m && k + bi + 0 < n) {
        es[thread_width * ti + tj].x = expf(a[n * (j + 0) + k + bi + 0]);
      }
      es[thread_width * ti + tj].y = 0.0f;
      if(j + 1 < m && k + bi + 0 < n) {
        es[thread_width * ti + tj].y = expf(a[n * (j + 1) + k + bi + 0]);
      }
      es[thread_width * ti + tj].z = 0.0f;
      if(j + 0 < m && k + bi + 1 < n) {
        es[thread_width * ti + tj].z = expf(a[n * (j + 0) + k + bi + 1]);
      }
      es[thread_width * ti + tj].w = 0.0f;
      if(j + 1 < m && k + bi + 1 < n) {
        es[thread_width * ti + tj].w = expf(a[n * (j + 1) + k + bi + 1]);
      }
      __syncthreads();
#pragma unroll 32
      for(tk = 0; tk < thread_height; tk++) {
        float4 e = es[thread_width * tk + tj];
        sum.x += e.x;
        sum.y += e.y;
        sum.x += e.z;
        sum.y += e.w;
      }
      __syncthreads();
    }
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = expf(a[n * (j + 0) + i + 0]) / sum.x;
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = expf(a[n * (j + 1) + i + 0]) / sum.y;
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = expf(a[n * (j + 0) + i + 1]) / sum.x;
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = expf(a[n * (j + 1) + i + 1]) / sum.y;
    }
  }

  __global__ void sqrt_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = sqrtf(a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = sqrtf(a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = sqrtf(a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = sqrtf(a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void sqrt_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = sqrtf(a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = sqrtf(a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = sqrtf(a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = sqrtf(a[n * (j + 1) + i + 1]);
    }
  }  
  
  __global__ void repeat_col_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = a[i + 0];
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = a[i + 0];
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = a[i + 1];
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = a[i + 1];
    }
  }

  __global__ void repeat_row_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = a[j + 0];
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = a[j + 1];
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = a[j + 0];
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = a[j + 1];
    }
  }
  
  __global__ void abs_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = fabsf(a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = fabsf(a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = fabsf(a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = fabsf(a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void abs_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = fabsf(a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = fabsf(a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = fabsf(a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = fabsf(a[n * (j + 1) + i + 1]);
    }
  }

  __global__ void pow_a_b(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = powf(a[m * (i + 0) + j + 0], b[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = powf(a[m * (i + 0) + j + 1], b[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = powf(a[m * (i + 1) + j + 0], b[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = powf(a[m * (i + 1) + j + 1], b[m * (i + 1) + j + 1]);
    }
  }

  __global__ void pow_at_b(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = powf(a[n * (j + 0) + i + 0], b[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = powf(a[n * (j + 1) + i + 0], b[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = powf(a[n * (j + 0) + i + 1], b[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = powf(a[n * (j + 1) + i + 1], b[m * (i + 1) + j + 1]);
    }
  }

  __global__ void pow_a_bt(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = powf(a[m * (i + 0) + j + 0], b[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = powf(a[m * (i + 0) + j + 1], b[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = powf(a[m * (i + 1) + j + 0], b[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = powf(a[m * (i + 1) + j + 1], b[n * (j + 1) + i + 1]);
    }
  }

  __global__ void pow_at_bt(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = powf(a[n * (j + 0) + i + 0], b[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = powf(a[n * (j + 1) + i + 0], b[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = powf(a[n * (j + 0) + i + 1], b[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = powf(a[n * (j + 1) + i + 1], b[n * (j + 1) + i + 1]);
    }
  }

  __global__ void pow_a_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = powf(a[m * (i + 0) + j + 0], b);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = powf(a[m * (i + 0) + j + 1], b);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = powf(a[m * (i + 1) + j + 0], b);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = powf(a[m * (i + 1) + j + 1], b);
    }
  }

  __global__ void pow_at_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = powf(a[n * (j + 0) + i + 0], b);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = powf(a[n * (j + 1) + i + 0], b);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = powf(a[n * (j + 0) + i + 1], b);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = powf(a[n * (j + 1) + i + 1], b);
    }
  }

  __global__ void rpow_a_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = powf(b, a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = powf(b, a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = powf(b, a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = powf(b, a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void rpow_at_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = powf(b, a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = powf(b, a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = powf(b, a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = powf(b, a[n * (j + 1) + i + 1]);
    }
  }

  __global__ void exp_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = expf(a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = expf(a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = expf(a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = expf(a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void exp_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = expf(a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = expf(a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = expf(a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = expf(a[n * (j + 1) + i + 1]);
    }
  }

  __global__ void ln_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = logf(a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = logf(a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = logf(a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = logf(a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void ln_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = logf(a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = logf(a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = logf(a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = logf(a[n * (j + 1) + i + 1]);
    }
  }

  __global__ void log2_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = log2f(a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = log2f(a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = log2f(a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = log2f(a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void log2_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = log2f(a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = log2f(a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = log2f(a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = log2f(a[n * (j + 1) + i + 1]);
    }
  }

  __global__ void log10_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = log10f(a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = log10f(a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = log10f(a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = log10f(a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void log10_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = log10f(a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = log10f(a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = log10f(a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = log10f(a[n * (j + 1) + i + 1]);
    }
  }

  __global__ void sin_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = sinf(a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = sinf(a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = sinf(a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = sinf(a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void sin_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = sinf(a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = sinf(a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = sinf(a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = sinf(a[n * (j + 1) + i + 1]);
    }
  }

  __global__ void cos_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = cosf(a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = cosf(a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = cosf(a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = cosf(a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void cos_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = cosf(a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = cosf(a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = cosf(a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = cosf(a[n * (j + 1) + i + 1]);
    }
  }

  __global__ void tan_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = tanf(a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = tanf(a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = tanf(a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = tanf(a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void tan_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = tanf(a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = tanf(a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = tanf(a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = tanf(a[n * (j + 1) + i + 1]);
    }
  }

  __global__ void asin_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = asinf(a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = asinf(a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = asinf(a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = asinf(a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void asin_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = asinf(a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = asinf(a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = asinf(a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = asinf(a[n * (j + 1) + i + 1]);
    }
  }

  __global__ void acos_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = acosf(a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = acosf(a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = acosf(a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = acosf(a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void acos_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = acosf(a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = acosf(a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = acosf(a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = acosf(a[n * (j + 1) + i + 1]);
    }
  }

  __global__ void atan_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = atanf(a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = atanf(a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = atanf(a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = atanf(a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void atan_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = atanf(a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = atanf(a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = atanf(a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = atanf(a[n * (j + 1) + i + 1]);
    }
  }

  __global__ void atan2_a_b(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = atan2f(a[m * (i + 0) + j + 0], b[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = atan2f(a[m * (i + 0) + j + 1], b[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = atan2f(a[m * (i + 1) + j + 0], b[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = atan2f(a[m * (i + 1) + j + 1], b[m * (i + 1) + j + 1]);
    }
  }

  __global__ void atan2_at_b(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = atan2f(a[n * (j + 0) + i + 0], b[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = atan2f(a[n * (j + 1) + i + 0], b[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = atan2f(a[n * (j + 0) + i + 1], b[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = atan2f(a[n * (j + 1) + i + 1], b[m * (i + 1) + j + 1]);
    }
  }

  __global__ void atan2_a_bt(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = atan2f(a[m * (i + 0) + j + 0], b[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = atan2f(a[m * (i + 0) + j + 1], b[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = atan2f(a[m * (i + 1) + j + 0], b[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = atan2f(a[m * (i + 1) + j + 1], b[n * (j + 1) + i + 1]);
    }
  }

  __global__ void atan2_at_bt(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = atan2f(a[n * (j + 0) + i + 0], b[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = atan2f(a[n * (j + 1) + i + 0], b[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = atan2f(a[n * (j + 0) + i + 1], b[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = atan2f(a[n * (j + 1) + i + 1], b[n * (j + 1) + i + 1]);
    }
  }

  __global__ void atan2_a_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = atan2f(a[m * (i + 0) + j + 0], b);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = atan2f(a[m * (i + 0) + j + 1], b);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = atan2f(a[m * (i + 1) + j + 0], b);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = atan2f(a[m * (i + 1) + j + 1], b);
    }
  }

  __global__ void atan2_at_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = atan2f(a[n * (j + 0) + i + 0], b);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = atan2f(a[n * (j + 1) + i + 0], b);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = atan2f(a[n * (j + 0) + i + 1], b);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = atan2f(a[n * (j + 1) + i + 1], b);
    }
  }

  __global__ void ratan2_a_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = atan2f(b, a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = atan2f(b, a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = atan2f(b, a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = atan2f(b, a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void ratan2_at_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = atan2f(b, a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = atan2f(b, a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = atan2f(b, a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = atan2f(b, a[n * (j + 1) + i + 1]);
    }
  }

  __global__ void sinh_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = sinhf(a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = sinhf(a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = sinhf(a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = sinhf(a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void sinh_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = sinhf(a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = sinhf(a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = sinhf(a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = sinhf(a[n * (j + 1) + i + 1]);
    }
  }

  __global__ void cosh_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = coshf(a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = coshf(a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = coshf(a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = coshf(a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void cosh_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = coshf(a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = coshf(a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = coshf(a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = coshf(a[n * (j + 1) + i + 1]);
    }
  }

  __global__ void asinh_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = asinhf(a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = asinhf(a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = asinhf(a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = asinhf(a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void asinh_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = asinhf(a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = asinhf(a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = asinhf(a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = asinhf(a[n * (j + 1) + i + 1]);
    }
  }

  __global__ void acosh_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = acoshf(a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = acoshf(a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = acoshf(a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = acoshf(a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void acosh_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = acoshf(a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = acoshf(a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = acoshf(a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = acoshf(a[n * (j + 1) + i + 1]);
    }
  }

  __global__ void atanh_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = atanhf(a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = atanhf(a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = atanhf(a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = atanhf(a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void atanh_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = atanhf(a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = atanhf(a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = atanhf(a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = atanhf(a[n * (j + 1) + i + 1]);
    }
  }

  __global__ void signum_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      if(!isnan(a[m * (i + 0) + j + 0])) {
        b[m * (i + 0) + j + 0] = (signbit(a[m * (i + 0) + j + 0]) ? -1.0 : 1.0);
      } else {
        b[m * (i + 0) + j + 0] = a[m * (i + 0) + j + 0];
      }
    }
    if(i + 0 < n && j + 1 < m) {
      if(!isnan(a[m * (i + 0) + j + 1])) {
        b[m * (i + 0) + j + 1] = (signbit(a[m * (i + 0) + j + 1]) ? -1.0 : 1.0);
      } else {
        b[m * (i + 0) + j + 1] = a[m * (i + 0) + j + 1];
      }
    }
    if(i + 1 < n && j + 0 < m) {
      if(!isnan(a[m * (i + 1) + j + 0])) {
        b[m * (i + 1) + j + 0] = (signbit(a[m * (i + 1) + j + 0]) ? -1.0 : 1.0);
      } else {
        b[m * (i + 1) + j + 0] = a[m * (i + 1) + j + 0];
      }
    }
    if(i + 1 < n && j + 1 < m) {
      if(!isnan(a[m * (i + 1) + j + 1])) {
        b[m * (i + 1) + j + 1] = (signbit(a[m * (i + 1) + j + 1]) ? -1.0 : 1.0);
      } else {
        b[m * (i + 1) + j + 1] = a[m * (i + 1) + j + 1];
      }
    }
  }

  __global__ void signum_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      if(!isnan(a[n * (j + 0) + i + 0])) {
        b[m * (i + 0) + j + 0] = (signbit(a[n * (j + 0) + i + 0]) ? -1.0 : 1.0);
      } else {
        b[m * (i + 0) + j + 0] = a[n * (j + 0) + i + 0];
      }
    }
    if(i + 0 < n && j + 1 < m) {
      if(!isnan(a[n * (j + 1) + i + 0])) {
        b[m * (i + 0) + j + 1] = (signbit(a[n * (j + 1) + i + 0]) ? -1.0 : 1.0);
      } else {
        b[m * (i + 0) + j + 1] = a[n * (j + 1) + i + 0];
      }
    }
    if(i + 1 < n && j + 0 < m) {
      if(!isnan(a[n * (j + 0) + i + 1])) {
        b[m * (i + 1) + j + 0] = (signbit(a[n * (j + 0) + i + 1]) ? -1.0 : 1.0);
      } else {
        b[m * (i + 1) + j + 0] = a[n * (j + 0) + i + 1];
      }
    }
    if(i + 1 < n && j + 1 < m) {
      if(!isnan(a[n * (j + 1) + i + 1])) {
        b[m * (i + 1) + j + 1] = (signbit(a[n * (j + 1) + i + 1]) ? -1.0 : 1.0);
      } else {
        b[m * (i + 1) + j + 1] = a[n * (j + 1) + i + 1];
      }
    }
  }

  __global__ void ceil_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = ceilf(a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = ceilf(a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = ceilf(a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = ceilf(a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void ceil_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = ceilf(a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = ceilf(a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = ceilf(a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = ceilf(a[n * (j + 1) + i + 1]);
    }
  }
  
  __global__ void floor_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = floorf(a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = floorf(a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = floorf(a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = floorf(a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void floor_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = floorf(a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = floorf(a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = floorf(a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = floorf(a[n * (j + 1) + i + 1]);
    }
  }

  __global__ void round_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = roundf(a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = roundf(a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = roundf(a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = roundf(a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void round_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = roundf(a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = roundf(a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = roundf(a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = roundf(a[n * (j + 1) + i + 1]);
    }
  }

  __global__ void trunc_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = truncf(a[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = truncf(a[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = truncf(a[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = truncf(a[m * (i + 1) + j + 1]);
    }
  }

  __global__ void trunc_at(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = truncf(a[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = truncf(a[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = truncf(a[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = truncf(a[n * (j + 1) + i + 1]);
    }
  }

  __global__ void max_a_b(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = fmaxf(a[m * (i + 0) + j + 0], b[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = fmaxf(a[m * (i + 0) + j + 1], b[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = fmaxf(a[m * (i + 1) + j + 0], b[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = fmaxf(a[m * (i + 1) + j + 1], b[m * (i + 1) + j + 1]);
    }
  }

  __global__ void max_at_b(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = fmaxf(a[n * (j + 0) + i + 0], b[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = fmaxf(a[n * (j + 1) + i + 0], b[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = fmaxf(a[n * (j + 0) + i + 1], b[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = fmaxf(a[n * (j + 1) + i + 1], b[m * (i + 1) + j + 1]);
    }
  }

  __global__ void max_a_bt(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = fmaxf(a[m * (i + 0) + j + 0], b[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = fmaxf(a[m * (i + 0) + j + 1], b[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = fmaxf(a[m * (i + 1) + j + 0], b[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = fmaxf(a[m * (i + 1) + j + 1], b[n * (j + 1) + i + 1]);
    }
  }

  __global__ void max_at_bt(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = fmaxf(a[n * (j + 0) + i + 0], b[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = fmaxf(a[n * (j + 1) + i + 0], b[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = fmaxf(a[n * (j + 0) + i + 1], b[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = fmaxf(a[n * (j + 1) + i + 1], b[n * (j + 1) + i + 1]);
    }
  }

  __global__ void max_a_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = fmaxf(a[m * (i + 0) + j + 0], b);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = fmaxf(a[m * (i + 0) + j + 1], b);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = fmaxf(a[m * (i + 1) + j + 0], b);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = fmaxf(a[m * (i + 1) + j + 1], b);
    }
  }

  __global__ void max_at_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = fmaxf(a[n * (j + 0) + i + 0], b);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = fmaxf(a[n * (j + 1) + i + 0], b);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = fmaxf(a[n * (j + 0) + i + 1], b);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = fmaxf(a[n * (j + 1) + i + 1], b);
    }
  }

  __global__ void min_a_b(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = fminf(a[m * (i + 0) + j + 0], b[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = fminf(a[m * (i + 0) + j + 1], b[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = fminf(a[m * (i + 1) + j + 0], b[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = fminf(a[m * (i + 1) + j + 1], b[m * (i + 1) + j + 1]);
    }
  }

  __global__ void min_at_b(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = fminf(a[n * (j + 0) + i + 0], b[m * (i + 0) + j + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = fminf(a[n * (j + 1) + i + 0], b[m * (i + 0) + j + 1]);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = fminf(a[n * (j + 0) + i + 1], b[m * (i + 1) + j + 0]);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = fminf(a[n * (j + 1) + i + 1], b[m * (i + 1) + j + 1]);
    }
  }

  __global__ void min_a_bt(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = fminf(a[m * (i + 0) + j + 0], b[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = fminf(a[m * (i + 0) + j + 1], b[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = fminf(a[m * (i + 1) + j + 0], b[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = fminf(a[m * (i + 1) + j + 1], b[n * (j + 1) + i + 1]);
    }
  }

  __global__ void min_at_bt(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = fminf(a[n * (j + 0) + i + 0], b[n * (j + 0) + i + 0]);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = fminf(a[n * (j + 1) + i + 0], b[n * (j + 1) + i + 0]);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = fminf(a[n * (j + 0) + i + 1], b[n * (j + 0) + i + 1]);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = fminf(a[n * (j + 1) + i + 1], b[n * (j + 1) + i + 1]);
    }
  }

  __global__ void min_a_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = fminf(a[m * (i + 0) + j + 0], b);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = fminf(a[m * (i + 0) + j + 1], b);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = fminf(a[m * (i + 1) + j + 0], b);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = fminf(a[m * (i + 1) + j + 1], b);
    }
  }

  __global__ void min_at_b_for_scalar(const float *a, float b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t j = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    if(i + 0 < n && j + 0 < m) {
      c[m * (i + 0) + j + 0] = fminf(a[n * (j + 0) + i + 0], b);
    }
    if(i + 0 < n && j + 1 < m) {
      c[m * (i + 0) + j + 1] = fminf(a[n * (j + 1) + i + 0], b);
    }
    if(i + 1 < n && j + 0 < m) {
      c[m * (i + 1) + j + 0] = fminf(a[n * (j + 0) + i + 1], b);
    }
    if(i + 1 < n && j + 1 < m) {
      c[m * (i + 1) + j + 1] = fminf(a[n * (j + 1) + i + 1], b);
    }
  }
}
