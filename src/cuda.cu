//
// Copyright (c) 2025-2026 Łukasz Szpakowski
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
#define TILE_SIZE       (1024 << 2)

#define MTHREAD_SIZE    16
#define MTILE_WIDTH     (MTHREAD_SIZE << 2)

#define MMA_TILE_WIDTH  64

extern "C" {
  __global__ void transpose_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    __shared__ float as[MTILE_WIDTH][MTHREAD_SIZE];
    __shared__ float bs[MTILE_WIDTH][MTHREAD_SIZE];
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 2;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 2;
    size_t k;
    size_t ti = threadIdx.x;
    size_t tj = threadIdx.y;
    size_t bi = ti << 2;
    size_t bj = tj << 2;
    size_t ii = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t ii1 = (ii + 0) & 3;
    size_t ii2 = (ii + 1) & 3;
    size_t ii3 = (ii + 2) & 3;
    size_t ii4 = (ii + 3) & 3;
    size_t jj = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    size_t jj1 = (jj + 0) & 3;
    size_t jj2 = (jj + 1) & 3;
    size_t jj3 = (jj + 2) & 3;
    size_t jj4 = (jj + 3) & 3;
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
    for(k = 0; k < l; k += MTHREAD_SIZE) {
      size_t tk;
      as[bi + ii1][tj] = 0.0f;
      if(i + 0 < n && k + tj < l) {
        as[bi + ii1][tj] = a[l * (i + 0) + k + tj];
      }
      as[bi + ii2][tj] = 0.0f;
      if(i + 1 < n && k + tj < l) {
        as[bi + ii2][tj] = a[l * (i + 1) + k + tj];
      }
      as[bi + ii3][tj] = 0.0f;
      if(i + 2 < n && k + tj < l) {
        as[bi + ii3][tj] = a[l * (i + 2) + k + tj];
      }
      as[bi + ii4][tj] = 0.0f;
      if(i + 3 < n && k + tj < l) {
        as[bi + ii4][tj] = a[l * (i + 3) + k + tj];
      }
      bs[bj + jj1][ti] = 0.0f;
      if(j + 0 < m && k + ti < l) {
        bs[bj + jj1][ti] = b[m * (k + ti) + j + 0];
      }
      bs[bj + jj2][ti] = 0.0f;
      if(j + 1 < m && k + ti < l) {
        bs[bj + jj2][ti] = b[m * (k + ti) + j + 1];
      }
      bs[bj + jj3][ti] = 0.0f;
      if(j + 2 < m && k + ti < l) {
        bs[bj + jj3][ti] = b[m * (k + ti) + j + 2];
      }
      bs[bj + jj4][ti] = 0.0f;
      if(j + 3 < m && k + ti < l) {
        bs[bj + jj4][ti] = b[m * (k + ti) + j + 3];
      }
      __syncthreads();
      for(tk = 0; tk < MTHREAD_SIZE; tk++) {
        ar1 = as[bi + ii1][tk];
        ar2 = as[bi + ii2][tk];
        ar3 = as[bi + ii3][tk];
        ar4 = as[bi + ii4][tk];
        br1 = bs[bj + jj1][tk];
        br2 = bs[bj + jj2][tk];
        br3 = bs[bj + jj3][tk];
        br4 = bs[bj + jj4][tk];
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
    __shared__ float as[MTILE_WIDTH][MTHREAD_SIZE];
    __shared__ float bs[MTILE_WIDTH][MTHREAD_SIZE];
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 2;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 2;
    size_t k;
    size_t ti = threadIdx.x;
    size_t tj = threadIdx.y;
    size_t bi = ti << 2;
    size_t bj = tj << 2;
    size_t ii = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t ii1 = (ii + 0) & 3;
    size_t ii2 = (ii + 1) & 3;
    size_t ii3 = (ii + 2) & 3;
    size_t ii4 = (ii + 3) & 3;
    size_t jj = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    size_t jj1 = (jj + 0) & 3;
    size_t jj2 = (jj + 1) & 3;
    size_t jj3 = (jj + 2) & 3;
    size_t jj4 = (jj + 3) & 3;
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
    for(k = 0; k < l; k += MTHREAD_SIZE) {
      size_t tk;
      as[bi + ii1][tj] = 0.0f;
      if(i + 0 < n && k + tj < l) {
        as[bi + ii1][tj] = a[n * (k + tj) + i + 0];
      }
      as[bi + ii2][tj] = 0.0f;
      if(i + 1 < n && k + tj < l) {
        as[bi + ii2][tj] = a[n * (k + tj) + i + 1];
      }
      as[bi + ii3][tj] = 0.0f;
      if(i + 2 < n && k + tj < l) {
        as[bi + ii3][tj] = a[n * (k + tj) + i + 2];
      }
      as[bi + ii4][tj] = 0.0f;
      if(i + 3 < n && k + tj < l) {
        as[bi + ii4][tj] = a[n * (k + tj) + i + 3];
      }
      bs[bj + jj1][ti] = 0.0f;
      if(j + 0 < m && k + ti < l) {
        bs[bj + jj1][ti] = b[m * (k + ti) + j + 0];
      }
      bs[bj + jj2][ti] = 0.0f;
      if(j + 1 < m && k + ti < l) {
        bs[bj + jj2][ti] = b[m * (k + ti) + j + 1];
      }
      bs[bj + jj3][ti] = 0.0f;
      if(j + 2 < m && k + ti < l) {
        bs[bj + jj3][ti] = b[m * (k + ti) + j + 2];
      }
      bs[bj + jj4][ti] = 0.0f;
      if(j + 3 < m && k + ti < l) {
        bs[bj + jj4][ti] = b[m * (k + ti) + j + 3];
      }
      __syncthreads();
      for(tk = 0; tk < MTHREAD_SIZE; tk++) {
        ar1 = as[bi + ii1][tk];
        ar2 = as[bi + ii2][tk];
        ar3 = as[bi + ii3][tk];
        ar4 = as[bi + ii4][tk];
        br1 = bs[bj + jj1][tk];
        br2 = bs[bj + jj2][tk];
        br3 = bs[bj + jj3][tk];
        br4 = bs[bj + jj4][tk];
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
    __shared__ float as[MTILE_WIDTH][MTHREAD_SIZE];
    __shared__ float bs[MTILE_WIDTH][MTHREAD_SIZE];
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 2;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 2;
    size_t k;
    size_t ti = threadIdx.x;
    size_t tj = threadIdx.y;
    size_t bi = ti << 2;
    size_t bj = tj << 2;
    size_t ii = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t ii1 = (ii + 0) & 3;
    size_t ii2 = (ii + 1) & 3;
    size_t ii3 = (ii + 2) & 3;
    size_t ii4 = (ii + 3) & 3;
    size_t jj = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    size_t jj1 = (jj + 0) & 3;
    size_t jj2 = (jj + 1) & 3;
    size_t jj3 = (jj + 2) & 3;
    size_t jj4 = (jj + 3) & 3;
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
    for(k = 0; k < l; k += MTHREAD_SIZE) {
      size_t tk;
      as[bi + ii1][tj] = 0.0f;
      if(i + 0 < n && k + tj < l) {
        as[bi + ii1][tj] = a[l * (i + 0) + k + tj];
      }
      as[bi + ii2][tj] = 0.0f;
      if(i + 1 < n && k + tj < l) {
        as[bi + ii2][tj] = a[l * (i + 1) + k + tj];
      }
      as[bi + ii3][tj] = 0.0f;
      if(i + 2 < n && k + tj < l) {
        as[bi + ii3][tj] = a[l * (i + 2) + k + tj];
      }
      as[bi + ii4][tj] = 0.0f;
      if(i + 3 < n && k + tj < l) {
        as[bi + ii4][tj] = a[l * (i + 3) + k + tj];
      }
      bs[bj + jj1][ti] = 0.0f;
      if(j + 0 < m && k + ti < l) {
        bs[bj + jj1][ti] = b[l * (j + 0) + k + ti];
      }
      bs[bj + jj2][ti] = 0.0f;
      if(j + 1 < m && k + ti < l) {
        bs[bj + jj2][ti] = b[l * (j + 1) + k + ti];
      }
      bs[bj + jj3][ti] = 0.0f;
      if(j + 2 < m && k + ti < l) {
        bs[bj + jj3][ti] = b[l * (j + 2) + k + ti];
      }
      bs[bj + jj4][ti] = 0.0f;
      if(j + 3 < m && k + ti < l) {
        bs[bj + jj4][ti] = b[l * (j + 3) + k + ti];
      }
      __syncthreads();
      for(tk = 0; tk < MTHREAD_SIZE; tk++) {
        ar1 = as[bi + ii1][tk];
        ar2 = as[bi + ii2][tk];
        ar3 = as[bi + ii3][tk];
        ar4 = as[bi + ii4][tk];
        br1 = bs[bj + jj1][tk];
        br2 = bs[bj + jj2][tk];
        br3 = bs[bj + jj3][tk];
        br4 = bs[bj + jj4][tk];
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
    __shared__ float as[MTILE_WIDTH][MTHREAD_SIZE];
    __shared__ float bs[MTILE_WIDTH][MTHREAD_SIZE];
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 2;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 2;
    size_t k;
    size_t ti = threadIdx.x;
    size_t tj = threadIdx.y;
    size_t bi = ti << 2;
    size_t bj = tj << 2;
    size_t ii = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t ii1 = (ii + 0) & 3;
    size_t ii2 = (ii + 1) & 3;
    size_t ii3 = (ii + 2) & 3;
    size_t ii4 = (ii + 3) & 3;
    size_t jj = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    size_t jj1 = (jj + 0) & 3;
    size_t jj2 = (jj + 1) & 3;
    size_t jj3 = (jj + 2) & 3;
    size_t jj4 = (jj + 3) & 3;
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
    for(k = 0; k < l; k += MTHREAD_SIZE) {
      size_t tk;
      as[bi + ii1][tj] = 0.0f;
      if(i + 0 < n && k + tj < l) {
        as[bi + ii1][tj] = a[n * (k + tj) + i + 0];
      }
      as[bi + ii2][tj] = 0.0f;
      if(i + 1 < n && k + tj < l) {
        as[bi + ii2][tj] = a[n * (k + tj) + i + 1];
      }
      as[bi + ii3][tj] = 0.0f;
      if(i + 2 < n && k + tj < l) {
        as[bi + ii3][tj] = a[n * (k + tj) + i + 2];
      }
      as[bi + ii4][tj] = 0.0f;
      if(i + 3 < n && k + tj < l) {
        as[bi + ii4][tj] = a[n * (k + tj) + i + 3];
      }
      bs[bj + jj1][ti] = 0.0f;
      if(j + 0 < m && k + ti < l) {
        bs[bj + jj1][ti] = b[l * (j + 0) + k + ti];
      }
      bs[bj + jj2][ti] = 0.0f;
      if(j + 1 < m && k + ti < l) {
        bs[bj + jj2][ti] = b[l * (j + 1) + k + ti];
      }
      bs[bj + jj3][ti] = 0.0f;
      if(j + 2 < m && k + ti < l) {
        bs[bj + jj3][ti] = b[l * (j + 2) + k + ti];
      }
      bs[bj + jj4][ti] = 0.0f;
      if(j + 3 < m && k + ti < l) {
        bs[bj + jj4][ti] = b[l * (j + 3) + k + ti];
      }
      __syncthreads();
      for(tk = 0; tk < MTHREAD_SIZE; tk++) {
        ar1 = as[bi + ii1][tk];
        ar2 = as[bi + ii2][tk];
        ar3 = as[bi + ii3][tk];
        ar4 = as[bi + ii4][tk];
        br1 = bs[bj + jj1][tk];
        br2 = bs[bj + jj2][tk];
        br3 = bs[bj + jj3][tk];
        br4 = bs[bj + jj4][tk];
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

#endif

  __global__ void mul_a_b_for_elems(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    __shared__ float es[TILE_SIZE];
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t k;
    size_t tile_width = blockDim.y << 1;
    size_t tile_height = blockDim.x << 1;
    size_t ti = threadIdx.x << 1;
    size_t tj = threadIdx.y << 1;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    for(k = 0; k < n; k += tile_height) {
      size_t tk;
      es[tile_width * (ti + 0) + tj + 0] = 0.0f;
      if(j + 0 < m && k + ti + 0 < n) {
        es[tile_width * (ti + 0) + tj + 0] = expf(a[m * (k + ti + 0) + j + 0]);
      }
      es[tile_width * (ti + 0) + tj + 1] = 0.0f;
      if(j + 1 < m && k + ti + 0 < n) {
        es[tile_width * (ti + 0) + tj + 1] = expf(a[m * (k + ti + 0) + j + 1]);
      }
      es[tile_width * (ti + 1) + tj + 0] = 0.0f;
      if(j + 0 < m && k + ti + 1 < n) {
        es[tile_width * (ti + 1) + tj + 0] = expf(a[m * (k + ti + 1) + j + 0]);
      }
      es[tile_width * (ti + 1) + tj + 1] = 0.0f;
      if(j + 1 < m && k + ti + 1 < n) {
        es[tile_width * (ti + 1) + tj + 1] = expf(a[m * (k + ti + 1) + j + 1]);
      }
      __syncthreads();
      for(tk = 0; tk < tile_height; tk += 2) {
        sum1 += es[tile_width * (tk + 0) + tj + 0];
        sum1 += es[tile_width * (tk + 1) + tj + 0];
        sum2 += es[tile_width * (tk + 0) + tj + 1];
        sum2 += es[tile_width * (tk + 1) + tj + 1];
      }
      __syncthreads();
    }
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = expf(a[m * (i + 0) + j + 0]) / sum1;
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = expf(a[m * (i + 0) + j + 1]) / sum2;
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = expf(a[m * (i + 1) + j + 0]) / sum1;
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = expf(a[m * (i + 1) + j + 1]) / sum2;
    }
  }

  __global__ void softmax_at(const float *a, float *b, size_t n, size_t m)
  {
    __shared__ float es[TILE_SIZE];
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
    size_t k;
    size_t tile_width = blockDim.y << 1;
    size_t tile_height = blockDim.x << 1;
    size_t ti = threadIdx.x << 1;
    size_t tj = threadIdx.y << 1;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    for(k = 0; k < n; k += tile_height) {
      size_t tk;
      es[tile_width * (ti + 0) + tj + 0] = 0.0f;
      if(j + 0 < m && k + ti + 0 < n) {
        es[tile_width * (ti + 0) + tj + 0] = expf(a[n * (j + 0) + k + ti + 0]);
      }
      es[tile_width * (ti + 0) + tj + 1] = 0.0f;
      if(j + 1 < m && k + ti + 0 < n) {
        es[tile_width * (ti + 0) + tj + 1] = expf(a[n * (j + 1) + k + ti + 0]);
      }
      es[tile_width * (ti + 1) + tj + 0] = 0.0f;
      if(j + 0 < m && k + ti + 1 < n) {
        es[tile_width * (ti + 1) + tj + 0] = expf(a[n * (j + 0) + k + ti + 1]);
      }
      es[tile_width * (ti + 1) + tj + 1] = 0.0f;
      if(j + 1 < m && k + ti + 1 < n) {
        es[tile_width * (ti + 1) + tj + 1] = expf(a[n * (j + 1) + k + ti + 1]);
      }
      __syncthreads();
      for(tk = 0; tk < tile_height; tk += 2) {
        sum1 += es[tile_width * (tk + 0) + tj + 0];
        sum1 += es[tile_width * (tk + 1) + tj + 0];
        sum2 += es[tile_width * (tk + 0) + tj + 1];
        sum2 += es[tile_width * (tk + 1) + tj + 1];
      }
      __syncthreads();
    }
    if(i + 0 < n && j + 0 < m) {
      b[m * (i + 0) + j + 0] = expf(a[n * (j + 0) + i + 0]) / sum1;
    }
    if(i + 0 < n && j + 1 < m) {
      b[m * (i + 0) + j + 1] = expf(a[n * (j + 1) + i + 0]) / sum2;
    }
    if(i + 1 < n && j + 0 < m) {
      b[m * (i + 1) + j + 0] = expf(a[n * (j + 0) + i + 1]) / sum1;
    }
    if(i + 1 < n && j + 1 < m) {
      b[m * (i + 1) + j + 1] = expf(a[n * (j + 1) + i + 1]) / sum2;
    }
  }

  __global__ void sqrt_a(const float *a, float *b, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x << 1;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y << 1;
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
