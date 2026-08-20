/*
 * Copyright (c) 2025-2026 Łukasz Szpakowski
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
__kernel void transpose_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = a[n2 * (j + 0) + i + 0];
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = a[n2 * (j + 1) + i + 0];
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = a[n2 * (j + 0) + i + 1];
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = a[n2 * (j + 1) + i + 1];
  }
}

__kernel void add_a_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[m2 * (i + 0) + j + 0] + b[m2 * (i + 0) + j + 0];
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[m2 * (i + 0) + j + 1] + b[m2 * (i + 0) + j + 1];
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[m2 * (i + 1) + j + 0] + b[m2 * (i + 1) + j + 0];
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[m2 * (i + 1) + j + 1] + b[m2 * (i + 1) + j + 1];
  }
}

__kernel void add_at_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[n2 * (j + 0) + i + 0] + b[m2 * (i + 0) + j + 0];
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[n2 * (j + 1) + i + 0] + b[m2 * (i + 0) + j + 1];
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[n2 * (j + 0) + i + 1] + b[m2 * (i + 1) + j + 0];
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[n2 * (j + 1) + i + 1] + b[m2 * (i + 1) + j + 1];
  }
}

__kernel void add_a_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[m2 * (i + 0) + j + 0] + b[n2 * (j + 0) + i + 0];
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[m2 * (i + 0) + j + 1] + b[n2 * (j + 1) + i + 0];
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[m2 * (i + 1) + j + 0] + b[n2 * (j + 0) + i + 1];
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[m2 * (i + 1) + j + 1] + b[n2 * (j + 1) + i + 1];
  }
}

__kernel void add_at_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[n2 * (j + 0) + i + 0] + b[n2 * (j + 0) + i + 0];
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[n2 * (j + 1) + i + 0] + b[n2 * (j + 1) + i + 0];
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[n2 * (j + 0) + i + 1] + b[n2 * (j + 0) + i + 1];
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[n2 * (j + 1) + i + 1] + b[n2 * (j + 1) + i + 1];
  }
}

__kernel void sub_a_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[m2 * (i + 0) + j + 0] - b[m2 * (i + 0) + j + 0];
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[m2 * (i + 0) + j + 1] - b[m2 * (i + 0) + j + 1];
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[m2 * (i + 1) + j + 0] - b[m2 * (i + 1) + j + 0];
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[m2 * (i + 1) + j + 1] - b[m2 * (i + 1) + j + 1];
  }
}

__kernel void sub_at_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[n2 * (j + 0) + i + 0] - b[m2 * (i + 0) + j + 0];
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[n2 * (j + 1) + i + 0] - b[m2 * (i + 0) + j + 1];
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[n2 * (j + 0) + i + 1] - b[m2 * (i + 1) + j + 0];
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[n2 * (j + 1) + i + 1] - b[m2 * (i + 1) + j + 1];
  }
}

__kernel void sub_a_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[m2 * (i + 0) + j + 0] - b[n2 * (j + 0) + i + 0];
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[m2 * (i + 0) + j + 1] - b[n2 * (j + 1) + i + 0];
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[m2 * (i + 1) + j + 0] - b[n2 * (j + 0) + i + 1];
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[m2 * (i + 1) + j + 1] - b[n2 * (j + 1) + i + 1];
  }
}

__kernel void sub_at_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[n2 * (j + 0) + i + 0] - b[n2 * (j + 0) + i + 0];
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[n2 * (j + 1) + i + 0] - b[n2 * (j + 1) + i + 0];
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[n2 * (j + 0) + i + 1] - b[n2 * (j + 0) + i + 1];
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[n2 * (j + 1) + i + 1] - b[n2 * (j + 1) + i + 1];
  }
}

__kernel void mul_a_b(__global const float *a, __global const float *b, __global float *c, __local float4 *as, __local float4 *bs, ulong n, ulong m, ulong l)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t l2 = (size_t) l;
  size_t i = get_global_id(1) << 3;
  size_t j = get_global_id(0) << 2;
  size_t k;
  size_t mthread_size = get_local_size(0);
  size_t ti = get_local_id(1);
  size_t tj = get_local_id(0);
  size_t ik = get_global_id(1);
  size_t jk = get_global_id(0);
  __private float4 ar1;
  __private float4 ar2;
  __private float4 br;
  __private float4 cr1 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr2 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr3 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr4 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr5 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr6 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr7 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr8 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  for(k = 0; k < l2; k += mthread_size) {
    size_t tk;
    size_t tjik = (tj + ik) % mthread_size;
    size_t tijk = (ti + jk) % mthread_size;
    as[mthread_size * ti + tjik].x = 0.0f;
    if(i + 0 < n2 && k + tj < l2) {
      as[mthread_size * ti + tjik].x = a[l2 * (i + 0) + k + tj];
    }
    as[mthread_size * ti + tjik].y = 0.0f;
    if(i + 1 < n2 && k + tj < l2) {
      as[mthread_size * ti + tjik].y = a[l2 * (i + 1) + k + tj];
    }
    as[mthread_size * ti + tjik].z = 0.0f;
    if(i + 2 < n2 && k + tj < l2) {
      as[mthread_size * ti + tjik].z = a[l2 * (i + 2) + k + tj];
    }
    as[mthread_size * ti + tjik].w = 0.0f;
    if(i + 3 < n2 && k + tj < l2) {
      as[mthread_size * ti + tjik].w = a[l2 * (i + 3) + k + tj];
    }
    as[mthread_size * (ti + mthread_size) + tjik].x = 0.0f;
    if(i + 4 < n2 && k + tj < l2) {
      as[mthread_size * (ti + mthread_size) + tjik].x = a[l2 * (i + 4) + k + tj];
    }
    as[mthread_size * (ti + mthread_size) + tjik].y = 0.0f;
    if(i + 5 < n2 && k + tj < l2) {
      as[mthread_size * (ti + mthread_size) + tjik].y = a[l2 * (i + 5) + k + tj];
    }
    as[mthread_size * (ti + mthread_size) + tjik].z = 0.0f;
    if(i + 6 < n2 && k + tj < l2) {
      as[mthread_size * (ti + mthread_size) + tjik].z = a[l2 * (i + 6) + k + tj];
    }
    as[mthread_size * (ti + mthread_size) + tjik].w = 0.0f;
    if(i + 7 < n2 && k + tj < l2) {
      as[mthread_size * (ti + mthread_size) + tjik].w = a[l2 * (i + 7) + k + tj];
    }
    bs[mthread_size * tj + tijk].x = 0.0f;
    if(j + 0 < m2 && k + ti < l2) {
      bs[mthread_size * tj + tijk].x = b[m2 * (k + ti) + j + 0];
    }
    bs[mthread_size * tj + tijk].y = 0.0f;
    if(j + 1 < m2 && k + ti < l2) {
      bs[mthread_size * tj + tijk].y = b[m2 * (k + ti) + j + 1];
    }
    bs[mthread_size * tj + tijk].z = 0.0f;
    if(j + 2 < m2 && k + ti < l2) {
      bs[mthread_size * tj + tijk].z = b[m2 * (k + ti) + j + 2];
    }
    bs[mthread_size * tj + tijk].w = 0.0f;
    if(j + 3 < m2 && k + ti < l2) {
      bs[mthread_size * tj + tijk].w = b[m2 * (k + ti) + j + 3];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll 16
    for(tk = 0; tk < mthread_size; tk++) {
      size_t tkik = (tk + ik) % mthread_size;
      size_t tkjk = (tk + jk) % mthread_size;
      ar1 = as[mthread_size * ti + tkik];
      ar2 = as[mthread_size * (ti + mthread_size) + tkik];
      br = bs[mthread_size * tj + tkjk];
      cr1 += ar1.x * br;
      cr2 += ar1.y * br;
      cr3 += ar1.z * br;
      cr4 += ar1.w * br;
      cr5 += ar2.x * br;
      cr6 += ar2.y * br;
      cr7 += ar2.z * br;
      cr8 += ar2.w * br;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = cr1.x;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = cr1.y;
  }
  if(i + 0 < n2 && j + 2 < m2) {
    c[m2 * (i + 0) + j + 2] = cr1.z;
  }
  if(i + 0 < n2 && j + 3 < m2) {
    c[m2 * (i + 0) + j + 3] = cr1.w;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = cr2.x;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = cr2.y;
  }
  if(i + 1 < n2 && j + 2 < m2) {
    c[m2 * (i + 1) + j + 2] = cr2.z;
  }
  if(i + 1 < n2 && j + 3 < m2) {
    c[m2 * (i + 1) + j + 3] = cr2.w;
  }
  if(i + 2 < n2 && j + 0 < m2) {
    c[m2 * (i + 2) + j + 0] = cr3.x;
  }
  if(i + 2 < n2 && j + 1 < m2) {
    c[m2 * (i + 2) + j + 1] = cr3.y;
  }
  if(i + 2 < n2 && j + 2 < m2) {
    c[m2 * (i + 2) + j + 2] = cr3.z;
  }
  if(i + 2 < n2 && j + 3 < m2) {
    c[m2 * (i + 2) + j + 3] = cr3.w;
  }
  if(i + 3 < n2 && j + 0 < m2) {
    c[m2 * (i + 3) + j + 0] = cr4.x;
  }
  if(i + 3 < n2 && j + 1 < m2) {
    c[m2 * (i + 3) + j + 1] = cr4.y;
  }
  if(i + 3 < n2 && j + 2 < m2) {
    c[m2 * (i + 3) + j + 2] = cr4.z;
  }
  if(i + 3 < n2 && j + 3 < m2) {
    c[m2 * (i + 3) + j + 3] = cr4.w;
  }
  if(i + 4 < n2 && j + 0 < m2) {
    c[m2 * (i + 4) + j + 0] = cr5.x;
  }
  if(i + 4 < n2 && j + 1 < m2) {
    c[m2 * (i + 4) + j + 1] = cr5.y;
  }
  if(i + 4 < n2 && j + 2 < m2) {
    c[m2 * (i + 4) + j + 2] = cr5.z;
  }
  if(i + 4 < n2 && j + 3 < m2) {
    c[m2 * (i + 4) + j + 3] = cr5.w;
  }
  if(i + 5 < n2 && j + 0 < m2) {
    c[m2 * (i + 5) + j + 0] = cr6.x;
  }
  if(i + 5 < n2 && j + 1 < m2) {
    c[m2 * (i + 5) + j + 1] = cr6.y;
  }
  if(i + 5 < n2 && j + 2 < m2) {
    c[m2 * (i + 5) + j + 2] = cr6.z;
  }
  if(i + 5 < n2 && j + 3 < m2) {
    c[m2 * (i + 5) + j + 3] = cr6.w;
  }
  if(i + 6 < n2 && j + 0 < m2) {
    c[m2 * (i + 6) + j + 0] = cr7.x;
  }
  if(i + 6 < n2 && j + 1 < m2) {
    c[m2 * (i + 6) + j + 1] = cr7.y;
  }
  if(i + 6 < n2 && j + 2 < m2) {
    c[m2 * (i + 6) + j + 2] = cr7.z;
  }
  if(i + 6 < n2 && j + 3 < m2) {
    c[m2 * (i + 6) + j + 3] = cr7.w;
  }
  if(i + 7 < n2 && j + 0 < m2) {
    c[m2 * (i + 7) + j + 0] = cr8.x;
  }
  if(i + 7 < n2 && j + 1 < m2) {
    c[m2 * (i + 7) + j + 1] = cr8.y;
  }
  if(i + 7 < n2 && j + 2 < m2) {
    c[m2 * (i + 7) + j + 2] = cr8.z;
  }
  if(i + 7 < n2 && j + 3 < m2) {
    c[m2 * (i + 7) + j + 3] = cr8.w;
  }
}

__kernel void mul_at_b(__global const float *a, __global const float *b, __global float *c, __local float4 *as, __local float4 *bs, ulong n, ulong m, ulong l)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t l2 = (size_t) l;
  size_t i = get_global_id(0) << 3;
  size_t j = get_global_id(1) << 2;
  size_t k;
  size_t mthread_size = get_local_size(0);
  size_t ti = get_local_id(0);
  size_t tj = get_local_id(1);
  size_t ik = get_global_id(0);
  size_t jk = get_global_id(1);
  __private float4 ar1;
  __private float4 ar2;
  __private float4 br;
  __private float4 cr1 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr2 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr3 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr4 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr5 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr6 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr7 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr8 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  for(k = 0; k < l2; k += mthread_size) {
    size_t tk;
    size_t tjik = (tj + ik) % mthread_size;
    size_t tijk = (ti + jk) % mthread_size;
    as[mthread_size * ti + tjik].x = 0.0f;
    if(i + 0 < n2 && k + tj < l2) {
      as[mthread_size * ti + tjik].x = a[n2 * (k + tj) + i + 0];
    }
    as[mthread_size * ti + tjik].y = 0.0f;
    if(i + 1 < n2 && k + tj < l2) {
      as[mthread_size * ti + tjik].y = a[n2 * (k + tj) + i + 1];
    }
    as[mthread_size * ti + tjik].z = 0.0f;
    if(i + 2 < n2 && k + tj < l2) {
      as[mthread_size * ti + tjik].z = a[n2 * (k + tj) + i + 2];
    }
    as[mthread_size * ti + tjik].w = 0.0f;
    if(i + 3 < n2 && k + tj < l2) {
      as[mthread_size * ti + tjik].w = a[n2 * (k + tj) + i + 3];
    }
    as[mthread_size * (ti + mthread_size) + tjik].x = 0.0f;
    if(i + 4 < n2 && k + tj < l2) {
      as[mthread_size * (ti + mthread_size) + tjik].x = a[n2 * (k + tj) + i + 4];
    }
    as[mthread_size * (ti + mthread_size) + tjik].y = 0.0f;
    if(i + 5 < n2 && k + tj < l2) {
      as[mthread_size * (ti + mthread_size) + tjik].y = a[n2 * (k + tj) + i + 5];
    }
    as[mthread_size * (ti + mthread_size) + tjik].z = 0.0f;
    if(i + 6 < n2 && k + tj < l2) {
      as[mthread_size * (ti + mthread_size) + tjik].z = a[n2 * (k + tj) + i + 6];
    }
    as[mthread_size * (ti + mthread_size) + tjik].w = 0.0f;
    if(i + 7 < n2 && k + tj < l2) {
      as[mthread_size * (ti + mthread_size) + tjik].w = a[n2 * (k + tj) + i + 7];
    }
    bs[mthread_size * tj + tijk].x = 0.0f;
    if(j + 0 < m2 && k + ti < l2) {
      bs[mthread_size * tj + tijk].x = b[m2 * (k + ti) + j + 0];
    }
    bs[mthread_size * tj + tijk].y = 0.0f;
    if(j + 1 < m2 && k + ti < l2) {
      bs[mthread_size * tj + tijk].y = b[m2 * (k + ti) + j + 1];
    }
    bs[mthread_size * tj + tijk].z = 0.0f;
    if(j + 2 < m2 && k + ti < l2) {
      bs[mthread_size * tj + tijk].z = b[m2 * (k + ti) + j + 2];
    }
    bs[mthread_size * tj + tijk].w = 0.0f;
    if(j + 3 < m2 && k + ti < l2) {
      bs[mthread_size * tj + tijk].w = b[m2 * (k + ti) + j + 3];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll 16
    for(tk = 0; tk < mthread_size; tk++) {
      size_t tkik = (tk + ik) % mthread_size;
      size_t tkjk = (tk + jk) % mthread_size;
      ar1 = as[mthread_size * ti + tkik];
      ar2 = as[mthread_size * (ti + mthread_size) + tkik];
      br = bs[mthread_size * tj + tkjk];
      cr1 += ar1.x * br;
      cr2 += ar1.y * br;
      cr3 += ar1.z * br;
      cr4 += ar1.w * br;
      cr5 += ar2.x * br;
      cr6 += ar2.y * br;
      cr7 += ar2.z * br;
      cr8 += ar2.w * br;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = cr1.x;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = cr1.y;
  }
  if(i + 0 < n2 && j + 2 < m2) {
    c[m2 * (i + 0) + j + 2] = cr1.z;
  }
  if(i + 0 < n2 && j + 3 < m2) {
    c[m2 * (i + 0) + j + 3] = cr1.w;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = cr2.x;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = cr2.y;
  }
  if(i + 1 < n2 && j + 2 < m2) {
    c[m2 * (i + 1) + j + 2] = cr2.z;
  }
  if(i + 1 < n2 && j + 3 < m2) {
    c[m2 * (i + 1) + j + 3] = cr2.w;
  }
  if(i + 2 < n2 && j + 0 < m2) {
    c[m2 * (i + 2) + j + 0] = cr3.x;
  }
  if(i + 2 < n2 && j + 1 < m2) {
    c[m2 * (i + 2) + j + 1] = cr3.y;
  }
  if(i + 2 < n2 && j + 2 < m2) {
    c[m2 * (i + 2) + j + 2] = cr3.z;
  }
  if(i + 2 < n2 && j + 3 < m2) {
    c[m2 * (i + 2) + j + 3] = cr3.w;
  }
  if(i + 3 < n2 && j + 0 < m2) {
    c[m2 * (i + 3) + j + 0] = cr4.x;
  }
  if(i + 3 < n2 && j + 1 < m2) {
    c[m2 * (i + 3) + j + 1] = cr4.y;
  }
  if(i + 3 < n2 && j + 2 < m2) {
    c[m2 * (i + 3) + j + 2] = cr4.z;
  }
  if(i + 3 < n2 && j + 3 < m2) {
    c[m2 * (i + 3) + j + 3] = cr4.w;
  }
  if(i + 4 < n2 && j + 0 < m2) {
    c[m2 * (i + 4) + j + 0] = cr5.x;
  }
  if(i + 4 < n2 && j + 1 < m2) {
    c[m2 * (i + 4) + j + 1] = cr5.y;
  }
  if(i + 4 < n2 && j + 2 < m2) {
    c[m2 * (i + 4) + j + 2] = cr5.z;
  }
  if(i + 4 < n2 && j + 3 < m2) {
    c[m2 * (i + 4) + j + 3] = cr5.w;
  }
  if(i + 5 < n2 && j + 0 < m2) {
    c[m2 * (i + 5) + j + 0] = cr6.x;
  }
  if(i + 5 < n2 && j + 1 < m2) {
    c[m2 * (i + 5) + j + 1] = cr6.y;
  }
  if(i + 5 < n2 && j + 2 < m2) {
    c[m2 * (i + 5) + j + 2] = cr6.z;
  }
  if(i + 5 < n2 && j + 3 < m2) {
    c[m2 * (i + 5) + j + 3] = cr6.w;
  }
  if(i + 6 < n2 && j + 0 < m2) {
    c[m2 * (i + 6) + j + 0] = cr7.x;
  }
  if(i + 6 < n2 && j + 1 < m2) {
    c[m2 * (i + 6) + j + 1] = cr7.y;
  }
  if(i + 6 < n2 && j + 2 < m2) {
    c[m2 * (i + 6) + j + 2] = cr7.z;
  }
  if(i + 6 < n2 && j + 3 < m2) {
    c[m2 * (i + 6) + j + 3] = cr7.w;
  }
  if(i + 7 < n2 && j + 0 < m2) {
    c[m2 * (i + 7) + j + 0] = cr8.x;
  }
  if(i + 7 < n2 && j + 1 < m2) {
    c[m2 * (i + 7) + j + 1] = cr8.y;
  }
  if(i + 7 < n2 && j + 2 < m2) {
    c[m2 * (i + 7) + j + 2] = cr8.z;
  }
  if(i + 7 < n2 && j + 3 < m2) {
    c[m2 * (i + 7) + j + 3] = cr8.w;
  }
}

__kernel void mul_a_bt(__global const float *a, __global const float *b, __global float *c, __local float4 *as, __local float4 *bs, ulong n, ulong m, ulong l)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t l2 = (size_t) l;
  size_t i = get_global_id(1) << 3;
  size_t j = get_global_id(0) << 2;
  size_t k;
  size_t mthread_size = get_local_size(0);
  size_t ti = get_local_id(1);
  size_t tj = get_local_id(0);
  size_t ik = get_global_id(1);
  size_t jk = get_global_id(0);
  __private float4 ar1;
  __private float4 ar2;
  __private float4 br;
  __private float4 cr1 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr2 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr3 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr4 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr5 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr6 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr7 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr8 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  for(k = 0; k < l2; k += mthread_size) {
    size_t tk;
    size_t tjik = (tj + ik) % mthread_size;
    size_t tijk = (ti + jk) % mthread_size;
    as[mthread_size * ti + tjik].x = 0.0f;
    if(i + 0 < n2 && k + tj < l2) {
      as[mthread_size * ti + tjik].x = a[l2 * (i + 0) + k + tj];
    }
    as[mthread_size * ti + tjik].y = 0.0f;
    if(i + 1 < n2 && k + tj < l2) {
      as[mthread_size * ti + tjik].y = a[l2 * (i + 1) + k + tj];
    }
    as[mthread_size * ti + tjik].z = 0.0f;
    if(i + 2 < n2 && k + tj < l2) {
      as[mthread_size * ti + tjik].z = a[l2 * (i + 2) + k + tj];
    }
    as[mthread_size * ti + tjik].w = 0.0f;
    if(i + 3 < n2 && k + tj < l2) {
      as[mthread_size * ti + tjik].w = a[l2 * (i + 3) + k + tj];
    }
    as[mthread_size * (ti + mthread_size) + tjik].x = 0.0f;
    if(i + 4 < n2 && k + tj < l2) {
      as[mthread_size * (ti + mthread_size) + tjik].x = a[l2 * (i + 4) + k + tj];
    }
    as[mthread_size * (ti + mthread_size) + tjik].y = 0.0f;
    if(i + 5 < n2 && k + tj < l2) {
      as[mthread_size * (ti + mthread_size) + tjik].y = a[l2 * (i + 5) + k + tj];
    }
    as[mthread_size * (ti + mthread_size) + tjik].z = 0.0f;
    if(i + 6 < n2 && k + tj < l2) {
      as[mthread_size * (ti + mthread_size) + tjik].z = a[l2 * (i + 6) + k + tj];
    }
    as[mthread_size * (ti + mthread_size) + tjik].w = 0.0f;
    if(i + 7 < n2 && k + tj < l2) {
      as[mthread_size * (ti + mthread_size) + tjik].w = a[l2 * (i + 7) + k + tj];
    }
    bs[mthread_size * tj + tijk].x = 0.0f;
    if(j + 0 < m2 && k + ti < l2) {
      bs[mthread_size * tj + tijk].x = b[l2 * (j + 0) + k + ti];
    }
    bs[mthread_size * tj + tijk].y = 0.0f;
    if(j + 1 < m2 && k + ti < l2) {
      bs[mthread_size * tj + tijk].y = b[l2 * (j + 1) + k + ti];
    }
    bs[mthread_size * tj + tijk].z = 0.0f;
    if(j + 2 < m2 && k + ti < l2) {
      bs[mthread_size * tj + tijk].z = b[l2 * (j + 2) + k + ti];
    }
    bs[mthread_size * tj + tijk].w = 0.0f;
    if(j + 3 < m2 && k + ti < l2) {
      bs[mthread_size * tj + tijk].w = b[l2 * (j + 3) + k + ti];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll 16
    for(tk = 0; tk < mthread_size; tk++) {
      size_t tkik = (tk + ik) % mthread_size;
      size_t tkjk = (tk + jk) % mthread_size;
      ar1 = as[mthread_size * ti + tkik];
      ar2 = as[mthread_size * (ti + mthread_size) + tkik];
      br = bs[mthread_size * tj + tkjk];
      cr1 += ar1.x * br;
      cr2 += ar1.y * br;
      cr3 += ar1.z * br;
      cr4 += ar1.w * br;
      cr5 += ar2.x * br;
      cr6 += ar2.y * br;
      cr7 += ar2.z * br;
      cr8 += ar2.w * br;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = cr1.x;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = cr1.y;
  }
  if(i + 0 < n2 && j + 2 < m2) {
    c[m2 * (i + 0) + j + 2] = cr1.z;
  }
  if(i + 0 < n2 && j + 3 < m2) {
    c[m2 * (i + 0) + j + 3] = cr1.w;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = cr2.x;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = cr2.y;
  }
  if(i + 1 < n2 && j + 2 < m2) {
    c[m2 * (i + 1) + j + 2] = cr2.z;
  }
  if(i + 1 < n2 && j + 3 < m2) {
    c[m2 * (i + 1) + j + 3] = cr2.w;
  }
  if(i + 2 < n2 && j + 0 < m2) {
    c[m2 * (i + 2) + j + 0] = cr3.x;
  }
  if(i + 2 < n2 && j + 1 < m2) {
    c[m2 * (i + 2) + j + 1] = cr3.y;
  }
  if(i + 2 < n2 && j + 2 < m2) {
    c[m2 * (i + 2) + j + 2] = cr3.z;
  }
  if(i + 2 < n2 && j + 3 < m2) {
    c[m2 * (i + 2) + j + 3] = cr3.w;
  }
  if(i + 3 < n2 && j + 0 < m2) {
    c[m2 * (i + 3) + j + 0] = cr4.x;
  }
  if(i + 3 < n2 && j + 1 < m2) {
    c[m2 * (i + 3) + j + 1] = cr4.y;
  }
  if(i + 3 < n2 && j + 2 < m2) {
    c[m2 * (i + 3) + j + 2] = cr4.z;
  }
  if(i + 3 < n2 && j + 3 < m2) {
    c[m2 * (i + 3) + j + 3] = cr4.w;
  }
  if(i + 4 < n2 && j + 0 < m2) {
    c[m2 * (i + 4) + j + 0] = cr5.x;
  }
  if(i + 4 < n2 && j + 1 < m2) {
    c[m2 * (i + 4) + j + 1] = cr5.y;
  }
  if(i + 4 < n2 && j + 2 < m2) {
    c[m2 * (i + 4) + j + 2] = cr5.z;
  }
  if(i + 4 < n2 && j + 3 < m2) {
    c[m2 * (i + 4) + j + 3] = cr5.w;
  }
  if(i + 5 < n2 && j + 0 < m2) {
    c[m2 * (i + 5) + j + 0] = cr6.x;
  }
  if(i + 5 < n2 && j + 1 < m2) {
    c[m2 * (i + 5) + j + 1] = cr6.y;
  }
  if(i + 5 < n2 && j + 2 < m2) {
    c[m2 * (i + 5) + j + 2] = cr6.z;
  }
  if(i + 5 < n2 && j + 3 < m2) {
    c[m2 * (i + 5) + j + 3] = cr6.w;
  }
  if(i + 6 < n2 && j + 0 < m2) {
    c[m2 * (i + 6) + j + 0] = cr7.x;
  }
  if(i + 6 < n2 && j + 1 < m2) {
    c[m2 * (i + 6) + j + 1] = cr7.y;
  }
  if(i + 6 < n2 && j + 2 < m2) {
    c[m2 * (i + 6) + j + 2] = cr7.z;
  }
  if(i + 6 < n2 && j + 3 < m2) {
    c[m2 * (i + 6) + j + 3] = cr7.w;
  }
  if(i + 7 < n2 && j + 0 < m2) {
    c[m2 * (i + 7) + j + 0] = cr8.x;
  }
  if(i + 7 < n2 && j + 1 < m2) {
    c[m2 * (i + 7) + j + 1] = cr8.y;
  }
  if(i + 7 < n2 && j + 2 < m2) {
    c[m2 * (i + 7) + j + 2] = cr8.z;
  }
  if(i + 7 < n2 && j + 3 < m2) {
    c[m2 * (i + 7) + j + 3] = cr8.w;
  }
}

__kernel void mul_at_bt(__global const float *a, __global const float *b, __global float *c, __local float4 *as, __local float4 *bs, ulong n, ulong m, ulong l)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t l2 = (size_t) l;
  size_t i = get_global_id(0) << 3;
  size_t j = get_global_id(1) << 2;
  size_t k;
  size_t mthread_size = get_local_size(0);
  size_t ti = get_local_id(0);
  size_t tj = get_local_id(1);
  size_t ik = get_global_id(0);
  size_t jk = get_global_id(1);
  __private float4 ar1;
  __private float4 ar2;
  __private float4 br;
  __private float4 cr1 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr2 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr3 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr4 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr5 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr6 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr7 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  __private float4 cr8 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
  for(k = 0; k < l2; k += mthread_size) {
    size_t tk;
    size_t tjik = (tj + ik) % mthread_size;
    size_t tijk = (ti + jk) % mthread_size;
    as[mthread_size * ti + tjik].x = 0.0f;
    if(i + 0 < n2 && k + tj < l2) {
      as[mthread_size * ti + tjik].x = a[n2 * (k + tj) + i + 0];
    }
    as[mthread_size * ti + tjik].y = 0.0f;
    if(i + 1 < n2 && k + tj < l2) {
      as[mthread_size * ti + tjik].y = a[n2 * (k + tj) + i + 1];
    }
    as[mthread_size * ti + tjik].z = 0.0f;
    if(i + 2 < n2 && k + tj < l2) {
      as[mthread_size * ti + tjik].z = a[n2 * (k + tj) + i + 2];
    }
    as[mthread_size * ti + tjik].w = 0.0f;
    if(i + 3 < n2 && k + tj < l2) {
      as[mthread_size * ti + tjik].w = a[n2 * (k + tj) + i + 3];
    }
    as[mthread_size * (ti + mthread_size) + tjik].x = 0.0f;
    if(i + 4 < n2 && k + tj < l2) {
      as[mthread_size * (ti + mthread_size) + tjik].x = a[n2 * (k + tj) + i + 4];
    }
    as[mthread_size * (ti + mthread_size) + tjik].y = 0.0f;
    if(i + 5 < n2 && k + tj < l2) {
      as[mthread_size * (ti + mthread_size) + tjik].y = a[n2 * (k + tj) + i + 5];
    }
    as[mthread_size * (ti + mthread_size) + tjik].z = 0.0f;
    if(i + 6 < n2 && k + tj < l2) {
      as[mthread_size * (ti + mthread_size) + tjik].z = a[n2 * (k + tj) + i + 6];
    }
    as[mthread_size * (ti + mthread_size) + tjik].w = 0.0f;
    if(i + 7 < n2 && k + tj < l2) {
      as[mthread_size * (ti + mthread_size) + tjik].w = a[n2 * (k + tj) + i + 7];
    }
    bs[mthread_size * tj + tijk].x = 0.0f;
    if(j + 0 < m2 && k + ti < l2) {
      bs[mthread_size * tj + tijk].x = b[l2 * (j + 0) + k + ti];
    }
    bs[mthread_size * tj + tijk].y = 0.0f;
    if(j + 1 < m2 && k + ti < l2) {
      bs[mthread_size * tj + tijk].y = b[l2 * (j + 1) + k + ti];
    }
    bs[mthread_size * tj + tijk].z = 0.0f;
    if(j + 2 < m2 && k + ti < l2) {
      bs[mthread_size * tj + tijk].z = b[l2 * (j + 2) + k + ti];
    }
    bs[mthread_size * tj + tijk].w = 0.0f;
    if(j + 3 < m2 && k + ti < l2) {
      bs[mthread_size * tj + tijk].w = b[l2 * (j + 3) + k + ti];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll 16
    for(tk = 0; tk < mthread_size; tk++) {
      size_t tkik = (tk + ik) % mthread_size;
      size_t tkjk = (tk + jk) % mthread_size;
      ar1 = as[mthread_size * ti + tkik];
      ar2 = as[mthread_size * (ti + mthread_size) + tkik];
      br = bs[mthread_size * tj + tkjk];
      cr1 += ar1.x * br;
      cr2 += ar1.y * br;
      cr3 += ar1.z * br;
      cr4 += ar1.w * br;
      cr5 += ar2.x * br;
      cr6 += ar2.y * br;
      cr7 += ar2.z * br;
      cr8 += ar2.w * br;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = cr1.x;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = cr1.y;
  }
  if(i + 0 < n2 && j + 2 < m2) {
    c[m2 * (i + 0) + j + 2] = cr1.z;
  }
  if(i + 0 < n2 && j + 3 < m2) {
    c[m2 * (i + 0) + j + 3] = cr1.w;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = cr2.x;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = cr2.y;
  }
  if(i + 1 < n2 && j + 2 < m2) {
    c[m2 * (i + 1) + j + 2] = cr2.z;
  }
  if(i + 1 < n2 && j + 3 < m2) {
    c[m2 * (i + 1) + j + 3] = cr2.w;
  }
  if(i + 2 < n2 && j + 0 < m2) {
    c[m2 * (i + 2) + j + 0] = cr3.x;
  }
  if(i + 2 < n2 && j + 1 < m2) {
    c[m2 * (i + 2) + j + 1] = cr3.y;
  }
  if(i + 2 < n2 && j + 2 < m2) {
    c[m2 * (i + 2) + j + 2] = cr3.z;
  }
  if(i + 2 < n2 && j + 3 < m2) {
    c[m2 * (i + 2) + j + 3] = cr3.w;
  }
  if(i + 3 < n2 && j + 0 < m2) {
    c[m2 * (i + 3) + j + 0] = cr4.x;
  }
  if(i + 3 < n2 && j + 1 < m2) {
    c[m2 * (i + 3) + j + 1] = cr4.y;
  }
  if(i + 3 < n2 && j + 2 < m2) {
    c[m2 * (i + 3) + j + 2] = cr4.z;
  }
  if(i + 3 < n2 && j + 3 < m2) {
    c[m2 * (i + 3) + j + 3] = cr4.w;
  }
  if(i + 4 < n2 && j + 0 < m2) {
    c[m2 * (i + 4) + j + 0] = cr5.x;
  }
  if(i + 4 < n2 && j + 1 < m2) {
    c[m2 * (i + 4) + j + 1] = cr5.y;
  }
  if(i + 4 < n2 && j + 2 < m2) {
    c[m2 * (i + 4) + j + 2] = cr5.z;
  }
  if(i + 4 < n2 && j + 3 < m2) {
    c[m2 * (i + 4) + j + 3] = cr5.w;
  }
  if(i + 5 < n2 && j + 0 < m2) {
    c[m2 * (i + 5) + j + 0] = cr6.x;
  }
  if(i + 5 < n2 && j + 1 < m2) {
    c[m2 * (i + 5) + j + 1] = cr6.y;
  }
  if(i + 5 < n2 && j + 2 < m2) {
    c[m2 * (i + 5) + j + 2] = cr6.z;
  }
  if(i + 5 < n2 && j + 3 < m2) {
    c[m2 * (i + 5) + j + 3] = cr6.w;
  }
  if(i + 6 < n2 && j + 0 < m2) {
    c[m2 * (i + 6) + j + 0] = cr7.x;
  }
  if(i + 6 < n2 && j + 1 < m2) {
    c[m2 * (i + 6) + j + 1] = cr7.y;
  }
  if(i + 6 < n2 && j + 2 < m2) {
    c[m2 * (i + 6) + j + 2] = cr7.z;
  }
  if(i + 6 < n2 && j + 3 < m2) {
    c[m2 * (i + 6) + j + 3] = cr7.w;
  }
  if(i + 7 < n2 && j + 0 < m2) {
    c[m2 * (i + 7) + j + 0] = cr8.x;
  }
  if(i + 7 < n2 && j + 1 < m2) {
    c[m2 * (i + 7) + j + 1] = cr8.y;
  }
  if(i + 7 < n2 && j + 2 < m2) {
    c[m2 * (i + 7) + j + 2] = cr8.z;
  }
  if(i + 7 < n2 && j + 3 < m2) {
    c[m2 * (i + 7) + j + 3] = cr8.w;
  }
}

__kernel void mul_a_b_for_elems(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[m2 * (i + 0) + j + 0] * b[m2 * (i + 0) + j + 0];
  }
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 1] = a[m2 * (i + 0) + j + 1] * b[m2 * (i + 0) + j + 1];
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[m2 * (i + 1) + j + 0] * b[m2 * (i + 1) + j + 0];
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[m2 * (i + 1) + j + 1] * b[m2 * (i + 1) + j + 1];
  }
}

__kernel void mul_at_b_for_elems(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[n2 * (j + 0) + i + 0] * b[m2 * (i + 0) + j + 0];
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[n2 * (j + 1) + i + 0] * b[m2 * (i + 0) + j + 1];
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[n2 * (j + 0) + i + 1] * b[m2 * (i + 1) + j + 0];
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[n2 * (j + 1) + i + 1] * b[m2 * (i + 1) + j + 1];
  }
}

__kernel void mul_a_bt_for_elems(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[m2 * (i + 0) + j + 0] * b[n2 * (j + 0) + i + 0];
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[m2 * (i + 0) + j + 1] * b[n2 * (j + 1) + i + 0];
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[m2 * (i + 1) + j + 0] * b[n2 * (j + 0) + i + 1];
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[m2 * (i + 1) + j + 1] * b[n2 * (j + 1) + i + 1];
  }
}

__kernel void mul_at_bt_for_elems(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[n2 * (j + 0) + i + 0] * b[n2 * (j + 0) + i + 0];
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[n2 * (j + 1) + i + 0] * b[n2 * (j + 1) + i + 0];
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[n2 * (j + 0) + i + 1] * b[n2 * (j + 0) + i + 1];
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[n2 * (j + 1) + i + 1] * b[n2 * (j + 1) + i + 1];
  }
}

__kernel void div_a_b_for_elems(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[m2 * (i + 0) + j + 0] / b[m2 * (i + 0) + j + 0];
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[m2 * (i + 0) + j + 1] / b[m2 * (i + 0) + j + 1];
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[m2 * (i + 1) + j + 0] / b[m2 * (i + 1) + j + 0];
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[m2 * (i + 1) + j + 1] / b[m2 * (i + 1) + j + 1];
  }
}

__kernel void div_at_b_for_elems(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[n2 * (j + 0) + i + 0] / b[m2 * (i + 0) + j + 0];
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[n2 * (j + 1) + i + 0] / b[m2 * (i + 0) + j + 1];
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[n2 * (j + 0) + i + 1] / b[m2 * (i + 1) + j + 0];
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[n2 * (j + 1) + i + 1] / b[m2 * (i + 1) + j + 1];
  }
}

__kernel void div_a_bt_for_elems(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[m2 * (i + 0) + j + 0] / b[n2 * (j + 0) + i + 0];
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[m2 * (i + 0) + j + 1] / b[n2 * (j + 1) + i + 0];
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[m2 * (i + 1) + j + 0] / b[n2 * (j + 0) + i + 1];
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[m2 * (i + 1) + j + 1] / b[n2 * (j + 1) + i + 1];
  }
}

__kernel void div_at_bt_for_elems(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[n2 * (j + 0) + i + 0] / b[n2 * (j + 0) + i + 0];
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[n2 * (j + 1) + i + 0] / b[n2 * (j + 1) + i + 0];
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[n2 * (j + 0) + i + 1] / b[n2 * (j + 0) + i + 1];
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[n2 * (j + 1) + i + 1] / b[n2 * (j + 1) + i + 1];
  }
}

__kernel void add_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[m2 * (i + 0) + j + 0] + b;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[m2 * (i + 0) + j + 1] + b;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[m2 * (i + 1) + j + 0] + b;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[m2 * (i + 1) + j + 1] + b;
  }
}

__kernel void add_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[n2 * (j + 0) + i + 0] + b;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[n2 * (j + 1) + i + 0] + b;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[n2 * (j + 0) + i + 1] + b;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[n2 * (j + 1) + i + 1] + b;
  }
}

__kernel void sub_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[m2 * (i + 0) + j + 0] - b;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[m2 * (i + 0) + j + 1] - b;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[m2 * (i + 1) + j + 0] - b;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[m2 * (i + 1) + j + 1] - b;
  }
}

__kernel void sub_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[n2 * (j + 0) + i + 0] - b;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[n2 * (j + 1) + i + 0] - b;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[n2 * (j + 0) + i + 1] - b;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[n2 * (j + 1) + i + 1] - b;
  }
}

__kernel void rsub_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = b - a[m2 * (i + 0) + j + 0];
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = b - a[m2 * (i + 0) + j + 1];
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = b - a[m2 * (i + 1) + j + 0];
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = b - a[m2 * (i + 1) + j + 1];
  }
}

__kernel void rsub_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = b - a[n2 * (j + 0) + i + 0];
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = b - a[n2 * (j + 1) + i + 0];
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = b - a[n2 * (j + 0) + i + 1];
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = b - a[n2 * (j + 1) + i + 1];
  }
}

__kernel void mul_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[m2 * (i + 0) + j + 0] * b;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[m2 * (i + 0) + j + 1] * b;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[m2 * (i + 1) + j + 0] * b;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[m2 * (i + 1) + j + 1] * b;
  }
}

__kernel void mul_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[n2 * (j + 0) + i + 0] * b;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[n2 * (j + 1) + i + 0] * b;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[n2 * (j + 0) + i + 1] * b;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[n2 * (j + 1) + i + 1] * b;
  }
}

__kernel void div_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[m2 * (i + 0) + j + 0] / b;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[m2 * (i + 0) + j + 1] / b;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[m2 * (i + 1) + j + 0] / b;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[m2 * (i + 1) + j + 1] / b;
  }
}

__kernel void div_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = a[n2 * (j + 0) + i + 0] / b;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = a[n2 * (j + 1) + i + 0] / b;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = a[n2 * (j + 0) + i + 1] / b;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = a[n2 * (j + 1) + i + 1] / b;
  }
}

__kernel void rdiv_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0  < m2) {
    c[m2 * (i + 0) + j + 0] = b / a[m2 * (i + 0) + j + 0];
  }
  if(i + 0 < n2 && j + 1  < m2) {
    c[m2 * (i + 0) + j + 1] = b / a[m2 * (i + 0) + j + 1];
  }
  if(i + 1 < n2 && j + 0  < m2) {
    c[m2 * (i + 1) + j + 0] = b / a[m2 * (i + 1) + j + 0];
  }
  if(i + 1 < n2 && j + 1  < m2) {
    c[m2 * (i + 1) + j + 1] = b / a[m2 * (i + 1) + j + 1];
  }
}

__kernel void rdiv_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0  < m2) {
    c[m2 * (i + 0) + j + 0] = b / a[n2 * (j + 0) + i + 0];
  }
  if(i + 0 < n2 && j + 1  < m2) {
    c[m2 * (i + 0) + j + 1] = b / a[n2 * (j + 1) + i + 0];
  }
  if(i + 1 < n2 && j + 0  < m2) {
    c[m2 * (i + 1) + j + 0] = b / a[n2 * (j + 0) + i + 1];
  }
  if(i + 1 < n2 && j + 1  < m2) {
    c[m2 * (i + 1) + j + 1] = b / a[n2 * (j + 1) + i + 1];
  }
}

__kernel void sigmoid_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = 1.0f / (1.0f + exp(-a[m2 * (i + 0) + j + 0]));
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = 1.0f / (1.0f + exp(-a[m2 * (i + 0) + j + 1]));
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = 1.0f / (1.0f + exp(-a[m2 * (i + 1) + j + 0]));
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = 1.0f / (1.0f + exp(-a[m2 * (i + 1) + j + 1]));
  }
}

__kernel void sigmoid_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = 1.0f / (1.0f + exp(-a[n2 * (j + 0) + i + 0]));
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = 1.0f / (1.0f + exp(-a[n2 * (j + 1) + i + 0]));
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = 1.0f / (1.0f + exp(-a[n2 * (j + 0) + i + 1]));
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = 1.0f / (1.0f + exp(-a[n2 * (j + 1) + i + 1]));
  }
}

__kernel void tanh_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = tanh(a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = tanh(a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = tanh(a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = tanh(a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void tanh_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = tanh(a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = tanh(a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = tanh(a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = tanh(a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void swish_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = a[m2 * (i + 0) + j + 0] / (1.0f + exp(-a[m2 * (i + 0) + j + 0]));
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = a[m2 * (i + 0) + j + 1] / (1.0f + exp(-a[m2 * (i + 0) + j + 1]));
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = a[m2 * (i + 1) + j + 0] / (1.0f + exp(-a[m2 * (i + 1) + j + 0]));
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = a[m2 * (i + 1) + j + 1] / (1.0f + exp(-a[m2 * (i + 1) + j + 1]));
  }
}

__kernel void swish_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = a[n2 * (j + 0) + i + 0] / (1.0f + exp(-a[n2 * (j + 0) + i + 0]));
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = a[n2 * (j + 1) + i + 0] / (1.0f + exp(-a[n2 * (j + 1) + i + 0]));
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = a[n2 * (j + 0) + i + 1] / (1.0f + exp(-a[n2 * (j + 0) + i + 1]));
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = a[n2 * (j + 1) + i + 1] / (1.0f + exp(-a[n2 * (j + 1) + i + 1]));
  }
}

__kernel void softmax_a(__global const float *a, __global float *b, __local float4 *es, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  size_t k;
  size_t thread_width = get_local_size(0);
  size_t thread_height = get_local_size(1);
  size_t tile_height = thread_height << 1;
  size_t ti = get_local_id(1);
  size_t tj = get_local_id(0);
  size_t bi = ti << 1;
  __private float2 sum = (float2) (0.0f, 0.0f);
  for(k = 0; k < n2; k += tile_height) {
    size_t tk;
    es[thread_width * ti + tj].x = 0.0f;
    if(j + 0 < m2 && k + bi + 0 < n2) {
      es[thread_width * ti + tj].x = exp(a[m2 * (k + bi + 0) + j + 0]);
    }
    es[thread_width * ti + tj].y = 0.0f;
    if(j + 1 < m2 && k + bi + 0 < n2) {
      es[thread_width * ti + tj].y = exp(a[m2 * (k + bi + 0) + j + 1]);
    }
    es[thread_width * ti + tj].z = 0.0f;
    if(j + 0 < m2 && k + bi + 1 < n2) {
      es[thread_width * ti + tj].z = exp(a[m2 * (k + bi + 1) + j + 0]);
    }
    es[thread_width * ti + tj].w = 0.0f;
    if(j + 1 < m2 && k + bi + 1 < n2) {
      es[thread_width * ti + tj].w = exp(a[m2 * (k + bi + 1) + j + 1]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll 32
    for(tk = 0; tk < thread_height; tk++) {
      __private float4 e = es[thread_width * tk + tj];
      sum += e.xy;
      sum += e.zw;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = exp(a[m2 * (i + 0) + j + 0]) / sum.x;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = exp(a[m2 * (i + 0) + j + 1]) / sum.y;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = exp(a[m2 * (i + 1) + j + 0]) / sum.x;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = exp(a[m2 * (i + 1) + j + 1]) / sum.y;
  }
}

__kernel void softmax_at(__global const float *a, __global float *b, __local float4 *es, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
  size_t k;
  size_t thread_width = get_local_size(1);
  size_t thread_height = get_local_size(0);
  size_t tile_height = thread_height << 1;
  size_t ti = get_local_id(0);
  size_t tj = get_local_id(1);
  size_t bi = ti << 1;
  __private float2 sum = (float2) (0.0f, 0.0f);
  for(k = 0; k < n2; k += tile_height) {
    size_t tk;
    es[thread_width * ti + tj].x = 0.0f;
    if(j + 0 < m2 && k + bi + 0 < n2) {
      es[thread_width * ti + tj].x = exp(a[n2 * (j + 0) + k + bi + 0]);
    }
    es[thread_width * ti + tj].y = 0.0f;
    if(j + 1 < m2 && k + bi + 0 < n2) {
      es[thread_width * ti + tj].y = exp(a[n2 * (j + 1) + k + bi + 0]);
    }
    es[thread_width * ti + tj].z = 0.0f;
    if(j + 0 < m2 && k + bi + 1 < n2) {
      es[thread_width * ti + tj].z = exp(a[n2 * (j + 0) + k + bi + 1]);
    }
    es[thread_width * ti + tj].w = 0.0f;
    if(j + 1 < m2 && k + bi + 1 < n2) {
      es[thread_width * ti + tj].w = exp(a[n2 * (j + 1) + k + bi + 1]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll 32
    for(tk = 0; tk < thread_height; tk++) {
      __private float4 e = es[thread_width * tk + tj];
      sum += e.xy;
      sum += e.zw;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = exp(a[n2 * (j + 0) + i + 0]) / sum.x;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = exp(a[n2 * (j + 1) + i + 0]) / sum.y;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = exp(a[n2 * (j + 0) + i + 1]) / sum.x;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = exp(a[n2 * (j + 1) + i + 1]) / sum.y;
  }
}

__kernel void sqrt_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = sqrt(a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = sqrt(a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = sqrt(a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = sqrt(a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void sqrt_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = sqrt(a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = sqrt(a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = sqrt(a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = sqrt(a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void repeat_col_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = a[i + 0];
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = a[i + 0];
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = a[i + 1];
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = a[i + 1];
  }
}

__kernel void repeat_row_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = a[j + 0];
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = a[j + 1];
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = a[j + 0];
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = a[j + 1];
  }
}

__kernel void abs_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = fabs(a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = fabs(a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = fabs(a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = fabs(a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void abs_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = fabs(a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = fabs(a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = fabs(a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = fabs(a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void pow_a_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = pow(a[m2 * (i + 0) + j + 0], b[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = pow(a[m2 * (i + 0) + j + 1], b[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = pow(a[m2 * (i + 1) + j + 0], b[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = pow(a[m2 * (i + 1) + j + 1], b[m2 * (i + 1) + j + 1]);
  }
}

__kernel void pow_at_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = pow(a[n2 * (j + 0) + i + 0], b[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = pow(a[n2 * (j + 1) + i + 0], b[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = pow(a[n2 * (j + 0) + i + 1], b[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = pow(a[n2 * (j + 1) + i + 1], b[m2 * (i + 1) + j + 1]);
  }
}

__kernel void pow_a_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = pow(a[m2 * (i + 0) + j + 0], b[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = pow(a[m2 * (i + 0) + j + 1], b[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = pow(a[m2 * (i + 1) + j + 0], b[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = pow(a[m2 * (i + 1) + j + 1], b[n2 * (j + 1) + i + 1]);
  }
}

__kernel void pow_at_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = pow(a[n2 * (j + 0) + i + 0], b[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = pow(a[n2 * (j + 1) + i + 0], b[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = pow(a[n2 * (j + 0) + i + 1], b[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = pow(a[n2 * (j + 1) + i + 1], b[n2 * (j + 1) + i + 1]);
  }
}

__kernel void pow_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = pow(a[m2 * (i + 0) + j + 0], b);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = pow(a[m2 * (i + 0) + j + 1], b);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = pow(a[m2 * (i + 1) + j + 0], b);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = pow(a[m2 * (i + 1) + j + 1], b);
  }
}

__kernel void pow_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = pow(a[n2 * (j + 0) + i + 0], b);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = pow(a[n2 * (j + 1) + i + 0], b);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = pow(a[n2 * (j + 0) + i + 1], b);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = pow(a[n2 * (j + 1) + i + 1], b);
  }
}

__kernel void rpow_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = pow(b, a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = pow(b, a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = pow(b, a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = pow(b, a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void rpow_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = pow(b, a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = pow(b, a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = pow(b, a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = pow(b, a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void exp_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = exp(a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = exp(a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = exp(a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = exp(a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void exp_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = exp(a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = exp(a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = exp(a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = exp(a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void ln_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = log(a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = log(a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = log(a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = log(a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void ln_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = log(a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = log(a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = log(a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = log(a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void log2_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = log2(a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = log2(a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = log2(a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = log2(a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void log2_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = log2(a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = log2(a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = log2(a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = log2(a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void log10_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = log10(a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = log10(a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = log10(a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = log10(a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void log10_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = log10(a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = log10(a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = log10(a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = log10(a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void sin_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = sin(a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = sin(a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = sin(a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = sin(a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void sin_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = sin(a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = sin(a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = sin(a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = sin(a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void cos_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = cos(a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = cos(a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = cos(a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = cos(a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void cos_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = cos(a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = cos(a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = cos(a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = cos(a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void tan_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = tan(a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = tan(a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = tan(a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = tan(a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void tan_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = tan(a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = tan(a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = tan(a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = tan(a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void asin_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = asin(a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = asin(a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = asin(a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = asin(a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void asin_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = asin(a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = asin(a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = asin(a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = asin(a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void acos_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = acos(a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = acos(a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = acos(a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = acos(a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void acos_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = acos(a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = acos(a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = acos(a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = acos(a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void atan_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = atan(a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = atan(a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = atan(a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = atan(a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void atan_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = atan(a[n2 * (j + 0)  + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = atan(a[n2 * (j + 1)  + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = atan(a[n2 * (j + 0)  + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = atan(a[n2 * (j + 1)  + i + 1]);
  }
}

__kernel void atan2_a_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = atan2(a[m2 * (i + 0) + j + 0], b[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = atan2(a[m2 * (i + 0) + j + 1], b[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = atan2(a[m2 * (i + 1) + j + 0], b[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = atan2(a[m2 * (i + 1) + j + 1], b[m2 * (i + 1) + j + 1]);
  }
}

__kernel void atan2_at_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = atan2(a[n2 * (j + 0) + i + 0], b[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = atan2(a[n2 * (j + 1) + i + 0], b[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = atan2(a[n2 * (j + 0) + i + 1], b[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = atan2(a[n2 * (j + 1) + i + 1], b[m2 * (i + 1) + j + 1]);
  }
}

__kernel void atan2_a_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = atan2(a[m2 * (i + 0) + j + 0], b[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = atan2(a[m2 * (i + 0) + j + 1], b[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = atan2(a[m2 * (i + 1) + j + 0], b[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = atan2(a[m2 * (i + 1) + j + 1], b[n2 * (j + 1) + i + 1]);
  }
}

__kernel void atan2_at_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = atan2(a[n2 * (j + 0) + i + 0], b[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = atan2(a[n2 * (j + 1) + i + 0], b[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = atan2(a[n2 * (j + 0) + i + 1], b[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = atan2(a[n2 * (j + 1) + i + 1], b[n2 * (j + 1) + i + 1]);
  }
}

__kernel void atan2_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = atan2(a[m2 * (i + 0) + j + 0], b);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = atan2(a[m2 * (i + 0) + j + 1], b);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = atan2(a[m2 * (i + 1) + j + 0], b);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = atan2(a[m2 * (i + 1) + j + 1], b);
  }
}

__kernel void atan2_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = atan2(a[n2 * (j + 0) + i + 0], b);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = atan2(a[n2 * (j + 1) + i + 0], b);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = atan2(a[n2 * (j + 0) + i + 1], b);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = atan2(a[n2 * (j + 1) + i + 1], b);
  }
}

__kernel void ratan2_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = atan2(b, a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = atan2(b, a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = atan2(b, a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = atan2(b, a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void ratan2_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = atan2(b, a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = atan2(b, a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = atan2(b, a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = atan2(b, a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void sinh_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = sinh(a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = sinh(a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = sinh(a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = sinh(a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void sinh_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = sinh(a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = sinh(a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = sinh(a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = sinh(a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void cosh_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = cosh(a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = cosh(a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = cosh(a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = cosh(a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void cosh_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = cosh(a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = cosh(a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = cosh(a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = cosh(a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void asinh_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = asinh(a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = asinh(a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = asinh(a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = asinh(a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void asinh_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = asinh(a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = asinh(a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = asinh(a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = asinh(a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void acosh_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = acosh(a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = acosh(a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = acosh(a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = acosh(a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void acosh_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = acosh(a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = acosh(a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = acosh(a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = acosh(a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void atanh_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = atanh(a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = atanh(a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = atanh(a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = atanh(a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void atanh_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = atanh(a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = atanh(a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = atanh(a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = atanh(a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void signum_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    if(!isnan(a[m2 * (i + 0) + j + 0])) {
      b[m2 * (i + 0) + j + 0] = (signbit(a[m2 * (i + 0) + j + 0]) ? -1.0 : 1.0);
    } else {
      b[m2 * (i + 0) + j + 0] = a[m2 * (i + 0) + j + 0];
    }
  }
  if(i + 0 < n2 && j + 1 < m2) {
    if(!isnan(a[m2 * (i + 0) + j + 1])) {
      b[m2 * (i + 0) + j + 1] = (signbit(a[m2 * (i + 0) + j + 1]) ? -1.0 : 1.0);
    } else {
      b[m2 * (i + 0) + j + 1] = a[m2 * (i + 0) + j + 1];
    }
  }
  if(i + 1 < n2 && j + 0 < m2) {
    if(!isnan(a[m2 * (i + 1) + j + 0])) {
      b[m2 * (i + 1) + j + 0] = (signbit(a[m2 * (i + 1) + j + 0]) ? -1.0 : 1.0);
    } else {
      b[m2 * (i + 1) + j + 0] = a[m2 * (i + 1) + j + 0];
    }
  }
  if(i + 1 < n2 && j + 1 < m2) {
    if(!isnan(a[m2 * (i + 1) + j + 1])) {
      b[m2 * (i + 1) + j + 1] = (signbit(a[m2 * (i + 1) + j + 1]) ? -1.0 : 1.0);
    } else {
      b[m2 * (i + 1) + j + 1] = a[m2 * (i + 1) + j + 1];
    }
  }
}

__kernel void signum_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    if(!isnan(a[n2 * (j + 0) + i + 0])) {
      b[m2 * (i + 0) + j + 0] = (signbit(a[n2 * (j + 0) + i + 0]) ? -1.0 : 1.0);
    } else {
      b[m2 * (i + 0) + j + 0] = a[n2 * (j + 0) + i + 0];
    }
  }
  if(i + 0 < n2 && j + 1 < m2) {
    if(!isnan(a[n2 * (j + 1) + i + 0])) {
      b[m2 * (i + 0) + j + 1] = (signbit(a[n2 * (j + 1) + i + 0]) ? -1.0 : 1.0);
    } else {
      b[m2 * (i + 0) + j + 1] = a[n2 * (j + 1) + i + 0];
    }
  }
  if(i + 1 < n2 && j + 0 < m2) {
    if(!isnan(a[n2 * (j + 0) + i + 1])) {
      b[m2 * (i + 1) + j + 0] = (signbit(a[n2 * (j + 0) + i + 1]) ? -1.0 : 1.0);
    } else {
      b[m2 * (i + 1) + j + 0] = a[n2 * (j + 0) + i + 1];
    }
  }
  if(i + 1 < n2 && j + 1 < m2) {
    if(!isnan(a[n2 * (j + 1) + i + 1])) {
      b[m2 * (i + 1) + j + 1] = (signbit(a[n2 * (j + 1) + i + 1]) ? -1.0 : 1.0);
    } else {
      b[m2 * (i + 1) + j + 1] = a[n2 * (j + 1) + i + 1];
    }
  }
}

__kernel void ceil_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = ceil(a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = ceil(a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = ceil(a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = ceil(a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void ceil_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = ceil(a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = ceil(a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = ceil(a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = ceil(a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void floor_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = floor(a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = floor(a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = floor(a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = floor(a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void floor_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = floor(a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = floor(a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = floor(a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = floor(a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void round_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = round(a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = round(a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = round(a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = round(a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void round_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = round(a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = round(a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = round(a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = round(a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void trunc_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = trunc(a[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = trunc(a[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = trunc(a[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = trunc(a[m2 * (i + 1) + j + 1]);
  }
}

__kernel void trunc_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = trunc(a[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = trunc(a[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = trunc(a[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = trunc(a[n2 * (j + 1) + i + 1]);
  }
}

__kernel void max_a_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = fmax(a[m2 * (i + 0) + j + 0], b[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = fmax(a[m2 * (i + 0) + j + 1], b[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = fmax(a[m2 * (i + 1) + j + 0], b[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = fmax(a[m2 * (i + 1) + j + 1], b[m2 * (i + 1) + j + 1]);
  }
}

__kernel void max_at_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = fmax(a[n2 * (j + 0) + i + 0], b[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = fmax(a[n2 * (j + 1) + i + 0], b[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = fmax(a[n2 * (j + 0) + i + 1], b[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = fmax(a[n2 * (j + 1) + i + 1], b[m2 * (i + 1) + j + 1]);
  }
}

__kernel void max_a_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = fmax(a[m2 * (i + 0) + j + 0], b[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = fmax(a[m2 * (i + 0) + j + 1], b[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = fmax(a[m2 * (i + 1) + j + 0], b[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = fmax(a[m2 * (i + 1) + j + 1], b[n2 * (j + 1) + i + 1]);
  }
}

__kernel void max_at_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = fmax(a[n2 * (j + 0) + i + 0], b[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = fmax(a[n2 * (j + 1) + i + 0], b[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = fmax(a[n2 * (j + 0) + i + 1], b[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = fmax(a[n2 * (j + 1) + i + 1], b[n2 * (j + 1) + i + 1]);
  }
}

__kernel void max_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = fmax(a[m2 * (i + 0) + j + 0], b);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = fmax(a[m2 * (i + 0) + j + 1], b);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = fmax(a[m2 * (i + 1) + j + 0], b);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = fmax(a[m2 * (i + 1) + j + 1], b);
  }
}

__kernel void max_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = fmax(a[n2 * (j + 0) + i + 0], b);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = fmax(a[n2 * (j + 1) + i + 0], b);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = fmax(a[n2 * (j + 0) + i + 1], b);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = fmax(a[n2 * (j + 1) + i + 1], b);
  }
}

__kernel void min_a_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = fmin(a[m2 * (i + 0) + j + 0], b[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = fmin(a[m2 * (i + 0) + j + 1], b[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = fmin(a[m2 * (i + 1) + j + 0], b[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = fmin(a[m2 * (i + 1) + j + 1], b[m2 * (i + 1) + j + 1]);
  }
}

__kernel void min_at_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = fmin(a[n2 * (j + 0) + i + 0], b[m2 * (i + 0) + j + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = fmin(a[n2 * (j + 1) + i + 0], b[m2 * (i + 0) + j + 1]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = fmin(a[n2 * (j + 0) + i + 1], b[m2 * (i + 1) + j + 0]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = fmin(a[n2 * (j + 1) + i + 1], b[m2 * (i + 1) + j + 1]);
  }
}

__kernel void min_a_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = fmin(a[m2 * (i + 0) + j + 0], b[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = fmin(a[m2 * (i + 0) + j + 1], b[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = fmin(a[m2 * (i + 1) + j + 0], b[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = fmin(a[m2 * (i + 1) + j + 1], b[n2 * (j + 1) + i + 1]);
  }
}

__kernel void min_at_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = fmin(a[n2 * (j + 0) + i + 0], b[n2 * (j + 0) + i + 0]);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = fmin(a[n2 * (j + 1) + i + 0], b[n2 * (j + 1) + i + 0]);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = fmin(a[n2 * (j + 0) + i + 1], b[n2 * (j + 0) + i + 1]);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = fmin(a[n2 * (j + 1) + i + 1], b[n2 * (j + 1) + i + 1]);
  }
}

__kernel void min_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = fmin(a[m2 * (i + 0) + j + 0], b);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = fmin(a[m2 * (i + 0) + j + 1], b);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = fmin(a[m2 * (i + 1) + j + 0], b);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = fmin(a[m2 * (i + 1) + j + 1], b);
  }
}

__kernel void min_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(1) << 1;
  size_t j = get_global_id(0) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = fmin(a[n2 * (j + 0) + i + 0], b);
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = fmin(a[n2 * (j + 1) + i + 0], b);
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = fmin(a[n2 * (j + 0) + i + 1], b);
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = fmin(a[n2 * (j + 1) + i + 1], b);
  }
}
