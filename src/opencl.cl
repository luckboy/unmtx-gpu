/*
 * Copyright (c) 2025 Łukasz Szpakowski
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
__kernel void transpose_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = a[n2 * j + i];
  }
}

__kernel void add_a_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[m2 * i + j] + b[m2 * i + j];
  }
}

__kernel void add_at_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[n2 * j + i] + b[m2 * i + j];
  }
}

__kernel void add_a_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[m2 * i + j] + b[n2 * j + i];
  }
}

__kernel void add_at_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[n2 * j + i] + b[n2 * j + i];
  }
}

__kernel void sub_a_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[m2 * i + j] - b[m2 * i + j];
  }
}

__kernel void sub_at_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[n2 * j + i] - b[m2 * i + j];
  }
}

__kernel void sub_a_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[m2 * i + j] - b[n2 * j + i];
  }
}

__kernel void sub_at_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[n2 * j + i] - b[n2 * j + i];
  }
}

__kernel void mul_a_b(__global const float *a, __global const float *b, __global float *c, __local float *as, __local float *bs, ulong n, ulong m, ulong l)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t l2 = (size_t) l;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  size_t k;
  size_t tile_width = get_local_size(0);
  size_t ti = get_local_id(0);
  size_t tj = get_local_id(1);
  size_t l2_i = l2 * i;
  size_t tile_width_ti = tile_width * ti;
  size_t tile_width_tj = tile_width * tj;
  size_t tile_width_3 = tile_width & ~((size_t) 3);
  float cij = 0.0f;
  for(k = 0; k < l2; k += tile_width) {
    size_t k_ti = k + ti;
    size_t k_tj = k + tj;
    size_t tk;
    as[tile_width_ti + tj] = 0.0f;
    if(i < n2 && k_tj < l2) {
      as[tile_width_ti + tj] = a[l2_i + k_tj];
    }
    bs[tile_width_ti + tj] = 0.0f;
    if(j < m2 && k_ti < l2) {
      bs[tile_width_ti + tj] = b[m2 * k_ti + j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(tk = 0; tk < tile_width_3; tk += 4) {
      __private float4 av;
      __private float4 bv;
      av.x = as[tile_width_ti + tk + 0];
      av.y = as[tile_width_ti + tk + 1];
      av.z = as[tile_width_ti + tk + 2];
      av.w = as[tile_width_ti + tk + 3];
      bv.x = bs[tile_width * (tk + 0) + tj];
      bv.y = bs[tile_width * (tk + 1) + tj];
      bv.z = bs[tile_width * (tk + 2) + tj];
      bv.w = bs[tile_width * (tk + 3) + tj];
      cij += dot(av, bv);
    }
    for(; tk < tile_width; tk++) {
      cij += a[tile_width_ti + tk] * b[tile_width * tk + tj];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i < n2 && j < m2) {
    c[m2 * i + j] = cij;
  }
}

__kernel void mul_at_b(__global const float *a, __global const float *b, __global float *c, __local float *as, __local float *bs, ulong n, ulong m, ulong l)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t l2 = (size_t) l;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  size_t k;
  size_t tile_width = get_local_size(0);
  size_t ti = get_local_id(0);
  size_t tj = get_local_id(1);
  size_t tile_width_ti = tile_width * ti;
  size_t tile_width_tj = tile_width * tj;
  size_t tile_width_3 = tile_width & ~((size_t) 3);
  size_t tk;
  float cij = 0.0f;
  for(k = 0; k < l2; k += tile_width) {
    size_t k_ti = k + ti;
    size_t k_tj = k + tj;
    as[tile_width_ti + tj] = 0.0f;
    if(i < n2 && k_tj < l2) {
      as[tile_width_ti + tj] = a[n2 * k_tj + i];
    }
    bs[tile_width_ti + tj] = 0.0f;
    if(j < m2 && k_ti < l2) {
      bs[tile_width_ti + tj] = b[m2 * k_ti + j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(tk = 0; tk < tile_width_3; tk += 4) {
      __private float4 av;
      __private float4 bv;
      av.x = as[tile_width_ti + tk + 0];
      av.y = as[tile_width_ti + tk + 1];
      av.z = as[tile_width_ti + tk + 2];
      av.w = as[tile_width_ti + tk + 3];
      bv.x = bs[tile_width * (tk + 0) + tj];
      bv.y = bs[tile_width * (tk + 1) + tj];
      bv.z = bs[tile_width * (tk + 2) + tj];
      bv.w = bs[tile_width * (tk + 3) + tj];
      cij += dot(av, bv);
    }
    for(; tk < tile_width; tk++) {
      cij += a[tile_width_ti + tk] * b[tile_width * tk + tj];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i < n2 && j < m2) {
    c[m2 * i + j] = cij;
  }
}

__kernel void mul_a_bt(__global const float *a, __global const float *b, __global float *c, __local float *as, __local float *bs, ulong n, ulong m, ulong l)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t l2 = (size_t) l;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  size_t k;
  size_t tile_width = get_local_size(0);
  size_t ti = get_local_id(0);
  size_t tj = get_local_id(1);
  size_t l2_i = l2 * i;
  size_t l2_j = l2 * j;
  size_t tile_width_ti = tile_width * ti;
  size_t tile_width_tj = tile_width * tj;
  size_t tile_width_3 = tile_width & ~((size_t) 3);
  float cij = 0.0f;
  for(k = 0; k < l2; k += tile_width) {
    size_t k_ti = k + ti;
    size_t k_tj = k + tj;
    size_t tk;
    as[tile_width_ti + tj] = 0.0f;
    if(i < n2 && k_tj < l2) {
      as[tile_width_ti + tj] = a[l2_i + k_tj];
    }
    bs[tile_width_ti + tj] = 0.0f;
    if(j < m2 && k_ti < l2) {
      bs[tile_width_ti + tj] = b[l2_j + k_ti];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(tk = 0; tk < tile_width_3; tk += 4) {
      __private float4 av;
      __private float4 bv;
      av.x = as[tile_width_ti + tk + 0];
      av.y = as[tile_width_ti + tk + 1];
      av.z = as[tile_width_ti + tk + 2];
      av.w = as[tile_width_ti + tk + 3];
      bv.x = bs[tile_width * (tk + 0) + tj];
      bv.y = bs[tile_width * (tk + 1) + tj];
      bv.z = bs[tile_width * (tk + 2) + tj];
      bv.w = bs[tile_width * (tk + 3) + tj];
      cij += dot(av, bv);
    }
    for(; tk < tile_width; tk++) {
      cij += a[tile_width_ti + tk] * b[tile_width * tk + tj];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i < n2 && j < m2) {
    c[m2 * i + j] = cij;
  }
}

__kernel void mul_at_bt(__global const float *a, __global const float *b, __global float *c, __local float *as, __local float *bs, ulong n, ulong m, ulong l)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t l2 = (size_t) l;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  size_t k;
  size_t tile_width = get_local_size(0);
  size_t ti = get_local_id(0);
  size_t tj = get_local_id(1);
  size_t l2_j = l2 * j;
  size_t tile_width_ti = tile_width * ti;
  size_t tile_width_tj = tile_width * tj;
  size_t tile_width_3 = tile_width & ~((size_t) 3);
  float cij = 0.0f;
  for(k = 0; k < l2; k += tile_width) {
    size_t k_ti = k + ti;
    size_t k_tj = k + tj;
    size_t tk;
    as[tile_width_ti + tj] = 0.0f;
    if(i < n2 && k_tj < l2) {
      as[tile_width_ti + tj] = a[n2 * k_tj + i];
    }
    bs[tile_width_ti + tj] = 0.0f;
    if(j < m2 && k_ti < l2) {
      bs[tile_width_ti + tj] = b[l2_j +  k_ti];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(tk = 0; tk < tile_width_3; tk += 4) {
      __private float4 av;
      __private float4 bv;
      av.x = as[tile_width_ti + tk + 0];
      av.y = as[tile_width_ti + tk + 1];
      av.z = as[tile_width_ti + tk + 2];
      av.w = as[tile_width_ti + tk + 3];
      bv.x = bs[tile_width * (tk + 0) + tj];
      bv.y = bs[tile_width * (tk + 1) + tj];
      bv.z = bs[tile_width * (tk + 2) + tj];
      bv.w = bs[tile_width * (tk + 3) + tj];
      cij += dot(av, bv);
    }
    for(; tk < tile_width; tk++) {
      cij += a[tile_width_ti + tk] * b[tile_width * tk + tj];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i < n2 && j < m2) {
    c[m2 * i + j] = cij;
  }
}

__kernel void mul_a_b_for_elems(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[m2 * i + j] * b[m2 * i + j];
  }
}

__kernel void mul_at_b_for_elems(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[n2 * j + i] * b[m2 * i + j];
  }
}

__kernel void mul_a_bt_for_elems(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[m2 * i + j] * b[n2 * j + i];
  }
}

__kernel void mul_at_bt_for_elems(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[n2 * j + i] * b[n2 * j + i];
  }
}

__kernel void div_a_b_for_elems(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[m2 * i + j] / b[m2 * i + j];
  }
}

__kernel void div_at_b_for_elems(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[n2 * j + i] / b[m2 * i + j];
  }
}

__kernel void div_a_bt_for_elems(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[m2 * i + j] / b[n2 * j + i];
  }
}

__kernel void div_at_bt_for_elems(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[n2 * j + i] / b[n2 * j + i];
  }
}

__kernel void add_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[m2 * i + j] + b;
  }
}

__kernel void add_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[n2 * j + i] + b;
  }
}

__kernel void sub_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[m2 * i + j] - b;
  }
}

__kernel void sub_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[n2 * j + i] - b;
  }
}

__kernel void rsub_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = b - a[m2 * i + j];
  }
}

__kernel void rsub_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = b - a[n2 * j + i];
  }
}

__kernel void mul_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[m2 * i + j] * b;
  }
}

__kernel void mul_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[n2 * j + i] * b;
  }
}

__kernel void div_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[m2 * i + j] / b;
  }
}

__kernel void div_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = a[n2 * j + i] / b;
  }
}

__kernel void rdiv_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = b / a[m2 * i + j];
  }
}

__kernel void rdiv_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = b / a[n2 * j + i];
  }
}

__kernel void sigmoid_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = 1.0f / (1.0f + exp(-a[m2 * i + j]));
  }
}

__kernel void sigmoid_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = 1.0f / (1.0f + exp(-a[n2 * j + i]));
  }
}

__kernel void tanh_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = tanh(a[m2 * i + j]);
  }
}

__kernel void tanh_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = tanh(a[n2 * j + i]);
  }
}

__kernel void softmax_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    size_t n2_3 = n2 & ~((size_t) 3);
    size_t k;
    float sum = 0.0f;
    for(k = 0; k < n2_3; k += 4) {
      __private float4 av;
      __private float4 tv;
      av.x = a[m2 * (k + 0) + j];
      av.y = a[m2 * (k + 1) + j];
      av.z = a[m2 * (k + 2) + j];
      av.w = a[m2 * (k + 3) + j];
      tv = exp(av);
      sum += tv.x + tv.y + tv.z + tv.w;
    }
    for(; k < n2; k++) {
      sum += exp(a[m2 * k + j]);
    }
    b[m2 * i + j] = exp(a[m2 * i + j]) / sum;
  }
}

__kernel void softmax_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    size_t n2_j = n2 * j;
    size_t n2_3 = n2 & ~((size_t) 3);
    size_t k;
    float sum = 0.0f;
    for(k = 0; k < n2_3; k += 4) {
      __private float4 av;
      __private float4 tv;
      av.x = a[n2_j + k + 0];
      av.y = a[n2_j + k + 1];
      av.z = a[n2_j + k + 2];
      av.w = a[n2_j + k + 3];
      tv = exp(av);
      sum += tv.x + tv.y + tv.z + tv.w;
    }
    for(; k < n2; k++) {
      sum += exp(a[n2_j + k]);
    }
    b[m2 * i + j] = exp(a[n2 * j + i]) / sum;
  }
}
