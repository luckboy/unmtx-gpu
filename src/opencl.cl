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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
  size_t k;
  size_t mthread_count = get_local_size(0);
  size_t mtile_width = mthread_count << 1;
  size_t ti = get_local_id(0);
  size_t tj = get_local_id(1);
  size_t bi = ti << 1;
  size_t bj = tj << 1;
  __private float ar1;
  __private float ar2;
  __private float br1;
  __private float br2;
  __private float cr11 = 0.0f;
  __private float cr12 = 0.0f;
  __private float cr21 = 0.0f;
  __private float cr22 = 0.0f;
  for(k = 0; k < l2; k += mthread_count) {
    size_t tk;
    as[mthread_count * (bi + 0) + tj] = 0.0f;
    if(i + 0 < n2 && k + tj < l2) {
      as[mthread_count * (bi + 0) + tj] = a[l2 * (i + 0) + k + tj];
    }
    as[mthread_count * (bi + 1) + tj] = 0.0f;
    if(i + 1 < n2 && k + tj < l2) {
      as[mthread_count * (bi + 1) + tj] = a[l2 * (i + 1) + k + tj];
    }
    bs[mtile_width * ti + bj + 0] = 0.0f;
    if(j + 0 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 0] = b[m2 * (k + ti) + j + 0];
    }
    bs[mtile_width * ti + bj + 1] = 0.0f;
    if(j + 1 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 1] = b[m2 * (k + ti) + j + 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(tk = 0; tk < mthread_count; tk++) {
      ar1 = as[mthread_count * (bi + 0) + tk];
      ar2 = as[mthread_count * (bi + 1) + tk];
      br1 = bs[mtile_width * tk + bj + 0];
      br2 = bs[mtile_width * tk + bj + 1];
      cr11 += ar1 * br1;
      cr12 += ar1 * br2;
      cr21 += ar2 * br1;
      cr22 += ar2 * br2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = cr11;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = cr12;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = cr21;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = cr22;
  }
}

__kernel void mul_at_b(__global const float *a, __global const float *b, __global float *c, __local float *as, __local float *bs, ulong n, ulong m, ulong l)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t l2 = (size_t) l;
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
  size_t k;
  size_t mthread_count = get_local_size(0);
  size_t mtile_width = mthread_count << 1;
  size_t ti = get_local_id(0);
  size_t tj = get_local_id(1);
  size_t bi = ti << 1;
  size_t bj = tj << 1;
  __private float ar1;
  __private float ar2;
  __private float br1;
  __private float br2;
  __private float cr11 = 0.0f;
  __private float cr12 = 0.0f;
  __private float cr21 = 0.0f;
  __private float cr22 = 0.0f;
  for(k = 0; k < l2; k += mthread_count) {
    size_t tk;
    as[mthread_count * (bi + 0) + tj] = 0.0f;
    if(i + 0 < n2 && k + tj < l2) {
      as[mthread_count * (bi + 0) + tj] = a[n2 * (k + tj) + i + 0];
    }
    as[mthread_count * (bi + 1) + tj] = 0.0f;
    if(i + 1 < n2 && k + tj < l2) {
      as[mthread_count * (bi + 1) + tj] = a[n2 * (k + tj) + i + 1];
    }
    bs[mtile_width * ti + bj + 0] = 0.0f;
    if(j + 0 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 0] = b[m2 * (k + ti) + j + 0];
    }
    bs[mtile_width * ti + bj + 1] = 0.0f;
    if(j + 1 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 1] = b[m2 * (k + ti) + j + 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(tk = 0; tk < mthread_count; tk++) {
      ar1 = as[mthread_count * (bi + 0) + tk];
      ar2 = as[mthread_count * (bi + 1) + tk];
      br1 = bs[mtile_width * tk + bj + 0];
      br2 = bs[mtile_width * tk + bj + 1];
      cr11 += ar1 * br1;
      cr12 += ar1 * br2;
      cr21 += ar2 * br1;
      cr22 += ar2 * br2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = cr11;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = cr12;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = cr21;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = cr22;
  }
}

__kernel void mul_a_bt(__global const float *a, __global const float *b, __global float *c, __local float *as, __local float *bs, ulong n, ulong m, ulong l)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t l2 = (size_t) l;
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
  size_t k;
  size_t mthread_count = get_local_size(0);
  size_t mtile_width = mthread_count << 1;
  size_t ti = get_local_id(0);
  size_t tj = get_local_id(1);
  size_t bi = ti << 1;
  size_t bj = tj << 1;
  __private float ar1;
  __private float ar2;
  __private float br1;
  __private float br2;
  __private float cr11 = 0.0f;
  __private float cr12 = 0.0f;
  __private float cr21 = 0.0f;
  __private float cr22 = 0.0f;
  for(k = 0; k < l2; k += mthread_count) {
    size_t tk;
    as[mthread_count * (bi + 0) + tj] = 0.0f;
    if(i + 0 < n2 && k + tj < l2) {
      as[mthread_count * (bi + 0) + tj] = a[l2 * (i + 0) + k + tj];
    }
    as[mthread_count * (bi + 1) + tj] = 0.0f;
    if(i + 1 < n2 && k + tj < l2) {
      as[mthread_count * (bi + 1) + tj] = a[l2 * (i + 1) + k + tj];
    }
    bs[mtile_width * ti + bj + 0] = 0.0f;
    if(j + 0 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 0] = b[l2 * (j + 0) + k + ti];
    }
    bs[mtile_width * ti + bj + 1] = 0.0f;
    if(j + 1 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 1] = b[l2 * (j + 1) + k + ti];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(tk = 0; tk < mthread_count; tk++) {
      ar1 = as[mthread_count * (bi + 0) + tk];
      ar2 = as[mthread_count * (bi + 1) + tk];
      br1 = bs[mtile_width * tk + bj + 0];
      br2 = bs[mtile_width * tk + bj + 1];
      cr11 += ar1 * br1;
      cr12 += ar1 * br2;
      cr21 += ar2 * br1;
      cr22 += ar2 * br2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = cr11;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = cr12;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = cr21;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = cr22;
  }
}

__kernel void mul_at_bt(__global const float *a, __global const float *b, __global float *c, __local float *as, __local float *bs, ulong n, ulong m, ulong l)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t l2 = (size_t) l;
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
  size_t k;
  size_t mthread_count = get_local_size(0);
  size_t mtile_width = mthread_count << 1;
  size_t ti = get_local_id(0);
  size_t tj = get_local_id(1);
  size_t bi = ti << 1;
  size_t bj = tj << 1;
  __private float ar1;
  __private float ar2;
  __private float br1;
  __private float br2;
  __private float cr11 = 0.0f;
  __private float cr12 = 0.0f;
  __private float cr21 = 0.0f;
  __private float cr22 = 0.0f;
  for(k = 0; k < l2; k += mthread_count) {
    size_t tk;
    as[mthread_count * (bi + 0) + tj] = 0.0f;
    if(i + 0 < n2 && k + tj < l2) {
      as[mthread_count * (bi + 0) + tj] = a[n2 * (k + tj) + i + 0];
    }
    as[mthread_count * (bi + 1) + tj] = 0.0f;
    if(i + 1 < n2 && k + tj < l2) {
      as[mthread_count * (bi + 1) + tj] = a[n2 * (k + tj) + i + 1];
    }
    bs[mtile_width * ti + bj + 0] = 0.0f;
    if(j + 0 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 0] = b[l2 * (j + 0) + k + ti];
    }
    bs[mtile_width * ti + bj + 1] = 0.0f;
    if(j + 1 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 1] = b[l2 * (j + 1) + k + ti];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(tk = 0; tk < mthread_count; tk++) {
      ar1 = as[mthread_count * (bi + 0) + tk];
      ar2 = as[mthread_count * (bi + 1) + tk];
      br1 = bs[mtile_width * tk + bj + 0];
      br2 = bs[mtile_width * tk + bj + 1];
      cr11 += ar1 * br1;
      cr12 += ar1 * br2;
      cr21 += ar2 * br1;
      cr22 += ar2 * br2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = cr11;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = cr12;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = cr21;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = cr22;
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

__kernel void softmax_a(__global const float *a, __global float *b, __local float *es, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  size_t k;
  size_t tile_width = get_local_size(0);
  size_t ti = get_local_id(0);
  size_t tj = get_local_id(1);
  size_t tile_width_ti = tile_width * ti;
  size_t tile_width_3 = tile_width & ~((size_t) 3);
  float sum = 0.0f;
  for(k = 0; k < n2; k += tile_width) {
    size_t k_ti = k + ti;
    size_t tk;
    es[tile_width_ti + tj] = 0.0f;
    if(j < m2 && k_ti < n2) {
      es[tile_width_ti + tj] = exp(a[m2 * k_ti + j]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(tk = 0; tk < tile_width_3; tk += 4) {
      __private float4 ev;
      ev.x = es[tile_width * (tk + 0) + tj];
      ev.y = es[tile_width * (tk + 1) + tj];
      ev.z = es[tile_width * (tk + 2) + tj];
      ev.w = es[tile_width * (tk + 3) + tj];
      sum += ev.x + ev.y + ev.z + ev.w;
    }
    for(; tk < tile_width_3; tk++) {
      sum += es[tile_width * tk + tj];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i < n2 && j < m2) {
    b[m2 * i + j] = exp(a[m2 * i + j]) / sum;
  }
}

__kernel void softmax_at(__global const float *a, __global float *b, __local float *es, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  size_t k;
  size_t tile_width = get_local_size(0);
  size_t ti = get_local_id(0);
  size_t tj = get_local_id(1);
  size_t tile_width_ti = tile_width * ti;
  size_t tile_width_3 = tile_width & ~((size_t) 3);
  size_t n2_j = n2 * j;
  float sum = 0.0f;
  for(k = 0; k < n2; k += tile_width) {
    size_t k_ti = k + ti;
    size_t tk;
    es[tile_width_ti + tj] = 0.0f;
    if(j < m2 && k_ti < n2) {
      es[tile_width_ti + tj] = exp(a[n2_j + k_ti]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(tk = 0; tk < tile_width_3; tk += 4) {
      __private float4 ev;
      ev.x = es[tile_width * (tk + 0) + tj];
      ev.y = es[tile_width * (tk + 1) + tj];
      ev.z = es[tile_width * (tk + 2) + tj];
      ev.w = es[tile_width * (tk + 3) + tj];
      sum += ev.x + ev.y + ev.z + ev.w;
    }
    for(; tk < tile_width_3; tk++) {
      sum += es[tile_width * tk + tj];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i < n2 && j < m2) {
    b[m2 * i + j] = exp(a[n2 * j + i]) / sum;
  }
}
