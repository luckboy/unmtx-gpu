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
  size_t i = get_global_id(0) << 2;
  size_t j = get_global_id(1) << 2;
  size_t k;
  size_t mthread_size = get_local_size(0);
  size_t mtile_width = mthread_size << 2;
  size_t ti = get_local_id(0);
  size_t tj = get_local_id(1);
  size_t bi = ti << 2;
  size_t bj = tj << 2;
  __private float ar1;
  __private float ar2;
  __private float ar3;
  __private float ar4;
  __private float br1;
  __private float br2;
  __private float br3;
  __private float br4;
  __private float cr11 = 0.0f;
  __private float cr12 = 0.0f;
  __private float cr13 = 0.0f;
  __private float cr14 = 0.0f;
  __private float cr21 = 0.0f;
  __private float cr22 = 0.0f;
  __private float cr23 = 0.0f;
  __private float cr24 = 0.0f;
  __private float cr31 = 0.0f;
  __private float cr32 = 0.0f;
  __private float cr33 = 0.0f;
  __private float cr34 = 0.0f;
  __private float cr41 = 0.0f;
  __private float cr42 = 0.0f;
  __private float cr43 = 0.0f;
  __private float cr44 = 0.0f;
  for(k = 0; k < l2; k += mthread_size) {
    size_t tk;
    as[mthread_size * (bi + 0) + tj] = 0.0f;
    if(i + 0 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 0) + tj] = a[l2 * (i + 0) + k + tj];
    }
    as[mthread_size * (bi + 1) + tj] = 0.0f;
    if(i + 1 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 1) + tj] = a[l2 * (i + 1) + k + tj];
    }
    as[mthread_size * (bi + 2) + tj] = 0.0f;
    if(i + 2 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 2) + tj] = a[l2 * (i + 2) + k + tj];
    }
    as[mthread_size * (bi + 3) + tj] = 0.0f;
    if(i + 3 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 3) + tj] = a[l2 * (i + 3) + k + tj];
    }
    bs[mtile_width * ti + bj + 0] = 0.0f;
    if(j + 0 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 0] = b[m2 * (k + ti) + j + 0];
    }
    bs[mtile_width * ti + bj + 1] = 0.0f;
    if(j + 1 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 1] = b[m2 * (k + ti) + j + 1];
    }
    bs[mtile_width * ti + bj + 2] = 0.0f;
    if(j + 2 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 2] = b[m2 * (k + ti) + j + 2];
    }
    bs[mtile_width * ti + bj + 3] = 0.0f;
    if(j + 3 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 3] = b[m2 * (k + ti) + j + 3];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(tk = 0; tk < mthread_size; tk++) {
      ar1 = as[mthread_size * (bi + 0) + tk];
      ar2 = as[mthread_size * (bi + 1) + tk];
      ar3 = as[mthread_size * (bi + 2) + tk];
      ar4 = as[mthread_size * (bi + 3) + tk];
      br1 = bs[mtile_width * tk + bj + 0];
      br2 = bs[mtile_width * tk + bj + 1];
      br3 = bs[mtile_width * tk + bj + 2];
      br4 = bs[mtile_width * tk + bj + 3];
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
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = cr11;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = cr12;
  }
  if(i + 0 < n2 && j + 2 < m2) {
    c[m2 * (i + 0) + j + 2] = cr13;
  }
  if(i + 0 < n2 && j + 3 < m2) {
    c[m2 * (i + 0) + j + 3] = cr14;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = cr21;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = cr22;
  }
  if(i + 1 < n2 && j + 2 < m2) {
    c[m2 * (i + 1) + j + 2] = cr23;
  }
  if(i + 1 < n2 && j + 3 < m2) {
    c[m2 * (i + 1) + j + 3] = cr24;
  }
  if(i + 2 < n2 && j + 0 < m2) {
    c[m2 * (i + 2) + j + 0] = cr31;
  }
  if(i + 2 < n2 && j + 1 < m2) {
    c[m2 * (i + 2) + j + 1] = cr32;
  }
  if(i + 2 < n2 && j + 2 < m2) {
    c[m2 * (i + 2) + j + 2] = cr33;
  }
  if(i + 2 < n2 && j + 3 < m2) {
    c[m2 * (i + 2) + j + 3] = cr34;
  }
  if(i + 3 < n2 && j + 0 < m2) {
    c[m2 * (i + 3) + j + 0] = cr41;
  }
  if(i + 3 < n2 && j + 1 < m2) {
    c[m2 * (i + 3) + j + 1] = cr42;
  }
  if(i + 3 < n2 && j + 2 < m2) {
    c[m2 * (i + 3) + j + 2] = cr43;
  }
  if(i + 3 < n2 && j + 3 < m2) {
    c[m2 * (i + 3) + j + 3] = cr44;
  }
}

__kernel void mul_at_b(__global const float *a, __global const float *b, __global float *c, __local float *as, __local float *bs, ulong n, ulong m, ulong l)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t l2 = (size_t) l;
  size_t i = get_global_id(0) << 2;
  size_t j = get_global_id(1) << 2;
  size_t k;
  size_t mthread_size = get_local_size(0);
  size_t mtile_width = mthread_size << 2;
  size_t ti = get_local_id(0);
  size_t tj = get_local_id(1);
  size_t bi = ti << 2;
  size_t bj = tj << 2;
  __private float ar1;
  __private float ar2;
  __private float ar3;
  __private float ar4;
  __private float br1;
  __private float br2;
  __private float br3;
  __private float br4;
  __private float cr11 = 0.0f;
  __private float cr12 = 0.0f;
  __private float cr13 = 0.0f;
  __private float cr14 = 0.0f;
  __private float cr21 = 0.0f;
  __private float cr22 = 0.0f;
  __private float cr23 = 0.0f;
  __private float cr24 = 0.0f;
  __private float cr31 = 0.0f;
  __private float cr32 = 0.0f;
  __private float cr33 = 0.0f;
  __private float cr34 = 0.0f;
  __private float cr41 = 0.0f;
  __private float cr42 = 0.0f;
  __private float cr43 = 0.0f;
  __private float cr44 = 0.0f;
  for(k = 0; k < l2; k += mthread_size) {
    size_t tk;
    as[mthread_size * (bi + 0) + tj] = 0.0f;
    if(i + 0 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 0) + tj] = a[n2 * (k + tj) + i + 0];
    }
    as[mthread_size * (bi + 1) + tj] = 0.0f;
    if(i + 1 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 1) + tj] = a[n2 * (k + tj) + i + 1];
    }
    as[mthread_size * (bi + 2) + tj] = 0.0f;
    if(i + 2 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 2) + tj] = a[n2 * (k + tj) + i + 2];
    }
    as[mthread_size * (bi + 3) + tj] = 0.0f;
    if(i + 3 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 3) + tj] = a[n2 * (k + tj) + i + 3];
    }
    bs[mtile_width * ti + bj + 0] = 0.0f;
    if(j + 0 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 0] = b[m2 * (k + ti) + j + 0];
    }
    bs[mtile_width * ti + bj + 1] = 0.0f;
    if(j + 1 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 1] = b[m2 * (k + ti) + j + 1];
    }
    bs[mtile_width * ti + bj + 2] = 0.0f;
    if(j + 2 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 2] = b[m2 * (k + ti) + j + 2];
    }
    bs[mtile_width * ti + bj + 3] = 0.0f;
    if(j + 3 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 3] = b[m2 * (k + ti) + j + 3];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(tk = 0; tk < mthread_size; tk++) {
      ar1 = as[mthread_size * (bi + 0) + tk];
      ar2 = as[mthread_size * (bi + 1) + tk];
      ar3 = as[mthread_size * (bi + 2) + tk];
      ar4 = as[mthread_size * (bi + 3) + tk];
      br1 = bs[mtile_width * tk + bj + 0];
      br2 = bs[mtile_width * tk + bj + 1];
      br3 = bs[mtile_width * tk + bj + 2];
      br4 = bs[mtile_width * tk + bj + 3];
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
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = cr11;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = cr12;
  }
  if(i + 0 < n2 && j + 2 < m2) {
    c[m2 * (i + 0) + j + 2] = cr13;
  }
  if(i + 0 < n2 && j + 3 < m2) {
    c[m2 * (i + 0) + j + 3] = cr14;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = cr21;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = cr22;
  }
  if(i + 1 < n2 && j + 2 < m2) {
    c[m2 * (i + 1) + j + 2] = cr23;
  }
  if(i + 1 < n2 && j + 3 < m2) {
    c[m2 * (i + 1) + j + 3] = cr24;
  }
  if(i + 2 < n2 && j + 0 < m2) {
    c[m2 * (i + 2) + j + 0] = cr31;
  }
  if(i + 2 < n2 && j + 1 < m2) {
    c[m2 * (i + 2) + j + 1] = cr32;
  }
  if(i + 2 < n2 && j + 2 < m2) {
    c[m2 * (i + 2) + j + 2] = cr33;
  }
  if(i + 2 < n2 && j + 3 < m2) {
    c[m2 * (i + 2) + j + 3] = cr34;
  }
  if(i + 3 < n2 && j + 0 < m2) {
    c[m2 * (i + 3) + j + 0] = cr41;
  }
  if(i + 3 < n2 && j + 1 < m2) {
    c[m2 * (i + 3) + j + 1] = cr42;
  }
  if(i + 3 < n2 && j + 2 < m2) {
    c[m2 * (i + 3) + j + 2] = cr43;
  }
  if(i + 3 < n2 && j + 3 < m2) {
    c[m2 * (i + 3) + j + 3] = cr44;
  }
}

__kernel void mul_a_bt(__global const float *a, __global const float *b, __global float *c, __local float *as, __local float *bs, ulong n, ulong m, ulong l)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t l2 = (size_t) l;
  size_t i = get_global_id(0) << 2;
  size_t j = get_global_id(1) << 2;
  size_t k;
  size_t mthread_size = get_local_size(0);
  size_t mtile_width = mthread_size << 2;
  size_t ti = get_local_id(0);
  size_t tj = get_local_id(1);
  size_t bi = ti << 2;
  size_t bj = tj << 2;
  __private float ar1;
  __private float ar2;
  __private float ar3;
  __private float ar4;
  __private float br1;
  __private float br2;
  __private float br3;
  __private float br4;
  __private float cr11 = 0.0f;
  __private float cr12 = 0.0f;
  __private float cr13 = 0.0f;
  __private float cr14 = 0.0f;
  __private float cr21 = 0.0f;
  __private float cr22 = 0.0f;
  __private float cr23 = 0.0f;
  __private float cr24 = 0.0f;
  __private float cr31 = 0.0f;
  __private float cr32 = 0.0f;
  __private float cr33 = 0.0f;
  __private float cr34 = 0.0f;
  __private float cr41 = 0.0f;
  __private float cr42 = 0.0f;
  __private float cr43 = 0.0f;
  __private float cr44 = 0.0f;
  for(k = 0; k < l2; k += mthread_size) {
    size_t tk;
    as[mthread_size * (bi + 0) + tj] = 0.0f;
    if(i + 0 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 0) + tj] = a[l2 * (i + 0) + k + tj];
    }
    as[mthread_size * (bi + 1) + tj] = 0.0f;
    if(i + 1 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 1) + tj] = a[l2 * (i + 1) + k + tj];
    }
    as[mthread_size * (bi + 2) + tj] = 0.0f;
    if(i + 2 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 2) + tj] = a[l2 * (i + 2) + k + tj];
    }
    as[mthread_size * (bi + 3) + tj] = 0.0f;
    if(i + 3 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 3) + tj] = a[l2 * (i + 3) + k + tj];
    }
    bs[mtile_width * ti + bj + 0] = 0.0f;
    if(j + 0 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 0] = b[l2 * (j + 0) + k + ti];
    }
    bs[mtile_width * ti + bj + 1] = 0.0f;
    if(j + 1 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 1] = b[l2 * (j + 1) + k + ti];
    }
    bs[mtile_width * ti + bj + 2] = 0.0f;
    if(j + 2 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 2] = b[l2 * (j + 2) + k + ti];
    }
    bs[mtile_width * ti + bj + 3] = 0.0f;
    if(j + 3 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 3] = b[l2 * (j + 3) + k + ti];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(tk = 0; tk < mthread_size; tk++) {
      ar1 = as[mthread_size * (bi + 0) + tk];
      ar2 = as[mthread_size * (bi + 1) + tk];
      ar3 = as[mthread_size * (bi + 2) + tk];
      ar4 = as[mthread_size * (bi + 3) + tk];
      br1 = bs[mtile_width * tk + bj + 0];
      br2 = bs[mtile_width * tk + bj + 1];
      br3 = bs[mtile_width * tk + bj + 2];
      br4 = bs[mtile_width * tk + bj + 3];
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
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = cr11;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = cr12;
  }
  if(i + 0 < n2 && j + 2 < m2) {
    c[m2 * (i + 0) + j + 2] = cr13;
  }
  if(i + 0 < n2 && j + 3 < m2) {
    c[m2 * (i + 0) + j + 3] = cr14;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = cr21;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = cr22;
  }
  if(i + 1 < n2 && j + 2 < m2) {
    c[m2 * (i + 1) + j + 2] = cr23;
  }
  if(i + 1 < n2 && j + 3 < m2) {
    c[m2 * (i + 1) + j + 3] = cr24;
  }
  if(i + 2 < n2 && j + 0 < m2) {
    c[m2 * (i + 2) + j + 0] = cr31;
  }
  if(i + 2 < n2 && j + 1 < m2) {
    c[m2 * (i + 2) + j + 1] = cr32;
  }
  if(i + 2 < n2 && j + 2 < m2) {
    c[m2 * (i + 2) + j + 2] = cr33;
  }
  if(i + 2 < n2 && j + 3 < m2) {
    c[m2 * (i + 2) + j + 3] = cr34;
  }
  if(i + 3 < n2 && j + 0 < m2) {
    c[m2 * (i + 3) + j + 0] = cr41;
  }
  if(i + 3 < n2 && j + 1 < m2) {
    c[m2 * (i + 3) + j + 1] = cr42;
  }
  if(i + 3 < n2 && j + 2 < m2) {
    c[m2 * (i + 3) + j + 2] = cr43;
  }
  if(i + 3 < n2 && j + 3 < m2) {
    c[m2 * (i + 3) + j + 3] = cr44;
  }
}

__kernel void mul_at_bt(__global const float *a, __global const float *b, __global float *c, __local float *as, __local float *bs, ulong n, ulong m, ulong l)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t l2 = (size_t) l;
  size_t i = get_global_id(0) << 2;
  size_t j = get_global_id(1) << 2;
  size_t k;
  size_t mthread_size = get_local_size(0);
  size_t mtile_width = mthread_size << 2;
  size_t ti = get_local_id(0);
  size_t tj = get_local_id(1);
  size_t bi = ti << 2;
  size_t bj = tj << 2;
  __private float ar1;
  __private float ar2;
  __private float ar3;
  __private float ar4;
  __private float br1;
  __private float br2;
  __private float br3;
  __private float br4;
  __private float cr11 = 0.0f;
  __private float cr12 = 0.0f;
  __private float cr13 = 0.0f;
  __private float cr14 = 0.0f;
  __private float cr21 = 0.0f;
  __private float cr22 = 0.0f;
  __private float cr23 = 0.0f;
  __private float cr24 = 0.0f;
  __private float cr31 = 0.0f;
  __private float cr32 = 0.0f;
  __private float cr33 = 0.0f;
  __private float cr34 = 0.0f;
  __private float cr41 = 0.0f;
  __private float cr42 = 0.0f;
  __private float cr43 = 0.0f;
  __private float cr44 = 0.0f;
  for(k = 0; k < l2; k += mthread_size) {
    size_t tk;
    as[mthread_size * (bi + 0) + tj] = 0.0f;
    if(i + 0 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 0) + tj] = a[n2 * (k + tj) + i + 0];
    }
    as[mthread_size * (bi + 1) + tj] = 0.0f;
    if(i + 1 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 1) + tj] = a[n2 * (k + tj) + i + 1];
    }
    as[mthread_size * (bi + 2) + tj] = 0.0f;
    if(i + 2 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 2) + tj] = a[n2 * (k + tj) + i + 2];
    }
    as[mthread_size * (bi + 3) + tj] = 0.0f;
    if(i + 3 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 3) + tj] = a[n2 * (k + tj) + i + 3];
    }
    bs[mtile_width * ti + bj + 0] = 0.0f;
    if(j + 0 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 0] = b[l2 * (j + 0) + k + ti];
    }
    bs[mtile_width * ti + bj + 1] = 0.0f;
    if(j + 1 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 1] = b[l2 * (j + 1) + k + ti];
    }
    bs[mtile_width * ti + bj + 2] = 0.0f;
    if(j + 2 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 2] = b[l2 * (j + 2) + k + ti];
    }
    bs[mtile_width * ti + bj + 3] = 0.0f;
    if(j + 3 < m2 && k + ti < l2) {
      bs[mtile_width * ti + bj + 3] = b[l2 * (j + 3) + k + ti];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(tk = 0; tk < mthread_size; tk++) {
      ar1 = as[mthread_size * (bi + 0) + tk];
      ar2 = as[mthread_size * (bi + 1) + tk];
      ar3 = as[mthread_size * (bi + 2) + tk];
      ar4 = as[mthread_size * (bi + 3) + tk];
      br1 = bs[mtile_width * tk + bj + 0];
      br2 = bs[mtile_width * tk + bj + 1];
      br3 = bs[mtile_width * tk + bj + 2];
      br4 = bs[mtile_width * tk + bj + 3];
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
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i + 0 < n2 && j + 0 < m2) {
    c[m2 * (i + 0) + j + 0] = cr11;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    c[m2 * (i + 0) + j + 1] = cr12;
  }
  if(i + 0 < n2 && j + 2 < m2) {
    c[m2 * (i + 0) + j + 2] = cr13;
  }
  if(i + 0 < n2 && j + 3 < m2) {
    c[m2 * (i + 0) + j + 3] = cr14;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    c[m2 * (i + 1) + j + 0] = cr21;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    c[m2 * (i + 1) + j + 1] = cr22;
  }
  if(i + 1 < n2 && j + 2 < m2) {
    c[m2 * (i + 1) + j + 2] = cr23;
  }
  if(i + 1 < n2 && j + 3 < m2) {
    c[m2 * (i + 1) + j + 3] = cr24;
  }
  if(i + 2 < n2 && j + 0 < m2) {
    c[m2 * (i + 2) + j + 0] = cr31;
  }
  if(i + 2 < n2 && j + 1 < m2) {
    c[m2 * (i + 2) + j + 1] = cr32;
  }
  if(i + 2 < n2 && j + 2 < m2) {
    c[m2 * (i + 2) + j + 2] = cr33;
  }
  if(i + 2 < n2 && j + 3 < m2) {
    c[m2 * (i + 2) + j + 3] = cr34;
  }
  if(i + 3 < n2 && j + 0 < m2) {
    c[m2 * (i + 3) + j + 0] = cr41;
  }
  if(i + 3 < n2 && j + 1 < m2) {
    c[m2 * (i + 3) + j + 1] = cr42;
  }
  if(i + 3 < n2 && j + 2 < m2) {
    c[m2 * (i + 3) + j + 2] = cr43;
  }
  if(i + 3 < n2 && j + 3 < m2) {
    c[m2 * (i + 3) + j + 3] = cr44;
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

__kernel void swish_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = a[m2 * i + j] / (1.0f + exp(-a[m2 * i + j]));
  }
}

__kernel void swish_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = a[n2 * j + i] / (1.0f + exp(-a[n2 * j + i]));
  }
}

__kernel void softmax_a(__global const float *a, __global float *b, __local float *es, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  size_t k;
  size_t tile_width = get_local_size(1);
  size_t tile_height = get_local_size(0);
  size_t ti = get_local_id(0);
  size_t tj = get_local_id(1);
  float sum = 0.0f;
  for(k = 0; k < n2; k += tile_height) {
    size_t tk;
    es[tile_width * ti + tj] = 0.0f;
    if(j < m2 && k + ti < n2) {
      es[tile_width * ti + tj] = exp(a[m2 * (k + ti) + j]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(tk = 0; tk < tile_height; tk++) {
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
  size_t tile_width = get_local_size(1);
  size_t tile_height = get_local_size(0);
  size_t ti = get_local_id(0);
  size_t tj = get_local_id(1);
  float sum = 0.0f;
  for(k = 0; k < n2; k += tile_height) {
    size_t tk;
    es[tile_width * ti + tj] = 0.0f;
    if(j < m2 && k + ti < n2) {
      es[tile_width * ti + tj] = exp(a[n2 * j + k + ti]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(tk = 0; tk < tile_height; tk++) {
      sum += es[tile_width * tk + tj];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i < n2 && j < m2) {
    b[m2 * i + j] = exp(a[n2 * j + i]) / sum;
  }
}

__kernel void sqrt_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = sqrt(a[m2 * i + j]);
  }
}

__kernel void sqrt_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = sqrt(a[n2 * j + i]);
  }
}

__kernel void repeat_col_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = a[i];
  }
}

__kernel void repeat_row_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = a[j];
  }
}

__kernel void abs_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = fabs(a[m2 * i + j]);
  }
}

__kernel void abs_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = fabs(a[n2 * j + i]);
  }
}

__kernel void pow_a_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = pow(a[m2 * i + j], b[m2 * i + j]);
  }
}

__kernel void pow_at_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = pow(a[n2 * j + i], b[m2 * i + j]);
  }
}

__kernel void pow_a_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = pow(a[m2 * i + j], b[n2 * j + i]);
  }
}

__kernel void pow_at_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = pow(a[n2 * j + i], b[n2 * j + i]);
  }
}

__kernel void pow_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = pow(a[m2 * i + j], b);
  }
}

__kernel void pow_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = pow(a[n2 * j + i], b);
  }
}

__kernel void rpow_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = pow(b, a[m2 * i + j]);
  }
}

__kernel void rpow_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = pow(b, a[n2 * j + i]);
  }
}

__kernel void exp_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = exp(a[m2 * i + j]);
  }
}

__kernel void exp_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = exp(a[n2 * j + i]);
  }
}

__kernel void ln_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = log(a[m2 * i + j]);
  }
}

__kernel void ln_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = log(a[n2 * j + i]);
  }
}

__kernel void log2_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = log2(a[m2 * i + j]);
  }
}

__kernel void log2_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = log2(a[n2 * j + i]);
  }
}

__kernel void log10_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = log10(a[m2 * i + j]);
  }
}

__kernel void log10_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = log10(a[n2 * j + i]);
  }
}

__kernel void sin_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = sin(a[m2 * i + j]);
  }
}

__kernel void sin_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = sin(a[n2 * j + i]);
  }
}

__kernel void cos_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = cos(a[m2 * i + j]);
  }
}

__kernel void cos_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = cos(a[n2 * j + i]);
  }
}

__kernel void tan_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = tan(a[m2 * i + j]);
  }
}

__kernel void tan_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = tan(a[n2 * j + i]);
  }
}

__kernel void asin_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = asin(a[m2 * i + j]);
  }
}

__kernel void asin_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = asin(a[n2 * j + i]);
  }
}

__kernel void acos_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = acos(a[m2 * i + j]);
  }
}

__kernel void acos_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = acos(a[n2 * j + i]);
  }
}

__kernel void atan_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = atan(a[m2 * i + j]);
  }
}

__kernel void atan_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = atan(a[n2 * j + i]);
  }
}

__kernel void atan2_a_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = atan2(a[m2 * i + j], b[m2 * i + j]);
  }
}

__kernel void atan2_at_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = atan2(a[n2 * j + i], b[m2 * i + j]);
  }
}

__kernel void atan2_a_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = atan2(a[m2 * i + j], b[n2 * j + i]);
  }
}

__kernel void atan2_at_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = atan2(a[n2 * j + i], b[n2 * j + i]);
  }
}

__kernel void atan2_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = atan2(a[m2 * i + j], b);
  }
}

__kernel void atan2_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = atan2(a[n2 * j + i], b);
  }
}

__kernel void ratan2_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = atan2(b, a[m2 * i + j]);
  }
}

__kernel void ratan2_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = atan2(b, a[n2 * j + i]);
  }
}

__kernel void sinh_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = sinh(a[m2 * i + j]);
  }
}

__kernel void sinh_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = sinh(a[n2 * j + i]);
  }
}

__kernel void cosh_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = cosh(a[m2 * i + j]);
  }
}

__kernel void cosh_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = cosh(a[n2 * j + i]);
  }
}

__kernel void asinh_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = asinh(a[m2 * i + j]);
  }
}

__kernel void asinh_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = asinh(a[n2 * j + i]);
  }
}

__kernel void acosh_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = acosh(a[m2 * i + j]);
  }
}

__kernel void acosh_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = acosh(a[n2 * j + i]);
  }
}

__kernel void atanh_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = atanh(a[m2 * i + j]);
  }
}

__kernel void atanh_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = atanh(a[n2 * j + i]);
  }
}

__kernel void signum_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    if(!isnan(a[m2 * i + j])) {
      b[m2 * i + j] = (signbit(a[m2 * i + j]) ? -1.0 : 1.0);
    } else {
      b[m2 * i + j] = a[m2 * i + j];
    }
  }
}

__kernel void signum_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    if(!isnan(a[n2 * j + i])) {
      b[m2 * i + j] = (signbit(a[n2 * j + i]) ? -1.0 : 1.0);
    } else {
      b[m2 * i + j] = a[n2 * j + i];
    }
  }
}

__kernel void ceil_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = ceil(a[m2 * i + j]);
  }
}

__kernel void ceil_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = ceil(a[n2 * j + i]);
  }
}

__kernel void floor_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = floor(a[m2 * i + j]);
  }
}

__kernel void floor_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = floor(a[n2 * j + i]);
  }
}

__kernel void round_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = round(a[m2 * i + j]);
  }
}

__kernel void round_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = round(a[n2 * j + i]);
  }
}

__kernel void trunc_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = trunc(a[m2 * i + j]);
  }
}

__kernel void trunc_at(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    b[m2 * i + j] = trunc(a[n2 * j + i]);
  }
}

__kernel void max_a_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = fmax(a[m2 * i + j], b[m2 * i + j]);
  }
}

__kernel void max_at_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = fmax(a[n2 * j + i], b[m2 * i + j]);
  }
}

__kernel void max_a_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = fmax(a[m2 * i + j], b[n2 * j + i]);
  }
}

__kernel void max_at_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = fmax(a[n2 * j + i], b[n2 * j + i]);
  }
}

__kernel void max_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = fmax(a[m2 * i + j], b);
  }
}

__kernel void max_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = fmax(a[n2 * j + i], b);
  }
}

__kernel void min_a_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = fmin(a[m2 * i + j], b[m2 * i + j]);
  }
}

__kernel void min_at_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = fmin(a[n2 * j + i], b[m2 * i + j]);
  }
}

__kernel void min_a_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = fmin(a[m2 * i + j], b[n2 * j + i]);
  }
}

__kernel void min_at_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = fmin(a[n2 * j + i], b[n2 * j + i]);
  }
}

__kernel void min_a_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = fmin(a[m2 * i + j], b);
  }
}

__kernel void min_at_b_for_scalar(__global const float *a, float b, __global float *c, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    c[m2 * i + j] = fmin(a[n2 * j + i], b);
  }
}
