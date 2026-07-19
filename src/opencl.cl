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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t ik = get_global_id(0);
  size_t jk = get_global_id(1);
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
    size_t tjik = (tj + ik) % mthread_size;
    size_t tijk = (ti + jk) % mthread_size;
    as[mthread_size * (bi + 0) + tjik] = 0.0f;
    if(i + 0 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 0) + tjik] = a[l2 * (i + 0) + k + tj];
    }
    as[mthread_size * (bi + 1) + tjik] = 0.0f;
    if(i + 1 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 1) + tjik] = a[l2 * (i + 1) + k + tj];
    }
    as[mthread_size * (bi + 2) + tjik] = 0.0f;
    if(i + 2 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 2) + tjik] = a[l2 * (i + 2) + k + tj];
    }
    as[mthread_size * (bi + 3) + tjik] = 0.0f;
    if(i + 3 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 3) + tjik] = a[l2 * (i + 3) + k + tj];
    }
    bs[mthread_size * (bj + 0) + tijk] = 0.0f;
    if(j + 0 < m2 && k + ti < l2) {
      bs[mthread_size * (bj + 0) + tijk] = b[m2 * (k + ti) + j + 0];
    }
    bs[mthread_size * (bj + 1) + tijk] = 0.0f;
    if(j + 1 < m2 && k + ti < l2) {
      bs[mthread_size * (bj + 1) + tijk] = b[m2 * (k + ti) + j + 1];
    }
    bs[mthread_size * (bj + 2) + tijk] = 0.0f;
    if(j + 2 < m2 && k + ti < l2) {
      bs[mthread_size * (bj + 2) + tijk] = b[m2 * (k + ti) + j + 2];
    }
    bs[mthread_size * (bj + 3) + tijk] = 0.0f;
    if(j + 3 < m2 && k + ti < l2) {
      bs[mthread_size * (bj + 3) + tijk] = b[m2 * (k + ti) + j + 3];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(tk = 0; tk < mthread_size; tk++) {
      size_t tkik = (tk + ik) % mthread_size;
      size_t tkjk = (tk + jk) % mthread_size;
      ar1 = as[mthread_size * (bi + 0) + tkik];
      ar2 = as[mthread_size * (bi + 1) + tkik];
      ar3 = as[mthread_size * (bi + 2) + tkik];
      ar4 = as[mthread_size * (bi + 3) + tkik];
      br1 = bs[mthread_size * (bj + 0) + tkjk];
      br2 = bs[mthread_size * (bj + 1) + tkjk];
      br3 = bs[mthread_size * (bj + 2) + tkjk];
      br4 = bs[mthread_size * (bj + 3) + tkjk];
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
  size_t ik = get_global_id(0);
  size_t jk = get_global_id(1);
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
    size_t tjik = (tj + ik) % mthread_size;
    size_t tijk = (ti + jk) % mthread_size;
    as[mthread_size * (bi + 0) + tjik] = 0.0f;
    if(i + 0 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 0) + tjik] = a[n2 * (k + tj) + i + 0];
    }
    as[mthread_size * (bi + 1) + tjik] = 0.0f;
    if(i + 1 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 1) + tjik] = a[n2 * (k + tj) + i + 1];
    }
    as[mthread_size * (bi + 2) + tjik] = 0.0f;
    if(i + 2 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 2) + tjik] = a[n2 * (k + tj) + i + 2];
    }
    as[mthread_size * (bi + 3) + tjik] = 0.0f;
    if(i + 3 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 3) + tjik] = a[n2 * (k + tj) + i + 3];
    }
    bs[mthread_size * (bj + 0) + tijk] = 0.0f;
    if(j + 0 < m2 && k + ti < l2) {
      bs[mthread_size * (bj + 0) + tijk] = b[m2 * (k + ti) + j + 0];
    }
    bs[mthread_size * (bj + 1) + tijk] = 0.0f;
    if(j + 1 < m2 && k + ti < l2) {
      bs[mthread_size * (bj + 1) + tijk] = b[m2 * (k + ti) + j + 1];
    }
    bs[mthread_size * (bj + 2) + tijk] = 0.0f;
    if(j + 2 < m2 && k + ti < l2) {
      bs[mthread_size * (bj + 2) + tijk] = b[m2 * (k + ti) + j + 2];
    }
    bs[mthread_size * (bj + 3) + tijk] = 0.0f;
    if(j + 3 < m2 && k + ti < l2) {
      bs[mthread_size * (bj + 3) + tijk] = b[m2 * (k + ti) + j + 3];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(tk = 0; tk < mthread_size; tk++) {
      size_t tkik = (tk + ik) % mthread_size;
      size_t tkjk = (tk + jk) % mthread_size;
      ar1 = as[mthread_size * (bi + 0) + tkik];
      ar2 = as[mthread_size * (bi + 1) + tkik];
      ar3 = as[mthread_size * (bi + 2) + tkik];
      ar4 = as[mthread_size * (bi + 3) + tkik];
      br1 = bs[mthread_size * (bj + 0) + tkjk];
      br2 = bs[mthread_size * (bj + 1) + tkjk];
      br3 = bs[mthread_size * (bj + 2) + tkjk];
      br4 = bs[mthread_size * (bj + 3) + tkjk];
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
  size_t ik = get_global_id(0);
  size_t jk = get_global_id(1);
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
    size_t tjik = (tj + ik) % mthread_size;
    size_t tijk = (ti + jk) % mthread_size;
    as[mthread_size * (bi + 0) + tjik] = 0.0f;
    if(i + 0 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 0) + tjik] = a[l2 * (i + 0) + k + tj];
    }
    as[mthread_size * (bi + 1) + tjik] = 0.0f;
    if(i + 1 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 1) + tjik] = a[l2 * (i + 1) + k + tj];
    }
    as[mthread_size * (bi + 2) + tjik] = 0.0f;
    if(i + 2 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 2) + tjik] = a[l2 * (i + 2) + k + tj];
    }
    as[mthread_size * (bi + 3) + tjik] = 0.0f;
    if(i + 3 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 3) + tjik] = a[l2 * (i + 3) + k + tj];
    }
    bs[mthread_size * (bj + 0) + tijk] = 0.0f;
    if(j + 0 < m2 && k + ti < l2) {
      bs[mthread_size * (bj + 0) + tijk] = b[l2 * (j + 0) + k + ti];
    }
    bs[mthread_size * (bj + 1) + tijk] = 0.0f;
    if(j + 1 < m2 && k + ti < l2) {
      bs[mthread_size * (bj + 1) + tijk] = b[l2 * (j + 1) + k + ti];
    }
    bs[mthread_size * (bj + 2) + tijk] = 0.0f;
    if(j + 2 < m2 && k + ti < l2) {
      bs[mthread_size * (bj + 2) + tijk] = b[l2 * (j + 2) + k + ti];
    }
    bs[mthread_size * (bj + 3) + tijk] = 0.0f;
    if(j + 3 < m2 && k + ti < l2) {
      bs[mthread_size * (bj + 3) + tijk] = b[l2 * (j + 3) + k + ti];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(tk = 0; tk < mthread_size; tk++) {
      size_t tkik = (tk + ik) % mthread_size;
      size_t tkjk = (tk + jk) % mthread_size;
      ar1 = as[mthread_size * (bi + 0) + tkik];
      ar2 = as[mthread_size * (bi + 1) + tkik];
      ar3 = as[mthread_size * (bi + 2) + tkik];
      ar4 = as[mthread_size * (bi + 3) + tkik];
      br1 = bs[mthread_size * (bj + 0) + tkjk];
      br2 = bs[mthread_size * (bj + 1) + tkjk];
      br3 = bs[mthread_size * (bj + 2) + tkjk];
      br4 = bs[mthread_size * (bj + 3) + tkjk];
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
  size_t ik = get_global_id(0);
  size_t jk = get_global_id(1);
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
    size_t tjik = (tj + ik) % mthread_size;
    size_t tijk = (ti + jk) % mthread_size;
    as[mthread_size * (bi + 0) + tjik] = 0.0f;
    if(i + 0 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 0) + tjik] = a[n2 * (k + tj) + i + 0];
    }
    as[mthread_size * (bi + 1) + tjik] = 0.0f;
    if(i + 1 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 1) + tjik] = a[n2 * (k + tj) + i + 1];
    }
    as[mthread_size * (bi + 2) + tjik] = 0.0f;
    if(i + 2 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 2) + tjik] = a[n2 * (k + tj) + i + 2];
    }
    as[mthread_size * (bi + 3) + tjik] = 0.0f;
    if(i + 3 < n2 && k + tj < l2) {
      as[mthread_size * (bi + 3) + tjik] = a[n2 * (k + tj) + i + 3];
    }
    bs[mthread_size * (bj + 0) + tijk] = 0.0f;
    if(j + 0 < m2 && k + ti < l2) {
      bs[mthread_size * (bj + 0) + tijk] = b[l2 * (j + 0) + k + ti];
    }
    bs[mthread_size * (bj + 1) + tijk] = 0.0f;
    if(j + 1 < m2 && k + ti < l2) {
      bs[mthread_size * (bj + 1) + tijk] = b[l2 * (j + 1) + k + ti];
    }
    bs[mthread_size * (bj + 2) + tijk] = 0.0f;
    if(j + 2 < m2 && k + ti < l2) {
      bs[mthread_size * (bj + 2) + tijk] = b[l2 * (j + 2) + k + ti];
    }
    bs[mthread_size * (bj + 3) + tijk] = 0.0f;
    if(j + 3 < m2 && k + ti < l2) {
      bs[mthread_size * (bj + 3) + tijk] = b[l2 * (j + 3) + k + ti];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(tk = 0; tk < mthread_size; tk++) {
      size_t tkik = (tk + ik) % mthread_size;
      size_t tkjk = (tk + jk) % mthread_size;
      ar1 = as[mthread_size * (bi + 0) + tkik];
      ar2 = as[mthread_size * (bi + 1) + tkik];
      ar3 = as[mthread_size * (bi + 2) + tkik];
      ar4 = as[mthread_size * (bi + 3) + tkik];
      br1 = bs[mthread_size * (bj + 0) + tkjk];
      br2 = bs[mthread_size * (bj + 1) + tkjk];
      br3 = bs[mthread_size * (bj + 2) + tkjk];
      br4 = bs[mthread_size * (bj + 3) + tkjk];
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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

__kernel void softmax_a(__global const float *a, __global float *b, __local float *es, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
  size_t k;
  size_t tile_width = get_local_size(1) << 1;
  size_t tile_height = get_local_size(0) << 1;
  size_t ti = get_local_id(0) << 1;
  size_t tj = get_local_id(1) << 1;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  for(k = 0; k < n2; k += tile_height) {
    size_t tk;
    es[tile_width * (ti + 0) + tj + 0] = 0.0f;
    if(j + 0 < m2 && k + ti + 0 < n2) {
      es[tile_width * (ti + 0) + tj + 0] = exp(a[m2 * (k + ti + 0) + j + 0]);
    }
    es[tile_width * (ti + 0) + tj + 1] = 0.0f;
    if(j + 1 < m2 && k + ti + 0 < n2) {
      es[tile_width * (ti + 0) + tj + 1] = exp(a[m2 * (k + ti + 0) + j + 1]);
    }
    es[tile_width * (ti + 1) + tj + 0] = 0.0f;
    if(j + 0 < m2 && k + ti + 1 < n2) {
      es[tile_width * (ti + 1) + tj + 0] = exp(a[m2 * (k + ti + 1) + j + 0]);
    }
    es[tile_width * (ti + 1) + tj + 1] = 0.0f;
    if(j + 1 < m2 && k + ti + 1 < n2) {
      es[tile_width * (ti + 1) + tj + 1] = exp(a[m2 * (k + ti + 1) + j + 1]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(tk = 0; tk < tile_height; tk += 2) {
      sum1 += es[tile_width * (tk + 0) + tj + 0];
      sum1 += es[tile_width * (tk + 1) + tj + 0];
      sum2 += es[tile_width * (tk + 0) + tj + 1];
      sum2 += es[tile_width * (tk + 1) + tj + 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = exp(a[m2 * (i + 0) + j + 0]) / sum1;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = exp(a[m2 * (i + 0) + j + 1]) / sum2;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = exp(a[m2 * (i + 1) + j + 0]) / sum1;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = exp(a[m2 * (i + 1) + j + 1]) / sum2;
  }
}

__kernel void softmax_at(__global const float *a, __global float *b, __local float *es, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
  size_t k;
  size_t tile_width = get_local_size(1) << 1;
  size_t tile_height = get_local_size(0) << 1;
  size_t ti = get_local_id(0) << 1;
  size_t tj = get_local_id(1) << 1;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  for(k = 0; k < n2; k += tile_height) {
    size_t tk;
    es[tile_width * (ti + 0) + tj + 0] = 0.0f;
    if(j + 0 < m2 && k + ti + 0 < n2) {
      es[tile_width * (ti + 0) + tj + 0] = exp(a[n2 * (j + 0) + k + ti + 0]);
    }
    es[tile_width * (ti + 0) + tj + 1] = 0.0f;
    if(j + 1 < m2 && k + ti + 0 < n2) {
      es[tile_width * (ti + 0) + tj + 1] = exp(a[n2 * (j + 1) + k + ti + 0]);
    }
    es[tile_width * (ti + 1) + tj + 0] = 0.0f;
    if(j + 0 < m2 && k + ti + 1 < n2) {
      es[tile_width * (ti + 1) + tj + 0] = exp(a[n2 * (j + 0) + k + ti + 1]);
    }
    es[tile_width * (ti + 1) + tj + 1] = 0.0f;
    if(j + 1 < m2 && k + ti + 1 < n2) {
      es[tile_width * (ti + 1) + tj + 1] = exp(a[n2 * (j + 1) + k + ti + 1]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(tk = 0; tk < tile_height; tk += 2) {
      sum1 += es[tile_width * (tk + 0) + tj + 0];
      sum1 += es[tile_width * (tk + 1) + tj + 0];
      sum2 += es[tile_width * (tk + 0) + tj + 1];
      sum2 += es[tile_width * (tk + 1) + tj + 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0] = exp(a[n2 * (j + 0) + i + 0]) / sum1;
  }
  if(i + 0 < n2 && j + 1 < m2) {
    b[m2 * (i + 0) + j + 1] = exp(a[n2 * (j + 1) + i + 0]) / sum2;
  }
  if(i + 1 < n2 && j + 0 < m2) {
    b[m2 * (i + 1) + j + 0] = exp(a[n2 * (j + 0) + i + 1]) / sum1;
  }
  if(i + 1 < n2 && j + 1 < m2) {
    b[m2 * (i + 1) + j + 1] = exp(a[n2 * (j + 1) + i + 1]) / sum2;
  }
}

__kernel void sqrt_a(__global const float *a, __global float *b, ulong n, ulong m)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
  if(i + 0 < n2 && j + 0 < m2) {
    b[m2 * (i + 0) + j + 0 ] = exp(a[m2 * (i + 0) + j + 0]);
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
  size_t i = get_global_id(0) << 1;
  size_t j = get_global_id(1) << 1;
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
