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
    c[m2 * i + j] = a[n2 * j + i] + b[n2 * j + i];
  }
}

__kernel void mul_a_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m, ulong l)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t l2 = (size_t) l;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    size_t l2_i = l2 * i;
    size_t l2_3 = l2 & ~((size_t) 3);
    size_t k;
    float cij = 0.0f;
    for(k = 0; k < l2_3; k += 4) {
      __private float4 av;
      __private float4 bv;
      av.x = a[l2_i + k + 0];
      av.y = a[l2_i + k + 1];
      av.z = a[l2_i + k + 2];
      av.w = a[l2_i + k + 3];
      bv.x = b[m2 * (k + 0) + j];
      bv.y = b[m2 * (k + 1) + j];
      bv.z = b[m2 * (k + 2) + j];
      bv.w = b[m2 * (k + 3) + j];
      cij += dot(av, bv);
    }
    for(; k < l2; k++) {
      cij += a[l2_i + k] * b[m2 * k + j];
    }
    c[m2 * i + j] = cij;
  }
}

__kernel void mul_at_b(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m, ulong l)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t l2 = (size_t) l;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    size_t l2_3 = l2 & ~((size_t) 3);
    size_t k;
    float cij = 0.0f;
    for(k = 0; k < l2_3; k += 4) {
      __private float4 av;
      __private float4 bv;
      av.x = a[n2 * (k + 0) + i];
      av.y = a[n2 * (k + 1) + i];
      av.z = a[n2 * (k + 2) + i];
      av.w = a[n2 * (k + 3) + i];
      bv.x = b[m2 * (k + 0) + j];
      bv.y = b[m2 * (k + 1) + j];
      bv.z = b[m2 * (k + 2) + j];
      bv.w = b[m2 * (k + 3) + j];
      cij += dot(av, bv);
    }
    for(; k < l2; k++) {
      cij += a[n2 * k + i] * b[m2 * k + j];
    }
    c[m2 * i + j] = cij;
  }
}

__kernel void mul_a_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m, ulong l)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t l2 = (size_t) l;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    size_t l2_i = l2 * i;
    size_t l2_j = l2 * j;
    size_t l2_3 = l2 & ~((size_t) 3);
    size_t k;
    float cij = 0.0f;
    for(k = 0; k < l2_3; k += 4) {
      __private float4 av;
      __private float4 bv;
      av.x = a[l2_i + k + 0];
      av.y = a[l2_i + k + 1];
      av.z = a[l2_i + k + 2];
      av.w = a[l2_i + k + 3];
      bv.x = b[l2_j + k + 0];
      bv.y = b[l2_j + k + 1];
      bv.z = b[l2_j + k + 2];
      bv.w = b[l2_j + k + 3];
      cij += dot(av, bv);
    }
    for(; k < l2; k++) {
      cij += a[l2_i + k] * b[l2_j + k];
    }
    c[m2 * i + j] = cij;
  }
}

__kernel void mul_at_bt(__global const float *a, __global const float *b, __global float *c, ulong n, ulong m, ulong l)
{
  size_t n2 = (size_t) n;
  size_t m2 = (size_t) m;
  size_t l2 = (size_t) l;
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  if(i < n2 && j < m2) {
    size_t l2_j = l2 * j;
    size_t l2_3 = l2 & ~((size_t) 3);
    size_t k;
    float cij = 0.0f;
    for(k = 0; k < l2_3; k += 4) {
      __private float4 av;
      __private float4 bv;
      av.x = a[n2 * (k + 0) + i];
      av.y = a[n2 * (k + 1) + i];
      av.z = a[n2 * (k + 2) + i];
      av.w = a[n2 * (k + 3) + i];
      bv.x = b[l2_j + k + 0];
      bv.y = b[l2_j + k + 1];
      bv.z = b[l2_j + k + 2];
      bv.w = b[l2_j + k + 3];
      cij += dot(av, bv);
    }
    for(; k < l2; k++) {
      cij += a[n2 * k + i] * b[l2_j + k];
    }
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
