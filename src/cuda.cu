//
// Copyright (c) 2025 Łukasz Szpakowski
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
extern "C" {
  __global__ void add_a_b(const float *a, const float *b, float *c, size_t n, size_t m)
  {
    size_t i = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    size_t j = ((size_t) blockDim.y) * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
      c[m * i + j] = a[m * i + j] + b[m * i + j];
    }
  }
}
