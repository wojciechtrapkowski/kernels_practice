#pragma once

#include <iostream>

namespace HostKernels {
void matmul(float *m1, float *m2, float *result, int row_1, int col_1,
            int row_2, int col_2) {
  if (col_1 != row_2) {
    throw std::invalid_argument(
        "Incompatible matrix dimensions for multiplication.");
  }

  for (int i = 0; i < row_1; i++) {
    for (int j = 0; j < col_2; j++) {
      for (int k = 0; k < col_1; k++) {
        result[i * col_2 + j] += m1[i * col_1 + k] * m2[k * col_2 + j];
      }
    }
  }
}
}; // namespace HostKernels