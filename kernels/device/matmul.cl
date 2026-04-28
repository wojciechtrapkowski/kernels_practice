
// __kernel void matmul(__global float *m1, __global float *m2, __global float
// *result,
//                    int row_1, int col_1, int row_2, int col_2) {

//   for (int i = 0; i < row_1; i++) {
//     for (int j = 0; j < col_2; j++) {
//       for (int k = 0; k < col_1; k++) {
//         result[i * col_2 + j] += m1[i * col_1 + k] * m2[k * col_2 + j];
//       }
//     }
//   }
// }

__kernel void matmul(__global float *m1, __global float *m2,
                     __global float *result, int row_1, int col_1, int row_2,
                     int col_2) {

  int i = get_global_id(0);
  int j = get_local_id(0);
  int wg = get_group_id(0);
  int wg_size = get_global_size(0) / get_local_size(0);
  printf("Global ID: %d, Work Group: %d Local ID: %d WG Size: %d\n", i, wg, j, wg_size);

  for (int i = get_group_id(0); i < row_1; i += wg_size) {
    for (int j = get_local_id(0); j < col_2; j += get_local_size(0)) {
      for (int k = 0; k < col_1; k++) {
        result[i * col_2 + j] += m1[i * col_1 + k] * m2[k * col_2 + j];
      }
    }
  }
}