__kernel void vadd(__global float *a, __global float *b, __global float *c, __global float* d,
                   const unsigned int count) {
  int i = get_global_id(0);
  int j = get_local_id(0);
  int wg = get_group_id(0);
  local int local_counter;
  printf("Global ID: %d, Work Group: %d Local ID: %d\n", i, wg, j);

  for (int i = get_global_id(0); i < count; i += get_global_size(0)) {
    d[i] = a[i] + b[i] + c[i];
  }
}