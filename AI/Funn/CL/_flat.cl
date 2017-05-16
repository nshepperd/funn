__kernel void relu(__global const float* x, const int x_offset,
                   __global float* y, const int y_offset) {
  const size_t i = get_global_id(0);
  y[i + y_offset] = fmax(0.0f, x[i + x_offset]);
}
__kernel void relu_back(__global const float* x, const int x_offset,
                        __global const float* dy, const int dy_offset,
                        __global float* dx, const int dx_offset) {
  const size_t i = get_global_id(0);
  dx[i + dx_offset] = (x[i + x_offset] > 0.0) ? dy[i + dy_offset] : 0.0;
}
