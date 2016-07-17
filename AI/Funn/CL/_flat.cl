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

__kernel void sigmoid(__global const float* x, const int x_offset,
                      __global float* y, const int y_offset) {
  const size_t i = get_global_id(0);
  y[i+y_offset] = 1.0 / (1.0 + exp (-x[i+x_offset]));
}
__kernel void sigmoid_back(__global const float* y, const int y_offset,
                           __global const float* dy, const int dy_offset,
                           __global float* dx, const int dx_offset) {
  const size_t i = get_global_id(0);
  dx[i+dx_offset] = dy[i+dy_offset] * y[i+y_offset] * (1 - y[i+y_offset]);
}

__kernel void fcdiff(const int m, const int n,
                     __global const float* pars, const int pars_offset,
                     __global const float* xs, const int xs_offset,
                     __global float* ys, const int ys_offset) {
  const int index = get_global_id(0);
  const int w_offset = m * index;
  float o = pars[m * n + index + pars_offset];
  for (int i = 0; i < m; i++) {
    o += pars[w_offset + i + pars_offset] * xs[i + xs_offset];
  }
  ys[index + ys_offset] = o;
}

__kernel void fcdiff_dws(const int m, const int n,
                         __global const float* xs, const int xs_offset,
                         __global const float* dys, const int dys_offset,
                         __global float* dws, const int dws_offset) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  dws[dws_offset + y * m + x] =
    xs[xs_offset + x] * dys[dys_offset + y];
}
