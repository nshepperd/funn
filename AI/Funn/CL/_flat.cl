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
  const int y = get_global_id(0);
  float o = pars[m * n + y + pars_offset];
  for (int x = 0; x < m; x++) {
    o += pars[y * m + x + pars_offset] * xs[x + xs_offset];
  }
  ys[y + ys_offset] = o;
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


__kernel void fcdiff_dxs(const int m, const int n,
                         __global const float* pars, const int pars_offset,
                         __global const float* dys, const int dys_offset,
                         __global float* dxs, const int dxs_offset) {
  const int x = get_global_id(0);
  float o = 0.0;
  for (int y = 0; y < n; y++) {
    o += pars[y * m + x + pars_offset] * dys[y + dys_offset];
  }
  dxs[x + dxs_offset] = o;
}


__kernel void fcdiff_dbs(__global const float* dys, const int dys_offset,
                         __global float* dbs, const int dbs_offset) {
  const int index = get_global_id(0);
  dbs[index + dbs_offset] = dys[index + dys_offset];
}
