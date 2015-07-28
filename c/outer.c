void outer_product(int n, int m, const double* input_u, const double* input_v, double* output) {
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < m; j++) {
      *output = input_u[i] * input_v[j];
      output++;
    }
  }
}

void vector_add(int n, double* target, const double* source) {
  for(int i = 0; i < n; i++) {
    *target += *source;
    target++; source++;
  }
}
