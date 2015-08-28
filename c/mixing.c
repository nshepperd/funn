void layer_resize_forward(const int A, const int B, const double* input, double* output) {
  for(int i = 0; i < B; i++) {
    output[i] = 0.0;
  }

  if(A < B) {
    // getting bigger
    int j = 0;
    while (j < B) {
      for(int i = 0; i < A && j < B; ++i, ++j) {
        output[j] = input[i];
      }
    }
  } else {
    // getting smaller
    for(int i = 0; i < B; ++i) {
      output[i] = input[i];
    }
  }
}

void layer_resize_backward(const int A, const int B, const double* delta, double* output) {
  for(int i = 0; i < A; i++) {
    output[i] = 0.0;
  }

  if(A < B) {
    // growing
    int j = 0;
    while (j < B) {
      for(int i = 0; i < A && j < B; ++i, ++j) {
        output[i] += delta[j];
      }
    }
  } else {
    // getting smaller
    for(int i = 0; i < B; ++i) {
      output[i] = delta[i];
    }
  }
}


void layer_mix_forward(const int N, const int* table, const double* params, const double* input, double* output) {
  for(int i = 0; i < N; i++) {
    output[i] = (input[table[0]] * params[0]
                 + input[table[1]] * params[1]
                 + input[table[2]] * params[2]);
    table += 3;
    params += 3;
  }
}

void layer_mix_backward(const int N, const int* table, const double* params, const double* delta, double* din) {
  /* for(int i = 0; i < N; i++) { */
  /*   din[i] = 0; */
  /* } */

  for(int i = 0; i < N; i++) {
    din[table[0]] += params[0] * delta[i];
    din[table[1]] += params[1] * delta[i];
    din[table[2]] += params[2] * delta[i];
    table += 3;
    params += 3;
  }
}

void layer_mix_backward_params(const int N, const int* table, const double* input, const double* delta, double* dparams) {
  for(int i = 0; i < N; i++) {
    dparams[0] = input[table[0]] * delta[i];
    dparams[1] = input[table[1]] * delta[i];
    dparams[2] = input[table[2]] * delta[i];
    table += 3;
    dparams += 3;
  }
}





void layer_mix2_forward(const int N, const int* table, const double* params, const double* input, double* output) {
  for(int i = 0; i < N; i++) {
    output[i] = (input[table[0]] * params[0]
                 + input[table[1]] * params[1]);
    table += 2;
    params += 2;
  }
}

void layer_mix2_backward(const int N, const int* table, const double* params, const double* delta, double* din) {
  for(int i = 0; i < N; i++) {
    din[table[0]] += params[0] * delta[i];
    din[table[1]] += params[1] * delta[i];
    table += 2;
    params += 2;
  }
}

void layer_mix2_backward_params(const int N, const int* table, const double* input, const double* delta, double* dparams) {
  for(int i = 0; i < N; i++) {
    dparams[0] = input[table[0]] * delta[i];
    dparams[1] = input[table[1]] * delta[i];
    table += 2;
    dparams += 2;
  }
}


void layer_mixN_forward(const int N, const int* table, const double* params, const double* input, double* output) {
  for(int i = 0; i < N; i++) {
    /* printf("%i %i %i\n", i, table[2*i], table[2*i+1]); */
    output[table[2*i+1]] += input[table[2*i]] * params[i];
  }
}

void layer_mixN_backward(const int N, const int* table, const double* params, const double* delta, double* din) {
  for(int i = 0; i < N; i++) {
    din[table[2*i]] += delta[table[2*i+1]] * params[i];
  }
}

void layer_mixN_backward_params(const int N, const int* table, const double* input, const double* delta, double* dparams) {
  for(int i = 0; i < N; i++) {
    dparams[i] = input[table[2*i]] * delta[table[2*i+1]];
  }
}
