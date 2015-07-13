void lstm_forward(const int n,
                  const double* ws, const double* hidden, const double* inputs,
                  double* new_hidden, double* outputs, double* store) {
  for(int i = 0; i < n; i++) {
    const double x = inputs[0];
    const double g_i = inputs[1];
    const double g_f = inputs[2];
    const double g_o = inputs[3];
    const double h_old = hidden[0];

    const double h_add = x * g_i;
    const double h_keep = h_old * g_f;
    const double h_new = h_keep * ws[0] + h_add * ws[1];
    const double y = h_new * g_o;

    store[0] = x;
    store[1] = g_i;
    store[2] = g_f;
    store[3] = g_o;
    store[4] = h_old;
    store[5] = h_add;
    store[6] = h_keep;
    store[7] = h_new;

    new_hidden[0] = h_new;
    outputs[0] = y;

    inputs += 4;
    hidden += 1;
    ws += 2;
    new_hidden += 1;
    outputs += 1;
    store += 8;
  }
}


void lstm_backward(const int n, const double* ws, const double* store, const double* delta_h, const double* delta_y,
                   double* d_ws, double* d_hidden, double* d_inputs) {
  for(int i = 0; i < n; i++) {
    const double x = store[0];
    const double g_i = store[1];
    const double g_f = store[2];
    const double g_o = store[3];
    const double h_old = store[4];
    const double h_add = store[5];
    const double h_keep = store[6];
    const double h_new = store[7];

    const double ws_keep = ws[0];
    const double ws_add = ws[1];

    const double dy = delta_y[0];
    const double dh_new = dy * g_o + delta_h[0];
    const double dg_o = dy * h_new;
    const double dh_keep = dh_new * ws_keep;
    const double dh_add = dh_new * ws_add;
    const double dws_keep = dh_new * h_keep;
    const double dws_add = dh_new * h_add;
    const double dx = dh_add * g_i;
    const double dg_i = dh_add * x;
    const double dg_f = dh_keep * h_old;
    const double dh_old = dh_keep * g_f;

    d_ws[0] = dws_keep;
    d_ws[1] = dws_add;

    d_hidden[0] = dh_old;

    d_inputs[0] = dx;
    d_inputs[1] = dg_i;
    d_inputs[2] = dg_f;
    d_inputs[3] = dg_o;

    store += 8;
    ws += 2;
    delta_y += 1;
    delta_h += 1;
    d_ws += 2;
    d_hidden += 1;
    d_inputs += 4;
  }
}
