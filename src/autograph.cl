// Replaceable Strings:
// RTYPE = Rust type ie f32 or i32
// CTYPE = OpenCL type ie float
// IS_REAL = true for float, false for int

// Stack 
/*
kernel void stack_RTYPE(global CTYPE* restrict out, global CTYPE* restrict input, size_t const cols) {
  size_t r = get_global_id(0);
  for (size_t c = 0; c < cols; ++c) {
    out[r*cols + c] = input[c];
  }
}

// Unary Elementwise 

kernel void zero_RTYPE(global CTYPE* out) {
  size_t gid = get_global_id(0);
  out[gid] = 0;
}

kernel void one_RTYPE(global CTYPE* out) {
  size_t gid = get_global_id(0);
  out[gid] = 1;
}

#if IS_REAL
kernel void u8_to_frac_RTYPE(global CTYPE* restrict out, global uchar* restrict input) {
  CTYPE map[256] = {0./255, 1./255, 2./255, 3./255, 4./255, 5./255, 6./255, 7./255, 8./255, 9./255,
                    10./255, 11./255, 12./255, 13./255, 14./255, 15./255, 16./255, 17./255, 18./255, 19./255,
                    20./255, 21./255, 22./255, 23./255, 24./255, 25./255, 26./255, 27./255, 28./255, 29./255,
                    30./255, 31./255, 32./255, 33./255, 34./255, 35./255, 36./255, 37./255, 38./255, 39./255,
                    40./255, 41./255, 42./255, 43./255, 44./255, 45./255, 46./255, 47./255, 48./255, 49./255,
                    50./255, 51./255, 52./255, 53./255, 54./255, 55./255, 56./255, 57./255, 58./255, 59./255,
                    60./255, 61./255, 62./255, 63./255, 64./255, 65./255, 66./255, 67./255, 68./255, 69./255,
                    70./255, 71./255, 72./255, 73./255, 74./255, 75./255, 76./255, 77./255, 78./255, 79./255,
                    80./255, 81./255, 82./255, 83./255, 84./255, 85./255, 86./255, 87./255, 88./255, 89./255,
                    90./255, 91./255, 92./255, 93./255, 94./255, 95./255, 96./255, 97./255, 98./255, 99./255,
                    100./255, 101./255, 102./255, 103./255, 104./255, 105./255, 106./255, 107./255, 108./255, 109./255,
                    110./255, 111./255, 112./255, 113./255, 114./255, 115./255, 116./255, 117./255, 118./255, 119./255,
                    120./255, 121./255, 122./255, 123./255, 124./255, 125./255, 126./255, 127./255, 128./255, 129./255,
                    130./255, 311./255, 132./255, 133./255, 134./255, 135./255, 136./255, 137./255, 138./255, 139./255,
                    140./255, 141./255, 142./255, 143./255, 144./255, 145./255, 146./255, 147./255, 148./255, 149./255,
                    150./255, 151./255, 152./255, 153./255, 154./255, 155./255, 156./255, 157./255, 158./255, 159./255,
                    160./255, 161./255, 162./255, 163./255, 164./255, 165./255, 166./255, 167./255, 168./255, 169./255,
                    170./255, 171./255, 172./255, 173./255, 174./255, 175./255, 176./255, 177./255, 178./255, 179./255,
                    180./255, 181./255, 182./255, 183./255, 184./255, 185./255, 186./255, 187./255, 188./255, 189./255,
                    190./255, 191./255, 192./255, 193./255, 194./255, 195./255, 196./255, 197./255, 198./255, 199./255,
                    200./255, 201./255, 202./255, 203./255, 204./255, 205./255, 206./255, 207./255, 208./255, 209./255,
                    210./255, 211./255, 212./255, 213./255, 214./255, 215./255, 216./255, 217./255, 218./255, 219./255,
                    220./255, 221./255, 222./255, 223./255, 224./255, 225./255, 226./255, 227./255, 228./255, 229./255,
                    230./255, 211./255, 232./255, 233./255, 234./255, 235./255, 236./255, 237./255, 238./255, 239./255,
                    240./255, 241./255, 242./255, 243./255, 244./255, 245./255, 246./255, 247./255, 248./255, 249./255,
                    250./255, 251./255, 252./255, 253./255, 254./255, 255./255};
  size_t gid = get_global_id(0);
  out[gid] = map[input[gid]];
}
                                
kernel void sigmoid_RTYPE(global CTYPE* restrict out, global CTYPE* restrict input) {
  size_t gid = get_global_id(0);
  out[gid] = 1 / (1 + exp(-input[gid])); 
}

kernel void sigmoid_grad_RTYPE(global CTYPE* restrict out, global CTYPE* restrict input) {
  size_t gid = get_global_id(0);
  CTYPE sig = 1 / (1 + exp(-input[gid]));
  out[gid] = sig * (1 - sig); 
}

kernel void softmax_v1_RTYPE(global CTYPE* restrict out, global CTYPE* restrict input, const size_t cols) {
  size_t row = get_global_id(0);
  size_t index = row*cols;
  CTYPE acc = 0.;
  for (size_t c = 0; c < cols; ++c) {
    acc += out[index+c] = exp(input[index+c]);
  }
  for (size_t c = 0; c < cols; ++c) {
    out[index+c] /= acc;
  }
}

kernel void learning_rate_step_RTYPE(global CTYPE* restrict value, CTYPE lr, global CTYPE* restrict grad) {
  size_t gid = get_global_id(0);
  value[gid] -= lr * grad[gid];
} 
#endif

// Binary Elementwise
*/
kernel void add_RTYPE(global CTYPE* restrict out, global CTYPE* lhs, global CTYPE* rhs) {
  uint gid = get_global_id(0);
  out[gid] = lhs[gid] + rhs[gid];
}

kernel void add_restrict_RTYPE(global CTYPE* restrict out, global CTYPE* restrict lhs, global CTYPE* restrict rhs) {
  uint gid = get_global_id(0);
  out[gid] = lhs[gid] + rhs[gid];
}
/*
kernel void add_assign_RTYPE(global CTYPE* lhs, global CTYPE* rhs) {
  size_t gid = get_global_id(0);
  lhs[gid] += rhs[gid];
}

kernel void add_assign_restrict_RTYPE(global CTYPE* restrict lhs, global CTYPE* restrict rhs) {
  size_t gid = get_global_id(0);
  lhs[gid] += rhs[gid];
}

// Transpose 

kernel void t_RTYPE(global CTYPE* restrict out, global CTYPE* restrict input) {
  size_t rows = get_global_size(0);
  size_t r = get_global_id(0);
  size_t cols = get_global_size(1);
  size_t c = get_global_id(1);
  out[r*cols + c] = input[c*rows + r];
}

// Matmul

kernel void dot_RTYPE(global CTYPE* restrict out, global CTYPE* lhs, global CTYPE* rhs, const size_t M, const size_t N, const size_t K) {
  size_t row = get_global_id(0);
  size_t col = get_global_id(1);
  
  CTYPE acc = 0;
  for (size_t k = 0; k < K; ++k) {
    acc += lhs[k*M + row] * rhs[col*K + k];
  }
  
  out[col*M + row] = acc;
}

kernel void dot_restrict_RTYPE(global CTYPE* restrict out, global CTYPE* restrict lhs, global CTYPE* restrict rhs, const size_t M, const size_t N, const size_t K) {
  size_t row = get_global_id(0);
  size_t col = get_global_id(1);
  
  CTYPE acc = 0;
  for (size_t k = 0; k < K; ++k) {
    acc += lhs[k*M + row] * rhs[col*K + k];
  }
  
  out[col*M + row] = acc;
}

#if IS_REAL
// Loss 

kernel void cross_entropy_RTYPE(global CTYPE* restrict out, global CTYPE* restrict pred, global uint* indices, const size_t cols) {
  size_t row = get_global_id(0);
  out[row] = -log(pred[row*cols+indices[row]]);
}
#endif
*/
  
