// Replaceable Strings:
// RTYPE = Rust type ie f32 or i32
// CTYPE = OpenCL type ie float
// CVECTOR = OpenCL vector type ie float8 
// IS_REAL = true for float, false for int

// Activations 

#if IS_REAL
kernel void sigmoid_RTYPE(global CTYPE* restrict out, global CTYPE* restrict input) {
  size_t gid = get_global_id(0);
  out[gid] = 1 / (1 + exp(-input[gid])); 
}
#endif

// Stack 

kernel void stack_RTYPE(global CTYPE* restrict out, global CTYPE* restrict input, size_t const cols) {
  size_t r = get_global_id(0);
  for (size_t c = 0; c < cols; ++c) {
    out[r*cols + c] = input[c];
  }
}

// Binary Elementwise

kernel void add_RTYPE(global CTYPE* restrict out, global CTYPE* lhs, global CTYPE* rhs) {
  size_t gid = get_global_id(0);
  out[gid] = lhs[gid] + rhs[gid];
}

kernel void add_RTYPE_restrict(global CTYPE* restrict out, global CTYPE* restrict lhs, global CTYPE* restrict rhs) {
  size_t gid = get_global_id(0);
  out[gid] = lhs[gid] + rhs[gid];
}

// Transpose 

kernel void transpose_v1_RTYPE(global CTYPE* restrict out, global CTYPE* restrict input) {
  size_t rows = get_global_size(0);
  size_t r = get_global_id(0);
  size_t cols = get_global_size(1);
  size_t c = get_global_id(1);
  out[r*cols + c] = input[c*rows + r];
}

// Matmul

kernel void matmul_v1_RTYPE(global CTYPE* restrict out, global CTYPE* lhs, global CTYPE* rhs, const size_t M, const size_t N, const size_t K) {
  size_t row = get_global_id(0);
  size_t col = get_global_id(1);
  
  CTYPE acc = 0;
  for (size_t k = 0; k < K; ++k) {
    acc += lhs[k*M + row] * rhs[col*K + k];
  }
  
  out[col*M + row] = acc;
}

kernel void matmul_v1_RTYPE_restrict(global CTYPE* restrict out, global CTYPE* restrict lhs, global CTYPE* restrict rhs, const size_t M, const size_t N, const size_t K) {
  size_t row = get_global_id(0);
  size_t col = get_global_id(1);
  
  CTYPE acc = 0;
  for (size_t k = 0; k < K; ++k) {
    acc += lhs[k*M + row] * rhs[col*K + k];
  }
  
  out[col*M + row] = acc;
}
  
