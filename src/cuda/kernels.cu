extern "C" {
  __global__ void u8_to_f32(const unsigned char* x, float* y, unsigned int len) {
    const float scale = 1.0f / 255.0f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    if (tid < len) {
      y[tid] = scale * x[tid];
    }
  }
  __global__ void u8_to_one_hot_f32(const unsigned char* x, unsigned int nclasses, float* y, unsigned int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    if (tid < len) {
      y[tid*nclasses+x[tid]] = 1.0f;
    }
  } 
  __global__ void add(const float* x1, const float* x2, float* y, unsigned int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
        y[tid] = x1[tid] + x2[tid];
    }
  } 
  __global__ void cross_entropy_forward(unsigned int batch_size, unsigned int nclasses, const float* x, const float* t, float* y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    if (tid < batch_size) {
      // compute max value of slice
      float m = x[tid*nclasses];
      for(int i = 1; i < nclasses; ++i) {
        m = fmaxf(x[tid*nclasses+i], m);
      } 
      // subtract max
      for(int i = 0; i < nclasses; ++i) {
        y[tid*nclasses+i] = x[tid*nclasses+i]-m;
      }
      // sum
      float s = 0.0f;
      for(int i = 0; i < nclasses; ++i) {
        s += expf(y[tid*nclasses+i]);
      }
      // compute ln(s)
      float ln_s = logf(s);
      // y = (ln_s - y) * t
      for(int i = 0; i < nclasses; ++i) {
        y[tid*nclasses+i] = (ln_s - y[tid*nclasses+i]) * t[tid*nclasses+i];
      }
    }
  }
  __global__ void cross_entropy_backward(const float* x, float* dx, const float* t, float* dy, unsigned int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    if (tid < len) {
      dx[tid] = dy[0] * (x[tid] - t[tid]);
    }
  }
  __global__ void reduce_sum_partial(const float* input, float* output, unsigned int len) {
    // from http://www.techdarting.com/2014/06/parallel-reduction-in-cuda.html
    // Load a segment of the input vector into shared memory
    __shared__ float partialSum[2*256];
    int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;

    if ((start + t) < len)
    {
        partialSum[t] = input[start + t];      
    }
    else
    {       
        partialSum[t] = 0.0;
    }
    if ((start + blockDim.x + t) < len)
    {   
        partialSum[blockDim.x + t] = input[start + blockDim.x + t];
    }
    else
    {
        partialSum[blockDim.x + t] = 0.0;
    }

    // Traverse reduction tree
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
    {
      __syncthreads();
        if (t < stride)
            partialSum[t] += partialSum[t + stride];
    }
    __syncthreads();

    // Write the computed sum of the block to the output vector at correct index
    if (t == 0 && (globalThreadId*2) < len)
    {
        output[blockIdx.x] = partialSum[t];
    }
  }
  __global__ void reduce_sum_final(const float* x, float* y, unsigned int len) {
    *y = 0;
    for(int i = 0; i < len; ++i) {
      *y += x[i];  
    }
  }
  __global__ void reverse_conv_filter(const float* x, float beta, float* y, unsigned int filter_len, unsigned int len) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < len) {
      if (beta == 0.0f) {
        for(int i = 0; i < filter_len; ++i) {
          y[tid*filter_len + i] = x[tid*filter_len + ((filter_len-1) - i)];
        }
      }
      else {
        for(int i = 0; i < filter_len; ++i) {
          y[tid*filter_len + i] = x[tid*filter_len + ((filter_len-1) - i)] + beta * y[tid*filter_len + i];
        }
      }
    }
  }
}
