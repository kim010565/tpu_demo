#pragma once

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>

#define CUDA_CHECK(call) cuda_check(call, __FILE__, __LINE__)
#define LAST_KERNEL_CHECK() kernel_check(__FILE__, __LINE__)

inline void cuda_check(cudaError_t err, const char *file, const int line) {
  if (err != cudaSuccess) {
    printf("cuda ERROR: %s:%d, ", file, line);
    printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
    exit(1);
  }
}

inline void kernel_check(const char *file, const int line) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("kernel ERROR: %s:%d, ", file, line);
    printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
    exit(1);
  }
}
