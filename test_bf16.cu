#include <random>
#include "test_bf16.h"

// fp32(D) = bf16(A)*bf16(B) + fp32(C), alayout=row, blayout=col
__global__ void mma_m16n8k16_bf16_kernel(const nv_bfloat16 *A, const nv_bfloat16 *B, float *D, float C = 0.0f) {
  // 通过特殊寄存器获取每个线程的laneid（与thread id的区别，thread id是一个block内所有线程的编号，laneid仅是一个warp内线程的编号）
  uint32_t laneid;
  asm("mov.u32 %0, %%laneid;" : "=r"(laneid) :);

  // 共享内存，用来存放矩阵A、B、D
  __shared__ nv_bfloat16 A_smem[16 * 16];
  __shared__ nv_bfloat16 B_smem[8 * 16];
  __shared__ float D_smem[16 * 8];

  // 每个线程根据laneid从全局内存搬移一部分数据到共享内存
  *((int4 *)(&A_smem[laneid * 8])) = *((int4 *)(&A[laneid * 8]));
  if (laneid < 16) {
    *((int4 *)(&B_smem[laneid * 8])) = *((int4 *)(&B[laneid * 8]));
  }

  // 线程间同步，保证所有线程的数据都已经搬好
  __syncthreads();
  // 每个线程上的寄存器
  float Rd[4];
  uint32_t Ra[4];
  uint32_t Rb[2];
  float Rc[4] = {C, C, C, C};

  // 每个线程从共享内存load对应的数据到寄存器，对应关系参考ptx mma & ldmatrix指令说明（或者参考https://blog.csdn.net/qq_40672115/article/details/150711245）
  uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[(laneid % 16) * 16 + (laneid / 16) * 8]);
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(Ra[0]), "=r"(Ra[1]), "=r"(Ra[2]), "=r"(Ra[3])
               : "r"(A_smem_lane_addr));
  uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[(laneid % 8) * 16 + ((laneid / 8) % 2) * 8]);
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(Rb[0]), "=r"(Rb[1]) : "r"(B_smem_lane_addr));

  // warp级mma计算
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(Rd[0]), "=f"(Rd[1]), "=f"(Rd[2]), "=f"(Rd[3])
      : "r"(Ra[0]), "r"(Ra[1]), "r"(Ra[2]), "r"(Ra[3]),
        "r"(Rb[0]), "r"(Rb[1]),
        "f"(Rc[0]), "f"(Rc[1]), "f"(Rc[2]), "f"(Rc[3]));
  // 线程间同步，保证后续各线程搬数都在warp级mma计算完成后
  __syncthreads();

  // 每个线程从寄存器store对应点的数据到共享内存
  D_smem[(laneid / 4) * 8 + (laneid % 4) * 2] = Rd[0];
  D_smem[(laneid / 4) * 8 + (laneid % 4) * 2 + 1] = Rd[1];
  D_smem[((laneid / 4) + 8) * 8 + (laneid % 4) * 2] = Rd[2];
  D_smem[((laneid / 4) + 8) * 8 + (laneid % 4) * 2 + 1] = Rd[3];

  // 线程间同步，保证所有线程的数据都已经搬好
  __syncthreads();

  // 每个线程根据laneid从共享内存搬移对应的数据到全局内存
  *((int4 *)(&D[laneid * 4])) = *((int4 *)(&D_smem[laneid * 4]));
}

void test_exp_diff_bf16() {
  uint16_t min_value = 0x1;
  nv_bfloat16 A[16 * 16] = {nv_bfloat16(0.0f)};
  nv_bfloat16 B[8 * 16] = {nv_bfloat16(0.0f)};
  float D[16 * 8] = {0.0f};

  for (uint32_t i = 0; i < 16 * 16; i++) {
    A[i] = *(nv_bfloat16 *)(&min_value);
  }
  for (uint32_t i = 0; i < 8 * 16; i++) {
    B[i] = nv_bfloat16(1.0f);
  }

  BF16 d_A[16];
  BF16 d_B[16];

  for (uint32_t i = 1; i < 50; i++) {
    if (A[0] < B[0]) {
      A[0] = nv_bfloat16(2.0f) * A[0];
    } else {
      B[0] = nv_bfloat16(2.0f) * B[0];
    }
    A[1] = A[0];
    B[1] = -B[0];

    double golden_rst = 0.0;
    for (uint32_t j = 0; j < 16; j++) {
      golden_rst += (double)A[j] * (double)B[j];
    }

    for (uint32_t j = 0; j < 16; j++) {
      d_A[j] = *(uint16_t *)(&A[j]);
      d_B[j] = *(uint16_t *)(&B[j]);
    }
    FP32 d_D = bf16_1x16x1<EXPAND_BIT_BF16, EXTRA_BIT_BF16>(d_A, d_B, FP32(0));
    uint32_t d_D_32 = d_D.to_uint();
    D[0] = *(float *)(&d_D_32);

    double D_double = (double)D[0];
    uint64_t golden_rst_tmp = *((uint64_t *)&golden_rst);
    uint64_t D_tmp = *((uint64_t *)&D_double);
    if (golden_rst_tmp != D_tmp) {
      printf("test_exp_diff_bf16: exp_diff=%d, golden_rst=%g(%lx), mma_rst=%g(%lx)\n", i, golden_rst, golden_rst_tmp, (double)D[0], D_tmp);
      break;
    }
  }
}

void test_special_bf16() {
  nv_bfloat16 A[16 * 16] = {nv_bfloat16(0.0f)};
  nv_bfloat16 B[8 * 16] = {nv_bfloat16(0.0f)};
  float D[16 * 8] = {0.0f};

  for (uint32_t i = 0; i < 16 * 16; i++) {
    A[i] = nv_bfloat16(1.0f);
  }
  for (uint32_t i = 0; i < 8 * 16; i++) {
    B[i] = nv_bfloat16(1.0f);
  }

  nv_bfloat16 *d_A;
  nv_bfloat16 *d_B;
  float *d_D;
  CUDA_CHECK(cudaMalloc(&d_A, 16 * 16 * sizeof(nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_B, 8 * 16 * sizeof(nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_D, 16 * 8 * sizeof(float)));
  BF16 my_A[16];
  BF16 my_B[16];

  for (uint32_t i = 0; i < 7; i++) {
    float C = 0.0f;
    FP32 my_C;
    for (uint32_t j = 0; j < 16; j++) {
      A[j] = nv_bfloat16(1.0f);
      B[j] = nv_bfloat16(1.0f);
    }
    if (0 == i) {
      A[0] = nv_bfloat16(std::numeric_limits<float>::infinity());
    } else if (1 == i) {
      A[0] = nv_bfloat16(-std::numeric_limits<float>::infinity());
    } else if (2 == i) {
      A[0] = nv_bfloat16(std::numeric_limits<float>::quiet_NaN());
    } else if (3 == i) {
      A[0] = nv_bfloat16(std::numeric_limits<float>::infinity());
      A[1] = nv_bfloat16(-std::numeric_limits<float>::infinity());
    } else if (4 == i) {
      C = std::numeric_limits<float>::infinity();
    } else if (5 == i) {
      C = std::numeric_limits<float>::quiet_NaN();
    } else if (6 == i) {
      C = std::numeric_limits<float>::infinity();
      A[0] = nv_bfloat16(-std::numeric_limits<float>::infinity());
    }

    double golden_rst = (double)C;
    for (uint32_t j = 0; j < 16; j++) {
      golden_rst += (double)A[j] * (double)B[j];
    }

    CUDA_CHECK(cudaMemcpy(d_A, A, 16 * 16 * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, 8 * 16 * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    mma_m16n8k16_bf16_kernel<<<1, 32>>>(d_A, d_B, d_D, C);
    LAST_KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpy(D, d_D, 16 * 8 * sizeof(float), cudaMemcpyDeviceToHost));

    for (uint32_t j = 0; j < 16; j++) {
      my_A[j] = *(uint16_t *)(&A[j]);
      my_B[j] = *(uint16_t *)(&B[j]);
    }
    my_C = *(uint32_t *)(&C);
    FP32 my_D = bf16_1x16x1<EXPAND_BIT_BF16, EXTRA_BIT_BF16>(my_A, my_B, my_C);
    uint32_t my_D_32 = my_D.to_uint();
    float my_D_f32 = *(float *)(&my_D_32);

    double D_double = (double)D[0];
    double my_D_double = (double)my_D_f32;
    uint64_t golden_rst_tmp = *((uint64_t *)&golden_rst);
    uint64_t D_tmp = *((uint64_t *)&D_double);
    uint64_t my_D_tmp = *((uint64_t *)&my_D_double);
    printf("test_special_bf16: i=%d, golden_rst=%g(%lx), mma_rst=%g(%lx), my_rst=%g(%lx)\n", i, golden_rst, golden_rst_tmp, (double)D[0], D_tmp, my_D_double, my_D_tmp);
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_D);
}

void test_ulp_bf16() {
  nv_bfloat16 A[16 * 16] = {nv_bfloat16(0.0f)};
  nv_bfloat16 B[8 * 16] = {nv_bfloat16(0.0f)};
  float D[16 * 8] = {0.0f};

  for (uint32_t i = 0; i < 16 * 16; i++) {
    A[i] = nv_bfloat16(1.0f);
  }
  for (uint32_t i = 0; i < 8 * 16; i++) {
    B[i] = nv_bfloat16(1.0f);
  }

  nv_bfloat16 *d_A;
  nv_bfloat16 *d_B;
  float *d_D;
  CUDA_CHECK(cudaMalloc(&d_A, 16 * 16 * sizeof(nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_B, 8 * 16 * sizeof(nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_D, 16 * 8 * sizeof(float)));
  BF16 my_A[16];
  BF16 my_B[16];

  uint64_t my_ave_ulp = 0;
  uint32_t my_max_ulp = 0;
  uint64_t nv_ave_ulp = 0;
  uint32_t nv_max_ulp = 0;
  uint32_t loop_num = 100000;
  float start_f = -100.0f;
  float end_f = 100.0f;
  // uint32_t golden_rst_32_max, my_D_32_max, D_32_max;
  for (uint32_t i = 0; i < loop_num; i++) {
    float C;
    FP32 my_C;
    float f_amax, f_amin;
    bool stop_flag = false;
    while (!stop_flag) {
      C = (float)rand() / (float)RAND_MAX * (end_f - start_f) + start_f;
      f_amax = start_f;
      f_amin = end_f;
      if (C != 0.0f) {
        f_amax = fabs(C);
        f_amin = fabs(C);
      }
      for (uint32_t j = 0; j < 16; j++) {
        A[j] = nv_bfloat16((float)rand() / (float)RAND_MAX * (end_f - start_f) + start_f);
        B[j] = nv_bfloat16((float)rand() / (float)RAND_MAX * (end_f - start_f) + start_f);
        float tmp = fabs((float)A[j] * (float)B[j]);
        if (tmp != 0.0f) {
          f_amax = fmax(f_amax, tmp);
          f_amin = fmin(f_amin, tmp);
        }
      }
      if ((f_amax / f_amin) < powf(2.0f, 26.0) && !(std::isinf(f_amax / f_amin) || std::isnan(f_amax / f_amin))) {
        stop_flag = true;
      }
    }

    double golden_rst = (double)C;
    for (uint32_t j = 0; j < 16; j++) {
      golden_rst += (double)A[j] * (double)B[j];
    }

    CUDA_CHECK(cudaMemcpy(d_A, A, 16 * 16 * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, 8 * 16 * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    mma_m16n8k16_bf16_kernel<<<1, 32>>>(d_A, d_B, d_D, C);
    LAST_KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpy(D, d_D, 16 * 8 * sizeof(float), cudaMemcpyDeviceToHost));

    for (uint32_t j = 0; j < 16; j++) {
      my_A[j] = *(uint16_t *)(&A[j]);
      my_B[j] = *(uint16_t *)(&B[j]);
    }
    my_C = *(uint32_t *)(&C);
    FP32 my_D = bf16_1x16x1<EXPAND_BIT_BF16, EXTRA_BIT_BF16>(my_A, my_B, my_C);
    uint32_t my_D_32 = my_D.to_uint();
    float my_D_f32 = *(float *)(&my_D_32);

    float golden_rst_f32 = (float)golden_rst;
    uint32_t golden_rst_32 = *(uint32_t *)(&golden_rst_f32);

    uint32_t D_32 = *(uint32_t *)(&D[0]);
    uint32_t my_ulp_tmp = std::abs((int64_t)golden_rst_32 - (int64_t)my_D_32);
    uint32_t nv_ulp_tmp = std::abs((int64_t)golden_rst_32 - (int64_t)D_32);

    // if (my_max_ulp < my_ulp_tmp) {
    // if (nv_max_ulp < nv_ulp_tmp) {
    //   golden_rst_32_max = golden_rst_32;
    //   my_D_32_max = my_D_32;
    //   D_32_max = D_32;
    // }

    my_ave_ulp += my_ulp_tmp;
    my_max_ulp = std::max(my_max_ulp, my_ulp_tmp);
    nv_ave_ulp += nv_ulp_tmp;
    nv_max_ulp = std::max(nv_max_ulp, nv_ulp_tmp);

    // printf("test_ulp_bf16: i=%d, golden_rst=%g(%x), mma_rst=%g(%x), my_rst=%g(%x)\n", i, golden_rst_f32, golden_rst_32, D[0], D_32, my_D_f32, my_D_32);
  }
  printf("test_ulp_bf16: my_ave_ulp=%g, nv_ave_ulp=%g; my_max_ulp=%d, nv_max_ulp=%d\n",
         (float)my_ave_ulp / (float)loop_num, (float)nv_ave_ulp / (float)loop_num,
         my_max_ulp, nv_max_ulp);
  // printf("%x %x %x\n", golden_rst_32_max, my_D_32_max, D_32_max);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_D);
}
