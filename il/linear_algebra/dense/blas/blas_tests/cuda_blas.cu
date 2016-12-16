// nvcc -std=c++11 cuda_blas.cu -o main -lcublas -lcurand

#include <cstdlib>
#include <cstdio>

#include <il/Timer.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

int main() {
  const int n = 16016;

  float* d_A;
  float* d_B;
  float* d_C;
  cudaMalloc(&d_A, n * n * sizeof(float));
  cudaMalloc(&d_B, n * n * sizeof(float));
  cudaMalloc(&d_C, n * n * sizeof(float));

  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, static_cast<std::size_t>(clock()));
  curandGenerateUniform(prng, d_A, n * n);
  curandGenerateUniform(prng, d_B, n * n);

  il::Timer timer{};
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B, n,
              &beta, d_C, n);
  cublasDestroy(handle);
  timer.stop();

  float* h_C = static_cast<float*>(malloc(n * n * sizeof(float)));
  cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);
  std::printf("Check: %8.4e\n", h_C[0]);
  free(h_C);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  long int m = n;
  std::printf("Time: %8.4e s\n", timer.time());
  std::printf("Gflops for CUDA: %8.4f Gflops\n", 1.0e-9 * 2 * m * m * m / timer.time());

  return 0;
}
