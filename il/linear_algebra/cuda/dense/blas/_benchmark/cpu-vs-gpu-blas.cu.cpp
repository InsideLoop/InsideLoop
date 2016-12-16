//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <cstdio>

#include <il/Timer.h>
#include <il/container/cuda/cuda_copy.h>
#include <il/linear_algebra/cuda/dense/blas/cuda_blas.h>
#include <il/linear_algebra/dense/blas/blas.h>

int main() {
  {
    const int n = 10000;
    il::Array2D<double> A{n, n};
    for (il::int_t j = 0; j < n; ++j) {
      for (il::int_t i = 0; i < n; ++i) {
        A(i, j) = 1.0 / (2 + i + j);
      }
    }
    il::Array2D<double> C{n, n, 0.0};

    il::CudaArray2D<double> A_d{n, n};
    il::CudaArray2D<double> B_d{n, n};
    il::CudaArray2D<double> C_d{n, n};

    il::Timer timer_product_host{};
    il::blas(1.0, A, A, 0.0, il::io, C);
    timer_product_host.stop();

    il::copy(A, il::io, A_d);
    il::Timer timer_copy1{};
    il::copy(A, il::io, B_d);
    il::copy(C, il::io, C_d);
    timer_copy1.stop();
    il::Timer timer_product{};
    il::blas(1.0, A_d, B_d, 0.0, il::io, C_d);
    timer_product.stop();
    il::Timer timer_copy2{};
    il::copy(C_d, il::io, C);
    timer_copy2.stop();

    const il::int_t ln = n;
    std::printf("=== For double ===\n");
    std::printf(
        "Time for matrix multiplication on CPU: %9.2f s, %9.2f Gflops\n",
        timer_product_host.time(),
        1.0e-9 * 2 * ln * ln * ln / timer_product_host.time());
    std::printf("               Time to copy to device: %9.2f s\n",
                timer_copy1.time());
    std::printf(
        "Time for matrix multiplication on GPU: %9.2f s, %9.2f Gflops\n",
        timer_product.time(), 1.0e-9 * 2 * ln * ln * ln / timer_product.time());
    std::printf("             Time to copy from device: %9.2f s\n",
                timer_copy2.time());
  }

  {
    const int n = 10000;
    il::Array2D<float> A{n, n};
    for (il::int_t j = 0; j < n; ++j) {
      for (il::int_t i = 0; i < n; ++i) {
        A(i, j) = 1.0f / (2 + i + j);
      }
    }
    il::Array2D<float> C{n, n, 0.0};

    il::CudaArray2D<float> A_d{n, n};
    il::CudaArray2D<float> B_d{n, n};
    il::CudaArray2D<float> C_d{n, n};

    il::Timer timer_product_host{};
    il::blas(1.0, A, A, 0.0, il::io, C);
    timer_product_host.stop();

    il::copy(A, il::io, A_d);
    il::Timer timer_copy1{};
    il::copy(A, il::io, B_d);
    il::copy(C, il::io, C_d);
    timer_copy1.stop();
    il::Timer timer_product{};
    il::blas(1.0, A_d, B_d, 0.0, il::io, C_d);
    timer_product.stop();
    il::Timer timer_copy2{};
    il::copy(C_d, il::io, C);
    timer_copy2.stop();

    const il::int_t ln = n;
    std::printf("=== For float ===\n");
    std::printf(
        "Time for matrix multiplication on CPU: %9.2f s, %9.2f Gflops\n",
        timer_product_host.time(),
        1.0e-9 * 2 * ln * ln * ln / timer_product_host.time());
    std::printf("               Time to copy to device: %9.2f s\n",
                timer_copy1.time());
    std::printf(
        "Time for matrix multiplication on GPU: %9.2f s, %9.2f Gflops\n",
        timer_product.time(), 1.0e-9 * 2 * ln * ln * ln / timer_product.time());
    std::printf("             Time to copy from device: %9.2f s\n",
                timer_copy2.time());
  }

  return 0;
}
