//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

// icpc -std=c++11 -O3 -xHost -ansi-alias -mkl=parallel -DNDEBUG -DIL_MKL
//   linear_solve_benchmark.cpp -o main -lbenchmark

#include <random>

#include <il/linear_algebra/dense/blas/blas.h>
#include "PartialLU.h"

#include <benchmark/benchmark.h>

// BM_LU_ARRAY2D (Fortran ordered) is usually faster than BM_LU_ARRAY_2C
// (C ordered)
// - 20% to 30% faster on OSX 10.11.2 with Intel compiler 16.0.1

static void BM_LU_MKL(benchmark::State& state) {
  while (state.KeepRunning()) {
    state.PauseTiming();

    std::random_device random_device{};
    std::mt19937 generator{random_device()};
    std::normal_distribution<double> normal{0.0, 1.0};

    const il::int_t n{state.range_x()};
    il::Array2D<double> A{n, n};
    for (il::int_t i = 0; i < n; ++i) {
      for (il::int_t j = 0; j < n; ++j) {
        A(i, j) = normal(generator);
      }
    }
    state.ResumeTiming();
    il::Status status{};
    il::LU lu{std::move(A), il::io, status};
    status.ignore_error();
  }
}

static void BM_LU_MKL_ALIGN(benchmark::State& state) {
  while (state.KeepRunning()) {
    state.PauseTiming();

    std::random_device random_device{};
    std::mt19937 generator{random_device()};
    std::normal_distribution<double> normal{0.0, 1.0};

    const il::int_t n{state.range_x()};
    il::Array2D<double> A{n, n, il::align, 256, 0};
    for (il::int_t i = 0; i < n; ++i) {
      for (il::int_t j = 0; j < n; ++j) {
        A(i, j) = normal(generator);
      }
    }
    state.ResumeTiming();
    il::Status status{};
    il::LU lu{std::move(A), il::io, status};
    status.ignore_error();
  }
}

BENCHMARK(BM_LU_MKL)->Arg(1000)->Arg(2000)->Arg(3000);
BENCHMARK(BM_LU_MKL_ALIGN)->Arg(1000)->Arg(2000)->Arg(3000);

BENCHMARK_MAIN();
