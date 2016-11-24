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

#include <il/Array.h>
#include <il/Array2D.h>
#include <il/Array2C.h>
#include <il/linear_algebra/blas.h>

#include <benchmark/benchmark.h>

// BM_LU_ARRAY2D (Fortran ordered) is usually faster than BM_LU_ARRAY_2C
// (C ordered)
// - 20% to 30% faster on OSX 10.11.2 with Intel compiler 16.0.1

static void BM_LU_ARRAY2D_NOTALIGNED(benchmark::State& state) {
  while (state.KeepRunning()) {
    state.PauseTiming();
    const il::int_t n{state.range_x()};
    il::Array2D<double> A{n, n};
    il::Array2D<double> B{n, n};
    il::Array2D<double> C{n, n};
    for (il::int_t i{0}; i < n; ++i) {
      for (il::int_t j{0}; j < n; ++j) {
        A(i, j) = 0.0;
        B(i, j) = 0.0;
        C(i, j) = 0.0;
      }
    }
    state.ResumeTiming();
    il::blas(1.0, A, B, 0.0, il::io, C);
  }
}

static void BM_LU_ARRAY2D(benchmark::State& state) {
  while (state.KeepRunning()) {
    state.PauseTiming();
    const il::int_t n{state.range_x()};
    il::Array2D<double> A{n, n, il::align, 256, 0};
    il::Array2D<double> B{n, n, il::align, 256, 0};
    il::Array2D<double> C{n, n, il::align, 256, 0};
    for (il::int_t i{0}; i < n; ++i) {
      for (il::int_t j{0}; j < n; ++j) {
        A(i, j) = 0.0;
        B(i, j) = 0.0;
        C(i, j) = 0.0;
      }
    }
    state.ResumeTiming();
    il::blas(1.0, A, B, 0.0, il::io, C);
  }
}

static void BM_LU_ARRAY2D_SHIFT(benchmark::State& state) {
  while (state.KeepRunning()) {
    state.PauseTiming();
    const il::int_t n{state.range_x()};
    il::Array2D<double> A{n, n, il::align, 256, 0};
    il::Array2D<double> B{n, n, il::align, 256, 128};
    il::Array2D<double> C{n, n, il::align, 256, 64};
    for (il::int_t i{0}; i < n; ++i) {
      for (il::int_t j{0}; j < n; ++j) {
        A(i, j) = 0.0;
        B(i, j) = 0.0;
        C(i, j) = 0.0;
      }
    }
    state.ResumeTiming();
    il::blas(1.0, A, B, 0.0, il::io, C);
  }
}

BENCHMARK(BM_LU_ARRAY2D_NOTALIGNED)->Arg(1987)->Arg(2048)->Arg(2112)->Arg(2304);
BENCHMARK(BM_LU_ARRAY2D)->Arg(1987)->Arg(2048)->Arg(2112)->Arg(2304);
BENCHMARK(BM_LU_ARRAY2D_SHIFT)->Arg(1987)->Arg(2048)->Arg(2112)->Arg(2304);

BENCHMARK_MAIN();
