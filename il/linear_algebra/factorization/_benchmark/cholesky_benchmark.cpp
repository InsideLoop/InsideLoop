//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <il/math.h>
#include <il/Array2D.h>
#include <il/LowerArray2D.h>
#include <il/linear_algebra/factorization/Cholesky.h>
#include <il/linear_algebra/factorization/LowerCholesky.h>

#include <benchmark/benchmark.h>

// icpc -std=c++11 -O3 -xHost -mkl=sequential -DNDEBUG -DIL_MKL
// -I/home/fayard/Documents/Projects/InsideLoop/InsideLoop
// -I/opt/gbenchmark/include
// -L/opt/gbenchmark/lib cholesky_benchmark.cpp -o main -lbenchmark
//
// The Cholesky factorization on a packed matrix is usually 2 times slower than
// with the full matrix.
// -> If you want to favor speed, use il::Array2D<double> which gives you the
//    full matrix and has a cost of n^2 elements for the memory
// -> If you want to favor low memory consumption, use il::LowerArray2D<double>
//    which gives you the half of the matrix and has a cost of n^2 / 2 elements
//    for the memory

static void BM_CHOLESKY_FULL(benchmark::State& state) {
  while (state.KeepRunning()) {
    state.PauseTiming();
    const il::int_t n{state.range_x()};
    il::Array2D<double> B{n, n};
    for (il::int_t j{0}; j < n; ++j) {
      for (il::int_t i{0}; i < n; ++i) {
        B(i, j) = 1.0 / (1 + il::abs(i - j));
      }
    }
    state.ResumeTiming();
    il::Error error{};
    il::Cholesky C{std::move(B), il::io, error};
    error.abort();
  }
}

static void BM_CHOLESKY_PACKED(benchmark::State& state) {
  while (state.KeepRunning()) {
    state.PauseTiming();
    const il::int_t n{state.range_x()};
    il::LowerArray2D<double> B{n};
    for (il::int_t j{0}; j < n; ++j) {
      for (il::int_t i{j}; i < n; ++i) {
        B(i, j) = 1.0 / (1 + il::abs(i - j));
      }
    }
    state.ResumeTiming();
    il::Error error{};
    il::LowerCholesky C{std::move(B), il::io, error};
    error.abort();
  }
}

BENCHMARK(BM_CHOLESKY_FULL)->Arg(100)->Arg(300)->Arg(1000)->Arg(3000);
BENCHMARK(BM_CHOLESKY_PACKED)->Arg(100)->Arg(300)->Arg(1000)->Arg(3000);

BENCHMARK_MAIN();
