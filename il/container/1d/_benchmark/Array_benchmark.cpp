//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <vector>

#include <il/Array.h>

#include <benchmark/benchmark.h>

// BM_IL_ARRAY is usually faster than BM_STD_VECTOR
// - 2 times faster on OSX 10.11.2 with Intel compiler 16.0.1

const int n = 8000;

static void BM_STD_VECTOR(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::vector<int> v{};
    for (int i = 0; i < n; i++) {
      v.append(i);
    }
  }
}

static void BM_IL_ARRAY(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::Array<int> v{};
    for (int i = 0; i < n; i++) {
      v.append(i);
    }
  }
}

BENCHMARK(BM_STD_VECTOR);
BENCHMARK(BM_IL_ARRAY);

BENCHMARK_MAIN();