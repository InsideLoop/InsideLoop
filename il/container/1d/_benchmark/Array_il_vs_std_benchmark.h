#include <benchmark/benchmark.h>

#include <vector>
#include <il/Array.h>

static void IlArray_Append(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::Array<double> v{};
    v.append(0.0);
    benchmark::DoNotOptimize(v.data());
  }
}

static void StdArray_Append(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::vector<double> v{};
    v.push_back(0.0);
    benchmark::DoNotOptimize(v.data());
  }
}

BENCHMARK(IlArray_Append);
BENCHMARK(StdArray_Append);
