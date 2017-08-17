#include <il/String.h>

static void StringConstruct_Small_0(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::String s0 = "1234567890123456789012";
    benchmark::DoNotOptimize(s0.data());
  }
}

static void StringConstruct_Large(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::String s0 = "12345678901234567890123";
    benchmark::DoNotOptimize(s0.data());
  }
}

BENCHMARK(StringConstruct_Small);
BENCHMARK(StringConstruct_Large);
