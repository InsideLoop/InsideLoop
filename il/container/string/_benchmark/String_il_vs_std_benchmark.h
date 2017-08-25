#include <benchmark/benchmark.h>

#include <il/String.h>
#include <string>

static void IlStringConstruct_0(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::String s0 = "123456789012345";
    benchmark::DoNotOptimize(s0.data());
  }
}

static void StdStringConstruct_0(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::string s0 = "123456789012345";
    benchmark::DoNotOptimize(s0.data());
  }
}

static void IlStringConstruct_1(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::String s0 = "1234567890123456789012";
    benchmark::DoNotOptimize(s0.data());
  }
}

static void StdStringConstruct_1(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::string s0 = "1234567890123456789012";
    benchmark::DoNotOptimize(s0.data());
  }
}

static void IlStringConstruct_2(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::String s0 = "12345678901234567890123";
    benchmark::DoNotOptimize(s0.data());
  }
}

static void StdStringConstruct_2(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::string s0 = "12345678901234567890123";
    benchmark::DoNotOptimize(s0.data());
  }
}

//static void IlStringAppend_0(benchmark::State& state) {
//  while (state.KeepRunning()) {
//    il::String s0 = "Hello";
//    il::String s1 = "world";
//    s0.append(" ", s1, "!");
//    benchmark::DoNotOptimize(s0.data());
//  }
//}
//
//static void StdStringAppend_0(benchmark::State& state) {
//  while (state.KeepRunning()) {
//    std::string s0 = "Hello";
//    std::string s1 = "world";
//    s0 = s0 + " " + s1 + "!";
//    benchmark::DoNotOptimize(s0.data());
//  }
//}

BENCHMARK(IlStringConstruct_0);
BENCHMARK(StdStringConstruct_0);
BENCHMARK(IlStringConstruct_1);
BENCHMARK(StdStringConstruct_1);
BENCHMARK(IlStringConstruct_2);
BENCHMARK(StdStringConstruct_2);
//BENCHMARK(IlStringAppend_0);
//BENCHMARK(StdStringAppend_0);
