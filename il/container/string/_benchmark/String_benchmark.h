#include <benchmark/benchmark.h>

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

static void StringAppend_0(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::String s0 = "hello";
    il::String s1 = "world!";
    s0.append(s1);
    benchmark::DoNotOptimize(s0.data());
    benchmark::DoNotOptimize(s1.data());
  }
}

static void StringAppend_1(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::String s0 = "hello";
    s0.append("world!");
    benchmark::DoNotOptimize(s0.data());
  }
}

static void StringJoin_0(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::String s0 = "hello";
    il::String s1 = "world";
    il::String s = il::join(s0, " ",  s1, "!");

    benchmark::DoNotOptimize(s0.data());
    benchmark::DoNotOptimize(s1.data());
    benchmark::DoNotOptimize(s.data());
  }
}

//static void StringFFJoinLarge(benchmark::State& state) {
//  while (state.KeepRunning()) {
//    auto j1 = il::join("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
//                       "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz");
//    auto j2 = il::join("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
//                       "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
//                       "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz");
//    auto j3 = il::join("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
//                       "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
//                       "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
//                       "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz");
//    auto j4 = il::join("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
//                       "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
//                       "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
//                       "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
//                       "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz");
//    benchmark::DoNotOptimize(j1.data());
//    benchmark::DoNotOptimize(j2.data());
//    benchmark::DoNotOptimize(j3.data());
//    benchmark::DoNotOptimize(j4.data());
//  }
//}


BENCHMARK(StringConstruct_Small_0);
BENCHMARK(StringConstruct_Large);
BENCHMARK(StringAppend_0);
BENCHMARK(StringAppend_1);
BENCHMARK(StringJoin_0);
//BENCHMARK(StringFFJoinLarge);
