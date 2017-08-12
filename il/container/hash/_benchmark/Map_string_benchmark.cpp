#include <iostream>

#include <benchmark/benchmark.h>

#include <il/String.h>
#include <il/Map.h>

static void BM_MapCString(benchmark::State& state) {
  il::Map<il::String, int> map{7};
  map.set("aaa", 0);
  map.set("bbb", 0);
  map.set("ccc", 0);
  map.set("ddd", 0);
  map.set("eee", 0);
  map.set("fff", 0);
  map.set("ggg", 0);
  il::int_t i;
  while (state.KeepRunning()) {
    i += map.searchCString("aaa");
    i += map.searchCString("bbb");
    i += map.searchCString("ccc");
    i += map.searchCString("ddd");
    i += map.searchCString("eee");
    i += map.searchCString("fff");
    i += map.searchCString("ggg");
  }
  std::cout << i << std::endl;
}

static void BM_SetMapCString(benchmark::State& state) {
  il::Map<il::String, int> map{7};
  while (state.KeepRunning()) {
    map.setCString("aaa", 0);
    map.setCString("bbb", 0);
    map.setCString("ccc", 0);
    map.setCString("ddd", 0);
    map.setCString("eee", 0);
    map.setCString("fff", 0);
    map.setCString("ggg", 0);
  }
}

static void BM_Map(benchmark::State& state) {
  il::Map<il::String, int> map{7};
  map.set("aaa", 0);
  map.set("bbb", 0);
  map.set("ccc", 0);
  map.set("ddd", 0);
  map.set("eee", 0);
  map.set("fff", 0);
  map.set("ggg", 0);
  il::int_t i;
  while (state.KeepRunning()) {
    i += map.search("aaa");
    i += map.search("bbb");
    i += map.search("ccc");
    i += map.search("ddd");
    i += map.search("eee");
    i += map.search("fff");
    i += map.search("ggg");
  }
  std::cout << i << std::endl;
}

static void BM_SetMap(benchmark::State& state) {
  il::Map<il::String, int> map{7};
  while (state.KeepRunning()) {
    map.set("aaa", 0);
    map.set("bbb", 0);
    map.set("ccc", 0);
    map.set("ddd", 0);
    map.set("eee", 0);
    map.set("fff", 0);
    map.set("ggg", 0);
  }
}

BENCHMARK(BM_MapCString);
BENCHMARK(BM_Map);
BENCHMARK(BM_SetMapCString);
BENCHMARK(BM_SetMap);


BENCHMARK_MAIN();

