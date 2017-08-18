#include <benchmark/benchmark.h>

#include <string>
#include <unordered_map>

#include <il/Array.h>
#include <il/Map.h>
#include <il/String.h>

static void IlMapInt_Insert(benchmark::State& state) {
  const il::int_t n = 100;
  il::Array<int> v{n};
  unsigned int k = 1234567891;
  for (il::int_t i = 0; i < n; ++i) {
    v[i] = static_cast<int>(k / 2);
    k = 93u * k + 117u;
  }

  while (state.KeepRunning()) {
    il::Map<int, int> map{n};
    for (il::int_t i = 0; i < n; ++i) {
      map.insert(v[i], 0);
    }
  }
}

static void StdMapInt_Insert(benchmark::State& state) {
  const std::size_t n = 100;
  il::Array<int> v{n};
  unsigned int k = 1234567891;
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = static_cast<int>(k / 2);
    k = 93u * k + 117u;
  }

  while (state.KeepRunning()) {
    std::unordered_map<int, int> map{n};
    for (std::size_t i = 0; i < n; ++i) {
      map[v[i]] = 0;
    }
  }
}

static void IlMapString_Insert(benchmark::State& state) {
  const il::int_t n = 100;
  il::Array<il::String> v{n};
  unsigned int k = 1234567891;
  for (il::int_t i = 0; i < n; ++i) {
    v[i] = il::String{reinterpret_cast<const char*>(&k), sizeof(unsigned int)};
    k = 93u * k + 117u;
  }

  while (state.KeepRunning()) {
    il::Map<il::String, int> map{n};
    for (il::int_t i = 0; i < n; ++i) {
      map.insert(v[i], 0);
    }
  }
}

static void StdMapString_Insert(benchmark::State& state) {
  const std::size_t n = 100;
  il::Array<std::string> v{n};
  unsigned int k = 1234567891;
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = std::string{reinterpret_cast<const char*>(&k), sizeof(unsigned int)};
    k = 93u * k + 117u;
  }

  while (state.KeepRunning()) {
    std::unordered_map<std::string, int> map{n};
    for (std::size_t i = 0; i < n; ++i) {
      map[v[i]] = 0;
    }
  }
}

static void IlMapString_Search(benchmark::State& state) {
  const il::int_t n = 100;
  il::Array<il::String> v{n};
  unsigned int k = 1234567891;
  for (il::int_t i = 0; i < n; ++i) {
    v[i] = il::String{reinterpret_cast<const char*>(&k), sizeof(unsigned int)};
    k = 93u * k + 117u;
  }
  il::Map<il::String, int> map{n};
  for (il::int_t i = 0; i < n; ++i) {
    map.insert(v[i], 0);
  }

  while (state.KeepRunning()) {
    int sum = 0;
    for (il::int_t i = 0; i < n; ++i) {
      const il::int_t j = map.search(v[i]);
      sum += map.value(j);
    }
    benchmark::DoNotOptimize(&sum);
  }
}

static void StdMapString_Search(benchmark::State& state) {
  const std::size_t n = 100;
  il::Array<std::string> v{n};
  unsigned int k = 1234567891;
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = std::string{reinterpret_cast<const char*>(&k), sizeof(unsigned int)};
    k = 93u * k + 117u;
  }
  std::unordered_map<std::string, int> map{};
  for (std::size_t i = 0; i < n; ++i) {
    map[v[i]] = 0;
  }

  while (state.KeepRunning()) {
    int sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
      sum += map[v[i]];
    }
    benchmark::DoNotOptimize(&sum);
  }
}

BENCHMARK(IlMapInt_Insert);
BENCHMARK(StdMapInt_Insert);
BENCHMARK(IlMapString_Insert);
BENCHMARK(StdMapString_Insert);
BENCHMARK(IlMapString_Search);
BENCHMARK(StdMapString_Search);
