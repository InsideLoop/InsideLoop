//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================
//
// Compile with
// g++ -std=c++11 -O3 -mavx2 -DNDEBUG append_benchmark.cpp -o append_benchmark
//     -lpthread -lbenchmark
//

#include <vector>

#include <benchmark/benchmark.h>

#include <il/Array.h>

void Append_1_Std(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::vector<double> v;
    v.push_back(0.0);
  }
}

void Append_1_Il(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::Array<double> v{};
    v.append(0.0);
  }
}

static void Append_2_Std(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::vector<double> v;
    v.push_back(0.0);
    v.push_back(0.0);
  }
}

static void Append_2_Il(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::Array<double> v{};
    v.append(0.0);
    v.append(0.0);
  }
}

static void Append_5_Std(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::vector<double> v;
    v.push_back(0.0);
    v.push_back(0.0);
    v.push_back(0.0);
    v.push_back(0.0);
    v.push_back(0.0);
  }
}

static void Append_5_Il(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::Array<double> v{};
    v.append(0.0);
    v.append(0.0);
    v.append(0.0);
    v.append(0.0);
    v.append(0.0);
  }
}

static void Append_10_Std(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 10;
    std::vector<double> v;
    for (il::int_t i = 0; i < n; ++i) {
      v.push_back(0.0);
    }
  }
}

static void Append_10_Il(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 10;
    il::Array<double> v{};
    for (il::int_t i = 0; i < n; ++i) {
      v.append(0.0);
    }
  }
}

static void Append_100_Std(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 100;
    std::vector<double> v;
    for (il::int_t i = 0; i < n; ++i) {
      v.push_back(0.0);
    }
  }
}

static void Append_100_Il(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 100;
    il::Array<double> v{};
    for (il::int_t i = 0; i < n; ++i) {
      v.append(0.0);
    }
  }
}

static void Append_1000_Std(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 1000;
    std::vector<double> v;
    for (il::int_t i = 0; i < n; ++i) {
      v.push_back(0.0);
    }
  }
}

static void Append_1000_Il(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 1000;
    il::Array<double> v{};
    for (il::int_t i = 0; i < n; ++i) {
      v.append(0.0);
    }
  }
}

static void Append_10000_Std(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 10000;
    std::vector<double> v;
    for (il::int_t i = 0; i < n; ++i) {
      v.push_back(0.0);
    }
  }
}

static void Append_10000_Il(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 10000;
    il::Array<double> v{};
    for (il::int_t i = 0; i < n; ++i) {
      v.append(0.0);
    }
  }
}

void Append_1_int_Std(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::vector<int> v;
    v.push_back(0);
  }
}

void Append_1_int_Il(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::Array<int> v{};
    v.append(0);
  }
}

static void Append_2_int_Std(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::vector<int> v;
    v.push_back(0);
    v.push_back(0);
  }
}

static void Append_2_int_Il(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::Array<int> v{};
    v.append(0);
    v.append(0);
  }
}

static void Append_5_int_Std(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::vector<int> v;
    v.push_back(0);
    v.push_back(0);
    v.push_back(0);
    v.push_back(0);
    v.push_back(0);
  }
}

static void Append_5_int_Il(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::Array<int> v{};
    v.append(0);
    v.append(0);
    v.append(0);
    v.append(0);
    v.append(0);
  }
}

static void Append_10_int_Std(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 10;
    std::vector<int> v;
    for (il::int_t i = 0; i < n; ++i) {
      v.push_back(0);
    }
  }
}

static void Append_10_int_Il(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 10;
    il::Array<int> v{};
    for (il::int_t i = 0; i < n; ++i) {
      v.append(0);
    }
  }
}

static void Append_100_int_Std(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 100;
    std::vector<int> v;
    for (il::int_t i = 0; i < n; ++i) {
      v.push_back(0);
    }
  }
}

static void Append_100_int_Il(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 100;
    il::Array<int> v{};
    for (il::int_t i = 0; i < n; ++i) {
      v.append(0);
    }
  }
}

static void Append_1000_int_Std(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 1000;
    std::vector<int> v;
    for (il::int_t i = 0; i < n; ++i) {
      v.push_back(0);
    }
  }
}

static void Append_1000_int_Il(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 1000;
    il::Array<int> v{};
    for (il::int_t i = 0; i < n; ++i) {
      v.append(0);
    }
  }
}

static void Append_10000_int_Std(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 10000;
    std::vector<int> v;
    for (il::int_t i = 0; i < n; ++i) {
      v.push_back(0);
    }
  }
}

static void Append_10000_int_Il(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 10000;
    il::Array<int> v{};
    for (il::int_t i = 0; i < n; ++i) {
      v.append(0);
    }
  }
}

void Append_1_long_long_int_Std(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::vector<long long int> v;
    v.push_back(0);
  }
}

void Append_1_long_long_int_Il(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::Array<long long int> v{};
    v.append(0);
  }
}

static void Append_2_long_long_int_Std(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::vector<long long int> v;
    v.push_back(0);
    v.push_back(0);
  }
}

static void Append_2_long_long_int_Il(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::Array<long long int> v{};
    v.append(0);
    v.append(0);
  }
}

static void Append_5_long_long_int_Std(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::vector<long long int> v;
    v.push_back(0);
    v.push_back(0);
    v.push_back(0);
    v.push_back(0);
    v.push_back(0);
  }
}

static void Append_5_long_long_int_Il(benchmark::State& state) {
  while (state.KeepRunning()) {
    il::Array<long long int> v{};
    v.append(0);
    v.append(0);
    v.append(0);
    v.append(0);
    v.append(0);
  }
}

static void Append_10_long_long_int_Std(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 10;
    std::vector<long long int> v;
    for (il::int_t i = 0; i < n; ++i) {
      v.push_back(0);
    }
  }
}

static void Append_10_long_long_int_Il(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 10;
    il::Array<long long int> v{};
    for (il::int_t i = 0; i < n; ++i) {
      v.append(0);
    }
  }
}

static void Append_100_long_long_int_Std(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 100;
    std::vector<long long int> v;
    for (il::int_t i = 0; i < n; ++i) {
      v.push_back(0);
    }
  }
}

static void Append_100_long_long_int_Il(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 100;
    il::Array<long long int> v{};
    for (il::int_t i = 0; i < n; ++i) {
      v.append(0);
    }
  }
}

static void Append_1000_long_long_int_Std(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 1000;
    std::vector<long long int> v;
    for (il::int_t i = 0; i < n; ++i) {
      v.push_back(0);
    }
  }
}

static void Append_1000_long_long_int_Il(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 1000;
    il::Array<long long int> v{};
    for (il::int_t i = 0; i < n; ++i) {
      v.append(0);
    }
  }
}

static void Append_10000_long_long_int_Std(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 10000;
    std::vector<long long int> v;
    for (il::int_t i = 0; i < n; ++i) {
      v.push_back(0);
    }
  }
}

static void Append_10000_long_long_int_Il(benchmark::State& state) {
  while (state.KeepRunning()) {
    const il::int_t n = 10000;
    il::Array<long long int> v{};
    for (il::int_t i = 0; i < n; ++i) {
      v.append(0);
    }
  }
}

BENCHMARK(Append_1_Std);
BENCHMARK(Append_1_Il);
BENCHMARK(Append_2_Std);
BENCHMARK(Append_2_Il);
BENCHMARK(Append_5_Std);
BENCHMARK(Append_5_Il);
BENCHMARK(Append_10_Std);
BENCHMARK(Append_10_Il);
BENCHMARK(Append_100_Std);
BENCHMARK(Append_100_Il);
BENCHMARK(Append_1000_Std);
BENCHMARK(Append_1000_Il);
BENCHMARK(Append_10000_Std);
BENCHMARK(Append_10000_Il);
BENCHMARK(Append_1_int_Std);
BENCHMARK(Append_1_int_Il);
BENCHMARK(Append_2_int_Std);
BENCHMARK(Append_2_int_Il);
BENCHMARK(Append_5_int_Std);
BENCHMARK(Append_5_int_Il);
BENCHMARK(Append_10_int_Std);
BENCHMARK(Append_10_int_Il);
BENCHMARK(Append_100_int_Std);
BENCHMARK(Append_100_int_Il);
BENCHMARK(Append_1000_int_Std);
BENCHMARK(Append_1000_int_Il);
BENCHMARK(Append_10000_int_Std);
BENCHMARK(Append_10000_int_Il);
BENCHMARK(Append_1_long_long_int_Std);
BENCHMARK(Append_1_long_long_int_Il);
BENCHMARK(Append_2_long_long_int_Std);
BENCHMARK(Append_2_long_long_int_Il);
BENCHMARK(Append_5_long_long_int_Std);
BENCHMARK(Append_5_long_long_int_Il);
BENCHMARK(Append_10_long_long_int_Std);
BENCHMARK(Append_10_long_long_int_Il);
BENCHMARK(Append_100_long_long_int_Std);
BENCHMARK(Append_100_long_long_int_Il);
BENCHMARK(Append_1000_long_long_int_Std);
BENCHMARK(Append_1000_long_long_int_Il);
BENCHMARK(Append_10000_long_long_int_Std);
BENCHMARK(Append_10000_long_long_int_Il);

BENCHMARK_MAIN()
