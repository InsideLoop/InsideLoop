//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <cstdio>
#include <random>
#include <unordered_map>

#include <il/HashMap.h>
#include <il/Timer.h>

#include <google/dense_hash_map>

int main() {
  // il::int_t n = 50000000;
  il::int_t n = 10;

  std::default_random_engine engine{};
  std::uniform_int_distribution<il::int_t> dist{
      0, std::numeric_limits<il::int_t>::max() / 2};
  il::Array<il::int_t> v{n};
  for (il::int_t k = 0; k < n; ++k) {
    v[k] = dist(engine);
  }

  il::Timer timer{};
  timer.start();
  il::HashMap<il::int_t, il::int_t> map{n};
  for (il::int_t k = 0; k < n; ++k) {
    const il::int_t key = v[k];
    il::int_t i = map.search(key);
    if (!map.found(i)) {
      map.insert(key, k, il::io, i);
    } else {
      map.value(i) = k;
    }
  }
  timer.stop();

  std::printf("      Time insert IL: %7.3f\n", timer.elapsed());

  timer.reset();
  timer.start();
  il::int_t sum = 0;
  for (il::int_t k = 0; k < n; ++k) {
    const il::int_t key = v[k];
    const il::int_t i = map.search(key);
    sum += map.value(i);
  }
  timer.stop();

  std::printf("      Time search IL: %7.3f\n", timer.elapsed());

  timer.reset();
  timer.start();
  google::dense_hash_map<il::int_t, il::int_t> map_google{
      static_cast<std::size_t>(n)};
  map_google.set_empty_key(std::numeric_limits<il::int_t>::max());
  for (il::int_t k = 0; k < n; ++k) {
    const il::int_t key = v[k];
    map_google[key] = k;
  }
  timer.stop();

  std::printf("  Time insert Google: %7.3f\n", timer.elapsed());

  timer.reset();
  timer.start();
  il::int_t sum_google = 0;
  for (il::int_t k = 0; k < n; ++k) {
    const il::int_t key = v[k];
    sum_google += map_google[key];
  }
  timer.stop();

  std::printf("  Time search Google: %7.3f\n", timer.elapsed());

  timer.reset();
  timer.start();
  std::unordered_map<il::int_t, il::int_t> map_stl{};
  for (il::int_t k = 0; k < n; ++k) {
    const il::int_t key = v[k];
    map_stl[key] = k;
  }
  timer.stop();

  std::printf("     Time insert STL: %7.3f\n", timer.elapsed());

  timer.reset();
  timer.start();
  il::int_t sum_stl = 0;
  for (il::int_t k = 0; k < n; ++k) {
    const il::int_t key = v[k];
    sum_stl += map_stl[key];
  }
  timer.stop();

  std::printf("     Time search STL: %7.3f\n", timer.elapsed());
  std::printf("Sum: %d %d\n", static_cast<int>(sum),
              static_cast<int>(sum_google));

  return 0;
}
