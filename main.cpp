//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <random>
#include <unordered_map>

#include <il/HashMap.h>
#include <il/Timer.h>
#include <il/format.h>

#include <llvm/ADT/DenseMap.h>
#include <google/dense_hash_map>

#define MYINT il::int_t

int main() {
  il::Timer timer{};
  const MYINT n = 1 << 26;
//  const MYINT n = 10;
  const bool inside_loop = true;
  MYINT sum_il = 0;
  const bool llvm = true;
  MYINT sum_llvm = 0;
  const bool google = true;
  MYINT sum_google = 0;
  const bool stl = true;
  MYINT sum_stl = 0;

  std::default_random_engine engine{};
  std::uniform_int_distribution<il::int_t> dist{
      0, std::numeric_limits<int>::max() / 4};
  il::Array<il::int_t> v{n};
  for (MYINT k = 0; k < n; ++k) {
    v[k] = dist(engine);
  }
//  il::print("{:>20}: {:>15n}\n", "n", n);

  //////////////////////////////////////////////////////////////////////////////
  // Timing LLVM
  //////////////////////////////////////////////////////////////////////////////
  if (llvm) {
    timer.start();
    llvm::DenseMap<il::int_t, il::int_t> map_llvm(n);
    timer.stop();
    il::print("{:>20}: {:>15.2f} ns\n", "Time init LLVM",
              1.0e9 * timer.elapsed() / n);
    timer.reset();

    timer.start();
    for (MYINT k = 0; k < n; ++k) {
      const il::int_t key = v[k];
      map_llvm[key] = k;
    }
    timer.stop();

    il::print("{:>20}: {:>15.2f} ns\n", "Time insert LLVM",
              1.0e9 * timer.elapsed() / n);

    timer.reset();
    timer.start();
    for (MYINT k = 0; k < n; ++k) {
      const il::int_t key = v[k];
      sum_llvm += map_llvm[key];
    }
    timer.stop();

    il::print("{:>20}: {:>15.2f} ns\n", "Time search LLVM",
              1.0e9 * timer.elapsed() / n);
    timer.reset();
  }


  //////////////////////////////////////////////////////////////////////////////
  // Timing Inside Loop
  //////////////////////////////////////////////////////////////////////////////
  if (inside_loop) {
    timer.start();
    il::HashMap<il::int_t, il::int_t> map{n};
    timer.stop();
    il::print("{:>20}: {:>15.2f} ns\n", "Time init IL",
              1.0e9 * timer.elapsed() / n);
    timer.reset();

    timer.start();
    for (MYINT k = 0; k < n; ++k) {
      const il::int_t key = v[k];
      MYINT i = map.search(key);
      if (!map.found(i)) {
        map.insert(key, k, il::io, i);
      } else {
        map.value(i) = k;
      }
    }
    timer.stop();

    il::print("{:>20}: {:>15.2f} ns\n", "Time insert IL",
              1.0e9 * timer.elapsed() / n);

    timer.reset();
    timer.start();
    for (MYINT k = 0; k < n; ++k) {
      const il::int_t key = v[k];
      const MYINT i = map.search(key);
      sum_il += map.value(i);
    }
    timer.stop();

    il::print("{:>20}: {:>15.2f} ns\n", "Time search IL",
              1.0e9 * timer.elapsed() / n);
    il::print("{:>20}: {:>15n}\n", "Number of slots", map.capacity());
    timer.reset();
  }

  //////////////////////////////////////////////////////////////////////////////
  // Timing Google Dense Hash
  //////////////////////////////////////////////////////////////////////////////
  if (google) {
    timer.start();
    google::dense_hash_map<il::int_t, il::int_t> map_google{
        static_cast<std::size_t>(n)};
    map_google.set_empty_key(std::numeric_limits<il::int_t>::max());
    for (MYINT k = 0; k < n; ++k) {
      const il::int_t key = v[k];
      map_google[key] = k;
    }
    timer.stop();

    il::print("{:>20}: {:>15.2f} ns\n", "Time insert Google",
              1.0e9 * timer.elapsed() / n);

    timer.reset();
    timer.start();
    for (MYINT k = 0; k < n; ++k) {
      const il::int_t key = v[k];
      sum_google += map_google[key];
    }
    timer.stop();

    il::print("{:>20}: {:>15.2f} ns\n", "Time search Google",
              1.0e9 * timer.elapsed() / n);
//    il::print("{:>20}: {:>15n}\n", "Number of slots",
//              map_google.bucket_count());
    timer.reset();
  }

  //////////////////////////////////////////////////////////////////////////////
  // Timing STL
  //////////////////////////////////////////////////////////////////////////////
  if (stl) {
    timer.start();
    std::unordered_map<il::int_t, il::int_t> map_stl{};
    for (MYINT k = 0; k < n; ++k) {
      const il::int_t key = v[k];
      map_stl[key] = k;
    }
    timer.stop();

    il::print("{:>20}: {:>15.2f} ns\n", "Time insert STL",
              1.0e9 * timer.elapsed() / n);

    timer.reset();
    timer.start();
    for (MYINT k = 0; k < n; ++k) {
      const il::int_t key = v[k];
      sum_stl += map_stl[key];
    }
    timer.stop();

    il::print("{:>20}: {:>15.2f} ns\n", "Time search STL",
              1.0e9 * timer.elapsed() / n);
    timer.reset();
  }

  il::print("\n{} {} {}\n", sum_il, sum_google, sum_stl);

  return 0;
}