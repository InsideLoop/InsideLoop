//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_ALIGNMENT_H
#define IL_ALIGNMENT_H

#include <cstdio>

#include <il/Array.h>
#include <il/Timer.h>

namespace il {

inline void alignment() {
  const il::int_t n = 8000;
  const il::int_t nb_loops = 10000000;

  {
    il::Array<char> a{n, 0};
    il::Array<char> b{n, 1};
    il::Array<char> c{n, 0};
    il::Timer timer{};
    for (il::int_t k = 0; k < nb_loops; ++k) {
      for (il::int_t i = 0; i < n; ++i) {
        b[i] = a[i] + b[i] + c[i];
      }
    }
    timer.stop();
    std::printf("Unaligned: %7.3e ns\n", timer.elapsed());
  }

  {
    il::Array<char> a{n, 0, il::align, 32};
    il::Array<char> b{n, 1, il::align, 32};
    il::Array<char> c{n, 0, il::align, 32};
    char* const a_data{a.data()};
    char* const b_data{b.data()};
    char* const c_data{c.data()};
    il::Timer timer{};
    for (il::int_t k = 0; k < nb_loops; ++k) {
#pragma omp simd aligned(a_data, b_data, c_data : IL_SIMD)
      for (il::int_t i = 0; i < n; ++i) {
        b_data[i] = a_data[i] + b_data[i] + c_data[i];
      }
    }
    timer.stop();
    std::printf("SIMD aligned: %7.3e ns\n", timer.elapsed());
  }

  {
    il::Array<char> a{n, 0, il::align, 32};
    il::Array<char> b{n, 1, il::align, 32};
    il::Array<char> c{n, 0, il::align, 32};
    char* const a_data{a.data()};
    char* const b_data{b.data()};
    char* const c_data{c.data()};
    il::Timer timer{};
    for (il::int_t k = 0; k < nb_loops; ++k) {
#pragma omp simd aligned(a_data, b_data, c_data : IL_CACHE)
      for (il::int_t i = 0; i < n; ++i) {
        b_data[i] = a_data[i] + b_data[i] + c_data[i];
      }
    }
    timer.stop();
    std::printf("Cache aligned: %7.3e ns\n", timer.elapsed());
  }
}

inline void conditional_assignment() {
  const il::int_t n = 12000;
  const il::int_t nb_loops = 100000;

  {
    il::Array<unsigned char> a{n};
    for (il::int_t i = 0; i < n; ++i) {
      a[i] = (i % 2 == 0) ? 10 : 245;
    }
    il::Array<unsigned char> b{n, 0};

    unsigned char* a_data{a.data()};
    unsigned char* b_data{b.data()};
    il::Timer timer{};
    for (il::int_t k = 0; k < nb_loops; ++k) {
#pragma omp simd
      for (il::int_t i = 0; i < n; ++i) {
        if (a_data[i] < 128) {
          b_data[i] = 128;
        }
      }
    }
    timer.stop();
    std::printf("Unaligned timing: %7.3e\n", timer.elapsed());
  }

  {
    il::Array<unsigned char> a{n, il::align, 32};
    for (il::int_t i = 0; i < n; ++i) {
      a[i] = (i % 2 == 0) ? 10 : 245;
    }
    il::Array<unsigned char> b{n, il::align, 32};

    unsigned char* a_data{a.data()};
    unsigned char* b_data{b.data()};
    il::Timer timer{};
    for (il::int_t k = 0; k < nb_loops; ++k) {
#pragma omp simd aligned(a_data, b_data : IL_SIMD)
      for (il::int_t i = 0; i < n; ++i) {
        if (a_data[i] < 128) {
          b_data[i] = 128;
        }
      }
    }
    timer.stop();
    std::printf("SIMD aligned timing: %7.3e\n", timer.elapsed());
  }

  {
    il::Array<unsigned char> a{n, il::align, il::cacheline};
    for (il::int_t i = 0; i < n; ++i) {
      a[i] = (i % 2 == 0) ? 10 : 245;
    }
    il::Array<unsigned char> b{n, il::align, il::cacheline};

    unsigned char* a_data{a.data()};
    unsigned char* b_data{b.data()};
    il::Timer timer{};
    for (il::int_t k = 0; k < nb_loops; ++k) {
#pragma omp simd aligned(a_data, b_data : IL_CACHE)
      for (il::int_t i = 0; i < n; ++i) {
        if (a_data[i] < 128) {
          b_data[i] = 128;
        }
      }
    }
    timer.stop();
    std::printf("Cache aligned timing: %7.3e\n", timer.elapsed());
  }
}
}  // namespace il

#endif  // IL_ALIGNMENT_H
