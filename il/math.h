//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_MATH_H
#define IL_MATH_H

#include <cmath>

#include <il/container/1d/Array.h>

namespace il {

template <typename T>
struct epsilon {
  static constexpr T value = 1.0;
};

template <>
struct epsilon<double> {
  static constexpr double value = 1.1102230246251565404e-16;
};

template <>
struct epsilon<long double> {
  static constexpr double value = 2.7105054312137610850e-20;
};

const double pi{3.1415926535897932385};

template <typename T>
T min(T a, T b) {
  return a <= b ? a : b;
}

template <typename T>
T min(T a, T b, T c, T d) {
  return min(min(a, b), min(c, d));
}

template <typename T>
T min(const il::Array<T>& A) {
  IL_ASSERT(A.size() > 0);
  T ans{A[0]};
  for (il::int_t k{0}; k < A.size(); ++k) {
    if (A[k] < ans) {
      ans = A[k];
    }
  }
  return ans;
}

template <typename T>
T max(T a, T b) {
  return a >= b ? a : b;
}

template <typename T>
T max(T a, T b, T c) {
  auto temp = T{max(b, c)};
  return a >= temp ? a : temp;
}

template <typename T>
T max(T a, T b, T c, T d) {
  return max(max(a, b), max(c, d));
}

template <typename T>
T max(const il::Array<T>& v) {
  IL_ASSERT(v.size() > 0);
  T max{v[0]};
  for (il::int_t k = 0; k < v.size(); ++k) {
    if (v[k] > max) {
      max = v[k];
    }
  }
  return max;
}

template <typename T>
T mean(const il::Array<T>& v) {
  IL_ASSERT(v.size() > 0);
  T ans{0};
  for (il::int_t i{0}; i < v.size(); ++i) {
    ans += v[i];
  }
  ans /= v.size();
  return ans;
}

template <typename T>
T sigma(const il::Array<T>& v) {
  IL_ASSERT(v.size() > 1);
  T mean{0};
  for (il::int_t i{0}; i < v.size(); ++i) {
    mean += v[i];
  }
  mean /= v.size();
  T sigma{0};
  for (il::int_t i{0}; i < v.size(); ++i) {
    sigma += (v[i] - mean) * (v[i] - mean);
  }
  sigma /= v.size() - 1;
  sigma = std::sqrt(sigma);
  return sigma;
}

template <typename T>
void transvection(il::Array<T>& x, il::int_t k1, T lambda, il::int_t k2) {
  x[k1] += lambda * x[k2];
}

template <typename T>
void dilation(il::Array<T>& x, il::int_t k, T lambda) {
  x[k] *= lambda;
}

template <typename T>
void swap(il::Array<T>& x, il::int_t k1, il::int_t k2) {
  T temp{x[k1]};
  x[k1] = x[k2];
  x[k2] = temp;
}

template <typename T>
T abs(T x) {
  return x >= 0 ? x : -x;
}

// Template for pow(x,N) where N is a positive il::int_t constant.
// General case, N is not a power of 2:
template <bool IsPowerOf2, il::int_t N, typename T>
class powN {
 public:
  static T p(T x) {
// Remove right-most 1-bit in binary representation of N:
#define N1 (N & (N - 1))
    return powN<(N1 & (N1 - 1)) == 0, N1, T>::p(x) *
           powN<true, N - N1, T>::p(x);
#undef N1
  }
};

// Partial template specialization for N a power of 2
template <il::int_t N, typename T>
class powN<true, N, T> {
 public:
  static T p(T x) {
    return powN<true, N / 2, T>::p(x) * powN<true, N / 2, T>::p(x);
  }
};

// Full template specialization for N = 1. This ends the recursion
template <typename T>
class powN<true, 1, T> {
 public:
  static T p(T x) { return x; }
};

// Full template specialization for N = 0
// This is used only for avoiding infinite loop if powN is
// erroneously called with IsPowerOf2 = false where it should be true.
template <typename T>
class powN<true, 0, T> {
 public:
  static T p(T x) {
    (void)x;

    return 1;
  }
};

// Function template for x to the power of N
template <il::int_t N, typename T>
static T ipow(T x) {
  // (N & N-1) == 0 if N is a power of 2
  return powN<(N & (N - 1)) == 0, N, T>::p(x);
}

template <typename T>
double ipow(T x, il::int_t n) {
  IL_ASSERT(n >= 0);
  T ans = 1;
  while (n != 0) {
    if (n & 1) {
      ans *= x;
    }
    x *= x;
    n >>= 1;
  }
  return ans;
}

inline std::uint32_t previous_power_of_2_32(std::uint32_t x) {
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x -= (x >> 1);

  return x;
}

inline std::uint32_t next_power_of_2_32(std::uint32_t x) {
  x -= 1;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x += 1;

  return x;
}

inline std::uint64_t previous_power_of_2_64(std::uint64_t x) {
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  x -= (x >> 1);

  return x;
}

inline std::uint64_t next_power_of_2_64(std::uint64_t x) {
  x -= 1;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  x += 1;

  return x;
}

// From http://chessprogramming.wikispaces.com/BitScan
static const int table_log2_64[64] = {
    63, 0,  58, 1,  59, 47, 53, 2,  60, 39, 48, 27, 54, 33, 42, 3,
    61, 51, 37, 40, 49, 18, 28, 20, 55, 30, 34, 11, 43, 14, 22, 4,
    62, 57, 46, 52, 38, 26, 32, 41, 50, 36, 17, 19, 29, 10, 13, 21,
    56, 45, 25, 31, 35, 16, 9,  12, 44, 24, 15, 8,  23, 7,  6,  5};

inline int next_log2_64(std::uint64_t x) {
  const il::int_t index = static_cast<il::int_t>(
      (next_power_of_2_64(x) * 0x07EDD5E59A4E28C2) >> 58);

  return table_log2_64[index];
}

static const int table_log2_32[32] = {
    0, 9,  1,  10, 13, 21, 2,  29, 11, 14, 16, 18, 22, 25, 3, 30,
    8, 12, 20, 28, 15, 17, 24, 7,  19, 27, 23, 6,  26, 5,  4, 31};

inline int next_log2_32(std::uint32_t x) {
  const il::int_t index =
      static_cast<il::int_t>((next_power_of_2_32(x) * 0x07C4ACDD) >> 27);

  return table_log2_64[index];
}

}

#endif  // IL_MATH_H