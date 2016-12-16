//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_NORM_H
#define IL_NORM_H

#include <il/math.h>
#include <il/container/1d/Array.h>
#include <il/container/2d/Array2D.h>
#include <il/container/2d/TriDiagonal.h>

namespace il {

enum class Norm { L1, L2, Linf };

template <typename T>
double norm(const il::Array<T>& x, Norm norm_type) {
  IL_ASSERT(x.size() > 0);

  auto norm = T{0};
  switch (norm_type) {
    case Norm::L1:
      for (il::int_t i{0}; i < x.size(); ++i) {
        norm += il::abs(x[i]);
      }
      break;
    case Norm::L2:
      for (il::int_t i{0}; i < x.size(); ++i) {
        norm += il::ipow<2>(x[i]);
      }
      break;
    case Norm::Linf:
      for (il::int_t i{0}; i < x.size(); ++i) {
        norm = il::max(norm, il::abs(x[i]));
      }
      break;
    default:
      IL_ASSERT(false);
  }
  return norm;
}

template <typename T>
double norm(const il::Array<T>& x, Norm norm_type, const il::Array<T>& alpha) {
  IL_ASSERT(x.size() > 0);
  IL_ASSERT(alpha.size() == x.size());

  auto norm = T{0};
  switch (norm_type) {
    case Norm::L1:
      for (il::int_t i{0}; i < x.size(); ++i) {
        norm += il::abs(x[i] / alpha[i]);
      }
      break;
    case Norm::L2:
      for (il::int_t i{0}; i < x.size(); ++i) {
        norm += il::ipow<2>(x[i] / alpha[i]);
      }
      break;
    case Norm::Linf:
      for (il::int_t i{0}; i < x.size(); ++i) {
        norm = il::max(norm, il::abs(x[i] / alpha[i]));
      }
      break;
    default:
      IL_ASSERT(false);
  }

  return norm;
}

template <typename T>
double norm(const il::Array2D<T>& A, Norm norm_type) {
  T ans{0};

  switch (norm_type) {
    case Norm::L1: {
      for (il::int_t j{0}; j < A.size(1); ++j) {
        T sum_column{0};
        for (il::int_t i{0}; i < A.size(0); ++i) {
          sum_column += il::abs(A(i, j));
        }
        ans = il::max(ans, sum_column);
      }
    } break;
    case Norm::Linf: {
      il::Array<T> sum_row{A.size(0), 0};
      for (il::int_t j{0}; j < A.size(1); ++j) {
        for (il::int_t i{0}; i < A.size(0); ++i) {
          sum_row[i] += il::abs(A(i, j));
        }
      }
      for (il::int_t i{0}; i < sum_row.size(); ++i) {
        ans = il::max(ans, sum_row[i]);
      }
    } break;
    default:
      IL_ASSERT(false);
  }

  return ans;
}

template <typename T>
double norm(const il::TriDiagonal<T>& A, Norm norm_type) {
  il::int_t n{A.size()};
  T ans{0};

  switch (norm_type) {
    case Norm::Linf: {
      il::Array<T> sum_row{n};
      sum_row[0] = il::abs(A(0, 0)) + il::abs(A(0, 1));
      for (il::int_t i{1}; i < n - 1; ++i) {
        sum_row[i] = il::abs(A(i, -1)) + il::abs(A(i, 0))+ il::abs(A(i, 1));
      }
      sum_row[n - 1] = il::abs(A(n - 1, -1)) + il::abs(A(n - 1, 0));
      for (il::int_t i{0}; i < sum_row.size(); ++i) {
        ans = il::max(ans, sum_row[i]);
      }
    } break;
    default:
      IL_ASSERT(false);
  }

  return ans;
}

}

#endif  // IL_NORM_H
