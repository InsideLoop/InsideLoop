//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_HEAT_H
#define IL_HEAT_H

#include <il/SparseArray2C.h>

namespace il {

template <typename Index>
void heat_1d(Index n, il::io_t, il::SparseArray2C<double, Index>& A,
             il::Array<double>& y) {
  il::Array<il::StaticArray<Index, 2>> position{};
  position.emplace_back(il::value, std::initializer_list<Index>{0, 0});
  position.emplace_back(il::value, std::initializer_list<Index>{0, 1});
  for (Index i = 1; i < n - 1; ++i) {
    position.emplace_back(il::value,
                          std::initializer_list<Index>{i, i - 1});
    position.emplace_back(il::value, std::initializer_list<Index>{i, i});
    position.emplace_back(il::value,
                          std::initializer_list<Index>{i, i + 1});
  }
  position.emplace_back(il::value,
                        std::initializer_list<Index>{n - 1, n - 2});
  position.emplace_back(il::value,
                        std::initializer_list<Index>{n - 1, n - 1});

  il::Array<Index> index{};
  A = il::SparseArray2C<double, Index>{n, position, il::io, index};
  const double a = 1.0;
  const double b = 2.0;
  Index k = 0;
  A[index[k++]] = b;
  A[index[k++]] = a;
  for (Index i = 1; i < n - 1; ++i) {
    A[index[k++]] = a;
    A[index[k++]] = b;
    A[index[k++]] = a;
  }
  A[index[k++]] = a;
  A[index[k++]] = b;

  y = il::Array<double>{n};
  y[0] = a + b;
  for (Index i = 1; i < n - 1; ++i) {
    y[i] = 2 * a + b;
  }
  y[n - 1] = a + b;
}

template <typename Index>
void heat_2d(Index n, il::io_t, il::SparseArray2C<double, Index>& A,
             il::Array<double>& y) {
  il::Array<il::StaticArray<Index, 2>> position{};
  for (Index i = 0; i < n; ++i) {
    for (Index j = 0; j < n; ++j) {
      const Index idx = i * n + j;
      if (i >= 1) {
        position.emplace_back(
            il::value, std::initializer_list<Index>{idx, (i - 1) * n + j});
      }
      if (j >= 1) {
        position.emplace_back(
            il::value, std::initializer_list<Index>{idx, i * n + (j - 1)});
      }
      position.emplace_back(il::value,
                            std::initializer_list<Index>{idx, idx});
      if (j < n - 1) {
        position.emplace_back(
            il::value, std::initializer_list<Index>{idx, i * n + (j + 1)});
      }
      if (i < n - 1) {
        position.emplace_back(
            il::value, std::initializer_list<Index>{idx, (i + 1) * n + j});
      }
    }
  }

  il::Array<Index> index{};
  A = il::SparseArray2C<double, Index>{n * n, position, il::io, index};
  const double a = -1.0;
  const double b = 5.0;
  Index k = 0;
  for (Index i = 0; i < n; ++i) {
    for (Index j = 0; j < n; ++j) {
      if (i >= 1) {
        A[k++] = a;
      }
      if (j >= 1) {
        A[k++] = a;
      }
      A[k++] = b;
      if (j < n - 1) {
        A[k++] = a;
      }
      if (i < n - 1) {
        A[k++] = a;
      }
    }
  }

  y = il::Array<double>{n * n};
  for (Index i = 0; i < n; ++i) {
    for (Index j = 0; j < n; ++j) {
      Index idx = i * n + j;
      double value = 0.0;
      if (i >= 1) {
        value += a;
      }
      if (j >= 1) {
        value += a;
      }
      value += b;
      if (j < n - 1) {
        value += a;
      }
      if (i < n - 1) {
        value += a;
      }
      y[idx] = value;
    }
  }
}

template <typename Index>
void heat_3d(Index n, il::io_t, il::SparseArray2C<double, Index>& A,
             il::Array<double>& y) {
  il::Array<il::StaticArray<Index, 2>> position{};
  for (Index k = 0; k < n; ++k) {
    for (Index j = 0; j < n; ++j) {
      for (Index i = 0; i < n; ++i) {
        const Index idx = (k * n + j) * n + i;
        if (k >= 1) {
          position.emplace_back(il::value, std::initializer_list<Index>{
              idx, ((k - 1) * n + j) * n + i});
        }
        if (j >= 1) {
          position.emplace_back(il::value, std::initializer_list<Index>{
              idx, (k * n + (j - 1)) * n + i});
        }
        if (i >= 1) {
          position.emplace_back(il::value, std::initializer_list<Index>{
              idx, (k * n + j) * n + (i - 1)});
        }
        position.emplace_back(il::value,
                              std::initializer_list<Index>{idx, idx});
        if (i < n - 1) {
          position.emplace_back(il::value, std::initializer_list<Index>{
              idx, (k * n + j) * n + (i + 1)});
        }
        if (j < n - 1) {
          position.emplace_back(il::value, std::initializer_list<Index>{
              idx, (k * n + (j + 1)) * n + i});
        }
        if (k < n - 1) {
          position.emplace_back(il::value, std::initializer_list<Index>{
              idx, ((k + 1) * n + j) * n + i});
        }
      }
    }
  }

  il::Array<Index> index{};
  A = il::SparseArray2C<double, Index>{n * n * n, position, il::io, index};
  const double a = -1.0;
  const double b = 7.0;
  Index idx = 0;
  for (Index k = 0; k < n; ++k) {
    for (Index j = 0; j < n; ++j) {
      for (Index i = 0; i < n; ++i) {
        if (k >= 1) {
          A[idx++] = a;
        }
        if (j >= 1) {
          A[idx++] = a;
        }
        if (i >= 1) {
          A[idx++] = a;
        }
        A[idx++] = b;
        if (i < n - 1) {
          A[idx++] = a;
        }
        if (j < n - 1) {
          A[idx++] = a;
        }
        if (k < n - 1) {
          A[idx++] = a;
        }
      }
    }
  }

  y = il::Array<double>{n * n * n};
  for (Index k = 0; k < n; ++k) {
    for (Index j = 0; j < n; ++j) {
      for (Index i = 0; i < n; ++i) {
        const Index idx = (k * n + j) * n + i;
        double value = 0.0;
        if (k >= 1) {
          value += a;
        }
        if (j >= 1) {
          value += a;
        }
        if (i >= 1) {
          value += a;
        }
        value += b;
        if (i < n - 1) {
          value += a;
        }
        if (j < n - 1) {
          value += a;
        }
        if (k < n - 1) {
          value += a;
        }
        y[idx] = value;
      }
    }
  }
}

}

#endif  // IL_HEAT_H
