//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <il/linear_algebra/sparse/_test/matrix/heat.h>

namespace il {

void heat_1d(il::int_t n, il::io_t, il::SparseArray2C<double>& A,
             il::Array<double>& y) {
  il::Array<il::StaticArray<il::int_t, 2>> position{};
  position.emplace_back(il::value, std::initializer_list<il::int_t>{0, 0});
  position.emplace_back(il::value, std::initializer_list<il::int_t>{0, 1});
  for (il::int_t i = 1; i < n - 1; ++i) {
    position.emplace_back(il::value,
                          std::initializer_list<il::int_t>{i, i - 1});
    position.emplace_back(il::value, std::initializer_list<il::int_t>{i, i});
    position.emplace_back(il::value,
                          std::initializer_list<il::int_t>{i, i + 1});
  }
  position.emplace_back(il::value,
                        std::initializer_list<il::int_t>{n - 1, n - 2});
  position.emplace_back(il::value,
                        std::initializer_list<il::int_t>{n - 1, n - 1});

  il::Array<il::int_t> index{};
  A = il::SparseArray2C<double>{n, position, il::io, index};
  const double a = 1.0;
  const double b = 2.0;
  il::int_t k = 0;
  A[index[k++]] = b;
  A[index[k++]] = a;
  for (il::int_t i = 1; i < n - 1; ++i) {
    A[index[k++]] = a;
    A[index[k++]] = b;
    A[index[k++]] = a;
  }
  A[index[k++]] = a;
  A[index[k++]] = b;

  y = il::Array<double>{n};
  y[0] = a + b;
  for (il::int_t i = 1; i < n - 1; ++i) {
    y[i] = 2 * a + b;
  }
  y[n - 1] = a + b;
}

void heat_2d(il::int_t n, il::io_t, il::SparseArray2C<double>& A,
             il::Array<double>& y) {
  il::Array<il::StaticArray<il::int_t, 2>> position{};
  for (il::int_t i = 0; i < n; ++i) {
    for (il::int_t j = 0; j < n; ++j) {
      const il::int_t idx = i * n + j;
      if (i >= 1) {
        position.emplace_back(
            il::value, std::initializer_list<il::int_t>{idx, (i - 1) * n + j});
      }
      if (j >= 1) {
        position.emplace_back(
            il::value, std::initializer_list<il::int_t>{idx, i * n + (j - 1)});
      }
      position.emplace_back(il::value,
                            std::initializer_list<il::int_t>{idx, idx});
      if (j < n - 1) {
        position.emplace_back(
            il::value, std::initializer_list<il::int_t>{idx, i * n + (j + 1)});
      }
      if (i < n - 1) {
        position.emplace_back(
            il::value, std::initializer_list<il::int_t>{idx, (i + 1) * n + j});
      }
    }
  }

  il::Array<il::int_t> index{};
  A = il::SparseArray2C<double>{n * n, position, il::io, index};
  const double a = -1.0;
  const double b = 5.0;
  il::int_t k = 0;
  for (il::int_t i = 0; i < n; ++i) {
    for (il::int_t j = 0; j < n; ++j) {
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
  for (il::int_t i = 0; i < n; ++i) {
    for (il::int_t j = 0; j < n; ++j) {
      il::int_t idx = i * n + j;
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

void heat_3d(il::int_t n, il::io_t, il::SparseArray2C<double>& A,
             il::Array<double>& y) {
  il::Array<il::StaticArray<il::int_t, 2>> position{};
  for (il::int_t k = 0; k < n; ++k) {
    for (il::int_t j = 0; j < n; ++j) {
      for (il::int_t i = 0; i < n; ++i) {
        const il::int_t idx = (k * n + j) * n + i;
        if (k >= 1) {
          position.emplace_back(il::value, std::initializer_list<il::int_t>{
              idx, ((k - 1) * n + j) * n + i});
        }
        if (j >= 1) {
          position.emplace_back(il::value, std::initializer_list<il::int_t>{
              idx, (k * n + (j - 1)) * n + i});
        }
        if (i >= 1) {
          position.emplace_back(il::value, std::initializer_list<il::int_t>{
              idx, (k * n + j) * n + (i - 1)});
        }
        position.emplace_back(il::value,
                              std::initializer_list<il::int_t>{idx, idx});
        if (i < n - 1) {
          position.emplace_back(il::value, std::initializer_list<il::int_t>{
              idx, (k * n + j) * n + (i + 1)});
        }
        if (j < n - 1) {
          position.emplace_back(il::value, std::initializer_list<il::int_t>{
              idx, (k * n + (j + 1)) * n + i});
        }
        if (k < n - 1) {
          position.emplace_back(il::value, std::initializer_list<il::int_t>{
              idx, ((k + 1) * n + j) * n + i});
        }
      }
    }
  }

  il::Array<il::int_t> index{};
  A = il::SparseArray2C<double>{n * n * n, position, il::io, index};
  const double a = -1.0;
  const double b = 7.0;
  il::int_t idx = 0;
  for (il::int_t k = 0; k < n; ++k) {
    for (il::int_t j = 0; j < n; ++j) {
      for (il::int_t i = 0; i < n; ++i) {
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
  for (il::int_t k = 0; k < n; ++k) {
    for (il::int_t j = 0; j < n; ++j) {
      for (il::int_t i = 0; i < n; ++i) {
        const il::int_t idx = (k * n + j) * n + i;
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
