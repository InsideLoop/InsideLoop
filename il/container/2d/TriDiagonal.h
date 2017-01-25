//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_TRIDIAGONAL_H
#define IL_TRIDIAGONAL_H

#include <il/Array.h>

namespace il {

template <typename T>
class TriDiagonal {
 private:
  il::int_t size_;
  il::Array<T> data_;

 public:
  TriDiagonal(il::int_t n) : data_{3 * n} { size_ = n; }
  const T& operator()(il::int_t i, il::int_t k) const {
    IL_EXPECT_BOUND(static_cast<il::uint_t>(i) <
        static_cast<il::uint_t>(size_));
    IL_EXPECT_BOUND(k >= -1 && k <= 1);
    return data_[size_ * (k + 1) + i];
  }
  T& operator()(il::int_t i, il::int_t k) {
    IL_EXPECT_BOUND(static_cast<il::uint_t>(i) <
                     static_cast<il::uint_t>(size_));
    IL_EXPECT_BOUND(k >= -1 && k <= 1);
    return data_[size_ * (k + 1) + i];
  }
  il::int_t size() const {
    return size_;
  }
  T* data_lower() { return data_.data() + 1; }
  T* data_diagonal() { return data_.data() + size_; }
  T* data_upper() { return data_.data() + 2 * size_; }
};

}

#endif  // IL_TRIDIAGONAL_H
