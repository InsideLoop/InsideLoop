//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_BANDARRAY2C_H
#define IL_BANDARRAY2C_H

#include <il/Array2C.h>

namespace il {

template <typename T>
class BandArray2C {
 private:
  il::int_t n0_;
  il::int_t n1_;
  il::int_t width_left_;
  il::int_t width_right_;
  il::int_t capacity_right_;
  il::Array2C<T> element_;

 public:
  BandArray2C(il::int_t n, il::int_t width);
  BandArray2C(il::int_t n0, il::int_t n1, il::int_t width_left,
              il::int_t width_right);
  const T& operator()(il::int_t i0, il::int_t k) const;
  T& operator()(il::int_t i0, il::int_t k);
  il::int_t size(il::int_t d) const;
  il::int_t widthLeft() const;
  il::int_t widthRight() const;
  il::int_t capacityRight() const;
  T* data();
};

template <typename T>
BandArray2C<T>::BandArray2C(il::int_t n, il::int_t width)
    : element_{n, 1 + 3 * width, 0} {
  n0_ = n;
  n1_ = n;
  width_left_ = width;
  width_right_ = width;
  capacity_right_ = 2 * width;
}

template <typename T>
BandArray2C<T>::BandArray2C(il::int_t n0, il::int_t n1, il::int_t width_left,
                            il::int_t width_right)
    : element_{n0, 1 + 2 * width_left + width_right, 0} {
  n0_ = n0;
  n1_ = n1;
  width_left_ = width_left;
  width_right_ = width_right;
  capacity_right_ = width_left + width_right;
}

template <typename T>
const T& BandArray2C<T>::operator()(il::int_t i0, il::int_t k) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) <
                   static_cast<std::size_t>(size(0)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0 + k) <
                   static_cast<std::size_t>(size(1)));
  IL_EXPECT_MEDIUM(k <= width_right_);
  IL_EXPECT_MEDIUM(k >= -width_left_);
  return element_(i0, width_left_ + k);
}

template <typename T>
T& BandArray2C<T>::operator()(il::int_t i0, il::int_t k) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) <
                   static_cast<std::size_t>(size(0)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0 + k) <
                   static_cast<std::size_t>(size(1)));
  IL_EXPECT_MEDIUM(k <= width_right_);
  IL_EXPECT_MEDIUM(k >= -width_left_);
  return element_(i0, width_left_ + k);
}

template <typename T>
il::int_t BandArray2C<T>::size(il::int_t d) const {
  return d == 0 ? n0_ : n1_;
}

template <typename T>
il::int_t BandArray2C<T>::widthLeft() const {
  return width_left_;
}

template <typename T>
il::int_t BandArray2C<T>::widthRight() const {
  return width_right_;
}

template <typename T>
il::int_t BandArray2C<T>::capacityRight() const {
  return capacity_right_;
}

template <typename T>
T* BandArray2C<T>::data() {
  return element_.data();
}
}  // namespace il

#endif  // IL_BANDARRAY2C_H
