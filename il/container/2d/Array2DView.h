//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_ARRAY2DVIEW_H
#define IL_ARRAY2DVIEW_H

#include <il/base.h>

namespace il {

template <typename T>
class ConstArray2DView {
 protected:
#ifdef IL_DEBUG_VISUALIZER
  il::int_t debug_size_0_;
  il::int_t debug_size_1_;
  il::int_t debug_stride_;
#endif
  T* data_;
  T* size_[2];
  T* stride_;
  short align_mod_;
  short align_r_;

 public:
  /* \brief Default constructor
  // \details It creates a ConstArray2DView of 0 rows and 0 columns.
  */
  ConstArray2DView();

  /* \brief Construct an il::ConstArray2DView<T> from a
  C-array (a pointer) and
  // its size and the stride
  //
  // void f(const double* data, il::int_t n, il::int_t p, il::int_t stride) {
  //   il::ConstArray2DView<double> A{data, n, p, stride};
  //   ...
  // }
  */
  ConstArray2DView(const T* data, il::int_t n0, il::int_t n1, il::int_t stride,
                   short align_mod, short align_r);

  /* \brief Accessor
  // \details Access (read only) the (i, j)-th element of the array view. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::ConstArray2DView<double> A{data, n, p};
  // std::cout << v(0, 0) << std::endl;
  */
  const T& operator()(il::int_t i0, il::int_t i1) const;

  /* \brief Get the size of the array view
  //
  // il::ConstArrayView<double> v{data, n, p, p};
  // for (il::int_t i = 0; i < v.size(0); ++i) {
  //   for (il::int_t j = 0; j < v.size(1); ++j) {
  //     A(i, j) = 1.0 / (i + j + 2);
  //   }
  // }
  */
  il::int_t size(il::int_t d) const;

  /* \brief Get a pointer to const to the first element of the array
  // \details One should use this method only when using C-style API
  */
  const T* data() const;

  /* \brief Memory distance (in sizeof(T)) in between A(i, j) and A(i + 1, j)
  */
  il::int_t stride(il::int_t d) const;
};

template <typename T>
ConstArray2DView<T>::ConstArray2DView() {
#ifdef IL_DEBUG_VISUALIZER
  debug_size_0_ = 0;
  debug_size_1_ = 0;
  debug_stride_ = 0;
#endif
  data_ = nullptr;
  size_[0] = nullptr;
  size_[1] = nullptr;
  stride_ = nullptr;
  align_mod_ = 0;
  align_r_ = 0;
}

template <typename T>
ConstArray2DView<T>::ConstArray2DView(const T* data, il::int_t n0, il::int_t n1,
                                      il::int_t stride, short align_mod,
                                      short align_r) {
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);
  IL_EXPECT_FAST(stride > 0);
  if (n0 > 0 && n1 > 0) {
    data_ = const_cast<T*>(data);
#ifdef IL_DEBUG_VISUALIZER
    debug_size_0_ = n0;
    debug_size_1_ = n1;
    debug_stride_ = stride;
#endif
    size_[0] = data_ + n0;
    size_[1] = data_ + n1;
    stride_ = data_ + stride;
    align_mod_ = align_mod;
    align_r_ = align_r;
  } else {
#ifdef IL_DEBUG_VISUALIZER
    debug_size_0_ = 0;
    debug_size_1_ = 0;
    debug_stride_ = 0;
#endif
    data_ = nullptr;
    size_[0] = nullptr;
    size_[1] = nullptr;
    stride_ = nullptr;
    align_mod_ = 0;
    align_r_ = 0;
  }
}

template <typename T>
const T& ConstArray2DView<T>::operator()(il::int_t i0, il::int_t i1) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) <
                   static_cast<std::size_t>(size(0)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) <
                   static_cast<std::size_t>(size(1)));
  return data_[i1 * (stride_ - data_) + i0];
}

template <typename T>
il::int_t ConstArray2DView<T>::size(il::int_t d) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(d) < static_cast<std::size_t>(2));
  return static_cast<il::int_t>(size_[d] - data_);
}

template <typename T>
const T* ConstArray2DView<T>::data() const {
  return data_;
}

template <typename T>
il::int_t ConstArray2DView<T>::stride(il::int_t d) const {
  IL_EXPECT_FAST(static_cast<std::size_t>(d) < static_cast<std::size_t>(2));
  return (d == 0) ? 1 : static_cast<il::int_t>(stride_ - data_);
}

////////////////////////////////////////////////////////////////////////////////

template <typename T>
class Array2DView : public ConstArray2DView<T> {
 public:
  /* \brief Default constructor
  // \details It creates a Array2DView of 0 rows and 0 columns.
  */
  Array2DView();

  /* \brief Construct an il::Array2DView<T> from a C-array (a
  pointer) and
  // its size and the stride
  //
  // void f(const double* data, il::int_t n, il::int_t, il::int_t stride) {
  //   il::Array2DView<double> A{data, n, p, stride};
  //   ...
  // }
  */
  Array2DView(T* data, il::int_t n0, il::int_t n1, il::int_t stride,
              short align_mod, short align_r);

  /* \brief Accessor
  // \details Access (read and write) the (i, j)-th element of the array view.
  // Bound checking is done in debug mode but not in release mode.
  */
  T& operator()(il::int_t i0, il::int_t i1);

  /* \brief Get a pointer to const to the first element of the array
  // \details One should use this method only when using C-style API
  */
  T* data();
};

template <typename T>
Array2DView<T>::Array2DView()
    : ConstArray2DView<T>{} {}

template <typename T>
Array2DView<T>::Array2DView(T* data, il::int_t n0, il::int_t n1,
                            il::int_t stride, short align_mod, short align_r)
    : ConstArray2DView<T>{data, n0, n1, stride, align_mod, align_r} {}

template <typename T>
T& Array2DView<T>::operator()(il::int_t i0, il::int_t i1) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) <
                   static_cast<std::size_t>(this->size(0)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) <
                   static_cast<std::size_t>(this->size(1)));
  return this->data_[i1 * (this->stride_ - this->data_) + i0];
}

template <typename T>
T* Array2DView<T>::data() {
  return this->data_;
}
}

#endif  // IL_ARRAY2DVIEW_H
