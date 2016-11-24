//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_ARRAY3DVIEW_H
#define IL_ARRAY3DVIEW_H

#include <il/base.h>

namespace il {

template <typename T>
class ConstArray3DView {
 protected:
#ifndef NDEBUG
  il::int_t debug_size_0_;
  il::int_t debug_size_1_;
  il::int_t debug_size_2_;
  il::int_t debug_stride_0_;
  il::int_t debug_stride_1_;
#endif
  T* data_;
  T* size_[3];
  T* stride_[2];

 public:
  /* \brief Default constructor
  // \details It creates a ConstArray3DView of 0 rows, 0 columns and 0 slices
  */
  ConstArray3DView();

  /* \brief Construct an il::ConstArray3DView<T> from a C-array (a pointer) and
  // its size and the stride
  //
  // il::ConstArray3DView<double> A{data, n, p, q, stride_0, stride_1};
  */
  ConstArray3DView(const T* data, il::int_t n, il::int_t p, il::int_t q,
                   il::int_t stride_0, il::int_t stride_1);

  /* \brief Accessor
  // \details Access (read only) the (i, j, k)-th element of the array view.
  // Bound  checking is done in debug mode but not in release mode.
  //
  // il::ConstArray3DView<double> A{data, n, p, q, p, q};
  // std::cout << v(0, 0, 0) << std::endl;
  */
  const T& operator()(il::int_t i, il::int_t j, il::int_t k) const;

  /* \brief Get the size of the array view
  //
  // il::ConstArray3DView<double> v{data, n, p, q, p, q};
  // for (il::int_t i{0}; i < v.size(0); ++i) {
  //   for (il::int_t j{0}; j < v.size(1); ++j) {
  //     for (il::int_t k{0}; k < v.size(2); ++k) {
  //       A(i, j, k) = 1.0 / (i + j + k + 3);
  //     }
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
ConstArray3DView<T>::ConstArray3DView() {
#ifndef NDEBUG
  debug_size_0_ = 0;
  debug_size_1_ = 0;
  debug_size_2_ = 0;
  debug_stride_0_ = 0;
  debug_stride_1_ = 0;
#endif
  data_ = nullptr;
  size_[0] = nullptr;
  size_[1] = nullptr;
  size_[2] = nullptr;
  stride_[0] = nullptr;
  stride_[1] = nullptr;
}

template <typename T>
ConstArray3DView<T>::ConstArray3DView(const T* data, il::int_t n, il::int_t p,
                                      il::int_t q, il::int_t stride_0,
                                      il::int_t stride_1) {
  IL_ASSERT(n >= 0);
  IL_ASSERT(p >= 0);
  IL_ASSERT(q >= 0);
  IL_ASSERT(stride_0 >= 0);
  IL_ASSERT(stride_1 >= 0);
  data_ = const_cast<T*>(data);
#ifndef NDEBUG
  debug_size_0_ = n;
  debug_size_1_ = p;
  debug_size_2_ = q;
  debug_stride_0_ = stride_0;
  debug_stride_1_ = stride_1;
#endif
  size_[0] = data_ + n;
  size_[1] = data_ + p;
  size_[2] = data_ + q;
  stride_[0] = data_ + stride_0;
  stride_[1] = data_ + stride_1;
}

template <typename T>
const T& ConstArray3DView<T>::operator()(il::int_t i, il::int_t j,
                                         il::int_t k) const {
  IL_ASSERT(static_cast<il::uint_t>(i) < static_cast<il::uint_t>(size(0)));
  IL_ASSERT(static_cast<il::uint_t>(j) < static_cast<il::uint_t>(size(1)));
  IL_ASSERT(static_cast<il::uint_t>(k) < static_cast<il::uint_t>(size(2)));
  return data_[(i * stride(0) + j) * stride(1) + k];
}

template <typename T>
il::int_t ConstArray3DView<T>::size(il::int_t d) const {
  IL_ASSERT(static_cast<il::uint_t>(d) < static_cast<il::uint_t>(3));
  return static_cast<il::int_t>(size_[d] - data_);
}

template <typename T>
const T* ConstArray3DView<T>::data() const {
  return data_;
}

template <typename T>
il::int_t ConstArray3DView<T>::stride(il::int_t d) const {
  IL_ASSERT(static_cast<il::uint_t>(d) < static_cast<il::uint_t>(2));
  return static_cast<il::int_t>(stride_[d] - data_);
}

////////////////////////////////////////////////////////////////////////////////

template <typename T>
class Array3DView : public ConstArray3DView<T> {
 public:
  /* \brief Default constructor
  // \details It creates a Array3DView of 0 rows, 0 columns and 0 slices
  */
  Array3DView();

  /* \brief Construct an il::Array3DView<T> from a C-array (a pointer) and
  // its size and the stride
  //
  // il::Array3DView<double> A{data, n, p, q, stride_0, stride_1};
  */
  Array3DView(T* data, il::int_t n, il::int_t p, il::int_t q,
              il::int_t stride_0, il::int_t stride_1);

  /* \brief Accessor
  // \details Access (read and write) the (i, j, k)-th element of the array
  // view. Bound  checking is done in debug mode but not in release mode.
  //
  // il::Array3DView<double> A{data, n, p, q, p, q};
  // std::cout << v(0, 0, 0) << std::endl;
  */
  T& operator()(il::int_t i, il::int_t j, il::int_t k);

  /* \brief Get a pointer to const to the first element of the array
  // \details One should use this method only when using C-style API
  */
  T* data();
};

template <typename T>
Array3DView<T>::Array3DView()
    : ConstArray3DView<T>{} {}

template <typename T>
Array3DView<T>::Array3DView(T* data, il::int_t n, il::int_t p, il::int_t q,
                            il::int_t stride_0, il::int_t stride_1)
    : ConstArray3DView<T>{data, n, p, q, stride_0, stride_1} {}

template <typename T>
T& Array3DView<T>::operator()(il::int_t i, il::int_t j, il::int_t k) {
  IL_ASSERT(static_cast<il::uint_t>(i) <
            static_cast<il::uint_t>(this->size(0)));
  IL_ASSERT(static_cast<il::uint_t>(j) <
            static_cast<il::uint_t>(this->size(1)));
  IL_ASSERT(static_cast<il::uint_t>(k) <
            static_cast<il::uint_t>(this->size(2)));
  return this->data_[(i * this->stride(0) + j) * this->stride(1) + k];
}

template <typename T>
T* Array3DView<T>::data() {
  return this->data_;
}
}

#endif  // IL_ARRAY3DVIEW_H
