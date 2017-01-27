//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_ARRAYVIEW_H
#define IL_ARRAYVIEW_H

#include <il/base.h>

namespace il {

template <typename T>
class ConstArrayView {
 protected:
#ifdef IL_DEBUG_VISUALIZER
  il::int_t debug_size_;
#endif
  T* data_;
  T* size_;
  short align_mod_;
  short align_r_;

 public:
  /* \brief Default constructor
  // \details It creates a ConstArrayView of size 0.
  */
  ConstArrayView();

  /* \brief Construct an il::ConstArrayView<T> from a C-array (a pointer) and
  // its size
  //
  // void f(const double* p, il::int_t n) {
  //   il::ConstArrayView<double> v{p, n};
  //   ...
  // }
  */
  explicit ConstArrayView(const T* data, il::int_t n, short align_mod = 0,
                          short align_r = 0);

  /* \brief Accessor
  // \details Access (read only) the i-th element of the array view. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::ConstArrayView<double> v{p, n};
  // std::cout << v[0] << std::endl;
  */
  const T& operator[](il::int_t i) const;

  /* \brief Accessor on the last element
  // \details In debug mode, calling this method on an array of size 0 aborts
  // the program. In release mode, it will lead to undefined behavior.
  */
  const T& back() const;

  /* \brief Get the size of the array view
  //
  // il::ConstArrayView<double> v{p, n};
  // for (il::int_t k = 0; k < v.size(); ++k) {
  //   std::cout << v[k] << std::endl;
  // }
  */
  il::int_t size() const;

  /* \brief Get a pointer to const to the first element of the array view
  // \details One should use this method only when using C-style API
  */
  const T* data() const;

  /* \brief Returns a pointer to const to the first element of the array view
  */
  const T* begin() const;

  /* \brief Returns a pointer to const to the one after the last element of
  //  the array view
  */
  const T* end() const;
};

template <typename T>
ConstArrayView<T>::ConstArrayView() {
#ifdef IL_DEBUG_VISUALIZER
  debug_size_ = 0;
#endif
  data_ = nullptr;
  size_ = nullptr;
  align_mod_ = 0;
  align_r_ = 0;
}

template <typename T>
ConstArrayView<T>::ConstArrayView(const T* data, il::int_t n, short align_mod,
                                  short align_r) {
  IL_EXPECT_FAST(n >= 0);
#ifdef IL_DEBUG_VISUALIZER
  debug_size_ = n;
#endif
  data_ = const_cast<T*>(data);
  size_ = const_cast<T*>(data) + n;
  align_mod_ = 0;
  align_r_ = 0;
}

template <typename T>
const T& ConstArrayView<T>::operator[](il::int_t i) const {
  IL_EXPECT_BOUND(static_cast<std::size_t>(i) <
                  static_cast<std::size_t>(size()));
  return data_[i];
}

template <typename T>
const T& ConstArrayView<T>::back() const {
  IL_EXPECT_FAST(size() > 0);
  return size_[-1];
}

template <typename T>
il::int_t ConstArrayView<T>::size() const {
  return static_cast<il::int_t>(size_ - data_);
}

template <typename T>
const T* ConstArrayView<T>::data() const {
  return data_;
}

template <typename T>
const T* ConstArrayView<T>::begin() const {
  return data_;
}

template <typename T>
const T* ConstArrayView<T>::end() const {
  return size_;
}

////////////////////////////////////////////////////////////////////////////////

template <typename T>
class ArrayView : public ConstArrayView<T> {
 public:
  /* \brief Default constructor
  // \details It creates a ConstArrayView of size 0.
  */
  ArrayView();

  /* \brief Construct an il::ArrayView<T> from a C-array (a pointer) and
  // its size
  //
  // void f(double* p, int n) {
  //   il::ArrayView<double> v{p, n};
  //   ...
  // }
  */
  explicit ArrayView(T* data, il::int_t n, short align_mod = 0,
                     short align_r = 0);

  /* \brief Accessor
  // \details Access (read or write) the i-th element of the array view. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::ArrayView<double> v{p, n};
  // v[0] = 0.0;
  // v[n] = 0.0; // Program is aborted in debug mode and has undefined
  //             // behavior in release mode
  */
  T& operator[](il::int_t i);

  /* \brief Accessor on the last element
  // \details In debug mode, calling this method on an array of size 0 aborts
  // the program. In release mode, it will lead to undefined behavior.
  */
  T& back();

  /* \brief Get a pointer to the first element of the array view
  // \details One should use this method only when using C-style API
  */
  T* data();

  /* \brief Returns a pointer to the first element of the array view
  */
  T* begin();

  /* \brief Returns a pointer to the one after the last element of the array
  // view
  */
  T* end();
};

template <typename T>
ArrayView<T>::ArrayView()
    : ConstArrayView<T>{} {}

template <typename T>
ArrayView<T>::ArrayView(T* data, il::int_t n, short align_mod, short align_r)
    : ConstArrayView<T>{data, n, align_mod, align_r} {}

template <typename T>
T& ArrayView<T>::operator[](il::int_t i) {
  IL_EXPECT_BOUND(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(this->size()));
  return this->data_[i];
}

template <typename T>
T& ArrayView<T>::back() {
  IL_EXPECT_FAST(this->size() > 0);
  return this->size_[-1];
}

template <typename T>
T* ArrayView<T>::data() {
  return this->data_;
}

template <typename T>
T* ArrayView<T>::begin() {
  return this->data_;
}

template <typename T>
T* ArrayView<T>::end() {
  return this->size_;
}
}

#endif  // IL_ARRAYVIEW_H
