//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_STATICARRAY_H
#define IL_STATICARRAY_H

// <cstring> is needed for memcpy
#include <cstring>
// <initializer_list> is needed for std::initializer_list<T>
#include <initializer_list>

#include <il/base.h>

namespace il {

template <typename T, il::int_t n>
class StaticArray {
  static_assert(n >= 0, "il::StaticArray<T, n>: n must be nonnegative");

 private:
  T data_[n > 0 ? n : 1];
  il::int_t size_ = n;

 public:
  /* \brief The default constructor
  // \details If T is a numeric value, the memory is
  // - (Debug mode) initialized to il::default_value<T>(). It is usually NaN
  //   if T is a floating point number or 666..666 if T is an integer.
  // - (Release mode) left uninitialized. This behavior is different from
  //   std::vector from the standard library which initializes all numeric
  //   values to 0.
  */
  StaticArray();

  /* \brief Construct a il::StaticArray<T, n> elements with a value
  /
  // // Construct a vector of double of length 5, initialized with 0.0
  // il::StaticArray<double, 5> v{0.0};
  */
  explicit StaticArray(const T& value);

  /* \brief Construct a il::StaticArray<T, n> from a brace-initialized list
  // \details In order to allow brace initialization in all cases, this
  // constructor has different syntax from the one found in std::array. You
  // must use the il::value value as a first argument. For instance:
  //
  // il::StaticArray<double, 4> v{il::value, {2.0, 3.14, 5.0, 7.0}};
  //
  // The length of the initializer list is checked against the vector length
  // in debug mode. In release mode, if the length do not match, the result
  // is undefined behavior.
  */
  StaticArray(il::value_t, std::initializer_list<T> list);

  /* \brief Accessor for a const il::StaticArray<T, n>
  // \details Access (read only) the i-th element of the array. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::StaticArray<il::int_t, 4> v{0};
  // std::cout << v[0] << std::endl;
  */
  const T& operator[](il::int_t i) const;

  /* \brief Accessor for a il::StaticArray<T, n>
  // \details Access (read only) the i-th element of the array. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::StaticArray<double, 4> v{};
  // v[0] = 0.0;
  // v[4] = 0.0; // Program is aborted in debug mode and has undefined
  //             // behavior in release mode
  */
  T& operator[](il::int_t i);

  /* \brief Accessor on the last element for a const il::SaticArray<T, n>
  // \details This method does not compile for empty vectors
  */
  const T& last() const;

  /* \brief Accessor on the last element
  // \details This method does not compile for empty vectors
  */
  T& last();

  /* \brief Get the size of the il::StaticArray<T, n>
  //
  // il::StaticArray<double, 4> v{};
  // for (il::int_t i{0}; i < v.size(); ++i) {
  //     v[i] = 1 / static_cast<double>(i);
  // }
  */
  il::int_t size() const;

  /* \brief Get a pointer to the first element of the array for a const
  // object
  // \details One should use this method only when using C-style API
  */
  const T* data() const;

  /* \brief Get a pointer to the first element of the array
  // \details One should use this method only when using C-style API
  */
  T* data();

  /* \brief Returns a pointer to const to the first element of the array
   */
  const T* begin() const;

  /* \brief Returns a pointer to the first element of the array
   */
  T* begin();

  /* \brief Returns a pointer to const to the one after the last element of
  //  the array
  */
  const T* end() const;

  /* \brief Returns a pointer to the one after the last element of the array
  */
  T* end();
};

template <typename T, il::int_t n>
StaticArray<T, n>::StaticArray() {
  if (std::is_pod<T>::value) {
#ifdef IL_DEFAULT_VALUE
    for (il::int_t i{0}; i < n; ++i) {
      data_[i] = il::default_value<T>();
    }
#endif
  }
}

template <typename T, il::int_t n>
StaticArray<T, n>::StaticArray(const T& value) {
  for (il::int_t i{0}; i < n; ++i) {
    data_[i] = value;
  }
}

template <typename T, il::int_t n>
StaticArray<T, n>::StaticArray(il::value_t, std::initializer_list<T> list) {
  IL_ASSERT(n == static_cast<il::int_t>(list.size()));
  if (std::is_pod<T>::value) {
    memcpy(data_, list.begin(), n * sizeof(T));
  } else {
    for (il::int_t i{0}; i < n; ++i) {
      data_[i] = *(list.begin() + i);
    }
  }
}

template <typename T, il::int_t n>
const T& StaticArray<T, n>::operator[](il::int_t i) const {
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i) < static_cast<il::uint_t>(n));
  return data_[i];
}

template <typename T, il::int_t n>
T& StaticArray<T, n>::operator[](il::int_t i) {
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i) < static_cast<il::uint_t>(n));
  return data_[i];
}

template <typename T, il::int_t n>
const T& StaticArray<T, n>::last() const {
  static_assert(n > 0,
                "il::StaticArray<T, n>: n must be positive to call last()");
  return data_[n - 1];
}

template <typename T, il::int_t n>
T& StaticArray<T, n>::last() {
  static_assert(n > 0,
                "il::StaticArray<T, n>: n must be positive to call last()");
  return data_[n - 1];
}

template <typename T, il::int_t n>
il::int_t StaticArray<T, n>::size() const {
  return n;
}

template <typename T, il::int_t n>
const T* StaticArray<T, n>::data() const {
  return data_;
}

template <typename T, il::int_t n>
T* StaticArray<T, n>::data() {
  return data_;
}

template <typename T, il::int_t n>
const T* StaticArray<T, n>::begin() const {
  return data_;
}

template <typename T, il::int_t n>
T* StaticArray<T, n>::begin() {
  return data_;
}

template <typename T, il::int_t n>
const T* StaticArray<T, n>::end() const {
  return data_ + n;
}

template <typename T, il::int_t n>
T* StaticArray<T, n>::end() {
  return data_ + n;
}
}

#endif  // IL_STATICARRAY_H
