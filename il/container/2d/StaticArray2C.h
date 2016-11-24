//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_STATICARRAY2D_H
#define IL_STATICARRAY2D_H

// <initializer_list> is needed for std::initializer_list<T>
#include <initializer_list>

#include <il/base.h>

namespace il {

template <typename T, il::int_t n0, il::int_t n1>
class StaticArray2D {
  static_assert(n0 >= 0,
                "il::StaticArray2D<T, n0, n1>: n0 must be nonnegative");
  static_assert(n1 >= 0,
                "il::StaticArray2D<T, n0, n1>: n1 must be nonnegative");

 private:
  T data_[n0 * n1 > 0 ? n0 * n1 : 1];

 public:
  /* \brief The default constructor
  // \details If T is a numeric value, the memory is
  // - (Debug mode) initialized to il::default_value<T>(). It is usually NaN
  //   if T is a floating point number or 666..666 if T is an integer.
  // - (Release mode) left uninitialized. This behavior is different from
  //   std::vector from the standard library which initializes all numeric
  //   values to 0.
  */
  StaticArray2D();

  /* \brief Construct a il::StaticArray2D<T, n0, n1> elements with a value
  /
  // // Construct a static array of 3 rows and 5 columns 5, initialized with 0.0
  // il::StaticArray2D<double, 3, 5> A{0.0};
  */
  StaticArray2D(const T& value);

  /* \brief Construct a il::StaticArray2D<T, n0, n1> from a brace-initialized
  list
  // \details In order to allow brace initialization in all cases, this
  // constructor has different syntax from the one found in std::array. You
  // must use the il::value value as a first argument. For instance:
  //
  // il::StaticArray2D<double, 2, 3> A{il::value,{2.0, 3.0, 5.0, 7.0, 8.0
  ,9.0}};
  //
  // The length of the initializer list is checked against the array length
  // in debug mode. In release mode, if the length do not match, the result
  // is undefined behavior.
  */
  StaticArray2D(il::value_t, std::initializer_list<T> list);

  /* \brief Accessor for a const il::StaticArray2D<T, n0, n1>
  // \details Access (read only) the (i, j)-th element of the array. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::StaticArray2D<double, 2, 5> A{0.0};
  // std::cout << A(0, 0) << std::endl;
  */
  const T& operator()(il::int_t i0, il::int_t i1) const;

  /* \brief Accessor for a  il::StaticArray2D<T, n0, n1>
  // \details Access (read and write) the (i, j)-th element of the array. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::StaticArray2D<double, 2, 5> A{0.0};
  // std::cout << A(0, 0) << std::endl;
  */
  T& operator()(il::int_t i0, il::int_t i1);

  /* \brief Get the size of the il::StaticArray2D<T, n0, n1>
  //
  // for (il::int_t i{0}; i < v.size(0); ++i) {
  //   for (il::int_t j{0}; j < v.size(1); ++j) {
  //     A(i, j) = 1.0 / (i + j + 2);
  //   }
  // }
  */
  il::int_t size(il::int_t d) const;

  /* \brief Get a pointer to the first element of the array for a const
  // object
  // \details One should use this method only when using C-style API
  */
  const T* data() const;

  /* \brief Get a pointer to the first element of the array
  // \details One should use this method only when using C-style API
  */
  T* data();
};

template <typename T, il::int_t n0, il::int_t n1>
StaticArray2D<T, n0, n1>::StaticArray2D() {
  if (std::is_pod<T>::value) {
#ifdef IL_DEFAULT_VALUE
    for (il::int_t k{0}; k < n0 * n1; ++k) {
      data_[k] = il::default_value<T>();
    }
#endif
  }
}

template <typename T, il::int_t n0, il::int_t n1>
StaticArray2D<T, n0, n1>::StaticArray2D(const T& value) {
  for (il::int_t k{0}; k < n0 * n1; ++k) {
    data_[k] = value;
  }
}

template <typename T, il::int_t n0, il::int_t n1>
StaticArray2D<T, n0, n1>::StaticArray2D(il::value_t,
                                        std::initializer_list<T> list) {
  IL_ASSERT(n0 * n1 == static_cast<il::int_t>(list.size()));
  for (il::int_t k{0}; k < n0 * n1; ++k) {
    data_[k] = *(list.begin() + k);
  }
};

template <typename T, il::int_t n0, il::int_t n1>
const T& StaticArray2D<T, n0, n1>::operator()(il::int_t i0,
                                              il::int_t i1) const {
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i0) < static_cast<il::uint_t>(n0));
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i1) < static_cast<il::uint_t>(n1));
  return data_[i1 * n0 + i0];
}

template <typename T, il::int_t n0, il::int_t n1>
T& StaticArray2D<T, n0, n1>::operator()(il::int_t i0, il::int_t i1) {
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i0) < static_cast<il::uint_t>(n0));
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i1) < static_cast<il::uint_t>(n1));
  return data_[i1 * n0 + i0];
}

template <typename T, il::int_t n0, il::int_t n1>
il::int_t StaticArray2D<T, n0, n1>::size(il::int_t d) const {
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(d) < static_cast<il::uint_t>(2));
  return d == 0 ? n0 : n1;
}

template <typename T, il::int_t n0, il::int_t n1>
const T* StaticArray2D<T, n0, n1>::data() const {
  return data_;
};

template <typename T, il::int_t n0, il::int_t n1>
T* StaticArray2D<T, n0, n1>::data() {
  return data_;
};
}

#endif  // IL_STATICARRAY2D_H
