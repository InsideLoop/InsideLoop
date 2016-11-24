//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_STATICARRAY3D_H
#define IL_STATICARRAY3D_H

// <initializer_list> is needed for std::initializer_list<T>
#include <initializer_list>

#include <il/container/3d/Array3DView.h>

namespace il {

template <class T, il::int_t n, il::int_t p, il::int_t q>
class StaticArray3D {
  static_assert(n >= 0, "il::StaticArray3D<T, n, p, q>: n must be nonnegative");
  static_assert(p >= 0, "il::StaticArray3D<T, n, p, q>: p must be nonnegative");
  static_assert(q >= 0, "il::StaticArray3D<T, n, p, q>: q must be nonnegative");

 private:
  T data_[n * p * q > 0 ? n* p* q : 1];

 public:
  /* \brief The default constructor
  // \details If T is a numeric value, the memory is
  // - (Debug mode) initialized to il::default_value<T>(). It is usually NaN
  //   if T is a floating point number or 666..666 if T is an integer.
  // - (Release mode) left uninitialized. This behavior is different from
  //   std::vector from the standard library which initializes all numeric
  //   values to 0.
  */
  StaticArray3D();

  /* \brief Construct a il::StaticArray3D<T, n, p, q> elements with a value
  /
  // // Construct a static array of 3 rows, 5 columns and 7 slices, initialized
  // // with 0.0.
  // il::StaticArray3D<double, 3, 5, 7> A{0.0};
  */
  StaticArray3D(const T& value);

  /* \brief Construct a il::StaticArray3D<T, n, p, q> from a brace-initialized
  // list
  // \details In order to allow brace initialization in all cases, this
  // constructor has different syntax from the one found in std::array. You
  // must use the il::value value as a first argument. For instance:
  //
  // // Construct an array of double with 2 rows, 3 columns and 2 slices from a
  // // list
  // il::StaticArray3D<double, 2, 3, 2> v{il::value,
  //                                      {2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
  //                                       2.5, 3.5, 4.5, 5.5, 6.5, 7.5}};
  */
  StaticArray3D(il::value_t, std::initializer_list<T> list);

  /* \brief Accessor for a const il::StaticArray3D<T, n, p, q>
  // \details Access (read only) the (i, j, k)-th element of the array. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::StaticArray3D<double, 2, 5, 10> A{0.0};
  // std::cout << A(0, 0, 0) << std::endl;
  */
  const T& operator()(il::int_t i, il::int_t j, il::int_t k) const;

  /* \brief Accessor for a il::StaticArray3D<T, n, p, q>
  // \details Access (read and write) the (i, j, k)-th element of the array.
  // Bound checking is done in debug mode but not in release mode.
  //
  // il::StaticArray3D<double, 2, 5, 10> A{0.0};
  // A(0, 0, 0) = 0.0;
  */
  T& operator()(il::int_t i, il::int_t j, il::int_t k);

  /* \brief Get the size of the il::StaticArray3D<T, n, p>
  //
  // for (il::int_t i{0}; i < v.size(0); ++i) {
  //   for (il::int_t j{0}; j < v.size(1); ++j) {
  //     A(i, j) = 1.0 / (i + j + 2);
  //   }
  // }
  */
  il::int_t size(il::int_t d) const;

  /* \brief Get an const array view to the container
  */
  ConstArray3DView<T> const_view() const;

  /* \brief Get an array view to the container
  */
  Array3DView<T> view();

  /* \brief Get an const array view to a subpart of the container with
  // the elements indexed by i with i_begin <= i < i_end, j with
  // j_begin <= j < j_end and k with k_begin <= k < k_end.
  */
  ConstArray3DView<T> const_view(il::int_t i_begin, il::int_t i_end,
                                 il::int_t j_begin, il::int_t j_end,
                                 il::int_t k_begin, il::int_t k_end) const;

  /* \brief Get an array view to a subpart of the container with
  // the elements indexed by i with i_begin <= i < i_end, j with
  // j_begin <= j < j_end and k with k_begin <= k < k_end.
  */
  Array3DView<T> view(il::int_t begin, il::int_t i_end, il::int_t j_begin,
                      il::int_t j_end, il::int_t k_begin, il::int_t k_end);

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

template <typename T, il::int_t n, il::int_t p, il::int_t q>
StaticArray3D<T, n, p>::StaticArray3D() {
  if (std::is_pod<T>::value) {
#ifndef NDEBUG
    for (il::int_t l{0}; l < n * p * q; ++l) {
      data_[l] = il::default_value<T>();
    }
#endif
  }
}

template <typename T, il::int_t n, il::int_t p, il::int_t q>
StaticArray3D<T, n, p>::StaticArray3D(const T& value) {
  for (il::int_t l{0}; l < n * p * q; ++l) {
    data_[l] = value;
  }
}

template <typename T, il::int_t n, il::int_t p, il::int_t q>
StaticArray3D<T, n, p, q>::StaticArray3D(il::value_t,
                                         std::initializer_list<T> list) {
  IL_ASSERT(n * p * q == static_cast<il::int_t>(list.size()));
  for (il::int_t l{0}; l < n * p * q; ++l) {
    data_[l] = *(list.begin() + l);
  }
}

template <typename T, il::int_t n, il::int_t p, il::int_t q>
const T& StaticArray3D<T, n, p, q>::operator()(il::int_t i, il::int_t j,
                                               il::int_t k) const {
  IL_ASSERT(static_cast<il::uint_t>(i) < static_cast<il::uint_t>(n));
  IL_ASSERT(static_cast<il::uint_t>(j) < static_cast<il::uint_t>(p));
  IL_ASSERT(static_cast<il::uint_t>(k) < static_cast<il::uint_t>(q));
  return data_[(i * p + j) * q + k];
}

template <typename T, il::int_t n, il::int_t p, il::int_t q>
T& StaticArray3D<T, n, p, q>::operator()(il::int_t i, il::int_t j,
                                         il::int_t k) {
  IL_ASSERT(static_cast<il::uint_t>(i) < static_cast<il::uint_t>(n));
  IL_ASSERT(static_cast<il::uint_t>(j) < static_cast<il::uint_t>(p));
  IL_ASSERT(static_cast<il::uint_t>(k) < static_cast<il::uint_t>(q));
  return data_[(i * p + j) * q + k];
}

template <typename T, il::int_t n, il::int_t p, il::int_t q>
il::int_t StaticArray3D<T, n, p, q>::size(il::int_t d) const {
  IL_ASSERT(static_cast<il::uint_t>(d) < static_cast<il::uint_t>(3));
  return d == 0 ? n : (d == 1 ? p : q);
}

template <typename T, il::int_t n, il::int_t p, il::int_t q>
ConstArray3DView<T> StaticArray3D<T, n, p, q>::const_view() const {
  return il::ConstArray3DView<T>{data_, n, p, q, p, q};
}

template <typename T, il::int_t n, il::int_t p, il::int_t q>
Array3DView<T> StaticArray3D<T, n, p, q>::view() {
  return il::Array3DView<T>{data_, n, p, q, p, q};
}

template <typename T, il::int_t n, il::int_t p, il::int_t q>
ConstArray3DView<T> StaticArray3D<T, n, p, q>::const_view(
    il::int_t i_begin, il::int_t i_end, il::int_t j_begin, il::int_t j_end,
    il::int_t k_begin, il::int_t k_end) const {
  IL_ASSERT(static_cast<il::uint_t>(i_begin) <
            static_cast<il::uint_t>(size(0)));
  IL_ASSERT(static_cast<il::uint_t>(i_end) <= static_cast<il::uint_t>(size(0)));
  IL_ASSERT(i_begin <= i_end);
  IL_ASSERT(static_cast<il::uint_t>(j_begin) <
            static_cast<il::uint_t>(size(1)));
  IL_ASSERT(static_cast<il::uint_t>(j_end) <= static_cast<il::uint_t>(size(1)));
  IL_ASSERT(j_begin <= j_end);
  IL_ASSERT(static_cast<il::uint_t>(k_begin) <
            static_cast<il::uint_t>(size(2)));
  IL_ASSERT(static_cast<il::uint_t>(k_end) <= static_cast<il::uint_t>(size(2)));
  IL_ASSERT(k_begin <= k_end);
  return ConstArray3DView<T>{data_ + (i_begin * p + j_begin) * q,
                             i_end - i_begin,
                             j_end - j_begin,
                             k_end - k_begin,
                             p,
                             q};
}

template <typename T, il::int_t n, il::int_t p, il::int_t q>
Array3DView<T> StaticArray3D<T, n, p, q>::view(
    il::int_t i_begin, il::int_t i_end, il::int_t j_begin, il::int_t j_end,
    il::int_t k_begin, il::int_t k_end) {
  IL_ASSERT(static_cast<il::uint_t>(i_begin) <
            static_cast<il::uint_t>(size(0)));
  IL_ASSERT(static_cast<il::uint_t>(i_end) <= static_cast<il::uint_t>(size(0)));
  IL_ASSERT(i_begin <= i_end);
  IL_ASSERT(static_cast<il::uint_t>(j_begin) <
            static_cast<il::uint_t>(size(1)));
  IL_ASSERT(static_cast<il::uint_t>(j_end) <= static_cast<il::uint_t>(size(1)));
  IL_ASSERT(j_begin <= j_end);
  IL_ASSERT(static_cast<il::uint_t>(k_begin) <
            static_cast<il::uint_t>(size(2)));
  IL_ASSERT(static_cast<il::uint_t>(k_end) <= static_cast<il::uint_t>(size(2)));
  IL_ASSERT(k_begin <= k_end);
  return Array3DView<T>{data_ + (i_begin * p + j_begin) * q,
                        i_end - i_begin,
                        j_end - j_begin,
                        k_end - k_begin,
                        p,
                        q};
}

template <typename T, il::int_t n, il::int_t p, il::int_t q>
const T* StaticArray3D<T, n, p, q>::data() const {
  return data_;
};

template <typename T, il::int_t n, il::int_t p, il::int_t q>
T* StaticArray3D<T, n, p, q>::data() {
  return data_;
};
}

#endif  // IL_STATICARRAY3D_H
