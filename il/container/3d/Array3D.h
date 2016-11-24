//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_ARRAY3D_H
#define IL_ARRAY3D_H

#include <il/base.h>

// <cstring> is needed for memcpy
#include <cstring>
// <initializer_list> is needed for std::initializer_list<T>
#include <initializer_list>
// <new> is needed for ::operator new
#include <new>
// <utility> is needed for std::move
#include <utility>

namespace il {

template <typename T>
class Array3D {
 private:
#ifdef IL_DEBUG_VISUALIZER
  il::int_t debug_align_;
  il::int_t debug_size_0_;
  il::int_t debug_size_1_;
  il::int_t debug_size_2_;
  il::int_t debug_capacity_0_;
  il::int_t debug_capacity_1_;
  il::int_t debug_capacity_2_;
#endif
  T* data_;
  T* size_[3];
  T* capacity_[3];

 public:
  /* \brief Default constructor
  // \details The size and the capacity of the array are set to 0 and no memory
  // allocation is done during the process.
  */
  Array3D();

  /* \brief Construct an il::Array3D<T> of n rows and p columns and q slices
  // \details The row size and the row capacity of the array are set to n. The
  // column size and the column capacity of the array are set to p. The slice
  // size and the slice capacity of the array are set to q.
  // - If T is a numeric value, the memory is
  //   - (Debug mode) initialized to il::default_value<T>(). It is usually NaN
  //     if T is a floating point number or 666..666 if T is an integer.
  //   - (Release mode) left uninitialized. This behavior is different from
  //     std::vector from the standard library which initializes all numeric
  //     values to 0.
  // - If T is an object with a default constructor, all objects are default
  //   constructed. A compile-time error is raised if T has no default
  //   constructor.
  //
  // // Construct an array of double with 3 rows, 5 columns and 7 slices.
  // il::Array3D<double> A{3, 5, 7};
  */
  explicit Array3D(il::int_t n0, il::int_t n1, il::int_t n2);

  /* \brief Construct an array of n rows and p columns with a value
  /
  // // Construct an array of double with 3 rows and 5 columns, initialized with
  // // 3.14.
  // il::Array2D<double> A{3, 5, 3.14};
  */
  explicit Array3D(il::int_t n, il::int_t p, il::int_t q, const T& x);

  /* \brief Construct an array of n rows, p columns and q slices using
  // constructor arguments
  /
  // // Construct an array of 3 by 5 by 7 il::Array<double>, all of length 9 and
  // // initialized with 3.14
  // il::Array3D<il::Array<double>> v{3, 5, 7, il::emplace, 9, 3.14};
  */
  template <typename... Args>
  explicit Array3D(il::int_t n, il::int_t p, il::int_t q, il::emplace_t,
                   Args&&... args);

  /* \brief Construct an array of n rows, p columns and q slices from a
  // brace-initialized list
  //
  // // Construct an array of double with 2 rows, 3 columns and 2 slices from a
  // // list
  // il::Array2D<double> v{2, 3, 2, il::value,
  //                       {2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
  //                        2.5, 3.5, 4.5, 5.5, 6.5, 7.5}};
  */
  explicit Array3D(il::value_t,
                   std::initializer_list<
                       std::initializer_list<std::initializer_list<T>>> list);

  /* \brief The copy constructor
  // \details The different size and capacity of the constructed il::Array3D<T>
  // are equal to the size of the source array.
  */
  Array3D(const Array3D<T>& A);

  /* \brief The move constructor
  */
  Array3D(Array3D<T>&& A);

  /* \brief The copy assignment
  // \details The size is the same as the one for the source array. The
  // capacity is not changed if it is enough for the copy to happen and is
  // set to the size of the source array if the initial capacity is too low.
  */
  Array3D& operator=(const Array3D<T>& A);

  /* \brief The move assignment
  */
  Array3D& operator=(Array3D<T>&& A);

  /* \brief The destructor
  */
  ~Array3D();

  /* \brief Accessor for a const il::3DArray<T>
  // \details Access (read only) the (i, j, k)-th element of the array. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::Array3D<double> A{4, 6, 7};
  // std::cout << A(3, 5, 6) << std::endl;
  */
  const T& operator()(il::int_t i, il::int_t j, il::int_t k) const;

  /* \brief Accessor for a il::3DArray<T>
  // \details Access (read and write) the (i, j, k)-th element of the array.
  // Bound checking is done in debug mode but not in release mode.
  //
  // il::Array3D<double> A{4, 6, 7};
  // A(0, 0, 0) = 3.14;
  */
  T& operator()(il::int_t i, il::int_t j, il::int_t k);

  /* \brief Get the size of the il::Array3D<T>
  // \details size(0) returns the number of rows of the array and size(1)
  // returns the number of columns of the same array and size(2) return the
  // number of slices of the array. The library has been designed in a way that
  // any compiler can prove that modifying A(i, j, k) can't
  // change the result of A.size(0), A.size(1) or A.size(2). As a consequence
  // a call to A.size(0), A.size(1) or A.size(2) are made just once at the very
  // beginning of the loop in the following example. It allows many
  // optimizations from the compiler, including automatic vectorization.
  //
  // il::Array3D<double> A{n, p, q};
  // for (il::int_t i{0}; i < v.size(0); ++i) {
  //   for (il::int_t j{0}; j < v.size(1); ++j) {
  //     for (il::int_t k{0}; k < v.size(2); ++k) {
  //       A(i, j, k) = 1.0 / (i + j + k + 3);
  //     }
  //   }
  // }
  */
  il::int_t size(il::int_t d) const;

  /* \brief Resizing an il::Array3D<T>
  // \details No reallocation is performed if the new size is <= to the
  // capacity for both rows, columns and slices. In this case, the capacity is
  // unchanged. When one of the sizes is > than the the capacity, reallocation
  // is done and the array gets the same capacity as its size.
  */
  void resize(il::int_t n, il::int_t p, il::int_t q);

  /* \brief Get the capacity of the il::Array3D<T>
  // \details capacity(0) gives the capacity in terms of rows and capacity(1)
  // gives the capacity in terms of columns and capacity(2) gives the capacity
  // in terms of slices.
  */
  il::int_t capacity(il::int_t d) const;

  /* \brief Change the capacity of the array to at least r, s columns and t
  // slices
  // \details If the row capacity is >= to r, the column capacity is >= to s,
  // and the slice capacity is >= to t, nothing is done. Otherwise, reallocation
  // is done and the new capacity is set to r, s and t
  */
  void reserve(il::int_t r, il::int_t s, il::int_t t);

  /* \brief Get a pointer to const to the first element of the array
  // \details One should use this method only when using C-style API
  */
  const T* data() const;

  /* \brief Get a pointer to the first element of the array
  // \details One should use this method only when using C-style API
  */
  T* data();

  /* \brief The memory position of A(i, j, k) is
  // data() + (i * stride(0) + j) * stride(1) + k
  */
  il::int_t stride(il::int_t d) const;

 private:
  /* \brief Used internally in debug mode to check the invariance of the object
  */
  void check_invariance() const;
};

template <typename T>
Array3D<T>::Array3D() {
#ifdef IL_DEBUG_VISUALIZER
  debug_size_0_ = 0;
  debug_size_1_ = 0;
  debug_size_2_ = 0;
  debug_capacity_0_ = 0;
  debug_capacity_1_ = 0;
  debug_capacity_2_ = 0;
#endif
  data_ = nullptr;
  size_[0] = nullptr;
  size_[1] = nullptr;
  size_[2] = nullptr;
  capacity_[0] = nullptr;
  capacity_[1] = nullptr;
  capacity_[2] = nullptr;
}

template <typename T>
Array3D<T>::Array3D(il::int_t n, il::int_t p, il::int_t q) {
  IL_ASSERT(n >= 0);
  IL_ASSERT(p >= 0);
  IL_ASSERT(q >= 0);
  il::int_t npq{n > 0 && p > 0 && q > 0
                    ? n * p * q
                    : (n > p && n > q ? n : (p > q ? p : q))};
  if (std::is_pod<T>::value) {
    data_ = new T[npq];
#ifndef NDEBUG
    for (il::int_t l{0}; l < n * p * q; ++l) {
      data_[l] = il::default_value<T>();
    }
#endif
  } else {
    data_ = static_cast<T*>(::operator new(npq * sizeof(T)));
    for (il::int_t l{0}; l < n * p * q; ++l) {
      new (data_ + l) T{};
    }
  }
#ifdef IL_DEBUG_VISUALIZER
  debug_size_0_ = n;
  debug_size_1_ = p;
  debug_size_2_ = q;
  debug_capacity_0_ = n;
  debug_capacity_1_ = p;
  debug_capacity_2_ = q;
#endif
  size_[0] = data_ + n;
  size_[1] = data_ + p;
  size_[2] = data_ + q;
  capacity_[0] = data_ + n;
  capacity_[1] = data_ + p;
  capacity_[2] = data_ + q;
}

template <typename T>
Array3D<T>::Array3D(il::int_t n, il::int_t p, il::int_t q, const T& x) {
  IL_ASSERT(n >= 0);
  IL_ASSERT(p >= 0);
  IL_ASSERT(q >= 0);
  il::int_t npq{n > 0 && p > 0 && q > 0
                    ? n * p * q
                    : (n > p && n > q ? n : (p > q ? p : q))};
  if (std::is_pod<T>::value) {
    data_ = new T[npq];
    for (il::int_t l{0}; l < n * p * q; ++l) {
      data_[l] = x;
    }
  } else {
    data_ = static_cast<T*>(::operator new(npq * sizeof(T)));
    for (il::int_t l{0}; l < n * p * q; ++l) {
      new (data_ + l) T(x);
    }
  }
#ifdef IL_DEBUG_VISUALIZER
  debug_size_0_ = n;
  debug_size_1_ = p;
  debug_size_2_ = q;
  debug_capacity_0_ = n;
  debug_capacity_1_ = p;
  debug_capacity_2_ = q;
#endif
  size_[0] = data_ + n;
  size_[1] = data_ + p;
  size_[2] = data_ + q;
  capacity_[0] = data_ + n;
  capacity_[1] = data_ + p;
  capacity_[2] = data_ + q;
}

template <typename T>
template <typename... Args>
Array3D<T>::Array3D(il::int_t n, il::int_t p, il::int_t q, il::emplace_t,
                    Args&&... args) {
  IL_ASSERT(n >= 0);
  IL_ASSERT(p >= 0);
  IL_ASSERT(q >= 0);
  il::int_t npq{n > 0 && p > 0 && q > 0
                    ? n * p * q
                    : (n > p && n > q ? n : (p > q ? p : q))};
  if (std::is_pod<T>::value) {
    data_ = new T[npq];
  } else {
    data_ = static_cast<T*>(::operator new(npq * sizeof(T)));
  }
  for (il::int_t l{0}; l < n * p * q; ++l) {
    new (data_ + l) T(args...);
  }
#ifdef IL_DEBUG_VISUALIZER
  debug_size_0_ = n;
  debug_size_1_ = p;
  debug_size_2_ = q;
  debug_capacity_0_ = n;
  debug_capacity_1_ = p;
  debug_capacity_2_ = q;
#endif
  size_[0] = data_ + n;
  size_[1] = data_ + p;
  size_[2] = data_ + q;
  capacity_[0] = data_ + n;
  capacity_[1] = data_ + p;
  capacity_[2] = data_ + q;
}

template <typename T>
Array3D<T>::Array3D(il::value_t,
                    std::initializer_list<
                        std::initializer_list<std::initializer_list<T>>> list) {
  const il::int_t n{static_cast<il::int_t>(list.size())};
  const il::int_t p{static_cast<il::int_t>(list.begin()->size())};
  const il::int_t q{static_cast<il::int_t>(list.begin()->begin()->size())};
  il::int_t npq{n > 0 && p > 0 && q > 0
                    ? n * p * q
                    : (n > p && n > q ? n : (p > q ? p : q))};
  if (std::is_pod<T>::value) {
    data_ = new T[npq];
    for (il::int_t i{0}; i < n; ++i) {
      IL_ASSERT((list.begin() + i)->size() == p);
      for (il::int_t j{0}; j < p; ++j) {
        IL_ASSERT(((list.begin() + i)->begin() + j)->size() == q);
        memcpy(data_ + (i * p + j) * q,
               ((list.begin() + i)->begin() + j)->begin(), q * sizeof(T));
      }
    }
  } else {
    data_ = static_cast<T*>(::operator new(npq * sizeof(T)));
    for (il::int_t i{0}; i < n; ++i) {
      IL_ASSERT((list.begin() + i)->size() == p);
      for (il::int_t j{0}; j < p; ++j) {
        IL_ASSERT(((list.begin() + i)->begin() + j)->size() == q);
        for (il::int_t k{0}; k < q; ++k) {
          new (data_ + (i * p + j) * q + k)
              T(*(((list.begin() + i)->begin() + j)->begin() + k));
        }
      }
    }
  }
#ifdef IL_DEBUG_VISUALIZER
  debug_size_0_ = n;
  debug_size_1_ = p;
  debug_size_2_ = q;
  debug_capacity_0_ = n;
  debug_capacity_1_ = p;
  debug_capacity_2_ = q;
#endif
  size_[0] = data_ + n;
  size_[1] = data_ + p;
  size_[2] = data_ + q;
  capacity_[0] = data_ + n;
  capacity_[1] = data_ + p;
  capacity_[2] = data_ + q;
}

template <typename T>
Array3D<T>::Array3D(const Array3D<T>& A) {
  const il::int_t n{A.size(0)};
  const il::int_t p{A.size(1)};
  const il::int_t q{A.size(2)};
  il::int_t npq{n > 0 && p > 0 && q > 0
                    ? n * p * q
                    : (n > p && n > q ? n : (p > q ? p : q))};
  if (std::is_pod<T>::value) {
    data_ = new T[npq];
    for (il::int_t i{0}; i < n; ++i) {
      for (il::int_t j{0}; j < p; ++j) {
        memcpy(data_ + (i * p + j) * q,
               A.data_ + (i * A.stride(0) + j) * A.stride(1), p * sizeof(T));
      }
    }
  } else {
    data_ = static_cast<T*>(::operator new(n * p * sizeof(T)));
    for (il::int_t i{0}; i < n; ++i) {
      for (il::int_t j{0}; j < p; ++j) {
        for (il::int_t k{0}; k < q; ++k) {
          new (data_ + (i * p + j) * q) T(A(i, j, k));
        }
      }
    }
  }
#ifdef IL_DEBUG_VISUALIZER
  debug_size_0_ = n;
  debug_size_1_ = p;
  debug_size_2_ = q;
  debug_capacity_0_ = n;
  debug_capacity_1_ = p;
  debug_capacity_2_ = q;
#endif
  size_[0] = data_ + n;
  size_[1] = data_ + p;
  size_[2] = data_ + q;
  capacity_[0] = data_ + n;
  capacity_[1] = data_ + p;
  capacity_[2] = data_ + q;
}

template <typename T>
Array3D<T>::Array3D(Array3D<T>&& A) {
#ifdef IL_DEBUG_VISUALIZER
  debug_size_0_ = A.debug_size_0_;
  debug_size_1_ = A.debug_size_1_;
  debug_size_2_ = A.debug_size_2_;
  debug_capacity_0_ = A.debug_capacity_0_;
  debug_capacity_1_ = A.debug_capacity_1_;
  debug_capacity_2_ = A.debug_capacity_2_;
#endif
  data_ = A.data_;
  size_[0] = A.size_[0];
  size_[1] = A.size_[1];
  size_[2] = A.size_[2];
  capacity_[0] = A.capacity_[0];
  capacity_[1] = A.capacity_[1];
  capacity_[2] = A.capacity_[2];
#ifdef IL_DEBUG_VISUALIZER
  A.debug_size_0_ = 0;
  A.debug_size_1_ = 0;
  A.debug_size_2_ = 0;
  A.debug_capacity_0_ = 0;
  A.debug_capacity_1_ = 0;
  A.debug_capacity_2_ = 0;
#endif
  A.data_ = nullptr;
  A.size_[0] = nullptr;
  A.size_[1] = nullptr;
  A.size_[2] = nullptr;
  A.capacity_[0] = nullptr;
  A.capacity_[1] = nullptr;
  A.capacity_[2] = nullptr;
}

template <typename T>
Array3D<T>& Array3D<T>::operator=(const Array3D<T>& A) {
  if (this != &A) {
    const il::int_t n{A.size(0)};
    const il::int_t p{A.size(1)};
    const il::int_t q{A.size(2)};
    il::int_t npq{n > 0 && p > 0 && q > 0
                      ? n * p * q
                      : (n > p && n > q ? n : (p > q ? p : q))};
    const bool needs_memory{n > capacity(0) || p > capacity(1) ||
                            q > capacity(2)};
    if (needs_memory) {
      if (std::is_pod<T>::value) {
        if (data_) {
          delete[] data_;
        }
        data_ = new T[npq];
        for (il::int_t i{0}; i < n; ++i) {
          for (il::int_t j{0}; j < p; ++j) {
            memcpy(data_ + (i * p + j) * q,
                   A.data_ + (i * A.stride(0) + j) * A.stride(1),
                   q * sizeof(T));
          }
        }
      } else {
        if (data_) {
          for (il::int_t i{0}; i < size(0); ++i) {
            for (il::int_t j{0}; j < size(1); ++j) {
              for (il::int_t k{0}; k < size(2); ++k) {
                (data_ + (i * stride(0) + j) * stride(1) + k)->~T();
              }
            }
          }
          ::operator delete(data_);
        }
        data_ = static_cast<T*>(::operator new(npq * sizeof(T)));
        for (il::int_t i{0}; i < n; ++i) {
          for (il::int_t j{0}; j < p; ++j) {
            for (il::int_t k{0}; k < q; ++k) {
              new (data_ + (i * p + j) * q + k) T(A(i, j, k));
            }
          }
        }
      }
#ifdef IL_DEBUG_VISUALIZER
      debug_size_0_ = n;
      debug_size_1_ = p;
      debug_size_2_ = q;
      debug_capacity_0_ = n;
      debug_capacity_1_ = p;
      debug_capacity_2_ = q;
#endif
      size_[0] = data_ + n;
      size_[1] = data_ + p;
      size_[2] = data_ + q;
      capacity_[0] = data_ + n;
      capacity_[1] = data_ + p;
      capacity_[2] = data_ + q;
    } else {
      if (std::is_pod<T>::value) {
        for (il::int_t i{0}; i < n; ++i) {
          for (il::int_t j{0}; j < p; ++j) {
            memcpy(data_ + (i * stride(0) + j) * stride(1),
                   A.data_ + (i * A.stride(0) + j) * A.stride(1),
                   q * sizeof(T));
          }
        }
      } else {
        for (il::int_t i{0}; i < n; ++i) {
          for (il::int_t j{0}; j < p; ++j) {
            for (il::int_t k{0}; k < q; ++k) {
              data_[(i * stride(0) + j) * stride(1) + k] = A(i, j, k);
            }
          }
        }
        for (il::int_t i{0}; i < size(0); ++i) {
          for (il::int_t j{i < n ? p : 0}; j < size(1); ++j) {
            for (il::int_t k{i < n && j < p ? q : 0}; k < size(2); ++k) {
              (data_ + (i * stride(0) + j) * stride(1) + k)->~T();
            }
          }
        }
      }
#ifdef IL_DEBUG_VISUALIZER
      debug_size_0_ = n;
      debug_size_1_ = p;
      debug_size_2_ = q;
#endif
      size_[0] = data_ + n;
      size_[1] = data_ + p;
      size_[2] = data_ + q;
    }
  }
  return *this;
}

template <typename T>
Array3D<T>& Array3D<T>::operator=(Array3D<T>&& A) {
  if (this != &A) {
    if (data_) {
      if (std::is_pod<T>::value) {
        delete[] data_;
      } else {
        for (il::int_t i{0}; i < size(0); ++i) {
          for (il::int_t j{0}; j < size(1); ++j) {
            for (il::int_t k{0}; k < size(2); ++k) {
              (data_ + (i * stride(0) + j) * stride(1) + k)->~T();
            }
          }
        }
        ::operator delete(data_);
      }
    }
#ifdef IL_DEBUG_VISUALIZER
    debug_size_0_ = A.debug_size_0_;
    debug_size_1_ = A.debug_size_1_;
    debug_size_2_ = A.debug_size_2_;
    debug_capacity_0_ = A.debug_capacity_0_;
    debug_capacity_1_ = A.debug_capacity_1_;
    debug_capacity_2_ = A.debug_capacity_2_;
#endif
    data_ = A.data_;
    size_[0] = A.size_[0];
    size_[1] = A.size_[1];
    size_[2] = A.size_[2];
    capacity_[0] = A.capacity_[0];
    capacity_[1] = A.capacity_[1];
    capacity_[2] = A.capacity_[2];
#ifdef IL_DEBUG_VISUALIZER
    A.debug_size_0_ = 0;
    A.debug_size_1_ = 0;
    A.debug_size_2_ = 0;
    A.debug_capacity_0_ = 0;
    A.debug_capacity_1_ = 0;
    A.debug_capacity_2_ = 0;
#endif
    A.data_ = nullptr;
    A.size_[0] = nullptr;
    A.size_[1] = nullptr;
    A.size_[2] = nullptr;
    A.capacity_[0] = nullptr;
    A.capacity_[1] = nullptr;
    A.capacity_[2] = nullptr;
  }
  return *this;
}

template <typename T>
Array3D<T>::~Array3D() {
#ifndef NDEBUG
  check_invariance();
#endif
  if (data_) {
    if (std::is_pod<T>::value) {
      delete[] data_;
    } else {
      for (il::int_t i{0}; i < size(0); ++i) {
        for (il::int_t j{0}; j < size(1); ++j) {
          for (il::int_t k{0}; k < size(2); ++k) {
            (data_ + (i * stride(0) + j * stride(1)) + k)->~T();
          }
        }
      }
      ::operator delete(data_);
    }
  }
}

template <typename T>
const T& Array3D<T>::operator()(il::int_t i, il::int_t j, il::int_t k) const {
  IL_ASSERT(static_cast<il::uint_t>(i) < static_cast<il::uint_t>(size(0)));
  IL_ASSERT(static_cast<il::uint_t>(j) < static_cast<il::uint_t>(size(1)));
  IL_ASSERT(static_cast<il::uint_t>(k) < static_cast<il::uint_t>(size(2)));
  return data_[(k * (capacity_[1] - data_) + j) * (capacity_[0] - data_) + i];
}

template <typename T>
T& Array3D<T>::operator()(il::int_t i, il::int_t j, il::int_t k) {
  IL_ASSERT(static_cast<il::uint_t>(i) < static_cast<il::uint_t>(size(0)));
  IL_ASSERT(static_cast<il::uint_t>(j) < static_cast<il::uint_t>(size(1)));
  IL_ASSERT(static_cast<il::uint_t>(k) < static_cast<il::uint_t>(size(2)));
  return data_[(k * (capacity_[1] - data_) + j) * (capacity_[0] - data_) + i];
}

template <typename T>
il::int_t Array3D<T>::size(il::int_t d) const {
  IL_ASSERT(static_cast<il::uint_t>(d) < static_cast<il::uint_t>(3));
  return static_cast<il::int_t>(size_[d] - data_);
}

template <typename T>
void Array3D<T>::resize(il::int_t n, il::int_t p, il::int_t q) {
  IL_ASSERT(n >= 0);
  IL_ASSERT(p >= 0);
  IL_ASSERT(q >= 0);
  if (n <= capacity(0) && p <= capacity(1) && q <= capacity(2)) {
    if (std::is_pod<T>::value) {
#ifndef NDEBUG
      for (il::int_t i{0}; i < n; ++i) {
        for (il::int_t j{i < size(0) ? size(1) : 0}; j < p; ++j) {
          for (il::int_t k{i < size(0) && j < size(1) ? size(2) : 0}; k < q;
               ++k) {
            data_[(i * stride(0) + j) * stride(1) + k] = il::default_value<T>();
          }
        }
      }
#endif
    } else {
      for (il::int_t i{0}; i < size(0); ++i) {
        for (il::int_t j{i < n ? p : 0}; j < size(1); ++j) {
          for (il::int_t k{i < n && j < p ? q : 0}; k < size(2); ++k) {
            (data_ + (i * stride(0) + j) * stride(1) + k)->~T();
          }
        }
      }
      for (il::int_t i{0}; i < n; ++i) {
        for (il::int_t j{i < size(0) ? size(1) : 0}; j < p; ++j) {
          for (il::int_t k{i < size(0) && j < size(1) ? size(2) : 0}; k < q;
               ++k) {
            new (data_ + (i * stride(0) + j) * stride(1) + k) T{};
          }
        }
      }
    }
  } else {
    il::int_t npq{n > 0 && p > 0 && q > 0
                      ? n * p * q
                      : (n > p && n > q ? n : (p > q ? p : q))};
    T* new_data;
    if (std::is_pod<T>::value) {
      new_data = new T[npq];
    } else {
      new_data = static_cast<T*>(::operator new(npq * sizeof(T)));
    }
    if (data_) {
      if (std::is_pod<T>::value) {
        for (il::int_t i{0}; i < size(0); ++i) {
          for (il::int_t j{0}; j < size(1); ++j) {
            memcpy(new_data + (i * p + j) * q,
                   data_ + (i * stride(0) + j) * stride(1),
                   size(2) * sizeof(T));
          }
        }
        delete[] data_;
      } else {
        for (il::int_t i{0}; i < size(0); ++i) {
          for (il::int_t j{0}; j < size(1); ++j) {
            for (il::int_t k{0}; k < size(2); ++k) {
              new (new_data + (i * p + j) * q + k)
                  T(std::move(data_[(i * stride(0) + j) * stride(1) + k]));
              (data_ + (i * stride(0) + j) * stride(1) + k)->~T();
            }
          }
        }
        ::operator delete(data_);
      }
    }
    if (std::is_pod<T>::value) {
#ifndef NDEBUG
      for (il::int_t i{0}; i < n; ++i) {
        for (il::int_t j{i < size(0) ? size(1) : 0}; j < p; ++j) {
          for (il::int_t k{i < size(0) && j < size(1) ? size(2) : 0}; k < q;
               ++k) {
            new_data[(i * p + j) * q + k] = il::default_value<T>();
          }
        }
      }
#endif
    } else {
      for (il::int_t i{0}; i < n; ++i) {
        for (il::int_t j{i < size(0) ? size(1) : 0}; j < p; ++j) {
          for (il::int_t k{i < size(0) && j < size(1) ? size(2) : 0}; k < q;
               ++k) {
            new (new_data + (i * p + j) * q + k) T{};
          }
        }
      }
    }
    data_ = new_data;
#ifdef IL_DEBUG_VISUALIZER
    debug_capacity_0_ = n;
    debug_capacity_1_ = p;
    debug_capacity_2_ = q;
#endif
    capacity_[0] = data_ + n;
    capacity_[1] = data_ + p;
    capacity_[2] = data_ + q;
  }
#ifdef IL_DEBUG_VISUALIZER
  debug_size_0_ = n;
  debug_size_1_ = p;
  debug_size_2_ = q;
#endif
  size_[0] = data_ + n;
  size_[1] = data_ + p;
  size_[2] = data_ + q;
}

template <typename T>
il::int_t Array3D<T>::capacity(il::int_t d) const {
  IL_ASSERT(static_cast<il::uint_t>(d) < static_cast<il::uint_t>(3));
  return static_cast<il::int_t>(capacity_[d] - data_);
}

template <typename T>
void Array3D<T>::reserve(il::int_t r, il::int_t s, il::int_t t) {
  IL_ASSERT(r >= 0);
  IL_ASSERT(s >= 0);
  IL_ASSERT(t >= 0);
  if (r > capacity(0) || s > capacity(1) || t > capacity(2)) {
    const il::int_t n_old{size(0)};
    const il::int_t p_old{size(1)};
    const il::int_t q_old{size(2)};
    const il::int_t rst{r > 0 && s > 0 && t > 0
                            ? r * s * t
                            : (r > s && r > s ? r : (s > t ? s : t))};
    T* new_data;
    if (std::is_pod<T>::value) {
      new_data = new T[rst];
    } else {
      new_data = static_cast<T*>(::operator new(rst * sizeof(T)));
    }
    if (data_) {
      if (std::is_pod<T>::value) {
        for (il::int_t i{0}; i < size(0); ++i) {
          for (il::int_t j{0}; j < size(0); ++j) {
            memcpy(new_data + (i * s + j) * t,
                   data_ + (i * stride(0) + j) * stride(1),
                   size(1) * sizeof(T));
          }
        }
        delete[] data_;
      } else {
        for (il::int_t i{0}; i < size(0); ++i) {
          for (il::int_t j{0}; j < size(1); ++j) {
            for (il::int_t k{0}; k < size(2); ++k) {
              new (new_data + (i * s + j) * t + k)
                  T(std::move(data_[(i * stride(0) + j) * stride(1) + k]));
              (data_ + (i * stride(0) + j) * stride(1) + k)->~T();
            }
          }
        }
        ::operator delete(data_);
      }
    }
    data_ = new_data;
#ifdef IL_DEBUG_VISUALIZER
    debug_capacity_0_ = r;
    debug_capacity_1_ = s;
    debug_capacity_2_ = t;
#endif
    size_[0] = data_ + n_old;
    size_[1] = data_ + p_old;
    size_[2] = data_ + q_old;
    capacity_[0] = data_ + r;
    capacity_[1] = data_ + s;
    capacity_[2] = data_ + t;
  }
}

template <typename T>
const T* Array3D<T>::data() const {
  return data_;
}

template <typename T>
T* Array3D<T>::data() {
  return data_;
}

template <typename T>
il::int_t Array3D<T>::stride(il::int_t d) const {
  IL_ASSERT(static_cast<il::uint_t>(d) < static_cast<il::uint_t>(2));
  return static_cast<il::int_t>(capacity_[d + 1] - data_);
}

template <typename T>
void Array3D<T>::check_invariance() const {
  if (data_ == nullptr) {
    IL_ASSERT(size_[0] == nullptr);
    IL_ASSERT(size_[1] == nullptr);
    IL_ASSERT(size_[2] == nullptr);
    IL_ASSERT(capacity_[0] == nullptr);
    IL_ASSERT(capacity_[1] == nullptr);
    IL_ASSERT(capacity_[2] == nullptr);
  } else {
    IL_ASSERT(size_[0] != nullptr);
    IL_ASSERT(size_[1] != nullptr);
    IL_ASSERT(size_[2] != nullptr);
    IL_ASSERT(capacity_[0] != nullptr);
    IL_ASSERT(capacity_[1] != nullptr);
    IL_ASSERT(capacity_[2] != nullptr);
    IL_ASSERT((size_[0] - data_) <= (capacity_[0] - data_));
    IL_ASSERT((size_[1] - data_) <= (capacity_[1] - data_));
    IL_ASSERT((size_[2] - data_) <= (capacity_[2] - data_));
  }
}
}

#endif  // IL_ARRAY3D_H