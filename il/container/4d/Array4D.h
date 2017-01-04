//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_ARRAY4D_H
#define IL_ARRAY4D_H

// <cstring> is needed for memcpy
#include <cstring>
// <initializer_list> is needed for std::initializer_list<T>
#include <initializer_list>
// <new> is needed for ::operator new
#include <new>
// <type_traits> is needed for std::is_pod
#include <type_traits>
// <utility> is needed for std::move
#include <utility>

#include <il/base.h>

namespace il {

template <typename T>
class Array4D {
 private:
#ifdef IL_DEBUG_VISUALIZER
  il::int_t debug_size_[4];
  il::int_t debug_capacity_[4];
#endif
  T* data_;
  T* size_[4];
  T* capacity_[4];

 public:
  /* \brief Default constructor
  // \details The size and the capacity of the array are set to 0 and no memory
  // allocation is done during the process.
  */
  Array4D();

  /* \brief Construct an il::Array4D<T> of n rows and p columns and q slices
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
  // il::Array4D<double> A{3, 5, 7};
  */
  explicit Array4D(il::int_t n0, il::int_t n1, il::int_t n2, il::int_t n3);

  /* \brief Construct an array of n rows and p columns with a value
  /
  // // Construct an array of double with 3 rows and 5 columns, initialized with
  // // 3.14.
  // il::Array2D<double> A{3, 5, 3.14};
  */
  explicit Array4D(il::int_t n0, il::int_t n1, il::int_t n2, il::int_t n3,
                   const T& x);

  /* \brief The copy constructor
  // \details The different size and capacity of the constructed il::Array4D<T>
  // are equal to the size of the source array.
  */
  Array4D(const Array4D<T>& A);

  /* \brief The move constructor
  */
  Array4D(Array4D<T>&& A);

  /* \brief The copy assignment
  // \details The size is the same as the one for the source array. The
  // capacity is not changed if it is enough for the copy to happen and is
  // set to the size of the source array if the initial capacity is too low.
  */
  Array4D& operator=(const Array4D<T>& A);

  /* \brief The move assignment
  */
  Array4D& operator=(Array4D<T>&& A);

  /* \brief The destructor
  */
  ~Array4D();

  /* \brief Accessor for a const il::4DArray<T>
  // \details Access (read only) the (i0, i1, i2, i3)-th element of the array.
  // Bound checking is done in debug mode but not in release mode.
  //
  // il::Array4D<double> A{4, 6, 7, 8};
  // std::cout << A(3, 5, 6, 4) << std::endl;
  */
  const T& operator()(il::int_t i0, il::int_t i1, il::int_t i2,
                      il::int_t i3) const;

  /* \brief Accessor for a il::4DArray<T>
  // \details Access (read and write) the (i0, i1, i2, i3)-th element of the
  // array. Bound checking is done in debug mode but not in release mode.
  //
  // il::Array4D<double> A{4, 6, 7, 8};
  // A(0, 0, 0, 0) = 3.14;
  */
  T& operator()(il::int_t i0, il::int_t i1, il::int_t i2, il::int_t i3);

  /* \brief Get the size of the il::Array4D<T>
  // \details size(0) returns the number of rows of the array and size(1)
  // returns the number of columns of the same array and size(2) return the
  // number of slices of the array. The library has been designed in a way that
  // any compiler can prove that modifying A(i, j, k) can't
  // change the result of A.size(0), A.size(1) or A.size(2). As a consequence
  // a call to A.size(0), A.size(1) or A.size(2) are made just once at the very
  // beginning of the loop in the following example. It allows many
  // optimizations from the compiler, including automatic vectorization.
  //
  // il::Array4D<double> A{n0, n1, n2, n3};
  // for (il::int_t i3 = 0; i3 < v.size(3); ++i3) {
  //   for (il::int_t i2 = 0; i2 < v.size(2); ++i2) {
  //     for (il::int_t i1 = 0; i1 < v.size(1); ++i1) {
  //       for (il::int_t i0 = 0; i0 < v.size(0); ++i0) {
  //         A(i0, i1, i2, i3) = 1.0 / (1 + i0 + i1 + i2 + i3);
  //       }
  //     }
  //   }
  // }
  */
  il::int_t size(il::int_t d) const;

  /* \brief Resizing an il::Array4D<T>
  // \details No reallocation is performed if the new size is <= to the
  // capacity for both rows, columns and slices. In this case, the capacity is
  // unchanged. When one of the sizes is > than the the capacity, reallocation
  // is done and the array gets the same capacity as its size.
  */
  void resize(il::int_t n0, il::int_t n1, il::int_t n2, il::int_t n3);

  /* \brief Get the capacity of the il::Array4D<T>
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
  void reserve(il::int_t r0, il::int_t r1, il::int_t r2, il::int_t r3);

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
  //  il::int_t stride(il::int_t d) const;

 private:
  /* \brief Used internally in debug mode to check the invariance of the object
  */
  void check_invariance() const;
};

template <typename T>
Array4D<T>::Array4D() {
#ifdef IL_DEBUG_VISUALIZER
  debug_size_[0] = 0;
  debug_size_[1] = 0;
  debug_size_[2] = 0;
  debug_size_[3] = 0;
  debug_capacity_[0] = 0;
  debug_capacity_[1] = 0;
  debug_capacity_[2] = 0;
  debug_capacity_[3] = 0;
#endif
  data_ = nullptr;
  size_[0] = nullptr;
  size_[1] = nullptr;
  size_[2] = nullptr;
  size_[3] = nullptr;
  capacity_[0] = nullptr;
  capacity_[1] = nullptr;
  capacity_[2] = nullptr;
  capacity_[3] = nullptr;
}

template <typename T>
Array4D<T>::Array4D(il::int_t n0, il::int_t n1, il::int_t n2, il::int_t n3) {
  IL_ASSERT_PRECOND(n0 >= 0);
  IL_ASSERT_PRECOND(n1 >= 0);
  IL_ASSERT_PRECOND(n2 >= 0);
  IL_ASSERT_PRECOND(n3 >= 0);
  il::int_t r0;
  il::int_t r1;
  il::int_t r2;
  il::int_t r3;
  if (n0 > 0 && n1 > 0 && n2 > 0 && n3 > 0) {
    r0 = n0;
    r1 = n1;
    r2 = n2;
    r3 = n3;
  } else if (n0 == 0 && n1 == 0 && n2 == 0 && n3 == 0) {
    r0 = 0;
    r1 = 0;
    r2 = 0;
    r3 = 0;
  } else {
    r0 = (n0 == 0) ? 1 : n0;
    r1 = (n1 == 0) ? 1 : n1;
    r2 = (n2 == 0) ? 1 : n2;
    r3 = (n3 == 0) ? 1 : n3;
  }
  const il::int_t r = r0 * r1 * r2 * r3;
  if (r > 0) {
    if (std::is_pod<T>::value) {
      data_ = new T[r];
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i3 = 0; i3 < n3; ++i3) {
        for (il::int_t i2 = 0; i2 < n2; ++i2) {
          for (il::int_t i1 = 0; i1 < n1; ++i1) {
            for (il::int_t i0 = 0; i0 < n0; ++i0) {
              data_[((i3 * r2 + i2) * r1 + i1) * r0 + i0] =
                  il::default_value<T>();
            }
          }
        }
      }
#endif
    } else {
      data_ = static_cast<T*>(::operator new(r * sizeof(T)));
      for (il::int_t i3 = 0; i3 < n3; ++i3) {
        for (il::int_t i2 = 0; i2 < n2; ++i2) {
          for (il::int_t i1 = 0; i1 < n1; ++i1) {
            for (il::int_t i0 = 0; i0 < n0; ++i0) {
              new (data_ + ((i3 * r2 + i2) * r1 + i1) * r0 + i0) T{};
            }
          }
        }
      }
    }
  } else {
    data_ = nullptr;
  }
#ifdef IL_DEBUG_VISUALIZER
  debug_size_[0] = n0;
  debug_size_[1] = n1;
  debug_size_[2] = n2;
  debug_size_[3] = n3;
  debug_capacity_[0] = r0;
  debug_capacity_[1] = r1;
  debug_capacity_[2] = r2;
  debug_capacity_[3] = r3;
#endif
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  size_[2] = data_ + n2;
  size_[3] = data_ + n3;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  capacity_[2] = data_ + r2;
  capacity_[3] = data_ + r3;
}

template <typename T>
Array4D<T>::Array4D(il::int_t n0, il::int_t n1, il::int_t n2, il::int_t n3,
                    const T& x) {
  IL_ASSERT_PRECOND(n0 >= 0);
  IL_ASSERT_PRECOND(n1 >= 0);
  IL_ASSERT_PRECOND(n2 >= 0);
  IL_ASSERT_PRECOND(n3 >= 0);
  il::int_t r0;
  il::int_t r1;
  il::int_t r2;
  il::int_t r3;
  if (n0 > 0 && n1 > 0 && n2 > 0 && n3 > 0) {
    r0 = n0;
    r1 = n1;
    r2 = n2;
    r3 = n3;
  } else if (n0 == 0 && n1 == 0 && n2 == 0 && n3 == 0) {
    r0 = 0;
    r1 = 0;
    r2 = 0;
    r3 = 0;
  } else {
    r0 = (n0 == 0) ? 1 : n0;
    r1 = (n1 == 0) ? 1 : n1;
    r2 = (n2 == 0) ? 1 : n2;
    r3 = (n3 == 0) ? 1 : n3;
  }
  const il::int_t r = r0 * r1 * r2 * r3;
  if (r > 0) {
    if (std::is_pod<T>::value) {
      data_ = new T[r];
      for (il::int_t i3 = 0; i3 < n3; ++i3) {
        for (il::int_t i2 = 0; i2 < n2; ++i2) {
          for (il::int_t i1 = 0; i1 < n1; ++i1) {
            for (il::int_t i0 = 0; i0 < n0; ++i0) {
              data_[((i3 * r2 + i2) * r1 + i1) * r0 + i0] = x;
            }
          }
        }
      }
    } else {
      data_ = static_cast<T*>(::operator new(r * sizeof(T)));
      for (il::int_t i3 = 0; i3 < n3; ++i3) {
        for (il::int_t i2 = 0; i2 < n2; ++i2) {
          for (il::int_t i1 = 0; i1 < n1; ++i1) {
            for (il::int_t i0 = 0; i0 < n0; ++i0) {
              new (data_ + ((i3 * r2 + i2) * r1 + i1) * r0 + i0) T(x);
            }
          }
        }
      }
    }
  } else {
    data_ = nullptr;
  }
#ifdef IL_DEBUG_VISUALIZER
  debug_size_[0] = n0;
  debug_size_[1] = n1;
  debug_size_[2] = n2;
  debug_size_[3] = n3;
  debug_capacity_[0] = r0;
  debug_capacity_[1] = r1;
  debug_capacity_[2] = r2;
  debug_capacity_[3] = r3;
#endif
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  size_[2] = data_ + n2;
  size_[3] = data_ + n3;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  capacity_[2] = data_ + r2;
  capacity_[3] = data_ + r3;
}

template <typename T>
Array4D<T>::Array4D(const Array4D<T>& A) {
  const il::int_t n0 = A.size(0);
  const il::int_t n1 = A.size(1);
  const il::int_t n2 = A.size(2);
  const il::int_t n3 = A.size(3);
  il::int_t r0;
  il::int_t r1;
  il::int_t r2;
  il::int_t r3;
  if (n0 > 0 && n1 > 0 && n2 > 0 && n3 > 0) {
    r0 = n0;
    r1 = n1;
    r2 = n2;
    r3 = n3;
  } else if (n0 == 0 && n1 == 0 && n2 == 0 && n3 == 0) {
    r0 = 0;
    r1 = 0;
    r2 = 0;
    r3 = 0;
  } else {
    r0 = (n0 == 0) ? 1 : n0;
    r1 = (n1 == 0) ? 1 : n1;
    r2 = (n2 == 0) ? 1 : n2;
    r3 = (n3 == 0) ? 1 : n3;
  }
  const il::int_t r = r0 * r1 * r2 * r3;
  if (std::is_pod<T>::value) {
    data_ = new T[r];
    for (il::int_t i3 = 0; i3 < n3; ++i3) {
      for (il::int_t i2 = 0; i2 < n2; ++i2) {
        for (il::int_t i1 = 0; i1 < n1; ++i1) {
          memcpy(data_ + ((i3 * r2 + i2) * r1 + i1) * r0,
                 A.data_ +
                     ((i3 * A.capacity(2) + i2) * A.capacity(1) + i1) *
                         A.capacity(0),
                 n0 * sizeof(T));
        }
      }
    }
  } else {
    data_ = static_cast<T*>(::operator new(r * sizeof(T)));
    for (il::int_t i3 = 0; i3 < n3; ++i3) {
      for (il::int_t i2 = 0; i2 < n2; ++i2) {
        for (il::int_t i1 = 0; i1 < n1; ++i1) {
          for (il::int_t i0 = 0; i0 < n0; ++i0) {
            new (data_ + ((i3 * r2 + i2) * r1 + i1) * r0 + i0)
                T(A(i0, i1, i2, i3));
          }
        }
      }
    }
  }
#ifdef IL_DEBUG_VISUALIZER
  debug_size_[0] = n0;
  debug_size_[1] = n1;
  debug_size_[2] = n2;
  debug_size_[3] = n3;
  debug_capacity_[0] = r0;
  debug_capacity_[1] = r1;
  debug_capacity_[2] = r2;
  debug_capacity_[3] = r3;
#endif
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  size_[2] = data_ + n2;
  size_[3] = data_ + n3;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  capacity_[2] = data_ + r2;
  capacity_[3] = data_ + r3;
}

template <typename T>
Array4D<T>::Array4D(Array4D<T>&& A) {
#ifdef IL_DEBUG_VISUALIZER
  debug_size_[0] = A.debug_size_[0];
  debug_size_[1] = A.debug_size_[1];
  debug_size_[2] = A.debug_size_[2];
  debug_size_[3] = A.debug_size_[3];
  debug_capacity_[0] = A.debug_capacity_[0];
  debug_capacity_[1] = A.debug_capacity_[1];
  debug_capacity_[2] = A.debug_capacity_[2];
  debug_capacity_[3] = A.debug_capacity_[3];
#endif
  data_ = A.data_;
  size_[0] = A.size_[0];
  size_[1] = A.size_[1];
  size_[2] = A.size_[2];
  size_[3] = A.size_[3];
  capacity_[0] = A.capacity_[0];
  capacity_[1] = A.capacity_[1];
  capacity_[2] = A.capacity_[2];
  capacity_[3] = A.capacity_[3];
#ifdef IL_DEBUG_VISUALIZER
  A.debug_size_[0] = 0;
  A.debug_size_[1] = 0;
  A.debug_size_[2] = 0;
  A.debug_size_[3] = 0;
  A.debug_capacity_[0] = 0;
  A.debug_capacity_[1] = 0;
  A.debug_capacity_[2] = 0;
  A.debug_capacity_[3] = 0;
#endif
  A.data_ = nullptr;
  A.size_[0] = nullptr;
  A.size_[1] = nullptr;
  A.size_[2] = nullptr;
  A.size_[3] = nullptr;
  A.capacity_[0] = nullptr;
  A.capacity_[1] = nullptr;
  A.capacity_[2] = nullptr;
  A.capacity_[3] = nullptr;
}

template <typename T>
Array4D<T>& Array4D<T>::operator=(const Array4D<T>& A) {
  if (this != &A) {
    const il::int_t n0 = A.size(0);
    const il::int_t n1 = A.size(1);
    const il::int_t n2 = A.size(2);
    const il::int_t n3 = A.size(3);
    il::int_t r0;
    il::int_t r1;
    il::int_t r2;
    il::int_t r3;
    if (n0 > 0 && n1 > 0 && n2 > 0 && n3 > 0) {
      r0 = n0;
      r1 = n1;
      r2 = n2;
      r3 = n3;
    } else if (n0 == 0 && n1 == 0 && n2 == 0 && n3 == 0) {
      r0 = 0;
      r1 = 0;
      r2 = 0;
      r3 = 0;
    } else {
      r0 = (n0 == 0) ? 1 : n0;
      r1 = (n1 == 0) ? 1 : n1;
      r2 = (n2 == 0) ? 1 : n2;
      r3 = (n3 == 0) ? 1 : n3;
    }
    const il::int_t r = r0 * r1 * r2 * r3;
    const bool needs_memory = r0 > capacity(0) || r1 > capacity(1) ||
                              r2 > capacity(2) || r3 > capacity(3);
    if (needs_memory) {
      if (std::is_pod<T>::value) {
        if (data_) {
          delete[] data_;
        }
        data_ = new T[r];
        for (il::int_t i3 = 0; i3 < n3; ++i3) {
          for (il::int_t i2 = 0; i2 < n2; ++i2) {
            for (il::int_t i1 = 0; i1 < n1; ++i1) {
              memcpy(data_ + ((i3 * r2 + i2) * r1 + i1) * r0,
                     A.data_ +
                         ((i3 * A.capacity(2) + i2) * A.capacity(1) + i1) *
                             A.capacity(0),
                     n0 * sizeof(T));
            }
          }
        }
      } else {
        if (data_) {
          for (il::int_t i3 = size(3) - 1; i3 >= 0; --i3) {
            for (il::int_t i2 = size(2) - 1; i2 >= 0; --i2) {
              for (il::int_t i1 = size(1) - 1; i1 >= 0; --i1) {
                for (il::int_t i0 = size(0) - 1; i0 >= 0; --i0) {
                  (data_ +
                   ((i3 * capacity(2) + i2) * capacity(1) + i1) * capacity(0) +
                   i0)->~T();
                }
              }
            }
          }
          ::operator delete(data_);
        }
        data_ = static_cast<T*>(::operator new(r * sizeof(T)));
        for (il::int_t i3 = 0; i3 < n3; ++i3) {
          for (il::int_t i2 = 0; i2 < n2; ++i2) {
            for (il::int_t i1 = 0; i1 < n1; ++i1) {
              for (il::int_t i0 = 0; i0 < n0; ++i0) {
                new (data_ + ((i3 * r2 + i2) * r1 + i1) * r0 + i0)
                    T(A(i0, i1, i2, i3));
              }
            }
          }
        }
      }
#ifdef IL_DEBUG_VISUALIZER
      debug_size_[0] = n0;
      debug_size_[1] = n1;
      debug_size_[2] = n2;
      debug_size_[3] = n3;
      debug_capacity_[0] = r0;
      debug_capacity_[1] = r1;
      debug_capacity_[2] = r2;
      debug_capacity_[3] = r3;
#endif
      size_[0] = data_ + n0;
      size_[1] = data_ + n1;
      size_[2] = data_ + n2;
      size_[3] = data_ + n3;
      capacity_[0] = data_ + r0;
      capacity_[1] = data_ + r1;
      capacity_[2] = data_ + r2;
      capacity_[3] = data_ + r3;
    } else {
      if (std::is_pod<T>::value) {
        for (il::int_t i3 = 0; i3 < n3; ++i3) {
          for (il::int_t i2 = 0; i2 < n2; ++i2) {
            for (il::int_t i1 = 0; i1 < n1; ++i1) {
              memcpy(data_ +
                         ((i3 * capacity(2) + i2) * capacity(1) + i1) *
                             capacity(0),
                     A.data_ +
                         ((i3 * A.capacity(2) + i2) * A.capacity(1) + i1) *
                             A.capacity(0),
                     n0 * sizeof(T));
            }
          }
        }
      } else {
        for (il::int_t i3 = 0; i3 < n3; ++i3) {
          for (il::int_t i2 = 0; i2 < n2; ++i2) {
            for (il::int_t i1 = 0; i1 < n1; ++i1) {
              for (il::int_t i0 = 0; i0 < n0; ++i0) {
                data_[((i3 * capacity(2) + i2) * capacity(1) + i1) *
                          capacity(0) +
                      i0] = A(i0, i1, i2, i3);
              }
            }
          }
        }
        for (il::int_t i3 = size(3) - 1; i3 >= 0; --i3) {
          for (il::int_t i2 = size(2) - 1; i2 >= 0; --i2) {
            for (il::int_t i1 = size(1) - 1; i1 >= 0; --i1) {
              for (il::int_t i0 = size(0) - 1;
                   i0 >= (((i1 < n1) && (i2 < n2) && (i3 < n3)) ? n0 : 0);
                   --i0) {
                (data_ +
                 ((i3 * capacity(2) + i2) * capacity(1) + i1) * capacity(0) +
                 i0)->~T();
              }
            }
          }
        }
      }
#ifdef IL_DEBUG_VISUALIZER
      debug_size_[0] = n0;
      debug_size_[1] = n1;
      debug_size_[2] = n2;
      debug_size_[3] = n3;
#endif
      size_[0] = data_ + n0;
      size_[1] = data_ + n1;
      size_[2] = data_ + n2;
      size_[3] = data_ + n3;
    }
  }
  return *this;
}

template <typename T>
Array4D<T>& Array4D<T>::operator=(Array4D<T>&& A) {
  if (this != &A) {
    if (data_) {
      if (std::is_pod<T>::value) {
        delete[] data_;
      } else {
        for (il::int_t i3 = size(3) - 1; i3 >= 0; --i3) {
          for (il::int_t i2 = size(2) - 1; i2 >= 0; --i2) {
            for (il::int_t i1 = size(1) - 1; i1 >= 0; --i1) {
              for (il::int_t i0 = size(0) - 1; i0 >= 0; --i0) {
                (data_ +
                 ((i3 * capacity(2) + i2) * capacity(1) + i1) * capacity(0) +
                 i0)->~T();
              }
            }
          }
        }
        ::operator delete(data_);
      }
    }
#ifdef IL_DEBUG_VISUALIZER
    debug_size_[0] = A.debug_size_[0];
    debug_size_[1] = A.debug_size_[1];
    debug_size_[2] = A.debug_size_[2];
    debug_size_[3] = A.debug_size_[3];
    debug_capacity_[0] = A.debug_capacity_[0];
    debug_capacity_[1] = A.debug_capacity_[1];
    debug_capacity_[2] = A.debug_capacity_[2];
    debug_capacity_[3] = A.debug_capacity_[3];
#endif
    data_ = A.data_;
    size_[0] = A.size_[0];
    size_[1] = A.size_[1];
    size_[2] = A.size_[2];
    size_[3] = A.size_[3];
    capacity_[0] = A.capacity_[0];
    capacity_[1] = A.capacity_[1];
    capacity_[2] = A.capacity_[2];
    capacity_[3] = A.capacity_[3];
#ifdef IL_DEBUG_VISUALIZER
    A.debug_size_[0] = 0;
    A.debug_size_[1] = 0;
    A.debug_size_[2] = 0;
    A.debug_size_[3] = 0;
    A.debug_capacity_[0] = 0;
    A.debug_capacity_[1] = 0;
    A.debug_capacity_[2] = 0;
    A.debug_capacity_[3] = 0;
#endif
    A.data_ = nullptr;
    A.size_[0] = nullptr;
    A.size_[1] = nullptr;
    A.size_[2] = nullptr;
    A.size_[3] = nullptr;
    A.capacity_[0] = nullptr;
    A.capacity_[1] = nullptr;
    A.capacity_[2] = nullptr;
    A.capacity_[3] = nullptr;
  }
  return *this;
}

template <typename T>
Array4D<T>::~Array4D() {
#ifndef NDEBUG
  check_invariance();
#endif
  if (data_) {
    if (std::is_pod<T>::value) {
      delete[] data_;
    } else {
      for (il::int_t i3 = size(3) - 1; i3 >= 0; --i3) {
        for (il::int_t i2 = size(2) - 1; i2 >= 0; --i2) {
          for (il::int_t i1 = size(1) - 1; i1 >= 0; --i1) {
            for (il::int_t i0 = size(0) - 1; i0 >= 0; --i0) {
              (data_ +
               ((i3 * capacity(2) + i2) * capacity(1) + i1) * capacity(0) +
               i0)->~T();
            }
          }
        }
      }
      ::operator delete(data_);
    }
  }
}

template <typename T>
const T& Array4D<T>::operator()(il::int_t i0, il::int_t i1, il::int_t i2,
                                il::int_t i3) const {
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i0) <
                   static_cast<il::uint_t>(size(0)));
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i1) <
                   static_cast<il::uint_t>(size(1)));
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i2) <
                   static_cast<il::uint_t>(size(2)));
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i3) <
                   static_cast<il::uint_t>(size(3)));
  return data_[((i3 * (capacity_[2] - data_) + i2) * (capacity_[1] - data_) +
                i1) *
                   (capacity_[0] - data_) +
               i0];
}

template <typename T>
T& Array4D<T>::operator()(il::int_t i0, il::int_t i1, il::int_t i2,
                          il::int_t i3) {
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i0) <
                   static_cast<il::uint_t>(size(0)));
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i1) <
                   static_cast<il::uint_t>(size(1)));
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i2) <
                   static_cast<il::uint_t>(size(2)));
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i3) <
                   static_cast<il::uint_t>(size(3)));
  return data_[((i3 * (capacity_[2] - data_) + i2) * (capacity_[1] - data_) +
                i1) *
                   (capacity_[0] - data_) +
               i0];
}

template <typename T>
il::int_t Array4D<T>::size(il::int_t d) const {
  IL_ASSERT(static_cast<il::uint_t>(d) < static_cast<il::uint_t>(4));
  return static_cast<il::int_t>(size_[d] - data_);
}

template <typename T>
void Array4D<T>::resize(il::int_t n0, il::int_t n1, il::int_t n2,
                        il::int_t n3) {
  IL_ASSERT_PRECOND(n0 >= 0);
  IL_ASSERT_PRECOND(n1 >= 0);
  IL_ASSERT_PRECOND(n2 >= 0);
  IL_ASSERT_PRECOND(n3 >= 0);
  if (n0 <= capacity(0) && n1 <= capacity(1) && n2 <= capacity(2) &&
      n3 <= capacity(3)) {
    if (std::is_pod<T>::value) {
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i3 = 0; i3 < n3; ++i3) {
        for (il::int_t i2 = 0; i2 < n2; ++i2) {
          for (il::int_t i1 = 0; i1 < n1; ++i1) {
            for (il::int_t i0 =
                     ((i1 < size(1) && i2 < size(2) && i3 < size(3)) ? size(0)
                                                                     : 0);
                 i0 < n0; ++i0) {
              data_[((i3 * capacity(2) + i2) * capacity(1) + i1) * capacity(0) +
                    i0] = il::default_value<T>();
            }
          }
        }
      }
#endif
    } else {
      for (il::int_t i3 = size(3) - 1; i3 >= 0; --i3) {
        for (il::int_t i2 = size(2) - 1; i2 >= 0; --i2) {
          for (il::int_t i1 = size(1) - 1; i1 >= 0; --i1) {
            for (il::int_t i0 = size(0) - 1;
                 i0 >= ((i1 < n1 && i2 < n2 && i3 < n3) ? n0 : 0); --i0) {
              (data_ +
               ((i3 * capacity(2) + i2) * capacity(1) + i1) * capacity(0) +
               i0)->~T();
            }
          }
        }
      }
    }
  } else {
    const il::int_t n0_old = size(0);
    const il::int_t n1_old = size(1);
    const il::int_t n2_old = size(2);
    const il::int_t n3_old = size(3);
    il::int_t r0;
    il::int_t r1;
    il::int_t r2;
    il::int_t r3;
    if (n0 > 0 && n1 > 0 && n2 > 0 && n3 > 0) {
      r0 = n0;
      r1 = n1;
      r2 = n2;
      r3 = n3;
    } else if (n0 == 0 && n1 == 0 && n2 == 0 && n3 == 0) {
      r0 = 0;
      r1 = 0;
      r2 = 0;
      r3 = 0;
    } else {
      r0 = (n0 == 0) ? 1 : n0;
      r1 = (n1 == 0) ? 1 : n1;
      r2 = (n2 == 0) ? 1 : n2;
      r3 = (n3 == 0) ? 1 : n3;
    }
    const il::int_t r = r0 * r1 * r2 * r3;
    T* new_data;
    if (std::is_pod<T>::value) {
      new_data = new T[r];
    } else {
      new_data = static_cast<T*>(::operator new(r * sizeof(T)));
    }
    if (data_) {
      if (std::is_pod<T>::value) {
        for (il::int_t i3 = 0; i3 < n3; ++i3) {
          for (il::int_t i2 = 0; i2 < n2; ++i2) {
            for (il::int_t i1 = 0; i1 < n1; ++i1) {
              memcpy(new_data + ((i3 * r2 + i2) * r1 + i1) * r0,
                     data_ +
                         ((i3 * capacity(2) + i2) * capacity(1) + i1) *
                             capacity(0),
                     size(0) * sizeof(T));
            }
          }
        }
        delete[] data_;
      } else {
        for (il::int_t i3 = 0; i3 < n3_old; ++i3) {
          for (il::int_t i2 = 0; i2 < n2_old; ++i2) {
            for (il::int_t i1 = 0; i1 < n1_old; ++i1) {
              for (il::int_t i0 = 0; i0 < n0_old; ++i0) {
                new (new_data + ((i3 * r2 + i2) * r1 + i1) * r0 + i0)
                    T(std::move(
                        data_[((i3 * capacity(2) + i2) * capacity(1) + i1) *
                                  capacity(0) +
                              i0]));
                (data_ +
                 ((i3 * capacity(2) + i2) * capacity(1) + i1) * capacity(0) +
                 i0)->~T();
              }
            }
          }
        }
        ::operator delete(data_);
      }
    }
    if (std::is_pod<T>::value) {
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i3 = 0; i3 < n3; ++i3) {
        for (il::int_t i2 = 0; i2 < n2; ++i2) {
          for (il::int_t i1 = 0; i1 < n1; ++i1) {
            for (il::int_t i0 =
                     ((i1 < n1_old && i2 < n2_old && i3 < n3_old) ? n0_old : 0);
                 i0 < n0; ++i0) {
              new_data[((i3 * r2 + i2) * r1 + i1) * r0 + i0] =
                  il::default_value<T>();
            }
          }
        }
      }
#endif
    } else {
      for (il::int_t i3 = 0; i3 < n3; ++i3) {
        for (il::int_t i2 = 0; i2 < n2; ++i2) {
          for (il::int_t i1 = 0; i1 < n1; ++i1) {
            for (il::int_t i0 =
                     ((i1 < n1_old && i2 < n2_old && i3 < n3_old) ? n0_old : 0);
                 i0 < n0; ++i0) {
              new (new_data + ((i3 * r2 + i2) * r1 + i1) * r0 + i0) T{};
            }
          }
        }
      }
    }
    data_ = new_data;
#ifdef IL_DEBUG_VISUALIZER
    debug_capacity_[0] = r0;
    debug_capacity_[1] = r1;
    debug_capacity_[2] = r2;
    debug_capacity_[3] = r3;
#endif
    capacity_[0] = data_ + r0;
    capacity_[1] = data_ + r1;
    capacity_[2] = data_ + r2;
    capacity_[3] = data_ + r3;
  }
#ifdef IL_DEBUG_VISUALIZER
  debug_size_[0] = n0;
  debug_size_[1] = n1;
  debug_size_[2] = n2;
  debug_size_[3] = n3;
#endif
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  size_[2] = data_ + n2;
  size_[3] = data_ + n3;
}

template <typename T>
il::int_t Array4D<T>::capacity(il::int_t d) const {
  IL_ASSERT(static_cast<il::uint_t>(d) < static_cast<il::uint_t>(4));
  return static_cast<il::int_t>(capacity_[d] - data_);
}

template <typename T>
void Array4D<T>::reserve(il::int_t r0, il::int_t r1, il::int_t r2,
                         il::int_t r3) {
  IL_ASSERT_PRECOND(r0 >= 0);
  IL_ASSERT_PRECOND(r1 >= 0);
  IL_ASSERT_PRECOND(r2 >= 0);
  IL_ASSERT_PRECOND(r3 >= 0);
  if (r0 > capacity(0) || r1 > capacity(1) || r2 > capacity(2) ||
      r3 > capacity(3)) {
    const il::int_t n0_old = size(0);
    const il::int_t n1_old = size(1);
    const il::int_t n2_old = size(2);
    const il::int_t n3_old = size(3);
    r0 = (r0 == 0) ? 1 : r0;
    r1 = (r1 == 0) ? 1 : r1;
    r2 = (r2 == 0) ? 1 : r2;
    r3 = (r3 == 0) ? 1 : r3;
    const il::int_t r = r0 * r1 * r2 * r3;
    T* new_data;
    if (std::is_pod<T>::value) {
      new_data = new T[r];
    } else {
      new_data = static_cast<T*>(::operator new(r * sizeof(T)));
    }
    if (data_) {
      if (std::is_pod<T>::value) {
        for (il::int_t i3 = 0; i3 < n3_old; ++i3) {
          for (il::int_t i2 = 0; i2 < n2_old; ++i2) {
            for (il::int_t i1 = 0; i1 < n1_old; ++i1) {
              memcpy(new_data + ((i3 * r2 + i2) * r1 + i1) * r0,
                     data_ +
                         ((i3 * capacity(2) + i2) * capacity(1) + i1) *
                             capacity(0),
                     n0_old * sizeof(T));
            }
          }
        }
        delete[] data_;
      } else {
        for (il::int_t i3 = 0; i3 < n3_old; ++i3) {
          for (il::int_t i2 = 0; i2 < n2_old; ++i2) {
            for (il::int_t i1 = 0; i1 < n1_old; ++i1) {
              for (il::int_t i0 = 0; i0 < n0_old; ++i0) {
                new (new_data + ((i3 * r2 + i2) * r1 + i1) * r0 + i0)
                    T(std::move(
                        data_[((i3 * capacity(2) + i2) * capacity(1) + i1) *
                                  capacity(0) +
                              i0]));
                (data_ +
                 ((i3 * capacity(2) + i2) * capacity(1) + i1) * capacity(0) +
                 i0)->~T();
              }
            }
          }
        }
        ::operator delete(data_);
      }
    }
    data_ = new_data;
#ifdef IL_DEBUG_VISUALIZER
    debug_capacity_[0] = r0;
    debug_capacity_[1] = r1;
    debug_capacity_[2] = r2;
    debug_capacity_[3] = r3;
#endif
    capacity_[0] = data_ + r0;
    capacity_[1] = data_ + r1;
    capacity_[2] = data_ + r2;
    capacity_[3] = data_ + r3;
    size_[0] = data_ + n0_old;
    size_[1] = data_ + n1_old;
    size_[2] = data_ + n2_old;
    size_[3] = data_ + n3_old;
  }
}

template <typename T>
const T* Array4D<T>::data() const {
  return data_;
}

template <typename T>
T* Array4D<T>::data() {
  return data_;
}

template <typename T>
void Array4D<T>::check_invariance() const {
  if (data_ == nullptr) {
    IL_ASSERT(size_[0] == nullptr);
    IL_ASSERT(size_[1] == nullptr);
    IL_ASSERT(size_[2] == nullptr);
    IL_ASSERT(size_[3] == nullptr);
    IL_ASSERT(capacity_[0] == nullptr);
    IL_ASSERT(capacity_[1] == nullptr);
    IL_ASSERT(capacity_[2] == nullptr);
    IL_ASSERT(capacity_[3] == nullptr);
  } else {
    IL_ASSERT(size_[0] != nullptr);
    IL_ASSERT(size_[1] != nullptr);
    IL_ASSERT(size_[2] != nullptr);
    IL_ASSERT(size_[3] != nullptr);
    IL_ASSERT(capacity_[0] != nullptr);
    IL_ASSERT(capacity_[1] != nullptr);
    IL_ASSERT(capacity_[2] != nullptr);
    IL_ASSERT(capacity_[3] != nullptr);
    IL_ASSERT((size_[0] - data_) <= (capacity_[0] - data_));
    IL_ASSERT((size_[1] - data_) <= (capacity_[1] - data_));
    IL_ASSERT((size_[2] - data_) <= (capacity_[2] - data_));
    IL_ASSERT((size_[3] - data_) <= (capacity_[3] - data_));
  }
}
}

#endif  // IL_ARRAY4D_H
