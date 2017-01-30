//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_ARRAY2D_H
#define IL_ARRAY2D_H

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
class Array2D {
 private:
#ifdef IL_DEBUG_VISUALIZER
  il::int_t debug_size_[2];
  il::int_t debug_capacity_[2];
#endif
  T* data_;
  T* size_[2];
  T* capacity_[2];
  short align_mod_;
  short align_r_;
  short new_shift_;

 public:
  /* \brief Default constructor
  // \details The size and the capacity of the array are set to 0 and no memory
  // allocation is done during the process.
  */
  Array2D();

  /* \brief Construct an il::Array2D<T> of n0 rows and n1 columns
  // \details The row size and the row capacity of the array are set to n0. The
  // column size and the column capacity of the array are set to n1.
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
  // // Construct an array of double with 3 rows and 5 columns
  // il::Array2D<double> A{3, 5};
  */
  explicit Array2D(il::int_t n0, il::int_t n1);

  /* \brief Construct an aligned array
  // \details The pointer data, when considered as an integer, satisfies
  // data_ = 0 (Modulo align_mod)
  */
  explicit Array2D(il::int_t n0, il::int_t n1, il::align_t, short align_mod);

  /* \brief Construct an aligned array
  // \details The pointer data, when considered as an integer, satisfies
  // data_ = align_r (Modulo align_mod)
  */
  explicit Array2D(il::int_t n0, il::int_t n1, il::align_t, short align_r,
                   short align_mod);

  /* \brief Construct an array of n rows and p columns with a value
  /
  // // Construct an array of double with 3 rows and 5 columns, initialized with
  // // 3.14.
  // il::Array2D<double> A{3, 5, 3.14};
  */
  explicit Array2D(il::int_t n0, il::int_t n1, const T& x);

  /* \brief Construct an array of n rows and p columns from a brace-initialized
  // list
  //
  // // Construct an array of double with 2 rows and 3 columns from a list
  // il::Array2D<double> v{2, 3, il::value, {2.0, 3.14, 5.0, 7.0, 8.0, 9.0}};
  */
  explicit Array2D(il::value_t,
                   std::initializer_list<std::initializer_list<T>> list);

  /* \brief The copy constructor
  // \details The different size and capacity of the constructed il::Array2D<T>
  // are equal to the size of the source array.
  */
  Array2D(const Array2D<T>& A);

  /* \brief The move constructor
  */
  Array2D(Array2D<T>&& A);

  /* \brief The copy assignment
  // \details The size is the same as the one for the source array. The
  // capacity is not changed if it is enough for the copy to happen and is
  // set to the size of the source array if the initial capacity is too low.
  */
  Array2D& operator=(const Array2D<T>& A);

  /* \brief The move assignment
  */
  Array2D& operator=(Array2D<T>&& A);

  /* \brief The destructor
  */
  ~Array2D();

  /* \brief Accessor for a const il::2DArray<T>
  // \details Access (read only) the (i, j)-th element of the array. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::Array2D<double> v{4, 6};
  // std::cout << v(3, 5) << std::endl;
  */
  const T& operator()(il::int_t i0, il::int_t i1) const;

  /* \brief Accessor for a il::2DArray<T>
  // \details Access (read or write) the (i, j)-th element of the array. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::Array2D<double> v{4, 6};
  // v(3, 5) = 0.0;
  // v(5, 3) = 0.0; // Program is aborted in debug mode and has undefined
  //                // behavior in release mode
  */
  T& operator()(il::int_t i0, il::int_t i1);

  /* \brief Get the size of the il::Array2D<T>
  // \details size(0) returns the number of rows of the array and size(1)
  // returns the number of columns of the same array. The library has been
  // designed in a way that any compiler can prove that modifying A(i, j) can't
  // change the result of A.size(0) or A.size(1). As a consequence
  // a call to A.size(0) and A.size(1) are made just once at the very beginning
  // of the loop in the following example. It allows many optimizations from the
  // compiler, including automatic vectorization.
  //
  // il::Array2D<double> A{n, p};
  // for (il::int_t i = 0; i < v.size(0); ++i) {
  //   for (il::int_t j = 0; j < v.size(1); ++j) {
  //     A(i, j) = 1.0 / (i + j + 2);
  //   }
  // }
  */
  il::int_t size(il::int_t d) const;

  /* \brief Resizing an il::Array2D<T>
  // \details No reallocation is performed if the new size is <= to the
  // capacity for both rows and columns. In this case, the capacity is
  // unchanged. When one of the sizes is > than the the capacity, reallocation
  // is done and the array gets the same capacity as its size.
  */
  void resize(il::int_t n0, il::int_t n1);

  /* \brief Get the capacity of the il::Array2D<T>
  // \details capacity(0) gives the capacity in terms of rows and capacity(1)
  // gives the capacity in terms of columns.
  */
  il::int_t capacity(il::int_t d) const;

  /* \brief Change the capacity of the array to at least r rows and s columns
  // \details If the row capacity is >= to r and the column capacity is >= to s,
  // nothing is done. Otherwise, reallocation is done and the new capacity is
  // set to r and s.
  */
  void reserve(il::int_t r0, il::int_t r1);

  /* \brief Get the alignment of the pointer returned by data()
  */
  short alignment() const;

  /* \brief Get a pointer to const to the first element of the array
  // \details One should use this method only when using C-style API
  */
  const T* data() const;

  /* \brief Get a pointer to the first element of the array
  // \details One should use this method only when using C-style API
  */
  T* data();

  /* \brief Get a pointer to the first element of the column
  // \details One should use this method only when using C-style API
  */
  const T* data(il::int_t i1) const;

  /* \brief Get a pointer to the first element of the column
  // \details One should use this method only when using C-style API
  */
  T* data(il::int_t i1);

  /* \brief The memory position of A(i0, i1) is
  // data() + i1 * stride(1) + i0
  */
  il::int_t stride(il::int_t d) const;

 private:
  T* allocate(il::int_t n, short align_mod, short align_r, il::io_t,
              short& new_shift);

  /* \brief Used internally in debug mode to check the invariance of the object
  */
  void check_invariance() const;
};

template <typename T>
Array2D<T>::Array2D() {
#ifdef IL_DEBUG_VISUALIZER
  debug_size_[0] = 0;
  debug_size_[1] = 0;
  debug_capacity_[0] = 0;
  debug_capacity_[1] = 0;
#endif
  data_ = nullptr;
  size_[0] = nullptr;
  size_[1] = nullptr;
  capacity_[0] = nullptr;
  capacity_[1] = nullptr;
  align_mod_ = 0;
  align_r_ = 0;
  new_shift_ = 0;
}

template <typename T>
Array2D<T>::Array2D(il::int_t n0, il::int_t n1) {
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);
  il::int_t r0;
  il::int_t r1;
  if (n0 > 0 && n1 > 0) {
    r0 = n0;
    r1 = n1;
  } else if (n0 == 0 && n1 == 0) {
    r0 = 0;
    r1 = 0;
  } else {
    r0 = (n0 == 0) ? 1 : n0;
    r1 = (n1 == 0) ? 1 : n1;
  }
  il::int_t r{r0 * r1};
  if (r > 0) {
    if (il::is_trivial<T>::value) {
      data_ = new T[r];
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = 0; i0 < n0; ++i0) {
          data_[i1 * r0 + i0] = il::default_value<T>();
        }
      }
#endif
    } else {
      data_ = static_cast<T*>(::operator new(r * sizeof(T)));
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = 0; i0 < n0; ++i0) {
          new (data_ + i1 * r0 + i0) T{};
        }
      }
    }
  } else {
    data_ = nullptr;
  }
#ifdef IL_DEBUG_VISUALIZER
  debug_size_[0] = n0;
  debug_size_[1] = n1;
  debug_capacity_[0] = r0;
  debug_capacity_[1] = r1;
#endif
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  align_mod_ = 0;
  align_r_ = 0;
  new_shift_ = 0;
}

template <typename T>
Array2D<T>::Array2D(il::int_t n0, il::int_t n1, il::align_t, short align_r,
                    short align_mod) {
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);
  IL_EXPECT_FAST(align_mod >= 0);
  IL_EXPECT_FAST(align_mod % sizeof(T) == 0);
  IL_EXPECT_FAST(align_r >= 0);
  IL_EXPECT_FAST(align_r < align_mod);
  IL_EXPECT_FAST(align_r % sizeof(T) == 0);
  align_mod_ = align_mod;
  align_r_ = align_r;
  il::int_t r0;
  il::int_t r1;
  if (n0 > 0 && n1 > 0) {
    if (il::is_trivial<T>::value && align_mod != 0) {
      const il::int_t nb_lanes{static_cast<il::int_t>(alignment() / sizeof(T))};
      r0 = ((n0 - 1) / nb_lanes + 1) * nb_lanes;
      r1 = n1;
    } else {
      r0 = n0;
      r1 = n1;
    }
  } else if (n0 == 0 && n1 == 0) {
    r0 = 0;
    r1 = 0;
  } else {
    r0 = (n0 == 0) ? 1 : n0;
    r1 = (n1 == 0) ? 1 : n1;
  }
  il::int_t r{r0 * r1};
  if (r > 0) {
    if (il::is_trivial<T>::value) {
      if (align_mod == 0) {
        data_ = new T[r];
        new_shift_ = 0;
      } else {
        data_ = allocate(r, align_mod, align_r, il::io, new_shift_);
      }
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = 0; i0 < n0; ++i0) {
          data_[i1 * r0 + i0] = il::default_value<T>();
        }
      }
#endif
    } else {
      data_ = static_cast<T*>(::operator new(r * sizeof(T)));
      new_shift_ = 0;
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = 0; i0 < n0; ++i0) {
          new (data_ + i1 * r0 + i0) T{};
        }
      }
    }
  } else {
    data_ = nullptr;
    new_shift_ = 0;
  }
#ifdef IL_DEBUG_VISUALIZER
  debug_size_[0] = n0;
  debug_size_[1] = n1;
  debug_capacity_[0] = r0;
  debug_capacity_[1] = r1;
#endif
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
}

template <typename T>
Array2D<T>::Array2D(il::int_t n0, il::int_t n1, il::align_t, short align_mod)
    : Array2D{n0, n1, il::align, 0, align_mod} {}

template <typename T>
Array2D<T>::Array2D(il::int_t n0, il::int_t n1, const T& x) {
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);
  il::int_t r0;
  il::int_t r1;
  if (n0 > 0 && n1 > 0) {
    r0 = n0;
    r1 = n1;
  } else if (n0 == 0 && n1 == 0) {
    r0 = 0;
    r1 = 0;
  } else {
    r0 = (n0 == 0) ? 1 : n0;
    r1 = (n1 == 0) ? 1 : n1;
  }
  il::int_t r{r0 * r1};
  if (r > 0) {
    if (il::is_trivial<T>::value) {
      data_ = new T[r];
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = 0; i0 < n0; ++i0) {
          data_[i1 * r0 + i0] = x;
        }
      }
    } else {
      data_ = static_cast<T*>(::operator new(r * sizeof(T)));
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = 0; i0 < n0; ++i0) {
          new (data_ + i1 * r0 + i0) T(x);
        }
      }
    }
  } else {
    data_ = nullptr;
  }
#ifdef IL_DEBUG_VISUALIZER
  debug_size_[0] = n0;
  debug_size_[1] = n1;
  debug_capacity_[0] = r0;
  debug_capacity_[1] = r1;
#endif
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  align_mod_ = 0;
  align_r_ = 0;
  new_shift_ = 0;
}

template <typename T>
Array2D<T>::Array2D(il::value_t,
                    std::initializer_list<std::initializer_list<T>> list) {
  const il::int_t n1{static_cast<il::int_t>(list.size())};
  const il::int_t n0{n1 > 0 ? static_cast<il::int_t>(list.begin()->size()) : 0};
  il::int_t r0;
  il::int_t r1;
  if (n0 > 0 && n1 > 0) {
    r0 = n0;
    r1 = n1;
    il::int_t r{r0 * r1};
    if (il::is_trivial<T>::value) {
      data_ = new T[r];
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        IL_EXPECT_FAST(static_cast<il::int_t>((list.begin() + i1)->size()) == n0);
        memcpy(data_ + i1 * r0, (list.begin() + i1)->begin(), n0 * sizeof(T));
      }
    } else {
      data_ = static_cast<T*>(::operator new(r * sizeof(T)));
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        IL_EXPECT_FAST(static_cast<il::int_t>((list.begin() + i1)->size()) == n0);
        for (il::int_t i0 = 0; i0 < n0; ++i0) {
          new (data_ + i1 * r0 + i0) T(*((list.begin() + i1)->begin() + i0));
        }
      }
    }
  } else if (n0 == 0 && n1 == 0) {
    r0 = 0;
    r1 = 0;
    data_ = nullptr;
  } else {
    r0 = (n0 == 0) ? 1 : n0;
    r1 = (n1 == 0) ? 1 : n1;
    il::int_t r{r0 * r1};
    if (il::is_trivial<T>::value) {
      data_ = new T[r];
    } else {
      data_ = static_cast<T*>(::operator new(r * sizeof(T)));
    }
  }
#ifdef IL_DEBUG_VISUALIZER
  debug_size_[0] = n0;
  debug_size_[1] = n1;
  debug_capacity_[0] = r0;
  debug_capacity_[1] = r1;
#endif
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  align_mod_ = 0;
  align_r_ = 0;
  new_shift_ = 0;
}

template <typename T>
Array2D<T>::Array2D(const Array2D<T>& A) {
  il::int_t n0{A.size(0)};
  il::int_t n1{A.size(1)};
  il::int_t r0;
  il::int_t r1;
  if (n0 > 0 && n1 > 0) {
    if (il::is_trivial<T>::value && A.align_mod_ != 0) {
      const il::int_t nb_lanes{
          static_cast<il::int_t>(A.alignment() / sizeof(T))};
      r0 = ((n0 - 1) / nb_lanes + 1) * nb_lanes;
      r1 = n1;
    } else {
      r0 = n0;
      r1 = n1;
    }
  } else if (n0 == 0 && n1 == 0) {
    r0 = 0;
    r1 = 0;
  } else {
    r0 = (n0 == 0) ? 1 : n0;
    r1 = (n1 == 0) ? 1 : n1;
  }
  const il::int_t r{r0 * r1};
  if (il::is_trivial<T>::value) {
    if (A.align_mod_ == 0) {
      data_ = new T[r];
      new_shift_ = 0;
    } else {
      data_ = allocate(r, A.align_mod_, A.align_r_, il::io, new_shift_);
    }
    for (il::int_t i1 = 0; i1 < n1; ++i1) {
      memcpy(data_ + i1 * r0, A.data() + i1 * A.stride(1), n0 * sizeof(T));
    }
  } else {
    data_ = static_cast<T*>(::operator new(r * sizeof(T)));
    new_shift_ = 0;
    for (il::int_t i1 = 0; i1 < n1; ++i1) {
      for (il::int_t i0 = 0; i0 < n0; ++i0) {
        new (data_ + i1 * r0 + i0) T(A(i0, i1));
      }
    }
  }
#ifdef IL_DEBUG_VISUALIZER
  debug_size_[0] = n0;
  debug_size_[1] = n1;
  debug_capacity_[0] = r0;
  debug_capacity_[1] = r1;
#endif
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  align_mod_ = A.align_mod_;
  align_r_ = A.align_r_;
}

template <typename T>
Array2D<T>::Array2D(Array2D<T>&& A) {
#ifdef IL_DEBUG_VISUALIZER
  debug_size_[0] = A.debug_size_[0];
  debug_size_[1] = A.debug_size_[1];
  debug_capacity_[0] = A.debug_capacity_[0];
  debug_capacity_[1] = A.debug_capacity_[1];
#endif
  data_ = A.data_;
  size_[0] = A.size_[0];
  size_[1] = A.size_[1];
  capacity_[0] = A.capacity_[0];
  capacity_[1] = A.capacity_[1];
  align_mod_ = A.align_mod_;
  align_r_ = A.align_r_;
  new_shift_ = A.new_shift_;
#ifdef IL_DEBUG_VISUALIZER
  A.debug_size_[0] = 0;
  A.debug_size_[1] = 0;
  A.debug_capacity_[0] = 0;
  A.debug_capacity_[1] = 0;
#endif
  A.data_ = nullptr;
  A.size_[0] = nullptr;
  A.size_[1] = nullptr;
  A.capacity_[0] = nullptr;
  A.capacity_[1] = nullptr;
  A.align_mod_ = 0;
  A.align_r_ = 0;
  A.new_shift_ = 0;
}

template <typename T>
Array2D<T>& Array2D<T>::operator=(const Array2D<T>& A) {
  if (this != &A) {
    const il::int_t n0{A.size(0)};
    const il::int_t n1{A.size(1)};
    const short align_mod{A.align_mod_};
    const short align_r{A.align_r_};
    if (n0 > capacity(0) || n1 > capacity(1) || align_mod_ != align_mod ||
        align_r_ != align_r) {
      il::int_t r0;
      il::int_t r1;
      if (n0 > 0 && n1 > 0) {
        if (il::is_trivial<T>::value && align_mod != 0) {
          const il::int_t nb_lanes{
              static_cast<il::int_t>(A.alignment() / sizeof(T))};
          r0 = (n0 / nb_lanes + 1) * nb_lanes;
          r1 = n1;
        } else {
          r0 = n0;
          r1 = n1;
        }
      } else {
        r0 = (n0 == 0) ? 1 : n0;
        r1 = (n1 == 0) ? 1 : n1;
      }
      const il::int_t r{r0 * r1};
      if (il::is_trivial<T>::value) {
        if (data_) {
          delete[](data_ - new_shift_);
        }
        if (align_mod == 0) {
          data_ = new T[r];
          new_shift_ = 0;
        } else {
          data_ = allocate(r, align_mod, align_r, il::io, new_shift_);
        }
        for (il::int_t i1 = 0; i1 < n1; ++i1) {
          memcpy(data_ + i1 * r0, A.data_ + i1 * A.stride(1), n0 * sizeof(T));
        }
      } else {
        if (data_) {
          for (il::int_t i1{size(1) - 1}; i1 >= 0; --i1) {
            for (il::int_t i0{size(0) - 1}; i0 >= 0; --i0) {
              (data_ + i1 * stride(1) + i0)->~T();
            }
          }
          ::operator delete(data_);
        }
        data_ = static_cast<T*>(::operator new(r * sizeof(T)));
        new_shift_ = 0;
        for (il::int_t i1 = 0; i1 < n1; ++i1) {
          for (il::int_t i0 = 0; i0 < n0; ++i0) {
            new (data_ + i1 * r0 + i0) T(A(i0, i1));
          }
        }
      }
#ifdef IL_DEBUG_VISUALIZER
      debug_size_[0] = n0;
      debug_size_[1] = n1;
      debug_capacity_[0] = r0;
      debug_capacity_[1] = r1;
#endif
      size_[0] = data_ + n0;
      size_[1] = data_ + n1;
      capacity_[0] = data_ + r0;
      capacity_[1] = data_ + r1;
      align_mod_ = align_mod;
      align_r_ = align_r;
    } else {
      if (il::is_trivial<T>::value) {
        for (il::int_t i1 = 0; i1 < n1; ++i1) {
          memcpy(data_ + i1 * stride(1), A.data_ + i1 * A.stride(1),
                 n0 * sizeof(T));
        }
      } else {
        for (il::int_t i1 = 0; i1 < n1; ++i1) {
          for (il::int_t i0 = 0; i0 < n0; ++i0) {
            data_[i1 * stride(1) + i0] = A(i0, i1);
          }
        }
        for (il::int_t i1{size(1) - 1}; i1 >= 0; --i1) {
          for (il::int_t i0{size(0) - 1}; i0 >= (i1 < n1 ? n0 : 0); --i0) {
            (data_ + i1 * stride(1) + i0)->~T();
          }
        }
      }
#ifdef IL_DEBUG_VISUALIZER
      debug_size_[0] = n0;
      debug_size_[1] = n1;
#endif
      size_[0] = data_ + n0;
      size_[1] = data_ + n1;
    }
  }
  return *this;
}

template <typename T>
Array2D<T>& Array2D<T>::operator=(Array2D<T>&& A) {
  if (this != &A) {
    if (data_) {
      if (il::is_trivial<T>::value) {
        delete[](data_ - new_shift_);
      } else {
        for (il::int_t i1{size(1) - 1}; i1 >= 0; --i1) {
          for (il::int_t i0{size(0) - 1}; i0 >= 0; --i0) {
            (data_ + i1 * stride(1) + i0)->~T();
          }
        }
        ::operator delete(data_);
      }
    }
#ifdef IL_DEBUG_VISUALIZER
    debug_size_[0] = A.debug_size_[0];
    debug_size_[1] = A.debug_size_[1];
    debug_capacity_[0] = A.debug_capacity_[0];
    debug_capacity_[1] = A.debug_capacity_[1];
#endif
    data_ = A.data_;
    size_[0] = A.size_[0];
    size_[1] = A.size_[1];
    capacity_[0] = A.capacity_[0];
    capacity_[1] = A.capacity_[1];
    align_mod_ = A.align_mod_;
    align_r_ = A.align_r_;
    new_shift_ = A.new_shift_;
#ifdef IL_DEBUG_VISUALIZER
    A.debug_size_[0] = 0;
    A.debug_size_[1] = 0;
    A.debug_capacity_[0] = 0;
    A.debug_capacity_[1] = 0;
#endif
    A.data_ = nullptr;
    A.size_[0] = nullptr;
    A.size_[1] = nullptr;
    A.capacity_[0] = nullptr;
    A.capacity_[1] = nullptr;
    A.align_mod_ = 0;
    A.align_r_ = 0;
    A.new_shift_ = 0;
  }
  return *this;
}

template <typename T>
Array2D<T>::~Array2D() {
#ifdef IL_INVARIANCE
  check_invariance();
#endif
  if (data_) {
    if (il::is_trivial<T>::value) {
      delete[](data_ - new_shift_);
    } else {
      for (il::int_t i1{size(1) - 1}; i1 >= 0; --i1) {
        for (il::int_t i0{size(0) - 1}; i0 >= 0; --i0) {
          (data_ + i1 * stride(1) + i0)->~T();
        }
      }
      ::operator delete(data_);
    }
  }
}

template <typename T>
const T& Array2D<T>::operator()(il::int_t i0, il::int_t i1) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) <
                   static_cast<std::size_t>(size(0)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) <
                   static_cast<std::size_t>(size(1)));
  return data_[i1 * (capacity_[0] - data_) + i0];
}

template <typename T>
T& Array2D<T>::operator()(il::int_t i0, il::int_t i1) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) <
                   static_cast<std::size_t>(size(0)));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) <
                   static_cast<std::size_t>(size(1)));
  return data_[i1 * (capacity_[0] - data_) + i0];
}

template <typename T>
il::int_t Array2D<T>::size(il::int_t d) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(d) < static_cast<std::size_t>(2));
  return static_cast<il::int_t>(size_[d] - data_);
}

template <typename T>
void Array2D<T>::resize(il::int_t n0, il::int_t n1) {
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);
  const il::int_t n0_old{size(0)};
  const il::int_t n1_old{size(1)};
  if (n0 > capacity(0) || n1 > capacity(1)) {
    il::int_t r0;
    il::int_t r1;
    if (n0 > 0 && n1 > 0) {
      if (il::is_trivial<T>::value && align_mod_ != 0) {
        const il::int_t nb_lanes{
            static_cast<il::int_t>(alignment() / sizeof(T))};
        r0 = (n0 / nb_lanes + 1) * nb_lanes;
        r1 = n1;
      } else {
        r0 = n0;
        r1 = n1;
      }
    } else if (n0 == 0 && n1 == 0) {
      r0 = 0;
      r1 = 0;
    } else {
      r0 = (n0 == 0) ? 1 : n0;
      r1 = (n1 == 0) ? 1 : n1;
    }
    const il::int_t r{r0 * r1};
    T* new_data;
    short new_shift;
    if (il::is_trivial<T>::value) {
      if (align_mod_ == 0) {
        new_data = new T[r];
        new_shift = 0;
      } else {
        new_data = allocate(r, align_mod_, align_r_, il::io, new_shift);
      }
      if (data_) {
        for (il::int_t i1 = 0; i1 < (n1 < n1_old ? n1 : n1_old); ++i1) {
          memcpy(new_data + i1 * r0, data_ + i1 * stride(1),
                 (n0 < n0_old ? n0 : n0_old) * sizeof(T));
        }
        delete[](data_ - new_shift_);
      }
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0{i1 < n1_old ? n0_old : 0}; i0 < n0; ++i0) {
          new_data[i1 * r0 + i0] = il::default_value<T>();
        }
      }
#endif
    } else {
      new_data = static_cast<T*>(::operator new(r * sizeof(T)));
      new_shift = 0;
      if (data_) {
        for (il::int_t i1{n1_old - 1}; i1 >= 0; --i1) {
          for (il::int_t i0{n0_old - 1}; i0 >= (i1 < n1 ? n0 : 0); --i0) {
            (data_ + i1 * stride(1) + i0)->~T();
          }
        }
        for (il::int_t i1{(n1 < n1_old ? n1 : n1_old) - 1}; i1 >= 0; --i1) {
          for (il::int_t i0{(n0 < n0_old ? n0 : n0_old) - 1}; i0 >= 0; --i0) {
            new (new_data + i1 * r0 + i0)
                T(std::move(data_[i1 * stride(1) + i0]));
            (data_ + i1 * stride(1) + i0)->~T();
          }
        }
        ::operator delete(data_);
      }
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0{i1 < n1_old ? n0_old : 0}; i0 < n0; ++i0) {
          new (new_data + i1 * r0 + i0) T{};
        }
      }
    }
    data_ = new_data;
#ifdef IL_DEBUG_VISUALIZER
    debug_capacity_[0] = n0;
    debug_capacity_[1] = n1;
#endif
    capacity_[0] = data_ + r0;
    capacity_[1] = data_ + r1;
    new_shift_ = new_shift;
  } else {
    if (il::is_trivial<T>::value) {
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0{i1 < n1_old ? n0_old : 0}; i0 < n1; ++i0) {
          data_[i1 * stride(1) + i0] = il::default_value<T>();
        }
      }
#endif
    } else {
      for (il::int_t i1{n1_old - 1}; i1 >= 0; --i1) {
        for (il::int_t i0{n0_old - 1}; i0 >= (i1 < n0 ? n1 : 0); --i0) {
          (data_ + i1 * stride(1) + i0)->~T();
        }
      }
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0{i1 < n1_old ? n0_old : 0}; i0 < n0; ++i0) {
          new (data_ + i1 * stride(1) + i0) T{};
        }
      }
    }
  }
#ifdef IL_DEBUG_VISUALIZER
  debug_size_[0] = n0;
  debug_size_[1] = n1;
#endif
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
}

template <typename T>
il::int_t Array2D<T>::capacity(il::int_t d) const {
  IL_EXPECT_FAST(static_cast<std::size_t>(d) < static_cast<std::size_t>(2));
  return static_cast<il::int_t>((d == 0 ? capacity_[0] : capacity_[1]) - data_);
}

template <typename T>
void Array2D<T>::reserve(il::int_t r0, il::int_t r1) {
  IL_EXPECT_FAST(r0 >= 0);
  IL_EXPECT_FAST(r1 >= 0);
  r0 = r0 > capacity(0) ? r0 : capacity(0);
  r1 = r1 > capacity(1) ? r1 : capacity(1);
  if (r0 > capacity(0) || r1 > capacity(1)) {
    il::int_t n0_old{size(0)};
    il::int_t n1_old{size(1)};
    if (il::is_trivial<T>::value && align_mod_ != 0) {
      const il::int_t nb_lanes{static_cast<il::int_t>(alignment() / sizeof(T))};
      r0 = (r0 / nb_lanes + 1) * nb_lanes;
    }
    il::int_t r{r0 * r1};
    T* new_data;
    short new_shift;
    if (il::is_trivial<T>::value) {
      if (align_mod_ == 0) {
        new_data = new T[r];
        new_shift = 0;
      } else {
        new_data = allocate(r, align_mod_, align_r_, il::io, new_shift);
      }
      if (data_) {
        for (il::int_t i1 = 0; i1 < size(1); ++i1) {
          memcpy(new_data + i1 * r0, data_ + i1 * stride(1),
                 size(0) * sizeof(T));
        }
        delete[](data_ - align_r_);
      }
    } else {
      new_data = static_cast<T*>(::operator new(r * sizeof(T)));
      new_shift = 0;
      for (il::int_t i1{size(1) - 1}; i1 >= 0; --i1) {
        for (il::int_t i0{size(0) - 1}; i0 >= 0; --i0) {
          new (new_data + i1 * r0 + i0)
              T(std::move(data_[i1 * stride(1) + i0]));
          (data_ + i1 * stride(1) + i0)->~T();
        }
      }
      ::operator delete(data_);
    }
    data_ = new_data;
#ifdef IL_DEBUG_VISUALIZER
    debug_capacity_[0] = r0;
    debug_capacity_[1] = r1;
#endif
    size_[0] = data_ + n0_old;
    size_[1] = data_ + n1_old;
    capacity_[0] = data_ + r0;
    capacity_[1] = data_ + r1;
    new_shift_ = new_shift;
  }
}

template <typename T>
short Array2D<T>::alignment() const {
  unsigned short a{static_cast<unsigned short>(align_mod_)};
  unsigned short b{static_cast<unsigned short>(align_r_)};
  while (b != 0) {
    unsigned short c{a};
    a = b;
    b = c % b;
  }
  return static_cast<short>(a);
}

template <typename T>
const T* Array2D<T>::data() const {
  return data_;
}

template <typename T>
T* Array2D<T>::data() {
  return data_;
}

template <typename T>
const T* Array2D<T>::data(il::int_t i1) const {
  return data_ + i1 * stride(1);
}

template <typename T>
T* Array2D<T>::data(il::int_t i1) {
  return data_ + i1 * stride(1);
}

template <typename T>
il::int_t Array2D<T>::stride(il::int_t d) const {
  IL_EXPECT_FAST(static_cast<std::size_t>(d) < static_cast<std::size_t>(2));
  return (d == 0) ? 1 : static_cast<il::int_t>(capacity_[0] - data_);
}

template <typename T>
T* Array2D<T>::allocate(il::int_t n, short align_mod, short align_r, il::io_t,
                        short& new_shift) {
  std::size_t size_in_T{
      static_cast<std::size_t>(n + (align_mod + align_r) / sizeof(T))};
  char* data{reinterpret_cast<char*>(new T[size_in_T])};
  std::size_t data_position{(std::size_t)data};
  std::size_t shift{data_position % align_mod};
  char* new_data{data + (align_mod - shift) + align_r};
  new_shift = static_cast<short>((new_data - data) / sizeof(T));
  return reinterpret_cast<T*>(new_data);
}

template <typename T>
void Array2D<T>::check_invariance() const {
#ifdef IL_DEBUG_VISUALIZER
  IL_EXPECT_FAST(debug_size_[0] == size_[0] - data_);
  IL_EXPECT_FAST(debug_size_[1] == size_[1] - data_);
  IL_EXPECT_FAST(debug_capacity_[0] == capacity_[0] - data_);
  IL_EXPECT_FAST(debug_capacity_[1] == capacity_[1] - data_);
#endif
  if (data_ == nullptr) {
    IL_EXPECT_FAST(size_[0] == nullptr);
    IL_EXPECT_FAST(size_[1] == nullptr);
    IL_EXPECT_FAST(capacity_[0] == nullptr);
    IL_EXPECT_FAST(capacity_[1] == nullptr);
    IL_EXPECT_FAST(align_mod_ == 0);
    IL_EXPECT_FAST(align_r_ == 0);
    IL_EXPECT_FAST(new_shift_ == 0);
  } else {
    IL_EXPECT_FAST(size_[0] != nullptr);
    IL_EXPECT_FAST(size_[1] != nullptr);
    IL_EXPECT_FAST(capacity_[0] != nullptr);
    IL_EXPECT_FAST(capacity_[1] != nullptr);
    IL_EXPECT_FAST((size_[0] - data_) <= (capacity_[0] - data_));
    IL_EXPECT_FAST((size_[1] - data_) <= (capacity_[1] - data_));
    if (!il::is_trivial<T>::value) {
      IL_EXPECT_FAST(align_mod_ == 0);
    }
    if (align_mod_ == 0) {
      IL_EXPECT_FAST(align_r_ == 0);
      IL_EXPECT_FAST(new_shift_ == 0);
    } else {
      IL_EXPECT_FAST(align_r_ < align_mod_);
      IL_EXPECT_FAST(((std::size_t)data_) % ((std::size_t)align_mod_) ==
                ((std::size_t)align_r_));
    }
  }
}
}

#endif  // IL_ARRAY2D_H
