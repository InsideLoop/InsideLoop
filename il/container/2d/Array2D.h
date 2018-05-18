//==============================================================================
//
// Copyright 2018 The InsideLoop Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//==============================================================================

#ifndef IL_ARRAY2D_H
#define IL_ARRAY2D_H

// <cstring> is needed for memcpy
#include <cstring>
// <initializer_list> is needed for std::initializer_list<T>
#include <initializer_list>
// <new> is needed for placement new
#include <new>
// <utility> is needed for std::move
#include <utility>

#include <il/container/1d/ArrayView.h>
#include <il/container/2d/Array2DView.h>
#include <il/core/memory/allocate.h>

namespace il {

template <typename T>
class Array2D {
 private:
  T* data_;
  T* size_[2];
  T* capacity_[2];
  short alignment_;
  short align_r_;
  short align_mod_;
  short shift_;

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
  //   - (Debug mode) initialized to il::defaultValue<T>(). It is usually NaN
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
  explicit Array2D(il::int_t n0, il::int_t n1, il::align_t,
                   il::int_t alignment);

  /* \brief Construct an aligned array
  // \details The pointer data, when considered as an integer, satisfies
  // data_ = align_r (Modulo align_mod)
  */
  explicit Array2D(il::int_t n0, il::int_t n1, il::align_t, il::int_t alignment,
                   il::int_t align_r, il::int_t align_mod);

  /* \brief Construct an array of n rows and p columns with a value
  /
  // // Construct an array of double with 3 rows and 5 columns, initialized with
  // // 3.14.
  // il::Array2D<double> A{3, 5, 3.14};
  */
  explicit Array2D(il::int_t n0, il::int_t n1, const T& x);

  template <typename... Args>
  explicit Array2D(il::int_t n0, il::int_t n1, il::emplace_t, Args&&... args);

  explicit Array2D(il::int_t n0, il::int_t n1, const T& x, il::align_t,
                   il::int_t alignment);

  explicit Array2D(il::int_t n0, il::int_t n1, const T& x, il::align_t,
                   il::int_t alignment, il::int_t align_r, il::int_t align_mod);

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
  void Resize(il::int_t n0, il::int_t n1);

  void Resize(il::int_t n0, il::int_t n1, const T& x);

  template <typename... Args>
  void Resize(il::int_t n0, il::int_t n1, il::emplace_t, Args&&... args);

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
  void Reserve(il::int_t r0, il::int_t r1);

  /* \brief Get the alignment of the pointer returned by data()
   */
  il::int_t alignment() const;

  il::Array2DView<T> view() const;

  il::Array2DView<T> view(il::Range range0, il::Range range1) const;

  il::ArrayView<T> view(il::Range range0, il::int_t i1) const;

  il::Array2DEdit<T> Edit();

  il::Array2DEdit<T> Edit(il::Range range0, il::Range range1);

  il::ArrayEdit<T> Edit(il::Range range0, il::int_t i1);

  /* \brief Get a pointer to const to the first element of the array
  // \details One should use this method only when using C-style API
  */
  const T* data() const;

  /* \brief Get a pointer to the first element of the array
  // \details One should use this method only when using C-style API
  */
  T* Data();

  /* \brief Get a pointer to the first element of the column
  // \details One should use this method only when using C-style API
  */
  const T* data(il::int_t i1) const;

  /* \brief Get a pointer to the first element of the column
  // \details One should use this method only when using C-style API
  */
  T* Data(il::int_t i1);

  /* \brief The memory position of A(i0, i1) is
  // data() + i1 * stride(1) + i0
  */
  il::int_t stride(il::int_t d) const;

 private:
  /* \brief Used internally in debug mode to check the invariance of the object
   */
  bool invariance() const;
};

template <typename T>
Array2D<T>::Array2D() {
  data_ = nullptr;
  size_[0] = nullptr;
  size_[1] = nullptr;
  capacity_[0] = nullptr;
  capacity_[1] = nullptr;
  alignment_ = 0;
  align_r_ = 0;
  align_mod_ = 0;
  shift_ = 0;
}

template <typename T>
Array2D<T>::Array2D(il::int_t n0, il::int_t n1) {
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);

  const il::int_t r0 = n0 > 0 ? n0 : (n1 == 0 ? 0 : 1);
  const il::int_t r1 = n1 > 0 ? n1 : (n0 == 0 ? 0 : 1);
  bool error = false;
  const il::int_t r = il::safeProduct(r0, r1, il::io, error);
  if (error) {
    il::abort();
  }
  if (r > 0) {
    data_ = il::allocateArray<T>(r);
    if (il::isTrivial<T>::value) {
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = 0; i0 < n0; ++i0) {
          data_[i1 * r0 + i0] = il::defaultValue<T>();
        }
      }
#endif
    } else {
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = 0; i0 < n0; ++i0) {
          new (data_ + i1 * r0 + i0) T{};
        }
      }
    }
  } else {
    data_ = nullptr;
  }
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  alignment_ = 0;
  align_r_ = 0;
  align_mod_ = 0;
  shift_ = 0;
}

template <typename T>
Array2D<T>::Array2D(il::int_t n0, il::int_t n1, il::align_t,
                    il::int_t alignment, il::int_t align_r,
                    il::int_t align_mod) {
  IL_EXPECT_FAST(il::isTrivial<T>::value);
  IL_EXPECT_FAST(sizeof(T) % alignof(T) == 0);
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);
  IL_EXPECT_FAST(align_mod > 0);
  IL_EXPECT_FAST(align_mod % alignof(T) == 0);
  IL_EXPECT_FAST(align_mod <= SHRT_MAX);
  IL_EXPECT_FAST(align_r >= 0);
  IL_EXPECT_FAST(align_r < align_mod);
  IL_EXPECT_FAST(align_r % alignof(T) == 0);
  IL_EXPECT_FAST(align_r <= SHRT_MAX);
  IL_EXPECT_FAST(alignment > 0);
  IL_EXPECT_FAST(alignment % alignof(T) == 0);
  IL_EXPECT_FAST(alignment <= SHRT_MAX);
  IL_EXPECT_FAST(align_r % alignment == 0);
  IL_EXPECT_FAST(align_mod % alignment == 0);

  il::int_t r0;
  il::int_t r1;
  if (n0 > 0 && n1 > 0) {
    if (il::isTrivial<T>::value) {
      const il::int_t nb_lanes = static_cast<il::int_t>(
          static_cast<std::size_t>(alignment) / alignof(T));
      bool error = false;
      r0 = il::safeUpperRound(n0, nb_lanes, il::io, error);
      if (error) {
        il::abort();
      }
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
  bool error = false;
  const il::int_t r = il::safeProduct(r0, r1, il::io, error);
  if (error) {
    il::abort();
  }
  if (r > 0) {
    il::int_t shift;
    data_ = il::allocateArray<T>(r, align_r, align_mod, il::io, shift);
    shift_ = static_cast<short>(shift);
#ifdef IL_DEFAULT_VALUE
    for (il::int_t i1 = 0; i1 < n1; ++i1) {
      for (il::int_t i0 = 0; i0 < n0; ++i0) {
        data_[i1 * r0 + i0] = il::defaultValue<T>();
      }
    }
#endif
  } else {
    data_ = nullptr;
    shift_ = 0;
  }
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  alignment_ = static_cast<short>(alignment);
  align_r_ = static_cast<short>(align_r);
  align_mod_ = static_cast<short>(align_mod);
}

template <typename T>
Array2D<T>::Array2D(il::int_t n0, il::int_t n1, il::align_t,
                    il::int_t alignment)
    : Array2D{n0, n1, il::align, alignment, 0, alignment} {}

template <typename T>
Array2D<T>::Array2D(il::int_t n0, il::int_t n1, const T& x) {
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);

  const il::int_t r0 = n0 > 0 ? n0 : (n1 == 0 ? 0 : 1);
  const il::int_t r1 = n1 > 0 ? n1 : (n0 == 0 ? 0 : 1);
  bool error = false;
  const il::int_t r = il::safeProduct(r0, r1, il::io, error);
  if (error) {
    il::abort();
  }
  if (r > 0) {
    data_ = il::allocateArray<T>(r);
    for (il::int_t i1 = 0; i1 < n1; ++i1) {
      for (il::int_t i0 = 0; i0 < n0; ++i0) {
        new (data_ + i1 * r0 + i0) T(x);
      }
    }
  } else {
    data_ = nullptr;
  }
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  alignment_ = 0;
  align_r_ = 0;
  align_mod_ = 0;
  shift_ = 0;
}

template <typename T>
template <typename... Args>
Array2D<T>::Array2D(il::int_t n0, il::int_t n1, il::emplace_t, Args&&... args) {
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);

  const il::int_t r0 = n0 > 0 ? n0 : (n1 == 0 ? 0 : 1);
  const il::int_t r1 = n1 > 0 ? n1 : (n0 == 0 ? 0 : 1);
  bool error = false;
  const il::int_t r = il::safeProduct(r0, r1, il::io, error);
  if (error) {
    il::abort();
  }
  if (r > 0) {
    data_ = il::allocateArray<T>(r);
    for (il::int_t i1 = 0; i1 < n1; ++i1) {
      for (il::int_t i0 = 0; i0 < n0; ++i0) {
        new (data_ + i1 * r0 + i0) T(std::forward<Args>(args)...);
      }
    }
  } else {
    data_ = nullptr;
  }
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  alignment_ = 0;
  align_r_ = 0;
  align_mod_ = 0;
  shift_ = 0;
}

template <typename T>
Array2D<T>::Array2D(il::int_t n0, il::int_t n1, const T& x, il::align_t,
                    il::int_t alignment, il::int_t align_r,
                    il::int_t align_mod) {
  IL_EXPECT_FAST(il::isTrivial<T>::value);
  IL_EXPECT_FAST(sizeof(T) % alignof(T) == 0);
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);
  IL_EXPECT_FAST(align_mod > 0);
  IL_EXPECT_FAST(align_mod % alignof(T) == 0);
  IL_EXPECT_FAST(align_mod <= SHRT_MAX);
  IL_EXPECT_FAST(align_r >= 0);
  IL_EXPECT_FAST(align_r < align_mod);
  IL_EXPECT_FAST(align_r % alignof(T) == 0);
  IL_EXPECT_FAST(align_r <= SHRT_MAX);
  IL_EXPECT_FAST(alignment > 0);
  IL_EXPECT_FAST(alignment % alignof(T) == 0);
  IL_EXPECT_FAST(alignment <= SHRT_MAX);
  IL_EXPECT_FAST(align_r % alignment == 0);
  IL_EXPECT_FAST(align_mod % alignment == 0);

  il::int_t r0;
  il::int_t r1;
  if (n0 > 0 && n1 > 0) {
    if (il::isTrivial<T>::value) {
      const il::int_t nb_lanes = static_cast<il::int_t>(
          static_cast<std::size_t>(alignment) / alignof(T));
      bool error = false;
      r0 = il::safeUpperRound(n0, nb_lanes, il::io, error);
      if (error) {
        il::abort();
      }
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
  bool error = false;
  const il::int_t r = il::safeProduct(r0, r1, il::io, error);
  if (error) {
    il::abort();
  }
  if (r > 0) {
    il::int_t shift;
    data_ = il::allocateArray<T>(r, align_r, align_mod, il::io, shift);
    shift_ = static_cast<short>(shift);
    for (il::int_t i1 = 0; i1 < n1; ++i1) {
      for (il::int_t i0 = 0; i0 < n0; ++i0) {
        data_[i1 * r0 + i0] = x;
      }
    }
  } else {
    data_ = nullptr;
    shift_ = 0;
  }
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  alignment_ = static_cast<short>(alignment);
  align_r_ = static_cast<short>(align_r);
  align_mod_ = static_cast<short>(align_mod);
}

template <typename T>
Array2D<T>::Array2D(il::int_t n0, il::int_t n1, const T& x, il::align_t,
                    il::int_t alignment)
    : Array2D{n0, n1, x, il::align, alignment, 0, alignment} {}

template <typename T>
Array2D<T>::Array2D(il::value_t,
                    std::initializer_list<std::initializer_list<T>> list) {
  bool error = false;
  const il::int_t n1 = il::safeConvert<il::int_t>(list.size(), il::io, error);
  if (error) {
    il::abort();
  }
  error = false;
  const il::int_t n0 =
      n1 > 0 ? il::safeConvert<il::int_t>(list.begin()->size(), il::io, error)
             : 0;
  if (error) {
    il::abort();
  }

  il::int_t r0;
  il::int_t r1;
  if (n0 > 0 && n1 > 0) {
    r0 = n0;
    r1 = n1;
    bool error = false;
    const il::int_t r = il::safeProduct(r0, r1, il::io, error);
    if (error) {
      il::abort();
    }
    data_ = il::allocateArray<T>(r);
    if (il::isTrivial<T>::value) {
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        IL_EXPECT_FAST(static_cast<il::int_t>((list.begin() + i1)->size()) ==
                       n0);
        memcpy(data_ + i1 * r0, (list.begin() + i1)->begin(), n0 * sizeof(T));
      }
    } else {
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        IL_EXPECT_FAST(static_cast<il::int_t>((list.begin() + i1)->size()) ==
                       n0);
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
    bool error = false;
    const il::int_t r = il::safeProduct(r0, r1, il::io, error);
    if (error) {
      il::abort();
    }
    data_ = il::allocateArray<T>(r);
  }
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  alignment_ = 0;
  align_r_ = 0;
  align_mod_ = 0;
  shift_ = 0;
}

template <typename T>
Array2D<T>::Array2D(const Array2D<T>& A) {
  const il::int_t n0 = A.size(0);
  const il::int_t n1 = A.size(1);
  il::int_t r0;
  il::int_t r1;
  if (n0 > 0 && n1 > 0) {
    if (il::isTrivial<T>::value && A.alignment_ != 0) {
      const il::int_t nb_lanes = static_cast<il::int_t>(
          static_cast<std::size_t>(A.alignment_) / alignof(T));
      bool error = false;
      r0 = il::safeUpperRound(n0, nb_lanes, il::io, error);
      if (error) {
        il::abort();
      }
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
  bool error = false;
  const il::int_t r = il::safeProduct(r0, r1, il::io, error);
  if (error) {
    il::abort();
  }
  if (il::isTrivial<T>::value) {
    if (A.alignment_ == 0) {
      data_ = il::allocateArray<T>(r);
      shift_ = 0;
    } else {
      il::int_t shift;
      data_ = il::allocateArray<T>(r, A.align_r_, A.align_mod_, il::io, shift);
      shift_ = static_cast<short>(shift);
    }
    for (il::int_t i1 = 0; i1 < n1; ++i1) {
      memcpy(data_ + i1 * r0, A.data() + i1 * A.capacity(0), n0 * sizeof(T));
    }
  } else {
    data_ = il::allocateArray<T>(r);
    shift_ = 0;
    for (il::int_t i1 = 0; i1 < n1; ++i1) {
      for (il::int_t i0 = 0; i0 < n0; ++i0) {
        new (data_ + i1 * r0 + i0) T(A(i0, i1));
      }
    }
  }
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
  capacity_[0] = data_ + r0;
  capacity_[1] = data_ + r1;
  alignment_ = A.alignment_;
  align_r_ = A.align_r_;
  align_mod_ = A.align_mod_;
}

template <typename T>
Array2D<T>::Array2D(Array2D<T>&& A) {
  data_ = A.data_;
  size_[0] = A.size_[0];
  size_[1] = A.size_[1];
  capacity_[0] = A.capacity_[0];
  capacity_[1] = A.capacity_[1];
  alignment_ = A.alignment_;
  align_r_ = A.align_r_;
  align_mod_ = A.align_mod_;
  shift_ = A.shift_;
  A.data_ = nullptr;
  A.size_[0] = nullptr;
  A.size_[1] = nullptr;
  A.capacity_[0] = nullptr;
  A.capacity_[1] = nullptr;
  A.alignment_ = 0;
  A.align_r_ = 0;
  A.align_mod_ = 0;
  A.shift_ = 0;
}

template <typename T>
Array2D<T>& Array2D<T>::operator=(const Array2D<T>& A) {
  if (this != &A) {
    const il::int_t n0 = A.size(0);
    const il::int_t n1 = A.size(1);
    const il::int_t alignment = A.alignment_;
    const il::int_t align_r = A.align_r_;
    const il::int_t align_mod = A.align_mod_;
    const bool need_memory = capacity(0) < n0 || capacity(1) < n1 ||
                             align_mod_ != align_mod || align_r_ != align_r ||
                             alignment_ != alignment;
    if (need_memory) {
      il::int_t r0;
      il::int_t r1;
      if (n0 > 0 && n1 > 0) {
        if (il::isTrivial<T>::value && alignment != 0) {
          const il::int_t nb_lanes = static_cast<il::int_t>(
              static_cast<std::size_t>(alignment) / alignof(T));
          bool error = false;
          r0 = il::safeUpperRound(n0, nb_lanes, il::io, error);
          if (error) {
            il::abort();
          }
          r1 = n1;
        } else {
          r0 = n0;
          r1 = n1;
        }
      } else {
        r0 = (n0 == 0) ? 1 : n0;
        r1 = (n1 == 0) ? 1 : n1;
      }
      bool error = false;
      const il::int_t r = il::safeProduct(r0, r1, il::io, error);
      if (error) {
        il::abort();
      }
      if (il::isTrivial<T>::value) {
        if (data_) {
          il::deallocate(data_ - shift_);
        }
        if (alignment == 0) {
          data_ = il::allocateArray<T>(r);
          shift_ = 0;
        } else {
          il::int_t shift;
          data_ = il::allocateArray<T>(r, align_r, align_mod, il::io, shift);
          shift_ = static_cast<short>(shift);
        }
        for (il::int_t i1 = 0; i1 < n1; ++i1) {
          memcpy(data_ + i1 * r0, A.data_ + i1 * A.capacity(0), n0 * sizeof(T));
        }
      } else {
        if (data_) {
          for (il::int_t i1 = size(1) - 1; i1 >= 0; --i1) {
            for (il::int_t i0 = size(0) - 1; i0 >= 0; --i0) {
              (data_ + i1 * stride(1) + i0)->~T();
            }
          }
          il::deallocate(data_);
        }
        data_ = il::allocateArray<T>(r);
        shift_ = 0;
        for (il::int_t i1 = 0; i1 < n1; ++i1) {
          for (il::int_t i0 = 0; i0 < n0; ++i0) {
            new (data_ + i1 * r0 + i0) T(A(i0, i1));
          }
        }
      }
      size_[0] = data_ + n0;
      size_[1] = data_ + n1;
      capacity_[0] = data_ + r0;
      capacity_[1] = data_ + r1;
      alignment_ = static_cast<short>(alignment);
      align_r_ = static_cast<short>(align_r);
      align_mod_ = static_cast<short>(align_mod);
    } else {
      if (il::isTrivial<T>::value) {
        for (il::int_t i1 = 0; i1 < n1; ++i1) {
          memcpy(data_ + i1 * capacity(0), A.data_ + i1 * A.capacity(0),
                 n0 * sizeof(T));
        }
      } else {
        for (il::int_t i1 = 0; i1 < n1; ++i1) {
          for (il::int_t i0 = 0; i0 < n0; ++i0) {
            data_[i1 * capacity(0) + i0] = A(i0, i1);
          }
        }
        for (il::int_t i1 = size(1) - 1; i1 >= 0; --i1) {
          for (il::int_t i0 = size(0) - 1; i0 >= (i1 < n1 ? n0 : 0); --i0) {
            (data_ + i1 * capacity(0) + i0)->~T();
          }
        }
      }
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
      if (!il::isTrivial<T>::value) {
        for (il::int_t i1 = size(1) - 1; i1 >= 0; --i1) {
          for (il::int_t i0 = size(0) - 1; i0 >= 0; --i0) {
            (data_ + i1 * capacity(0) + i0)->~T();
          }
        }
      }
      il::deallocate(data_ - shift_);
    }
    data_ = A.data_;
    size_[0] = A.size_[0];
    size_[1] = A.size_[1];
    capacity_[0] = A.capacity_[0];
    capacity_[1] = A.capacity_[1];
    alignment_ = A.alignment_;
    align_r_ = A.align_r_;
    align_mod_ = A.align_mod_;
    shift_ = A.shift_;
    A.data_ = nullptr;
    A.size_[0] = nullptr;
    A.size_[1] = nullptr;
    A.capacity_[0] = nullptr;
    A.capacity_[1] = nullptr;
    A.alignment_ = 0;
    A.align_r_ = 0;
    A.align_mod_ = 0;
    A.shift_ = 0;
  }
  return *this;
}

template <typename T>
Array2D<T>::~Array2D() {
  IL_EXPECT_FAST_NOTHROW(invariance());

  if (data_) {
    if (!il::isTrivial<T>::value) {
      for (il::int_t i1 = size(1) - 1; i1 >= 0; --i1) {
        for (il::int_t i0 = size(0) - 1; i0 >= 0; --i0) {
          (data_ + i1 * capacity(0) + i0)->~T();
        }
      }
    }
    il::deallocate(data_ - shift_);
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

  return size_[d] - data_;
}

template <typename T>
void Array2D<T>::Resize(il::int_t n0, il::int_t n1) {
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);

  const il::int_t n0_old = size(0);
  const il::int_t n1_old = size(1);
  const bool need_memory = n0 > capacity(0) || n1 > capacity(1);
  if (need_memory) {
    il::int_t r0;
    il::int_t r1;
    if (n0 > 0 && n1 > 0) {
      if (il::isTrivial<T>::value && alignment_ != 0) {
        const il::int_t nb_lanes = static_cast<il::int_t>(
            static_cast<std::size_t>(alignment_) / alignof(T));
        bool error = false;
        r0 = il::safeUpperRound(n0, nb_lanes, il::io, error);
        if (error) {
          il::abort();
        }
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
    bool error = false;
    const il::int_t r = il::safeProduct(r0, r1, il::io, error);
    if (error) {
      il::abort();
    }
    T* new_data;
    il::int_t new_shift;
    if (il::isTrivial<T>::value) {
      if (alignment_ == 0) {
        new_data = il::allocateArray<T>(r);
        new_shift = 0;
      } else {
        new_data =
            il::allocateArray<T>(r, align_r_, align_mod_, il::io, new_shift);
      }
      if (data_) {
        for (il::int_t i1 = 0; i1 < (n1 < n1_old ? n1 : n1_old); ++i1) {
          memcpy(new_data + i1 * r0, data_ + i1 * capacity(0),
                 (n0 < n0_old ? n0 : n0_old) * sizeof(T));
        }
        il::deallocate(data_ - shift_);
      }
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = i1 < n1_old ? n0_old : 0; i0 < n0; ++i0) {
          new_data[i1 * r0 + i0] = il::defaultValue<T>();
        }
      }
#endif
    } else {
      new_data = il::allocateArray<T>(r);
      new_shift = 0;
      if (data_) {
        for (il::int_t i1 = n1_old - 1; i1 >= 0; --i1) {
          for (il::int_t i0 = n0_old - 1; i0 >= (i1 < n1 ? n0 : 0); --i0) {
            (data_ + i1 * capacity(0) + i0)->~T();
          }
        }
        for (il::int_t i1 = (n1 < n1_old ? n1 : n1_old) - 1; i1 >= 0; --i1) {
          for (il::int_t i0 = (n0 < n0_old ? n0 : n0_old) - 1; i0 >= 0; --i0) {
            new (new_data + i1 * r0 + i0)
                T(std::move(data_[i1 * capacity(0) + i0]));
            (data_ + i1 * capacity(0) + i0)->~T();
          }
        }
        il::deallocate(data_);
      }
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = i1 < n1_old ? n0_old : 0; i0 < n0; ++i0) {
          new (new_data + i1 * r0 + i0) T{};
        }
      }
    }
    data_ = new_data;
    capacity_[0] = data_ + r0;
    capacity_[1] = data_ + r1;
    shift_ = static_cast<short>(new_shift);
  } else {
    if (il::isTrivial<T>::value) {
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = (i1 < n1_old ? n0_old : 0); i0 < n0; ++i0) {
          data_[i1 * capacity(0) + i0] = il::defaultValue<T>();
        }
      }
#endif
    } else {
      for (il::int_t i1 = n1_old - 1; i1 >= 0; --i1) {
        for (il::int_t i0 = n0_old - 1; i0 >= (i1 < n1 ? n0 : 0); --i0) {
          (data_ + i1 * capacity(0) + i0)->~T();
        }
      }
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = i1 < n1_old ? n0_old : 0; i0 < n0; ++i0) {
          new (data_ + i1 * capacity(0) + i0) T{};
        }
      }
    }
  }
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
}

template <typename T>
void Array2D<T>::Resize(il::int_t n0, il::int_t n1, const T& x) {
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);

  const il::int_t n0_old = size(0);
  const il::int_t n1_old = size(1);
  const bool need_memory = n0 > capacity(0) || n1 > capacity(1);
  if (need_memory) {
    il::int_t r0;
    il::int_t r1;
    if (n0 > 0 && n1 > 0) {
      if (il::isTrivial<T>::value && alignment_ != 0) {
        const il::int_t nb_lanes = static_cast<il::int_t>(
            static_cast<std::size_t>(alignment_) / alignof(T));
        bool error = false;
        r0 = il::safeUpperRound(n0, nb_lanes, il::io, error);
        if (error) {
          il::abort();
        }
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
    bool error = false;
    const il::int_t r = il::safeProduct(r0, r1, il::io, error);
    if (error) {
      il::abort();
    }
    T* new_data;
    il::int_t new_shift;
    if (il::isTrivial<T>::value) {
      if (alignment_ == 0) {
        new_data = il::allocateArray<T>(r);
        new_shift = 0;
      } else {
        new_data =
            il::allocateArray<T>(r, align_r_, align_mod_, il::io, new_shift);
      }
      if (data_) {
        for (il::int_t i1 = 0; i1 < (n1 < n1_old ? n1 : n1_old); ++i1) {
          memcpy(new_data + i1 * r0, data_ + i1 * capacity(0),
                 (n0 < n0_old ? n0 : n0_old) * sizeof(T));
        }
        il::deallocate(data_ - shift_);
      }
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = i1 < n1_old ? n0_old : 0; i0 < n0; ++i0) {
          new_data[i1 * r0 + i0] = x;
        }
      }
    } else {
      new_data = il::allocateArray<T>(r);
      new_shift = 0;
      if (data_) {
        for (il::int_t i1 = n1_old - 1; i1 >= 0; --i1) {
          for (il::int_t i0 = n0_old - 1; i0 >= (i1 < n1 ? n0 : 0); --i0) {
            (data_ + i1 * capacity(0) + i0)->~T();
          }
        }
        for (il::int_t i1 = (n1 < n1_old ? n1 : n1_old) - 1; i1 >= 0; --i1) {
          for (il::int_t i0 = (n0 < n0_old ? n0 : n0_old) - 1; i0 >= 0; --i0) {
            new (new_data + i1 * r0 + i0)
                T(std::move(data_[i1 * capacity(0) + i0]));
            (data_ + i1 * capacity(0) + i0)->~T();
          }
        }
        il::deallocate(data_);
      }
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = i1 < n1_old ? n0_old : 0; i0 < n0; ++i0) {
          new (new_data + i1 * r0 + i0) T{x};
        }
      }
    }
    data_ = new_data;
    capacity_[0] = data_ + r0;
    capacity_[1] = data_ + r1;
    shift_ = static_cast<short>(new_shift);
  } else {
    if (il::isTrivial<T>::value) {
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = (i1 < n1_old ? n0_old : 0); i0 < n0; ++i0) {
          data_[i1 * capacity(0) + i0] = x;
        }
      }
    } else {
      for (il::int_t i1 = n1_old - 1; i1 >= 0; --i1) {
        for (il::int_t i0 = n0_old - 1; i0 >= (i1 < n1 ? n0 : 0); --i0) {
          (data_ + i1 * capacity(0) + i0)->~T();
        }
      }
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = i1 < n1_old ? n0_old : 0; i0 < n0; ++i0) {
          new (data_ + i1 * capacity(0) + i0) T{x};
        }
      }
    }
  }
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
}

template <typename T>
template <typename... Args>
void Array2D<T>::Resize(il::int_t n0, il::int_t n1, il::emplace_t,
                        Args&&... args) {
  IL_EXPECT_FAST(n0 >= 0);
  IL_EXPECT_FAST(n1 >= 0);

  const il::int_t n0_old = size(0);
  const il::int_t n1_old = size(1);
  const bool need_memory = n0 > capacity(0) || n1 > capacity(1);
  if (need_memory) {
    il::int_t r0;
    il::int_t r1;
    if (n0 > 0 && n1 > 0) {
      if (il::isTrivial<T>::value && alignment_ != 0) {
        const il::int_t nb_lanes = static_cast<il::int_t>(
            static_cast<std::size_t>(alignment_) / alignof(T));
        bool error = false;
        r0 = il::safeUpperRound(n0, nb_lanes, il::io, error);
        if (error) {
          il::abort();
        }
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
    bool error = false;
    const il::int_t r = il::safeProduct(r0, r1, il::io, error);
    if (error) {
      il::abort();
    }
    T* new_data;
    il::int_t new_shift;
    if (il::isTrivial<T>::value) {
      if (alignment_ == 0) {
        new_data = il::allocateArray<T>(r);
        new_shift = 0;
      } else {
        new_data =
            il::allocateArray<T>(r, align_r_, align_mod_, il::io, new_shift);
      }
      if (data_) {
        for (il::int_t i1 = 0; i1 < (n1 < n1_old ? n1 : n1_old); ++i1) {
          memcpy(new_data + i1 * r0, data_ + i1 * capacity(0),
                 (n0 < n0_old ? n0 : n0_old) * sizeof(T));
        }
        il::deallocate(data_ - shift_);
      }
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = i1 < n1_old ? n0_old : 0; i0 < n0; ++i0) {
          new_data[i1 * r0 + i0] = T(std::forward<Args>(args)...);
        }
      }
    } else {
      new_data = il::allocateArray<T>(r);
      new_shift = 0;
      if (data_) {
        for (il::int_t i1 = n1_old - 1; i1 >= 0; --i1) {
          for (il::int_t i0 = n0_old - 1; i0 >= (i1 < n1 ? n0 : 0); --i0) {
            (data_ + i1 * capacity(0) + i0)->~T();
          }
        }
        for (il::int_t i1 = (n1 < n1_old ? n1 : n1_old) - 1; i1 >= 0; --i1) {
          for (il::int_t i0 = (n0 < n0_old ? n0 : n0_old) - 1; i0 >= 0; --i0) {
            new (new_data + i1 * r0 + i0)
                T(std::move(data_[i1 * capacity(0) + i0]));
            (data_ + i1 * capacity(0) + i0)->~T();
          }
        }
        il::deallocate(data_);
      }
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = i1 < n1_old ? n0_old : 0; i0 < n0; ++i0) {
          new (new_data + i1 * r0 + i0) T{std::forward<Args>(args)...};
        }
      }
    }
    data_ = new_data;
    capacity_[0] = data_ + r0;
    capacity_[1] = data_ + r1;
    shift_ = static_cast<short>(new_shift);
  } else {
    if (il::isTrivial<T>::value) {
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = (i1 < n1_old ? n0_old : 0); i0 < n0; ++i0) {
          data_[i1 * capacity(0) + i0] = T(std::forward<Args>(args)...);
        }
      }
    } else {
      for (il::int_t i1 = n1_old - 1; i1 >= 0; --i1) {
        for (il::int_t i0 = n0_old - 1; i0 >= (i1 < n1 ? n0 : 0); --i0) {
          (data_ + i1 * capacity(0) + i0)->~T();
        }
      }
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = i1 < n1_old ? n0_old : 0; i0 < n0; ++i0) {
          new (data_ + i1 * capacity(0) + i0) T{std::forward<Args>(args)...};
        }
      }
    }
  }
  size_[0] = data_ + n0;
  size_[1] = data_ + n1;
}

template <typename T>
il::int_t Array2D<T>::capacity(il::int_t d) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(d) < static_cast<std::size_t>(2));

  return capacity_[d] - data_;
}

template <typename T>
void Array2D<T>::Reserve(il::int_t r0, il::int_t r1) {
  IL_EXPECT_FAST(r0 >= 0);
  IL_EXPECT_FAST(r1 >= 0);

  r0 = (r0 > capacity(0)) ? r0 : capacity(0);
  r1 = (r1 > capacity(1)) ? r1 : capacity(1);
  const bool need_memory = r0 > capacity(0) || r1 > capacity(1);
  if (need_memory) {
    const il::int_t n0_old = size(0);
    const il::int_t n1_old = size(1);
    if (il::isTrivial<T>::value && alignment_ != 0) {
      const il::int_t nb_lanes = static_cast<il::int_t>(
          static_cast<std::size_t>(alignment_) / alignof(T));
      bool error = false;
      r0 = il::safeUpperRound(r0, nb_lanes, il::io, error);
      if (error) {
        il::abort();
      }
    }
    bool error = false;
    const il::int_t r = il::safeProduct(r0, r1, il::io, error);
    if (error) {
      il::abort();
    }
    T* new_data;
    il::int_t new_shift;
    if (il::isTrivial<T>::value) {
      if (alignment_ == 0) {
        new_data = il::allocateArray<T>(r);
        new_shift = 0;
      } else {
        new_data =
            il::allocateArray<T>(r, align_r_, align_mod_, il::io, new_shift);
      }
      if (data_) {
        for (il::int_t i1 = 0; i1 < size(1); ++i1) {
          memcpy(new_data + i1 * r0, data_ + i1 * capacity(0),
                 size(0) * sizeof(T));
        }
        il::deallocate(data_ - shift_);
      }
    } else {
      new_data = il::allocateArray<T>(r);
      new_shift = 0;
      for (il::int_t i1 = size(1) - 1; i1 >= 0; --i1) {
        for (il::int_t i0 = size(0) - 1; i0 >= 0; --i0) {
          new (new_data + i1 * r0 + i0)
              T(std::move(data_[i1 * capacity(0) + i0]));
          (data_ + i1 * capacity(0) + i0)->~T();
        }
      }
      il::deallocate(data_);
    }
    data_ = new_data;
    size_[0] = data_ + n0_old;
    size_[1] = data_ + n1_old;
    capacity_[0] = data_ + r0;
    capacity_[1] = data_ + r1;
    shift_ = static_cast<short>(new_shift);
  }
}

template <typename T>
il::int_t Array2D<T>::alignment() const {
  return alignment_;
}

template <typename T>
il::Array2DView<T> Array2D<T>::view() const {
  return il::Array2DView<T>{data(), size(0), size(1), stride(1), 0, 0};
}

template <typename T>
il::ArrayView<T> Array2D<T>::view(il::Range range0, il::int_t i1) const {
  return il::ArrayView<T>{data() + i1 * size(0) + range0.begin,
                          range0.end - range0.begin};
}

template <typename T>
il::Array2DView<T> Array2D<T>::view(il::Range range0, il::Range range1) const {
  IL_EXPECT_FAST(static_cast<std::size_t>(range0.begin) <
                 static_cast<std::size_t>(size(0)));
  IL_EXPECT_FAST(static_cast<std::size_t>(range0.end) <=
                 static_cast<std::size_t>(size(0)));
  IL_EXPECT_FAST(static_cast<std::size_t>(range1.begin) <
                 static_cast<std::size_t>(size(1)));
  IL_EXPECT_FAST(static_cast<std::size_t>(range1.end) <=
                 static_cast<std::size_t>(size(1)));

  const il::int_t the_stride = stride(1);
  return il::Array2DView<T>{data_ + range1.begin * the_stride + range0.begin,
                            range0.end - range0.begin,
                            range1.end - range1.begin,
                            the_stride,
                            0,
                            0};
}

template <typename T>
il::Array2DEdit<T> Array2D<T>::Edit() {
  return il::Array2DEdit<T>{Data(), size(0), size(1), stride(1), 0, 0};
}

template <typename T>
il::Array2DEdit<T> Array2D<T>::Edit(il::Range range0, il::Range range1) {
  IL_EXPECT_FAST(static_cast<std::size_t>(range0.begin) <
                 static_cast<std::size_t>(size(0)));
  IL_EXPECT_FAST(static_cast<std::size_t>(range0.end) <=
                 static_cast<std::size_t>(size(0)));
  IL_EXPECT_FAST(static_cast<std::size_t>(range1.begin) <
                 static_cast<std::size_t>(size(1)));
  IL_EXPECT_FAST(static_cast<std::size_t>(range1.end) <=
                 static_cast<std::size_t>(size(1)));

  const il::int_t the_stride = stride(1);
  return il::Array2DEdit<T>{data_ + range1.begin * the_stride + range0.begin,
                            range0.end - range0.begin,
                            range1.end - range1.begin,
                            the_stride,
                            0,
                            0};
}

template <typename T>
il::ArrayEdit<T> Array2D<T>::Edit(il::Range range0, il::int_t i1) {
  return il::ArrayEdit<T>{data() + i1 * stride(1) + range0.begin,
                          range0.end - range0.begin};
}

template <typename T>
const T* Array2D<T>::data() const {
  return data_;
}

template <typename T>
T* Array2D<T>::Data() {
  return data_;
}

template <typename T>
const T* Array2D<T>::data(il::int_t i1) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) <
                   static_cast<std::size_t>(size(1)));

  return data_ + i1 * capacity(0);
}

template <typename T>
T* Array2D<T>::Data(il::int_t i1) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) <
                   static_cast<std::size_t>(size(1)));

  return data_ + i1 * capacity(0);
}

template <typename T>
il::int_t Array2D<T>::stride(il::int_t d) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(d) < static_cast<std::size_t>(2));

  return (d == 0) ? 1 : static_cast<il::int_t>(capacity_[0] - data_);
}

template <typename T>
bool Array2D<T>::invariance() const {
  bool ans = true;

  if (data_ == nullptr) {
    ans = ans && (size_[0] == nullptr);
    ans = ans && (size_[1] == nullptr);
    ans = ans && (capacity_[0] == nullptr);
    ans = ans && (capacity_[1] == nullptr);
    ans = ans && (alignment_ == 0);
    ans = ans && (align_r_ == 0);
    ans = ans && (align_mod_ == 0);
    ans = ans && (shift_ == 0);
  } else {
    ans = ans && (size_[0] != nullptr);
    ans = ans && (size_[1] != nullptr);
    ans = ans && (capacity_[0] != nullptr);
    ans = ans && (capacity_[1] != nullptr);
    ans = ans && ((size_[0] - data_) <= (capacity_[0] - data_));
    ans = ans && ((size_[1] - data_) <= (capacity_[1] - data_));
    if (il::isTrivial<T>::value) {
      ans = ans && (alignment_ >= 0);
      ans = ans && (align_r_ >= 0);
      ans = ans && (align_mod_ >= 0);
      ans = ans && (align_r_ % alignof(T) == 0);
      ans = ans && (align_mod_ % alignof(T) == 0);
      ans = ans && (alignment_ % alignof(T) == 0);
      if (alignment_ > 0) {
        ans = ans && (align_r_ % alignment_ == 0);
        ans = ans && (align_mod_ > 0);
        ans = ans && (align_mod_ % alignment_ == 0);
        ans = ans && (align_r_ < align_mod_);
        ans = ans && (reinterpret_cast<std::size_t>(data_) %
                          static_cast<std::size_t>(align_mod_) ==
                      static_cast<std::size_t>(align_r_));
      } else {
        ans = ans && (align_r_ == 0);
        ans = ans && (align_mod_ == 0);
      }
    } else {
      ans = ans && (align_r_ == 0);
      ans = ans && (align_mod_ == 0);
      ans = ans && (alignment_ == 0);
    }
  }
  return ans;
}

}  // namespace il

#endif  // IL_ARRAY2D_H
