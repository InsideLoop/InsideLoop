//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_ARRAY_H
#define IL_ARRAY_H

// <cstring> is needed for memcpy
#include <cstring>
// <initializer_list> is needed for std::initializer_list<T>
#include <initializer_list>
// <utility> is needed for std::move
#include <utility>

#include <il/base.h>
#include <il/core/memory/allocate.h>

namespace il {

template <typename T>
class Array {
 private:
  T* data_;
  T* size_;
  T* capacity_;
  short align_r_;
  short align_mod_;
  short shift_;

 public:
  /* \brief Default constructor
  // \details The size and the capacity of the array are set to 0. The alignment
  // is undefined. No memory allocation is done during the process.
  */
  Array();

  /* \brief Construct an array of n elements
  // \details The size and the capacity of the array are set to n.
  // - If T is a numeric value, the memory is
  //   - (Debug mode) initialized to il::default_value<T>(). It is usually
  NaN
  //     if T is a floating point number or 666..666 if T is an integer.
  //   - (Release mode) left uninitialized. This behavior is different from
  //     std::vector from the standard library which initializes all numeric
  //     values to 0.
  // - If T is an object with a default constructor, all objects are default
  //   constructed. A compile-time error is raised if T has no default
  //   constructor.
  //
  // // Construct an array of double of size 5
  // il::Array<double> v{5};
  */
  explicit Array(il::int_t n);

  /* \brief Construct an aligned array of n elements
  // \details The pointer data, when considered as an integer, satisfies
  // data_ = 0 (Modulo align_mod)
  */
  explicit Array(il::int_t n, il::align_t, il::int_t align_mod);

  /* \brief Construct an aligned array of n elements
  // \details The pointer data, when considered as an integer, satisfies
  // data_ = align_r (Modulo align_mod)
  */
  explicit Array(il::int_t n, il::align_t, il::int_t align_r,
                 il::int_t align_mod);

  /* \brief Construct an array of n elements with a value
  //
  // // Construct an array of double of length 5, initialized with 3.14
  // il::Array<double> v{5, 3.14};
  */
  explicit Array(il::int_t n, const T& x);

  /* \brief Construct an array of n elements with a value
  //
  // // Construct an array of double of length 5, initialized with 3.14
  // il::Array<double> v{5, 3.14};
  */
  explicit Array(il::int_t n, const T& x, il::align_t, il::int_t align_r,
                 il::int_t align_mod);

  /* \brief Construct an array from a brace-initialized list
  // \details The size and the capacity of the il::Array<T> is adjusted
  to
  // the size of the initializer list. The tag il::value is used to allow brace
  // initialization of il::Array<T> everywhere.
  //
  // // Construct an array of double from a list
  // il::Array<double> v{il::value, {2.0, 3.14, 5.0, 7.0}};
  */
  explicit Array(il::value_t, std::initializer_list<T> list);

  /* \brief The copy constructor
  // \details The size and the capacity of the constructed il::Array<T>
  are
  // equal to the size of the source array.
  */
  Array(const Array<T>& v);

  /* \brief The move constructor
  */
  Array(Array<T>&& v);

  /* \brief The copy assignment
  // \details The size is the same as the one for the source array. The
  // capacity is not changed if it is enough for the copy to happen and is
  // set to the size of the source array if the initial capacity is too low.
  */
  Array& operator=(const Array<T>& v);

  /* \brief The move assignment
  */
  Array& operator=(Array<T>&& v);

  /* \brief The destructor
  */
  ~Array();

  /* \brief Accessor for a const il::Array<T>
  // \details Access (read only) the i-th element of the array. Bound checking
  // is done in debug mode but not in release mode.
  //
  // il::Array<double> v{4};
  // std::cout << v[0] << std::endl;
  */
  const T& operator[](il::int_t i) const;

  /* \brief Accessor for an il::Array<T>
  // \details Access (read or write) the i-th element of the array. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::Array<double> v{4};
  // v[0] = 0.0;
  // v[4] = 0.0; // Program is aborted in debug mode and has undefined
  //             // behavior in release mode
  */
  T& operator[](il::int_t i);

  /* \brief Accessor to the last element of a const il::Array<T>
  // \details In debug mode, calling this method on an array of size 0 aborts
  // the program. In release mode, it will lead to undefined behavior.
  */
  const T& back() const;

  /* \brief Accessor to the last element of a il::Array<T>
  // \details In debug mode, calling this method on an array of size 0 aborts
  // the program. In release mode, it will lead to undefined behavior.
  */
  T& back();

  /* \brief Get the size of the il::Array<T>
  // \details The library has been designed in a way that any compiler can prove
  // that modifying v[k] can't change the result of v.size(). As a consequence
  // a call to v.size() is made just once at the very beginning of the loop
  // in the following example. It allows many optimizations from the compiler,
  // including automatic vectorization.
  //
  // il::Array<double> v{n};
  // for (il::int_t k = 0; k < v.size(); ++k) {
  //     v[k] = 1.0 / (k + 1);
  // }
  */
  il::int_t size() const;

  /* \brief Resizing an il::Array<T>
  // \details No reallocation is performed if the new size is <= to the
  // capacity. In this case, the capacity is unchanged. When the size is > than
  // the current capacity, reallocation is done and the array gets the same
  // capacity as its size.
  */
  void resize(il::int_t n);

  /* \brief Get the capacity of the il::Array<T>
  */
  il::int_t capacity() const;

  /* \brief Change the capacity of the array to at least p
  // \details If the capacity is >= to p, nothing is done. Otherwise,
  // reallocation is done and the new capacity is set to p.
  */
  void reserve(il::int_t r);

  /* \brief Add an element at the end of the array
  // \details Reallocation is done only if it is needed. In case reallocation
  // happens, then new capacity is roughly (3/2) the previous capacity.
  */
  void append(const T& x);

  /* \brief Add an element at the end of the array
  // \details Reallocation is done only if it is needed. In case reallocation
  // happens, then new capacity is roughly (3/2) the previous capacity.
  */
  void append(T&& x);

  /* \brief Construct an element at the end of the array
  // \details Reallocation is done only if it is needed. In case reallocation
  // happens, then new capacity is roughly (3/2) the previous capacity.
  */
  template <typename... Args>
  void append(il::emplace_t, Args&&... args);

  /* \brief Get the alignment of the pointer returned by data()
  */
  il::int_t alignment() const;

  /* \brief Get a pointer to const to the first element of the array
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

  /* \brief Returns a pointer to const to the first element of the array
  */
  T* begin();

  /* \brief Returns a pointer to const to the one after the last element of
  //  the array
  */
  const T* end() const;

  /* \brief Returns a pointer to const to the one after the last element of
  //  the array
  */
  T* end();

 private:
  /* \brief Used internally to increase the capacity of the array
  */
  void increase_capacity(il::int_t r);

  /* \brief Used internally in debug mode to check the invariance of the object
  */
  bool invariance() const;
};

template <typename T>
Array<T>::Array() {
  data_ = nullptr;
  size_ = nullptr;
  capacity_ = nullptr;
  align_r_ = 0;
  align_mod_ = 0;
  shift_ = 0;
}

template <typename T>
Array<T>::Array(il::int_t n) {
  IL_EXPECT_FAST(n >= 0);

  if (n > 0) {
    data_ = il::allocate_array<T>(n);
    if (il::is_trivial<T>::value) {
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i = 0; i < n; ++i) {
        data_[i] = il::default_value<T>();
      }
#endif
    } else {
      for (il::int_t i = 0; i < n; ++i) {
        new (data_ + i) T{};
      }
    }
  } else {
    data_ = nullptr;
  }
  size_ = data_ + n;
  capacity_ = data_ + n;
  align_r_ = 0;
  align_mod_ = 0;
  shift_ = 0;
}

template <typename T>
Array<T>::Array(il::int_t n, il::align_t, il::int_t align_r,
                il::int_t align_mod) {
  IL_EXPECT_FAST(il::is_trivial<T>::value);
  IL_EXPECT_FAST(sizeof(T) == alignof(T));
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_FAST(align_mod > 0);
  IL_EXPECT_FAST(align_mod % alignof(T) == 0);
  IL_EXPECT_FAST(align_mod <= std::numeric_limits<short>::max());
  IL_EXPECT_FAST(align_r >= 0);
  IL_EXPECT_FAST(align_r < align_mod);
  IL_EXPECT_FAST(align_r % alignof(T) == 0);
  IL_EXPECT_FAST(align_r <= std::numeric_limits<short>::max());

  if (n > 0) {
    il::int_t shift;
    data_ = il::allocate_array<T>(n, align_r, align_mod, il::io, shift);
    align_r_ = static_cast<short>(align_r);
    align_mod_ = static_cast<short>(align_mod);
    shift_ = static_cast<short>(shift);
#ifdef IL_DEFAULT_VALUE
    for (il::int_t i = 0; i < n; ++i) {
      data_[i] = il::default_value<T>();
    }
#endif
  } else {
    data_ = nullptr;
    align_r_ = 0;
    align_mod_ = 0;
    shift_ = 0;
  }
  size_ = data_ + n;
  capacity_ = data_ + n;
}

template <typename T>
Array<T>::Array(il::int_t n, il::align_t, il::int_t align_mod)
    : Array{n, il::align, 0, align_mod} {}

template <typename T>
Array<T>::Array(il::int_t n, const T& x) {
  IL_EXPECT_FAST(n >= 0);

  if (n > 0) {
    data_ = il::allocate_array<T>(n);
    for (il::int_t i = 0; i < n; ++i) {
      new (data_ + i) T(x);
    }
  } else {
    data_ = nullptr;
  }
  size_ = data_ + n;
  capacity_ = data_ + n;
  align_r_ = 0;
  align_mod_ = 0;
  shift_ = 0;
}

template <typename T>
Array<T>::Array(il::int_t n, const T& x, il::align_t, il::int_t align_r,
                il::int_t align_mod) {
  IL_EXPECT_FAST(il::is_trivial<T>::value);
  IL_EXPECT_FAST(sizeof(T) == alignof(T));
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_FAST(align_mod > 0);
  IL_EXPECT_FAST(align_mod % alignof(T) == 0);
  IL_EXPECT_FAST(align_mod <= std::numeric_limits<short>::max());
  IL_EXPECT_FAST(align_r >= 0);
  IL_EXPECT_FAST(align_r < align_mod);
  IL_EXPECT_FAST(align_r % alignof(T) == 0);
  IL_EXPECT_FAST(align_r <= std::numeric_limits<short>::max());

  if (n > 0) {
    il::int_t shift;
    data_ = il::allocate_array<T>(n, align_r, align_mod, il::io, shift);
    align_r_ = static_cast<short>(align_r);
    align_mod_ = static_cast<short>(align_mod);
    shift_ = static_cast<short>(shift);
    for (il::int_t i = 0; i < n; ++i) {
      data_[i] = x;
    }
  } else {
    data_ = nullptr;
    align_r_ = 0;
    align_mod_ = 0;
    shift_ = 0;
  }
  size_ = data_ + n;
  capacity_ = data_ + n;
}

template <typename T>
Array<T>::Array(il::value_t, std::initializer_list<T> list) {
  bool error = false;
  const il::int_t n = il::safe_convert<il::int_t>(list.size(), il::io, error);
  if (error) {
    std::abort();
  }

  if (n > 0) {
    data_ = il::allocate_array<T>(n);
    if (il::is_trivial<T>::value) {
      memcpy(data_, list.begin(), n * sizeof(T));
    } else {
      for (il::int_t i = 0; i < n; ++i) {
        new (data_ + i) T(*(list.begin() + i));
      }
    }
  } else {
    data_ = nullptr;
  }
  size_ = data_ + n;
  capacity_ = data_ + n;
  align_r_ = 0;
  align_mod_ = 0;
  shift_ = 0;
}

template <typename T>
Array<T>::Array(const Array<T>& v) {
  const il::int_t n = v.size();
  const il::int_t align_r = v.align_r_;
  const il::int_t align_mod = v.align_mod_;
  if (align_mod == 0) {
    data_ = il::allocate_array<T>(n);
    align_r_ = 0;
    align_mod_ = 0;
    shift_ = 0;
  } else {
    il::int_t shift;
    data_ = il::allocate_array<T>(n, align_r, align_mod, il::io, shift);
    align_r_ = static_cast<short>(align_r);
    align_mod_ = static_cast<short>(align_mod);
    shift_ = static_cast<short>(shift);
  }
  if (il::is_trivial<T>::value) {
    memcpy(data_, v.data_, n * sizeof(T));
  } else {
    for (il::int_t i = 0; i < n; ++i) {
      new (data_ + i) T(v.data_[i]);
    }
  }
  size_ = data_ + n;
  capacity_ = data_ + n;
}

template <typename T>
Array<T>::Array(Array<T>&& v) {
  data_ = v.data_;
  size_ = v.size_;
  capacity_ = v.capacity_;
  align_r_ = v.align_r_;
  align_mod_ = v.align_mod_;
  shift_ = v.shift_;
  v.data_ = nullptr;
  v.size_ = nullptr;
  v.capacity_ = nullptr;
  v.align_r_ = 0;
  v.align_mod_ = 0;
  v.shift_ = 0;
}

template <typename T>
Array<T>& Array<T>::operator=(const Array<T>& v) {
  if (this == &v) {
    return *this;
  }

  const il::int_t n = v.size();
  const il::int_t align_r = v.align_r_;
  const il::int_t align_mod = v.align_mod_;
  const bool needs_memory =
      n > capacity() || align_mod_ != align_mod || align_r_ != align_r;
  if (needs_memory) {
    if (data_) {
      if (!il::is_trivial<T>::value) {
        for (il::int_t i = size() - 1; i >= 0; --i) {
          (data_ + i)->~T();
        }
      }
      il::deallocate(data_ - shift_);
    }
    if (align_mod == 0) {
      data_ = il::allocate_array<T>(n);
      align_r_ = 0;
      align_mod_ = 0;
      shift_ = 0;
    } else {
      il::int_t shift;
      data_ = il::allocate_array<T>(n, align_r, align_mod, il::io, shift);
      align_r_ = static_cast<short>(align_r);
      align_mod_ = static_cast<short>(align_mod);
      shift_ = static_cast<short>(shift);
    }
    if (il::is_trivial<T>::value) {
      memcpy(data_, v.data_, n * sizeof(T));
    } else {
      for (il::int_t i = 0; i < n; ++i) {
        new (data_ + i) T(v.data_[i]);
      }
    }
    size_ = data_ + n;
    capacity_ = data_ + n;
  } else {
    if (il::is_trivial<T>::value) {
      memcpy(data_, v.data_, n * sizeof(T));
    } else {
      for (il::int_t i = 0; i < n; ++i) {
        data_[i] = v.data_[i];
      }
      for (il::int_t i = size() - 1; i >= n; --i) {
        (data_ + i)->~T();
      }
    }
    size_ = data_ + n;
  }
  return *this;
}

template <typename T>
Array<T>& Array<T>::operator=(Array<T>&& v) {
  if (this == &v) {
    return *this;
  }

  if (data_) {
    if (!il::is_trivial<T>::value) {
      for (il::int_t i = size() - 1; i >= 0; --i) {
        (data_ + i)->~T();
      }
    }
    il::deallocate(data_ - shift_);
  }
  data_ = v.data_;
  size_ = v.size_;
  capacity_ = v.capacity_;
  align_r_ = v.align_r_;
  align_mod_ = v.align_mod_;
  shift_ = v.shift_;
  v.data_ = nullptr;
  v.size_ = nullptr;
  v.capacity_ = nullptr;
  v.align_r_ = 0;
  v.align_mod_ = 0;
  v.shift_ = 0;
  return *this;
}

template <typename T>
Array<T>::~Array() {
  IL_EXPECT_FAST(invariance());

  if (data_) {
    if (!il::is_trivial<T>::value) {
      for (il::int_t i = size() - 1; i >= 0; --i) {
        (data_ + i)->~T();
      }
    }
    il::deallocate(data_ - shift_);
  }
}

template <typename T>
const T& Array<T>::operator[](il::int_t i) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(size()));
  return data_[i];
}

template <typename T>
T& Array<T>::operator[](il::int_t i) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(size()));
  return data_[i];
}

template <typename T>
const T& Array<T>::back() const {
  IL_EXPECT_MEDIUM(size() > 0);
  return size_[-1];
}

template <typename T>
T& Array<T>::back() {
  IL_EXPECT_MEDIUM(size() > 0);
  return size_[-1];
}

template <typename T>
il::int_t Array<T>::size() const {
  return size_ - data_;
}

template <typename T>
void Array<T>::resize(il::int_t n) {
  IL_EXPECT_FAST(n >= 0);

  if (n <= capacity()) {
    if (il::is_trivial<T>::value) {
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i = size(); i < n; ++i) {
        data_[i] = il::default_value<T>();
      }
#endif
    } else {
      for (il::int_t i = size() - 1; i >= n; --i) {
        (data_ + i)->~T();
      }
      for (il::int_t i = size(); i < n; ++i) {
        new (data_ + i) T{};
      }
    }
  } else {
    const il::int_t n_old = size();
    increase_capacity(n);
    if (il::is_trivial<T>::value) {
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i = n_old; i < n; ++i) {
        data_[i] = il::default_value<T>();
      }
#endif
    } else {
      for (il::int_t i = n_old; i < n; ++i) {
        new (data_ + i) T{};
      }
    }
  }
  size_ = data_ + n;
}

template <typename T>
il::int_t Array<T>::capacity() const {
  return static_cast<il::int_t>(capacity_ - data_);
}

template <typename T>
void Array<T>::reserve(il::int_t r) {
  IL_EXPECT_FAST(r >= 0);

  if (r > capacity()) {
    increase_capacity(r);
  }
}

template <typename T>
void Array<T>::append(const T& x) {
  if (size_ == capacity_) {
    const il::int_t n = size();
    bool error = false;
    il::int_t new_capacity =
        n > 1 ? il::safe_sum(n, n / 2, il::io, error)
              : il::safe_sum(n, static_cast<il::int_t>(1), il::io, error);
    if (error) {
      std::abort();
    }
    T x_copy = x;
    increase_capacity(new_capacity);
    new (size_) T(std::move(x_copy));
  } else {
    new (size_) T(x);
  }
  ++size_;
}

template <typename T>
void Array<T>::append(T&& x) {
  if (size_ == capacity_) {
    const il::int_t n = size();
    bool error = false;
    il::int_t new_capacity =
        n > 1 ? il::safe_sum(n, n / 2, il::io, error)
              : il::safe_sum(n, static_cast<il::int_t>(1), il::io, error);
    if (error) {
      std::abort();
    }
    increase_capacity(new_capacity);
  }
  if (il::is_trivial<T>::value) {
    *size_ = std::move(x);
  } else {
    new (size_) T(std::move(x));
  }
  ++size_;
}

template <typename T>
template <typename... Args>
void Array<T>::append(il::emplace_t, Args&&... args) {
  if (size_ == capacity_) {
    const il::int_t n = size();
    bool error = false;
    il::int_t new_capacity =
        n > 1 ? il::safe_sum(n, n / 2, il::io, error)
              : il::safe_sum(n, static_cast<il::int_t>(1), il::io, error);
    if (error) {
      std::abort();
    }
    increase_capacity(new_capacity);
  };
  new (size_) T(args...);
  ++size_;
}

template <typename T>
il::int_t Array<T>::alignment() const {
  unsigned int a = static_cast<unsigned int>(align_mod_);
  unsigned int b = static_cast<unsigned int>(align_r_);
  while (b != 0) {
    unsigned int c = a;
    a = b;
    b = c % b;
  }
  return static_cast<il::int_t>(a);
}

template <typename T>
const T* Array<T>::data() const {
  return data_;
}

template <typename T>
T* Array<T>::data() {
  return data_;
}

template <typename T>
const T* Array<T>::begin() const {
  return data_;
}

template <typename T>
T* Array<T>::begin() {
  return data_;
}

template <typename T>
const T* Array<T>::end() const {
  return size_;
}

template <typename T>
T* Array<T>::end() {
  return size_;
}

template <typename T>
void Array<T>::increase_capacity(il::int_t r) {
  IL_EXPECT_FAST(capacity() < r);

  const il::int_t n = size();
  T* new_data;
  short new_shift;
  if (align_mod_ == 0) {
    new_data = il::allocate_array<T>(r);
    new_shift = 0;
  } else {
    il::int_t shift;
    new_data = il::allocate_array<T>(n, align_r_, align_mod_, il::io, shift);
    new_shift = static_cast<short>(shift);
  }
  if (data_) {
    if (il::is_trivial<T>::value) {
      memcpy(new_data, data_, n * sizeof(T));
    } else {
      for (il::int_t i = n - 1; i >= 0; --i) {
        new (new_data + i) T(std::move(data_[i]));
        (data_ + i)->~T();
      }
    }
    il::deallocate(data_ - shift_);
  }
  data_ = new_data;
  size_ = data_ + n;
  capacity_ = data_ + r;
  shift_ = new_shift;
}

template <typename T>
bool Array<T>::invariance() const {
  bool ans = true;

  if (data_ == nullptr) {
    ans = ans && (size_ == nullptr);
    ans = ans && (capacity_ == nullptr);
  } else {
    ans = ans && (size_ != nullptr);
    ans = ans && (capacity_ != nullptr);
    ans = ans && ((size_ - data_) <= (capacity_ - data_));
  }
  if (!il::is_trivial<T>::value) {
    ans = ans && (align_mod_ == 0);
  }
  if (align_mod_ == 0) {
    ans = ans && (align_r_ == 0);
    ans = ans && (shift_ == 0);
  } else {
    ans = ans && (align_r_ < align_mod_);
    ans = ans && (reinterpret_cast<std::size_t>(data_) %
                      static_cast<std::size_t>(align_mod_) ==
                  static_cast<std::size_t>(align_r_));
  }
  return ans;
}
}

#endif  // IL_ARRAY_H
