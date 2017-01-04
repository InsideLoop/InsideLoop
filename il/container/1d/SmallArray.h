//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_SMALLARRAY_H
#define IL_SMALLARRAY_H

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

template <typename T, il::int_t small_size>
class SmallArray {
  static_assert(small_size >= 1,
                "il::SmallArray<T, small_size>: small_size must be positive");

 private:
#ifdef IL_DEBUG_VISUALIZER
  il::int_t debug_size_;
  il::int_t debug_capacity_;
  bool debug_small_data_used_;
#endif
  T* data_;
  T* size_;
  T* capacity_;
  alignas(T) char small_data_[small_size * sizeof(T)];

 public:
  /* \brief Default constructor
  // \details The size is set to 0 and the capacity is set to small_size.
  */
  SmallArray();

  /* \brief Construct a small array of n elements
  // \details The size and the capacity of the array are set to n.
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
  // // Construct a small array of double of size 5 with a stack memory of
  // // size 10.
  // il::SmallArray<double, 10> v{5};
  */
  explicit SmallArray(il::int_t n);

  /* \brief Construct a small array of n elements from a value
  // \details Initialize the array of length n with a given value.
  /
  // // Construct an array of double of length 5, initialized with 0.0
  // il::SmallArray<double, 10> v{5, 0.0};
  */
  explicit SmallArray(il::int_t n, const T& x);

  /* \brief Construct a il::SmallArray<T, small_size> from a brace-initialized
  // list
  // \details The size and the capacity of the il::SmallArray<T, small_size> is
  // adjusted to the size of the initializer list. The tag il::value is used to
  // allow brace initialization of il::Array<T> everywhere.
  //
  // // Construct an array of double from a list
  // il::SmallArray<double, 4> v{il::value, {2.0, 3.14, 5.0, 7.0}};
  */
  explicit SmallArray(il::value_t, std::initializer_list<T> list);

  /* \brief The copy constructor
  // \details The size of the constructed array is the same as the one for the
  // source array. However, its capacity is the same as its size even though
  // the source array had a larger capacity.
  */
  SmallArray(const SmallArray<T, small_size>& A);

  /* \brief The move constructor
  */
  SmallArray(SmallArray<T, small_size>&& A);

  /* \brief The copy assignment
  // \details The size is the same as the one for the source array. The
  // capacity is not changed if it is enough for the copy to happen and is
  // set to the size of the source array if the initial capacity is too low.
  */
  SmallArray& operator=(const SmallArray<T, small_size>& A);

  /* \brief The move assignment
  */
  SmallArray& operator=(SmallArray<T, small_size>&& A);

  /* \brief The destructor
  // \details If T is an object, they are destructed from the last one to the
  // first one. Then, the allocated memory is released.
  */
  ~SmallArray();

  /* \brief Accessor for a const object
  // \details Access (read only) the i-th element of the array. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::SmallArray<double, 10> v{4};
  // std::cout << v[0] << std::endl;
  */
  const T& operator[](il::int_t i) const;

  /* \brief Accessor
  // \details Access (read or write) the i-th element of the array. Bound
  // checking is done in debug mode but not in release mode.
  //
  // il::SmallArray<double, 10> v{4};
  // v[0] = 0.0;
  // v[4] = 0.0; // Program is aborted in debug mode and has undefined
  //             // behavior in release mode
  */
  T& operator[](il::int_t i);

  /* \brief Accessor to the last element of a const
  // il::SmallArray<T, small_size>
  // \details In debug mode, calling this method on an array of size 0 aborts
  // the program. In release mode, it will lead to undefined behavior.
  */
  const T& back() const;

  /* \brief Accessor to the last element of a il::SmallArray<T, small_size>
  // \details In debug mode, calling this method on an array of size 0 aborts
  // the program. In release mode, it will lead to undefined behavior.
  */
  T& back();

  /* \brief Get the size of the array
  //
  // il::SmallArray<double, 10> v{4};
  // for (il::int_t k = 0; k < v.size(); ++k) {
  //     v[k] = 1.0 / (k + 1);
  // }
  */
  il::int_t size() const;

  /* \brief Change the size of the array
  // \details No reallocation is performed if the new size is <= to the
  // capacity. In this case, the capacity is unchanged. When the size is > at
  // the current capacity, reallocation is done and the array gets the same
  // capacity as its size.
  */
  void resize(il::int_t n);

  /* \brief Get the capacity of the array
  */
  il::int_t capacity() const;

  /* \brief Change the capacity of the array to at least p
  // \details If the capacity is >= to p, nothing is done. Otherwise,
  // reallocation is done and the new capacity is set to p.
  */
  void reserve(il::int_t p);

  /* \brief Add an element at the end of the array
  // \details Reallocation is done only if it is needed. In case reallocation
  // happens, then new capacity is roughly (3/2) the previous capacity.
  */
  void append(const T& x);

  /* \brief Construct an element at the end of the array
  // \details Reallocation is done only if it is needed. In case reallocation
  // happens, then new capacity is roughly (3/2) the previous capacity.
  */
  template <typename... Args>
  void emplace_back(Args&&... args);

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

  /* \brief Returns a pointer to const to the one after the last element of
  //  the array
  */
  const T* end() const;

 private:
  /* \brief Used internally to check if the stack array is used
  */
  bool small_data_used() const;

  /* \brief Used internally to increase the capacity of the array
  */
  void increase_capacity(il::int_t r);

  /* \brief Used internally in debug mode to check the invariance of the object
  */
  void check_invariance() const;
};

template <typename T, il::int_t small_size>
SmallArray<T, small_size>::SmallArray() {
#ifdef IL_DEBUG_VISUALIZER
  debug_size_ = 0;
  debug_capacity_ = small_size;
  debug_small_data_used_ = true;
#endif
  data_ = reinterpret_cast<T*>(small_data_);
  size_ = data_;
  capacity_ = data_ + small_size;
}

template <typename T, il::int_t small_size>
SmallArray<T, small_size>::SmallArray(il::int_t n) {
  IL_ASSERT(n >= 0);
  if (n <= small_size) {
#ifdef IL_DEBUG_VISUALIZER
    debug_size_ = n;
    debug_capacity_ = small_size;
    debug_small_data_used_ = true;
#endif
    data_ = reinterpret_cast<T*>(small_data_);
    size_ = data_ + n;
    capacity_ = data_ + small_size;
  } else {
#ifdef IL_DEBUG_VISUALIZER
    debug_size_ = n;
    debug_capacity_ = n;
    debug_small_data_used_ = false;
#endif
    data_ = static_cast<T*>(::operator new(n * sizeof(T)));
    size_ = data_ + n;
    capacity_ = data_ + n;
  }
  if (std::is_pod<T>::value) {
#ifdef IL_DEFAULT_VALUE
    for (il::int_t i{0}; i < n; ++i) {
      data_[i] = il::default_value<T>();
    }
#endif
  } else {
    for (il::int_t i{0}; i < n; ++i) {
      new (data_ + i) T{};
    }
  }
}

template <typename T, il::int_t small_size>
SmallArray<T, small_size>::SmallArray(il::int_t n, const T& x) {
  IL_ASSERT(n >= 0);
  if (n <= small_size) {
#ifdef IL_DEBUG_VISUALIZER
    debug_size_ = n;
    debug_capacity_ = small_size;
    debug_small_data_used_ = true;
#endif
    data_ = reinterpret_cast<T*>(small_data_);
    size_ = data_ + n;
    capacity_ = data_ + small_size;
  } else {
#ifdef IL_DEBUG_VISUALIZER
    debug_size_ = n;
    debug_capacity_ = n;
    debug_small_data_used_ = false;
#endif
    data_ = static_cast<T*>(::operator new(n * sizeof(T)));
    size_ = data_ + n;
    capacity_ = data_ + n;
  }
  for (il::int_t i{0}; i < n; ++i) {
    new (data_ + i) T(x);
  }
}

template <typename T, il::int_t small_size>
SmallArray<T, small_size>::SmallArray(il::value_t,
                                      std::initializer_list<T> list) {
  const il::int_t n{static_cast<il::int_t>(list.size())};
  if (n <= small_size) {
#ifdef IL_DEBUG_VISUALIZER
    debug_size_ = n;
    debug_capacity_ = small_size;
    debug_small_data_used_ = true;
#endif
    data_ = reinterpret_cast<T*>(small_data_);
    size_ = data_ + n;
    capacity_ = data_ + small_size;
  } else {
#ifdef IL_DEBUG_VISUALIZER
    debug_size_ = n;
    debug_capacity_ = n;
    debug_small_data_used_ = false;
#endif
    data_ = static_cast<T*>(::operator new(n * sizeof(T)));
    size_ = data_ + n;
    capacity_ = size_;
  }
  if (std::is_pod<T>::value) {
    memcpy(data_, list.begin(), n * sizeof(T));
  } else {
    for (il::int_t i{0}; i < n; ++i) {
      new (data_ + i) T(*(list.begin() + i));
    }
  }
}

template <typename T, il::int_t small_size>
SmallArray<T, small_size>::SmallArray(const SmallArray<T, small_size>& A) {
  const il::int_t n{A.size()};
  if (n <= small_size) {
#ifdef IL_DEBUG_VISUALIZER
    debug_size_ = n;
    debug_capacity_ = small_size;
    debug_small_data_used_ = true;
#endif
    data_ = reinterpret_cast<T*>(small_data_);
    size_ = data_ + n;
    capacity_ = data_ + small_size;
  } else {
#ifdef IL_DEBUG_VISUALIZER
    debug_size_ = n;
    debug_capacity_ = n;
    debug_small_data_used_ = false;
#endif
    data_ = static_cast<T*>(::operator new(n * sizeof(T)));
    size_ = data_ + n;
    capacity_ = size_;
  }
  if (std::is_pod<T>::value) {
    memcpy(data_, A.data_, n * sizeof(T));
  } else {
    for (il::int_t i{0}; i < n; ++i) {
      new (data_ + i) T(A.data_[i]);
    }
  }
}

template <typename T, il::int_t small_size>
SmallArray<T, small_size>::SmallArray(SmallArray<T, small_size>&& A) {
  const il::int_t n{A.size()};
  if (A.small_data_used()) {
    data_ = reinterpret_cast<T*>(small_data_);
    if (std::is_pod<T>::value) {
      memcpy(data_, A.data_, n * sizeof(T));
    } else {
      for (il::int_t i{0}; i < n; ++i) {
        new (data_ + i) T(std::move(A.data_[i]));
        (data_ + i)->~T();
      }
    }
#ifdef IL_DEBUG_VISUALIZER
    debug_size_ = n;
    debug_capacity_ = small_size;
    debug_small_data_used_ = true;
#endif
    size_ = data_ + n;
    capacity_ = data_ + small_size;
  } else {
#ifdef IL_DEBUG_VISUALIZER
    debug_size_ = A.debug_size_;
    debug_capacity_ = A.debug_capacity_;
    debug_small_data_used_ = false;
#endif
    data_ = A.data_;
    size_ = A.size_;
    capacity_ = A.capacity_;
  }
#ifdef IL_DEBUG_VISUALIZER
  A.debug_size_ = 0;
  A.debug_capacity_ = 0;
  A.debug_small_data_used_ = false;
#endif
  A.data_ = reinterpret_cast<T*>(A.small_data_);
  A.size_ = A.data_ + 0;
  A.capacity_ = A.data_ + small_size;
}

template <typename T, il::int_t small_size>
SmallArray<T, small_size>& SmallArray<T, small_size>::operator=(
    const SmallArray<T, small_size>& A) {
  if (this != &A) {
    const il::int_t n{A.size()};
    const bool needs_memory{capacity() < n};
    if (needs_memory) {
      if (std::is_pod<T>::value) {
        if (!small_data_used()) {
          delete[] data_;
        }
        data_ = new T[n];
        memcpy(data_, A.data_, n * sizeof(T));
      } else {
        for (il::int_t i{0}; i < size(); ++i) {
          (data_ + i)->~T();
        }
        if (!small_data_used()) {
          ::operator delete(data_);
        }
        data_ = static_cast<T*>(::operator new(n * sizeof(T)));
        for (il::int_t i{0}; i < n; ++i) {
          new (data_ + i) T(A.data_[i]);
        }
      }
#ifdef IL_DEBUG_VISUALIZER
      debug_size_ = n;
      debug_capacity_ = n;
      debug_small_data_used_ = false;
#endif
      size_ = data_ + n;
      capacity_ = data_ + n;
    } else {
      if (!std::is_pod<T>::value) {
        for (il::int_t i{n}; i < size(); ++i) {
          (data_ + i)->~T();
        }
      }
      for (il::int_t i{0}; i < n; ++i) {
        data_[i] = A.data_[i];
      }
#ifdef IL_DEBUG_VISUALIZER
      debug_size_ = n;
#endif
      size_ = data_ + n;
    }
  }
  return *this;
}

template <typename T, il::int_t small_size>
SmallArray<T, small_size>& SmallArray<T, small_size>::operator=(
    SmallArray<T, small_size>&& A) {
  if (this != &A) {
    if (std::is_pod<T>::value) {
      if (!small_data_used()) {
        delete[] data_;
      }
    } else {
      for (il::int_t i{0}; i < size(); ++i) {
        (data_ + i)->~T();
      }
      if (!small_data_used()) {
        ::operator delete(data_);
      }
    }
    const il::int_t n{A.size()};
    if (A.small_data_used()) {
      data_ = reinterpret_cast<T*>(small_data_);
      if (std::is_pod<T>::value) {
        memcpy(data_, A.data_, n * sizeof(T));
      } else {
        for (il::int_t i{0}; i < n; ++i) {
          new (data_ + i) T(std::move(A.data_[i]));
          (data_ + i)->~T();
        }
      }
#ifdef IL_DEBUG_VISUALIZER
      debug_size_ = n;
      debug_capacity_ = small_size;
      debug_small_data_used_ = true;
#endif
      size_ = data_ + n;
      capacity_ = data_ + small_size;
    } else {
#ifdef IL_DEBUG_VISUALIZER
      debug_size_ = A.debug_size_;
      debug_capacity_ = A.debug_capacity_;
      debug_small_data_used_ = false;
#endif
      data_ = A.data_;
      size_ = A.size_;
      capacity_ = A.capacity_;
    }
#ifdef IL_DEBUG_VISUALIZER
    A.debug_size_ = 0;
    A.debug_capacity_ = small_size;
    A.debug_small_data_used_ = true;
#endif
    data_ = reinterpret_cast<T*>(small_data_);
    A.size_ = A.data_ + 0;
    A.capacity_ = A.data_ + small_size;
  }
  return *this;
}

template <typename T, il::int_t small_size>
SmallArray<T, small_size>::~SmallArray() {
#ifdef IL_INVARIANCE
  check_invariance();
#endif
  if (std::is_pod<T>::value) {
    if (!small_data_used()) {
      delete[] data_;
    }
  } else {
    for (il::int_t i{0}; i <= size(); ++i) {
      (data_ + i)->~T();
    }
    if (!small_data_used()) {
      ::operator delete(data_);
    }
  }
}

template <typename T, il::int_t small_size>
const T& SmallArray<T, small_size>::operator[](il::int_t i) const {
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i) <
                   static_cast<il::uint_t>(size()));
  return data_[i];
}

template <typename T, il::int_t small_size>
T& SmallArray<T, small_size>::operator[](il::int_t i) {
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i) <
                   static_cast<il::uint_t>(size()));
  return data_[i];
}

template <typename T, il::int_t small_size>
const T& SmallArray<T, small_size>::back() const {
  IL_ASSERT(size() > 0);
  return size_[-1];
}

template <typename T, il::int_t small_size>
T& SmallArray<T, small_size>::back() {
  IL_ASSERT(size() > 0);
  return size_[-1];
}

template <typename T, il::int_t small_size>
il::int_t SmallArray<T, small_size>::size() const {
  return static_cast<il::int_t>(size_ - data_);
}

template <typename T, il::int_t small_size>
void SmallArray<T, small_size>::resize(il::int_t n) {
  IL_ASSERT(n >= 0);
  if (n <= capacity()) {
    if (std::is_pod<T>::value) {
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i{size()}; i < n; ++i) {
        data_[i] = il::default_value<T>();
      }
#endif
    } else {
      for (il::int_t i{0}; i < size(); ++i) {
        (data_ + i)->~T();
      }
      for (il::int_t i{size()}; i < n; ++i) {
        new (data_ + i) T{};
      }
    }
  } else {
    const il::int_t n_old{size()};
    increase_capacity(n);
    if (std::is_pod<T>::value) {
#ifdef IL_DEFAULT_VALUE
      for (il::int_t i{n_old}; i < n; ++i) {
        data_[i] = il::default_value<T>();
      }
#endif
    } else {
      for (il::int_t i{n_old}; i < n; ++i) {
        new (data_ + i) T{};
      }
    }
  }
#ifdef IL_DEBUG_VISUALIZER
  debug_size_ = n;
#endif
  size_ = data_ + n;
}

template <typename T, il::int_t small_size>
il::int_t SmallArray<T, small_size>::capacity() const {
  return static_cast<il::int_t>(capacity_ - data_);
}

template <typename T, il::int_t small_size>
void SmallArray<T, small_size>::reserve(il::int_t r) {
  IL_ASSERT(r >= 0);
  if (r > capacity()) {
    increase_capacity(r);
  }
}

template <typename T, il::int_t small_size>
void SmallArray<T, small_size>::append(const T& x) {
  if (size_ == capacity_) {
    const il::int_t n{size()};
    increase_capacity(n > 1 ? (3 * n) / 2 : n + 1);
  }
  new (size_) T(x);
#ifdef IL_DEBUG_VISUALIZER
  ++debug_size_;
#endif
  ++size_;
}

template <typename T, il::int_t small_size>
template <typename... Args>
void SmallArray<T, small_size>::emplace_back(Args&&... args) {
  if (size_ == capacity_) {
    const il::int_t n{size()};
    increase_capacity(n > 1 ? (3 * n) / 2 : n + 1);
  };
  new (size_) T(args...);
#ifdef IL_DEBUG_VISUALIZER
  ++debug_size_;
#endif
  ++size_;
}

template <typename T, il::int_t small_size>
T* SmallArray<T, small_size>::data() {
  return data_;
}

template <typename T, il::int_t small_size>
const T* SmallArray<T, small_size>::data() const {
  return data_;
}

template <typename T, il::int_t small_size>
const T* SmallArray<T, small_size>::begin() const {
  return data_;
}

template <typename T, il::int_t small_size>
const T* SmallArray<T, small_size>::end() const {
  return size_;
}

template <typename T, il::int_t small_size>
bool SmallArray<T, small_size>::small_data_used() const {
  return data_ == reinterpret_cast<const T*>(small_data_);
}

template <typename T, il::int_t small_size>
void SmallArray<T, small_size>::increase_capacity(il::int_t r) {
  IL_ASSERT(size() <= r);
  const il::int_t n{size()};
  T* new_data;
  if (std::is_pod<T>::value) {
    new_data = new T[r];
  } else {
    new_data = static_cast<T*>(::operator new(r * sizeof(T)));
  }
  if (std::is_pod<T>::value) {
    memcpy(new_data, data_, n * sizeof(T));
    if (!small_data_used()) {
      delete[] data_;
    }
  } else {
    for (il::int_t i{0}; i < n; ++i) {
      new (new_data + i) T(std::move(data_[i]));
      (data_ + i)->~T();
    }
    if (!small_data_used()) {
      ::operator delete(data_);
    }
  }
#ifdef IL_DEBUG_VISUALIZER
  debug_capacity_ = r;
#endif
  data_ = new_data;
  size_ = data_ + n;
  capacity_ = data_ + r;
}

template <typename T, il::int_t small_size>
void SmallArray<T, small_size>::check_invariance() const {
  IL_ASSERT(size_ - data_ >= 0);
  IL_ASSERT(capacity_ - data_ >= 0);
  IL_ASSERT((size_ - data_) <= (capacity_ - data_));
}
}

#endif  // IL_SMALLARRAY_H
