//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_LOWERARRAY2D_H
#define IL_LOWERARRAY2D_H

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
class LowerArray2D {
 private:
#ifdef IL_DEBUG_VISUALIZER
  il::int_t debug_size_;
  il::int_t debug_capacity_;
#endif
  T* data_;
  T* size_;
  T* capacity_;

 public:
  LowerArray2D();
  LowerArray2D(il::int_t n);
  LowerArray2D(const LowerArray2D<T>& A);
  LowerArray2D(LowerArray2D<T>&& A);
  LowerArray2D<T>& operator=(const LowerArray2D<T>& A);
  LowerArray2D<T>& operator=(LowerArray2D<T>&& A);
  ~LowerArray2D();
  const T& operator()(il::int_t i0, il::int_t i1) const;
  T& operator()(il::int_t i0, il::int_t i1);
  il::int_t size() const;
  il::int_t capacity() const;
  T* data();
};

template <typename T>
LowerArray2D<T>::LowerArray2D() {
#ifdef IL_DEBUG_VISUALIZER
  debug_size_ = 0;
  debug_capacity_ = 0;
#endif
  data_ = nullptr;
  size_ = nullptr;
  capacity_ = nullptr;
}

template <typename T>
LowerArray2D<T>::LowerArray2D(il::int_t n) {
  IL_ASSERT(n >= 0);
  if (n > 0) {
    const il::int_t nb_elements{(n * (n + 1)) / 2};
    if (std::is_pod<T>::value) {
      data_ = new T[nb_elements];
#ifdef IL_DEFAULT_VALUE
      for (il::int_t k{0}; k < nb_elements; ++k) {
        data_[k] = il::default_value<T>();
      }
#endif
    } else {
      data_ = static_cast<T*>(::operator new(nb_elements * sizeof(T)));
      for (il::int_t k{0}; k < nb_elements; ++k) {
        new (data_ + k) T{};
      }
    }
  } else {
    data_ = nullptr;
  }
#ifdef IL_DEBUG_VISUALIZER
  debug_size_ = n;
  debug_capacity_ = n;
#endif
  size_ = data_ + n;
  capacity_ = data_ + n;
}

template <typename T>
LowerArray2D<T>::LowerArray2D(const LowerArray2D<T>& A) {
  const il::int_t n{A.size()};
  if (n > 0) {
    const il::int_t nb_elements{(n * (n + 1)) / 2};
    if (std::is_pod<T>::value) {
      data_ = new T[nb_elements];
      memcpy(data_, A.data_, nb_elements * sizeof(T));
    } else {
      data_ = static_cast<T*>(::operator new(nb_elements * sizeof(T)));
      for (il::int_t k{0}; k < nb_elements; ++k) {
        new (data_ + k) T{A.data_[k]};
      }
    }
  } else {
    data_ = nullptr;
  }
#ifdef IL_DEBUG_VISUALIZER
  debug_size_ = n;
  debug_capacity_ = n;
#endif
  size_ = data_ + n;
  capacity_ = data_ + n;
}

template <typename T>
LowerArray2D<T>::LowerArray2D(LowerArray2D<T>&& A) {
#ifdef IL_DEBUG_VISUALIZER
  debug_size_ = A.debug_size_;
  debug_capacity_ = A.debug_capacity_;
#endif
  data_ = A.data_;
  size_ = A.size_;
  capacity_ = A.capacity_;
#ifdef IL_DEBUG_VISUALIZER
  A.debug_size_ = 0;
  A.debug_capacity_ = 0;
#endif
  A.data_ = nullptr;
  A.size_ = nullptr;
  A.capacity_ = nullptr;
}

template <typename T>
LowerArray2D<T>& LowerArray2D<T>::operator=(const LowerArray2D<T>& A) {
  if (this != &A) {
    const il::int_t n{A.size()};
    const il::int_t nb_elements{(n * (n + 1)) / 2};
    const bool needs_memory{n > capacity()};
    if (needs_memory) {
      if (std::is_pod<T>::value) {
        if (data_) {
          delete[] data_;
        }
        data_ = new T[nb_elements];
        memcpy(data_, A.data_, nb_elements * sizeof(T));
      } else {
        if (data_) {
          for (il::int_t k{nb_elements - 1}; k >= 0; --k) {
            (data_ + k)->~T();
          }
          ::operator delete(data_);
        }
        data_ = static_cast<T*>(::operator new(nb_elements * sizeof(T)));
        for (il::int_t k{0}; k < nb_elements; ++k) {
          new (data_ + k) T{A.data_[k]};
        }
      }
#ifdef IL_DEBUG_VISUALIZER
      debug_size_ = n;
      debug_capacity_ = n;
#endif
      size_ = data_ + n;
      capacity_ = data_ + n;
    } else {
      if (std::is_pod<T>::value) {
        memcpy(data_, A.data_, n * sizeof(T));
      } else {
        for (il::int_t k{0}; k < nb_elements; ++k) {
          data_[k] = A.data_[k];
        }
        const il::int_t nb_elements_old{(size() * (size() + 1)) / 2};
        for (il::int_t k{nb_elements_old - 1}; k >= nb_elements; --k) {
          (data_ + k)->~T();
        }
      }
#ifdef IL_DEBUG_VISUALIZER
      debug_size_ = n;
#endif
      size_ = data_ + n;
    }
  }
  return *this;
}

template <typename T>
LowerArray2D<T>& LowerArray2D<T>::operator=(LowerArray2D<T>&& A) {
  if (this != &A) {
    if (data_) {
      if (std::is_pod<T>::value) {
        delete[] data_;
      } else {
        const il::int_t nb_elements{(size() * (size() + 1)) / 2};
        for (il::int_t k{nb_elements - 1}; k >= 0; --k) {
          (data_ + k)->~T();
        }
        ::operator delete(data_);
      }
    }
#ifdef IL_DEBUG_VISUALIZER
    debug_size_ = A.debug_size_;
    debug_capacity_ = A.debug_capacity_;
#endif
    data_ = A.data_;
    size_ = A.size_;
    capacity_ = A.capacity_;
#ifdef IL_DEBUG_VISUALIZER
    A.debug_size_ = 0;
    A.debug_capacity_ = 0;
#endif
    A.data_ = nullptr;
    A.size_ = nullptr;
    A.capacity_ = nullptr;
  }
  return *this;
}

template <typename T>
LowerArray2D<T>::~LowerArray2D() {
  if (data_) {
    if (std::is_pod<T>::value) {
      delete[] data_;
    } else {
      const il::int_t nb_elements{(size() * (size() + 1)) / 2};
      for (il::int_t k{nb_elements - 1}; k >= 0; --k) {
        (data_ + k)->~T();
      }
      ::operator delete(data_);
    }
  }
}

template <typename T>
const T& LowerArray2D<T>::operator()(il::int_t i0, il::int_t i1) const {
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i0) <
                   static_cast<il::uint_t>(size()));
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i1) <
                   static_cast<il::uint_t>(size()));
  IL_ASSERT_BOUNDS(i1 <= i0);
  return data_[(i1 * (2 * (size_ - data_) - (1 + i1))) / 2 + i0];
}

template <typename T>
T& LowerArray2D<T>::operator()(il::int_t i0, il::int_t i1) {
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i0) <
                   static_cast<il::uint_t>(size()));
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i1) <
                   static_cast<il::uint_t>(size()));
  IL_ASSERT_BOUNDS(i1 <= i0);
  return data_[(i1 * (2 * (size_ - data_) - (1 + i1))) / 2 + i0];
}

template <typename T>
il::int_t LowerArray2D<T>::size() const {
  return static_cast<il::int_t>(size_ - data_);
}

template <typename T>
il::int_t LowerArray2D<T>::capacity() const {
  return static_cast<il::int_t>(capacity_ - data_);
}

template <typename T>
T* LowerArray2D<T>::data() {
  return data_;
}
}

#endif  // IL_LOWERARRAY2D_H
