//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_SET_H
#define IL_SET_H

#include <il/container/hash/HashFunction.h>

namespace il {

template <typename T, typename F = HashFunction<T>>
class Set {
 private:
  T* bucket_;
  il::int_t nb_elements_;
  il::int_t nb_tombstones_;
  int p_;

 public:
  Set();
  Set(const Set<T, F>& set);
  Set(Set<T, F>&& set);
  Set& operator=(const Set<T, F>& set);
  Set& operator=(Set<T, F>&& set);
  ~Set();

  il::int_t search(const T& x) const;
  void insert(const T& x, il::io_t, il::int_t i);
  void insert(const T& x);
  bool found(il::int_t i) const;
  bool contains(const T& x) const;
  void clear();
  const T& operator[](il::int_t i) const;
  il::int_t first() const;
  il::int_t sentinel() const;
  il::int_t next(il::int_t i) const;
  il::int_t nbElements() const;

 private:
  il::int_t nbBuckets() const;
  il::int_t nbBuckets(int p) const;
  void reserveWithP(int p);
};

template <typename T, typename F>
Set<T, F>::Set() {
  bucket_ = nullptr;
  nb_elements_ = 0;
  nb_tombstones_ = 0;
  p_ = -1;
}

template <typename T, typename F>
Set<T, F>::Set(const Set<T, F>& set) {
  p_ = set.p_;
  nb_elements_ = 0;
  nb_tombstones_ = 0;
  if (p_ >= 0) {
    const il::int_t m = nbBuckets(p_);
    bucket_ = il::allocateArray<T>(m);
    for (il::int_t i = 0; i < m; ++i) {
      F::constructEmpty(il::io, bucket_ + i);
    }
    for (il::int_t i = set.first(); i < set.sentinel(); i = set.next(i)) {
      insert(set[i]);
    }
  }
}

template <typename T, typename F>
Set<T, F>::Set(Set<T, F>&& set) {
  bucket_ = set.bucket_;
  p_ = set.p_;
  nb_elements_ = set.nb_elements_;
  nb_tombstones_ = set.nb_tombstones_;
  set.bucket_ = nullptr;
  set.p_ = -1;
  set.nb_elements_ = 0;
  set.nb_tombstones_ = 0;
}

template <typename T, typename F>
Set<T, F>& Set<T, F>::operator=(const Set<T, F>& set) {
  const int p = set.p_;
  const int old_p = p_;
  if (p >= 0) {
    if (old_p >= 0) {
      const il::int_t old_m = nbBuckets(old_p);
      for (il::int_t i = 0; i < old_m; ++i) {
        if (!F::isEmpty(bucket_ + i) && !F::isTombstone(bucket_ + i)) {
          (bucket_ + i)->~T();
        }
      }
    }
    il::deallocate(bucket_);
    const il::int_t m = nbBuckets(p);
    bucket_ = il::allocateArray<T>(m);
    for (il::int_t i = 0; i < m; ++i) {
      if (F::isEmpty(set.bucket_[i])) {
        F::constructEmpty(il::io, bucket_ + i);
      } else if (F::isTombstone(set.bucket_[i])) {
        F::constructTombstone(il::io, bucket_ + i);
      } else {
        new (bucket_ + i) T(set.bucket_[i]);
      }
    }
    p_ = p;
  } else {
    bucket_ = nullptr;
    p_ = -1;
  }
  nb_elements_ = set.nb_elements_;
  nb_tombstones_ = set.nb_tombstones_;
  return *this;
}

template <typename T, typename F>
Set<T, F>& Set<T, F>::operator=(Set<T, F>&& set) {
  if (this != &set) {
    const int old_p = p_;
    if (old_p >= 0) {
      const il::int_t old_m = nbBuckets(old_p);
      for (il::int_t i = 0; i < old_m; ++i) {
        if (!F::is_empy(bucket_ + i) && !F::isTombstone(bucket_ + i)) {
          (&((bucket_ + i)->value))->~V();
          (&((bucket_ + i)->key))->~K();
        }
      }
      il::deallocate(bucket_);
    }
    bucket_ = set.bucket_;
    p_ = set.p_;
    nb_elements_ = set.nb_elements_;
    nb_tombstones_ = set.nb_tombstones_;
    set.bucket_ = nullptr;
    set.p_ = -1;
    set.nb_elements_ = 0;
    set.nb_tombstones_ = 0;
  }
  return *this;
}

template <typename T, typename F>
Set<T, F>::~Set() {
  if (p_ >= 0) {
    const il::int_t m = nbBuckets();
    for (il::int_t i = 0; i < m; ++i) {
      if (!F::isEmpty(bucket_[i]) && !F::isTombstone(bucket_[i])) {
        (bucket_ + i)->~T();
      }
    }
    il::deallocate(bucket_);
  }
}

template <typename T, typename F>
il::int_t Set<T, F>::search(const T& x) const {
  IL_EXPECT_MEDIUM(!F::isEmpty(x));
  IL_EXPECT_MEDIUM(!F::isTombstone(x));

  if (p_ == -1) {
    return -1;
  }

  const std::size_t mask = (static_cast<std::size_t>(1) << p_) - 1;
  std::size_t i = F::hash(x, p_);
  std::size_t i_tombstone = -1;
  std::size_t delta_i = 1;
  while (true) {
    if (F::isEmpty(bucket_[i])) {
      return (i_tombstone == static_cast<std::size_t>(-1))
                 ? -(1 + static_cast<il::int_t>(i))
                 : -(1 + static_cast<il::int_t>(i_tombstone));
    } else if (F::isTombstone(bucket_[i])) {
      i_tombstone = i;
    } else if (F::isEqual(bucket_[i], x)) {
      return static_cast<il::int_t>(i);
    }
    i += delta_i;
    i &= mask;
    ++delta_i;
  }
}

template <typename T, typename F>
void Set<T, F>::insert(const T& x, il::io_t, il::int_t i) {
  IL_EXPECT_FAST(!found(i));

  il::int_t i_local = -(1 + i);
  const il::int_t m = nbBuckets();
  if (4 * (static_cast<std::size_t>(nb_elements_) + 1) >
      3 * static_cast<std::size_t>(m)) {
    if (p_ + 1 <= static_cast<int>(sizeof(std::size_t) * 8 - 2)) {
      reserveWithP(p_ == -1 ? 1 : p_ + 1);
    } else {
      il::abort();
    }
    il::int_t j = search(x);
    i_local = -(1 + j);
  }
  new (bucket_ + i_local) T(x);
  ++nb_elements_;
  i = i_local;
}

template <typename T, typename F>
void Set<T, F>::insert(const T& x) {
  il::int_t i = search(x);
  if (!found(i)) {
    insert(x, il::io, i);
  }
}

template <typename T, typename F>
bool Set<T, F>::found(il::int_t i) const {
  return i >= 0;
}

template <typename T, typename F>
bool Set<T, F>::contains(const T& x) const {
  const il::int_t i = search(x);
  return i >= 0;
}

template <typename T, typename F>
void Set<T, F>::clear() {
  if (p_ >= 0) {
    const il::int_t m = nbBuckets();
    for (il::int_t i = 0; i < m; ++i) {
      if (!F::isEmpty(bucket_[i]) && !F::isTombstone(bucket_[i])) {
        (bucket_ + i)->~T();
      }
    }
  }
  nb_elements_ = 0;
}

template <typename T, typename F>
const T& Set<T, F>::operator[](il::int_t i) const {
  return bucket_[i];
}

template <typename T, typename F>
il::int_t Set<T, F>::first() const {
  const il::int_t m = static_cast<il::int_t>(static_cast<std::size_t>(1) << p_);

  il::int_t i = 0;
  while (i < m && (F::isEmpty(bucket_[i]) || F::isTombstone(bucket_[i]))) {
    ++i;
  }
  return i;
}

template <typename T, typename F>
il::int_t Set<T, F>::sentinel() const {
  return static_cast<il::int_t>(static_cast<std::size_t>(1) << p_);
  ;
}

template <typename T, typename F>
il::int_t Set<T, F>::next(il::int_t i) const {
  const il::int_t m = static_cast<il::int_t>(static_cast<std::size_t>(1) << p_);

  ++i;
  while (i < m && (F::isEmpty(bucket_[i]) || F::isTombstone(bucket_[i]))) {
    ++i;
  }
  return i;
}

template <typename T, typename F>
il::int_t Set<T, F>::nbElements() const {
  return nb_elements_;
}

template <typename T, typename F>
il::int_t Set<T, F>::nbBuckets() const {
  return (p_ >= 0) ? static_cast<il::int_t>(static_cast<std::size_t>(1) << p_)
                   : 0;
}

template <typename T, typename F>
il::int_t Set<T, F>::nbBuckets(int p) const {
  return (p >= 0) ? static_cast<il::int_t>(static_cast<std::size_t>(1) << p)
                   : 0;
}

template <typename T, typename F>
void Set<T, F>::reserveWithP(int p) {
  T* old_bucket_ = bucket_;
  const il::int_t old_m = nbBuckets(p_);
  const il::int_t m = nbBuckets(p);

  bucket_ = il::allocateArray<T>(m);
  for (il::int_t i = 0; i < m; ++i) {
    F::constructEmpty(il::io, bucket_ + i);
  }
  p_ = p;
  nb_elements_ = 0;
  nb_tombstones_ = 0;

  if (p_ >= 0) {
    for (il::int_t i = 0; i < old_m; ++i) {
      if (!F::isEmpty(old_bucket_[i]) && !F::isTombstone(old_bucket_[i])) {
        il::int_t new_i = search(old_bucket_[i]);
        insert(std::move(old_bucket_[i]), il::io, new_i);
        (old_bucket_ + i)->~T();
      }
    }
    il::deallocate(old_bucket_);
  }
}

}  // namespace il

#endif  // IL_SET_H
