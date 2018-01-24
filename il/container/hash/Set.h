//==============================================================================
//
// Copyright 2017 The InsideLoop Authors. All Rights Reserved.
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

  il::Location search(const T& x) const;
  void add(const T& x, il::io_t, il::Location i);
  void add(const T& x);
  bool found(il::Location i) const;
  bool contains(const T& x) const;
  void clear();
  const T& operator[](il::int_t i) const;
  il::Location first() const;
  il::Location sentinel() const;
  il::Location next(il::Location i) const;
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
      add(set[i]);
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
il::Location Set<T, F>::search(const T& x) const {
  IL_EXPECT_MEDIUM(!F::isEmpty(x));
  IL_EXPECT_MEDIUM(!F::isTombstone(x));

  if (p_ == -1) {
    return il::Location{-1};
  }

  const std::size_t mask = (static_cast<std::size_t>(1) << p_) - 1;
  std::size_t i = F::hash(x, p_);
  std::size_t i_tombstone = -1;
  std::size_t delta_i = 1;
  while (true) {
    if (F::isEmpty(bucket_[i])) {
      return il::Location{(i_tombstone == static_cast<std::size_t>(-1))
                              ? -(1 + static_cast<il::int_t>(i))
                              : -(1 + static_cast<il::int_t>(i_tombstone))};
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
void Set<T, F>::add(const T& x, il::io_t, il::Location& i) {
  IL_EXPECT_FAST(!found(i));

  il::int_t i_local = -(1 + i.index());
  const il::int_t m = nbBuckets();
  if (4 * (static_cast<std::size_t>(nb_elements_) + 1) >
      3 * static_cast<std::size_t>(m)) {
    if (p_ + 1 <= static_cast<int>(sizeof(std::size_t) * 8 - 2)) {
      reserveWithP(p_ == -1 ? 1 : p_ + 1);
    } else {
      il::abort();
    }
    il::Location j = search(x);
    i_local = -(1 + j.index());
  }
  new (bucket_ + i_local) T(x);
  ++nb_elements_;
  i.setIndex(i_local);
}

template <typename T, typename F>
void Set<T, F>::add(const T& x) {
  il::Location i = search(x);
  if (!found(i)) {
    add(x, il::io, i);
  }
}

template <typename T, typename F>
bool Set<T, F>::found(il::Location i) const {
  return i.index() >= 0;
}

template <typename T, typename F>
bool Set<T, F>::contains(const T& x) const {
  const il::Location i = search(x);
  return i.index() >= 0;
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
const T& Set<T, F>::operator[](il::Location i) const {
  return bucket_[i.index()];
}

template <typename T, typename F>
il::Location Set<T, F>::first() const {
  const il::int_t m = static_cast<il::int_t>(static_cast<std::size_t>(1) << p_);

  il::int_t i = 0;
  while (i < m && (F::isEmpty(bucket_[i]) || F::isTombstone(bucket_[i]))) {
    ++i;
  }
  return il::Location{i};
}

template <typename T, typename F>
il::Location Set<T, F>::sentinel() const {
  return il::Location{
      static_cast<il::int_t>(static_cast<std::size_t>(1) << p_)};
}

template <typename T, typename F>
il::Location Set<T, F>::next(il::Location i) const {
  const il::int_t m = static_cast<il::int_t>(static_cast<std::size_t>(1) << p_);

  il::int_t i_local = i.index();
  ++i_local;
  while (i_local < m &&
         (F::isEmpty(bucket_[i_local]) || F::isTombstone(bucket_[i_local]))) {
    ++i_local;
  }
  return i_local;
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
        il::Location new_i = search(old_bucket_[i]);
        add(std::move(old_bucket_[i]), il::io, new_i);
        (old_bucket_ + i)->~T();
      }
    }
    il::deallocate(old_bucket_);
  }
}

}  // namespace il

#endif  // IL_SET_H
