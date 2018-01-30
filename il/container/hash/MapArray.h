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

#ifndef IL_MAPARRAY_H
#define IL_MAPARRAY_H

#include <il/Array.h>
#include <il/Map.h>

namespace il {

template <typename K, typename V, typename F = HashFunction<K>>
class MapArray {
 private:
  il::Array<il::KeyValue<K, V>> array_;
  il::Map<K, il::int_t, F> map_;
#ifdef IL_DEBUG_CLASS
  std::size_t hash_;
#endif

 public:
  MapArray();
  MapArray(il::int_t n);
  MapArray(il::value_t, std::initializer_list<il::KeyValue<K, V>> list);
  void set(const K& key, const V& value);
  void set(const K& key, V&& value);
  void set(K&& key, const V& value);
  void set(K&& key, V&& value);
  void set(const K& key, const V& value, il::io_t, il::Location& i);
  void set(const K& key, V&& value, il::io_t, il::Location& i);
  void set(K&& key, const V& value, il::io_t, il::Location& i);
  void set(K&& key, V&& value, il::io_t, il::Location& i);
  il::int_t size() const;
  il::int_t capacity() const;
  il::Location search(const K& key) const;
  template <il::int_t m>
  il::Location searchCString(const char (&key)[m]) const;
  bool found(il::Location i) const;
  const K& key(il::Location i) const;
  const V& value(il::Location i) const;
  V& value(il::Location i);
  il::Location next(il::Location i) const;
  il::Location first() const;
  il::Location sentinel() const;
  il::int_t nbElements() const;
};

template <typename K, typename V, typename F>
MapArray<K, V, F>::MapArray() : array_{}, map_{} {
#ifdef IL_DEBUG_CLASS
  hash_ = 0;
#endif
}

template <typename K, typename V, typename F>
MapArray<K, V, F>::MapArray(il::int_t n) : array_{}, map_{n} {
  array_.reserve(n);
#ifdef IL_DEBUG_CLASS
  hash_ = 0;
#endif
}

template <typename K, typename V, typename F>
MapArray<K, V, F>::MapArray(il::value_t,
                            std::initializer_list<il::KeyValue<K, V>> list) {
  const il::int_t n = static_cast<il::int_t>(list.size());
  if (n > 0) {
    array_.reserve(n);
    map_.reserve(n);
    for (il::int_t i = 0; i < n; ++i) {
      array_.append(il::emplace, (list.begin() + i)->key,
                    (list.begin() + i)->value);
      map_.set((list.begin() + i)->key, i);
    }
  }
#ifdef IL_DEBUG_CLASS
  hash_ = static_cast<std::size_t>(n);
#endif
};

template <typename K, typename V, typename F>
void MapArray<K, V, F>::set(const K& key, const V& value) {
  const il::int_t i = array_.size();
  array_.append(il::emplace, key, value);
  map_.set(key, i);
#ifdef IL_DEBUG_CLASS
  hash_ = F::hash(key, map_.hash());
#endif
}

template <typename K, typename V, typename F>
void MapArray<K, V, F>::set(const K& key, V&& value) {
  const il::int_t i = array_.size();
  array_.append(il::emplace, key, std::move(value));
  map_.set(key, i);
#ifdef IL_DEBUG_CLASS
  hash_ = F::hash(key, map_.hash());
#endif
}

template <typename K, typename V, typename F>
void MapArray<K, V, F>::set(K&& key, V&& value) {
  const il::int_t i = array_.size();
  array_.append(il::emplace, key, std::move(value));
  map_.set(std::move(key), i);
#ifdef IL_DEBUG_CLASS
  hash_ = F::hash(key, map_.hash());
#endif
}

template <typename K, typename V, typename F>
void MapArray<K, V, F>::set(const K& key, const V& value, il::io_t,
                            il::Location& i) {
  const il::int_t j = array_.size();
  array_.append(il::emplace, key, value);
  map_.set(key, j, il::io, i);

#ifdef IL_DEBUG_CLASS
  hash_ = F::hash(key, map_.hash());
  i.setIndex(j, hash_);
#else
  i.setIndex(j);
#endif
}

template <typename K, typename V, typename F>
void MapArray<K, V, F>::set(const K& key, V&& value, il::io_t,
                            il::Location& i) {
  const il::int_t j = array_.size();
  array_.append(il::emplace, key, std::move(value));
  map_.set(key, j, il::io, i);
#ifdef IL_DEBUG_CLASS
  hash_ = F::hash(key, map_.hash());
  i.setIndex(j, hash_);
#else
  i.setIndex(j);
#endif
}

template <typename K, typename V, typename F>
void MapArray<K, V, F>::set(K&& key, V&& value, il::io_t, il::Location& i) {
  const il::int_t j = array_.size();
  array_.append(il::emplace, key, std::move(value));
  map_.set(std::move(key), j, il::io, i);
#ifdef IL_DEBUG_CLASS
  hash_ = F::hash(key, map_.hash());
  i.setIndex(j, hash_);
#else
  i.setIndex(j);
#endif
}

template <typename K, typename V, typename F>
il::int_t MapArray<K, V, F>::size() const {
  return array_.size();
}

template <typename K, typename V, typename F>
il::int_t MapArray<K, V, F>::capacity() const {
  return array_.capacity();
}

template <typename K, typename V, typename F>
il::Location MapArray<K, V, F>::search(const K& key) const {
  const il::Location i = map_.search(key);
#ifdef IL_DEBUG_CLASS
  return map_.found(i) ? il::Location{map_.value(i), map_.hash()}
                       : il::Location{i.index(), map_.hash()};
#else
  return map_.found(i) ? il::Location{map_.value(i)} : il::Location{i.index()};
#endif
}

template <typename K, typename V, typename F>
template <il::int_t m>
il::Location MapArray<K, V, F>::searchCString(const char (&key)[m]) const {
  const il::Location i = map_.search(key);
#ifdef IL_DEBUG_CLASS
  return map_.found(i) ? il::Location{map_.value(i), map_.hash()}
                       : il::Location{i.index(), map_.hash()};
#else
  return map_.found(i) ? il::Location{map_.value(i)} : il::Location{i.index()};
#endif
};

template <typename K, typename V, typename F>
bool MapArray<K, V, F>::found(il::Location i) const {
  return i.index() >= 0;
}

template <typename K, typename V, typename F>
const K& MapArray<K, V, F>::key(il::Location i) const {
  return array_[i.index()].key;
}

template <typename K, typename V, typename F>
const V& MapArray<K, V, F>::value(il::Location i) const {
  return array_[i.index()].value;
}

template <typename K, typename V, typename F>
V& MapArray<K, V, F>::value(il::Location i) {
  return array_[i.index()].value;
}

template <typename K, typename V, typename F>
il::Location MapArray<K, V, F>::next(il::Location i) const {
#ifdef IL_DEBUG_CLASS
  return il::Location{i.index() + 1, map_.hash()};
#else
  return il::Location{i.index() + 1};
#endif
}

template <typename K, typename V, typename F>
il::Location MapArray<K, V, F>::first() const {
#ifdef IL_DEBUG_CLASS
  return il::Location{0, hash_};
#else
  return il::Location{0};
#endif
}

template <typename K, typename V, typename F>
il::Location MapArray<K, V, F>::sentinel() const {
#ifdef IL_DEBUG_CLASS
  return il::Location{array_.size(), map_.hash()};
#else
  return il::Location{array_.size()};
#endif
}

template <typename K, typename V, typename F>
il::int_t MapArray<K, V, F>::nbElements() const {
  return array_.size();
};

}  // namespace il

#endif  // IL_MAPARRAY_H
