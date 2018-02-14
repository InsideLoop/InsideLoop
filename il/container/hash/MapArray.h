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
  void Set(const K& key, const V& value);
  void Set(const K& key, V&& value);
  void Set(K&& key, const V& value);
  void Set(K&& key, V&& value);
  void Set(const K& key, const V& value, il::io_t, il::Spot& i);
  void Set(const K& key, V&& value, il::io_t, il::Spot& i);
  void Set(K&& key, const V& value, il::io_t, il::Spot& i);
  void Set(K&& key, V&& value, il::io_t, il::Spot& i);
  il::int_t size() const;
  il::int_t capacity() const;
  il::Spot search(const K& key) const;
  template <il::int_t m>
  il::Spot searchCString(const char (&key)[m]) const;
  bool found(il::Spot i) const;
  const K& key(il::Spot i) const;
  const V& value(il::Spot i) const;
  V& Value(il::Spot i);
  il::Spot next(il::Spot i) const;
  il::Spot spotBegin() const;
  il::Spot spotEnd() const;
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
  array_.Reserve(n);
#ifdef IL_DEBUG_CLASS
  hash_ = 0;
#endif
}

template <typename K, typename V, typename F>
MapArray<K, V, F>::MapArray(il::value_t,
                            std::initializer_list<il::KeyValue<K, V>> list) {
  const il::int_t n = static_cast<il::int_t>(list.size());
  if (n > 0) {
    array_.Reserve(n);
    map_.Reserve(n);
    for (il::int_t i = 0; i < n; ++i) {
      array_.Append(il::emplace, (list.begin() + i)->key,
                    (list.begin() + i)->value);
      map_.Set((list.begin() + i)->key, i);
    }
  }
#ifdef IL_DEBUG_CLASS
  hash_ = static_cast<std::size_t>(n);
#endif
};

template <typename K, typename V, typename F>
void MapArray<K, V, F>::Set(const K& key, const V& value) {
  const il::int_t i = array_.size();
  array_.Append(il::emplace, key, value);
  map_.Set(key, i);
#ifdef IL_DEBUG_CLASS
  hash_ = F::hash(key, map_.hash());
#endif
}

template <typename K, typename V, typename F>
void MapArray<K, V, F>::Set(const K& key, V&& value) {
  const il::int_t i = array_.size();
  array_.Append(il::emplace, key, std::move(value));
  map_.Set(key, i);
#ifdef IL_DEBUG_CLASS
  hash_ = F::hash(key, map_.hash());
#endif
}

template <typename K, typename V, typename F>
void MapArray<K, V, F>::Set(K&& key, V&& value) {
  const il::int_t i = array_.size();
  array_.Append(il::emplace, key, std::move(value));
  map_.Set(std::move(key), i);
#ifdef IL_DEBUG_CLASS
  hash_ = F::hash(key, map_.hash());
#endif
}

template <typename K, typename V, typename F>
void MapArray<K, V, F>::Set(const K& key, const V& value, il::io_t,
                            il::Spot& i) {
  const il::int_t j = array_.size();
  array_.Append(il::emplace, key, value);
  map_.Set(key, j, il::io, i);

#ifdef IL_DEBUG_CLASS
  hash_ = F::hash(key, map_.hash());
  i.SetIndex(j, hash_);
#else
  i.SetIndex(j);
#endif
}

template <typename K, typename V, typename F>
void MapArray<K, V, F>::Set(const K& key, V&& value, il::io_t, il::Spot& i) {
  const il::int_t j = array_.size();
  array_.Append(il::emplace, key, std::move(value));
  map_.Set(key, j, il::io, i);
#ifdef IL_DEBUG_CLASS
  hash_ = F::hash(key, map_.hash());
  i.SetIndex(j, hash_);
#else
  i.SetIndex(j);
#endif
}

template <typename K, typename V, typename F>
void MapArray<K, V, F>::Set(K&& key, V&& value, il::io_t, il::Spot& i) {
  const il::int_t j = array_.size();
  array_.Append(il::emplace, key, std::move(value));
  map_.Set(std::move(key), j, il::io, i);
#ifdef IL_DEBUG_CLASS
  hash_ = F::hash(key, map_.hash());
  i.SetIndex(j, hash_);
#else
  i.SetIndex(j);
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
il::Spot MapArray<K, V, F>::search(const K& key) const {
  const il::Spot i = map_.search(key);
#ifdef IL_DEBUG_CLASS
  return map_.found(i) ? il::Spot{map_.value(i), map_.hash()}
                       : il::Spot{i.index(), map_.hash()};
#else
  return map_.found(i) ? il::Spot{map_.value(i)} : il::Spot{i.index()};
#endif
}

template <typename K, typename V, typename F>
template <il::int_t m>
il::Spot MapArray<K, V, F>::searchCString(const char (&key)[m]) const {
  const il::Spot i = map_.search(key);
#ifdef IL_DEBUG_CLASS
  return map_.found(i) ? il::Spot{map_.value(i), map_.hash()}
                       : il::Spot{i.index(), map_.hash()};
#else
  return map_.found(i) ? il::Spot{map_.value(i)} : il::Spot{i.index()};
#endif
};

template <typename K, typename V, typename F>
bool MapArray<K, V, F>::found(il::Spot i) const {
  return i.index() >= 0;
}

template <typename K, typename V, typename F>
const K& MapArray<K, V, F>::key(il::Spot i) const {
  return array_[i.index()].key;
}

template <typename K, typename V, typename F>
const V& MapArray<K, V, F>::value(il::Spot i) const {
  return array_[i.index()].value;
}

template <typename K, typename V, typename F>
V& MapArray<K, V, F>::Value(il::Spot i) {
  return array_[i.index()].value;
}

template <typename K, typename V, typename F>
il::Spot MapArray<K, V, F>::next(il::Spot i) const {
#ifdef IL_DEBUG_CLASS
  return il::Spot{i.index() + 1, map_.hash()};
#else
  return il::Spot{i.index() + 1};
#endif
}

template <typename K, typename V, typename F>
il::Spot MapArray<K, V, F>::spotBegin() const {
#ifdef IL_DEBUG_CLASS
  return il::Spot{0, hash_};
#else
  return il::Spot{0};
#endif
}

template <typename K, typename V, typename F>
il::Spot MapArray<K, V, F>::spotEnd() const {
#ifdef IL_DEBUG_CLASS
  return il::Spot{array_.size(), map_.hash()};
#else
  return il::Spot{array_.size()};
#endif
}

template <typename K, typename V, typename F>
il::int_t MapArray<K, V, F>::nbElements() const {
  return array_.size();
};

}  // namespace il

#endif  // IL_MAPARRAY_H
