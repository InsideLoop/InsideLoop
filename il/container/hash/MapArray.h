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

 public:
  MapArray();
  MapArray(il::int_t n);
  void set(const K& key, const V& value);
  void set(const K& key, V&& value);
  void set(K&& key, const V& value);
  void set(K&& key, V&& value);
  void insert(const K& key, const V& value, il::io_t, il::int_t& i);
  void insert(const K& key, V&& value, il::io_t, il::int_t& i);
  void insert(K&& key, const V& value, il::io_t, il::int_t& i);
  void insert(K&& key, V&& value, il::io_t, il::int_t& i);
  il::int_t size() const;
  il::int_t capacity() const;
  il::int_t search(const K& key) const;
  bool found(il::int_t i) const;
  const K& key(il::int_t i) const;
  const V& value(il::int_t i) const;
  V& value(il::int_t i);
  il::int_t next(il::int_t i) const;
  il::int_t first() const;
  il::int_t sentinel() const;
};

template <typename K, typename V, typename F>
MapArray<K, V, F>::MapArray() : array_{}, map_{} {}

template <typename K, typename V, typename F>
MapArray<K, V, F>::MapArray(il::int_t n) : array_{}, map_{n} {
  array_.reserve(n);
}

template <typename K, typename V, typename F>
void MapArray<K, V, F>::set(const K& key, const V& value) {
  const il::int_t i = array_.size();
  array_.append(il::emplace, key, value);
  map_.insert(key, i);
}

template <typename K, typename V, typename F>
void MapArray<K, V, F>::set(const K& key, V&& value) {
  const il::int_t i = array_.size();
  //  array_.append(il::KeyValue<K, V>{key, std::move(value)});
  array_.append(il::emplace, key, std::move(value));
  map_.insert(key, i);
}

template <typename K, typename V, typename F>
void MapArray<K, V, F>::set(K&& key, V&& value) {
  const il::int_t i = array_.size();
  array_.append(il::emplace, key, std::move(value));
  map_.insert(std::move(key), i);
}

template <typename K, typename V, typename F>
void MapArray<K, V, F>::insert(const K& key, const V& value, il::io_t,
                               il::int_t& i) {
  const il::int_t j = array_.size();
  array_.append(il::emplace, key, value);
  map_.insert(key, j, il::io, i);
  i = j;
}

template <typename K, typename V, typename F>
void MapArray<K, V, F>::insert(const K& key, V&& value, il::io_t,
                               il::int_t& i) {
  const il::int_t j = array_.size();
  array_.append(il::emplace, key, std::move(value));
  map_.insert(key, j, il::io, i);
  i = j;
}

template <typename K, typename V, typename F>
void MapArray<K, V, F>::insert(K&& key, V&& value, il::io_t, il::int_t& i) {
  const il::int_t j = array_.size();
  array_.append(il::emplace, key, std::move(value));
  map_.insert(std::move(key), j, il::io, i);
  i = j;
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
il::int_t MapArray<K, V, F>::search(const K& key) const {
  const il::int_t i = map_.search(key);
  return map_.found(i) ? map_.value(i) : i;
}

template <typename K, typename V, typename F>
bool MapArray<K, V, F>::found(il::int_t i) const {
  return i >= 0;
}

template <typename K, typename V, typename F>
const K& MapArray<K, V, F>::key(il::int_t i) const {
  return array_[i].key;
}

template <typename K, typename V, typename F>
const V& MapArray<K, V, F>::value(il::int_t i) const {
  return array_[i].value;
}

template <typename K, typename V, typename F>
V& MapArray<K, V, F>::value(il::int_t i) {
  return array_[i].value;
}

template <typename K, typename V, typename F>
il::int_t MapArray<K, V, F>::next(il::int_t i) const {
  return i + 1;
}

template <typename K, typename V, typename F>
il::int_t MapArray<K, V, F>::first() const {
  return 0;
}

template <typename K, typename V, typename F>
il::int_t MapArray<K, V, F>::sentinel() const {
  return array_.size();
}

}  // namespace il

#endif  // IL_MAPARRAY_H
