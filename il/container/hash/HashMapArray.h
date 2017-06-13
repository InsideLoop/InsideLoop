//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_HASHMAPARRAY_H
#define IL_HASHMAPARRAY_H

#include <il/Array.h>
#include <il/HashMap.h>

namespace il {

template <typename K, typename V, typename F = HashFunction<K>>
class HashMapArray {
 private:
  il::Array<il::KeyValue<K, V>> array_;
  il::HashMap<K, il::int_t, F> map_;

 public:
  HashMapArray();
  HashMapArray(il::int_t n);
  void set(const K& key, const V& value);
  void set(const K& key, V&& value);
  void set(K&& key, V&& value);
  void insert(const K& key, const V& value);
  void insert(const K& key, V&& value);
  void insert(const K& key, const V& value, il::io_t, il::int_t& i);
  void insert(const K& key, V&& value, il::io_t, il::int_t& i);
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
  il::int_t broom() const;
};

template <typename K, typename V, typename F>
HashMapArray<K, V, F>::HashMapArray() : array_{}, map_{} {}

template <typename K, typename V, typename F>
HashMapArray<K, V, F>::HashMapArray(il::int_t n) : array_{}, map_{n} {
  array_.reserve(n);
}

template <typename K, typename V, typename F>
void HashMapArray<K, V, F>::set(const K& key, const V& value) {
  const il::int_t i = array_.size();
  array_.append(il::emplace, key, value);
  map_.set(key, i);
}

template <typename K, typename V, typename F>
void HashMapArray<K, V, F>::set(const K& key, V&& value) {
  const il::int_t i = array_.size();
  //  array_.append(il::KeyValue<K, V>{key, std::move(value)});
  array_.append(il::emplace, key, std::move(value));
  map_.set(key, i);
}

template <typename K, typename V, typename F>
void HashMapArray<K, V, F>::set(K&& key, V&& value) {
  const il::int_t i = array_.size();
  array_.append(il::emplace, key, std::move(value));
  map_.set(std::move(key), i);
}

template <typename K, typename V, typename F>
void HashMapArray<K, V, F>::insert(const K& key, const V& value) {
  const il::int_t i = array_.size();
  array_.append(il::emplace, key, value);
  map_.insert(key, i);
}

template <typename K, typename V, typename F>
void HashMapArray<K, V, F>::insert(const K& key, V&& value) {
  const il::int_t i = array_.size();
  array_.append(il::emplace, key, value);
  map_.insert(key, i);
}

template <typename K, typename V, typename F>
void HashMapArray<K, V, F>::insert(const K& key, const V& value, il::io_t,
                                   il::int_t& i) {
  const il::int_t j = array_.size();
  array_.append(il::emplace, key, value);
  map_.insert(key, j, il::io, i);
  i = j;
}

template <typename K, typename V, typename F>
void HashMapArray<K, V, F>::insert(const K& key, V&& value, il::io_t,
                                   il::int_t& i) {
  const il::int_t j = array_.size();
  array_.append(il::emplace, key, std::move(value));
  map_.insert(key, j, il::io, i);
  i = j;
}

template <typename K, typename V, typename F>
void HashMapArray<K, V, F>::insert(K&& key, V&& value, il::io_t, il::int_t& i) {
  const il::int_t j = array_.size();
  array_.append(il::emplace, key, std::move(value));
  map_.insert(std::move(key), j, il::io, i);
  i = j;
}

template <typename K, typename V, typename F>
il::int_t HashMapArray<K, V, F>::size() const {
  return array_.size();
}

template <typename K, typename V, typename F>
il::int_t HashMapArray<K, V, F>::capacity() const {
  return array_.capacity();
}

template <typename K, typename V, typename F>
il::int_t HashMapArray<K, V, F>::search(const K& key) const {
  const il::int_t i = map_.search(key);
  return map_.found(i) ? map_.value(i) : i;
}

template <typename K, typename V, typename F>
bool HashMapArray<K, V, F>::found(il::int_t i) const {
  return i >= 0;
}

template <typename K, typename V, typename F>
const K& HashMapArray<K, V, F>::key(il::int_t i) const {
  return array_[i].key;
}

template <typename K, typename V, typename F>
const V& HashMapArray<K, V, F>::value(il::int_t i) const {
  return array_[i].value;
}

template <typename K, typename V, typename F>
V& HashMapArray<K, V, F>::value(il::int_t i) {
  return array_[i].value;
}

template <typename K, typename V, typename F>
il::int_t HashMapArray<K, V, F>::next(il::int_t i) const {
  return i + 1;
}

template <typename K, typename V, typename F>
il::int_t HashMapArray<K, V, F>::first() const {
  return 0;
}

template <typename K, typename V, typename F>
il::int_t HashMapArray<K, V, F>::broom() const {
  return array_.size();
}

}  // namespace il

#endif  // IL_HASHMAPARRAY_H
