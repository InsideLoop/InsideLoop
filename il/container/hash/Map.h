//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_MAP_H
#define IL_MAP_H

#include <il/Array.h>
#include <il/Status.h>
#include <il/container/hash/HashFunction.h>
#include <il/math.h>

namespace il {

template <typename K, typename V>
struct KeyValue {
  K key;
  V value;
  KeyValue(const K& the_key, const V& the_value)
      : key{the_key}, value{the_value} {};
  KeyValue(const K& the_key, V&& the_value)
      : key{the_key}, value{std::move(the_value)} {};
  KeyValue(K&& the_key, const V& the_value)
      : key{std::move(the_key)}, value{the_value} {};
  KeyValue(K&& the_key, V&& the_value)
      : key{std::move(the_key)}, value{std::move(the_value)} {};
};

template <typename K, typename V, typename F>
class MapIterator {
 private:
  KeyValue<K, V>* pointer_;
  KeyValue<K, V>* end_;

 public:
  MapIterator(KeyValue<K, V>* pointer, KeyValue<K, V>* end);
  KeyValue<K, V>& operator*() const;
  KeyValue<K, V>* operator->() const;
  bool operator==(const MapIterator& it) const;
  bool operator!=(const MapIterator& it) const;
  MapIterator& operator++();
  MapIterator& operator++(int);

 private:
  void advancePastEmptyBuckets();
};

template <typename K, typename V, typename F>
MapIterator<K, V, F>::MapIterator(KeyValue<K, V>* pointer,
                                  KeyValue<K, V>* end) {
  pointer_ = pointer;
  end_ = end;
}

template <typename K, typename V, typename F>
KeyValue<K, V>& MapIterator<K, V, F>::operator*() const {
  return *pointer_;
}

template <typename K, typename V, typename F>
KeyValue<K, V>* MapIterator<K, V, F>::operator->() const {
  return pointer_;
}

template <typename K, typename V, typename F>
bool MapIterator<K, V, F>::operator==(const MapIterator& it) const {
  return pointer_ == it.pointer_;
}

template <typename K, typename V, typename F>
bool MapIterator<K, V, F>::operator!=(const MapIterator& it) const {
  return pointer_ != it.pointer_;
}

template <typename K, typename V, typename F>
MapIterator<K, V, F>& MapIterator<K, V, F>::operator++() {
  ++pointer_;
  advancePastEmptyBuckets();
  return *this;
}

template <typename K, typename V, typename F>
MapIterator<K, V, F>& MapIterator<K, V, F>::operator++(int) {
  MapIterator tmp = *this;
  ++pointer_;
  advancePastEmptyBuckets();
  return tmp;
}

template <typename K, typename V, typename F>
void MapIterator<K, V, F>::advancePastEmptyBuckets() {
  const K empty_key = F::empty_key();
  const K tombstone_key = F::tombstone_key();
  while (pointer_ != end_ && (F::isEqual(pointer_->key, empty_key) ||
                              F::isEqual(pointer_->key, tombstone_key))) {
    ++pointer_;
  }
}

template <typename K, typename V, typename F = HashFunction<K>>
class Map {
 private:
  KeyValue<K, V>* slot_;
  // The number of slot is equal to 2^p_ if p_ >= 0 and 0 if p_ = -1;
  int p_;
  il::int_t nb_element_;
  il::int_t nb_tombstone_;

 public:
  Map();
  Map(il::int_t n);
  Map(il::value_t, std::initializer_list<il::KeyValue<K, V>> list);
  Map(const Map<K, V, F>& map);
  Map(Map<K, V, F>&& map);
  Map& operator=(const Map<K, V, F>& map);
  Map& operator=(Map<K, V, F>&& map);
  ~Map();
  void set(const K& key, const V& value, il::io_t, il::int_t& i);
  void set(const K& key, V&& value, il::io_t, il::int_t& i);
  void set(K&& key, V&& value, il::io_t, il::int_t& i);
  void set(const K& key, const V& value);
  void set(K&& key, const V& value);
  void set(const K& key, V&& value);
  void set(K&& key, V&& value);
  il::int_t search(const K& key) const;
  bool hasFound(il::int_t i) const;
  bool notFound(il::int_t i) const;
  void insert(const K& key, const V& value, il::io_t, il::int_t& i);
  void insert(const K& key, V&& value, il::io_t, il::int_t& i);
  void insert(K&& key, const V& value, il::io_t, il::int_t& i);
  void insert(K&& key, V&& value, il::io_t, il::int_t& i);
  void insert(const K& key, const V& value);
  void insert(const K& key, V&& value);
  void erase(il::int_t i);
  const K& key(il::int_t i) const;
  const V& value(il::int_t i) const;
  V& value(il::int_t i);
  bool isEmpty() const;
  il::int_t size() const;
  il::int_t capacity() const;
  il::int_t first() const;
  il::int_t sentinel() const;
  il::int_t next(il::int_t i) const;
  void reserve(il::int_t r);
  MapIterator<K, V, F> begin();
  MapIterator<K, V, F> end();
  double load() const;
  double displaced() const;
  double displacedTwice() const;

 private:
  void grow(il::int_t r);
};

template <typename K, typename V, typename F>
Map<K, V, F>::Map() {
  slot_ = nullptr;
  p_ = -1;
  nb_element_ = 0;
  nb_tombstone_ = 0;
}

template <typename K, typename V, typename F>
Map<K, V, F>::Map(il::int_t n) {
  IL_EXPECT_FAST(n >= 0);

  // To be large
  n = 2 * n;

  if (n > 0) {
    const int p = 1 + il::next_log2_32(n);
    const il::int_t m = 1 << p;
    slot_ = static_cast<KeyValue<K, V>*>(
        ::operator new(m * sizeof(KeyValue<K, V>)));
    for (il::int_t i = 0; i < m; ++i) {
      F::constructEmpty(il::io, reinterpret_cast<K*>(slot_ + i));
    }
    p_ = p;
  } else {
    slot_ = nullptr;
    p_ = -1;
  }
  nb_element_ = 0;
  nb_tombstone_ = 0;
}

template <typename K, typename V, typename F>
Map<K, V, F>::Map(il::value_t, std::initializer_list<il::KeyValue<K, V>> list) {
  const il::int_t n = static_cast<il::int_t>(list.size());

  if (n > 0) {
    const int p = 1 + il::next_log2_32(n);
    const il::int_t m = 1 << p;
    slot_ = static_cast<KeyValue<K, V>*>(
        ::operator new(m * sizeof(KeyValue<K, V>)));
    for (il::int_t i = 0; i < m; ++i) {
      F::constructEmpty(il::io, reinterpret_cast<K*>(slot_ + i));
    }
    p_ = p;
  } else {
    slot_ = nullptr;
    p_ = -1;
  }
  nb_element_ = 0;
  nb_tombstone_ = 0;
  for (il::int_t k = 0; k < n; ++k) {
    il::int_t i = search((list.begin() + k)->key);
    IL_EXPECT_FAST(notFound(i));
    insert((list.begin() + k)->key, (list.begin() + k)->value, il::io, i);
  }
}

template <typename K, typename V, typename F>
Map<K, V, F>::Map(const Map<K, V, F>& map) {
  const il::int_t p = map.p_;
  if (p >= 0) {
    const il::int_t m = 1 << p;
    slot_ = static_cast<KeyValue<K, V>*>(
        ::operator new(m * sizeof(KeyValue<K, V>)));
    for (il::int_t i = 0; i < m; ++i) {
      if (F::isEmpty(map.slot_[i].key)) {
        F::constructEmpty(il::io, reinterpret_cast<K*>(slot_ + i));
      } else if (F::isTombstone(map.slot_[i].key)) {
        F::constructTombstone(il::io, reinterpret_cast<K*>(slot_ + i));
      } else {
        new (const_cast<K*>(&((slot_ + i)->key))) K(map.slot_[i].key);
        new (&((slot_ + i)->value)) V(map.slot_[i].value);
      }
    }
    p_ = p;
  } else {
    slot_ = nullptr;
    p_ = -1;
  }
  nb_element_ = map.nb_element_;
  nb_tombstone_ = map.nb_tombstone_;
}

template <typename K, typename V, typename F>
Map<K, V, F>::Map(Map<K, V, F>&& map) {
  slot_ = map.slot_;
  p_ = map.p_;
  nb_element_ = map.nb_element_;
  nb_tombstone_ = map.nb_tombstone_;
  map.slot_ = nullptr;
  map.p_ = -1;
  map.nb_element_ = 0;
  map.nb_tombstone_ = 0;
}

template <typename K, typename V, typename F>
Map<K, V, F>& Map<K, V, F>::operator=(const Map<K, V, F>& map) {
  const il::int_t p = map.p_;
  const il::int_t old_p = p_;
  if (p >= 0) {
    if (old_p >= 0) {
      const il::int_t old_m = 1 << old_p;
      for (il::int_t i = 0; i < old_m; ++i) {
        if (!F::is_empy(slot_ + i) && !F::isTombstone(slot_ + i)) {
          (&((slot_ + i)->value))->~V();
          (&((slot_ + i)->key))->~K();
        }
      }
    }
    ::operator delete(slot_);
    const il::int_t m = 1 << p;
    slot_ = static_cast<KeyValue<K, V>*>(
        ::operator new(m * sizeof(KeyValue<K, V>)));
    for (il::int_t i = 0; i < m; ++i) {
      if (F::isEmpty(map.slot_[i].key)) {
        F::constructEmpty(il::io, reinterpret_cast<K*>(slot_ + i));
      } else if (F::isTombstone(map.slot_[i].key)) {
        F::constructTombstone(il::io, reinterpret_cast<K*>(slot_ + i));
      } else {
        new (const_cast<K*>(&((slot_ + i)->key))) K(map.slot_[i].key);
        new (&((slot_ + i)->value)) V(map.slot_[i].value);
      }
    }
    p_ = p;
  } else {
    slot_ = nullptr;
    p_ = -1;
  }
  nb_element_ = map.nb_element_;
  nb_tombstone_ = map.nb_tombstone_;
}

template <typename K, typename V, typename F>
Map<K, V, F>& Map<K, V, F>::operator=(Map<K, V, F>&& map) {
  if (this != &map) {
    const il::int_t old_p = p_;
    if (old_p >= 0) {
      const il::int_t old_m = 1 << old_p;
      for (il::int_t i = 0; i < old_m; ++i) {
        if (!F::is_empy(slot_ + i) && !F::isTombstone(slot_ + i)) {
          (&((slot_ + i)->value))->~V();
          (&((slot_ + i)->key))->~K();
        }
      }
      ::operator delete(slot_);
    }
    slot_ = map.slot_;
    p_ = map.p_;
    nb_element_ = map.nb_element_;
    nb_tombstone_ = map.nb_tombstone_;
    map.slot_ = nullptr;
    map.p_ = -1;
    map.nb_element_ = 0;
    map.nb_tombstone_ = 0;
  }
}

template <typename K, typename V, typename F>
Map<K, V, F>::~Map() {
  if (p_ >= 0) {
    const il::int_t m = 1 << p_;
    for (il::int_t i = 0; i < m; ++i) {
      if (!F::isEmpty(slot_[i].key) && !F::isTombstone(slot_[i].key)) {
        (&((slot_ + i)->value))->~V();
        (&((slot_ + i)->key))->~K();
      }
    }
    ::operator delete(slot_);
  }
}

template <typename K, typename V, typename F>
void Map<K, V, F>::set(const K& key, const V& value, il::io_t, il::int_t& i) {
  i = search(key);
  if (notFound(i)) {
    insert(key, value, il::io, i);
  } else {
    (slot_ + i)->value.~V();
    new (&((slot_ + i)->value)) V(value);
  }
}

template <typename K, typename V, typename F>
void Map<K, V, F>::set(const K& key, V&& value, il::io_t, il::int_t& i) {
  i = search(key);
  if (notFound(i)) {
    insert(key, value, il::io, i);
  } else {
    (slot_ + i)->value.~V();
    new (&((slot_ + i)->value)) V(value);
  }
}

template <typename K, typename V, typename F>
void Map<K, V, F>::set(K&& key, V&& value, il::io_t, il::int_t& i) {
  i = search(key);
  if (notFound(i)) {
    insert(std::move(key), std::move(value), il::io, i);
  } else {
    (slot_ + i)->value.~V();
    new (&((slot_ + i)->value)) V(std::move(value));
  }
}

template <typename K, typename V, typename F>
void Map<K, V, F>::set(const K& key, const V& value) {
  il::int_t i = search(key);
  if (notFound(i)) {
    insert(key, value, il::io, i);
  } else {
    (slot_ + i)->value.~V();
    new (&((slot_ + i)->value)) V(value);
  }
}

template <typename K, typename V, typename F>
void Map<K, V, F>::set(const K& key, V&& value) {
  il::int_t i = search(key);
  if (notFound(i)) {
    insert(key, std::move(value), il::io, i);
  } else {
    (slot_ + i)->value.~V();
    new (&((slot_ + i)->value)) V(std::move(value));
  }
}

template <typename K, typename V, typename F>
void Map<K, V, F>::set(K&& key, const V& value) {
  il::int_t i = search(key);
  if (notFound(i)) {
    insert(std::move(key), value, il::io, i);
  } else {
    (slot_ + i)->value.~V();
    new (&((slot_ + i)->value)) V(std::move(value));
  }
}

template <typename K, typename V, typename F>
void Map<K, V, F>::set(K&& key, V&& value) {
  il::int_t i = search(key);
  if (notFound(i)) {
    insert(std::move(key), std::move(value), il::io, i);
  } else {
    (slot_ + i)->value.~V();
    new (&((slot_ + i)->value)) V(std::move(value));
  }
}

template <typename K, typename V, typename F>
il::int_t Map<K, V, F>::search(const K& key) const {
  IL_EXPECT_MEDIUM(!F::isEmpty(key));
  IL_EXPECT_MEDIUM(!F::isTombstone(key));

  if (p_ == -1) {
    return -1;
  }

  const il::int_t mask = (1 << p_) - 1;
  il::int_t i = static_cast<il::int_t>(F::hash(key, p_));
  il::int_t i_tombstone = -1;
  il::int_t delta_i = 1;
  while (true) {
    if (F::isEmpty(slot_[i].key)) {
      return (i_tombstone == -1) ? -(1 + i) : -(1 + i_tombstone);
    } else if (F::isTombstone(slot_[i].key)) {
      i_tombstone = i;
    } else if (F::isEqual(slot_[i].key, key)) {
      return i;
    }

    i += delta_i;
    i &= mask;
    ++delta_i;
  }
}

template <typename K, typename V, typename F>
bool Map<K, V, F>::hasFound(il::int_t i) const {
  return i >= 0;
}

template <typename K, typename V, typename F>
bool Map<K, V, F>::notFound(il::int_t i) const {
  return i < 0;
}

template <typename K, typename V, typename F>
void Map<K, V, F>::insert(const K& key, const V& value, il::io_t,
                          il::int_t& i) {
  IL_EXPECT_FAST(notFound(i));

  // FIXME: What it the place is a tombstone. We should update the number of
  // tombstones in the hash table.

  il::int_t i_local = -(1 + i);
  if (2 * (nb_element_ + 1) > (1 << p_)) {
    grow(il::next_power_of_2_32(2 * (nb_element_ + 1)));
    il::int_t j = search(key);
    i_local = -(1 + j);
  }
  new (const_cast<K*>(&((slot_ + i_local)->key))) K(key);
  new (&((slot_ + i_local)->value)) V(value);
  ++nb_element_;
  i = i_local;
}

template <typename K, typename V, typename F>
void Map<K, V, F>::insert(const K& key, V&& value, il::io_t, il::int_t& i) {
  IL_EXPECT_FAST(notFound(i));

  // FIXME: What it the place is a tombstone. We should update the number of
  // tombstones in the hash table.

  il::int_t i_local = -(1 + i);
  if (2 * (nb_element_ + 1) > (1 << p_)) {
    grow(il::next_power_of_2_32(2 * (nb_element_ + 1)));
    il::int_t j = search(key);
    i_local = -(1 + j);
  }
  new (const_cast<K*>(&((slot_ + i_local)->key))) K(key);
  new (&((slot_ + i_local)->value)) V(value);
  ++nb_element_;
  i = i_local;
}

template <typename K, typename V, typename F>
void Map<K, V, F>::insert(K&& key, const V& value, il::io_t, il::int_t& i) {
  IL_EXPECT_FAST(notFound(i));

  // FIXME: What it the place is a tombstone. We should update the number of
  // tombstones in the hash table.

  il::int_t i_local = -(1 + i);
  if (2 * (nb_element_ + 1) > (1 << p_)) {
    grow(il::next_power_of_2_32(2 * (nb_element_ + 1)));
    il::int_t j = search(key);
    i_local = -(1 + j);
  }
  new (const_cast<K*>(&((slot_ + i_local)->key))) K(std::move(key));
  new (&((slot_ + i_local)->value)) V(value);
  ++nb_element_;
  i = i_local;
}

template <typename K, typename V, typename F>
void Map<K, V, F>::insert(K&& key, V&& value, il::io_t, il::int_t& i) {
  IL_EXPECT_FAST(notFound(i));

  // FIXME: What it the place is a tombstone. We should update the number of
  // tombstones in the hash table.

  il::int_t i_local = -(1 + i);
  if (2 * (nb_element_ + 1) > (1 << p_)) {
    grow(il::next_power_of_2_32(2 * (nb_element_ + 1)));
    il::int_t j = search(key);
    i_local = -(1 + j);
  }
  new (const_cast<K*>(&((slot_ + i_local)->key))) K(std::move(key));
  new (&((slot_ + i_local)->value)) V(std::move(value));
  ++nb_element_;
  i = i_local;
}

template <typename K, typename V, typename F>
void Map<K, V, F>::insert(const K& key, const V& value) {
  const il::int_t i = search(key);
  IL_EXPECT_FAST(notFound(i));

  // FIXME: What it the place is a tombstone. We should update the number of
  // tombstones in the hash table.

  il::int_t i_local = -(1 + i);
  if (2 * (nb_element_ + 1) > (1 << p_)) {
    grow(il::next_power_of_2_32(2 * (nb_element_ + 1)));
    il::int_t j = search(key);
    i_local = -(1 + j);
  }
  new (const_cast<K*>(&((slot_ + i_local)->key))) K(key);
  new (&((slot_ + i_local)->value)) V(value);
  ++nb_element_;
}

template <typename K, typename V, typename F>
void Map<K, V, F>::insert(const K& key, V&& value) {
  const il::int_t i = search(key);
  IL_EXPECT_FAST(notFound(i));

  // FIXME: What it the place is a tombstone. We should update the number of
  // tombstones in the hash table.

  il::int_t i_local = -(1 + i);
  if (2 * (nb_element_ + 1) > (1 << p_)) {
    grow(il::next_power_of_2_32(2 * (nb_element_ + 1)));
    il::int_t j = search(key);
    i_local = -(1 + j);
  }
  new (const_cast<K*>(&((slot_ + i_local)->key))) K(key);
  new (&((slot_ + i_local)->value)) V(value);
  ++nb_element_;
}

template <typename K, typename V, typename F>
void Map<K, V, F>::erase(il::int_t i) {
  IL_EXPECT_FAST(hasFound(i));

  (&((slot_ + i)->key))->~K();
  F::constructTombstone(il::io, reinterpret_cast<K*>(slot_ + i));
  (&((slot_ + i)->value))->~V();
  --nb_element_;
  ++nb_tombstone_;
  return;
}

template <typename K, typename V, typename F>
const K& Map<K, V, F>::key(il::int_t i) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>((p_ >= 0) ? (1 << p_) : 0));

  return slot_[i].key;
}

template <typename K, typename V, typename F>
const V& Map<K, V, F>::value(il::int_t i) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>((p_ >= 0) ? (1 << p_) : 0));

  return slot_[i].value;
}

template <typename K, typename V, typename F>
V& Map<K, V, F>::value(il::int_t i) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>((p_ >= 0) ? (1 << p_) : 0));

  return slot_[i].value;
}

template <typename K, typename V, typename F>
il::int_t Map<K, V, F>::size() const {
  return nb_element_;
}

template <typename K, typename V, typename F>
il::int_t Map<K, V, F>::capacity() const {
  return (p_ >= 0) ? (1 << p_) : 0;
}

template <typename K, typename V, typename F>
void Map<K, V, F>::reserve(il::int_t r) {
  IL_EXPECT_FAST(r >= 0);

  grow(il::next_power_of_2_32(r));
}

template <typename K, typename V, typename F>
double Map<K, V, F>::load() const {
  IL_EXPECT_MEDIUM(p_ >= 0);

  return static_cast<double>(nb_element_) / capacity();
}

template <typename K, typename V, typename F>
double Map<K, V, F>::displaced() const {
  const il::int_t m = capacity();
  il::int_t nb_displaced = 0;
  for (il::int_t i = 0; i < m; ++i) {
    if (!F::isEmpty(slot_[i].key) && !F::isTombstone(slot_[i].key)) {
      const il::int_t hashed =
          static_cast<il::int_t>(F::hash(slot_[i].key, p_));
      if (i != hashed) {
        ++nb_displaced;
      }
    }
  }

  return static_cast<double>(nb_displaced) / nb_element_;
}

template <typename K, typename V, typename F>
double Map<K, V, F>::displacedTwice() const {
  const il::int_t m = capacity();
  const il::int_t mask = (1 << p_) - 1;
  il::int_t nb_displaced_twice = 0;
  for (il::int_t i = 0; i < m; ++i) {
    if (!F::isEmpty(slot_[i].key) && !F::isTombstone(slot_[i].key)) {
      const il::int_t hashed =
          static_cast<il::int_t>(F::hash(slot_[i].key, p_));
      if (i != hashed && (((i - 1) - hashed) & mask) != 0) {
        ++nb_displaced_twice;
      }
    }
  }

  return static_cast<double>(nb_displaced_twice) / nb_element_;
}

template <typename K, typename V, typename F>
bool Map<K, V, F>::isEmpty() const {
  return nb_element_ == 0;
}

template <typename K, typename V, typename F>
MapIterator<K, V, F> Map<K, V, F>::begin() {
  if (isEmpty()) {
    return end();
  } else {
    il::int_t i = 0;
    while (true) {
      if (!F::isEmpty(slot_[i].key) && !F::isTombstone(slot_[i].key)) {
        return MapIterator<K, V, F>{slot_ + i, slot_ + nb_element_};
      }
      ++i;
    }
  }
}

template <typename K, typename V, typename F>
MapIterator<K, V, F> Map<K, V, F>::end() {
  return MapIterator<K, V, F>{slot_ + nb_element_, slot_ + nb_element_};
}

template <typename K, typename V, typename F>
il::int_t Map<K, V, F>::first() const {
  const il::int_t m = il::int_t{1} << p_;

  il::int_t i = 0;
  while (i < m && (F::isEmpty(slot_[i].key) || F::isTombstone(slot_[i].key))) {
    ++i;
  }
  return i;
}

template <typename K, typename V, typename F>
il::int_t Map<K, V, F>::sentinel() const {
  return il::int_t{1} << p_;
}

template <typename K, typename V, typename F>
il::int_t Map<K, V, F>::next(il::int_t i) const {
  const il::int_t m = il::int_t{1} << p_;

  ++i;
  while (i < m && (F::isEmpty(slot_[i].key) || F::isTombstone(slot_[i].key))) {
    ++i;
  }
  return i;
}

template <typename K, typename V, typename F>
void Map<K, V, F>::grow(il::int_t r) {
  IL_EXPECT_FAST(r >= capacity());

  KeyValue<K, V>* old_slot_ = slot_;
  const il::int_t old_m = (p_ == -1) ? 0 : (1 << p_);
  const int p = il::next_log2_32(r);
  const il::int_t m = 1 << p;

  slot_ =
      static_cast<KeyValue<K, V>*>(::operator new(m * sizeof(KeyValue<K, V>)));
  for (il::int_t i = 0; i < m; ++i) {
    F::constructEmpty(il::io, reinterpret_cast<K*>(slot_ + i));
  }
  p_ = p;
  nb_element_ = 0;
  nb_tombstone_ = 0;

  if (p_ >= 0) {
    for (il::int_t i = 0; i < old_m; ++i) {
      if (!F::isEmpty(old_slot_[i].key) && !F::isTombstone(old_slot_[i].key)) {
        il::int_t new_i = search(old_slot_[i].key);
        insert(std::move(old_slot_[i].key), std::move(old_slot_[i].value),
               il::io, new_i);
        (&((old_slot_ + i)->key))->~K();
        (&((old_slot_ + i)->value))->~V();
      }
    }
    ::operator delete(old_slot_);
  }
}
}  // namespace il

#endif  // IL_MAP_H
