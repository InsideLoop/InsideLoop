//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_HASHMAP_H
#define IL_HASHMAP_H

#include <il/Array.h>
#include <il/container/hash/HashFunction.h>
#include <il/core/Status.h>

namespace il {

template <typename K, typename V>
struct KeyValue {
  const K key;
  V value;
};

template <typename K, typename V, typename F>
class HashMapIterator {
 private:
  KeyValue<K, V>* pointer_;
  KeyValue<K, V>* end_;

 public:
  HashMapIterator(KeyValue<K, V>* pointer, KeyValue<K, V>* end);
  KeyValue<K, V>& operator*() const;
  KeyValue<K, V>* operator->() const;
  bool operator==(const HashMapIterator& it) const;
  bool operator!=(const HashMapIterator& it) const;
  HashMapIterator& operator++();
  HashMapIterator& operator++(int);

 private:
  void advance_past_empty_buckets();
};

template <typename K, typename V, typename F>
HashMapIterator<K, V, F>::HashMapIterator(KeyValue<K, V>* pointer,
                                          KeyValue<K, V>* end) {
  pointer_ = pointer;
  end_ = end;
}

template <typename K, typename V, typename F>
KeyValue<K, V>& HashMapIterator<K, V, F>::operator*() const {
  return *pointer_;
}

template <typename K, typename V, typename F>
KeyValue<K, V>* HashMapIterator<K, V, F>::operator->() const {
  return pointer_;
}

template <typename K, typename V, typename F>
bool HashMapIterator<K, V, F>::operator==(const HashMapIterator& it) const {
  return pointer_ == it.pointer_;
}

template <typename K, typename V, typename F>
bool HashMapIterator<K, V, F>::operator!=(const HashMapIterator& it) const {
  return pointer_ != it.pointer_;
}

template <typename K, typename V, typename F>
HashMapIterator<K, V, F>& HashMapIterator<K, V, F>::operator++() {
  ++pointer_;
  advance_past_empty_buckets();
  return *this;
}

template <typename K, typename V, typename F>
HashMapIterator<K, V, F>& HashMapIterator<K, V, F>::operator++(int) {
  HashMapIterator tmp = *this;
  ++pointer_;
  advance_past_empty_buckets();
  return tmp;
}

template <typename K, typename V, typename F>
void HashMapIterator<K, V, F>::advance_past_empty_buckets() {
  const K empty_key = F::empty_key();
  const K tombstone_key = F::tombstone_key();
  while (pointer_ != end_ && (F::is_equal(pointer_->key, empty_key) ||
                              F::is_equal(pointer_->key, tombstone_key))) {
    ++pointer_;
  }
}

template <typename K, typename V, typename F = HashFunction<K>>
class HashMap {
 private:
  KeyValue<K, V>* slot_;
  il::int_t nb_slot_;
  il::int_t nb_element_;
  il::int_t nb_tombstone_;

 public:
  HashMap();
  HashMap(il::int_t n);
  HashMap(il::value_t, std::initializer_list<il::KeyValue<K, V>> list);
  HashMap(const HashMap<K, V, F>& map);
  HashMap(HashMap<K, V, F>&& map);
  HashMap& operator=(const HashMap<K, V, F>& map);
  HashMap& operator=(HashMap<K, V, F>&& map);
  ~HashMap();
  il::int_t search(const K& key) const;
  bool found(il::int_t i) const;
  void insert(const K& key, const V& value, il::io_t, il::int_t& i);
  void erase(il::int_t i);
  const K& key(il::int_t i) const;
  const V& value(il::int_t i) const;
  V& value(il::int_t i);
  bool empty() const;
  il::int_t size() const;
  il::int_t capacity() const;
  void reserve(il::int_t r);
  double load() const;
  double displaced() const;
  double displaced_twice() const;
  HashMapIterator<K, V, F> begin();
  HashMapIterator<K, V, F> end();

 private:
  void grow(il::int_t r);
  static il::int_t next_power_of_2(il::int_t i);
};

template <typename K, typename V, typename F>
HashMap<K, V, F>::HashMap() {
  slot_ = nullptr;
  nb_slot_ = 0;
  nb_element_ = 0;
  nb_tombstone_ = 0;
}

template <typename K, typename V, typename F>
HashMap<K, V, F>::HashMap(il::int_t n) {
  IL_ASSERT_PRECOND(n >= 0);
  if (n > 0) {
    n = next_power_of_2(2 * n);
    slot_ = static_cast<KeyValue<K, V>*>(
        ::operator new(n * sizeof(KeyValue<K, V>)));
    for (il::int_t i = 0; i < n; ++i) {
      F::set_empty(il::io, reinterpret_cast<K*>(slot_ + i));
    }
  } else {
    slot_ = nullptr;
  }
  nb_slot_ = n;
  nb_element_ = 0;
  nb_tombstone_ = 0;
}

template <typename K, typename V, typename F>
HashMap<K, V, F>::HashMap(il::value_t,
                          std::initializer_list<il::KeyValue<K, V>> list) {
  const il::int_t nb_element = static_cast<il::int_t>(list.size());
  il::int_t n = nb_element;
  if (n > 0) {
    n = next_power_of_2(2 * n);
    slot_ = static_cast<KeyValue<K, V>*>(
        ::operator new(n * sizeof(KeyValue<K, V>)));
    for (il::int_t i = 0; i < n; ++i) {
      F::set_empty(il::io, reinterpret_cast<K*>(slot_ + i));
    }
  } else {
    slot_ = nullptr;
  }
  nb_slot_ = n;
  nb_element_ = 0;
  nb_tombstone_ = 0;
  for (il::int_t k = 0; k < nb_element; ++k) {
    il::int_t i = search((list.begin() + k)->key);
    IL_ASSERT(!found(i));
    insert((list.begin() + k)->key, (list.begin() + k)->value, il::io, i);
  }
}

template <typename K, typename V, typename F>
HashMap<K, V, F>::HashMap(const HashMap<K, V, F>& map) {
  const il::int_t n = map.nb_slot_;
  if (n > 0) {
    slot_ = static_cast<KeyValue<K, V>*>(
        ::operator new(n * sizeof(KeyValue<K, V>)));
    for (il::int_t i = 0; i < n; ++i) {
      if (F::is_empty(map.slot_[i].key)) {
        F::set_empty(il::io, reinterpret_cast<K*>(slot_ + i));
      } else if (F::is_tombstone(map.slot_[i].key)) {
        F::set_tombstone(il::io, reinterpret_cast<K*>(slot_ + i));
      } else {
        new (const_cast<K*>(&((slot_ + i)->key))) K(map.slot_[i].key);
        new (&((slot_ + i)->value)) V(map.slot_[i].value);
      }
    }
  } else {
    slot_ = nullptr;
  }
  nb_slot_ = n;
  nb_element_ = map.nb_element_;
  nb_tombstone_ = map.nb_tombstone_;
}

template <typename K, typename V, typename F>
HashMap<K, V, F>::HashMap(HashMap<K, V, F>&& map) {
  slot_ = map.slot_;
  nb_slot_ = map.nb_slot_;
  nb_element_ = map.nb_element_;
  nb_tombstone_ = map.nb_tombstone_;
  map.slot_ = nullptr;
  map.nb_slot_ = 0;
  map.nb_element_ = 0;
  map.nb_tombstone_ = 0;
}

template <typename K, typename V, typename F>
HashMap<K, V, F>& HashMap<K, V, F>::operator=(const HashMap<K, V, F>& map) {
  const il::int_t n = map.nb_slot_;
  if (n > 0) {
    for (il::int_t i = 0; i < nb_slot_; ++i) {
      if (!F::is_empy(slot_ + i) && !F::is_tombstone(slot_ + i)) {
        (&((slot_ + i)->value))->~V();
        (&((slot_ + i)->key))->~K();
      }
    }
    ::operator delete(slot_);
    slot_ = static_cast<KeyValue<K, V>*>(
        ::operator new(n * sizeof(KeyValue<K, V>)));
    for (il::int_t i = 0; i < n; ++i) {
      if (F::is_empty(map.slot_[i].key)) {
        F::set_empty(il::io, reinterpret_cast<K*>(slot_ + i));
      } else if (F::is_tombstone(map.slot_[i].key)) {
        F::set_tombstone(il::io, reinterpret_cast<K*>(slot_ + i));
      } else {
        new (const_cast<K*>(&((slot_ + i)->key))) K(map.slot_[i].key);
        new (&((slot_ + i)->value)) V(map.slot_[i].value);
      }
    }
  } else {
    slot_ = nullptr;
  }
  nb_slot_ = n;
  nb_element_ = map.nb_element_;
  nb_tombstone_ = map.nb_tombstone_;
}

template <typename K, typename V, typename F>
HashMap<K, V, F>& HashMap<K, V, F>::operator=(HashMap<K, V, F>&& map) {
  if (this != &map) {
    if (slot_ != nullptr) {
      for (il::int_t i = 0; i < nb_slot_; ++i) {
        if (!F::is_empy(slot_ + i) && !F::is_tombstone(slot_ + i)) {
          (&((slot_ + i)->value))->~V();
          (&((slot_ + i)->key))->~K();
        }
      }
      ::operator delete(slot_);
    }
    slot_ = map.slot_;
    nb_slot_ = map.nb_slot_;
    nb_element_ = map.nb_element_;
    nb_tombstone_ = map.nb_tombstone_;
    map.slot_ = nullptr;
    map.nb_slot_ = 0;
    map.nb_element_ = 0;
    map.nb_tombstone_ = 0;
  }
}

template <typename K, typename V, typename F>
HashMap<K, V, F>::~HashMap() {
  if (slot_ != nullptr) {
    for (il::int_t i = 0; i < nb_slot_; ++i) {
      if (!F::is_empty(slot_[i].key) && !F::is_tombstone(slot_[i].key)) {
        (&((slot_ + i)->value))->~V();
        (&((slot_ + i)->key))->~K();
      }
    }
    ::operator delete(slot_);
  }
}

template <typename K, typename V, typename F>
il::int_t HashMap<K, V, F>::search(const K& key) const {
  IL_EXPECT_FAST(!F::is_empty(key));
  IL_EXPECT_FAST(!F::is_tombstone(key));

  if (nb_slot_ == 0) {
    return -(1 + nb_slot_);
  }

  const std::size_t nb_slot_minus_one = static_cast<std::size_t>(nb_slot_) - 1;
  il::int_t i = static_cast<il::int_t>(F::hash_value(key) & nb_slot_minus_one);
  il::int_t i_tombstone = -1;
  il::int_t delta_i = 1;
  while (true) {
    if (F::is_empty(slot_[i].key)) {
      return (i_tombstone == -1) ? -(1 + i) : -(1 + i_tombstone);
    } else if (F::is_tombstone(slot_[i].key)) {
      i_tombstone = i;
    } else if (F::is_equal(slot_[i].key, key)) {
      return i;
    }

    i += delta_i;
    ++delta_i;
    i &= nb_slot_minus_one;
  }
}

template <typename K, typename V, typename F>
bool HashMap<K, V, F>::found(il::int_t i) const {
  return i >= 0;
}

template <typename K, typename V, typename F>
void HashMap<K, V, F>::insert(const K& key, const V& value, il::io_t,
                              il::int_t& i) {
  IL_ASSERT_PRECOND(!found(i));

  il::int_t i_local = -(1 + i);
  if (2 * (nb_element_ + 1) >= nb_slot_) {
    grow(next_power_of_2(nb_element_ + 1));
    il::int_t j = search(key);
    i_local = -(1 + j);
  }
  new (const_cast<K*>(&((slot_ + i_local)->key))) K(key);
  new (&((slot_ + i_local)->value)) V(value);
  ++nb_element_;
  i = i_local;
}

template <typename K, typename V, typename F>
void HashMap<K, V, F>::erase(il::int_t i) {
  IL_ASSERT_PRECOND(found(i));

  (&((slot_ + i)->key))->~K();
  F::set_tombstone(il::io, reinterpret_cast<K*>(slot_ + i));
  (&((slot_ + i)->value))->~V();
  --nb_element_;
  ++nb_tombstone_;
  return;
}

template <typename K, typename V, typename F>
const K& HashMap<K, V, F>::key(il::int_t i) const {
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i) <
                   static_cast<il::uint_t>(nb_slot_));

  return slot_[i].key;
}

template <typename K, typename V, typename F>
const V& HashMap<K, V, F>::value(il::int_t i) const {
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i) <
                   static_cast<il::uint_t>(nb_slot_));

  return slot_[i].value;
}

template <typename K, typename V, typename F>
V& HashMap<K, V, F>::value(il::int_t i) {
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(i) <
                   static_cast<il::uint_t>(nb_slot_));

  return slot_[i].value;
}

template <typename K, typename V, typename F>
il::int_t HashMap<K, V, F>::size() const {
  return nb_element_;
}

template <typename K, typename V, typename F>
il::int_t HashMap<K, V, F>::capacity() const {
  return nb_slot_;
}

template <typename K, typename V, typename F>
void HashMap<K, V, F>::reserve(il::int_t r) {
  IL_ASSERT_PRECOND(r >= 0);

  grow(next_power_of_2(r));
}

template <typename K, typename V, typename F>
double HashMap<K, V, F>::load() const {
  return static_cast<double>(nb_element_) / nb_slot_;
}

template <typename K, typename V, typename F>
double HashMap<K, V, F>::displaced() const {
  const std::size_t nb_slot_minus_one = static_cast<std::size_t>(nb_slot_) - 1;
  il::int_t nb_displaced = 0;
  for (il::int_t i = 0; i < nb_slot_; ++i) {
    if (!F::is_empty(slot_[i].key) &&
        !F::is_tombstone(slot_[i].key)) {
      const il::int_t hashed = static_cast<il::int_t>(
          F::hash_value(slot_[i].key) & nb_slot_minus_one);
      if (i != hashed) {
        ++nb_displaced;
      }
    }
  }

  return static_cast<double>(nb_displaced) / nb_element_;
}

template <typename K, typename V, typename F>
double HashMap<K, V, F>::displaced_twice() const {
  const std::size_t nb_slot_minus_one = static_cast<std::size_t>(nb_slot_) - 1;
  il::int_t nb_displaced_twice = 0;
  for (il::int_t i = 0; i < nb_slot_; ++i) {
    if (!F::is_empty(slot_[i].key) &&
        !F::is_tombstone(slot_[i].key)) {
      const il::int_t hashed = static_cast<il::int_t>(
          F::hash_value(slot_[i].key) & nb_slot_minus_one);
      if (i != hashed && (((i - 1) - hashed) & nb_slot_minus_one) != 0) {
        ++nb_displaced_twice;
      }
    }
  }

  return static_cast<double>(nb_displaced_twice) / nb_element_;
}

template <typename K, typename V, typename F>
bool HashMap<K, V, F>::empty() const {
  return nb_element_ == 0;
}

template <typename K, typename V, typename F>
HashMapIterator<K, V, F> HashMap<K, V, F>::begin() {
  if (empty()) {
    return end();
  } else {
    il::int_t i = 0;
    while (true) {
      if (!F::is_empty(slot_[i].key) &&
          !F::is_tombstone(slot_[i].key)) {
        return HashMapIterator<K, V, F>{slot_ + i, slot_ + nb_element_};
      }
      ++i;
    }
  }
}

template <typename K, typename V, typename F>
HashMapIterator<K, V, F> HashMap<K, V, F>::end() {
  return HashMapIterator<K, V, F>{slot_ + nb_element_, slot_ + nb_element_};
}

template <typename K, typename V, typename F>
void HashMap<K, V, F>::grow(il::int_t r) {
  IL_ASSERT(r >= nb_slot_);

  KeyValue<K, V>* old_slot_ = slot_;
  il::int_t old_nb_slot_ = nb_slot_;
  slot_ =
      static_cast<KeyValue<K, V>*>(::operator new(r * sizeof(KeyValue<K, V>)));
  for (il::int_t i = 0; i < r; ++i) {
    F::set_empty(il::io, reinterpret_cast<K*>(slot_ + i));
  }
  nb_slot_ = r;
  nb_element_ = 0;
  nb_tombstone_ = 0;

  if (old_nb_slot_ > 0) {
    for (il::int_t i = 0; i < old_nb_slot_; ++i) {
      if (!F::is_empty(old_slot_[i].key) &&
          !F::is_tombstone(old_slot_[i].key)) {
        il::int_t new_i = search(old_slot_[i].key);
        IL_ASSERT(found(new_i));
        insert(old_slot_[i].key, old_slot_[i].value, il::io, new_i);
        (&((old_slot_ + i)->value))->~V();
      }
    }
    ::operator delete(old_slot_);
  }
}

template <typename K, typename V, typename F>
il::int_t HashMap<K, V, F>::next_power_of_2(il::int_t i) {
  IL_ASSERT(i >= 0);

  i |= (i >> 1);
  i |= (i >> 2);
  i |= (i >> 4);
  i |= (i >> 8);
  i |= (i >> 16);
  if (sizeof(il::int_t) == 8) {
    i |= (i >> 32);
  }
  i = i + 1;

  IL_ASSERT(i >= 0);
  return i;
}
}

#endif  // IL_HASHMAP_H
