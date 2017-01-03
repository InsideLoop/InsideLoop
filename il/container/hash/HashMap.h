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

template <typename K, typename V, typename F>
struct KeyValue {
  K key;
  V value;

  KeyValue();
};

template <typename K, typename V, typename F>
KeyValue<K, V, F>::KeyValue() {
  key = F::empty_key();
  value = V{};
}

template <typename K, typename V, typename F>
class HashMapIterator {
 private:
  KeyValue<K, V, F>* pointer_;
  KeyValue<K, V, F>* end_;

 public:
  HashMapIterator(KeyValue<K, V, F>* pointer, KeyValue<K, V, F>* end);
  KeyValue<K, V, F>& operator*() const;
  KeyValue<K, V, F>* operator->() const;
  bool operator==(const HashMapIterator& it) const;
  bool operator!=(const HashMapIterator& it) const;
  HashMapIterator& operator++();
  HashMapIterator& operator++(int);

 private:
  void advance_past_empty_buckets();
};

template <typename K, typename V, typename F>
HashMapIterator<K, V, F>::HashMapIterator(KeyValue<K, V, F>* pointer,
                                              KeyValue<K, V, F>* end) {
  pointer_ = pointer;
  end_ = end;
}

template <typename K, typename V, typename F>
KeyValue<K, V, F>& HashMapIterator<K, V, F>::operator*() const {
  return *pointer_;
}

template <typename K, typename V, typename F>
KeyValue<K, V, F>* HashMapIterator<K, V, F>::operator->() const {
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
  il::Array<KeyValue<K, V, F>> bucket_;
  il::int_t nb_entries_;
  il::int_t nb_tombstones_;

 public:
  HashMap(il::int_t nb_entries = 0);
  il::int_t search(const K& key) const;
  bool found(il::int_t i) const;
  void insert(const K& key, const K& value, il::io_t, il::int_t& i);
  void insert(const K& key, const K& value);
  void erase(il::int_t i);
  const KeyValue<K, V, F>& operator[](il::int_t i) const;
  KeyValue<K, V, F>& operator[](il::int_t i);
  bool empty() const;
  il::int_t nb_entries() const;
  HashMapIterator<K, V, F> begin();
  HashMapIterator<K, V, F> end();

 private:
  void grow(il::int_t n);
  static il::int_t next_power_of_2(il::int_t i);
  static il::int_t nb_bucket(il::int_t nb_entries);
};

template <typename K, typename V, typename F>
HashMap<K, V, F>::HashMap(il::int_t nb_entries)
    : bucket_{nb_bucket(nb_entries)} {
  nb_entries_ = 0;
  nb_tombstones_ = 0;
}

template <typename K, typename V, typename F>
il::int_t HashMap<K, V, F>::search(const K& key) const {
  const K empty_key = F::empty_key();
  const K tombstone_key = F::tombstone_key();
  IL_ASSERT_PRECOND(!F::is_equal(key, empty_key));
  IL_ASSERT_PRECOND(!F::is_equal(key, tombstone_key));

  const il::int_t nb_bucket = bucket_.size();
  if (nb_bucket == 0) {
    return -(1 + nb_bucket);
  }

  il::int_t i = F::hash_value(key) & (nb_bucket - 1);
  il::int_t i_tombstone = -1;
  il::int_t delta_i = 1;
  for (il::int_t k = 0; k < nb_bucket; ++k) {
    if (F::is_equal(bucket_[i].key, key)) {
      return i;
    }

    if (F::is_equal(bucket_[i].key, empty_key)) {
      return (i_tombstone == -1) ? -(1 + i) : -(1 + i_tombstone);
    }

    if (F::is_equal(bucket_[i].key, tombstone_key) && i_tombstone == -1) {
      i_tombstone = i;
    }

    i += delta_i;
    ++delta_i;
    i &= (nb_bucket - 1);
  }
  return -(1 + nb_bucket);
}

template <typename K, typename V, typename F>
bool HashMap<K, V, F>::found(il::int_t i) const {
  return i >= 0;
}

template <typename K, typename V, typename F>
void HashMap<K, V, F>::insert(const K& key, const K& value, il::io_t, il::int_t& i) {
  IL_ASSERT_PRECOND(!found(i));

  il::int_t i_local = -(1 + i);
  if (nb_entries_ >= bucket_.size()) {
    grow(nb_bucket(nb_entries_));
    il::int_t j = search(key);
    i_local = -(1 + search(key));
  }
  bucket_[i_local].key = key;
  bucket_[i_local].value = value;
  ++nb_entries_;
};

template <typename K, typename V, typename F>
void HashMap<K, V, F>::insert(const K& key, const K& value) {
  il::int_t i = search(key);
  IL_ASSERT(!found(i));
  insert(key, value, il::io, i);
};

template <typename K, typename V, typename F>
void HashMap<K, V, F>::erase(il::int_t i) {
  bucket_[i].key = F::tombstone_key();
  bucket_[i].value = V{};
  --nb_entries_;
  ++nb_tombstones_;
  return;
};

template <typename K, typename V, typename F>
const KeyValue<K, V, F>& HashMap<K, V, F>::operator[](il::int_t i) const {
  return bucket_[i];
};

template <typename K, typename V, typename F>
KeyValue<K, V, F>& HashMap<K, V, F>::operator[](il::int_t i) {
  return bucket_[i];
};

template <typename K, typename V, typename F>
il::int_t HashMap<K, V, F>::nb_entries() const {
  return nb_entries_;
}

template <typename K, typename V, typename F>
bool HashMap<K, V, F>::empty() const {
  return nb_entries_ == 0;
}

template <typename K, typename V, typename F>
HashMapIterator<K, V, F> HashMap<K, V, F>::begin() {
  if (empty()) {
    return end();
  } else {
    const K empty_key = F::empty_key();
    const K tombstone_key = F::tombstone_key();
    il::int_t i = 0;
    while (true) {
      if (!F::is_equal(bucket_[i].key, empty_key) &&
          !F::is_equal(bucket_[i].key, tombstone_key)) {
        return HashMapIterator<K, V, F>{bucket_.data() + i,
                                          bucket_.data() + bucket_.size()};
      }
      ++i;
    }
  }
}

template <typename K, typename V, typename F>
HashMapIterator<K, V, F> HashMap<K, V, F>::end() {
  return HashMapIterator<K, V, F>{bucket_.data() + bucket_.size(),
                                    bucket_.data() + bucket_.size()};
}

template <typename K, typename V, typename F>
void HashMap<K, V, F>::grow(il::int_t n) {
  IL_ASSERT(n >= bucket_.size());

  const K empty_key = F::empty_key();
  const K tombstone_key = F::tombstone_key();

  il::Array<KeyValue<K, V, F>> old_bucket{std::move(bucket_)};
  bucket_.resize(n);
  nb_entries_ = 0;
  nb_tombstones_ = 0;

  for (il::int_t i = 0; i < old_bucket.size(); ++i) {
    if (!F::is_equal(old_bucket[i].key, empty_key) &&
        !F::is_equal(old_bucket[i].key, tombstone_key)) {
      insert(old_bucket[i].key, old_bucket[i].value);
    }
  }
}

template <typename K, typename V, typename F>
il::int_t HashMap<K, V, F>::next_power_of_2(il::int_t i) {
  // Only works with 32 bits integers
  IL_ASSERT(i >= 0);

  i |= (i >> 1);
  i |= (i >> 2);
  i |= (i >> 4);
  i |= (i >> 8);
  i |= (i >> 16);
  i = i + 1;

  IL_ASSERT(i >= 0);
  return i;
}

template <typename K, typename V, typename F>
il::int_t HashMap<K, V, F>::nb_bucket(il::int_t nb_entries) {
  return nb_entries == 0 ? 1 : next_power_of_2(4 * nb_entries / 3 + 1);
}
}

#endif  // IL_HASHMAP_H
