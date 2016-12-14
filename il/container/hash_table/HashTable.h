//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_HASHTABLE_H
#define IL_HASHTABLE_H

#include <il/Array.h>
#include <il/container/hash_table/HashFunction.h>
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
class HashTableIterator {
 private:
  KeyValue<K, V, F>* pointer_;
  KeyValue<K, V, F>* end_;

 public:
  HashTableIterator(KeyValue<K, V, F>* pointer, KeyValue<K, V, F>* end);
  KeyValue<K, V, F>& operator*() const;
  KeyValue<K, V, F>* operator->() const;
  bool operator==(const HashTableIterator& it) const;
  bool operator!=(const HashTableIterator& it) const;
  HashTableIterator& operator++();
  HashTableIterator& operator++(int);

 private:
  void advance_past_empty_buckets();
};

template <typename K, typename V, typename F>
HashTableIterator<K, V, F>::HashTableIterator(KeyValue<K, V, F>* pointer,
                                              KeyValue<K, V, F>* end) {
  pointer_ = pointer;
  end_ = end;
}

template <typename K, typename V, typename F>
KeyValue<K, V, F>& HashTableIterator<K, V, F>::operator*() const {
  return *pointer_;
}

template <typename K, typename V, typename F>
KeyValue<K, V, F>* HashTableIterator<K, V, F>::operator->() const {
  return pointer_;
}

template <typename K, typename V, typename F>
bool HashTableIterator<K, V, F>::operator==(const HashTableIterator& it) const {
  return pointer_ == it.pointer_;
}

template <typename K, typename V, typename F>
bool HashTableIterator<K, V, F>::operator!=(const HashTableIterator& it) const {
  return pointer_ != it.pointer_;
}

template <typename K, typename V, typename F>
HashTableIterator<K, V, F>& HashTableIterator<K, V, F>::operator++() {
  ++pointer_;
  advance_past_empty_buckets();
  return *this;
}

template <typename K, typename V, typename F>
HashTableIterator<K, V, F>& HashTableIterator<K, V, F>::operator++(int) {
  HashTableIterator tmp = *this;
  ++pointer_;
  advance_past_empty_buckets();
  return tmp;
}

template <typename K, typename V, typename F>
void HashTableIterator<K, V, F>::advance_past_empty_buckets() {
  const K empty_key = F::empty_key();
  const K tombstone_key = F::tombstone_key();
  while (pointer_ != end_ && (F::is_equal(pointer_->key, empty_key) ||
                              F::is_equal(pointer_->key, tombstone_key))) {
    ++pointer_;
  }
}

template <typename K, typename V, typename F = HashFunction<K>>
class HashTable {
 private:
  il::Array<KeyValue<K, V, F>> bucket_;
  il::int_t nb_entries_;
  il::int_t nb_tombstones_;

 public:
  HashTable(il::int_t nb_entries = 0);
  il::int_t nb_entries() const;
  V value(const K& key, il::io_t, il::Status &status) const;
  void insert(const K& key, const V& value, il::io_t, il::Status &status);
  void erase(const K& key, il::io_t, il::Status &status);
  bool empty() const;
  HashTableIterator<K, V, F> begin();
  HashTableIterator<K, V, F> end();

 private:
  bool search(const K& key, il::io_t, il::int_t& i) const;
  void grow(il::int_t n);
  static il::int_t next_power_of_2(il::int_t i);
  static il::int_t nb_bucket(il::int_t nb_entries);
};

template <typename K, typename V, typename F>
HashTable<K, V, F>::HashTable(il::int_t nb_entries)
    : bucket_{nb_bucket(nb_entries)} {
  nb_entries_ = 0;
  nb_tombstones_ = 0;
}

template <typename K, typename V, typename F>
il::int_t HashTable<K, V, F>::nb_entries() const {
  return nb_entries_;
}

template <typename K, typename V, typename F>
V HashTable<K, V, F>::value(const K& key, il::io_t, il::Status &status) const {
  il::int_t i;
  bool found = search(key, il::io, i);
  if (found) {
    status.set(il::ErrorCode::ok);
    return bucket_[i].value;
  } else {
    status.set(il::ErrorCode::not_found);
    return V{};
  }
}

template <typename K, typename V, typename F>
void HashTable<K, V, F>::insert(const K& key, const V& value, il::io_t,
                                il::Status &status) {
  il::int_t i;
  if (nb_entries_ >= bucket_.size() - 1) {
    grow(nb_bucket(nb_entries_));
  }
  bool found = search(key, il::io, i);
  if (found) {
    status.set(il::ErrorCode::already_there);
    return;
  } else {
    bucket_[i].key = key;
    bucket_[i].value = value;
    ++nb_entries_;
    status.set(il::ErrorCode::ok);
    return;
  }
}

template <typename K, typename V, typename F>
void HashTable<K, V, F>::erase(const K& key, il::io_t, il::Status &status) {
  il::int_t i;
  bool found = search(key, il::io, i);
  if (found) {
    bucket_[i].key = F::tombstone_key();
    bucket_[i].value = V{};
    --nb_entries_;
    ++nb_tombstones_;
    status.set(il::ErrorCode::ok);
    return;
  } else {
    status.set(il::ErrorCode::not_found);
    return;
  }
}

template <typename K, typename V, typename F>
bool HashTable<K, V, F>::empty() const {
  return nb_entries_ == 0;
}

template <typename K, typename V, typename F>
HashTableIterator<K, V, F> HashTable<K, V, F>::begin() {
  if (empty()) {
    return end();
  } else {
    const K empty_key = F::empty_key();
    const K tombstone_key = F::tombstone_key();
    il::int_t i = 0;
    while (true) {
      if (!F::is_equal(bucket_[i].key, empty_key) &&
          !F::is_equal(bucket_[i].key, tombstone_key)) {
        return HashTableIterator<K, V, F>{bucket_.data() + i,
                                          bucket_.data() + bucket_.size()};
      }
      ++i;
    }
  }
}

template <typename K, typename V, typename F>
HashTableIterator<K, V, F> HashTable<K, V, F>::end() {
  return HashTableIterator<K, V, F>{bucket_.data() + bucket_.size(),
                                    bucket_.data() + bucket_.size()};
}

template <typename K, typename V, typename F>
bool HashTable<K, V, F>::search(const K& key, il::io_t, il::int_t& i) const {
  const il::int_t nb_buckets = bucket_.size();
  if (nb_buckets == 0) {
    i = -1;
    return false;
  }

  const K empty_key = F::empty_key();
  const K tombstone_key = F::tombstone_key();
  IL_ASSERT(!F::is_equal(key, empty_key) && !F::is_equal(key, tombstone_key));

  il::int_t k = F::hash_value(key) & (nb_buckets - 1);
  il::int_t k_tombstone = -1;
  il::int_t delta_k = 1;
  while (true) {
    if (F::is_equal(bucket_[k].key, key)) {
      i = k;
      return true;
    }

    if (F::is_equal(bucket_[k].key, empty_key)) {
      i = (k_tombstone == -1) ? k : k_tombstone;
      return false;
    }

    if (F::is_equal(bucket_[k].key, tombstone_key) && k_tombstone == -1) {
      k_tombstone = k;
    }

    k += delta_k;
    ++delta_k;
    k &= (nb_buckets - 1);
  }
}

template <typename K, typename V, typename F>
void HashTable<K, V, F>::grow(il::int_t n) {
  IL_ASSERT(n >= bucket_.size());

  const K empty_key = F::empty_key();
  const K tombstone_key = F::tombstone_key();

  il::Array<KeyValue<K, V, F>> old_bucket{std::move(bucket_)};
  bucket_.resize(n);
  nb_entries_ = 0;
  nb_tombstones_ = 0;

  il::Status status{};
  for (il::int_t i = 0; i < old_bucket.size(); ++i) {
    if (!F::is_equal(old_bucket[i].key, empty_key) &&
        !F::is_equal(old_bucket[i].key, tombstone_key)) {
      insert(old_bucket[i].key, old_bucket[i].value, il::io, status);
      //      IL_ASSERT(!!status.ok());
    }
  }
  status.ignore_error();
}

template <typename K, typename V, typename F>
il::int_t HashTable<K, V, F>::next_power_of_2(il::int_t i) {
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
il::int_t HashTable<K, V, F>::nb_bucket(il::int_t nb_entries) {
  return nb_entries == 0 ? 1 : next_power_of_2(4 * nb_entries / 3 + 1);
}
}

#endif  // IL_HASHTABLE_H
