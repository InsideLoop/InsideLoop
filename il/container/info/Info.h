//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_INFO_H
#define IL_INFO_H

#include <il/base.h>
#include <cstring>

namespace il {

class Info {
 private:
  struct Large {
    unsigned char* data;
    std::size_t size;
    std::size_t capacity;
  };
  union {
    unsigned char small_[sizeof(Large)];
    Large large_;
  };
  constexpr static il::int_t max_small_size_ =
      static_cast<il::int_t>(sizeof(Large)) - 1;
  constexpr static unsigned char category_extract_mask_ = 0x80;
  constexpr static std::size_t capacity_extract_mask_ =
      ~(static_cast<std::size_t>(0x80) << (8 * (sizeof(std::size_t) - 1)));

 public:
  Info();
  Info(const Info& other);
  Info(Info&& other);
  Info& operator=(const Info& other);
  Info& operator=(Info&& other);

  bool empty() const;
  void clear();

  void set(const char* key, bool value);
  void set(const char* key, int value);
  void set(const char* key, il::int_t value);
  void set(const char* key, double value);
  void set(const char* key, const char* value);

  bool to_bool(const char* key) const;
  int to_int(const char* key) const;
  il::int_t to_integer(const char* key) const;
  double to_double(const char* key) const;
  const char* as_c_string(const char* key) const;

  il::int_t search(const char* key) const;
  bool found(il::int_t i) const;

  bool is_bool(il::int_t i) const;
  bool is_int(il::int_t i) const;
  bool is_integer(il::int_t i) const;
  bool is_double(il::int_t i) const;
  bool is_string(il::int_t i) const;

  bool to_bool(il::int_t i) const;
  int to_int(il::int_t i) const;
  il::int_t to_integer(il::int_t i) const;
  double to_double(il::int_t i) const;
  const char* as_c_string(il::int_t i) const;

 private:
  il::int_t size() const;
  il::int_t capacity() const;
  const unsigned char* data() const;
  unsigned char* data();
  void resize(il::int_t n);

  bool small() const;
  void set_small_size(il::int_t n);
  void set_large_capacity(il::int_t r);
  il::int_t large_capacity() const;

  void set(int value, unsigned char* data);
  void set(il::int_t value, unsigned char* data);
  void set(double value, unsigned char* data);

  int get_int(il::int_t i) const;
  il::int_t get_integer(il::int_t i) const;
};

inline Info::Info() { set_small_size(0); }

inline Info::Info(const Info& other) {
  const il::int_t size = other.size();
  if (size <= max_small_size_) {
    std::memcpy(small_, other.data(), size);
    set_small_size(size);
  } else {
    large_.data = new unsigned char[size];
    std::memcpy(large_.data, other.data(), size);
    large_.size = size;
    set_large_capacity(size);
  }
}

inline Info::Info(Info&& other) {
  const il::int_t size = other.size();
  if (size <= max_small_size_) {
    std::memcpy(small_, other.data(), size);
    set_small_size(size);
  } else {
    large_ = other.large_;
    other.set_small_size(0);
  }
}

inline Info& Info::operator=(const Info& other) {
  const il::int_t size = other.size();
  if (size <= max_small_size_) {
    if (!small()) {
      delete[] large_.data;
    }
    std::memcpy(small_, other.data(), size);
    set_small_size(size);
  } else {
    if (size <= capacity()) {
      std::memcpy(large_.data, other.data(), size);
      large_.size = size;
    } else {
      if (!small()) {
        delete[] large_.data;
      }
      large_.data = new unsigned char[size];
      std::memcpy(large_.data, other.data(), size);
      large_.size = size;
      set_large_capacity(size);
    }
  }
  return *this;
}

inline Info& Info::operator=(Info&& other) {
  if (this != &other) {
    const il::int_t size = other.size();
    if (size <= max_small_size_) {
      if (!small()) {
        delete[] large_.data;
      }
      std::memcpy(small_, other.data(), size);
      set_small_size(size);
    } else {
      large_ = other.large_;
      other.set_small_size(0);
    }
  }
  return *this;
}

inline void Info::set(const char* key, int value) {
  const int key_length = static_cast<int>(strlen(key));
  const int n = key_length + 2 + 2 * static_cast<int>(sizeof(int));
  il::int_t i = size();
  resize(i + n);

  unsigned char* p = data();

  set(n, p + i);
  i += sizeof(int);

  for (il::int_t j = 0; j < key_length + 1; ++j) {
    p[i + j] = key[j];
  }
  i += key_length + 1;

  p[i] = 3;
  ++i;

  set(value, p + i);
}

inline void Info::set(const char* key, il::int_t value) {
  const int key_length = static_cast<int>(strlen(key));
  const int n =
      key_length + 2 + static_cast<int>(sizeof(int) + sizeof(il::int_t));
  il::int_t i = size();
  resize(i + n);

  unsigned char* p = data();

  set(n, p + i);
  i += sizeof(int);

  for (il::int_t j = 0; j < key_length + 1; ++j) {
    p[i + j] = key[j];
  }
  i += key_length + 1;

  p[i] = 0;
  ++i;

  set(value, p + i);
}

inline void Info::set(const char* key, double value) {
  const int key_length = static_cast<int>(strlen(key));
  const int n = key_length + 2 + static_cast<int>(sizeof(int) + sizeof(double));
  il::int_t i = size();
  resize(i + n);

  unsigned char* p = data();

  set(n, p + i);
  i += sizeof(int);

  for (il::int_t j = 0; j < key_length + 1; ++j) {
    p[i + j] = key[j];
  }
  i += key_length + 1;

  p[i] = 1;
  ++i;

  set(value, p + i);
}

inline void Info::set(const char* key, const char* value) {
  const int key_length = static_cast<int>(strlen(key));
  const int value_length = static_cast<int>(strlen(value));
  const int n = key_length + value_length + 3 + static_cast<int>(sizeof(int));
  il::int_t i = size();
  resize(i + n);

  unsigned char* p = data();

  set(n, p + i);
  i += sizeof(int);

  for (il::int_t j = 0; j < key_length + 1; ++j) {
    p[i + j] = key[j];
  }
  i += key_length + 1;

  p[i] = 2;
  ++i;

  for (il::int_t j = 0; j < value_length + 1; ++j) {
    p[i + j] = value[j];
  }
}

inline il::int_t Info::search(const char* key) const {
  const unsigned char* p = data();
  il::int_t i = 0;
  bool found = false;
  il::int_t j;
  while (!found && i < size()) {
    const int step = get_int(i);

    j = 0;
    while (key[j] != '\0' && p[i + sizeof(int) + j] == key[j]) {
      ++j;
    }
    if (p[i + sizeof(int) + j] == '\0') {
      found = true;
    } else {
      i += step;
    }
  }

  return found ? i + sizeof(int) + j + 1 : -1;
}

inline bool Info::found(il::int_t i) const { return i >= 0; }

inline bool Info::is_int(il::int_t i) const { return data()[i] == 3; }

inline bool Info::is_integer(il::int_t i) const { return data()[i] == 0; }

inline bool Info::is_double(il::int_t i) const { return data()[i] == 1; }

inline bool Info::is_string(il::int_t i) const { return data()[i] == 2; }

inline int Info::to_int(il::int_t i) const {
  IL_EXPECT_FAST(is_int(i));

  const unsigned char* p = data();
  const il::int_t n = static_cast<il::int_t>(sizeof(int));
  union {
    int local_value;
    unsigned char local_raw[n];
  };
  for (il::int_t j = 0; j < n; ++j) {
    local_raw[j] = p[i + 1 + j];
  }
  return local_value;
}

inline il::int_t Info::to_integer(il::int_t i) const {
  IL_EXPECT_FAST(is_integer(i));

  const unsigned char* p = data();
  const il::int_t n = static_cast<il::int_t>(sizeof(il::int_t));
  union {
    il::int_t local_value;
    unsigned char local_raw[n];
  };
  for (il::int_t j = 0; j < n; ++j) {
    local_raw[j] = p[i + 1 + j];
  }
  return local_value;
}

inline double Info::to_double(il::int_t i) const {
  IL_EXPECT_FAST(is_double(i));

  const unsigned char* p = data();
  const il::int_t n = static_cast<il::int_t>(sizeof(double));
  union {
    double local_value;
    unsigned char local_raw[n];
  };
  for (il::int_t j = 0; j < n; ++j) {
    local_raw[j] = p[i + 1 + j];
  }
  return local_value;
}

inline const char* Info::as_c_string(il::int_t i) const {
  IL_EXPECT_FAST(is_string(i));

  return reinterpret_cast<const char*>(data()) + i + 1;
}

inline int Info::to_int(const char* key) const {
  const il::int_t i = search(key);
  IL_ENSURE(found(i));

  return to_int(i);
}

inline il::int_t Info::to_integer(const char* key) const {
  const il::int_t i = search(key);
  IL_ENSURE(found(i));

  return to_integer(i);
}

inline double Info::to_double(const char* key) const {
  const il::int_t i = search(key);
  IL_ENSURE(found(i));

  return to_double(i);
}

inline const char* Info::as_c_string(const char* key) const {
  const il::int_t i = search(key);
  IL_ENSURE(found(i));

  return as_c_string(i);
}

inline void Info::set(int value, unsigned char* data) {
  const il::int_t n = static_cast<il::int_t>(sizeof(int));
  union {
    int local_value;
    unsigned char local_raw[n];
  };
  local_value = value;
  for (int j = 0; j < n; ++j) {
    data[j] = local_raw[j];
  }
}

inline void Info::set(il::int_t value, unsigned char* data) {
  const il::int_t n = static_cast<il::int_t>(sizeof(il::int_t));
  union {
    il::int_t local_value;
    unsigned char local_raw[n];
  };
  local_value = value;
  for (il::int_t j = 0; j < n; ++j) {
    data[j] = local_raw[j];
  }
}

inline void Info::set(double value, unsigned char* data) {
  const il::int_t n = static_cast<il::int_t>(sizeof(double));
  union {
    double local_value;
    unsigned char local_raw[n];
  };
  local_value = value;
  for (il::int_t j = 0; j < n; ++j) {
    data[j] = local_raw[j];
  }
}

inline int Info::get_int(il::int_t i) const {
  const unsigned char* p = data();
  const il::int_t n = static_cast<il::int_t>(sizeof(int));
  union {
    int local_value;
    unsigned char local_raw[n];
  };
  for (il::int_t j = 0; j < n; ++j) {
    local_raw[j] = p[i + j];
  }
  return local_value;
}

inline il::int_t Info::get_integer(il::int_t i) const {
  const unsigned char* p = data();
  const il::int_t n = static_cast<il::int_t>(sizeof(il::int_t));
  union {
    il::int_t local_value;
    unsigned char local_raw[n];
  };
  for (il::int_t j = 0; j < n; ++j) {
    local_raw[j] = p[i + j];
  }
  return local_value;
}

inline il::int_t Info::size() const {
  return small() ? small_[max_small_size_] : large_.size;
}

inline il::int_t Info::capacity() const {
  return small() ? max_small_size_ : large_capacity();
}

inline bool Info::small() const {
  return (small_[max_small_size_] & category_extract_mask_) == 0;
}

inline void Info::set_small_size(il::int_t n) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(n) <=
                   static_cast<std::size_t>(max_small_size_));

  small_[max_small_size_] = static_cast<unsigned char>(n);
}

inline void Info::set_large_capacity(il::int_t r) {
  large_.capacity =
      static_cast<std::size_t>(r) |
      (static_cast<std::size_t>(0x80) << ((sizeof(std::size_t) - 1) * 8));
}

inline il::int_t Info::large_capacity() const {
  return large_.capacity & capacity_extract_mask_;
}

inline const unsigned char* Info::data() const {
  return small() ? small_ : large_.data;
}

inline unsigned char* Info::data() { return small() ? small_ : large_.data; }

inline bool Info::empty() const { return size() == 0; }

inline void Info::resize(il::int_t n) {
  IL_EXPECT_FAST(n >= 0);

  const il::int_t old_size = size();
  if (small()) {
    if (n <= max_small_size_) {
      set_small_size(n);
    } else {
      unsigned char* new_data = new unsigned char[n];
      std::memcpy(new_data, small_, old_size);
      large_.data = new_data;
      large_.size = n;
      set_large_capacity(n);
    }
  } else {
    if (n <= capacity()) {
      large_.size = n;
    } else {
      unsigned char* new_data = new unsigned char[n];
      std::memcpy(new_data, large_.data, old_size);
      delete[] large_.data;
      large_.data = new_data;
      large_.size = n;
      set_large_capacity(n);
    }
  }
}

inline void Info::clear() {
  if (!small()) {
    delete[] large_.data;
  }
  set_small_size(0);
}

}  // namespace il

#endif  // IL_INFO_H
