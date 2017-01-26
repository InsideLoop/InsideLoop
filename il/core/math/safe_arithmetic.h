//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_SAFE_ARITHMETIC_H
#define IL_SAFE_ARITHMETIC_H

#include <limits>

#include <il/base.h>
#include <il/core/Status.h>

namespace il {

////////////////////////////////////////////////////////////////////////////////
// int
////////////////////////////////////////////////////////////////////////////////

// We use the fact that long long is stricly wider than int which is not part
// of the standard.
// - The C++11 standard states that long long is at least 64 bits
// - All data models (LP32, ILP32, LLP64, LP64) work with an int that is 32 bits
//   or less
inline int safe_sum(int a, int b, il::io_t, bool& error) {
  const long long ll_a = a;
  const long long ll_b = b;
  const long long ll_sum = ll_a + ll_b;
  if (ll_sum > std::numeric_limits<int>::max()) {
    error = true;
    return 0;
  } else if (ll_sum < std::numeric_limits<int>::min()) {
    error = true;
    return 0;
  } else {
    return static_cast<int>(ll_sum);
  }
}

// We use the fact that long long is stricly wider than int (check comment on
// safe_add).
inline int safe_substraction(int a, int b, il::io_t, bool& error) {
  const long long ll_a = a;
  const long long ll_b = b;
  const long long ll_sum = ll_a - ll_b;
  if (ll_sum > std::numeric_limits<int>::max()) {
    error = true;
    return 0;
  } else if (ll_sum < std::numeric_limits<int>::min()) {
    error = true;
    return 0;
  } else {
    return static_cast<int>(ll_sum);
  }
}

// We use the fact that long long is stricly wider than int (check comment on
// safe_add).
inline int safe_product(int a, int b, il::io_t, bool& error) {
  const long long ll_a = a;
  const long long ll_b = b;
  const long long ll_sum = ll_a * ll_b;
  if (ll_sum > std::numeric_limits<int>::max()) {
    error = true;
    return 0;
  } else if (ll_sum < std::numeric_limits<int>::min()) {
    error = true;
    return 0;
  } else {
    return static_cast<int>(ll_sum);
  }
}

inline int safe_division(int a, int b, il::io_t, bool& error) {
  if (b == 0 || (b == -1 && a == std::numeric_limits<int>::min())) {
    error = true;
    return 0;
  } else {
    return a / b;
  }
}

////////////////////////////////////////////////////////////////////////////////
// il::int_t
////////////////////////////////////////////////////////////////////////////////

inline il::int_t safe_sum(il::int_t a, il::int_t b, il::io_t, bool& error) {
  if ((b > 0 && a > std::numeric_limits<il::int_t>::max() - b) ||
      (b < 0 && a < std::numeric_limits<il::int_t>::min() - b)) {
    error = true;
    return 0;
  } else {
    return a + b;
  }
}

inline il::int_t safe_substraction(il::int_t a, il::int_t b, il::io_t,
                                   bool& error) {
  if ((b > 0 && a < std::numeric_limits<il::int_t>::min() + b) ||
      (b < 0 && a > std::numeric_limits<il::int_t>::max() + b)) {
    error = true;
    return 0;
  } else {
    return a - b;
  }
}

inline il::int_t safe_product(il::int_t a, il::int_t b, il::io_t, bool& error) {
  if (a > 0) {
    if (b > 0) {
      if (a > std::numeric_limits<il::int_t>::max() / b) {
        error = true;
        return 0;
      }
    } else {
      if (b < std::numeric_limits<il::int_t>::min() / a) {
        error = true;
        return 0;
      }
    }
  } else {
    if (b > 0) {
      if (a < std::numeric_limits<il::int_t>::min() / b) {
        error = true;
        return 0;
      }
    } else {
      if (a != 0 && b < std::numeric_limits<il::int_t>::max() / a) {
        error = true;
        return 0;
      }
    }
  }
  return a * b;
}

inline il::int_t safe_division(il::int_t a, il::int_t b, il::io_t,
                               bool& error) {
  if (b == 0 || (b == -1 && a == std::numeric_limits<il::int_t>::min())) {
    error = true;
    return 0;
  } else {
    return a / b;
  }
}

////////////////////////////////////////////////////////////////////////////////
// std::size_t
////////////////////////////////////////////////////////////////////////////////

inline std::size_t safe_sum(std::size_t a, std::size_t b, il::io_t,
                            bool& error) {
  if (a > (std::numeric_limits<std::size_t>::max() - b)) {
    error = true;
    return 0;
  }
  return a + b;
}

inline std::size_t safe_product(std::size_t a, std::size_t b, il::io_t,
                                bool& error) {
  if (b > 0) {
    if (a > std::numeric_limits<std::size_t>::max() / b) {
      error = true;
      return 0;
    }
  }
  error = error || false;
  return a * b;
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

template <typename T1, typename T2>
T1 safe_convert(T2 n, il::io_t, bool& error) {
  error = true;
  return T1{};
};

template <>
inline il::int_t safe_convert(std::size_t n, il::io_t, bool& error) {
  if (n > std::numeric_limits<il::int_t>::max()) {
    error = true;
    return 0;
  }
  return static_cast<il::int_t>(n);
}

template <>
inline std::size_t safe_convert(il::int_t n, il::io_t, bool& error) {
  if (n < 0) {
    error = true;
    return 0;
  }
  return static_cast<std::size_t>(n);
}
}
#endif  // IL_SAFE_ARITHMETIC_H
