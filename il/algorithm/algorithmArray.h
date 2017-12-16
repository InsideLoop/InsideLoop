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

#ifndef IL_ALGORITHM_ARRAY_H
#define IL_ALGORITHM_ARRAY_H

#include <algorithm>

#include <il/Array.h>

namespace il {

template <typename T>
struct MinMax {
  T min;
  T max;
};

template <typename T>
MinMax<T> minMax(const il::Array<T>& v, il::int_t i_begin, il::int_t i_end) {
  IL_EXPECT_FAST(i_begin < i_end);

  MinMax<T> ans{};
  ans.min = std::numeric_limits<T>::max();
  ans.max = -std::numeric_limits<T>::max();
  for (il::int_t i = i_begin; i < i_end; ++i) {
    if (v[i] < ans.min) {
      ans.min = v[i];
    }
    if (v[i] > ans.max) {
      ans.max = v[i];
    }
  }
  return ans;
}


//template <typename T, typename F>
//bool allOf(const il::Array<T>& v, const F& f) {
//  il::int_t i = 0;
//  while (i < v.size() && f(v[i])) {
//    ++i;
//  }
//  return (i == v.size());
//};
//
//template <typename T, typename F>
//bool anyOf(const il::Array<T>& v, const F& f) {
//  il::int_t i = 0;
//  while (i < v.size() && !f(v[i])) {
//    ++i;
//  }
//  return (i < v.size());
//};
//
//template <typename T>
//bool count(const il::Array<T>& v, const T& x) {
//  il::int_t ans = 0;
//  for (il::int_t i = 0; i < v.size(); ++i) {
//    if (v[i] == x) {
//      ++ans;
//    }
//  }
//  return ans;
//};
//
//template <typename T, typename F>
//bool count(const il::Array<T>& v, const F& f) {
//  il::int_t ans = 0;
//  for (il::int_t i = 0; i < v.size(); ++i) {
//    if (f(v[i])) {
//      ++ans;
//    }
//  }
//  return ans;
//};

//template <typename T>
//il::Array<il::int_t> searchAll(const il::Array<T>& v, const T& x) {
//  il::Array<il::int_t> ans{};
//  for (il::int_t i = 0; i < v.size(); ++i) {
//    if (v[i] == x) {
//      ans.append(i);
//    }
//  }
//  return ans;
//};

//template <typename T, typename F>
//il::Array<il::int_t> searchAll(const il::Array<T>& v, const F& f) {
//  il::Array<il::int_t> ans{};
//  for (il::int_t i = 0; i < v.size(); ++i) {
//    if (f(v[i])) {
//      ans.append(i);
//    }
//  }
//  return ans;
//};


template <typename T>
il::int_t search(const il::Array<T>& v, const T& x) {
  il::int_t i = 0;
  while (i < v.size() && v[i] != x) {
    ++i;
  }
  return (i < v.size()) ? i : -1;
};

template <typename T, typename F>
il::int_t search(const il::Array<T>& v, const F& f) {
  il::int_t i = 0;
  while (i < v.size() && !f(v[i])) {
    ++i;
  }
  return (i < v.size()) ? i : -1;
};

template <typename T>
T min(const il::Array<T>& v) {
  IL_EXPECT_FAST(v.size() > 0);

  il::int_t min_value = v[0];
  for (il::int_t i = 1; i < v.size(); ++i) {
    if (v[i] < min_value) {
      min_value = v[i];
    }
  }
  return min_value;
}

template <typename T>
T max(const il::Array<T>& v) {
  IL_EXPECT_FAST(v.size() > 0);

  il::int_t max_value = v[0];
  for (il::int_t i = 1; i < v.size(); ++i) {
    if (v[i] > max_value) {
      max_value = v[i];
    }
  }
  return max_value;
}

template <typename T>
il::int_t findMin(const il::Array<T>& v) {
  IL_EXPECT_FAST(v.size() > 0);

  il::int_t min_index = 0;
  T min_value = v[0];
  for (il::int_t i = 1; i < v.size(); ++i) {
    if (v[i] < min_value) {
      min_index = i;
      min_value = v[i];
    }
  }
  return min_index;
}

template <typename T>
il::int_t findMax(const il::Array<T>& v) {
  IL_EXPECT_FAST(v.size() > 0);

  il::int_t max_index = 0;
  T max_value = v[0];
  for (il::int_t i = 1; i < v.size(); ++i) {
    if (v[i] > max_value) {
      max_index = i;
      max_value = v[i];
    }
  }
  return max_index;
}

template <typename T>
T mean(const il::Array<T>& v) {
  IL_EXPECT_FAST(v.size() > 0);

  T ans = 0;
  for (il::int_t i = 0; i < v.size(); ++i) {
    ans += v[i];
  }
  ans /= v.size();
  return ans;
}

//template <typename T>
//T variance(const il::Array<T>& v) {
//  IL_EXPECT_FAST(v.size() > 1);
//
//  T mean = 0;
//  for (il::int_t i = 0; i < v.size(); ++i) {
//    mean += v[i];
//  }
//  mean /= v.size();
//  T variance = 0;
//  for (il::int_t i = 0; i < v.size(); ++i) {
//    variance += (v[i] - mean) * (v[i] - mean);
//  }
//  variance /= v.size();
//  return variance;
//}

//template <typename T, typename F>
//void map(const F& f, il::io_t, il::Array<T>& v) {
//  for (il::int_t i = 0; i < v.size(); ++i) {
//    v[i] = f(v[i]);
//  }
//};

template <typename T>
void sort(il::io_t, il::Array<T>& v) {
  std::sort(v.begin(), v.end());
}

template <typename T>
il::Array<T> sort(const il::Array<T>& v) {
  il::Array<T> ans = v;
  std::sort(ans.begin(), ans.end());
  return ans;
}

template <typename T>
il::int_t binarySearch(const il::Array<T>& v, const T& x) {
  il::int_t i_begin = 0;
  il::int_t i_end = v.size();
  while (i_end > i_begin + 1) {
    const il::int_t i = i_begin + (i_end - i_begin) / 2;
    if (v[i] <= x) {
      i_begin = i;
    } else {
      i_end = i;
    }
  }
  return (i_end == i_begin + 1 && v[i_begin] == x) ? i_begin : -1;
}

}  // namespace il

#endif  // IL_ALGORITHM_ARRAY_H
