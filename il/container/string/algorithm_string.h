//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_ALGORITHM_STRING_H
#define IL_ALGORITHM_STRING_H

#include <cstdio>

#include <il/String.h>
#include <il/StringView.h>
#include <il/container/string/view_string.h>

namespace il {

inline il::String toString(il::int_t n) {
  char tmp[11];
  std::sprintf(tmp, "%td", n);
  return il::String{tmp};
}

inline il::ConstStringView removeWhitespaceLeft(ConstStringView string) {
  il::int_t i = 0;
  while (i < string.size() && (string[i] == ' ' || string[i] == '\t')) {
    ++i;
  }
  string.removePrefix(i);

  return string;
}

inline il::int_t search(ConstStringView a, ConstStringView b) {
  const il::int_t na = a.size();
  const il::int_t nb = b.size();
  il::int_t k = 0;
  bool found = false;
  while (!found && k + na <= nb) {
    il::int_t i = 0;
    found = true;
    while (found && i < na) {
      if (a[i] != b[k + i]) {
        found = false;
      }
      ++i;
    }
    if (found) {
      return k;
    }
    ++k;
  }
  return -1;
}

inline il::int_t search(const char* a, ConstStringView b) {
  return il::search(il::view(a), b);
}

inline il::int_t search(const String& a, const String& b) {
  return il::search(il::view(a), il::view(b));
}

inline il::int_t search(const char* a, const String& b) {
  return il::search(ConstStringView{a}, il::view(b));
}

inline il::int_t count(char c, ConstStringView a) {
  IL_EXPECT_FAST(static_cast<unsigned char>(c) < 128);

  il::int_t ans = 0;
  for (il::int_t i = 0; i < a.size(); ++i) {
    if (a[i] == c) {
      ++ans;
    }
  }
  return ans;
}

}  // namespace il

#endif  // IL_ALGORITHM_STRING_H
