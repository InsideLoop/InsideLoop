//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_FORMAT_IL_H
#define IL_FORMAT_IL_H

#include <il/io/format/format.h>
#include <il/String.h>
#include <string>

namespace il {

template <typename... Args>
il::String format(Args&&... args) {
  std::string s = fmt::format(args...);
  il::String ans{s.data(), static_cast<il::int_t>(s.size())};
  return ans;
}

}

#endif  // IL_FORMAT_IL_H
