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
