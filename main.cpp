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

#include <il/Array.h>
#include <il/Array2D.h>
#include <il/Array2C.h>
#include <il/Array3D.h>
#include <il/Array3C.h>

int main() {
  il::Array<il::int_t> z{il::value, {1, 2, 3}};
  il::Array<double> a{il::value, {1.0, 2.0, 3.0}};
  il::Array2D<double> b{il::value, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}};
  il::Array2C<double> c{il::value, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}};
  il::Array3D<double> d{il::value, {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
                                    {{11.0, 12.0, 13.0}, {14.0, 15.0, 16.0}}}};
  il::Array3C<double> e{il::value, {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
                                    {{11.0, 12.0, 13.0}, {14.0, 15.0, 16.0}}}};

  return 0;
}
