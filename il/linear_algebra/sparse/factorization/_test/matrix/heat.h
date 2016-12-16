//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_HEAT_H
#define IL_HEAT_H

#include <il/SparseArray2C.h>

namespace il {

void heat_1d(il::int_t n, il::io_t, il::SparseArray2C<double>& A,
             il::Array<double>& y);
void heat_2d(il::int_t n, il::io_t, il::SparseArray2C<double>& A,
             il::Array<double>& y);
void heat_3d(il::int_t n, il::io_t, il::SparseArray2C<double>& A,
             il::Array<double>& y);

}


#endif  // IL_HEAT_H
