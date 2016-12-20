//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_SPARSE_DOT_H
#define IL_SPARSE_DOT_H

#include <il/SparseMatrixCSR.h>

#ifdef IL_MKL
#include <mkl_spblas.h>
#endif

namespace il {

////////////////////////////////////////////////////////////////////////////////
// BLAS Level 2
////////////////////////////////////////////////////////////////////////////////

// A is a matrix, x, y are vectors
// C <- A.B
inline il::SparseMatrixCSR<int, double> dot(il::io_t,
                                          il::SparseMatrixCSR<int, double> &A,
                                          il::SparseMatrixCSR<int, double> &B) {
  IL_ASSERT_PRECOND(A.size(1) == B.size(0));

  const char trans = 'n';
  const MKL_INT sort = 0;
  const MKL_INT m = A.size(0);
  const MKL_INT n = A.size(1);
  const MKL_INT k = B.size(1);
  const MKL_INT nzmax = 0;
  MKL_INT info;

  il::Array<double> element{};
  il::Array<int> column{};
  il::Array<int> row{m + 1};
  MKL_INT request = 1;

  int *A_row_data = A.row_data();
  int *A_column_data = A.column_data();
  for (int i = 0; i <= A.size(0); ++i) {
    ++A_row_data[i];
  }
  for (int i = 0; i <= A.nb_nonzeros(); ++i) {
    ++A_column_data[i];
  }
  int *B_row_data = B.row_data();
  int *B_column_data = B.column_data();
  if (&A != &B) {
    for (int i = 0; i <= B.size(0); ++i) {
      ++B_row_data[i];
    }
    for (int i = 0; i <= B.nb_nonzeros(); ++i) {
      ++B_column_data[i];
    }
  }

  mkl_dcsrmultcsr(&trans, &request, &sort, &m, &n, &k, A.element_data(),
                  A.column_data(), A.row_data(), B.element_data(),
                  B.column_data(), B.row_data(), element.data(), column.data(),
                  row.data(), &nzmax, &info);
  IL_ASSERT(info == 0);

  element.resize(row[m] - 1);
  column.resize(row[m] - 1);
  request = 2;
  mkl_dcsrmultcsr(&trans, &request, &sort, &m, &n, &k, A.element_data(),
                  A.column_data(), A.row_data(), B.element_data(),
                  B.column_data(), B.row_data(), element.data(), column.data(),
                  row.data(), &nzmax, &info);
  IL_ASSERT(info == 0);

  for (int i = 0; i <= A.size(0); ++i) {
    --A_row_data[i];
  }
  for (int i = 0; i <= A.nb_nonzeros(); ++i) {
    --A_column_data[i];
  }
  if (&A != &B) {
    for (int i = 0; i <= B.size(0); ++i) {
      --B_row_data[i];
    }
    for (int i = 0; i <= B.nb_nonzeros(); ++i) {
      --B_column_data[i];
    }
  }
  for (int i = 0; i < row.size(); ++i) {
    --row[i];
  }
  for (int i = 0; i < column.size(); ++i) {
    --column[i];
  }

  return il::SparseMatrixCSR<int, double>{m, k, std::move(column), std::move(row),
                                        std::move(element)};
}

}

#endif  // IL_SPARSE_DOT_H
