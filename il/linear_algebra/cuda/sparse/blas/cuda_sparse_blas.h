//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_CUDA_SPARSE_BLAS_H
#define IL_CUDA_SPARSE_BLAS_H

#include <il/linear_algebra/cuda/sparse/blas/CusparseHandle.h>

namespace il {

inline void blas(float alpha, const il::CudaSparseMatrixCSR<float> &A,
                 const il::CudaArray<float> &x, float beta, il::io_t,
                 il::CudaArray<float> &y, il::CusparseHandle &handle) {
  IL_ASSERT_PRECOND(A.size(1) == x.size());
  IL_ASSERT_PRECOND(A.size(0) == y.size());

  const cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  const int m = static_cast<int>(A.size(0));
  const int n = static_cast<int>(A.size(1));
  const int nnz = static_cast<int>(A.nb_nonzero());
  cusparseMatDescr_t descrA;
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

  const cusparseStatus_t status = cusparseScsrmv(
      handle.handle(), transA, m, n, nnz, &alpha, descrA, A.element_data(),
      A.row_data(), A.column_data(), x.data(), &beta, y.data());
  IL_ASSERT(status == CUSPARSE_STATUS_SUCCESS);
}

}

#endif  // IL_CUDA_SPARSE_BLAS_H
