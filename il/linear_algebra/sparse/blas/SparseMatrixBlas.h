//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_SPARSEMATRIXOPTIMIZED_H
#define IL_SPARSEMATRIXOPTIMIZED_H

#include <il/SparseMatrixCSR.h>

#ifdef IL_MKL
#include <mkl_spblas.h>

namespace il {

template <typename Index, typename T>
class SparseMatrixBlas {
 private:
  sparse_matrix_t sparse_matrix_handle_;

 public:
  SparseMatrixBlas(il::io_t, il::SparseMatrixCSR<Index, T> &A);
  ~SparseMatrixBlas();
  void set_nb_matrix_vector(il::int_t n);
  sparse_matrix_t handle();
};

template <typename Index, typename T>
SparseMatrixBlas<Index, T>::SparseMatrixBlas(il::io_t,
                                             il::SparseMatrixCSR<Index, T> &A) {
  const sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
  const MKL_INT rows = A.size(0);
  const MKL_INT cols = A.size(1);
  MKL_INT *rows_start = A.row_data();
  MKL_INT *rows_end = A.row_data() + 1;
  MKL_INT *col_indx = A.column_data();
  T *values = A.element_data();

  sparse_status_t status =
      mkl_sparse_d_create_csr(&sparse_matrix_handle_, indexing, rows, cols,
                              rows_start, rows_end, col_indx, values);
  IL_EXPECT_FAST(status == SPARSE_STATUS_SUCCESS);
}

template <typename Index, typename T>
SparseMatrixBlas<Index, T>::~SparseMatrixBlas() {
  sparse_status_t status = mkl_sparse_destroy(sparse_matrix_handle_);
  IL_EXPECT_FAST(status == SPARSE_STATUS_SUCCESS);
}

template <typename Index, typename T>
void SparseMatrixBlas<Index, T>::set_nb_matrix_vector(il::int_t n) {
  IL_EXPECT_FAST(n > 0);

  const sparse_operation_t operation = SPARSE_OPERATION_NON_TRANSPOSE;
  matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  sparse_status_t status =
      mkl_sparse_set_mv_hint(sparse_matrix_handle_, operation, descr, n);
  IL_EXPECT_FAST(status == SPARSE_STATUS_SUCCESS);
}

template <typename Index, typename T>
sparse_matrix_t SparseMatrixBlas<Index, T>::handle() {
  return sparse_matrix_handle_;
}

/////// For float

template <typename Index>
class SparseMatrixBlas<Index, float> {
 private:
  sparse_matrix_t sparse_matrix_handle_;

 public:
  SparseMatrixBlas(il::io_t, il::SparseMatrixCSR<Index, float> &A);
  ~SparseMatrixBlas();
  void set_nb_matrix_vector(il::int_t n);
  sparse_matrix_t handle();
};

template <typename Index>
SparseMatrixBlas<Index, float>::SparseMatrixBlas(
    il::io_t, il::SparseMatrixCSR<Index, float> &A) {
  const sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
  const MKL_INT rows = A.size(0);
  const MKL_INT cols = A.size(1);
  MKL_INT *rows_start = A.row_data();
  MKL_INT *rows_end = A.row_data() + 1;
  MKL_INT *col_indx = A.column_data();
  float *values = A.element_data();

  sparse_status_t status =
      mkl_sparse_s_create_csr(&sparse_matrix_handle_, indexing, rows, cols,
                              rows_start, rows_end, col_indx, values);
  IL_EXPECT_FAST(status == SPARSE_STATUS_SUCCESS);
}

template <typename Index>
SparseMatrixBlas<Index, float>::~SparseMatrixBlas() {
  sparse_status_t status = mkl_sparse_destroy(sparse_matrix_handle_);
  IL_EXPECT_FAST(status == SPARSE_STATUS_SUCCESS);
}

template <typename Index>
void SparseMatrixBlas<Index, float>::set_nb_matrix_vector(il::int_t n) {
  IL_EXPECT_FAST(n > 0);

  const sparse_operation_t operation = SPARSE_OPERATION_NON_TRANSPOSE;
  matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  sparse_status_t status =
      mkl_sparse_set_mv_hint(sparse_matrix_handle_, operation, descr, n);
  IL_EXPECT_FAST(status == SPARSE_STATUS_SUCCESS);
}

template <typename Index>
sparse_matrix_t SparseMatrixBlas<Index, float>::handle() {
  return sparse_matrix_handle_;
}

}  // namespace il

#endif  // IL_MKL

#endif  // IL_SPARSEMATRIXOPTIMIZED_H
