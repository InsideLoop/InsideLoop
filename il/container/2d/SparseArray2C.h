//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_SPARSEARRAY2C_H
#define IL_SPARSEARRAY2C_H

#include <il/Array.h>
#include <il/Array2C.h>
#include <il/SmallArray.h>
#include <il/StaticArray.h>
#include <il/linear_algebra/dense/blas/norm.h>
#include <il/math.h>

namespace il {

template <typename T>
class SparseArray2C {
 private:
  il::int_t n0_;
  il::int_t n1_;
  il::Array<T> element_;
  il::Array<il::int_t> column_;
  il::Array<il::int_t> row_;

 public:
  SparseArray2C();
  SparseArray2C(il::int_t height, il::int_t width, il::Array<il::int_t> column,
                il::Array<il::int_t> row);
  SparseArray2C(il::int_t height, il::int_t width, il::Array<il::int_t> column,
                il::Array<il::int_t> row, il::Array<double> element);
  template <il::int_t n>
  SparseArray2C(il::int_t width, il::int_t height,
                const il::Array<il::SmallArray<il::int_t, n>>& column);
  SparseArray2C(il::int_t n,
                const il::Array<il::StaticArray<il::int_t, 2>>& position,
                il::io_t, il::Array<il::int_t>& index);
  const T& operator[](il::int_t k) const;
  T& operator[](il::int_t k);
  const T& operator()(il::int_t i, il::int_t k) const;
  T& operator()(il::int_t i, il::int_t k);
  T element(il::int_t k) const;
  il::int_t row(il::int_t i) const;
  il::int_t column(il::int_t k) const;
  il::int_t size(il::int_t d) const;
  il::int_t nb_nonzeros() const;
  const T* element_data() const;
  T* element_data();
  const il::int_t* row_data() const;
  il::int_t* row_data();
  const il::int_t* column_data() const;
  il::int_t* column_data();
};

template <typename T>
SparseArray2C<T>::SparseArray2C() : element_{}, column_{}, row_{} {
  n0_ = 0;
  n1_ = 0;
}

template <typename T>
SparseArray2C<T>::SparseArray2C(il::int_t height, il::int_t width,
                                il::Array<il::int_t> column,
                                il::Array<il::int_t> row)
    : n0_{height},
      n1_{width},
      element_{column.size()},
      column_{std::move(column)},
      row_{std::move(row)} {
  IL_ASSERT(row_.size() == height + 1);
}

template <typename T>
SparseArray2C<T>::SparseArray2C(il::int_t height, il::int_t width,
                                il::Array<il::int_t> column,
                                il::Array<il::int_t> row,
                                il::Array<double> element)
    : n0_{height},
      n1_{width},
      element_{std::move(element)},
      column_{std::move(column)},
      row_{std::move(row)} {
  IL_ASSERT(row_.size() == height + 1);
}

template <typename T>
template <il::int_t n>
SparseArray2C<T>::SparseArray2C(
    il::int_t width, il::int_t height,
    const il::Array<il::SmallArray<il::int_t, n>>& column)
    : n1_{width}, n0_{height}, element_{}, column_{}, row_{height + 1} {
  il::int_t nb_nonzero{0};
  for (il::int_t i{0}; i < column.size(); ++i) {
    nb_nonzero += column[i].size();
  }

  element_.resize(nb_nonzero);
  column_.resize(nb_nonzero);
  row_[0] = 0;
  for (il::int_t i{0}; i < column.size(); ++i) {
    for (il::int_t k{0}; k < column[i].size(); ++k) {
      column_[row_[i] + k] = column[i][k];
    }
    row_[i + 1] = row_[i] + column[i].size();
  }
}

template <typename T>
SparseArray2C<T>::SparseArray2C(
    il::int_t n, const il::Array<il::StaticArray<il::int_t, 2>>& position,
    il::io_t, il::Array<il::int_t>& index)
    : element_{}, column_{}, row_{} {
  IL_ASSERT(n >= 0);
  //
  // element_
  // column_
  // row_
  n0_ = n;
  n1_ = n;

  const il::int_t nb_entries = position.size();
  index.resize(nb_entries);

  // Compute the numbers of entries per Row. After this section, the
  // array nb_entries_per_Row will contains the numbers of entries in
  // the row i. The variable max_entries_per_row will contain the
  //  maximum number of entry in a row.
  il::Array<il::int_t> nb_entries_per_row{n, 0};
  for (il::int_t l = 0; l < nb_entries; ++l) {
    ++nb_entries_per_row[position[l][0]];
  }
  const il::int_t max_entries_per_row = il::max(nb_entries_per_row);

  // Suppose that we have the matrix entered with the following entries:
  // (0, 0) - (2, 2) - (1, 2) - (1, 1) - (2, 2)
  // and the following values
  // a      - d      - c      - b      - e
  //
  // It would give us the following matrix
  // a 0 0
  // 0 b c
  // 0 0 d+e
  // There would be one entry in the first row, 2 entries in the second
  // one and 2 entries in the last one.
  // The integer entry_of_RowIndex(i, p) is set here to be the position
  // of the p-th entry of the row i in the list of entries of the matrix.
  // For instance entry_of_RowIndex(0, 0) = 0, entry_of_RowIndex(1, 0)=2,
  // entry_of_RowIndex(2, 1) = 4. The integer col_of_RowIndex is set
  // the same way and give the column number.
  // The array row_of_entry[l] gives us in which row the l-th entry is.
  // For instance row_of_entry[2] = 1
  // We first set those array without considering sorting the entries
  // of a line.
  il::Array2C<il::int_t> entry_of_rowIndex{n, max_entries_per_row};
  il::Array2C<il::int_t> col_of_rowIndex{n, max_entries_per_row};
  for (il::int_t i = 0; i < n; ++i) {
    nb_entries_per_row[i] = 0;
  }
  for (il::int_t l = 0; l < nb_entries; ++l) {
    il::int_t i = position[l][0];
    il::int_t p = nb_entries_per_row[i];
    entry_of_rowIndex(i, p) = l;
    col_of_rowIndex(i, p) = position[l][1];
    ++nb_entries_per_row[i];
  }
  // For each row, we sort them according to their column.
  for (il::int_t i = 0; i < n; ++i) {
    for (il::int_t p = 0; p < nb_entries_per_row[i] - 1; ++p) {
      il::int_t min_col = n;
      il::int_t min_p = -1;
      for (il::int_t q = p; q < nb_entries_per_row[i]; ++q) {
        if (col_of_rowIndex(i, q) < min_col) {
          min_col = col_of_rowIndex(i, q);
          min_p = q;
        }
      }
      il::int_t tmp_entry = entry_of_rowIndex(i, p);
      il::int_t tmp_col = col_of_rowIndex(i, p);
      entry_of_rowIndex(i, p) = entry_of_rowIndex(i, min_p);
      col_of_rowIndex(i, p) = col_of_rowIndex(i, min_p);
      entry_of_rowIndex(i, min_p) = tmp_entry;
      col_of_rowIndex(i, min_p) = tmp_col;
    };
  }

  // We count the number of non-zero elements per Row
  // which is less than the number of entries.
  il::Array<il::int_t> nb_elements_per_row{n};
  il::int_t nb_elements = 0;
  for (il::int_t i = 0; i < n; ++i) {
    il::int_t nb_elements_row = 0;
    il::int_t last_newCol = -1;
    for (il::int_t p = 0; p < nb_entries_per_row[i]; ++p) {
      if (col_of_rowIndex(i, p) != last_newCol) {
        last_newCol = col_of_rowIndex(i, p);
        ++nb_elements_row;
      }
    }
    nb_elements_per_row[i] = nb_elements_row;
    nb_elements += nb_elements_row;
  }

  // We then fill column_, row_
  element_.resize(nb_elements);
  for (il::int_t k = 0; k < element_.size(); ++k) {
    element_[k] = 0;
  }

  column_.resize(nb_elements);
  row_.resize(n + 1);
  row_[0] = 0;
  il::int_t k = -1;
  for (il::int_t i = 0; i < n; ++i) {
    il::int_t last_newCol = -1;
    for (il::int_t p = 0; p < nb_entries_per_row[i]; ++p) {
      if (col_of_rowIndex(i, p) != last_newCol) {
        ++k;
        last_newCol = col_of_rowIndex(i, p);
        column_[k] = col_of_rowIndex(i, p);
      }
      index[entry_of_rowIndex(i, p)] = k;
    }
    row_[i + 1] = row_[i] + nb_elements_per_row[i];
  }
}

template <typename T>
T const& SparseArray2C<T>::operator[](il::int_t k) const {
  return element_[k];
}

template <typename T>
T& SparseArray2C<T>::operator[](il::int_t k) {
  return element_[k];
}

template <typename T>
T const& SparseArray2C<T>::operator()(il::int_t i, il::int_t k) const {
  IL_ASSERT(static_cast<il::uint_t>(i) < static_cast<il::uint_t>(n0_));
  IL_ASSERT(static_cast<il::uint_t>(row_[i] + k) <
            static_cast<il::uint_t>(row_[i + 1]));
  return element_[row_[i] + k];
}

template <typename T>
T& SparseArray2C<T>::operator()(il::int_t i, il::int_t k) {
  IL_ASSERT(static_cast<il::uint_t>(i) < static_cast<il::uint_t>(n0_));
  IL_ASSERT(static_cast<il::uint_t>(row_[i] + k) <
            static_cast<il::uint_t>(row_[i + 1]));
  return element_[row_[i] + k];
}

template <typename T>
il::int_t SparseArray2C<T>::size(il::int_t d) const {
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(d) < static_cast<il::uint_t>(2));
  return (d == 0) ? n0_ : n1_;
}

template <typename T>
il::int_t SparseArray2C<T>::nb_nonzeros() const {
  return element_.size();
}

template <typename T>
const T* SparseArray2C<T>::element_data() const {
  return element_.data();
}

template <typename T>
T* SparseArray2C<T>::element_data() {
  return element_.data();
}

template <typename T>
const il::int_t* SparseArray2C<T>::row_data() const {
  return row_.data();
}

template <typename T>
il::int_t* SparseArray2C<T>::row_data() {
  return row_.data();
}

template <typename T>
const il::int_t* SparseArray2C<T>::column_data() const {
  return column_.data();
}

template <typename T>
il::int_t* SparseArray2C<T>::column_data() {
  return column_.data();
}

template <typename T>
T SparseArray2C<T>::element(il::int_t k) const {
  return element_[k];
}

template <typename T>
il::int_t SparseArray2C<T>::row(il::int_t i) const {
  return row_[i];
}

template <typename T>
il::int_t SparseArray2C<T>::column(il::int_t k) const {
  return column_[k];
}

inline double norm(const il::SparseArray2C<double>& A, Norm norm_type,
                   const il::Array<double>& beta,
                   const il::Array<double>& alpha) {
  IL_ASSERT(alpha.size() == A.size(0));
  IL_ASSERT(beta.size() == A.size(1));

  double norm = 0.0;
  switch (norm_type) {
    case Norm::Linf:
      for (il::int_t i = 0; i < A.size(0); ++i) {
        double sum = 0.0;
        for (il::int_t k = A.row(i); k < A.row(i + 1); ++k) {
          sum += il::abs(A[k] * alpha[A.column(k)] / beta[i]);
        }
        norm = il::max(norm, sum);
      }
      break;
    default:
      IL_ASSERT(false);
  }

  return norm;
}
}

#endif  // IL_SPARSEARRAY2C_H