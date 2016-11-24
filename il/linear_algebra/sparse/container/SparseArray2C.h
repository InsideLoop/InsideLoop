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
#include <il/SmallArray.h>

namespace il {

template <typename T>
class SparseArray2C {
 private:
  il::int_t height_;
  il::int_t width_;
  il::Array<T> element_;
  il::Array<il::int_t> column_;
  il::Array<il::int_t> row_;

 public:
  SparseArray2C(il::int_t height, il::int_t width, il::Array<il::int_t> column,
                il::Array<il::int_t> row);
  template <il::int_t n>
  SparseArray2C(il::int_t width, il::int_t height,
                const il::Array<il::SmallArray<il::int_t, n>>& column);
  const T& operator[](il::int_t k) const;
  T& operator[](il::int_t k);
  const T& operator()(il::int_t i, il::int_t k) const;
  T& operator()(il::int_t i, il::int_t k);
  T element(il::int_t k) const;
  il::int_t row(il::int_t i) const;
  il::int_t column(il::int_t k) const;
  il::int_t size() const;
  il::int_t size(il::int_t d) const;
  il::int_t nb_nonzeros() const;
  T* element_data();
  il::int_t* row_data();
  il::int_t* column_data();
};

template <typename T>
SparseArray2C<T>::SparseArray2C(il::int_t height, il::int_t width,
                                il::Array<il::int_t> column,
                                il::Array<il::int_t> row)
    : height_{height},
      width_{width},
      element_{column.size()},
      column_{std::move(column)},
      row_{std::move(row)} {
  IL_ASSERT(row_.size() == height + 1);
}

template <typename T>
template <il::int_t n>
SparseArray2C<T>::SparseArray2C(
    il::int_t width, il::int_t height,
    const il::Array<il::SmallArray<il::int_t, n>>& column)
    : width_{width}, height_{height}, element_{}, column_{}, row_{height + 1} {
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
T const& SparseArray2C<T>::operator[](il::int_t k) const {
  return element_[k];
}

template <typename T>
T& SparseArray2C<T>::operator[](il::int_t k) {
  return element_[k];
}

template <typename T>
T const& SparseArray2C<T>::operator()(il::int_t i, il::int_t k) const {
  IL_ASSERT(static_cast<il::uint_t>(i) < static_cast<il::uint_t>(height_));
  IL_ASSERT(static_cast<il::uint_t>(row_[i] + k) <
            static_cast<il::uint_t>(row_[i + 1]));
  return element_[row_[i] + k];
}

template <typename T>
T& SparseArray2C<T>::operator()(il::int_t i, il::int_t k) {
  IL_ASSERT(static_cast<il::uint_t>(i) < static_cast<il::uint_t>(height_));
  IL_ASSERT(static_cast<il::uint_t>(row_[i] + k) <
            static_cast<il::uint_t>(row_[i + 1]));
  return element_[row_[i] + k];
}

template <typename T>
il::int_t SparseArray2C<T>::size() const {
  return element_.size();
}

template <typename T>
il::int_t SparseArray2C<T>::size(il::int_t d) const {
  switch (d) {
    case 0:
      return height_;
      break;
    case 1:
      return width_;
      break;
    default:
      abort();
  }
}

template <typename T>
il::int_t SparseArray2C<T>::nb_nonzeros() const {
  return element_.size();
}

template <typename T>
T* SparseArray2C<T>::element_data() {
  return element_.data();
}

template <typename T>
il::int_t* SparseArray2C<T>::row_data() {
  return row_.data();
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
  return row_[i + 1] - row_[i];
}

template <typename T>
il::int_t SparseArray2C<T>::column(il::int_t k) const {
  return column_[k];
}
};

#endif  // IL_SPARSEARRAY2C_H