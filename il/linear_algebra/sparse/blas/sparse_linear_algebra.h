template <typename T>
double norm(const il::SparseArray2D<T> &A, Norm norm_type,
            const il::Array<T> &beta, const il::Array<T> &alpha) {
  IL_ASSERT(alpha.size() == A.size(1));
  IL_ASSERT(beta.size() == A.size(0));

  auto norm = T{0.0};
  switch (norm_type) {
    case Norm::infinity:
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