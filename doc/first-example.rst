First example
=============

In order to get a quick feeling about the API, here is a small code playing
around with Hilbert matrices.

.. code-block:: cpp

    #include <il/linearAlgebra.h>

    int main() {
      // The il::int_t is the signed version of std::size_t. We use signed
      // integers for both convenience and performance.
      //
      const il::int_t n = 100;

      // Create a n by n matrix, using Fortran ordering. At construction,
      // elements are initialized to NaN in debug mode and uninitialized in
      // release mode. The matrix is then initialized to a Hilbert matrix.
      //
      il::Array2D<double> hilbert(n, n);
      for (il::int_t j = 0; j < n; ++j) {
        for (il::int_t i = 0; i < n; ++i) {
          hilbert(i, j) = 1.0 / (1 + i + j);
        }
      }

      // Create an array x of length n, filled with 1.0. For Matlab users, bare
      // in mind that one dimensional arrays are neither row or column matrices.
      //
      il::Array<double> x(n, 1.0);

      // The function il::dot computes the product of a matrix and a vector. It
      // is also used to compute matrix products and more generally tensor
      // products.
      //
      il::Array<double> y = il::dot(A, x);

      // Play around with hilbert, x and y. Expect some numerical instabilities
      // thanks to Hilbert :-)

      return 0;
    }
