InsideLoop's documentation
==========================

InsideLoop is a **C++11 library** for **high performance scientific
applications** running on processors (including **Xeon** and **Xeon Phi**) and
coprocessors (**Cuda**). This library has been designed to provide you with:

- Efficient containers such as **array**, **multi-dimensional arrays**,
  **string** a **hash set** and a  **hash map**
- A **simple code** base that could be extended easily
- **Efficient debugging** mode
- Almost **no coupling in between classes** so you can extract from InsideLoop
  the containers or the solvers you use. For instance, if you only use arrays
  and multidimensional arrays, all you need are a few files.
- C++ wrappers for the linear algebra libraries provided by Intel (Math Kernel
  Library, also known as the **MKL**) and NVidia (**cuBLAS**, **cuSPARSE**)
- An **Open Source** licence allowing its integration in both **free software**
  and **commercial** products

**InsideLoop** should appeal to **scientific programmers** looking for
easy-to-use containers and wrappers around the best numerical libraries
available. It should be very **friendly** to **Fortran** and **C** programmers,
and even **C++** programmers who like to keep simple things simple.

.. warning::
   InsideLoop is still a work in progress. Even though some containers have been
   used in production code, the libray is still experimental. Moreover, the API
   is not stabilized yet.

In order to get a quick feeling about **InsideLoop**, here is a small code that
computes the Hilbert matrix of size n and compute the product y of this matrix
with a vector x.

.. code-block:: cpp

    #include <il/Array.h>
    #include <il/Array2D.h>
    #include <il/linear_algebra.h>

    int main() {
      const il::int_t n = 100;

      il::Array2D<double> hilbert(n, n);
      for (il::int_t j = 0; j < n; ++j) {
        for (il::int_t i = 0; i < n; ++i) {
          hilbert(i, j) = 1.0 / (1 + i + j);
        }
      }

      il::Array<double> x(n, 1.0);
      il::Array<double> y = il::dot(A, x);

      // Play around with hilbert, x and y
      // Expect some numerical instabilities thanks to Hilbert :-)

      return 0;
    }

Guide
^^^^^

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   license.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
