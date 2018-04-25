.. role:: cpp(code)

    :language: cpp

Linear Algebra
==============

Provided you have access to a BLAS library such as the MKL or OpenBLAS,
InsideLoop gives you access to those optimized libraries in a much more friendly
way than the original API.

The way InsideLoop deals with vectors and matrices is very similar to what
Mathematica and Numpy (the standard linear algebra package of Python) do. For
Matlab users, there are few differences, the main one being that vectors are
not special case of matrices: with InsideLoop there is no notion of row vector
or column vector. All our linear algebra functions can work with the following
types: :cpp:`float`, :cpp:`double`, :cpp:`std::complex<float>` and
:cpp:`std::complex<double>`. Here are how the basic linear algebra notions are
represented:


- **Vectors**: We use the :cpp:`il::Array<T>` family of object to represent
  vectors.

- **Matrices**: We use the :cpp:`il::Array2D<T>` family of objects to represent
  matrices in column major order (also known as Fortran order). We also provide
  functions that work with matrices defined as :cpp:`il::Array2C<T>` which
  are stored in row major order but we encourage newcomers to use the column
  major order which is more common in numerical libraries.

- **Tensors**: InsideLoop also provides some operations on tensors of larger
  rank. For instance, we use :cpp:`il::Arrray3D<T>` and :cpp:`il::Arrray4D<T>`
  to represent tensors of rank 3 and 4 respectively. Vectors are tensors of rank
  1 and matrices are tensors of rank 2.

Dot
---

The main operation in linear algebra is the :cpp:`dot` operation that implements
a tensor contraction of two tensors. It takes two tensors and returns a tensor.


.. code-block:: cpp

    #include <il/blas.h>

    auto c = il::dot(a, b)

- **Vector/Vector**: In case both :cpp:`a` and :cpp:`b` are :cpp:`il::Array<T>`,
  the dot operation will return the scalar product of :cpp:`a` and :cpp:`b`
  which is:

  .. math::

      \alpha = \sum_{k = 0}^{n-1} a_k \cdot b_k

  The :cpp:`il::dot` funcion will receive two :cpp:`il::Array<T>` of the same
  size and will return a :cpp:`T`. In debug mode, if those arrays do not have
  the same size, the program will abort. In release mode, such a situation leads
  to undefined behavior.

   .. code-block:: cpp

       #include <il/blas.h>

       const il::int_t n = 1000;
       il::Array<double> a{n};
       il::Array<double> b{n};
       for (il::int_t i = 0; i < n; ++i) {
         a[i] = 1.0 / (1 + i);
         b[i] = 1.0 / (2 + i);
       }

       const double alpha = il::dot(a, b);

  When :cpp:`T` is a complex number, the usual dot product is defined by:

  .. math::

       \alpha = \sum_{k = 0}^{n-1} \overline{a_k} \cdot b_k

  and is computed with the call:

  .. code-block:: cpp

       #include <il/blas.h>

       const il::int_t n = 1000;
       il::Array<std::complex<double>> a{n};
       il::Array<std::complex<double>> b{n};
       for (il::int_t i = 0; i < n; ++i) {
         a[i] = 1.0 / std::complex<double>{1.0, static_cast<double>(i)};
         b[i] = 1.0 / std::complex<double>{2.0, static_cast<double>(i)};
       }

       const std::complex<double> alpha = il::dot(a, il::Dot::Star, b);

  For those who prefer the convention where we take the conjugate
  of the second vector, one can also use the call:

  .. code-block:: cpp

       #include <il/blas.h>

       const il::int_t n = 1000;
       il::Array<std::complex<double>> a{n};
       il::Array<std::complex<double>> b{n};
       for (il::int_t i = 0; i < n; ++i) {
         a[i] = 1.0 / std::complex<double>{1.0, static_cast<double>(i)};
         b[i] = 1.0 / std::complex<double>{2.0, static_cast<double>(i)};
       }

       const std::complex<double> alpha = il::dot(a, b, il::Dot::Star);


- **Matrix/Vector**: When :math:`A` is a matrix and :math:`x` is a vector, the
  dot operation will return the classic matrix/vector product
  :math:`y = A\cdot x` defined by

  .. math::

      y_{i_0} = \sum_{i_1 = 0}^{n_1 - 1} A_{i_0, i_1} x_{i_1}

  where :cpp:`n1` is the size of the array :cpp:`x` and the number of columns
  of the matrix :cpp:`A`. Again, those dimensions are checked at runtime and
  will abort if there is a problem in debug mode.

  .. code-block:: cpp

       #include <il/blas.h>

       const il::int_t n0 = 1000;
       const il::int_t n1 = 2000;
       il::Array2D<double> A{n0, n1};
       il::Array<double> x{n1};
       for (il::int_t i1 = 0; i1 < n1; ++i1) {
         for (il::int_t i0 = 0; i0 < n0; ++i0) {
           A(i0, i1) = 1.0 / (2 + i0 + i1);
         }
         x[i1] = 1.0;
       }

       il::Array<double> y = il::dot(A, x);

- **Matrix/Matrix**: Finally, when :cpp:`A` and :cpp:`B` are both matrices, the
  dot operation returns the classic matrix/matrix product defined by:

  .. math::

      C_{i_0, i_1} = \sum_{k = 0}^{n - 1} A_{i_0, k} B_{k, i_1}

  where :cpp:`n` is the number of columns of :cpp:`A` and the
  number of rows of :cpp:`B`.

  .. code-block:: cpp

       #include <il/blas.h>

       const il::int_t n0 = 1000;
       const il::int_t n = 500;
       const il::int_t n1 = 2000;
       il::Array2D<double> A{n0, n};
       for (il::int_t i1 = 0; i1 < n; ++i1) {
         for (il::int_t i0 = 0; i0 < n0; ++i0) {
           A(i0, i1) = 1.0 / (2 + i0 + i1);
         }
       }
       il::Array2D<double> B{n, n1};
       for (il::int_t i1 = 0; i1 < n1; ++i1) {
         for (il::int_t i0 = 0; i0 < n; ++i0) {
           B(i0, i1) = 1.0 / (2 + i0 + i1);
         }
       }

       il::Array2D<double> C = il::dot(A, B);

Blas
----

The previous function :cpp:`il::dot` returns a tensor and
therefore allocates some memory. But sometimes it might be that we already have
some memory allocated, or that we need to add the result of the product of 2
matrices to a third one which already exists. For that purpose, we offer the
:cpp:`il::blas` family of functions that work with view of arrays.


The default operation on BLAS, is the following

.. math::

    C = \alpha A\cdot B + \beta C

and is implemented with the Inside Loop library as

.. code-block:: cpp

     #include <il/blas.h>

     il::blas(alpha, A.view(), B.view(), beta, il::io, C.Edit());

For instance, il you want to compute the product of the two matrices :cpp:`A`
and :cpp:`B` and want to save the result in :cpp:`C` which has its memory
already allocated, you can run:


.. code-block:: cpp

     #include <il/blas.h>

     const il::int_t n0 = 500;
     const il::int_t n1 = 1000;
     const il::int_t n2 = 2000;
     il::Array2D<double> A{n0, n1, 0.0};
     il::Array2D<double> B{n1, n2, 0.0};
     il::Array2D<double> C{n0, n2};

     il::blas(1.0, A.view(), B.view(), 0.0, il::io, C.Edit());

One can also use this function to generate products with the transpose of the
matrices you deal with. For instance, if you want to add :math:`A^{T}\cdot B`
to :math:`C`, one can issue the call:

.. code-block:: cpp

     #include <il/blas.h>

     il::blas(1.0, A.view(), il::Dot::Transpose, B.view(), 1.0, il::io, C.Edit());

One can also work with complex matrices and use the transpose/conjugate of the
matrix A or B. For that, all you need to do is adding the value
:cpp:`il::Dot::Star` just after the matrix A or B.
