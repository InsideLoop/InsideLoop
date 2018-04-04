.. role:: cpp(code)

    :language: cpp

Linear Algebra
==============

Provided you have access to a BLAS library such as the MKL or OpenBLAS, the
InsideLoop library will give you access to those optimized libraries in a
much more friendly way than the original API.

The way the InsideLoop library deals with vectors and matrices is very similar
to what Numpy (the standard linear algebra package of Python) does. It is also
quite similar to Mathematica, but quite different from Matlab. For Matlab users,
all you need is to forget about the notion of row vector and column vector. With
Inside Loop, there is just one kind of vector, and this vector is not a special
case of a Matrix. Now that you are aware of this main difference, let's talk
about how Inside Loop deals with vectors and matrices.

- **Vectors**: We use the :cpp:`il::Array<T>` family of object to represent
  vectors where `T` can be :cpp:`float` or :cpp:`double`. Other types such
  as :cpp:`std::complex<flat>` and :cpp:`std::complex<double>` are under
  implementation.

- **Matrices**: We use the :cpp:`il::Array2D<T>` family of objects to represent
  matrices in column major order (also known as Fortran order). We also provide
  functions that work with matrices defined as :cpp:`il::Array2C<T>` and
  therefore stored in row major order (also known as C order, which is the
  origin or the name), but we encourage newcomers to use the column major
  order as it is better supported by many external libraries for historical
  reasons.

- **Tensors**: Inside Loop also provides some operations on tensors of larger
  rank. For instance, we use :cpp:`il::Arrray3D<T>` to represent tensors of
  rank 3. Vectors are tensors of rank 1 and matrices are tensors of rank 2.

Dot
---

The main operation in linear algebra is the :cpp:`dot` operation that implements
a tensor contraction of two tensors. It takes two tensors and returns a tensor.


.. code-block:: cpp

    #include <il/dot.h>

    auto c = il::dot(a, b)

- **Vector/Vector**: In case both :cpp:`a` and :cpp:`b` are :cpp:`il::Array<T>`,
  the dot operation will return the scalar product of :cpp:`a` and :cpp:`b`
  which is:

  .. math::

      \alpha = \sum_{k = 0}^{n-1} a_k \cdot b_k

  As a consequence, in this case, :cpp:`il::dot` will receive two
  :cpp:`il::Array<T>` of the same size and will return a :cpp:`T`. In debug
  mode, if those arrays do not have the same size, the program will abort. The
  result is undefined behavior in release mode.

   .. code-block:: cpp

       #include <il/dot.h>

       const il::int_t n = 1000;
       il::Array<double> a{n, 0.0};
       il::Array<double> b{n, 0.0};
       const double alpha = il::dot(a, b);

- **Matrix/Vector**: When :cpp:`A` is a matrix and :cpp:`x` is a vector, the dot
  operation will return the classic matrix/vector product :cpp:`y` defined by

  .. math::

      y_{i_0} = \sum_{i_1 = 0}^{n_1 - 1} a_{i_0, i_1} x_{i_1}

  where :cpp:`n1` is the size of the array :cpp:`x` and the number of columns
  of the matrix :cpp:`A`. Again, those dimensions are checked at runtime and
  will abort if there is a problem in debug mode.

  .. code-block:: cpp

       #include <il/dot.h>

       const il::int_t n0 = 1000;
       const il::int_t n1 = 2000;
       il::Array2D<double> A{n0, n1, 0.0};
       il::Array<double> x{n1, 0.0};
       il::Array<double> y = il::dot(A, x);

- **Matrix/Matrix**: Finally, when :cpp:`A` and :cpp:`B` are both matrices, the
  dot operation returns the classic matrix/matrix product defined by:

  .. math::

      C_{i_0, i_1} = \sum_{k = 0}^{n - 1} A_{i_0, k} B_{k, i_1}

  where :cpp:`n` is the number of columns of :cpp:`A` and :cpp:`n` is the
  number of rows of :cpp:`B` which should be equal.

  .. code-block:: cpp

       #include <il/dot.h>

       const il::int_t n0 = 1000;
       const il::int_t n = 500;
       const il::int_t n1 = 2000;
       il::Array2D<double> A{n0, n, 0.0};
       il::Array2D<double> B{n, n1, 0.0};
       il::Array2D<double> C = il::dot(A, B);

Blas
----

Unfortunately, the previous :cpp:`il::dot` returns a tensor and therefore
allocates some memory. But sometimes it might be that we already have some
memory allocated, or that we need to add the result of the product of 2 matrices
to a third one which already exists.

Some libraries such as Eigen provide very friendly ways to do this kind of
operations using both operator overloading and template metaprogramming.
Unfortunately, it makes programs that use them harder to debug and much harder
to profile. Inside Loop has chosen a different way, with a syntax which is a
bit less intuitive, but which allows to get the best performance with any
compiler and which makes debugging and profiling much easier. With Inside Loop
you'll be able to step into those functions without feeling the pain of
template meta-programming.

The default operation on BLAS, is the following

.. math::

    C = \alpha A\cdot B + \beta C

and is implemented with the Inside Loop library as

.. code-block:: cpp

     #include <il/blas.h>

     il::blas(alpha, A, B, beta, il::io, C);

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
     il::blas(1.0, A, B, 0.0, il::io, C);

One can also use this function to generate products with the transpose of the
matrices you deal with. For instance, if you want to add :math:`A^{T}\cdot B`
to :math:`C`, one can issue the call:

.. code-block:: cpp

     #include <il/blas.h>

     il::blas(1.0, A, il::MatrixOperator::Transpose, B, 1.0, il::io, C);
