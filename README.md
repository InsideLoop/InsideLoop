![InsideLoop icon](http://www.insideloop.io/wp-content/uploads/2014/09/inside-loop-logo-front.png)

InsideLoop is a **C++11 library** for **high performance scientific applications**
running on **processors** and **coprocessors**. In
order to get the most of current hardware, data locality is of uttermost
importance. Unfortunately, the Standard Template Library (STL) has been designed
for generic programming and hides its memory layout to the programmer.
InsideLoop stays away from generic programming and template meta-programming
techniques. It should appeal to Fortran and C programmers who want to benefit
from C++ features but don't want a complex library to get in their way to high
performance computing.

InsideLoop provides arrays, multi-dimensional arrays and a hash table. It has
been designed to provide you with:

- Containers for modern parallel architectures
- C++ wrappers for the Math Kernel Library (MKL) from Intel
- Portable performance across compilers
- Smooth integration into your projects
- Efficient debugging mode

## Debugability and efficiency for dynamic arrays

Our first container, `il::Array` is meant to be a replacement for `std::vector`.

```cpp
#include <il/Array.h>

int main() {
  const il::int_t n = 100;
  il::Array<double> v(n);
  for (il::int_t i = 0; i < v.size(); ++i) {
    v[i] = 1.0 / (1 + i);
  }
  
  return 0;
}
```

As `std::vector`, it owns its memory which is allocated at construction and
released at destruction. Its size can be decided at run-time, and changed during
the lifetime of the object. The array can also be moved efficiently. Almost all
methods available for `std::vector` are also available for `il::Array`. However, it
differs from the standard library container on the following points:

- **Bounds checking**: By default, bound checking is available for all our
  containers in debug mode. For `il::Array<T>`, `v[i]` are bounds checked in
  debug mode but not in release mode. 
- **Indices**: Our containers use signed integers for indices. By default, we
  use `std::ptrdiff_t` which is typedefed to `il::int_t` (it is a 64 bit signed
  integer on 64-bit macOS, Linux and Windows
  platforms), but the type might be changed to `int` (which is
  a 32-bit signed integer on those platforms).
- **Initialization**: When T is a numeric type (int, float, double, etc), the
  construction of an `il::Array<T>` of size n does not initialize its memory in
  release mode.
- **Vectorization**: InsideLoop library has been designed to allow many
  loops to be vectorized automatically by the compiler, and to allow easy
  access to the underlying structure of the objects when guided or manual
  vectorization is needed.
  
These choices are important for both debugability and performance. We use
**signed integers** because it makes it easier to write correct programs.
Suppose you want to write a fonction that takes an array of
double v and returns the largest index k such that `a <= k < b` and `v[k]` is
equal to 0. With signed integers, the code is straightforward:
```cpp
#include <il/Array.h>

il::int_t f(const il::Array<double>& v, a, b) {
  IL_ASSERT_PRECOND(a >= 0);
  IL_ASSERT_PRECOND(b <= v.size());
  
  for (il::int_t k = b - 1; k >= a; --k) {
    if (v[k] == 0.0) {
      return k;
    }
  }
  
  return -1;
}
```
If you write the same code with unsigned integers, you'll get a bug if
`a == 0` and v does not contain any zero. In this case, k will go down to
0, and then the cyclic nature of unsigned integers will make k jump from `0`
to `2^32 - 1 == 4'294'967'295`. An out-of-bound access and a crash is ready to happen. Writing a
correct code with unsigned integers can be done, but is a bit more tricky. We cannot
count all the bugs which have been discovered because of unsigned integers. Their
usage in the C++ standard library was a mistake as acknowledged by
Bjarne Stroustrup, Herb Sutter and Chandler Carruth in this
[video](https://www.youtube.com/watch?v=Puio5dly9N8) at 42:38 and 1:02:50. Using
InsideLoop's library allows programmers to stay away from them. Moreover, the fact
that signed overflow is undefined behaviour in C/C++ allows optimizations which
are not available to the compiler when dealing with unsigned integers.

We don't enforce **initialization** of numeric types for debugability and
performance reason. Let's look at the following code which stores the cosine
of various numbers in an array. It has an obvious bug as `v[n - 1]` is not set.

```cpp
#include <cmath>

#include <il/Array.h>

int main() {
  const il::int_t n = 2000000000;
  const double x = 0.001;
  il::Array<double> v(n);
  for (il::int_t i = 0; i < n - 1; ++i) {
    v[i] = std::cos(i * x);
  }
  
  return 0;
}
```
With InsideLoop, in debug mode, at construction, all `v[k]` are set to `NaN` which
will propagate very fast in your code and will make the bug easy to track. With
`std::vector`, `v[n - 1]` would have been set to 0 which would make the error harder
to track. In release mode, none of the `v[k]` will be set at
construction and the array `v` will be set only once and not twice as in the
standard library.

Once the bug is fixed, you might want to use OpenMP for
multithreading. With InsideLoop's library, all you need is to add a pragma before the loop
to get efficient multithreading.

```cpp
#include <cmath>

#include <il/Array.h>

int main() {
  const il::int_t n = 2000000000;
  const double x = 0.001;
  il::Array<double> v(n);
#pragma omp parallel for
  for (il::int_t i = 0; i < v.size(); ++i) {
    v[i] = std::cos(i * x);
  }
  
  return 0;
}
```

The same code would be much slower with std::vector on a dual-socket workstation
because of non uniform memory allocation (NUMA) effects. As all the memory is
initialized to `0.0` by the standard library, the first touch policy will map
all the memory used by v to the RAM close to one of the processors. During the OpenMP
loop, half of the vector will be taken care by the other socket and the memory
will have to go trough the Quick Path Interconnect (QPI). The performance of the
 loop will be limited. Unfortunately, there is no easy solution for this problem
 with `std::vector` (unless you really want to deal with
custom allocators) and this is one of the many reason `std::vector` is
not used in high performance computing programs.

Concerning **vectorization**, unfortunately, pointer aliasing 
makes vectorization impossible for the compiler in many situations. It has to
be taken care either by working directly with pointers and using the `restrict`
keyword or by using OpenMP 4 pragmas such as `#pragma omp simd`. InsideLoop
library allows to get access to the underlying structure of the containers
when such manual optimization is needed. Moreover, for efficient vectorization,
it is sometimes useful to align memory to the vector width which is 32 bytes
on AVX processors. This can be easily done with the constructor
`il::Array<double> v(n, il::align, 32)`. You can even misalign arrays
on AVX for teaching purposes with `il::Array<double> w(n, il::align, 16, 32)`.
  
## Allocation on the stack for static and small arrays

Insideloop's library also provides `il::StaticArray<T, n>` which is a replacement for
`std::array<T, n>` and uses the stack for its storage.

It is also useful to work with small arrays whose size
is not known at compile time but for which we know that most of them will have
a small size. For those cases, InsideLoop provides the object `il::SmallArray<T, n>`
which is a dynamic array whose elements are allocated on the stack when its size
is below n and allocated on the heap when its size is larger than n. For instance,
the sparsity pattern of a sparse matrix for which most rows contains no more than
5 non-zero elements could be efficiently represented and filled with the following code:
 
```cpp
#include <il/Array.h>
#include <il/SmallArray.h>

il::Array<il::SmallArray<double, 5>> A{n};
for (il::int_t i = 0; i < n; ++i) {
  LOOP {
    const il::int_t column = ...;
    A[i].push_back(column);
  }
}
```

This structure will allow us to use few memory allocations.

## Multidimensional arrays

InsideLoop's library provides efficient multidimensional arrays, in Fortran
and C order. The object `il::Array2D<T>` represents a 2-dimensional array, with
Fortran ordering: the memory will successively contains `A(0, 0)`, `A(1, 0)`,
..., `A(nb_row - 1, 0)`, `A(0, 1)`, `A(1, 1)`, ..., `A(nb_row - 1, 1)`, ...,
`A(nb_row - 1, nb_col - 1)`. This ordering is the one used in Fortran, and for
that reason, most BLAS libraries (including the MKL) are better optimized for
it. Here is a code that constructs an Hilbert matrix:

```cpp
#include <il/Array2D.h>

const il::int_t n = 63;
il::Array2D<double> A(n, n);
for (il::int_t j = 0; A.size(1); ++j) {
  for (il::int_t i = 0; A.size(0); ++i) {
    A(i, j) = 1.0 / (2 + i + j);
  }
}
```

Most operations available on `il::Array<T>` are also available for
`il::Array2D<T>`. For instance, it is possible to `resize` the object and
`reserve` memory for it. It is also possible to align a multidimensional array.
The object will include some padding at the end of each column so the first
element of each column is aligned.

```cpp
#include <il/Array2D.h>

const il::int_t n = 63;
il::Array2D<double> A(n, n, il::align, 32);
for (il::int_t j = 0; A.size(1); ++j) {
  for (il::int_t i = 0; A.size(0); ++i) {
    A(i, j) = 1.0 / (2 + i + j);
  }
}
```

For those who want a 2-dimensional array with C ordering, `il::Array2C<T>`
provides such an object. We also provide `il::Array3D<T>` and `il::Array4D<T>`
objects.

A `il::StaticArray2D<T, n0, n1>` is also available for 2-dimensional arrays
whose size is known at compile time. Their elements are stored on the stack
instead of the heap.

## Linear algebra on processors optimized by  the Intel MKL

Matrices are represented as 2-dimensional arrays and many routines are available
to help solving linear algebra problems.

You can easily multiply a matrix by a vector with the following code.
```cpp
#include <il/Array2D.h>

const il::int_t n = 1000;
il::Array2D<double> A(n, n);
// Fill the matrix
il::Array<double> x(n);
// Fill x

il::Array<double> y = il::dot(A, x);
```

Matrix multiplication is also available with the same function

```cpp
#include <il/Array2D.h>
#include <il/linear_algebra/dense/blas/dot.h>

const il::int_t n = 1000;
il::Array2D<double> A(n, n);
// Fill the matrix A
il::Array2D<double> B(n, n);
// Fill the matrix B

il::Array2D<double> C = il::dot(A, B);
```
which can be used for any tensor contraction.

It allows the best performance available on Intel processors as it uses the
MKL library behind the scene. If a matrix C is already available
and you want to add the result of A.B to C, you can use the following blas
function which won't allocate any memory:

```cpp
#include <il/Array2D.h>
#include <il/linear_algebra/dense/blas/blas.h>

const il::int_t n = 1000;
il::Array2D<double> A(n, n);
// Fill the matrix A
il::Array2D<double> B(n, n);
// Fill the matrix B
il::Array2D<double> C(n, n);
// Fill the matrix C

const double alpha = 1.0;
const double beta = 1.0;
// Perform C = alpha A.B + beta C
il::blas(alpha, A, B, beta, il::io, C);
```

As you can see, InsideLoop is quite different from other linear algebra packages
such as Eigen. We don't use expression templates which would allow to write
`C += A * B` and expect good performance. But it allows us to have
a better control of the memory and the different operations. It also allows us
to have a code that you can understand and hack easily if you need a new feature.
You are more likely to understand the errors of the compiler if something
goes wrong. 

You can also solve system of linear equations:

```cpp
// Code to solve the system A.x = y
#include <il/Array2D.h>
#include <il/linear_algebra/dense/factorization/LU.h>

const il::int_t n = 1000;
il::Array2D<double> A(n, n);
// Fill the matrix
il::Array<double> y(n);
// Fill y

il::Status status;
il::LU<il::Array2D<double>> lu_decomposition(A, il::io, status);
if (!status.ok()) {
  // The matrix is singular to the machine precision. You should deal with the error.
}

il::Array<double> x = lu_decomposition.solve(y);
```

This code shows many design patterns available in the InsideLoop library. At
first we define the matrix and the second member of the linear system. Then,
we do the LU decomposition with partial pivoting of the matrix A, which is then
stored in the object lu_decomposition. As the matrix might be singular with
respect to machine precision (the algorithm finds no pivot in the process of
the LU factorization), an error has to be given back to the user. Such errors
are given back through `il::Status` objects which are passed by reference to the
function. The tag `il::io` is here to distinguish in between input and output
parameters. All parameters at its left are passed by value or constant reference
(they are input parameters) and all parameters at its right are passed by
reference (they are input/output parameters). It is mandatory to check the
status of the function before the end of the scope. Otherwise the object
status will abort the program when destroyed. You can also choose to ignore
the status of the code with `status.ignore_error()` or even abort the program
if there is any error with `status.abort_on_error()`. This interface allows
 an explicit handling of error and makes it impossible to ignore them. It also
 allows InsideLoop to be used within codes that don't allow exceptions.
 
For those who are familiar with LAPACK which is used behind the scene, it is
known that the LU decomposition is done inplace by the routine. As A is
on the left of the `il::io` tag, it shows clearly that A cannot be mutated. A
careful inspection of the signature of the constructor of
`il::PartialLU<il::Array2D<double>>` shows that A is passed by value. In case
you don't need A anymore after the construction of `lu_decomposition` and you
don't want to pay the price of a copy, you can move A into lu_decomposition.
It is also known that the solving part of LAPACK functions works inplace. As a
consequence, we can even avoid memory allocation for x. 

```cpp
// Code to solve the system A.x = y
#include <il/Array2D.h>
#include <il/linear_algebra/dense/factorization/LU.h>

const il::int_t n = 1000;
il::Array2D<double> A(n, n);
// Fill the matrix
il::Array<double> y(n);
// Fill y

il::Status status;
il::LU<il::Array2D<double>> lu_decomposition(std::move(A), il::io, status);
if (!status.ok()) {
  // The matrix is singular to the machine precision. You should deal with the error.
}

il::Array<double> x = lu_decomposition.solve(std::move(y));
```

This example shows that InsideLoop has explicit interfaces, allows explicit error
handling, and the interfaces have been carefully crafted to get the same memory
consumption and the same peformance of direct calls to the MKL. As our code is
as simple as possible, it will be very easy for you to add new methods to our
objects if you want to change the interface to get better performance in a case
we overlooked.

## Linear algebra on Cuda devices

We also provide containers for Cuda Devices. The folling code creates an array
on the main memory and copies it to the memory of your GPU.

```cpp
#include <il/Array.h>
#include <il/CudaArray.h>

const int n = 1000000;
il::Array<double> v_host(n);
for (il::int_t i = 0; i < v_host.size(); ++i) {
    v[i] = 1.0 / (1 + i);
}

il::CudaArray<double> v_gpu(n);
il::copy(v_host, il::io, v_gpu);
```

Of course, BLAS operations such as matrix multiplication are availble and runs
the cuBLAS library provided by NVidia. Their usage is similar to what is available
on the processor. The following code creates matrices in the main memory, then
transfers it to the CUDA device where the BLAS operation is performed. Its result
is then copied back to the main memory.

```cpp
#include <il/Array2D.h>
#include <il/CudaArray2D.h>

const int n = 10000;
il::Array2D<double> A(n, 0);
il::Array2D<double> B(n, n);
// Fill matrices A and B
il::Array<double> C(n, n, 0.0);

il::CudaArray<double> A_gpu(n, n);
il::CudaArray<double> B_gpu(n, n);
il::CudaArray<double> C_gpu(n, n);
il::copy(A, il::io, A_gpu);
il::copy(B, il::io, B_gpu);
il::copy(C, il::io, C_gpu);
il::blas(1.0, A_gpu, B_gpu, 0.0, il::io, C_gpu);
il::copy(C_gpu, il::io, C);

```

## A Platform for core i7, Xeon, Xeon Phi and Cuda devices

This makes it very easy to compare state of the art libraries on processors
and on a co-processor. For instance, here are the timings for multiplying two
10000x10000 matrices of floats and doubles on different devices using both the
MKL from Parallel Studio XE 2017 and cuBLAS from CUDA 8.0:

|                                                 |Time (float) | TFlops (float) | Time (double) | Tflops (double) |
|-------------------------------------------------|:-----------:|:--------------:|:-------------:|:---------------:|
|Core i7, 4 cores, Haswell (MacBook Pro 2014)     |     6.25 s  |   0.32 TFlops  |   12.78 s     |   0.15 TFlops   |
|Dual-Xeon E5-2660, 2x14 cores, Broadwell         |     1.36 s  |   1.45 TFlops  |    2.72 s     |   0.73 TFlops   |
|NVidia Titan X Pascal                            |     0.20 s  |   10.1 TFlops  |    5.40 s     |   0.37 TFlops   |


## Hash Table

We finally provide a hash table implemented using open addressing and quadratic
probing. This hash table has better performance than the one provided by the
standard library `std::unordered_map`.

## Remarks, feature request, bug report

Please send then to `fayard@insideloop.io`
