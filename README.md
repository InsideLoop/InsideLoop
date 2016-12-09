![InsideLoop icon](http://www.insideloop.io/wp-content/uploads/2014/09/inside-loop-logo-front.png)

InsideLoop is a C++11 library for high performance scientific applications. In
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

## The pitfalls of std::vector for High Performance Computing

Our first container, il::Array is meant to be a replacement for std::vector.

```cpp
#include <il/Array.h>

int main() {
  const int n = 100;
  il::Array<double> v(n);
  for (int i = 0; i < v.size(); ++i) {
    v[i] = 1.0 / (1 + i);
  }
  
  return 0;
}
```

As a std::vector, it owns its memory which is allocated at construction and
released at destruction. Its size can be decided at run-time, and change during
the lifetime of the object. It can me moved efficiently. Almost all methods
available for std::vector are also available for il::Array. However, it differs
from the standard library container on the following points:

- **Indices**: Our container uses signed integers for indices. The default is to
  use int (which is a 32 bit signed integer on 64-bit Linux and Windows
  platforms), but the type can globally be changed to std::ptrdiff_t (which is
  a 64-bit signed integer on 64-bit Linux and Windows platform) if your
  application needs to work with large arrays.
- **Initialization**: When T is a numeric type (int, float, double, etc), the
  construction of an il::Array<T> of size T does not initialize its memory.
  
These choices are important for both debugability and performance. We use
**signed integers** because it makes it easier to write correct programs.
Suppose you want to write a fonction that takes an array of
double v and returns the largest index k such that a <= k < b and v[k] is
equal to 0. Here is the code you need to write with signed integers:
```cpp
#include <il/Array.h>

int f(const il::Array<double>& v, a, b) {
  IL_ASSERT_PRECOND(a >= 0);
  IL_ASSERT_PRECOND(b <= v.size());
  
  for (int k = b - 1; k >= a; --k) {
    if (v[k] == 0.0) {
      return k;
    }
  }
  
  return -1;
}
```
It you write the same code with unsigned integers, you'll get a bug when
a == 0 and when v does not contain any zero. In this case, k will go down to
0, and then the cyclic nature of unsigned integers will make k go from 0
to 2^32 - 1 = 4'294'967'295 and provides an out-of-bound access. Writing a
correct code with unsigned integers can be done, but a bit more tricky.

We don't enforce initialization of numeric types for debugability and
performance reason. Let's look at the following code which stores the inverse
of integers in an array and has an obvious bug as v[n - 1] is not set.

```cpp
#include <il/Array.h>

int main() {
  const int n = 1000000000;
  il::Array<double> v(n);
  for (int i = 0; i < n - 1); ++i) {
    v[i] = 1.0 / (1 + i);
  }
  
  return 0;
}
```
With InsideLoop, in debug mode, at construction, all v[k] are set to NaN which
will propagate very fast in your code and will make the bug easy to track. With
std::vector, v[n - 1] would have been set to 0 which would make the error harder
to track. Of course, in release mode, none of the v[k] will be set at
construction.
Once the bug is fixed, you might want to use OpenMP for efficient
multithreading. With InsideLoop, all you need is to add a pragma before the loop
to get efficient multithreading.

```cpp
#include <il/Array.h>

int main() {
  const int n = 1000000000;
  il::Array<double> v(n);
#pragma omp parallel for
  for (int i = 0; i < v.size(); ++i) {
    v[i] = 1.0 / (1 + i);
  }
  
  return 0;
}
```

The same code would be much slower with std::vector on a dual-socket workstation
because of non uniform memory allocation (NUMA) effects. As all the memory is
initialized to 0.0 by the standard library, the first touch policy will map
all the memory of v to the RAM close to one of the processor. During the OpenMP
loop, half of the vector will have to go trough the Quick Path Interconnect
(QPI) which will limit the memory performance of the loop. Unfortunately, there
is no easy solution for this problem (unless you really want to deal with
custom allocators) and this is one of the most important reason std::vector are
not used in high performance computing programs.
