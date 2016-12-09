![InsideLoop icon](http://www.insideloop.io/wp-content/uploads/2014/09/inside-loop-logo-front.png)

InsideLoop is a C++ library for high performance scientific applications. In
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

## Tutorial

InsideLoop comes with many containers, such as a dynamic array which is meant
to be used as a replacement for std::vector.

```cpp
#include <il/Array.h>

int main() {
  const int n = 100;
  il::Array<double> v{n};
  for (int i = 0; i < v.size(); ++i) {
    v[i] = 1.0 / (1 + i);
  }
  
  return 0;
}
```

