InsideLoop is a **C++11 library** for **high performance scientific applications**
running on processors (including **Xeon**) and coprocessors (**Cuda**). This
library has been designed to provide you with:

- Efficient containers:
  - **Arrays** and **multi-dimensional arrays** with different allocation
    policies
  - **sets** and **maps** implemented using open addressing with
    quadratic probing
  - **Unicode Strings** implemented with small size optimization
  - An efficient **dynamic** type
- IO using the binary **Numpy** file format for arrays and the textual **TOML**
  file format for structured data
- A pure **C++11** library with **no dependency** for easy integration in both
  **free software** and **commercial products** 
- For those who can afford a library dependency, an easy access to the best
  numerical libraries from Intel (**MKL**) and nVidia (**cuBLAS** and
  **cuSPARSE**)

In a few words, **InsideLoop** should appeal to **scientific programmers**
looking for easy-to-use containers and wrappers around the best numerical
libraries available today. 

Note that for the time being, **InsideLoop is a work in progress**. Although
some parts of it have been used in production code, it should be used at your
own risk. The API getting stabilized.

The documentation is currently begin written and is available here: [Documentation](http://www.insideloop.io/doc-insideloop/_build/html/index.html)

## Remarks, feature request, bug report

Please send them to `fayard@insideloop.io`
