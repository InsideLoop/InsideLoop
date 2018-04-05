.. InsideLoop documentation master file, created by
   sphinx-quickstart on Sat Feb  3 08:50:07 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to InsideLoop's documentation!
======================================

InsideLoop is a **C++11 library** for **high performance scientific applications**
running on processors (**Intel**, **AMD** and **ARM**) and coprocessors (**Nvidia Cuda**). This
library has been designed to provide you with:

- Efficient containers:

  - **Arrays** and **multi-dimensional arrays** with different allocation policies
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
own risk. The API is getting stabilized and specifications documented here
will be enforced through the whole 0.x.x serie.

.. toctree::
   :caption: Getting started
   :maxdepth: 1

   CompilingFirstProgram.rst

.. toctree::
   :caption: The containers
   :maxdepth: 1

   Array.rst
   ArrayView.rst
   Array2D.rst
   String.rst
   Map.rst

.. toctree::
   :caption: Linear algebra
   :maxdepth: 1

   LinearAlgebra.rst

.. toctree::
   :caption: Performance guide
   :maxdepth: 1

   integer.rst

Remarks, feature request, bug report
====================================

Please send them to `fayard@insideloop.io`

