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
  the containers or the solvers you use.
- C++ wrappers for the linear algebra libraries provided by Intel (**MKL**) and
  NVidia (**cuBLAS**, **cuSPARSE**)
- An **Open Source** licence allowing its integration in both **free software**
  and **commercial** products

**InsideLoop** should appeal to **scientific programmers** looking for
easy-to-use containers and wrappers around the best numerical libraries
available. It should be very **friendly** to **Fortran** and **C** programmers,
and even **C++** programmers who like to keep simple things simple.

Guide
^^^^^

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   first-example.rst
   license.rst

.. warning::
**InsideLoop** is a work in progress. Even though some containers have been
   used in production code, the libray is still experimental. Moreover, the API
   is not stabilized yet.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
