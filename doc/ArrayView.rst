.. role:: cpp(code)

    :language: cpp

ArrayView/ArrayEdit
===================

The object :cpp:`il::ArrayView<T>` is similar to :cpp:`il::Array<T>` in many
ways except that it does not own its memory. It is your responsability as a
programmer to ensure that the memory which is viewed by this object is valid
during the lifetime of the object. The semantics of copying and assignement
are also different as the copy the view of the array and not the array itself.
As the name suggests, it only allows read access to the elements of the array.
For those who need read and write access to the underlying memory, the object
you are looking for is :cpp:`il::ArrayEdit<T>`. Moreover, one cannot change
the size of an array view: of you want to few a smaller part of the array, you
have to create a new array view. Finally there is no concept of capacity for
array view.

Overview
--------

Let's start with a simple example where we create an array view from a
traditional array.

.. code-block:: cpp

    #include <iostream>

    #include <il/Array.h>
    #include <il/ArrayView.h>

    void printArrayView(il::ArrayView<double> v) {
      for (il::int_t k = 0; k < v.size(); ++k) {
        std::cout << "Index: " << k << ", Value: " << v[k] << std::endl;
      }
    }

    il::Array<double> a{il::value, {1.0, 5.0, 7.0, 9.0}};

    il::ArrayView<double> v = a.view(il::Range{1, 3});
    a[2] = 7.5;

    printArrayView(v);

This code will print the following:

.. code-block:: bash

    Index: 0, Value: 5.0
    Index: 1, Value: 7.5

From this small code, we can infer many different things:

- Once the array :cpp:`a` that contains 1.0, 5.0, 7.0, 9.0 is created, we
  define the view :cpp:`v` that looks at all the elements of the array from
  the index 1 (included) to the index 3 (excluded). As a consequence, the
  view :cpp:`v` looks at the elements of :cpp:`a` which are at the index
  position 1 and 2.

- After the view has been created, we modify one of the elements of the array
  :cpp:`a`. As we can see in the output of the program, the view :cpp:`v`
  can see that :cpp:`a` has been modified. The memory where the elements are
  stored, which is owned by the object :cpp:`a` is the same as the memory
  which is viewed by :cpp:`v`. We say that they alias.

- When we print the view :cpp:`v`, note that the original indices of the
  elements are not preserved: the indices are related to the view, not to the
  original array.

- We finally see that array views are passed to function by copy and not by
  reference. Only the view is part of their identity and not the elements
  they refer to, they are cheap to copy.

An array view can be used in different situations.

- **It can be used to pack a pointer and a size into a single object**.
  This is very useful when you get a container from another library or even
  another language. If you want to pack it into an object understood by
  InsideLoop, an array view is the way to go. For instance, if you get an old
  style C array with a pointer and a length, it is a good idea to pack it into
  an array view to benefit from bounds checking in debug mode.

  .. code-block:: cpp

      #include <il/ArrayView.h>

      bool contains_zero(const double* p, int n) {
        const il::ArrayView<double> v{p, n};
        bool ans = false;

        for (il::int_t i = 0; i < v.size(); ++i) {
          if (v[i] == 0.0) {
            ans = true;
          }
        }

        return ans;
      }

  In release mode, there is no performance penalty in using an array view
  instead of a raw pointer. The container has been designed so that all
  optimizations are allowed by the compiler.

- **It can be used to access parts of an array**. An array view can be used to
  access some parts of an array. for instance, suppose that the function
  :cpp:`sum` above is available to you and you want to know if the first
  half of an array has elements whose sum is larger than the second part. This
  could be done with the following code:

  .. code-block:: cpp

    #include <il/Array.h>
    #include <il/ArrayView.h>

    double sum(il::ArrayView<double> v);

    bool isFirstHalfLarger(const il::Array<double>& a) {
      const il::int_t n = a.size();

      const il::ArrayView<double> v0 = a.view(il::Range{0, n / 2});
      const il::ArrayView<double> v1 = a.view(il::Range{n / 2, n});

      return sum(v0) > sum(v1);
    }

- **It can be used to access parts of arrays of rank 2 or more**.
  We can generate array views from arrays of larger rank. For
  instance, given a 2 dimension array in column major order, the following code
  returns the column whose sum is the largest one:

  .. code-block:: cpp

    #include <il/Array2D.h>
    #include <il/ArrayView.h>

    double sum(il::ArrayView<double> v);

    il::int_t largestColumn(const il::Array2D<double>& A) {
      const il::int_t n0 = A.size(0);

      double maximum = -std::numeric_limits<double>::max();
      il::int_t i1_max = -1;

      for (il::int_t i1 = 0; i1 < A.size(1); ++i) {
        const il::ArrayView<double> v = A.view(il::Range{0, n0}, i1);
        const double value = sum(v);
        if (value >= maximum) {
          maximum = value;
          i1_max = i1;
        }
      }

      return i1_max;
    }

  Bare in mind that the elements of an array view should be contiguous in
  memory. As a consequence, it is not possible to create the array view of a row
  for a :cpp:`il::Array2D<double>`.

Construction
------------

There are many different ways to construct an array view.

1. The default constructor construct an array of size 0. On cannot do much with
   such a view, but having a default constructor is handy in C++.

   .. code-block:: cpp

       #include <il/ArrayView.h>

       il::ArrayView<double> v{};

2. If you have a const pointer :cpp:`data` to an array of elements of type
   :cpp:`T` and this array is of size :cpp:`n`:, one can easily construct an
   array view from this information:

   .. code-block:: cpp

       #include <il/ArrayView.h>

       void f(const double* data, int n) {
           il::ArrayView<double> v{data, n};
           ...
       }

3. But the most common way to get an array view is to get it from an array or
   from a multidimensional array.

   - If :cpp:`a` is an array and you want to create a view on the whole array,
     you could use the following method:

     .. code-block:: cpp

         #include <il/ArrayView.h>

         void f(const il::Array<double>& a) {
           il::ArrayView<double> v = a.view();
         }

   - But sometimes, all you need is access to a subpart of the array. This can
     also be done directly from the array itself. The following code will
     create a from on the array :cpp:`a` from the index :cpp:`n0` (included)
     to the index :cpp:`n1` (excluded).

     .. code-block:: cpp

         #include <il/ArrayView.h>

         void f(const il::Array<double>& a) {
           const il::int_t n0 = a.size() / 4;
           const il::int_t n1 = a.size() / 2;
           il::ArrayView<double> v = a.view(il::Range{n0, n1});
           ...
         }

   - For those who want to create an array view on the row of a
     :cpp:`il::Array2C<T>` which stores elements in row major order, we also
     have methods for that, directly from the object. The following code creates
     a view on the first row of the matrix.

     .. code-block:: cpp

         #include <il/Array2C.h>
         #include <il/ArrayView.h>

         void f(const il::Array2C<double>& A) {
           il::ArrayView<double> v = A.view(0, il::Range{0, A.size(1)});
           ...
         }

     the same method exists to create a view on a column of a
     :cpp:`il::Array2D<T>` but one cannot create a view on the column of a
     row major 2 dimensional array or on the row of a column major 2
     dimensional array.

   - Finally, it is straigthforward to create an array view from an array view.
     The methods previously available for an array are also available for an
     array view. For instance, the following code will split the view into 2
     parts before processing them.

     .. code-block:: cpp

         #include <il/ArrayView.h>

         void f(il::ArrayView<double> v) {
           il::ArrayView<double> v0 = v.view(il::Range{0, v.size() / 2});
           il::ArrayView<double> v1 = v.view(il::Range{v.size() / 2, v.size());
           ...
         }


Copy construction and assignement
---------------------------------

Copy construction and assigment in C++ are designed to provide a fresh copy of
an object with the same identity. The identity of :cpp:`il::ArrayView<T>` is
defined by the memory location is points to and the size of the view. As a
consequence, the semantics of the copy for array views is trivial. Moreover,
as these objects are cheap to copy, they should be passed by value.

In order to have a clear understanding of the semantics of a copy. Let's have
a look at the output of this code:

.. code-block:: cpp

    #include <iostream>

    #include <il/Array.h>
    #include <il/ArrayView.h>

    il::Array<double> a{il::value, {1.0, 5.0, 7.0, 9.0}};
    il::ArrayView<double> v0 = a.view();
    il::ArrayView<double> v1 = v0;

    a[1] = 5.5;
    std::cout << "Value: " << v1[1] << std::endl;

which prints :bash:`Value: 5.5`.

As they don't own their values, it is useless to use the move constructor or
the move copy with array views.

Destruction
-----------

As they don't own their value, nothing has to be done in their destructor.

Accessing the elements
----------------------

Elements of an array view are accessed the same way as an regular array. We
also provide a bracket operator, a :cpp:`back()` method and array views can
be walked through with the syntax of the C++11 for loop.

Note that, as the name suggests, array views only provide read access to their
underlying elements. If you need write access, the object you are looking for
is :cpp:`il::ArrayEdit<T>`.

Raw access for C functions
--------------------------

Much like a regular array, one can have direct access to the underlying pointer
of an array view. We provide the :cpp:`data()` method for such an access.

ArrayEdit<T>
------------

In case you want to have write access to some location of the memory, you should
use :cpp:`il::ArrayEdit<T>` that behaves exactly like an array view but also
provides write access to the memory.

:cpp:`il::ArrayEdit<T>` is implemented as an object inheriting from
:cpp:`il::ArrayView<T>` and does not have any data member that the view does not
have. As a consequence, every function that takes an :cpp:`il::ArrayView<T>` as
a parameter can take an :cpp:`il::ArrayEdit<T>` with no change.
