.. role:: cpp(code)

    :language: cpp

Array2D
=======

Overview
--------

We provide a two-dimensional dynamic array :cpp:`il::Array2D<T>` that stores its
element in column-major order, also known as Fortran order: the elements
:cpp:`a(i0, i1)` and :cpp:`a(i0 + 1, a1)` are stored next to each other in
memory. This array is quite similar to the linear array :cpp:`il::Array<T>` in
many ways: it owns its memory, its size can be dynamically changed, and it
provides capacities in both dimension which also both more efficient resizing
and padding for those who are concerned about memory alignement.

Let's start with our first example where we create such an object which will be
interpreted as a matrix:

.. code-block:: cpp

    #include <il/Array2D.h>

    const il::int_t n = 100;
    il::Array2D<double> M{n, 2 * n};
    for (il::int_t j = 0; j < M.size(1); ++j) {
      for (il::int_t i = 0; i < M.size(0); ++i) {
        M(i, j) = 1.0 / (2 + i + j);
      }
    }

We have created a matrix with 100 rows and 200 columns. Our container provides
sizes for both dimensions which can be accessed with the :cpp:`size(0)` and
:cpp:`size(1)` methods which return an integer of type :cpp:`il::int_t`. The
array is stored in column-major order. As a consequence, this code can
be vectorized efficiently by most compilers.

In order to clearly understand how a :cpp:`il::Array2D<T>` array is layed out
in memory, here is such an array that as a size of 2 and a capacity of 3 in the
first dimension (the rows for a matrix), and a size of 3 and a capacity of 4
in the second dimension (the columns for a matrix)

.. image:: array2d-matrix.png
    :scale: 20 %
    :alt: alternate text
    :align: center

And here is how this array is layed out in memory

.. image:: array2d-memory.png
    :scale: 20 %
    :alt: alternate text
    :align: center

As you can see, the elements :cpp:`M(i0, i1)` and :cpp:`M(i0 + 1, i1)` are
close to each other in memory. But, if the capacity is different from the size
in the first dimension, there might be a gap in between the last element of
the first column and the first element of the second column. This property
could me very useful if you want to resize efficiently your array, but it is
also needed for people who want to get the very best from memory transfer and
vectorization: these people might need to have the first element of teach column
at the beginning of a cacheline. Our two-dimensional array implementation allows
this kind of optimization.

There are many different use cases for such a two-dimensional array. The most
common ones are described below:

- **Matrix**. In scientific computing, the most common usage for a
  two-dimensional array is a matrix. Matrices can be stored in column-major
  order or row-major order and :cpp:`il::Array2D<T>` uses the column-major
  order. This should be your preferred way to store matrices. Even though the
  same performance can we achieved with column-major and row-major order
  matrices, most numerical libraries have been optimized for column-major order
  storage and some of them still gives better performance for this storage.

- **Images**. Images are also commonly stored in two dimensional arrays. For
  instance, a grayscale landscape image with a width of 1000 pixels and a height
  of 500 pixels can be stored in a
  :cpp:`il::Array2D<unsigned char> img{1000, 500}` array. To access the pixel
  at position :cpp:`(x, y)` in the image, all you need is to type
  :cpp:`img(x, y)`. The usual convention is that pixels are stored from left
  to right within one row of the image, and that rows are stored from'
  the top to the bottom of the image. For instance, the pixel at the top left
  corner of the image is :cpp:`img(0, 0)` and the pixel at the bottom right
  corner of the image is :cpp:`img(999, 499)`. Even though, it gives you the
  feeling of working upside down, most libraries store image in such a way for
  historical reasons.

  .. image:: array2d-image.png
      :scale: 20 %
      :alt: alternate text
      :align: center

  Again, the Fortran order is the right order as :cpp:`img(x, y)` and
  :cpp:`img(x + 1, y)` are next to each other in memory.

- **Data**. You can store more general data in an `il::Array2D<T>`. For
  instance, imagine that you need to store in memory 1000 points of the 3
  dimensional space. It might be a very good idea to store those elements in
  a :cpp:`il::Array2D<float>`.

  .. code-block:: cpp

      #include <il/Array2D.h>

      const il::int_t nb_points = 1000;
      const il::int_t dim = 3;
      il::Array2D<float> point{nb_points, dim};

  If you want to access the x-coordinate of the point of index :cpp:`i`, all
  you have to do is type :cpp:`point(i, 0)`. Bare in mind that, as
  :cpp:`il::Array2D<T>` stores element in Fortran order, in memory, you'll find
  all the x-coordinates of the points, then all the y-coordinates of the points
  and finally all the z-coordinates of the points. This seem surprising at first
  as most people expect to have the coordinates of a given point next to each
  other, but this layout, also known as the "structure of arrays" layout is
  the most efficient for many algorithms.

As you can see, unlike many linear algebra libraries out there,
:cpp:`il::Array2D<T>` has been designed to hold different kind of data. Even
though we also provide a :cpp:`il::Array2C<T>` container that stores data in
row-major order (also known as C-major order), we encourage you to use
this data structure which is at least as efficient and sometimes more efficient
in most cases.

Construction
------------

There are many different ways to construct a :cpp:`il::Array2D<T>`.

1. The default constructor construct a :cpp:`il::Array2D<T>` of size 0 and
   capacity 0 in both dimensions. It does not allocate any memory and is
   extremely fast.

   .. code-block:: cpp

       #include <il/Array2D.h>

       il::Array2D<unsigned char> v{};

2. It is also possible to construct a :cpp:`il::Array2D<T>` giving its sizes
   in both dimensions :cpp:`n0` and :cpp:`n1`. An array of these sizes will
   be constructed. Most of the time, the capacity will be set respectively to
   :cpp:`n0` and :cpp:`n1`, but this is something you cannot count on as
   some parameters of the library might have been tweaked to align memory
   and create a larger capacity in the first dimension.

   .. code-block:: cpp

       #include <il/Array2D.h>

       const il::int_t n0 = 1000;
       const il::int_t n1 = 500;
       il::Array2D<unsigned char> img{n0, n1};

   As with :cpp:`il::Array<T>`, the initialization behavior of the elements
   depends upon the type :cpp:`T`.

   - For numeric types such as :cpp:`bool`, :cpp:`unsigned char`, :cpp:`char`,
     :cpp:`int`, :cpp:`il::int_t`, :cpp:`float`, :cpp:`double` the memory should
     be considered as uninitialized. Therefore, for such arrays, reading the
     element :cpp:`img(x, y)` before it has been set results in undefined
     behaviour. In practice:

     - In *release mode*, the memory is left *uninitialized*.
     - In *debug mode*, the memory is set to a special value that makes bugs
       that come from noninitialization easier to track. Floating point types
       are set to :cpp:`NaN` and integers are set to the largest number
       :cpp:`1234...` that the type can hold.

   - For objects, the elements are default constructed. As a consequence they
     are properly initialized. As a consequence, this constructor must be called
     with types :cpp:`T` with a default constructor. However, the
     :cpp:`il::Array2D<T>` container also comes with constructors and methods that
     works with types that do not provide any default constructor.

3. One can also explicitly ask to construct an array of sizes :cpp:`n0` and
   :cpp:`n1` with a default value supplied as a second argument. For instance,
   if you want to create an array of floating point with all the elements
   initialized to :cpp:`0.0`, one can use the following code:

   .. code-block:: cpp

       #include <il/Array2D.h>

       const il::int_t n = 1000;
       il::Array2D<double> a{n, n, 0.0};

4. The previous constructor is only available for types that can be copied.
   Unfortunately, there are some types such as :cpp:`std::atomic<int>` that
   cannot be copied. Other types may be expensive to copy. As a consequence,
   another constructor is available where you do not provide a default object,
   but arguments to construct this default object.

   .. code-block:: cpp

       #include <atomic>
       #include <il/Array2D.h>

       const il::int_t n = 1000;
       il::Array2D<std::atomic<int>> a{n, n, il::emplace, 0};

5. Finally, in case you want to explicitly initialize the element to a list of
   values, the following constructor can be used.

   .. code-block:: cpp

       #include <il/Array2D.h>

       il::Array2D<double> M{il::value, {{0.0, 1.0}, {2.0, 3.0}, {4.0, 5.0}}

   This code creates a matrix :cpp:`M` with 2 rows and 3 columns. The entries
   are given column after column. As a consequence we will get the following
   matrix:

   .. image:: array2d-small-matrix.png
      :scale: 20 %
      :alt: alternate text
      :align: center

Arrays own their memory. As a consequence, it is impossible to construct an
array from a pointer that has been allocated by :cpp:`malloc`, :cpp:`new` or
any other library. If you want to warp such a pointer inside an InsideLoop
object, the type you are looking for are :cpp:`il::ArrayView2D<T>` and
:cpp:`il::ArrayEdit2D<T>`.

Copy construction and assignement
---------------------------------

Copy construction and assigment works in the same way as :cpp:`il::Array<T>`.
The identity of an :cpp:`il::Array2D<T>` object is defined by its sizes in both
dimensions and the elements it holds. As a consequence, when you copy construct
an two-dimensional array, you are guarenteed to have a new array with the same
sizes as the original and with elements that compare equal. The same warning
is still true for capacities which are not part of the identity of the object
and therefore are usually not conserved during a copy or an assignement.

Destruction
-----------

The destruction of a two-dimensional array releases its memory. If :cpp:`T` is a
class, all the objects contained in the array will be destructed in an
unspecified order before the memory is released.

Accessing the elements
----------------------

1. **Read access with the parenthesis operator**.
   The method :cpp:`size(il::int_t d)` returns the size of the two-dimensional
   array along dimension :cpp:`d` as an :cpp:`il::int_t`. The elements can be
   accessed through the parenthesis operator. For instance, here is a code
   that walks an image on the row :cpp:`y` and walks from right to left and
   returns the psition of the first white pixel (with a value larger than 128).

   .. code-block:: cpp

       #include <il/Array2D.h>

       il::int_t findWhite(const il::Array<unsigned char>& img, il::int_t y) {
         IL_EXPECT_FAST(y >= 0 && y < img.size(1));

         for (il::int_t x = img.size(0) - 1; x >= 0; --x) {
           if (img(x, y) >= 128) {
             return x;
           }
         }
         return -1;
       }

   Accessing elements out of bound is forbidden. The behavior of the library is
   different, wether we are in debug mode or in release mode.

   - *Debug mode*: In debug mode, the library will call :cpp:`std::abort()`
     which will let you easily find the culprit with a debugger.
   - *Release mode*: In release mode, no bounds checking is done and accessing
     arrays out of bounds will result in undefined behavior. But this will
     let the compiler do a lot of optimization, such as vectorization which would
     be impossible in C++ with bounds checking.

2. **Write access with the parenthesis operator**.
   The parenthesis operator can also be used on a arrays for write access.

To be continued...

