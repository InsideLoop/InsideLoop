Array
=====

@param context

The dynamic array `il::Array<T>` is a container that stores elements of type
`T` in contiguous locations of the memory. Its size `n` can change during the
lifetime of the object. The following code creates an array of `n` floating
point and stores the inverse of the first `n` positive integers.

.. code-block:: cpp

    const il::int_t n = 100;
    il::Array<double> v{n};
    for (il::int_t i = 0; i < n; ++i) {
      v[i] = 1.0 / (1 + i);
    }

In this example, the size of the array is defined at the construction of the
object and is fixed in this code. But their size an grow or shrink during the
lifetime of the object. You can use the `Append` method to add an element at
the end of an array. The following function takes an array `v` and returns
the array containing all the positive elements of `v`.

.. code-block:: cpp

    il::Array<double> f(const il::Array<double>& v) {
      il::Array<double> ans{};
      for (il::int_t i = 0; i < v.size(); ++i) {
        if (v[i] > 0) {
          ans.Append(v[i]);
        }
      }
      return ans;
    }

The default constructor creates an array of size 0 and every time the `Append`
method is called, the array size grows by 1 and the new element is placed at
the end of the array.
To store its elements, the array `v` allocates memory on the heap. As an array
must always store all of its elements in contiguous locations of the memory,
when its size grows from `n0` to `n1`, the array might need to allocate a new
memory chunk of size `n1`, and copy the first `n0` elements to the new
location. As memory allocation takes some time, `il::Array<T>` tries to
minimize the number of such allocations. In order to do that, an array has
a capacity `r` which is larger that its size `n`. The array has the room to
store `r` elements on the memory chunk that has been allocated, and as long as
the size of the array stays below `n`, resizing the array is a fast operation.
When the size of an array `n` reaches its capacity, appending an element to
the array triggers an allocation of a new chunk of memory that can store about
`2n` elements, the first `n` elements of the array are copied to the
new location, and the new element is stored at the end. We say that the growth
factor of the dynamic array is 2. No memory allocation will be triggered until
the size of the array has reached `2n`.
If you have an upper bound for the number of elements that will be used in the
sequence, you can set the capacity at the very beginning of the algorithm. In
the previous algorithm, as we know that `ans` will contain less elements than
`v`, we can decide to set the capacity of `ans` to the size of `v` at the very
beginning.

.. code-block:: cpp

    il::Array<double> f(const il::Array<double>& v) {
      il::Array<double> ans{};
      ans.Reserve(v.size());
      for (il::int_t i = 0; i < v.size(); ++i) {
        if (v[i] > 0) {
          ans.Append(v[i]);
        }
      }
      return ans;
    }

That way, we can be sure that only one memory allocation will occur in this
function. Note that the capacity of an array is not part of the identity of
the object: the copy of an `il::Array<T>` is not guaranteed to have the same
capacity of the original array.

Construction
------------

There are many different ways to construct an array.

.. code-block:: cpp

    il::Array<double> v{};

The default constructor construct an array of size 0 and capacity 0. As a
consequence, it does not allocate any memory.

.. code-block:: cpp

    const il::int_t n = 1000;
    il::Array<double> v{n};

It is also possible to construct an array giving its size `n`. An array of
size `n` (and usually of the same capacity) will be constructed. The
initialization behavior of the elements depends upon the type `T`.

- For numeric types such as `bool`, `unsigned char`, `char`, `int`, `il::int_t`,
  `float`, `double` the memory should be considered as initialized. Therefore, 
  for such arrays, reading the element `v[k]` before it has been set results
  in undefined behaviour. In practice:
  - In release mode, the memory is left uninitialized.
  - In debug mode, the memory is set to a special value that makes bugs that
    come from non initialization easier to track. Floating point types are
    set to `NaN` and integers are set to the largest number `1234...` that
    the type can hold.
- For objects, the elements are default constructed. As a consequence they
  are properly initialized.

.. code-block:: cpp

    const il::int_t n = 1000;
    il::Array<double> v{n, 0.0};

 In case you want to explictly initialized the element to a default value, this
 constructor can be used.

.. code-block:: cpp

    il::Array<double> v{il::value, {1.0, 2.0}}

 Sometimes, we want to initialize an array of a small size whose values are
 known. This constructor, using initializer list, is useful for that.

Copy construction and assignement
---------------------------------

.. code-block:: cpp

    void f(const il::array<double>& v) {
      il::Array<double> w = v;
      ...
    }


An array owns its memory. As a consequence, when an array `w` is constructed as
a copy of the array `v`, any changed made to one of the array will not affect
the other one. The identity of an array is defined by its size and the elements
that it contains. As a consequence, if `w` is constructed from `v`, it will
allocate its own memory and copy all the elements of `v` in it. Note that the
capacity of `w` might be different from the one of `v`. Usually, `w` will have
the same capacity as its size.

.. code-block:: cpp

    il::Array<double> v{};
    ...
    il::Array<double> w = std::move(v);

If you want to transfer the ownership of the memory of `v` to `w`, you can
use the move constructor that will assign the memory chunk previously used by
`v` to the newly constructed array `w`. After the move construction has been
completed, `v` will have a size of 0 and a capacity of 0.

.. code-block:: cpp

    il::Array<double> v{};
    il::Array<double> w{};
    ...
    w = v;

The assignement follows the same behaviour. If the capacity of w is larger
than the size of v, no memory allocation will be done and all the elements will
be copied from `v` to `w`. Otherwise, a new memory chunk will be allocated to
copy the elements of `v` and the old memory chunk will be released.

.. code-block:: cpp

    il::Array<double> v{};
    il::Array<double> w{};
    ...
    w = std::move(v);

Nothing special has to be said about move assignement.

Destruction
-----------

The destruction of an array releases its memory. If `T` is a class, all the
objects contained in the array will be destructed in an unspecified order
before the memory is released.

Accessing the elements
----------------------

The method `size()` returns the size of the array and the elements can
be accessed through the bracket operator. As a consequence, the following code
can be used to compute the sum of all the elements of an array.


.. code-block:: cpp

    double sum(const il::Array<double>& v) {
      double ans = 0.0;
      for (il::int_t i = 0; i < v.size(); ++i) {
        ans += v[i];
      }
      return ans;
    }

The array has been implemented in such a way that any smart compiler won't
call the `v.size()` at every iteration. Therefore it can be kept in the
loop without any performance hit.

The last element of an array can be accessed with the `back()` method. In case
you want to write to this location, you can use the `Back` method.

.. code-block:: cpp

    il::Array<double> v{n};;
    ...
    v.Back() = 5.0;
    std::cout << "The last element is now: " << v.back() << std::endl;

Changing the size of an array
-----------------------------

The size of an array can be changed with the method `Resize(n)`. If the
original array is larger than `n`, the first `n` elements of the array will be
kept. Is the original array is smaller thant `n`, the new elements will be
or not initialized depending upon the type `T` in the same way things happened
for the constructor.

In case you want to specify a value for the arrays that might grow, one
can use the following syntax.

.. code-block:: cpp

    il::Array<double> v{};
    ...
    v.Resize(n, 0.0);

If you want to change the capacity of an array, you can use the `Reserve(n)`
method which makes the capacity of the array at least equal to `n`. Usually,
if `n` is less than the current capacity, nothing will be done. But if `n`
is larger than the crurrent capacity, the new capacity will be equal to `n`.
The current capacity of an array is returned by the method `capacity()`.

In order to add one element at the end of an array, one can use the `Append(x)`
method. The move semantics are also available and you can use the code:

.. code-block:: cpp

    il::Array<il::Array<double>> v{};
    il::Array<double> x{};
    ...
    v.Append(std::move(x));

to move `x` to the last position of the array. It is also possible to construct
in place some elements with the following syntax.

.. code-block:: cpp

    il::Array<il::Array<double>> v{};
    v.Append(il::emplace, 10, 0.0);

This code will construct an array of double of size 10, filled with `0.0` as
the first element of `v`.

View of an array
----------------

Its is sometimes useful to create a view of an array. A view of an array does
not own its memory but allows to view its elements. For instance, the
following code

.. code-block:: cpp

    il::Array<double> w{};
    ...
    il::ArrayView<double> v = w.view();

allows `v` to have a view on the elements of `w`. Any change made to `w` will
be reflected in its view `v`. Be aware that, because of reallocation, any
change in the size or the capacity of `w` might invalidate the view `v`.
Working with `v` after such a change has been made to `w` results in undefined
behaviour.

Note that views can be used to have a look at some parts of an array. For
instance, the following code

.. code-block:: cpp

    il::Array<double> w{};
    ...
    il::ArrayView<double> v0 = w.view{il::Range{0, w.size() / 2}};
    il::ArrayView<double> v1 = w.view{il::Range{w.size() / 2, w.size()}};

will create a view on the first half of the array and another one on the second
part.

It is also possible to create an `il::ArrayEdit<T>` from an array. It works
the same way but also allows to change the elements of the array.

Raw access
----------

When working with other C++ libraries or other languages, it might be useful
to get a raw pointer to the first element of the array. This can be done with
the `data()` and `Data()` methods. The first one returns a pointer to const
while the second one returns a pointer that can be used to write to the array.
