.. role:: cpp(code)

    :language: cpp

.. role:: bash(code)

    :language: bash

String
======

Overview
--------

Quite often, people wonder why InsideLoop library provides a string type as
one already exists in the standard library. We often answer this question
with 2 words: **unicode** and **performance**.

- **Unicode**. Text encoding has been a major pain
  in the digital world. We had :bash:`ASCII` which links a table of 128
  "characters" to the first 128 integers, :bash:`ISO8859-1` that fills the
  reminding 128 spots available for european people and many other standards that
  fill those last 128 spots available for different regions of the word. As things
  would be too simple, text documents do not encode which table they use for their
  encoding which is a major reasons why you might sometimes receive emails with
  strange characters. Unicode has been designed to solve this problem and to give
  a table that contains all the "characters" of the world. When we talk about
  "characters", it is a very broad concept as smileys are part of this table. But
  more interesting is the fact that the notion of "character" might be fuzzy in
  some languages. To be more accurate with the unicode standard, we will now talk
  about code points or runes. At the beginning, people thought that 16 bits would
  be enough to store all these runes. But it turns out that
  :math:`2^{16} = 65 536` is not big enough for all the runes of the world. But
  people discovered that too late and many platforms, including Windows has
  already started to use an encoding that could not hold all the runes of the
  world. This encoding has been adapted to handle the new runes and an encoding
  called :bash:`UTF-16` has emerged from that. One of the problem of this encoding
  is that it uses 16 bits for most of its characters whereas most of the
  characters used today in computer science are from the :bash:`ASCII` table. As
  a consequence, the :bash:`UTF-8` encoding has been designed to be efficient
  both in terms of memory and parsing performance. This is the encoding that
  dominates the web today and should be the core of any string object. But as
  history has its dark corners, we have to deal with the fact that some filenames
  on most platform can sometimes not fit in a unicode string. As a consequence,
  we designed our own :cpp:`il::String` container that:

  - most of the times, will handle a unicode string stored in :bash:`UTF-8`
  - can store array of bytes

  It contains its own internal flag to know if the string it holds is a valid
  :bash:`UTF-8` string. Many methods available in the :cpp:`std::string` container
  have been removed as the don't make sense with a unicode string. For instance,
  accessing the i-th element with :cpp:`str[i]` would be too dangerous for
  non-specialist who usually believe that they access the i-th character in such
  a way. This is not true in many languages, including european languages such
  as french or german. Some operations which are known to be very inefficient
  such as :cpp:`+` for string concatenation have also been removed. We provide
  a different API for that which is much more efficient.

- **Performance**. Moreover, our :cpp:`il::String` container has been optimized
  in a unique way for every platform. We use small string optimization where
  every string with a length of 22 or less is stored on the stack. If you use
  the :cpp:`std::string` from your vendor, you might get different optimization
  wether you use :bash:`libstdc++` (from gcc), :bash:`libc++` (from clang) or
  Visual Studio.

But enough talk, let's go into our first example:

.. code-block:: cpp

    #include <il/String.h>

    il::String name = "François Fayard";

This code stores a name into an :cpp:`il::String`. The string which does contain
a 'ç' character used in french will be stored as :bash:`UTF-8`. To be sure that
no problems occur with your compiler, your file needs to be saved in
:bash:`UTF-8` which we recommend. The string :cpp:`name` will contain internally
a flag that states that the string is a valid unicode string. This is checked in
debug mode but not in release mode for performance reasons. In case you do not
use :bash:`UTF-8` to save your source files (this is very bad), you should
replace the code with

.. code-block:: cpp

    #include <il/String.h>

    il::String name = u8"François Fayard";

to enforce the correct conversion to :bash:`UTF-8`.

In case you want to know the size of that string, it is accessible with the
:cpp:`size()` method. Bare in mind that the size of a string is not the number
of characters of that string (which has no meaning, or a very contrived one),
but the number of bytes used to store the string.

.. code-block:: cpp

    #include <iostream>
    #include <il/String.h>

    il::String name = "François Fayard";
    std::cout << "Size: " << name.size() << std::endl;

The above code, gives a size of 16. There are 14 characters that belong to the
:bash:`ASCII` table and that need one byte in :bash:`UTF-8`. There is also one
character that needs 2 bytes to be represented. This explains the size of 16.
Bare in mind that the unicode representation of a string is not unique. As a
consequence, some platforms might give a different size for the same string even
though they are all encoded in :bash:`UTF-8`!

One can append another string to the previous one with

.. code-block:: cpp

    #include <il/String.h>

    ...
    name.Append(" wishes you a great day!");

You can query your string to know if contains an :bash:`UTF-8` string with the
following method

.. code-block:: cpp

    #include <iostream>
    #include <il/String.h>

    ...
    if (name.isUtf8()) {
      std::cout << "Congratulation, your string is valid UTF-8" << std::endl;
    } else {
      std::cout << "All you have is an array of bytes" << std::endl;
    }

This method just checks a flag in the string. As a consequence, it is extremely
fast. Strings that are build from strings that come from the source file are
supposed to be :bash:`UTF-8` by default and are considered as such unless
otherwise stated. But strings that come from untrusted sources such as a text
file or filenames are flagged as array of bytes. They can only be flagged as
:bash:`UTF-8` strings once they have been checked.

If you want to display the string, one can easily get access to it as an
array of char that ends with the null character :cpp:`'\0'` needed by many
APIs.

.. code-block:: cpp

    #include <iostream>
    #include <il/String.h>

    ...
    std::cout << name.asCString() << std::endl;
