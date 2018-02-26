The C++ language comes with many integer types inherited from C. They differ by the range of integers they can represent and how overflow is handled. In integer arithmetic, overflow occurs when the result of an operation such as `+`, `-`, `*` and `/` lies outside of the range that can be represented by the type. In this article, we'll give you some recommendations on how to choose the best integer type for your problem. Those recommendations are based on both convenience and efficiency. Because efficiency is platform dependent, we'll restrict ourselves to general purpose CPUs such as the ones produced by Intel, AMD and ARM. Most devices, wether they are phones running iOS/Android, small devices such as the Raspberry Pi, laptops and desktops running Windows/macOS/Linux, and even servers, all fall into that category. Those platforms have in common the usage of two's complement to represent signed integers, pointers having the same size as a general purpose register, and the availability of vector instructions.

C++ offers both signed and unsigned integer types. The signed types are called `signed char`, `short`, `int`, `long` and `long long` and differs by their storage size. A n-bit signed type can handle any integer from -2^(n-1) to 2^(n-1) - 1 included. Those signed types have their unsigned equivalent that use the same number of bits. They are called `unsigned char`, `unsigned short`, `unsigned int`, `unsigned long` and `unsigned long long`. An unsigned type of n bits can hold any integer in between 0 and 2^n - 1 included. On the 64-bit Linux system I am working with, a `signed char` has a width of 8 bits, a `short` use 16 bits, an `int` use 32 bits, and both `long` and `long long` use 64 bits.

Signed types and their unsigned counterparts do not only differ in the range of values they can hold. They also have a very different way of handling overflow.

Arithmetic of signed types
==========================

In order to fix ideas, we'll work with the `int` type, but all signed types behave similarly. Wether your platform is 32-bit or 64-bit, this type should have a width of n = 32 bits. Therefore, the `int` type can store any integer in the range from -2^(n-1) = -2'147'483'648 to 2^(n-1) - 1 = 2'147'483'647 included. All arithmetic operations `+`, `-` (negation), `-` (difference), `*`, `/` can overflow. For instance, if a = 2^(n-1) - 1 is the largest `int` and b = 1, a + b cannot be represented as an `int`. Similarly, if a = -2^(n-1) is the smallest `int`, -a cannot be represented by an `int`. There are many ways to handle overflow, but for signed integers, C and therefore C++ have decided to take a radical solution: the programmer has the responsability to never let this situation happen. In case the programmer does not respect this contract, the behavior of the whole program is undefined. Practically, the result of an operation that overflows could depend upon your compiler and the optimization flags you have used. Overflow could also crash your program, leak sensitive information and even corrupt your data. An old prank in gcc 1.17 was starting the game "Towers of Hanoi" when undefined behavior was detected at runtime. In order to experience undefined behavior in a gentle manner, let's write a small program where n = 2^30 has been carefuly chosen.

.. code:: cpp
    #include <iostream>

    int f(int a) { return (4 * a) / 4; }

    int main() {
      int a = 1073741824;
      int b = f(a);

      std::cout << b << std::endl;

      return 0;
    }

Ignoring overflow, one might expect that such a program would print 1'073'741'824. But experience shows that it really depends upon your compiler options. On a Linux system, with Gcc 5.4.0, if compiled without any optimization (`g++ -O0 main.cpp -o main`), the program prints 0. But if compiled with optimizations (`g++ -O2 main.cpp -o main`), it prints 1'073'741'824. The reason for this undefined behavior is that 4 * a = 2^32 cannot be represented as an int. As a consequence, we get undefined behavior and the compiler can safely ignore those problems. When optimizations are turned off, the compiler generates assembly instructions for both the multiplication and the division by 4. As we are on a system using two's complement representation of signed integers, every operation is computed modulo 2^32. As a consequence 4 * a = 2^32 is reduced to 0. When optimizations are turned on, the compiler decides to simplify (4 * a) / 4 into a and does not generate a single assembly instruction.

Arithmetic of unsigned types
============================

The C++ language also come with unsigned integers. We'll work with the type `unsigned int` that should have a width of n = 32 bits on your platform. But all unsigned integers obey to the same rules. The range of this type goes from 0 to 2^n - 1 = 4'294'967'295 included.
Contrary to what happens with signed integers, arithmetic operations do not overflow but wrap around: the result of every operation is reduced modulo 2^n = 4'294'967'296. For instance, if a = 4'294'967'295 is the largest unsigned int and b = 1, the result of a + b is 0 because 4'294'967'295 + 1 = 0 mod 2^n. This behavior similar to the one we used to have in our old cars with mechanical odometers. The same thing happen if we substract b = 1 from a = 0. As -1 cannot be represented with an `unsigned int` type, its value modulo 2^32 is computed and we obtain the largest `unsigned int`, 4'294'967'295.
One of the advantage of that choice is that operations for + and - are always well defined and follow most of the rules we have in mathematics. For instance (a + b) - b = a. But things get tricky when inequality is involved. For instance, `a <= b` does not imply `a + c <= b + c`. Have a try with a = 2^n - 2, b = 2^n - 1 and c = 1. Things get even worse with `*` and `/`. For instance, with unsigned integers, (4 * a) / 4 is not always equal to a. If a = 1'073'741'824, 4 * a = 4'294'967'296 which is reduced to 0. Therefore, with unsigned integers, (4 * a) / 4 is equal to 0 and not the expected a. If the function `f` in our previous program had to be compiled with `unsigned int` instead of `int`, the compiler could not have replaced (4 * a) / a with a. This phenomenon where some optimizations can be done with signed integers but not with unsigned integers is quite common: the arithmetic of signed integer allows many operations to be optimized away. Moreover, the arithmetic of unsigned integers is full of traps and many bugs find their roots in the misconception most programmers have about it. Here is a code that hides such a bug. Given an horizontal line of an 8-bit grayscale image and a range on that line, it tries to find the first "white" pixel (a gray value above 128), from right to left.

.. code:: cpp
    unsigned int find(unsigned int k_left, unsigned int k_right,
                      unsigned char* image_line) {
      for (unsigned int k = k_right; k >= k_left, --k) {
        if (image_line[k] >= 128) {
          return k;
        }
      }
      return -1;
    }

First, let's rule out the -1 that might be surprising to some programmers when you are expected to return an unsigned integer. This value is returned when no pixels with a value above 128 has been found and should be treaded as an "error" by the caller of this function. As an `unsigned int` is expected, the integer -1 is reduced modulo 2^32 which leads to 4'294'967'295, the largest `unsigned int`. Even though such a way to return an error is dangerous, it is extremely fast and is a common pattern. But I would like to focus on another point, much more dangerous. The bug that hides in this code does not show up often and is therefore extremely nasty. This code was running fine, but one day, a user called the function with kx_left = 0 with an image on which all pixels were black. The program crashed. The reason for this is that an unsigned int is always nonnegative. Therefore, the condition k >= 0 is always met. When k = 0, the operation --k decrements the index to 4'294'967'295. Then, the program tries to access image_line[k] which is likely to be out of bounds and can cause a crash. In order to fix this bug, the loop can be changed to:

.. code:: cpp
    for (unsigned int k = k_right; k != k_left - 1, --k)

But, with this fix, the code is not as straightforward to understand. It could also lead to another bug if k_right = 2^n - 1 and k_left = 0 even though it is very unlikely to happen. The best fix was to move away from unsigned integer. I hope that you understand that the arithmetic on unsigned integers is tricky, prevents some compiler optimizations and leads to nasty bugs. Threfore, unsigned integers should be avoided unless you really need to compute modulo 2^n.

In order to conclude our review of the signed and unsigned integers types, you should know that mixing them might give unexpected behavior. Try running the following code.

.. code:: cpp
    #include <iostream>

    int main() {
      const int a = -1;
      const unsigned int b = 3;

      if (a >= b) {
        std::cout << "The world of computer integers is strange!" << std::endl;
      }

      return 0;
    }

The reason why a >= b evaluates to true is that the signed `int` a is promoted to an `unsigned int` before the comparison is made. The value -1 is reduced modulo 2^32 which leads to 4'294'967'295. As a consequence, a >= b evaluates to true.

The width of different integer types
====================================

The C++ standard states that the width of integer types should obey to the following rules:

- width(signed char) <= width(short) <= width(int) <= width (long) <= width(long long)
- width(signed char) >= 8
- width(short) >= 16
- width(long) >= 32
- width(long long) >= 64

But it does not offer any other guarantee. The following table shows the width of those integers on both Unix (Linux, macOS, iOS, Android) and Windows in their 32-bit and 64-bit versions.

======= ======= ============ ===== === ==== =========
        pointer signed  char short int long long long
======= ======= ============ ===== === ==== =========
Unix         64            8    16  32   64        64
Unix         32            8    16  32   32        64
Windows      32            8    16  32   32        64
Windows      64            8    16  32   32        64
======= ======= ============ ===== === ==== =========

The unsigned versions have the same width. Be aware that even though some type may have the same width and the same signedness, they are still considered different by the type system.





We'll see later that for performance and practical reasons, using an integer that has the same size as the pointer is often needed. Unfortunately, the C++ language does not provide a type that fulfills this requirement. As a consequence, the standard commitee decided to introduce the following aliases: `std::size_t` and `std::ptrdiff_t`. The alias `std::size_t` is an unsigned integer and has the same width as the pointers. This type has been used extensively in the C++ standard library. The alias `std::ptrdiff_t` is a signed integer and has the same width. On my 64-bit Linux system, `std::size_t` is an alias to `unsigned long` and `std::ptrdiff_t` is an alias to `long`. On a 64-bit Windows, `std::size_t` is an alias for `unsigned long long` and `std::ptrdiff_t` is an alias for `long long`.

Because it is often needed to work with integers with a known size, some aliases have been defined for this purpose. The aliases `std::int8_t` and `std::uint8_t` are aliases for 8-bit integers, respectively signed and unsigned. The standard also provides the aliases `std::int16_t`, `std::uint16_t`, `std::int32_t`, `std::uint32_t`, `std::int64_t`, `std::uint64_t` whose names speak for themselves. As some exotic architectures do not provide integer types of those width, these aliases might not be defined, but most platforms provide them.

Performance of different integers
=================================

When it comes to performance, we cannot abstract ourselves from the underlying platform. All the benchmarks given have been run on an 64-bit Intel CPU. But similar results should be obtained on most 64-bit Intel, AMD and ARM processors that powers your your phone, your laptop, workstation or server.

Performance of loners
=====================

We 'll first discuss performance of integers that do not come in arrays. Their usage might impact the performance of the program in 2 ways. Obviously, the time it takes to perform arithmetic operations on them has an influence. But one should also consider that overflow for signed integers leading to undefined behavior, compilers can sometimes avoid some computations. And there is nothing faster than an operation that never occured.

Let's first try with a small experiment where we compile the following code where a function that compute the sum of two different signed integers is generated for different widths.

.. code:: cpp
    std::int8_t f(std::int8_t a, std::int8_t b) { return a + b; }
    std::int16_t f(std::int16_t a, std::int16_t b) { return a + b; }
    std::int32_t f(std::int32_t a, std::int32_t b) { return a + b; }
    std::int64_t f(std::int64_t a, std::int64_t b) { return a + b; }

Surprisingly, all those functions generate the same assembly on x86-64:

.. code:: assembly
    addl %esi, %edi
    movl %edi, %eax
    ret

This code just adds the two integers on the first line, move the result into the register used to return values, and then return from the function. On a 64-bit platform, general purpose registers can hold 64 bits of data, but such registers can only hold a single integer. As we can see from the assembly, those registers are used to hold all kinds of integers and it turns out that the `addl` instruction operates on all the 64 bits of the register. It turns out that, because of the way integers are stored, this operation gives the right result even for shorter integers. We can deduce that the `+` operation will be as fast on all kind of integers. The same experience shows that both `-` and `*` generates the same assembly code no matter which signed integer is used. Therefore, the performance is the same for these operations for every width.

Divisions of integers are up to 10 times slower that addition and multiplication and should be considered as an expensive operations. A quick inspection of the assembly generated by a division shows that different integers size lead to different assembly operations. But a benchmark is needed to get the difference in performance. Fro instance, to test the speed of the division of two `std::int16_t` variables, we ran the following code with n = 1'000'000'000.

.. code:: cpp
    std::int16_t f(std::int16_t a, std::int16_t b, int n) {
      std::int16_t ans = a;
      for (int k = 0; k < n; ++k) {
        ans /= b;
      }
      return ans;
    }

Our CPU was running at 3.3 GHz during the benchmarks, which gives 3'300'000'000 cycles per second. Timing the function call allowed us to get the duration, in cycles, of one loop. We repeated the experience with different size of integers and we obtained the following results:

======== ====== ====== ====== ======
          8 bit 16 bit 32 bit 64 bit
======== ====== ====== ====== ======
  signed  21 cy  21 cy  21 cy  38 cy
unsigned  21 cy  21 cy  21 cy  31 cy
======== ====== ====== ====== ======

As a consequence, on 64 bit processors, code that relies heavily on integer divisions should use 32 bits integers if possible. Using smaller integers does not improve performance. If 64 bit integers are needed, using unsigned integers gives a performance boost.

Finally, it should be noted that a common case is the division by 2 when the factor 2 is known at compile time. As integers are stored in a binary format, this operation can be performed much more efficiently. For every integer size, wether the integer is signed or unsigned, a loop takes 3 cycles. Dividing by 3 takes 5 cycles and dividing by 7 takes 7 cycles. As a consequence, large performance boost should be expected when the number used to divide is known at compile time.

As a conclusion, +, - and * generates exactly the same code for signed and unsigned integers no matter which width they use. Division by a number known at compile time also gives the same performance on signed and unsigned integer. But if you need to divide a number by another on which you have no information at compile time, you'll get better speed if you use 32 bits integers. If 64 bits integers are needed, you'll get a bit more performance when using unsigned integers.

Performance of integers as array indices
========================================

Another common usage for integer is array indexing. In order to understand the impact of the size of integers on array indexing, we compiled the following function on x86-64:

.. code:: cpp
    double f(const double* p, int k) {
      return p[k];
    }

It generated the following code

.. code:: assembly
    movslq %esi, %rsi
    vmovsd (%rdi,%rsi,8), %xmm0
    ret

The first operation converts the 32 bits integer k into a 64 bits integer. The second line gets the element from memory and put it into `%xmm0` which is the register used to return floating points. If you change the type of `k` from `int` to `std::ptrdiff_t` which is 64 bits on this platform, the assembly code generated is reduced to:

.. code:: assembly
    vmovsd (%rdi,%rsi,8), %xmm0
    ret

We clearly see that no integer conversion is done, and we might expect a faster operation. This might look scary to people using `int` types for indices in loops. But for signed integers, the compiler can happily promote our `int` to a 64 bits integer. As a consequence, `int` are almost always as efficient as 64 bits signed integers for array indexing on 64-bit platforms. It is still possible to build corner case benchmarks that shows a performance difference in between 32 bits and 64 bits signed integers for array indexing. But they are quite rare.
Using 64 bits unsigned integers give the same performance for array indexing if no heavy transformation of the loop is required by the compiler. Unfortunately, when loops transformation such as unrolling and vectorization, which could provide performance boost by a factor of 10, there are still many cases where compilers do a better job at producing good code when the index of the loop is signed. There are many cases where changing the integer from unsigned to signed gave a 5x performane boost, just because was able to generate vectorized code with the signed integer. Most of the time, it should have been to generate the same code, even with unsigned integers. But, as we have seen, unsigned integers make the job of the compiler more difficult, and compilers give up more quickly with them. I prefer not to show some code where we see that, because the difference in performance between signed and unsigned index can be seen only with some very specific version of the compilers. But this behaviour is still seen with recent compilers such as Gcc 7, Clang 5 and Icpc 18.

Performance of arrays of integers
=================================

When integers don't come as loners but with millions of their friends, such as in arrays, the general rule is to make as small as you can. The reason is that many algorithms are bandwidth bound or compute bound. If the algorithm is bandwidth bound, it means that the bottleneck of the program is the memory transfer in between the memory (the RAM or the caches inside a processor) and the registers of the processor. If we use 8 bits integers instead of 32 bits integers, the array size will be divided by 4, and the transfer and therefor the program will be 4 times faster. If the algorithm is compute bound and process array of elements, most of the time, they can be processed with vector instructions. On modern Intel CPU, a vector width is 512 bits wide. Therefore, we can pack 16 `int` in such registers. If we use `std::int16_t`, we can pack 32 of them. As the operations on vector registers that contain `int` are as fast as those that act on vector registers containing `std::int16_t`, we end up getting twice the speed with `std::int16_t` than with `int`. Because of these two reasons, when integers come in arrays, one should make it as small as possible. Usually those types are used:

- `std::uint8_t`: When 8 bits are enough, as most of the time, we only need nonnegative integers. As a consequence, to get the largest range and get that extra bit, people use `std::uint8_t`. This is quite common in image processing where every pixel of an image is represented by one (or 3 for color images) std::uint8_t that can be in between 0 (black) and 255 (white). One should be careful when doing arithmetic with those kind of integers, must most of the time, those integers are converted to 32 bits floating points before any transformation on the image is needed.
- `std::uint16_t`: When 16 bits are enough, for the same kind of reasons, people also use unsigned integer to get that extra bit. 16 bits integers are also used in image processing.
- `std::int32_t`: When 32 bits are enough, it is often recommended to switch to signed integers as they are much more friendly when it comes to their arithmetic properties.

InsideLoop's standard integer type, `il::int_t`
===============================================

As stated above the `int` type has a width of 32 bits on most 64-bit platforms. As a consequence, it cannot be safely used to store an index of an array of std::uint8_t that exceeds 2'147'483'648 elements which would require more than 2 GB of memory. This is a dramatic change from what we used to have on 16-bit and 32-bit platforms where the `int` type could be safely used as an index for arrays. If you need to be portable accross platforms, using `long` is not the solution either as it has a width of 32 bits on 64-bit version of Windows. Using `long long` would be enough for any platforms, but using such a type would be highly inefficient on 32-bit platforms where such an integer would not fit into a register. This is the reason why the standard came with 2 aliases called std::size_t and std::ptrdiff_t. The first one has been designed so that it can represent any memory size that can be handled by the platform. It is an unsigned integer. The second one has been designed so that it can represent the differrence of two pointers refering to elements of the same array. It is a signed integer and has been designed to be the defacto standard for array indices. Those types are both 32 bit wide on 32 bit platforms and 64 bit wide on 64-bit platforms.

The story could have been more simple if everyone had adopted std::ptrdiff_t for array indices. Unfortunately this is not the case. The C++ standard library has done a major mistake to use the unsigned type std::size_t for its array indices. This mistake has been acknowledged by Bjarne Stroustrup, the creator of C++, but it is now too late to change it. As a consequence many C++ codes are poluted with unsigned integers to conform to the standard library. Some companies such as Google advise to use the type `int` for array indices. As they rely on the standard library for their containers, it explains a lot of the warnings you get when compiling their software. The following pattern is often seen in their code:

.. code:: cpp
    double f(const std::vector<double>& v) {
      double ans = 0.0;
      for (int i = 0; i < v.size(); ++i) {
        ...
      }
      return ans;
    }

The warning that most compiler raise concerns the comparaison `i < v.size()` where we compare the signed integer i with the unsigned integer `v.size()` which is of type `std::size_t`. As explained above, comparison of signed and unsigned integers can be extremely dangerous. The standard library tries to push programmers away from writing loops. Unfortunately, many data structures used in high performance computing cannot be used efficiently with iterators.

Therefore, InsideLoop promotes the use of `std::ptrdiff_t` for indices. Some high performance libraries such as Eigen have made the same choice. We even go as far as promoting it as the standard integer type as it provides the widest range of integers that can be represented efficiently by the underlying platform. The name `std::ptrdiff_t` being rather unintuitive and long to type, the InsideLooop library using the alias `il::int_t`.

In order to make a long story short, the standard integer type used in the Inside Loop API is `il::int_t`. It is an alias for `std::ptrdiff_t` which is a 32 bits signed integer on 32-bit platforms and a 64 bits signed integer on 64 bit platforms.


When should I use something else than `il::int_t` ?
===================================================

Now that integers types have no secret for you, come the difficult question of choosing which type is right for you. Here is my advice:

- General purpose type: ptrdiff_t
  This type is signed and has the largest range that can be processed efficiently on your platform (32 bits on 32-bit platform and 64 bits on 64-bit platform). It's only problem as a general purpose type is it's weird name.
- Use small types when you store and process large arrays of integers
  When we need to store and process thousands or even billions of integers of the same type, size does matter for efficiency. We need to use types with small size to reduce the memory used by the application and to increase its speed when data is transferred in between the CPU and the RAM and when data is processed in vector registers. Using arrays of data types that are twice as small as another one allow to transfer twice the number of elements within the same time and allow vector registers to process twice the number of elements within the same time. For algorithms that are compute bound or bandwidth bound (which is often the case in High Performance Computing), dividing the storage size by two makes the algorithm twice as fast.
  In image processing, people often use 8-bit or 16-bit integers. As the intensity of a pixel is a nonnegative value, and we deal with very small types where every bit can make a difference, people use unsigned types. With std::uint8_t, they can represent all the numbers in between 0 and 255 included and with unsigned short, they can represent all the numbers in between 0 and 65'535 included. In some highly specialized deep learning applications where integers are used instead of floating points, unsigned integers of 16 bits can be used.
  When 32 bits or 64 bits are needed, use the signed version. Unless you really know what you are doing, I highly suggest you not to use unsigned int because you know that your integer is positive and you think you need "an extra bit".
- Use unsigned types when you need to do modulo 2^n arithmetic
  This type of computation does not arise frequently in high performance computing. It is the right signedness to choose when you compute hash values for instance. But their usage is often really limited.

In a nutshell, my advice is the following:
- General purpose integer: ptrdiff_t
- Integers to be stored and process in large arrays: std::uint8_t, unsigned short, int, ptrdiff_t
- Integers to do modulo 2^n arithmetic: size_t

Some other choices are often made in programs and libraries.
- The most common choice is to use size_t as the general purpose integer in many C++ programs. I understand programmers who need to live in the continuity of the C++ STL who made the bad decision to choose the size_t type as a general purpose integer. Programmers who claim that they use size_t because the integers they deal with are nonnegative are wrong, period. Others are concerned about security and are scared of the undefined behaviour of overflow with signed integers. But companies such as Google took the decision to use signed integers instead of unsigned integers even in parts of the code where security matters because it was possible to design tools to detect overflows which could not be done with unsigned integers as "overflow" might be expected by the programmer with those types. In a nutshell, except for those people who decided to work in the continuity of the C++ STL, I beleive that those who use size_t as a choice are wrong.
- Another common choice is to use int as the general purpose integer. This is the choice made by many reputed software company such as Google. When 32 bits is not enough, they decide to move to a 64 bits signed integer. Although it is an very good choice for Google for many of the software they develop, in scientific applications it is not a good choice as some arrays we deal with can be extremely large and might have more than 2'147'483'647 elements. Even Google acknowledged that point and the default integer for array indices in TensorFlow is ptrdiff_t as used in the Eigen library.

As a consequence, the InsideLoop library is using ptrdiff_t as it's general purpose integer. In order to have a "better" name, we typedef it to il::int_t. All you have to know is that it is a signed integer which is 32 bits large on 32-bit platform and 64 bits large on 64-bit platforms.




Integer minimum. The C++ standard redirects to the C standard section 5.2.4.2.1 (version C11)

=== On Linux
fayard@Grisbouille:Type$ ./type-32-clang 
std::ptrdiff_t is an alias for int
std::size_t is an alias for unsigned int
fayard@Grisbouille:Type$ ./type-32-gcc
std::ptrdiff_t is an alias for int
std::size_t is an alias for unsigned int
fayard@Grisbouille:Type$ ./type-32-gcc7
std::ptrdiff_t is an alias for int
std::size_t is an alias for unsigned int
fayard@Grisbouille:Type$ ./type-32-icpc 
std::ptrdiff_t is an alias for int
std::size_t is an alias for unsigned int

fayard@Grisbouille:Type$ ./type-64-clang 
std::ptrdiff_t is an alias for long
std::size_t is an alias for unsigned long
fayard@Grisbouille:Type$ ./type-64-gcc
std::ptrdiff_t is an alias for long
std::size_t is an alias for unsigned long
fayard@Grisbouille:Type$ ./type-64-gcc7
std::ptrdiff_t is an alias for long
std::size_t is an alias for unsigned long
fayard@Grisbouille:Type$ ./type-64-icpc 
std::ptrdiff_t is an alias for long
std::size_t is an alias for unsigned long
fayard@Grisbouille:Type$ ./type-64-pgc++ 
std::ptrdiff_t is an alias for long
std::size_t is an alias for unsigned long

=== On Mac
Cheetah:Desktop fayard$ ./type-32-clang 
std::ptrdiff_t is an alias for int         <== Because of historical reasons. See Stackoverflow
std::size_t is an alias for unsigned long  <== "Strange typedef for std::ptrdiff_t and std::size_t"
Cheetah:Desktop fayard$ ./type-64-clang 
std::ptrdiff_t is an alias for long
std::size_t is an alias for unsigned long



Byte: Today, a byte is almost always 8 bit. However, that wasn't always the case and there's no "standard" or something that dictates this. Since 8 bits is a convenient number to work with it became the de facto standard.

Word: The natural size with which a processor is handling data (the register size). The most common word sizes encountered today are 8, 16, 32 and 64 bits, but other sizes are possible. For examples, there were a few 36 bit machines, or even 12 bit machines.

The byte is the smallest addressable unit for a CPU. If you want to set/clear single bits, you first need to fetch the corresponding byte from memory, mess with the bits and then write the byte back to memory.

The word by contrast is biggest chunk of bits with which a processor can do processing (like addition and subtraction) at a time. That definition is a bit fuzzy, as some processor might have different word sizes for different tasks (integer vs. floating point processing for example). The word size is what the majority of operations work with.

There are also a few processors who have a different pointer size: for example, the 8086 is a 16-bit processor which means its registers are 16 bit wide. But its pointers (addresses) are 20 bit wide and were calculated by combining two 16 bit registers in a certain way.
