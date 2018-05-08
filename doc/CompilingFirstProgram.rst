.. role:: cpp(code)

    :language: cpp

.. role:: bash(code)

    :language: bash

Compiling your first program
============================

In this tutorial, we'll explain how to compile and run your first program
using the InsideLoop library on both Linux/macOS and Windows.

Linux
-----

On Linux, one should first download the latest version of the InsideLoop
library with the following command on your bash prompt. You are welcome to
install the InsiedLoop library wherever you want. We'll install it in a folder
called :bash:`Library` inside your home folder.

.. code:: bash

    fayard@Grisbouille:~$ mkdir ~/Library
    fayard@Grisbouille:~$ cd ~/Library
    fayard@Grisbouille:~$ git clone https://github.com/InsideLoop/InsideLoop.git

- **Compile your first program**.
  To compile our first program we need to create a file :bash:`main.cpp` with
  the text editor of your choice and type your first InsideLoop program which
  computes the first 100'000'000 values of :math:`\cos(k)` and benchmarks this
  operation.

  .. code:: cpp

      #include <cmath>
      #include <iostream>

      #include <il/Array.h>
      #include <il/Timer.h>

      int main() {
        const il::int_t n = 100000000;
        il::Array<double> v{n, 0.0};
        const double alpha = 1.0;

        il::Timer timer{};
        timer.Start();
        for (il::int_t k = 0; k < v.size(); ++k) {
          v[k] = std::cos(k * alpha);
        }
        timer.Stop();

        std::cout << "Time: " << timer.time() << "s" << std::endl;

        return 0;
      }

  To compile and run your program in debug mode, you should use

  .. code:: bash

      fayard@Grisbouille:~$ g++ -g -std=c++11 -O0 -I~/Library/InsideLoop main.cpp -o main
      fayard@Grisbouille:~$ ./main
      Time: 4.44s

  For release mode, you can use the following commands:

  .. code:: bash

      fayard@Grisbouille:~$ g++ -g -std=c++11 -O3 -march=native -DNDEBUG -I~/Library/InsideLoop main.cpp -o main
      fayard@Grisbouille:~$ ./main
      Time: 3.24s

  If you want to use clang, the same commands can be used:

  .. code:: bash

      fayard@Grisbouille:~$ clang++ -g -std=c++11 -O3 -march=native -DNDEBUG -I~/Library/InsideLoop main.cpp -o main
      fayard@Grisbouille:~$ ./main
      Time: 1.68e-7s

  As you can see, clang is smart enough to realize that nothig is done with the
  array. As a consequence, the loop is removed from the final code which does
  nothing.

  If you use the Intel compilers, one can issue the following command:

  .. code:: bash

      fayard@Grisbouille:~$ icpc -g -std=c++11 -O3 -xHost -DNDEBUG -I~/Library/InsideLoop main.cpp -o main
      fayard@Grisbouille:~$ ./main
      Time: 0.66s

  This time, we get a faster code than gcc. It is obviously not as fast as clang
  as the computation is really done.

- **Using the MKL for basic linear algebra**.
  Let's write our first program that can do a matrix multiplication and
  benchmark the performance of your computer.

  .. code:: cpp

      #include <iostream>

      #include <il/Array2D.h>
      #include <il/Timer.h>
      #include <il/math.h>
      #include <il/blas.h>

      int main() {
        const il::int_t n = 2000;
        const il::int_t nb_times = 10;
        il::Array2D<float> A{n, n, 0.0};
        il::Array2D<float> B{n, n, 0.0};

        // Warmup
        il::Array2D<float> C = il::dot(A, B);

        il::Timer timer{};
        timer.Start();
        for (il::int_t k = 0; k < nb_times; ++k) {
          il::Array2D<float> D = il::dot(A, B);
        }
        timer.Stop();

        const il::int_t nb_flops = nb_times * 2 * il::ipow<3>(n);

        std::cout << "Gflops per second: " << 1.0e-9 * nb_flops / timer.time()
                  << std::endl;

        return 0;
      }

  If you use the Intel compilers, it can be easily compiled with

  .. code:: bash

      fayard@Grisbouille:~$ icpc -std=c++11 -O3 -xHost -mkl=parallel -DNDEBUG -DIL_MKL
        -I~/Library/InsideLoop main.cpp -o main
      fayard@Grisbouille:~$ ./main
      Gflops per second: 2930.51

  For other compilers, it is better to go to the "Intel MKL Link Line Advisor".
  On my workstation, with gcc, one need to use the following command line

  .. code:: bash

      fayard@Grisbouille:~$ g++ -std=c++11 -O3 -march=native -m64 -DNDEBUG -DIL_MKL
        -I~/Library/InsideLoop -I${MKLROOT}/include -L${MKLROOT}/lib/intel64
        -Wl,--no-as-needed main.cpp -o main
        -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
      fayard@Grisbouille:~$ ./main
      Gflops per second: 2916.56

  The same command line works with Clang:

  .. code:: bash

      fayard@Grisbouille:~$ clang++ -std=c++11 -O3 -march=native -m64 -DNDEBUG -DIL_MKL
        -I~/Library/InsideLoop -I${MKLROOT}/include -L${MKLROOT}/lib/intel64
        -Wl,--no-as-needed main.cpp -o main
        -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
      fayard@Grisbouille:~$ ./main
      Gflops per second: 2915.1

- **Friendly debugging with GDB**.
  Out of the box, C++ programs can be painful to debug for 2 reasons. The first
  one is that when you debug, you want to debug your program and the underlying
  library you are using. For instance, you don't want to step into a function
  call of the library such as an array accessor. Moreover, you are not
  interested in the internals of an object :cpp:`il::Array<T>` which is composed
  of 3 pointers. You are interested to the elements that are contained in this
  array. In order to solve these problems, we provide some configuration files
  in the folder :cpp:`prettyPrint/gdb`. Copy the :cpp:`gdbinit` file to your
  home directory as :cpp:`.gdbinit` and edit the file. You should replace the
  line

  .. code:: bash

      sys.path.insert(0, '/home/fayard/Library/InsideLoop/prettyPrint/gdb')

  with the full path where is stored the file :cpp:`printers.py` that contains
  all the pretty printers.

  Once it is configured, IDE such as CLion should display your arrays as such
  in the debugger.

  .. image:: debugger.png
       :scale: 100 %
       :alt: alternate text
       :align: center

Windows
-------

On Windows 10, using Visual Studio 2017, here are the steps you should do.

- First, go to https://gtihub.com/InsideLoop/InsideLoop and download a ZIP
  file of the latest version of the library.

  .. image:: download-github-windows.png
      :scale: 100 %
      :alt: alternate text
      :align: center

- Once it has been downloaded, go to your :bash:`Downloads` folder, right-click
  on the :bash:`InsideLoop-master` package and choose :bash:`Extract All...`.
  If you leave the default folder, it should extract all the library in the
  :bash:`Downloads` folder.

- Start Visual Studio 2017, and click on the menu :bash:`File -> New -> Project`.
  Then, click on :bash:`Visual C++` and choose :bash:`Windows Console Application`.

  .. image:: vs2017-console.png
      :scale: 100 %
      :alt: alternate text
      :align: center

  And then click :bash:`OK`.

- You can then type your code in the :bash:`main.cpp` file. Please keep the
  special include :cpp:`#include "stdafx.h"` at the top of the file. It is
  specific to Windows.

- Before you can compile, you need to tell Visual Studio where the InsideLoop
  library is located. For that, click on the menu :bash:`Project -> ConsoleApplication Properties...`.
  Then go to :bash:`VC++ Directories` and edit :bash:`Include Directories`.

  .. image:: vs2017-include.png
      :scale: 100 %
      :alt: alternate text
      :align: center

  Click on the folder icon on the top right of the Window and navigate to
  :bash:`C:\Users\YourName\Downloads\InsideLoop-master\InsideLoop-master`
  and make sure that this folder contains among a few folders, a folder
  called :bash:`il`. Then, click :bash:`Select Folder`, :bash:`OK`, :bash:`OK`

  .. image:: vs2017-include-path.png
      :scale: 100 %
      :alt: alternate text
      :align: center

- Then, click on the menu :bash:`Build -> Build Solution`. After the program
  has compiled, you can start it with :bash:`Debug -> Start Debugging`.


