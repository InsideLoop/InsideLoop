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

Then create a file :bash:`main.cpp` with the text editor of your choice and
type your first InsideLoop program which computes the first 100'000'000 values
of :math:`\cos(k)` and times this operation.

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

    fayard@Grisbouille:~$ g++ -g -std=c++11 -O0 -I~/Library main.cpp -o main
    fayard@Grisbouille:~$ ./main
    Time: 4.44s

For release mode, you can use the following commands:

.. code:: bash

    fayard@Grisbouille:~$ g++ -g -std=c++11 -O3 -march=native -DNDEBUG -I~/Library main.cpp -o main
    fayard@Grisbouille:~$ ./main
    Time: 3.24s

If you want to use clang, the same commands can be used:

.. code:: bash

    fayard@Grisbouille:~$ clang++ -g -std=c++11 -O3 -march=native -DNDEBUG -I~/Library main.cpp -o main
    fayard@Grisbouille:~$ ./main
    Time: 1.68e-7s

As you can see, clang is smart enough to realize that nothig is done with the
array. As a consequence, the loop is removed from the final code which does
nothing.

If you use the Intel compilers, one can issue the following command:

.. code:: bash

    fayard@Grisbouille:~$ icpc -g -std=c++11 -O3 -xHost -DNDEBUG -I~/Library main.cpp -o main
    fayard@Grisbouille:~$ ./main
    Time: 0.66s

This time, we get a faster code than gcc. It is obviously not as fast as clang
as the computation is really done.

Note: The Gcc compiler is know to use very slow calls when computing :cpp:`sin`
and :cpp:`cos` functions. Even though the Intel compiler is notoriously good
for number crunching, one should not expect such a difference in between
compilers for most programs.


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


