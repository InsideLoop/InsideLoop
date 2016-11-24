InsideLoop is a C++ library for high performance scientific applications. In
order to get the most of current hardware, data locality is of uttermost
importance. Unfortunately, the Standard Template Library (STL) has been
designed for generic programming and hides its memory layout to the programmer.
InsideLoop stays away from generic programming and template meta-programming
techniques. It should appeal to Fortran and C programmers who want to benefit
from C++ features but don't want a complex library to get in their way to
high performance computing.

InsideLoop provides arrays, multi-dimensional arrays and a hash table. It has
been designed to provide you with:

- Containers for modern parallel architectures
- C++ wrappers for the Math Kernel Library (MKL) from Intel
- Portable performance across compilers
- Smooth integration into your projects
- Efficient debugging mode

////////////////////////////////////////////////////////////////////////////////
Installing debugging visualizer for Visual Studio:

To install the debug visualizer for Visual Studio, copy the file
visualizer/visual_studio/il.natvis in one of the following directory, depending
wether or not you want the installation to be available to all users.
%VSINSTALLDIR%\Common7\Packages\Debugger\Visualizers
%USERPROFILE%\My Documents\Visual Studio 2015\Visualizers
