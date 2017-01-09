#===============================================================================
#
#                                   InsideLoop
#
#  This file is distributed under the University of Illinois Open Source
#  License. See LICENSE.txt for details.
#
#===============================================================================

# To compile with icpc with full optimizations
# cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpc
#
# One can use -march=native with gcc and clang

cmake_minimum_required(VERSION 2.8)

set(PACKAGE_VERSION_MAJOR "0")
set(PACKAGE_VERSION_MINOR "0")

project(InsideLoop CXX)
project(InsideLoopUnitTest CXX)
project(Program CXX)

set(IL_OPENMP 0)
set(IL_MKL 1)
set(IL_PNG 1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11")

# We want all possible warnings and treat them as errors
if (UNIX)
    if (UNIX)
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -pedantic")
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -pedantic")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -pedantic")
        endif()
    endif()
endif()

################################################################################
# Choose release options
################################################################################

if (IL_OPENMP)
    if (UNIX)
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -xHost -ansi-alias -DNDEBUG")
        endif()
    endif()
endif()

################################################################################
# Choose debug options
################################################################################

# For the sanitizers
if (ADDRESS_SANITIZER)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    endif()
endif()

################################################################################
# Choose parallel framework
################################################################################

# For OpenMP
if (IL_OPENMP)
    if (UNIX)
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
#            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#            link_directories(/opt/clang-3.9.0/lib)
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
        endif()
    endif()
endif()

# For TBB
if (IL_TBB)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIL_TBB")
#    target_link_libraries(InsideLoop tbb)
endif()

# For Cilk
if (IL_CILK)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIL_CILK")
endif()

################################################################################
# Choose math framework
################################################################################

# For Intel MKL
if (IL_MKL)
    if (UNIX)
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -DIL_MKL")
            set(CMAKE_MKL_LIBRARIES mkl_intel_lp64 mkl_core mkl_intel_thread iomp5 dl pthread m)
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -DIL_MKL")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
            set(CMAKE_MKL_LIBRARIES mkl_intel_lp64 mkl_core mkl_intel_thread iomp5 dl pthread m)
            link_directories($ENV{MKLROOT}/lib/intel64_lin)
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl=parallel -DIL_MKL")
        endif()
    endif()
elseif (IL_OPENBLAS)
    if (APPLE)
        set(CMAKE_OPENBLAS_LIBRARIES "openblas")
    endif()
endif()

################################################################################
# Choose graphic framework
################################################################################

# For PNG
if (IL_PNG)
    set(CMAKE_PNG_LIBRARIES "png")
endif()

################################################################################

set(SOURCE_FILES
    il/Array.h
    il/Array2D.h
    il/Array2C.h
    il/Array3D.h
    il/Array4D.h
    il/Array4C.h
    il/ArrayView.h
    il/BandArray2C.h
    il/base.h
    il/CudaArray2D.h
    il/format.h
    il/HashMap.h
    il/linear_algebra.h
    il/LowerArray2D.h
    il/math.h
    il/SmallArray.h
    il/SparseMatrixCSR.h
    il/StaticArray.h
    il/StaticArray2D.h
    il/StaticArray2C.h
    il/StaticArray3D.h
    il/String.h
    il/Timer.h
    il/TriDiagonal.h
    il/UpperArray2D.h
    il/container/1d/Array.h
    il/container/1d/ArrayView.h
    il/container/1d/SmallArray.h
    il/container/1d/StaticArray.h
    il/container/2d/Array2D.h
    il/container/2d/Array2C.h
    il/container/2d/BandArray2C.h
    il/container/2d/Array2DView.h
    il/container/2d/LowerArray2D.h
    il/container/2d/SparseMatrixCSR.h
    il/container/2d/StaticArray2D.h
    il/container/2d/StaticArray2C.h
    il/container/2d/TriDiagonal.h
    il/container/2d/UpperArray2D.h
    il/container/3d/Array3D.h
    il/container/3d/Array3DView.h
    il/container/3d/StaticArray3D.h
    il/container/4d/Array4D.h
    il/container/4d/Array4C.h
    il/container/cuda/1d/CudaArray.h
    il/container/cuda/2d/CudaArray2D.h
    il/container/cuda/2d/CudaSparseMatrixCSR.h
    il/container/cuda/cuda_copy.h
    il/container/hash/HashFunction.h
    il/container/hash/HashMap.h
    il/container/string/String.h
    il/core/base.h
    il/core/Status.h
    il/io/ppm.h
    il/io/numpy.h
    il/io/numpy.cpp
    il/io/png.h
    il/io/png.cpp
    il/io/format/format.cc
    il/linear_algebra/dense/blas/blas.h
    il/linear_algebra/dense/blas/dot.h
    il/linear_algebra/dense/factorization/linear_solve.h
    il/linear_algebra/dense/blas/norm.h
    il/linear_algebra/dense/blas/blas.h
    il/linear_algebra/dense/blas/dot.h
    il/linear_algebra/dense/blas/norm.h
    il/linear_algebra/dense/factorization/LU.h
    il/linear_algebra/dense/factorization/Cholesky.h
    il/linear_algebra/dense/factorization/Householder.h
    il/linear_algebra/dense/factorization/Eigen.h
    il/linear_algebra/dense/factorization/Singular.h
    il/linear_algebra/sparse/blas/_code/conjugate_gradient_blaze.h
    il/linear_algebra/sparse/blas/sparse_blas.h
    il/linear_algebra/sparse/blas/sparse_dot.h
    il/linear_algebra/sparse/blas/sparse_linear_algebra.h
    il/linear_algebra/sparse/blas/SparseMatrixBlas.h
    il/linear_algebra/cuda/dense/blas/cuda_blas.h
    il/unit/time.h
    il/random/sobol.h)

################################################################################
# A test program
################################################################################


add_executable(InsideLoop ${SOURCE_FILES} main.cpp)
target_link_libraries(InsideLoop ${CMAKE_MKL_LIBRARIES} ${CMAKE_PNG_LIBRARIES})
target_include_directories(InsideLoop PRIVATE ${CMAKE_SOURCE_DIR} /opt/llvm-3.9.0/include /opt/google-hash/include /opt/blaze-3.0 /opt/cuda-8.0/include)

################################################################################
# For Unit Tests
################################################################################

set(UNIT_TEST_FILES
    il/container/1d/_test/Array_test.cpp
    il/container/1d/_test/Dummy_test.h
    il/container/1d/_test/Dummy_test.cpp
    il/container/2d/_test/Array2D_test.cpp
    il/container/2d/_test/Array2C_test.cpp
    il/container/3d/_test/Array3D_test.cpp
    il/linear_algebra/dense/blas/_test/blas_test.cpp
    il/linear_algebra/dense/blas/_test/linear_solve_test.cpp
    il/linear_algebra/dense/blas/_test/dot_test.cpp
    il/linear_algebra/dense/factorization/_test/Eigen_test.cpp
    il/linear_algebra/sparse/factorization/_test/Pardiso_test.cpp
    il/linear_algebra/sparse/factorization/_test/GmresIlu0_test.cpp
    gtest/src/gtest-all.cc)

add_executable(InsideLoopUnitTest ${SOURCE_FILES} ${UNIT_TEST_FILES} test.cpp)

target_include_directories(InsideLoopUnitTest PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/gtest)
target_link_libraries(InsideLoopUnitTest ${CMAKE_MKL_LIBRARIES} ${CMAKE_PNG_LIBRARIES})

# For unit tests: The precondition of our fonctions are checked with assert
# macros that terminate the program in debug mode. In order to test those macros
# in our unit tests, the flag IL_UNIT_TEST turns them into exceptions that can
# be caught and checked.
target_compile_options(InsideLoopUnitTest PRIVATE "-DIL_UNIT_TEST")
