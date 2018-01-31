#===============================================================================
#
# Copyright 2017 The InsideLoop Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#===============================================================================
#
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
set(IL_BLAS 1)
set(IL_MKL 1)
set(IL_OPENBLAS 0)
set(IL_PNG 0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17")

# We want all possible warnings and treat them as errors
if (UNIX)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
#        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -pedantic")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -pedantic")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -pedantic")
    endif()
elseif (WIN32)
    if (MINGW)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -pedantic")
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
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined -fno-omit-frame-pointer")
#        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory -fno-omit-frame-pointer")
#        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -fno-omit-frame-pointer")
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
    elseif (WIN32)
        if (MINGW)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
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
        if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -DIL_MKL -DIL_BLAS")
            set(CMAKE_MKL_LIBRARIES mkl_intel_lp64 mkl_core mkl_intel_thread iomp5 dl pthread m)
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -DIL_MKL -DIL_BLAS")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
            set(CMAKE_MKL_LIBRARIES mkl_intel_lp64 mkl_core mkl_intel_thread iomp5 dl pthread m)
            link_directories($ENV{MKLROOT}/lib/intel64_lin)
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl=parallel -DIL_MKL -DIL_BLAS")
        endif()
    endif()
elseif (IL_OPENBLAS)
    if (UNIX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIL_OPENBLAS -DIL_BLAS")
        set(CMAKE_OPENBLAS_LIBRARIES "openblas")
    elseif(WIN32)
        if (MINGW)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIL_OPENBLAS -DIL_BLAS")
            set(CMAKE_OPENBLAS_LIBRARIES "openblas")
        endif()
    elseif (APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIL_MKL -DIL_OPENBLAS -DIL_BLAS")
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
    il/algorithmArray.h
    il/algorithmArray2D.h
    il/Array.h
    il/Array2D.h
    il/Array2C.h
    il/Array2DView.h
    il/Array3C.h
    il/Array3D.h
    il/Array4D.h
    il/Array4C.h
    il/ArrayView.h
    il/BandArray2C.h
    il/core.h
    il/CudaArray2D.h
    il/data.h
    il/Dynamic.h
    il/format.h
    il/Map.h
    il/Info.h
    il/linear_algebra.h
    il/LowerArray2D.h
    il/math.h
    il/norm.h
    il/numpy.h
    il/png.h
    il/Queue.h
    il/Set.h
    il/SmallArray.h
    il/SparseMatrixCSR.h
    il/StaticArray.h
    il/StaticArray2D.h
    il/StaticArray2C.h
    il/StaticArray3D.h
    il/StaticArray4D.h
    il/Status.h
    il/String.h
    il/string_util.h
    il/Timer.h
    il/Type.h
    il/toml.h
    il/TriDiagonal.h
    il/UTF16String.h
    il/unicode.h
    il/UpperArray2D.h
    il/algorithm/algorithmArray.h
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
    il/container/3d/Array3C.h
    il/container/3d/Array3D.h
    il/container/3d/Array3DView.h
    il/container/3d/StaticArray3D.h
    il/container/4d/Array4D.h
    il/container/4d/Array4C.h
    il/container/4d/StaticArray4D.h
    il/container/cuda/1d/CudaArray.h
    il/container/cuda/2d/CudaArray2D.h
    il/container/cuda/2d/CudaSparseMatrixCSR.h
    il/container/cuda/cuda_copy.h
    il/container/hash/HashFunction.h
    il/container/hash/Map.h
    il/container/hash/MapArray.h
    il/container/hash/Set.h
    il/container/hash/Spot.h
    il/container/string/String.h
    il/container/string/UTF16String.h
    il/container/string/unicode.h
    il/container/string/view_string.h
    il/container/string/algorithm_string.h
    il/container/dynamic/Type.h
    il/container/dynamic/Dynamic.h
    il/container/info/Info.h
    il/container/info/Status.h
    il/container/queue/Queue.h
    il/core/core.h
    il/core/math/safe_arithmetic.h
    il/core/memory/allocate.h
    il/io/io_base.h
    il/io/ppm/ppm.h
    il/io/numpy/numpy.h
    il/io/numpy/numpy.cpp
    il/io/toml/toml.h
    il/io/toml/toml.cpp
    il/io/png/png.h
#    il/io/png/png.cpp
    il/io/filepack/filepack.h
    il/io/format/format.cc
    il/linear_algebra/dense/blas/blas.h
    il/linear_algebra/dense/blas/dot.h
    il/linear_algebra/dense/norm.h
    il/linear_algebra/dense/factorization/linearSolve.h
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
#    il/linear_algebra/matrixFree/solver/Gmres.cpp
    il/unit/time.h
    il/random/sobol.h)

################################################################################
# A test program
################################################################################


add_executable(InsideLoop ${SOURCE_FILES} main.cpp)
target_link_libraries(InsideLoop ${CMAKE_MKL_LIBRARIES} ${CMAKE_PNG_LIBRARIES} ${CMAKE_OPENBLAS_LIBRARIES})
target_include_directories(InsideLoop PRIVATE ${CMAKE_SOURCE_DIR} /opt/eigen-3.3.2 /opt/blaze-3.0 /opt/cuda-8.0/include)

if (APPLE)
    if (IL_MKL)
        add_custom_command(TARGET InsideLoop POST_BUILD COMMAND /usr/bin/install_name_tool -change @rpath/libmkl_intel_lp64.dylib $ENV{MKLROOT}/lib/libmkl_intel_lp64.dylib $<TARGET_FILE:InsideLoop>)
        add_custom_command(TARGET InsideLoop POST_BUILD COMMAND /usr/bin/install_name_tool -change @rpath/libmkl_core.dylib $ENV{MKLROOT}/lib/libmkl_core.dylib $<TARGET_FILE:InsideLoop>)
        add_custom_command(TARGET InsideLoop POST_BUILD COMMAND /usr/bin/install_name_tool -change @rpath/libmkl_intel_thread.dylib $ENV{MKLROOT}/lib/libmkl_intel_thread.dylib $<TARGET_FILE:InsideLoop>)
        add_custom_command(TARGET InsideLoop POST_BUILD COMMAND /usr/bin/install_name_tool -change @rpath/libiomp5.dylib $ENV{MKLROOT}/../compiler/lib/libiomp5.dylib $<TARGET_FILE:InsideLoop>)
    endif()
endif()

################################################################################
# For Unit Tests
################################################################################

set(IL_FOLDER "${CMAKE_SOURCE_DIR}/il")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIL_FOLDER='\"${IL_FOLDER}\"'")

set(UNIT_TEST_FILES
    il/container/1d/_test/Array_test.cpp
    il/container/1d/_test/Dummy_test.h
    il/container/1d/_test/Dummy_test.cpp
    il/container/2d/_test/Array2D_test.cpp
    il/container/2d/_test/Array2C_test.cpp
    il/container/3d/_test/Array3D_test.cpp
    il/container/string/_test/String_test.cpp
    il/container/dynamic/_test/Dynamic_test.cpp
    il/container/info/_test/Info_test.cpp
    il/core/math/_test/safe_arithmetic_test.cpp
    il/linear_algebra/dense/_test/norm_test.cpp
    il/linear_algebra/dense/blas/_test/blas_test.cpp
    il/linear_algebra/dense/blas/_test/linear_solve_test.cpp
    il/linear_algebra/dense/blas/_test/dot_test.cpp
    il/linear_algebra/dense/blas/_test/cross_test.cpp
    il/linear_algebra/dense/factorization/_test/Eigen_test.cpp
    il/linear_algebra/dense/factorization/_test/Singular_test.cpp
    il/linear_algebra/sparse/factorization/_test/Pardiso_test.cpp
    il/linear_algebra/sparse/factorization/_test/GmresIlu0_test.cpp
    il/io/_test/numpy_test.cpp
    il/io/toml/_test/toml_valid_test.cpp
    gtest/src/gtest-all.cc
    il/core/_test/Status_test.cpp
    il/MapArray.h
    il/container/string/_test/StringView_test.cpp)

add_executable(InsideLoopUnitTest ${SOURCE_FILES} ${UNIT_TEST_FILES} test.cpp)

target_include_directories(InsideLoopUnitTest PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/gtest)
target_link_libraries(InsideLoopUnitTest ${CMAKE_MKL_LIBRARIES} ${CMAKE_PNG_LIBRARIES} ${CMAKE_OPENBLAS_LIBRARIES})

# For unit tests: The precondition of our fonctions are checked with assert
# macros that terminate the program in debug mode. In order to test those macros
# in our unit tests, the flag IL_UNIT_TEST turns them into exceptions that can
# be caught and checked.
target_compile_options(InsideLoopUnitTest PRIVATE "-DIL_UNIT_TEST")

if (APPLE)
    if (IL_MKL)
        add_custom_command(TARGET InsideLoopUnitTest POST_BUILD COMMAND /usr/bin/install_name_tool -change @rpath/libmkl_intel_lp64.dylib $ENV{MKLROOT}/lib/libmkl_intel_lp64.dylib $<TARGET_FILE:InsideLoopUnitTest>)
        add_custom_command(TARGET InsideLoopUnitTest POST_BUILD COMMAND /usr/bin/install_name_tool -change @rpath/libmkl_core.dylib $ENV{MKLROOT}/lib/libmkl_core.dylib $<TARGET_FILE:InsideLoopUnitTest>)
        add_custom_command(TARGET InsideLoopUnitTest POST_BUILD COMMAND /usr/bin/install_name_tool -change @rpath/libmkl_intel_thread.dylib $ENV{MKLROOT}/lib/libmkl_intel_thread.dylib $<TARGET_FILE:InsideLoopUnitTest>)
        add_custom_command(TARGET InsideLoopUnitTest POST_BUILD COMMAND /usr/bin/install_name_tool -change @rpath/libiomp5.dylib $ENV{MKLROOT}/../compiler/lib/libiomp5.dylib $<TARGET_FILE:InsideLoopUnitTest>)
    endif()
endif()

################################################################################
# For Benchmarks
################################################################################

foreach(p
        CMP0054 # CMake 3.1
        CMP0056 # export EXE_LINKER_FLAGS to try_run
        )
    if(POLICY ${p})
        cmake_policy(SET ${p} NEW)
    endif()
endforeach()

option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." ON)
option(BENCHMARK_ENABLE_EXCEPTIONS "Enable the use of exceptions in the benchmark library." ON)
option(BENCHMARK_ENABLE_LTO "Enable link time optimisation of the benchmark library." OFF)
option(BENCHMARK_USE_LIBCXX "Build and test using libc++ as the standard library." OFF)
option(BENCHMARK_BUILD_32_BITS "Build a 32 bit version of the library" OFF)

# Make sure we can import out CMake functions
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Read the git tags to determine the project version
include(GetGitVersion)
get_git_version(GIT_VERSION)

# Tell the user what versions we are using
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" VERSION ${GIT_VERSION})
message("-- Version: ${VERSION}")

# The version of the libraries
set(GENERIC_LIB_VERSION ${VERSION})
string(SUBSTRING ${VERSION} 0 1 GENERIC_LIB_SOVERSION)

# Import our CMake modules
include(CheckCXXCompilerFlag)
include(AddCXXCompilerFlag)
include(CXXFeatureCheck)

# Try and enable C++11. Don't use C++14 because it doesn't work in some
# configurations.
add_cxx_compiler_flag(-std=c++11)
if (NOT HAVE_CXX_FLAG_STD_CXX11)
    add_cxx_compiler_flag(-std=c++0x)
endif()

# Turn compiler warnings up to 11
#add_cxx_compiler_flag(-Wall)

#add_cxx_compiler_flag(-Wextra)
#add_cxx_compiler_flag(-Wshadow)
#add_cxx_compiler_flag(-Werror RELEASE)
#add_cxx_compiler_flag(-Werror RELWITHDEBINFO)
#add_cxx_compiler_flag(-Werror MINSIZEREL)
#add_cxx_compiler_flag(-pedantic)
#add_cxx_compiler_flag(-pedantic-errors)
#add_cxx_compiler_flag(-Wshorten-64-to-32)
#add_cxx_compiler_flag(-Wfloat-equal)
#add_cxx_compiler_flag(-fstrict-aliasing)
#if (NOT BENCHMARK_ENABLE_EXCEPTIONS)
#    add_cxx_compiler_flag(-fno-exceptions)
#endif()
#if (NOT BENCHMARK_USE_LIBCXX)
#    add_cxx_compiler_flag(-Wzero-as-null-pointer-constant)
#endif()
#if (HAVE_CXX_FLAG_FSTRICT_ALIASING)
#    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "Intel") #ICC17u2: Many false positives for Wstrict-aliasing
#        add_cxx_compiler_flag(-Wstrict-aliasing)
#    endif()
#endif()
# ICC17u2: overloaded virtual function "benchmark::Fixture::SetUp" is only partially overridden
# (because of deprecated overload)
add_cxx_compiler_flag(-wd654)
add_cxx_compiler_flag(-Wthread-safety)
if (HAVE_CXX_FLAG_WTHREAD_SAFETY)
    cxx_feature_check(THREAD_SAFETY_ATTRIBUTES)
endif()

# On most UNIX like platforms g++ and clang++ define _GNU_SOURCE as a
# predefined macro, which turns on all of the wonderful libc extensions.
# However g++ doesn't do this in Cygwin so we have to define it ourselfs
# since we depend on GNU/POSIX/BSD extensions.
if (CYGWIN)
    add_definitions(-D_GNU_SOURCE=1)
endif()

# Link time optimisation
#if (BENCHMARK_ENABLE_LTO)
#    add_cxx_compiler_flag(-flto)
#    if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
#        find_program(GCC_AR gcc-ar)
#        if (GCC_AR)
#            set(CMAKE_AR ${GCC_AR})
#        endif()
#        find_program(GCC_RANLIB gcc-ranlib)
#        if (GCC_RANLIB)
#            set(CMAKE_RANLIB ${GCC_RANLIB})
#        endif()
#    endif()
#endif()

# Coverage build type
set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG}" CACHE STRING
        "Flags used by the C++ compiler during coverage builds."
        FORCE)
set(CMAKE_EXE_LINKER_FLAGS_COVERAGE
        "${CMAKE_EXE_LINKER_FLAGS_DEBUG}" CACHE STRING
        "Flags used for linking binaries during coverage builds."
        FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE
        "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}" CACHE STRING
        "Flags used by the shared libraries linker during coverage builds."
        FORCE)
mark_as_advanced(
        CMAKE_CXX_FLAGS_COVERAGE
        CMAKE_EXE_LINKER_FLAGS_COVERAGE
        CMAKE_SHARED_LINKER_FLAGS_COVERAGE)
set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel Coverage."
        FORCE)
add_cxx_compiler_flag(--coverage COVERAGE)

if (BENCHMARK_USE_LIBCXX)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_cxx_compiler_flag(-stdlib=libc++)
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    add_cxx_compiler_flag(-nostdinc++)
    message("libc++ header path must be manually specified using CMAKE_CXX_FLAGS")
    # Adding -nodefaultlibs directly to CMAKE_<TYPE>_LINKER_FLAGS will break
    # configuration checks such as 'find_package(Threads)'
    list(APPEND BENCHMARK_CXX_LINKER_FLAGS -nodefaultlibs)
    # -lc++ cannot be added directly to CMAKE_<TYPE>_LINKER_FLAGS because
    # linker flags appear before all linker inputs and -lc++ must appear after.
    list(APPEND BENCHMARK_CXX_LIBRARIES c++)
  else()
    message(FATAL "-DBENCHMARK_USE_LIBCXX:BOOL=ON is not supported for compiler")
  endif()
endif(BENCHMARK_USE_LIBCXX)

# C++ feature checks
# Determine the correct regular expression engine to use
cxx_feature_check(STD_REGEX)
cxx_feature_check(GNU_POSIX_REGEX)
cxx_feature_check(POSIX_REGEX)
if(NOT HAVE_STD_REGEX AND NOT HAVE_GNU_POSIX_REGEX AND NOT HAVE_POSIX_REGEX)
  message(FATAL_ERROR "Failed to determine the source files for the regular expression backend")
endif()
if (NOT BENCHMARK_ENABLE_EXCEPTIONS AND HAVE_STD_REGEX AND NOT HAVE_GNU_POSIX_REGEX AND NOT HAVE_POSIX_REGEX)
  message(WARNING "Using std::regex with exceptions disabled is not fully supported")
endif()
cxx_feature_check(STEADY_CLOCK)
# Ensure we have pthreads
find_package(Threads REQUIRED)

set(BENCHMARK_FILES
    gbenchmark/src/benchmark.cc
    gbenchmark/src/benchmark_register.cc
    gbenchmark/src/colorprint.cc
    gbenchmark/src/commandlineflags.cc
    gbenchmark/src/complexity.cc
    gbenchmark/src/console_reporter.cc
    gbenchmark/src/counter.cc
    gbenchmark/src/csv_reporter.cc
    gbenchmark/src/json_reporter.cc
    gbenchmark/src/reporter.cc
    gbenchmark/src/sleep.cc
    gbenchmark/src/string_util.cc
    gbenchmark/src/sysinfo.cc
    gbenchmark/src/timers.cc
    il/container/hash/_benchmark/Map_il_vs_std_benchmark.h
    il/container/string/_benchmark/String_benchmark.h
    il/container/string/_benchmark/String_il_vs_std_benchmark.h
        il/linear_algebra/matrixFree/solver/Gmres.cpp il/linear_algebra/matrixFree/solver/Gmres.h il/File.h)

# C++ feature checks
# Determine the correct regular expression engine to use
cxx_feature_check(STD_REGEX)
cxx_feature_check(GNU_POSIX_REGEX)
cxx_feature_check(POSIX_REGEX)
if(NOT HAVE_STD_REGEX AND NOT HAVE_GNU_POSIX_REGEX AND NOT HAVE_POSIX_REGEX)
    message(FATAL_ERROR "Failed to determine the source files for the regular expression backend")
endif()
if (NOT BENCHMARK_ENABLE_EXCEPTIONS AND HAVE_STD_REGEX
        AND NOT HAVE_GNU_POSIX_REGEX AND NOT HAVE_POSIX_REGEX)
    message(WARNING "Using std::regex with exceptions disabled is not fully supported")
endif()
cxx_feature_check(STEADY_CLOCK)
# Ensure we have pthreads
find_package(Threads REQUIRED)

add_executable(InsideLoopBenchmark ${SOURCE_FILES} ${BENCHMARK_FILES} benchmark.cpp)

target_include_directories(InsideLoopBenchmark PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/gbenchmark/include)
target_link_libraries(InsideLoopBenchmark ${CMAKE_MKL_LIBRARIES} ${CMAKE_PNG_LIBRARIES} ${CMAKE_OPENBLAS_LIBRARIES} "pthread")

if (APPLE)
    if (IL_MKL)
        add_custom_command(TARGET InsideLoopBenchmark POST_BUILD COMMAND /usr/bin/install_name_tool -change @rpath/libmkl_intel_lp64.dylib $ENV{MKLROOT}/lib/libmkl_intel_lp64.dylib $<TARGET_FILE:InsideLoopBenchmark>)
        add_custom_command(TARGET InsideLoopBenchmark POST_BUILD COMMAND /usr/bin/install_name_tool -change @rpath/libmkl_core.dylib $ENV{MKLROOT}/lib/libmkl_core.dylib $<TARGET_FILE:InsideLoopBenchmark>)
        add_custom_command(TARGET InsideLoopBenchmark POST_BUILD COMMAND /usr/bin/install_name_tool -change @rpath/libmkl_intel_thread.dylib $ENV{MKLROOT}/lib/libmkl_intel_thread.dylib $<TARGET_FILE:InsideLoopBenchmark>)
        add_custom_command(TARGET InsideLoopBenchmark POST_BUILD COMMAND /usr/bin/install_name_tool -change @rpath/libiomp5.dylib $ENV{MKLROOT}/../compiler/lib/libiomp5.dylib $<TARGET_FILE:InsideLoopBenchmark>)
    endif()
endif()
