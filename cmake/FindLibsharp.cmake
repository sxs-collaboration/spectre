# Distributed under the MIT License.
# See LICENSE.txt for details.

# Find libsharp: https://github.com/Libsharp/libsharp
# If not in one of the default paths specify -D LIBSHARP_ROOT=/path/to/libsharp
# to search there as well.

include (CheckCXXSourceRuns)

# find the libsharp include directory
find_path(LIBSHARP_INCLUDE_DIRS sharp_cxx.h
    PATH_SUFFIXES include
    HINTS ${LIBSHARP_ROOT}/include/)

find_library(LIBSHARP_LIBFFTPACK
    NAMES libfftpack libfftpack.a
    PATH_SUFFIXES lib64 lib
    HINTS ${LIBSHARP_ROOT})

find_library(LIBSHARP_LIBRARIES
    NAMES libsharp libsharp.a
    PATH_SUFFIXES lib64 lib
    HINTS ${LIBSHARP_ROOT})

find_library(LIBSHARP_LIBCUTILS
    NAMES libc_utils libc_utils.a
    PATH_SUFFIXES lib64 lib
    HINTS ${LIBSHARP_ROOT})


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Libsharp
    REQUIRED_VARS LIBSHARP_INCLUDE_DIRS
    LIBSHARP_LIBRARIES LIBSHARP_LIBFFTPACK LIBSHARP_LIBCUTILS)

mark_as_advanced(LIBSHARP_INCLUDE_DIRS)
mark_as_advanced(LIBSHARP_LIBRARIES)
mark_as_advanced(LIBSHARP_FFTPACK)
mark_as_advanced(LIBSHARP_LIBCUTILS)
