# Distributed under the MIT License.
# See LICENSE.txt for details.

# Find LIBXSMM: https://github.com/hfp/libxsmm
# If not in one of the default paths specify -D LIBXSMM_ROOT=/path/to/LIBXSMM
# to search there as well.

# find the LIBXSMM include directory
find_path(LIBXSMM_INCLUDE_DIRS libxsmm.h
    PATH_SUFFIXES include
    HINTS ${LIBXSMM_ROOT})

find_library(LIBXSMM_LIBRARIES
    NAMES libxsmm.a
    PATH_SUFFIXES lib64 lib
    HINTS ${LIBXSMM_ROOT})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LIBXSMM
    DEFAULT_MSG LIBXSMM_INCLUDE_DIRS LIBXSMM_LIBRARIES)
mark_as_advanced(LIBXSMM_INCLUDE_DIRS LIBXSMM_LIBRARIES)
