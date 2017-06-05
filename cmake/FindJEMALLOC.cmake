# Distributed under the MIT License.
# See LICENSE.txt for details.

# Find jemalloc: https://github.com/jemalloc/jemalloc
# If not in one of the default paths specify -D JEMALLOC_ROOT=/path/to/jemalloc
# to search there as well.

# find the jemalloc include directory
find_path(JEMALLOC_INCLUDE_DIRS jemalloc/jemalloc.h
    PATH_SUFFIXES include
    HINTS ${JEMALLOC_ROOT})

find_library(JEMALLOC_LIBRARIES
    NAMES jemalloc
    PATH_SUFFIXES lib64 lib
    HINTS ${JEMALLOC_ROOT})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(JEMALLOC
    DEFAULT_MSG JEMALLOC_INCLUDE_DIRS JEMALLOC_LIBRARIES)
mark_as_advanced(JEMALLOC_INCLUDE_DIRS JEMALLOC_LIBRARIES)
