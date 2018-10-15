# Distributed under the MIT License.
# See LICENSE.txt for details.

# Find ssht: https://github.com/astro-informatics/ssht
# If not in one of the default paths specify -D SSHT_ROOT=/path/to/ssht to
# search there as well.

include (CheckCXXSourceRuns)

# find the ssht include directory
find_path(SSHT_INCLUDE_DIRS ssht.h
    PATH_SUFFIXES include
    HINTS ${SSHT_ROOT}/include/)

find_library(SSHT_LIBRARIES
    NAMES ssht libssht.a
    PATH_SUFFIXES lib64 lib
    HINTS ${SSHT_ROOT})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SSHT
    DEFAULT_MSG SSHT_INCLUDE_DIRS SSHT_LIBRARIES)
mark_as_advanced(SSHT_INCLUDE_DIRS SSHT_LIBRARIES)
