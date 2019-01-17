# Distributed under the MIT License.
# See LICENSE.txt for details.

# Find tcmalloc: https://github.com/gperftools/gperftools
# If not in one of the default paths specify -D TCMALLOC_ROOT=/path/to/tcmalloc
# to search there as well.

# find the tcmalloc include directory
find_path(TCMALLOC_INCLUDE_DIRS gperftools/tcmalloc.h
    PATH_SUFFIXES include
    HINTS ${TCMALLOC_ROOT})

find_library(TCMALLOC_LIBRARIES
    NAMES tcmalloc
    PATH_SUFFIXES lib64 lib
    HINTS ${TCMALLOC_ROOT})

# Extract version info from header
file(READ
  "${TCMALLOC_INCLUDE_DIRS}/gperftools/tcmalloc.h"
  TCMALLOC_FIND_HEADER_CONTENTS)

string(REGEX MATCH "#define TC_VERSION_MAJOR [0-9]+"
  TCMALLOC_MAJOR_VERSION "${TCMALLOC_FIND_HEADER_CONTENTS}")
string(REPLACE "#define TC_VERSION_MAJOR " ""
  TCMALLOC_MAJOR_VERSION
  "${TCMALLOC_MAJOR_VERSION}")

string(REGEX MATCH "#define TC_VERSION_MINOR [0-9]+"
  TCMALLOC_MINOR_VERSION "${TCMALLOC_FIND_HEADER_CONTENTS}")
string(REPLACE "#define TC_VERSION_MINOR " ""
  TCMALLOC_MINOR_VERSION
  "${TCMALLOC_MINOR_VERSION}")


set(TCMALLOC_VERSION
  "${TCMALLOC_MAJOR_VERSION}.${TCMALLOC_MINOR_VERSION}"
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  TCMALLOC
  FOUND_VAR TCMALLOC_FOUND
  REQUIRED_VARS TCMALLOC_INCLUDE_DIRS TCMALLOC_LIBRARIES
  VERSION_VAR TCMALLOC_VERSION
  )
mark_as_advanced(TCMALLOC_INCLUDE_DIRS TCMALLOC_LIBRARIES
  TCMALLOC_MAJOR_VERSION TCMALLOC_MINOR_VERSION
  TCMALLOC_VERSION)
