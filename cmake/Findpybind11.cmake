# Distributed under the MIT License.
# See LICENSE.txt for details.

find_path(
    pybind11_INCLUDE_DIRS pybind11/pybind11.h
    PATH_SUFFIXES include
    HINTS ${pybind11_ROOT} ENV pybind11_ROOT
    DOC "Pybind11 include directory. Use pybind11_ROOT to set a search dir."
)

set(pybind11_INCLUDE_DIR ${pybind11_INCLUDE_DIRS})

# Extract version info from header
file(READ
  "${pybind11_INCLUDE_DIR}/pybind11/detail/common.h"
  pybind11_FIND_HEADER_CONTENTS)

string(REGEX MATCH "#define PYBIND11_VERSION_MAJOR [0-9]+"
  pybind11_MAJOR_VERSION "${pybind11_FIND_HEADER_CONTENTS}")
string(REPLACE "#define PYBIND11_VERSION_MAJOR " ""
  pybind11_MAJOR_VERSION
  "${pybind11_MAJOR_VERSION}")

string(REGEX MATCH "#define PYBIND11_VERSION_MINOR [0-9]+"
  pybind11_MINOR_VERSION "${pybind11_FIND_HEADER_CONTENTS}")
string(REPLACE "#define PYBIND11_VERSION_MINOR " ""
  pybind11_MINOR_VERSION
  "${pybind11_MINOR_VERSION}")

string(REGEX MATCH "#define PYBIND11_VERSION_PATCH [0-9a-zA-Z]+"
  pybind11_SUBMINOR_VERSION "${pybind11_FIND_HEADER_CONTENTS}")
string(REPLACE "#define PYBIND11_VERSION_PATCH " ""
  pybind11_SUBMINOR_VERSION
  "${pybind11_SUBMINOR_VERSION}")

set(pybind11_VERSION
  "${pybind11_MAJOR_VERSION}.${pybind11_MINOR_VERSION}.${pybind11_SUBMINOR_VERSION}"
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  pybind11
  FOUND_VAR pybind11_FOUND
  REQUIRED_VARS pybind11_INCLUDE_DIR pybind11_INCLUDE_DIRS
  VERSION_VAR pybind11_VERSION
  )
mark_as_advanced(pybind11_INCLUDE_DIR pybind11_INCLUDE_DIRS
  pybind11_MAJOR_VERSION pybind11_MINOR_VERSION pybind11_SUBMINOR_VERSION
  pybind11_VERSION)
