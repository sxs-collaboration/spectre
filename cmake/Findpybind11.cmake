# Distributed under the MIT License.
# See LICENSE.txt for details.

find_path(
  pybind11_INCLUDE_DIR
  PATH_SUFFIXES include
  NAMES pybind11/pybind11.h
  HINTS ${pybind11_ROOT}
  DOC "Pybind11 include directory. Used pybind11_ROOT to set a search dir."
  )

set(pybind11_INCLUDE_DIRS ${pybind11_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  pybind11
  DEFAULT_MSG pybind11_INCLUDE_DIR pybind11_INCLUDE_DIRS
  )
mark_as_advanced(pybind11_INCLUDE_DIR pybind11_INCLUDE_DIRS)
