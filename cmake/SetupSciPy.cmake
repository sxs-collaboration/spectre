# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(SciPy REQUIRED)

message(STATUS "SciPy incl: " ${SCIPY_INCLUDE_DIRS})
message(STATUS "SciPy vers: " ${SCIPY_VERSION})

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "SciPy Version:  ${SCIPY_VERSION}\n"
  )

add_library(SciPy INTERFACE IMPORTED)
set_property(TARGET SciPy PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES ${SCIPY_INCLUDE_DIRS})

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  SciPy
  )
