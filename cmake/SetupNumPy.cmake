# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(NumPy 1.10 REQUIRED)

message(STATUS "NumPy incl: " ${NUMPY_INCLUDE_DIRS})
message(STATUS "NumPy vers: " ${NUMPY_VERSION})

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "NumPy Version:  ${NUMPY_VERSION}\n"
  )

add_library(NumPy INTERFACE IMPORTED)
set_property(TARGET NumPy PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES ${NUMPY_INCLUDE_DIR})

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  NumPy
  )
