# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(SciPy REQUIRED)

message(STATUS "SciPy incl: " ${SCIPY_INCLUDE_DIRS})
message(STATUS "SciPy vers: " ${SCIPY_VERSION})

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "SciPy Version:  ${SCIPY_VERSION}\n"
  )
