# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(LAPACK REQUIRED)
message(STATUS "LAPACK libs: " ${LAPACK_LIBRARIES})
list(APPEND SPECTRE_LIBRARIES ${LAPACK_LIBRARIES})
file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "LAPACK_LIBRARIES:  ${LAPACK_LIBRARIES}\n"
  )
