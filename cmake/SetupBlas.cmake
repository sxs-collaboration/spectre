# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(BLAS REQUIRED)
message(STATUS "BLAS libs: " ${BLAS_LIBRARIES})
list(APPEND SPECTRE_LIBRARIES ${BLAS_LIBRARIES})
file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "BLAS_LIBRARIES:  ${BLAS_LIBRARIES}\n"
  )
