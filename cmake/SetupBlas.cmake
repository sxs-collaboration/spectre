# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(BLAS REQUIRED)
message(STATUS "BLAS libs: " ${BLAS_LIBRARIES})
file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "BLAS_LIBRARIES:  ${BLAS_LIBRARIES}\n"
  )

add_library(Blas INTERFACE IMPORTED)
set_property(TARGET Blas
  APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${BLAS_LIBRARIES})

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  Blas
  )
