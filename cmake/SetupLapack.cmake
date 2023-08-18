# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(LAPACK REQUIRED)
message(STATUS "LAPACK libs: " ${LAPACK_LIBRARIES})

file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
  "LAPACK_LIBRARIES: ${LAPACK_LIBRARIES}\n"
  )

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  LAPACK::LAPACK
  )
