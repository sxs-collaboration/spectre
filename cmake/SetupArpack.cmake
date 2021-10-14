# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Arpack REQUIRED)

add_library(Arpack INTERFACE IMPORTED)

message(STATUS "Arpack libs: " ${ARPACK_LIBRARIES})

file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
  "ARPACK_LIBRARIES: ${ARPACK_LIBRARIES}\n"
  )

set_property(TARGET Arpack
  APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${ARPACK_LIBRARIES})

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  Arpack
  )
