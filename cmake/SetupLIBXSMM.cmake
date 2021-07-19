# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(LIBXSMM 1.16.1 REQUIRED)

message(STATUS "LIBXSMM libs: " ${LIBXSMM_LIBRARIES})
message(STATUS "LIBXSMM incl: " ${LIBXSMM_INCLUDE_DIRS})
message(STATUS "LIBXSMM vers: " ${LIBXSMM_VERSION})

file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
  "LIBXSMM version: ${LIBXSMM_VERSION}\n"
  )

add_library(Libxsmm INTERFACE IMPORTED)
set_property(TARGET Libxsmm
  APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${LIBXSMM_INCLUDE_DIRS})
set_property(TARGET Libxsmm
  APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${LIBXSMM_LIBRARIES})
# LIBXSMM falls back to blas, so we need to link against Blas with it as well.
set_property(TARGET Libxsmm
  APPEND PROPERTY INTERFACE_LINK_LIBRARIES Blas)

add_interface_lib_headers(
  TARGET Libxsmm
  HEADERS
  libxsmm.h
  )

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  Libxsmm
  )
