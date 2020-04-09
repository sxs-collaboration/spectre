# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(GSL REQUIRED)

message(STATUS "GSL libs: ${GSL_LIBRARIES}")
message(STATUS "GSL incl: ${GSL_INCLUDE_DIR}")
message(STATUS "GSL vers: ${GSL_VERSION}")

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "GSL Version:  ${GSL_VERSION}\n"
  )

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  GSL::gsl GSL::gslcblas
  )
