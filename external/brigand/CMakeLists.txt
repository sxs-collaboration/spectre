# Distributed under the MIT License.
# See LICENSE.txt for details.

set(BRIGAND_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/external/brigand/include)

message(STATUS "Brigand include: ${BRIGAND_INCLUDE_DIR}")

add_library(Brigand INTERFACE IMPORTED GLOBAL)
target_include_directories(Brigand SYSTEM INTERFACE ${BRIGAND_INCLUDE_DIR})

add_interface_lib_headers(
  TARGET Brigand
  HEADERS
  brigand/brigand.hpp
)

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  Brigand
)
