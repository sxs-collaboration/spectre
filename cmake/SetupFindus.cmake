# Distributed under the MIT License.
# See LICENSE.txt for details.

if (NOT ("${SPECTRE_PARALLEL_LIB}" STREQUAL "findus"))
  return()
endif()

find_package(findus REQUIRED)

message(STATUS "Found findus at: ${findus_DIR}")

add_library(Charmxx::charmxx INTERFACE IMPORTED)
target_link_libraries(
  Charmxx::charmxx
  INTERFACE
  findus::findus
)
add_interface_lib_headers(
  TARGET
  Charmxx::charmxx
  HEADERS
  charm++.h
)

add_library(Charmxx::pup INTERFACE IMPORTED)
target_link_libraries(
  Charmxx::pup
  INTERFACE
  findus::findus
)
add_interface_lib_headers(
  TARGET
  Charmxx::pup
  HEADERS
  pup.h
  pup_stl.h
)

set_property(TARGET SpectreFlags
  APPEND PROPERTY
  INTERFACE_COMPILE_OPTIONS
  -DSPECTRE_USE_FINDUS
)

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  Charmxx::charmxx Charmxx::pup
  findus::findus
)
