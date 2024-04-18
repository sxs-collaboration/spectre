# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Deadlock)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  PrintDgElementArray.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  DataStructures
  DiscontinuousGalerkin
  Parallel
  Printf
  Time
  Utilities
  )
