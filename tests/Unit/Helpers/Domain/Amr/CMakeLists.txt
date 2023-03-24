# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "DomainAmrHelpers")

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  NeighborFlagHelpers.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/Unit
  HEADERS
  NeighborFlagHelpers.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Amr
  DomainStructure
  Utilities
  )
