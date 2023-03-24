# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "DomainStructureHelpers")

add_spectre_library(${LIBRARY} ${SPECTRE_TEST_LIBS_TYPE})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  NeighborHelpers.cpp
  OrientationMapHelpers.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/Unit
  HEADERS
  NeighborHelpers.hpp
  OrientationMapHelpers.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DomainStructure
  Utilities
  PRIVATE
  Framework
  )
