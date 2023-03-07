# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "DomainBoundaryConditionsHelpers")

add_spectre_library(${LIBRARY} ${SPECTRE_TEST_LIBS_TYPE})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BoundaryCondition.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/Unit
  HEADERS
  BoundaryCondition.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Domain
  DomainBoundaryConditions
  DomainCreators
  DomainStructure
  Options
  Parallel
  Utilities
  PRIVATE
  Framework
  )
