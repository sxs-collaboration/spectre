# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "DomainHelpers")

set(LIBRARY_SOURCES
  DomainTestHelpers.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Boost::boost
  CoordinateMaps
  FunctionsOfTime

  PRIVATE
  DomainBoundaryConditionsHelpers
  DomainCreators
  ErrorHandling
  Framework

  PUBLIC
  DataStructures
  Domain
  DomainStructure
  Utilities
  )

add_subdirectory(BoundaryConditions)
