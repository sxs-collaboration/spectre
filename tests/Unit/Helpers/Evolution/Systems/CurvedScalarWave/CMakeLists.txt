# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "CurvedScalarWaveHelpers")

set(LIBRARY_SOURCES
  TestHelpers.cpp
  Worldtube/TestHelpers.cpp
  )

add_spectre_library(${LIBRARY} ${SPECTRE_TEST_LIBS_TYPE} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  DomainBoundaryConditions
  DomainBoundaryConditionsHelpers
  DomainCreators
  Framework
  )
