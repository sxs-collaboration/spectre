# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_BoundaryEvolvedFields")

set(LIBRARY_SOURCES
  Test_InitializeBoundaryEvolvedFields.cpp
  Test_Tags.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DgSubcell
  DiscontinuousGalerkin
  Domain
  DomainStructure
  Evolution
  Spectral
  Time
  Utilities
  )
