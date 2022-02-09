# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GeneralRelativity")

set(LIBRARY_SOURCES
  Test_Christoffel.cpp
  Test_ComputeGhQuantities.cpp
  Test_ComputeSpacetimeQuantities.cpp
  Test_IndexManipulation.cpp
  Test_InterfaceNullNormal.cpp
  Test_KerrSchildCoords.cpp
  Test_ProjectionOperators.cpp
  Test_Ricci.cpp
  Test_Tags.cpp
  Test_Transform.cpp
  Test_WeylElectric.cpp
  Test_WeylMagnetic.cpp
  Test_WeylPropagating.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/GeneralRelativity/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CoordinateMaps
  DataBoxTestHelpers
  DataStructures
  DataStructuresHelpers
  Domain
  DomainStructure
  GeneralizedHarmonic
  GeneralRelativity
  GeneralRelativityHelpers
  GeneralRelativitySolutions
  LinearOperators
  Spectral
  Utilities
)
