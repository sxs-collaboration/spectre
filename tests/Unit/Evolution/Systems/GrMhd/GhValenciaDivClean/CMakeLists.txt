# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GhValenciaDivClean")

set(LIBRARY_SOURCES
  BoundaryConditions/Test_ProductOfConditions.cpp
  BoundaryCorrections/Test_ProductOfCorrections.cpp
  Subcell/Test_FixConservativesAndComputePrims.cpp
  Subcell/Test_InitialDataTci.cpp
  Subcell/Test_PrimitiveGhostData.cpp
  Subcell/Test_PrimsAfterRollback.cpp
  Test_Constraints.cpp
  Test_Tags.cpp
  Test_StressEnergy.cpp
  Test_TimeDerivativeTerms.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/GrMhd/GhValenciaDivClean"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DomainBoundaryConditionsHelpers
  DomainHelpers
  Framework
  GeneralRelativityHelpers
  GeneralRelativitySolutions
  GeneralizedHarmonic
  GhGrMhdSolutions
  GhValenciaDivClean
  HydroHelpers
  ValenciaDivClean
  )
