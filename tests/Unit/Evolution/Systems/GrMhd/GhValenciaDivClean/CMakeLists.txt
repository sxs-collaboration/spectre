# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GhValenciaDivClean")

set(LIBRARY_SOURCES
  Actions/Test_SetInitialData.cpp
  BoundaryConditions/Test_ConstraintPreservingFreeOutflow.cpp
  BoundaryConditions/Test_DirichletAnalytic.cpp
  BoundaryConditions/Test_DirichletFreeOutflow.cpp
  BoundaryConditions/Test_ProductOfConditions.cpp
  BoundaryCorrections/Test_ProductOfCorrections.cpp
  FiniteDifference/Test_BoundaryConditionGhostData.cpp
  FiniteDifference/Test_Derivatives.cpp
  FiniteDifference/Test_FilterOptions.cpp
  FiniteDifference/Test_Filters.cpp
  FiniteDifference/Test_MonotonisedCentral.cpp
  Subcell/Test_FixConservativesAndComputePrims.cpp
  Subcell/Test_InitialDataTci.cpp
  Subcell/Test_NeighborPackagedData.cpp
  Subcell/Test_PrimitiveGhostData.cpp
  Subcell/Test_PrimsAfterRollback.cpp
  Subcell/Test_ResizeAndComputePrimitives.cpp
  Subcell/Test_TimeDerivative.cpp
  Test_Characteristics.cpp
  Test_Constraints.cpp
  Test_Tags.cpp
  Test_SetPiFromGauge.cpp
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
