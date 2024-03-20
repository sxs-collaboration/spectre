# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ValenciaDivClean")

set(LIBRARY_SOURCES
  Actions/Test_NumericInitialData.cpp
  BoundaryConditions/Test_DirichletAnalytic.cpp
  BoundaryConditions/Test_DemandOutgoingCharSpeeds.cpp
  BoundaryConditions/Test_HydroFreeOutflow.cpp
  BoundaryConditions/Test_Periodic.cpp
  BoundaryConditions/Test_Reflective.cpp
  BoundaryCorrections/Test_Hll.cpp
  BoundaryCorrections/Test_Rusanov.cpp
  FiniteDifference/Test_BoundaryConditionGhostData.cpp
  FiniteDifference/Test_MonotonicityPreserving5.cpp
  FiniteDifference/Test_MonotonisedCentral.cpp
  FiniteDifference/Test_PositivityPreservingAdaptiveOrder.cpp
  FiniteDifference/Test_Tag.cpp
  FiniteDifference/Test_Wcns5z.cpp
  Subcell/Test_ComputeFluxes.cpp
  Subcell/Test_FixConservativesAndComputePrims.cpp
  Subcell/Test_NeighborPackagedData.cpp
  Subcell/Test_PrimitiveGhostData.cpp
  Subcell/Test_PrimsAfterRollback.cpp
  Subcell/Test_ResizeAndComputePrimitives.cpp
  Subcell/Test_SetInitialRdmpData.cpp
  Subcell/Test_SwapGrTags.cpp
  Subcell/Test_TciOnDgGrid.cpp
  Subcell/Test_TciOnFdGrid.cpp
  Subcell/Test_TciOptions.cpp
  Subcell/Test_TimeDerivative.cpp
  Test_Characteristics.cpp
  Test_ComovingMagneticFieldMagnitude.cpp
  Test_ConservativeFromPrimitive.cpp
  Test_FixConservatives.cpp
  Test_Flattener.cpp
  Test_Fluxes.cpp
  Test_PrimitiveFromConservative.cpp
  Test_QuadrupoleFormula.cpp
  Test_SetVariablesNeededFixingToFalse.cpp
  Test_Sources.cpp
  Test_Tags.cpp
  Test_TimeDerivativeTerms.cpp
  Test_ValenciaDivClean.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DomainBoundaryConditionsHelpers
  DomainHelpers
  DomainStructure
  Framework
  GeneralRelativityHelpers
  GrMhdAnalyticData
  GrMhdSolutions
  Hydro
  HydroHelpers
  Importers
  Utilities
  ValenciaDivClean
  )
