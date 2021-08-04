# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ValenciaDivClean")

set(LIBRARY_SOURCES
  BoundaryConditions/Test_DirichletAnalytic.cpp
  BoundaryConditions/Test_Outflow.cpp
  BoundaryConditions/Test_Periodic.cpp
  BoundaryCorrections/Test_Hll.cpp
  BoundaryCorrections/Test_Rusanov.cpp
  FiniteDifference/Test_MonotisedCentral.cpp
  FiniteDifference/Test_Tag.cpp
  Limiters/Test_CharacteristicHelpers.cpp
  Limiters/Test_Flattener.cpp
  Limiters/Test_KxrcfTci.cpp
  Limiters/Test_VariablesToLimit.cpp
  Subcell/Test_ComputeFluxes.cpp
  Subcell/Test_FixConservativesAndComputePrims.cpp
  Subcell/Test_GrTagsForHydro.cpp
  Subcell/Test_InitialDataTci.cpp
  Subcell/Test_PrimitiveGhostData.cpp
  Subcell/Test_PrimsAfterRollback.cpp
  Subcell/Test_ResizeAndComputePrimitives.cpp
  Subcell/Test_SwapGrTags.cpp
  Subcell/Test_TciOnDgGrid.cpp
  Subcell/Test_TciOnFdGrid.cpp
  Subcell/Test_TciOptions.cpp
  Subcell/Test_TimeDerivative.cpp
  Test_Characteristics.cpp
  Test_ConservativeFromPrimitive.cpp
  Test_FixConservatives.cpp
  Test_Fluxes.cpp
  Test_PrimitiveFromConservative.cpp
  Test_SetVariablesNeededFixingToFalse.cpp
  Test_Sources.cpp
  Test_Tags.cpp
  Test_TimeDerivativeTerms.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/GrMhd/ValenciaDivClean"
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
  HydroHelpers
  ValenciaDivClean
  ValenciaDivCleanLimiters
  )
