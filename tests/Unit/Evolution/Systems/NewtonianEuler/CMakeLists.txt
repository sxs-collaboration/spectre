# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_NewtonianEuler")

set(LIBRARY_SOURCES
  BoundaryConditions/Test_DirichletAnalytic.cpp
  BoundaryConditions/Test_DemandOutgoingCharSpeeds.cpp
  BoundaryConditions/Test_Periodic.cpp
  BoundaryConditions/Test_Reflection.cpp
  BoundaryCorrections/Test_Hll.cpp
  BoundaryCorrections/Test_Hllc.cpp
  BoundaryCorrections/Test_Rusanov.cpp
  FiniteDifference/Test_AoWeno.cpp
  FiniteDifference/Test_MonotonisedCentral.cpp
  FiniteDifference/Test_Tag.cpp
  Subcell/Test_ComputeFluxes.cpp
  Subcell/Test_InitialDataTci.cpp
  Subcell/Test_NeighborPackagedData.cpp
  Subcell/Test_PrimitiveGhostData.cpp
  Subcell/Test_PrimsAfterRollback.cpp
  Subcell/Test_ResizeAndComputePrimitives.cpp
  Subcell/Test_TciOnDgGrid.cpp
  Subcell/Test_TciOnFdGrid.cpp
  Subcell/Test_TimeDerivative.cpp
  Test_Characteristics.cpp
  Test_ConservativeFromPrimitive.cpp
  Test_Fluxes.cpp
  Test_InternalEnergyDensity.cpp
  Test_KineticEnergyDensity.cpp
  Test_MachNumber.cpp
  Test_PrimitiveFromConservative.cpp
  Test_RamPressure.cpp
  Test_SoundSpeedSquared.cpp
  Test_Sources.cpp
  Test_SpecificKineticEnergy.cpp
  Test_Tags.cpp
  Test_TimeDerivativeTerms.cpp
  )

add_subdirectory(Limiters)
add_subdirectory(Sources)

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/NewtonianEuler/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CoordinateMaps
  DataStructures
  DiscontinuousGalerkin
  Domain
  DomainBoundaryConditionsHelpers
  DomainHelpers
  Framework
  Hydro
  NewtonianEuler
  NewtonianEulerAnalyticData
  NewtonianEulerSolutions
  NewtonianEulerSources
  Spectral
  Utilities
  )
