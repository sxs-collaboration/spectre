# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Test_ForceFree)

set(LIBRARY_SOURCES
  BoundaryConditions/Test_Periodic.cpp
  BoundaryConditions/Test_DirichletAnalytic.cpp
  BoundaryCorrections/Test_Rusanov.cpp
  FiniteDifference/Test_AdaptiveOrder.cpp
  FiniteDifference/Test_MonotonisedCentral.cpp
  FiniteDifference/Test_Tags.cpp
  FiniteDifference/Test_Wcns5z.cpp
  Subcell/Test_ComputeFluxes.cpp
  Subcell/Test_GhostData.cpp
  Subcell/Test_NeighborPackagedData.cpp
  Subcell/Test_SetInitialRdmpData.cpp
  Subcell/Test_TciOnDgGrid.cpp
  Subcell/Test_TciOnFdGrid.cpp
  Subcell/Test_TciOptions.cpp
  Test_Characteristics.cpp
  Test_ElectricCurrentDensity.cpp
  Test_ElectromagneticVariables.cpp
  Test_Fluxes.cpp
  Test_MaskNeutronStarInterior.cpp
  Test_Sources.cpp
  Test_Tags.cpp
  Test_TimeDerivativeTerms.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Domain
  ForceFree
  ForceFreeAnalyticData
  ForceFreeSolutions
  GeneralRelativityHelpers
  Utilities
  )
