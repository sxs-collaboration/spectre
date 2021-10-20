# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Test_ScalarAdvection)

set(LIBRARY_SOURCES
  BoundaryConditions/Test_Periodic.cpp
  BoundaryCorrections/Test_Rusanov.cpp
  FiniteDifference/Test_MonotisedCentral.cpp
  FiniteDifference/Test_Tag.cpp
  Subcell/Test_ComputeFluxes.cpp
  Subcell/Test_GhostData.cpp
  Subcell/Test_InitialDataTci.cpp
  Subcell/Test_TciOnDgGrid.cpp
  Subcell/Test_TciOnFdGrid.cpp
  Subcell/Test_VelocityAtFace.cpp
  Test_Characteristics.cpp
  Test_Fluxes.cpp
  Test_Tags.cpp
  Test_TimeDerivativeTerms.cpp
  Test_VelocityField.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/ScalarAdvection"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ScalarAdvection
  DataStructures
  Domain
  Utilities
  )
