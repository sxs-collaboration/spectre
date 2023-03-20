# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Test_ForceFree)

set(LIBRARY_SOURCES
  BoundaryConditions/Test_Periodic.cpp
  BoundaryCorrections/Test_Rusanov.cpp
  Test_Characteristics.cpp
  Test_ElectricCurrentDensity.cpp
  Test_Fluxes.cpp
  Test_Sources.cpp
  Test_Tags.cpp
  Test_TimeDerivativeTerms.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/ForceFree"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Domain
  ForceFree
  GeneralRelativityHelpers
  Utilities
  )
