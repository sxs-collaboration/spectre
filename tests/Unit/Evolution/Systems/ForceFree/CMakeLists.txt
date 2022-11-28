# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Test_ForceFree)

set(LIBRARY_SOURCES
  BoundaryConditions/Test_Periodic.cpp
  Test_Characteristics.cpp
  Test_Fluxes.cpp
  Test_Tags.cpp
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
