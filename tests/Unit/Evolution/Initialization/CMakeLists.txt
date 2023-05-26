# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EvolutionInitialization")

set(LIBRARY_SOURCES
  Test_ConservativeSystem.cpp
  Test_DgDomain.cpp
  Test_Evolution.cpp
  Test_NonconservativeSystem.cpp
  Test_SetVariables.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Initialization/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Amr
  DataStructures
  Domain
  Evolution
  Hydro
  Utilities
  )
