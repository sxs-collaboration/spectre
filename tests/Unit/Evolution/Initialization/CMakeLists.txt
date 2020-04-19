# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EvolutionInitialization")

set(LIBRARY_SOURCES
  Test_ConservativeSystem.cpp
  Test_DgDomain.cpp
  Test_NonconservativeSystem.cpp
  Test_Tags.cpp
  Test_SetVariables.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Initialization/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Domain;Evolution;Utilities"
  )

add_dependencies(
  ${LIBRARY}
  module_ConstGlobalCache
  )
