# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Limiters)

set(LIBRARY "Test_EvolutionDg")

set(LIBRARY_SOURCES
  Actions/Test_ComputeTimeDerivative.cpp
  Test_InboxTags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/DiscontinuousGalerkin/"
  "${LIBRARY_SOURCES}"
  "Boost::boost;Evolution"
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
