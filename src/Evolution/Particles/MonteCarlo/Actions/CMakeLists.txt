# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  FluidCouplingAction.hpp
  InitializeMonteCarlo.hpp
  Labels.hpp
  TimeStepActions.hpp
  TriggerMonteCarloEvolution.hpp
  )
