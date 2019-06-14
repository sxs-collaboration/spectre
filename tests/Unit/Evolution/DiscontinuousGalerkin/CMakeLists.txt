# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EvolutionDiscontinuousGalerkin")

set(LIBRARY_SOURCES
  Test_Filtering.cpp
  Test_InitializeElement.cpp
  Test_ObserveErrorNorms.cpp
  Test_ObserveFields.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/DiscontinuousGalerkin/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Domain;DomainCreators;ErrorHandling;Spectral;Time;Utilities"
  )

add_dependencies(
  ${LIBRARY}
  module_ConstGlobalCache
  )

add_subdirectory(SlopeLimiters)
