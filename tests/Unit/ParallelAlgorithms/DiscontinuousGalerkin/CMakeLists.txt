# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelDiscontinuousGalerkin")

set(LIBRARY_SOURCES
  Test_InitializeDomain.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/DiscontinuousGalerkin/"
  "${LIBRARY_SOURCES}"
  "CoordinateMaps;DataStructures;Domain;DiscontinuousGalerkin;ErrorHandling;Utilities"
  )

add_dependencies(
  ${LIBRARY}
  module_ConstGlobalCache
  )
