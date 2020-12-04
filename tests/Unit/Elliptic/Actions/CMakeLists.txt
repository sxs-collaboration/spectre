# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EllipticActions")

set(LIBRARY_SOURCES
  Test_InitializeAnalyticSolution.cpp
  Test_InitializeSystem.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/Actions/"
  "${LIBRARY_SOURCES}"
  "CoordinateMaps;DataStructures;Domain;DomainCreators;DomainStructure;Elliptic;ErrorHandling;Parallel;AnalyticSolutions;Utilities"
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
