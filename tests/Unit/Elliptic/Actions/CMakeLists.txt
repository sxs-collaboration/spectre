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
  "DataStructures;Domain;ErrorHandling"
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
