# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EllipticActions")

set(LIBRARY_SOURCES
  Test_InitializeAnalyticSolution.cpp
  Test_InitializeBackgroundFields.cpp
  Test_InitializeFields.cpp
  Test_InitializeFixedSources.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/Actions/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  AnalyticSolutions
  CoordinateMaps
  DataStructures
  Domain
  DomainCreators
  DomainStructure
  Elliptic
  EllipticDg
  ErrorHandling
  Parallel
  Utilities
  )
