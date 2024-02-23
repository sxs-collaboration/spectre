# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EllipticDG")

set(LIBRARY_SOURCES
  Test_DgOperator.cpp
  Test_Penalty.cpp
  Test_Tags.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  AmrCriteria
  AnalyticSolutions
  DataStructures
  DiscontinuousGalerkin
  Domain
  DomainCreators
  DomainStructure
  Elliptic
  EllipticDg
  ErrorHandling
  Parallel
  ParallelAmr
  Poisson
  PoissonSolutions
  Spectral
  Utilities
  Xcts
  XctsSolutions
  )

add_subdirectory(Actions)
add_subdirectory(SubdomainOperator)
