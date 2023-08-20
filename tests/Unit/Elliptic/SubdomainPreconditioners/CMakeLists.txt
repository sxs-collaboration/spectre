# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EllipticSubdomainPreconditioners")

set(LIBRARY_SOURCES
  Test_MinusLaplacian.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Convergence
  CoordinateMaps
  DataStructures
  Domain
  DomainStructure
  EllipticSubdomainPreconditioners
  Options
  Parallel
  ParallelSchwarz
  Utilities
  )
