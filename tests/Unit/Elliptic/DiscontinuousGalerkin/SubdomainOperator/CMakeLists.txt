# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EllipticDgSubdomainOperator")

set(LIBRARY_SOURCES
  Test_SubdomainOperator.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  ConstitutiveRelations
  DataStructures
  Domain
  DomainBoundaryConditions
  DomainCreators
  DomainStructure
  Elasticity
  ElasticityActions
  ElasticityBoundaryConditions
  Elliptic
  EllipticDg
  EllipticDgSubdomainOperator
  ErrorHandling
  Parallel
  ParallelSchwarz
  Poisson
  PoissonBoundaryConditions
  Spectral
  Utilities
  )
