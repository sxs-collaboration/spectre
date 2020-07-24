# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EllipticDgSubdomainOperator")

set(LIBRARY_SOURCES
  Test_SubdomainOperator.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/DiscontinuousGalerkin/SubdomainOperator"
  "${LIBRARY_SOURCES}"
  "ConstitutiveRelations;DataStructures;Domain;ErrorHandling;Elasticity;EllipticDgTestHelpers;Poisson"
  )
