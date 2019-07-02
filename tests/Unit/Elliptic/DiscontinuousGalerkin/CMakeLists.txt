# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EllipticDG")

set(LIBRARY_SOURCES
  Test_ImposeInhomogeneousBoundaryConditions.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/DiscontinuousGalerkin/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Domain;ErrorHandling"
  )

add_dependencies(
  ${LIBRARY}
  module_ConstGlobalCache
  )
