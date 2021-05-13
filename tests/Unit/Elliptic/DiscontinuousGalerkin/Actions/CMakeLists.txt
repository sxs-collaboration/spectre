# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EllipticDgActions")

set(LIBRARY_SOURCES
  Test_InitializeDomain.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/DiscontinuousGalerkin/Actions/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CoordinateMaps
  DataStructures
  Domain
  DomainCreators
  DomainStructure
  EllipticDg
  Parallel
  Spectral
  Utilities
  )
