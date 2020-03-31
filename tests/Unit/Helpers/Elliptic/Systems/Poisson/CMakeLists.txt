# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PoissonTestHelpers")

set(LIBRARY_SOURCES
  DgSchemes.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  DomainCreators

  PRIVATE
  DiscontinuousGalerkin
  EllipticDgTestHelpers
  LinearOperators
  Poisson
  )
