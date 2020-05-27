# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PoissonDgTestHelpers")

set(LIBRARY_SOURCES
  FirstOrder.cpp
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
