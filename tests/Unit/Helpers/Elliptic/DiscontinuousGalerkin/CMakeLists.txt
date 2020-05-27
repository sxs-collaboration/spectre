# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "EllipticDgTestHelpers")

set(LIBRARY_SOURCES
  TestHelpers.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  DiscontinuousGalerkin
  Domain
  LinearOperators
  Spectral
  Utilities
  )
