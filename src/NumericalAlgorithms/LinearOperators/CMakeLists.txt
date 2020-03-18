# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY LinearOperators)

set(LIBRARY_SOURCES
  ApplyMatrices.cpp
  CoefficientTransforms.cpp
  DefiniteIntegral.cpp
  Divergence.cpp
  ExponentialFilter.cpp
  IndefiniteIntegral.cpp
  Linearize.cpp
  MeanValue.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Blas
  DataStructures
  ErrorHandling
  Spectral

  INTERFACE
  Domain # Domain should be PUBLIC but then there's a cyclic dependency
  )
