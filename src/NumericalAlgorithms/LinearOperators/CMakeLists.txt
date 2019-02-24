# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY LinearOperators)

set(LIBRARY_SOURCES
  ApplyMatrices.cpp
  CoefficientTransforms.cpp
  DefiniteIntegral.cpp
  IndefiniteIntegral.cpp
  Linearize.cpp
  MeanValue.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE Domain
  INTERFACE ErrorHandling
  INTERFACE Spectral
  )
