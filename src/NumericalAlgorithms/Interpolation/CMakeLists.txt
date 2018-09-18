# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Interpolation)

set(LIBRARY_SOURCES
  BarycentricRational.cpp
  IrregularInterpolant.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE Domain
  INTERFACE ErrorHandling
  INTERFACE Spectral
  )
