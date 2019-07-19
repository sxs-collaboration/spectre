# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Interpolation)

set(LIBRARY_SOURCES
  BarycentricRational.cpp
  InterpolationTargetApparentHorizon.cpp
  InterpolationTargetKerrHorizon.cpp
  InterpolationTargetLineSegment.cpp
  InterpolationTargetWedgeSectionTorus.cpp
  IrregularInterpolant.cpp
  RegularGridInterpolant.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

add_dependencies(
  ${LIBRARY}
  module_ConstGlobalCache
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE Domain
  INTERFACE ErrorHandling
  INTERFACE Spectral
  )
