# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Interpolation)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BarycentricRational.cpp
  BarycentricRationalSpanInterpolator.cpp
  CubicSpanInterpolator.cpp
  CubicSpline.cpp
  InterpolationTargetApparentHorizon.cpp
  InterpolationTargetKerrHorizon.cpp
  InterpolationTargetLineSegment.cpp
  InterpolationTargetSpecifiedPoints.cpp
  InterpolationTargetWedgeSectionTorus.cpp
  IrregularInterpolant.cpp
  LinearSpanInterpolator.cpp
  RegularGridInterpolant.cpp
  SpanInterpolator.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AddTemporalIdsToInterpolationTarget.hpp
  BarycentricRational.hpp
  BarycentricRationalSpanInterpolator.hpp
  CMakeLists.txt
  CleanUpInterpolator.hpp
  CubicSpanInterpolator.hpp
  CubicSpline.hpp
  InitializeInterpolationTarget.hpp
  InitializeInterpolator.hpp
  Interpolate.hpp
  InterpolatedVars.hpp
  InterpolationTarget.hpp
  InterpolationTargetApparentHorizon.hpp
  InterpolationTargetDetail.hpp
  InterpolationTargetKerrHorizon.hpp
  InterpolationTargetLineSegment.hpp
  InterpolationTargetReceiveVars.hpp
  InterpolationTargetSpecifiedPoints.hpp
  InterpolationTargetWedgeSectionTorus.hpp
  Interpolator.hpp
  InterpolatorReceivePoints.hpp
  InterpolatorReceiveVolumeData.hpp
  InterpolatorRegisterElement.hpp
  Intrp.hpp
  IntrpOptionHolders.hpp
  IrregularInterpolant.hpp
  LagrangePolynomial.hpp
  LinearSpanInterpolator.hpp
  PointInfoTag.hpp
  RegularGridInterpolant.hpp
  SpanInterpolator.hpp
  Tags.hpp
  TryToInterpolate.hpp
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ApparentHorizons
  Blas
  Boost::boost
  DataStructures
  Domain
  DomainStructure
  ErrorHandling
  EventsAndTriggers
  GSL::gsl
  Options
  Spectral
  INTERFACE
  SystemUtilities
  )

add_subdirectory(Actions)
add_subdirectory(Callbacks)
add_subdirectory(Events)
add_subdirectory(Python)
