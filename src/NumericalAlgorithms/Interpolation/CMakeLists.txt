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
  IrregularInterpolant.cpp
  LinearSpanInterpolator.cpp
  RegularGridInterpolant.cpp
  SpanInterpolator.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BarycentricRational.hpp
  BarycentricRationalSpanInterpolator.hpp
  CubicSpanInterpolator.hpp
  CubicSpline.hpp
  IrregularInterpolant.hpp
  LagrangePolynomial.hpp
  LinearSpanInterpolator.hpp
  RegularGridInterpolant.hpp
  SpanInterpolator.hpp
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
  DgSubcell
  Domain
  DomainStructure
  ErrorHandling
  EventsAndTriggers
  GSL::gsl
  Logging
  Options
  Spectral
  INTERFACE
  SystemUtilities
  )

add_subdirectory(Python)
