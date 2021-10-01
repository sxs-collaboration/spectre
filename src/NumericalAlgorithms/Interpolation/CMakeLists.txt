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

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Blas
  DataStructures
  ErrorHandling
  Options
  Spectral
  Utilities
  PRIVATE
  Boost::boost
  Domain
  GSL::gsl
  )

add_subdirectory(Python)
