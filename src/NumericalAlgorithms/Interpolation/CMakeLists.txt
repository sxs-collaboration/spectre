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
  LinearLeastSquares.cpp
  LinearRegression.cpp
  LinearSpanInterpolator.cpp
  PolynomialInterpolation.cpp
  PredictedZeroCrossing.cpp
  RegularGridInterpolant.cpp
  SpanInterpolator.cpp
  ZeroCrossingPredictor.cpp
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
  LinearLeastSquares.hpp
  LinearRegression.hpp
  MultiLinearSpanInterpolation.hpp
  LinearSpanInterpolator.hpp
  PolynomialInterpolation.hpp
  PredictedZeroCrossing.hpp
  RegularGridInterpolant.hpp
  SpanInterpolator.hpp
  ZeroCrossingPredictor.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Blas
  DataStructures
  ErrorHandling
  Options
  Serialization
  Spectral
  Utilities
  PRIVATE
  Boost::boost
  GSL::gsl
  )

add_subdirectory(Python)
