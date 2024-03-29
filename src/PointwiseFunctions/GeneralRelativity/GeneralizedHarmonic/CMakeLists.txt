# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Christoffel.cpp
  CovariantDerivOfExtrinsicCurvature.cpp
  DerivSpatialMetric.cpp
  ExtrinsicCurvature.cpp
  GaugeSource.cpp
  Phi.cpp
  Pi.cpp
  Ricci.cpp
  SecondTimeDerivOfSpacetimeMetric.cpp
  SpacetimeDerivOfDetSpatialMetric.cpp
  SpacetimeDerivOfNormOfShift.cpp
  SpacetimeDerivativeOfSpacetimeMetric.cpp
  SpatialDerivOfLapse.cpp
  SpatialDerivOfShift.cpp
  TimeDerivOfLapse.cpp
  TimeDerivOfLowerShift.cpp
  TimeDerivOfShift.cpp
  TimeDerivOfSpatialMetric.cpp
  TimeDerivativeOfSpacetimeMetric.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Christoffel.hpp
  ConstraintGammas.hpp
  CovariantDerivOfExtrinsicCurvature.hpp
  DerivSpatialMetric.hpp
  ExtrinsicCurvature.hpp
  GaugeSource.hpp
  Phi.hpp
  Pi.hpp
  Ricci.hpp
  SecondTimeDerivOfSpacetimeMetric.hpp
  SpacetimeDerivOfDetSpatialMetric.hpp
  SpacetimeDerivOfNormOfShift.hpp
  SpacetimeDerivativeOfSpacetimeMetric.hpp
  SpatialDerivOfLapse.hpp
  SpatialDerivOfShift.hpp
  TimeDerivOfLapse.hpp
  TimeDerivOfLowerShift.hpp
  TimeDerivOfShift.hpp
  TimeDerivOfSpatialMetric.hpp
  TimeDerivativeOfSpacetimeMetric.hpp
)

add_subdirectory(Python)
