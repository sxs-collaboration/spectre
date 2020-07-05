# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  GeneralRelativity
  PRIVATE
  DerivSpatialMetric.cpp
  ExtrinsicCurvature.cpp
  GaugeSource.cpp
  Phi.cpp
  Pi.cpp
  SpacetimeDerivativeOfSpacetimeMetric.cpp
  SpacetimeDerivOfDetSpatialMetric.cpp
  SpacetimeDerivOfNormOfShift.cpp
  SpatialDerivOfLapse.cpp
  SpatialDerivOfShift.cpp
  TimeDerivativeOfSpacetimeMetric.cpp
  TimeDerivOfLapse.cpp
  TimeDerivOfLowerShift.cpp
  TimeDerivOfShift.cpp
  TimeDerivOfSpatialMetric.cpp
  )

spectre_target_headers(
  GeneralRelativity
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ConstraintGammas.hpp
  DerivSpatialMetric.hpp
  ExtrinsicCurvature.hpp
  GaugeSource.hpp
  Phi.hpp
  Pi.hpp
  SpacetimeDerivativeOfSpacetimeMetric.hpp
  SpacetimeDerivOfDetSpatialMetric.hpp
  SpacetimeDerivOfNormOfShift.hpp
  SpatialDerivOfLapse.hpp
  SpatialDerivOfShift.hpp
  TimeDerivativeOfSpacetimeMetric.hpp
  TimeDerivOfLapse.hpp
  TimeDerivOfLowerShift.hpp
  TimeDerivOfShift.hpp
  TimeDerivOfSpatialMetric.hpp
)
