# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GeneralRelativity)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Christoffel.cpp
  DerivativesOfSpacetimeMetric.cpp
  ExtrinsicCurvature.cpp
  IndexManipulation.cpp
  InverseSpacetimeMetric.cpp
  KerrSchildCoords.cpp
  Lapse.cpp
  Ricci.cpp
  Shift.cpp
  SpacetimeMetric.cpp
  SpacetimeNormalOneForm.cpp
  SpacetimeNormalVector.cpp
  SpatialMetric.cpp
  TimeDerivativeOfSpacetimeMetric.cpp
  WeylElectric.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Christoffel.hpp
  DerivativesOfSpacetimeMetric.hpp
  DetAndInverseSpatialMetric.hpp
  ExtrinsicCurvature.hpp
  IndexManipulation.hpp
  InverseSpacetimeMetric.hpp
  KerrSchildCoords.hpp
  Lapse.hpp
  Ricci.hpp
  Shift.hpp
  SpacetimeMetric.hpp
  SpacetimeNormalOneForm.hpp
  SpacetimeNormalVector.hpp
  SpatialMetric.hpp
  Tags.hpp
  TagsDeclarations.hpp
  TimeDerivativeOfSpacetimeMetric.hpp
  WeylElectric.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC DataStructures
  INTERFACE ErrorHandling
  )

add_subdirectory(GeneralizedHarmonic)
