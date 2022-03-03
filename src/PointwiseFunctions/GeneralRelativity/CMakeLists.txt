# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GeneralRelativity)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Christoffel.cpp
  DerivativeSpatialMetric.cpp
  DerivativesOfSpacetimeMetric.cpp
  ExtrinsicCurvature.cpp
  IndexManipulation.cpp
  InterfaceNullNormal.cpp
  InverseSpacetimeMetric.cpp
  KerrSchildCoords.cpp
  Lapse.cpp
  ProjectionOperators.cpp
  Ricci.cpp
  Shift.cpp
  SpacetimeMetric.cpp
  SpacetimeNormalOneForm.cpp
  SpacetimeNormalVector.cpp
  SpatialMetric.cpp
  TimeDerivativeOfSpacetimeMetric.cpp
  TimeDerivativeOfSpatialMetric.cpp
  Transform.cpp
  WeylElectric.cpp
  WeylMagnetic.cpp
  WeylPropagating.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Christoffel.hpp
  DerivativeSpatialMetric.hpp
  DerivativesOfSpacetimeMetric.hpp
  DetAndInverseSpatialMetric.hpp
  ExtrinsicCurvature.hpp
  IndexManipulation.hpp
  InterfaceNullNormal.hpp
  InverseSpacetimeMetric.hpp
  KerrSchildCoords.hpp
  Lapse.hpp
  ProjectionOperators.hpp
  Ricci.hpp
  Shift.hpp
  SpacetimeMetric.hpp
  SpacetimeNormalOneForm.hpp
  SpacetimeNormalVector.hpp
  SpatialMetric.hpp
  Tags.hpp
  TagsDeclarations.hpp
  TimeDerivativeOfSpacetimeMetric.hpp
  TimeDerivativeOfSpatialMetric.hpp
  Transform.hpp
  WeylElectric.hpp
  WeylMagnetic.hpp
  WeylPropagating.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC DataStructures
  INTERFACE ErrorHandling
  )

add_subdirectory(GeneralizedHarmonic)
add_subdirectory(Tags)
