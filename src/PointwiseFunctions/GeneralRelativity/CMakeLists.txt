# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GeneralRelativity)

set(LIBRARY_SOURCES
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

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC DataStructures
  INTERFACE ErrorHandling
  )

add_subdirectory(GeneralizedHarmonic)
