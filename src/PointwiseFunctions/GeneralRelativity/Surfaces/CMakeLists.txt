# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY StrahlkorperGr)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AreaElement.cpp
  EuclidianAreaElement.cpp
  Expansion.cpp
  ExtrinsicCurvature.cpp
  GradUnitNormalOneForm.cpp
  InverseSurfaceMetric.cpp
  Mass.cpp
  RadialDistance.cpp
  RicciScalar.cpp
  Spin.cpp
  SurfaceIntegralOfScalar.cpp
  SurfaceIntegralOfVector.cpp
  UnitNormalOneForm.cpp
)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AreaElement.hpp
  ComputeItems.hpp
  EuclidianAreaElement.hpp
  ExtrinsicCurvature.hpp
  GradUnitNormalOneForm.hpp
  InverseSurfaceMetric.hpp
  Mass.hpp
  RadialDistance.hpp
  RicciScalar.hpp
  Spin.hpp
  SurfaceIntegralOfScalar.hpp
  SurfaceIntegralOfVector.hpp
  Expansion.hpp
  Tags.hpp
  TagsDeclarations.hpp
  UnitNormalOneForm.hpp
)

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  SphericalHarmonics
  PRIVATE
  ErrorHandling
  GeneralRelativity
  LinearAlgebra
  Utilities
  )
