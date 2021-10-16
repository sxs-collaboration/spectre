# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DomainTimeDependence)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  CompositionCubicScaleAndUniformRotationAboutZAxis.cpp
  CompositionUniformTranslation.cpp
  CubicScale.cpp
  None.cpp
  RegisterDerivedWithCharm.cpp
  SphericalCompression.cpp
  UniformRotationAboutZAxis.cpp
  UniformTranslation.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  CompositionCubicScaleAndUniformRotationAboutZAxis.hpp
  CompositionUniformTranslation.hpp
  CubicScale.hpp
  GenerateCoordinateMap.hpp
  None.hpp
  OptionTags.hpp
  RegisterDerivedWithCharm.hpp
  SphericalCompression.hpp
  TimeDependence.hpp
  UniformRotationAboutZAxis.hpp
  UniformTranslation.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  CoordinateMaps
  DataStructures
  Options
  Utilities
  PRIVATE
  ErrorHandling
  INTERFACE
  Domain
  FunctionsOfTime
  )
