# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DomainTimeDependence)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  CubicScale.cpp
  None.cpp
  ScalingAndZRotation.cpp
  RegisterDerivedWithCharm.cpp
  RotationAboutZAxis.cpp
  Shape.cpp
  SphericalCompression.cpp
  UniformTranslation.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  CubicScale.hpp
  GenerateCoordinateMap.hpp
  None.hpp
  RegisterDerivedWithCharm.hpp
  RotationAboutZAxis.hpp
  ScalingAndZRotation.hpp
  Shape.hpp
  SphericalCompression.hpp
  TimeDependence.hpp
  UniformTranslation.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  CoordinateMaps
  DataStructures
  DomainStructure
  FunctionsOfTime
  Options
  Utilities
  PRIVATE
  GeneralRelativitySolutions
  ErrorHandling
  )
