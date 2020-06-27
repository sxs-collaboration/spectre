# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DomainTimeDependence)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  CubicScale.cpp
  None.cpp
  RegisterDerivedWithCharm.cpp
  UniformRotationAboutZAxis.cpp
  UniformTranslation.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Composition.hpp
  Composition.tpp
  CubicScale.hpp
  GenerateCoordinateMap.hpp
  None.hpp
  RegisterDerivedWithCharm.hpp
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
