# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Domain)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AreaElement.cpp
  Block.cpp
  BlockLogicalCoordinates.cpp
  CreateInitialElement.cpp
  Domain.cpp
  DomainHelpers.cpp
  ElementDistribution.cpp
  ElementLogicalCoordinates.cpp
  ElementMap.cpp
  ElementToBlockLogicalMap.cpp
  FaceNormal.cpp
  FlatLogicalMetric.cpp
  InterfaceLogicalCoordinates.cpp
  JacobianDiagnostic.cpp
  MinimumGridSpacing.cpp
  RadiallyCompressedCoordinates.cpp
  SizeOfElement.cpp
  TagsTimeDependent.cpp
)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AreaElement.hpp
  Block.hpp
  BlockLogicalCoordinates.hpp
  CreateInitialElement.hpp
  DiagnosticInfo.hpp
  Domain.hpp
  DomainHelpers.hpp
  ElementDistribution.hpp
  ElementLogicalCoordinates.hpp
  ElementMap.hpp
  ElementToBlockLogicalMap.hpp
  FaceNormal.hpp
  FlatLogicalMetric.hpp
  InterfaceComputeTags.hpp
  InterfaceHelpers.hpp
  InterfaceLogicalCoordinates.hpp
  JacobianDiagnostic.hpp
  MinimumGridSpacing.hpp
  RadiallyCompressedCoordinates.hpp
  SizeOfElement.hpp
  Tags.hpp
  TagsCharacteristicSpeeds.hpp
  TagsTimeDependent.hpp
)

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  CoordinateMaps
  DataStructures
  DomainBoundaryConditions
  DomainStructure
  ErrorHandling
  Options
  Spectral
  Utilities
  PRIVATE
  LinearOperators
  INTERFACE
  FunctionsOfTime
  Time
)

add_subdirectory(Amr)
add_subdirectory(BoundaryConditions)
add_subdirectory(CoordinateMaps)
add_subdirectory(Creators)
add_subdirectory(FunctionsOfTime)
add_subdirectory(Protocols)
add_subdirectory(Python)
add_subdirectory(Structure)
add_subdirectory(Tags)
